# Expert Parallelism (EP) 深度面试指南

> 目标岗位：大模型训练 Infra 专家
> 核心参考：DeepSeek-V4 Technical Report Section 3.1、DeepSeek-V3、Comet、DeepGEMM

---

## 一、EP 的数学本质与通信模型

### 1.1 EP 要解决什么问题

MoE 层的参数量与专家数成正比。以 DeepSeek-V4-Pro 为例：

- 384 个路由专家，每个专家包含 SwiGLU 的 gate/up/down 三个矩阵
- 单个专家参数量 = 3 × h × d_ff = 3 × 7168 × 3072 ≈ 66M 参数
- 384 个专家总参数 ≈ 25B（仅 MoE 部分）
- 单卡 H100 (80GB) 装不下

因此必须将不同专家分布到不同 GPU 上——这就是 EP。

### 1.2 通信量的精确计算

MoE 层核心是两次 All-to-All：

**Dispatch（Token → Expert GPU）**：
```
输入：每个 rank 持有 batch 中的一部分 token
操作：将每个 token 的 hidden state 发送到其 top-k 专家所在的 GPU
输出：每个 rank 收到分配给本地专家的所有 token
```

**Combine（Expert GPU → Token GPU）**：
```
输入：每个 rank 计算完本地专家的输出
操作：将专家输出发送回 token 原始的 rank
输出：每个 rank 恢复其原始 batch 顺序
```

**通信量公式**（per token-expert pair）：

| Stage | 精度 | 每 token-expert 字节数 |
|-------|------|----------------------|
| Dispatch (V3) | BF16 | 2h |
| Dispatch (V4) | FP8 | h |
| Combine (V3) | BF16 | 2h |
| Combine (V4) | BF16 | 2h |
| **V4 合计** | 混合 | **3h** |

### 1.3 计算-通信比的精确推导（核心公式，必须会背）

这是 DeepSeek-V4 论文中最重要的公式之一。

**每个 token-expert pair 的计算量**：

SwiGLU 包含三个矩阵乘法：
- Gate projection: W_gate ∈ R^{h × d_ff} → 2·h·d_ff FLOPs
- Up projection: W_up ∈ R^{h × d_ff} → 2·h·d_ff FLOPs
- Down projection: W_down ∈ R^{d_ff × h} → 2·h·d_ff FLOPs
- 合计：**6·h·d_ff FLOPs**

**每个 token-expert pair 的通信量**（V4）：
- FP8 Dispatch: h 字节
- BF16 Combine: 2h 字节
- 合计：**3h 字节**

**关键比值**：
```
V_comp / V_comm = 6·h·d_ff / 3·h = 2·d_ff
```

**这导出了通信隐藏条件**：

```
C / B ≤ V_comp / V_comm = 2·d_ff
```

其中 C = 峰值算力 (FLOP/s)，B = 互联带宽 (Bytes/s)。

**代入具体数字**：

| 模型 | d_ff | C/B 阈值 (FLOPs/Byte) | 每 GB/s 可隐藏的 TFLOP/s |
|------|------|----------------------|-------------------------|
| V4-Flash | 2048 | 4096 | 4.1 |
| V4-Pro | 3072 | **6144** | **6.1** |

**实际含义**：对于 V4-Pro，每 1 GB/s 的互联带宽可以隐藏 6.1 TFLOP/s 的计算。一旦带宽超过此阈值，再加带宽是浪费。

**面试时建议直接说出**：`C/B ≤ 2d_ff = 6144 FLOPs/Byte`（V4-Pro），然后解释推导过程。

### 1.4 为什么 FP8 Dispatch + BF16 Combine 是混合精度？

这并非随意选择：

- **Dispatch 用 FP8**：token hidden state 作为 GEMM 输入，FP8 对精度不敏感。且 dispatch 通信量占总通信量的 1/3（h 字节 vs 总共 3h），量化收益高。
- **Combine 用 BF16**：专家输出需要与其他专家的输出做加权求和（routing weights），且要加到 residual stream 上。Combine 结果直接参与后续 Attention 的 QKV 计算，对精度敏感。因此保留 BF16。
- **等效精度**：这样混合的总通信精度等效于 ~FP12，实际训练无精度损失。

---

## 二、细粒度 EP：从粗粒度到 Wave 调度的完整推理链

### 2.1 为什么需要细粒度

**传统粗粒度 EP** 的执行顺序：
```
时间轴：
|──── Dispatch All-to-All ────|──── Expert Compute ────|──── Combine All-to-All ────|
                              通信和计算完全串行
```

问题：即使总计算时间 > 总通信时间，因为串行执行，通信延迟仍然暴露在关键路径上。

**Comet (Zhang et al., 2025b) 的改进**：
```
时间轴：
|── Dispatch ──|────────────────────────────────────|
               |── Linear-1 ──|── Linear-2 ──|      |
                              |── Combine ───|      |
               Dispatch 和 Linear-1 重叠，Combine 和 Linear-2 重叠
               理论加速：1.42×
```

Comet 的问题：重叠粒度仍然太粗。Dispatch 必须全部完成才能开始任何计算，Combine 必须等全部计算完成。

### 2.2 V4 的 Wave 调度：三级流水线

**核心思想**：将 E 个专家均分为 W 个 wave，每个 wave 有 E/W 个专家。

每个 wave 经历三级流水：
```
Wave i:   | Dispatch_i | Compute_i | Combine_i |
Wave i+1:              | Dispatch_{i+1} | Compute_{i+1} | Combine_{i+1} |
Wave i+2:                               | Dispatch_{i+2} | Compute_{i+2} | ...
```

**稳态执行（三路并发）**：
```
当前 wave 的计算 || 下一 wave 的 Dispatch || 上一 wave 的 Combine
```

**关键设计决策——Wave 数量 W 的选择**：

- W 太小（如 W=1）：退化为粗粒度 EP，无重叠
- W 太大（如 W=E）：每 wave 仅 1 个专家，token 分布可能极不均衡
- 最优 W：使得每 wave 的计算时间与 Dispatch/Combine 通信时间可比，以实现完美流水线

V4 论文未明确给出 W，但从性能数据可以反推：在 V4-Flash（256 专家）上，理论加速 1.92×（对比 naive 的 1.0× 和 Comet 的 1.42×），说明 W 足够大以实现有效重叠。

### 2.3 为什么 Wave 调度对 RL Rollout 特别有效

RL rollout 场景的特点：
- **极小 batch**：通常单条或几条 sequence
- **token 数远小于 expert 数**：很多专家收到 0 token
- **通信延迟占比极高**：传统 EP 通信开销可能超过计算本身

Wave 调度的增益：
- 只对有 token 的 wave 做通信和计算
- 空 wave 直接跳过，无任何开销
- 因此对小 batch 加速更大（最高 1.96× vs 常规 1.50-1.73×）

---

## 三、MegaMoE Mega-Kernel 的实现细节

### 3.1 为什么需要单一融合 Kernel

独立 kernel launch 的问题：
- 每个 kernel launch 有 CPU→GPU 调度延迟（~5-20μs per launch）
- 每个 kernel 之间需要全局同步（implicit barrier at kernel boundary）
- 中间结果需要写回 global memory 再由下一个 kernel 读取
- GPU SM 在 kernel 边界处空闲

Mega-Kernel 解决方案：
- 单一 kernel 内包含全部逻辑：token 重排 → All-to-All → GEMM → SwiGLU → All-to-All → 输出重组
- 使用 CUDA cooperative groups 或 persistent threads 实现 SM 间协调
- 中间结果保持 on-chip（shared memory / registers）或通过 SM-to-SM 通信

### 3.2 Pull vs Push Dispatch 的底层原因

V4 选择 **Pull 模式**而非传统的 Push：

**Push 模式**（发送方驱动）：
```
GPU A: "我有 token 要发给 GPU B 的 Expert 3"
GPU A: cudaMemcpy (or NCCL send) → GPU B
问题：每个 wave 都需要异步通知 B，B 需要知道数据已就绪
在 wave 级别（每 wave 可能仅几十 μs），通知延迟成为瓶颈
```

**Pull 模式**（接收方驱动）：
```
GPU B: "我负责 Expert 3, 5, 7, 我需要从其他 GPU 拉取发给这些 expert 的 token"
GPU B: cudaMemcpy (从 GPU A) ← GPU A 的内存
好处：B 自己控制节奏，不需要显式通知，通信发起方与计算方是同一个 GPU
```

Pull 模式之所以可行，是因为：
- 路由结果（token → expert 映射）在 EP 组内是确定的
- 每个 GPU 知道哪些专家在本地，可以精确计算需要从哪些 rank 拉多少 token
- 计算方主动拉取 = 零通知延迟

### 3.3 Wave Quantization（波次量化）问题

这是一个容易被忽略但实际很重要的工程问题。

**定义**：当最后一波（tail wave）的 token 数不满足完整的 wave 大小时，GPU SM 利用率下降，出现「尾波量化」损失。

**产生原因**：
- 每 wave 的 token 数 = total_tokens / W，通常不是 SM 数（或 wave 内 GEMM tile 数）的整数倍
- 最后一波可能只有部分 tile 有实际计算，其余 tile 空转
- 小 batch 下尤其严重（total_tokens 本身就小）

**V4 的解决方案**（Section 3.3）：
- **双核策略**：
  1. 满波用「单 SM 核」：整个 GEMM 由一个 SM 完成，确保满波吞吐（避免 split-k 导致的 batch 不变性问题）
  2. 尾波用「多 SM 核」：多个 SM 并发处理，降低尾波延迟
- 精心设计的累加顺序确保两种核的结果 bit-identical

---

## 四、EP 下的负载均衡：从 Aux Loss 到无辅助损失

### 4.1 为什么 Auxiliary Loss 有问题

传统 MoE（如 GShard, Switch Transformer）：
```
Loss = LM_Loss + α × Load_Balance_Loss
Load_Balance_Loss = E × Σ_i (f_i × P_i)
其中 f_i = fraction of tokens routed to expert i
     P_i = average routing probability for expert i
```

问题：
- α 是超参，需要调。α 太小 → 负载不均衡，token dropping 严重；α 太大 → 损害 LM 性能
- 辅助损失和 LM 损失存在竞争，本质上是多目标优化的 trade-off
- 训练初期路由不成熟时，Aux Loss 可能引导路由到次优状态

### 4.2 DeepSeek 无辅助损失策略的完整机制

**Step 1**：计算原始路由亲和分数
```
s_i = Sqrt(Softplus(W_route · x))  // V4 用 Sqrt(Softplus)，V3 用 Sigmoid
```

为什么 V4 改为 Sqrt(Softplus)？
- Sigmoid 在饱和区梯度消失，路由训练慢
- Softplus 是 ReLU 的平滑版本，梯度更健康
- Sqrt 压缩大值，防止少数 token 的亲和分数过大主导路由

**Step 2**：加 per-expert bias
```
s'_i = s_i + b_i
```
其中 b_i 是每个专家独立的可学习/可调偏置。

**Step 3**：Top-K 选择
```
selected_experts = TopK({s'_i | i ∈ all_routed_experts}, k)
```

**Step 4**：偏置更新（非梯度更新！这是关键）

不通过反向传播更新 b_i，而是基于负载统计直接调整：
```
if expert_i_overloaded:
    b_i -= η_bias  // 减少该专家的 bias，降低后续被选中的概率
elif expert_i_underloaded:
    b_i += η_bias  // 增加该专家的 bias，提高后续被选中的概率
```
其中 η_bias = 0.001（V4 超参），更新频率通常为每 step。

**关键区分**：b_i 的更新走的是启发式规则，**不参与梯度计算**。这意味着负载均衡信号和 LM 优化信号完全解耦。

### 4.3 序列级平衡损失

V4 在无辅助损失基础上增加了 lightweight 的序列级平衡损失（权重 0.0001）：

作用：防止单条长序列内部出现极端不均衡（如某条 64K 序列的所有 token 都路由到某几个专家）。

为什么需要？因为 per-expert bias 是全局统计量，无法感知单序列内的分布异常。

---

## 五、EP + ZeRO + DP 的混合并行：梯度同步细节

### 5.1 并行维度布局

DeepSeek V4 的典型并行配置：
```
Total GPUs = DP × TP × PP × EP
```

- **DP (Data Parallel)**：同一组参数在不同的 DP rank 上处理不同的 micro-batch
- **TP (Tensor Parallel)**：单层内的矩阵乘法切分到多个 GPU
- **PP (Pipeline Parallel)**：不同层放在不同 GPU，用 DualPipe 1F1B 调度
- **EP (Expert Parallel)**：不同专家在不同 GPU

### 5.2 梯度同步的三条路径

**路径 1：稠密参数（Attention, Embedding, Norm）**
```
存储方式：ZeRO-1/2 分片在 DP 维度
同步方式：reduce-scatter → all-gather（标准 ZeRO）
精度：FP32
```

**路径 2：MoE 专家参数**
```
存储方式：EP 维度上每个 rank 持有不同专家
同步方式：DP 维度上的梯度需要 all-reduce
精度：BF16 随机舍入 + All-to-All + 本地 FP32 求和（两阶段规约）
```

**路径 3：Muon 特殊处理**
```
Muon 需要完整的梯度矩阵（不能切分），与 ZeRO 冲突
解决方案：限制 ZeRO 并行度上限 + 背包算法均衡分配
         + 同形状参数合并批处理 Newton-Schulz
         + 超出限制的 DP group 做冗余 Muon 更新
```

### 5.3 两阶段梯度规约的细节

传统做法（有问题）：
```
All-Reduce (FP32) → 精确但通信量大
All-Reduce (BF16) → 通信量小但累加误差大（低精度加法树）
```

V4 的两阶段方案：
```
Phase 1: All-to-All 交换 BF16 梯度（不累加，仅交换）
Phase 2: 各 rank 本地 FP32 求和
```

为什么这比 All-Reduce 更好？
- All-Reduce 的 ring/tree 算法中，每跳都在低精度累加 → 误差传播
- All-to-All + 本地求和 → 只有一次低精度传输，精度损失最小
- 对于 MoE 梯度，All-to-All 天然与 EP 通信模式匹配（Expert 参数已经分布在 EP rank 上）

### 5.4 Muon + ZeRO 的背包算法

问题：Muon 需要完整矩阵做 Newton-Schulz 正交化，但 ZeRO 按元素切分。

V4 的方案：
- 设定 ZeRO 并行度上限（每个 rank 最多管 5 个参数矩阵）
- 把稠密参数按形状分组，用**背包算法**分配到各 ZeRO rank
- 目标是让各 rank 的参数总量均衡
- Padding 到相同大小以便 reduce-scatter
- 额外内存开销 < 10%

对于超出 ZeRO 规模的 DP group：
- 在这些 group 上**冗余计算** Muon 更新
- 用额外计算换内存（典型的 compute-memory trade-off）

---

## 六、MoE 反向的确定性（Determinism）

### 6.1 为什么 MoE 反向天然非确定性

**来源 1：原子加顺序不确定**

MoE 反向中，多个 token 可能路由到同一个专家。这些 token 对专家权重的梯度贡献需要累加。

```
// 伪代码：多个 SM 并发写同一个梯度 buffer
atomicAdd(&grad_W[i][j], local_gradient);  // ← 浮点加法不可交换！
```

FP32 加法不满足结合律：(a+b)+c ≠ a+(b+c) 在浮点下。因此原子加的顺序不同 → 结果不同 → 不可复现。

**来源 2：跨 rank 的 token 数量变化**

不同 step 中，路由到每个专家的 token 数量不同。EP 通信 buffer 的大小取决于 token 数量：
- Buffer 大小变化 → 内存布局变化 → 累加顺序变化 → 结果不同

### 6.2 V4 的解决方案

**MoE 反向确定性方案（Section 3.3）**：

1. **Token 顺序预处理**：在 dispatch 前，将每个 rank 要处理的 token 按固定规则排序（如按 token ID 或 expert ID），确保同一批 token 在不同 run 中处理顺序一致。

2. **Rank 间 Buffer 隔离**：不同 rank 发来的数据写入不同的 buffer 区域，避免多 rank 竞争同一 buffer 位置。

3. **Per-SM 独立 Buffer + 确定性规约**：每个 SM 先累加到自己的私有 buffer，最后做一次确定性全局求和（预先排好顺序的 tree reduction）。

4. **禁止 Split-KV（注意力）**：Flash Attention 的 split-KV 会引入非确定性（K/V 切分方式不同 → 累加顺序不同）。V4 的注意力 kernel 不在 KV 维度做 split。

### 6.3 Batch 不变性（Batch Invariance）

更强的保证：同一 batch 数据，无论 batch 内样本如何排列，结果 bit-identical。

实现要点：
- 不用 Split-K GEMM（小 batch 场景）
- 双核注意力策略（满波单 SM 核 + 尾波多 SM 核）
- 所有 reduction 使用确定性顺序

---

## 七、路由机制的深度解析

### 7.1 路由亲和分数：Sigmoid → Sqrt(Softplus) 的数学原因

```
V2/V3: s_i = Sigmoid(W_route · x)    → 输出范围 (0, 1)
V4:    s_i = Sqrt(Softplus(W_route · x)) → 输出范围 (0, +∞)，但增长被 sqrt 抑制
```

| 激活函数 | 输出范围 | 梯度行为 | 问题 |
|---------|---------|---------|------|
| Sigmoid | (0, 1) | 远离 0 时梯度消失 | 路由 scores 全部接近 0.5，区分度差 |
| Softplus | (0, +∞) | 正值始终有梯度 | 大值无界，可能少数 token 垄断专家 |
| Sqrt(Softplus) | (0, +∞) | 正值有梯度，但大值被压缩 | 平衡了区分度与极端值控制 |

### 7.2 Hash 路由：为什么放在最前几层

V4 将初始 3 个 Transformer block 的 FFN 从 Dense 替换为 Hash MoE + 后续层使用 Learnable Router MoE。

Hash 路由原理：
```
expert_id = hash(token_id) % num_experts
```

为什么有效？
- 输入层（embedding 附近）的路由信息量少，learnable router 容易退化为静态分配
- Hash 路由可以看作是 learnable router 的一个廉价替代
- 确定性的分配使得训练更稳定
- 没有路由训练开销

### 7.3 Anticipatory Routing：解决训练尖峰

**问题**：万亿参数 MoE 训练中，偶尔出现 loss spike。回滚能恢复，但治标不治本。

**观察**：spike 与 MoE 异常值及路由机制的 feedback loop 有关。

**Anticipatory Routing 原理**：
```
step t 的输入特征用参数 θ_t 计算
但路由选择用历史参数 θ_{t-Δt} 计算的路由索引
```

为什么这能缓解 spike？
- θ_t 刚更新，可能产生 outlier 路由分布
- θ_{t-Δt} 是已验证稳定的参数，路由分布更平滑
- 用旧路由 + 新特征 = 破坏了可能导致 spike 的 feedback loop

**代价**：
- 需要在 t-Δt 预取数据并缓存路由索引（额外通信和存储）
- Wall-time 额外开销 ~20%
- 仅在 spike 检测到时短暂启用，稳定后关闭

### 7.4 容量约束移除

V3 对每个专家的路由 token 数有上限约束（capacity constraint），V4 移除了此约束。

**为什么 V3 需要**：防止某些专家被过载导致显存溢出（每个 expert 的计算 buffer 需要预先分配固定大小）。

**为什么 V4 可以移除**：
- 无辅助损失策略使负载天然更均衡
- 序列级平衡损失进一步防止单序列极端不均衡
- 细粒度 EP 的 Wave 调度中，buffer 可以在 wave 间复用，降低了预分配压力
- 动态 buffer 分配 + 更灵活的内存管理

---

## 八、EP 性能分析：实测数据与理论限速

### 8.1 理论加速比分析

论文 Figure 5 的理论加速比：

```
Naive:                     1.00× (baseline)
Comet (overlap L1/Dispatch + L2/Combine): 1.42×
V4 (wave-based):           1.92×
```

推导：假设通信占比为 p，计算占比为 (1-p)，则：
- Naive 时间：T = T_comm + T_comp
- Comet 时间：T = max(T_dispatch, T_L1) + max(T_L2, T_combine) ≈ max(p, 1-p)·T
- V4（W 个 wave）：T = T_comm/W + T_comp/W（流水线填充后每个 wave 的开销被摊销）

对于 V4-Flash 的配置（256 专家，每 token 6 专家），p ≈ 通信时间 / 总时间 ≈ 0.35。理论上完美流水线加速比 = 1/(1-p) ≈ 1.54×。但 Wave 调度消除了尾延迟，实际效果更好（达到 1.92× 理论值）。

### 8.2 实际性能数据

| 场景 | 加速比 | 说明 |
|------|--------|------|
| 通用推理 | 1.50-1.73× | batch size 中等 |
| RL Rollout | 最高 1.96× | 极小 batch，通信占比极高 |
| 高速 Agent 服务 | 最高 1.96× | 延迟敏感，需流水线 |

### 8.3 功耗墙问题

融合 mega-kernel 的一个反直觉问题：**同时满载计算、内存带宽、网络带宽 → 总功耗超过 GPU TDP → 触发降频 → 性能反而下降**。

这是一个真实的硬件限制。V4 论文建议硬件厂商为这类全并发负载预留更多功耗预算。

---

## 九、EP 的容错与可抢占设计（后训练场景）

### 9.1 为什么 RL/OPD Rollout 需要特殊容错

后训练阶段（RL + On-Policy Distillation）的 rollout 特点：
- 单个请求的生成可能很长（thinking mode 可达数万 token）
- 集群级可抢占调度（任何任务随时可能被抢占）
- 硬件故障常见（大规模集群中几乎是常态）

### 9.2 Token 粒度 WAL（Write-Ahead Log）

V4 后训练基础设施的核心设计：

```
每生成一个新 token：
  1. 计算 logits
  2. 采样得到 token
  3. 立即追加到 WAL（persistent storage）
  4. 更新 KV Cache

抢占发生时：
  1. 保存所有未完成请求的 KV Cache 到持久化存储
  2. WAL 已记录当前已生成的 token 序列

恢复时：
  1. 读取 WAL + 保存的 KV Cache
  2. 从断点继续解码（不需要从头重新生成！）
```

**为什么不能简单重新生成**：从头重新生成未被抢占的请求会引入**长度偏差**——只有被抢占的请求需要重新生成，而它们往往恰好是长序列请求。这会导致训练数据中长序列的系统性偏差。

### 9.3 Batch 不变性与容错的配合

如果生成过程是 batch 不变且确定性的，可以用固定 PRNG seed 完全复现。但 V4 选择 WAL 方案，因为：
- 保持确定性需要额外工程成本
- WAL + KV 恢复的开销远小于重新生成

---

## 十、深度面试问答（15 题）

### Q1: 推导 MoE 层的 computation-communication ratio，并解释其工程意义。

在 MoE 层中，每个 token-expert pair：
- 计算量 = 6·h·d_ff FLOPs（SwiGLU 三个矩阵乘，各 2·h·d_ff）
- 通信量 = 3h 字节（FP8 dispatch h 字节 + BF16 combine 2h 字节）
- 比值 = 2·d_ff

因此通信可被隐藏的充分条件是：C/B ≤ 2·d_ff。

对 V4-Pro (d_ff=3072)：阈值 6144 FLOPs/Byte。给定 400GB/s NVLink，最多可隐藏 2.46 PFLOP/s 计算，远超单卡 FP8 峰值 ~2 PFLOP/s。因此 NVLink 下通信可完全隐藏。给定 50GB/s InfiniBand，最多可隐藏 307 TFLOP/s，远低于峰值，通信成为瓶颈。

工程意义：这个公式是硬件选型的核心依据——给定模型配置，可以直接算需要多少带宽才能通信不成为瓶颈。

### Q2: Wave 调度与 Comet 的重叠方案有什么本质区别？

Comet 是**两段式重叠**：Dispatch 与 Linear-1 重叠，Combine 与 Linear-2 重叠。但它需要 dispatch 全部完成才能开始计算。

Wave 是**多段流水线重叠**：将专家切成 W 个 wave，每个 wave 独立流水线。关键区别：
1. 流水线深度：Comet = 2 段，Wave = W 段
2. 启动延迟：Wave 只需等第一个 wave 的 dispatch 完成即可开始计算
3. 尾延迟：Wave 的尾波可以用多 SM 核加速
4. 对 batch 大小的鲁棒性：Wave 在极小 batch 下增益更大

### Q3: 为什么 DeepSeek 选择 Pull 而非 Push 做 Dispatch？

Push 的发送方需要通知接收方数据已就绪。在细粒度 wave 调度中，每个 wave 粒度极小（可能几十 μs），通知延迟不可忽略。

Pull 模式中，每个 GPU 知道自己负责哪些专家，可以直接从其他 rank 拉取数据。计算方 = 接收方 = 发起方，通信节奏由计算方控制，不需要额外信令。

代价：发送方需要提前准备好数据并暴露可读 buffer，增加一些内存和同步开销。

### Q4: 无辅助损失负载均衡的本质是什么？为什么比 Aux Loss 好？

本质是将负载均衡信号与梯度优化信号**解耦**。Aux Loss 通过梯度影响路由参数，与 LM 优化目标竞争梯度方向。无辅助损失通过启发式规则直接调整 per-expert bias（不经过反向传播），两个信号独立。

这种设计的好处：
1. 不需要调 α 超参
2. LM 性能不被负载均衡目标损害
3. 收敛更稳定（路由不会在 LM 和负载均衡之间振荡）

### Q5: FP4 QAT 在 MoE EP 中的角色是什么？

V4 后训练阶段对 MoE 专家权重做 MXFP4 量化感知训练：
- 前向：FP32 master → 量化到 FP4 → 反量化到 FP8 → FP8 GEMM
- 反向：对 FP8 权重求梯度，STE 回传 FP32 master
- 推理：直接用原生 FP4 权重，显存减少 ~50%

关键设计：FP4(E2M1) → FP8(E4M3) 反量化。FP8 的动态范围比 FP4 大得多（E4M3 的指数范围是 E2M1 的 8 倍），因此 FP4 的细粒度 scale 可以被 FP8 完全吸收。

### Q6: EP 训练中，为什么 MoE 梯度的同步要用两阶段 All-to-All + 本地求和，而不是直接 All-Reduce？

All-Reduce（ring 或 tree）的每一跳都在低精度（BF16）做累加，误差在 N 跳中累积。N 越大，误差越大。

两阶段方案先用 All-to-All 在 BF16 精度下纯传输（不累加），然后各 rank 在 FP32 本地求和。只有一次低精度传输，无误差传播。

这和 EP 的通信模式天然吻合——MoE 参数已经按 EP 维度分布，All-to-All 的通信拓扑与 EP 的 Dispatch/Combine 一致。

### Q7: 解释 Wave Quantization 问题及其解决方案。

当 token 总数不能被 wave 内的 SM 数量或 tile 大小整除时，尾波（最后几个 wave 或最后一个 wave）的计算资源利用率下降。

V4 的解决方案：双核策略：
- 满波：单 SM 做完整个 GEMM（避免 split-k 导致 batch 不变性问题）
- 尾波：多 SM 并发处理，降低延迟
- 精心设计的累加顺序保证两种核结果 bit-identical

### Q8: MoE EP 为什么对 batch 大小敏感？

小 batch 下：
- 每个专家收到的 token 数可能为 0
- 通信时间相对计算时间占比更高
- Wave 调度中空 wave 多，流水线效率低

大 batch 下：
- 负载天然更均衡（统计意义上）
- 通信可被大型 GEMM 隐藏
- Wave 调度接近完美流水线

这就是为什么 V4 的加速比在 RL rollout（小 batch）下最高 1.96×——小 batch 场景的优化空间最大。

### Q9: 为什么把前几层的 Dense FFN 换成 Hash MoE？

输入层的 embedded token 信息量有限，learnable router 的区分度差。Hash 路由用 token ID 的哈希函数确定性分配专家，避免了低质量的路由训练。同时 Hash MoE 增加了模型容量（比 Dense FFN 更多参数），且无路由训练开销。

3 层之后 token 通过 Attention 已经融合了上下文信息，此时 learnable router 才能有效工作。

### Q10: EP 通信的功耗墙是什么意思？怎么解决？

Mega-kernel 使得 GPU 的计算单元（Tensor Core）、内存带宽（HBM）、网络带宽（NVLink/NIC）三者同时满载。总功耗 = P_compute + P_memory + P_network > GPU TDP → 硬件降频 → 实际性能低于理论峰值。

这是硬件的物理限制，软件层面只能缓解（如降低 SM 频率换取更多功耗给网络）。

### Q11: EP 中，token 如何在各个 rank 间路由？路由决策在哪里发生？

所有 rank 对自己持有的 token 独立计算路由亲和分数。由于 embedding 和 attention 输出在每个 rank 上都是完整的（TP 内是切分的，但 EP 前会通过 All-Reduce 或 All-Gather 恢复），每个 rank 可以独立决定自己 token 的 top-k 专家。

然后通过 All-to-All 将 token 发送到对应专家所在的 EP rank。

注意：这要求所有 rank 上的 gating network 权重是一致的（EP 维度上 gating 参数是复制的，不是切分的）。Gating 网络本身很小（h × E），复制的开销可忽略。

### Q12: 对比 V2/V3/V4 在 EP 策略上的演进。

| 维度 | V2 | V3 | V4 |
|------|----|----|-----|
| 专家数 | 160 | 256 | 384/256 |
| 激活专家/token | 6 | 8 | 6 |
| 通信重叠 | 无 | Dispatch+Compute 粗粒度重叠 | Wave 级细粒度重叠 |
| 融合程度 | 分离 kernel | 分离 kernel + 调度重叠 | 单一 Mega-Kernel |
| Dispatch 精度 | BF16 | BF16 | FP8 |
| 负载均衡 | Aux Loss | Aux-loss-free (bias) | Aux-loss-free + Sequence Balance |
| 容量约束 | 有 | 有 | 无 |
| 路由激活 | Sigmoid | Sigmoid | Sqrt(Softplus) |
| 混合 ZeRO | 无 | 无 | 背包算法 + 两阶段规约 |
| 确定性 | 不保证 | 部分保证 | Batch 不变 + 确定性 |

### Q13: EP 和 TP 如何协作？什么情况下选 EP 优先，什么情况选 TP 优先？

EP 和 TP 是正交的并行维度，可以同时使用：

- EP 切的是「专家维度」——不同专家在不同 GPU
- TP 切的是「层内矩阵维度」——同一个专家的矩阵乘被切分

选择优先级：
- 专家数量多 → 优先 EP（天然并行度）
- 单个专家矩阵大 → 优先 TP
- 通信瓶颈大 → 优先 EP（EP 的通信量是 O(token·h)，TP 是 O(h²)）
- 小 batch → EP 通信占比高，TP 更合适

DeepSeek V4 实际用了 EP（切专家）+ TP（切 Attention 的 QKV 和 MoE 的 large GEMM）+ DP + PP 的四维混合。

### Q14: 讲一下 MegaMoE 开源实现中包含的关键优化。

MegaMoE 是 DeepGEMM 的一部分（GitHub PR #304），关键优化包括：

1. **Persistent Kernel**：单个 kernel 持续运行，处理所有 wave，消除 kernel relaunch 开销
2. **SM 级别的生产者-消费者流水线**：不同 SM 承担不同 wave 的通信/计算角色
3. **TileLang 代码生成**：用 DSL 自动生成融合 kernel，平衡开发效率和运行效率
4. **Host Codegen**：消除 Python 运行时检查开销（从数十/数百 μs → <1μs per call）
5. **混合精度计算**：FP8 GEMM + BF16 后处理 + FP32 累加

### Q15: 如果让你为一个全新的 MoE 架构设计 EP 方案，你会考虑哪些因素？

1. **专家数量与每 token 激活数** → 决定 EP 并行度上限和通信量
2. **单专家参数量** → 决定是否需要 TP 与之配合
3. **互联拓扑** → NVLink 带宽 vs 网络带宽，决定 wave 粒度
4. **Batch 大小预期** → 训练/推理的 batch 分布，影响波次调度策略
5. **负载均衡策略** → 是否允许 token dropping，capacity factor 设计
6. **精度需求** → Dispatch/Combine 可以用多低的精度
7. **确定性需求** → 是否需要 batch 不变，如何设计 atomics 和 reduction
8. **硬件特性** → GPU/NPU 的通信原语、SM 架构、功耗限制
9. **容错需求** → 是否需要可抢占、可恢复的通信方案
10. **与训练框架的整合** → autograd、checkpoint、ZeRO 的兼容性

---

## 十一、一周复习计划

| 天 | 主题 | 核心内容 |
|---|------|---------|
| **Day 1** | **EP 并行（本文档）** | 数学推导、Wave 调度、Mega-Kernel、负载均衡、混合并行 |
| **Day 2** | **TP + PP + DualPipe** | 张量切分的数学，1F1B vs DualPipe，通信量与计算量分析 |
| **Day 3** | **ZeRO + 混合并行** | ZeRO-1/2/3 的通信量、FSDP、与 EP 的交互 |
| **Day 4** | **MoE 架构深度** | GShard → Switch Transformer → DeepSeekMoE → Hash MoE |
| **Day 5** | **Attention + KV Cache** | MLA 数学推导、GQA/MQA、Flash Attention、CSA/HCA |
| **Day 6** | **训练稳定性 + 精度** | FP8/FP4 混合精度、Loss Spike 机理、Muon 优化器 |
| **Day 7** | **综合模拟面试** | 串联所有知识点，练习从「推导公式」角度回答 |

---

## 十二、必背公式与关键数字

### 必背公式

| 公式 | 含义 |
|------|------|
| V_comp = 6·h·d_ff | 每 token-expert pair 的 SwiGLU 计算量 |
| V_comm = 3h (FP8+B16) | 每 token-expert pair 的通信量 |
| C/B ≤ 2d_ff | 通信可隐藏条件 |
| C/B ≤ 6144 (V4-Pro) | 代入 d_ff=3072 |
| 每 GB/s 带宽隐藏 6.1 TFLOP/s | 工程经验值 |

### 必背数字

| 数字 | 含义 |
|------|------|
| 1.50-1.73× | MegaMoE 通用推理加速 |
| 1.96× | RL Rollout 场景加速 |
| 1.42× | Comet 理论加速 |
| 1.92× | V4 Wave 调度理论加速 |
| 0.001 | Aux-loss-free bias 更新速度 |
| 0.0001 | Sequence balance loss 权重 |
| 6.7% | mHC 引入的额外 wall-time 开销 |
| 20% | Anticipatory Routing 额外 wall-time 开销 |
| < 1μs | Host Codegen 后的 kernel launch 开销 |
| 99.7% | FP4 indexer 量化后的 KV 召回率 |

---

## 十三、推荐阅读

1. **DeepSeek-V4 Technical Report** Section 3.1 — 细粒度 EP 原文（本文档主要来源）
2. **DeepSeek-V3 Technical Report** (arXiv:2412.19437) — DualPipe, FP8 training, auxiliary-loss-free
3. **DeepSeek-V2 Technical Report** (arXiv:2405.04434) — DeepSeekMoE architecture
4. **Comet** (Zhang et al., 2025b) — Fine-grained computation-communication overlapping for MoE
5. **DeepGEMM** (github.com/deepseek-ai/DeepGEMM) — MegaMoE 开源实现
6. **GShard** (arXiv:2006.16668) — MoE 大规模并行的基础工作
7. **Switch Transformer** (arXiv:2101.03961) — Capacity Factor, token dropping
8. **MegaBlocks** (arXiv:2211.15841) — Block-sparse MoE, 负载均衡
