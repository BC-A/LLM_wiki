# Expert Parallelism (EP)

> **一句话概述**: 将 MoE 层的不同专家分布到不同 GPU 上，通过集合通信在专家间路由 token，突破单卡无法容纳全部专家的显存限制。
> **核心问题**: 如何在「专家分布的通信开销」与「单卡显存/计算限制」之间取得平衡？

---

## 一、为什么需要 EP

### 1.1 问题：专家参数太多，单卡放不下

MoE（Mixture of Experts）用多个「专家」子网络替代普通 FFN，每个 token 只激活其中 top-k 个。这带来巨大的参数膨胀。以 DeepSeek-V3 为例：

- 256 个路由专家 + 1 个共享专家
- 每个专家包含 SwiGLU 的 gate/up/down 三个矩阵
- 单个专家参数量 ≈ 3 × h × d_ff ≈ 3 × 4096 × 2048 ≈ 25M
- 一层 MoE 的总参数量 ≈ 256 × 25M ≈ 6.4B

单卡 H100（80GB）装不下，更不用说训练时还有优化器状态和激活值。因此必须将不同专家分布到不同 GPU 上——这就是 **Expert Parallelism (EP)**。

> EP 解决的是「专家参数太多」的问题，与 TP 解决「单个矩阵太大」是正交的。

### 1.2 MoE 层的前向流程

在进入 EP 的具体机制之前，先明确 MoE 层在整个 Transformer 中的位置和计算步骤：

```
Input (hidden_states) [B, S, h]
  │
  ├─→ Router (Gating Network)
  │     每个 rank 独立对自己持有的 token 做路由决策
  │     Router 参数在所有 EP rank 上复制（不切分），因为参数量极小（h × E ≈ 1M）
  │
  ├─→ Token Dispatch
  │     将 token 按路由结果发送到对应专家所在的 rank
  │
  ├─→ Expert Compute
  │     每个 rank 对自己持有的本地专家执行 SwiGLU GEMM
  │
  └─→ Token Combine
        将专家输出发送回 token 原始的 rank，按 routing weights 加权求和
```

Router 为什么在所有 rank 上复制而不是切分？参数量极小（4096 × 256 ≈ 1M），复制开销可以忽略。如果切分，每个 rank 只知道部分专家的路由分数，需要额外通信来协调 → 得不偿失。

---

## 二、EP 的通信方式：从简单到高效

知道了 MoE 层需要把 token 分发到对应专家的 GPU，怎么实现？在讨论具体方案之前，先搞清楚三个核心数字——参数量、计算量、通信量。后续所有的设计抉择都源于这三者之间的比例关系。

### 2.1 参数量、计算量、通信量

先看清数字——后面所有 EP 的设计决策都源于这三者之间的比例。以 **DeepSeek-V4-Pro** 的一个 MoE 层为例：

```
h = 7168, d_ff = 3072, num_experts = 384, topk = 6
FP8 dispatch + BF16 combine（V4 混合精度方案）
```

#### 参数量

每个专家是标准的 SwiGLU MLP，三个权重矩阵：

```
W_gate: [h, d_ff] = [7168, 3072] → 22M 参数
W_up:   [h, d_ff] = [7168, 3072] → 22M 参数
W_down: [d_ff, h] = [3072, 7168] → 22M 参数
──────────────────────────────────────
单个专家 ≈ 66M 参数
一层 MoE ≈ 384 × 66M ≈ 25.3B 参数
```

对比 Attention 层（QKV + output 投影 ≈ 4·h² ≈ 205M 参数），MoE 层的参数量是它的 ~120 倍。单卡 H100 80GB 装不下——即使只存 BF16 权重也需要 25.3B × 2 bytes ≈ 50GB，还没算优化器状态和激活值。这就是 EP 存在的根本原因。

#### 计算量

每个 token 只激活 topk=6 个专家，每次激活走完整的 SwiGLU 路径（gate/up 矩阵乘 + 激活 + down 矩阵乘）：

```
每个 token-expert pair:
  3 个 GEMM × 2·h·d_ff = 6·h·d_ff = 6 × 7168 × 3072 ≈ 132M FLOPs

每个 token（topk=6）:
  6 × 132M ≈ 792M FLOPs
```

对比 Attention 层每 token 约 `4·h² = 205M` FLOPs，MoE 层的计算量约是它的 4 倍。但因为只有 6/384 ≈ 1.6% 的专家被激活，这些计算是**稀疏的**——每个 token 只触及了模型的一小部分。

#### 通信量

每个 token 需要在 Dispatch 时把 hidden state 发给 topk 个专家所在的 rank，Combine 时把结果收回来做加权聚合：

```
Dispatch（FP8，1 byte/element）:
  per token-expert: h × 1 = 7168 bytes ≈ 7KB
  per token:        topk × 7KB = 6 × 7KB = 42KB

Combine（BF16，2 bytes/element，精度敏感保留高精度）:
  per token-expert: h × 2 = 14336 bytes ≈ 14KB
  per token:        topk × 14KB = 6 × 14KB = 84KB
  ─────────────────────────────────────────
  per token 总通信量: 42KB + 84KB = 126KB
```

**为什么 Dispatch 用 FP8 而 Combine 用 BF16？Dispatch 发的 token hidden state 是 GEMM 的输入，对精度相对宽容；Combine 发回的专家输出要做加权求和并加到 residual stream 上，对精度敏感。** 一省一保，这是 V4 混合精度通信的核心权衡。

#### 计算-通信比：决定 EP 是瓶颈还是可隐藏

##### 1. 模型侧：单token-expert配对的计算/通信配比
先算出一组token分配给单个专家时，对应的总计算量与总传输数据量：
```
# 单次配对浮点运算总量
V_comp = 6·h·d_ff = 6 × 7168 × 3072 ≈ 132M FLOPs

# 单次配对双向通信总字节数
V_comm = h(FP8权重) + 2h(BF16激活梯度) = 3h = 21504 bytes
```
两者做比值，消去隐藏参数h，得到模型固有的**计算通信负荷比**：
```
Load_Ratio = V_comp / V_comm = (6·h·d_ff) / 3h = 2·d_ff
代入d_ff=3072：Load_Ratio = 2 × 3072 = 6144 FLOP/Byte
```
这个比值含义：模型业务本身约束——**每1字节跨卡传输数据，配套必须完成6144次浮点运算**。
这个数值代表通信背后绑定的计算工作量，数值越大，同等通信量下计算任务越繁重。

##### 2. 硬件侧：算力带宽供给能力比
V4专家层GEMM采用FP8精度计算，硬件算力`C`取FP8峰值算力；
All-to-All为双向收发并行传输，带宽`B`取双向聚合总带宽。
定义硬件**算力供给比** `Supply_Ratio = C/B`：代表硬件每1Byte/s带宽，一秒内最多能同步支撑多少FLOP/s的计算。

| 硬件链路 | C (FP8峰值算力) | B (双向聚合带宽) | Supply_Ratio=C/B (FLOP/Byte) |
|--------|----------------|-----------------|-----------------------------|
| H100 NVSwitch 节点内NVLink | ~2 PFLOP/s | ~900 GB/s | ~2200 |
| InfiniBand NDR 跨节点单网卡 | ~2 PFLOP/s | ~50 GB/s | ~40000 |

##### 掩盖判定核心逻辑
想要**GEMM计算完全掩盖通信延迟**，需要满足：硬件一秒每字节带宽能供给的算力 ≤ 模型每字节通信绑定的计算量
也就是判定公式：
$$\boldsymbol{\frac{C}{B} \le 2\cdot d_{ff}}$$
通俗解释：
硬件算力的“供给速度”追不上模型计算的“任务负荷”，传输数据的耗时足够GPU完整跑完配套计算，GPU不会空闲等待数据。
**数字越大 = GPU 计算速度相对传输速度快得越多，越容易算完等通信；
数字越小 = 传输速度跟得上计算速度，不容易卡通信。**

##### 3. V4-Pro（d_ff=3072）瓶颈判定
模型负荷阈值：$2d_{ff}=6144$
- 节点内NVLink：2200 ≤ 6144 ✅
  硬件供给算力远小于模型负荷，通信耗时被GEMM完全隐藏，节点内EP通信无性能瓶颈
- 跨节点IB：40000 > 6144 ❌
  硬件算力供给过剩、带宽速度跟不上算力，GPU快速算完一批数据后，必须空转等待All-to-All传输，跨节点通信成为硬性瓶颈

这就是V3设计DualPipe、V4设计Wave流水线调度的底层根源：跨节点IB带宽算力匹配度极差，只能依靠多层流水线，异步重叠All-to-All通信与GEMM计算，分摊屏蔽通信延迟。

##### 4. V4-Flash（d_ff=2048）对比验证
更小的中间维度，模型计算负荷同步降低：
负荷阈值：$2d_{ff}=2\times2048=4096$
- NVLink：2200 ≤ 4096 ✅ 依旧可以完全隐藏通信
- IB：40000 ≫ 4096 ❌ 瓶颈问题反而加剧

规律总结：
$d_{ff}$ 数值越大 → 同等通信量绑定的计算量越重 → 计算耗时更长、更容易盖住传输等待；
大d_ff宽FFN结构的模型，天然更适配多机跨节点EP并行训练。

##### 补充延伸
EP并行里的All-to-All、DualPipe、Wave调度三类通信方案，核心取舍逻辑一致：在不同专家并行规模、不同链路带宽约束下，权衡**额外冗余计算开销**和**全局通信总传输量**，找到整体吞吐最优平衡点。

下面三种通信方式，本质上是在不同的 EP 规模和 bandwidth 约束下，如何在「冗余计算」和「通信量」之间做权衡。

### 2.2 方式一：AllGather（最简单，但浪费计算）

最直观的思路：既然每个 rank 都不知道其他 rank 的 token 要去哪个专家，那就**让所有 rank 看到所有 token**。

```
AllGather 方式:

  Rank 0                    Rank 1                    Rank 2
  [token_a, token_b]       [token_c, token_d]       [token_e, token_f]
       │                        │                        │
       └──────── AllGather ─────┼────────────────────────┘
                                │
              每个 rank 都有 [token_a, b, c, d, e, f]
                                │
              每个 rank 只算自己负责的专家
              Rank 0: Expert 0..15 处理 token_a, c（只取路由到这些专家的）
              Rank 1: Expert 16..31 处理 token_b, d, e
                                │
       ┌──────── ReduceScatter ─┴────────────────────────┐
       │                        │                        │
  Rank 0                    Rank 1                    Rank 2
  [expert_out_a,c]         [expert_out_b,d,e]        [expert_out_f]
```

**优点**：实现简单，用标准的 AllGather + ReduceScatter 就能完成。

**致命问题**：每个 rank 收到的都是**所有 token**，只是每个 rank 只计算其中路由到本地专家的那部分。当 EP 较大时（如 ep_size=64），每个 rank 只负责 1/64 的专家，但处理了 100% 的 token → 99% 的计算是无效的。

**适用场景**：EP 很小（≤4）时还能接受，通信简单、kernel 成熟。

在 Megatron-LM 中这就是 `moe_token_dispatcher_type="allgather"`（也是默认值）。

### 2.3 方式二：All-to-All（按需路由，EP 的标配）

改进思路：**只把 token 发给真正需要它的 rank**。

```
All-to-All 方式:

  Rank 0                    Rank 1                    Rank 2
  [token_a→Expert 5]       [token_c→Expert 20]      [token_e→Expert 0]
  [token_b→Expert 18]      [token_d→Expert 35]      [token_f→Expert 18]
       │                        │                        │
       │  Expert 0..15 在 Rank 0, Expert 16..31 在 Rank 1, Expert 32..47 在 Rank 2
       │
       └──────── All-to-All ────┼────────────────────────┘
                                │
  Rank 0 收到: token_a(E5) + token_e(E0)          ← 只收到路由到本地专家的
  Rank 1 收到: token_b(E18) + token_c(E20) + token_f(E18)
  Rank 2 收到: token_d(E35)
                                │
              每个 rank 计算本地专家
              Rank 0: Expert 0 处理 token_e, Expert 5 处理 token_a
              Rank 1: Expert 18 处理 token_b+token_f, Expert 20 处理 token_c
                                │
       ┌──────── All-to-All ────┴────────────────────────┐
       │                        │                        │
  Rank 0                    Rank 1                    Rank 2
  [结果聚合回原顺序]        [结果聚合回原顺序]        [结果聚合回原顺序]
```

「只发给需要的 rank」意味着发送方和接收方之间的数据量是**不均匀的**——这正是 All-to-All 的语义：每个 rank 向不同 rank 发送**不同数量**的数据。

#### All-to-All vs All-Reduce：为什么不能用 All-Reduce？

这个问题经常被问到。两种集合通信的语义完全不同：

| 维度 | All-to-All | All-Reduce |
| ---- | ---------- | ---------- |
| **做什么** | 数据重新分布（redistribution） | 数据聚合（aggregation） |
| **每个 rank 得到** | 来自各 rank 的**不同数据片段** | 所有 rank 数据的**求和/平均结果** |
| **数据量变化** | 总量不变，重新分配 | 归约后数据量减少 |
| **典型场景** | EP dispatch/combine、SP attention | DP 梯度同步、TP 矩阵乘结果聚合 |

EP 用 All-Reduce 会发生什么？所有 rank 的 token hidden states 会被求和 → 不同 token 的数据被混在一起 → 完全破坏 token 独立性。EP 需要的是「把我的数据发给正确的人，不是求和」。

#### 两步 All-to-All：Dispatch + Combine

EP 需要两次 All-to-All：

- **Dispatch**：token → expert rank（把每个 token 的 hidden state 发到其 top-k 专家所在的 GPU）
- **Combine**：expert rank → token rank（把专家输出发回 token 原来所在的 GPU，做加权聚合）

**通信量**（per token-expert pair）：

| 阶段 | BF16 | FP8 |
|------|------|-----|
| Dispatch | 2h 字节 | h 字节 |
| Combine | 2h 字节 | h 字节 |
| **合计** | **4h 字节** | **2h 字节** |

通信能否被 GEMM 掩盖，取决于硬件的 C/B 是否满足 §2.1 推导的 `C/B ≤ 2·d_ff`。结论是：节点内 NVLink 下 EP 通信不是瓶颈，跨节点 IB 是——所以 DualPipe（V3）和 Wave 调度（V4）的核心目的都是**跨节点时把 All-to-All 和 GEMM 重叠起来**，隐藏 IB 的带宽短板。

节点内 NVLink 下通信不是瓶颈；跨节点 IB 必须靠 DualPipe 或 Wave 调度来重叠。

### 2.4 方式三：Flex / Fused（消除中间搬运）

All-to-All 解决了计算冗余，但它自身还有一个隐藏开销：**permute（重排）**。

为什么需要重排？来看 Dispatch 前的实际数据布局：

```
Rank 0 有 4 个 token，路由结果:
  A → Expert 5  (在 Rank 1)
  B → Expert 20 (在 Rank 1)
  C → Expert 3  (在 Rank 0)
  D → Expert 35 (在 Rank 2)

当前 tensor 顺序: [A, B, C, D]           ← 按 token 索引排列
All-to-All 需要:  [C, A, B, D]           ← 按目标 rank 分组
                  └Rank0┘ └─Rank1─┘ └Rank2┘

所以需要一次 permute，把数据从「token 顺序」重排为「目标 rank 顺序」。
```

收到数据后也一样——接收方拿到的 tensor 是按**源 rank** 分组的，但本地专家需要按 **expert ID** 组织 token：

```
Rank 1 收到:  [A(from Rank0), B(from Rank0), E(from Rank2)]
               └── 按源 rank 排列 ──┘

但本地 Expert 20 需要 [B, E]，Expert 18 需要 [A]
→ 需要再 permute 一次
```

**传统做法（三个独立 kernel）**：

```
permute → All-to-All → permute
  ↑          ↑           ↑
  Kernel1    Kernel2     Kernel3
  写HBM      写HBM       写HBM
```

每个 kernel 的结果都写回 HBM，下一个 kernel 再读出来。两次 permute 的数据只是中间状态，对用户毫无意义，却各占了一次 HBM 读写。

**Flex 的做法（融合单个 kernel）**：

```
┌──────────── 单个 kernel ────────────┐
│                                      │
│  HBM → 读入 → [shared mem 重排]      │
│                → [shared mem A2A]    │
│                → [shared mem 重排]   │
│                → 写出 → HBM          │
│                                      │
│  中间数据始终在 on-chip shared mem   │
└──────────────────────────────────────┘
```

融合后，permute 的中间结果不写 HBM，在 shared memory 里流转。好处有两个：
1. 省了两次 HBM 读写（HBM 带宽是 GPU 最稀缺的资源之一）
2. 省了两次 kernel launch（每次 launch 有 ~5-20μs 的 CPU→GPU 调度延迟）

Megatron-LM 中 `moe_token_dispatcher_type="flex"` 对应这种方式，底层由 **DeepEP** 或 **HybridEP** 提供融合 kernel。

> DeepEP 的实现细节见下文第五节。

### 三种方式的选择建议

| 方式 | Megatron 参数 | 何时用 |
|------|-------------|--------|
| AllGather | `allgather` | EP ≤ 4，快速原型 |
| All-to-All | `alltoall` | EP ≥ 8，训练标配 |
| Flex (Fused) | `flex` | H100+，大规模训练，追求极致性能 |

> **关键认知**：方式一到方式三的演进，本质是逐步消除**不必要的数据搬运**——AllGather 搬运了不需要的 token，分离的 All-to-All 搬运了中间状态的 permute 结果，Flex 连中间结果都不写了。

---

## 三、Megatron-LM 中怎么配 EP

理解原理后，来看 Megatron-LM 中 EP 是怎么配置和运作的。

> 代码路径：[/home/jsy/LLM/training-frameworks/Megatron-LM](https://github.com/NVIDIA/Megatron-LM)

### 3.1 基本参数

EP 的配置分布在两个核心类中：

**`ModelParallelConfig`** 定义 EP 并行度：

```python
# megatron/core/model_parallel_config.py
expert_model_parallel_size: int = 1
"""将 MoE 专家分布到 expert model parallel group 的 GPU 上"""

expert_tensor_parallel_size: Optional[int] = None  # 默认 = tensor_model_parallel_size
"""专家层的张量并行大小，可以和 attention 层的 TP 不同"""
```

**CLI 完整示例**：

```bash
# 核心并行度
--tensor-model-parallel-size 2       # Attention/Embedding 的 TP
--expert-model-parallel-size 8       # EP 并行度
--expert-tensor-parallel-size 2      # 专家层的 TP（默认等于上面那个）
--pipeline-model-parallel-size 2     # PP

# MoE 架构
--num-experts 64                     # 专家总数（必须能被 ep_size 整除）
--moe-router-topk 8                  # 每 token 激活的专家数
--moe-ffn-hidden-size 2048           # 专家 FFN 隐藏层维度

# Token 分发方式
--moe-token-dispatcher-type alltoall # allgather / alltoall / flex
--moe-enable-deepep                  # 启用 DeepEP（flex 模式需要）
```

### 3.2 专家到 Rank 的分配

专家按**连续分块**分配到各 EP rank：

```python
# megatron/core/transformer/moe/moe_layer.py
ep_size = get_pg_size(self.ep_group)
ep_rank = get_pg_rank(self.ep_group)
self.num_local_experts = self.config.num_moe_experts // ep_size
local_expert_indices_offset = ep_rank * self.num_local_experts
self.local_expert_indices = [local_expert_indices_offset + i
                             for i in range(self.num_local_experts)]
```

例如 `num_experts=64, ep_size=4` → Rank 0 持有专家 0-15，Rank 1 持有 16-31，依此类推。

**约束**：
- `ep_size ≤ num_experts`（每 rank 至少 1 个专家）
- `num_experts % ep_size == 0`（Megatron-LM 硬性要求）
- 实践中每个 rank 通常持有 **1-4 个专家**——多了计算太粗（不利于负载均衡），少了通信占比太高

---

## 四、EP 如何与其他并行维度组合

真实训练中不可能只用 EP 一种并行。这一节串起来讲 EP 和 TP / DP / CP / PP 组合时的**行为、约束和性能考量**。每种组合自然会引出它需要的进程组。

### 4.0 从 Attention 进程组到 MoE 进程组：EP 切在哪

Megatron-LM 内部有两套进程组，分别服务于 Attention 层和 MoE 层。两套进程组**用同一批物理 GPU**（同一个 `world_size`），但分组方式不同。理解两层之间的映射关系，是掌握混合并行的起点。

#### Attention 进程组

这是不含 EP 的进程组，Megatron 中由 `decoder_rank_generator` 建：

```
world_size = TP × PP × CP × DP_attn
```

四个维度都是用户通过 CLI 显式指定的（TP、PP、CP），`DP_attn` 由 `world_size / (TP × PP × CP)` 自动算出。

#### MoE 进程组

引入了 EP，由 `expert_decoder_rank_generator` 建：

```
world_size = expert_TP × PP × EP × DP_expert
```

这里 CP 固定为 1（MoE 进程组中没有 CP 维度）。

| 维度 | 来源 | 说明 |
|------|------|------|
| `PP` | 与 Attention 进程组相同 | 流水线 stage 不变，同一 stage 内的 Attention 和 MoE 层必须同组 |
| `expert_TP` | CLI `--expert-tensor-parallel-size`，**默认等于 TP** | 专家层的张量并行度。如果不显式指定，就取 `TP` 的值。可以设得比 TP 小（如 TP=4 但 expert_TP=1，专家层不做张量切分） |
| `EP` | CLI `--expert-model-parallel-size` | 用户显式指定，必须满足 `num_experts % EP == 0` |
| `DP_expert` | 自动算出 `world_size / (expert_TP × PP × EP)` | 专家层的数据并行度，不单独配置 |

#### EP 到底从哪「切」出来的

两套进程组用同一批 GPU，所以：

```
TP × PP × CP × DP_attn  =  expert_TP × PP × EP × DP_expert
```

消掉两边的 `PP`（相同），并假设 `expert_TP = TP`（最常见的默认情况）：

```
CP × DP_attn  =  EP × DP_expert
```

**这就是 EP 的来源：EP 是从 `CP × DP_attn` 这个「池子」里分出来的。** 当 `CP = 1` 时，退化为 `DP_attn = EP × DP_expert`——EP 直接切 `DP_attn`。

#### 用一个具体的并行配置走一遍

**配置**: `TP=2, PP=1, CP=2, EP=4`，`expert_TP` 不设（默认等于 TP），32 张 GPU。

```
Attention 进程组:  32 = 2(TP) × 1(PP) × 2(CP) × 8(DP_attn)
MoE 进程组:       32 = 2(expert_TP) × 1(PP) × 4(EP) × 4(DP_expert)

验证: CP × DP_attn = 2 × 8 = 16,  EP × DP_expert = 4 × 4 = 16  ✓
```

在这 32 张 GPU 上，**同一批 GPU 在 Attention 层和 MoE 层被编入不同的进程组**：

```
32 GPU, TP=2, CP=2, EP=4

Attention 层 (TP×CP=4 GPU/组, DP_attn=8 组):
  [0,1]  [2,3]  │  [4,5]  [6,7]  │  [8,9] [10,11] │ [12,13] [14,15] │ ...
   TP0    TP1    │   TP2    TP3    │   TP4    TP5   │   TP6    TP7    │
  └── CP0 ──┘   │  └── CP1 ──┘    │  └── CP2 ──┘  │  └── CP3 ──┘   │
  └───── DP0 ───┘ └───── DP1 ────┘ └───── DP2 ───┘ └───── DP3 ───┘  ...

MoE 层 (TP×EP=8 GPU/组, DP_expert=4 组):
  [0,1]  [2,3]  [4,5]  [6,7]  │  [8,9] [10,11] [12,13] [14,15]  │ ...
   TP0    TP1    TP2    TP3    │   TP4    TP5     TP6     TP7    │
  └────────── EP0 ──────────┘  │  └──────────── EP1 ──────────┘ │
  └────────── DP0 ──────────┘  ┘  └──────────── DP1 ──────────┘ ┘
```

**关键变化**：进入 MoE 层时，CP 消失（MoE 进程组中 CP=1），替换为 EP=4。Attention 的 `CP × DP_attn = 16` 个「数据+序列分片」被重新组织为 MoE 的 `EP × DP_expert = 16` 个「专家+数据分片」。EP 吃掉的就是原来 CP 和 DP_attn 的份额。

#### Rank Order：为什么 TP 必须排在最前面

上面说两套进程组用同一批 GPU 但分组不同，具体怎么「分组」是由 `order` 参数控制的。Megatron 默认 `order = "tp-cp-ep-dp-pp"`，这个顺序直接决定了**哪些 GPU 物理上更近，通信更快**。

RankGenerator 的核心公式（[`generate_masked_orthogonal_rank_groups`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/parallel_state.py#L251)）：

```
global_rank = 最内层 + 次内层 × size_最内层 + 次外层 × size_最内层 × size_次内层 + ...
```

以 `order = "tp-cp-ep-dp-pp"`、`size = [2, 1, 1, 4, 2]` 为例（即 TP=2, CP=1, EP=1, DP=4, PP=2，Attention 网格）：

```
global_rank = tp_rank + dp*tp + pp*tp*dp
```

这意味着 **TP 的 stride 是 1，DP 的 stride 是 tp_size，PP 的 stride 是 tp_size × dp_size**。结果就是：

```
TP 组：GPU 0,1 在一起 → 相邻 GPU，NVLink 直连 → 带宽 ~900 GB/s
DP 组：GPU 0,2,4,6 在一起 → 跨节点（如果 DP > 每节点 GPU 数）→ 走 IB → 带宽 ~50 GB/s
PP 组：GPU 0,8 在一起 → 跨节点 → 走 IB
```

**排在前面的维度，通信落在同一 NVLink 域内**——这就是为什么 TP、CP 必须排前面（带宽需求极高），DP、PP 排后面（对带宽不敏感，跨几跳也行）。

对于 MoE 层，Expert RankGenerator 的 order 是 `"tp-ep-dp-pp"`（CP=1，EP 顶上 CP 的位置）：

```
Attention:  tp (最内) → cp → dp → pp (最外)
MoE:        tp (最内) → ep → dp → pp (最外)
```

**EP 排在了 CP 的位置，也就是第二内层**。这意味着当 EP ≤ 每节点 GPU 数时，EP 的 All-to-All 完全在节点内走 NVLink，延迟极低；一旦 EP 超过该值，EP 组跨越节点，部分通信走 IB，延迟上升。这和 TP 不能超过每节点 GPU 数是同一个约束来源。

#### 几个由公式导出的工程约束

- **TP ≤ 单节点 GPU 数（通常 8）**：TP 通信是 All-Reduce 走 NVLink。如果 TP 组跨越节点边界，部分通信走 IB → 带宽从 ~900 GB/s 掉到 ~50 GB/s → 极慢。
- **EP 的上限受 `CP × DP_attn` 约束**：`EP = (CP × DP_attn) / DP_expert`，`DP_expert` 至少为 1 → `EP ≤ CP × DP_attn`。GPU 总数固定的情况下，EP 不能无限增大。
- **EP 增大 → DP_expert 减小 → expert 层 micro-batch 变小 → GEMM 效率下降**。这跟 DP 减小导致计算效率下降是同一个道理。
- **CP 和 EP 没有直接冲突**，但都会挤占 `DP_expert` 的「预算」。

### 4.1 EP + TP：权重到底怎么切

EP 和 TP 经常一起开，但它们切的是完全不同的东西。

#### EP 不切权重，只分专家

EP 做的事很简单：把 E 个专家按索引连续分块，分给不同 rank。权重本身是完整的——EP Group 0 的 rank 持有 Expert 0..E/EP-1 的**完整**矩阵，EP Group 1 持有下一批，依此类推。EP 本身不做任何矩阵切分。

#### TP 切权重，跟 Attention 层一样

TP 切的是单个专家内部的权重矩阵。SwiGLU 专家有三个矩阵，以 `h = 4096, d_ff = 1536` 为例：

```
Expert k 的完整权重（无 TP）:
  W_gate: [h, d_ff] = [4096, 1536]
  W_up:   [h, d_ff] = [4096, 1536]
  W_down: [d_ff, h] = [1536, 4096]
```

开启 TP=2 后，切分方式和 Megatron 的 Attention 层完全一致——**gate/up 列切，down 行切**：

```
TP 列切（W_gate, W_up）—— 切 dim 1，即 d_ff:
  Rank 0: [h, d_ff/TP] = [4096, 768]    ← 列前半
  Rank 1: [h, d_ff/TP] = [4096, 768]    ← 列后半

TP 行切（W_down）—— 切 dim 0，即 d_ff:
  Rank 0: [d_ff/TP, h] = [768, 4096]    ← 行前半
  Rank 1: [d_ff/TP, h] = [768, 4096]    ← 行后半
```

为什么 gate/up 列切、down 行切？因为 SwiGLU 的计算是 `down( gate(x) ⊙ up(x) )`。gate 和 up 的输出在 d_ff 维度上做 element-wise 乘，所以它们在 d_ff 维度上列切之后，每个 TP rank 独立做激活，不需要通信。Down 的输入是激活后的 d_ff 维度 → 行切匹配。

#### EP + TP 叠加时，每个 rank 上是什么

设 `EP=2, TP=2, num_experts=4`。EP 先把专家分两半，TP 再把每半里的专家矩阵切分：

```
GPU 0 (EP rank 0, TP rank 0):  持有 Expert 0,1 的以下部分:
  W_gate[0][4096, 768]  W_up[0][4096, 768]  W_down[0][768, 4096]
  W_gate[1][4096, 768]  W_up[1][4096, 768]  W_down[1][768, 4096]

GPU 1 (EP rank 0, TP rank 1):  持有 Expert 0,1 的另一半:
  W_gate[0][4096, 768]  W_up[0][4096, 768]  W_down[0][768, 4096]
  W_gate[1][4096, 768]  W_up[1][4096, 768]  W_down[1][768, 4096]

GPU 2 (EP rank 1, TP rank 0):  持有 Expert 2,3:
  （同上结构，只是 Expert 索引不同）

GPU 3 (EP rank 1, TP rank 1):  持有 Expert 2,3 的另一半:
```

**EP 和 TP 是嵌套关系**：先 EP 决定哪个 rank 管哪些专家，再 TP 决定每个专家的矩阵怎么切。不同 EP group 之间没有任何权重共享。

#### 通信流程

一个 token 路由到 Expert 0（在 EP Group 0）的完整路径：

```
Token 在 GPU 2, hidden = [1, 4096]

Dispatch:
  → All-to-All (EP):     发到 EP Group 0 (GPU 0,1)
  → AllGather (TP):       GPU 0,1 各自拿到对方收到的 token（需要处理同一批 token）
  → GEMM:                 GPU 0: [1,4096]×W_gate[0][4096,768] → [1,768] partial
                          GPU 1: [1,4096]×W_gate[0][4096,768] → [1,768] partial

Combine:
  → ReduceScatter (TP):   两个 partial 求和 → 完整 [1,4096]
  → All-to-All (EP):      发回 GPU 2
```

EP 通信让 token 到达正确的专家组，TP 通信让同一专家组内的不同 rank 协力完成 GEMM。

#### `expert_tp_size` 可以 ≠ `tp_size`

Megatron-LM 允许专家层使用不同于 Attention 层的 TP 大小（默认相等）。例如 DeepSeek-V3：Attention 用 TP=4，但 MoE 层有 256 个专家、EP=64，每 rank 只持有 256/64 = 4 个专家，单个专家矩阵不大 → `expert_tp_size=1`，专家层不做 TP，只有 EP 通信。

### 4.2 EP + DP：Expert Data Parallel

当 EP 开了之后，**同一套专家参数可能出现在多个 DP rank 上**（除非 `EP == num_experts` 即每个 rank 只 1 个专家）。

这些持有相同专家的 DP rank 处理的是**不同的 micro-batch 数据**，训练后各自产生一份该专家的梯度。这些梯度需要同步——这就是 Expert DP Group 的作用：

```python
# megatron/core/parallel_state.py
# _EXPERT_DATA_PARALLEL_GROUP:
# 包含持有相同专家的不同 DP rank
# 在这些 rank 之间对专家梯度做 All-Reduce
```

**重要特殊情形**：当 `EP == num_experts` 时（DeepSeek-V3 的配置），每个 rank 只有 1 个专家 → 不同 DP rank 持有的专家完全不同 → Expert DP Group 退化为单 rank 组 → **不需要 Expert DP 的梯度同步**。这正是 V3 可以省掉大量 DP 通信的原因之一。

### 4.3 EP + CP：为什么不在同一个进程组里

先说结论：**EP 和 CP 可以在同一个训练中同时使用，只是不在同一套进程组里。** 这不是什么互斥 bug，而是两种并行各自服务于完全不同的目的。

#### CP 为什么存在

CP 切的是序列长度。Attention 层需要做 cross-token 计算——每个 token 要 attend 到序列中所有其他 token。当序列太长（128K+）时，单卡放不下全序列的激活 → 用 CP 把序列切成多段，通过 Ring / All-to-All 交换 KV，让每个 GPU 仍能计算完整的 attention。

#### CP 为什么在 MoE 层没有意义

MoE 的 FFN 是 **per-token 独立计算**——每个 token 只跟自己路由到的专家交互，token 之间没有依赖。所以 MoE 层不需要「看到全序列」，自然也不需要 CP。

```
Attention:  token_i 需要看到 token_0 ... token_n → 需要 CP 来跨 GPU 共享 KV
MoE FFN:    token_i 只跟自己路由到的专家交互 → per-token 独立 → CP 无用
```

因此 Megatron 在 Expert 进程组中直接把 CP 固定为 1——不是不能用，是用不着。

#### Attention 的 CP 切出来的 token 怎么进入 MoE

Attention 输出后，每个 rank 上只有 1/CP 序列的 token。这些 token 进入 MoE 层时，就是该 rank 上的「本地 token」——MoE 层不关心这些 token 在完整序列中的位置，只关心每个 token 的 hidden state 要发给哪个专家。EP 的 dispatch 照常工作。

```
Attention 层 (CP=2):
  GPU 0: token_0 ... token_{n/2-1}   ← 序列前半
  GPU 1: token_{n/2} ... token_{n-1}  ← 序列后半

进入 MoE 层:
  GPU 0: 用自己的 token_{0..n/2-1} 走 Router → Dispatch → Expert Compute
  GPU 1: 用自己的 token_{n/2..n-1} 走 Router → Dispatch → Expert Compute
```

唯一的实际影响是：CP 越大，进入 MoE 的 per-rank token 数越少 → EP 通信延迟占比越高 → 小 batch 时尤其明显。这和 DP 小了 GEMM 效率降是同一个道理。

### 4.4 EP + PP：流水线中的 EP

PP 切层（不同 stage 持有不同层），EP 切专家（同一 MoE 层的不同专家）。两者没有直接冲突。

唯一需要注意的约束：Attention 层和 MoE 层在 PP 分组上必须一致（它们是交替堆叠的，同一 PP stage 内的 Attention 和 MoE 必须同组）。

另外 PP 的 bubble 问题会影响所有并行维度的利用率——EP 的 All-to-All 发生在 PP 的 forward/backward 阶段内，bubble 期间 GPU 空闲，EP 通信自然也停了。

### 4.5 通信量全景对比

| 并行 | 通信操作 | 带宽需求 | 跨节点可行？ | 为什么？ |
|------|---------|---------|------------|---------|
| DP | All-Reduce（梯度） | **低** | ✅ | 每 micro-batch 才做一次，梯度通信量 ∝ 参数量/DP |
| TP | All-Reduce / AllGather（激活） | **极高** | ❌ | 每层每 GEMM 都做，通信量 ∝ B×S×h，dense |
| CP | Ring P2P（传 K/V chunk） | **极高** | ❌ | 每层 CP-1 次 ring step，K/V 全量交换，通信量 ∝ CP×S×h，dense |
| PP | P2P Send/Recv | **低-中** | ✅ | 只传层边界激活，通信量 ∝ B×S×h |
| **EP** | **All-to-All** | **中-高** | **✅** | 稀疏通信——只有路由到的 topk 专家参与，通信量 ∝ topk × tokens × h |

**关键区分：dense vs sparse**

- **TP 和 CP 都是 dense 通信**，所有 token/所有激活都参与，所以带宽需求极高，通常不能超出 NVLink 域（单节点 8 卡）。
- **EP 是 sparse 通信**，每个 token 只发往 topk 个专家（通常 6-8），不是所有专家 rank 都参与。所以 EP 的总通信量远小于 dense 操作，跨节点走 IB 也能接受（但需要用 DualPipe 或 Wave 调度来隐藏延迟）。

---

## 五、DeepEP：高性能 EP 通信库

Megatron 原生的 All-to-All 用的是 NCCL 通用接口。DeepSeek 针对 MoE 场景的特点（数据量不均匀、需要拓扑感知）写了专用库 **DeepEP**，也是 Megatron-LM `flex` dispatcher 的后端。

> 代码路径：`code/deepseek-ref/DeepEP`

### 5.1 它做了什么

DeepEP 的核心优化是**感知物理拓扑的融合 All-to-All**：

```
通用 NCCL All-to-All:
  permute kernel → NCCL All-to-All → permute kernel
  问题: 3 个独立 launch，NCCL 不区分 NVLink vs RDMA

DeepEP:
  单个 kernel 内完成 permute + 数据传输
  内部区分:
    - NVLink 域内（scale-up）: 直接 GPU-to-GPU via symmetric memory + TMA
    - 跨节点（scale-out）: NCCL Gin put via RDMA
```

**混合模式**自动区分物理拓扑：

```
32 GPU, 8 节点 × 4 GPU/节点:
  scale-out_ranks = 8   ← RDMA 域（跨节点）
  scale-up_ranks  = 4   ← NVLink 域（节点内）
  
  Dispatch 时:
    节点内: TMA load → shared memory → TMA store (NVLink)
    跨节点: TMA load → shared memory → Gin put (RDMA)
```

### 5.2 关键技术点

- **对称内存**：所有 rank 的通信 buffer 映射到同一虚拟地址空间，GPU 可直接访问 peer GPU 显存
- **Pull 模式**：接收方主动从发送方拉数据（而不是发送方推 → 等通知），适合细粒度调度（V4 的 Wave 调度依赖这个）
- **FP8 原生支持**：dispatch 用 FP8 省一半带宽（V3/V4 的关键优化）
- **JIT 编译**：kernel 运行时编译，无需预装 CUDA toolchain

### 5.3 Megatron 中的集成

```python
# megatron/core/transformer/moe/fused_a2a.py
from deep_ep import Buffer

def get_buffer(group, hidden_bytes):
    """获取通信 buffer，自动计算 NVLink 和 RDMA 各需要多少空间"""
    # Buffer 内部根据 group.size() 自动判断拓扑
    buffer = Buffer(group, num_nvl_bytes, num_rdma_bytes)
    return buffer

# FusedDispatch / FusedCombine: 融合 permute + All-to-All
```

启用方式：`--moe-token-dispatcher-type flex --moe-enable-deepep`

---

## 六、关键数字与速查

### 并行度约束

| 约束 | 值 | 原因 |
|------|-----|------|
| TP ≤ 8 | 通常 | NVLink 域 ≤ 8 GPU，超出走 IB 带宽骤降 |
| EP ≤ num_experts | 硬性 | 每 rank 至少 1 个专家 |
| 推荐 experts/rank | 1-4 | 太少→通信占比高，太多→计算太粗不利于负载均衡 |
| EP 取 2 的幂 | 惯例 | 便于进程组划分和负载对称 |

### Token Dispatcher 对比

| Dispatcher | 底层通信 | 有计算冗余？ | 中间结果写 HBM？ | 适用 |
|-----------|---------|------------|----------------|------|
| `allgather` | AG + RS (TP×EP联合组) | 有（每个 rank 算所有 token） | 有 | EP ≤ 4，快速原型 |
| `alltoall` | A2A (EP组) + AG/RS (TP组) 分离 | 无 | 有（permute 结果） | EP ≥ 8，训练标配 |
| `flex` | DeepEP / HybridEP 融合 A2A | 无 | **无**（kernel内流转） | H100+，最优性能 |

---

## 七、常见问题 (FAQ)

### Q1: EP 和 TP 什么时候用哪个？
**答**：两者正交，可以同时用。EP 切专家维度（通信量 O(topk·h)），TP 切矩阵维度（通信量 O(h²)）。专家多 → EP 优先；单专家矩阵大（d_ff 大） → 加 TP。实际训练中往往是两者一起用，只是各自的 size 根据模型配置调。

### Q2: 为什么 allgather dispatcher 是默认值，但大家都用 alltoall？
**答**：`allgather` 是安全默认（不需要复杂的通信拓扑感知），适合初次跑通。一旦 EP ≥ 8，allgather 的计算冗余不可接受，必须切到 alltoall。

### Q3: EP + CP 互斥怎么理解？真不能同时用吗？
**答**：是说它们在 Megatron-LM 的同一个 RankGenerator 中互斥，不是说整个模型不能同时用。Attention 层用 CP（切序列）、MoE 层用 EP（切专家）是完全可行的——它们在不同的进程组体系中。只是进入 MoE 层时，CP 切分的序列片段被当作独立 token 处理。

### Q4: EP 的通信量跟 batch size 有关吗？
**答**：有关系。Batch 越大 → 每个 rank 的 token 越多 → All-to-All 总数据量增大 → 但单个 token 分摊的通信延迟更小（计算也更密集） → GEMM 更容易隐藏通信。Batch 小的时候（如推理），通信延迟占比高，是 EP 的痛点。

### Q5: 为什么 V3 要设 EP == num_experts（每个 rank 1 个专家）？
**答**：好处：(1) Expert DP Group 退化为单 rank → 省掉专家梯度的 DP 通信；(2) 每个 rank 只算 1 个专家的 GEMM，计算粒度最细 → 最适合 DualPipe 的细粒度调度。代价是 EP 通信跨 rank 更多 → 依赖节点限制路由和 DualPipe 来隐藏通信。

---

## 八、延伸阅读

- 📖 [Megatron-LM MoE 源码](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/transformer/moe) — token_dispatcher.py, moe_layer.py, experts.py, fused_a2a.py
- 📖 [DeepEP GitHub](https://github.com/deepseek-ai/DeepEP) — 高性能 EP 通信库
- 📖 [GShard](papers/GShard%20MoE%20Sharding.pdf) — MoE 大规模并行的基础工作
- 📖 [DeepSpeed-MoE](papers/DeepSpeed-MoE%20EP.pdf) — DeepSpeed 中的 EP 实现
- 📖 [Megatron-LM TP + PP + DP](papers/Megatron-LM%20TP%20PP%20DP.pdf) — Megatron 的混合并行方法论
- 📖 [混合精度训练](topics/mixed-precision-training.md) — EP 通信中 FP8 dispatch / BF16 combine 的精度选择
- 📖 [DeepSeek-V3 中文笔记](papers-zh/DeepSeek-V3.md) — V3 的 DualPipe + EP 通信重叠
- 📖 [DeepSeek-V4 技术报告](papers/DeepSeek-V4%20Technical%20Report.pdf) — V4 的细粒度 EP 方案

> **V3/V4 的 EP 深度分析**：见 [notes/expert-parallelism/deepseek-ep-in-depth.md](notes/expert-parallelism/deepseek-ep-in-depth.md)，涵盖 DeepEP 架构、DualPipe EP 重叠、Wave 调度、Mega-Kernel、EP 确定性等进阶话题。
