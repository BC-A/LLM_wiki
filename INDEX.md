# LLM Training Knowledge Index 🧠

> 大模型训练知识速查表。每个知识点 2-3 行讲清楚核心，细节链接到论文/笔记。
> 格式：**What → Why it matters → How it works (one line)**

---

## 分布式并行 (Distributed Parallelism)

### 数据并行 (DP / Data Parallelism)
- **What**: 每张卡持有完整模型副本，处理不同 batch 数据，梯度做 All-Reduce。
- **Why**: 最简单、通信量最小的并行方式，但要求单卡能装下完整模型。
- **Key insight**: ZeRO 系列将 DP 的内存冗余转化为通信换空间。
- 📖 [ZeRO](papers/ZeRO%20Memory%20Optimization.pdf) · [FSDP](papers/PyTorch%20FSDP.pdf)

### 张量并行 (TP / Tensor Parallelism)
- **What**: 将单层内的矩阵乘法沿行/列切分到多个 GPU，中间结果通过 All-Reduce / All-Gather 同步。
- **Why**: 突破单卡显存限制，但通信量与矩阵维度平方成正比（O(h²)），需要高速 NVLink。
- **Key numbers**: Megatron-LM TP 将 Attention 的 QKV 矩阵沿列切，FFN 的 gate/up 沿列切、down 沿行切。
- 📖 [Megatron-LM TP](papers/Megatron-LM%20TP.pdf) · [TP+PP+DP](papers/Megatron-LM%20TP%20PP%20DP.pdf)

### 流水线并行 (PP / Pipeline Parallelism)
- **What**: 不同层放在不同 GPU，micro-batch 流水线执行。
- **Why**: 通信量小（只传层间激活），适合跨节点。但 naive 实现有大量 bubble（空闲等待）。
- **1F1B**: 每个 GPU 交替做 forward/backward，bubble = (PP-1)/(PP+micro_batches)。
- **DualPipe (DeepSeek-V3)**: 双向流水线 + 计算-通信重叠，bubble 更小。
- 📖 [DualPipe → DeepSeek-V3 3.2.1](papers-zh/DeepSeek-V3.md#321-dualpipe-与计算-通信重叠) · [TP+PP+DP](papers/Megatron-LM%20TP%20PP%20DP.pdf)

### 专家并行 (EP / Expert Parallelism) — [📝 基础笔记](topics/expert-parallelism.md) · [🔬 V3→V4 深度分析](notes/expert-parallelism/deepseek-ep-in-depth.md)
- **What**: MoE 的不同专家分布在不同 GPU，token 通过 All-to-All 路由到对应专家的 GPU。
- **Why**: 天然适合 MoE 架构，通信量 = O(token · h)，远小于 TP。但 batch 小时通信占比高。
- **Megatron-LM 配置**: `--expert-model-parallel-size` + `--num-experts`，专家连续分块分配到 EP rank。
- **EP + TP 组合**: EP 切专家维度，TP 切层内矩阵维度 → 先 EP All-to-All → 再 TP AllGather/ReduceScatter。
- **EP + CP**: 互斥（不在同一 RankGenerator），CP 作用于 Attention，EP 作用于 MoE。
- **DeepEP**: NCCL Gin 后端，混合模式（节点内 NVLink + 节点间 RDMA），融合 permute + All-to-All。
- **V3 DualPipe**: PP 级计算-通信重叠 + 节点限制路由（M≤4）→ 隐藏跨节点 IB 延迟。
- **V4 Wave 调度**: 专家级细粒度三级流水线（Dispatch→Compute→Combine）+ Pull dispatch + MegaMoE 融合核。
- 📖 [EP 基础详解](topics/expert-parallelism.md) · [V3→V4 EP 深度分析](notes/expert-parallelism/deepseek-ep-in-depth.md)

### 序列并行 (SP / Sequence Parallelism)
- **What**: 将长序列沿 token 维度切分到多个 GPU，减少单卡激活内存。
- **Why**: 长上下文训练（128K+）时，激活内存与序列长度成正比。
- **Ulysses**: All-to-All 在序列维度换 attention head 维度，每 GPU 始终有完整 head。
- **Ring Attention**: 环形传递 KV blocks，通信与计算重叠。
- 📖 [DeepSpeed Ulysses](papers/DeepSpeed%20Ulysses%20SP.pdf) · [Ring Attention](papers/Ring%20Attention%20CP.pdf) · [Selective Recomputation SP](papers/Selective%20Recomputation%20SP.pdf)

### 上下文并行 (CP / Context Parallelism)
- **What**: 长序列的 KV cache 切分到多 GPU，用环形或 all-to-all 通信共享 KV。
- **与 SP 的区别**: CP 关注推理（KV cache 分片），SP 关注训练（激活分片）。两者常联用。
- 📖 [Ring Attention](papers/Ring%20Attention%20CP.pdf)

### 自动并行 / SPMD
- **GSPMD**: 用 sharding annotation 描述切分方案，编译器自动推导通信。
- **Alpa**: 两层搜索 — 算子间并行（PP）+ 算子内并行（TP/DP），自动找到最优配置。
- 📖 [GSPMD](papers/GSPMD%20SPMD.pdf) · [Alpa](papers/Alpa%20Parallelism%20Search.pdf)

---

## 内存优化 (Memory Optimization)

### ZeRO-1/2/3
- **ZeRO-1**: 分片优化器状态 (12× 参数量 → 分到 DP group)。
- **ZeRO-2**: + 分片梯度 → 每个 rank 只存自己负责的梯度。
- **ZeRO-3**: + 分片模型参数 → 前向/反向时按需 All-Gather，用完丢弃。通信量 = 参数量的 1.5×。
- **核心思想**: DP 下每张卡的参数/梯度/优化器状态是冗余的，消除冗余 = 显存换通信。
- 📖 [ZeRO](papers/ZeRO%20Memory%20Optimization.pdf) · [ZeRO-Infinity](papers/ZeRO-Infinity%20NVMe%20Offload.pdf)

### 激活重计算 (Activation Recomputation / Checkpointing)
- **What**: 反向传播时重新计算 activation 而非从内存读取，省显存、多花一次前向时间。
- **Selective Recomputation**: 只重计算显存大、重算快的 op（如 Attention），不重算显存小、重算慢的 op。
- 📖 [Selective Recomputation](papers/Selective%20Recomputation%20SP.pdf)

### 混合精度训练 (Mixed Precision Training) — [📝 专题笔记](topics/mixed-precision-training.md) · [🔬 DeepSeek-V3 FP8 详解](notes/mixed-precision-training/deepseek-v3-fp8-training.md) · [🔬 DeepSeek-V4 FP8/FP4 详解](notes/mixed-precision-training/deepseek-v4-mixed-precision.md)

> 系统性地整理：浮点格式 → BF16/FP8/FP4 训练方案 → 精度敏感算子 → 论文分析大纲。

#### 浮点数格式速查
- **FP32 (1-8-23)**: 训练权威副本（master weight），范围 ±3.4×10³⁸，精度 ~7 位十进制有效数字。
- **FP16 (1-5-10)**: 范围仅 ±65504，需要 Loss Scaling 防止梯度 underflow。机器精度 \(2^{-10} \approx 9.77\times 10^{-4}\)。
- **BF16 (1-8-7)**: 指数与 FP32 相同（8 位）→ 动态范围等同 FP32 → **不需要 Loss Scaling**。当前大模型训练主流。
- **FP8 E4M3 (1-4-3)**: 精度优先（3 位尾数），范围 ±448 → 用于前向传播（激活值分布稳定）。
- **FP8 E5M2 (1-5-2)**: 范围优先（5 位指数）→ 用于反向传播（梯度动态范围大）。
- **FP4 E2M1 (1-2-1)**: 仅 16 个可表示值 → QAT 模拟训练（FP32 master → FP4 quant → FP8 dequant → GEMM）。
- 📖 [格式对比详表](topics/mixed-precision-training.md#11-浮点数表示格式-floating-point-formats)

#### 混合精度训练方案对比
| 方案 | 精度敏感 op | 需要 Loss Scaling | 细粒度量化 | 代表模型 |
|------|------------|------------------|-----------|---------|
| FP16 + AMP | 保持 FP32 | ✅ 必须 | ❌ | BERT, GPT-2 |
| BF16 | 保持 FP32/BF16 | ❌ 不需要 | ❌ | LLaMA, V3/V4 |
| FP8 (V3) | 保持 BF16/FP32 | ❌ (tile scaling 替代) | ✅ 1×128 / 128×128 | DeepSeek-V3 |
| FP4 QAT (V4) | 保持 BF16/FP32 | ❌ (QAT 内置) | ✅ per-group | DeepSeek-V4 |

#### 核心经验知识
- **精度敏感算子**: Embedding > Softmax > Norm > Loss ≈ Optimizer Step >> GEMM ≈ RoPE。非线性和统计量操作保持高精度，线性操作（GEMM）可用低精度。
- **细粒度量化**: Per-tensor 量化中 outlier 主导 scale factor → 大部分值精度被浪费。Per-tile（1×128 或 128×128）独立缩放 → 精度大幅提升。
- **提升累加精度**: H800 FP8 Tensor Core 仅 14 位累加精度 → 每 128 个元素做一次 FP32 partial sum 提升。
- **BF16 vs FP16 选择**: A100+ 硬件上无脑选 BF16 — 动态范围远大于精度优势。
- 📖 [精度敏感算子详细列表](topics/mixed-precision-training.md#15-精度敏感算子-precision-sensitive-operations)
- 📖 [FP8 Formats](papers/FP8%20Formats.pdf) · [DeepSeek-V3 FP8](papers-zh/DeepSeek-V3.md#33-fp8-训练) · [V4 FP8/FP4](notes/mixed-precision-training/deepseek-v4-mixed-precision.md) · [V4 FP4 QAT](papers-zh/DeepSeek-V4-technical-report.zh.md)

### CPU/NVMe Offload
- **ZeRO-Infinity**: 将优化器状态、梯度、参数 offload 到 CPU DRAM 甚至 NVMe SSD。
- 📖 [ZeRO-Infinity](papers/ZeRO-Infinity%20NVMe%20Offload.pdf)

---

## 注意力机制 (Attention)

### FlashAttention (FA1/2/3)
- **What**: IO-aware 精确注意力 — 在 SRAM 中完成 softmax 的 tile-by-tile 计算，避免写回 HBM。
- **FA-3**: Hopper 架构专用 — 异步 TMA 加载 + warp specialization + FP8 低精度。
- **Key numbers**: FA-2 加速 ~2-4×，FA-3 在 H100 上再快 ~1.5-2×。
- 📖 [FlashAttention-3](papers/FlashAttention-3.pdf)

### MLA (Multi-head Latent Attention)
- **What**: DeepSeek-V2/V3/V4 的核心推理加速技术。对 KV 做低秩压缩 (d_c ≪ d_h·n_h)，推理时只缓存压缩后的 latent vector。
- **Why**: KV cache 是推理内存瓶颈，MLA 将 cache 量减少 ~10×，同时保持 MHA 的建模能力。
- **数学**: 压缩维度 d_c=512（KV），d_c'=1536（Q）。KV 用联合压缩，Q 独立压缩。
- **RoPE 解耦**: 解耦的 key 分量单独存，只做位置编码不加压缩（避免 RoPE 破坏低秩结构）。
- 📖 [DeepSeek-V2](papers/DeepSeek-V2%20MLA%20MoE.pdf) · [DeepSeek-V3 2.1.1](papers-zh/DeepSeek-V3.md#211-多头潜在注意力)

### GQA / MQA
- **MQA**: 所有 query heads 共享同一个 KV head → KV cache 减少 head 倍。
- **GQA**: N 个 query heads 共享 1 个 KV head（N>1）→ trade-off 性能和 cache 量。
- **MLA vs GQA**: MLA 用学习到的低秩压缩替代手工分组，理论上更灵活。

### Ring Attention
- **What**: 沿序列维度环形传递 KV blocks，每个 GPU 轮流做 attention。通信和计算可重叠。
- 📖 [Ring Attention](papers/Ring%20Attention%20CP.pdf)

---

## MoE 架构 (Mixture of Experts)

### MoE 基础
- **What**: FFN 替换为多个「专家」子网络，每个 token 只激活 top-k 个专家。
- **Why**: 模型参数可以很大（稀疏激活），而计算量保持可控。
- **Key trade-off**: 参数量 ↑ vs 通信量 ↑ vs 负载均衡难度 ↑。
- 📖 [GShard](papers/GShard%20MoE%20Sharding.pdf)

### DeepSeekMoE
- **细粒度专家**: 专家数更多、每个更小 → 灵活组合知识。
- **共享专家**: 部分专家对所有 token 激活 → 捕获通用知识，路由专家负责特化。
- **路由机制**:
  - V2/V3: Sigmoid + top-K + 无辅助损失负载均衡（per-expert bias 直接调，不走梯度）。
  - V4: Sqrt(Softplus) 替代 Sigmoid（更健康梯度），移除容量约束。
- 📖 [DeepSeekMoE](papers/DeepSeekMoE.pdf) · [DeepSeek-V3 2.1.2](papers-zh/DeepSeek-V3.md#212-带无辅助损失负载均衡的-deepseekmoe)

### 负载均衡 (Load Balancing)
- **Aux Loss 问题**: 辅助损失与 LM 损失竞争梯度方向 → α 超参难调 → 损害模型性能。
- **无辅助损失策略 (V3/V4)**: Per-expert bias 不参与梯度计算，直接按负载统计调整 (±0.001/step)。负载均衡信号与 LM 优化目标完全解耦。
- **序列级平衡损失**: 权重 0.0001，防止单条长序列内极端不均衡。
- 📖 [DeepSeek-V3 2.1.2](papers-zh/DeepSeek-V3.md#212-带无辅助损失负载均衡的-deepseekmoe)

### 节点限制路由
- **What**: 限制每个 token 最多发送到 M 个节点（V3 中 M=4），减少跨节点 IB 通信。
- **Why**: NVLink 带宽 >> IB 带宽 → 优先节点内通信 → 降低通信瓶颈。
- 📖 [DeepSeek-V3 2.1.2](papers-zh/DeepSeek-V3.md#212-带无辅助损失负载均衡的-deepseekmoe)

### Hash MoE (V4)
- **What**: 前几层用 token_id 的哈希函数决定路由，替代 learnable router。
- **Why**: 输入层 token embedding 信息量少，learnable router 区分度差 → Hash 路由是廉价替代。
- 📖 [DeepSeek-V4 技术报告](papers/DeepSeek-V4%20Technical%20Report.pdf) (Section 3.1)

---

## 训练框架与 Infra

### Megatron-LM 性能优化 — [📝 专题笔记](topics/megatron-performance-optimization.md)

> 系统性地整理 Megatron-LM 针对内存墙/通信墙/计算墙的完整优化方案，涵盖并行策略、融合算子、FP8 训练、Activation Offloading、CUDA Graph 等 30+ 项优化技术。

- **内存墙**: Distributed Optimizer (ZeRO-1), Precision-Aware Optimizer, CPU Offload, Fine-grained Recomputation, Activation Offloading
- **通信墙**: DP/TP/PP/EP 通信重叠, BF16/FP8 通信, Shared Expert Overlap, 1F1B A2A Overlap
- **计算墙**: GroupedGEMM, Router/Permute/CE Fusion, FP8 GEMM, CUDA Graph, Manual GC
- **MoE 专用**: Flex Dispatcher (DeepEP/HybridEP), Parallel Folding, FP8 Routing Map Padding
- 📖 [完整配置模板](topics/megatron-performance-optimization.md#32-最完整的训练配置模板) · [并行策略决策树](topics/megatron-performance-optimization.md#42-并行策略决策树) · [MoE README](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/README.md)

### DualPipe (V3)
- **What**: 双向流水线 + 在一个 forward-backward 对内重叠计算和通信。
- **Key**: All-to-All 和 PP 通信可被完全隐藏 → 跨节点 MoE 训练的通信瓶颈得以解决。
- **vs 1F1B**: Bubble 更小，但需保留两份参数（内存换时间）。
- 📖 [DeepSeek-V3 3.2.1](papers-zh/DeepSeek-V3.md#321-dualpipe-与计算-通信重叠)

### MegaMoE / Mega-Kernel (V4)
- **What**: 单一融合 kernel 包含全部 EP 操作（token 重排 → All-to-All → GEMM → SwiGLU → All-to-All → 输出重组）。
- **Why**: 消除 kernel relaunch 开销、中间结果写回 HBM、implicit barrier → 加速 1.50-1.96×。
- **Pull Dispatch**: 接收方主动拉取（而非发送方推送）→ 零通知延迟。
- 📖 [Expert Parallelism 基础](topics/expert-parallelism.md) · [V3→V4 EP 深度分析](notes/expert-parallelism/deepseek-ep-in-depth.md)

### 训练容错
- **ByteRobust**: 万卡级容错 — 故障检测 → 自动恢复 → 继续训练，最小化 GPU 浪费。
- **WAL (V4 Rollout)**: Token 粒度 Write-Ahead Log → 抢占恢复时从断点继续，不重新生成。
- 📖 [ByteRobust](papers/ByteRobust%20Fault%20Tolerance.pdf)

### 数据中心网络
- **Astral**: 面向 LLM 训练的 RDMA 监控和网络架构。
- 📖 [Astral](papers/Astral%20Datacenter%20Network.pdf)

---

## 强化学习 / 后训练 (RL / Post-Training)

### RLHF 框架
- **PPO (RLHF)**: 经典四模型架构 — Policy / Reference / Critic / Reward。Critic 与 Policy 同规模，计算量大。
- **GRPO**: 去掉 Critic 模型，用同一 prompt 的多条 response 的相对分数估计优势函数。省一半显存。
- **DPO**: 直接优化偏好对（不用 reward 模型），隐式 reward = log(π_θ / π_ref)。
- 📖 [GRPO → DeepSeek-V3 5.2.2](papers-zh/DeepSeek-V3.md#522-组相对策略优化) · [DPO](papers/DPO%20Algorithm.pdf) · [Scaling RLHF](papers/Scaling%20RLHF.pdf)

### GRPO 关键公式
```
Advantage = (r_i - mean(r_group)) / std(r_group)  // 组内归一化
Loss = min(ratio · A, clip(ratio, 1-ε, 1+ε) · A) - β · KL(π_θ || π_ref)
```
- **Ratio clipping**: 防止单步更新过大（同 PPO）。
- **KL 惩罚**: 防止偏离 reference 太远。
- 📖 [GRPO Analysis](papers/GRPO%20Analysis.pdf) · [DeepSeekMath GRPO](papers/DeepSeekMath%20GRPO.pdf)

### 蒸馏 (Distillation)
- **R1 → V3 蒸馏**: 用 R1 的长 CoT 数据训练 V3，将推理能力迁移到标准 LLM。
- **Trade-off**: 蒸馏提升推理性能，但显著增加平均响应长度。
- 📖 [DeepSeek-R1](papers/DeepSeek-R1%20RL%20Reasoning.pdf) · [DeepSeek-V3 5.4.1](papers-zh/DeepSeek-V3.md#541-从-deepseek-r1-蒸馏)

### 后训练框架
- **OpenRLHF**: 开源 RLHF 框架，支持 PPO/DPO/GRPO。
- **ROLL (阿里)**: 异步 RL 框架 — Rollout 和 Train 解耦，GPU 利用率更高。
- 📖 [OpenRLHF](papers/OpenRLHF%20Framework.pdf) · [ROLL](papers/ROLL%20RL%20Framework.pdf) · [ROLL Flash Async](papers/ROLL%20Flash%20Async.pdf)

### REINFORCE++ / Lite PPO / DPO
- 各种简化 RL 算法，目标都是降低 RLHF 的计算和工程复杂度。
- 📖 [REINFORCE++](papers/REINFORCE++%20Algorithm.pdf) · [ROLL Tricks](papers/ROLL%20RL4LLM%20Tricks.pdf)

---

## 模型系列

| 模型 | 核心创新 | 参数量 | 论文/笔记 |
|------|----------|--------|-----------|
| DeepSeek-V2 | MLA + DeepSeekMoE 架构验证 | 236B (21B active) | [V2](papers/DeepSeek-V2%20MLA%20MoE.pdf) |
| DeepSeek-V3 | FP8 训练 + DualPipe + Aux-Loss-Free + MTP | 671B (37B active) | [中文笔记](papers-zh/DeepSeek-V3.md) |
| DeepSeek-V4 | 1M 上下文 + MegaMoE + 细粒度 EP + CSA/HCA | — | [中文笔记](papers-zh/DeepSeek-V4-technical-report.zh.md) |
| DeepSeek-R1 | RL 推理蒸馏 + 长 CoT | — | [R1](papers/DeepSeek-R1%20RL%20Reasoning.pdf) |
| DeepSeekMath | GRPO 数学推理 | 7B | [DeepSeekMath](papers/DeepSeekMath%20GRPO.pdf) |
| Megatron-LM | TP (1D 切分) + TP+PP+DP 组合 | — | [TP](papers/Megatron-LM%20TP.pdf) · [TP+PP+DP](papers/Megatron-LM%20TP%20PP%20DP.pdf) |

---

## 常见面试问题速答

### Q1: ZeRO-1/2/3 的区别？
**一句话**: ZeRO-1 分片优化器状态 → ZeRO-2 +分片梯度 → ZeRO-3 +分片参数。越往后越省显存，通信量越大。
📖 [ZeRO](papers/ZeRO%20Memory%20Optimization.pdf)

### Q2: TP vs EP 什么时候用哪个？
**TP**: 层内矩阵切分，通信量 O(h²)，需要高速 NVLink。**EP**: MoE 专家切分，通信量 O(token · h)，适合跨节点。专家多 → EP 优先；单专家大 → TP 优先。两者可同时使用（EP 切专家维度，TP 切层内矩阵维度）。
📖 [Expert Parallelism FAQ](topics/expert-parallelism.md#q1-ep-和-tp-什么时候用哪个)

### Q3: PP 的 bubble 怎么算？DualPipe 怎么减少的？
**1F1B bubble**: (PP-1) / (PP + micro_batches)。DualPipe 用双向流水线 + 计算-通信重叠，bubble ≈ (PP/2-1)(F&B+B-3W)。
📖 [DeepSeek-V3 3.2.1](papers-zh/DeepSeek-V3.md#321-dualpipe-与计算-通信重叠)

### Q4: MLA 为什么省 KV cache？
KV 通过低秩矩阵压缩到 d_c 维（如 512），远小于 n_h × d_h（如 128×128=16384）。推理只缓存压缩后的 latent + RoPE key。
📖 [DeepSeek-V3 2.1.1](papers-zh/DeepSeek-V3.md#211-多头潜在注意力)

### Q5: GRPO 和 PPO 的核心区别？
GRPO 去掉 Critic 模型 → 用同 prompt 的 group 内相对分数估算 baseline → 省一半显存，适合大规模 MoE 训练。
📖 [DeepSeek-V3 5.2.2](papers-zh/DeepSeek-V3.md#522-组相对策略优化)

### Q6: 无辅助损失负载均衡为什么比 Aux Loss 好？
Aux Loss 的梯度与 LM loss 竞争 → 多目标优化的 trade-off → α 难调。无辅助损失用启发式规则直接调 bias，不经过梯度 → 两个优化目标解耦。
📖 [DeepSeek-V3 2.1.2](papers-zh/DeepSeek-V3.md#212-带无辅助损失负载均衡的-deepseekmoe)

### Q7: FP8 训练为什么能成功？有哪些关键技巧？
(1) 细粒度量化（1×128 tile-level 而非 per-tensor）；(2) 提高累加精度（CUDA Core FP32 累加，每 128 元素提升一次）；(3) 关键 op 保留高精度（Embedding/Attention/Norm/Gate/Loss）；(4) 缩放在线计算（不存历史 max）。
📖 [DeepSeek-V3 FP8 详解](notes/mixed-precision-training/deepseek-v3-fp8-training.md) · [DeepSeek-V3 3.3](papers-zh/DeepSeek-V3.md#33-fp8-训练) · [混合精度训练专题](topics/mixed-precision-training.md)

### Q8: BF16 和 FP16 应该选哪个？为什么大模型都在用 BF16？
BF16 动态范围 = FP32（8 位指数相同）→ 不需要 loss scaling → 训练更稳定，零心智负担。FP16 精度虽高（10 位 vs 7 位尾数），但动态范围小（最大 65504）→ 必须精细调整 loss scale → 大模型训练中得不偿失。A100+ 硬件上无脑选 BF16。
📖 [混合精度训练 FAQ](topics/mixed-precision-training.md#q1-bf16-和-fp16-到底选哪个)

### Q9: EP 通信为什么批大小敏感？
小 batch → 很多专家收到 0 token → 通信占比高 → Wave 调度空 wave 多。大 batch → 负载自然均衡 → 通信被 GEMM 隐藏。
📖 [Expert Parallelism §4.5](topics/expert-parallelism.md#45-通信量全景对比)

---

## 关键数字一览

| 数字 | 含义 |
|------|------|
| 671B / 37B | DeepSeek-V3 总参数 / 激活参数 |
| 2.788M H800 GPU hrs | DeepSeek-V3 完整训练成本 |
| 14.8T tokens | DeepSeek-V3 预训练数据量 |
| 2048 H800 | DeepSeek-V3 训练集群规模 |
| 128K | DeepSeek-V3/V4 最大上下文长度 |
| 1.8× | MTP 推测解码加速比 |
| 1.50-1.96× | MegaMoE EP 加速范围 |
| 6144 FLOPs/Byte | V4-Pro 计算-通信比阈值 (2·d_ff) |
| 0.001 | 无辅助损失 bias 更新速度 |
| d_c=512 | MLA KV 压缩维度 |
| 256 (V3) / 384 (V4) | 路由专家数量 |
| 8 (V3) / 6 (V4) | 每 token 激活专家数 |
| 3 层 | V4 Hash MoE 层数（前 3 个 Transformer block） |
| 1-8-23 / 1-5-10 / 1-8-7 / 1-4-3 / 1-5-2 | 各浮点格式的 (sign, exponent, mantissa) 位分配 |
| 65504 | FP16 正常数最大值（overflow 阈值，需 loss scaling 的原因） |
| 448 | FP8 E4M3 正常数最大值 |
| 14 位 | H800 FP8 Tensor Core 累加精度 |
| 128 | V3 FP8 GEMM 累加精度提升窗口 |
| 1×128 / 128×128 | 激活 / 权重 FP8 细粒度量化 tile size |
| 2× / 4× | FP8 / FP4 相对 BF16 的理论吞吐提升 |

---

## 学习路线建议

```
1. 基础并行 → TP → PP → DP → ZeRO
2. 内存优化 → 混合精度训练（FP32→BF16→FP8→FP4）→ Checkpointing → Offload
3. MoE 架构 → GShard → DeepSeekMoE → EP 基础 → EP 深度（DeepEP/V3/V4）→ 负载均衡
4. 注意力   → FA-2 → GQA/MQA → MLA → Ring Attention
5. 训练系统 → Megatron-LM 优化 → DualPipe → MegaMoE → 容错
6. 后训练   → PPO → DPO → GRPO → 蒸馏
```

---

## 目录结构

```
INDEX.md          ← 你在这里（知识速查索引）
README.md         ← 项目介绍 & 论文清单
papers/           ← 论文 PDF
papers-zh/        ← 中文翻译 / 精读笔记
topics/           ← 专题深度笔记
notes/            ← 进阶分析笔记（结合论文+源码）
  expert-parallelism/    ← EP 深度分析（DeepEP + V3/V4）
  mixed-precision-training/ ← 混合精度训练深度分析
code/             ← 参考实现
interview/        ← 面试准备资料
```

> **维护原则**: 读完新论文 → 翻译到 `papers-zh/` → 将核心知识点以 2-3 行加到 INDEX.md → 如果有足够深度 → 写专题笔记到 `topics/`。
