# Megatron-LM 高性能训练优化指南 (Megatron-LM Performance Optimization)

> **一句话概述**: Megatron-LM 围绕「内存墙、通信墙、计算墙」三大瓶颈，提供了一整套从并行策略、融合算子、量化训练到 activation offloading 的生产级优化方案，支撑 DeepSeek-V3 等数千亿 MoE 模型的高效训练。
> **核心问题**: 在大规模 MoE 训练中，如何系统性地突破内存容量、通信带宽、计算吞吐三重限制？

---

## 一、先验知识 / 基础知识 (Prerequisites)

### 1.1 大规模训练的三堵墙

大模型训练的性能瓶颈可以归纳为三类：

```
                    ┌─────────────────┐
                    │   Compute Wall  │
                    │  GPU 算力不足    │
                    │  GEMM 受限于     │
                    │  Tensor Core 吞吐 │
                    └────────┬────────┘
                             │
    ┌────────────────────────┼────────────────────────┐
    │                        │                        │
    ▼                        ▼                        ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│  Memory Wall  │   │Communication  │   │  Compute Wall  │
│  显存容量不足   │   │    Wall       │   │  Kernel Launches│
│               │   │  跨节点通信     │   │  + Python     │
│ 参数 + 优化器  │   │  带宽瓶颈      │   │  Overhead     │
│ + 激活值      │   │               │   │               │
└───────────────┘   └───────────────┘   └───────────────┘
```

| 瓶颈 | 表象 | 根源 | 优化方向 |
|------|------|------|---------|
| **内存墙** | OOM、batch size 受限 | 模型参数 + 优化器状态 + 激活值 > GPU HBM | 重计算、offloading、量化、分布式优化器 |
| **通信墙** | GPU 空闲等待通信 | 跨节点带宽 (IB/RoCE) << NVLink 带宽 | 通信-计算重叠、FP8 通信、融合调度 |
| **计算墙** | GPU 利用率低 | Kernel launch overhead、非融合小 op、非 Tensor Core 路径 | 融合算子(GrouppedGEMM)、FP8/FP4 GEMM、CUDA Graph |

> **关键认知**: 这三堵墙不是独立的——一个优化可能同时缓解多个瓶颈（如 FP8 训练既减少显存又降低通信量又提升计算吞吐），也可能 trade-off（如重计算省显存但多花计算）。

### 1.2 Megatron-LM 的优化架构全景

Megatron-LM 采用分层优化架构：

```
┌─────────────────────────────────────────────────────┐
│              顶层：并行策略 (Parallelism)             │
│  DP · TP · PP · EP · CP · SP  → 分布模型到集群       │
├─────────────────────────────────────────────────────┤
│              中层：融合与调度 (Fusion & Scheduling)    │
│  GroupedGEMM · Router Fusion · Permute Fusion       │
│  CUDA Graph · Comm Overlap · 1F1B A2A Overlap       │
├─────────────────────────────────────────────────────┤
│              底层：精度与内存 (Precision & Memory)     │
│  FP8/FP4 · Activation Offload · Recompute           │
│  Distributed Optimizer · Manual GC · Straggler Det   │
└─────────────────────────────────────────────────────┘
```

---

## 二、核心内容 (Core Techniques)

### 2.1 内存墙优化 (Memory Wall)

#### 2.1.1 分布式优化器 (Distributed Optimizer)

**原理**: 将优化器状态 (Adam 的 m, v) 沿 DP 维度切分，等价于 ZeRO-1。

```
传统方案: 每个 DP rank 持有完整优化器状态
  ┌─────────────────────────────────────────┐
  │ Rank 0: 完整 m, v  (2× 参数量)          │
  │ Rank 1: 完整 m, v  (2× 参数量)  ← 冗余! │
  │ Rank 2: 完整 m, v  (2× 参数量)          │
  └─────────────────────────────────────────┘

Distributed Optimizer: 每个 rank 只存自己负责的参数片段的优化器状态
  ┌─────────────────────────────────────────┐
  │ Rank 0: m[0:1/3], v[0:1/3]  (2/3× 参数量)│
  │ Rank 1: m[1/3:2/3], v[1/3:2/3]         │
  │ Rank 2: m[2/3:3], v[2/3:3]             │
  └─────────────────────────────────────────┘
```

**配置**:
```bash
--use-distributed-optimizer
```

**节省**: 优化器显存 = (2× 参数量) / DP_size（Adam 的 m+v，FP32 共计 8 bytes/param）

#### 2.1.2 Precision-Aware Optimizer

将 Adam 的一阶矩 (m) 和二阶矩 (v) 从 FP32 降到 BF16 存储，减少 50% 优化器显存。

```bash
--use-precision-aware-optimizer
--exp-avg-dtype bf16       # 一阶矩 m: FP32 → BF16
--exp-avg-sq-dtype bf16    # 二阶矩 v: FP32 → BF16
```

| 组件 | FP32 (default) | BF16 (precision-aware) | 节省 |
|------|---------------|----------------------|------|
| m (一阶矩) | 4 bytes/param | 2 bytes/param | 50% |
| v (二阶矩) | 4 bytes/param | 2 bytes/param | 50% |
| **合计** | **8 bytes/param** | **4 bytes/param** | **50%** |

> **Note**: 精度损失极小——Adam 的 m/v 主要用于计算自适应学习率 `m / (sqrt(v) + ε)`，BF16 的尾数精度（7 bits ≈ 2 位十进制）对二阶统计量来说足够。

#### 2.1.3 Optimizer CPU Offload

将优化器状态和梯度卸载到 CPU 内存，GPU 只保留模型参数和激活值。

```bash
--optimizer-cpu-offload
```

**适用场景**: 显存极度紧张但 CPU 内存充足的场景。通信开销：GPU ↔ CPU 通过 PCIe，带宽 ~32 GB/s（仅为 HBM 的 1/100）。

#### 2.1.4 细粒度激活重计算 (Fine-grained Recomputation)

与传统的 full layer checkpointing 不同，Megatron 支持**模块级别**的选择性重计算。

```bash
--recompute-granularity selective
--recompute-modules layernorm moe_act mlp
```

支持的模块:

| 模块 | 类型 | 说明 |
|------|------|------|
| `layernorm` | output-discarding | 重算 input/pre_mlp layernorm，节省归一化层激活值 |
| `moe_act` | output-discarding | 重算 GroupedMLP 激活函数，省 GroupedGEMM 中间结果 |
| `mla_up_proj` | output-discarding | 重算 MLA up projection + RoPE |
| `core_attn` | standard checkpointing | 重算 attention 子模块 |
| `mlp` | standard checkpointing | 重算 dense MLP（用于 hybrid 模型如 DeepSeek-V3） |
| `moe` | standard checkpointing | 重算 MoE 层 |

> **关键区分**: `output-discarding` 只重算输出（前向仅丢弃输出，反向重新计算），比 `standard checkpointing`（需要重新运行整个前向）更轻量。

#### 2.1.5 细粒度 Activation Offloading

将激活值异步卸载到 CPU 内存，D2H (Device-to-Host) 和 H2D (Host-to-Device) 传输与计算重叠。

```bash
--fine-grained-activation-offloading
--offload-modules attn_norm core_attn attn_proj mlp_norm expert_fc1 moe_act
```

**工作原理**:
```
Forward:  [Compute layer N] ──→ [Async D2H: offload activation N]
                                    ↓ (overlaps with...)
          [Compute layer N+1] ──→ [Async D2H: offload activation N+1]

Backward: [Async H2D: prefetch activation N] ──→ [Compute backward N]
                      ↓ (overlaps with...)
          [Async H2D: prefetch activation N+1] ──→ [Compute backward N+1]
```

**Trade-off**: GPU-CPU 带宽换显存。PCIe 5.0 ×16 ≈ 64 GB/s vs HBM3 ≈ 3.35 TB/s → 约 50× 带宽差。

**环境变量**:
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # 防止内存碎片化
```

### 2.2 通信墙优化 (Communication Wall)

#### 2.2.1 DP 通信重叠 (Data Parallel Comm Overlap)

| 优化 | 说明 | 配置 |
|------|------|------|
| **Gradient Reduce Overlap** | 反向计算与梯度 All-Reduce 异步重叠 | `--overlap-grad-reduce` |
| **Param Gather Overlap** | 前向计算与参数 All-Gather 异步重叠 | `--overlap-param-gather` |
| **BF16 Gradient Reduce** | 梯度通信用 BF16 替代 FP32，通信量减半 | `--grad-reduce-in-fp32 false` |
| **FP8 Param Gather** | 参数 All-Gather 用 FP8，通信量减半 | `--fp8-param-gather` |

**Gradient Reduce Overlap 原理**:
```
无重叠:
  [Compute backward L1] → [All-Reduce grad L1] → [Compute backward L2] → [All-Reduce grad L2]
                           ↑ idle                                    ↑ idle

有重叠:
  [Compute backward L1] → [Compute backward L2] → [Compute backward L3]
    [Async AR L1]           [Async AR L2]           [Async AR L3]
```

#### 2.2.2 TP 通信重叠 (Tensor Parallel Comm Overlap)

Tensor Parallelism 每层需要 2 次 All-Reduce（attention output + MLP output）。`--tp-comm-overlap` 将通信与后续计算重叠。

```bash
--tp-comm-overlap                  # 启用 TP 通信重叠
--tensor-model-parallel-size >= 2  # 需要 TP ≥ 2
--sequence-parallel                # 必须开启 SP
```

**两种模式**:
- **Bulk**: 无依赖的通信批量发出（适合无依赖场景）
- **Pipelined**: 有依赖关系的通信分阶段发出（适合有依赖场景）

#### 2.2.3 PP 通信重叠 (Pipeline Parallel Comm Overlap)

```bash
--overlap-p2p-comm                 # 重叠 PP P2P 通信
--num-layers-per-virtual-pipeline-stage N  # VPP 增加重叠机会
```

PP 的层间传递激活（forward）和梯度（backward），`--overlap-p2p-comm` 将 P2P 通信与不依赖该通信结果的计算重叠。

#### 2.2.4 EP 通信重叠 (Expert Parallel Comm Overlap)

MoE 训练中，EP 的 All-to-All 通信是主要瓶颈之一。Megatron 提供两种重叠方案：

**方案 1: 1F1B A2A Overlap（batch 级别）**

```bash
--overlap-moe-expert-parallel-comm
--delay-wgrad-compute             # 延迟权重梯度计算以扩大重叠窗口
```

```
Micro-batch N:         [Dispatch] [Expert Compute] [Combine] [Wgrad]
Micro-batch N+1:       [Dispatch] [Expert Compute] [Combine] [Wgrad]
                               ↑ A2A of N+1 overlaps with Compute of N
```

**方案 2: Shared Expert Overlap（层内级别）**

```bash
--moe-shared-expert-overlap       # 共享专家计算与 EP 通信并发
```

```
共享专家 ([s, h] → [s, d_ff] → [s, h]) 与
EP dispatch (All-to-All) 并发执行（独立 CUDA stream）
```

> **适用场景**: DeepSeek-V3 规模模型（256 专家、EP=64+），EP 通信量可能超过 NVLink 带宽 → 使用 1F1B A2A Overlap。

### 2.3 计算墙优化 (Compute Wall)

#### 2.3.1 GroupedGEMM

MoE 层每个 token 被路由到不同专家 → 朴素实现需要逐个专家执行 GEMM（多个小 GEMM），GPU 利用率低。

**GroupedGEMM** 将多个独立的小 GEMM 打包成一个 kernel launch，大幅提升 GPU 利用率。

```
朴素实现:
  for expert in local_experts:
      output[expert] = input[tokens_of_expert] @ W[expert]
  → N 次独立 kernel launch，每次可能只处理几十个 token

GroupedGEMM:
  grouped_gemm([input_0, input_1, ..., input_N],
               [W_0, W_1, ..., W_N])
  → 1 次 kernel launch，batch 后的 M/N/K 维度更规整
```

**配置**:
```bash
--moe-grouped-gemm
```

#### 2.3.2 Router Fusion

将 Router 的计算流程——projection（Linear）→ Top-K 选择 → Softmax → aux loss—融合为更少的 kernel。

```bash
--moe-router-fusion
```

**融合范围**:

| 操作 | 融合前 | 融合后 |
|------|--------|--------|
| Router Linear | 独立 kernel | 合并到融合 kernel |
| TopK selection | 独立 kernel | 合并 |
| Softmax | 独立 kernel | 合并 |
| Aux loss compute | 独立 kernel | 合并 |

#### 2.3.3 Permute Fusion

Token 在 dispatch（按专家重排）和 combine（按 token 重排）过程中需要 permutation 操作。融合后的 permute kernel 消除多次内存搬运。

```bash
--moe-permute-fusion
```

#### 2.3.4 Cross-Entropy Loss Fusion

将 Cross-entropy loss 的计算融合为较少 kernel。

```bash
--cross-entropy-loss-fusion
--cross-entropy-fusion-impl native
```

#### 2.3.5 Manual GC (Python Garbage Collection)

Python GC 在大规模训练中可能造成不可预测的暂停。手动控制 GC interval 消除 jitter。

```bash
--manual-gc              # 启用手动 GC
--manual-gc-interval 10  # 每 10 个 training step 触发一次 GC
```

#### 2.3.6 Straggler Detection

检测集群中的慢节点（straggler），帮助识别网络/硬件异常。

参考: `megatron/core/README_STRAGGLER.md`

### 2.4 并行策略选择指南 (Parallelism Strategy)

Megatron-LM 支持 DP + TP + PP + EP + CP + SP 全部 6 种并行方式。以下是官方推荐的选择指南：

#### 2.4.1 五大原则

| 优先级 | 原则 | 说明 |
|--------|------|------|
| **1** | 最小化模型并行，最大化数据并行 | TP/EP/PP 尽量小，只要不 OOM 就行 |
| **2** | EP 和 TP 通信限制在 NVLink 域内 | EP × TP size ≤ 单节点 GPU 数（通常是 8） |
| **3** | 跨节点缩放用 PP | 层间通信量小（只传激活），适合跨 IB/RoCE |
| **4** | Expert 层优先用 EP 而非 TP | EP: O(token · h) 通信 vs TP: O(h²) 通信 + EP 天然适配 MoE |
| **5** | 长序列 (≥8K) 启用 CP | Context Parallelism 对长序列激活显存更友好 |

#### 2.4.2 MoE Parallel Folding（MoE 并行折叠）

**核心思想**: 将 Attention 层和 MoE 层的并行策略**解耦**，各自使用最优配置。

```
传统混合并行:
  Attention layers: TP × CP × DP × PP
  MoE layers:       TP × CP × DP × PP  ← 与 Attention 绑定！

MoE Parallel Folding:
  Attention layers: TP × CP × DP × PP   ← 高 TP 优化 Attention GEMM
  MoE layers:       ETP × EP × EDP × PP ← 低/零 ETP，高 EP 优化 MoE
```

**四大好处**:

1. **打破 EP ≤ DP 约束**: EP 不再受 DP size 限制 → 可以大幅增加专家并行度
2. **降低最小 GPU 需求**: CP 和 EP fold 在一起
3. **独立优化**: Attention 用高 TP（大 GEMM 效率高），MoE 用 ETP=1（避免冗余通信）
4. **高带宽通信留在 NVLink 域**: EP A2A 和 TP AR 不跨节点

#### 2.4.3 并行策略速查表

| 并行类型 | 切分维度 | 通信操作 | 通信量 | 最佳场景 |
|---------|---------|---------|--------|---------|
| **DP** | Batch | All-Reduce (grad) | O(参数量) / step | 任何时候，基础并行 |
| **TP** | 矩阵行/列 | All-Reduce ×2/layer | O(b·s·h) per layer | 大矩阵单卡放不下 |
| **PP** | Layer | P2P Send/Recv | O(b·s·h) per boundary | 跨节点深度缩放 |
| **EP** | Expert | All-to-All ×2/layer | O(b·s·h) per layer | MoE 专家分布 |
| **CP/SP** | Sequence | All-to-All / Ring P2P | O(b·s·h) | 长序列 (≥8K) |
| **DP (ZeRO)** | Param/Grad/Opt | All-Gather + Reduce-Scatter | ~1.5× 参数量 | 单卡放不下模型 |

### 2.5 Token Dispatch 策略

MoE 的核心操作——将 token 发送到对应专家——有 4 种实现方式。

| Dispatcher | 通信方式 | 适合场景 | 配置 |
|------------|---------|---------|------|
| **allgather** | All-Gather tokens → 每个 GPU 本地计算所有专家 → ReduceScatter | TP-only, 小 EP | `--moe-token-dispatcher-type allgather` |
| **alltoall** | NCCL All-to-All dispatch + combine | 标准 EP > 1 | `--moe-token-dispatcher-type alltoall` |
| **flex + DeepEP** | 融合 permute + A2A，跨节点去重 token | 跨节点 EP, H100 | `--moe-token-dispatcher-type flex --moe-flex-dispatcher-backend deepep` |
| **flex + HybridEP** | TMA + IBGDA 原生 MNNVL 支持 | GB200 NVL72 | `--moe-token-dispatcher-type flex --moe-flex-dispatcher-backend hybridep` |

**DeepEP Flex Dispatcher 详解**:
- 将 token permutation 和 All-to-All 融合为单一 kernel
- 跨节点时去除冗余 token（同一 token 被发送到同一节点的多个 rank → 只传一份）
- 将节点内 (NVLink) 和节点间 (RDMA) 通信融合在一次操作中

### 2.6 FP8 训练 (FP8 Training for MoE)

Megatron-LM 支持三种 FP8 精度方案，对 MoE 模型有专门的优化。

#### 2.6.1 FP8 Recipe

| Recipe | 缩放粒度 | 格式 | 平台 |
|--------|---------|------|------|
| **Per-tensor** | 整张 tensor | E4M3/E5M2 hybrid | Hopper, Blackwell |
| **Blockwise** | 激活 1×128, 权重 128×128 | E4M3 | Hopper (production-proven on DeepSeek-V3) |
| **MXFP8** | 1×32 | E4M3 + E8M0 | Blackwell |

**配置**:
```bash
--fp8-format e4m3              # 或 hybrid, mx
--fp8-recipe blockwise         # 或 per_tensor, mx
```

#### 2.6.2 MoE 专用 FP8 优化

| 优化 | 说明 | 配置 |
|------|------|------|
| **Routing Map Padding** | Padding routing map（非 token）以对齐 M 维度到 16/32 | `--moe-router-padding-for-fp8` |
| **FP8 Param Gather** | FP32 master → FP8（跳过 BF16 中间副本）→ 参数 All-Gather | `--fp8-param-gather` |
| **FP8 Dispatch** | Token dispatch 用 FP8 格式，通信量减半 | (built-in when FP8 enabled) |

#### 2.6.3 FP8 对三堵墙的贡献

| 墙 | 收益 | 量化 |
|----|------|------|
| **内存** | 激活值存 FP8 而非 BF16 | -50% 激活显存 |
| **内存** | 消除 BF16 weight 副本 | FP32→FP8 直接 cast |
| **通信** | EP dispatch 通信量减半 | -50% A2A volume |
| **通信** | 参数 All-Gather 通信量减半 | -50% param gather |
| **计算** | FP8 Tensor Core 比 BF16 快 | ~2× GEMM 吞吐 |

> 📖 详细 FP8 原理参考: [混合精度训练专题](mixed-precision-training.md)

### 2.7 CUDA Graph

CUDA Graph 将 CPU 端的 kernel launch 序列预先录制 → 回放时消除 launch overhead，对**小 batch / 浅层模型 / 大量小 op 的 MoE** 场景收益最大。

| 实现 | 说明 | 配置 |
|------|------|------|
| **local** | 用 MCore 内置 Graph Manager 录制 per-layer graph | `--cuda-graph-impl local` |
| **transformer_engine** | 用 TE 的 `make_graphed_callables()` | `--cuda-graph-impl transformer_engine` |
| **full_iteration** | 整个 forward-backward 迭代录制为一个 graph | `--cuda-graph-impl full_iteration` |

**MoE 特殊处理**: MoE 的动态形状（无 expert capacity factor / dropless）导致 graph 录制困难 → 使用 `--cuda-graph-modules attn` 只对 attention 层录制。

```bash
# 推荐：MoE 训练时只 capture attention 层
--cuda-graph-modules attn
--cuda-graph-impl transformer_engine

# 推理时控制 scope
--inference-cuda-graph-scope layer   # 或 block
```

### 2.8 其他重要功能

#### 2.8.1 Upcycling (Dense → MoE)

将已有的 dense checkpoint 转换为 MoE 模型继续训练：

```bash
--moe-use-upcycling                        # 启用 upcycling
--moe-upcycling-granularity N              # 1=默认(复制MLP), >1=细粒度(专家 hidden dim 更小)
--load /path/to/dense/checkpoint           # 加载 dense 模型
--save /path/to/moe/checkpoint             # 保存转换后的 MoE 模型
```

两种策略:
- **Default**: 每个专家 = dense MLP 的完整副本
- **Granular**: 专家 hidden dim = dense FFN hidden dim / N（如 N=4 → 专家只有原 MLP 的 1/4 大小，但数量 ×4）

#### 2.8.2 Distributed Checkpointing for MoE

MoE 模型 checkpoint 支持与 dense 模型不同的并行模式保存/加载（如训练时 EP=64，加载时 EP=8）。

#### 2.8.3 FSDP2 Support

Megatron-LM 支持 FSDP2 作为额外的分布式策略。参考 `megatron/core/distributed/README.md`。

#### 2.8.4 Per-Layer Logging

```bash
--moe-per-layer-logging   # 每层 MoE 的负载均衡、token 分布等统计
```

---

## 三、代码与配置分析 (Code & Config Analysis)

### 3.1 关键源码文件

| 文件 | 功能 |
|------|------|
| [moe_layer.py](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/moe_layer.py) | MoE 层主入口，forward/backward 流程 |
| [token_dispatcher.py](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/token_dispatcher.py) | 三种 dispatcher 实现 (alltoall/flex/allgather) |
| [fused_a2a.py](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/fused_a2a.py) | DeepEP/HybridEP fused dispatch/combine kernel |
| [experts.py](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/experts.py) | TEGroupedMLP (GroupedGEMM), SequentialMLP, InferenceGroupedMLP |
| [router.py](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/router.py) | TopK / Group Top-K router, load balancing |
| [shared_experts.py](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/shared_experts.py) | 共享专家 MLP + 通信重叠 |
| [moe_utils.py](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/moe_utils.py) | permute/unpermute, capacity, FP8 alignment |
| [upcycling_utils.py](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/upcycling_utils.py) | Dense→MoE 转换逻辑 |
| [paged_stash.py](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/paged_stash.py) | Paged memory stash（内存管理） |

### 3.2 最完整的训练配置模板

```bash
# ======== 基础并行策略 ========
--tensor-model-parallel-size 4
--pipeline-model-parallel-size 4
--expert-model-parallel-size 8
--sequence-parallel
--context-parallel-size 1
--use-distributed-optimizer

# ======== MoE 配置 ========
--num-experts 256
--moe-router-topk 8
--moe-token-dispatcher-type flex
--moe-flex-dispatcher-backend deepep
--moe-grouped-gemm

# ======== 负载均衡 ========
--moe-router-load-balancing-type aux_loss
# 或: 无辅助损失
# --moe-router-enable-expert-bias
# --moe-router-bias-update-rate 1e-3

# ======== 计算优化 ========
--moe-router-fusion
--moe-permute-fusion
--cross-entropy-loss-fusion
--cross-entropy-fusion-impl native

# ======== 通信重叠 ========
--overlap-param-gather
--overlap-grad-reduce
--grad-reduce-in-fp32 false
--tp-comm-overlap
--overlap-moe-expert-parallel-comm
--delay-wgrad-compute
--moe-shared-expert-overlap

# ======== FP8 训练 ========
--fp8-format e4m3
--fp8-recipe blockwise
--fp8-param-gather
--moe-router-padding-for-fp8

# ======== CUDA Graph ========
--cuda-graph-impl transformer_engine
--cuda-graph-modules attn

# ======== 重计算与 Offload ========
--recompute-granularity selective
--recompute-modules layernorm moe_act

# ======== 内存与 GC ========
--manual-gc
--manual-gc-interval 10
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export NCCL_NVLS_ENABLE=0

# ======== Upcycling (可选) ========
# --moe-use-upcycling
# --moe-upcycling-granularity 4
# --load /path/to/dense/ckpt
```

### 3.3 参考示例脚本

| 示例 | 路径 | 关键配置 |
|------|------|---------|
| LLaMA-3 8B FP8 | `examples/llama/train_llama3_8b_h100_fp8.sh` | FP8 + overlap + distributed optimizer |
| Mixtral 8×7B | `examples/mixtral/train_mixtral_8x7b_distributed.sh` | EP=8 + GroupedGEMM + alltoall |
| GPT-3 175B | `examples/gpt3/train_gpt3_175b_distributed.sh` | TP=8 + PP=16 |

---

## 四、关键数字与速查 (Quick Reference)

### 4.1 性能优化效果估算

| 优化 | 显存节省 | 通信节省 | 计算加速 | 适用条件 |
|------|---------|---------|---------|---------|
| Distributed Optimizer | ~(1 - 1/DP) opt states | - | - | DP > 1 |
| Precision-Aware Opt | 50% opt states | - | - | Always |
| FP8 Training | ~50% activations | 50% EP dispatch + param gather | ~2× GEMM | H100+ |
| BF16 Grad Reduce | - | 50% grad comm | - | Always |
| GroupedGEMM | - | - | 1.5-3× (小batch 更明显) | MoE |
| Router Fusion | - | - | 减少 kernel launch | MoE |
| Permute Fusion | - | - | 减少 2× 内存搬运 | MoE |
| CUDA Graph | - | - | 减少 launch overhead | 小batch/浅层 |
| EP A2A Overlap | - | 隐藏 A2A 通信 | - | EP > 1 |
| TP Comm Overlap | - | 隐藏 AR 通信 | - | TP ≥ 2 + SP |
| Activation Offload | ~N × activations → CPU | 增加 D2H/H2D | -0~5% (重叠) | CPU DRAM 充足 |

### 4.2 并行策略决策树

```
单卡能否放下模型？
  ├── 能 → DP only (最简单)
  └── 否 → 单层矩阵太大？
            ├── 是 → 加 TP (NVLink 域内)
            └── 否 → 总层数太多？
                      ├── 是 → 加 PP (跨节点)
                      └── 否 → MoE 专家太多？
                                ├── 是 → 加 EP
                                └── 否 → 序列太长？
                                          ├── 是 → 加 CP/SP
                                          └── 否 → 减少 batch size 或加 ZeRO
```

### 4.3 命令速查

```bash
# 快速检查吞吐
--log-throughput

# 内存碎片
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# NCCL 内存开销
export NCCL_NVLS_ENABLE=0

# 读取 Megatron MoE README
cat megatron/core/transformer/moe/README.md
```

---

## 五、常见问题 (FAQ)

### Q1: Megatron-LM 的 Distributed Optimizer 和 ZeRO-1 是一回事吗？

**是的，等效。** 两者都沿 DP 维度分片优化器状态，每个 rank 只存储自己负责的参数片段的 m/v。区别在于 Megatron 的实现与自身的 TP/PP/EP 机制深度集成。

### Q2: 什么时候用 GroupedGEMM，什么时候用 SequentialMLP？

- **GroupedGEMM**: batch 足够大时（每个专家都有一定量的 token）→ 合并 GEMM 提升 GPU 利用率
- **SequentialMLP**: batch 极小（推理、小 batch）→ 逐个执行专家 GEMM，避免 GroupedGEMM 的打包开销

Megatron-LM 内部会自动选择（根据 `--moe-grouped-gemm` flag）。

### Q3: alltoall vs flex 哪个 dispatcher 更好？

| 场景 | 推荐 |
|------|------|
| 单节点 EP (≤8 GPUs) | `alltoall` — 纯 NVLink，NCCL 足够 |
| 跨节点 EP (>8 GPUs) | `flex + deepep` — 融合 permute+A2A + 去重 |
| GB200 NVL72 | `flex + hybridep` — TMA+IBGDA 原生支持 |
| TP-only, 小 EP | `allgather` — 免 A2A 通信 |

### Q4: FP8 训练的精度有保障吗？

看 recipe。**Per-tensor** 最粗糙但最快；**blockwise**（1×128 / 128×128）在 DeepSeek-V3 上验证过生产可用；**MXFP8** 最精细但需要 Blackwell。核心技巧是 (1) 细粒度 tile-level scaling + (2) 提高累加精度 + (3) 精度敏感 op 保留高精度。

### Q5: CUDA Graph 在 MoE 上怎么用？

MoE 的 token 分布是动态的 → graph 录制困难。解决方案是只对 **attention 层**录制 CUDA Graph（`--cuda-graph-modules attn`），MoE 层正常执行。或者使用 `full_iteration` 模式录制整个 iteration（需要 capacity factor / padding 保证形状固定）。

### Q6: Activation Offload 会影响训练速度吗？

理论上不会（D2H/H2D 与计算异步重叠），实际上取决于:
- CPU↔GPU 带宽（PCIe 5.0 更有利于重叠）
- Offload 的模块量（越多越难完全隐藏）
- 计算密度（GEMM 越重越容易隐藏通信）

### Q7: MoE Parallel Folding 和普通混合并行有什么本质区别？

普通混合并行: Attention 和 MoE **共享**同一套并行配置 → EP 受 DP 约束（`EP ≤ DP`）。

MoE Parallel Folding: Attention 和 MoE **独立**并行配置 → EP 可以 >> DP → 更适合大量专家的场景（如 256 专家）。

### Q8: `--delay-wgrad-compute` 有什么用？

延迟权重梯度计算可以为 EP A2A Overlap 创造更大的重叠窗口。正常流程: dispatch → expert compute → combine → wgrad。延迟后：将 wgrad 推迟到下一个 micro-batch 的 dispatch 之后，使当前 micro-batch 的 combine 与下一个 micro-batch 的 dispatch 可以重叠。

---

## 六、延伸阅读 (Further Reading)

- 📖 [Megatron-LM MoE README (优化技术全集)](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/README.md) — 本文大部分内容的第一手来源
- 📖 [Expert Parallelism 基础详解](expert-parallelism.md) — EP 的通信模式、A2A 原语、EP+TP 组合等
- 📖 [混合精度训练专题](mixed-precision-training.md) — FP32/FP16/BF16/FP8/FP4 格式对比与训练方案
- 📖 [Megatron-LM 官方并行指南](https://github.com/NVIDIA/Megatron-LM/blob/main/docs/user-guide/parallelism-guide.md)
- 📖 [Megatron-LM Fine-Grained Activation Offloading](https://github.com/NVIDIA/Megatron-LM/blob/main/docs/user-guide/features/fine_grained_activation_offloading.md)
- 📖 [Megatron-LM Distributed Optimizer](https://github.com/NVIDIA/Megatron-LM/blob/main/docs/user-guide/features/dist_optimizer.md)
- 📖 [DeepSeek-V3 中文笔记](../papers-zh/DeepSeek-V3.md) — FP8 训练 + DualPipe + Aux-Loss-Free 实战
- 📖 [DeepSeek-V4 中文笔记](../papers-zh/DeepSeek-V4-technical-report.zh.md) — MegaMoE + 细粒度 EP
- 📖 [Megatron-LM TP 论文](../papers/Megatron-LM%20TP.pdf) · [TP+PP+DP 论文](../papers/Megatron-LM%20TP%20PP%20DP.pdf)
- 📖 [Straggler Detection Utility](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/README_STRAGGLER.md)
