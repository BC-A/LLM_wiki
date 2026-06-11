# DeepSeek 专家并行 (EP) 深度分析：从 V3 到 V4

> 本文档是对 [Expert Parallelism 基础](topics/expert-parallelism.md) 的进阶补充，结合 DeepSeek-V3/V4 论文和 DeepEP 源码，分析 EP 在真实大规模训练中的工程方案。
>
> 阅读前提：已理解 EP 的基本概念（All-to-All、dispatch/combine、三种通信方式），熟悉 Megatron-LM 中 EP 的配置方式。

---

## 目录

1. [DeepEP 通信库架构（共同基座）](#1-deepep-通信库架构)
2. [V3 EP 策略：DualPipe + 拓扑感知通信](#2-v3-ep-策略)
3. [V4 EP 策略：细粒度 Wave 调度 + MegaMoE](#3-v4-ep-策略)
4. [V3 → V4 技术演进总结](#4-v3--v4-技术演进总结)

---

## 1. DeepEP 通信库架构

DeepEP 是 DeepSeek 为 MoE 场景定制的集合通信库，也是 Megatron-LM `flex` dispatcher 的后端。它在 V2 版本后从 NVSHMEM 切换到 NCCL Gin 后端，完全重写了 EP 内核。

> 源码路径：`code/deepseek-ref/DeepEP`

### 1.1 整体设计：ElasticBuffer 统一接口

DeepEP V2 将所有 EP 操作统一到一个 `ElasticBuffer` 接口下，替代了 V1 中分离的"高吞吐"和"低延迟"两套 API。

```python
# 核心模型
from deep_ep import ElasticBuffer, EPHandle, EventOverlap

_buffer = ElasticBuffer(
    group,                                  # dist.ProcessGroup
    num_max_tokens_per_rank=num_max_tokens,
    hidden=hidden,
    num_topk=num_topk,
    use_fp8_dispatch=True,                  # FP8 dispatch + BF16 combine
    allow_hybrid_mode=True,                 # 混合模式：RDMA + NVLink
    allow_multiple_reduction=True,          # combine 中允许多次归约
    prefer_overlap_with_compute=True,       # 倾向用更少的 SM
    deterministic=False,                    # 确定性路由
)
```

**设计原则**：

- **弹性**：buffer 大小由 MoE 配置自动计算（`get_buffer_size_hint`），也支持手动指定
- **自分析**：SM 数量和 QP 数量通过带宽建模自动计算，无需手动调优
- **拓扑感知**：自动检测 NVLink 域和 RDMA 域的物理拓扑

### 1.2 拓扑感知：Scale-up vs Scale-out

DeepEP 的核心抽象是将通信分为两个域：

```
32 GPU = 4 节点 × 8 GPU/节点（H800 标准配置）:

  num_rdma_ranks    = 4   ← physical RDMA domain (跨节点 InfiniBand)
  num_nvlink_ranks  = 8   ← physical NVLink domain (节点内 NVSwitch)

  logical_rank_idx = scaleout_rank_idx × num_scaleup_ranks + scaleup_rank_idx
```

> DeepEP 区分为 logical 和 physical domain。Physical domain 由硬件拓扑决定（节点内 NVSwitch 连接 = NVLink domain，跨节点 = RDMA domain）。Logical domain 由 `get_logical_domain_size()` 根据 EP 并行配置自动计算，是 physical domain 的重新映射。

两个域完全在 GPU kernel 内部协同工作：

- **Scale-up (NVLink)**：通过对称内存 + TMA（Tensor Memory Accelerator）实现 GPU-to-GPU 直接访问
- **Scale-out (RDMA)**：通过 NCCL Gin 的 `ncclDeviceWrite` / `ncclDeviceRead` 原语实现跨节点数据传输

**为什么需要区分这两个域？** 因为它们的带宽特征截然不同：

| 域 | 带宽 | 延迟 | 语义 |
|----|------|------|------|
| NVLink（scale-up）| ~900 GB/s | ~1μs | 对称内存 + TMA load/store |
| RDMA（scale-out）| ~50 GB/s | ~10μs | NCCL Gin put/get |

如果不区分，统一走最慢的路径 → 浪费 NVLink 带宽。DeepEP 在单个 kernel 内同时使用两种路径，让 IB 和 NVLink 的通信**完全重叠**。

### 1.3 通信模式：Hybrid vs Direct

DeepEP 支持两种通信模式，由 `allow_hybrid_mode` 控制：

#### Direct 模式（纯 NVLink）

适用于单节点或 `ep_size ≤ 每节点 GPU 数`：

```
所有 rank 通过 NVLink 直接通信
每个 warp = 一个 channel
每个 channel 独立拥有一个 QP（Queue Pair）
```

代码路径：`dispatch.cuh` / `combine.cuh`

#### Hybrid 模式（RDMA + NVLink 混合）

适用于多节点：

```
  发送端                               接收端
  ┌──────────────┐                  ┌──────────────┐
  │ Notify warp  │ ── RDMA atomic ─→│ Notify warp  │  ← 通知接收方"数据准备好了"
  │ Scaleout warp│ ── RDMA put ────→│              │  ← 跨节点传输
  │ Forward warp │ ── NVLink TMA ──→│ Forward warp │  ← 节点内转发
  └──────────────┘                  └──────────────┘
```

三种 warp 角色在同一 kernel 内协同：

1. **Notify warp**：通过 RDMA atomic 操作更新远程标志位，告知"有数据到达"
2. **Scaleout warp**：负责跨节点 RDMA 传输——将数据写入远程 GPU 显存
3. **Forward warp**：处理节点内 NVLink 转发——从共享 buffer 读到本地，或从本地发到同节点其他 GPU

代码路径：`hybrid_dispatch.cuh` / `hybrid_combine.cuh`

**关键设计**：

```cpp
// hybrid_dispatch.cuh — warp 角色分配
if (warp_idx < kNumNotifyWarps) {
    // Notify: RDMA atomic 更新远程通知标志
} else if (warp_idx < kNumNotifyWarps + kNumScaleoutWarps) {
    // Scaleout: RDMA put 跨节点传输
} else {
    // Forward: NVLink TMA 节点内转发
}
```

通道数计算公式：

```
kNumChannels = kNumScaleoutWarps × kNumSMs
kNumMaxTokensPerChannel = ceil_div(kNumMaxTokensPerRank, kNumChannels)
```

每个 SM 上有多个 channel（每个 warp 一个），每个 channel 独立处理一部分 token，最大化并行度。

### 1.4 同步机制：GPU Barrier

DeepEP 实现了自己的 GPU 级 barrier（不依赖 CUDA `__syncthreads` 跨 block 同步），这是融合 kernel 正确性的基石。

```cpp
// barrier.cuh — 两种 barrier 实现

// NVLink barrier: 用 LSA (Load Store Access) 对称内存
// RDMA barrier: 用 NCCL Gin 的跨节点原语
template <bool kIsScaleupNVLink, ...>
void gpu_barrier(gin, workspace_layout, ...) {
    // Phase counter 机制：phase ∈ {0,1} 交替使用，避免等待的 ABA 问题
    // 每个 SM 一个线程参与，通过 atomicAdd 计数
    // 超时检测：clock64() 计时 → 超时 → ptx::trap() 挂起
}
```

**为什么需要自定义 barrier？**

- CUDA 的 `__syncthreads` 只在 block 内有效，不能跨 block（即跨 SM）
- NCCL 的 barrier 是 host-side 操作，需要 CPU 参与 → 无法在 kernel 内使用
- DeepEP 的融合 kernel 需要在 GPU kernel 内跨 SM、跨节点同步

Phase counter 机制避免 ABA 问题：

```
Round 0: phase=0, sign=1 → 所有 rank 写入 +1 → counter=num_ranks → 通过
Round 1: phase=1, sign=0 → 所有 rank 写入 -1 → counter=0 → 通过
Round 2: phase=0, sign=1 → ...（counter 的 phase 和 sign 交替翻转）
```

### 1.5 EPHandle：路由元数据管理

`EPHandle` 是 dispatch 返回的元数据对象，combin 用它来逆路由 token。关键字段：

```python
class EPHandle:
    # 路由信息
    topk_idx: Tensor              # [num_tokens, num_topk] 路由到的专家索引
    num_recv_tokens_per_expert_list: list  # 每个专家的 token 数（CPU side）

    # 前缀和（GPU side），用于确定 token 在 buffer 中的偏移
    psum_num_recv_tokens_per_scaleup_rank: Tensor  # 每个 scaleup rank 收到的 token 前缀和
    psum_num_recv_tokens_per_expert: Tensor         # 每个专家收到的 token 前缀和

    # Buffer 槽位
    recv_src_metadata: Tensor     # 源 token 索引和 buffer slot 索引
    dst_buffer_slot_idx: Tensor    # dispatch 的目标 buffer slot

    # Hybrid 模式专用
    token_metadata_at_forward: Tensor  # 跨节点转发的 token 元数据
    channel_linked_list: Tensor        # 每个 channel 的链表（跨节点转发队列）
```

**推理中的 handle 缓存**：当门控决策不变时（常见于 decode 阶段），可以复用缓存 handle 跳过 CPU 同步：

```python
# 训练：每次 dispatch 都重新计算路由布局
recv_x, recv_topk_idx, recv_topk_weights, handle, event = _buffer.dispatch(
    x, topk_idx=topk_idx, topk_weights=topk_weights, num_experts=num_experts)

# 推理 decode（handle 缓存）：跳过 CPU sync
if cached_handle is not None:
    recv_x, _, _, handle, event = _buffer.dispatch(
        x, handle=cached_handle)  # 不传 topk_idx
```

### 1.6 精度支持：FP8 Dispatch + BF16 Combine

```python
# FP8 dispatch: x 传 tuple (data_fp8, scale_factors)
recv_x, recv_topk_idx, recv_topk_weights, handle, event = _buffer.dispatch(
    x=(x_fp8, x_scale),   # FP8 数据 + per-tile scale factors
    ...
)

# BF16 combine: x 传普通 tensor
combined_x, event = _buffer.combine(x=x_bf16, handle=handle)
```

**为什么 dispatch FP8 而 combine BF16？** dispatch 发的是 GEMM 输入，对精度相对宽容；combine 发的是专家输出，要做加权求和并入 residual stream，精度敏感。这个设计与 V4 混合精度方案一致。

在 kernel 内部，FP8 数据以 `sf_pack_t` 形式打包传输——scale factor 和数据打包在一起，避免额外传输。

### 1.7 Buffer 大小与 SM 分配

DeepEP V2 通过带宽建模自动计算最优 SM 数量，替代了 V1 的 auto-tuning：

```python
def get_theoretical_num_sms(self, num_experts, num_topk, ...):
    # 1. 计算期望的 top-k scale-out rank 数（概率组合）
    num_expected_scaleout_topk = get_expected_topk(num_scaleout_ranks)
    num_expected_topk = get_expected_topk(num_ranks)

    # 2. 计算每种操作的数据量
    sm_read = 1 / num_expected_topk           # 读 token
    sm_write = 1 / num_expected_topk          # 写 send buffer
    rdma_traffic = ...                        # 跨节点数据量
    nvlink_traffic = ...                      # 节点内数据量

    # 3. 找瓶颈: RDMA 和 NVLink 哪个先饱和
    bounded_traffic, bounded_gbs = (rdma_traffic, rdma_gbs) \
        if (rdma_traffic/rdma_gbs) > (nvlink_traffic/nvlink_gbs) \
        else (nvlink_traffic, nvlink_gbs)

    # 4. SM = max(瓶颈带宽/数据量 × 读带宽比, 瓶颈带宽/数据量 × 写带宽比)
    num_sms = max(
        bounded_gbs / bounded_traffic * sm_read / sm_read_gbs,
        bounded_gbs / bounded_traffic * sm_write / sm_write_gbs,
    )
    return align(max(4, ceil(num_sms * 1.25)), 2)  # 1.25× margin, 偶数对齐
```

SM 使用量的实际对比（来自 DeepEP README）：

| 场景 | 拓扑 | Dispatch BW | Combine BW | SM 用量 |
|------|------|-------------|------------|---------|
| 训练（CX7 IB）| EP 8×2 | 90 GB/s（RDMA）| 81 GB/s（RDMA）| 12 |
| 训练（CX7 IB）| EP 8×4 | 61 GB/s（RDMA）| 61 GB/s（RDMA）| 6 |
| 节点内（NVLink）| EP 8 | 726 GB/s（NVLink）| 740 GB/s（NVLink）| 64 (最大性能) |
| 节点内（NVLink）| EP 8 | 643 GB/s（NVLink）| 675 GB/s（NVLink）| 24 (最小 SM) |

> V2 相比 V1 的最显著变化：训练场景 SM 用量从 24 降至 4-6，同时性能持平或更优。

### 1.8 JIT 编译系统

DeepEP 所有 kernel 在运行时通过轻量级 JIT 编译，无需预装 CUDA toolchain：

```
Python API → JIT module → 生成 .cu 源码 → nvcc 编译 → PTX → cubin → 加载
                              ↓
                    缓存到 $HOME/.deep_ep
```

为什么不预编译？因为 kernel 的模板参数（`kNumSMs, kNumRanks, kNumHiddenBytes, kNumExperts, kNumTopk, ...`）在每个部署场景下不同，预编译会导致组合爆炸。JIT 在首次运行时根据实际配置编译，后续从缓存加载。

关键环境变量：

- `EP_JIT_CACHE_DIR`：缓存目录，默认 `$HOME/.deep_ep`
- `EP_JIT_DEBUG`：打印编译命令用于调试
- `EP_JIT_DUMP_SASS`：导出 SASS 汇编用于性能分析

---

## 2. V3 EP 策略

> 论文来源：[DeepSeek-V3 §3.2](papers-zh/DeepSeek-V3.md#32-训练框架)
> 对应 DeepEP 版本：V1（legacy，NVSHMEM 后端）

### 2.1 V3 的 EP 配置

> 如需回看 EP 的基础配置方式（EP + TP/DP 如何组合、Megatron 参数等），见 [Expert Parallelism §3-4](topics/expert-parallelism.md#三megatron-lm-中怎么配-ep)。

```
V3 训练:
  num_experts = 256 (路由) + 1 (共享)
  topk = 8 (路由) + 1 (共享) = 9
  ep_size = 64 (跨 8 节点)
  tp_size = 1 (专家层不用 TP！)
  pp_size = 16
  world_size = 2048 / (64 × 16) = 2 DP

V3 推理 prefilling:
  ep_size = 320, 每 GPU 1 个专家 + 冗余专家
```

**关键设计决策**：EP=64 且 TP=1。这意味着每个 rank 持有 256/64 = 4 个专家的完整矩阵（不切分）。单个专家的矩阵不大（4× 的 d_ff），不需要 TP 的额外切分。TP 的通信开销被完全省掉，代价是 EP 通信跨节点 → 必须用 DualPipe 来隐藏。

### 2.2 节点限制路由

V3 的 MoE 层使用了**节点限制路由（Node-Limited Routing）**：

```
每个 token 最多路由到 M = 4 个节点
选择规则：按每个节点内专家的最高 K/M 个亲和度分数之和排序，取 top-M 节点
```

**为什么 M=4？** NVLink 带宽（160 GB/s H800）约为 IB（50 GB/s）的 3.2 倍。如果每个 token 路由到更多节点，IB 流量占比增大 → 通信瓶颈加剧。M=4 意味着 topk=8 个专家平均每节点 2 个，充分利用 NVLink 而不过度占用 IB。

在 DeepEP 的 SM 计算中，`num_expected_scaleout_topk` 量化的就是这个效果——路由到多少个不同的跨节点 rank。

### 2.3 DualPipe：在 PP 层面掩盖 EP 通信

DualPipe 是 V3 最核心的 EP 通信隐藏策略。它的核心问题是：**EP=64 跨节点时，通信时间 ≈ 计算时间（1:1），如果串行执行 → 一半 GPU 时间在等通信。**

#### 原理

将单个 Transformer block 的前向+反向分解为细粒度组件并重新排列：

```
一个 block 对 = {
    Forward:  Attention → Dispatch → MLP → Combine
    Backward: Input Grad(Attention) → Weight Grad(Attention)
            → Input Grad(MLP) → Weight Grad(MLP)
}
```

DeepEP V1 内核对应的 Dispatch/Combine 通信在这里发生。DualPipe 利用 V1 的 NVSHMEM 通信后端，将每个组件的执行时间与另一个组件的通信时间对齐。

关键重叠策略（图 4 的核心）：

```
时间 ──────────────────────────────────────────────→
         ┌──── 前向 Attention ────┐
PP 通信: ░░░░░░░░░░░░░░░░░░░░░░░░          ← PP bubble 也被填了计算
         ┌──── Dispatch ──────────┐
         ┌── 专家 GEMM ───────────┐
                   ┌── Combine ───┐
                        ┌── 后向 ──┐  ← 前向 Dispatch 与后向 compute 重叠

结果：All-to-All 和 PP 通信都可以被完全隐藏。
```

#### Bubble 对比

| 方法 | Bubble | 参数副本 | 激活内存 |
|------|--------|---------|---------|
| 1F1B | (PP-1)(F+B) | 1× | 1 |
| ZB1P | (PP-1)(F+B-2W) | 1× | 1 |
| **DualPipe** | **(PP/2-1)(F&B+B-3W)** | **2×** | 1 + 1/PP |

DualPipe 的代价是保留**两份参数副本**——但因为 EP 很大（64），每 rank 持有的参数已经是完整模型的 1/64 → 两份副本也不多。

#### V4 对比

V4 不再依赖 DualPipe 级别的粗粒度重叠，转而使用 MoE 层内部的细粒度 Wave 调度（见 §3.1），这两个方案本质是同一问题在不同粒度上的解法。

### 2.4 V3 的跨节点 A2A 内核设计（对应 DeepEP V1）

V3 论文描述的内核用了 **warp specialization** 技术（即 DeepEP V1 的 legacy kernel）：

```
20 SM 划分为 10 个通信 channel

Dispatch 过程中，每个 channel 内的 warp 分工:
  Warp Group A: IB 发送 (RDMA put)
  Warp Group B: IB → NVLink 转发 (共享内存中转)
  Warp Group C: NVLink 接收 (TMA load)

Combine 过程中:
  Warp Group A: NVLink 发送
  Warp Group B: NVLink → IB 转发 + 累加
  Warp Group C: IB 接收 + 累加
```

**与 DeepEP V2 的对应**：

- V1 的 IB warp → V2 的 Scaleout warp（NCCL Gin 替代 NVSHMEM）
- V1 的转发 warp → V2 的 Forward warp
- V1 的 NVLink warp → V2 合并到 Forward warp + TMA

V1 需要 20 SM，V2 降至 4-6 SM（Hybrid 模式）或 24 SM（Direct 最大性能模式），吞吐量反而提升 1.3×。

---

## 3. V4 EP 策略

> 论文来源：[DeepSeek-V4 §3.1](papers-zh/DeepSeek-V4-technical-report.zh.md#31-专家并行中的细粒度通信-计算重叠)
> 对应 DeepEP 版本：V2（NCCL Gin 后端）

### 3.1 V4 的 EP 配置

```
V4-Pro:
  num_experts = 384 (路由) + 1 (共享)
  topk = 6
  h = 7168, d_ff = 3072
  FP8 dispatch + BF16 combine

V4-Flash:
  d_ff = 2048 (更小的中间维度)
```

**V4 相对于 V3 的关键变化**：**移除了节点限制路由**。V3 的 M=4 节点限制意味着 topk=8 专家最多分布在 4 个节点，这约束了路由灵活性。V4 通过更高效的 EP 方案，让 topk=6 专家可以自由选择任何节点，通信开销仍能被计算隐藏。

### 3.2 计算-通信比分析：为什么可以移除节点限制

V4 论文从理论上解释了为什么通信可以被计算完全隐藏。

> 相关背景公式推导见 [Expert Parallelism §2.1](topics/expert-parallelism.md#21-参数量计算量通信量)，此处只做简要回顾。

**核心判定公式**：

$$\frac{C}{B} \le 2 \cdot d_{ff}$$

其中 C = GPU 峰值算力（FP8），B = 互联带宽。推导：

```
每个 token-expert pair:
  V_comp = 6·h·d_ff (3 个 GEMM 的 FLOPs)
  V_comm = 3h 字节 (FP8 dispatch + BF16 combine)

Load_Ratio = V_comp / V_comm = 2·d_ff
```

**含义**：硬件每 Byte/s 带宽能提供的算力 ＜ 模型每 Byte 通信绑定的计算量 → 通信被计算完全掩盖。

**V4-Pro 场景（d_ff=3072）**：

| 链路 | Supply_Ratio (C/B) | Load_Ratio (2·d_ff) | 结论 |
|------|-------------------|---------------------|------|
| NVLink | ~2200 | 6144 | ✅ 计算 > 通信 → 可隐藏 |
| InfiniBand | ~40000 | 6144 | ❌ 算力供给过剩 → 通信成瓶颈 |

**V4-Flash（d_ff=2048）**：Load_Ratio=4096，IB 侧的 mismatch 更严重。d_ff 越小，通信占比越高。

节点内 NVLink 天然满足条件 → 单节点 EP 无需重叠。跨节点 IB 不满足 → 必须用 Wave 调度做细粒度重叠。这与 V3 用 DualPipe 处理跨节点通信的逻辑一致，但 V4 用更细粒度的方案。

### 3.3 Wave 调度：专家级细粒度通信-计算重叠

V4 提出了一种**比 DualPipe 更细粒度**的重叠策略——将专家按波次（wave）拆分并流水线执行。这本质上是将 MoE 层的一次完整执行（Dispatch → GEMM → Combine）拆成多个子任务，让它们在时间上错开，形成流水线。

#### 3.3.1 Wave 划分：怎么拆

设每个 EP rank 持有 `E_local = num_experts / ep_size` 个专家。将这些本地专家拆分为 W 个 wave，每个 wave 包含 `E_local / W` 个专家。

**具体例子（V4-Pro，ep_size=32）**：

```
num_experts = 384, ep_size = 32
E_local = 384 / 32 = 12 个专家/rank

W = 3 个 wave:
  Wave 0: Expert 0, 1, 2, 3     ← 本地专家索引
  Wave 1: Expert 4, 5, 6, 7
  Wave 2: Expert 8, 9, 10, 11

每个 wave 持有 4 个专家的权重矩阵
```

**W 选多少？** 这是一个权衡：

- W 太小（如 W=1）→ 退化为传统 All-to-All → 无重叠
- W 太大（如 W=E_local，每 wave 1 个专家）→ 流水线启动/排空开销大，且单个专家的 GEMM 太小 → 不能充分占用 Tensor Core
- 实践中 W 由 `E_local` 和每个专家期望收到的 token 数决定：如果每个专家收到的 token 太少（<128），合并到同一 wave 以保证 GEMM tile 足够大

在 DeepGEMM MegaMoE 中，`expert_alignment` 参数间接控制 wave 内的专家粒度——对齐越大，每 wave 处理的 token 越粗，越接近传统非 wave 模式。

#### 3.3.2 流水线的三个阶段

Wave 调度将每 wave 的处理拆为三个流水线阶段，由不同的 warp group 负责：

```
┌──────────────────────────────────────────────────────────────┐
│                    单个 MegaMoE kernel                       │
│                                                              │
│  Warp Group A (通信 warp):  负责 Dispatch — 从远程拉 token   │
│  Warp Group B (计算 warp):  负责 Linear-1 + SwiGLU + Linear-2 │
│  Warp Group C (通信 warp):  负责 Combine — 将结果发回远程     │
│                                                              │
│  三组 warp 同时运行，处理不同的 wave                           │
│  Group A 在处理 Wave k+1 的 dispatch                          │
│  Group B 在处理 Wave k   的 GEMM                   ← 重叠!   │
│  Group C 在处理 Wave k-1 的 combine                          │
└──────────────────────────────────────────────────────────────┘
```

#### 3.3.3 逐 wave 执行时间线

下面展开具体的时间线，看数据如何在各 wave 之间流转：

```
时间 ────────────────────────────────────────────────────────────────→

阶段:        │<── 流水线预热 ──>│<────── 稳定状态 ──────>│<─ 排空 ──>│
             │                  │                         │           │
Wave 0:      │                  │                         │           │
  Dispatch   ████████           │                         │           │
  Linear-1          ████████    │                         │           │
  SwiGLU                  ██    │                         │           │
  Linear-2                 ████ │                         │           │
  Combine                      ████████                   │           │
             │                  │                         │           │
Wave 1:      │                  │                         │           │
  Dispatch         ████████     │                         │           │
  Linear-1                ████████                        │           │
  SwiGLU                        ██                       │           │
  Linear-2                       ████████                 │           │
  Combine                                ████████         │           │
             │                  │                         │           │
Wave 2:      │                  │                         │           │
  Dispatch               ████████                         │           │
  Linear-1                              ████████          │           │
  SwiGLU                                      ██          │           │
  Linear-2                                     ████████   │           │
  Combine                                              ████           │
             │                  │                         │           │
             ▼                  ▼                         ▼           │
        3 种操作串行       3 种操作同时执行              逐渐退出      │
        (仅 Wave 0)       (不同 wave 的 Dispatch /        (最后 wave   │
                           Compute / Combine 并行)        的收尾)      │
```

**关键同步点**：

1. **Dispatch → Linear-1**：当前 wave 的所有 token 到达本地 buffer → 通信 warp 通过 named barrier 通知计算 warp "数据就绪"
2. **Linear-2 → Combine**：当前 wave 的 GEMM 完成 → 计算结果在 shared memory / register → 通信 warp 开始 RDMA write / NVLink store 发回源 rank
3. **跨 wave 不直接同步**：Wave k 的 Linear-2 完成不等 Wave k+1 的 Dispatch 完成 —— 它们在不同的 warp group 上独立推进

#### 3.3.4 Warp specialization 在 Wave 调度中的角色

Wave 调度依赖 warp specialization 实现三类 warp 的并行执行。以 MegaMoE 在 H100（132 SM）上的配置为例：

```
H100, 132 SM, EP=32, W=3:

  SM 分配:
    通信 SM (Dispatch):  ~12 SM  ← 每个 SM 1 个 warp = 12 个通信 channel
    计算 SM (GEMM):      ~108 SM ← Tensor Core 密集
    通信 SM (Combine):   ~12 SM  ← 复用或独立分配

  Warp 角色:
    - Dispatch warp: 每个 warp = 1 个 RDMA/NVLink channel
      负责从远程 GPU 拉取对应 wave 的 token
    - Compute warp: 每个 warp 处理一个 GEMM tile
      负责 Linear-1(tile) → SwiGLU → Linear-2(tile)
    - Combine warp: 将 GEMM 输出写回远程 GPU 的 buffer

  执行模型（persistent kernel）:
    所有 warp 从同一个 grid launch，kernel 不退出的持续循环:
      while (还有未处理的 wave):
        if (我是通信 warp): 处理当前 wave 的 dispatch/combine
        if (我是计算 warp): 处理当前 wave 的 GEMM
        通过 device-wide barrier 同步 → 推进到下一个 wave
```

**Named barrier**（CUDA Hopper 的 `mbarrier`）在这里扮演核心角色——它允许同一个 SM 上的不同 warp 在 shared memory 中低成本同步，而不需要通过 global memory 的 atomic 操作。

在 DeepEP 的 dispatch kernel 中可以看到这个模式：

```cpp
// dispatch.cuh — 通信 warp 内部的 named barrier 同步
if (warp_idx < kNumNotifyWarps) {
    // Notify warps: 等所有 notify warp 都完成 RDMA atomic
    ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);
    // 现在可以安全地更新远程标志位
    ...
    ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);
}

// 计算 warp 收到信号后开始 GEMM
// ... (在 MegaMoE 中，这一步由 DeepGEMM 的 GEMM warp 处理)
```

#### 3.3.5 Pull-based 通信在 Wave 调度中的作用

Wave 调度天然需要 pull 模式（§3.4）。原因：如果发送方逐 wave 推送 token，每个 wave 都需要一次 "wave X 的 token 准备好了" 的通知 → 通知延迟 × W 次 → 流水线根本跑不起来。

Pull 模式让接收方完全控制节奏：接收方知道本地有哪些专家（按 wave 组织），主动按 wave 顺序去远程拉 token：

```
对于 EP rank 0，持有 Expert 0-11:

  当前需要处理 Wave 1 (Expert 4-7):
    1. Dispatch warp 主动发起 RDMA read / NVLink TMA load:
       - 从 rank i 拉取路由到 Expert 4 的 token
       - 从 rank j 拉取路由到 Expert 5 的 token
       - ...
    2. 数据到达 → named barrier 释放 → 计算 warp 开始 GEMM
    3. 同时 Dispatch warp 已经开始拉 Wave 2 (Expert 8-11) 的数据
    4. Combine warp 在发送 Wave 0 (Expert 0-3) 的计算结果
```

#### 3.3.6 与 DeepEP dispatch/combine 的配合

在 V4 的实际实现中，Wave 调度**不完全在 DeepEP 层面**——DeepEP 负责单个 dispatch/combine 操作的高效通信，Wave 调度由 MegaMoE 在更上层编排：

```
┌─ MegaMoE Kernel ───────────────────────────────────────┐
│                                                        │
│  for wave_id in range(num_waves):                      │
│                                                        │
│    # Phase 1: Dispatch（如果有上一个 wave 的 combine）  │
│    if wave_id > 0:                                     │
│      deep_ep combine(outputs[wave_id-1])  ← 发回上一波 │
│                                                        │
│    # Phase 2: Dispatch（拉取当前 wave 的 token）         │
│    deep_ep dispatch(get_tokens_for_experts(expert_ids[wave_id])) │
│                                                        │
│    # Phase 3: GEMM（与 Phase 1/2 的通信 warp 并行）      │
│    deep_gemm fp8_gemm(l1_weights[wave_id], tokens)     │
│    swiglu(...)                                         │
│    deep_gemm fp8_gemm(l2_weights[wave_id], activated)  │
│                                                        │
│  # 最后 wave 的 combine                                 │
│  deep_ep combine(outputs[-1])                          │
│                                                        │
└──────────────────────────────────────────────────────────┘
```

**而 MegaMoE 的"融合"更进一步**：上面的伪代码是逻辑视图，实际 MegaMoE 将所有阶段编译到**同一个 CUDA kernel** 中，通过 warp specialization 让通信 warp 和计算 warp 在同一 grid launch 内并行推进不同 wave。这样：
- 中间数据（dispatch 结果、GEMM tile、SwiGLU 输出）全在 shared memory + register 中流转，不写 HBM
- 无需多次 kernel launch → 省 CPU→GPU 调度延迟
- Wave 之间的同步通过 on-chip named barrier，延迟 ~10 cycles（对比 global memory atomic ~500 cycles）

#### 3.3.7 与 DualPipe 的区别

| 维度 | DualPipe（V3）| Wave 调度（V4）|
|------|--------------|----------------|
| 粒度 | Transformer block 级别 | 专家 wave 级别（~4 个专家/wave）|
| 流水线 | 2 向（前缀/后缀微批次）| 3 级（Dispatch/Compute/Combine）|
| 依赖 | 需要 PP > 1 | **不需要 PP**，单层就能做 |
| 通信操作 | 整层 token 一起发 | **按 wave 分发**，每次只传部分专家需要的 token |
| 实现方式 | 调度器层面重排 block 顺序 | Kernel 内部 warp specialization |
| 同步机制 | CUDA stream 同步 | On-chip named barrier（~10 cycles）|
| 极端 batch 性能 | 小 batch 效果差 | 小 batch 加速更明显（1.96×）|

**Wave 调度的关键优势**：不需要 PP。这意味着即使 `PP=1`，也能享受通信-计算重叠——这对推理（PP 通常是 1）至关重要。

#### 3.3.8 为什么小 batch 加速更明显？

推理（特别是 RL rollout）中 batch 往往很小。在这种情况下：

- **传统做法**：所有专家一起做 All-to-All → 通信时间固定，计算量随 batch 线性减小 → 通信占比飙升，GPU 大量时间空转等待
- **Wave 调度**：每 wave 只传输一小部分 token → 通信延迟分散到多个 wave → 每个 wave 的 compute 仍能掩盖其 dispatch/combine → 减少空转

**加速比的来源**：

```
传统（串行）:
  Total = Dispatch_all + GEMM_all + Combine_all
        = D + G + C

Wave 调度（3 wave, 理想重叠）:
  Total ≈ max(D/3 + C/3, G/3) + 2 × max(D/3, C/3)  ← 预热 + 排空
        ≈ D/3 + G/3 + C/3  （稳定状态下完全重叠）
        ≈ (D + G + C) / 3

  理想加速比 ≈ 3×
  实际 1.50~1.96×  ← 受限于预热/排空开销 + 细粒度 warp 调度的不完美负载均衡
```

大 batch 时 `G` 远大于 `D+C`，重叠收益有限（本来就计算 bound）。小 batch 时 `D+C` 接近甚至超过 `G`，Wave 调度的重叠收益才真正体现出来。

### 3.4 Pull-based Dispatch

V4 的 dispatch 采用 **pull（拉取）模式**，而非传统的 push（推送）：

```
Push (传统):
  发送方: 准备好数据 → 通知接收方 → 等待确认 → 发送
  问题: 通知延迟在细粒度场景下成为瓶颈

Pull (V4):
  接收方: 主动从远程 GPU 读数据 (RDMA read / NVLink TMA load)
  优势: 无需通知 → 零通知延迟 → 适合细粒度调度

  在 kernel 中的表现:
    Forward warp: TMA load from peer GPU's symmetric memory
    Scaleout warp: ncclDeviceRead from remote GPU
```

Pull 模式天然适合 Wave 调度：每个 wave 的接收方知道需要哪些专家的 token → 直接去读，不需要发送方逐波次通知。

在 DeepEP 源码中，这体现在 barrier 之后的 forwarding 逻辑——接收方的 Forward warp 在 barrier 通过后直接通过 TMA load 拉取数据。

### 3.5 MegaMoE：单融合巨型核

V4 开源了 **MegaMoE**（作为 DeepGEMM 组件），将 MoE 层的**全部操作融合进一个 kernel**：

```
传统做法（多个 kernel launch）:
  permute → dispatch → GEMM(Linear-1) → SwiGLU → GEMM(Linear-2) → combine → unpermute
    ↑         ↑            ↑               ↑           ↑            ↑          ↑
    Kernel1   Kernel2      Kernel3        Kernel4     Kernel5     Kernel6   Kernel7
    每个 kernel 结果写 HBM，下一个再读出来

MegaMoE（一个 kernel）:
  ┌─ 单个 kernel ───────────────────────────────────────────┐
  │                                                          │
  │  SMEM: [token重排] → [dispatch] → [GEMM-1 tile] →       │
  │        [SwiGLU tile] → [GEMM-2 tile] → [combine] →      │
  │        [结果聚合]                                        │
  │                                                          │
  │  中间数据始终在 shared memory + register 中流转           │
  └──────────────────────────────────────────────────────────┘
```

**加速来源**：

1. 省掉 6 次 kernel launch 开销（每次 ~5μs CPU→GPU 延迟）
2. 省掉 6 次中间结果写回 HBM（HBM 带宽是 GPU 最稀缺的资源）
3. Wave 调度天然融入 kernel —— 每个 wave 就是一个 warp group 的迭代
4. 省掉 implicit barrier——传统做法每个 kernel 后自动同步，MegaMoE 内部按需同步

**性能**：
- 通用推理：1.50-1.73× vs 非融合基线
- RL rollout / 延迟敏感：高达 1.96×

### 3.6 批量不变性与确定性

V4 在 EP 通信中实现了两种关键正确性属性：

#### 批不变性（Batch Invariance）

**定义**：任意 token 的输出在位级别相同，无论它在批次中的位置如何。

对 EP 的影响：
- Dispatch：每个 token 独立路由，天然批不变
- Combine：不同 rank 发回来的结果做 weighted sum → 累加顺序必须与位置无关

V4 采用 **buffer 隔离 + 确定性归约**。在 DeepEP 中，这由 `deterministic=True` 参数和 `dispatch_deterministic_prologue.cuh` 控制——每个 source rank 的数据写到接收 buffer 的预分配区域，避免动态竞争。

#### 确定性

V4 论文特别提到 MoE 反向的确定性问题：

```
问题: 多个 SM 同时向同一接收 rank 写数据时，
      atomicAdd 的浮点累加顺序不确定 → 反向梯度不确定

解决:
  1. 每个 rank 内部：token 顺序预处理 → 确定每个 token 的 buffer 位
  2. 跨 rank：buffer 隔离 → 不同源 rank 写不同区域
  3. 全局归约：分区域累加后确定性求和
```

在 DeepEP 源码中，`dispatch_deterministic_prologue.cuh` 实现了 token 顺序预处理机制——在 dispatch 之前先对 token 按目标 rank+expert 排序，保证无论 batch 顺序如何，同一组 token 的写入位置确定。

---

## 4. V3 → V4 技术演进总结

### 4.1 演进路线图

```
V3 (2024)                            V4 (2025)
─────────                            ─────────
DualPipe (PP 级重叠)          →     Wave 调度 (专家级重叠)
节点限制路由 (M=4)             →     移除节点限制
Push dispatch                  →     Pull dispatch
DeepEP V1 (NVSHMEM, 20 SM)    →     DeepEP V2 (NCCL Gin, 4-6 SM)
分离 kernel (7 launches)       →     MegaMoE (1 kernel 融合)
非确定性                       →     批不变 + 确定性
```

### 4.2 核心设计思想的演进

**从粗粒度到细粒度**：V3 的 DualPipe 在 Transformer block 级别做重叠，依赖 PP 存在且需要 2× 参数。V4 的 Wave 调度在专家级别做重叠，不需要 PP，更适合推理和小 batch。

**从 Push 到 Pull**：Push 需要发送方通知 → 接收方确认 → 再传输 → 细粒度时延迟不可忍受。Pull 让接收方直接去读 → 零通知延迟 → 天然适合波次调度。

**从专用通信库到通用通信基座**：DeepEP V2 不仅是 EP 通信库——它还支持 PP（send/recv）、CP（0 SM copy engine）、Engram（远程 KV cache 获取）、AGRS（all-gather reduce-scatter）。这是一套完整的分布式训练通信基座。

### 4.3 硬件设计建议（来自 V4 论文）

> 这些建议基于 DeepSeek 在 EP 通信优化上的实践经验，对 AI 硬件设计有参考价值。

1. **计算-通信比平衡点**：每 GB/s 互联带宽应匹配 ~6000 FLOP/s 的算力（FP8）。一旦达到此阈值，增加带宽的边际收益递减；硅片面积应用于提升算力。

2. **功耗预算**：极端核融合使计算、内存、网络同时高负载 → 功耗限制成为关键瓶颈。未来硬件需要为此类 fully concurrent workloads 提供足够功率余量。

3. **通信原语**：硬件支持低延迟跨 GPU 信令将使 Push 模式可行，实现更自然的通信模式。

4. **激活函数**：建议用无指数/除法的低成本逐元素激活函数替代 SwiGLU → GEMM 后处理不被阻塞 → 整体吞吐提升。

### 4.4 V3 和 V4 方案的选择建议

| 场景 | 推荐方案 | 原因 |
|------|---------|------|
| 单节点 EP（≤8 GPU）| DeepEP Direct 模式 | NVLink 带宽充足，无需重叠 |
| 多节点训练，PP>1 | DualPipe + DeepEP Hybrid | PP 级重叠足够，工程成熟 |
| 多节点训练，PP=1 | Wave 调度 + MegaMoE | 无 PP 依赖的单层重叠 |
| 推理 prefill | DeepEP Hybrid + 双微批次重叠 | 吞吐优先 |
| 推理 decode | Wave 调度 + handle 缓存 | 延迟优先，小 batch |
| RL rollout | Wave 调度（MegaMoE）| 延迟敏感，长尾小 batch |

---

## 5. 延伸阅读

- 📖 [Expert Parallelism 基础](topics/expert-parallelism.md) — EP 概念、三种通信方式、Megatron 配置
- 📖 [DeepSeek-V3 中文笔记 §3.2](papers-zh/DeepSeek-V3.md#32-训练框架) — DualPipe + A2A 内核
- 📖 [DeepSeek-V4 技术报告 §3.1-3.3](papers-zh/DeepSeek-V4-technical-report.zh.md#31-专家并行中的细粒度通信-计算重叠) — Wave 调度 + 批量不变性
- 📖 [DeepEP GitHub](https://github.com/deepseek-ai/DeepEP) — 源码
- 📖 [混合精度训练](notes/mixed-precision-training/) — FP8 dispatch / BF16 combine 的精度选择
- 📖 V4 MegaMoE 论文（待发布）— 单融合巨型核的完整设计

---

> **维护说明**：本文档聚焦于 DeepSeek 的 EP 工程实现和 V3→V4 演进。EP 的基础概念、Megatron 配置、并行组合等见 [topics/expert-parallelism.md](topics/expert-parallelism.md)。如发现本文档与最新论文/代码不一致，优先以源码为准。
