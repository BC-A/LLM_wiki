# DeepSeek-V4 混合精度训练：FP8 训练 + FP4 后训练量化

> **一句话概述**: DeepSeek-V4 在 V3 的 FP8 训练基础上，引入 **FP4 量化感知训练（QAT）** 用于 MoE 专家权重和 CSA 索引器，配合 **UE8M0 幂次缩放**、**KV 混合存储（BF16 RoPE + FP8 非 RoPE）**、**Hadamard 旋转** 等技术，形成覆盖训练-后训练-推理的全链路低精度体系。
> **参考论文**: DeepSeek-V4 Technical Report | [arXiv](https://arxiv.org/abs/2505.18206)
> **参考代码**: [code/deepseek-ref/V4/inference/](../../code/deepseek-ref/V4/inference/)
> **前置阅读**: [DeepSeek-V3 FP8 训练](./deepseek-v3-fp8-training.md) — V4 继承了 V3 的 FP8 GEMM 方案，本文重点讲 V4 的增量创新，重复内容简略带过。

---

## 一、总览：V4 的精度全景图

### 1.1 与 V3 的差异速览

V3 只有一套精度方案：**FP8 用于 FFN/MoE GEMM，其余 BF16/FP32**。
V4 有三套精度方案，用于不同阶段：

| 阶段 | 精度方案 | 用途 |
|------|----------|------|
| **预训练** | FP8（继承 V3） | FFN/MoE GEMM 前向+反向 |
| **后训练 QAT** | FP4 → FP8 无损反量化 | MoE 专家权重（减少显存）+ CSA 索引器 QK（加速长上下文注意力） |
| **推理** | KV 混合精度 + FP4 原生权重 | KV Cache 近半节省 + 专家权重直接 FP4 加载 |

### 1.2 一张表：每个组件用什么精度

| 组件 | V3 精度 | V4 精度 | 变化 & 原因 |
|------|---------|---------|------------|
| **FFN/MoE GEMM** | FP8 E4M3（1×128 / 128×128 tile） | FP8 E4M3（继承） | 无变化，V3 方案已成熟 |
| **MoE 专家权重（存储）** | FP8 E4M3 | **FP4 E2M1（QAT 后）** ⚡ 新 | 显存减半，未来硬件 FP4×FP8 吞吐 1.33× |
| **QAT 中专家 GEMM 计算** | — | **FP8×FP8**（FP4 weight → FP8 dequant） | FP4→FP8 反量化无损，完全复用 FP8 训练框架 |
| **MLA Attention GEMM** | BF16 | BF16 | 不变 |
| **KV Cache（RoPE 维度）** | BF16 | BF16 | 位置编码精度敏感，保持 BF16 |
| **KV Cache（非 RoPE 维度）** | BF16 | **FP8（1×64 tile）** ⚡ 新 | 近半 KV Cache 节省 |
| **CSA 索引器 QK** | N/A（V4 新组件）| **FP4** ⚡ 新 | QK 全路径 FP4 缓存/加载/计算，加速长上下文注意力（详见 §七） |
| **CSA 索引器 索引分数** | N/A | FP32 → **BF16**（QAT 后）| top-k 选择器 2× 加速，召回率 99.7%（详见 §七） |
| **Embedding** | BF16 | BF16 | 不变 |
| **RMSNorm** | BF16 | BF16（FP32 计算内部） | 不变 |
| **Softmax / SiLU** | BF16 / FP32 | BF16 / FP32 | 不变 |
| **MoE Gate** | BF16 | BF16 | 不变 |
| **Output Head + Loss** | BF16 + FP32 | BF16 + FP32 | 不变 |

### 1.3 精度体系的层次结构

```
V4 精度体系（由粗到细）

预训练（FP8）
  └── FFN/MoE GEMM: FP8 E4M3，UE8M0 幂次缩放
  └── 其余所有 op: BF16/FP32（与 V3 相同）
  └── KV Cache 模拟: BF16 (RoPE) + FP8 (非 RoPE)

后训练 QAT（FP4 引入）
  └── MoE 专家权重: FP32 master → FP4 quant → FP8 dequant → FP8 GEMM
  └── CSA 索引器 QK: FP4 全路径
  └── 索引分数: FP32 → BF16

推理部署
  └── 专家权重: 直接 FP4 加载（非模拟）
  └── QK 计算: 直接 FP4
  └── KV Cache: 混合 BF16/FP8 存储
```

---

### 1.4 关键技术一览（阅读路线图）

V4 在 V3 的 FP8 训练基础上引入了 **5 项关键技术**。在深入细节之前，先用一句话说清每项技术「解决什么问题、怎么做」：

| # | 技术 | 解决的问题 | 一句话 |
|---|------|-----------|--------|
| ❶ | **UE8M0 幂次缩放** | Scale factor 存 FP32 太占空间（每个 tile 4 bytes） | 把 scale 压缩到 8-bit 纯指数（只取 2 的幂次），存储降为 **1/4**，乘除用位运算替代 |
| ❷ | **FP4 量化感知训练（QAT）** | MoE 数百个专家的权重占显存太大（FP8 仍不够） | 后训练阶段把专家权重量化到 **FP4**，显存再减半；QAT 让模型在训练中就适应量化误差 |
| ❸ | **FP4→FP8 无损反量化** | FP4 只有 16 个值，直接计算精度太低 | 利用 FP4 ⊆ FP8（值集包含关系）→ FP4 cast 到 FP8 只是尾数补零、零信息损失 → 复用 FP8 GEMM |
| ❹ | **KV Cache 混合精度** | 长上下文推理时 KV Cache 是显存瓶颈 | 把 KV 拆成两半：RoPE 维度保持 BF16（位置敏感），非 RoPE 维度用 FP8（内容鲁棒），整体省 **~44%** |
| ❺ | **Hadamard 旋转** | 异常值（outlier）主导 scale，浪费其余元素的量化精度 | 量化前用 Hadamard 正交矩阵旋转 → 信息均匀分散到所有维度 → 稀释异常值 → 量化更均匀 |

此外，FP8 GEMM 内核从 V3 的 Triton 迁移到了 **TileLang**（SMT 求解器辅助生成，§四），CSA 索引器的 QK 路径也用了 FP4 全链路量化（详见 §七）。

**阅读建议**：§二~§六每章独立，可按需跳读。如果只关心某几项技术，直接跳到对应章节即可。

### 1.5 速查手册：V4 混合精度全貌

读完本节即可掌握 V4 混合精度训练的完整体系，细节在各章中按需查阅。

#### 1.5.1 三阶段精度流水线

```
预训练（FP8，继承 V3）
  │
  │  FFN/MoE GEMM: FP8 E4M3 × FP8 E4M3 → FP32 accum
  │  Scale: UE8M0 (8-bit 纯指数, 2 的幂次)
  │  Tile: 激活 1×128, 权重 128×128
  │  其余 op (Attn GEMM/Embedding/Norm/Softmax/Gate/Loss): BF16/FP32
  │  耗时: 14.8T tokens, 671B params
  │
  ▼
后训练 QAT（FP4 引入）  ← V4 增量
  │
  │  MoE 专家权重: FP32 master → FP4 E2M1 quant → FP8 dequant → FP8 GEMM
  │    量化: w_fp4 = round(w / scale),  scale = 2^ceil(log2(amax/6))
  │    反量化: w_fp8 = cast(w_fp4, FP4→FP8)  ← 尾数补零, 数学无损
  │    GEMM: FP8×FP8 (复用 V3 内核)
  │    反向: STE 穿过 round+cast → 梯度直达 FP32 master
  │    Tile: 权重 1×32 (更细, 补偿 FP4 粗粒度)
  │
  │  CSA 索引器: Q→Hadamard→FP4,  K→Hadamard→FP4,  QK→BF16 index
  │  索引分数: FP32 → BF16 (top-k 召回率 99.7%)
  │  步数: 少量 (几千步, 权重已稳定, 仅微调适应 FP4 舍入)
  │
  ▼
推理部署
  │
  │  专家权重: FP4 直接加载 → FP4→FP8 cast → FP8 GEMM (内存加载减半)
  │  KV Cache: 混合存储 — RoPE 维度 BF16 + 非 RoPE 维度 FP8 (省 ~44%)
  │  CSA: QK 原生 FP4 计算 + BF16 index + FP32 attention softmax
```

#### 1.5.2 五项关键技术（一页纸讲完）

**❶ UE8M0 幂次缩放（§二）**

Scale factor 从 FP32 (4 bytes) 压缩到 8-bit 纯指数 (1 byte)，值只能是 2 的幂次。

```
scale = 2^ceil(log2(amax / max_val))    # max_val: 448 (FP8) / 6 (FP4)
```

- **为什么安全**：ceil 保证 scale 只大不小 → 量化永不溢出。scale 偏大 → 量化值偏小 → FP8 指数自适应降低步长 → 相对舍入误差 ~6% 不变。
- **为什么比 V3 好**：反量化乘 2^k = 指数移位，零舍入误差（V3 乘 FP32 scale 反而多一层 FP32 乘法舍入）。
- **代价**：FP8 范围利用率从 100% 降到 ~73%（可接受）。

**❷ FP4 量化感知训练（§三）**

后训练阶段将 MoE 专家权重从 FP8 再压到 FP4，显存减半。

```
FP32 master → [round(w / scale)] → FP4 (存储, 4-bit)
                                    ↓ cast (尾数补零, 无损)
                                   FP8 → FP8×FP8 GEMM (复用 V3)
                                    ↑
                               STE: 梯度直传 FP32 master
```

- **为什么只在后训练用**：FP4 仅 16 个值，预训练中权重剧烈变化时信息丢失严重；后训练权重已稳定，QAT 只需微调适应 FP4 舍入，几千步即可收敛。
- **为什么 FP4→FP8 是「无损」的**：FP4 的 16 个值全部包含在 FP8 的 256 个值中，cast 只是把 1-bit 尾数补零为 3-bit，数值精确不变。真正误差仅来自 FP4 round 那一步。
- **为什么不需要 V3 的「提升累加」**：FP4 权重 block 仅 32（不是 128），Tensor Core 累加 32 次误差小很多，乘 scale 时顺带完成精度提升。

**❸ KV Cache 混合精度（§五）**

```
KV 向量: [非 RoPE 维度 (448) | RoPE 维度 (64)]
            ↓ FP8 (1×64 tile)    ↓ BF16
            内容信息, 精度鲁棒      位置信息, 精度敏感
```

- 非 RoPE 维度占 448/512 ≈ 87.5%，BF16→FP8 减半后，KV Cache 整体省 ~44%。
- 非 RoPE 维度 tile size 选 64 而非 128：448 ÷ 64 = 7 恰好整除。
- Block size 64 也用于 CSA Compressor（无旋转模式下）。

**❹ Hadamard 旋转（§六）**

```
x_rotated = H @ x / sqrt(d)    # H: Hadamard 正交矩阵, 元素 ±1
```

量化前将信息均匀旋转到所有维度 → 异常值被稀释 → 每个维度的量化 scale 更均匀 → FP4 精度损失可控。用于 CSA Compressor 和 Indexer 的量化前处理。

**❺ FP8 GEMM 内核升级（§四）**

V3 的 Triton kernel → V4 的 TileLang kernel + 三项算法优化：

| 优化 | V3 | V4 | 收益 |
|------|-----|-----|------|
| Scale 格式 | FP32 | UE8M0 (8-bit) | Scale 存储 1/4 |
| Scale 乘法 | `a_s[:,None] * b_s[None,:]` 形状 (M,N) | 预合并为 `Scale_C[i]` 形状 (M,) | 中间张量省 128× |
| Post-GEMM 指令 | 3 条 (MUL+MUL+ADD) | 1 条 FMA | 延迟被流水线自然吸收 |

#### 1.5.3 精度分配总表（速查）

```
组件                    V3         V4           变化原因
────────────────────────────────────────────────────────
FFN/MoE GEMM           FP8        FP8           继承, 成熟方案
MoE 专家权重(存储)      FP8        FP4 (QAT后)   显存减半, FP4×FP8 吞吐 1.33×
MoE 专家 GEMM(计算)     FP8×FP8    FP8×FP8       FP4→FP8 cast 后复用
Attention GEMM         BF16       BF16          不变 (memory-bound)
KV Cache               BF16       混合 BF16/FP8  省 ~44%
CSA Indexer QK         — (新)     FP4 全路径     计算量 O(seq²), 加速收益高
CSA Indexer 分数        — (新)     BF16 (QAT后)   top-k 召回率 99.7%
Embedding              BF16       BF16          不变
RMSNorm / Softmax      BF16/FP32  BF16/FP32      不变
MoE Gate / Loss        BF16/FP32  BF16/FP32      不变
MoE 通信               FP8→BF16   FP8→BF16      不变 (计算可隐藏通信)
Muon 梯度同步           — (新)     BF16 随机舍入   通信量减半
```

#### 1.5.4 核心公式速查

| 公式 | 用途 | 位置 |
|------|------|------|
| `scale = 2^ceil(log2(amax / 448))` | FP8 激活/权重量化 (UE8M0) | §2.3 |
| `scale = 2^ceil(log2(amax / 6))` | FP4 权重量化 (UE8M0) | §3.2 |
| `w_fp8 = cast(w_fp4, FP4→FP8)` | FP4→FP8 无损精度扩展 | §3.3 |
| `C_ij = (∑ A_ik·B_jk) × s_a[i] × s_b[j]` | FP8 GEMM + 反量化 | §4.2.1 |
| `Scale_C[i] = s_a[i] × s_b` | V4 预合并 scale (同 N block 内 s_b 共享) | §4.2.1 |
| `C_accum += C_local × Scale_C[i]` | V4 后 GEMM 单条 FMA | §4.2.2 |
| `∂L/∂w = STE(∂L/∂w_fp4) ≜ ∂L/∂w_fp4` | FP4 QAT 反向 (直通估计器) | §3.4 |
| `x_rot = H @ x / √d` | Hadamard 旋转稀释异常值 | §6 |

#### 1.5.5 关键数值

| 数值 | 含义 |
|------|------|
| **FP8 E4M3 范围** | [−448, 448]，256 个值，3-bit 尾数 |
| **FP4 E2M1 范围** | [−6, 6]，16 个值，1-bit 尾数 |
| **UE8M0 范围** | 2⁻¹²⁷ ~ 2¹²⁷，256 个 2 的幂次值 |
| **Tile sizes (FP8)** | 激活 1×128，权重 128×128 |
| **Tile sizes (FP4)** | 激活 128 (不变)，权重 1×32 |
| **KV Cache tile** | 非 RoPE 维度 1×64 |
| **KV Cache 节省** | ~44%（对于 head_dim=512, rope_dim=64） |
| **FP4 显存节省** | FP8→FP4：专家权重存储减半 |
| **QAT 步数** | 几千步（远少于预训练的百万步级） |
| **CSA 索引召回率** | 99.7%（BF16 索引分数） |

#### 1.5.6 设计原则速记

1. **精度敏感操作永不降级**：Embedding、RMSNorm、Softmax、Attention GEMM、RoPE 维度一律 BF16/FP32。
2. **计算密集型操作激进降级**：MoE GEMM（FP8→FP4）、KV Cache（FP8）、CSA QK（FP4），用计算量换精度。
3. **Scale 只需「指数精度」**：UE8M0 保持完整的 8-bit 动态范围，丢掉尾数不影响量化质量。
4. **QAT 在后训练才引入 FP4**：预训练 FP8 已充分验证，FP4 是部署优化而非训练优化。
5. **从 FP32 master 每步重新量化**（FP4 QAT）：没有持久缓存 → 没有 tile 布局固化 → 转置不撕裂 tile。
6. **能合并就合并**：预合并 act_scale × weight_scale → 1 条 FMA 替代 3 条指令 → 不需要双 warpgroup。`

---

## 二、V4 核心创新一：UE8M0 幂次缩放格式

### 2.1 什么是 UE8M0

V3 的 scale factor 存为 **FP32**（32-bit）。V4 改为 **UE8M0**（8-bit，仅 8 位指数，无符号位，无尾数，也叫 **E8M0**）。

| 格式 | 位数 | 指数 | 尾数 | 可表示值 | 备注 |
|------|------|------|------|----------|------|
| FP32 scale (V3) | 32 | 8 | 23 | 任意实数 | 范围大、精度高，但占用多 |
| UE8M0 scale (V4) | **8** | 8 | 0 | **仅为 2 的幂次** | 256 个值：2⁻¹²⁷, 2⁻¹²⁶, ..., 2¹²⁷ |

UE8M0 = **U**nsigned **E**xponent **8**-bit **M**antissa **0**-bit。只有指数，所以值只能是 2 的整数次幂。

### 2.2 使用幂次缩放的原因有三

**计算层面**: `y = x / scale` 和 `dequant = y × scale` 中的除法/乘法，在 scale 是 2^k 时等价于**指数偏移**，可以用位运算实现，无需浮点除法。

**存储层面**: scale 从 32-bit 降到 8-bit，scale tensor 大小降为 1/4。

**精度层面**: scale 的精度损失由细粒度 tile 量化补偿——每个 128 长度的 tile 独立缩放，即使 scale 只取 2 的幂次，局部误差也有限。

### 2.3 如何计算 UE8M0 scale

V3 计算方式：
```
amax = max(|x_tile|)
scale = amax / 448.0            # FP32 任意值
```

V4 计算方式（`kernel.py:36-37`）：
```python
def fast_round_scale(amax, fp8_max_inv):
    # 1. amax * (1/448) → 得到精确 scale
    # 2. 用 bit 操作取 ceil(log2(scale)) → 找到下一个 2 的幂次
    # 3. 用 bit 操作计算 2^exp → 得到 2^k
    return fast_pow2(fast_log2_ceil(amax * fp8_max_inv))
```

具体 bit 操作实现（`kernel.py:22-33`）：
```python
def fast_log2_ceil(x):
    """通过 IEEE 754 bit 操作计算 ceil(log2(x))，避免慢速 log/ceil 内置函数"""
    bits_x = T.reinterpret("uint32", x)      # float32 → uint32 位模式
    exp_x = (bits_x >> 23) & 0xFF            # 提取指数位
    man_bits = bits_x & ((1 << 23) - 1)       # 提取尾数位
    # 如果尾数 ≠ 0，指数需 +1（取 ceiling）
    return T.Cast("int32", exp_x - 127 + T.if_then_else(man_bits != 0, 1, 0))

def fast_pow2(x):
    """通过 IEEE 754 bit 操作计算 2^x"""
    bits_x = (x + 127) << 23                 # 构造 float32 位模式
    return T.reinterpret("float32", bits_x)
```

### 2.3.1 完整例子：一个 tile 从 V3 到 V4

假设某个 `1×128` 激活 tile，其绝对值最大值 `amax = 5.1`。

**V3 计算方式（FP32 scale）**：

```
scale = amax / 448 = 5.1 / 448 ≈ 0.0113839
scale 存储为 FP32（32-bit 任意值）
```

量化：`x_q = x / 0.0113839` → 最大量化值是 `5.1 / 0.0113839 = 448.0`，恰好填满 FP8 E4M3 的范围 [−448, 448]。

**V4 计算方式（UE8M0 scale）**：

```
Step 1: scale_raw = amax × (1/448) = 5.1 × 0.002232 = 0.0113839

Step 2: log2(0.0113839) = log2(5.1/448) ≈ −6.457

Step 3: ceil(−6.457) = −6    ← 注意：ceil 是向正无穷取整，−6 > −6.457，因此 −6 是 −6.457 的 ceiling

Step 4: scale = 2^(−6) = 1/64 = 0.015625
```

V4 的 scale 被**向上取整**（向正无穷方向）到最近的 2 的幂次：`0.015625 > 0.0113839`。

量化：`x_q = x / 0.015625` → 最大量化值是 `5.1 / 0.015625 = 326.4`，在 [−448, 448] 内 ✓

**对比**：

| | scale 值 | scale 存储 | FP8 中的范围利用率 |
|------|------------|-------------|--------------------|
| V3 (FP32) | 0.0113839 | 32-bit 任意值 | 448.0 / 448 = 100% |
| V4 (UE8M0) | 0.015625 = 2⁻⁶ | 8-bit（仅指数 −6） | 326.4 / 448 ≈ 73% |

V4 的代价是范围利用率从 100% 降到 ~73%，但换来了 scale 存储从 32-bit → 8-bit。

### 2.3.2 不变量证明：|x/scale| ≤ 448 恒成立

这是 V4 设计的核心不变量。证明很简单：

```
scale = 2^ceil(log2(amax / 448))

由于 ceil(t) ≥ t（向上取整），有：
  ceil(log2(amax / 448)) ≥ log2(amax / 448)

两边取 2^：
  scale = 2^ceil(log2(amax/448)) ≥ 2^log2(amax/448) = amax / 448

即 scale ≥ amax / 448，移项得：
  amax / scale ≤ 448

对于 tile 内任意元素 x，有 |x| ≤ amax，所以：
  |x / scale| ≤ amax / scale ≤ 448 ✓
```

**一句话**：scale 只会被放大（向上取整），不会缩小，所以量化后的值一定不会超过 FP8 E4M3 的表示上限 448。

### 2.3.3 UE8M0 的存储编码

UE8M0 只有 8 位指数，它的 bit pattern 就是 IEEE 754 float32 指数位的直接截取：

```
scale = 0.015625 = 2^(−6)
在 float32 中: 指数 bias = 127, 存储指数 = −6 + 127 = 121 = 0b01111001

UE8M0 只存这 8 位: 0b01111001 = 0x79
```

这就是为什么叫 E8M0 — **E**xponent **8**-bit, **M**antissa **0**-bit。它本质上是把一个 float32 的符号位和尾数全部扔掉，只留指数。

**举个例子直观感受**：一个 `(M, K)` 的激活矩阵，`K=7168`，用 128 的 tile size：

```
scale tensor 形状: (M, 7168 // 128) = (M, 56)

V3:  56 个 FP32 scale = 56 × 4 bytes = 224 bytes per token
V4:  56 个 UE8M0 scale = 56 × 1 byte = 56 bytes per token

SCALE 存储降为 1/4
```

### 2.3.4 8 位指数是否够用？

一个自然的问题是：8 位指数会不会溢出？答案是不会，因为 **UE8M0 的 8 位指数和 BF16/FP32 的指数位宽完全一样**。

| 格式 | 指数位宽 | 动态范围 |
|------|----------|----------|
| FP32 | 8-bit | 2⁻¹²⁶ ~ 2¹²⁷ |
| BF16 | 8-bit | 2⁻¹²⁶ ~ 2¹²⁷ |
| **UE8M0** | **8-bit** | **2⁻¹²⁷ ~ 2¹²⁷** |

三种格式的动态范围相同。为什么？因为 IEEE 754 浮点数的范围只由指数位数决定，尾数只影响精度。

**关键观察**：scale 是从数据本身算出来的 — `scale ≈ amax / 448`。数据本身的 `amax` 能存在 BF16/FP32 里（意味着它的指数在 8-bit 范围内），那 scale 的指数也在同一个范围内。UE8M0 只是丢掉了尾数精度，**没有丢掉指数范围**。

具体来说几点保障：

1. **下限不会 underflow**：激活值经过 RMSNorm 后，`amax` 通常在 [0.1, 100] 量级，对应的 scale 指数在 [−10, 10] 附近，远没到 E8M0 的下限 2⁻¹²⁷。

2. **上限不会 overflow**：即使极端情况（比如 embedding 输出 `amax = 1000`），scale = 2^ceil(log2(1000/448)) ≈ 2^2 = 4，指数 = 2，远小于上限 2¹²⁶。

3. **最极端情况**：假设某个 tile 全是最大 BF16 值（~3.4×10³⁸），scale ≈ 10³⁶，指数 ~120，仍在 E8M0 范围（上限 126）内。但这种情况在正常训练中不会出现 — RMSNorm 把激活值约束在合理范围内。

**一句话**：UE8M0 不是「缩小了范围来省空间」，而是「保持了和 FP32/BF16 相同的指数范围，只去掉了不需要的尾数精度」。

### 2.3.5 去掉尾数精度为何不损害模型？

这引出一个更核心的问题。UE8M0 把 scale 从任意实数 round 到了 2 的幂次，比如 `0.01138 → 0.015625`，误差 ~37%。为什么 scale 差这么多，量化质量不崩？

**先理解 scale 的作用**：scale 不是模型的「参数」，它是量化的「辅助信息」。它的任务很简单：把数据线性缩放到 FP8 的 [−448, 448] 范围内。反量化时用**同一个 scale** 乘回去。

```
量化-反量化全过程（对 tile 中某个值 x）：

  V3:  x → x_q = round(x / 0.01138, fp8) → x̂ = x_q × 0.01138 ≈ x
  V4:  x → x_q = round(x / 0.01563, fp8) → x̂ = x_q × 0.01563 ≈ x
       ↑ scale 被 round 到 2⁻⁶                         ↑ 反量化用同样的 scale
```

**关键洞察：量化 + 反量化是一个近似的恒等变换，scale 的绝对大小不重要，只要量化值 x_q 不溢出 FP8 范围。**

具体来说，三个原因让尾数精度不重要：

**原因 1：scale 的误差方向有利，只影响 FP8 范围利用率**

```
V3:  amax=5.1, scale=0.01138 → 量化值范围 [−448, 448]，利用率 100%
V4:  amax=5.1, scale=0.01563 → 量化值范围 [−326, 326]，利用率 ~73%
```

V4 的 scale 更大 → 量化值的绝对值更小 → 不会溢出，但 FP8 的动态范围没用满。**代价是 FP8 的相对量化误差略微增大**（因为值小了，同样的 FP8 尾数步长代表的相对误差变大）。

但 scale 最多偏大 ~2 倍（向上取整到最近的 2 的幂次）。而 FP8 的 3-bit 尾数精度足以容忍这一偏差。

**原因 2：细粒度 tile 量化把误差隔离在每个 128 元素内**

每个 tile 有独立的 scale，tile 之间互不影响。一个 tile 的 scale 被 round 到 0.015625，不会影响其他 tile。对比 per-tensor 量化：

```
Per-tensor:  整个矩阵一个 scale，如果一个 outlier 让 scale 变大，
             所有位置都受影响 → 灾难性精度损失

Per-tile (V4): 每个 128×1 block 独立 scale，
               outlier 只影响自己所在 tile → 误差被隔离
```

**原因 3：ceil 的误差到底去了哪里？**

这个质疑是合理的：ceil 让 V4 的 scale 最多偏大 ~2 倍，直觉上「scale 不准 → 反量化结果不准」。下面用一个 2×2 矩阵 + 2 维向量的完整 GEMM 走一遍，看误差到底出在哪。

```
数据:
  x = [3.7, 2.8]        激活向量, amax=3.7
  W = [[2.5, 1.0],      权重矩阵, amax=2.5
       [0.8, -2.2]]
  真值: x @ W = [3.7×2.5 + 2.8×0.8,  3.7×1.0 + 2.8×(-2.2)]
             = [11.49, -2.46]

══════════════ V3: FP32 scale ═══════════════
  act_scale = 3.7/448 = 0.008259
  w_scale   = 2.5/448 = 0.005580

  量化:  x / act_scale  = [448.0, 339.1]
         → FP8 round:   = [448,   352  ]  ← step=32 at exp=8, 339→352
         W / w_scale    = [[448.0, 179.2], [143.4, -394.2]]
         → FP8 round:   = [[448,   176  ], [144,   -384 ]]  ← step=16 at lower exp

  GEMM:  C_fp8 = [448,352] @ [[448,176],[144,-384]]
               = [448×448+352×144, 448×176+352×(-384)]
               = [251392, -56320]

  反量化: C = C_fp8 × 0.008259 × 0.005580
           = C_fp8 × 0.00004609      ← FP32 乘法, 有舍入
           = [11.590, -2.596]
  
  误差: [|11.590-11.49|, |-2.596-(-2.46)|] = [0.100, 0.136]

══════════════ V4: UE8M0 scale ═══════════════
  act_scale = 2^ceil(log2(3.7/448)) = 2^ceil(-6.92) = 2⁻⁶ = 0.015625  ← ceil! 比理想值大 1.89×
  w_scale   = 2^ceil(log2(2.5/448)) = 2^ceil(-7.48) = 2⁻⁷ = 0.0078125 ← ceil! 比理想值大 1.40×

  量化:  x / act_scale  = [236.8, 179.2]
         → FP8 round:   = [240,   176  ]  ← step=16 at exp=7, 237→240
         W / w_scale    = [[320.0, 128.0], [102.4, -281.6]]
         → FP8 round:   = [[320,   128  ], [104,   -288 ]]

  GEMM:  C_fp8 = [240,176] @ [[320,128],[104,-288]]
               = [240×320+176×104, 240×128+176×(-288)]
               = [95104, -19968]

  反量化: C = C_fp8 × 2⁻⁶ × 2⁻⁷ = C_fp8 × 2⁻¹³ = C_fp8 / 8192
           = [11.609375, -2.4375]  ← 除以 8192 = 指数移位, 完全精确

  误差: [|11.609-11.49|, |-2.438-(-2.46)|] = [0.119, 0.022]
```

**逐项拆解误差来源**：

```
V3 总误差 [0.100, 0.136] 来自:
  ├── FP8 round: x 的 339.1→352 产生 ~1.5% 相对误差
  ├── FP8 round: W 的 179.2→176, 143.4→144, -394.2→-384 各自 ~1-2%
  └── 反量化乘法: FP32 × 0.00004609 产生 ~10⁻⁷ 量级的额外舍入

V4 总误差 [0.119, 0.022] 来自:
  ├── ceil 让 act_scale 大了 1.89× → x_q 小了 ~2× → 但 FP8 步长也小了 ~2×:
  │   V3 的 x_q 在 exp=8, step=32; V4 的 x_q 在 exp=7, step=16
  │   → FP8 相对精度不变 (都是 ~6%)
  ├── ceil 让 w_scale 大了 1.40× → 同样的自适应补偿
  │   而且巧合的是 320, 128 都是 FP8 的精确值, 所以 W 的量化误差反而更小
  └── 反量化乘法: C_fp8 / 8192 → 指数移位, 零舍入误差 ← 这一步 V4 比 V3 精确
```

以上分析可以提炼为一个核心结论：**ceil 误差不等于反量化误差**。

ceil 让 scale 变大，但 FP8 的步长是**指数适应的** — scale 大 → 量化值小 → 指数降低 → 步长等比缩小。所以 FP8 的相对舍入误差 ~6% 是一个近似常数，不受 scale 大 1.1× 还是 2× 的影响。

而反量化这一步，V4 因为乘以 2 的幂次（纯指数移位），**完全消除了反量化乘法本身带来的舍入误差**。V3 反而会多一层 FP32 乘法误差（虽然很小，~10⁻⁷ 量级）。

```
真正的误差公式:

  reconstruction_error ≈ FP8_round_error + dequant_mul_error
                       ≈      ~6%           +      ~0 (V4) / ~10⁻⁷ (V3)
                       ↑ 由 FP8 3-bit 尾数决定   ↑ V4 消除了这一项
                       ↑ 与 scale 是 FP32 还是 UE8M0 无关
```

**总结**：

```
scale 为什么不需要精度？
  ├── 量化误差由 FP8 3-bit 尾数决定, 与 scale 精度无关
  │   └── scale 大 → x_q 小 → FP8 步长自适应变小 → 相对误差恒定 ~6%
  ├── 反量化乘 2^k 在 FP32 中无损 — V4 反而比 V3 少了一层舍入
  ├── ceil 误差方向有利 — scale 只变大, 量化永不溢出
  └── 细粒度 tile — 每个 tile 独立, ceil 误差不扩散
```

### 2.4 在代码中的体现

V4 的 `Linear` 类初始化权重 scale 时（`model.py:137-142`）：
```python
if dtype == torch.float8_e4m3fn:
    self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
    # Scale 存为 float8_e8m0fnu（UE8M0），只有 8-bit
    scale_out_features = (out_features + 128 - 1) // 128
    scale_in_features = (in_features + 128 - 1) // 128
    self.weight.scale = nn.Parameter(
        torch.empty(scale_out_features, scale_in_features, dtype=torch.float8_e8m0fnu))
```

注意 `scale_out_features` = `N // 128`（不是 V3 的 `M//128 × N//128`）。V4 的 FP8 权重 scale 形状为 `(N//128, K//128)` — 这对应 `fp8_gemm_kernel` 中 `scales_b` 的签名：
```python
scales_b: T.Tensor[(T.ceildiv(N, group_size), T.ceildiv(K, group_size)), scale_dtype]
```

---

## 三、V4 核心创新二：FP4 量化感知训练（QAT）

> **⚠️ 适用范围**：FP4 QAT **仅在后训练阶段使用**，预训练仍使用 FP8（继承 V3 方案）。这一点在 §1.1 的表格中已有标注，但本章展开细节前有必要先说清楚**为什么**。

**为什么预训练不用 FP4？**

原因有四，由核心到外围：

1. **FP4 粒度太粗，无法支撑从零学习**。FP4 E2M1 只有 16 个可表示值，尾数仅 1-bit。预训练中权重在剧烈变化，梯度经 STE 穿过 FP4 的 round 时会丢失大量信息——FP8 的 3-bit 尾数（256 个值）是经验验证过的预训练精度下限，FP4 再砍 2-bit 尾数会导致收敛变慢甚至不收敛。

2. **预训练的显存瓶颈不是权重**。预训练时显存大头是激活值（activation）和优化器状态（AdamW 的 m/v），权重本身占比有限。把权重从 FP8 压到 FP4 省出的显存，对预训练的 batch size 或序列长度几乎没有帮助——但在推理部署时权重就是显存的主体，FP4 的收益才能兑现。

3. **后训练阶段权重已稳定，QAT 只是「微调适应」**。后训练（SFT + RL）的 step 数远少于预训练，权重变化幅度小。此时引入 FP4 量化，模型需要学习的不是新知识，而是「容忍 FP4 的粗粒度舍入误差」——这是一个相对简单的适应性任务，少量 steps 即可收敛。

4. **预训练 FP8 方案已被 V3 充分验证**。V3 在 671B 规模、14.8T token 上证明了 FP8 预训练稳定且无损。V4 没必要在这个成熟方案上冒险换 FP4。

**一句话**：FP4 是**部署优化**（让最终模型更小、推理更快），不是**训练优化**（让预训练更省）。预训练保持 FP8，后训练 QAT 引入 FP4 → 推理时直接加载 FP4，形成「FP8 预训练 → FP4 QAT → FP4 推理」的完整链路。

### 3.1 全流程概览：一张图看懂 FP4 QAT

DeepSeek-V4 的 MoE 层有数百个专家（Flash: 256, Pro: 384），专家权重是 GPU 显存的主要占用。FP4 相比 FP8 将权重存储量再减半。

但 FP4 E2M1 只有 **16 个可表示值**（1-bit 符号 + 2-bit 指数 + 1-bit 尾数），直接拿来做矩阵乘精度太低。V4 的做法是：**权重存 FP4，计算时「无损」反量化到 FP8，复用成熟的 FP8 GEMM**。

整个 FP4 QAT 的前向链路分三步，后续 §3.2~§3.4 按此顺序展开：

```
FP32 master weight
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│ ① 量化 (FP32 → FP4)                         ← §3.2      │
│   w_fp4 = round(w / scale)     scale = 2^ceil(log2(amax/6)) │
│   → 存为 FP4 (4-bit), scale 为 UE8M0 (8-bit)              │
│   ⚠️ 误差唯一来源：FP4 的粗粒度 round                     │
└─────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│ ② 反量化 + 计算 (FP4 → FP8 → GEMM)          ← §3.3      │
│   w_fp8 = cast(w_fp4, FP4→FP8)  ← 尾数补零, 逐元素, 无损 │
│   C += (act_fp8 @ w_fp8^T) × scale_act × scale_w         │
│   → 完全复用 V3 的 FP8 GEMM 算力                          │
└─────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│ ③ 反向传播 (STE)                             ← §3.4      │
│   梯度直接穿过 round 和 cast → 更新 FP32 master           │
└─────────────────────────────────────────────────────────┘
     │
     ▼
  output (BF16)
```

**一句话**：误差只在第①步的 FP4 round 中产生。第②步反量化在数学上是精确的——FP4 的值集是 FP8 值集的子集，cast 只是尾数补零；scale 是 2 的幂次，乘法等于指数加法。QAT 的目标就是让模型学会容忍第①步的 FP4 舍入误差。

> **FP4 E2M1 格式参考**（共 16 个值，范围 [−6, 6]，bias=1，隐式前导 1）：
>

| 位模式 | 符号 | 指数 (存储值 → 实际值) | 尾数 (1-bit, 隐式+1) | 值 |
|--------|------|------------------------|----------------------|-----|
| 0 0 0 0 | + | 2⁻¹ (00 → 0.5) | 1.0 | 0.5 |
| 0 0 0 1 | + | 2⁻¹ (00 → 0.5) | 1.5 | 0.75 |
| 0 0 1 0 | + | 2⁰ (01 → 1.0) | 1.0 | 1.0 |
| 0 0 1 1 | + | 2⁰ (01 → 1.0) | 1.5 | 1.5 |
| 0 1 0 0 | + | 2¹ (10 → 2.0) | 1.0 | 2.0 |
| 0 1 0 1 | + | 2¹ (10 → 2.0) | 1.5 | 3.0 |
| 0 1 1 0 | + | 2² (11 → 4.0) | 1.0 | 4.0 |
| 0 1 1 1 | + | 2² (11 → 4.0) | 1.5 | 6.0 |
| 1 × × × | − | 同上 | 同上 | 相应负值 |

共 16 个可表示值，正值集合 {0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}，范围 [−6, 6]。粒度非常粗——各值之间间隔很大。（注：全零位模式 `0b0000` 通常保留为真正的零值，即上述 0.5 可能以 0 替代，具体取决于实现。）

### 3.2 第一步：FP32 → FP4 量化

Expert 包含三个 Linear 层（gate/up/down），当 `expert_dtype = "fp4"` 时权重全部存为 FP4。以 w2（down 投影）为例，维度 `[dim, inter_dim]`。

**存储格式**（`model.py:131-137`）：

```
FP4 数据:  shape [dim, inter_dim // 2], dtype = float4_e2m1fn_x2
           → 每字节存 2 个 FP4 值, inter_dim 方向被「对折」

FP4 scale: shape [dim, inter_dim // 32], dtype = float8_e8m0fnu (UE8M0)
           → 每 32 个元素 (沿 inter_dim) 共享一个 scale（1×32 tile）
```

对比 V3 的 FP8 权重（128×128 tile），FP4 的 tile 更细。原因是 FP4 动态范围更小（E2M1 仅 2 个指数位 vs FP8 E4M3 的 4 个），需要更细粒度的 scale 来补偿。

**量化过程（具体例子）**。假设 w2 的第 row 行、第 col 列，FP32 master 值为 `1.875`：

```
原始值 (FP32 master):  w = 1.875

col 所在的 32-group (group = col // 32):
  该 group 的 amax = 4.8 (假设)
  scale = 2^ceil(log2(4.8 / 6))      ← FP4 max=6, 所以除 6 而非 448
        = 2^ceil(log2(0.8)) = 2^ceil(-0.322) = 2^0 = 1.0

  w_fp4 = round(1.875 / 1.0, fp4)
        → FP4 E2M1 可表示值: {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, ...}
        → 最近值是 2.0

存储:
  fp4_data[row, col//2] 中存 2.0 (4 bits, 与相邻元素共享一个字节)
  fp4_scale[row, group] 中存 1.0 = 2^0 (8 bits, UE8M0)
```

**量化误差**：原始值 1.875 → 量化值 2.0，相对误差 ~6.7%。这是 FP4 QAT **唯一**的信息损失环节——下一节的 FP4→FP8 反量化和 GEMM 都不会再引入新误差。QAT 的训练目标就是让模型学会容忍这个量级的权重扰动。

### 3.3 第二步：FP4 → FP8 反量化 + GEMM

前向传播时，`self.w2(x)` 调用链: `linear(x, w2_fp4)` → `fp4_gemm(x_fp8, x_scale, w2_fp4, w2_scale)`。

fp4_gemm 内部每 32 个 K 维元素执行一轮，每轮四个子步骤：

```
for k in range(K // 32):
    a     = load_act_fp8(k)        # FP8 激活, [block_M, 32]
    b_fp4 = load_wt_fp4(k)         # FP4 权重, [block_N, 32]

    b_fp8 = cast(b_fp4, FP4→FP8)  # 子步骤 A: 精度扩展 → 尾数补零
                                     #   2.0(FP4) → 2.0(FP8)

    scale_w = wt_scale[k]          # 子步骤 B: 取预计算好的 UE8M0 scale
    scale_a = act_scale[k // 4]

    C_fp8 = a_fp8 @ b_fp8^T       # 子步骤 C: FP8×FP8 MMA（裸值相乘）

    C += C_fp8 * scale_a * scale_w # 子步骤 D: 乘 scale 完成反量化
```

**为什么「无损」**

论文说的「无损」不是指「FP4 能还原 FP32」——那不可能（FP4 只有 16 个值）。它特指：**FP4 的存储信息在反量化到 FP8 时不会产生额外舍入**。两个原因：

1. **值集包含**：FP4 的 16 个值 ⊆ FP8 的 256 个值。`cast(2.0, FP4→FP8)` 只是把 1-bit 尾数 `0` 补零成 3-bit 尾数 `000`，表示的数值完全不变。
2. **Scale 是 2 的幂次**：`w_fp8 × 2^k` 在 FP32 中等价于指数位加 k，不产生舍入。

回顾 §3.2 的例子：`w_fp4 = 2.0, scale_w = 1.0` → 反量化得 `2.0 × 1.0 = 2.0`，精确。但 `1.875 → 2.0` 的误差是 §3.2 量化 round 时产生的，与反量化无关——**反量化不会让误差变大**。

**补充：什么时候反量化会「有损」？**

只有当反量化值超出 FP8 上限 448 时（overflow），但实际没有观察到：

```
group A (正常):  w_fp4 = 6.0,  scale = 2^3 = 8    → 48   ≤ 448 ✓
group B (极端):  w_fp4 = 6.0,  scale = 2^8 = 256  → 1536 > 448 ✗
```

相邻 32-group 的 scale 差异需要超过 32× 才会触发溢出，论文经验验证实际 Expert 权重不存在这种极端分布。

### 3.4 第三步：反向传播（STE）

#### 3.4.1 STE（Straight-Through Estimator，直通估计器）

STE是一种处理不可微操作的梯度近似方法，常用于量化、二值化等场景。

**先说直觉**。神经网络训练靠的是「反向传播求梯度 → 梯度下降更新参数」。但 FP4 量化里有个 `round`（四舍五入），它的函数图像是一个「楼梯」：

```
round(x)
  ↑
3 ┤·········○
2 ┤·····○   :          ← 每个「台阶」上斜率为 0（平的 → 梯度 = 0）
1 ┤·○   :   :
0 ├○───○───○──→ x        ← 台阶边缘处斜率不存在（跳变 → 梯度无定义）
```

`round(1.3) = 1`，`round(1.30001) = 1`——输入变了，输出不变 → 梯度为 0。而 `round(1.4999) = 1` 但 `round(1.5001) = 2`——在 1.5 处梯度根本不存在。

如果正常求导，梯度传到 `round` 就断了：要么是 0（梯度消失），要么无定义。上游的 FP32 master weight 收不到任何更新信号。

**STE 的解法非常直接**：**前向照常 round，反向假装没 round——将 round 视为恒等映射，导数恒为 1。**

```
Forward:   x=1.875 → round(1.875) → 2.0    （真实量化，有舍入误差）
Backward:  梯度 g ←──────── 1 ────────       （假装 round 是 f(x)=x，导数恒为 1）
```

**为什么这个做法能 work？** 三个原因：

1. **梯度方向大致正确**。`round` 改的值很小（FP4 下约 ±6.7% 相对误差），梯度方向不会完全反转。传回去的梯度虽然不精确，但指向的大方向是对的。
2. **QAT 步数很少**。FP4 QAT 只在后训练阶段跑少量 steps，模型已从预训练学到好的表示，只需微调适应 FP4 的舍入误差，梯度稍有偏差也能收敛。
3. **没有更好的替代方案**。`round` 本质是离散函数，不存在真正的梯度。所有 QAT 方法都必须用近似梯度替代，STE 是最简单也最有效的。

**数学表述**：

```
w_fp4 = round(w_fp32 / scale)  ← round 函数不可微

STE: ∂w_fp4 / ∂w_fp32 ≜ 1    （假装 round 是恒等映射）
```

**V4 中 STE 跨过了两个不可微操作**——FP32→FP4 的 round + FP4→FP8 的 cast：

```
Forward:  FP32 master → [round · ¹⁄scale] → FP4 → [cast: 尾数补零] → FP8 → FP8 GEMM → loss
                                                                       ↑
Backward: ∂L/∂W_fp32 ←──────────────── STE 直接穿过 ─────────────────←── FP8 GEMM backward
          梯度照常更新 master
```

FP4→FP8 的 cast 是纯位操作（尾数补零，精度扩展），同样没有梯度。STE 一口气把两个都跳过了——**FP8 GEMM backward 产出的梯度直接成为 FP32 master 的梯度**，中间不做任何变换。实现上，PyTorch 中只需给 `torch.round` 和 `torch.cast` 注册恒等梯度 hook 即可。

#### 3.4.2 与 V3 的对比：V3 需要「转置后重新量化」，V4 不需要

要理解这句话，必须先理清**线性层的反向传播到底需要什么**，以及 **V3 和 V4 的根本架构差异**。

**第一步：线性层的反向传播需要转置权重**

对于线性层 `Y = X @ W`（X 为输入，W 为权重），反向传播需计算两个梯度：

```
Dgrad（输入梯度）: dX = dY @ W^T    ← 需要转置后的权重
Wgrad（权重梯度）: dW = X^T @ dY    ← 需要转置后的输入
```

Dgrad 需要 `W^T`——这就是「转置权重」需求的来源。

**第二步：V3 的反向传播——为什么非要用 FP8？**

V3 的核心设计是 **FP8 训练全链路**：不仅前向用 FP8 GEMM，反向的 Dgrad 和 Wgrad 也用 FP8 GEMM（[V3 §四](./deepseek-v3-fp8-training.md#四反向传播中的-fp8简述)）。这是 V3 的精髓——如果反向退回到 BF16/FP32，训练吞吐会大幅下降，FP8 训练的加速收益就没了。

所以对 V3 来说，Dgrad 的计算是：

```
dX = dY @ W^T    ← 这个 GEMM 要在 FP8 精度下执行
                 ← 也就是说 W^T 必须是 FP8 格式的
```

问题来了：**W 以 FP8 格式持久存储，但其量化布局（tile→scale 的对应关系）是按 `(Out, In)` 形状设计的。**

**第三步：V3 的困境——tile 量化与转置的矛盾**

V3 的权重存储（[V3 §2.1](./deepseek-v3-fp8-training.md#L247-L264)）：

```
W_fp8   shape: (Out, In)           ← FP8 数据, 持久存储
W_scale shape: (Out/128, In/128)   ← 每个 scale 对应一个 128×128 block
```

每个 scale 值 `s_ij` 是从 `W[i*128:(i+1)*128, j*128:(j+1)*128]` 这个 block 的 `amax` 算出来的。现在 Dgrad 需要 W^T，W^T 的 shape 是 `(In, Out)`。如果按同样的 128×128 tile 去切 W^T：

```
以 W(256×256) 为例：

  原始 W (Out×In):                转置 W^T (In×Out):
  ┌──────────┬──────────┐          ┌──────────┬──────────┐
  │ scale₀₀  │ scale₀₁  │          │ ？？？    │ ？？？    │
  │W[0:128,  │W[0:128,  │          │W^T[0:128,│W^T[0:128,│
  │ 0:128]   │ 128:256]  │          │ 0:128]   │ 128:256] │
  ├──────────┼──────────┤          ├──────────┼──────────┤
  │ scale₁₀  │ scale₁₁  │          │ ？？？    │ ？？？    │
  │W[128:256,│W[128:256,│          │W^T[128:  │W^T[128:  │
  │ 0:128]   │ 128:256]  │          │ 256,0:128│ 256,128: │
  └──────────┴──────────┘          └──────────┴──────────┘

  W^T[0:128, 128:256] 来自 W[128:256, 0:128]
  → 这个新 block 混入了两个不同 scale（scale₁₀ 和 scale₁₁）管辖的元素
  → 没有一个单一的 scale 能正确地反量化这个新 block
```

**第四步：V3 能「直达 FP32」吗？**

一个直接的思路是：**把 FP8 权重全部反量化到 FP32，转置后直接用，不行吗？**

技术上**可以**，但这样做的代价是：

```
方案 A（V3 实际做法）:  FP8 GEMM 做反向 → 快，内存带宽省，但需要 W^T 也是 FP8
方案 B（反量化到 FP32）:  反量化 W → FP32 → 转置 → FP32 GEMM 做反向 → 慢
```

方案 B 完全抹杀了 FP8 训练的意义——V3 整个设计的目标就是让 GEMM 在 FP8 下跑，省内存带宽、省计算。如果反向退回 FP32，训练吞吐会掉一大截。

所以 V3 面临的是一个**架构层面的两难**：

| 方案 | 做法 | 代价 |
|------|------|------|
| 存两份 FP8 | W 和 W^T 各存一份 FP8，各自独立量化 | 权重显存翻倍 |
| 运行时重量化 | 反量化→转置→重新量化到 FP8 | 额外计算开销 |

V3 选择了后者（或等效方案）。

**第五步：V4 为什么没有这个问题**

V3 和 V4 区别在于**低精度版本是「持久缓存」还是「每步重量化」**：

```
V3 架构:
  FP32 master weight ──→ 训练开始时量化一次 → FP8 持久缓存
                           ↑
                  之后几百万步都用这份 FP8，不再重新量化
                  （重新量化太贵，预训练跑不起）

  → Dgrad 需要 W^T → FP8 缓存是固定 tile 布局的 → tile 矛盾 → 必须重新量化

V4 架构:
  FP32 master weight ──→ 每个 forward 都重新量化 → FP4（临时，不缓存）
                           ↑
                  每次从 FP32 master 现场量化
                  （QAT 只有几千步，开销可接受）

  → Dgrad 需要 W^T → 现场量化时选对 layout 就行（或 FP4 转置后逐元素 cast）
  → 没有持久缓存 → 也就没有 tile 布局固化的问题
```

**真正的区别**：

| | V3 | V4 |
|---|-----|-----|
| FP32 master | ✅ 有 | ✅ 有 |
| 低精度版本 | **持久缓存**（训前量化一次，几百万步复用） | **每步重量化**（从 FP32 master 现场算） |
| 为什么这样设计 | 预训练 step 数巨大，每步重量化不可接受 | QAT 只有几千步，每步重量化开销可忽略 |
| 转置问题 | 持久缓存的 tile 布局是固化的 → 转置撕裂 tile | 没有固化布局 → 现场量化直接按需要的 layout 来 |

**一句话总结**：V3 为了性能把 FP8 权重「冻结」成了持久缓存→ tile 布局固化→ 转置就出问题。V4 不缓存 FP4，每步从 FP32 master 现场量化→ 没有固化布局→ 转置不是问题。代价是每步多一次量化计算，但 QAT 步数少，这个代价完全可接受。

### 3.5 推理部署：直接使用 FP4

后训练完成后，推理阶段**不需要 FP32 master**——直接加载 FP4 权重做计算：

```
推理时:
  FP4 weight（从 checkpoint 直接加载）
  → fp4_gemm(): FP4→FP8 cast → FP8×FP8 GEMM
```

目前硬件上 FP4×FP8 的峰值 FLOPS 与 FP8×FP8 相同（因为 FP4→FP8 cast 后仍走 FP8 Tensor Core），但：
- **内存加载减半**（FP4 vs FP8）
- **未来硬件**: FP4×FP8 理论上可达 1.33× FP8×FP8 的峰值吞吐

### 3.6 代码：fp4_gemm_kernel 详解

前面 §3.2~§3.4 走完了 FP4 QAT 的概念链路（量化 → 反量化+GEMM → 反向传播 STE）。本节将 §3.3 的伪代码映射到实际的 GPU kernel 实现，展示 **cast（FP4→FP8）、取 scale、MMA（矩阵乘）、乘 scale** 这四个子步骤如何翻译为 TileLang 代码。

`fp4_gemm_kernel`（`kernel.py:442-515`）：

```python
# 关键参数
act_group_size = 128     # 激活: 1×128 tile, FP8
weight_group_size = 32   # 权重: 1×32 tile, FP4
block_K = 32             # 每次加载 32 个 K 维元素（匹配 weight_group_size）

for k in T.Pipelined(K_iters, num_stages=2):
    T.copy(A[by*block_M, k*block_K], A_shared)      # 加载 FP8 激活
    T.copy(B[bx*block_N, k*block_K], B_fp4_shared)   # 加载 FP4 权重

    # FP4 → FP8 cast（必须经过 FP32 中间类型）
    for i, j in T.Parallel(block_N, block_K):
        B_shared[i, j] = T.Cast(FP8, T.Cast(FP32, B_fp4_shared[i, j]))

    # Weight scale: per 32 on K（1 scale per 32-element group）
    for i in T.Parallel(block_N):
        scale_b_frag[i] = T.Cast(FP32, scales_b[bx*block_N + i, k])

    # Act scale: per 128 on K, 索引为 k // 4（4个 32-group = 1个 128-group）
    for i in T.Parallel(block_M):
        scale_a_frag[i] = T.Cast(FP32, scales_a[by*block_M + i, k // n_sub])

    T.gemm(A_shared, B_shared, C_local, transpose_B=True)  # FP8×FP8 MMA

    for i, j in T.Parallel(block_M, block_N):
        C_local_accum[i, j] += C_local[i, j] * scale_a_frag[i] * scale_b_frag[j]
```

**关键差异 vs V3 FP8 GEMM**：
1. 权重是 FP4，加载后 cast 到 FP8
2. 权重 block size 是 **32**（不是 128）
3. 每个 k iteration 加载 32 个 K 维元素 → 正好对应一个 FP4 weight scale group
4. 激活 block size 仍是 128 → `k // n_sub` 索引：每 4 个 k iteration 对应同一个激活 scale
5. **不需要 V3 风格的提升累加** — 原因见下方展开。

**关于第 5 点的展开：什么是「提升累加」，为什么 fp4_gemm 不需要？**

V3 的 FP8 GEMM 有一个关键设计——**提升累加**（[V3 §3.6-3.7](./deepseek-v3-fp8-training.md#L450-L536)）。要理解它，先要了解一个硬件事实：

> NVIDIA Tensor Core 在做 FP8 矩阵乘时，内部的累加器精度只有 **~14 位**（FP32 是 24 位）。这是硬件为了吞吐量做的取舍——更宽的累加器需要更多芯片面积和功耗。[^1]

这意味着，每累加一个乘积，就会引入微小的舍入误差。累加的次数越多，误差累积越大。V3 的 FP8 GEMM 每次处理 128 个 K 维元素，意味着 Tensor Core 内部要累加 128 次——实测当 K=4096 时，如果不加干预，最大相对误差可达 **~2%**，足以影响大模型训练的收敛。

V3 的解决方案是**每 128 个元素（即每次 WGMMA 完成后）把 Tensor Core 的 14 位部分和「提升」到 CUDA Core 的 FP32 寄存器中**，在那里乘 scale 并累加到 FP32 的 running sum：

```
V3 的做法（每 128 个 K 元素一次）:
  Tensor Core MMA（14 位累加器）
       ↓ partial（~14 位精度, 误差已累积 128 次）
  [提升到 FP32] × scale_a × scale_b
       ↓
  FP32 running sum（24 位精度, 跨所有 K 迭代）
```

这个「提升」操作本身有开销（Tensor Core → CUDA Core 的数据搬运），V3 通过**双 warpgroup** 来隐藏：一个 warpgroup 做提升时，另一个在 Tensor Core 上跑 MMA。

**V4 的 fp4_gemm 为什么不需要？** 因为 FP4 权重的 block size 是 **32** 而不是 128——每次 GEMM 只累加 32 个乘积：

```
V4 fp4_gemm（每 32 个 K 元素一次）:
  Tensor Core MMA（14 位累加器）
       ↓ partial（误差只累积了 32 次，~1/4 of V3）
  [scale_a × scale_b]
       ↓
  C_local_accum += C_local × scale  ← 乘 scale 即完成 FP32 提升
```

两个因素叠加：

1. **累加次数少**：32 次 vs 128 次 → 14 位累加器的误差累积只有 V3 的 ~1/4，即使在最坏情况下也可忽略。
2. **scale 乘法本身就是提升**：`C_local_accum += C_local * scale` 这条 multiply-add 在 FP32 寄存器中执行，Tensor Core 的 14 位输出被乘入 FP32 后自然获得 24 位精度。不需要额外的「提升到 CUDA Core」步骤。

**一句话**：V3 因为 K chunk = 128，14 位累加器误差大到需要显式提升；V4 fp4_gemm 因为 K chunk = 32，误差本身就小，scale 乘法顺带完成了精度提升，不需要额外的提升步骤。

[^1]: **关于「~14 位」的说明**：NVIDIA 官方文档声明 Tensor Core 的 FP8 MMA 指令使用 FP32 累加器（24-bit 尾数）。此处「~14 位」为 DeepSeek-V3 论文中的**实测有效精度**，可能反映 Tensor Core 内部微架构的逐周期舍入行为，并非 NVIDIA 公开标称值。文中基于此值的定量分析（如「K=4096 时最大相对误差 ~2%」）来自 V3 论文的实测数据。

---

## 四、逐步分析：FP8 GEMM（继承 V3，但换了格式和框架）

### 4.1 V3 vs V4：什么变了，什么没变

V3 和 V4 的 FP8 GEMM 遵循相同的**三阶段结构**：

```text
① 量化:    BF16 激活/权重 → FP8 + scale
② GEMM:    FP8 Tensor Core 矩阵乘 → FP32 accumulator
③ 反量化:  accumulator × act_scale × weight_scale → BF16 输出
```

V4 在 V3 的基础上做了三个升级：

| 阶段 | V3 | V4 | 变化动机 |
|------|-----|-----|---------|
| 量化 | Triton kernel，FP32 scale | TileLang kernel，**UE8M0** scale，bit 操作计算 | Scale 存储降为 1/4（详见 §二） |
| GEMM | Triton `tl.dot`，FP32 accumulator | TileLang `T.gemm`，FP32 accumulator | 框架迁移到 TileLang（SMT 辅助调度） |
| Scale 乘回 | `acc += dot * a_s[:, None] * b_s[None, :]` | `C_local_accum += C_local * Scale_C_shared[i]`（**预合并** act_scale × weight_scale） | 省 shared memory + 减少一次乘法 |

**一句话**：算法不变（三阶段 FP8 GEMM），格式和框架升级——UE8M0、TileLang、预合并 scale。

### 4.2 内核详解

在阅读 kernel 代码之前，先标出 V4 相比 V3 的 **4 个关键变化**，带着这些去看代码会更清晰：

| # | 变化 | 对应行 | 要点 |
|---|------|--------|------|
| ❶ | **Scale 预合并** | `Scale_C_shared[i] = scale_a × Scale_B` | V3 是 `a_s[:, None] * b_s[None, :]`（形状 32×128），V4 先乘好存为 `(32,)`，省 shared memory |
| ❷ | **4 级流水线** | `num_stages=4` | 加载 / GEMM / scale 乘 / 写回 四阶段流水重叠，隐藏内存延迟 |
| ❸ | **权重 scale 共享** | `scales_b[bx, k]` | 同一 block_N 的 128 个 N 共享一个权重 scale（形状 `N//128 × K//128`） |
| ❹ | **无 CUDA Core 提升** | `C_local_accum += C_local × scale` | Tensor Core 的 FP32 输出直接在寄存器中，下游 multiply-add 直接消费 |

完整 kernel（`kernel.py:204-254`）：

```python
@T.prim_func
def fp8_gemm_kernel_(
    A: T.Tensor[(M, K), FP8],           # 激活: FP8, (M, K)
    B: T.Tensor[(N, K), FP8],           # 权重: FP8, (N, K), 已转置存储
    C: T.Tensor[(M, N), out_dtype],     # 输出
    scales_a: T.Tensor[(M, K//128), scale_dtype],   # 激活 scale: per 1×128
    scales_b: T.Tensor[(N//128, K//128), scale_dtype], # 权重 scale: per 128×128
):
    with T.Kernel(T.ceildiv(N, 128), T.ceildiv(M, 32), threads=128) as (bx, by):
        A_shared = T.alloc_shared((32, 128), FP8)
        B_shared = T.alloc_shared((128, 128), FP8)
        C_local = T.alloc_fragment((32, 128), accum_dtype)         # Tensor Core 输出
        C_local_accum = T.alloc_fragment((32, 128), accum_dtype)   # Scale 校正后

        K_iters = T.ceildiv(K, 128)
        for k in T.Pipelined(K_iters, num_stages=4):
            T.copy(A[by * 32, k * 128], A_shared)      # Load FP8 act tile
            T.copy(B[bx * 128, k * 128], B_shared)      # Load FP8 weight tile

            # 预合并 act scale × weight scale（逐行广播）
            Scale_B = T.Cast(FP32, scales_b[bx * 128 // 128, k])
            for i in T.Parallel(32):
                Scale_C_shared[i] = T.Cast(FP32, scales_a[by*32 + i, k]) * Scale_B

            T.gemm(A_shared, B_shared, C_local, transpose_B=True)  # FP8×FP8 MMA

            # Scale 校正累加
            for i, j in T.Parallel(32, 128):
                C_local_accum[i, j] += C_local[i, j] * Scale_C_shared[i]

        T.copy(C_local_accum, C_shared)
        T.copy(C_shared, C[by * 32, bx * 128])
```

**与 V3 的关键差异**:

1. **Scale 合并时机不同**: V3 是在 `tl.dot` 之后乘两个 scale。V4 在每个 k iteration 中**先合并 act_scale × weight_scale = Scale_C**，再乘到 accumulator。结果数学上等价，但 V4 的 `Scale_C_shared[i]` 只需 `(32,)` 的形状（比 V3 的 `a_s[:, None] * b_s[None, :]` 中间张量 `(32, 128)` 更省 shared memory）。数学推导见下方 §4.2.1。

2. **4 级流水线**: `num_stages=4` 意味着同时有 4 个 k iteration 的不同阶段在流水线中（加载/GEMM/scale/写回），更好地隐藏内存延迟。

3. **权重 scale 的索引**: `scales_b[bx * 128 // 128, k]` — 权重 scale 形状是 `(N//128, K//128)`，对固定 `bx`（N 方向 block）和 `k`（K 方向 block）取一个 scale 值。这意味着**同一 block_N 内的所有 128 个 N 共享同一个权重 scale**（因为 `bx * 128 // 128 = bx`）。

4. **无显式 CUDA Core 提升步骤**: V3 在 `tl.dot` 后每 N_C=128 做一次 FP32 CUDA Core 累加，并依赖双 warpgroup 隐藏开销。V4 通过预合并 scale 将 post-GEMM 操作压缩为单条 FMA，不再需要显式提升。数学推导见下方 §4.2.2。

#### 4.2.1 Scale 预合并的数学推导

为什么 V3 和 V4 数学等价，但 V4 更省存储？下面给出严格推导。

**符号定义**

设一个 GEMM tile 的维度为：
- 激活 \(A_{\text{fp8}} \in \mathbb{Z}^{M \times K}\)（FP8 已量化值，即 \(A_{\text{fp8}} = \text{round}(A_{\text{true}} / s_a)\) 的整数表示）
- 权重 \(B_{\text{fp8}} \in \mathbb{Z}^{N \times K}\)（FP8 已量化值，类似）
- 输出 \(C_{\text{true}} \in \mathbb{R}^{M \times N}\) 是最终 BF16 结果

其中 \(M = 32\)（激活 tile 的行数），\(N = 128\)（权重 tile 的行数，即输出维度）。

**量化与反量化的基本关系**

对于每个元素，量化值 × scale = 原始值（忽略舍入误差）：

\[
A_{\text{true}}[i, k] = A_{\text{fp8}}[i, k] \cdot s_a[i], \quad
B_{\text{true}}[j, k] = B_{\text{fp8}}[j, k] \cdot s_b[j]
\]

其中 \(s_a[i]\) 是激活第 i 行所在 tile 的 scale，\(s_b[j]\) 是权重第 j 行所在 tile 的 scale。

**GEMM 输出与 scale 的关系**

矩阵乘法的输出元素为 \(C[i,j] = \sum_k A_{\text{fp8}}[i,k] \cdot B_{\text{fp8}}[j,k]\)（即 `C_local`，Tensor Core 的裸输出）。反量化到真实值：

\[
C_{\text{true}}[i, j] = \sum_k A_{\text{true}}[i,k] \cdot B_{\text{true}}[j,k]
= \sum_k \big(A_{\text{fp8}}[i,k] \cdot s_a[i]\big) \cdot \big(B_{\text{fp8}}[j,k] \cdot s_b[j]\big)
\]

由于 \(s_a[i]\) 和 \(s_b[j]\) 不依赖 k，可以提到求和外面：

\[
\boxed{C_{\text{true}}[i,j] = \underbrace{\left(\sum_k A_{\text{fp8}}[i,k] \cdot B_{\text{fp8}}[j,k]\right)}_{C_{\text{local}}[i,j]} \cdot s_a[i] \cdot s_b[j]}
\tag{1}
\]

**V3 的做法：二维 scale 矩阵**

V3 直接在 accumulator 上乘两个 scale：

```python
# V3: a_s 形状 (M,), b_s 形状 (N,)
# 广播为 (M, N) 后逐元素乘
acc += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
```

其中 `a_s[:, None] * b_s[None, :]` 产生一个形状为 `(M, N)` 的中间张量。对于 `M=32, N=128`，这个临时矩阵有 **4096 个元素**，占用 shared memory / 寄存器。

**V4 的优化：利用权重 scale 的块共享性质**

V4 的关键观察来自权重 scale 的存储布局。V4 的 `scales_b` 形状是 `(N//128, K//128)`——每个 scale 管一个 **128×128** 的权重 block。在当前 GEMM tile 内（`M=32, N=128`），所有 128 个 N 维元素**恰好落在同一个 scale block 内**，即：

\[
s_b[0] = s_b[1] = \dots = s_b[127] \triangleq s_b^{\text{block}}
\]

因此公式 (1) 可以简化：

\[
C_{\text{true}}[i, j] = C_{\text{local}}[i, j] \cdot s_a[i] \cdot \underbrace{s_b^{\text{block}}}_{\text{不依赖 } j}
\]

V4 利用这一点，**在 GEMM 之前**把两个 scale 预合并：

\[
\text{Scale\_C}[i] = s_a[i] \cdot s_b^{\text{block}}
\]

然后对 accumulator 只需要乘一个一维向量：

```python
# V4: Scale_C 形状 (M,), 无需广播为 (M, N)
Scale_B = scales_b[bx, k]                         # 标量: 当前 block 的 weight scale
Scale_C_shared[i] = scales_a[by*32 + i, k] * Scale_B  # (M,) 一维向量
C_local_accum[i, j] += C_local[i, j] * Scale_C_shared[i]  # 只对 i 索引
```

**存储量对比**

| | V3 | V4 |
|---|-----|-----|
| scale 中间张量形状 | `(M, N)` = `(32, 128)` | `(M,)` = `(32,)` |
| 元素数 | 4096 | 32 |
| 节省 | — | **128×** |

**等价性验证**

在同一个 N block 内 \(s_b[j] \equiv s_b^{\text{block}}\)，因此：

\[
\boxed{
\begin{aligned}
\text{V3:}&\quad C_{\text{true}}[i,j] = C_{\text{local}}[i,j] \times s_a[i] \times s_b[j] \\
\text{V4:}&\quad C_{\text{true}}[i,j] = C_{\text{local}}[i,j] \times \underbrace{\big(s_a[i] \times s_b^{\text{block}}\big)}_{\text{Scale\_C}[i],\ \text{只依赖 } i}
\end{aligned}
}
\]

两式数学上严格相等。V4 将二维系数矩阵压缩为一维向量，存储量从 \(O(MN)\) 降至 \(O(M)\)。

**前提条件**

这个优化成立的前提是 **同一 block_N 内的所有权重 scale 相同**。V4 的 128×128 tile 权重量化恰好满足这一点——一个 scale 管 128 个连续输出 channel。如果权重量化 tile 更细（比如 1×32，像 FP4 那样），这个优化就不适用了。这也解释了为什么 fp4_gemm 的 scale 乘法和 fp8_gemm 不同：fp4_gemm 的 weight scale 是 per-32 而不是 per-128，无法做同样的预合并。

#### 4.2.2 为什么不需要显式的「提升到 CUDA Core」

先回顾 V3 为什么要「提升」。然后从计算流程上推导 V4 为什么不需要。

**硬件背景：Tensor Core 的累加器精度**

NVIDIA Tensor Core 在做 FP8 矩阵乘时，内部的累加器精度只有 **~14 位**（FP32 是 24 位）[^1]。设 Tensor Core 计算 \(C = A_{\text{fp8}} \times B_{\text{fp8}}^T\)，其中 \(A \in \mathbb{Z}^{M \times 128}, B \in \mathbb{Z}^{N \times 128}\)，每个输出元素是 128 次乘加：

\[
C[i,j] = \sum_{k=1}^{128} A[i,k] \cdot B[j,k]
\]

每次乘法产生一个精确的 16 位乘积（两个 FP8 → 乘积最多需 8+8-1=15 位），但累加器只有 ~14 位，意味着每累加一次就可能丢弃 ~1-2 位。累加 128 次后，最大相对误差约 \(O(\sqrt{128} \times 2^{-14}) \approx 0.07\%\)（V3 论文实测）。

**V3 的做法：显式提升**

V3 的 Triton kernel 中，每次 WGMMA（128 个 K 元素）完成后：

```python
# V3: tl.dot 输出 ~14 位 partial → 乘 scale → 累加到 FP32 accumulator
accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
```

这条语句实际上执行了三件事：

```
① tl.dot(a, b)           → partial:  ~14 位精度, 形状 (M, N)
② partial * a_s * b_s    → 乘以两个 scale, 结果提升到 FP32
③ += accumulator         → 累加到 FP32 running sum
```

步骤 ②-③ 在 **CUDA Core 的 FP32 寄存器**中执行（`tl.dot` 把 partial 写进寄存器文件，紧接的乘加指令消费它）。V3 称这步为「提升到 FP32 累加」。由于这步有开销（两个 scale 的 2D 广播乘法 + FP32 累加），V3 用**双 warpgroup** 来隐藏：一个 warpgroup 做提升时，另一个跑 MMA。

**V4 的分析：为什么不需要显式提升**

V4 的 TileLang kernel 对应的代码：

```python
# V4: 先预合并 scale → GEMM → 单次 FMA
Scale_C_shared[i] = scales_a[by*32 + i, k] * Scale_B  # 预合并, 标量操作
T.gemm(A_shared, B_shared, C_local, transpose_B=True)   # FP8×FP8 MMA
C_local_accum[i, j] += C_local[i, j] * Scale_C_shared[i] # 单次 FMA
```

与 V3 对比，从计算流程上分析：

**（a）操作数量**

| | V3 | V4 |
|---|-----|-----|
| post-GEMM 乘法次数（per element） | 2 次（× a_s, × b_s） | **1 次**（× Scale_C） |
| 中间 broadcast 形状 | `(M,)` → `(M,1)` × `(N,)` → `(1,N)` = `(M,N)` | 无（Scale_C 已是一维） |
| post-GEMM 指令类型 | MUL + MUL + ADD | **单个 FMA** |

V4 的 `C_local * Scale_C_shared[i] + C_local_accum` 是一条 **FMA（Fused Multiply-Add）指令**。V3 需要先算 `a_s[:, None] * b_s[None, :]`（2D broadcast），再乘以 partial，再加到 accumulator——至少 3 条指令。V4 只需 1 条。

**（b）为什么指令少就意味着「不需要显式提升」**

V3 的「提升」之所以成为一个需要特殊处理的步骤（双 warpgroup），是因为 post-GEMM 的运算链太长——3 条指令的延迟无法被 GPU 的指令级并行完全吸收，必须在 warp 级别做切换。

V4 将 post-GEMM 压缩为单条 FMA 后：

1. **延迟足够低**：单条 FMA 的延迟（~4 cycles on H800）可以被 Tensor Core 的流水线自然吸收，不需要 warp 切换。
2. **寄存器压力更低**：不需要存 `(M,N)` 的 2D broadcast 中间结果（V3 需要 32×128=4096 个临时值）。
3. **TileLang 的 4 级流水线**（`num_stages=4`）进一步将 FMA 的延迟隐藏在 GEMM 的加载/执行/写回流水线中。

**（c）精度提升仍然发生，只是不显式标出**

V3 的「提升到 FP32」和 V4 的 `C_local_accum += C_local * scale` 在精度语义上是等价的：

```
V3:  partial(~14b) → × scale_a × scale_b → FP32 result → += FP32 accum
V4:  C_local(~14b) → × Scale_C           → FP32 result → += FP32 accum
```

两者都是把 ~14 位的 Tensor Core 输出乘以 FP32 的 scale，结果自然获得 FP32 的 24 位精度，再加入 FP32 的 running sum。V4 不需要一个单独的「提升」步骤，因为这个提升**已经隐含在 scale 乘法的 FMA 指令里了**。

**总结**

\[
\boxed{
\begin{aligned}
\text{V3:}&\quad \text{acc} \mathrel{+}= \underbrace{\text{tl.dot}(a,b)}_{\text{14-bit}} \times \underbrace{a_s \otimes b_s}_{\text{2D broadcast, 2 MULs}} \quad\rightarrow\quad \text{3 条指令, 需双 warpgroup 隐藏} \\
\text{V4:}&\quad \text{C\_accum} \mathrel{+}= \underbrace{\text{T.gemm}(A,B)}_{\text{14-bit}} \times \underbrace{\text{Scale\_C}[i]}_{\text{1D, 预合并}} \quad\rightarrow\quad \text{1 条 FMA, 流水线自然吸收}
\end{aligned}
}
\]

V4 不是「不做提升」，而是**把提升融合进了 scale 乘法**，并且因为预合并 scale 将操作数从 3 条指令压缩到 1 条 FMA，不再需要双 warpgroup 来隐藏延迟。

### 4.3 激活量化中的 inplace 模式

V4 新增了 `act_quant(inplace=True)` 模式（`kernel.py:84-91`）：

```python
if inplace:
    # 量化后再反量化回 BF16 — 用于 QAT 模拟
    for i, j in T.Parallel(blk_m, group_size):
        y_local[i, j] = T.Cast(
            out_dtype,
            T.Cast(compute_dtype, T.Cast(out_dtype,
                T.clamp(x_local[i, j] / s_local[i], fp8_min, fp8_max)
            )) * s_local[i],
        )
```

流程：`BF16 → FP8(量化) → BF16(反量化)`，输出仍是 BF16，但经历了 FP8 的精度损失。用于：
- **QAT 模拟**: 训练时模拟推理的量化误差，但保持 BF16 数据流
- **KV Cache 模拟**: attention 中非 RoPE 维度用 `inplace=True` 模拟 FP8 存储

inplace 模式的 KV Cache 模拟是下一章的前奏——§五将展开 V4 如何系统性地用混合精度压缩 KV Cache。

---

## 五、逐步分析：KV Cache 混合精度

### 5.1 动机

V3 的 KV Cache 全部以 BF16 存储，但它是长上下文推理的主要内存瓶颈（复杂度 O(seq_len × n_layers × head_dim)）。V4 将 KV 拆分为两部分：

- **RoPE 维度**（`rope_head_dim=64`）：旋转位置编码作用于这些维度 → 位置信息，精度敏感 → **保持 BF16**
- **非 RoPE 维度**（`head_dim - rope_head_dim=448`）：内容信息，对精度相对鲁棒 → **FP8 量化**

对于 head_dim=512 的 V4，非 RoPE 维度占 448/512 ≈ 87.5%，这部分从 BF16 切到 FP8 后，**KV Cache 整体节省 ~44%**。

### 5.2 代码实现

Attention 中 KV 的量化（`model.py:506`）：
```python
kv = self.wkv(x)
kv = self.kv_norm(kv)
apply_rotary_emb(kv[..., -rd:], freqs_cis)        # RoPE 维度保持 BF16
# 非 RoPE 维度做 FP8 inplace 量化模拟
act_quant(kv[..., :-rd], 64, scale_fmt, scale_dtype, True)
#                                         ^ block_size=64 (不是 128!)
#                                                    ^ inplace=True: 量化后反量化回 BF16
```

**注意 block_size=64**：非 RoPE 维度使用 64 而非 128 的 tile size，原因是 448 ÷ 64 = 7 恰好整除（而 448 ÷ 128 = 3.5 不整除，边界 tile 需特殊处理）。

### 5.3 CSA Compressor 中的量化

Compressor 在压缩 KV 后也做量化（`model.py:370-372`）：
```python
if self.rotate:
    kv = rotate_activation(kv)             # Hadamard 旋转
    fp4_act_quant(kv, fp4_block_size, True)  # FP4 inplace
else:
    act_quant(kv[..., :-rd], 64, scale_fmt, scale_dtype, True)  # FP8 inplace
```

带 Hadamard 旋转的 Compressor 用 **FP4** 而非 FP8 — 旋转后信息均匀分布，FP4 精度损失更可控。Hadamard 旋转的原理详见 §六。

CSA 系统中的另一个关键组件——**索引器（Indexer）**，负责从压缩后的 KV Cache 中选取 top-k 位置——的精度设计详见 §七。

---

## 六、逐步分析：Hadamard 旋转

### 6.1 动机

低精度量化（尤其是 FP4）的一个核心问题是**异常值（outlier）**——少数维度的值远大于其他维度，导致 scale 被异常值主导，大部分值的量化精度被浪费。

Hadamard 矩阵是正交矩阵（所有元素为 ±1，按 1/√d 缩放），作用是将信息**均匀旋转**到所有维度：
```
x_rotated = hadamard_transform(x) / sqrt(d)
```
旋转后每个维度是原始所有维度的混合 → 异常值被「稀释」→ 量化更均匀 → 精度损失更小。

### 6.2 在 V4 中的使用

`rotate_activation` 函数（`model.py:247-251`）：
```python
def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.bfloat16
    from fast_hadamard_transform import hadamard_transform
    return hadamard_transform(x, scale=x.size(-1) ** -0.5)
```

使用位置：
- **CSA Compressor**（`compress_ratio=4` 的层）：压缩后的 KV 先做 Hadamard 旋转，再做 FP4 量化
- **CSA Indexer**：Q、KV 都先做 Hadamard 旋转，再做 FP4 量化

在无旋转模式下（`compress_ratio ≠ 4`），Compressor 仅对非 RoPE 维度做 FP8 inplace 量化，不使用 Hadamard 旋转。

---

## 七、逐步分析：CSA 索引器的混合精度

### 7.1 索引器的精度设计

CSA 索引器（`Indexer` 类，`model.py:380-433`）负责从压缩 KV Cache 中选择 top-k 位置。其精度设计包含三层：

**第一层：QK 路径用 FP4**

```python
q = self.wq_b(qr)
q = q.unflatten(-1, (self.n_local_heads, self.head_dim))
apply_rotary_emb(q[..., -rd:], freqs_cis)
q = rotate_activation(q)              # Hadamard 旋转（见 §六）
fp4_act_quant(q, fp4_block_size, True) # FP4 inplace 模拟
self.compressor(x, start_pos)          # Compressor 内部也做 FP4/FP8 量化
```

Q 和 KV 经过 Hadamard 旋转 + FP4 量化后，进入索引分数的计算。

**第二层：索引分数从 FP32 → BF16**

```python
# QAT 后，索引分数 I 从 FP32 量化为 BF16
# top-k 选择器 2× 加速，召回率 99.7%
index_score = torch.einsum("bshd,btd->bsht", q, self.kv_cache[:bsz, :end_pos // ratio])
index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
```

**第三层：注意力分数在 FP32 中计算**

虽然 `sparse_attn_kernel` 内部使用 BF16 加载 Q 和 KV，但 softmax 计算在 FP32 中执行（`acc_s` 和 `acc_o` 都是 FP32 累加器，`kernel.py:308-309`）。

### 7.2 精度选择的逻辑

```
CSA 索引器精度选择:
  QK 计算:  FP4 → 极低精度，计算量大（O(seq²)），加速收益最高
  索引分数: BF16 → top-k 选择对微小误差不敏感（召回率 99.7%）
  注意力分数: BF16 + FP32 softmax → softmax 精度敏感
```

---

## 八、MoE 通信精度

V3 的 MoE 通信方案为 **FP8 Dispatch + BF16 Combine**。
V4 沿用了这一方案，但细化了通信-计算重叠条件（论文 §3.1）：

> 对于 V4-Pro，每个 token-expert 对需要 6hd FLOPs（SwiGLU gate + up + down）但仅需 3h bytes 通信（FP8 Dispatch + BF16 Combine）

这意味着计算-通信比远大于所需阈值，通信可被完全隐藏在计算中。

---

## 九、其他精度相关细节

### 9.1 Muon 优化器的梯度量化

V4 引入 Muon 优化器，其梯度同步时使用 **BF16 随机舍入**量化（论文 §3.4.1）：

> 「我们观察到 Muon 中的 Newton-Schulz 迭代在使用 BF16 矩阵乘法计算时保持稳定。利用这一点，我们进一步以随机舍入方式将要在数据并行排名间同步的 MoE 梯度量化为 BF16 精度，将通信量减半。」

关键在于使用「随机舍入」而非「最近舍入」——随机舍入是无偏估计，能避免系统性偏差的累积。

同步后不直接在 BF16 累加器上做 reduce-scatter，而是改为两阶段：
1. All-to-All 交换局部梯度
2. 每个 rank 在 **FP32** 中做局部求和

这一设计保持了数值稳健性。

### 9.2 批不变性和确定性

V4 用 **DeepGEMM** 替代 cuBLAS（后者不保证批不变性）。在大多数场景放弃 split-k，通过优化使非 split-k 实现达到甚至超越 split-k 性能（§3.3）。

MoE 反向的确定性通过「令牌顺序预处理 + 跨排名缓冲区隔离」保证。

### 9.3 RMSNorm 内部仍用 FP32

V4 的 RMSNorm 与 V3 相同（`model.py:183-196`）：
```python
def forward(self, x):
    x = x.float()                                # BF16 → FP32
    var = x.square().mean(-1, keepdim=True)
    x = x * torch.rsqrt(var + self.eps)          # FP32 计算
    return (self.weight * x).to(dtype)            # 写回原精度
```

---

## 十、与 V3 的关键差异总结

| 维度 | V3 | V4 |
|------|-----|-----|
| **训练 FP8** | Triton kernel，FP32 scale | TileLang kernel + **UE8M0** scale |
| **专家权重精度** | FP8（训练时即 FP8） | FP8（预训练）→ **FP4（后训练 QAT）** |
| **FP4 QAT** | 无 | FP32 master → FP4 quant → FP8 **无损** dequant → FP8 GEMM |
| **FP4 块大小** | N/A | **1×32**（权重 tile 沿 K 维，激活仍为 128），block_size=32 |
| **KV Cache** | 全 BF16 | **BF16（RoPE） + FP8（非 RoPE，1×64 tile）** |
| **Hadamard 旋转** | 无 | CSA Compressor/Indexer 量化前旋转 |
| **FP4×FP8 GEMM** | 无 | 激活 FP8 + 权重 FP4 → FP8 cast → FP8 MMA |
| **Scale 合并** | `a_s[:, None] * b_s[None, :]` | 预合并为 `Scale_C_shared[i]`（省 shared memory） |
| **内核框架** | Triton | **TileLang**（SMT 求解器辅助） |
| **梯度量化** | 无 | Muon MoE 梯度 BF16 随机舍入同步 |

### 10.1 继承自 V3 的部分

以下 V3 的创新在 V4 中全部保留，无变化：
1. **核心 FP8 方案**: 细粒度 tile 量化（1×128 / 128×128）+ FP32 累加
2. **精度敏感 op 清单**: Embedding / RMSNorm / Softmax / SiLU / Gate / Loss 全部 BF16/FP32
3. **FP8 不用于 Attention GEMM**: Attention 保持 BF16（memory-bound + Softmax 敏感）
4. **MoE 通信**: FP8 Dispatch + BF16 Combine
5. **E4M3 统一格式**: 不区分前向/反向格式（与标准 E5M2 反传不同）

---

## 十一、代码路径索引

### TileLang 内核
- [`kernel.py:40-102`](../../code/deepseek-ref/V4/inference/kernel.py#L40-L102) — `act_quant_kernel`: FP8 激活量化（支持 round_scale 和 inplace）
- [`kernel.py:128-200`](../../code/deepseek-ref/V4/inference/kernel.py#L128-L200) — `fp4_quant_kernel`: FP4 激活量化（E8M0 scale, bit 操作）
- [`kernel.py:204-254`](../../code/deepseek-ref/V4/inference/kernel.py#L204-L254) — `fp8_gemm_kernel`: FP8×FP8 GEMM（UE8M0 scale, 预合并 scale）
- [`kernel.py:442-515`](../../code/deepseek-ref/V4/inference/kernel.py#L442-L515) — `fp4_gemm_kernel`: FP8×FP4 混合 GEMM
- [`kernel.py:277-352`](../../code/deepseek-ref/V4/inference/kernel.py#L277-L352) — `sparse_attn_kernel`: 稀疏注意力（FP32 accumulator + online softmax）

### 模型层
- [`model.py:108-121`](../../code/deepseek-ref/V4/inference/model.py#L108-L121) — `linear()`: 双精度分发（FP4 → fp4_gemm, FP8 → fp8_gemm, BF16 → F.linear）
- [`model.py:123-152`](../../code/deepseek-ref/V4/inference/model.py#L123-L152) — `Linear`: 权重初始化（FP4/FP8/BF16 三种格式 + 对应 scale）
- [`model.py:247-251`](../../code/deepseek-ref/V4/inference/model.py#L247-L251) — `rotate_activation()`: Hadamard 旋转
- [`model.py:279-377`](../../code/deepseek-ref/V4/inference/model.py#L279-L377) — `Compressor`: KV 压缩 + 量化（FP4/FP8 两路）
- [`model.py:380-433`](../../code/deepseek-ref/V4/inference/model.py#L380-L433) — `Indexer`: CSA 索引器（FP4 QK + BF16 索引分数）
- [`model.py:436-543`](../../code/deepseek-ref/V4/inference/model.py#L436-L543) — `Attention`: MLA（KV 混合精度存储）
- [`model.py:587-606`](../../code/deepseek-ref/V4/inference/model.py#L587-L606) — `Expert`: SwiGLU FFN（FP32 内部计算）
- [`model.py:609-644`](../../code/deepseek-ref/V4/inference/model.py#L609-L644) — `MoE`: 专家路由 + FP32 累加

---

## 参考文献

1. DeepSeek-V4 Technical Report — [arXiv](https://arxiv.org/abs/2505.18206) · [中文翻译](../../papers-zh/DeepSeek-V4-technical-report.zh.md)
2. DeepSeek-V3 Technical Report §3.3 — [中文翻译](../../papers-zh/DeepSeek-V3.md#33-fp8-训练)
3. Rouhani et al., 2023 — Microscaling (MX) Formats — MXFP4/MXFP8/MXFP6 规范
4. Wang et al., 2026 — TileLang: A DSL for Composable GPU Kernel Development
5. Jordan et al., 2024 — Muon: Momentum Orthogonalized by Newton-Schulz
6. [DeepSeek-V3 FP8 训练详解](./deepseek-v3-fp8-training.md) — 本文档的前置阅读
