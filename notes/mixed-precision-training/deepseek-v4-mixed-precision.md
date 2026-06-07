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
| **CSA 索引器 QK** | N/A（V4 新组件）| **FP4** ⚡ 新 | QK 全路径 FP4 缓存/加载/计算，加速长上下文注意力 |
| **CSA 索引器 索引分数** | N/A | FP32 → **BF16**（QAT 后）| top-k 选择器 2× 加速，召回率 99.7% |
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

## 二、V4 核心创新一：UE8M0 幂次缩放格式

### 2.1 什么是 UE8M0

V3 的 scale factor 存为 **FP32**（32-bit）。V4 改为 **UE8M0**（8-bit，仅 8 位指数，无符号位，无尾数，也叫 **E8M0**）。

| 格式 | 位数 | 指数 | 尾数 | 可表示值 | 备注 |
|------|------|------|------|----------|------|
| FP32 scale (V3) | 32 | 8 | 23 | 任意实数 | 范围大、精度高，但占用多 |
| UE8M0 scale (V4) | **8** | 8 | 0 | **仅为 2 的幂次** | 256 个值：2⁻¹²⁷, 2⁻¹²⁶, ..., 2¹²⁷ |

UE8M0 = **U**nsigned **E**xponent **8**-bit **M**antissa **0**-bit。只有指数，所以值只能是 2 的整数次幂。

### 2.2 为什么用幂次缩放

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

Step 3: ceil(−6.457) = −6    ← 向负无穷方向取整？不，是向正无穷：−6 > −6.457

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

### 2.3.2 为什么 `|x / scale| ≤ 448` 恒成立

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

### 2.3.4 为什么 8 位指数不会溢出？

这个问题可以反过来想：**UE8M0 的 8 位指数和 BF16/FP32 的指数位宽完全一样**。

| 格式 | 指数位宽 | 动态范围 |
|------|----------|----------|
| FP32 | 8-bit | 2⁻¹²⁷ ~ 2¹²⁶ |
| BF16 | 8-bit | 2⁻¹²⁷ ~ 2¹²⁶ |
| **UE8M0** | **8-bit** | **2⁻¹²⁷ ~ 2¹²⁶** |

三种格式的动态范围相同。为什么？因为 IEEE 754 浮点数的范围只由指数位数决定，尾数只影响精度。

**关键观察**：scale 是从数据本身算出来的 — `scale ≈ amax / 448`。数据本身的 `amax` 能存在 BF16/FP32 里（意味着它的指数在 8-bit 范围内），那 scale 的指数也在同一个范围内。UE8M0 只是丢掉了尾数精度，**没有丢掉指数范围**。

具体来说几点保障：

1. **下限不会 underflow**：激活值经过 RMSNorm 后，`amax` 通常在 [0.1, 100] 量级，对应的 scale 指数在 [−10, 10] 附近，远没到 E8M0 的下限 2⁻¹²⁷。

2. **上限不会 overflow**：即使极端情况（比如 embedding 输出 `amax = 1000`），scale = 2^ceil(log2(1000/448)) ≈ 2^2 = 4，指数 = 2，远小于上限 2¹²⁶。

3. **最极端情况**：假设某个 tile 全是最大 BF16 值（~3.4×10³⁸），scale ≈ 10³⁶，指数 ~120，仍在 E8M0 范围（上限 126）内。但这种情况在正常训练中不会出现 — RMSNorm 把激活值约束在合理范围内。

**一句话**：UE8M0 不是「缩小了范围来省空间」，而是「保持了和 FP32/BF16 相同的指数范围，只去掉了不需要的尾数精度」。

### 2.3.5 为什么去掉尾数精度不会损害模型？

这是更核心的问题。UE8M0 把 scale 从任意实数 round 到了 2 的幂次，比如 `0.01138 → 0.015625`，误差 ~37%。为什么 scale 差这么多，量化质量不崩？

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

但这最多让 scale 误差达到 ~2 倍（向上取整到最近的 2 的幂次，误差 ≤ 2×），对应的 FP8 精度损失在 3-bit 尾数下仍然可接受。

**原因 2：细粒度 tile 量化把误差隔离在每个 128 元素内**

每个 tile 有独立的 scale，tile 之间互不影响。一个 tile 的 scale 被 round 到 0.015625，不会影响其他 tile。对比 per-tensor 量化：

```
Per-tensor:  整个矩阵一个 scale，如果一个 outlier 让 scale 变大，
             所有位置都受影响 → 灾难性精度损失

Per-tile (V4): 每个 128×1 block 独立 scale，
               outlier 只影响自己所在 tile → 误差被隔离
```

**原因 3：反向来看 — ceil 的误差到底去了哪里？**

这是个合理质疑。ceil 操作让 V4 的 scale 可能比理想值大最多 ~2 倍，直觉上「scale 不准了」=「反量化结果不准了」。但用一个 2×2 矩阵 + 2 元素向量的完整 GEMM 算一遍，看看误差到底在哪。

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

**核心结论：ceil 误差不等于反量化误差**。

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

### 3.1 为什么用 FP4

DeepSeek-V4 的 MoE 层有数百个专家（Flash: 256, Pro: 384），专家权重是 GPU 显存的主要占用。FP4 相比 FP8 将权重存储量再减半。

FP4 E2M1 格式：

| 位模式 | 符号 | 指数 | 尾数 | 值 |
|--------|------|------|------|-----|
| 0 0 0 0 | + | 2⁰ | 0.0 | 0 |
| 0 0 0 1 | + | 2⁰ | 0.5 | 0.5 |
| 0 0 1 0 | + | 2⁰ | 1.0 | 1.0 |
| 0 0 1 1 | + | 2⁰ | 1.5 | 1.5 |
| 0 1 0 0 | + | 2¹ | 0.0 | 2.0 |
| ... | | | | |
| 1 1 1 1 | − | 2³ | 1.5 | −12.0 |

共 16 个可表示值，范围 [−12, 12]。粒度非常粗——每个值之间间隔很大。

### 3.2 从 Expert 的前向传播出发

先看 Expert 做了什么（`model.py:587-606`）：

```python
class Expert(nn.Module):
    def __init__(self, dim, inter_dim, dtype=None):
        self.w1 = Linear(dim, inter_dim, dtype=dtype)   # gate 投影
        self.w2 = Linear(inter_dim, dim, dtype=dtype)   # down 投影
        self.w3 = Linear(dim, inter_dim, dtype=dtype)   # up 投影

    def forward(self, x):        # x: [tokens, dim]
        gate = self.w1(x)        # gate 投影 → FP8 GEMM
        up   = self.w3(x)        # up 投影   → FP8 GEMM
        x = F.silu(gate) * up    # SwiGLU 激活 (BF16)
        return self.w2(x)        # down 投影 → FP8 GEMM
```

当 `expert_dtype = "fp4"` 时，`w1, w2, w3` 的权重存为 FP4。以 w2 为例（down 投影），维度为 `[dim, inter_dim]`。

**FP4 权重存储格式**（`model.py:131-137`）：

```
FP4 数据:  shape [dim, inter_dim // 2], dtype = float4_e2m1fn_x2
           → 每字节存 2 个 FP4 值, inter_dim 方向被「对折」

FP4 scale: shape [dim, inter_dim // 32], dtype = float8_e8m0fnu (UE8M0)
           → 每 32 个元素 (沿 inter_dim) 共享一个 scale
           → 这就是 1×32 tile 量化
```

**拿一个具体元素来看**。假设 w2 的第 row 行、第 col 列（沿 inter_dim = K 方向），原始 FP32 值为 `1.875`：

```
原始值 (FP32 master):  w = 1.875

FP4 量化 (col 所在的 32-group, group = col // 32):
  该 group 的 amax = 4.8 (假设)
  scale = 2^ceil(log2(4.8 / 6))      ← FP4 max=6, 所以除 6 而非 448
        = 2^ceil(log2(0.8)) = 2^ceil(-0.322) = 2^0 = 1.0

  w_fp4 = round(1.875 / 1.0, fp4)
        → FP4 E2M1 可表示值: {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
        → 最近值是 2.0

存储:
  fp4_data[row, col//2] 中存 2.0 (4 bits)
  fp4_scale[row, group] 中存 1.0 = 2^0 (8 bits, UE8M0)
```

**前向传播时**，`self.w2(x)` 调用链: `linear(x, w2_fp4)` → `fp4_gemm(x_fp8, x_scale, w2_fp4, w2_scale)`。

fp4_gemm 内部（`kernel.py:466-513`），每次处理 32 个 K 元素：

```python
for k in range(K // 32):
    a = load_act_fp8(k)            # FP8 激活, [block_M, 32]
    b = load_wt_fp4(k)             # FP4 权重, [block_N, 32]

    b_fp8 = cast(b, FP4→FP8)      # ← 步骤 1: 精度扩展, 不乘 scale
                                     #   2.0(FP4) → 2.0(FP8)

    scale_w = wt_scale[k]          # = 1.0 (这个 32-group 的 UE8M0 scale)
    scale_a = act_scale[k // 4]    # 激活 scale (每 128 K 一个)

    C_fp8 = a_fp8 @ b_fp8^T       # ← 步骤 2: FP8×FP8 MMA (裸值相乘)

    C += C_fp8 * scale_a * scale_w # ← 步骤 3: 乘 scale 完成「反量化」
                                     #   C_fp8[i,j] * scale_a[i] * scale_w[j]
```

**【关键】「反量化」被拆成了两步**：

```
步骤 1 (FP4→FP8 cast): 把 FP4 的 1-bit 尾数扩展到 FP8 的 3-bit 尾数
  → 2.0 在两种格式中都是精确的 → 无信息损失 ✓

步骤 3 (乘 scale): C_fp8 × scale_w = C_fp8 × 1.0 
  → 乘 2^k (UE8M0) 在 FP32 中是纯指数移位 → 无舍入 ✓
```

**但注意**：步骤 1+3 合起来，重构出的是 `w_fp4 × scale_w = 2.0 × 1.0 = 2.0`，而原始值是 `1.875`。这 `2.0 ≠ 1.875` 的误差来自 **FP4 量化时的 round**（FP32 → FP4 的精度损失），不是来自反量化。反量化本身（2.0 × 1.0 = 2.0）在数值上是精确的。

### 3.3 「FP4→FP8 无损反量化」精确含义

论文说的「无损」不是指「FP4 权重能完美还原 FP32 原始值」（那不可能，FP4 只有 16 个值），而是指 **FP4 的存储信息在反量化到 FP8/FP32 时不会产生额外的舍入**：

```
「无损」= 反量化操作本身不引入新误差

  w_fp4 × scale_w  →  w_fp8/fp32  这一步是精确的

  因为:
    1. w_fp4 (FP4 的 16 个值) ⊆ FP8 的可表示值集合 (256 个值)
       → FP4→FP8 cast 只是尾数补零, 不丢信息
    2. scale_w = 2^k → 乘法在 FP32 中等于指数加法, 不产生舍入

「有损」= 反量化结果超出 FP8 表示范围 → overflow/underflow
  
  需要: w_fp4_max × max(scale_w) ≤ 448 (FP8 上限)
  由于 FP8 比 FP4 多 2 个指数位 (16× vs 4× range)
  相邻 32-group 的 scale 差异 ≤ 4× 就不会溢出
  → 论文经验验证实际权重满足此条件
```

用刚才的例子验证：

```
w_fp4 = 2.0, scale_w = 1.0
→ w_fp4 × scale_w = 2.0 ≤ 448 ✓ (不溢出)
→ 2.0 在 FP8 中精确可表示 ✓
→ 反量化无损 ✓ (但原始值 1.875 → 2.0 的损失在 FP4 round 阶段已发生)
```

### 3.4 什么时候会有损

如果同一个 Expert 权重矩阵中，某个 32-group 的 scale 极大、另一个极小：

```
group A (正常):  w_fp4 = 6.0,  scale = 2^3 = 8
                  6.0 × 8 = 48  → FP8 可表示 ✓

group B (极端):  w_fp4 = 6.0,  scale = 2^8 = 256
                  6.0 × 256 = 1536 > 448 → FP8 overflow ✗ 有损!
```

论文经验验证了训练后的 Expert 权重不存在这种极端情况 — 相邻 32-group 的 scale 分布足够均匀。

### 3.5 反向传播：直通估计器（STE）

QAT 中的反向传播不需要额外设计：

```
Forward:  FP32 master → FP4 quant → FP8 dequant → FP8 GEMM → loss
Backward: d(loss)/d(FP4) ← [ FP8 GEMM backward ] ← d(loss)/d(FP8_weight)

关键: 梯度直接流过 FP4→FP8 的 cast（无梯度操作），
      等效于 Straight-Through Estimator (STE)
```

这也避免了 V3 中「转置权重需要重新量化」的问题——FP4 的 transpose 就是 FP4 的 transpose，反量化到 FP8 后再做 GEMM 即可。

### 3.6 推理部署：直接使用 FP4

后训练完成后，推理阶段**不需要 FP32 master**——直接加载 FP4 权重做计算：

```
推理时:
  FP4 weight（从 checkpoint 直接加载）
  → fp4_gemm(): FP4→FP8 cast → FP8×FP8 GEMM
```

目前硬件上 FP4×FP8 的峰值 FLOPS 与 FP8×FP8 相同（因为 FP4→FP8 cast 后仍走 FP8 Tensor Core），但：
- **内存加载减半**（FP4 vs FP8）
- **未来硬件**: FP4×FP8 理论上可达 1.33× FP8×FP8 的峰值吞吐

### 3.7 代码中的 FP4 GEMM

`fp4_gemm_kernel`（`kernel.py:442-515`）实现了混合精度 GEMM：

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
5. **不需要 V3 风格的提升累加** — 因为仍走 FP8 Tensor Core，且 scale 直接乘在 accumulator 上

---

## 四、逐步分析：FP8 GEMM（继承 V3，但换了格式和框架）

### 4.1 V4 FP8 GEMM 的流程

与 V3 相同的三阶段结构（详见 [V3 文档 §3.6](./deepseek-v3-fp8-training.md#36-阶段-2fp8-gemm全流程)），但有三个变化：

| 阶段 | V3 | V4 |
|------|-----|-----|
| 量化 | Triton kernel，FP32 scale | TileLang kernel，**UE8M0** scale，bit 操作计算 |
| GEMM | Triton `tl.dot`，FP32 accumulator | TileLang `T.gemm`，FP32 accumulator |
| Scale 乘回 | `acc += dot * a_s[:, None] * b_s[None, :]` | `C_local_accum += C_local * Scale_C_shared[i]`（预合并了 a_s × b_s） |

### 4.2 V4 FP8 GEMM kernel 详解

`fp8_gemm_kernel`（`kernel.py:204-254`）：

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

1. **Scale 合并时机不同**: V3 是在 `tl.dot` 之后乘两个 scale。V4 在每个 k iteration 中**先合并 act_scale × weight_scale = Scale_C**，再乘到 accumulator。结果数学上等价，但 V4 的 `Scale_C_shared[i]` 只需 `(32,)` 的形状（比 V3 的 `a_s[:, None] * b_s[None, :]` 中间张量 `(32, 128)` 更省 shared memory）。

2. **4 级流水线**: `num_stages=4` 意味着同时有 4 个 k iteration 的不同阶段在流水线中（加载/GEMM/scale/写回），更好地隐藏内存延迟。

3. **权重 scale 的索引**: `scales_b[bx * 128 // 128, k]` — 权重 scale 形状是 `(N//128, K//128)`，对固定 `bx`（N 方向 block）和 `k`（K 方向 block）取一个 scale 值。这意味着**同一 block_N 内的所有 128 个 N 共享同一个权重 scale**（因为 `bx * 128 // 128 = bx`）。

4. **无显式 CUDA Core 提升步骤**: V3 在 `tl.dot` 后每 N_C=128 做一次 FP32 CUDA Core 累加。V4 通过 `C_local_accum` 做 scale 校正累加，本质上也是 FP32 累加（`accum_dtype=FP32`），但不需要「提升到 CUDA Core」这个额外步骤——**Tensor Core 的 FP32 输出直接在寄存器中，`C_local_accum[i, j] += C_local[i, j] * Scale_C_shared[i]` 这条 multiply-add 可以融合在 Tensor Core 输出的下游**。

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

---

## 五、逐步分析：KV Cache 混合精度

### 5.1 动机

V3 的 KV Cache 全部存 BF16。但 KV Cache 是长上下文推理的主要内存瓶颈（O(seq_len × n_layers × head_dim)）。V4 将 KV 拆分为两部分：

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

**注意 block_size=64**：非 RoPE 维度用 64 而非 128 的 tile size。因为 448 ÷ 64 = 7，整除（448 ÷ 128 = 3.5 不整除）。

### 5.3 CSA Compressor 中的量化

Compressor 在压缩 KV 后也做量化（`model.py:370-372`）：
```python
if self.rotate:
    kv = rotate_activation(kv)             # Hadamard 旋转
    fp4_act_quant(kv, fp4_block_size, True)  # FP4 inplace
else:
    act_quant(kv[..., :-rd], 64, scale_fmt, scale_dtype, True)  # FP8 inplace
```

带 Hadamard 旋转的 Compressor 用 **FP4** 而非 FP8 — 旋转后信息均匀分布，FP4 精度损失更可控。

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

在「无旋转」模式下（`compress_ratio ≠ 4`），Compressor 只对非 RoPE 维度做 FP8 inplace 量化，不用 Hadamard。

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

V3 的 MoE 通信：**FP8 Dispatch + BF16 Combine**。
V4 保持相同方案，但细化了通信-计算重叠条件（论文 §3.1）：

> 对于 V4-Pro，每个 token-expert 对需要 6hd FLOPs（SwiGLU gate + up + down）但仅需 3h bytes 通信（FP8 Dispatch + BF16 Combine）

这意味着计算-通信比 **C/B** 远大于所需值，通信可以被完全隐藏在计算中。

---

## 九、其他精度相关细节

### 9.1 Muon 优化器的梯度量化

V4 引入 Muon 优化器，其梯度同步时使用 **BF16 随机舍入**量化（论文 §3.4.1）：

> 「我们观察到 Muon 中的 Newton-Schulz 迭代在使用 BF16 矩阵乘法计算时保持稳定。利用这一点，我们进一步以随机舍入方式将要在数据并行排名间同步的 MoE 梯度量化为 BF16 精度，将通信量减半。」

关键是「随机舍入」而非「最近舍入」— 随机舍入是无偏估计，避免系统性偏差累积。

同步后不直接在 BF16 累加器上做 reduce-scatter，而是改为两阶段：
1. All-to-All 交换局部梯度
2. 每个 rank 在 **FP32** 中做局部求和

这保持数值稳健性。

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
| **训练 FP8** | Triton kernel，FP32 scale | TileLang kernel，**UE8M0** scale |
| **专家权重精度** | FP8（训练时即 FP8） | FP8（预训练）→ **FP4（后训练 QAT）** |
| **FP4 QAT** | 无 | FP32 master → FP4 quant → FP8 **无损** dequant → FP8 GEMM |
| **FP4 块大小** | N/A | **1×32**（权重沿 K 维），**block_size=32** |
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
