# 混合精度训练 (Mixed Precision Training)

> **一句话概述**: 在训练过程中混合使用多种数值精度（FP32/FP16/BF16/FP8/FP4），在保持模型精度的前提下降低显存占用和提升计算吞吐。
> **核心问题**: 如何在「数值精度」与「计算效率/显存开销」之间找到最优 trade-off？

---

## 一、先验知识 / 基础知识 (Prerequisites)

### 1.1 浮点数表示格式 (Floating Point Formats)

浮点数的一般表示形式：

\[
x = (-1)^{s} \times 2^{e - \text{bias}} \times (1 + m)
\]

其中 \(s\) 为符号位，\(e\) 为指数位（编码值），\(m\) 为尾数位（小数部分，隐含整数部分 1）。指数为全 0 时进入 subnormal（次正规数）范围，此时隐含位变为 0。

> **总览表（复习时只看这个）**:

| 格式 | 位分配 (S-E-M) | 指数偏置 | 正常数范围 | 机器精度 \(2^{-M}\) | DL 中的角色 |
|------|---------------|---------|-----------|-------------------|------------|
| **FP32** | 1-8-23 | 127 | ±[1.18e-38, 3.40e38] | 1.19e-7 | Master weight 权威副本 |
| **FP16** | 1-5-10 | 15 | ±[6.1e-5, **65504**] | 9.77e-4 | 早期混合精度，需 Loss Scaling |
| **BF16** | 1-8-7 | 127 | **同 FP32** | 7.81e-3 | 当前主流，无需 Loss Scaling |
| **FP8 E4M3** | 1-4-3 | 7 | ±[0.00195, 448] | 0.125 | 前向传播（精度优先） |
| **FP8 E5M2** | 1-5-2 | 15 | ±[6.1e-5, 57344] | 0.25 | 反向传播（范围优先） |
| **FP4 E2M1** | 1-2-1 | — | ±[0.5, 6.0] | — | QAT 模拟，不可直接训练 |

> **关键 takeaways**:
> - BF16 的动态范围 = FP32（8 位指数相同）→ **不需要 Loss Scaling**，这是它碾压 FP16 的根本原因。
> - FP8 按用途分化：E4M3 偏精度（Fprop），E5M2 偏范围（Dgrad/Wgrad）。
> - FP4 仅 16 个可表示值 → 不能直接训练，必须走 QAT（FP32 master → FP4 quant → FP8 dequant → GEMM）。
> - 计算吞吐 (H100 Tensor Core, vs BF16): FP4 ~4×, FP8 ~2×, FP16/BF16 ~1×, FP32 无 Tensor Core 很慢。

---

#### 1.1.1 FP32 — 训练权威副本

- **Why matters**: 深度学习的"黄金标准"精度。FP32 master weight 是几乎所有混合精度方案的基石——模型的"权威副本"始终保存在 FP32。
- **关键属性**: 8 位指数（bias=127）→ 动态范围极大；23 位尾数 → 精度 ~7 位十进制有效数字。
- **Downside**: 显存开销大（4 bytes/element），无 Tensor Core 加速 → 训练时只在 optimizer step 和精度敏感 op 中使用。

#### 1.1.2 FP16 — 动态范围小是致命伤

- **Why matters**: 最早被用于混合精度训练的格式。CUDA Tensor Core 原生支持 FP16 GEMM。
- **核心问题**: **动态范围太小** — 正常数最大值仅 65504（overflow），最小仅 6.1e-5（underflow）。梯度值经常超出这个范围 → **必须使用 Loss Scaling**（见 §1.4）。
- **当前地位**: 已被 BF16 取代，除非硬件不支持 BF16（V100 及更早）。

#### 1.1.3 BF16 — 当前主流

- **Why matters**: 动态范围 = FP32（8 位指数相同），**不需要 Loss Scaling**，是当前主流大模型训练的默认选择（V3、V4 均大量使用 BF16）。
- **关键洞察**: BF16 等于 FP32 的高 16 位截断 — BF16 → FP32 转换是 trivial 的（低位补零），FP32 → BF16 需要最近舍入（round-to-nearest-even）。
- **Trade-off**: 精度比 FP16 低 ~4×（7 位 vs 10 位尾数），但对大模型训练来说，**动态范围的重要性远大于尾数精度**——梯度 underflow 导致训练卡住的灾难性后果 >> 舍入误差的微小精度损失。

#### 1.1.4 FP8 — 分化设计，各司其职

- **E4M3（高精度型）**: 尾数位多 → 精度更高 → 适合**前向传播**（Fprop、激活值）。能表示 ±448 范围覆盖绝大多数激活值。无 Inf 表示（保留给 NaN）。
- **E5M2（大动态范围型）**: 指数位多 → 动态范围大 → 适合**反向传播**（Dgrad、Wgrad）。梯度值可能非常小，需要更大动态范围。有 Inf 和 NaN。
- **Why matters**: H100/H800 的 FP8 Tensor Core 吞吐是 BF16 的 2×，DeepSeek-V3 首次在大规模训练（671B MoE）中验证了 FP8 全链路训练的可行性。
- **核心技巧**: 不是直接替换 BF16 → FP8，而是配合细粒度量化（§1.3）+ 累加精度提升 + 精度敏感 op 保留 BF16（§1.5）— 三位一体。

#### 1.1.5 FP4 — 极限压缩，QAT 必需

- **Why matters**: DeepSeek-V4 用 FP4 做 QAT（量化感知训练），不是直接训 FP4 — 而是 FP32 master → 量化到 FP4 → 反量化到 FP8 → GEMM。FP4 压缩比 4×（vs BF16），推理时原生 FP4 GEMM 获得 2× 加速。
- **核心限制**: 可表示值仅 16 个（E2M1）→ 单靠 FP4 本身精度不够 → 必须依赖细粒度分组缩放（per-group scaling）+ QAT 在训练时补偿量化误差。
- **与 FP8 的关系**: FP4 QAT 的前向计算仍依赖 FP8 GEMM（FP4 → dequant → FP8 → Tensor Core）— FP4 是存储/通信格式，FP8 是计算格式。

### 1.2 次正规数 (Subnormal / Denormal Numbers)

- **What**: 当指数位全为 0 时，隐含整数位变为 0（而非 1），可表示比最小正常数更接近 0 的值。
- **Why matters**: 
  - FP16 的正常数下限约为 \(6.1\times 10^{-5}\)，梯度值经常低于此值 → 进入 subnormal 范围 → 精度进一步下降。
  - Subnormal 值的计算在 GPU 上可能触发微码处理 → **性能急剧下降**。
  - BF16 虽然也有 subnormal，但其正常数下限与 FP32 相同（\(1.18\times 10^{-38}\)），实际训练中几乎不会遇到 → 这是 BF16 不需要 loss scaling 的根本原因。

### 1.3 量化基础：缩放因子与分组量化 (Scale Factor & Group Quantization)

#### Per-Tensor 量化

对整个张量使用一个缩放因子：

\[
x_{\text{quant}} = \text{round}\left(\frac{x_{\text{fp32}}}{s}\right), \quad s = \frac{\max(|x|)}{q_{\max}}
\]

- **问题**: 如果张量中存在 outlier（数值远大于其他元素），缩放因子会被 outlier 主导 → 大部分正常值的量化精度被浪费。
- **典型场景**: Transformer 的激活值中经常出现 channel-wise outlier（某些 channel 的值比平均值大 10-100×）。

#### Per-Tile / Per-Group 量化（细粒度量化）

将张量切分为 tile（如 `1×128` 或 `128×128`），每个 tile 独立缩放：

\[
s_{\text{tile}} = \frac{\max(|x_{\text{tile}}|)}{q_{\max}}
\]

- **DeepSeek-V3 方案**: 激活量化用 `1×128` tile（每个 token 的每 128 个 channel 为一组），权重量化用 `128×128` tile。
- **优势**: Outlier 只影响其所在 tile 的缩放因子，其他 tile 不受影响 → 整体量化精度大幅提升。
- **开销**: 需要存储每个 tile 的 scale factor（BF16 精度），额外显存开销约为 `2 / tile_size × 100%`。对于 `1×128` tile → ~1.56% 额外开销。

#### 缩放因子的在线计算（Online Scaling）

- DeepSeek-V3 不保存历史 max 值，而是在每次前向/反向时实时计算 scale factor。
- 好处: (1) 不需要维护 running statistics；(2) 不会因为历史 max 过大导致当前 step 精度下降。
- 代价: 额外的 max reduction kernel，但这部分开销可与 GEMM 重叠。

### 1.4 混合精度训练的常规流程

#### 标准 BF16 混合精度流程

```
┌─────────────────────────────────────────────────────────┐
│  FP32 Master Weights (权威副本)                         │
│       ↓ cast to BF16                                    │
│  BF16 Forward Pass                                       │
│       ↓                                                  │
│  BF16 Backward Pass (梯度计算)                           │
│       ↓ cast to FP32                                    │
│  FP32 Gradient → FP32 Optimizer Update (Adam/AdamW)     │
│       ↓                                                  │
│  FP32 Master Weights (更新后)                            │
└─────────────────────────────────────────────────────────┘
```

**关键点**:
- **Master Weight 永远是 FP32**: 保证参数更新的累积精度（AdamW 的 \(m_t, v_t\) 也是 FP32）。
- **前向和反向用低精度**: BF16 GEMM 在 Tensor Core 上计算，吞吐高。
- **无需 Loss Scaling**: BF16 的动态范围与 FP32 一致，梯度几乎不会 underflow。
- **精度敏感的 op 保留高精度**: Embedding、Norm、Softmax、Loss 等（详见 1.5 节）。

#### FP16 混合精度流程（需 Loss Scaling）

```
┌─────────────────────────────────────────────────────────┐
│  FP32 Master Weights                                     │
│       ↓ cast to FP16                                    │
│  FP16 Forward Pass                                       │
│       ↓                                                  │
│  Loss × Loss Scale (放大)                                │
│       ↓                                                  │
│  FP16 Backward Pass (梯度被放大)                         │
│       ↓ Unscale (梯度 / Loss Scale)                     │
│  FP32 Gradient → Optimizer Update                        │
│       ↓                                                  │
│  FP32 Master Weights                                     │
└─────────────────────────────────────────────────────────┘
```

- **Loss Scaling 的必要性**: FP16 正常数下限 \(6.1\times 10^{-5}\)，很多梯度值（尤其是靠近输出层的）天然很小 → underflow → 梯度为零 → 训练卡住。
- **动态 Loss Scaling**: 训练过程中自动调整 scale factor。检测到 inf/NaN 时减小 scale；连续 N 步无 overflow 时增大 scale。
- **FP16 vs BF16 的选择**: 如果硬件支持 BF16（A100+），应**优先使用 BF16**，完全避免 loss scaling 的复杂性。

### 1.5 精度敏感算子 (Precision-Sensitive Operations)

这是大模型训练中的**核心经验知识**。以下 op 通常需要保留高精度（FP32/BF16），不能直接使用更低精度（FP8）：

| 算子 / 组件 | 敏感程度 | 原因 | 处理方案 |
|------------|---------|------|---------|
| **Embedding** | 🔴 极高 | 词表大（~100K），embedding 行数多，每行 4096+ 维 → 量化误差累积严重。Embedding 参与 token 的初始表示，误差会传播到整个网络。 | 保持 BF16/FP32，不量化到 FP8 |
| **Attention Softmax** | 🔴 高 | 指数运算 \(e^x\) 对输入误差极度放大。FP8 下 softmax 的指数映射误差可导致 attention score 分布畸变。 | 保持 FP32/BF16（FA-3 中 attention 内部可部分用 FP8，但 softmax 本身保持高精度） |
| **LayerNorm / RMSNorm** | 🔴 高 | 归一化操作的统计量（mean/std）对精度敏感。分母中的 \(1/\sqrt{\text{Var}}\) 在小方差时被放大。 | 保持 FP32/BF16 计算统计量和归一化 |
| **Cross-Entropy Loss** | 🔴 极高 | 最终输出 → 直接影响训练信号。Log-softmax 在概率接近 0 时输出极大负值 → FP8 无法表示。 | 保持 FP32 |
| **GELU / SiLU 激活函数** | 🟡 中 | 近似计算（如 tanh 近似）对精度有一定容忍度，但 FP8 下近似误差可能累积。 | 可用 FP8 计算（需 tile scaling），关键路径保留 BF16 |
| **梯度累加 (All-Reduce)** | 🟡 中 | 多卡梯度求和时，FP16/BF16 的精度损失可能累积。 | 梯度 All-Reduce 通常保持 FP32（ZeRO 框架默认） |
| **Optimizer Step (AdamW)** | 🔴 极高 | 更新量 \(lr \times m_t / (\sqrt{v_t} + \epsilon)\) 往往远小于参数值 → FP8 下完全 underflow。 | 必须 FP32（master weight + optimizer state） |
| **SwiGLU Gate/Up/Down** | 🟢 低 | 矩阵乘法对精度相对鲁棒，FP8/FP4 GEMM 下仍能提供有效梯度。 | 可用 FP8 GEMM（需细粒度量化 + FP32 累加） |
| **MoE Router (Gate)** | 🟡 中 | Router 输出决定 token 路由 → 精度损失可能改变专家分配 → 影响训练动态。 | 保持 BF16 或更高 |
| **Multi-Token Prediction (MTP) Heads** | 🟡 中 | MTP 的多个输出头独立计算 loss → 精度要求与主 head 类似。 | 保持 BF16/FP32 |
| **RoPE (旋转位置编码)** | 🟢 低 | 三角函数计算，输入范围有限 → FP8 可接受。 | 可用 FP8/BF16 |
| **All-to-All (EP 通信)** | 🟡 中 | V3 中用 BF16 dispatch + combine，V4 中 dispatch 用 FP8（省一半带宽）但 combine 保持 BF16（精度敏感）。 | 见 V3/V4 论文 |

**经验法则 (Rule of Thumb)**:
- **非线性操作** → 保持高精度（softmax、norm、activation 的第一层近似）
- **线性操作（GEMM）** → 可用低精度（有细粒度量化 + FP32 累加兜底）
- **统计量计算** → 保持高精度（mean、variance、running statistics）
- **最终输出** → 保持高精度（loss、logits）

### 1.6 累加精度 (Accumulation Precision)

- **What**: GEMM 中，中间累加结果用多少位存储。
- **Why matters**: H100/H800 的 FP8 Tensor Core 只保留 **14 位累加精度**（远低于 FP32 的 23 位）。当内维维度 \(K\) 较大时，累加误差可能显著。
- **DeepSeek-V3 的做法**: GPU 上每 128 个元素做一次 partial sum，将 FP8 Tensor Core 的 14 位部分和提升到 CUDA Core 的 FP32 做下一次累加 → 每 128 个元素「刷新」一次精度。
- **对应公式**: 内维维度 = \(K\)，每 \(N_{\text{acc}}\) 个元素提升一次，总提升次数 = \(\lceil K / N_{\text{acc}} \rceil\)，其中 \(N_{\text{acc}} = 128\)（V3 选择的窗口大小）。

---

## 二、核心内容 (Core Content)

> **方案对比总览（复习时只看这个）**:

| 维度 | FP16 + AMP | BF16 | FP8 (V3) | FP4 QAT (V4) |
|------|-----------|------|----------|--------------|
| **动态范围** | 小 (max 65504) | 大 (=FP32) | 中 (E4M3: 448, E5M2: 57344) | 极小 (~6) |
| **精度 (尾数位)** | 10 位 | 7 位 | 3 位 / 2 位 | 1 位 |
| **需要 Loss Scaling** | ✅ 必须 | ❌ 不需要 | ❌ (tile scaling 替代) | ❌ (QAT 内置) |
| **需要细粒度量化** | ❌ | ❌ | ✅ (1×128 / 128×128) | ✅ (per-group) |
| **显存节省 vs FP32** | ~2× | ~2× | ~4× | ~8× |
| **计算吞吐 vs BF16** | 1× | 1× | ~2× | ~4× (理论) |
| **精度敏感 op** | 保持 FP32 | 保持 FP32/BF16 | 保持 BF16/FP32 | 保持 BF16/FP32 |
| **成熟度** | 🟢 成熟 | 🟢 成熟 | 🟡 前沿 | 🔴 研究阶段 |
| **代表模型** | BERT, GPT-2 | LLaMA, Falcon, V3/V4 | DeepSeek-V3/V3.2 | DeepSeek-V4 |

> **关键 takeaways**:
> - **BF16 是当前最优 baseline**: 零心智负担 + 动态范围充足 + 硬件原生支持。
> - **FP8 是 V3 的工程突破**: 首次在 671B 规模验证可行，2× 加速，但需要定制 kernel + 细粒度量化 + 精度敏感 op 保护。
> - **FP4 是 V4 的研究前沿**: 不能直接训练，必须 QAT。训练时 FP4 只是存储/通信格式，计算仍走 FP8 GEMM。
> - **核心规律**: 每降一级精度，都需要额外的「保护措施」来补偿精度损失 — FP16 用 Loss Scaling，FP8 用 Tile Scaling + 累加提升，FP4 用 QAT。

---

### 2.1 混合精度训练的发展路线

```
2020 ─── FP16 + Loss Scaling (NVIDIA AMP)
  │      问题: Loss scale 敏感，动态调整不稳定
  │
2021 ─── BF16 Training (A100 原生支持)
  │      优势: 无 loss scaling，动态范围 = FP32
  │      现状: 当前主流，V3/V4 均大面积使用
  │
2022 ─── FP8 格式标准 (Micikevicius et al.)
  │      E4M3 for Fprop, E5M2 for Dgrad/Wgrad
  │
2024 ─── FP8 大规模训练验证 (DeepSeek-V3)
  │      首次在 671B MoE 上全链路 FP8 训练
  │      关键技术: 细粒度量化、提升累加精度、精度敏感 op 保留 BF16
  │
2025 ─── FP4 QAT (DeepSeek-V4)
         训练时 FP32 master → 量化到 FP4 → 反量化到 FP8 → GEMM
         推理时原生 FP4 GEMM → 2× 加速 vs FP8
```

### 2.2 BF16 混合精度训练 — 当前主流方案

#### 标准配置

| 组件 | 精度 |
|------|------|
| Master Weight | FP32 |
| Optimizer State (\(m_t, v_t\)) | FP32 |
| Forward Activation | BF16 |
| Backward Gradient | BF16 |
| 精度敏感 op (Embedding/Norm/Softmax/Loss) | FP32 / BF16 |
| All-Reduce (梯度通信) | FP32 |

#### 为什么 BF16 成为主流？

1. **零心智负担**: 不需要 loss scaling → 不需要调 scale factor → 训练更稳定。
2. **动态范围充足**: 与 FP32 相同 → 不会因为 overflow/underflow 卡住训练。
3. **转换廉价**: BF16 ↔ FP32 只是截断/补零 → 无复杂舍入逻辑。
4. **硬件支持广泛**: A100/H100/H800/B200 均原生支持 BF16 Tensor Core。

#### 局限

- 尾数仅 7 位 → **精度只有 FP32 的 ~1/4000** → 对于小数值的表示能力差。
- 大 batch + 长时间训练时，BF16 的累积舍入误差可能在极罕见情况下影响收敛（大多数情况下无影响）。

### 2.3 FP8 训练 — DeepSeek-V3 方案

#### FP8 使用策略

| 操作 | 输入精度 | 权重精度 | 输出精度 | 累加精度 |
|------|---------|---------|---------|---------|
| Fprop (前向 GEMM) | FP8 E4M3 | FP8 E4M3 | BF16 | FP32 (每 128 提升) |
| Dgrad (反向输入梯度) | FP8 E5M2 | FP8 E5M2 | BF16 | FP32 |
| Wgrad (反向权重梯度) | FP8 E5M2 | FP8 E5M2 | FP32 | FP32 |
| Attention | BF16 | BF16 | BF16 | — |
| Norm | BF16/FP32 | BF16/FP32 | BF16/FP32 | — |
| Embedding | BF16 | BF16 | BF16 | — |
| Loss | FP32 | FP32 | FP32 | — |

#### 细粒度量化的具体方案

```
激活量化 (1×128 tile):
┌─────────────────────────────────────┐
│ Token 1 │ ch1..128 │ ch129..256 │...│  → 每个 1×128 tile 独立 scale
│ Token 2 │ ...      │ ...        │...│
│ ...                                  │
└─────────────────────────────────────┘

权重量化 (128×128 tile):
┌─────────────────────────────────────┐
│ Out ch 1..128   │ In 1..128   │...│  → 每个 128×128 tile 独立 scale
│ Out ch 129..256 │ ...         │...│
│ ...                                  │
└─────────────────────────────────────┘
```

**为什么激活用 1×128（而非 128×128）？**
- 激活的量化发生在前向/反向传播的「边缘」— 即 GEMM 输入/输出处。
- 前向: Activation (BF16) → quantize to FP8 → GEMM → dequantize to BF16。
- 反向: Gradient (BF16) → quantize to FP8 → GEMM → dequantize to BF16。
- 每个 token 的激活值可能分布差异大 → 用 1×128 tile 可以更细粒度地按 token 分组 → 减少 outlier token 对其他 token 的影响。

### 2.4 FP4 QAT — DeepSeek-V4 方案（大纲）

#### 训练流程

```
FP32 Master Weight
     ↓ Quantize (per-group, round-to-nearest)
FP4 Weight
     ↓ Dequantize to FP8
FP8 Activation × FP8 Weight → FP8 GEMM (Tensor Core)
     ↓ STE (Straight-Through Estimator) 回传梯度
FP32 Weight Gradient → Optimizer Update
```

#### 核心挑战
- FP4 只有 16 个可表示值 → 量化误差远大于 FP8。
- QAT 中量化/反量化的 stride loss 需要在训练时被优化。
- 需要特殊的 GPU kernel 支持 FP4 量化/反量化（当前开源生态不成熟）。

#### V4 的具体方案
- 详见 → 后续论文分析章节（3.3）。

---

## 三、论文与代码分析 (Paper & Code Analysis)

### 3.1 FP8 Formats for Deep Learning (Micikevicius et al., 2022)

- **论文**: [arXiv 2209.05433](https://arxiv.org/abs/2209.05433) | 📄 [本地 PDF](papers/FP8%20Formats.pdf)
- **核心创新**: 提出 FP8 的两种格式（E4M3 for Fprop, E5M2 for Dgrad/Wgrad），定义混合精度训练中 FP8 的使用规范。
- **与本主题的关联**: FP8 训练的「格式标准」— DeepSeek-V3 的 FP8 方案几乎完全遵循此论文的格式定义。

**分析大纲**:
- [ ] **格式设计的理论依据**: 为什么选 E4M3 和 E5M2？设计空间中的 trade-off 分析。
- [ ] **FP8 混合精度训练的通用框架**: 哪些 op 用 FP8，哪些用更高精度？
- [ ] **与 BF16 的对比实验**: FP8 vs BF16 在不同模型/任务上的精度损失。
- [ ] **硬件层面的考虑**: Tensor Core 的 FP8 支持特性（吞吐、精度、限制）。
- [ ] **对 DeepSeek-V3 的影响**: V3 在哪些方面遵循/扩展了此文的标准？

### 3.2 DeepSeek-V3: FP8 Mixed Precision Training

- **论文**: [arXiv 2412.19437](https://arxiv.org/abs/2412.19437) | 📄 [本地 PDF](papers/DeepSeek-V3%20FP8%20Infra.pdf) | 📝 [中文笔记](papers-zh/DeepSeek-V3.md)
- **核心创新**: **首次在 671B 超大规模 MoE 模型上验证了全链路 FP8 训练的可行性**，并系统性地解决了 FP8 训练的精度和效率问题。
- **与本主题的关联**: 混合精度训练在工业界的「天花板」— 定义了什么是「可行的」FP8 训练方案。

**分析大纲**:
- [ ] **Section 3.3 完整精读**: FP8 格式选择、细粒度量化方案、精度提升策略。
- [ ] **FP8 GEMM 的精度-效率分析**: 累加精度提升窗口大小（128）的 ablations。
- [ ] **精度敏感 op 的 FP8 处理**: 哪些 op 保留了 BF16/FP32？为什么？
- [ ] **FP8 通信**: All-to-All dispatch/combine 是否也用了 FP8？
- [ ] **训练稳定性分析**: FP8 训练下的 loss curve、梯度范数、溢出率等。
- [ ] **代码走读**: `code/deepseek-ref/` 中 FP8 相关的实现（量化 kernel、GEMM wrapper 等）。

### 3.3 DeepSeek-V4: FP4 Quantization-Aware Training

- **论文**: [arXiv (V4 Technical Report)] | 📄 [本地 PDF](papers/DeepSeek-V4%20Technical%20Report.pdf) | 📝 [中文笔记](papers-zh/DeepSeek-V4-technical-report.zh.md)
- **核心创新**: 引入 FP4 QAT 训练流程（FP32 master → FP4 quant → FP8 dequant → GEMM），推理时用原生 FP4 GEMM 获得 2× 加速。
- **与本主题的关联**: 混合精度训练的「前沿」— 将 low-bit 训练推到 4-bit，进一步压缩训练/推理的精度边界。

**分析大纲**:
- [ ] **FP4 格式定义**: E2M1 的具体编码方案和可表示值集合。
- [ ] **QAT 训练流程**: 量化→反量化→FP8 GEMM→STE 回传的完整计算图。
- [ ] **与 FP8 的对比**: FP4 QAT 相比 FP8 训练的精度损失和效率收益。
- [ ] **MegaMoE 中的 FP4**: FP4 在 MoE 层的具体使用方式（专家权重压缩比）。
- [ ] **推理阶段的 FP4**: V4 推理中 FP4 GEMM 的加速比和精度 trade-off。

### 3.4 LongCat / Kimi / 其他前沿方案（TBD）

- **LongCat**: (待补充论文和代码)
  - [ ] 论文核心观点
  - [ ] 混合精度训练方案
  - [ ] 与 DeepSeek 系列的对比

- **Kimi (Moonshot)**: (待补充论文和代码)
  - [ ] 论文核心观点
  - [ ] 混合精度训练方案
  - [ ] 与 DeepSeek 系列的对比

---

## 四、关键数字与速查 (Quick Reference)

| 数字 / 事实 | 含义 | 来源 |
|------------|------|------|
| 1-8-23 / 1-5-10 / 1-8-7 / 1-4-3 / 1-5-2 / 1-2-1 | 各格式的 (sign, exponent, mantissa) 位分配 | IEEE 754 / 各论文 |
| \(2^{-23} \approx 1.19\times 10^{-7}\) | FP32 机器精度 | IEEE 754 |
| \(2^{-7} \approx 7.81\times 10^{-3}\) | BF16 机器精度 | — |
| \(6.1\times 10^{-5}\) | FP16 正常数最小正值（underflow 阈值） | IEEE 754 |
| 65504 | FP16 正常数最大值（overflow 阈值） | IEEE 754 |
| 448 | FP8 E4M3 正常数最大值 | Micikevicius 2022 |
| 14 位 | H100/H800 FP8 Tensor Core 累加精度 | NVIDIA |
| 128 | V3 FP8 GEMM 累加精度提升窗口大小 | DeepSeek-V3 |
| 1×128 / 128×128 | 激活 / 权重的细粒度量化 tile size | DeepSeek-V3 |
| ~1.56% | 1×128 tile 量化的 scale factor 显存开销 | 计算值 |
| 2× | FP8 vs BF16 的理论计算吞吐提升 | NVIDIA H100 spec |
| 4× | FP4 vs BF16 的理论计算吞吐提升 | NVIDIA H100 spec (B200) |
| 3 层 | V4 使用 Hash MoE 的前几层（这些层对精度更敏感） | DeepSeek-V4 |
| \(\lceil K/128 \rceil\) | V3 累加精度提升操作的总执行次数 | DeepSeek-V3 |

---

## 五、常见问题 (FAQ)

### Q1: BF16 和 FP16 到底选哪个？
**答**: 在 A100 及以上硬件上，**优先选 BF16**。原因：(1) 不需要 loss scaling，训练更稳定；(2) 动态范围 = FP32，梯度不会 underflow；(3) 转换廉价。FP16 的唯一优势是精度更高（10 位 vs 7 位尾数），但这个优势在大模型训练中远不及动态范围重要。

### Q2: FP8 训练中为什么 Fprop 用 E4M3 而 Dgrad/Wgrad 用 E5M2？
**答**: **Fprop** 处理的是激活值（已通过 LayerNorm 归一化，分布稳定），范围可控 → 偏重精度（E4M3，3 位尾数）。**Dgrad/Wgrad** 处理梯度，梯度天然具有大动态范围（某些层梯度极小、某些极大）→ 偏重动态范围（E5M2，5 位指数）。这是 Micikevicius et al. (2022) 的核心设计决策。

### Q3: 细粒度量化和 Loss Scaling 有什么本质区别？
**答**: **Loss Scaling** 是全局的、单一的缩放因子，通过放大梯度来防止 FP16 underflow，但会同等放大 noise。**细粒度量化** 是局部的、per-tile 的缩放，只解决量化表示的范围匹配问题，不改变梯度的相对大小。两者解决不同问题：loss scaling 解决「梯度太小 FP16 表示不了」，细粒度量化解决「outlier 让大多数值的量化精度被浪费」。

### Q4: 为什么 embedding 对精度如此敏感？
**答**: Embedding 表是一个 `vocab_size × hidden_dim` 的大矩阵。每个 token 只取其中一行。如果 embedding 矩阵有量化误差，误差直接进入 token 的初始表示 → 贯穿整个 Transformer → 被残差连接保留到最后 → 直接影响 loss。相当于「源头污染」。此外，embedding 行的数值往往非常接近（语义相近的词 embedding 也相近），量化舍入可能破坏这种语义相近关系。

### Q5: FP4 QAT 和直接 FP4 训练有什么区别？
**答**: **直接 FP4 训练**: 所有计算（前向+反向+更新）都在 FP4 → 几乎不可能，FP4 的表示能力（16 个值）不足以直接承载梯度。**FP4 QAT**: 只在 GEMM 时用 FP4（权重和激活量化为 FP4 → 反量化到 FP8 → 用 FP8 做 GEMM），但 master weight、梯度、optimizer 仍然是 FP32。本质是「训练时的 FP4 模拟 + 推理时的 FP4 原生化」。

### Q6: 累加精度不足会导致什么问题？如何检测？
**答**: 当 GEMM 内维维度 \(K\) 很大时，14 位累加精度可能导致部分和的舍入误差累积，最终误差大于 FP32 的理论误差。**检测方法**: (1) 对比 FP8 GEMM 和 FP32 GEMM 的输出，计算 relative error；(2) 在训练中监控特定层的梯度范数是否异常；(3) Ablation 累加精度窗口大小，观察收敛行为。**V3 的解决**: 每 128 个元素提升到 FP32 累加。

---

## 六、延伸阅读 (Further Reading)

- 📖 [FP8 Formats for Deep Learning](papers/FP8%20Formats.pdf) — FP8 格式标准
- 📖 [DeepSeek-V3 FP8 Infra](papers/DeepSeek-V3%20FP8%20Infra.pdf) — FP8 大规模训练验证
- 📖 [DeepSeek-V3 中文笔记](papers-zh/DeepSeek-V3.md) — 论文精读
- 📖 [DeepSeek-V4 Technical Report](papers/DeepSeek-V4%20Technical%20Report.pdf) — FP4 QAT 方案
- 📖 [DeepSeek-V4 中文笔记](papers-zh/DeepSeek-V4-technical-report.zh.md)
- 📖 [Megatron-LM 性能优化指南](topics/megatron-performance-optimization.md) — FP8 训练在 Megatron-LM 中的完整配置（FP8 recipe、MoE FP8 dispatch、FP8 Param Gather）
- 📖 [Expert Parallelism 深度笔记](topics/expert-parallelism.md) — EP 通信与精度/效率的关联
- 📖 [INDEX.md — FP8/FP4 低精度训练](INDEX.md#fp8--fp4-低精度训练)
