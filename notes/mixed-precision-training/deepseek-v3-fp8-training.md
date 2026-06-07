# DeepSeek-V3 FP8 混合精度训练

> **一句话概述**: DeepSeek-V3 首次在 671B 超大规模 MoE 模型上验证了全链路 FP8 训练的可行性，核心方案是「细粒度 tile 量化（1×128 激活 / 128×128 权重）+ 提升累加精度（每 128 元素提升至 FP32）+ 精度敏感 op 保留 BF16/FP32」三位一体。
> **参考论文**: DeepSeek-V3 Technical Report §3.3 | [arXiv 2412.19437](https://arxiv.org/abs/2412.19437)
> **参考代码**: [code/deepseek-ref/V3/inference/](../../code/deepseek-ref/V3/inference/)

---

## 一、总览：FP8 用在哪里？（总分）

### 1.1 一句话总结

在 DeepSeek-V3 中，**FP8 只用于 FFN / MoE 层的 GEMM**（Fprop 前向、Dgrad 输入梯度反向、Wgrad 权重梯度反向）。以下组件虽然也包含 GEMM，但刻意保持在 BF16/FP32：

| 组件 | 包含的 GEMM | 精度 | 为什么不量化到 FP8 |
|------|-----------|------|--------------------|
| **FFN/MoE 层** | w1, w2, w3 (gate/up/down) | **FP8** ✅ | 计算量大且对精度鲁棒 → FP8 收益最大 |
| **MLA Attention** | W^Q, W^K, W^V, W^O, Q·K^T, Attn·V | BF16 ❌ | Softmax 对误差极度敏感（指数放大效应）；Attention 是 memory-bound → FP8 加速收益有限 |
| **MoE Gate** | W_gate | BF16 ❌ | 路由决策敏感 — 微小误差可能改变 token → expert 分配 |
| **Embedding** | 无（gather 操作） | BF16 ❌ | 查表操作不是 GEMM；源头误差会贯穿整个网络 |
| **RMSNorm** | 无（element-wise 统计） | BF16 ❌ | 平方、均值、开方对精度敏感 |
| **SiLU** | 无（element-wise 非线性） | BF16 ❌ | 指数运算 `σ(x)` 对精度敏感 |
| **Output Head** | W_head | BF16 ❌ | 最终输出，直接影响 loss |
| **Cross-Entropy Loss** | 无（log-softmax + reduction） | FP32 ❌ | log 运算在概率接近 0 时的梯度极大 |

**一句话**：FP8 不是「所有 GEMM 都用」，而是「只用在大计算量、对精度鲁棒的 FFN GEMM 上」。Attention 虽然也有 GEMM，但它天然是 memory-bound 且被 Softmax 绑定，FP8 加速收益小、风险大 → 不如保持 BF16 稳妥。

### 1.2 一张图说清楚（前向流程中）

```
Token → [Embedding: BF16]          ← 查表操作，不是 GEMM
           ↓
    ┌─────────────────────────────────────────────────┐
    │  Transformer Block × 61                         │
    │                                                 │
    │  1. RMSNorm (BF16)              ← element-wise │
    │  2. MLA Attention (全程 BF16)   ← QK^T, AV 也是 GEMM，但保持 BF16 │
    │  3. RMSNorm (BF16)              ← element-wise │
    │  4. MoE / FFN:                                  │
    │     ├─ Gate (BF16)              ← W_gate GEMM，但保持 BF16 │
    │     ├─ Expert w1: [BF16→FP8] → ★ FP8 GEMM → [FP8→BF16]  │
    │     ├─ Expert w3: [BF16→FP8] → ★ FP8 GEMM → [FP8→BF16]  │
    │     ├─ SiLU activation (BF16)   ← element-wise │
    │     ├─ gate * up (BF16)         ← element-wise │
    │     ├─ Expert w2: [BF16→FP8] → ★ FP8 GEMM → [FP8→BF16]  │
    │     └─ Shared Expert: 同上（★ 三个 FP8 GEMM）   │
    └─────────────────────────────────────────────────┘
           ↓
    RMSNorm (BF16) → Output Head (BF16) → Loss (FP32)
```

**★ = FP8 介入的位置**。可以看到：
- Attention 虽然内部有 6 个 GEMM（W^Q/W^K/W^V + W^O 投影 + QK^T + AV），但全部保持 BF16
- Gate 虽然是一次矩阵乘法（`scores = x @ W_gate^T`），但保持 BF16
- **只有 FFN/MoE 的 w1, w2, w3 三个 GEMM 用 FP8** — 因为这三者占据了训练中绝大多数的矩阵乘法计算量

### 1.3 先理解本质：为什么需要「按 tile 量化」？（逐级展开）

这是理解 V3 FP8 的最关键概念。我们用一个具体的小例子逐级展开。

---

**级别 0：不加量化的 BF16 GEMM（基线）**

```
输入 x (BF16)             权重 W (BF16)              输出 y (BF16)
┌──────────┐       ┌──────────────┐         ┌──────────┐
│ 0.5  -2.0│   ×   │ 1.0   0.5  T│    =    │  ·   ·   │
│-0.1   3.0│       │-0.5   2.0   │         │  ·   ·   │
└──────────┘       └──────────────┘         └──────────┘
  M × K                 N × K                  M × N
```

没有任何量化，所有值都是 BF16。这就是普通的 `F.linear(x, weight)`。

---

**级别 1：Per-Tensor 量化（一整张 tensor 共用一个 scale）— 有缺陷的方案**

```
想把 BF16 → FP8，需要把数值缩放到 FP8 E4M3 的表示范围 [−448, 448]。

Per-tensor 做法：
  对整个 x 矩阵算一个 scale: s = max(|x|) / 448
  然后用 x / s 把所有值映射到 FP8

例子：x = [[0.5, -2.0],
          [-0.1, 100.0]]    ← 注意 100.0 是 outlier

  s = max(0.5, 2.0, 0.1, 100.0) / 448 = 100.0 / 448 ≈ 0.223

  x / s = [[2.24,  -8.97],
           [-0.45, 448.0 ]]

  → 转为 FP8 E4M3：[[2.25, -9.0], [-0.448, 448.0]]

问题：
  -0.1 / 0.223 = -0.448 → FP8 表示 -0.448（精度损失不大）
  但 0.5 / 0.223 = 2.24 → FP8 只能表示 2.25（3 位尾数精度有限）

  核心矛盾：outlier 100.0 主导了 scale，导致大部分正常值（0.5, -2.0, -0.1）
  被「压缩」到 FP8 的极小范围里，白白浪费了精度。
```

**图解**：

```
Per-tensor: 整个矩阵 1 个 scale
┌─────────────────────────────────┐
│ 0.5  -2.0  │            100.0  │   ← outlier 拉大了 scale
│-0.1   3.0  │                   │
└─────────────────────────────────┘
       ↑ 所有值共用一个 scale，outlier 让所有人「陪绑」
```

---

**级别 2：Per-Tile 量化（细粒度分组独立 scale）— V3 的方案**

```
把 x 按 1×128 的 tile 切分，每个 tile 独立算 scale：

  Tile 1: [0.5, -2.0, ..., -0.1]  → amax=3.0  → s₁=3.0/448≈0.0067
  Tile 2: [100.0, 0.3, ..., 1.2]  → amax=100  → s₂=100/448≈0.223

  Tile 1 量化：值 / 0.0067 → 全部在 [-448, 448] 内，充分利用 FP8 精度
  Tile 2 量化：值 / 0.223  → 同样充分利用 FP8 精度

  ★ outlier 100.0 只污染 Tile 2，Tile 1 完全不受影响！
```

**图解**：

```
Per-tile (V3): 每个 1×128 tile 一个独立 scale
┌──────────────┬──────────────┬─────┐
│ tile 1       │ tile 2       │ ... │
│ s₁=0.0067    │ s₂=0.223     │     │  ← 每个 tile 独立 scale
└──────────────┴──────────────┴─────┘
  正常值精度完好    outlier 被隔离

术语统一（这三个词在本文中指向同一概念）：
  tile  = block = group = 量化的基本分组单元
  scale = 缩放因子 = 该 tile 内数值 × scale = 原始值
  per-tile scaling = per-group quantization = 细粒度量化
```

---

**级别 3：代码中如何体现？**

现在回头看 `act_quant` 函数，每一步都能对上：

```python
# kernel.py: act_quant（Triton kernel 版本）
@triton.jit
def act_quant_kernel(x_ptr, y_ptr, s_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)                 # 当前处理第几个 tile
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs).to(tl.float32)     # 加载一个 tile（128 个元素）

    amax = tl.max(tl.abs(x))                     # ← 一次 reduction：找该 tile 的 |max|
    amax = tl.maximum(amax, 1e-4)                # ← 防止除以零
    s = amax / 448.                              # ← 确保 |x/s| ≤ 448（FP8 E4M3 上限）

    y = x / s                                     # ← 量化：除以 scale
    y = y.to(y_ptr.dtype.element_ty)              # ← 转为 float8_e4m3fn
    tl.store(y_ptr + offs, y)
    tl.store(s_ptr + pid, s)                      # ← 存该 tile 的 scale（1 个 FP32 数）

# 调用：
#   x:        (M, 7168)  BF16
#   x_fp8:    (M, 7168)  FP8   ← 量化后的值
#   scale:    (M, 56)    FP32  ← 7168/128=56 个 scale（每行 56 个 tile）
```

**输入/输出形状变换（这是理解的关键）**：

```
假设 batch=4 tokens, hidden_dim=7168（V3 的实际维度）

x.shape  = (4, 7168)        BF16 输入
                ↓ act_quant(x, block_size=128)
x_fp8.shape = (4, 7168)      FP8  量化数据（每个元素 1 byte）
scale.shape = (4, 56)        FP32 scale（7168 ÷ 128 = 56 个 tile/token）
                                ↑
                    每个 token 每 128 channel = 1 个 scale
                    所以每个 token 有 56 个 scale
```

**对应到 GEMM 中的使用**：

```python
# GEMM: y = x @ W^T，内维 K=7168
# 内维被切成 k = 7168/128 = 56 个 segment

# 激活 x: (4, 7168) in FP8, scale (4, 56)
# 权重 W: (2048, 7168) in FP8, scale (2048/128, 7168/128) = (16, 56)

for i in range(56):  # 遍历每个 128-element segment
    # 取第 i 个 tile:
    x_tile = x[:, i*128:(i+1)*128]        # (4, 128) FP8
    w_tile = W[:, i*128:(i+1)*128]        # (2048, 128) FP8

    # FP8 Tensor Core 矩阵乘法:
    partial = x_tile @ w_tile.T            # (4, 2048), 14 位累加

    # 乘入 scale（★ 在 FP32 中完成）
    x_scale_i = scale[:, i]               # (4,)   — 每个 token 该 tile 的 scale
    w_scale_i = W_scale[:, i]             # (16,)  — 每 128 output ch 该 tile 的 scale
    acc += partial * x_scale_i[:, None] * w_scale_i[None, :]

y = acc  # (4, 2048) in FP32 → 转为 BF16 输出
```

**为什么激活用 1×128 而权重用 128×128？**

```
激活 (1×128 tile):
  每行 = 1 token × 128 channels → token 之间不共享 scale
  → 某个「异常 token」不会污染其他 token 的量化

权重 (128×128 block):
  每 128 output channels × 128 input channels → 二维分组
  → 权重是静态的，128×128 足够细粒度，且存储开销可控
  → 如果用 1×128 存权重，scale 数量太大（Out × In/128 vs Out/128 × In/128）
```

---

### 1.4 小结：术语速查

| 术语 | 含义 | 激活上的形状 | 权重上的形状 |
|------|------|------------|------------|
| **tile / block / group** | 量化的基本分组单元 | 1×128（1 token × 128 ch） | 128×128（128 out × 128 in） |
| **scale factor** | 该 tile 的缩放因子 `s = amax/448` | 每个 tile 1 个 FP32 值 | 每个 block 1 个 FP32 值 |
| **per-tile scaling / per-group quantization** | 细粒度量化（相对于 per-tensor） | 激活 shape `(M, K/128)` | 权重 shape `(Out/128, In/128)` |
| **act_quant** | 激活在线量化：`BF16 → FP8 + scale` | 输入 `(M, K)`→输出 `(M,K) fp8` + `(M,K/128) fp32` | — |
| **weight_dequant** | 权重反量化：`FP8 × scale → BF16` | — | 输入 `(Out,In) fp8` + `(Out/128,In/128) scale` → `(Out,In) bf16` |

---

## 二、权重存储格式：FP8 + Per-128×128-Block Scale

理解了 tile 量化，权重的存储就很自然了。训练/推理时权重不是临时量化的，而是**预先存成 FP8 + scale 的格式**。

### 2.1 存储布局

以 w1: `Linear(7168 → 2048)` 为例，权重矩阵 shape 为 `(2048, 7168)`（PyTorch 的 `(out_features, in_features)` 布局）：

```
W 数据（FP8 E4M3）:        W_scale（FP32）:
┌─────────────────────┐    ┌──────────────┐
│ 2048 rows            │    │ 16 rows       │  2048/128 = 16
│ 7168 columns         │    │ 56 columns    │  7168/128 = 56
│                     │    │              │
│ ██ ██ ██ ... ██     │    │ s₀₀ s₀₁ ... │  每个 s_ij 对应
│ ...                  │    │ ...          │  128×128 的 FP8 数据块
│ ██ ██ ██ ... ██     │    │ s₁₅,₀ ...   │
└─────────────────────┘    └──────────────┘
 每个元素 1 byte            每个 scale 4 bytes
 = 2048×7168 ≈ 14 MB       = 16×56×4 ≈ 3.5 KB
                                 ↑ 额外开销仅 ~0.025%
```

**Scale 数量计算**：

```
W.shape = (2048, 7168)
W_scale.shape = (ceil(2048/128), ceil(7168/128)) = (16, 56)

显存开销比 = (16×56×4) / (2048×7168×1) = 3584 / 14680064 ≈ 0.024%
```

### 2.2 与标准 FP8 格式的区别

V3 统一使用 **E4M3**（4 位指数 + 3 位尾数），与 Micikevicius et al. (2022) 推荐的分化格式不同：

| | 标准做法（Micikevicius 2022）| DeepSeek-V3 |
|---|---|---|
| Fprop 输入格式 | E4M3（精度优先）| **E4M3** |
| Dgrad 输入格式 | **E5M2**（范围优先）| **E4M3**（统一）|
| Wgrad 输入格式 | **E5M2**（范围优先）| **E4M3**（统一）|

**为什么 V3 敢统一用 E4M3？**

E5M2 的存在意义是「指数多 1 位 → 动态范围大 2× → 能覆盖梯度的大动态范围」。但细粒度 tile 量化让每个小 group 独立缩放 → 在 group 内部，数值范围天然被控制 → 不再需要 E5M2 的大范围。等价地说：

```
Per-tensor 量化:
  整层梯度 1 个 scale → 动态范围必须覆盖 [极小梯度, 极大梯度]
  → E5M2 的 5 位指数是刚需

Per-tile 量化:
  每个 tile 独立 scale → tile 内数值范围可控
  → E4M3 的 4 位指数 + 3 位尾数就够用
  → 反而多出来的 1 位尾数（E4M3: 3 位 vs E5M2: 2 位）提升了精度
```

这就是论文 §3.3.2 说的 "Mantissa over Exponent"。

> **复习**: 浮点格式位分配和动态范围 → [混合精度训练基础知识](mixed-precision-training.md#11-浮点数表示格式-floating-point-formats)。

---

## 三、Forward 流程详解（逐 op 走读）

现在以**一个 token 的完整前向传播**为例，逐步追踪 FP8 在哪些 op 介入、如何介入。

### 3.1 Embedding: BF16（不量化）

```
Token ID → Embedding Lookup → BF16 hidden vector (dim=7168)
```

- **精度**: BF16（或 FP32，取决于配置）
- **为什么不量化到 FP8**: Embedding 是「源头」— 量化误差直接进入 token 初始表示，被残差连接保留到最后，等效于「源头污染」。详见 [精度敏感算子分析](mixed-precision-training.md#15-精度敏感算子-precision-sensitive-operations)。

对应代码 [model.py:89-128]：
```python
y = F.embedding(x, self.weight)  # weight 始终是 BF16/FP32
```

### 3.2 RMSNorm: BF16（不量化）

```
hidden (BF16) → RMSNorm(x) = x / sqrt(mean(x²) + ε) * weight → hidden (BF16)
```

- **精度**: BF16（或 FP32）
- **为什么不量化**: `1/sqrt(Var)` 在小方差时对精度极度敏感。RMSNorm 的输出将被送入 Attention 和 FFN，在 FP8 GEMM 的入口处再量化。

对应代码 [model.py:270-294]：
```python
return F.rms_norm(x, (self.dim,), self.weight, self.eps)  # 全程 BF16
```

### 3.3 MLA Attention: 全程 BF16（不量化到 FP8）

```
hidden (BF16)
  ├─→ W^DQ(bf16) → compress_Q → W^UQ(bf16) → Q
  ├─→ W^DKV(bf16) → compress_KV → W^UK(bf16) → K
  │                               → W^UV(bf16) → V
  ├─→ Q·K^T → softmax → ·V   ← 全程 BF16/FP32
  └─→ W^O(bf16) → output (BF16)
```

- **精度**: Attention 内所有线性投影 + Softmax 保持 BF16 或 FP32
- **为什么不量化**: 
  - Softmax 的指数运算对精度极度敏感（见 [§1.5](mixed-precision-training.md#15-精度敏感算子-precision-sensitive-operations)）
  - MLA 的压缩/解压缩涉及低秩投影，量化误差可能在压缩和上投影中双重放大
  - Attention 在整体计算中的占比不高 → 保留 BF16 的边际成本低

对应代码 [model.py:396-497]，所有 Linear 的输入/输出都是 BF16。

### 3.4 MoE Gate: BF16（不量化）

```
hidden (BF16) → Linear(W_gate) → sigmoid → TopK → weights, indices
```

- **精度**: BF16（Gate 权重不可量化到 FP8）
- **为什么不量化**: Gate 的 s_i,t = sigmoid(u_t^T · e_i) 决定 token → expert 路由。量化误差改变亲和度 → 改变专家分配 → 训练动态被扰动。

对应代码 [model.py:566-598]：
```python
scores = linear(x, self.weight)   # weight 是 BF16
# sigmoid / softmax → topk → weights, indices
```

### 3.5 ★ MoE Expert FFN: FP8 GEMM 的核心舞台

这才进入 DeepSeek-V3 FP8 训练的**核心环节**。Expert 内部是标准的 SwiGLU FFN：

```python
# model.py:622-633 — Expert.forward(x)
def forward(self, x):  # x: BF16, shape (num_tokens, dim)
    return self.w2(F.silu(self.w1(x)) * self.w3(x))
```

```
x (BF16, num_tokens × 7168)
│
├─→ w1: Linear(7168 → 2048)   ★ FP8 GEMM
│     └→ gate = SiLU(x @ W1^T)
│
├─→ w3: Linear(7168 → 2048)   ★ FP8 GEMM
│     └→ up   = x @ W3^T
│
├─→ gate * up  (element-wise, BF16)   ← 激活乘法保持 BF16
│
└─→ w2: Linear(2048 → 7168)   ★ FP8 GEMM
      └→ output = (gate * up) @ W2^T
```

**标 ★ 的三个 GEMM（w1, w2, w3）是 FP8 训练的全部舞台**。下面展开单个 GEMM 的完整量化和计算流程。

### 3.6 ★ FP8 GEMM 全流程展开（Fprop 核心）

以 `w1: Linear(7168 → 2048)` 为例，设输入 `x` 为 `(M, 7168)` 的 BF16 张量（M = batch tokens 数）。这个 GEMM 计算 `y = x @ W^T`。

```
═══════════════════════════════════════════════════════════════
阶段 0: 权重早已存为 FP8 + scale（离线转换或训练时维护）
═══════════════════════════════════════════════════════════════

W1 权重矩阵 shape (2048, 7168):
  - FP8 数据: float8_e4m3fn, ~14 MB
  - Scale:   FP32, shape (16, 56)    ← (2048/128) × (7168/128) 个 per-block scale

═══════════════════════════════════════════════════════════════
阶段 1: 激活量化 — act_quant(x, block_size=128)
         x: BF16 (M, 7168) → x_fp8: FP8 (M, 7168), scale: FP32 (M, 56)
═══════════════════════════════════════════════════════════════

沿最后一维（In=7168），每 128 个 channel 一组，独立计算 amax → scale → 量化。
（详细过程见 §1.3 级别 3 的代码展开。）

  结果:
    x_fp8:  (M, 7168)  in float8_e4m3fn
    scale:  (M, 56)    in FP32          ← 7168 / 128 = 56

为什么是 1×128（而非 128×128）？
  - 激活按 token 独立量化 → 不同 token 的 outlier 不会互相污染
  - 激活和权重都用 128×128 会导致训练不稳定（论文附录 B.2 有详述）

═══════════════════════════════════════════════════════════════
阶段 2: FP8 GEMM 累加 — 这是 DeepSeek-V3 最核心的创新
═══════════════════════════════════════════════════════════════

内维 K = 7168，按 128 切分为 56 个 segment。每次处理一个 segment（128 个内维元素），
循环 56 次完成整个 GEMM。

for segment in range(56):

    —— 2a: 加载 segment ——
      x_tile = x_fp8[:, segment*128 : (segment+1)*128]   # (M, 128)
      W_tile = W_fp8[:, segment*128 : (segment+1)*128]   # (N, 128)

    —— 2b: FP8 Tensor Core MMA ——
      partial = tl.dot(x_tile, W_tile.T)    # (M, N)，对这 128 个内维元素做矩阵乘累加

      tl.dot 是什么？
        tl = triton.language，是 Triton kernel 的 DSL（类似 CUDA 的 C++）。
        tl.dot(a, b) = 调用 GPU Tensor Core 的 MMA 指令，计算 a × b^T。

      关键问题：Tensor Core 的「累加精度」只有 ~14 位。

      什么是「累加精度」？
        矩阵乘法就是一堆「乘然后加」。每个输出元素 y[0,0] =
          x[0]·W[0] + x[1]·W[1] + ... + x[127]·W[127]
        这个不断累加的 running sum 存在「累加器」寄存器里。
        该寄存器能保留多少位有效数字 = 累加精度。
        FP8 Tensor Core 的累加器只有 ~14 位（FP32 是 24 位），是 NVIDIA
        为了高吞吐做的硬件取舍（更宽的累加器需要更多芯片面积和功耗）。

      后果: 如果不加干预，K=4096 时最大相对误差可达 ~2%（论文实测）。

    —— 2c: ★ 乘入 scale + 提升到 FP32 累加（核心创新）——
      acc_fp32 += partial * x_scale[:, segment, None] * W_scale_expanded

      这里有一个广播细节：激活 scale (M,) 和权重 scale (16,) 的 shape 跟
      partial (M, 2048) 都不一样，怎么乘？

        激活 scale: shape (M,)
          → x_scale[:, segment, None] 变成 (M, 1)，广播到 (M, 2048)
          → 含义：每个 token 的所有 2048 个输出共享同一个激活 scale（因为一个
             tile 内 128 个 channel 使用同一个缩放因子是对的——量化时就是这样分的组）

        权重 scale: shape (16,)   ← W_scale[:, segment]，每个值管 128 个输出 ch
          → 需要 expand: 每个值重复 128 次 → (2048,) → [None, :] 为 (1, 2048)
          → 含义：每 128 个连续输出 channel 共享同一个权重 scale（因为权重是按
            128×128 block 存的，每个 scale 管 128 行 × 128 列）

      在真实的 Triton kernel 中（kernel.py:154）:
        b_s_ptrs = b_s_ptr + (offs_n // BLOCK_SIZE_K) * k
        其中 BLOCK_SIZE_K = 128，所以 offs_n // 128 决定了用哪个 scale，
        同一 128-channel group 内的所有列自动指向同一个 scale 值。
        → b_s 直接就是 (BLOCK_N,) shape，无需显式 repeat_interleave。

      这一步在 CUDA Core 的 FP32 寄存器中完成，而非 Tensor Core。
      （Tensor Core 和 CUDA Core 共享 SM 上的寄存器文件 — tl.dot() 把 partial
      写进寄存器，紧接的乘加指令从同一个寄存器读出，零拷贝接力。）

    —— 2d: 双 warpgroup 并发隐藏开销 ——
      H800 每个 SM 可同时运行 2 个 warpgroup（wg = 4 warps = 128 线程）:
        WG 0: [MMA seg0][提升到 FP32][MMA seg2][提升][...]
        WG 1: [MMA seg1][提升到 FP32][MMA seg3][...]
      当一个 wg 在做「提升」时，另一个在 Tensor Core 上跑 MMA → 零气泡。

结果: acc_fp32 (M, 2048) in FP32

═══════════════════════════════════════════════════════════════
阶段 3: 输出转换 — FP32 → BF16
═══════════════════════════════════════════════════════════════

output = acc_fp32.to(torch.bfloat16)   # (M, 2048) in BF16

  此后被 SiLU 消费（BF16），或参与 gate * up 的 element-wise 乘法，
  然后进入下一个 FP8 GEMM。
```

对应到真实 Triton 代码（[kernel.py:120-172]）的关键行：

```python
for i in range(k):                     # k = 56
    a = tl.load(a_ptrs)                # 2a: 加载 FP8 激活
    b = tl.load(b_ptrs)                # 2a: 加载 FP8 权重
    a_s = tl.load(a_s_ptrs)            # 拿当前 segment 的 scale
    b_s = tl.load(b_s_ptrs)
    accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
    #             ^^^^^^^^^ 2b         ~~~~~~~~~~~ 2c ~~~~~~~~~~
    #  tl.dot = Tensor Core MMA（14 位累加）
    #  * scale + += accumulator = 在 FP32 寄存器中完成（2c）
```

### 3.7 为什么是每 128 个元素？

这是 V3 论文 §3.3.2 通过实验确定的：

```
H800 FP8 Tensor Core 累加精度: ~14 位

K = 7168:
  不提升（全在 Tensor Core 14 位累加):
    → 当 K=4096 时，最大相对误差 ≈ 2%（论文实测）
    → 大模型训练中 2% 的梯度偏差足以影响收敛

  每个元素都提升:
    → 精度最高，但 Tensor Core ↔ CUDA Core 数据搬运开销太大

  N_C = 128（每 4 个 WGMMA 提升一次):
    → 论文测试的最小有效间隔
    → 绝对误差 ≈ O(√128 × 2^(-14)) ≈ 0.07%，在训练噪声范围内
    → 开销被双 warpgroup 并发完全隐藏
```

**一个直观类比**：用只能记 4 位数字的草稿纸加 7168 个数，每 128 个誊到 8 位正式账本上。草稿纸的误差最多累积 128 次，不会蔓延到全部 7168 次。

### 3.8 SiLU 激活：BF16

```
gate = silu(x @ W1^T)   # BF16
up   = x @ W3^T          # BF16（刚从 FP8 GEMM 输出）
y    = gate * up          # element-wise 乘法，BF16
```

- **精度**: BF16
- **原因**: SiLU `x·σ(x)` 涉及 sigmoid → 指数运算 → 精度敏感。虽然论文标记为 🟡 中，但 SiLU 的输入和输出都是 BF16，不额外量化到 FP8（只在 GEMM 入口量化）。

### 3.9 Shared Expert: 同样的 FP8 GEMM 流程

```
shared_expert: MLP(dim=7168, inter_dim=2048)  # V3 配置中 inter_dim = n_shared_experts × moe_inter_dim
  → w1: FP8 GEMM → SiLU
  → w3: FP8 GEMM
  → gate * up (BF16)
  → w2: FP8 GEMM
```

与 routed expert 完全相同的 FP8 流程。代码中 `MLP` 和 `Expert` 使用相同的 `Linear` 类 → 自动继承 FP8 逻辑。

### 3.10 最终输出：FP32

```
hidden (BF16)
  → RMSNorm (BF16)
  → Output Head Linear (BF16) → logits
  → Cross-Entropy Loss (FP32)
```

- **Output Head**: 保持 BF16（V3 不将其量化到 FP8，代码中 `self.head = ColumnParallelLinear(..., dtype=torch.get_default_dtype())` 即 BF16）
- **Loss**: 必须 FP32。`log_softmax` 在概率接近 0 时输出极大负值，FP8/FP16 无法安全表示。

---

## 四、反向传播中的 FP8（简述）

反向传播与前向对称，三种 GEMM 都用 FP8：

| GEMM 类型 | 输入 | 输入精度 | 权重精度 | 输出精度 | 量化 tile |
|-----------|------|---------|---------|---------|-----------|
| **Fprop** | 激活 x | FP8 E4M3 | FP8 E4M3 | BF16 | 1×128 |
| **Dgrad** | 输出梯度 ∂L/∂y | FP8 E4M3 | FP8 E4M3 | BF16 | 1×128 |
| **Wgrad** | 激活 x | FP8 E4M3 | FP8 E4M3 | FP32 | 1×128 |

**要点**：
- V3 统一用 E4M3（而非标准 E5M2 for Dgrad/Wgrad）— 细粒度量化使 E4M3 动态范围也能覆盖梯度
- Wgrad 输出是 FP32（不是 BF16）— 因为梯度要累加到 FP32 master weight
- 激活在前向时被缓存在 FP8 格式中，反向直接复用，节省显存

---

## 五、FP8 在通信和优化器中的应用

### 5.1 激活通信（All-to-All Dispatch）

```
MoE Dispatch 前:
  激活 (BF16) → act_quant(scale_fmt="ue8m0") → FP8 → IB 发送
  ↑ scale 四舍五入到 2 的整数幂（ue8m0 格式）
  ↑ 原因: 简化反向的 scale 计算，避免量化误差传播
```

- **Dispatch**: BF16 → FP8 量化后发送，省一半带宽
- **Combine**: 保持 BF16（精度敏感 — 多 expert 输出求和，量化误差可能累积）

### 5.2 注意力后激活的特殊处理（E5M6）

注意力之后的线性算子的输入激活（同时用于 Attn 反向）使用定制的 **E5M6** 格式：
- 额外增加指数位 → 更大动态范围 → 保护需要发往 Attn 反向的激活
- 量化 tile 从 1×128（前向）转为 128×1（反向）
- 缩放因子四舍五入为 2 的整数幂

### 5.3 优化器状态：BF16

AdamW 的一阶矩 m_t 和二阶矩 v_t 以 **BF16** 存储（而非传统 FP32）：
- 实验表明无可见性能下降
- Master weight 和梯度仍保持 FP32

---

## 六、配置切换：BF16 vs FP8 模式

从推理代码 [model.py:16](model.py:16) 可以看到，V3 支持两种模式：

```python
gemm_impl: Literal["bf16", "fp8"] = "bf16"  # 默认 BF16
```

| 模式 | GEMM 实现 | 权重格式 | 适用场景 |
|------|----------|---------|---------|
| `bf16` | `weight_dequant(FP8→BF16) + torch F.linear(BF16)` | FP8 存储 → 运行时反量化 | 默认推理（精度优先）|
| `fp8` | `act_quant + fp8_gemm(FP8)` | FP8 直接计算 | 训练 / 追求最大吞吐 |

V3 的**训练**使用 FP8 模式；社区推理通常用 BF16 模式（简单可靠，且 V3 官方推荐）。

---

## 七、关键代码路径速查

| 功能 | 文件 | 行号 | 说明 |
|------|------|------|------|
| 激活量化 | [kernel.py](../../code/deepseek-ref/V3/inference/kernel.py) | 10-57 | `act_quant_kernel`, `act_quant` — 1×128 tile 量化 |
| 权重反量化 | [kernel.py](../../code/deepseek-ref/V3/inference/kernel.py) | 60-110 | `weight_dequant_kernel`, `weight_dequant` — 128×128 block 反量化 |
| FP8 GEMM | [kernel.py](../../code/deepseek-ref/V3/inference/kernel.py) | 118-196 | `fp8_gemm_kernel`, `fp8_gemm` — 带 per-group scale 的 GEMM |
| 统一线性层入口 | [model.py](../../code/deepseek-ref/V3/inference/model.py) | 131-163 | `linear` — BF16/FP8 双路径 |
| Linear 层 | [model.py](../../code/deepseek-ref/V3/inference/model.py) | 166-205 | `Linear.__init__` — FP8 weight + scale 参数初始化 |
| FP8→BF16 转换工具 | [fp8_cast_bf16.py](../../code/deepseek-ref/V3/inference/fp8_cast_bf16.py) | 1-108 | 离线 FP8 checkpoint → BF16 checkpoint |

---

## 八、与通用混合精度文档的关系

本文是 [混合精度训练](mixed-precision-training.md) 的 DeepSeek-V3 专项深入分析。通用概念（浮点格式、量化基础、Loss Scaling、精度敏感算子分类）请参见该文档。

**V3 在混合精度演进中的位置**：

```
FP16+AMP → BF16 → FP8 (V3) → FP4 QAT (V4)
                        ↑
                   本文分析的方案
```

---

## 九、延伸阅读

- 📖 [DeepSeek-V3 Technical Report §3.3](https://arxiv.org/abs/2412.19437) — FP8 训练原始章节
- 📝 [DeepSeek-V3 中文笔记](../../papers-zh/DeepSeek-V3.md) — 论文精读
- 📖 [混合精度训练（通用）](../../topics/mixed-precision-training.md) — 基础知识 + FP8/FP4 总览
- 📖 [FP8 Formats for Deep Learning (Micikevicius 2022)](../../papers/FP8%20Formats.pdf) — FP8 格式标准
- 📝 [DeepSeek-V4 Technical Report 中文笔记](../../papers-zh/DeepSeek-V4-technical-report.zh.md) — FP4 QAT 演进
- 💻 [code/deepseek-ref/V3/inference/](../../code/deepseek-ref/V3/inference/) — FP8 推理实现
