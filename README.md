# LLM Training Wiki

大模型训练知识库，覆盖分布式训练基础设施、模型架构、强化学习后训练等方向。

## 目录结构

```
├── papers/          # 论文 PDF（38 篇，覆盖并行策略/注意力/MoE/RL 等）
│   └── README.md    # 论文索引表
├── notes/           # 论文精读笔记 / 中文翻译
├── topics/          # 按技术主题整理的专题笔记
├── interview/       # 面试准备资料
├── code/deepseek-ref/  # 参考实现（DeepSeek V3/V3.2/V4 推理代码）
└── scripts/         # 工具脚本
```

## 知识地图

### 分布式并行
| 主题 | 论文 | 笔记 |
|------|------|------|
| 张量并行 (TP) | [Megatron-LM TP](papers/Megatron-LM%20TP.pdf) | — |
| TP + PP + DP | [Megatron-LM TP PP DP](papers/Megatron-LM%20TP%20PP%20DP.pdf) | — |
| 专家并行 (EP) | [DeepSpeed-MoE EP](papers/DeepSpeed-MoE%20EP.pdf) | [Expert Parallelism](topics/expert-parallelism.md) |
| 序列并行 (SP) | [Megatron-LM TP PP DP](papers/Megatron-LM%20TP%20PP%20DP.pdf), [DeepSpeed Ulysses](papers/DeepSpeed%20Ulysses%20SP.pdf) | — |
| 上下文并行 (CP) | [Ring Attention CP](papers/Ring%20Attention%20CP.pdf) | — |
| 统一 SPMD | [GSPMD SPMD](papers/GSPMD%20SPMD.pdf) | — |
| 自动并行搜索 | [Alpa Parallelism Search](papers/Alpa%20Parallelism%20Search.pdf) | — |

### 内存优化
| 主题 | 论文 |
|------|------|
| ZeRO-1/2/3 | [ZeRO Memory Optimization](papers/ZeRO%20Memory%20Optimization.pdf) |
| ZeRO-Infinity (NVMe) | [ZeRO-Infinity NVMe Offload](papers/ZeRO-Infinity%20NVMe%20Offload.pdf) |
| 选择性重计算 | [Selective Recomputation SP](papers/Selective%20Recomputation%20SP.pdf) |
| PyTorch FSDP | [PyTorch FSDP](papers/PyTorch%20FSDP.pdf) |

### 注意力机制
| 主题 | 论文 |
|------|------|
| FlashAttention-3 | [FlashAttention-3](papers/FlashAttention-3.pdf) |
| Ring Attention | [Ring Attention CP](papers/Ring%20Attention%20CP.pdf) |
| 混合注意力 (CSA + HCA) | [DeepSeek-V4 Technical Report](papers/DeepSeek-V4%20Technical%20Report.pdf) |

### MoE 架构
| 主题 | 论文 |
|------|------|
| GShard | [GShard MoE Sharding](papers/GShard%20MoE%20Sharding.pdf) |
| DeepSeekMoE | [DeepSeekMoE](papers/DeepSeekMoE.pdf) |
| DeepSeek-V2 (MLA + MoE) | [DeepSeek-V2 MLA MoE](papers/DeepSeek-V2%20MLA%20MoE.pdf) |
| MegaBlocks (稀疏 MoE) | [MegaBlocks MoE Sparse](papers/MegaBlocks%20MoE%20Sparse.pdf) |

### 训练框架与 Infra
| 主题 | 论文 |
|------|------|
| DeepSeek-V3 (FP8 Infra) | [DeepSeek-V3 FP8 Infra](papers/DeepSeek-V3%20FP8%20Infra.pdf) |
| DeepSeek-V4 (1M Context) | [DeepSeek-V4 Technical Report](papers/DeepSeek-V4%20Technical%20Report.pdf) |
| FP8 格式 | [FP8 Formats](papers/FP8%20Formats.pdf) |
| AXLearn | [AXLearn Heterogeneous](papers/AXLearn%20Heterogeneous.pdf) |
| MegatronApp (任务管理) | [MegatronApp Management](papers/MegatronApp%20Management.pdf) |
| ByteRobust (容错) | [ByteRobust Fault Tolerance](papers/ByteRobust%20Fault%20Tolerance.pdf) |

### 强化学习 / 后训练
| 主题 | 论文 |
|------|------|
| DeepSeek-R1 (RL 推理) | [DeepSeek-R1 RL Reasoning](papers/DeepSeek-R1%20RL%20Reasoning.pdf) |
| DeepSeekMath (GRPO) | [DeepSeekMath GRPO](papers/DeepSeekMath%20GRPO.pdf) |
| GRPO 机制分析 | [GRPO Analysis](papers/GRPO%20Analysis.pdf) |
| REINFORCE++ | [REINFORCE++ Algorithm](papers/REINFORCE++%20Algorithm.pdf) |
| DPO | [DPO Algorithm](papers/DPO%20Algorithm.pdf) |
| Scaling RLHF | [Scaling RLHF](papers/Scaling%20RLHF.pdf) |
| DeepSpeed-Chat RLHF | [DeepSpeed-Chat RLHF](papers/DeepSpeed-Chat%20RLHF.pdf) |
| OpenRLHF | [OpenRLHF Framework](papers/OpenRLHF%20Framework.pdf) |
| ROLL 框架 | [ROLL RL Framework](papers/ROLL%20RL%20Framework.pdf) |
| ROLL RL4LLM Tricks | [ROLL RL4LLM Tricks](papers/ROLL%20RL4LLM%20Tricks.pdf) |
| ROLL Flash Async | [ROLL Flash Async](papers/ROLL%20Flash%20Async.pdf) |

### 模型系列
| 模型 | 论文 |
|------|------|
| DeepSeek-LLM | [DeepSeek-LLM](papers/DeepSeek-LLM.pdf) |
| DeepSeek-Coder | [DeepSeek-Coder](papers/DeepSeek-Coder.pdf), [DeepSeek-Coder-V2](papers/DeepSeek-Coder-V2.pdf) |
| DeepSeek-V2 | [DeepSeek-V2 MLA MoE](papers/DeepSeek-V2%20MLA%20MoE.pdf) |
| DeepSeek-V3 | [DeepSeek-V3 FP8 Infra](papers/DeepSeek-V3%20FP8%20Infra.pdf), [DeepSeek-V3.2](papers/DeepSeek-V3.2.pdf) |
| DeepSeek-V4 | [DeepSeek-V4 Technical Report](papers/DeepSeek-V4%20Technical%20Report.pdf) |
| DeepSeek-R1 | [DeepSeek-R1 RL Reasoning](papers/DeepSeek-R1%20RL%20Reasoning.pdf) |

### 其他
| 主题 | 论文 |
|------|------|
| 数据中心网络 | [Astral Datacenter Network](papers/Astral%20Datacenter%20Network.pdf) |
| 容错训练 | [ByteRobust Fault Tolerance](papers/ByteRobust%20Fault%20Tolerance.pdf) |
