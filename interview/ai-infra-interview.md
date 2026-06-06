# 大模型训练 Infra 面试大纲 — 内行知识细节

## 分布式并行
### 张量并行（TP）与序列并行（SP）
- Megatron TP 中 ColumnLinear/RowLinear 的通信原语分别是 all-reduce 还是 all-gather+reduce-scatter，二者在 NVLink 带宽下的 crossover 点在哪里（面试侧重：是否手写过 TP 切分代码）
- SP 的三种实现分支：Megatron-SP（仅切 LayerNorm/RMSNorm 和 Dropout 的激活，用 all-reduce 聚合）、Ring Attention（切 attention 的 Q/K/V 并做环形 P2P 传输）、USP（DeepSpeed-Ulysses 的 all-to-all + Ring 的复合策略），各自的通信量公式推导和跨机/机内适用边界（面试侧重：是否在真实集群上踩过 SP 选型坑）
- TP + SP 组合时 layer_norm 的 forward/backward 通信重叠是否真的零开销——Megatron-Core 里的 `_allreduce` 和 layernorm 的 cuda stream 编排细节（面试侧重：是否看过具体 CUDA kernel launch 序列）
- 当 TP size 超过单机 8 卡时，跨机 TP 的 NVLink 域截断如何导致通信时间从 10us 级恶化到 ms 级，以及为什么几乎没有人在万卡训练中对 attention 做跨机 TP（面试侧重：是否理解物理拓扑对并行策略的硬约束）

### 流水线并行（PP）
- Megatron 的 interleaved 1F1B 调度中，micro-batch 数 m 与 pipeline stage 数 p 的约束关系（m >= 2p 时 pipeline bubble 才能小于某个下界），以及 warmup/cooldown 阶段的显存峰值分布（面试侧重：是否调过 PP 的 schedule 参数）
- PP 与 TP 组合的设备网格映射：为什么优先将 TP group 放在同一 NVLink 域内（如 DGX 8xA100 内），而将 PP stage 跨机放置——以及违反此原则时的真实性能退化数据（面试侧重：是否在物理集群上做过拓扑感知的策略搜索）
- Zero-Bubble PP（ZB-H1/ZB-H2）中通过手工注入 backward 的「虚假」micro-batch 来填满 bubble 的技巧，以及为什么这种 trick 在大 batch size 下会因为 optimizer step 的延迟反而退化（面试侧重：是否追踪过最新的 PP 调度论文并做过复现）
- PP 的显存不均衡根因：首尾 stage 为何比中间 stage 少存一份 micro-batch 激活，以及 interleaved 调度是否缓解了这一问题（面试侧重：是否能用公式推导显存分布）

### 数据并行（DP）与 ZeRO
- ZeRO-1/2/3 中 optimizer state / gradient / parameter 三者分片后的 all-gather 时机差异：ZeRO-2 的 gradient 在 reduce-scatter 后即释放，而 ZeRO-3 的 parameter 需要在 forward 前 all-gather、用完立即释放——这两条时序线上的通信-显存 tradeoff（面试侧重：是否在代码里跟过 ZeRO 的通信原语）
- ZeRO-3 的 `stage3_max_live_parameters` 和 `stage3_prefetch_bucket_size` 两个参数如何影响 OOM 与吞吐——参数取值过小导致频繁 all-gather，过大导致显存尖峰拖慢 throughput（面试侧重：是否有大规模 ZeRO-3 的调参经验）
- FSDP2 的 `reshard_after_forward` 与 ZeRO-3 的参数释放语义是否完全等价——以及 FSDP2 基于 DTensor 的子类重写如何影响与 torch.compile 的兼容性（面试侧重：是否理解 FSDP 的 DTensor 表示与 autograd hook 的交互）
- DP 的 gradient all-reduce 与 TP 的 all-reduce 在通信量上的差异：DP 通信量与参数量成正比，TP 与激活量成正比乘以层数——以此为据推导混合并行的最优配置方向（面试侧重：是否具备并行策略的数学建模能力）

### 专家并行（EP / MoE）
- EP 中 all-to-all 通信的容量因子（capacity factor）如何影响 token drop 率和通信量：capacity factor=1.0 时每个 expert 最多接收 (tokens/num_experts) 个 token，超出部分直接丢弃，capacity factor=1.25 时保留 25% 冗余——如何在模型质量与吞吐间 tradeoff（面试侧重：是否理解 token drop 与 capacity factor 的量化关系）
- DeepSpeed-MoE 的 hierarchical all-to-all（先在机内 NVLink 做 all-to-all，再跨机 IB 做 all-to-all）相比扁平 all-to-all 的通信量缩减比率的推导（面试侧重：是否清楚 all-to-all 的拓扑感知优化）
- Expert 负载不均的常见解法：auxiliary load balancing loss 的权重调参、expert choice 路由（让 expert 主动选 token 而非 token 选 expert）、capacity factor 的动态调整——各自的收敛影响与工程代价（面试侧重：是否在 MoE 训练中实际诊断过 load imbalance）
- EP + TP 的组合：expert 内部的 MLP 是否再做 TP 切分——以及当 expert 参数量很小（如 Mixtral 的单个 expert 仅 ~160M）时做 TP 引入的通信延迟可能超过计算收益（面试侧重：是否理解 EP 下 TP 的收益临界点）

### 并行策略的组合与冲突
- 5D 并行 (TP+PP+DP+EP+SP) 的设备网格映射：如何将 2D/3D 的 device mesh 扩展到 5D，以及 PyTorch 的 `DeviceMesh` 从 2D 起每个维度如何映射到物理拓扑（面试侧重：是否在真实万卡规模上做过 mesh 设计）
- 并行策略的自动搜索（如 AlphaComm、FlexFlow、Alpa 等）在万卡规模下为何几乎不可用——搜索空间爆炸、profile 开销不可接受、模型变更后策略复用率低（面试侧重：是否批判性地评估过自动并行方案）

## 显存优化
### 激活重计算（Activation Checkpointing / Recomputation）
- Megatron 的 `recompute_method` 中 `uniform`（均匀放置 checkpoint 边界）vs `block`（按 Transformer block 边界放置）在 PP interleaved 调度下的显存峰值差异——`block` 方式在 warmup 期可能因 micro-batch 堆积导致更高峰值（面试侧重：是否对不同 recompute 策略做过显存 profiling）
- 选择性重计算（selective recompute）：哪些 op 的激活重计算开销最小——attention 的 softmax 输出（小张量但重计算需要完整 attention 矩阵）vs MLP 的 GELU 输出（大张量但计算快）的选择逻辑（面试侧重：是否理解不同 op 的激活大小/重计算代价比）
- FlashAttention 的 backward 重计算与手动 activation checkpointing 的二选一问题——FA 的 backward 已通过 recompute 省去 QK^T 和 attention 矩阵的存储，此时在 FA 外部再加 checkpointing 是否还有收益（面试侧重：是否理解 FA 内部的显存机制）

### 精度与数值格式
- BF16 训练中，为什么 optimizer 的 master weights 必须保持 FP32 而 forward/backward 可以用 BF16——以及 Megatron 的 `--fp16-lm-cross-entropy` 为什么在计算 cross-entropy loss 时必须回到 FP32 以避免 NaN（面试侧重：是否遇到过 BF16 训练的数值不稳定并定位到具体 op）
- FP8 训练中 transformer engine 的 delayed scaling 策略——amax 的 history buffer 长度如何影响 scaling factor 的稳定性与精度损失，以及为什么 delayed scaling 在 MoE 的 expert 路由不均衡时容易产生数值溢出（面试侧重：是否在 FP8 训练中调试过 loss spike）
- FP8 中 E4M3 与 E5M2 两种格式的使用分工：E4M3（更高精度、更窄范围）用于 forward 的矩阵乘，E5M2（更宽范围、更低精度）用于 backward 的梯度乘——为什么 backward 需要更大的动态范围（面试侧重：是否理解混合 FP8 格式的设计动机）
- Micro-scaling（MXFP8/MXFP6/MXFP4）与 per-tensor scaling 的区别：block-wise 量化中 32 个元素共享一个 scaling factor，其硬件支持现状（NV 的 Blackwell 是否原生支持）及对训练精度的实际影响（面试侧重：是否追踪最新的数值格式标准）

### Offload 层级
- ZeRO-Offload 的 1-CPU-Adam 设计：为什么选择把 optimizer state + FP32 optimizer step offload 到 CPU 而非 offload forward/backward——forward/backward 的计算密度和通信量的比值推导（面试侧重：是否手工算过 offload 策略的通信-计算 overlap）
- NVMe offload（ZeRO-Infinity）相比 CPU offload 的带宽从 ~50GB/s 降到 ~5GB/s，但容量从 ~1TB 扩展到 ~10TB——为什么 NVMe offload 在实际中几乎只在 batch size 扩展时使用，而很少用于常规训练加速（面试侧重：是否理解 offload 带宽的实际 wall clock time 占比）
- PagedAttention 式的 KV-cache offload 思想能否用于训练的 activation offload——为什么训练中的激活是「用完即弃」而非「持久复用」，使得训练 offload 的核心是 optimizer state 而非 activations（面试侧重：是否区分推理和训练 offload 的本质差异）

### 显存生命周期与碎片化
- PyTorch 的 CUDA caching allocator 在 training loop 中的 `empty_cache()` 调用时机：为什么在 grad all-reduce 完成和 optimizer step 之间的窗口调用可能导致显存回收不彻底，以及 `torch.cuda.memory_stats` 中 `allocated_bytes` 和 `reserved_bytes` 的差值来源（面试侧重：是否用 memory profiler 诊断过实际 OOM）
- Tensor 分片后的碎片化：DTensor 的 `redistribute` 操作中隐式的 `_to_copy` 产生的临时 GPU tensor 为何在 profiler 中难以追踪（面试侧重：是否在生产中遇到过分片策略变更时的显存尖峰）

## 通信优化
### 网络协议与拓扑
- InfiniBand 的 SHARP（Scalable Hierarchical Aggregation and Reduction Protocol）在 all-reduce 中将 reduce 操作 offload 到交换机——但为什么在混合精度训练中 SHARP 默认仅对 FP32 生效，对 BF16/FP16 需要额外配置且版本有严格限制（面试侧重：是否在 IB 集群上配置过 SHARP 并测量过收益）
- RoCE v2 相比 InfiniBand 的 PFC（Priority Flow Control）痛点：无损网络要求下 PFC 的 pause frame 可能引发拥塞蔓延（congestion spreading）和死锁——以及为什么大模型训练的突发通信模式特别容易触发此问题（面试侧重：是否在 RoCE 网络上遇到过通信 hang 并定位到 PFC）
- NVLink 带宽中 unidirectional vs bidirectional 的实际利用——NCCL 的 ring all-reduce 是否真的能充分利用双向带宽，以及在 TP all-reduce 场景下 NVSwitch 的 all-to-all 延迟与 ring 方式的 crossover 节点数（面试侧重：是否针对 GPU 拓扑手写过 nccl-tests 的 microbenchmark）

### 自定义通信与 NCCL 深度使用
- NCCL 的 `NCCL_ALGO` 环境变量强制指定 all-reduce 算法（Tree vs Ring vs CollnetDirect）在什么场景下优于 NCCL 的自动调优——以及为什么 CollnetDirect（使用 SHARP）在跨机通信中收益显著但在机内反而退化（面试侧重：是否针对自己的模型规模手动调过 NCCL 算法）
- NCCL 的 `NCCL_PROTO` 中 Simple vs LL（Low Latency）vs LL128 协议的实际选择逻辑——Simple 协议对大数据量（>128KB）使用、LL/LL128 对小数据量使用，以及 LL128 的 128-byte 粒度为何能减少控制消息开销（面试侧重：是否理解 NCCL 内部的协议栈）
- 手写 CUDA kernel 替代 NCCL 的场景——例如 MoE 的 all-to-all 中，P2P 直连可以按 expert 粒度做 token 路由感知的通信调度，而 NCCL 的 all-to-all 是全局同步的——手写版的通信-计算 overlap 如何实现以及额外工程成本（面试侧重：是否有手写集合通信 kernel 的经验）
- DeepSpeed 的 `comm_backend` 选择：为什么在 InfiniBand 上用 `nccl` backend 而在 RoCE 或以太网上考虑 `ccl`（oneCCL）或 `gloo`——不同 backend 的初始化开销和集合通信算法覆盖度的差异（面试侧重：是否在多网卡混合环境中调试过通信问题）

### 通信-计算 Overlap
- Megatron 的 `overlap_param_gather` 与 `overlap_grad_reduce` 如何利用 CUDA stream 将通信藏在计算后面——以及为什么 overlap 在跨机（IB/RoCE）场景收益更显著，而在机内 NVLink 场景可能因为 kernel launch 开销反而不如不 overlap（面试侧重：是否测量过 overlap 在不同网络环境下的实际收益）
- 梯度分桶（gradient bucketing）的大小选取：桶太小导致通信次数增加、桶太大导致第一块算完后迟迟等不到最后一块完成才能发通信——DDP/FSDP 中 `bucket_cap_mb` 的默认 25MB 在实际大模型（>70B）下为何需要调整（面试侧重：是否调过梯度分桶参数）

### 通信死锁场景
- TP + PP 组合中，PP 的 P2P send/recv 与 TP 的 all-reduce 在 NCCL 内部使用同一通信队列时的死锁风险：当 PP 的 recv 阻塞等待 upstream stage 发送，而 upstream stage 的 NCCL all-reduce 因为 ring 算法中下一节点的 recv 未就绪而无法完成时触发死锁（面试侧重：是否在生产中遇到过 NCCL 死锁并定位到 root cause）
- ZeRO-3 的 implicit all-gather 导致的通信死锁：在 `forward()` 中对同一 parameter group 做多次 all-gather 时，如果第二次 all-gather 依赖第一次 all-gather 释放的显存，而 NCCL 内部 buffer 因为第一次通信未完成而无法回收——如何用 `wait()` 或 stream sync 规避（面试侧重：是否 Debug 过 ZeRO-3 的通信 hang）

## 框架差异
### Megatron-Core vs DeepSpeed
- Megatron-Core 的 `TransformerBlock` 定义与 DeepSpeed 的直接封装 HuggingFace 模型的根本差异：Megatron 自行管理 tensor 的 partition 边界（`mpu` 的 `VocabParallel`、`ColumnParallelLinear` 等），DeepSpeed 通过 hook 机制注入通信——这导致 Megatron 可以做 weight 预分片（从 checkpoint 加载时即按 TP/PP 切好），而 DeepSpeed 需要先加载完整权重再分片（面试侧重：是否处理过大规模模型加载时 OOM 的问题）
- Megatron 的 `gpt_bias_swiglu_fusion` 等 kernel fusion 是在 Megatron-Core 内部通过 Apex 或 TE 实现的，DeepSpeed 依赖编译器（如 `torch.compile` / `deepseed.ops`）做后处理——这两种 fusion 策略在 MoE 的 expert 计算中的实际性能差距（面试侧重：是否 benchmark 过两个框架的 MoE 训练吞吐）
- DeepSpeed 的 ZeRO-3 + CPU Offload 与 Megatron 的 TP+PP+SP 显存优化路线的根本分歧：ZeRO 以通信换显存（每层参数的 all-gather 通信），TP 以额外的 all-reduce 通信换显存——在跨机带宽有限时 ZeRO-3 的 each-layer all-gather 为何比 TP 的 all-reduce「更贵」（面试侧重：是否能用带宽模型推导两种路线的优劣边界）

### FSDP2（PyTorch 原生）
- FSDP2 的 `fully_shard` API 与 FSDP1 的 `FullyShardedDataParallel` 的根本架构变化：FSDP2 基于 DTensor 和 `torch.compile` 做图级别优化，而 FSDP1 在 autograd hook 层面拦截——FSDP2 的通信可以被 `torch.compile` 的图重排优化，但 DTensor 的 `from_local` 和 `redistribute` 可能在 compile 的图中引入不可融合的同步点（面试侧重：是否尝试过 FSDP2 + torch.compile 的组合）
- FSDP2 中 `reshard_after_forward` 的默认 True 与 ZeRO-3 的行为等价，但 FSDP2 的 `reshard_after_forward=False`（即不释放参数）在什么场景下反而提升吞吐——当模型可以完整放入单卡显存时，保留参数可以省去下一轮 all-gather，但失去了 ZeRO 的显存优势（面试侧重：是否理解 FSDP/ZeRO 参数释放的 tradeoff）
- PyTorch 的 DTensor 在 TP 场景下 `redistribute` 从 replicate 到 shard 的转换中，隐式插入的 `all_gather / reduce_scatter` 与手写的 `dist.all_gather` 相比是否有额外开销（面试侧重：是否跟过 DTensor 的具体通信原语代码）

### 训练框架的核心数据结构差异
- Megatron 的 `ModelParallelEmbedding` 对 embedding 表做 VocabParallel 切分，其反向传播中 `all-reduce` 的语义是 sum 还是 identity——以及为什么 embedding 的梯度 all-reduce 必须是 reduce-scatter + all-gather 的组合而非简单的 reduce-scatter（面试侧重：是否手动推导过 VocabParallel 的前反向通信语义）
- DeepSpeed 的 `deepspeed.runtime.engine.DeepSpeedEngine` 对 `backward()` 做了 override，在 loss 的 attribute 中注入 `backward` hook——这种 monkey-patch 方式的 backward 拦截与 PyTorch 标准的 `autograd.backward` 兼容问题（面试侧重：是否遇到过自定义 loss 在 DeepSpeed 下的 backward 报错）

## 算子与编译器
### Attention 算子
- FlashAttention-2 相比 FA-1 的核心改进不在 forward 而在 backward：FA-1 的 backward 对 Q/K/V 梯度需要各自遍历一次序列长度维度，FA-2 通过 `torch.vmap` 式的 batch 并行化将 backward 的 HBM 访问量从 O(N^2) 降到 O(N)——但为什么 FA-2 的 backward 仍在 causal mask 场景下有冗余计算（面试侧重：是否实际对比过 FA-1/FA-2/FA-3 的 wall clock time）
- FlashAttention-3 引入的 warp-specialization 和 asynchronous TMA（Tensor Memory Accelerator）是 Hopper 架构独有特性——为什么 FA-3 在 H100 上比 FA-2 快但在 A100 上无法运行，以及 FA-3 在 GQA（Grouped Query Attention）下的 warp 分工策略如何变化（面试侧重：是否在 Hopper 架构上部署过 FA-3）
- PagedAttention 与 FlashInfer 的 block-sparse attention 在训练中的适用性：为什么 vLLM 式的 paged KV-cache 在训练中不可用（训练需要完整 KV），而 FlashAttention 的 window attention / sliding window 是训练可用的局部 attention 方案（面试侧重：是否区分推理和训练 attention 优化的适用边界）

### 融合算子与数值稳定性
- FusedAdam vs AdamW 的数值差异来源：Adam 的 `bias_correction` 项 (1-beta^step) 在 step=1 时近 0，FusedAdam 是否在 CUDA kernel 中对此做了 guard——以及当 step 很大时 `bias_correction` 趋近于 1 后 fused kernel 是否有跳过该分支的优化（面试侧重：是否在训练初期遇到过 fused optimizer 的 loss 抖动）
- RMSNorm 的 CUDA kernel 中 eps 的加法位置：先做 `x^2.sum() + eps` 再 sqrt 的 naive 实现在大激活值下会导致精度丢失，正确做法是在求和时做 double-precision accumulation——以及 Megatron 的 `FusedRMSNorm` 是否做了此优化（面试侧重：是否理解 layernorm/RMSNorm 的数值精度细节）
- SwiGLU 的融合策略：Megatron 的 `gpt_bias_swiglu_fusion` 将 gate_proj 和 up_proj 合并为一个更大的 GEMM（行数翻倍）后再用 element-wise silu + multiply——与分开做两个小 GEMM 再加融合相比，哪种方式更适应 Tensor Core 的 tile 尺寸（面试侧重：是否理解 Tensor Core 的 M/N/K 对齐对 kernel 效率的影响）

### 编译器与图捕获
- `torch.compile` 在训练中的图捕获失败的主导原因：data-dependent control flow（如 MoE 的 top-k 路由、dynamic batch size 的 padding）、non-tensor 的输出（如 attention weights 的返回）、以及 DTensor `redistribute` 的隐式同步——各自的规避方案与性能代价（面试侧重：是否在真实训练脚本上尝试过 `torch.compile` 并解决过图断点）
- `torch.compile` 的 `mode="reduce-overhead"` vs `mode="max-autotune"`：前者通过 CUDA graph 减少 kernel launch overhead 但会固化输入 shape，后者在 Triton matmul 中做 exhaustive autotune 但预热的 wall clock time 可能超过训练本身——在 MoE 的 dynamic shape 场景下，`reduce-overhead` 会因为 shape 不匹配反复重新 compile 导致吞吐退化（面试侧重：是否对 torch.compile 的 mode 做过针对性 benchmark）
- Triton kernel 的 autotune 空间爆炸：一个 matmul 的 BLOCK_M、BLOCK_N、BLOCK_K、num_warps、num_stages 五个参数的笛卡尔积可达数万配置——实际 Triton autotuner 如何在 100 次内收敛到近似最优配置，以及 `triton.Config` 的 `pre_hook` 如何利用已知 shape 做剪枝（面试侧重：是否手写过 Triton kernel 并调过 autotune 参数）

## 容错与弹性
### Checkpoint 的一致性
- 异步 checkpoint（如 Megatron 的 `--async-save`）利用独立 CUDA stream 将 GPU tensor copy 到 CPU 再写入磁盘，但如果 DP group 中的不同 rank 在 checkpoint 边界进入了不同的 micro-batch 迭代次数时——DP 的 gradient all-reduce 是否可能因为 checkpoint 的 CPU copy 延迟而出现数值不一致（面试侧重：是否在生产中验证过异步 checkpoint 的可恢复性）
- DeepSpeed 的 `zero_to_fp32` checkpoint 合并过程中，ZeRO-3 的分片参数如何在 CPU 端合并为完整 FP32 权重——以及当模型过大无法放入单节点内存时，`zero_to_fp32` 的 per-parameter 加载合并策略为何会导致 OOM（面试侧重：是否在 >100B 模型上做过 checkpoint 合并）
- 分布式 checkpoint（`torch.distributed.checkpoint`）的 `FileSystemReader` 和 `ShardedTensor` 的 metadata 一致性：当 save 过程中某个 rank 的 checkpoint 不完整（如被 OOM 中断），其他 rank 的 checkpoint shard 如何检测到不一致——以及 `DCP` 的 `load_state_dict` 对此错误处理的缺失（面试侧重：是否在生产中遇到过加载不完整 checkpoint 导致的静默数据损坏）

### 慢节点绕行（Straggler Mitigation）
- 慢节点的检测方案：NCCL 的 `ncclRemoteError` 能否区分「节点已死」与「节点慢」——以及 DeepSpeed 的 `autotuner` 的 `--strangler` 如何通过 heartbeat 超时区分（面试侧重：是否在万卡集群上处理过慢节点问题）
- 慢节点的绕过代价：DP group 中剔除一个 rank 后需要重建 NCCL communicator、重新分发数据、且 checkpoint 需要从其他 rank 重新加载——整个过程的总 wall clock time 可能超过慢节点等待时间（面试侧重：是否量化过弹性训练的恢复开销与收益）
- 华为的「弹性训练」方案（mindtorch 的 `mindspore.communication` ）与 PyTorch 生态的 torchrun + torchrunx 方案的能力差距——PyTorch 生态在节点增减时需要重启整个 training process group，而某些闭源方案可以在进程级别做热替换（面试侧重：是否对比过不同厂商的弹性方案）

### 错误检测与恢复
- GPU 的 Xid 错误码解读：Xid 63/64（ECC page retirement）是否需要立即停机，Xid 31/43/45（GPU fell off bus）是否一定是硬件故障——以及如何根据 Xid 频率设置自动化踢出策略（面试侧重：是否有 GPU 错误的运维排障经验）
- NCCL 的 `NCCL_DEBUG=INFO` 输出中 `timeout` 的根因分析：是 IB 链路故障、GPU 计算卡死、还是 collective 通信的 barrier 不匹配——以及如何用 `ib_write_bw` 和 `dcgmi diag` 做链路级排查（面试侧重：是否有完整的通信故障排查工具链）

## 超大规模挑战
### 负载不均
- MoE 的 token-to-expert 负载不均 vs expert 的 compute load 不均：capacity factor 解决的是前者（token 数的均衡），但不同 expert 的序列长度分布差异导致的 compute latency 差异（后者）只能通过 expert 内部的 micro-batch 并行度调整缓解——以及这两层不均之间的关系（面试侧重：是否真正理解 MoE 负载不均的双层结构）
- 数据加载的 global batch 不均：DDP 中不同 rank 的 data loader 的 `num_workers` 不足或 CPU 负载不均时，某些 rank 的数据预处理耗时远长于其他 rank，导致 GPU 空等——如何通过 `torch.profiler` 定位是 DataLoader 瓶颈还是通信瓶颈（面试侧重：是否用 profiler 诊断过端到端训练的性能短板）
- PP 的 micro-batch 计算不均：不同 micro-batch 的序列长度由 padding 策略决定，`packed` sequence 可以消除此不均但引入了 attention mask 的稀疏化及额外的 kernel launch 开销，以及 packed sequence 在 TP 切分下的边界处理（面试侧重：是否在生产中使用过 sequence packing）

### 收敛稳定性
- Batch Size 的 scaling law 偏差：当 global batch size 超过某临界值（通常 ~4-8M tokens）后，进一步增加 batch size 不仅不提升收敛速度反而可能导致 loss 发散——以及如何通过调整 learning rate 的 sqrt scaling 而非 linear scaling 来缓解（面试侧重：是否有训练 scaling law 的实际调参经验）
- 不同并行策略下的 gradient noise scale 差异：TP 内部 all-reduce 是精确求和（每步完全同步），DP 的 gradient 在 all-reduce 后才统一步长——但当 TP size 大而 DP size 小时，过少的梯度样本导致 gradient noise 偏高，其与 batch size、learning rate 的交互是否会引发训练不稳定（面试侧重：是否理解并行策略与优化动力学的耦合）
- FP16/BF16 训练中的 loss scale 增长爆表与截断：`dynamic_loss_scale` 的 window size 和 min scale 的设定原则——window size 太小导致频繁触发 scale 降低，太大导致增长过慢浪费数值范围（面试侧重：是否处理过 loss scale 调参导致的收敛问题）

### 能耗优化
- GPU 的 Power capping（如 `nvidia-smi -pl`）降低 TDP 可减少功耗但性能退化非线性——因为 GPU 计算单元频率下降但显存带宽几乎不变，导致计算 bound 的 kernel（如 GEMM）退化显著而 memory bound 的 kernel（如 Layernorm）几乎不受影响（面试侧重：是否在集群级别做过 power capping 实验）
- 数据中心 PUE 与训练任务的能耗分配：万卡训练的空调/液冷能耗与 GPU 能耗在同一量级——液冷系统中冷却液温度每升高 1°C 造成的 GPU leakage power 增加是否可忽略（面试侧重：是否关注过训练任务的总能耗模型）

## 数据工程
### 数据去重
- MinHash vs SimHash 的语义去重选择：MinHash（Jaccard 相似度估计）适合捕捉文本的重叠度但计算量大，SimHash 适合快速近邻查找但 hash bits 数（通常 64 或 128）需要根据语料规模调整——以及为什么在万卡训练中，去重算法的选择会影响模型对长尾知识的记忆（面试侧重：是否有在大规模语料上实施过去重 pipeline）
- 精确去重（Exact Dedup）中 suffix array 的构建对内存的要求，以及大规模语料下如何分块构造 suffix array——构建 1TB 文本的 suffix array 需要多少内存（面试侧重：是否手工算过去重系统的资源需求）

### Tokenizer 与数据预处理
- BPE tokenizer 的增量训练：新增 token 时 embedding 矩阵和 LM head 矩阵的维度扩展后，如何初始化新增行——随机初始化会导致 loss spike，用已有 token 的平均 embedding 做 warm start 可缓解但需精心设计平均策略（面试侧重：是否处理过 tokenizer 扩展后的训练恢复）
- 多语种语料的采样比率（sampling ratio）对模型能力的 tradeoff：过采样低资源语种提升多语言能力但稀释高资源语种性能——以及 SMoE 模型中不同 expert 对不同语种的自然分化是否是采样策略的替代方案（面试侧重：是否在语料配比上调过参）

## 性能剖析
### MFU 计算
- MFU 公式中的「理论 peak FLOPS」是在 boost clock 还是 base clock 下计算——实际训练受 TDP 和散热限制几乎不可能达到 boost clock，因此用 boost peak 计算的 MFU 普遍偏低，以及为什么不同论文/厂商的 MFU 数字不可直接对比（面试侧重：是否理解 MFU 标准化的陷阱）
- MFU 计算中需要扣除的 overhead：通信时间、数据加载停顿、checkpoint 保存、profiling 开销——PaLM 论文的 MFU 计算方式与 Megatron 团队的 MFU 计算方式是否一致，以及各自未扣除的隐藏开销（面试侧重：是否批判性地审视过论文中报告的 MFU 数字）
- 高 MFU 的关键瓶颈在 GEMM 的 tile 利用率而非 peak ops——GEMM 的 Wave 量化（一个 SM 上需要多少个 thread block tile 才能隐藏 memory latency）决定了实际能达到的 flops 占比，以及为什么 H100 比 A100 更难达到高 MFU（更多 SM 需要更大的 GEMM 规模才能填满）（面试侧重：是否理解 GPU 架构层面对 MFU 的约束）

### Rofline 与 Kernel 分析
- A100 到 H100 的 roofline 迁移：H100 的 compute roof 提升 ~3x 但 memory bandwidth 仅提升 ~1.5x——这意味着原本在 A100 上 compute bound 的 kernel 在 H100 上可能变为 memory bound，以及这对并行策略选择的影响（面试侧重：是否根据硬件架构做过并行策略的重新评估）
- `torch.profiler` 与 Nsight Systems 的 profiler overhead 对训练性能的影响——Nsight 的 GPU sampling 模式 overhead 可控制在 2% 以内但仅提供统计信息，trace 模式 overhead 10-50% 但提供完整的 kernel launch 序列——以及如何用 `nsys` 的 `--trace` 选项做针对性裁剪（面试侧重：是否在万卡训练中做过 profiling 而不引入 measurable overhead）
- Kernel launch overhead 在 Transformer 这种大量小 kernel （如 dropout、layernorm、attention mask）场景下的累积效应——CUDA Graph 可将 kernel launch 时间压缩近零，但要求 input shape 固定（面试侧重：是否用 CUDA Graph 优化过训练吞吐）
