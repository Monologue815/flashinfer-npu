# Attention 数值特殊值契约

> 状态：Numerics policy v1  
> 日期：2026-08-13  
> 范围：Host conformance oracle；尚未声称 FlashInfer CUDA 或昇腾 backend 已通过

## 1. 目的

普通有限输入不能覆盖推理系统中的所有数值路径。全 mask、空 KV、量化极值、dot-product
overflow 以及 NaN 输入如果没有显式规则，不同 Python/C++/AI Core reduction 顺序会产生不同
结果。`AttentionNumericsPolicy` 将 row-wise softmax 的特殊值行为版本化并提供 fingerprint，
后续 backend 必须用同一组 trace/property cases 证明兼容。

## 2. Policy v1

| Row 状态 | 概率/支持集 | Output | LSE |
| --- | --- | --- | --- |
| 有限 logits | subtract-max stable softmax | 加权 V | `max + log(sum(exp))` |
| custom/all mask 或空 KV | 空支持集 | 全零 | `-inf` |
| 任意可见 logit 为 NaN | 整行传播 NaN | 全 NaN | NaN |
| 一个或多个 logit 为 `+inf` | 只在 `+inf` 位置均分 | 对这些 V 求均值 | `+inf` |
| 所有可见 logits 为 `-inf` | 空支持集 | 全零 | `-inf` |

零概率项不参与 V reduction，因此未选中的 `NaN/Inf` V 不会通过 `0 * value` 污染结果。
被选中的 V 仍使用确定顺序的 IEEE 浮点加权和；例如同时归一到 `+inf` 和 `-inf` V 时可得到
NaN。

`q_scale/k_scale/v_scale` 与量化 scale 已在进入数值循环前要求 finite；量化 scale 还必须
严格为正。Q/K/V tensor data 本身不做 O(numel) 的预扫描，而是在 row 运算中按上述规则
传播。这一点适合真实异步 backend：frontend 不应为了数值检查隐式同步设备。

## 3. 与 mask 的区别

mask 在计算 QK 前移除 key；被 mask 的 Q/K/V 无论包含何值都不应被读取或影响输出。
`-inf` logit 仍属于可见 key，但 v1 在整行均为 `-inf` 时把它归一为零支持集。两条路径的
observable result 相同，诊断原因可以不同。

## 4. 当前证据与兼容性边界

Host tests 覆盖：

- 大有限 logits 的稳定 softmax；
- NaN 位于不同 key 顺序时结果一致；
- 多个 `+inf` key 均分，有限 key 即使 V 为 Inf 也不污染；
- 全 `-inf`、全 mask、paged 空 KV 的零 output/`-inf` LSE；
- trace JSON 对 NaN/±Inf 使用显式对象编码，不产生非标准 JSON token。

这是本项目的 v1 correctness gate，不是尚未测量的上游或硬件事实。真实 Torch CPU、
FlashInfer CUDA 对照以及 torch_npu/昇腾 functional backend 应分别记录验证结果；如果某个
backend 的原生语义不同，必须作为 capability gap 暴露，不能静默改写 oracle。
