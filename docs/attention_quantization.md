# Attention 量化 KV Cache 框架设计

> 状态：Host reference contract v0.4  
> 日期：2026-08-21  
> 范围：只定义并验证 Attention 消费量化 KV 的框架语义；不包含 NPU kernel

## 1. 目标与边界

当前目标是在保持 FlashInfer Attention public signature 的前提下，为昇腾推理场景增加
可版本化、可调度、可独立验证的量化 KV Cache 契约。它需要先回答四个问题：

1. 量化数据的逻辑 shape 与物理存储 shape 如何对应。
2. scale/zero-point 的 shape 和索引如何从逻辑坐标推导。
3. Attention 在什么位置反量化 K/V。
4. plan、run、workload fingerprint 如何保证使用同一个量化语义。

本阶段不声称支持 CANN、torch_npu、FP8、NVFP4、MX、设备 swizzle 或任何 NPU
性能结果。Host oracle 是 correctness contract，不是生产 fallback。

## 2. 分层模型

```mermaid
flowchart LR
    API["FlashInfer-compatible Attention facade"] --> P["AttentionPlanSpec"]
    API --> QKV["ReferenceQuantizedKVData"]
    P --> QS["QuantSpec: immutable semantics"]
    QKV --> K["K storage + K scale/zero"]
    QKV --> V["V storage + V scale/zero"]
    P --> W["WorkloadSpec / fingerprint"]
    QKV --> R["Consumer-side indexed dequant"]
    R --> A["Reference Attention math"]
```

- `QuantSpec` 描述 scheme、storage/compute/accumulator dtype、granularity、axis、
  group size、zero-point、packing order 和 physical layout。
- `ReferenceQuantizedTensor` 绑定一个逻辑 tensor 的量化 storage、scale 和可选
  zero-point，并提供按逻辑坐标读取的语义。
- `ReferenceQuantizedKVData` 为 K/V 保留独立的 scale/zero tensor；当前两者要求使用
  相同的 `QuantSpec` 配置，但 scale/zero 数值不共享。
- `AttentionPlanSpec.kv_quant_spec` 进入 `WorkloadSpec.quant_specs`，因此量化与非量化
  plan、以及不同量化粒度的 plan，具有不同 fingerprint。

## 3. 数学语义

对逻辑坐标 `i`：

```text
K(i) = (Kq(i) - Kzero(scale_index(i))) * Kscale(scale_index(i))
V(i) = (Vq(i) - Vzero(scale_index(i))) * Vscale(scale_index(i))
```

对称量化没有 zero-point 时按 `zero = 0` 处理。scale 必须使用
`QuantSpec.scale_dtype`，所有元素必须有限且大于零。显式 zero-point 使用 `int32`，
shape 与 scale 完全相同，且数值必须落入逻辑量化 dtype 的范围。

Reference executor 在读取每个 K/V 元素时反量化，不先物化完整 FP16/BF16 cache。
这冻结了未来 fused consumer kernel 的语义边界。Host 计算使用 Python 浮点数；
`compute_dtype` 和 `accumulator_dtype` 当前属于 plan/dispatch contract，不模拟逐指令舍入。

## 4. 逻辑 shape 与 scale shape

`QuantSpec.axis` 始终引用逻辑 tensor 轴，并允许负轴。scale 轴顺序与 `axis` 声明顺序一致。

- `granularity="tensor"`：`axis=None`，scale shape 为标量 `()`。
- `channel/token/page`：scale shape 为每个声明轴对应的逻辑维度。
- `group/block`：`group_size` 与 `axis` 一一对应，scale 的每一维是
  `ceil(logical_dim / group_size)`；索引是 `logical_index // group_size`。

KV 逻辑 shape：

| Cache | Layout | K logical shape | V logical shape |
| --- | --- | --- | --- |
| Ragged | NHD | `[token, head, qk_dim]` | `[token, head, vo_dim]` |
| Ragged | HND | `[head, token, qk_dim]` | `[head, token, vo_dim]` |
| Paged | NHD | `[page, slot, head, qk_dim]` | `[page, slot, head, vo_dim]` |
| Paged | HND | `[page, head, slot, qk_dim]` | `[page, head, slot, vo_dim]` |

典型 scale 设计：

| 语义 | NHD axis | HND axis | scale shape |
| --- | --- | --- | --- |
| Ragged per-token/head | `(0, 1)` | `(1, 0)` | `[token, head]` |
| Ragged per-token/head/group-D | `(0, 1, 2)` | `(1, 0, 2)` | `[token, head, ceil(D/G)]`，group size 对应 `(1,1,G)` |
| Paged per-page/head | `(0, 2)` | `(0, 1)` | `[page, head]` |
| Paged per-slot/head/group-D | `(0,1,2,3)` | `(0,2,1,3)` | 轴声明顺序对应的 group 网格 |

同一个 K/V `QuantSpec` 可以因 `qk_dim != vo_dim` 推导出不同的末维 scale shape；
这是合法的，因为两者绑定不同的 scale tensor。

## 5. INT8、UINT8 与 packed INT4

当前 Host contract 支持：

| `storage_dtype` | 物理 `ReferenceTensor.dtype` | 逻辑范围 |
| --- | --- | --- |
| `int8` | `int8` | `[-128, 127]` |
| `uint8` | `uint8` | `[0, 255]` |
| `int4_packed` | `uint8` | `[-8, 7]` |
| `uint4_packed` | `uint8` | `[0, 15]` |

INT4 沿逻辑 tensor 的最后一维打包，物理末维为 `ceil(logical_last_dim / 2)`。
`packing_order` 必须是：

- `low_nibble_first`：偶数逻辑元素位于低 4 bit。
- `high_nibble_first`：偶数逻辑元素位于高 4 bit。

有符号 INT4 使用 4-bit two's-complement 解码。逻辑末维为奇数时，最后一个 byte
未使用的 nibble 必须为零；这使 checksum、缓存复用和后续设备 converter 的行为稳定。

## 6. Public facade 接入

不增加或改写现有 FlashInfer 参数位置：

- single prefill/decode：K、V 传入两个 `ReferenceQuantizedTensor` 即显式选择 Host
  量化 oracle。
- batch prefill/decode 和 mixed `BatchAttention`：`plan(..., kv_data_type=quant_spec)`；
  run 传入 `ReferenceQuantizedKVData`。
- `kv_cache_sf` 仍保留上游 NVFP4 含义，当前显式报未实现，不能借用该参数表达 INT8/INT4。

把 `QuantSpec` 对象作为 `kv_data_type` 是本项目的框架扩展：Python signature 与上游一致，
但运行时类型契约不是 upstream exact parity。未来 Torch frontend 可以增加清晰命名的
量化 cache wrapper；不能把额外语义隐藏在裸字符串 dtype 中。

## 7. Plan/run 一致性

run 必须同时匹配：

- storage dtype；
- 完整 `QuantSpec`；
- KV layout、head count、QK/VO dimension、page size 和设备；
- quantized K/V 的逻辑 shape、storage shape、scale shape 和 zero-point shape。

量化 plan 收到普通 K/V，或普通 plan 收到量化 K/V，必须失败。即使 storage dtype
同为 `int8`，granularity、axis、group size、zero-point 或 packing order 不同也必须失败。

## 8. 当前验证与后续门禁

Host tests 已覆盖 INT8/UINT8、对称/非对称、per-tensor/per-token/group、INT4 两种
nibble order、奇数 padding、独立 K/V scale，以及 single、ragged、paged 和 mixed
Attention 端到端消费。mixed 属性测试进一步覆盖 NHD/HND、GQA、共享/重复 page、packed
INT4、asymmetric UINT8 channel scale、per-head runtime K/V scale、sliding window 与
plan/runtime soft-cap，并逐元素对比显式解量化后的 dense oracle。

Corpus v4 明确包含 paged INT4 multi-request/shared-page、groupwise paged
decode/GQA/QK-VO 不同维度，以及以下两个 mixed 联合门禁：

- packed INT4 + 奇数维 + shared/repeated page + window + runtime soft-cap + per-head scale；
- asymmetric UINT8 + per-head channel scale/zero-point + HND + GQA + runtime soft-cap/scale。

`window_left` 属于归一化 `AttentionPlanSpec` 和 executor/backend 的内部契约；上游
`BatchAttention.plan` public signature 当前没有对应参数，因此框架不扩写该公开接口。
window 组合只在内部 plan/reference contract 中验证，避免产生伪 upstream parity。

量化准确度 v1 已通过 paired dense/quantized trace 将量化误差与未来 backend 执行误差
分开，详见 [`attention_accuracy.md`](attention_accuracy.md)。内置 accuracy corpus 覆盖精确
INT8、lossy asymmetric UINT8、奇数维 packed INT4 和必须拒绝的 scale overflow。

进入 Torch/NPU 层前仍需补齐：

1. 扩展更多 scale 极值、累加 dtype、shape/head mapping 的格式化预算矩阵。
2. Torch metadata wrapper 已有协议级验证；真实 Torch/torch_npu 的 stride/device/stream 与 allocator lifetime acceptance。
3. physical layout descriptor/catalog、三组件 shape、conversion plan 与 KV POD/binary ABI v2 已冻结；仍需真实昇腾 descriptor、converter artifact，以及 v2 packet/provider 接线。
4. capability profile v1 已冻结完整 QuantSpec predicate；仍需用真实 SoC/CANN tuple 生成 evidence。
5. accuracy report 已进一步绑定 launch packet、成功 provider completion 与 lifecycle trace；仍需可信 runner attestation。

FP8、NVFP4、MX、真实非逻辑 layout descriptor/converter、K/V 不同 `QuantSpec` 配置和
量化 packed combined-KV allocation 仍是显式 gap。
