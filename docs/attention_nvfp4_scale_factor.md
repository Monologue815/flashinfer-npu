# Attention NVFP4 KV scale-factor 契约

## 范围

`inspect_attention_nvfp4_kv_scale_factors()` 定义公开 `kv_cache_sf` 进入 provider lowering 前的
Host 侧 metadata 契约。它对齐 FlashInfer 的 NVFP4 KV 表达，但不实现量化、反量化、layout
转换或 NPU 算子。

metadata 检查形成 `AttentionNvfp4ScaleFactorView`。只有显式构造
`AttentionOperatorNvfp4ScaleFactorBinding` 并把它与精确 operation catalog 行绑定后，
`AttentionOperatorNvfp4ScaleFactorRunAdapter` 才能把 scale factor 加入 lowered call。
仓库没有为任何真实 provider 注册该 binding，因此现有生产运行路径继续失败关闭。

scale-factor 不能脱离 packed K/V storage 单独形成可执行授权。二者的联合 shape、layout、
设备和 alias 闭合由
[`Attention NVFP4 packed KV 联合输入契约`](attention_nvfp4_packed_kv.md)定义。

## 公开结构

single Attention 只接受 `(k_scales, v_scales)`；scale tensor 与 K/V 的逻辑排列一致，最后一维
为对应原始 head dimension 除以 16。

paged Attention 接受两种形式：

- separate：`(k_scales, v_scales)`；
- combined：单个 tensor，第二维长度为 2，索引 0/1 分别对应 K/V。

对 page 数为 `P`、page size 为 `S`、KV head 数为 `H`、head dimension 为 `D` 的等维 K/V：

| layout | separate K 或 V | combined |
|---|---|---|
| NHD | `[P, S, H, D/16]` | `[P, 2, S, H, D/16]` |
| HND | `[P, H, S, D/16]` | `[P, 2, H, S, D/16]` |

separate 形式允许 key/value head dimension 不同，并分别推导形状；combined 形式要求两者相等。
single 模式使用相同的 NHD/HND 轴顺序，但没有 page 维。

## 失败关闭规则

- K/V head dimension 必须能被 16 整除；
- scale dtype 必须是 `float8_e4m3fn`；
- scale tensor 必须与 provider query 位于同一设备；
- 当前已注册的 scale-factor layout 只有线性连续布局；
- paged scale tensor 的 page 容量必须覆盖 page table 中每个引用；
- tuple 必须恰好包含 K、V 两个 tensor；
- single Attention 不接受 combined scale tensor；
- combined tensor 不允许 K/V head dimension 不同。

检查器只读取注入的 tensor metadata inspector，不访问 tensor 数据、不导入 provider package、
不分配 device tensor，也不执行算子。未来若某个 Ascend provider 使用非线性 scale layout，
必须新增带 provenance 的 layout descriptor，而不能把任意非连续 strides 当作等价输入。

## Operation 参数绑定

一个 binding 固定 provider id、operation id、完整 NVFP4 `QuantSpec`、线性 scale-factor layout，
并声明以下一种或两种映射：

- combined：公开 combined tensor 映射到一个 provider keyword；
- separate：公开 K/V tuple 映射到两个不同 provider keywords。

所有目标参数必须同时是 operation catalog 的 keyword arguments 和 quant arguments。缺失 K/V
之一、重复参数名、operation/provider 漂移、未声明 quant 参数或 positional-only 参数都会在
lowering 前失败。

binding 接受的 `QuantSpec` 必须同时满足：

- symmetric、无 zero point；
- packed storage dtype 为 `uint8`；
- scale dtype 为 `float8_e4m3fn`；
- 最后一维每 16 个逻辑值共享一个 block scale；
- `axis=(-1,)`；
- 使用非 `logical` physical layout；
- 明确声明 provider packing order。

`attention_nvfp4_kv_quant_spec()` 用于构造该逻辑契约，但不会猜测昇腾 provider 的 physical
layout 或 nibble packing；两项必须由接入模块命名。`infer_attention_nvfp4_packed_storage_shape()`
只冻结两个 E2M1 值存入一个 byte 的 shape 关系，因此逻辑最后一维 `D` 对应 packed storage
最后一维 `D/2`。它不执行编码，也不决定两个值在 byte 内的顺序。

lowering 会先比较 active plan 与 binding 的完整 QuantSpec fingerprint。layout、packing、block
size、scale dtype 或 compute/accumulator dtype 任一漂移，都会在 tensor metadata observation
和 base adapter 调用前失败。

adapter 先执行本文的 metadata contract，再应用 provider access policy 的 alignment 和 output
alias 规则。验证后的 scale-factor views 进入 `validated_input_views`，以便 completion validation、
execution receipt 和离线证据链继续追踪输入身份。adapter 随后清除内部 request 的
`kv_cache_sf`，让已有 base adapter 完成其余参数 lowering，最后再注入精确目标参数；与 base
参数或 input-view 名称冲突时拒绝组合。

这一层只生成 `AttentionLoweredOperatorCall`，不解析 callable、不调用 package，也不执行 NPU
算子。真实 provider 必须通过 contribution/bundle bootstrap 交付相同 binding 身份后，才能
进入自动路由。

生产 runtime 不应只组合 scale-factor adapter。公开 packed KV route 应使用
`AttentionOperatorNvfp4PackedKVBinding` 和 `AttentionOperatorNvfp4PackedKVRunAdapter`，使
storage 与 scale 在同一个 lowering 边界闭合；scale-factor adapter 保留为参数映射的底层契约。

## 与量化绑定的关系

`kv_cache_sf` 是 NVFP4 的 per-block scale-factor 载体，与公开 `k_scale`/`v_scale` calibration
参数不同，也不能复用现有 INT8/INT4 的 key/value scale source。后续 operation binding 必须：

1. 将本文已经固定的 NVFP4 storage `QuantSpec` 与
   `AttentionNvfp4PackedLayoutDescriptor`、联合 packed-KV view 和 capability rule 绑定；
2. 使用本文的 operation binding 声明 provider callable 接收 combined 还是 separate scale factors；
3. 把本契约产生的 tensor views 纳入 alignment、alias 和 execution identity；
4. 在 lowering 中显式消费 `kv_cache_sf`，且不与 `k_scale`/`v_scale` 混淆；
5. 只有 CANN 或 flash-attention-npu 的精确 operation、版本、能力与证据同时匹配时才允许执行。

FlashInfer 对 `kv_cache_sf` 的公开说明见其
[single prefill API](https://docs.flashinfer.ai/generated/flashinfer.prefill.single_prefill_with_kv_cache.html)
和 [Attention API](https://docs.flashinfer.ai/api/attention.html)。本仓库的通用量化授权模型见
[Attention quantization](attention_quantization.md)。
