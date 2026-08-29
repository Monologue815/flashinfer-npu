# Attention NVFP4 packed KV 联合输入契约

## 1. 目的与边界

NVFP4 Attention 不能只验证 `kv_cache_sf`。框架必须证明 packed K/V bytes、block scale、
`AttentionPlanSpec.kv_quant_spec` 和 provider 声明的物理布局属于同一个量化语义，才能把输入交给
CANN 或 flash-attention-npu 的候选算子。

`inspect_attention_nvfp4_packed_kv_input()` 完成这一 Host metadata 闭合，并返回
`AttentionNvfp4PackedKVView`。它不读取 tensor 数据，不执行 E2M1 编码或解码，不导入外部算子包，
不分配 NPU tensor，也不调用 Attention 算子。

本契约支持 FlashInfer Attention 当前公开 NVFP4 形态：

- single Attention 的 separate `(K, V)` storage 与 `(K scale, V scale)`；
- paged Attention 的 separate K/V；
- paged Attention 的 combined K/V，第二维长度为 2。

公开 canonical 格式由 `flashinfer_nvfp4_kv_quant_spec()` 唯一表示：每个 `uint8` 保存两个 E2M1
值，低 nibble 对应偶数逻辑元素，高 nibble 对应奇数逻辑元素；每 16 个逻辑值使用一个
`float8_e4m3fn` scale。paged wrapper 的 `plan(..., kv_data_type=uint8)` 按 FlashInfer 语义解释
为该 NVFP4 格式，`run(..., kv_cache_sf=...)` 提供 K/V scale。通用 UINT8 量化必须传显式
`QuantSpec`，不会与这个保留语义混淆。

ragged NVFP4、layout conversion 和真实 provider 执行不由本文隐式推断。需要这些能力时，provider
必须交付新的精确契约和证据。

## 2. 为什么使用专用 view

通用 `QuantizedTensorView` 的 group/block scale shape 只保留 `QuantSpec.axis` 声明的量化轴。
例如 NVFP4 的 `axis=(-1,)`、`group_size=(16,)` 会得到 `(D/16,)`。这是已有 INT8/INT4
通用契约的既定含义，不能为了 NVFP4 改成另一种解释。

FlashInfer NVFP4 KV scale 则保留 K/V 的 page、token/slot 和 head 外层坐标，仅把最后一维从
`D` 变成 `D/16`。因此本仓库使用独立的 `AttentionNvfp4PackedKVView`：

- 不改变通用 INT8/INT4 scale shape；
- 显式冻结 NVFP4 的完整外层索引；
- 同时验证 packed storage 与 scale-factor；
- 允许未来 provider 为其他 physical layout 注册不同的、可审核的 descriptor，而不是复用一个
  含义不相同的通用 view。

## 3. Layout descriptor

`AttentionNvfp4PackedLayoutDescriptor` 固定以下身份：

- `physical_layout`：必须与 plan 的 `QuantSpec.physical_layout` 完全相同；
- `packing_order`：必须与 plan 的 `QuantSpec.packing_order` 完全相同；
- storage shape rule：`linear_last_dim_e2m1_pair_v1`；
- scale shape rule：`logical_outer_dims_d16_v1`；
- storage 与 scale 各自的最小地址对齐。

当前 storage shape rule 表示两个逻辑 E2M1 值存入一个 `uint8`，所以逻辑 shape
`[..., D]` 对应 storage shape `[..., D/2]`。公开 canonical descriptor 同时固定低 nibble/高
nibble 的逻辑顺序；该顺序进入 QuantSpec、descriptor 和 binding fingerprint。

scale shape rule 保留全部外层维，只把逻辑最后一维变为 `D/16`。`D` 必须能被 16 整除。

descriptor 是 provider 声明，不是框架猜测。CANN 或 flash-attention-npu 接入模块只有在其真实
tensor ABI 可直接消费公开 canonical 格式时才能复用该 descriptor；否则必须在 provider 内部
注册显式 converter 和新的版本化目标 descriptor。转换后的私有布局不参与 facade plan，也不
改变模型侧接口。

## 4. Shape 契约

设 page 数为 `P`、page size 为 `S`、KV head 数为 `H`、head dimension 为 `D`。

### 4.1 Paged NHD

| 组件 | separate K 或 V | combined K/V |
|---|---|---|
| logical | `[P, S, H, D]` | K/V 共享该 logical shape |
| packed storage | `[P, S, H, D/2]` | `[P, 2, S, H, D/2]` |
| scale factor | `[P, S, H, D/16]` | `[P, 2, S, H, D/16]` |

### 4.2 Paged HND

| 组件 | separate K 或 V | combined K/V |
|---|---|---|
| logical | `[P, H, S, D]` | K/V 共享该 logical shape |
| packed storage | `[P, H, S, D/2]` | `[P, 2, H, S, D/2]` |
| scale factor | `[P, H, S, D/16]` | `[P, 2, H, S, D/16]` |

combined 形式要求 K/V head dimension 相等。separate 形式允许 `head_dim_qk != head_dim_vo`，
K/V 的 storage 和 scale 末维分别计算。

### 4.3 Single

single NHD logical shape 为 `[T, H, D]`，HND 为 `[H, T, D]`。packed storage 最后一维为
`D/2`，scale 最后一维为 `D/16`。single 只接受 separate K/V，不接受插入 K/V 轴的 combined
tensor。

## 5. 检查顺序

联合检查严格按以下语义闭合：

1. plan 必须携带由公开输入规范化得到的 canonical NVFP4 `QuantSpec`；
2. descriptor 的 layout 与 packing 必须和完整 QuantSpec 相同；
3. storage 必须是合法的 separate 或 combined 结构；
4. paged storage 的 page 容量必须覆盖 page table 的最大引用；
5. 从 plan metadata 推导 K/V logical shape；
6. 从 descriptor 推导 packed storage shape 和完整 scale shape；
7. `kv_cache_sf` 必须通过独立 scale-factor 契约，且结构与 storage 相同；
8. storage dtype 必须是 `uint8`，scale dtype 必须是 `float8_e4m3fn`；
9. 所有组件必须位于 provider query 的同一设备，满足连续布局和 descriptor 对齐；
10. packed storage、K/V storage 和 scale-factor 之间不得发生 storage alias。

返回 view 携带 plan fingerprint、完整 QuantSpec、layout descriptor 及所有命名 TensorView。
这些身份可以继续进入 lowering、completion validation 和 execution receipt，但 view 自身不会使
任何 provider 获得执行权限。

## 6. 联合 lowering 边界

`AttentionOperatorNvfp4PackedKVBinding` 把两项已经审核的身份组成一个不可变 binding：

- `AttentionOperatorNvfp4ScaleFactorBinding`：provider/operation、完整 QuantSpec，以及
  combined 或 separate scale-factor 参数映射；
- `AttentionNvfp4PackedLayoutDescriptor`：packed storage/scale shape rule、packing order 和
  最小对齐。

两部分必须引用同一个 physical layout、packing order 和完整 QuantSpec。binding fingerprint
覆盖两者，避免只替换 layout descriptor 或参数映射后仍复用旧的审核身份。

`AttentionOperatorNvfp4PackedKVRunAdapter` 是公开 packed KV 与 provider lowering 之间唯一的
联合消费边界：

1. active plan 的完整 QuantSpec 不匹配时，只有未提供 `kv_cache_sf` 才允许交给其他 adapter；
2. active plan 命中 binding 时，`kv_cache_sf` 必须存在；
3. adapter 调用本文的联合 inspector，并再次应用 provider access policy；
4. 输出不得与 packed storage 或 scale-factor alias；
5. adapter 从交给 base adapter 的内部 request 中清除 `kv_cache_sf`，防止重复消费；
6. base adapter 完成 query/KV 等普通参数 lowering 后，adapter 才注入精确的 scale keyword；
7. packed storage 与 scale-factor 的全部命名 view 进入 `validated_input_views`；
8. 参数名或 view 名与其他 adapter 冲突时失败关闭。

不匹配的量化 plan 不会被该 adapter 截获。这样通用 INT8/INT4/FP8 quantization adapter 与
NVFP4 packed route 可以在同一 operation runtime 中保持互斥语义，而不是根据 `uint8` dtype 或
`kv_cache_sf` 参数名猜测格式。

`AttentionOperatorNvfp4PackedKVRunAdapterFactory` 只在 provider device 已确定后绑定上述 adapter。
factory 构造、binding 校验和 `lower()` 都不解析 package callable，也不执行外部代码。

## 7. Runtime spec 注册

provider 通过 `AttentionOperatorPackageRuntimeSpec.nvfp4_packed_kv_bindings` 注册一个或多个联合
binding。bootstrap 把 operation capability rules 中的全部 QuantSpec 划分为两个互斥集合：

- `quantization_bindings`：通用 INT8/INT4/FP8 quantized-input route；
- `nvfp4_packed_kv_bindings`：公开 packed KV + `kv_cache_sf` route。

两个集合不能包含相同 QuantSpec fingerprint，其并集必须与该 operation 的 capability QuantSpec
集合完全相等。缺失 binding、没有 capability 的孤立 binding、重复 routing、operation/provider
identity 漂移都会在 package metadata probe 和 callable import 之前失败。

通用 quantization adapter 会获得明确的 delegated QuantSpec 集合。遇到 NVFP4 plan 时它只把请求
交给联合 adapter，遇到自己拥有的 QuantSpec 时才解析项目 quantized-input wrapper；因此 adapter
顺序不再承担隐式格式选择。联合 factory 以完整 binding 集合构造一个 fingerprint-indexed route，
没有命中的 `kv_cache_sf` 仍失败关闭。

`AttentionOperatorPackageRuntimeDeclaration` 保存
`nvfp4_packed_kv_binding_fingerprints`。layout、packing、alignment、operation 参数映射或完整
QuantSpec 任一变化都会改变 declaration fingerprint，并被 `validate_runtime_spec()` 识别为审核后
漂移。声明只保存 SHA-256 身份，不序列化 tensor、adapter 对象或 callable。

## 8. Provider 接入边界

真实 provider 接入需要同时提供：

1. capability rule：证明目标 mode、dtype、layout、head dimension 和完整 NVFP4 QuantSpec 可用；
2. packed layout descriptor：与外部 package 的实际 storage/scale ABI 一致；
3. operation catalog 与精确参数 binding：声明 combined 或 separate `kv_cache_sf` 参数；
4. runtime bootstrap：在 `nvfp4_packed_kv_bindings` 注册联合 binding，并由框架把 factory 加入
   operation 的 adapter chain；
5. package/version/artifact evidence：防止同名参数或不同版本被误当成相同语义；
6. completion 与准确性证据：证明真实执行结果属于被选中的 plan 和 operation。

框架只在上述身份全部闭合时自动选择算子。缺少 descriptor、参数 binding 或版本证据时，候选
provider 必须失败关闭，不能退化成基于参数名的猜测。

scale-factor 的公开形态和 operation 参数注入规则见
[`attention_nvfp4_scale_factor.md`](attention_nvfp4_scale_factor.md)，通用量化模型见
[`attention_quantization.md`](attention_quantization.md)。
