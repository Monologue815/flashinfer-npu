# Attention 量化 KV 物理布局与转换计划

> 状态：Physical layout contract v1  
> 日期：2026-08-21  
> 范围：layout schema、shape/coordinate、tensor admission 与 conversion plan；不执行 converter 或 NPU kernel

## 1. 目的与边界

`NHD/HND` 是 Attention KV 的逻辑轴语义，不等于设备上的物理排布。INT4 packing、轴
permutation、block/tile、padding、scale/zero-point 排布和 alignment 必须在进入 backend 前
成为显式对象，不能隐藏在 shape、stride、裸 dtype 字符串或 adapter 的 `.contiguous()` 中。

v1 不登记任何真实 Ascend ND/NZ/分形布局，也不声称知道某个 CANN 版本的内部格式。
`synthetic.*` descriptor 只用于表达框架合同。真实 layout 必须来自固定 runtime tuple 和
官方/实测证据后单独注册。

## 2. 三层 shape

```text
Attention logical shape
  -> sub-byte logical storage shape
  -> registered physical component shape
```

例如 logical INT4 最后一维 `D=33`：

1. packed storage last dimension 先变为 `ceil(33/2)=17` byte；
2. 若 byte 轴按 16 block，则物理 outer/inner shape 为 `[2, 16]`；
3. 末 block 中超出 17 byte 的坐标属于 canonical padding，不能反向映射成逻辑元素。

packing 必须先于 physical blocking。否则 nibble order、奇数维 padding和 tile padding 会被
混为一层，无法建立可逆坐标证明。

## 3. Axis transform

`QuantPhysicalAxisTransform` 使用：

- `input_rank`；
- 每个输入轴的 `axis_blocks`；
- `physical_axes`；
- required alignment；
- canonical padding value。

物理轴 token `oN` 表示输入轴 N 的 outer coordinate，`iN` 表示 block-local coordinate。
每个 outer 轴必须恰好出现一次；只有 block size 大于 1 的轴必须恰好出现一个 inner 轴。
token 顺序同时表达 permutation 和 inner-axis placement。

框架提供 logical-storage ↔ physical coordinate 映射。反向遇到 padding 返回 `None`，越界则
报 schema error。这只做整数坐标验证，不读取或移动 tensor data。

## 4. Layout descriptor 与 catalog

`QuantPhysicalLayoutDescriptor` 是一个 layout bundle，包含：

- 唯一 `layout_id`，与 `QuantSpec.physical_layout` 精确相等；
- 支持的 storage dtype；
- storage transform 与 forward/inverse converter id；
- 可选 scale transform 与 converter id；
- 可选 zero-point transform 与 converter id；
- backend 必需 feature 名称。

storage、scale、zero-point 分开建模，因为它们的 rank、padding value 和 alignment 可以不同。
未声明 scale/zero transform 表示该组件保持 logical layout，不表示继承 storage transform。

`QuantPhysicalLayoutCatalog` 要求 layout id 唯一，支持严格字典 round-trip 和 fingerprint。
当前默认 catalog 为空；因此非逻辑 `QuantSpec` 在没有显式 catalog 时继续失败。

## 5. Conversion plan

`plan_quant_layout_conversion()` 只允许改变 `physical_layout`，scheme、dtype、granularity、axis、
group size、zero-point、packing order 等任一量化语义变化都会失败。

- logical → physical：按组件产生 forward converter step；
- physical → logical：产生 inverse converter step；
- physical A → physical B：必须显式产生 A inverse → logical → B forward；
- 相同 layout：零步骤 no-op。

每步固定 component、source/destination layout、converter id、input/output shape 和 output
alignment。plan 固定 source/destination QuantSpec、所有组件 shape 与 step fingerprint，但不包含
converter artifact，也不执行数据搬运。

## 6. Tensor contract 接入

非逻辑 `QuantizedTensorView` 必须携带准确的 descriptor，并验证：

- storage/scale/zero-point 的物理 shape；
- component dtype/device/alias；
- 每个物理 component alignment；
- K/V 使用完全相同的 layout descriptor。

descriptor fingerprint 进入 `attention_run_tensor_signature()`，所以 graph/capture identity 不会把
同名 layout 的不同 descriptor 当成同一个 tensor ABI。

## 7. Launch ABI v1/v2 明确门禁

现有 `AttentionKVCacheViewPOD` v1 只携带 `QuantSpec` fingerprint，没有独立的 layout descriptor
fingerprint。因此 `materialize_attention_kv_cache_view()` 明确拒绝任何非逻辑 physical layout。

`FlashInferNpuKVCacheViewV2` 已冻结为独立 192-byte ABI，并增加：

- descriptor/catalog fingerprint；
- physical-layout access code；
- physical-layout capability binding fingerprint；
- dispatch receipt fingerprint。

component physical shape/stride/alignment 继续由 v2 指向的逐组件 `TensorViewV1` 固定。非逻辑
layout 当前只允许 `KERNEL_NATIVE`：`bind_attention_kv_physical_layout()` 必须闭合 exact catalog、
profile/rule/environment、kernel artifact/binary ABI、evidence digest 与 dispatch receipt，并证明
descriptor required features 同时出现在 rule、kernel constraint 和 observed environment。

conversion plan 只描述预期数据变换，不是执行或 completion evidence，不能冒充 physical-layout
binding。converter pipeline 在拥有 output lease、artifact/ABI 和 completion record 前仍不得发射。

当前 ownership-complete `AttentionLaunchPacket` 仍是 v1 packet，只接受 KV POD v1。KV POD v2 和
binary ABI v2 已完成独立 Host materialization/capability gate，但尚未接入 packet/provider。因此
当前仍不存在任何可运行的非逻辑布局设备路径。

在这些字段进入 binary ABI 前，不能通过仅修改 `QuantSpec.physical_layout` 绕过门禁。

## 8. 真实昇腾接入顺序

1. 针对固定 SoC/CANN/runtime tuple 收集官方格式定义与可复现实测证据。
2. 注册真实 descriptor；不得用 `synthetic.*` 或猜测名称。
3. 为 forward/inverse converter 建立独立 artifact、workspace、stream/lifetime 和 protocol contract。
4. 用坐标采样、padding canonicalization 和 round-trip corpus 验证 converter。
5. 将 descriptor/feature 精确绑定 capability rule、kernel descriptor 和 dispatch receipt。
6. 将已冻结的 KV POD/binary ABI v2 接入 ownership-complete packet/provider。
7. correctness 与 accuracy execution binding 通过后，才进入性能门禁。

当前完成的是上述步骤的框架前置条件，不是任何真实 layout 或 converter 的实现证据。
