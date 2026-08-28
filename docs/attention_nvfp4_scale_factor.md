# Attention NVFP4 KV scale-factor 契约

## 范围

`inspect_attention_nvfp4_kv_scale_factors()` 定义公开 `kv_cache_sf` 进入 provider lowering 前的
Host 侧 metadata 契约。它对齐 FlashInfer 的 NVFP4 KV 表达，但不实现量化、反量化、layout
转换或 NPU 算子。

本阶段只形成 `AttentionNvfp4ScaleFactorView`，不把任何 provider 标记为 eligible，也不把
scale factor 注入外部 callable。没有后续精确 operation binding 时，现有运行路径继续失败关闭。

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

## 与量化绑定的关系

`kv_cache_sf` 是 NVFP4 的 per-block scale-factor 载体，与公开 `k_scale`/`v_scale` calibration
参数不同，也不能复用现有 INT8/INT4 的 key/value scale source。后续 operation binding 必须：

1. 固定 NVFP4 storage `QuantSpec` 和 physical layout；
2. 声明 provider callable 接收 combined 还是 separate scale factors；
3. 把本契约产生的 tensor views 纳入 alignment、alias 和 execution identity；
4. 在 lowering 中显式消费 `kv_cache_sf`，且不与 `k_scale`/`v_scale` 混淆；
5. 只有 CANN 或 flash-attention-npu 的精确 operation、版本、能力与证据同时匹配时才允许执行。

FlashInfer 对 `kv_cache_sf` 的公开说明见其
[single prefill API](https://docs.flashinfer.ai/generated/flashinfer.prefill.single_prefill_with_kv_cache.html)
和 [Attention API](https://docs.flashinfer.ai/api/attention.html)。本仓库的通用量化授权模型见
[Attention quantization](attention_quantization.md)。
