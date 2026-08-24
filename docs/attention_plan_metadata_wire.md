# Attention Plan Metadata Wire v1

> 状态：Framework canonical wire protocol v1  
> 日期：2026-08-20  
> 范围：六种 Attention mode 的 Host materialize/decode；不包含设备内存拷贝或 NPU launcher

## 1. 目标与边界

`AttentionFrameworkPlan` 是 Python 语义对象，不能直接成为稳定 C ABI。Plan metadata wire v1
把 kernel 热路径需要的固定 scalar、CSR offset、page table 和 mask segment offset 物化为唯一
little-endian byte representation。

它只保存 planner 生成的数据：

- single prefill/decode 的长度与 scalar config；
- paged/ragged/mixed 的 `qo_indptr`、`kv_indptr`、page indices、last-page length、KV length；
- custom mask 的 per-request segment offset。

它不保存 custom mask 内容、per-head Q/K/V scale、profiler output 或任何 allocator 地址。这些是
run-time tensor，必须通过 `FlashInferNpuAttentionAuxiliaryViewV1` 和 storage lease 进入 launcher。

## 2. Canonical byte layout

```text
FlashInferNpuAttentionPlanHeaderV1       160 bytes
FlashInferNpuAttentionPlanConfigV1       128 bytes
FlashInferNpuAttentionMetadataSectionV1   32 bytes × N
section body 0                            int32[], 8-byte padded
...
section body N-1                          int32[], 8-byte padded
```

`plan_metadata_nbytes` 是整个 blob 的 byte 数；header 的 `payload_nbytes` 是 header 后面的 byte
数，因此必须满足：

```text
plan_metadata_nbytes == 160 + payload_nbytes
```

所有 struct 是 8-byte aligned、little-endian、fixed-width primitive。所有 reserved 和 section
padding bytes 必须为零，不允许 trailing bytes、重叠 section、乱序 directory 或等价的第二种
编码。

## 3. Header 与 config

160-byte header 固定 schema/mode/flags、payload byte 数，以及 plan、admission、dispatch 和
binary ABI 四个 SHA-256 fingerprint。旧 plan、旧 capability 选择或旧 binary ABI 不能复用
新的 metadata blob。

128-byte config 固定：

- section count 与 32-byte directory entry size；
- batch、Q/KV head、QK/VO dimension、page size、decode `q_len_per_req`；
- position encoding、KV layout、Q/KV/output dtype 与固定 `int32` index dtype；
- window left/right；
- softmax scale、soft cap、RoPE scale/theta；
- total Q/KV token 数与 custom-mask element/byte 数。

mode code v1：

| Code | Mode |
| ---: | --- |
| 1 | single prefill |
| 2 | single decode |
| 3 | batch paged prefill |
| 4 | batch ragged prefill |
| 5 | batch paged decode |
| 6 | mixed paged Attention |

plan flags v1 分别表达 effective causal、custom mask、packed mask、FP16 QK reduction、profiler
和 quantized KV。未知 bit 必须拒绝。

INT4 的逻辑格式与物理 storage 明确分离：config dtype 可为 `INT4`、`INT4_PACKED` 或
`UINT4_PACKED`；KV component TensorView 仍报告真实 `uint8` storage，完整解释由 QuantSpec
fingerprint 固定。

## 4. Mode-specific sections

section directory 固定 kind、element type、flags、payload-relative offset、element count 和 byte
数。v1 section element type只有 signed `int32`。

| Mode | 必需 section |
| --- | --- |
| single prefill/decode | 无 |
| paged prefill | QO indptr、KV indptr、KV indices、last-page length |
| ragged prefill | QO indptr、KV indptr |
| paged decode | KV indptr、KV indices、last-page length |
| mixed paged | QO indptr、KV indptr、KV indices、KV length |

有 custom mask 时，三个 prefill mode 额外携带 `MASK_INDPTR`。unpacked mask 的 segment 长度为
`qo_len * kv_len`；packed mask 为每个 request 独立计算 `ceil(qo_len * kv_len / 8)`，然后做
前缀和。这保留 FlashInfer segment-packbits 边界，不能把整个 batch 连续 pack 后再切分。

## 5. Strict decode 与资源边界

`AttentionPlanMetadataWire.from_bytes()` 在建立 domain metadata 前执行：

- 默认 64 MiB 总 byte、16 sections、每 section 16M elements 上限；
- header/config/directory 完整性和 byte-count equality；
- enum、flags、entry size、INT32 count 与 canonical offset；
- padding/reserved 为零、section 唯一且按 kind 排序；
- 用现有 `PagedKVMetadata`、`RaggedKVMetadata` 等 schema 重建并再次验证 CSR/page 语义；
- batch/token totals、page size、mask segment offset 与 config 交叉验证。

`materialize_attention_plan_metadata()` 只接受当前 canonical binary ABI fingerprint。
`validate_plan()` 重新物化完整 blob 并做 byte equality；只比较 plan id 或长度不被接受。

当前测试覆盖六种 mode、dense/INT8/packed-INT4、packed/unpacked mask、空页请求、固定 blob
fingerprint、INT32 overflow、未知 enum/flag、截断、错 offset、非零 padding/reserved、decode
limits 和 stale plan。它不证明 blob 已复制到 NPU 或任何 kernel 能消费该结构。

