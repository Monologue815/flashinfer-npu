# Attention 调用指南

## 调用者与集成层的分工

模型代码使用 FlashInfer 风格的 Attention 函数或 wrapper。批量调用分为两步：
`plan()` 描述本次工作负载，`run()` 提供实际 tensor 并返回结果。调用者不传入
provider 名称、算子句柄、内部 binder 或可执行 plan；框架在规划时自动选择符合条件的实现。

部署集成层负责安装经过审核的算子版本、能力声明、参数适配和运行时依赖。
下文的 batch mask 路径还要求显式配置 mask 参数映射、完成事件记录器和
`AttentionBatchRuntimeOwner`。这些是进程初始化配置，不是模型每次调用的参数。
当前仓库没有默认启用真实 CANN 或 flash-attention-npu 实现；仅安装这些包并不能使
下面的 NPU 路径立即可执行。真实设备数值正确性与性能也不能由框架契约推导。

## 量化 KV 与 mask 的 paged prefill

以下代码使用显式量化输入，这是 FlashInfer-NPU 对量化 KV 数据描述的扩展。
`plan()` / `run()` 的参数位置保持兼容，但 `QuantSpec` 和
`AttentionOperatorQuantizedKVInput` 不是上游 FlashInfer 的通用输入类型。
上游裸 FP8、NVFP4 参数的独立契约见[量化说明](attention_quantization.md)。

示例假定集成已在构造 wrapper 前完成，并由应用提供下列输入；它不会创建或复制设备 tensor。

| 输入 | 约束 |
| --- | --- |
| `workspace_buffer` | NPU 上的一维 uint8 workspace，容量满足所选实现 |
| `qo_indptr`、`paged_kv_indptr`、`paged_kv_indices`、`paged_kv_last_page_len` | 合法批次与页表元数据；当前前端要求可直接读取的 Host 整数元数据，不隐式搬运设备索引 |
| `num_qo_heads`、`num_kv_heads`、`head_dim_qk`、`head_dim_vo`、`page_size` | 与 Q、KV 和所选实现一致的整数配置 |
| `q_dtype`、`o_dtype` | 规划时的 query/output dtype 名称 |
| `quant_spec` | 完整且被 provider 支持的量化语义；本例要求对称量化，无 zero-point |
| `k_storage`、`v_storage`、`k_scale`、`v_scale` | 独立 K/V 存储及 scale tensor，shape、dtype、device 与量化语义匹配 |
| `k_logical_shape`、`v_logical_shape` | 量化前的逻辑 shape；本例 NHD paged 布局为 `[page, slot, head, dim]` |
| `q` | 本次 query tensor，shape 为 `[总 QO token 数, Q head 数, QK dim]` |
| `custom_mask`、`packed_custom_mask` | 不使用时为 `None`；格式见下节 |

<!-- example: paged_quant_mask -->
```python
from flashinfer_npu.prefill import BatchPrefillWithPagedKVCacheWrapper
from flashinfer_npu.attention import AttentionOperatorQuantizedKVInput

attention = BatchPrefillWithPagedKVCacheWrapper(
    workspace_buffer, kv_layout="NHD", backend="auto"
)
attention.plan(
    qo_indptr, paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len,
    num_qo_heads, num_kv_heads, head_dim_qk, page_size,
    head_dim_vo=head_dim_vo,
    q_data_type=q_dtype, kv_data_type=quant_spec, o_data_type=o_dtype,
    custom_mask=custom_mask, packed_custom_mask=packed_custom_mask,
)
kv_cache = AttentionOperatorQuantizedKVInput(
    quant_spec=quant_spec,
    key_storage=k_storage, value_storage=v_storage,
    key_scale=k_scale, value_scale=v_scale,
    key_logical_shape=k_logical_shape, value_logical_shape=v_logical_shape,
)
output, lse = attention.run(q, kv_cache, return_lse=True)
```

这条链路不会在框架中反量化 KV 或计算 Attention；框架检查输入契约，将 storage、scale
及 mask 映射到所选实现的准确参数。无匹配实现时明确失败，不自动改用 Host 计算。

## mask 与计划复用

批量 bool mask 是 rank-1、按请求拼接的行优先 allow-mask：`True` 表示可见。
第 `i` 个请求的逻辑长度是 `qo_len[i] * kv_len[i]`。
packed mask 是 rank-1 uint8、little-endian 位序，每个请求独立补齐到整字节；总长度为
`sum(ceil(qo_len[i] * kv_len[i] / 8))`，不能先把整个批次拼接后统一取整。
同时提供两种 mask 时，packed 输入优先，bool 输入不读取。无效 packed 输入不会回退到 bool。

直接借用路径不打包、不取反、不 reshape，也不复制 mask。mask 的内容和存储身份必须在
使用该计划的调用完成前保持有效且不被并发修改。元数据检查不等于内容不可变证明。
需要换 mask、改 shape、页表或量化配置时重新 `plan()`；失败的重新规划保留原有计划。
不再使用 mask 时，重新规划并令两种 mask 参数都为 `None`。

如果上述规划信息未变，可以多次 `run()`。下面假设 `q_next` 与 `q` 的规划约束一致，
上一次调用已完成，且所选实现支持调用者提供的输出 buffer：

<!-- example: reuse_plan_buffers -->
```python
output_again, lse_again = attention.run(
    q_next, kv_cache, return_lse=True, out=output, lse=lse
)
```

复用同一输出分配前，应用/集成层必须确保没有未完成调用仍在写入它；非阻塞的框架
资源回收不会替应用等待设备完成。不同调用可以在同一个有效 plan 下传入新的 KV storage/scale，
但每次输入仍须满足该 plan 的量化语义及元数据约束。

## 自动选择与诊断

规划时先依据模式、dtype、量化语义、mask 格式、布局和环境排除不兼容实现，再按安装的
优先级和评分策略选择。运行时复用已绑定实现，不重新评分，不在失败后隐式换算子重试。
诊断是可选的，只读选择信息不参与下一次运行：

<!-- example: inspect_selection -->
```python
selection = attention.plan_selection
selected_operation = selection.operation_id
```

batch ragged prefill 使用同样的 `plan()` / `run()` 分工，但 `run(q, k, v)` 保持分离的
K/V 参数；显式量化时二者分别使用 `AttentionOperatorQuantizedTensorInput`。
single prefill/decode 则是一次性函数，调用者不额外传 plan。
single custom-mask 与 provider graph 路径目前仍未开放。

## 生命周期与可运行的 Host 示例

模型调用者无需手动获取或释放内部 plan。集成服务必须保留 runtime owner，并在服务关闭时
调用其非阻塞、可重试的 `close()`；未完成或记录失败的调用保持资源所有权，不能靠超时、
Python 函数返回或垃圾回收推断完成。具体恢复与关闭协议见
[调用资源生命周期](attention_operator_call_retention.md)。

没有真实 provider 集成时，可在仓库目录运行 `python3 -m examples.attention_plan_run`，
体验已有 CPU reference 的混合批次、INT8 输入和 plan/buffer 复用。
它使用 Host 容器演示参考语义，不是 NPU 算子，也不是自动生产 fallback。

设计细节见 [plan/run 与调度](attention_plan_run_dispatch_design.md)、
[能力边界](support_matrix.md)和[公开参数归属](attention_frontend_contract.md)。
