# Attention Plan/Run 与后端选择设计

> 状态：检查点 026，已将量化 KV tensor metadata validation、量化 KV run lowering、exact QuantSpec/API binding、原子 registry 快照、声明式 bootstrap、自动选择、evidence-bearing package authority、plan-time provider tensor
> 物化与受 runtime identity 保护的 callable execution 收入公共 `BatchAttention` 生命周期，
> 并实现 CANN v2 与 flash-attention-npu v3 的纯框架分页 lowering；尚未导入或调用 CANN、torch_npu、
> flash-attention-npu 或任何 NPU 算子。

## 1. 上游事实

FlashInfer 当前存在两类相关入口，不能混为一个接口：

1. `BatchPrefillWithPagedKVCacheWrapper`、
   `BatchDecodeWithPagedKVCacheWrapper` 等 wrapper 在构造时接受
   `backend="auto"`。`plan()` 根据 workload 决定模块并把底层
   `plan_info` 保存到 wrapper；`run()` 不接收 plan 参数。
2. holistic `attention.BatchAttention(kv_layout="NHD", device="cuda")`
   当前不暴露 `backend` 参数。它的 `plan()` 同样返回 `None`，内部保存
   JIT module 与 `plan_info`，`run()` 直接消费这份状态。

当前上游的 `auto` 不是通用 provider 插件遍历器。标准 Attention selector
主要根据设备、dtype、位置编码、自定义 mask 和 head dimension 在 FA2/FA3
之间选择；部分 wrapper 另有显式分支选择 cuDNN、TRT-LLM、CuTe DSL 或
特定 FMHA 实现。

因此，本项目对齐的是它的公共语义和 plan/run 生命周期，不复制 CUDA
模块装载结构，也不把 NVIDIA backend 名称伪装成昇腾能力。

## 2. 冻结的用户接口

普通用户只使用 wrapper，不管理底层 plan handle：

```python
wrapper = BatchPrefillWithPagedKVCacheWrapper(
    workspace,
    kv_layout="NHD",
    backend="auto",
)

wrapper.plan(
    qo_indptr,
    paged_kv_indptr,
    paged_kv_indices,
    paged_kv_last_page_len,
    num_qo_heads,
    num_kv_heads,
    head_dim_qk,
    page_size,
    causal=True,
)

output = wrapper.run(q, paged_kv_cache)
```

公共契约为：

- `plan()` 返回 `None`；成功后原子替换 wrapper 内部的 active plan。
- `run()` 没有 plan 参数，只执行当前 active plan；未 plan 时失败。
- 影响实现选择、workspace 或数学语义的字段由 `plan()` 冻结。
- `run()` 只接收热路径 tensor、输出 buffer 和已声明可动态变化的值。
- 再次调用 `plan()` 可替换 active plan；旧 plan 不再被后续 `run()` 使用。

`plan_state`、未来的 `selected_backend`/`explain_plan()` 仅用于诊断，不能成为
正常调用必需参数。

## 3. 昇腾内部映射

后续实现保持同一外部接口，在 `plan()` 内部完成：

```text
FlashInfer-compatible arguments
  -> AttentionPlanSpec + normalized metadata
  -> provider capability matching
  -> selected provider adapter
  -> provider-owned opaque plan
  -> wrapper active plan
```

首批 provider adapter 目标是：

- CANN/torch_npu 已发布 Attention 能力；
- `flash-attention-npu` 已发布 Attention 能力；
- 仅用于小规模框架验证且必须显式选择的 Host reference。

adapter 负责参数翻译和调用已有包，框架本身不复制这些包的算子。provider
不可用或能力不匹配时必须记录拒绝原因；`reference` 不参与生产 `auto`。

## 4. 与上游保持一致的边界

| 项目 | 决策 |
| --- | --- |
| wrapper `plan()` / `run()` | 精确保持生命周期 |
| plan 是否由用户传给 `run()` | 否 |
| `backend="auto"` 默认值 | 对有该参数的上游 wrapper 保持 |
| holistic `BatchAttention` 的 backend 参数 | 不擅自增加 |
| backend 名称 | 使用昇腾 provider 名称，不伪装成 FA2/FA3 |
| provider 实现方式 | 内部 SPI；这是昇腾适配结构，不声称上游已有通用 SPI |
| fallback 到 Host reference | 禁止静默 fallback |

检查点 004 才定义 provider SPI 的最小协议；检查点 003 不发现包、不加载运行时、
不调用任何算子。

## 5. 检查点 004：operator provider SPI

框架级 `AttentionOperatorProvider` 与现有低层 `AttentionLauncherProvider`
职责不同：

- operator provider 代表 CANN/torch_npu 或第三方 Python 包，负责无设备初始化的
  availability probe，并声明其拥有的 `AttentionBackendCapabilityProfile`；
- launcher provider 只处理已经选定的 AOT/builtin artifact、ABI、提交与完成事件。

004 冻结三个操作：显式注册、惰性 `probe()`、`capability_profiles()`。注册本身
不得导入包或初始化设备；`discover()` 才收集 probe，并校验 provider id、profile
唯一归属以及禁止 provider 声明 Host reference。缺包不是异常，也不能假装支持；
probe 必须保留明确拒绝原因。

预留 provider id 为 `cann` 与 `flash_attention_npu`。它们只是稳定身份，不代表
当前已经接入对应包。004 仍然没有 prepare、run、fallback 或自动选择行为。

## 6. 检查点 005：dispatch 到 provider 的绑定

provider 不能绕过已有的 plan/capability/evidence/kernel/ABI 选择链。顺序冻结为：

```text
AttentionFrameworkPlan
  -> select_attention_dispatch(...)
  -> AttentionDispatchReceipt
  -> bind_attention_operator_provider(...)
  -> AttentionOperatorProviderSelection
```

`AttentionDispatchReceipt` 仍是算子实现选择的权威记录；operator provider 只根据
receipt 中已经选定的 profile id、profile fingerprint 和 backend 解析唯一所有者。
`provider="auto"` 接受唯一可用所有者；显式 provider policy 必须命中该所有者，
否则立即失败。package 不可用、profile 过期、backend 漂移或没有发现 provider
都会进入结构化候选报告，不允许回退到 Host reference。

005 仍不调用 provider 的 prepare/run。下一步才把 selection 与 provider-owned
opaque plan 保存进 wrapper active plan。

## 7. 检查点 006：wrapper-owned active plan

provider 执行侧通过独立的 `AttentionOperatorPlanFactory.prepare()` 生成
`AttentionPreparedOperatorPlan`。它包含可验证的 plan/selection identity、稳定的
implementation id、opaque plan token 和不参与序列化的 provider 私有状态。

`AttentionOperatorPlanSession.plan()` 先在临时 candidate 中完成 prepare 与整条
identity 校验，全部成功后才替换 active plan。因此：

- 公共 `plan()` 语义仍返回 `None`；
- `run()` 将只读取 session 内的 active plan；
- prepare 抛错或返回陈旧身份时，旧 active plan 保持不变；
- opaque state 不暴露为用户必须传递的 plan handle；
- 006 只冻结 plan 生命周期，尚未定义或调用 provider `run()`。

## 8. 对 NVIDIA FlashInfer 实现方式的判断

本项目需要保持一致的不是 CUDA 代码组织，而是以下可观察语义：

- wrapper 在 `plan()` 中选择并缓存 module/implementation 与 `plan_info`；
- `plan()` 返回 `None`，用户不保存也不向 `run()` 回传 plan handle；
- `run()` 消费 wrapper 内部的 module 与 `plan_info`，热路径不重新选择实现；
- `backend="auto"` 是 wrapper 的 plan-time policy，不是 run-time 参数；
- holistic `attention.BatchAttention` 当前构造函数没有 `backend` 参数。

上游源码也说明其内部不是一个通用第三方 provider registry：常规 selector
主要返回 FA2/FA3，部分 wrapper 再对 cuDNN、TRT-LLM、CuTe DSL、CUTLASS/FHMA
等实现做专用分支。因而本项目的 CANN/flash-attention-npu provider SPI 是昇腾
内部扩展；它必须隐藏在相同的公共 plan/run 生命周期后面。

上游核对来源：

- [BatchPrefillWithPagedKVCacheWrapper](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/prefill.py)
- [holistic BatchAttention](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/attention/_core.py)
- [determine_attention_backend](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/utils.py)

## 9. FlashInfer plan 字段到昇腾 provider 的映射边界

下表只描述参数 lowering，不代表 provider 已经通过能力验证。实际可选范围必须由
带版本与环境证据的 capability profile 决定。

| FlashInfer 语义 | CANN/torch_npu 候选 lowering | flash-attention-npu 候选 lowering |
| --- | --- | --- |
| prefill/decode/mixed mode | `npu_fused_infer_attention_score` 由 query 的 S=1/非1进入增量/全量分支 | plan 时在 `flash_attn_func`、`flash_attn_varlen_func`、`flash_attn_with_kvcache` 间确定一个 operation |
| Q/K/V 或 paged KV cache | `query`/`key`/`value`，paged 场景附加 `block_table` | dense 使用 q/k/v；KV cache 使用 q/k_cache/v_cache；paged 参数 v2 为 `block_table`、v3 为 `page_table` |
| batch 变长元数据 | eager torch_npu 的 `actual_seq_lengths`/`actual_seq_lengths_kv` 是 Host IntArray | varlen 使用 `cu_seqlens_q/k`；KV cache 使用 `cache_seqlens` |
| head 数 | `num_heads`、`num_key_value_heads` | 从 tensor shape 推导，Q heads 必须可被 KV heads 整除 |
| softmax scale | 文档版本使用 `scale_value`，示例存在 `scale` 命名；adapter 必须按探测到的包版本绑定 | `softmax_scale` |
| causal/window/custom mask | `atten_mask`、`pre_tokens`、`next_tokens`、`sparse_mode` 组合，不能机械一一映射 | 公开接口有 `causal`、`window_size`；没有等价的任意 custom mask 参数 |
| layout | `input_layout`，paged KV 另有专门排布约束 | 公开接口以 tensor shape/layout 约定表达，不接收 FlashInfer `NHD/HND` 字符串 |
| page size | `block_size`；所核对文档版本要求 128–512 且为 128 倍数 | v2 README 要求 page block size 为 256 倍数；v3 声明可为任意值 |
| quantization | 显式 dequant/quant/antiquant scale/offset，key/value 支持 INT8 与 INT4(INT32 storage) | v3 签名出现 descale 参数，但当前 README 能力表仍标记 FP8 不支持；验证前不得发布量化 profile |
| output/LSE | 返回 `(output, softmax_lse)`，由 `softmax_lse_flag` 控制 | 不同函数用 `return_attn_probs` 或 `return_softmax_lse` 控制，adapter 统一成 wrapper 返回语义 |

这里尤其不能把 TorchAir 图模式的 Device tensor sequence lengths 与 eager
torch_npu 的 Host IntArray 混为同一 ABI；未来如支持 TorchAir，应发布独立 provider
operation/profile。

## 10. 检查点 007：非执行 run lowering

`AttentionOperatorRunRequest` 在内部镜像上游 holistic `BatchAttention.run` 的参数，
并绑定 active plan fingerprint 与 generation。`AttentionOperatorRunAdapter.lower()`
只能读取已经发布的 active plan，将其翻译为可检查的
`AttentionLoweredOperatorCall`：

- operation id 必须等于 plan 阶段冻结的 `implementation_id`；
- provider、active plan fingerprint 与 generation 必须完全匹配；
- 用户传入的非默认 run 字段必须被显式消费或验证，不能静默丢弃；
- plan-time block table、sequence lengths 等可从 provider opaque state 进入调用描述；
- lowering 不导入包、不执行算子、不做二次 dispatch。

因此 007 验证的是框架调用边界，不是算子正确性。下一检查点才会把这个内部请求
接入公共 wrapper 的 provider 路径；真实 package import/call 仍需单独授权。

## 11. 检查点 008：wrapper-owned operator session

`AttentionOperatorWrapperSession` 把 plan factory、run adapter、active plan 与
request 构造全部收进 wrapper 内部。它的 `run` 参数名和 holistic
`BatchAttention.run` 完全一致，不出现 plan、request、provider 或 adapter 参数。

内部 `plan()` 会先在临时 session 完成 provider prepare；只有 factory 与 adapter
身份一致且 active plan 全部校验通过后，才原子替换当前运行状态。prepare 失败或
adapter 身份错误时，旧 plan 与旧 adapter 继续服务后续 run。008 的 run 仍然只
返回非执行 call description，以便在没有 NPU 算子的前提下验证完整框架生命周期。

## 12. 检查点 009：版本化 operation catalog

仅保存 `torch_npu.npu_fused_infer_attention_score` 之类的 callable 名称不够：
CANN 的旧版 FIA、FIA v2 以及 flash-attention-npu v2/v3/v4 的参数名、page table
名称、LSE 返回能力和量化参数都不同。009 增加打包的
`attention_operator_operations.json`，每个 operation id 都包含 API version，例如：

- `cann.torch_npu.npu_fused_infer_attention_score@6.0.0`；
- `cann.torch_npu.npu_fused_infer_attention_score_v2@7.3.0`；
- `flash_attention_npu.flash_attn_with_kvcache@v2` / `@v3`；
- `flash_attention_npu.flash_attn_varlen_func@v2` / `@v3` / `@v4`。

每个 `AttentionOperatorOperationSpec` 冻结 callable path、候选 Attention mode、
位置参数、关键字参数、可能返回值、可变参数、Host sequence 参数、量化参数、
page table 参数、LSE 控制参数与上游来源。`bind_attention_operator_operation()`
要求 prepared plan 的 `implementation_id` 精确命中带版本 operation id，并校验
provider 与 mode，拒绝只给函数名的模糊绑定。

该 catalog 只是已发布 API 的语法快照，不是 capability evidence。即使某签名包含
量化参数，也不能据此声明当前 SoC/CANN/package 组合支持相应 plan；生产 dispatch
仍必须通过现有 capability profile、accuracy evidence、kernel/ABI receipt 链。

## 13. 检查点 010：catalog-bound lowering

`AttentionOperatorWrapperSession` 构造时接收内部 operation catalog。plan 阶段先完成
provider prepare，再将 `implementation_id` 绑定到 exact operation fingerprint；
两者全部成功后才同时发布 active plan、run adapter 与 operation binding。

run 阶段在生成非执行 call description 后继续验证：

- 位置参数名称与顺序必须和该 API variant 完全一致；
- keyword 与 return name 必须属于 catalog 声明；
- cache append 等 API 的可变参数必须被明确标记；
- 当调用参数启用 LSE 时，返回契约必须包含 `softmax_lse`；
- call 的 provider、operation id 和 active plan fingerprint 必须匹配 binding。

catalog binding 失败发生在原子发布之前，因此模糊 operation id 或签名漂移的重新
plan 不会破坏上一份可运行状态。010 仍不解析 Python callable，也不调用算子。

## 14. 检查点 011：callable observation 与签名绑定

真实 package availability/version 与 callable 解析必须分层：

1. `AttentionOperatorProviderProbe` 报告 package metadata，注册阶段不导入包；
2. provider-owned inspector 在允许的 plan 阶段提交
   `AttentionObservedOperatorCallable`；
3. 框架只将 observation 与 versioned operation spec 做纯比较。

`AttentionObservedCallableSignature` 记录 required positional 参数、其余 keyword
参数以及 `*args`/`**kwargs`。精确绑定要求参数名称与顺序完全一致，并拒绝用可变
参数掩盖 API 漂移。`AttentionOperatorCallableBinding` 同时冻结 provider probe、
package version、API variant、callable path、operation fingerprint 与 signature
fingerprint。

011 只提供 provider inspector protocol 和 fake Python callable 的签名归一化工具，
没有默认的 torch_npu/flash-attention-npu importer。测试函数体如果被执行会立即
失败，从而证明当前检查点仅观察签名、不调用算子。

## 15. 检查点 012：原子 runtime authority

`AttentionOperatorPlanFactory` 现在必须在 prepare 前声明带版本 `operation_id`，并且
返回的 `AttentionPreparedOperatorPlan.implementation_id` 必须保持一致。factory
不能在 prepare 内临时改选其他 API。

`AttentionOperatorRuntimeBinding` 关闭以下身份链：

```text
active plan
  -> provider selection + provider probe fingerprint
  -> versioned operation binding + operation fingerprint
  -> callable binding + package/signature observation fingerprint
```

wrapper 只有在 callable binding 与 selection、catalog operation、mode 全部一致后
才调用 prepare；随后再生成 active plan、operation binding 与 runtime binding，并
一次性发布。陈旧 callable 在 prepare 前就被拒绝，旧 active runtime 保持可用。
run 同时核验 runtime binding 与 active plan fingerprint，仍不解析或执行 callable。

## 16. 检查点 013：首批真实 API 语义的纯框架 adapter

013 不再使用任意 fake keyword 模拟 provider，而是针对两个已冻结 operation id
生成确定的 prepared state 与 call description：

- `cann.torch_npu.npu_fused_infer_attention_score_v2@7.3.0`；
- `flash_attention_npu.flash_attn_with_kvcache@v3`。

共同的 plan-time 转换为：

1. 从 FlashInfer CSR `indptr + indices` 构造二维
   `[batch, max_pages_per_request]` 页表；有效 page id 原序保留，未使用单元以 0
   确定性填充，并由逐请求 KV 长度限定为不可读区域；
2. 页表、序列长度和 provider 静态参数保存在 opaque prepared state，`run()`
   不重新解析 CSR，也不重新选择 operation；
3. 需要 device tensor 的 plan 数据先保存为 `AttentionOperatorTensorPlan`；它是
   可检查的物化配方，不是假 Tensor，也不会导入 torch_npu/flash_attn；
4. lowering 继续返回 `AttentionLoweredOperatorCall`，不执行 callable。

CANN v2 的 TND+PageAttention 映射为：

| FlashInfer plan 数据 | CANN v2 参数 |
| --- | --- |
| `qo_indptr` | `actual_seq_qlen = qo_indptr[1:]`，即 TND 累计 query 长度 |
| paged KV token lengths | `actual_seq_kvlen`，PageAttention 下为逐请求 KV token 数 |
| CSR page table | 二维 int32 `block_table` |
| `page_size` | `block_size` |
| heads / scale | `num_query_heads`、`num_key_value_heads`、`softmax_scale` |
| causal | symbolic 2048x2048 right-down causal mask + `sparse_mode=3` |
| flat Q | `input_layout="TND"` |

该 7.3.0 binding 只接受文档可直接承接的 HND 分页 KV、float16/bfloat16、
128/192 head dimension 与 128–512（128 倍数）block size。滑窗、自定义 mask、
输出 buffer 和未经验证的分页量化均在 plan/run 边界失败。

flash-attention-npu v3 映射为：

| FlashInfer plan 数据 | v3 参数 |
| --- | --- |
| `qo_indptr` | int32 `cu_seqlens_q`（保留首个 0）与 `max_seqlen_q` |
| paged KV token lengths | int32 `cache_seqlens` |
| CSR page table | 二维 int32 `page_table` |
| causal/window/scale | `causal`、`window_size`、`softmax_scale` |
| separate K/V cache | 位置参数 `k_cache`、`v_cache` |

v3 只接受 README 明确描述的 NHD page pool 与 float16/bfloat16。虽然函数签名
包含 `q_descale/k_descale/v_descale`，同一 README 的能力表仍标记 FP8 不支持，
因此 013 不发布量化 binding，也不把 FlashInfer `k_scale/v_scale` 静默传入。

官方语义来源：

- [CANN/torch_npu FIA v2 7.3.0](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-npu_fused_infer_attention_score_v2.md)
- [flash-attention-npu README](https://github.com/MinghuasLab/flash-attention-npu/blob/main/README.md)

## 17. 检查点 014：公共 BatchAttention 自动 runtime

013 验证了内部 adapter，但用户仍可能被迫面对 factory、run adapter、receipt 与
callable binding。014 将这些对象全部收进公共 `BatchAttention` 后面，构造、plan
和 run 参数仍保持上游 holistic 接口：

```python
wrapper = BatchAttention(kv_layout="HND", device="npu:0")
wrapper.plan(qo_indptr, kv_indptr, kv_indices, kv_len_arr, ...)
output, lse = wrapper.run(q, (k_cache, v_cache))
```

内部生命周期变为：

```text
public plan arguments
  -> prepare AttentionFrameworkPlan（尚未发布）
  -> device runtime resolver 自动选择完整 provider candidate
  -> provider prepare + operation/callable/runtime binding
  -> 同时发布 framework plan、operator session 与 executor

public run arguments
  -> 使用已发布 operator session 生成并校验 lowered call
  -> 已绑定 executor 返回公共 output/LSE
```

为了保证原子性，`AttentionFrameworkSession` 增加 `prepare_plan()` 与
`commit_prepared_plan()`：resolver 不可用、签名漂移或 provider prepare 失败发生时，
候选 framework plan 不会覆盖旧 plan。成功 plan 之后，多次 run 只复用当前 executor，
不会重新 resolve。

`AttentionOperatorRuntimeResolverRegistry` 按 device type 路由 package integration；
它不出现在 `BatchAttention` 构造或 plan/run 参数中。当前默认 registry 仍为空，真实
CANN/flash-attention-npu package resolver 与 executor 尚未安装；检查点测试使用 fake
resolver/executor 验证公共返回路径，既不导入包也不执行算子。

CANN v2 TND PageAttention 与 flash-attention-npu v3 的 `cu_seqlens_q` 均能表达
逐请求不同 query 长度，因此 catalog 和两个 adapter 在 014 中扩展到
`batch_mixed_paged`。这使公共 holistic `BatchAttention` 可以保持单 wrapper 的 mixed
prefill/decode 语义，而无需向用户暴露两个 provider wrapper。

## 18. 检查点 015：确定性 implementation auto selection

device resolver 下面增加版本化 `AttentionOperatorRuntimeImplementationRegistry`。
每个 implementation 固定：

- `provider_id`；
- 带 API version 的 `operation_id`；
- 显式整数 `priority`；
- 无副作用的 `rejection_reasons(plan, device)`；
- 仅在被选中后调用的 `resolve(plan, device)`。

自动选择先对所有候选生成 `AttentionOperatorRuntimeResolutionReport`，再取唯一最高
priority 的 accepted candidate。注册顺序不参与结果。如果没有候选、全部拒绝，或
最高优先级存在多个候选，plan 失败并携带完整结构化报告；任何候选都不会被执行性
resolve。低优先级 accepted candidate 可以作为明确配置的后备，但不能掩盖最高层
的歧义。

选中 implementation 返回的 complete runtime 仍要经过 identity 校验：plan
fingerprint、provider selection、factory operation 和 executor operation 必须与候选
一致，不能在 `resolve()` 内偷换 provider/API。随后公共 wrapper 才进入 014 定义的
原子 prepare/publish 流程。

015 的 fake implementation 只验证算法和 authority closure。真实 CANN 与
flash-attention-npu implementation 仍需分别提供：包 availability/version probe、
capability/evidence-bearing dispatch receipt、callable observation 和 executor；本项
没有导入或调用任何外部算子包。

## 19. 检查点 016：plan-time provider tensor materialization

013 的 `AttentionOperatorTensorPlan` 是精确的 tensor 配方，但不能直接传给真实
Python callable。016 在 logical factory/adapter 外增加一对内部组合器：

```text
logical provider factory
  -> AttentionMaterializingPlanFactory
  -> provider tensor materializer（仅 tensor 创建，不调用 Attention）
  -> AttentionMaterializedOperatorPlanState

logical run adapter
  -> AttentionMaterializingRunAdapter
  -> 用 plan-owned tensor 替换 lowered call 中的配方
```

物化项包括：

- CANN：二维 `block_table` 与 symbolic 2048x2048 causal mask；
- flash-attention-npu v3：`cu_seqlens_q`、`cache_seqlens` 与二维 `page_table`。

每个 `AttentionMaterializedOperatorTensor` 绑定 provider、materializer id、原 tensor
plan fingerprint、role 与 device；实际 tensor 作为 opaque value，不进入可序列化身份。
同一 plan role 只能有一个配方，materializer 不能返回原配方冒充 tensor，也不能为
其他 provider 物化。

materialization 发生在候选 provider 已选中之后、wrapper active plan 发布之前。
成功后 run 只复用 plan-owned tensor，不重新创建；任一 tensor 创建失败会中止候选
plan，旧 framework/operator/runtime state 全部保持。016 测试使用 fake tensor
materializer，不导入 torch/torch_npu，也不执行 Attention 算子。

## 20. 检查点 017：受最终 plan 身份保护的 callable execution

provider resolver 可以提交经过 operation catalog 与 Python signature 验证的 callable，
但 resolver 返回时 wrapper 尚未完成 provider prepare，因此候选 executor 不能提前声称
拥有执行授权。017 将授权时点固定为：

```text
resolve candidate（executor 未绑定）
  -> provider prepare
  -> active plan + operation binding + runtime binding
  -> executor.bind_runtime(runtime_binding)
  -> 原子发布 framework/operator/executor 三份状态
```

`AttentionInjectedCallableExecutor` 只接受外部注入的 callable；它不解析模块路径、不导入
package，也不初始化设备。构造时重新观察 callable signature，并要求其 fingerprint 与
callable binding 完全一致。每次执行前继续校验 provider、带版本 operation id、active plan
fingerprint、位置参数顺序、keyword 集合与 return name 集合，随后才进行一次精确 Python
调用。

单返回值保持 callable 原值；多返回值要求 tuple/list 数量与 lowering 的 return names
一致，并统一公开为 tuple。只有 callable 成功返回且返回契约通过后才签发
`AttentionOperatorExecutionReceipt`；签名漂移、陈旧 plan、返回数量漂移或 callable 异常
都不会产生成功收据。callable 自身异常原样传播，便于 provider 保留真实错误语义。

runtime binding 失败仍发生在 `commit_prepared_plan()` 之前，因此重新 plan 失败不会拆散
已有的 framework plan、operator session 与 executable。017 的 5 项增量测试和 418 项
全量测试仅使用注入的 Python fake callable；没有导入或调用 CANN、torch_npu、
flash-attention-npu 或 NPU runtime。

## 21. 检查点 018：惰性外部 package 与 callable resolution

017 从一个已经注入的 callable 开始，尚未定义真实 package integration 如何安全地产生
该对象。018 增加 `AttentionOperatorPackageResolver`，将过程分成两个明确阶段：

```text
metadata stage
  -> 查询 distribution version（不 import package）
  -> 对照 adapter 明确声明的 exact supported versions

callable stage（仅 metadata accepted 后）
  -> 解析 catalog 中的 exact callable path
  -> inspect 实际 signature
  -> provider probe + observation + callable binding
  -> 产生未绑定、未执行的 AttentionInjectedCallableExecutor
```

`AttentionOperatorPackageCompatibility` 的 package version 集合属于 adapter 声明，不能
用 operation 的 API version 猜测，也不能使用宽泛的“版本大于某值”替代已验证组合。
package 可安装、可 import 仍不等于 plan capability：018 不创建 dispatch receipt、不发布
capability profile，也不把 resolver 放进默认 NPU registry。完整执行仍必须依次经过
capability/evidence dispatch、provider selection、provider prepare 和 017 的最终 runtime
binding。

框架提供通用 `ImportlibAttentionOperatorPackageLoader`，但不会默认实例化或注册它。
`package_version()` 只读取 Python distribution metadata；`resolve_callable()` 仅解析受信任
operation catalog 的 module path，并且从不调用 callable。实际 CANN 与
flash-attention-npu integration 后续必须显式提供 compatibility、tensor materializer、
capability evidence 与 runtime implementation。

018 的 9 项增量测试覆盖：构造零副作用、metadata-only explain、缺包、未授权版本、
authority 后 metadata 漂移、非 callable、签名漂移、成功产生未绑定 executor 以及显式
importlib path 解析。全量测试除 Python 标准库解析测试外全部使用 fake loader，没有导入任何 NPU
package、初始化设备或调用 Attention 算子。

## 22. 检查点 019：package runtime implementation 闭环

019 将 018 的未绑定 package candidate 接入 015 的自动 implementation registry、016 的
tensor materialization 与 017 的最终 plan binding。一个完整候选由以下注入组件组成：

- `AttentionOperatorPlanGate`：纯 plan/API admission，只返回拒绝原因；
- `AttentionOperatorPackageResolver`：distribution metadata 与 exact callable binding；
- `AttentionOperatorRuntimeAuthorityResolver`：生成 capability/evidence-bearing dispatch receipt
  与 provider selection；
- logical plan factory/run adapter：provider 参数规划与 lowering；
- tensor materializer：把 plan recipe 建成 provider tensor，但不调用 Attention；
- injected callable executor：在最终 active plan 发布前接受 runtime binding。

固定顺序为：

```text
plan gate + package metadata explain（无 import）
  -> provider probe
  -> capability/evidence authority（仍无 import）
  -> 再次核验 package metadata 未漂移
  -> resolve + inspect exact callable
  -> provider prepare + tensor materialization
  -> active runtime binding
  -> 原子发布 framework plan / operator session / executor
```

`AttentionOperatorRuntimeAuthority` 同时绑定 framework plan、provider probe、operation
fingerprint、dispatch receipt 与 provider selection。authority 错误或陈旧、包版本在授权后
变化、callable 签名漂移、prepare/materialization 失败都会发生在公共状态提交之前。
已经成功发布的旧 runtime 因此可以在失败重规划后继续 run。

019 强制完整 package implementation 提供 tensor materializer，避免把
`AttentionOperatorTensorPlan` 配方误传给真实 callable。它仍不提供任何默认 provider，
也不创建真实 CANN/flash-attention-npu capability profile。6 项增量测试使用 fake package、
authority、materializer 与 callable 验证两次 run 只物化一次、authority 先于 import、拒绝
路径零执行副作用和失败重规划原子性；全量 433 项 Host 测试通过。

## 23. 检查点 020：provider plan gate 与 factory 单一规则源

019 要求每个 package implementation 提供纯 `AttentionOperatorPlanGate`，但 CANN v2 与
flash-attention-npu v3 的实际支持条件此前只存在于 factory `prepare()`。如果 integration
另写一份 gate，自动选择与 prepare 很容易随版本演进产生漂移。

020 将两个 binding 的 plan admission 提取为无副作用解释函数：

- `explain_cann_v2_paged_plan()`；
- `explain_flash_attention_npu_v3_paged_plan()`。

对应的 `CannV2PagedPlanGate` 与 `FlashAttentionNpuV3PagedPlanGate` 直接返回完整原因集合；
两个 factory 在 authority 校验后调用同一个解释函数，并以首个原因失败。因此 gate 用于
auto-selection 时可以展示全部不匹配维度，prepare 又不会出现另一套隐藏条件。

CANN gate 覆盖 paged mode、position/custom mask、soft cap、profiler、未验证 quant、正序列
长度、HND、float16/bfloat16、dtype 一致性、128/192 head dimension、GQA ratio、window、
128–512 page size 和 batch 上限。flash-attention-npu v3 gate 复用共同规则，并增加 NHD、
float16/bfloat16 与 dense dtype 一致性要求；v3 文档允许的 page size 不被擅自收窄。

020 的 5 项增量测试验证 gate protocol、无 package import、确定性多原因报告、量化拒绝，
以及 6 组接受/拒绝计划中 gate 与 factory 结果严格一致。全量 438 项 Host 测试通过；
没有导入 NPU package、初始化设备或调用 Attention 算子。

## 24. 检查点 021：evidence-bearing runtime authority resolver

019 定义了 `AttentionOperatorRuntimeAuthorityResolver` protocol，但测试 authority 仍由
手工伪造 receipt。021 增加 `AttentionEvidenceOperatorRuntimeAuthorityResolver`，直接组合
项目已有的完整授权链：

```text
AttentionFrameworkPlan
  -> capability profile + exact runtime environment
  -> conformance evidence validation
  -> bound KernelDescriptor
  -> artifact + launch ABI + binary ABI provenance
  -> AttentionDispatchReceipt
  -> provider profile ownership binding
  -> AttentionOperatorRuntimeAuthority
```

resolver 构造时固定 exact operation、profiles、descriptors、observed environment、backend
policy、可选 tuning 顺序、numerics policy 与 evidence replay policy。构造阶段立即交叉校验
profile/kernel binding，但不加载 artifact、import package 或访问设备。`authorize()` 只接受
同一 operation fingerprint、可用且同 provider 的 package probe，以及 `npu[:index]` device。

authority 新增 device identity；019 的组合器在 callable import 前和 package resolve 后都
校验该 device、plan、probe 与 operation。环境、evidence、profile、kernel、artifact/ABI、
backend policy 或 provider ownership 任一漂移都会阻止 callable import 或 runtime 发布。
只有 `select_attention_dispatch()` 接受的 descriptor 才能被 tuning 选中。

021 的 7 项增量测试使用既有 synthetic capability/evidence fixtures，验证完整 receipt 可再次
revalidate、环境漂移、backend 排除、provider/operation/device 身份拒绝、tuned kernel gate
和零 package import。全量 445 项 Host 测试通过；这些 synthetic evidence 不构成真实 NPU
能力声明，默认 provider registry 仍为空。

## 25. 检查点 022：声明式 package runtime bootstrap

021 之前，package resolver、evidence authority、plan gate、provider lowering、tensor
materializer 与自动 implementation registry 虽然都已存在，但只能由测试或未来集成代码逐件
手工拼装。022 增加单一 composition root：`AttentionOperatorPackageRuntimeSpec`。

每个 spec 必须显式提交：

- catalog 中的 exact operation id、adapter version 与允许的 exact distribution versions；
- priority、capability profiles、bound kernel descriptors 与 observed runtime environment；
- plan gate、logical plan factory、logical run adapter 与 provider tensor materializer；
- backend/tuning/numerics/evidence replay policy。

`build_attention_operator_package_runtime()` 将一个 spec 组装为完整 candidate；
`build_attention_operator_runtime_resolvers()` 将多个 candidate 放入确定性 implementation
registry，并只对 `npu` device type 安装 resolver。公共 `BatchAttention` 构造器仍没有 backend、
provider 或 plan handle 参数，它从 `build_default_attention_operator_runtime_resolvers()` 获得
内部 registry，因此继续保持 FlashInfer 风格的 `plan()`/`run()` 使用面。

bootstrap 构建阶段不读取 distribution metadata、不 import callable、不加载 artifact、不初始化
设备、不物化 tensor，也不调用算子。`explain()` 才允许 metadata-only probe；选中 candidate 后，
`resolve()` 依次完成 evidence authority 和 callable binding；tensor materialization 仍只发生在
wrapper `plan()`。默认 packaged spec tuple 目前为空，避免在没有真实 environment/evidence/
materializer 的情况下把已安装 package 误当成可运行能力。

022 的 8 项增量测试验证空默认入口、零副作用构建、NPU 路由、metadata-only explain、完整
evidence-before-import resolve、缺包短路、跨 provider 身份拒绝以及重复 operation/version/tuning
声明拒绝。全量 453 项 Host 测试通过；未导入外部 Attention package 或使用 NPU runtime。

## 26. 检查点 023：公共 wrapper 的原子 runtime registry 快照

022 产生 immutable resolver registry，但公共 `BatchAttention` 构造时仍直接读取一个模块变量，
没有正式约束集成更新与既有 wrapper 的隔离关系。023 增加 process-bootstrap 安装边界：

- `attention_operator_runtime_registry_snapshot()` 在锁内捕获 registry 与单调 generation；
- `install_attention_operator_runtime_resolvers()` 只接受空 registry 或唯一 `npu` 路由，并支持
  `expected_generation` compare-and-swap，避免并发 bootstrap 静默覆盖；
- 每个 NPU `BatchAttention` 在构造时捕获一个 snapshot，随后只使用该 immutable registry；
- 后续安装或恢复只影响之后构造的 wrapper，已存在且已规划/未规划的 wrapper 都不换 authority；
- snapshot/install 不执行 resolver、不 probe package、不访问设备。

安装入口属于 library/provider integration control，不进入模型调用面。`BatchAttention` 构造签名
仍只有 `kv_layout` 与 `device`；`plan()` 不暴露 provider，`run()` 不接收 plan。真实 package
integration 将在进程初始化阶段安装已验证的 spec tree，模型使用者仍只调用同一个 wrapper。

023 的 8 项增量测试验证默认快照、未来实例隔离、恢复隔离、stale generation 拒绝、仅 NPU
路由、零 resolver/device side effect、snapshot schema 与公共签名。全量 461 项 Host 测试通过；
默认 registry 仍为空，未导入或调用任何外部 Attention 实现。

## 27. 检查点 024：QuantSpec 到 provider API 参数的闭合绑定

capability profile 能证明某个 backend/kernel 接受 exact `QuantSpec`，operation catalog 能证明
某个 package API 暴露一组 `quant_arguments`，但两者此前没有语义连接。只凭参数名含有
`quant`/`scale` 不能推导它消费的是 KV scale、输出 scale、query scale 还是其他格式。024 增加：

- `AttentionOperatorQuantArgumentBinding`：把逻辑来源（K/V scale、K/V zero-point、可选
  runtime K/V scale）映射到一个 catalog argument；
- `AttentionOperatorQuantizationBinding`：绑定 exact `QuantSpec`、provider、operation、完整参数
  source 集、runtime scale policy 与 quantized KV input contract；
- `validate_attention_operator_quantization_bindings()`：要求 operation 候选 mode 涉及的所有
  capability rule `QuantSpec` 与 API bindings 构成完全相同的集合；
- `AttentionOperatorQuantizationPlanGate`：在 provider 自身 gate 之外再次按 exact QuantSpec
  fingerprint admission，缺失或不同粒度/axis/group/zero-point/packing 的 plan 都拒绝。

对称量化至少必须独立映射 K scale 与 V scale；非对称量化还必须独立映射 K zero-point 与
V zero-point。runtime `k_scale`/`v_scale` 是反量化之外的公共运行时倍率，不能默认混同于 KV
storage scale；只有 policy=`argument` 且存在专属 source binding 才能声明消费，否则维持 reject。

bootstrap 构建时执行上述闭合验证，早于 distribution metadata probe、callable import、device
访问与 operator execution。现有 CANN v2/flash-attention-npu v3 paged gate 仍明确拒绝量化；024
没有为真实 package 声明任何 QuantSpec，只冻结未来启用量化 plan 之前必须满足的框架门禁。

024 的 8 项增量测试验证 canonical binding、对称/非对称来源、runtime scale policy、capability/
catalog exact-set、一处曾可构造的 synthetic 假支持拒绝、非 catalog/跨 operation/重复 binding、
exact plan gate 与 base reason 保留。全量 469 项 Host 测试通过；没有导入或调用 NPU 算子。

## 28. 检查点 025：量化 KV provider run lowering

024 冻结了语义 binding，但尚未把公共 `run(q, kv_cache, ...)` 输入转换为 provider API 的
实际 argument values。025 增加 `AttentionOperatorQuantizedKVInput`，在不改变 public signature
的前提下承载六个 opaque provider tensor/object：K/V storage、K/V scale、可选独立 K/V
zero-point，并携带 exact `QuantSpec`。

`AttentionOperatorQuantizationRunAdapter` 由 bootstrap 自动包裹 provider logical run adapter：

1. dense plan 原样委托；量化 plan 必须收到 `AttentionOperatorQuantizedKVInput`；
2. input QuantSpec fingerprint 必须与 active plan 及选中 binding 完全相同；
3. 先把 KV input 解包为 provider adapter 原有的 `(key_storage, value_storage)`；
4. base lowering 成功后，再按 binding 注入 K/V scale 与可选 zero-point keyword；
5. runtime `k_scale`/`v_scale` policy=`reject` 时显式拒绝，policy=`argument` 时注入专属参数；
6. 注入参数不得覆盖 base lowering 已产生的 provider keyword；
7. 最终 call 继续由 operation binding 校验，整个过程只产生 call description，不执行 callable。

025 的 wrapper 只检查 provider-independent 的存在性与语义身份；tensor metadata 安全边界由
026 补齐。physical layout converter 和 allocator/stream 生命周期仍留给后续检查点。现有真实
provider gate 仍未开启量化能力。

025 的 8 项增量测试验证 symmetric/asymmetric input、storage 解包、独立 scale 注入、普通/漂移
QuantSpec 拒绝、runtime scale reject/argument 两种策略、keyword collision、plan materialization
复用与零 callable execution。全量 477 项 Host 测试通过；未使用 NPU runtime 或算子。

## 29. 检查点 026：量化 provider tensor metadata validation

025 的 opaque tensor 可以被正确解包，但仅凭对象存在不能证明 provider 接收到的实际 storage、
scale 和 zero-point 与 active plan 一致。026 新增可注入的
`AttentionOperatorTensorMetadataInspector`；其唯一能力是把 opaque tensor 读取为既有
`TensorView`，不拥有 package import、tensor allocation、device data access 或 operator call。

bootstrap 对所有量化 binding 强制要求 inspector。runtime 先完成 provider/device 选择，再通过
`AttentionOperatorQuantizationRunAdapterFactory` 把校验 adapter 包在 materialization adapter
之外，因此 run-time 校验可以使用精确设备实例（例如 `npu:0`），同时 provider logical adapter
仍只看到原有 opaque plan state。

校验流程复用既有 `TensorView -> QuantizedTensorView -> KVCacheView` 契约：

1. inspector 只读取 K/V storage、scale 和可选 zero-point 的 metadata；
2. 每个 component 必须是无内部重叠的 contiguous view，且 storage 范围有效；
3. storage dtype/physical shape、scale dtype/shape 和 zero-point int32/shape 必须匹配 exact
   `QuantSpec`；
4. K/V component 必须共享 active runtime 的精确 device，且 component/KV storage 不得别名；
5. paged KV 的实际 page 数从 logical physical storage metadata 推导，并必须覆盖 plan 引用的
   最大 page index；ragged/single KV token 数必须与 plan metadata 完全一致；
6. 所有校验成功后才允许 base provider lowering 与 quant argument 注入。

当前 provider metadata 路径只授权 `physical_layout="logical"`；非 logical layout 必须在后续由
provider 声明并绑定精确 `QuantPhysicalLayoutDescriptor`，不能依靠猜测放行。026 的 8 项增量
测试覆盖合法 lowering、dtype/shape/page capacity、scale、asymmetric zero-point、stride、精确
device、alias、inspector bootstrap/output 门禁。全量 485 项 Host 测试通过；没有导入或调用
NPU package/runtime/operator。
