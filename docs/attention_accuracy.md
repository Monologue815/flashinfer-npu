# Attention 量化准确度与误差归因契约

> 状态：Accuracy contract v1.2  
> 日期：2026-08-21  
> 范围：Host reference 语义与未来 backend 的误差预算；不包含 NPU kernel

## 1. 目的

量化 Attention 对拍存在两种性质不同的误差：

1. 高精度 KV 转换为 INT8/UINT8/INT4 后产生的量化误差；
2. backend 相对已冻结量化语义产生的执行误差。

如果只把设备输出与高精度输出比较，这两种误差会混在一起，既可能把合法量化损失错误
归因给 kernel，也可能让 kernel 误差消耗量化预算。v1 因此固定为两段独立比较：

```text
dense high-precision reference
        |
        | quantization_output / quantization_lse budget
        v
quantized indexed-dequant reference
        |
        | backend_output / backend_lse budget
        v
candidate backend result
```

candidate 缺省为 quantized reference 自身，只用于验证量化 corpus；接入 functional/NPU
backend 后必须传入真实结果。Host reference 不会被生产 dispatch 自动选择。

## 2. 预算与指标

`AttentionAccuracyBudget` 分别保存四组 `AttentionErrorTolerance`：

- quantization output；
- quantization LSE；
- backend output；
- backend LSE。

有限值逐元素使用：

```text
abs(candidate - reference) <= atol + rtol * abs(reference)
```

每个 tensor 独立记录：

- numel、finite pair 数；
- 匹配和不匹配的 non-finite 数；
- tolerance violation 数；
- max absolute error；
- 对称 max relative error；
- finite pair RMSE。

默认预算是严格的 `atol=0, rtol=0`。项目不会提供一个可以被所有量化格式静默共享的
宽松默认值；每个格式和 workload 必须声明自己的预算。

## 3. 非有限值规则

NaN 只与 NaN 匹配，`+inf`/`-inf` 必须符号完全一致。non-finite mismatch 无论预算多大
都视为 violation。因此，有限 scale 在反量化乘法中产生 overflow 时，不能被巨大 atol
掩盖。该规则与 `AttentionNumericsPolicy` 的 row-wise softmax 特殊值语义一致。

候选结果允许位于不同 device，但 output/LSE 的 shape、dtype 和 presence 必须与 reference
一致。device 差异是未来 CPU/NPU 对拍的正常情况，不属于数值误差。

## 4. Trace 配对身份

`evaluate_attention_accuracy()` 只接受严格配对的 dense/quantized trace：

- mode、heads、QK/VO dimension、layout、causal/mask/position/window/soft-cap 相同；
- metadata、query 和 runtime scale 输入相同；
- KV cache 的逻辑容量、layout、structure 和 device 相同；
- 只允许 KV dtype、QuantSpec 和 KV 数值表示不同；
- 左侧必须是 `ReferenceKVData`，右侧必须是 `ReferenceQuantizedKVData`。

这可阻止把不同 workload 的输出误当成量化误差。报告绑定 dense input fingerprint、
quantized input fingerprint、candidate result fingerprint 和 budget fingerprint。

## 5. 版本化证据模型

budget、metrics、report、case 和 corpus 均有严格字段校验。报告中的
`quantization_within_budget`、`backend_within_budget`、`passes` 是派生字段；反序列化时会
重新计算并拒绝被篡改的 verdict。

Accuracy corpus 必须同时表达正向样例和预期拒绝样例。正向样例覆盖精确与有损量化，
预期拒绝样例覆盖非有限 scale、overflow 和结构不匹配。预期拒绝是判定规则，不代表 backend
执行失败，也不能被登记为通过 correctness 的 capability evidence。

具体 case 清单、预期 verdict 和覆盖统计由版本化 corpus 自身维护，文档不复制动态测试
清单或某次 replay 结果。

## 6. Accuracy ↔ dispatch 证据绑定

`AttentionAccuracyDispatchBinding` 是 post-dispatch 记录，不是 kernel 选择权。它把以下
身份固定在一个可序列化 fingerprint 中：

- accuracy corpus/case、dense/quantized input、budget、report 和 candidate result；
- dispatch receipt、plan/admission/workload；
- profile/rule/exact environment/capability evidence；
- kernel/artifact/logical ABI/binary ABI 和 backend。

`bind_attention_accuracy_dispatch()` 在创建 binding 前重新执行 paired trace 与 candidate
对拍，要求量化层和 backend 层都通过，并重新调用 dispatch receipt 的完整
profile/descriptor/environment/corpus validator。以下情况不能创建 binding：

- accuracy case 不属于声明 corpus；
- case 本身是 expected rejection，例如 scale overflow；
- report 不是由当前 case、budget 和 candidate 重新计算得到；
- quantized trace 与 receipt 对应不同 workload；
- profile、runtime tuple、kernel、artifact、ABI、capability evidence 或 workspace 已变化。

binding 的 `validate()` 会从所有原始 authority 重建记录并逐字段比较，不能只检查保存的
SHA-256。当前 SHA-256 是一致性标识，不是签名；`runner` 也是声明字段，不构成可信执行
证明。synthetic profile/kernel/result 只能用于框架拒绝逻辑，绝不登记为 packaged
capability 或真实 Ascend evidence。

## 7. Accuracy ↔ provider execution 绑定

`AttentionAccuracyExecutionBinding` 在 post-dispatch binding 之上继续固定：

- 完整 `AttentionLaunchPacket`、execution identity、dispatch receipt 和 launch lease；
- stream context、runtime id/generation；
- resolved launcher 与 provider probe；
- 成功 submit result、成功 completion result、submission/event id；
- 自动捕获的 provider protocol trace；
- 与 accuracy report 相同的 candidate result fingerprint。

构造器要求 candidate shape/dtype/LSE presence 与 packet 的 output lease 一致；resolved
launcher 必须重新验证 packet；submit/completion 必须指向同一 packet、launcher、probe 和
completion event。protocol trace 必须以 packet 为 subject、使用同一 stream，包含与 submit
和 completion wire record 对应的 evidence fingerprint，并以 `COMPLETED -> RELEASED` 结束。
同步失败、异步失败、runtime quiescence 或未完成路径均不能绑定 passing accuracy result。

`result_origin` 固定为 `runner_declared_post_completion`。当前框架只能证明 runner 声称
candidate 是 completion 后读取的结果，并能证明所有结构身份一致；它无法仅靠 SHA-256
证明 runner 没有伪造 tensor 内容。synthetic provider、地址、artifact 和输出只属于
contract fixture，不构成 Ascend 测量。

验证时需要分别重放两层：

1. `AttentionAccuracyDispatchBinding.validate()` 重建数值、profile、kernel、artifact 与
   dispatch authority；
2. `AttentionAccuracyExecutionBinding.validate()` 重建 packet、provider completion 与
   lifecycle 连接。

后者引用前者 fingerprint，但不会代替前者对原始 authority 的重验证。

## 8. 后续 backend 接线门禁

真实 functional/NPU backend 接入时需要：

1. 用同一 quantized trace 生成 candidate，不能重新量化或修改 runtime scale；
2. backend 只消费 backend budget，不能借用 quantization budget；
3. execution binding 必须同时绑定 provider completion 和 launch packet，并提供可信 runner attestation；
4. 分格式、shape、head mapping 和累加 dtype 制定有证据的预算，不使用全局万能阈值；
5. 性能门禁必须在 correctness 通过后独立执行。

当前实现不声明任何 Ascend dtype 精度、AI Core 累加顺序或 CANN 行为。它只建立未来设备
结果必须满足的可重放框架边界。
