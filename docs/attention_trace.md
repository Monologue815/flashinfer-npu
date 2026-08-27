# Attention Conformance Trace 与 Replay 设计

> 状态：Host trace schema v1  
> 日期：2026-08-05  
> 范围：小型 correctness case 的跨进程重放，不是生产请求数据采集

## 1. 两类 trace 不混用

项目需要区分：

1. **Conformance trace**：包含微型 Q/K/V、mask、scale 和期望 output/LSE，用于
   correctness 回归、framework replay 和后续 backend 对拍。当前 `AttentionTrace` 属于此类。
2. **Workload trace**：只包含 shape、dtype、layout、metadata bucket、feature、环境和
   dispatch 结果，不记录用户 tensor 数据，用于 benchmark、autotune 和线上诊断。该格式后续单独定义。

Conformance trace 不能直接用于采集真实推理流量；workload trace 也不能替代数值 oracle。

## 2. v1 JSON 结构

```text
schema_version: 1
kind: attention_conformance
plan_spec: AttentionPlanSpec 的完整版本化字段
metadata: single / paged / paged_prefill / ragged / mixed_paged
inputs:
  q: ReferenceTensor
  kv_data: dense 或 quantized KV
  custom_mask_data / alibi_slopes
  q_scale / k_scale / v_scale
  logits_soft_cap / return_lse
expected:
  output: ReferenceTensor
  lse: ReferenceTensor 或 null
```

Quantized KV 记录：

- 逻辑 cache spec 和完整 `QuantSpec`；
- K/V 各自的逻辑 shape、packed/unpacked storage、scale 和 zero-point；
- K/V 共用的量化配置，但不共享 scale/zero 数值。

普通 dense KV 禁止携带非空 `quant_spec`，防止 trace 表面声明量化、实际按整数浮点值直接计算。

## 3. JSON 与非有限数

Python `json` 默认可能输出非标准裸 `NaN`/`Infinity`。v1 禁止这种形式，统一使用：

```json
{"nonfinite":"nan"}
{"nonfinite":"+inf"}
{"nonfinite":"-inf"}
```

这允许全 mask 或空 KV request 的 `LSE=-inf` 被标准 JSON parser 读取。Codec 与 Host
numerics v1 已能保存、重放 NaN/Inf case；真实设备 frontend/backend 是否通过仍属于逐 backend
capability evidence，不能从 codec 能力推导设备支持。

Corpus 的 canonical JSON 会把有限浮点规范化为 15 位有效数字，并把 `-0.0` 统一为
`0.0`。Host oracle 的超越函数可能因 Python/libm 版本在 double 的最后一两个 bit 上不同；
这些低于 correctness 容差的差异不应改变 corpus fingerprint。15 位规则同时保留远高于
当前数值门禁所需的精度。单条 `AttentionTrace` 的输入身份和完整身份合同不因此改写；
该规范化只属于 corpus JSON/整体 corpus fingerprint 边界。

## 4. Capture、身份与验证

```python
trace = AttentionTrace.capture(
    spec=spec,
    metadata=metadata,
    q=q,
    kv_data=kv_data,
)

payload = trace.to_json(indent=2)
restored = AttentionTrace.from_json(payload)
result = restored.replay(atol=1e-6, rtol=1e-6)
```

- `input_fingerprint` 覆盖 plan、metadata 和所有输入，不覆盖期望结果；oracle 更新不会改变 case 身份。
- `fingerprint` 覆盖完整 trace；任何期望 output/LSE 变化都会改变它。
- replay 先重建 plan 并执行全部 plan/run schema 校验，再运行 Host oracle。
- output/LSE 的 shape、dtype、device 必须完全相同；有限数使用 `atol + rtol*abs(expected)`；
  NaN 只匹配 NaN，Inf 必须符号相同。
- mismatch 只报告 tensor 名和 flat index，不打印输入或输出数值。

## 5. CLI

```bash
flashinfer-npu attention-replay case.json
python3 -m flashinfer_npu attention-replay case.json --atol 1e-6 --rtol 1e-6
```

可选 `--no-validate` 只执行而不比较已记录 oracle；它不能作为 correctness gate。
库 API 与 CLI 默认都通过 JSON envelope 拒绝大于 16 MiB 的内容，可用 `--max-bytes`
进一步收紧。完整 depth/node/array/string/tensor/case limits 见
[`attention_json_envelope.md`](attention_json_envelope.md)。

安全边界：

- 只解析严格 JSON；重复 key、裸 NaN/Infinity、超 depth/node/tensor envelope 先失败，未知 version/kind/field 随后失败。
- 不使用 pickle、动态 import、代码执行或用户指定 Python class。
- trace 中的 cache spec、tensor shape 和 data 长度重新经过构造器校验。
- replay 输出只包含 fingerprint、mode、shape、dtype 和验证状态。

## 6. 当前 edge policy

已经冻结的 Host framework 行为：

| 情况 | 行为 |
| --- | --- |
| 全部 key 被 mask | output 为零，LSE 为 `-inf` |
| Paged 空 KV request | output 为零，LSE 为 `-inf` |
| 任一可见 logit 为 NaN | 整行 output/LSE 传播 NaN |
| 一个或多个 logit 为 `+inf` | 只在 `+inf` key 间均分，LSE 为 `+inf` |
| 全部可见 logits 为 `-inf` | output 为零，LSE 为 `-inf` |
| 量化 scale 为 NaN/Inf/非正数 | schema 阶段拒绝 |
| INT4 奇数末维的未使用 nibble 非零 | schema 阶段拒绝 |
| 量化 plan + dense KV | schema 阶段拒绝 |
| oracle output 漂移 | replay 返回 mismatch，不改变 input fingerprint |

metadata 资源 admission 已有 framework v1，但具体生产 backend limits 尚未生成。仍待验证：
真实 Torch/NPU tensor 的特殊值行为、所有请求为空时的生产 backend、parser envelope 与
device stream/alias 生命周期。

## 7. Backend 接入门禁

未来每个 Torch/NPU Attention backend 必须能消费同一 conformance trace 的逻辑内容，并：

1. 用 Host oracle 生成或校验 expected output/LSE。
2. 将量化误差和 kernel 数值误差分开统计。
3. 记录 backend、SoC/CANN capability 和实际 dispatch descriptor，但不改写 input identity。
4. 对 unsupported QuantSpec 明确失败，不自动转换成另一个粒度或物理格式。
5. 将 trace corpus 按 mode/layout/dtype/quant/mask/position encoding 建立覆盖矩阵。

## 8. Corpus 与覆盖矩阵

`AttentionTraceCorpus` 将多个带 oracle 的 trace 组成版本化、可整体 replay 的 JSON。
case id 和 input fingerprint 均必须唯一。覆盖特征从 plan/metadata/KV data 自动推导，
不接受调用者手工声明标签。

Coverage policy 必须覆盖 mode、layout、cache kind、storage、scheme、granularity、causal、
position、mask、MHA/GQA、head dimension relation、空请求、numerical edge、window、
soft-cap、KV runtime scale，以及 single/ragged/paged/mixed 的量化消费。

Policy 还必须支持联合 selector，避免“每个 feature 分别出现过，但关键组合从未出现”造成
虚假覆盖。例如 paged + packed INT4、groupwise decode + GQA + QK/VO 不同维度，以及 mixed
batch 中量化格式、物理页面复用、window、soft-cap 和 runtime scale 的组合，都应作为联合
能力约束表达。

具体 case 清单和 coverage 统计由版本化 corpus 与 policy 生成，不复制到设计文档。

```bash
# 输出内置 corpus JSON
python3 -m flashinfer_npu attention-corpus --pretty

# 计算覆盖并重放所有 case
python3 -m flashinfer_npu attention-coverage --replay --require-complete

# 检查外部 corpus
python3 -m flashinfer_npu attention-coverage corpus.json --require-complete
```
