# Attention 外部 provider 接入指南

本文定义如何把 CANN/aclnn 或 flash-attention-npu 中已经存在的 Attention 算子接到
FlashInfer-NPU。接入层只负责选择、计划、参数转换、调用和结果验收，不复制或重新实现
外部包中的算子。

## 1. 使用者看到的接口

模型代码只使用 FlashInfer 对齐的 Attention API：

- single prefill/decode 使用无状态函数；
- batch prefill/decode、ragged 和 mixed Attention 使用 wrapper 的 `plan()` / `run()`；
- `plan()` 根据完整语义、环境、能力与证据自动选择 provider；
- `run()` 只能执行已经选定并绑定的 plan，不接受 provider 名称、算子句柄或回退列表；
- batch wrapper 可通过只读 `last_run_receipt` 取得最近一次成功 provider 调用的原子证据；
  无状态 single 函数仍保持 tensor 或 `(tensor, lse)` 返回面，不增加 provider 返回对象。

provider 的安装属于进程启动配置，不属于模型调用接口。已经创建的 wrapper 持有不可变
registry generation；之后安装的新集成不会改变它的 plan/run 权限。

## 2. 已连接与默认未启用的边界

仓库已经连接以下框架层能力：

1. FlashInfer 风格的前端参数、wrapper-owned plan/run 生命周期；
2. operation catalog、package 版本观察与 callable signature 校验；
3. capability profile、kernel/artifact descriptor 与 evidence authority；
4. plan gate、优先级选择、不可变 active plan 和失败关闭；
5. tensor/metadata materialization、access policy 与 quantization binding；
6. logical run adapter、caller-buffer adapter 和 provider callable 执行；
7. provider 返回数量、tensor metadata、输入/输出 alias 与 caller buffer 身份校验；
8. execution receipt、completion receipt 与 `AttentionOperatorRunReceipt` 原子发布；
9. JIT plan/artifact/module/callable/runtime-executor 的可选绑定链。

默认 registry 仍为空。仓库没有声明任何可运行的 CANN 或 flash-attention-npu 版本，也不
因检测到某个 Python 包、某个函数名或 NPU 设备就自动声称支持。真实 provider 必须以一
组完整、版本固定且经过证据约束的 runtime spec 显式安装。

## 3. 接入单位

一个接入单位是“一个精确外部 operation 的一个适配版本”，由
`AttentionOperatorPackageRuntimeSpec` 表达。不能用一个宽泛声明覆盖同名但签名或语义
不同的包版本。

runtime spec 至少绑定：

- `operation_id`、adapter version、priority 和允许的 package versions；
- 被观察的 SoC、CANN、驱动、固件、Python、PyTorch、torch_npu 与外部包环境；
- capability profiles、kernel descriptors 和 evidence manifest 身份；
- pure plan gate、logical plan factory 与 logical run adapter；
- tensor materializer、metadata inspector 和 access policy；
- 量化参数映射与非逻辑 physical layout 目录/证据（如适用）；
- JIT resolver/binder 链（仅在该 provider 确实使用 JIT 时）。

`validate_provider_results` 默认必须保持 `True`。只允许无真实 tensor 结果的合成测试
适配器显式关闭；生产 provider 不得用关闭结果校验来绕过返回值、alias 或 caller buffer
契约。

## 4. 接入顺序

### 4.1 固定外部 API

先选择精确 package version 和 operation，保存其官方来源、导入路径、callable 名称与
signature。catalog 必须逐项声明：

- positional/keyword 参数名称和是否必需；
- output、LSE、workspace、stream、metadata 与量化参数归属；
- 返回值数量、顺序和可为空条件；
- caller-provided buffer 是否被原位写入；
- 该 operation 覆盖的 Attention mode、layout、dtype、mask、positional encoding 与
  head mapping。

参数名称相似不是语义证据。适配器不得静默删除参数、改变默认值，或用另一种 Attention
语义近似执行。

### 4.2 固定环境与能力证据

能力声明必须绑定到可复现的环境 tuple，而不是仅绑定 provider 名称。每个 capability
profile 只列出已经证实的组合；每个 kernel descriptor 绑定具体实现、ABI、artifact 或
外部 package operation 的来源。

量化能力还必须证明 storage format、packing、signedness、scale granularity、axis、K/V
独立 scale、zero point、runtime multiplier、group size、padding 和 physical blocking。
非逻辑 physical layout 没有 descriptor、converter provenance 和 evidence 时保持不 eligible。

### 4.3 实现纯计划映射

`plan_gate` 只检查请求和声明能力，给出结构化接受或拒绝原因；不得导入包、分配 device
tensor 或执行算子。`logical_factory` 把选定的 framework plan 转成 provider-owned plan
对象，并将其身份绑定到 active plan。

自动选择发生在 `plan()`：候选必须同时通过 catalog、package version、plan gate、
authority、capability/evidence 和资源约束。priority 只在多个候选都具备完整权限时参与
排序，不能覆盖缺失证据。

### 4.4 实现 tensor 与量化适配

materializer 负责 provider 所需 metadata、辅助表、workspace view 或外部 plan object；
metadata inspector 和 access policy 负责 shape、stride、storage capacity、alignment、
device、stream、writable 与 alias 规则。

每一个 `QuantSpec` semantic source 必须映射到 catalog 中精确的 package argument。
K/V scale、K/V zero point、Q/K/V/output runtime scale 和 per-head scale 彼此独立，不得
因 shape 恰好相同而合并来源。paged/mixed 的公开 `kv_cache` 槽可承载
`AttentionOperatorQuantizedKVInput`；single/ragged 的独立 K/V 槽可承载
`AttentionOperatorQuantizedTensorInput`。这些对象不暴露 provider plan 或 callable。

### 4.5 实现调用与完成校验

logical run adapter 只把已经验证的 lowered call 映射到外部 callable 参数。算子返回后，
框架先检查返回 arity，再检查 output/LSE tensor metadata、caller buffer identity 和允许的
输入 alias；只有完成校验成功，才同时发布 execution/completion 的原子 run receipt。

运行期失败不触发另一个 provider 的隐式 retry。需要换 provider、shape、dtype、layout、
量化方案、workspace 或 JIT specialization 时，必须重新 `plan()`，避免一次调用产生无法
审计的混合语义。

### 4.6 构建并安装 registry

集成模块在进程 bootstrap 阶段构建 resolver，并原子安装：

```python
# 示意代码：所有对象都必须由具体集成模块按本指南提供。
spec = AttentionOperatorPackageRuntimeSpec(
    operation_id=operation_id,
    priority=priority,
    adapter_version=adapter_version,
    supported_package_versions=package_versions,
    profiles=profiles,
    descriptors=descriptors,
    observed_environment=observed_environment,
    plan_gate=plan_gate,
    logical_factory=logical_factory,
    logical_run_adapter=logical_run_adapter,
    tensor_materializer=tensor_materializer,
    tensor_metadata_inspector=tensor_metadata_inspector,
    tensor_access_policy=tensor_access_policy,
    quantization_bindings=quantization_bindings,
)

registry = build_attention_operator_runtime_resolvers(
    (spec,),
    operation_catalog=operation_catalog,
    package_loader=package_loader,
)
install_attention_operator_runtime_resolvers(
    registry,
    operation_catalog=operation_catalog,
    expected_generation=current_generation,
)
```

安装完成后，模型侧代码仍然只调用公开 Attention API。不得为了接入方便而增加
`provider="cann"`、`kernel_id=...` 或 callable 参数。

## 5. CANN 与 flash-attention-npu 的独立性

两者是独立 provider，不能共享未经证明的能力行：

| 项目 | CANN/aclnn adapter | flash-attention-npu adapter |
|---|---|---|
| API 来源 | 精确 CANN/aclnn operation 与版本 | 精确 Python package operation 与版本 |
| callable 解析 | 对应运行时符号/封装入口 | 对应模块、属性和 Python signature |
| plan/metadata | 按该 operation 的 executor/workspace/metadata 规则 | 按该包暴露的 plan 与 tensor 参数规则 |
| stream/error | 按 CANN/torch_npu 的真实所有权声明 | 按该包的真实同步、stream 与异常行为声明 |
| capability/evidence | 仅归属于该 CANN operation | 仅归属于该 package operation |

如果两者最终调用同一底层实现，也必须通过 provenance/evidence 显式证明；不能据此推断
参数语义、数值边界或 physical layout 自动相同。

## 6. 发布门禁

只有在以下条件同时满足后，provider 才能加入默认 runtime declaration 和 support matrix
的 enabled 能力：

1. package/API version、signature 与环境 tuple 已固定；
2. catalog、plan gate 与 rejection reason 覆盖所有声明组合；
3. capability、numerics、kernel/artifact 与 physical layout evidence 完整；
4. tensor、stream、workspace、lifetime、alias 和 caller-buffer 契约已验证；
5. 量化 semantic source 到 package argument 的映射闭合；
6. provider 返回值通过严格 completion validation；
7. plan、runtime binding 与原子 run receipt 的身份一致；
8. 未支持组合稳定地失败关闭，不会回退为近似语义。

未满足其中任一项时，可以保留适配代码和显式集成声明，但默认 registry 必须继续禁用该
operation，文档也不得将其描述为可用能力。
