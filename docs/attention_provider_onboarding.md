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
- exact provider-operation plan scoring policy（由审核后的 manifest 绑定）；
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

生产集成应优先把本次部署的完整 operations、runtime specs、scoring policies 和 loader
routes 交给 `assemble_attention_operator_provider_integration_bundle()`。该入口先形成最终
scoped catalog 和 manifest，再绑定 policies、生成 declarations、组合 routed loader 与
bundle，避免不同 provider 使用局部 catalog 提前生成声明。完整流程见
[Attention provider bundle assembly](attention_provider_bundle_assembly.md)。

当 provider 适配由独立模块交付时，每个模块应把自己的四类完整输入封装为单 provider
`AttentionOperatorProviderIntegrationContribution`；部署层使用
`assemble_attention_operator_provider_integration_contributions()` 合并。provider 模块不拥有
全局 catalog，部署层也不复用局部 declarations。所有权、漂移校验和 provenance 传播见
[Attention provider contribution](attention_provider_contributions.md)。
生产部署应再提供 `AttentionOperatorProviderContributionManifest`，固定已经审核的
contribution bindings；仅检测到模块或外部包不能自动获得路由权限。清单加载、精确匹配与
升级规则见
[Attention provider contribution approval manifest](attention_provider_contribution_manifest.md)。
adapter 模块通过显式 `AttentionOperatorProviderContributionSourceRegistry` 注入，不允许环境
扫描或隐式 entry-point 注册；source-set、factory 身份与生成结果的校验见
[Attention provider contribution source registry](attention_provider_contribution_sources.md)。
如果 adapter factory 位于独立 package，使用精确 source declaration 与注入式 loader；禁止
按环境扫描结果隐式 import。流程见
[Attention provider contribution adapter loading](attention_provider_contribution_loading.md)。

在构建 resolver 前，集成模块先用
`describe_attention_operator_package_runtime()` 从 runtime spec 生成
`AttentionOperatorPackageRuntimeDeclaration`。这是纯数据审计快照，包含 operation/catalog、
包版本、环境、profile、kernel descriptor、量化/physical-layout evidence、结果校验开关和
适配组件类型身份，但不包含 callable、opaque plan 或 tensor。声明的创建、JSON 加载和
`validate_runtime_spec()` 漂移校验均不得查询外部包、导入 provider 或访问 device。

声明通过评审后，集成模块在进程 bootstrap 阶段构建 resolver，并原子安装：

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

# scoring_manifest 先通过 bounded JSON loader；其 identity 集合必须与
# 本次全部 runtime specs 完全一致。
(spec,) = bind_attention_operator_plan_scoring_manifest(
    (spec,),
    scoring_manifest,
)

declaration = describe_attention_operator_package_runtime(
    spec,
    operation_catalog=operation_catalog,
)
declaration.validate_runtime_spec(
    spec,
    operation_catalog=operation_catalog,
)

registration = AttentionDeclaredOperatorPackageRuntimeSpec(
    declaration=declaration,
    runtime_spec=spec,
)

# 多 package/provider 部署可先把每个 operation 的专属 loader 组合为
# AttentionOperatorRoutedPackageLoader；单 package 部署也可以直接使用其 loader。

bundle = AttentionOperatorProviderIntegrationBundle(
    bundle_id=bundle_id,
    operation_catalog=operation_catalog,
    registrations=(registration,),
    scoring_manifest=scoring_manifest,
    package_loader=package_loader,
)
installed = install_attention_operator_provider_integration_bundle(
    bundle,
    expected_generation=current_generation,
)
```

评分 manifest 必须在生成 declaration 之前绑定。这样 declaration 审核的是最终 policy
fingerprint，而不是尚未绑定 scorer 的中间 spec。绑定不会覆盖已有 custom scorer 或不同
policy；缺失 policy、孤立 policy、重复 runtime spec identity 均在 package metadata probe、
callable 解析和 registry 发布之前失败。多 provider 安装时一次传入完整 spec 集合，不能
逐项绑定并逐项发布。

安装完成后，模型侧代码仍然只调用公开 Attention API。不得为了接入方便而增加
`provider="cann"`、`kernel_id=...` 或 callable 参数。

declaration 只证明“集成配置的身份可审计且没有漂移”，不证明包已安装、callable 可解析、
capability 可运行或设备结果正确。后续 metadata probe、authority、callable binding、plan
和 completion gate 仍按既定顺序失败关闭。低层
`build_attention_operator_runtime_resolvers()` 和
`install_declared_attention_operator_runtime_resolvers()` 保留给框架组合、兼容路径与合成
检查；真实 provider bootstrap 使用完整 bundle 与单一 bundle installer，
避免 catalog、声明、评分 manifest 或 loader 身份被分开替换。完整 bundle 契约见
[Attention provider 集成包](attention_provider_integration_bundle.md)；多 loader 的精确路由
见 [Attention package loader 路由](attention_package_loader_routing.md)。

bundle installer 将 resolver、operation catalog 和每个
`(provider_id, operation_id, declaration_fingerprint)` 作为同一 registry generation 原子
发布；传入的 scoring manifest 同时压缩为只含 manifest/policy fingerprints 的非执行
binding，bundle 自身也压缩为包含 catalog、manifest、loader type/id 和 registration
fingerprints 的非执行 binding，并进入同一快照。installer 要求 registration 已在
declaration 生成前绑定同一 manifest，不会在安装时重写未审核的 spec。新建 wrapper
捕获这一不可变快照；成功
`plan()` 后，`plan_selection` 的只读
`runtime_declaration_fingerprint` 指向所选 operation 的审核声明。旧 wrapper 不随之后的
安装变化，legacy/合成 registry 则明确返回 `None`，不能伪装成 declaration-bound 集成。
同一指纹也进入成功 provider 调用的 `AttentionOperatorRunReceipt`；执行或 completion
失败时不发布 receipt，重新 plan 时 active-plan 身份更新但声明身份保持为该 wrapper 捕获
的 registry generation。manifest-bound scorer 返回的 score 必须携带结构化 policy id 与
fingerprint；runtime 在 plan commit 前与 snapshot 中所选 operation 的 binding 比对。成功
`plan_selection` 和 run receipt 都发布同一 manifest/policy 身份，不能把 source/reason 文本
解析成授权依据。bundle 安装路径还会在 runtime 构造和 plan commit 前复核 catalog、
declarations、manifest 与所选 operation；`plan_selection` 和成功 run receipt 同时发布同一
bundle id/fingerprint，离线 scoring audit 再与 wrapper 捕获的 registry snapshot 精确比对。

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
