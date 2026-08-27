# Attention provider 集成包

## 目标

真实的 CANN/aclnn 或 flash-attention-npu 接入不能由若干彼此独立的启动参数拼装。
FlashInfer-NPU 使用 `AttentionOperatorProviderIntegrationBundle` 作为生产接入的完整审核
单位，并通过 `install_attention_operator_provider_integration_bundle()` 一次性安装。

该集成包只管理 Attention 框架配置，不实现算子，也不把 provider 句柄暴露给模型代码。
模型侧仍然只创建 Attention wrapper、提交 `plan()`，随后调用 `run()`；provider 与具体
operation 由框架自动选择。

## 完整性边界

一个集成包必须同时提供四类对象：

1. `AttentionOperatorOperationCatalog`：允许被路由的精确外部 API 签名；
2. `AttentionDeclaredOperatorPackageRuntimeSpec` 集合：已经审核的 runtime spec 与声明；
3. `AttentionOperatorPlanScoringManifest`：每个 operation 的声明式选择策略；
4. `AttentionOperatorPackageLoader`：在 plan 阶段观察包版本并解析已授权 callable 的边界。

catalog、registrations 和 scoring manifest 的 `(provider_id, operation_id)` 集合必须完全
相等。缺失、孤立或重复 identity 都会使集成包构造失败。这里要求使用本次部署的 scoped
catalog，而不是把与本次安装无关的 operation 留在一个宽泛 catalog 中。

每个 registration 必须先将同一 scoring manifest 的 policy 绑定到 runtime spec，再生成
declaration。安装入口不会把 policy 临时写入一个尚未审核的 spec。这样 declaration
fingerprint 覆盖最终的 plan scorer 类型与 policy fingerprint。

## Loader 身份

runtime declaration 描述一个 operation 的适配器与受支持 package version；package
loader 则是多个 operation 共享的解析边界。集成包单独把下面两个值纳入 bundle
fingerprint：

- 非空且不含空白的 `loader_id`；
- loader 实例的稳定、模块限定类型名称。

局部类、lambda 或匿名动态类型不能成为生产 loader 身份。bundle 构造时会冻结 loader
的 id/type，安装前再次核对；两者之间发生身份漂移必须在 package probe 和 registry 发布
之前失败。安装进入 registry 的 identity-bound loader 还会在每次 package metadata 或
callable observation 前复核同一身份，因此安装后的漂移也不能接触外部包。loader 的其他
可变内部状态不进入 fingerprint；允许的 package versions 已由每个 runtime declaration
固定，而实际观察到的版本与 callable signature 会在 plan-time resolution report 中记录
并继续接受独立校验。

当同一 bundle 中的 CANN 与 flash-attention-npu operation 需要不同 delegate 时，使用
`AttentionOperatorRoutedPackageLoader` 组成一个精确、可审计的 bundle loader。它的
loader id 内含完整 route fingerprint，且每条 route 与 scoped catalog operation 一一对应；
具体约束见 [Attention package loader 路由](attention_package_loader_routing.md)。

## 构建与安装

示意流程如下：

```python
from flashinfer_npu.attention import (
    AttentionDeclaredOperatorPackageRuntimeSpec,
    AttentionOperatorProviderIntegrationBundle,
    bind_attention_operator_plan_scoring_manifest,
    describe_attention_operator_package_runtime,
    install_attention_operator_provider_integration_bundle,
)

bound_specs = bind_attention_operator_plan_scoring_manifest(
    runtime_specs,
    scoring_manifest,
)
registrations = tuple(
    AttentionDeclaredOperatorPackageRuntimeSpec(
        declaration=describe_attention_operator_package_runtime(
            spec,
            operation_catalog=operation_catalog,
        ),
        runtime_spec=spec,
    )
    for spec in bound_specs
)

bundle = AttentionOperatorProviderIntegrationBundle(
    bundle_id="deployment.attention.providers.v1",
    operation_catalog=operation_catalog,
    registrations=registrations,
    scoring_manifest=scoring_manifest,
    package_loader=package_loader,
)

installed = install_attention_operator_provider_integration_bundle(
    bundle,
    expected_generation=current_generation,
)
```

构建 bundle、计算 fingerprint、构建 resolver 和发布 registry 都是 Host metadata 操作。
这些步骤不得查询 package version、导入 CANN/flash-attention-npu、解析 callable、探测 NPU
或执行算子。外部包观察仍然推迟到用户提交完整 plan 之后。

## 原子发布与快照

安装入口先完成整包校验和 resolver 构造，再比较 `expected_generation`，最后在一个锁保护
的提交中同时发布：

- resolver registry；
- scoped operation catalog；
- 全部 runtime declaration bindings；
- scoring manifest binding；
- provider integration bundle binding；
- 新 registry generation。

bundle binding 是非执行数据，只包含 bundle、catalog、manifest、loader 和 registration
的身份与 fingerprint。`attention_operator_runtime_registry_snapshot()` 会重新检查这些身份
是否与同一快照中的 catalog、manifest 和 declarations 一致。任何一项漂移都不能构造有效
快照。

每个新建 NPU wrapper 捕获一个不可变快照；以后安装的新 bundle 不会改变已有 wrapper 的
权限。legacy 或合成 registry 安装会明确清除 bundle binding，不能伪装成经过完整 bundle
审核的生产集成。

## Plan、run 与审计证据链

wrapper 从快照创建内部 runtime 时会传入同一 bundle binding。runtime 构造阶段先核对它
与 catalog、全部 declaration bindings 和 scoring manifest binding 完全一致；`plan()`
选出 operation 后，再确认该 operation 的 declaration fingerprint 属于 bundle 的精确
registration set，之后才提交 active plan。

成功 planning 后，只读 `plan_selection` 发布 bundle id/fingerprint。它们必须成对出现，
并且只能与完整的 runtime declaration、scoring manifest/policy 和 resolution evidence
同时出现。reference、legacy 和未使用 bundle 的兼容路径保持这两个可选字段为空。

只有 provider 执行与 completion validation 都成功，原子 run receipt 才复制相同的 bundle
id/fingerprint；执行失败或结果验收失败不发布回执。纯 Host 的
`verify_attention_plan_scoring_chain()` 进一步要求 snapshot、selection 与可选 run receipt
中的 bundle 身份完全一致，并把这两个字段写入不可执行的审计报告。这样 bundle 安装、
自动 plan 选择和实际成功调用形成一条可复核身份链，而不把 bundle 变成 `run()` 参数。

## Plan-time 与 run-time 边界

bundle 安装成功并不证明外部包已经安装、当前环境可用或算子结果正确。`plan()` 仍需按
顺序完成 package version observation、callable signature、capability/evidence、资源、
评分唯一胜者和 active-plan 身份校验。`run()` 只消费被冻结的 active plan，并继续执行
tensor、量化参数、alias、caller buffer 和 completion 校验；失败后不会隐式换 provider。

因此 bundle 回答的是“这一代 registry 是否由一组完整且一致的审核配置产生”，而不是
跳过既有的运行时证据门禁。

## 不属于集成包的内容

- NPU kernel、Ascend C 源码或从其他仓库复制的算子；
- 模型侧 provider、operation、kernel id 或 callable 参数；
- 自动联网安装 provider package；
- 在线 tuning、运行时失败回退或未经重新 `plan()` 的 provider 切换；
- 声称某个尚未经过真实环境证据验证的能力已经可用。

外部 provider 的完整适配流程见
[Attention 外部 provider 接入指南](attention_provider_onboarding.md)，评分与选择证据见
[Attention plan scoring policy](attention_plan_scoring_policy.md) 和
[Attention plan scoring audit](attention_plan_scoring_audit.md)。
