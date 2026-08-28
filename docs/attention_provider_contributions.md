# Attention provider contribution

## 目标

`AttentionOperatorProviderIntegrationContribution` 是外部 Attention provider 自己拥有的
最小完整接入单元。CANN/aclnn 适配模块与 flash-attention-npu 适配模块可以分别构造、版本化
和审核自己的 contribution；部署层再把已接受的 contributions 合并为一个全局
`AttentionOperatorProviderIntegrationBundle`。

contribution 只描述框架接入内容，不实现算子，也不证明外部包或 NPU 当前可用。构造、计算
fingerprint 和合并均为 Host 侧纯元数据操作，不查询 package version、不导入 provider、
不解析 callable、不访问设备，也不执行 Attention。

## 所有权边界

一个 contribution 只能属于一个 `provider_id`，但可以包含该 provider 的一个或多个精确
operation。每个 operation 必须同时贡献：

- `AttentionOperatorOperationSpec`：外部 API 的精确签名和来源；
- `AttentionOperatorPackageRuntimeSpec`：框架适配组件、能力证据和量化映射；
- `AttentionOperatorPlanScoringPolicy`：该 operation 的声明式 plan 评分；
- `AttentionOperatorPackageLoaderRoute`：该 operation 到审核 loader delegate 的精确路由。

四个集合的 `(provider_id, operation_id)` 必须完全相等。缺失、孤立、重复或属于另一
provider 的 identity 都会使 contribution 构造失败。contribution 不允许只提供一个 package
名称后由部署层猜测 callable、策略或 runtime spec。

provider 适配模块拥有上述局部输入；部署层拥有最终 catalog 名称、合并 scoring manifest
身份、bundle 身份、接受哪些 contributions 以及何时原子安装。provider 模块不能直接修改
全局 catalog 或逐项发布 registry。

## 独立审核身份

contribution 构造时会生成内部 scoped catalog 和 scoring manifest，将 policy 一次性绑定到
runtime specs，并生成局部 runtime declarations。随后每个 operation 形成一条不可执行绑定：

```text
(provider_id,
 operation_id,
 operation_fingerprint,
 local_declaration_fingerprint,
 scoring_policy_fingerprint,
 loader_route_fingerprint)
```

这些绑定连同 `contribution_id`、provider、局部 catalog/manifest 身份一起形成 contribution
fingerprint。绑定不保存 loader、callable、plan、tensor 或其他可执行对象，适合进入部署审核
记录和最终 bundle binding。

局部 declaration 的作用是发现适配组件、policy 或 loader route 在审核后发生漂移。它不是
最终安装声明，因为它绑定的是 contribution 自己的 scoped catalog。

## 全局合并

部署层使用
`assemble_attention_operator_provider_integration_contributions()` 合并 contributions：

```python
from flashinfer_npu.attention import (
    AttentionOperatorProviderIntegrationContribution,
    assemble_attention_operator_provider_integration_contributions,
)

cann = AttentionOperatorProviderIntegrationContribution(
    contribution_id="cann.attention.release.v1",
    operations=cann_operations,
    runtime_specs=cann_runtime_specs,
    scoring_policies=cann_scoring_policies,
    package_loader_routes=cann_loader_routes,
)

flash_attention_npu = AttentionOperatorProviderIntegrationContribution(
    contribution_id="flash_attention_npu.attention.release.v1",
    operations=flash_operations,
    runtime_specs=flash_runtime_specs,
    scoring_policies=flash_scoring_policies,
    package_loader_routes=flash_loader_routes,
)

bundle = assemble_attention_operator_provider_integration_contributions(
    bundle_id="deployment.attention.providers.v1",
    catalog_name="deployment-attention-catalog-v1",
    scoring_manifest_id="deployment.attention.scoring.v1",
    contributions=(cann, flash_attention_npu),
    approval_manifest=approved_contributions,
)
```

合并入口先重新验证每个 contribution，包括当前适配组件身份和 delegate loader id/type；然后
规范排序并展开四类输入。最终 catalog 与 scoring manifest 形成后，框架再次绑定 policies，
并针对最终全局 catalog 重新生成 declarations。局部 declarations 永远不会被拼接或安装。

该顺序保证最终 registration 的 catalog fingerprint 指向同一部署 catalog，同时保留每个
provider contribution 的独立审核来源。contribution 输入顺序不影响最终结果；重复
`contribution_id`、operation identity 重叠或全局集合冲突会在 package probe 前失败。

## 身份传播

最终 bundle 保存按 `contribution_id` 排序的 contribution bindings，并要求它们覆盖每个
registration identity 且恰好覆盖一次。所有 contribution fingerprints 都进入 bundle 的
canonical representation，因此会传递到：

- provider integration bundle fingerprint；
- 原子 registry snapshot 中的 bundle binding；
- `plan_selection` 的 bundle identity；
- 成功 run receipt 的 bundle identity；
- 离线 plan scoring audit 的 bundle identity。

修改 contribution 的 id、operation、适配声明、policy 或 loader route 都会改变 contribution
fingerprint，并进一步改变最终 bundle fingerprint。运行接口不增加 contribution 参数；模型
使用者仍只调用 wrapper 的 `plan()` 和 `run()`。

## 与直接 bundle assembly 的关系

`assemble_attention_operator_provider_integration_bundle()` 继续作为完整四集合的低层组装入口，
适合单一部署模块或框架定向组合。多个独立 provider 模块共同进入同一部署时，优先使用
contribution 入口，使 provider 所有权、部署所有权和最终 provenance 保持清晰。

生产部署还应使用有界、不可执行的 approval manifest 固定本次允许的完整 contribution
bindings；缺失、多余或漂移项都会在组装前失败。详见
[Attention provider contribution approval manifest](attention_provider_contribution_manifest.md)。
独立 adapter factory 的显式注入、零自动发现和物化顺序见
[Attention provider contribution source registry](attention_provider_contribution_sources.md)。

最终 bundle 的原子安装与运行身份链见
[Attention provider 集成包](attention_provider_integration_bundle.md)，固定的全局派生顺序见
[Attention provider bundle assembly](attention_provider_bundle_assembly.md)，多 package 委托
边界见 [Attention package loader 路由](attention_package_loader_routing.md)。
