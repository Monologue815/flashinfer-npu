# Attention provider contribution source registry

## 目标

`AttentionOperatorProviderContributionSourceRegistry` 定义外部 Attention adapter 模块进入
部署组装流程的受控来源边界。它解决的是“由谁构造已审核 contribution”，而不是“运行时
调用哪个算子”。CANN/aclnn adapter 与 flash-attention-npu adapter 可以分别提供 factory，
部署代码显式把需要的 sources 组成一个不可变 registry。

框架不扫描 Python entry points、不遍历已安装包、不按模块名自动 import，也不因为发现
CANN、torch_npu 或 flash-attention-npu 就获得路由权限。source registry 必须由部署方显式
构造，并与 contribution approval manifest 同时使用。

## Factory 契约

每个 source 包含：

- `source_id` 与 `source_version`；
- 唯一 `provider_id` 与预期 `contribution_id`；
- 实现 `AttentionOperatorProviderContributionFactory` 的注入 factory；
- 构造时冻结的 factory id 与稳定、模块限定类型名称。

factory 只有一个职责：通过 `build_contribution()` 返回精确的
`AttentionOperatorProviderIntegrationContribution`。factory 是 adapter 配置工厂，不是外部
provider package loader。它不得查询 package version、导入 provider callable、探测 NPU、
创建 device tensor 或执行算子；这些操作仍属于 plan/run 的后续门禁。

source 在调用 factory 前后都会复核 factory id/type。返回值必须是 contribution，且
provider id 和 contribution id 与 source 声明完全相同；随后还会重新验证 contribution 的
operation、runtime spec、policy、loader route 和组件身份。匿名局部类型、动态 lambda、空白
factory id 或调用过程中身份自修改都会失败关闭。

## Registry 与不可执行 binding

registry 要求 source id、contribution id 和 `(provider_id, contribution_id)` 唯一，并按
`source_id` 规范排序。构造 registry 只读取 factory id/type，不调用
`build_contribution()`。

registry fingerprint 覆盖 registry id 和全部 source bindings。binding 只包含：

```text
(source_id,
 source_version,
 provider_id,
 contribution_id,
 factory_id,
 factory_type)
```

它不包含 factory 对象、函数、module、loader、callable、runtime spec 或 provider handle，
因此可以作为不可执行 provenance 进入最终 bundle snapshot。

## 两阶段精确校验

`materialize(approval_manifest)` 使用两个独立门禁：

1. 在执行任何 factory 之前，registry 的 `(provider_id, contribution_id)` 集合必须与 approval
   manifest 完全一致；缺失或多余 source 会零调用失败。
2. 全部 sources 生成 contributions 后，approval manifest 再逐项核对完整 contribution
   bindings，包括 contribution、operation、局部 declaration、scoring policy 和 loader route
   fingerprints。

第二阶段失败时不会返回部分 contribution 集合。factory 必须保持纯 Host、无外部副作用；
框架不能回滚任意第三方 factory 自己产生的副作用，因此生产 adapter factory 禁止在此阶段
观察或修改外部运行环境。

## 从 sources 组装 bundle

部署层使用统一入口：

```python
from flashinfer_npu.attention import (
    AttentionOperatorProviderContributionSource,
    AttentionOperatorProviderContributionSourceRegistry,
    assemble_attention_operator_provider_integration_sources,
)

cann_source = AttentionOperatorProviderContributionSource(
    source_id="cann.attention.adapter-source.v1",
    source_version="1.0.0",
    provider_id="cann",
    contribution_id="cann.attention.release.v1",
    factory=cann_contribution_factory,
)

flash_source = AttentionOperatorProviderContributionSource(
    source_id="flash_attention_npu.attention.adapter-source.v1",
    source_version="1.0.0",
    provider_id="flash_attention_npu",
    contribution_id="flash_attention_npu.attention.release.v1",
    factory=flash_contribution_factory,
)

sources = AttentionOperatorProviderContributionSourceRegistry(
    registry_id="deployment.attention.adapter-sources.v1",
    sources=(cann_source, flash_source),
)

bundle = assemble_attention_operator_provider_integration_sources(
    bundle_id="deployment.attention.providers.v1",
    catalog_name="deployment-attention-catalog-v1",
    scoring_manifest_id="deployment.attention.scoring.v1",
    source_registry=sources,
    approval_manifest=approved_contributions,
)
```

该入口严格执行：source-set 校验、factory 物化、approval manifest 校验、全局 catalog/scoring
manifest 构造、最终 declarations 再生成、routed loader 组合和 bundle 构造。任一步失败都
不会发布 registry generation。

## 身份传播与运行边界

最终 bundle 和 bundle binding 保存 source registry binding，并要求它覆盖每个 provider
contribution 恰好一次，同时要求 approval manifest 存在。source registry id、source
version 或 factory id/type 变化会改变 registry fingerprint，继而改变 bundle fingerprint。
该身份经 registry snapshot、plan selection、成功 run receipt 和 scoring audit 传播；模型侧
`plan()` / `run()` 不新增 source 或 factory 参数。

source factory 构造 contribution 并不调用实际算子。最终 routed package loader 仍在用户提交
完整 plan 后才观察被选候选的 package version 与 callable；capability/evidence、量化映射、
workspace、tensor 和 completion validation 继续按原有顺序执行。运行失败不会返回 source
registry 重新生成或隐式切换 provider。

contribution 内容见 [Attention provider contribution](attention_provider_contributions.md)，
部署允许清单见
[Attention provider contribution approval manifest](attention_provider_contribution_manifest.md)，
最终安装边界见 [Attention provider 集成包](attention_provider_integration_bundle.md)。

当 factories 来自独立 adapter packages 时，部署方使用版本固定的 source declarations 和
注入式 factory loader 显式解析；adapter package/version/factory path 会进入 source origin
binding。详见
[Attention provider contribution adapter loading](attention_provider_contribution_loading.md)。
Python 部署可以使用其中的显式 importlib factory loader；它不会扫描或自动注册 adapter。
