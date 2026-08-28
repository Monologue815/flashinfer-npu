# Attention provider contribution adapter loading

## 目标

source registry 接收已经构造的 adapter factories；
`AttentionOperatorProviderContributionSourceDeclarationRegistry` 进一步定义这些 factories 如何
从版本固定的 adapter packages 显式加载。它是部署 bootstrap 的受控代码加载边界，不是
Python 环境自动发现机制，也不是实际 Attention provider callable 的加载器。

框架不提供默认扫描、不读取 entry points、不按包是否安装自动注册。部署方必须显式构造
declarations、注入 factory loader，并提供已经审核的 contribution approval manifest。

## Source declaration

每条 `AttentionOperatorProviderContributionSourceDeclaration` 固定：

- source id/version、provider id 与预期 contribution id；
- adapter package 名称和允许的精确 package versions；
- 模块限定的 factory path；
- 预期 factory id 与稳定类型名称；
- schema version 与完整 declaration fingerprint。

declaration 是不可执行数据，不包含 factory、module、callable、runtime spec、tensor 或
provider handle。source id、contribution id、provider-contribution identity 和
`(adapter_package_name, factory_path)` 在同一 registry 中必须唯一。

这里的 adapter package 是负责构造 FlashInfer-NPU framework contribution 的集成模块。它与
实际执行 Attention 的 CANN/flash-attention-npu provider package 是两个加载边界。即使二者
最终由同一个发行包提供，adapter factory 也只能在 bootstrap 生成 Host 配置，不能提前解析
或调用 provider Attention 算子。

## 注入式 factory loader

`AttentionOperatorProviderContributionFactoryLoader` 提供三个显式能力：

- 稳定 `loader_id`；
- `package_version(package_name)`；
- `resolve_factory(factory_path)`。

框架只定义协议，不在当前阶段提供默认 importlib 实现，也不会自行安装 adapter package。
部署系统可以注入受控 loader；loader 的 id/type 在 declaration registry 构造时冻结，并在每次
package observation 和 factory resolution 后重新验证。局部类、lambda、空白 id 或调用过程
中的 loader 身份变化都会失败关闭。

## 加载顺序

`load_sources(approval_manifest)` 保持以下固定顺序：

1. 复核 factory loader id/type；
2. 在任何 package observation 或 import 前，要求 declarations 的
   `(provider_id, contribution_id)` 集合与 approval manifest 完全相等；
3. 按 source id 规范排序；
4. 对每个 adapter package 只观察一次版本，并要求它属于 declaration 的精确允许集合；
5. 通过注入 loader 解析精确 factory path；
6. 核对 factory protocol、factory id 与稳定类型；
7. 生成带 origin binding 的 `AttentionOperatorProviderContributionSource`；
8. 形成不可变 source registry，尚不调用 `build_contribution()`。

共享同一 adapter package 的多条 declaration 必须接受同一个已观察版本。缺失 declaration、
多余 declaration、package version 不支持、factory path 解析失败或 factory identity 漂移都会
在 contribution factory 执行之前停止。

## Source origin binding

显式加载的 source 保存不可执行
`AttentionOperatorProviderContributionSourceOriginBinding`：

```text
(adapter_package_name,
 observed_package_version,
 factory_path,
 factory_loader_id,
 factory_loader_type,
 declaration_fingerprint)
```

origin binding 进入 source binding 和 source registry fingerprint。随后 source registry 物化
contributions、approval manifest 复核完整 contribution fingerprints、部署层重新生成全局
declarations，最终 source registry binding 进入 provider integration bundle。因此 adapter
declaration、观察版本、factory path 或 loader identity 的变化都会传递为新的 bundle
fingerprint。

## 一步组装入口

```python
from flashinfer_npu.attention import (
    AttentionOperatorProviderContributionSourceDeclaration,
    AttentionOperatorProviderContributionSourceDeclarationRegistry,
    assemble_attention_operator_provider_integration_source_declarations,
)

cann = AttentionOperatorProviderContributionSourceDeclaration(
    source_id="cann.attention.adapter-source.v1",
    source_version="1.0.0",
    provider_id="cann",
    contribution_id="cann.attention.release.v1",
    adapter_package_name="flashinfer-npu-cann-adapter",
    supported_package_versions=("1.0.0",),
    factory_path="flashinfer_npu_cann.attention.build_contribution",
    factory_id="flashinfer_npu_cann.attention.factory.v1",
    factory_type="flashinfer_npu_cann.attention.AttentionContributionFactory",
)

declared_sources = AttentionOperatorProviderContributionSourceDeclarationRegistry(
    registry_id="deployment.attention.adapter-sources.v1",
    declarations=(cann, flash_attention_npu),
    factory_loader=controlled_factory_loader,
)

bundle = assemble_attention_operator_provider_integration_source_declarations(
    bundle_id="deployment.attention.providers.v1",
    catalog_name="deployment-attention-catalog-v1",
    scoring_manifest_id="deployment.attention.scoring.v1",
    source_declarations=declared_sources,
    approval_manifest=approved_contributions,
)
```

该入口依次执行显式 factory 加载、source registry 物化、完整 contribution approval、最终
catalog/scoring/declaration/route 组装。它返回尚未安装的 bundle；部署方仍使用 bundle
installer 和 expected registry generation 原子发布。

## 与实际 provider package 的边界

adapter factory 加载成功只说明审核的配置代码可以构造 contribution，不说明 CANN 或
flash-attention-npu 的实际 Attention operation 已安装、签名正确或具备当前 plan 能力。
contribution 内的 routed provider package loader 仍在 `plan()` 后续门禁观察精确 provider
package version 与 callable，继续执行 capability/evidence、量化映射、workspace、tensor、
execution 和 completion validation。

模型侧 API 不接收 declaration、factory loader、source registry、contribution、provider 或
callable 参数。用户仍只调用 wrapper `plan()` 与 `run()`；bootstrap provenance 通过 bundle
fingerprint 进入 plan/run 审计链。

source registry 语义见
[Attention provider contribution source registry](attention_provider_contribution_sources.md)，
部署 approval 见
[Attention provider contribution approval manifest](attention_provider_contribution_manifest.md)，
provider callable 路由见
[Attention package loader 路由](attention_package_loader_routing.md)。
