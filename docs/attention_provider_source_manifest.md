# Attention provider source declaration manifest

## 目标

`AttentionOperatorProviderContributionSourceDeclarationManifest` 是 adapter source declarations
的版本化、不可执行交付格式。它允许部署方在不持有 factory loader、factory 对象或 provider
包的情况下审核“允许加载哪些 adapter 配置来源”，随后再把审核结果绑定到受控 loader。

manifest 不扫描 Python 环境、不安装 package、不导入 module、不构造 contribution，也不访问
NPU。它只描述完整 source declaration 集合。

## Schema

manifest 包含：

- 稳定 `manifest_id`；
- 规范排序的 `AttentionOperatorProviderContributionSourceDeclaration` 集合；
- 固定 kind 与 schema version；
- 由完整 canonical JSON representation 计算的 SHA-256 fingerprint。

每条 declaration 固定 source/provider/contribution identity、adapter distribution 名称、允许的
精确 versions、factory path、factory id/type。source id、contribution id、
provider-contribution identity 和 package/factory location 均必须唯一。输入顺序不影响
manifest fingerprint。

manifest 不包含 factory loader id/type，因为 loader 是部署时注入的可执行边界；绑定之后的
declaration registry 会同时记录 manifest identity 与 loader identity。

## 有界 JSON

`load_attention_operator_provider_contribution_source_declaration_manifest()` 先应用共享
`AttentionJsonEnvelopeLimits`，限制字节、嵌套、节点、数组、对象字段和字符串；再应用
`AttentionOperatorProviderContributionSourceDeclarationManifestLimits`：

- 最大 declaration 数量；
- 每条 declaration 最大 package-version 数量；
- 全部 declarations 的 package-version 总数。

JSON 必须使用精确字段集合、固定 kind 和受支持 schema version。重复 key、未知字段、错误
数组结构、非法 id/path/version 或超限输入都会在构造任何 declaration registry 前失败。
loader 同时返回 envelope usage，供部署审计输入规模；usage 不参与运行授权。

## Loader binding

通过 `AttentionOperatorProviderContributionSourceDeclarationRegistry.from_manifest()` 将 data-only
manifest 与显式 factory loader 绑定。该步骤只冻结 manifest id/fingerprint、loader id/type 和
declarations，不读取 distribution metadata，也不 import module。

随后 `load_sources(approval_manifest)` 才按固定顺序执行：

1. contribution approval manifest 与 source declarations 集合核对；
2. adapter distribution version observation；
3. 精确 factory path resolution 和 factory id/type 核对；
4. source origin binding 构造；
5. source registry 形成；
6. factory 生成 contributions，并再次接受完整 approval manifest 校验；
7. 最终全局 bundle assembly。

每个显式加载的 source origin 同时记录 source declaration fingerprint 和 declaration manifest
id/fingerprint。修改 manifest id、任一 declaration、允许版本或 factory identity 都会改变
source registry fingerprint，并进一步改变 bundle fingerprint。

## 一步组装

```python
from flashinfer_npu.attention import (
    AttentionOperatorProviderContributionSourceDeclarationManifest,
    assemble_attention_operator_provider_integration_source_manifest,
)

source_manifest = AttentionOperatorProviderContributionSourceDeclarationManifest(
    manifest_id="deployment.attention.adapter-sources.v1",
    declarations=(cann_source, flash_attention_npu_source),
)

bundle = assemble_attention_operator_provider_integration_source_manifest(
    bundle_id="deployment.attention.providers.v1",
    catalog_name="deployment-attention-catalog-v1",
    scoring_manifest_id="deployment.attention.scoring.v1",
    source_manifest=source_manifest,
    factory_loader=controlled_factory_loader,
    approval_manifest=approved_contributions,
)
```

返回值仍是尚未安装的 provider integration bundle。部署方使用既有 bundle installer 和期望
registry generation 原子发布；模型侧 `plan()` / `run()` 不接收 source manifest 或 loader。

adapter distribution 加载与错误边界见
[Attention provider contribution adapter loading](attention_provider_contribution_loading.md)，
最终 contribution 审核见
[Attention provider contribution approval manifest](attention_provider_contribution_manifest.md)。
生产部署应再用
[Attention provider 顶层 bootstrap manifest](attention_provider_bootstrap.md)
把本 manifest、approval manifest、factory loader 与最终 bundle 命名固定为一个审核身份。
