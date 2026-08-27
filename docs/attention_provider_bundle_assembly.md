# Attention provider bundle assembly

## 目标

`assemble_attention_operator_provider_integration_bundle()` 是生产 Attention provider 配置
的规范组装入口。它接收 CANN、flash-attention-npu 或其他外部适配模块提供的完整输入集合，
按固定顺序派生最终 `AttentionOperatorProviderIntegrationBundle`。

assembly 只处理 Host 侧声明对象。它不查询 package version、不导入 provider、不解析
callable、不探测设备、不创建 NPU tensor，也不执行算子。

## 输入

调用方必须显式提供：

- `bundle_id`：最终部署 bundle 身份；
- `catalog_name`：本次部署的 scoped operation catalog 名称；
- `scoring_manifest_id`：合并后的选择策略 manifest 身份；
- `operations`：所有允许进入自动路由的精确 operation specs；
- `runtime_specs`：每个 operation 的完整框架适配 spec；
- `scoring_policies`：每个 provider-operation 的声明式 plan policy；
- `package_loader_routes`：每个 operation 到审核 loader delegate 的精确 route。

operation、runtime spec、policy 和 route 的 `(provider_id, operation_id)` 集合必须完全相等。
assembly 不补默认 operation、不复用相似 policy，也不根据包名推断缺失 route。

## 固定组装顺序

assembly 按以下顺序完成全部派生：

1. 从完整 operation 集合构造并规范排序 scoped catalog；
2. 从完整 policy 集合构造并规范排序 scoring manifest；
3. 将 manifest policy 一次性绑定到整个 runtime-spec 集合；
4. 从最终 bound specs 和最终 catalog 生成 runtime declarations；
5. 生成 declaration-bound registrations；
6. 从最终 catalog 和全部 routes 生成 routed package loader；
7. 构造并复核最终 provider integration bundle。

这个顺序保证 declaration 覆盖最终 plan-scorer policy fingerprint，并绑定最终合并 catalog
的 fingerprint。不能先为各 provider 生成绑定局部 catalog 的 declaration，再把这些
declaration 拼到另一个全局 catalog；那会造成声明身份漂移。

如果 runtime spec 已绑定同一 declarative policy，操作保持幂等。已有 custom scorer、不同
policy、缺失/孤立 identity、重复 operation 或 route 字段漂移都会在 bundle 返回之前失败，
不会被 assembly 覆盖或降级。

## 单 provider 与多 provider

单 provider 和多 provider 使用同一个入口。多 provider 输入只是包含更多完整 operation
行，例如 CANN 与 flash-attention-npu 各自贡献 operation、runtime spec、policy 和 loader
route。输入顺序不影响最终 catalog、manifest、routed-loader 或 bundle fingerprint。

```python
from flashinfer_npu.attention import (
    assemble_attention_operator_provider_integration_bundle,
)

bundle = assemble_attention_operator_provider_integration_bundle(
    bundle_id="deployment.attention.providers.v1",
    catalog_name="deployment-attention-catalog-v1",
    scoring_manifest_id="deployment.attention.scoring.v1",
    operations=all_operations,
    runtime_specs=all_runtime_specs,
    scoring_policies=all_scoring_policies,
    package_loader_routes=all_loader_routes,
)
```

assembly 总是生成 `AttentionOperatorRoutedPackageLoader`，即使只有一个 operation。这样单
provider 与多 provider 的 package/callable 授权语义相同，不需要在扩容时切换 bootstrap
模型。

## 审核与安装

assembly 返回的是完整、可计算 fingerprint 的 reviewable bundle，不代表部署方已经接受
它，也不证明外部包或设备可用。部署流程应固定输入来源和最终 bundle fingerprint，再调用
`install_attention_operator_provider_integration_bundle()`，并使用期望 registry generation
防止并发覆盖。

安装之后，package、capability/evidence、评分唯一胜者、tensor/量化契约和 completion
validation 仍按原有 plan/run 门禁执行。assembly 不能绕过这些运行时检查。

低层手工构造 catalog、manifest、declarations、routed loader 与 bundle 的 API 继续保留给
框架组合和定向检查；生产适配模块应优先使用 assembly 入口，减少顺序错误和不完整集合。

当 CANN 与 flash-attention-npu 由不同适配模块独立维护时，每个模块先生成只属于自身
provider 的 `AttentionOperatorProviderIntegrationContribution`，部署层再调用
`assemble_attention_operator_provider_integration_contributions()`。该入口复用本文的全局组装
顺序，并把 contribution fingerprints 纳入最终 bundle；局部 declarations 只用于审核，不会
直接进入全局 registrations。完整所有权与合并契约见
[Attention provider contribution](attention_provider_contributions.md)。

最终 bundle 契约见
[Attention provider 集成包](attention_provider_integration_bundle.md)，loader route 约束见
[Attention package loader 路由](attention_package_loader_routing.md)。
