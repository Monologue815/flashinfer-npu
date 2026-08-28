# Attention provider 顶层 bootstrap manifest

## 目标

`AttentionOperatorProviderIntegrationBootstrapManifest` 是一次生产 Attention provider
部署的顶层、纯数据授权。部署方只需审核这一份清单，即可固定最终 bundle、source
declaration manifest、contribution approval manifest 和 factory loader 的完整身份关系。

它不包含 Python factory、provider callable、tensor、plan 或设备句柄，不扫描环境、不安装
package、不导入 module，也不访问 NPU。清单本身不使任何算子获得执行权限。

## 身份闭包

manifest 固定以下字段：

| 身份 | 作用 |
|---|---|
| `bootstrap_id` | 本次部署审核或发布的稳定名称 |
| `bundle_id` | 最终 provider integration bundle 身份 |
| `catalog_name` | 最终 scoped operation catalog 名称 |
| `scoring_manifest_id` | 自动 plan 选择策略集合身份 |
| `source_manifest_id/fingerprint` | 允许加载的 adapter distribution、版本和 factory path 集合 |
| `contribution_manifest_id/fingerprint` | 允许进入全局 catalog 的 provider contribution 集合 |
| `factory_loader_id/type` | 解释 source declarations 的唯一受控代码边界 |

`fingerprint` 由完整 canonical representation 计算。仅修改发布 id、任一下级 manifest、loader
身份或最终 bundle 命名，都会产生新的 bootstrap fingerprint。

## 组装与失败边界

`assemble_attention_operator_provider_integration_bootstrap()` 按固定顺序处理输入：

1. 将 source manifest id/fingerprint 与顶层清单精确比较；
2. 将 contribution approval manifest id/fingerprint 与顶层清单精确比较；
3. 将 factory loader id/type 与顶层清单精确比较；
4. 仅在三类身份全部一致后，观察 adapter distribution 的精确版本并解析 factory；
5. 生成 provider-owned contributions，并按 approval manifest 做集合与 fingerprint 复核；
6. 形成全局 catalog、scoring manifest、declarations、routed package loader 和 bundle；
7. 复核每个 source origin 都携带同一 source manifest 与 factory loader 身份；
8. 把 bootstrap id/fingerprint 纳入 bundle fingerprint 和非执行 registry binding。

前三步失败不会读取 distribution metadata 或解析 factory。后续任一步失败都不会发布新的
registry generation。adapter factory 是 Host 侧配置工厂，必须保持纯构造语义；它不能执行
provider 算子或产生设备副作用。

## 一步安装

```python
from flashinfer_npu.attention import (
    AttentionOperatorProviderIntegrationBootstrapManifest,
    install_attention_operator_provider_integration_bootstrap,
)

bootstrap_manifest = (
    AttentionOperatorProviderIntegrationBootstrapManifest.from_inputs(
        bootstrap_id="deployment.attention.bootstrap.v1",
        bundle_id="deployment.attention.providers.v1",
        catalog_name="deployment-attention-catalog-v1",
        scoring_manifest_id="deployment.attention.scoring.v1",
        source_manifest=source_manifest,
        approval_manifest=contribution_manifest,
        factory_loader=controlled_factory_loader,
    )
)

installed = install_attention_operator_provider_integration_bootstrap(
    bootstrap_manifest=bootstrap_manifest,
    source_manifest=source_manifest,
    approval_manifest=contribution_manifest,
    factory_loader=controlled_factory_loader,
    expected_generation=current_generation,
)
```

installer 在最终发布点再次检查 `expected_generation`，并把 resolver、catalog、scoring
authority、declaration bindings、source provenance、bundle identity 和 bootstrap identity
作为一个不可分割的 registry generation 发布。并发更新不会被静默覆盖。

需要在审核系统或配置文件中保存清单时，使用 `to_json()` 和
`load_attention_operator_provider_integration_bootstrap_manifest()`。JSON loader 先应用共享的
有界 envelope 规则，再要求精确字段集合、固定 kind、受支持 schema version、合法 id 和
SHA-256 fingerprint 格式。JSON 只保存身份引用；下级 manifests 仍分别交付，并在组装时按
fingerprint 重新核对。

## 模型侧边界

bootstrap 只属于进程部署层。模型使用者仍只创建公开 Attention wrapper，调用 `plan()`，
再调用 `run()`：

- `plan()` 在内部 registry generation 中自动选择唯一 eligible provider operation；
- `run()` 只执行已绑定的 active plan；
- 使用者不传 bootstrap、bundle、provider、operation、callable 或 plan handle；
- `plan_selection` 和成功 run receipt 只读地携带 bundle identity，bootstrap identity 已经通过
  bundle fingerprint 形成传递授权链。

因此顶层清单加强了部署可审核性，但不扩大或改变 FlashInfer 对齐的模型调用接口。

下级 source 数据格式见
[Attention provider source declaration manifest](attention_provider_source_manifest.md)，
contribution allow-list 见
[Attention provider contribution approval manifest](attention_provider_contribution_manifest.md)，
最终原子发布单元见
[Attention provider integration bundle](attention_provider_integration_bundle.md)。
