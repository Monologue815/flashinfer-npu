# Attention provider contribution approval manifest

## 目标

`AttentionOperatorProviderContributionManifest` 是部署方对 provider contributions 的精确
允许清单。provider 模块可以独立构造 contribution，但“模块存在”“包已安装”或“名字匹配”
都不代表部署已授权它进入自动 Attention 路由。只有 contribution binding 与审核清单完全
一致，部署层才可以组装最终 provider integration bundle。

该清单是不可执行、可序列化的 Host 元数据。加载和校验过程不导入 CANN、
flash-attention-npu 或其他 provider，不查询 package version，不解析 callable，不探测 NPU，
也不执行算子。

## 数据模型

一个 manifest 包含：

- `manifest_id`：部署审核版本的稳定身份；
- `contribution_bindings`：按 `contribution_id` 规范排序的完整绑定集合；
- schema version 与固定 kind；
- 由完整 canonical representation 计算的 SHA-256 fingerprint。

每个 contribution binding 继续保存 contribution id/fingerprint、provider id，以及每个
operation 的 operation、局部 declaration、scoring policy 和 loader route fingerprints。
manifest 不保存 contribution 对象本身，也不保存 runtime spec、loader、callable、plan、
tensor 或 provider handle。

同一 manifest 内的 `contribution_id` 必须唯一，各 contribution 的
`(provider_id, operation_id)` 不能重叠。清单顺序不影响 manifest fingerprint。

## 有界加载

`load_attention_operator_provider_contribution_manifest()` 先使用共享的
`AttentionJsonEnvelopeLimits` 限制编码字节、嵌套、节点、数组和字符串，再使用
`AttentionOperatorProviderContributionManifestLimits` 限制：

- contribution 数量；
- 所有 contribution 的 operation 总数。

JSON 必须使用精确字段集合、固定 kind 和受支持 schema version；重复 key、非标准数值、
错误 binding 结构、非法 id 或 fingerprint 均失败关闭。加载结果同时返回 envelope usage，
便于部署系统审计输入规模，但 usage 不参与运行路由授权。

## 精确匹配与组装

部署层将审核清单传给 contribution assembly：

```python
from flashinfer_npu.attention import (
    AttentionOperatorProviderContributionManifest,
    assemble_attention_operator_provider_integration_contributions,
)

approval = AttentionOperatorProviderContributionManifest(
    manifest_id="deployment.attention.contributions.v1",
    contribution_bindings=(
        cann_contribution.binding,
        flash_attention_npu_contribution.binding,
    ),
)

bundle = assemble_attention_operator_provider_integration_contributions(
    bundle_id="deployment.attention.providers.v1",
    catalog_name="deployment-attention-catalog-v1",
    scoring_manifest_id="deployment.attention.scoring.v1",
    contributions=(cann_contribution, flash_attention_npu_contribution),
    approval_manifest=approval,
)
```

assembly 先重新验证 contribution 当前的组件与 loader 身份，再把规范排序后的完整 bindings
与 manifest 逐项比较。比较同时区分：

- `missing`：清单允许但本次没有提供；
- `orphan`：本次提供但清单未允许；
- `drifted`：id 相同，但 contribution 或任一 operation 绑定发生变化。

任一集合非空都会在 catalog 合并、package probe 和 registry 发布之前失败。框架不会按
provider 名称接受替代项，也不会忽略缺失 provider 或使用旧 fingerprint。

清单验证通过后，assembly 才按最终部署 catalog 重新生成 declarations，并构造 routed
loader 与 bundle。清单只授权配置集合，不证明外部包、当前环境、capability evidence 或
算子结果；这些仍由 plan/run 门禁验证。

## 身份传播与升级

最终 bundle 和其非执行 binding 同时保存 approval manifest id/fingerprint。两者进入 bundle
canonical representation，因此清单变更会改变 bundle fingerprint，并经现有链路传递到
registry snapshot、plan selection、成功 run receipt 和离线 scoring audit。模型侧 API 不
新增 manifest 参数。

升级某个 provider contribution 时，部署方必须生成新的 contribution fingerprint、更新并
重新审核 manifest，然后组装并原子安装新的 bundle generation。已有 wrapper 继续使用其
创建时捕获的旧 generation；框架不在一次 `run()` 中热替换清单或 provider。

未绑定 approval manifest 的低层 contribution assembly 路径仍保留给框架组合和定向开发；
生产部署应提供 manifest，确保“发现”“审核”“组装”“安装”四个阶段分离。

provider 所有权和局部审核内容见
[Attention provider contribution](attention_provider_contributions.md)，最终原子安装见
[Attention provider 集成包](attention_provider_integration_bundle.md)。
