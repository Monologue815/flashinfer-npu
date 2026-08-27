# Attention package loader 路由

## 目的

一个生产 `AttentionOperatorProviderIntegrationBundle` 可以同时包含 CANN/aclnn 与
flash-attention-npu operation，但两类集成不一定使用同一种包观察和 callable 解析方式。
`AttentionOperatorRoutedPackageLoader` 在单一 bundle loader 边界内组合多个 delegate，
避免强迫所有 provider 共享一个实现，也避免在模型接口中增加 provider 选择参数。

该组件只负责 loader 路由，不注册算子、不导入外部包、不探测 NPU，也不改变 Attention
`plan()` / `run()` 接口。

## 精确路由单位

每个 `AttentionOperatorPackageLoaderRoute` 绑定：

- `provider_id` 与 `operation_id`；
- catalog 声明的精确 `package_name`；
- catalog 声明的精确 `callable_path`；
- 一个实现 `AttentionOperatorPackageLoader` 协议的 delegate；
- 构造时冻结的 delegate `loader_id` 和稳定类型名称。

route 推荐通过 `from_catalog_operation(operation, loader)` 创建。复合 loader 构造时要求
route identity 集合与 scoped operation catalog 完全相等，并逐项复核 provider、package 和
callable。缺失、孤立、重复或字段漂移都会在任何外部包观察前失败。

路由不使用名称前缀、正则、import fallback 或“第一个能导入的包”。
`package_version(package_name)` 只接受 catalog 中的精确 package；
`resolve_callable(callable_path)` 只接受 catalog 中的精确 callable。

## 消除歧义

同一个 package name 可能服务多个版本化 operation，同一个 callable path 也可能由多个
API version 共享。只有这些 route 指向同一个 delegate 对象时才允许共享；若同一个查询键
指向不同 delegate，复合 loader 构造失败。这样 package metadata observation 与 callable
resolution 不会因 route 顺序产生不同结果。

不同 package 或 callable 可以指向 loader id/type 相同但实例不同的 delegate；它们仍由
各自的精确 route 隔离。

## 身份与漂移

每个 route 提供不含可执行对象的 binding。复合 loader 的 `routing_fingerprint` 覆盖：

- scoped catalog name/fingerprint；
- 全部规范排序后的 route bindings；
- 每个 delegate 的 loader id/type。

公开给 bundle 的 `loader_id` 由固定前缀和 routing fingerprint 组成，因此 bundle fingerprint
间接覆盖完整 loader 路由图。route 输入顺序不影响 fingerprint。

复合 loader 在读取自身 `loader_id` 以及每次 package/callable observation 前复核全部 delegate
身份。任一 delegate 在构造后改变 loader id/type，整个路由器立即失败；不会只让未漂移的
另一条 provider route 继续运行。bundle 安装前的 loader 身份复核也会触发这一检查，所以
构造 bundle 之后、发布 registry 之前的内部漂移不能被隐藏。

## 使用方式

下面只展示框架组合方式，不表示仓库已经启用这些具体 operation：

```python
from flashinfer_npu.attention import (
    AttentionOperatorPackageLoaderRoute,
    AttentionOperatorRoutedPackageLoader,
)

routes = tuple(
    AttentionOperatorPackageLoaderRoute.from_catalog_operation(
        operation,
        loader_by_operation_id[operation.operation_id],
    )
    for operation in scoped_catalog.operations
)

package_loader = AttentionOperatorRoutedPackageLoader(
    operation_catalog=scoped_catalog,
    routes=routes,
)
```

随后把 `package_loader` 作为普通 `AttentionOperatorPackageLoader` 传给
`AttentionOperatorProviderIntegrationBundle`。bundle 构建和安装只读取冻结身份；真实
delegate 的 `package_version()` 与 `resolve_callable()` 仍然分别推迟到 plan 的 metadata
observation 和 callable resolution 阶段。

## 与自动选择的关系

复合 loader 不决定哪个 provider 胜出。resolver 仍先对每个 catalog operation 完成既有的
package、capability/evidence、资源与评分流程，再由唯一最高分产生 active plan。路由器只
保证“某个候选需要观察外部包时，使用的是该 operation 已审核的 loader”。运行期失败仍不
触发另一 route 或 provider 的隐式 fallback。

完整生产接入单位见
[Attention provider 集成包](attention_provider_integration_bundle.md)，接入步骤见
[Attention 外部 provider 接入指南](attention_provider_onboarding.md)。
