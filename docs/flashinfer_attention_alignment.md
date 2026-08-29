# FlashInfer Attention 对标边界

本文定义 FlashInfer-NPU 在 Attention 范围内与 NVIDIA FlashInfer 保持一致的
公开编程模型，以及为了适配昇腾软件栈而必须新增的内部机制。它不是开发进度记录，
也不把昇腾 provider 架构描述成 NVIDIA FlashInfer 已有的实现。

## 1. 对标基线

接口语义以 FlashInfer 官方资料为准：

- [Attention API](https://docs.flashinfer.ai/api/attention.html)
- [Attention API source index](https://github.com/flashinfer-ai/flashinfer/blob/main/docs/api/attention.rst)
- [Decode implementation](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/decode.py)
- [Sparse implementation](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/sparse.py)
- [Single prefill with injected JIT module](https://docs.flashinfer.ai/generated/flashinfer.prefill.single_prefill_with_kv_cache_with_jit_module.html)

版本化文档用于确定用户可见合同；主分支源码用于理解 backend/JIT 的内部选择方式。
以后升级对标基线时，应先更新 API parity 数据，再评审本文所列生命周期是否发生变化。

## 2. 核心结论

FlashInfer 的常规 Attention 使用方式确实是 wrapper 持有状态的 `plan()` / `run()`：

1. 用户创建 wrapper，并向 `plan()` 提交 metadata、shape 和 Attention 语义。
2. `plan()` 在 wrapper 内缓存执行所需状态；用户不接收或保存可执行 plan handle。
3. 用户随后调用 `run()`，只传入本次执行的 tensor 和公开 run 选项，不把 plan
   作为参数传回。
4. backend/module 的确定属于实现内部；已有 plan 的 `run()` 不重新选择另一个实现。
5. 重新规划由下一次显式 `plan()` 触发，而不是在执行失败后静默发生。

因此，FlashInfer-NPU 的模型侧主接口也必须保持“提交语义并执行”，不能要求用户理解
CANN、flash-attention-npu、Ascend C、包版本、算子名或 JIT artifact。

不过，“常规路径不暴露 module handle”不等于 FlashInfer 完全没有公开 JIT 接口。
上游明确提供 `*_with_jit_module` 低层入口，让高级用户注入预编译 JIT module 和附加参数；
官方文档同时建议普通用户使用不带该后缀的函数。FlashInfer-NPU 保留这类兼容入口，
但将它们与面向模型的自动路由接口严格分开。

## 3. 用户可见语义对照

| 语义 | NVIDIA FlashInfer | FlashInfer-NPU 设计 |
| --- | --- | --- |
| batch 生命周期 | wrapper 调用 `plan()` 后复用 `run()` | 相同；active plan 由 wrapper 私有持有 |
| plan 返回值 | `plan(...) -> None`，状态缓存在实例内 | 相同；不向用户返回可执行对象 |
| run 入参 | tensor 与公开 run 选项，不传 plan | 相同；不接收 provider/module/callable handle |
| 自动选择 | `backend="auto"` 在支持该参数的接口内选择可用实现 | `auto` 在私有 registry snapshot 内选择已验证的昇腾 operation |
| 显式 backend | 部分 classic/single 接口允许上游定义的 override | 只保留兼容所需的公开参数；普通生产路径使用 `auto` |
| unified batch | `BatchAttention` 统一处理 batch 内 prefill/decode | `BATCH_MIXED_PAGED` 使用同样的统一 facade |
| JIT 高级入口 | 独立的 `*_with_jit_module` 低层 API | 保留独立兼容层，不混入常规 `plan()` / `run()` |
| 重规划 | 用户显式再次调用 `plan()` | 相同；`run()` 不 fallback、不重选、不隐式 replan |
| workspace | wrapper 拥有或接收 workspace，plan 可物化 metadata | 相同所有权语义；具体容量由选中 provider 声明 |
| paged NVFP4 | `plan(..., kv_data_type=uint8)`，`run(..., kv_cache_sf=...)` | 相同公开参数；内部生成固定 NVFP4 QuantSpec 后自动选择 provider |

`backend="auto"` 是兼容入口，不是把 provider 名称变成模型配置。当前公共接口不会新增
`cann`、`flash_attention_npu`、`ascendc` 或具体 `kernel_id` 作为模型侧 backend 值。
集成者可在进程 bootstrap 时安装 provider；模型代码只看到统一 Attention 接口。

`backend="reference"` 是本仓库 Host oracle 的显式开发入口，不代表生产 NPU fallback。
`auto` 没有满足证据要求的 NPU candidate 时必须失败，不能静默改用 Host 计算。

NVFP4 的公开输入同样遵守这条边界。模型代码只提交 FlashInfer-compatible packed `uint8` K/V
和 `kv_cache_sf`；框架在内部固定 E2M1 nibble 顺序、每 16 个值一个 E4M3 scale 的计划语义。
若选中的昇腾算子要求不同物理布局，转换由该 provider 的受审核 adapter 完成，不能变成新的
公开 layout/provider 参数。

## 4. 相同合同与昇腾适配的边界

NVIDIA FlashInfer 并不是一个让第三方 CANN/FA-NPU 包注册进来的通用 provider registry。
它拥有自己的 CUDA backend、预编译/JIT module 和 backend 判定逻辑。FlashInfer-NPU
无法复制这些 CUDA 实现，因此对标分为两层：

### 4.1 必须保持一致的高层合同

- API family、参数语义、layout、shape 与返回值约定；
- wrapper-owned `plan()` / `run()` 状态机；
- plan-time 选择与物化，run-time 复用；
- 默认自动选择，不要求模型代码选择算子；
- 不兼容请求需要显式 replan；
- 常规路径与高级 injected-JIT 路径分离。

### 4.2 FlashInfer-NPU 特有的内部适配

- immutable provider registry snapshot；
- CANN、flash-attention-npu 与未来 Ascend C adapter；
- package/version/callable、capability 和 operator evidence；
- provider plan、workspace、quantization lowering 与 result contract；
- declaration、artifact、ABI、runtime binding 与审计 receipt。

这些机制用于在多个昇腾算子来源之间实现可复核的自动选择。它们是对 FlashInfer
高层使用方式的昇腾实现方案，不声称与上游内部类或源码结构一一对应。

## 5. `plan()` 的责任

公开 `plan()` 只接收 workload metadata 和 Attention 语义。内部按以下顺序完成：

```text
public metadata
  -> canonical framework plan
  -> immutable registry/catalog snapshot
  -> pure candidate admission and deterministic selection
  -> exact package/runtime declaration binding
  -> provider metadata/workspace/JIT materialization
  -> atomic active-plan publication
```

计划阶段可以编译或装载 JIT artifact、转换 page metadata、准备 workspace 和绑定精确
callable，但只能通过集成层已安装的受信依赖完成。计划失败时，不能发布半完成状态；
已有 active plan 保持不变。

`plan_selection` 只是一份可选只读诊断，包含 mode、provider operation、backend、registry
generation 和声明 fingerprint 等非执行信息。它不是 plan handle，不能交回 `run()`，也不能
用于绕过 wrapper 的生命周期。

## 6. `run()` 的责任

公开 `run()`：

- 验证 tensor metadata 与 active plan 兼容；
- 将运行时 tensor/scale/output buffer 降低为选中 operation 的精确参数；
- 调用 plan 阶段已绑定的 executor；
- 验证 completion 和返回 schema；
- 返回 output 或 `(output, lse)`。

`run()` 不探测包、不更改 provider、不重新编译、不重新规划，也不在失败后调用第二个
候选。需要另一个实现或另一组静态语义时，调用方显式再次执行 `plan()`。

## 7. 当前能力边界

框架已经定义 provider 接入、自动选择和生命周期合同，但仓库默认不声明任何可生产执行的
NPU Attention operation。只有当精确版本的 CANN 或 flash-attention-npu 能力具备完整
package、callable、capability、quantization、workspace 和结果证据后，集成层才可以安装它。

当前不声称：

- 已实现或复制 NVIDIA CUDA kernel；
- 已验证任何真实 NPU Attention 数值正确性或性能；
- 任意已安装的昇腾包都会被自动信任；
- Host reference 是 NPU 执行失败时的自动降级；
- provider registry 与 NVIDIA FlashInfer 内部架构相同；
- advanced JIT 兼容入口等同于可用的 Ascend 编译器或动态加载器。

后续真实算子接入必须保持本文的高层接口不变，只扩展内部声明与 adapter。
