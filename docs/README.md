# FlashInfer-NPU 文档指南

`docs/` 只保存仓库需要长期维护的设计与使用说明，不保存开发过程记录。

## 文档范围

- 总体分层、模块职责和依赖方向；
- FlashInfer Attention API 对标范围与兼容策略；
- `plan()` / `run()` 生命周期、自动算子选择和 provider 接入契约；
- metadata、workspace、tensor、量化 KV、JIT、artifact 与 launcher ABI；
- capability、correctness、trace、发布和扩展所必须满足的规范；
- 仓库当前明确支持、尚未支持和禁止隐式推断的能力边界。

## 不进入文档的内容

- 检查点编号和开发进度流水账；
- 某一次本地或远端测试的通过数量、控制台输出和耗时；
- 临时验证目录、机器地址、账号或环境操作记录；
- wheel 文件名、哈希和一次性安装结果；
- 会随测试集合变化而失效的 case 数量、coverage 数量或通过率。

这些信息分别属于测试代码、CI 输出、版本历史和受控的验证记录，不是仓库设计文档。
文档可以定义测试与证据必须遵守的规则，但不复制某次执行结果。

## 阅读顺序

1. [`architecture.md`](architecture.md)：总体目标、分层和架构决策。
2. [`attention_framework.md`](attention_framework.md)：Attention 范围、模式和核心语义。
3. [`attention_frontend_contract.md`](attention_frontend_contract.md)：公开 API 与参数归属。
4. [`attention_plan_run_dispatch_design.md`](attention_plan_run_dispatch_design.md)：wrapper
   持有的 `plan()` / `run()` 生命周期与自动 provider 选择。
5. [`attention_quantization.md`](attention_quantization.md)：量化 KV Cache 语义。
6. [`attention_workspace.md`](attention_workspace.md)：workspace 查询、容量和所有权。
7. [`attention_capability_profile.md`](attention_capability_profile.md) 与
   [`support_matrix.md`](support_matrix.md)：外部算子包进入可执行路由前必须满足的能力声明。
8. [`attention_jit_framework.md`](attention_jit_framework.md)：JIT 目录、缓存身份与加载边界。

其余文件细化 trace、数值、tensor、artifact、ABI、launch 和 execution identity 等独立契约。

## 动态事实来源

- Attention API 对标状态：`flashinfer_npu/data/attention_api_parity.json`；
- backend capability：`flashinfer_npu/data/attention_capabilities.json`；
- kernel/artifact 声明：对应的版本化 manifest；
- conformance case 与覆盖统计：版本化 corpus 和 coverage policy；
- 测试是否通过：CI 或当前测试命令的输出。

设计文档解释这些事实源的含义和约束，但不手工维护它们的动态统计值。
