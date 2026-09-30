# 验证记录：2026-09-30

环境：本地独立 Python 3.12.14、PyTorch 2.9.0 CPU；上游修订见 `sources.lock.json`。
没有运行 GPU kernel，没有产生设备正确性或性能结果。

- 10/10 题完成原始定义与 workload 的 Pydantic/shape 审计，共 160 个 workload。
- 10 项软件回归测试通过，覆盖错误候选的拒绝与结果范围标记。
- 以下 8 题各使用 `suite.json` 冻结的最小输入规模原始 workload，完成 2 轮 CPU
  reference selfcheck：L1/001、011、048、058、069；L2/018、035、060。
- L2/056 的 CPU 尝试在 120 秒时由外部进程超时停止。未产生数值失败结论，未通过
  改小固定维度或另换 workload 来掩盖这个结果。
- L2/024 单份输入约 12 GiB，未在本地执行 CPU selfcheck；保留完整原始题目和测试入口，
  后续在受控设备上验证。

这些是题库整理和测试入口验证，不是 160 个 workload 的设备验证，不是对参考算法的
独立证明。`reference_selfcheck` 没有检查任何优化候选。

原始本地过程记录保存在被 Git 忽略的 `.local/` 中。每次设备执行需新建报告，明确
host/runtime、原始 workload UUID、输入种子、轮数、失败阶段和数值比较结果。
