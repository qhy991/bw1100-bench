# 复用 Cake 的范围

本次核对的 Open-Cake source revision 是 `f09a3e859859e848f33a21b63412a306eea8738f`。
先阅读 owning repository 中的以下文件，再使用其运行命令：

- `docs/dcu-gfx938-design.md`：环境说明明确区分被拒绝的 `docker exec` 与已用过的
  `docker run --rm`；DTK 的 `LD_LIBRARY_PATH`、`ROCM_PATH` 和临时目录是实际条件。
- `src/open_cake_ir/evaluation/local_broker.py`：Hygon 采用本地串行 admission，
  不是必须使用 NVIDIA 集群的 `gpu-run`。同一用户/锁范围的串行不等于物理 GPU 独占。
- `src/open_cake_ir/evaluation/triton_hip.py`：绑定实际 HIP target、设备和 admission。
- `findings/2026-09-23-001-bw1100-1-executor-host.json`：同为 gfx938，两个 host
  的内核与环境不同，需要分别记录，不能共用旧 Executor 身份。
- `docs/adr/0066-sol-execbench-task-import.md`：Cake 的导入约定采用固定 shape、
  新的独立 CPU oracle 和更严格的逐元素门；它明确不继承原始全部 workload 域。

因此本项目复用其容器环境经验，不把当前 10 道题冒称为已完成 Cake IR 导入。
L1/L2 的原始 ABI 包含 BF16、int64、bool、动态 shape 和复杂子图；Cake 的某个算子
或某个固定 shape 通过不能替代本工作集的 160 个原始 workload。

`scripts/dtk.sh` 复用标准临时容器命令，并在 GPU 模式直接调用当前 Cake checkout 的
Hygon `local_broker` admission。它不改 Docker 包装程序、用户组或 socket，也不声称
本地锁能排除其他用户和容器。调用方应先观察 KFD 占用、选定 HCU，并给出新收据路径。
这条路径记录本地 admission 与 benchmark 正确性；它不是 Cake 的正式 Executor receipt。

现场验证表明：此节点的 PyTorch 2.11/DTK 镜像在普通容器 UID 下 `rocminfo` 退出 8，
在现有 vLLM 使用的 root/privileged 身份下才识别一张 `BW1101/gfx938`。GPU 模式
据此采用后者，同时维持只读根、无网络和只读 hyhal 挂载。CPU 模式继续用宿主 UID。

历史 runbook 不是当前运行资格。每台机器仍应核对具体镜像、PyTorch/Triton 版本、
HCU 枚举与实际占用。不得由一次 `docker exec` 拒绝推断所有 GPU 执行路径都不可用。
