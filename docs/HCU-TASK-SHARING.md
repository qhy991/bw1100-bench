# 同一 HCU 上运行多个 agent Task

一个 agent 绑定 HCU id 不代表它持续占有 GPU。模型请求、源代码编辑和 CPU-only
准备不持有卡锁。正确性、配对计时、profiling 等设备作业通过同一个
`scripts/hcu_run.py` 获取 per-user HCU 锁，结束并观察设备释放后才交还锁。

两个 agent 可以交错运行：A 编辑/请求模型时，B 测试；B 编辑时，A 测试。
同卡设备作业串行执行。多个 kernel 同时运行会互相影响时钟、缓存和带宽，
不能用这些干扰下的延迟作为同一个配对性能协议的结果。

## 后继入口

`--queue-timeout SECONDS` 在既有卡锁上有界等待，默认0保持此前立即拒绝行为。
`--latest-start-epoch EPOCH` 由 Task 的 node deadline 提供。获取锁前和完成设备/镜像
探测后均检查该截止时间；已超时请求不启动 GPU 容器。队列等待包含在 agent 的
三小时 wall budget 中，GPU执行的 `--timeout` 单独保持原有含义。

`dtk.sh gpu` 和 `rocprof.sh` 使用同一个入口。未来 Task 通过这些变量传入：

```sh
export BWBENCH_QUEUE_TIMEOUT=900
export BWBENCH_TIMEOUT=600
export BWBENCH_LATEST_START_EPOCH="$(python3 - <<'PY'
import json,os
from pathlib import Path
d=json.loads(Path('campaign/deadline.json').read_text())
p=json.loads(Path('campaign/protocol.json').read_text())
end=d['stop_at_epoch']
print(min(end-p['final_handoff_minutes']*60,
          end-int(os.environ['BWBENCH_TIMEOUT'])-30))
PY
)"
bash scripts/dtk.sh gpu IMAGE .local/job-001/admission.json COMMAND ARGS
```

每个后继 Task 仍有独立目录、模型上下文、cache、deadline、candidate和结果。
等待时间记录在原 admission/terminal receipt 的 `queue_wait_s`；没有新增 allocator
或独立队列数据库。超时、设备未空闲或释放不完整继续拒绝，不自动换卡。
本地锁只协调同一用户、同一入口的作业，不能证明其他用户的物理独占。
等待不保证FIFO或避免所有饥饿；有界超时是明确的退出条件。

## 与旧 Run 的兼容边界

同时共享同一卡的各个新 Task 都必须使用排队入口。旧 fail-fast Task若遇到新
Task持锁，其请求仍会拒绝。不能仅更新新增 Task 就假定它与正在运行的旧 Task
组成了兼容的共享队列。

现有冻结 Run 完成后才启用这个后继执行入口；不替换它们的脚本、时限、候选或
结果。若重开完整十题共享实验，保持同一Compiler/强社区baseline/原始测试和容差，
明确记录资源调度为一个新的协议版本。当前六题的历史结果仍按其原协议解释。

先采用每卡两个agent。增加更多Task会增加排队和主机CPU压力；卡utilization、
队列等待、完成候选数和合格性能结果分别观测。GPU作业内仍有CPU比较阶段，
单纯增加agent不能使那些持锁区间同时服务另一任务。后续分离CPU oracle/比较
阶段需要单独保持原始输入、数值和测量协议的验证。

## 验证范围

`tests/test_hcu_queue.py` 使用真实进程和文件锁检查两个任务不重叠、等待后执行、
有界退出、旧fail-fast、过期admission与探测后过期拒绝。它不使用GPU或Docker。
设备资格化与实际多Task利用率/性能变化是后续证据，不能从锁测试推断。
