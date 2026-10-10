# Frozen fresh-search launch

This successor maintains the existing external Ralph campaign and common assay. It
prepares the frozen plan's explicit nonempty task subset of the twelve-task suite for
both Compiler conditions. It does not launch GPU work during preparation, select a
winning version, or dispatch the next
53 development tasks. The evolve owner reviews all Bench terminal records first.

Freeze with Open-Cake `tools/evolve.py freeze-bench` using this clean Bench commit,
`measurement_protocol: campaign/protocol.json`, the exact community baseline map,
`knowledge: none`, `reference_access: known_kernel_reproduction`, and
`scaffold: bw1100-bench@<this commit>:campaign/TASK.md`. The shared author budget is
10800 seconds including 1800 for confirmation. Name the model with effort explicitly.
Each selected task must have every declared
replicate in both conditions; unselected tasks cannot appear in its allocation or
baseline map. A later subset is a separately named comparison segment, not a rewrite
or replacement of an earlier cohort or its failed Runs.
Every original 16-workload, ten-round gate remains unchanged. Each added GEMM baseline
must pass that full gfx938 gate before the owner authorizes the cohort to launch.

Freeze a separate routing JSON before any preparation. It contains `hosts` keyed by
SSH host identity and `assignments` containing every plan allocation exactly once.
Each assignment has `workspace`, `host`, `root` and `hcu`. `workspace` is the original
plan allocation identity; `root` is its physical host path. They are distinct fields.
Each task's two conditions must use the same host and HCU. Physical `(host, root)` pairs
must be unique, canonical and absent. HCU0 is excluded.

Each host entry contains:

- `home`: the existing account home used by hmz authentication and logs. It is not
  changed on disk and does not establish filesystem isolation.
- `image`: qualified immutable image ID; `gateway`: clean source `root` and `commit`.
- `compiler_roots`: clean `control` and `successor` source checkouts at the plan commits.
- `data_root`: materialized suite `.data`, `sol_root`: pinned SOL-ExecBench source,
  `flag_gems_root`: the pinned FlagGems source used by the fixed baseline qualification.
- `capacity_per_hcu`: explicit author concurrency from 1 to 3, and
  `start_window_seconds`: the positive bound for admitting new authors.

For example, one assignment is
`{"workspace":"/intent/runs/t001-r1-control","host":"node-a","root":"/experiments/cohort/t001-r1-control","hcu":1}`.
The corresponding successor assignment uses `node-a` and HCU1 too. Freeze all
assignments before choosing a local subset. Do not edit routing after preparation.

```bash
python3 scripts/prepare_bench.py --plan /intent/plan.json \
  --routing /intent/routing.json --host node-a --output /intent/node-a/allocation.json
```

Preparation makes fresh Git clones, copies original data and dependencies, overlays
only `dtk.sh`, `hcu_run.py`, its required `hcu_device_identity.py`, `rocprof.sh`
and `verify_rocprof_csv.py` from the qualified host gateway, and commits
the Run's binding. It copies no prior candidate, outcome or author session. The full
plan and routing remain recorded. The source checkout's author identity must already
be configured. Preparation reads both Git identity fields before creating any Run and
copies them into each Run's repository-local configuration. It does not change global
Git configuration or HOME. No raw data is added to Git.

The owner must inspect `campaign/binding.json`, full baseline qualification, current
runtime/device mapping and release state before dispatch. Reconciliation checks the
actual source and controls again immediately before each Run's clock starts. The owner
uses `hmz` beside its own Python executable and prepends that bin directory to PATH.
A bounded `hmz --help` check runs before deadline/intake creation, without a Provider
request. A missing, non-executable or unusable entry therefore cannot consume a Run
budget. Start the queue with the qualified environment's absolute Python path.

```bash
python3 scripts/bench_queue.py --allocation /intent/node-a/allocation.json \
  --state-directory /intent/node-a/owner
```

Run this owner under the existing persistent terminal/tmux mechanism. This is only an
author sequencer; the qualified HCU gateway owns every device lease. `LAUNCH.json` and
per-Run reservations are create-only, so reconnect by observing the same owner PID and
`STATUS.json`. Never start a replacement owner after an observation timeout. A missing
DONE, nonzero owner exit, missing release record or live owned container remains
attention in the terminal report. Unverified release blocks subsequent authors on that
HCU; an empty container listing cannot clear it. A start-window expiry leaves unstarted allocations
visible. Review each final confirmation, failed and unknown attempt before choosing
the next Compiler. The script never performs automatic promotion or another batch.

The inherited assay uses complete callable wall time, 10 warmups, 30 forward pairs,
30 reverse pairs and 30 A/A pairs per original workload. Search candidates must pass
all original correctness cells, caller ownership and precision checks. The owner
nominates one fixed source and confirms it inside the same three-hour budget. Final
confirmation failure cannot become an accepted endpoint. A confirmed, source-bound
nominee is sealed as `completed_pending_owner_review`. Emission replay validates the
declared source, but does not prove the wrapper executed every declared receipt.
After all Runs terminate and release is verified, the evolve owner reviews the frozen
wrapper, receipt closure and rounding chain in a separate comparison disposition.
The owner does not rewrite DONE, change a nominee, reopen search or add measurements
outside the original budget. Until this review, the aggregate must say attribution is
pending and cannot automatically adopt a version or dispatch the next 53 tasks.

Historical implementation source was the retained round4 `engine/campaign` and its
finite queue. This tracked successor removes historical seed paths and fixed ten-task
allocation while preserving the assay, adapter and gateway responsibilities. It adds
explicit original BF16/FP16 operands and FP32 accumulation checks for the two GEMMs.
