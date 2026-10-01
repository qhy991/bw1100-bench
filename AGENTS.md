# bw1100-bench

- `suite.json` owns the 10-task selection, difficulty rationale, smoke UUIDs, default
  seed and number of correctness rounds. `sources.lock.json` owns upstream revisions.
- Raw dataset and generated problems stay in ignored `.data/`. Never upload them,
  even to this private repository. Preserve their upstream license and source binding.
- Reuse pinned upstream `gen_inputs` and `compute_error_stats`; keep any stricter
  local gates explicit. Do not silently change shape, precision or tolerances.
- CPU/selfcheck/smoke/partial results never become full device correctness or timing.
  The runner is correctness-only; it is not the official NVIDIA evaluator.
- The GPU entry is owned by this repository: `scripts/dtk.sh gpu` calls
  `scripts/hcu_run.py`, with no dependency on another project. Inspect live HCU
  and KFD occupancy before admission; leave existing services untouched.
- The task-local per-user lock serializes this suite only. It is not physical
  GPU exclusivity and cannot rule out other users or containers.
- Implement first, then run focused CPU regression tests with
  `python -m unittest discover -s tests -v`. Run `bwbench.py audit` after materialization.
- Preserve failed and historical reports; all execution output paths are create-only.
