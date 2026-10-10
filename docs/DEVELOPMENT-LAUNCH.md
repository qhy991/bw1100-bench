# Bounded registered-task development

`development/` maintains the existing external registered-Task adapter. It uses the
selected Cake Compiler's original Workload, five cases and comparator. Its callable
samples are diagnostic development evidence. They are not independent Bench scores.
The independent `campaign/` Bench evaluator and its community baselines are unchanged.

After the completed Bench comparison has an explicit owner disposition, prepare one
fresh finite development round. `scripts/prepare_development.py` requires `--plan`,
`--contracts`, `--disposition`, `--host` and `--output`. It performs CPU checks and makes
create-only Run directories; it launches neither an author nor a GPU process.

The disposition must name `selected_compiler`, matching the plan's full immutable
`compiler` commit. There is no default selected version. A Bench control condition and
the previously adopted development version need not be the same Compiler.

The private plan contains:

- `kind: registered_task_development`, `target: gfx938`, `compiler`, and `adapter_commit`.
- `controls`: model, `wall_time_seconds: 10800`, `search_seconds: 9000`,
  `confirmation_seconds: 1800`, `token_limits: null`, `capacity_per_hcu: 1`, and
  `reference_access: known_kernel_reproduction_with_declared_development_incumbent`.
- `hosts`: each host's explicit qualified HCU subset, home, owner root, clean Compiler
  root, immutable image, gateway root/commit, and bounded `start_window_seconds`.
- `assignments`: exactly the original 53 distinct tasks, each with its host, fixed HCU,
  fresh root, and `inherited` source file, historical Compiler, candidate ID, candidate
  path and endpoint path. Keep these operational paths in private experiment records.

The original contracts file contains `tasks` with each task's rows, columns, depth,
case IDs, ABI and whole Workload. Preparation compares the selected Compiler's factory
output, every case ABI and both starter/inherited emission against these inputs.
It preserves the historical candidate in `campaign/inherited/` with provenance.
The canonical starter is generated separately inside the new Run budget. Historical
outcomes and samples are never copied into the new evaluations directory. Both starter
and inherited source require new device qualification on all original cases.

The prepared host allocation is consumed by the existing `scripts/bench_queue.py`
with `--allocation` and `--state-directory` equal to the plan's owner root. Run it with
the qualified hmz environment's Python. Before any queue reservation or Run deadline,
every Run checks its frozen binding and the executable `hmz` beside that Python.
The shared owner permits one development author per HCU, retains every failure, and
never restarts an existing owner. Missing release evidence or a development HCU stop
record blocks subsequent Runs on that HCU. Other assigned HCUs may continue.

Fresh physical admission, provider preflight and Bench disposition remain prerequisites
of actual launch. Preparation, CPU fixtures and a clean source commit grant none of
those qualifications. Keep old Runs and old attempt budgets unchanged.
