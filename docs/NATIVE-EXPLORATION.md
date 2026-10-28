# Explore a concrete Cake limitation with HIP

Use this procedure when an optimization mechanism has a demonstrated Cake IR
expressibility or backend lowering limitation. First retain the smallest Cake
attempt, the exact diagnostic or missing public vocabulary, and the supported
transforms you considered. A malformed candidate, an unknown tool error or a
resource wait does not establish an IR gap.

A HIP probe is required for that mechanism within the current Run's remaining
search budget. If the budget or infrastructure prevents it, record the specific
reason with the same command's `--skip-reason` option. Keep failed, inconclusive
and unexecuted probes. Do not restart a Run, extend its budget or repeat a probe
for a preferred result.

## Prepare the owned source and gap evidence

Put the complete own Python, HIP and C++ source closure in a dedicated directory
under `campaign/candidates/`. Supported source suffixes are `.py`, `.hip`, `.cpp`,
`.cc`, `.c`, `.cu`, `.h`, `.hpp` and `.hh`. Do not import another Run's source or
results. Include every own helper and header. Symlinks and escaping paths are
refused. Keep the request JSON outside the source directory.

The Python entry file exposes two fixed callables.

- `prepare()` performs CPU-only compilation before any GPU lease. It takes no
  arguments. Read `HIP_PROBE_BUILD_DIR` for the persistent build directory,
  `HIP_PROBE_TARGET` for `gfx938` and `HIP_PROBE_BUILD_FLAGS` for the JSON flag
  list. `TORCH_EXTENSIONS_DIR` names the same persistent directory. Return a
  dictionary with a nonempty `artifacts` list of paths relative to that directory.
  Optional `metadata` records compiler versions, commands and loaded module names.
  Retain compiler stdout and stderr. Do not use `/tmp` or `HOME` for build artifacts.
- `run()` loads those compiled artifacts and implements the original task ABI.
  Bench calls it with the original positional arguments and ordered returns.
  Registered development calls it with the original named input tensors and
  expects its original ordered output tensors. Keep exact output count, shape,
  dtype, contiguous layout and logical device `cuda:0`. Do not compile or prepare
  inputs on the device path. Importing the entry file must be safe without a GPU.

Use `Path(__file__).parent` for sources. Use `HIP_PROBE_BUILD_DIR` to locate retained
binaries when `run()` loads the module. Compilation does not establish that the
GPU path executed the declared HIP kernel. The independent reviewer must inspect
that path, its complete source and its fixed rounding chain.

The request is an own JSON file under `campaign/candidates/`. It has these fields.

| Field | Required content |
| --- | --- |
| `target` | `gfx938` |
| `source_dir` | Relative dedicated source directory |
| `entry` | Relative Python entry file inside `source_dir` |
| `build_flags` | JSON list of the exact native build flags |
| `gap.hypothesis` | Mechanism and falsifiable expected effect |
| `gap.operation` | Affected operation in the original task |
| `gap.limitation` | Concrete expressibility or lowering limitation |
| `gap.supported_transforms` | Supported transforms considered and why they cannot express this mechanism |
| `gap.proposed_owner` | Candidate, IR, lowering, toolchain or another proposed maintenance owner |
| `gap.counterexamples` | Known failure domains, or a specific statement of what remains untested |
| `gap.evidence` | Nonempty distinct list of relative files in this Run's `campaign/` |

For example, an evidence list can name an own minimal Schedule and its retained
assessment. The helper saves exact diagnostic bytes as `.source` files and records
their original paths. A diagnostic is never relabelled as an admission receipt.

## Run one bounded probe

Run this command on the host from the prepared Run.

```bash
python3 campaign/hip_probe.py run --id hip-mechanism-01 --request campaign/candidates/hip-request.json
```

The ID is create-only. The helper reconciles the original prepared binding,
Compiler and gateway before work and before every stage. It reads the immutable
image and assigned HCU from the binding. CPU preparation, queue waiting and device
work spend the same original search window. The final 30 minutes remain the owner's.
Registered development also honors its existing batch STOP and `SEARCH_CLOSE`.

The helper first freezes source and gap evidence in
`campaign/engineering/hip-mechanism-01/`. It compiles in the CPU container and
retains `prepare.log`, `build.json` and artifact hashes. It then uses only the
qualified `scripts/hcu_run.py` gateway. Every stage needs its matching admission,
terminal, complete release and completion inside the original search cutoff
before the next stage starts. A failure or missing receipt stops this probe.
An unknown or incomplete native development release creates the existing batch STOP.
Fully released candidate failures and late completions remain separate failures.
The frozen source directory rejects added, removed or symlink members as well as
changed bytes. Generated files belong in the probe's separate build directory.

For Bench, the stages call the existing full 16-workload, ten-round numerical
checker, caller guards and unprofiled paired wall timer. Forward, reverse and A/A
samples keep the original fixed community baseline and timing algorithm. The raw
native timing report carries `implementation_kind=native_hip`. The frozen author
condition remains unchanged.

For registered development, `--native-manifest` is a separate device input. It is
not an emission receipt. The device adapter derives the full output ABI from the
original Workload and checks it before flattening values. `compare-native` uses
`compare_tile_outputs` for all original task-owned cases and writes `comparison.json`
inside this engineering record. The callable samples remain diagnostic.

If execution cannot fit, use a new ID and the original gap dossier to retain the
specific reason. The skip request can omit `source_dir`, `entry` and `build_flags`.

```bash
python3 campaign/hip_probe.py run --id hip-mechanism-unexecuted --request campaign/candidates/gap-request.json --skip-reason 'The remaining search window cannot fund the required build and full original checks; see the retained deadline observation.'
```

Do not use this option as a generic waiver. Name the actual budget or infrastructure
evidence in `gap.evidence`. A skip has no correctness, timing or qualification claim.

## Record the Compiler lead

Read each probe's `outcome.json`, source snapshot, build log and stage receipts.
The author context retains failures and incomplete probes as well as successful
measurements. A successful pipeline reports `measurements_retained_pending_review`.
It has no Cake precision PASS, accepted Cake outcome, nominee or score.

In this Run's evolution notes and handoff, name the native mechanism, original
semantics and rounding boundaries, observed cases and counterexamples, actual
execution-source evidence, missing Cake commitment and proposed next owner.
Distinguish measured numeric behavior from a proposed Compiler improvement.
Preserve failures and No promotion conclusions. A native-versus-community timing
observation is not a same-lease comparison against your Cake candidate or a hardware
ceiling. Keep profiling diagnostics separate and use the existing admitted
profiling path only when they fit the same search budget.

Only Cake-generated candidates continue through `campaign/evaluations`, nomination
and owner confirmation. Native records stay in `campaign/engineering` and do not
change those gates or authorize Compiler edits during the Run.
