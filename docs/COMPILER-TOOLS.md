# Pinned Compiler tools for hmz authors

Bench and registered development Runs use the same author-tool adapter. Their
existing evaluators, deadlines, final nomination rules and host owner remain
distinct where their task contracts differ. This adapter starts no Provider or
device work and writes no acceptance or Run terminal.

## What reaches the author

Preparation invokes `campaign/compiler_tools.py freeze` against the clean
`.deps/cake-ir` checkout. The Compiler's `TRANSFORMATIONS` registry and Lab's
`transformation_surface` produce `campaign/compiler-api.json`. It is tracked in
the prepared commit and binds the condition's Compiler. Both launchers check it
before intake; both actual hmz flow functions append its contents on every agent
call, including after context compaction. A changed catalog or condition is
refused. The same projection code serves both Compiler arms; a new API does not
appear in an older Compiler's catalog.

The external engineering task permits these public transforms subject to the
selected Compiler's guards. The catalog is not an applicability prediction or
new hardware qualification. It includes target-specific entries; calling a Metal
pass on gfx938 can be refused. Automatic lowering cleanup, such as live constexpr
pruning, does not require a transform action.

## Inspect, transform, and evaluate

Author or select a permitted own Cake source, then inspect its exact stage names:

```bash
python3 campaign/compiler_tools.py inspect --parent campaign/candidates/starter.py
```

Independent Bench Runs have no starter or inherited development candidate. Use a
Schedule you authored under `campaign/schedules/` or `campaign/candidates/`.
Development Runs retain their canonical starter and separately declared incumbent.
Use the incumbent path from `campaign/inherited/provenance.json`.

Write a request under `campaign/candidates/`. Copy the actual stage from `inspect`;
the example below assumes `inspect` returned `gemm_fp32`. Replace that name with
the actual stage; the parameters are a candidate to test, not a recommended optimum:

```json
{
  "action": "transform",
  "parent": "campaign/candidates/starter.py",
  "transformation": "specialize_fp32_contraction",
  "parameters": {
    "stage": "gemm_fp32",
    "row_tile": 16,
    "column_tile": 64,
    "k_tile": 64,
    "num_warps": 4,
    "num_stages": 1,
    "schedule_id": "mma_trial_1",
    "entry_point": "mma_trial_1"
  }
}
```

```bash
python3 campaign/compiler_tools.py transform --id mma-001 --request campaign/candidates/request.json
python3 campaign/compiler_tools.py verify --id mma-001
```

The adapter uses the pinned Lab `resolve_action`, not a second matcher or rewrite
implementation. A parent must be a Cake Schedule/Program, not the final Torch wrapper. It supplies only a snapshot of the explicitly named own parent.
Files outside the Run, Compiler examples, and symlinks escaping the Run are not
parents. A Bench Run cannot use development-inherited material. A tool-generated
parent must first replay as its original Program or stage.

Each create-only action directory under `campaign/compiler-actions/` retains:

- `request.json`: Compiler identity, action, parameters and observed start time.
- `parent.json`: exact parent input supplied to the existing action resolver.
- `result.json`: applied/refused result, original resolver identities and reason.
- `program.json` and `stage-000.json`, etc.: complete generated Program and its
  stage Schedules, only when applied.

Refusals consume the existing wall time and remain in author feedback. An
interrupted directory remains unknown, cannot reuse its id, and is not silently
recreated. `verify` replays the retained request and parent, checks the Program and
all generated stages, and writes nothing. Later edits to the original author file
do not change the retained snapshot. New transform requests stop at the existing
search deadline or after the development owner approves early closure; read-only
replay remains available afterward.

For Bench, emit each required stage with the existing bridge and construct the
complete original-ABI callable. The bridge records and verifies the originating
tool stage in its emission receipt. Emit/evaluate the generated stage at its original
path; copying it elsewhere is a directly authored snapshot without automatic origin
binding. Then use the original `campaign/evaluate.py`
path and its caller, precision, numerical and timing gates.

```bash
python3 campaign/cake_bridge.py emit campaign/compiler-actions/mma-001/stage-000.json --output-dir campaign/generated/mma-001
```

For a registered development task, its existing evaluator accepts the generated
Schedule JSON as well as Cake Python:

```bash
python3 campaign/development_evaluate_owner.py --candidate campaign/compiler-actions/mma-001/stage-000.json --id mma-001
```

The retained candidate keeps its original `.json` or `.py` format. Confirmation
rechecks the exact original ABI, emission and declared transform source. Inherited
JSON candidates also retain their format in a later prepared Run. This does not
add multi-stage development evaluation: each submitted Schedule must still match
the original complete task ABI. Do not submit a partial stage as a full task.

Directly authored candidates remain valid. Use the tool for explicit Compiler
rewrite calls so the action is recorded. Tool records establish CPU derivation,
not that a wrapper executed every declared artifact or that the kernel is correct
or faster. Existing independent wrapper/rounding review remains required. These
files are ordinary external engineering records, not native EvidenceStore events,
proof of filesystem isolation, or a claim that all possible direct Python calls
were observed. The prompt shows the latest 20 action summaries and an omitted
count; complete records stay in the Run.

## One engineering path, existing responsibilities

| Owner | Responsibility after this change |
| --- | --- |
| Compiler registry and guards | API declarations, matching, rewriting and lowering |
| Lab pure author functions | API projection, parent parsing and action resolution |
| External preparation and hmz flows | Frozen catalog delivery and Run-local tool observations |
| Existing Bench/development owner | Budget, stopping, nomination, confirmation and terminal records |
| Independent Bench and HCU gateway | Original references, measurement and device admission/release |

The native Lab execution loop is not instantiated by this adapter. Native
`tools/launch_task.py`, task runtime composition, Study execution/audit, and
historical replay still consume it. They must not be deleted merely because
BW1100 engineering uses hmz. No new queue, common acceptance schema or parallel
result writer is added. Lab contracts are reused without pretending an external
directory is a native sealed Run.

The local C550 Bench currently exposes its independent correctness checker, not
this Hygon launch owner. A C550 successor must bind its actual author entry,
Compiler, original task and qualified MACA evaluator before adoption. Do not copy
HCU selectors, claim HIP/MACA measurement equivalence, or migrate an active Run.
The catalog/action mechanism itself does not grant device support.

## Adoption and software checks

Select this adapter commit in a new frozen Bench plan or development allocation.
Preparation creates the catalog once. Do not copy this file or prompt into an
already prepared/running cohort, reset deadlines, or replace failed attempts.
Catalog delivery is a scaffold change and must be recorded in version comparisons.

Run existing CPU contracts with the pinned SOL source/dependencies. To include
real Compiler integration tests, provide a clean successor checkout:

```bash
BWBENCH_CAKE_SOURCE=/absolute/clean/cake-checkout python3 -m unittest discover -s tests -v
```

Those checks execute real CPU transformations and both flow functions with an
in-memory author fixture. They do not invoke a model, compile native GPU code or
establish a hardware speedup. Production runs still require their usual admission.

Set `BWBENCH_CONTROL_SOURCE` to a clean pre-feature checkout to additionally verify
that the old condition does not advertise the successor-only MMA pass.
