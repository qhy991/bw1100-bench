# Targeted3h experiment: Gate/up independent replication or GQA continuation

Read root AGENTS.md, campaign/AGENTS.md, intake.json, its group plan, protocol.json,
deadline.json, STATUS/JOURNAL, docs/RALPH-PROFILING.md and the installed profiling
and ROCm Wiki skills through Read. Use Claude/GLM-5.3:high and the assigned HCU.
The plan's experiment_kind determines what this run means; do not mix the modes.

FRESH_REPLICATION (groups a/b/c/d, L1/048): independently optimize the original
Gate/up GELU-tanh task from its shipped-community baseline only. Do not read ANY
prior custom candidate, old experiment history, previous score, other replicate
or other arm. The same untouched baseline and high-level reference are visible
in both arms. These are two new independent3h pairs testing the ORIGINAL3h
question. They are NOT replications of the6h continuation endpoint. No old
winner is seeded, and no later-winning algorithm is supplied in this task.

CONTINUATION (groups e/f, L1/001): read campaign/prior/seed.json, ENDPOINT,
JOURNAL and STATUS and the selected parent. Read campaign/ANALYSIS-GQA.json;
these are shared high-level hypotheses, not another arm's code. Read-only access
to your OWN parent root is allowed for missing evidence. Never read another arm.
Initialize the byte-identical inherited winner through evaluate.py as seed00
(any unique retry ID is allowed; initializer identity is determined by source,
not ID). Then search new mechanisms and compare them directly with the parent.
Report incremental3h and cumulative9h separately; this is not an independent run.

Common authoring treatment:
- direct_triton writes custom Triton kernels directly; no Cake or handwritten
  HIP/CUDA/C++ GPU computation. Existing shared community operators are allowed.
- cake_ir writes gfx938 Schedules, assesses and lowers through the unchanged
  Compiler at .deps/cake-ir (commit in plan) using cake_bridge.py. All custom GPU
  computation must be unchanged Compiler emission. Shared library calls are
  allowed with coverage recorded. Do not patch generated source or the Compiler.
  Read IR_GUIDE.md and compiler/AUTHORING_CONTRACT.md; corpus examples are syntax
  resources. CPU prefilter localized Findings and supported resource analysis
  before GPU. Missing timing calibration remains unmodeled.

Original contracts:
L1/048 has two BF16 projections, GELU-tanh (despite SwiGLU in title), original
projection/activation rounding and original output shape. Denominator is
baselines/l1_048_flaggems_gelu.py including both GEMMs and activation.
L1/001 has both GQA training gradients, original mask/dropout scale, FP32 softmax
backward chain and GQA reduction/rounding. Denominator is the native ATen
training-backward/vendor-GEMM composition in the plan. Never substitute a weak
reference denominator, change precision/tolerance, or silently approximate a
required FP32 intermediate. All16 workloads x10 original rounds remain mandatory.

CUDA graph replay/persistent scratch storage is allowed equally if every call
recomputes from CURRENT inputs and preserves the output contract. No answer
memoization or tensor-content/expected-output route selection. Public metadata
dispatch is allowed and included in complete-call timing. All graph computation
in Cake must still use unchanged emitted kernels or recorded community calls.

Budget: model authoring, CPU compile, GPU checks, profiling and timing all count
inside the same3h opportunity. Final15min are reserved for CPU handoff, no new
GPU admission. The outer process timeout enforces the deadline. First success is
not the stopping rule; continue meaningful distinct improvements. Preserve all
failed/refused attempts. Avoid stale hash rituals, repeated settled screens and
model sleep loops. Search exhaustion is an agent-reported limit only.

Fresh-smoke the unchanged baseline; its original full qualification is settled.
Collect bounded actual baseline rocprof via campaign/admit.sh profile (one <=6
counter group, real kernel regex, validated CSV+REPORT) before bottleneck claims.
Profile the best candidate on the same workload to support counter claims, or
retain the exact failed command and a specific unavailable/not-relevant reason.
Profiled durations are not speed scores. Prewarm AITER/Triton separately using
persistent AITER_JIT_DIR=/work/.local/aiter-jit-cache and
TRITON_CACHE_DIR=/work/.local/triton-cache passed explicitly via env in commands.

GPU entry ONLY campaign/admit.sh gpu|profile, assigned HCU, unique outputs and
BWBENCH_TIMEOUT<=900. Image is pinned in intake. Do not use HCU0 or stop services.
CPU work uses scripts/dtk.sh cpu IMAGE /usr/bin/timeout -k10s180s ... (use separate
arguments -k 10s 180s). Keep dataset/raw results/logs ignored and historical.

Candidate authoring lives in campaign/candidates (one self-contained Python
source besides pinned libraries, immutable parent fallback where applicable,
cake_bridge and declared emission receipts). Schedules/generated artifacts live
in campaign/schedules and campaign/generated. Emit with a unique output-dir:
  bash scripts/dtk.sh cpu IMAGE /usr/bin/timeout -k 10s 180s \
    python3 campaign/cake_bridge.py emit campaign/schedules/NEW.json \
    --output-dir campaign/generated/UNIQUE
Load pre-emitted callables using campaign.cake_bridge.load in the adapter; no
Compiler work inside GPU timing. Keep all receipt/assessment/source associations.

ONLY canonical evaluate.py promotes a candidate:
  python3 campaign/evaluate.py --candidate campaign/candidates/CAND.py --id UNIQUE \
    --profile-evidence RELATIVE/REPORT.md
Cake also supplies --cake-artifact RECEIPT for every executed emission. Evidence
paths must exist BEFORE the call; a specific profile-skip-reason is allowed when
justified. Run evaluate.py on the HOST, never inside GPU admission: it owns all
GPU admissions. It freezes source, verifies emissions, performs full160 gate,
then common complete-call wall timing (warmup10,30samples/direction, actual A/B
and B/A, baseline A/A, mutation guard and source binding).

Fresh replication ranks conservative community-relative geomean across all16
UUIDs, requiring robust improvement >max(A/A drift,1%) and no material per-cell
regression. Continuation additionally compares with the inherited parent on the
SAME inputs using the SAME timer; it ranks parent-relative geomean and blocks
material regression from both denominators. An unchanged seed is an initializer,
not a newly optimized candidate. Public-shape fallback to immutable parent is
allowed; include dispatch cost and dependency binding. Record time-to-first-full-
correct and first improvement separately from best performance at3h.

Update STATUS/JOURNAL and commit candidate/Schedule/harness sources on this root's
branch; do not push/merge or message other tasks. Checkpoints30/60/120/165/180
come from immutable evaluation epochs. At last2minutes, finish.py derives
ENDPOINT/DONE from the ledger; never author early DONE or change the checker.
No after-budget edits get search credit. Independent common-HCU confirmation is
post-budget and never additional search.

If at least two NEW structural hypotheses have real evidence and an accepted
candidate/parent initialization exists, SEARCH-EXHAUSTED.json may name reason
and new_hypothesis_evidence (>=2 relative files from THIS run). The controller
holds the incumbent without repeated model calls. It is not a global optimum
proof or permission to invent a result. Otherwise continue useful search.
