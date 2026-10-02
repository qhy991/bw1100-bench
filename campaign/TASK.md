# Fixed3h best-result comparison: direct Triton versus Cake Compiler

Read root AGENTS.md, campaign/AGENTS.md, campaign/intake.json and its group plan,
campaign/deadline.json, STATUS.md/JOURNAL.md, docs/RALPH-PROFILING.md and both
installed skill entrypoints through Read. The injected intake names the DCU
profiling skill; also Read /data3/testuser01/.agents/skills/rocm-kernelwiki/SKILL.md.
Read the assigned baseline and original task definition/reference/workloads.

The group plan owns ONE original task, its community denominator, assigned HCU,
authoring arm and protocol.json: exactly the same3h opportunity, including
model authoring, compilation, GPU checks, timing and profiling. The last15min
is reserved equally for handoff; GPU admission must fit the remaining budget.
Other groups are independent. Both arms use Claude/GLM-5.3:high, the same prompt,
reference visibility, skills, data, dtype/tolerance, budget and common wall timer.
This is an exploratory matched-budget comparison with one run per arm/task.
The primary endpoint is the best qualified performance attained within3h, not
the first successful candidate. Record the same time-performance checkpoints.
It is not an Open-Cake Lab Campaign or a fully isolated scientific experiment.
No statistical treatment-effect estimate is justified by one pair per task.

Only the authoring surface differs:
- direct_triton: write custom kernels directly in Triton. Do not use Cake Compiler,
  generated Cake code, HIP/CUDA/C++ or previous optimization winners.
- cake_ir: write complete gfx938 Schedules and use the unchanged standalone Cake
  Compiler at .deps/cake-ir (commit in plan) via campaign/cake_bridge.py to assess
  and lower them. ALL newly authored custom GPU computation must come from
  unchanged Compiler emission. No hand-written Triton/HIP/C++ kernel, patch of
  generated source, or fallback pretending to be Cake. Existing shared community
  GEMMs/operators remain allowed for both arms; explicitly record that coverage.

This is known-kernel reproduction/optimization: both arms may read the original
high-level reference and frozen community baseline implementation. Neither may
read any previous campaign's custom candidate, winning source, timings, logs or
other arm's work. Stay inside this root, the installed skills, and pinned library
source. Do not search/list other experiment directories. A prior winner is not a
seed for this fresh pilot. Do not change the Compiler, gateway, baseline, suite,
source locks, reference or tolerances. No new project-wide dependency is required.

Task semantics:
- L1/069: residual + RMSNorm. Residual addition is rounded to BF16 before the
  FP32 normalization chain; preserve supplied weights and epsilon. Denominator
  baselines/l1_069_vllm_aiter_fused.py is the actual vLLM->AITER fused path.
- L1/048: two projection GEMMs followed by GELU-tanh times up, despite the title
  saying SwiGLU. Preserve intermediate BF16 projection rounding. Denominator
  baselines/l1_048_flaggems_gelu.py includes both GEMMs and the activation.
- L1/001: GQA training backward, both original gradient outputs; FP32 softmax
  backward chain and original dropout/masking/group-reduction rounding. Denominator
  baselines/l1_001_aten_training_backward.py uses native backward + vendor GEMMs.

Explore structurally distinct candidates/hypotheses. Cake uses public Compiler
+Verifier before GPU: construct/assess, retain localized Findings and static
analysis, then lower only eligible candidates. Rank using supported resource facts;
Hygon calibration gaps stay unmodeled, not borrowed from another target. A refusal
is a measured capability limit to report, not permission to edit the Compiler.
Read .deps/cake-ir/docs/IR_GUIDE.md and compiler/AUTHORING_CONTRACT.md in the Cake
arm; examples in corpus/schedules are syntax resources, not ready-made task winners.
CPU authoring helper:
  bash scripts/dtk.sh cpu IMAGE /usr/bin/timeout -k 10s 180s \
    python3 campaign/cake_bridge.py emit campaign/schedules/NEW.json \
    --output-dir campaign/generated/UNIQUE
Retain all Schedule/assessment/source/receipt files. candidate.py may load an
already generated callable using from campaign.cake_bridge import load, then
load('campaign/generated/UNIQUE/receipt.json'). Use only public metadata dispatch;
reshape/transpose/output binding is adapter work, tensor-content computation is not.
Compiler loading/lowering is a CPU authoring phase outside all GPU timing.

Fresh-smoke the frozen baseline (prior full160 qualification is settled). Before
GPU-bottleneck claims collect a bounded baseline rocprof diagnostic, then profile
any accepted candidate on the same original workload with the same counters.
Use campaign/admit.sh profile, matched target kernel rows, <=6 metrics/group,
CSV validation and REPORT.md. Record concrete task-specific failure/skip reasons;
profiled duration is never a score. Prewarm AITER/Triton outside rocprof with
persistent AITER_JIT_DIR=/work/.local/aiter-jit-cache and
TRITON_CACHE_DIR=/work/.local/triton-cache passed explicitly via env in commands.

GPU entry ONLY campaign/admit.sh gpu|profile, assigned HCU from intake,
BWBENCH_TIMEOUT<=900 and unique admission/output paths. Image:
sha256:3ad0ae7192b8f9bafdf5b48fc414f8785f3c2463005e6b25290b7f75146ff260
Do not use HCU0 or another group's GPU or stop any existing service. All CPU
container work must have /usr/bin/timeout -k 10s 180s inside the container.
No new GPU admissions in final15min. Keep rejected and failed artifacts.

Search probes may use smoke or bounded shape screens through admission, but
ONLY campaign/evaluate.py promotes a candidate into the best-result ledger. It
freezes a self-contained candidate into campaign/candidates/frozen_ID.py, replays
Cake emissions when applicable, runs all16 original workloads x10 rounds, then
uses the SAME unmodified paired_wall.py for both arms. Do not add an alternative
speed harness or edit frozen candidates/receipts. Your candidate must be one Python
source besides pinned community libraries, cake_bridge, and declared emission
receipts; move custom helper code into that source before canonical evaluation.

Host invocation (correct HOME inherited from launcher):
  python3 campaign/evaluate.py --candidate campaign/candidates/CAND.py --id UNIQUE \
    --profile-evidence .local/profile/BASELINE/REPORT.md
Cake adds one --cake-artifact campaign/generated/UNIQUE/receipt.json for EACH
executed emission. Profile the current best on the same representative workload
when making counter claims; record an actual skip reason if unavailable.
The common evaluator owns HCU, environment, full gate,30sample paired wall timing,
A/A and source binding. It writes campaign/evaluations/ID/outcome.json. Only status
accepted can enter the incumbent ledger. A full correctness gate alone is not an
accepted performance result. Declare any explicit numerical/capability refusal.

Ranking is predeclared in protocol.json: geometric mean across all16 original
UUIDs of the CONSERVATIVE directional speedup min(forward_ratio,reverse_ratio).
Require at least one robust win and no material per-cell regression, using
max(A/A drift,1%). Include the complete callable and all public dispatch cost.
The frozen community baseline is the initial incumbent (speedup1), never weaken it.
Do not time profiled durations, cache outputs or hide regressions in an average.

Record candidate order and one falsifiable hypothesis per mechanism change.
After every accepted evaluation, update the incumbent if its canonical ranking
improves. Reaching a first valid candidate or a first win DOES NOT END THE RUN.
Continue distinct improvements, resource/tile/launch tuning and public dispatch
until the fixed search window expires. Avoid duplicate settled evaluations;
negative results prune that hypothesis, not the whole experiment. Record first
full-correct and first robust-win elapsed time as secondary endpoints. Do not
read previous pilot or historical winners; these successors start fresh.

At30,60,120 and165minutes record the best accepted candidate available at that
time. These checkpoints are derived from immutable outcome completion times,
not backfilled assertions. Preserve all successful and failed outcomes. The
primary180minute endpoint is the highest ranked qualified candidate completed
within the budget. The final15min permits CPU handoff, not new GPU jobs. No
candidate discovery or editing after deadline gets credited to the3h endpoint.

Before stop, commit authoring sources and write a factual README/JOURNAL. At the
last2minutes, run python3 campaign/finish.py ONCE to derive ENDPOINT.json and
DONE.json from the canonical accepted ledger. The flow also invokes it between
turns if needed. Never author an early DONE or change the stopping checker.
finish.py selects the best attained result; earlier successes remain checkpoints.
Independent same-HCU comparison of the two frozen endpoints is a separate
confirmation phase after budget, with no candidate edits or extra search credit.

Write candidate sources in campaign/candidates and harnesses in campaign/tools,
Schedules in campaign/schedules and emitted artifacts in campaign/generated.
Update STATUS/JOURNAL and commit source/authoring artifacts on this arm's branch.
Raw results/logs/dataset stay ignored. Do not push/merge or message other agents.

The owner finalizer owns ENDPOINT.json and DONE.json. Accepted outcomes need a
full original gate, canonical timing, source binding and executed Cake receipts.
The owner independently replays Schedule->emission before accepting Cake results.
If there is no accepted custom candidate, retain the baseline incumbent and the
actual failures/refusals; do not claim an optimized artifact. Search failures and
provider/infrastructure faults are separate. Keep within-budget artifacts intact
for common independent confirmation after freeze.
