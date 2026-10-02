# Matched engineering pilot: direct Triton versus the standalone Cake Compiler

Read root AGENTS.md, campaign/AGENTS.md, campaign/intake.json and its group plan,
campaign/deadline.json, STATUS.md/JOURNAL.md, docs/RALPH-PROFILING.md and both
installed skill entrypoints through Read. The injected intake names the DCU
profiling skill; also Read /data3/testuser01/.agents/skills/rocm-kernelwiki/SKILL.md.
Read the assigned baseline and original task definition/reference/workloads.

The group plan owns ONE original task, its community denominator, assigned HCU,
authoring arm and 3h maximum wall budget (165min search +15min handoff).
Other groups are independent. Both arms use Claude/GLM-5.3:high, the same prompt,
reference visibility, skills, data, dtype/tolerance, budget and common wall timer.
This is an engineering feasibility pilot with one run per arm/task. It is not an
Open-Cake Lab Campaign, an isolated filesystem-custody experiment, or a scientific
causal estimate. Do not invent a Lab acceptance or a treatment-effect conclusion.

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

Every accepted candidate needs bwbench.py check on all original16 workloads x10
rounds, same seeds/precision/tolerances, with current source binding. Both arms
MUST measure with the unmodified common campaign/paired_wall.py, no alternate
speed harness or CUDA-event substitute. After full gate:
  HIP_VISIBLE_DEVICES=HCU BWBENCH_TIMEOUT=900 bash campaign/admit.sh gpu IMAGE \
    campaign/results/NEW-latency-admission.json env \
    AITER_JIT_DIR=/work/.local/aiter-jit-cache TRITON_CACHE_DIR=/work/.local/triton-cache \
    python3 campaign/paired_wall.py --candidate /work/campaign/candidates/CAND.py \
    --gate campaign/results/FULL-GATE.json --output campaign/results/NEW-latency.json
This measures whole run(*inputs), warmup10,30 samples/direction, actual alternating
baseline/candidate and candidate/baseline order, A/A, input-mutation guard and
absolute wall microseconds. Robust improvement exceeds max(A/A drift,1%) with
no material regression in either direction. Include public dispatch overhead.
Do not claim an arm comparison from separate score formats or profiled time.

Record candidate order, hypotheses, first full-correct checkpoint, first robust
win checkpoint, elapsed time, GPU admissions, accepted result and any coverage
limitations. The same stopping policy applies to both arms: stop early when two
structurally distinct hypotheses have been evaluated or specifically refused and
one candidate is accepted; otherwise continue useful work to the timebox, then
hand off an honest no_robust_gain or blocked outcome. Never fill time with settled
checks. This pilot records the attained endpoint and time-to-first-valid-candidate;
it does not estimate best performance at a fully consumed3h search budget.

Write candidate sources in campaign/candidates and harnesses in campaign/tools,
Schedules in campaign/schedules and emitted artifacts in campaign/generated.
Update STATUS/JOURNAL and commit source/authoring artifacts on this arm's branch.
Raw results/logs/dataset stay ignored. Do not push/merge or message other agents.

Finish README and DONE.json covering exactly the assigned task. Each row has task,
status accepted/no_robust_gain/blocked, reason, evidence paths and profile_evidence
or a specific profile_skip_reason. Accepted rows also need candidate_source,
correctness_source_sha256 at gate, correctness_report and latency_report. Common
latency JSON must bind current sources and all16 UUIDs. A Cake accepted row also
needs cake_artifacts listing every executed generated receipt, with kernel/library
coverage described in README. The owner independently replays each Schedule and
checks exact emitted bytes before accepting the Cake outcome. Do not edit the
completion checker to bypass failures. Confirm released receipts/no live owned
containers, then return; the derived Ralph flow stops at validated completion.
