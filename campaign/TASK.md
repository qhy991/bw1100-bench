# Optimize the assigned original SOL-ExecBench tasks

Use fresh-session humanize2 Ralph with Claude/GLM-5.3 on BW1100-1. Read root
AGENTS.md, campaign/AGENTS.md, campaign/intake.json, the referenced group plan,
campaign/deadline.json, STATUS.md and JOURNAL.md each round. In the first round
use the Read tool on the profiling skill path in the injected intake, on
docs/RALPH-PROFILING.md, and on
/data3/testuser01/.agents/skills/rocm-kernelwiki/SKILL.md. Record actual reads.
The official installed Ralph is unchanged; this local derived flow adds a
validated DONE check to stop after completion instead of repeating terminal work.

The plan owns the task order, frozen community baseline source/hash, smoke UUID,
HCU and per-task timebox. All seven baselines already passed original 16 workloads
x10 rounds on gfx938 with the pinned image. Read baselines/README.md and the
source before optimizing. Fresh smoke each baseline; retain its full qualification
provenance. Do not repeat settled baseline full gates simply to fill time.
Community primitive compositions are legitimate denominators for tasks lacking a
whole shipped callable; they do not establish a universally fastest implementation.
Never substitute reference.py or a reconstructed weak denominator.

Candidate custom Triton/HIP/C++ and GEMM implementations ARE ALLOWED. The requirement
for unchanged community source applies to BASELINES, not optimization candidates.
Implement original semantics rather than restricting yourself to library glue.
Reuse historical sources only as prior art, copy into this campaign and requalify.

Semantic constraints by task:
- L1/058: stable expert bucketing, original token order, exact int32 indices and
  257 prefix boundaries. Denominator is native stable sort + searchsorted.
- L1/001: training GQA backward, original softmax/dropout precision and reductions;
  ATen training backward plus vendor GEMMs is the denominator.
- L2/056: ten original gradient outputs, supplied intermediates, BF16 rounding
  chain and FP32 norm gradients; native ATen backward/GEMM denominator.
- L1/011: supplied inv_freq and attention scalar are authoritative. Baseline is
  shipped Transformers 5.16.1 RoPE with qualified eager/compiled shape dispatcher.
  Earlier weak-baseline speedup is not evidence for this denominator.
- L2/018: head_dim72 and ragged attention; BF16 scaled scores BEFORE softmax and
  BF16 probability BEFORE PV are mandatory. Ordinary FP32 online FlashAttention
  is not equivalent. Denominator is qualified materialized FlagGems composition.
- L2/024: true FP32 expert projection/activation/combine with final BF16 output,
  top8 and 256 experts. BF16 fused inference MoE is not equivalent. Input ~12GiB.
- L2/060: cyclic whole-head repetition, zero padding to64, original BF16 norm,
  FP32 intermediates and per-site raw/cumulative gates. Baseline unchanged FLA
  state JIT uses BV32/warps4/stages1 for 64KiB LDS. Every baseline and candidate
  correctness/profile/timing command must pass TRITON_F32_DEFAULT=ieee and
  FLA_TRIL_PRECISION=ieee; keep the same persistent cache across these stages.

For each task collect a bounded baseline rocprof diagnostic through
campaign/admit.sh profile (which calls scripts/rocprof.sh) before making GPU
bottleneck claims. Use original generated smoke and a representative larger
UUID; one counter group <=6, actual target kernel regex, CSV validator and
REPORT.md. If unavailable or not decision relevant, preserve failed command,
terminal receipt and concrete task-specific reason; do not silently skip.
Name one falsifiable optimization hypothesis before each mechanism change.
For a claimed counter improvement, also profile the accepted candidate on the
same workload. Profiling duration is never a speed score. Warm JIT separately;
use TRITON_CACHE_DIR=/work/.local/triton-cache and, where relevant,
AITER_JIT_DIR=/work/.local/aiter-jit-cache. GPU subprocesses inherit only command
environment; pass needed variables explicitly via env after the receipt argument.

All GPU work, including checks/profiling/timing, uses ONLY campaign/admit.sh
gpu|profile, with HIP_VISIBLE_DEVICES set to intake.hcu, BWBENCH_TIMEOUT<=900,
unique receipts under campaign/results or .local/profile, and pinned image
sha256:3ad0ae7192b8f9bafdf5b48fc414f8785f3c2463005e6b25290b7f75146ff260.
Never use HCU0 or another group's HCU; never stop another container/service.
CPU container work uses scripts/dtk.sh cpu IMAGE /usr/bin/timeout -k 10s 180s ...
to ensure a bounded inner process even if the outer tool is interrupted.

Respect each task's timebox from its first work, move to the next with an honest
checkpoint, and retain the final15 minutes of the3h ceiling for handoff. No new
GPU admissions in that final window. No repeated settled tests or idle checks to
consume the budget. Stop early if all assigned tasks have evidence-backed outcomes.

Before accepting any candidate run bwbench.py check on all original16 workloads
x10 rounds with original seeds/tolerances/precision and no modifications to
reference inputs. Measure complete run(*inputs), separately from profiling,
using pinned gen_inputs, identical original inputs, warmup, input-mutation guard,
absolute microseconds, forward/reverse paired orders and baseline A/A drift.
No material regression in either direction; a win must exceed max(A/A drift,1%).
Public shape/dtype dispatch is allowed; include dispatch overhead. Do not cache
answers or choose a route from tensor contents/expected outputs.

Update STATUS/JOURNAL with factual evidence paths and decisions. Commit candidate
sources and harnesses on this experiment's branch; raw receipts stay ignored.
To finish, write campaign/README.md and campaign/DONE.json covering EXACTLY the
assigned task IDs. Every row needs task, status (accepted/no_robust_gain/blocked),
nonempty factual reason, evidence (list of relative file paths), and either
profile_evidence (relative files) or a specific profile_skip_reason. Negative
outcomes require the concrete tested hypothesis or external blocker, not a
generic assertion that custom kernels are forbidden.

Accepted rows ALSO need candidate_source, correctness_source_sha256 (hash at full
gate), correctness_report, latency_report. The latency JSON must bind current
candidate_source_sha256 and baseline_source_sha256, and rows for all16 UUIDs.
Each row has workload_uuid, inputs_unmutated_after_measurement=true, baseline_us and candidate_us
objects with median/forward_median/reverse_median, and aa_control_baseline_us with
abs_drift. Keep original UUID in each row. The completion checker rejects missing
gates, altered baseline, stale source binding, timing regressions, or live owned
containers. If rejected fix the handoff evidence; do not edit the checker.
