# Three-hour profiled Ralph campaign with qualified community baselines

Run the installed official humanize2 `ralph_loop` using Claude/GLM-5.3 on
BW1100-1 node4. Start with `campaign/AGENTS.md`, `campaign/STATUS.md`,
`campaign/JOURNAL.md`, `campaign/deadline.json`, root `AGENTS.md`,
`docs/RALPH-PROFILING.md`, and both installed skill entrypoints named in the
launch intake. Read the exact DCU profiling skill through the Read tool in
the first round. Record its path/hash and the Read in JOURNAL; an intake
receipt alone is not proof of agent use. Do not change the suite, source locks,
reference, workload inputs, tolerances, admission runner or installed skills.

Optimize these three unchanged tasks in this order:

1. `L1/069_rms_norm`: denominator `baselines/l1_069_vllm_aiter_fused.py`.
   This is the shipped vLLM 0.29.0 -> AITER 0.1.5 fused residual RMSNorm path,
   previously full-correct on 16 original workloads x 10 rounds. Preserve the
   original residual BF16 rounding. Review historical accepted candidates only
   as prior source evidence; copy any reused candidate into this campaign and
   requalify it. Describe gains as AITER-relative, not globally best-baseline.
2. `L1/048_fused_gate_up_projection_with_swiglu`: denominator
   `baselines/l1_048_flaggems_gelu.py`, two library GEMMs followed by the shipped
   Hygon FlagGems 5.4.0dev fused GELU-tanh-and-mul. The actual reference is
   GELU-tanh, despite the title. Keep the intermediate BF16 projection rounding.
   The denominator is the whole original callable, including both projections;
   an activation-only measurement cannot become whole-task speedup.
3. `L2/035_convnextv2_block_with_grn`: denominator
   `baselines/l2_035_timm_convnextv2.py`, timm 1.0.28 ConvNeXtBlock with GRN
   and the supplied original weights bound by functional_call. Optimize the
   FP32 block while preserving layouts, both epsilons and the supplied weights.
   Distinguish a measured Python binding/dispatch gain from a GPU-kernel gain.

These are community implementations with prior full gfx938 correctness;
that does not prove universal performance optimality. Inspect the baseline
source before changing any kernel. If another shipped, exact-semantics path
is available, qualify it before treating it as a stronger denominator. Never
weaken the denominator to the Torch reference or a handwritten formula.

The total wall-clock budget is three hours shared by all tasks. Use at most
50 minutes for task 1, 50 minutes for task 2, 65 minutes for task 3, and retain
the final 15 minutes for terminal checks and a factual handoff. Stop starting
GPU jobs in those final 15 minutes. At each timebox preserve the checkpoint,
including negative results, then move on. Do not spend this optimization
budget repairing the seven unqualified baseline tasks; their next actions
are in `campaign/BASELINE-QUEUE.md` for a separate qualification phase.

For each task, fresh smoke-check the frozen baseline then run all 16 original
workloads x 10 rounds through `bwbench.py check`. Once qualified, collect one
bounded profile of the actual baseline callable using original generated
inputs, representative smoke and larger workload UUIDs from `bwbench.py audit`.
Keep each hardware counter request to one group of at most six metrics, bind
the expected observed target kernel name, and retain CSV, validation JSON,
admission/terminal receipts, source hashes and a REPORT.md. Analyze using the
installed skill helper. If the question requires write traffic or cache
counters, collect a separate group rather than assume missing metrics are zero.

For AITER, first prewarm the same source/image/HCU outside rocprof with
`AITER_JIT_DIR=/work/.local/aiter-jit-cache`, then use that persistent cache
inside the profile. First-call JIT under rocprof previously corrupted native
target discovery. Also account for FlagGems/Triton and MIOpen compilation;
exclude compilation/import/source-verification setup from steady-state timing.
Use create-only paths under `campaign/results/` and `.local/profile/`.

Implement candidates only under `campaign/candidates/` and harnesses under
`campaign/tools/`. Candidate optimizations are owned by Ralph. The owner
prepared the contract, baseline bindings and launch route. Before every GPU
mechanism change, name one falsifiable hypothesis and its expected counter
effect. Repeat bounded profiling if the claimed GPU bottleneck changes, and
profile the accepted candidate on the same workload when making a counter
comparison. If profiling fails, preserve the exact failed command and terminal
receipt, record `profile_unavailable` with a concrete cause, and leave the GPU
bottleneck claim unqualified. An empty filtered torch.profiler list is not a
completed diagnostic and must not silently substitute for rocprof.

Run each candidate through the same complete correctness gate. After baseline
and candidate pass, measure their complete `run(*inputs)` callables without a
profiler using pinned SOL-ExecBench `gen_inputs` and frozen seeds, identical
original inputs, separate warmup, absolute median microseconds, paired
forward/reverse order and A/A controls. Preserve both source hashes. Report
every workload cell's drift and ratios; a combined average cannot hide a
regression or failed direction. A robust win must exceed max(A/A drift, 1%)
in each reported direction. Shape-based dispatch may use public shapes/dtypes
only; include dispatch cost. Never read outputs or expected answers to choose
a route. No timing claim is accepted from profiled duration.

Use HCU1 only after observed-idle admission. HCU0 previously held a timed-out
FlagGems container; never assume it is released or stop another service. The
only GPU entry is this bench's `scripts/dtk.sh gpu` / `scripts/rocprof.sh`.
Each command must have a new receipt path and `BWBENCH_TIMEOUT` at most 900s.
The pinned image ID is
`sha256:3ad0ae7192b8f9bafdf5b48fc414f8785f3c2463005e6b25290b7f75146ff260`.
For example, substitute the original task/adapter and a new result name:

```bash
HIP_VISIBLE_DEVICES=1 BWBENCH_TIMEOUT=900 bash scripts/dtk.sh gpu IMAGE \
  campaign/results/UNIQUE-admission.json \
  env AITER_JIT_DIR=/work/.local/aiter-jit-cache \
  python3 bwbench.py check --task TASK --device cuda:0 \
  --candidate /work/baselines/ADAPTER.py --workloads all --rounds 10 \
  --output campaign/results/UNIQUE-check.json
```

Read the profiler invocation in `docs/RALPH-PROFILING.md` before collecting.
Each final task row in STATUS must identify baseline source/version, full
correctness receipts, profile evidence or explicit cause, candidate source,
paired timing status, decision and next action. Commit accepted source changes
on this experiment branch; keep raw results ignored. Update JOURNAL as facts
change, and leave a final handoff in `campaign/README.md`.
