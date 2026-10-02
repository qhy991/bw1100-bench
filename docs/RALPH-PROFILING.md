# Diagnostic profiling gate for BW1100 Ralph operator campaigns

The source-locked SOL-ExecBench reference is the correctness oracle. A
candidate speed claim still requires the original complete device correctness
gate and paired **no-profiler** timing with forward/reverse order and A/A
controls. A rocprof run is a separate diagnostic lane, never its score.

Before a fresh Ralph optimization round edits a GPU kernel, it must read the
installed `dcu-rocprof-report-skill/SKILL.md` and the relevant source/workload
contract. Once a community baseline is correct on gfx938, collect one bounded
profile of its actual callable or record a specific `profile_unavailable` or
`not_decision_relevant` reason. Repeat profiling when a new hypothesis claims a
different GPU bottleneck. For a pure Python dispatch hypothesis, record the
GPU kernel count/identity if available and use paired callable latency for
the host-side mechanism. An empty filtered `torch.profiler` list is not a
diagnostic profile or evidence that rocprof is unavailable.

Use **only** this repository's `scripts/dtk.sh gpu` admission. The skill's
standalone `profile_container.sh` launches a separate raw Docker container and
does not own this suite's HCU lock or terminal receipt. `scripts/rocprof.sh`
adapts the skill's verified rocprof v1 command to the existing gateway. Keep
at most one hardware counter group, approximately six metrics, per run. For
example, after freezing an original workload and preparing an exact-callable
harness under ignored `.local/profile/<run>/`:

```text
pmc: Wavefronts VALUInsts SALUInsts SFetchInsts FETCH_SIZE GPUBusy
```

```bash
HIP_VISIBLE_DEVICES=1 BWBENCH_TIMEOUT=600 \
  bash scripts/rocprof.sh IMAGE \
  .local/profile/RUN/admission.json \
  .local/profile/RUN/pmc.txt \
  .local/profile/RUN/reports/metrics.csv \
  'Rmsnorm2dFwd' \
  python3 /work/.local/profile/RUN/harness.py
```

Substitute the pinned image, create-only run name and selected observed-idle
HCU. The harness must use an original source-locked workload and call only the
arm whose bottleneck is being diagnosed, with JIT warmup accounted for.
`scripts/rocprof.sh` fails when the CSV lacks a requested metric or the
expected kernel row, and creates `metrics-validation.json` with both input
hashes and the matched-row count. Also inspect the matching
`admission-terminal.json` for exit 0 and an observed released HCU. Analyze
the CSV with the skill's `helpers/analyze_csv.py` and
record the source hash, image ID, workload UUID, selected HCU, job ID, actual
counter values, and a bounded conclusion. A counter-backed GPU bottleneck
claim needs those artifacts. If collection fails, retain the failure receipt
and leave the causal claim unqualified; do not replace it with a profiled
duration or a guessed bottleneck.

The historical three-task 2026-10-01 Ralph run predates this gate. Its
L1/069 AITER-relative and L1/011 adapter-relative no-profiler paired latency
receipts remain valid for their exact workload cells; L2/060 has no speed
claim. The run produced no usable rocprof report. Do not relabel its empty
`torch.profiler` kernel-name probe as a completed profile.

The first independent route check on 2026-10-02 (`.local/profile/route-smoke-001/`)
failed before the AITER kernel because first-call JIT compilation under
rocprof corrupted `aicc --offload-arch=native` target discovery. A new
non-profile admission prewarmed the exact original L1/069 smoke workload into
persistent ignored `.local/aiter-jit-cache`. The next create-only rocprof run
at `.local/profile/route-smoke-003/` completed as job `bw-179e2692e8cd` on
HCU1; the validation JSON found 3 `Rmsnorm2dFwd` rows and bound the CSV/PMC
hashes. Its local `REPORT.md` gives the actual counters and missing metrics.
This validates the gateway and skill parser, **not** an additional speed claim
or a complete GPU bottleneck diagnosis.
