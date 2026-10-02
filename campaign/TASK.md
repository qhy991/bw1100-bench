# Ninety-minute community-baseline qualification, within the three-hour window

Use the installed official humanize2 Ralph loop with Claude/GLM-5.3 on node4
HCU2. The separate three-task optimization is already running on HCU1.
Read `campaign/AGENTS.md`, `campaign/STATUS.md`, `campaign/JOURNAL.md`,
`campaign/deadline.json`, root `AGENTS.md`, `baselines/README.md`,
`suite.json`, `sources.lock.json` and `docs/CORRECTNESS.md` in each fresh round.
First use the Read tool on the exact installed profiler skill and the wiki
skill, and record path/hash/Read in JOURNAL. The intake alone proves readability,
not agent use. The total wall budget is 90 minutes, with final 15 minutes for
terminal checks/handoff; stop starting GPU jobs in that final interval.

Five tasks already have exact-original full gfx938 community gates: L1/069,
L1/048, L2/035, plus newly qualified L1/011 Transformers eager/Inductor RoPE
and L1/001 ATen training-backward composition. Their source/evidence authority
is root `baselines/README.md`. Do not redo settled gates or optimize their
candidates in this phase. Preserve the ten original task IDs and workloads.

Work through the five remaining gaps, at most about 15 minutes each, in order:

1. L1/058 stable expert sorting: inspect shipped Hygon-capable stable routing
   or sort pipelines (AITER/FlagGems/other community code). The prior FlagGems
   argsort route timed out and its HCU0 container remains quarantined. Do not
   rerun that unchanged route. VLLM alignment's padded/block output needs exact
   stable flattened indices and 257 offsets; matching names is insufficient.
   The original already uses vendor Torch stable sort/bincount/cumsum. Merely
   copying that reference sequence is not a newly qualified stronger baseline.
2. L2/018 ragged vision attention: inspect shipped community attention modules
   or optimized primitives that preserve the original BF16 intermediate
   rounding. The installed FlashAttention adapter failed the numeric smoke;
   FP32 online softmax is not automatically equivalent to the original
   materialized probability path. Source-analyze the earliest divergence first.
3. L2/024 256-expert MoE: the earlier installed vLLM fused route VMFaulted.
   Inspect its weight/layout/index contracts before any device retry. Screen
   a distinct shipped AITER/FlagGems/vendor path only after source/ABI checks;
   preserve exact top-8, expert count, supplied weights and combine semantics.
4. L2/060 chunk gated delta rule: FLA's public entry cumulatively integrates
   log gates, while the original mixes cumulative intra-chunk factors with
   raw per-token inter-chunk factors. A gate/layout-only adapter is not assumed
   equivalent. Prove a possible transformation or find an actual community
   composition that matches both paths; otherwise document the semantic gap.
   Never relabel a edited FLA kernel or reference reconstruction as shipped.
5. L2/056 whole decoder backward: inspect community training modules/native
   backward operators, not vLLM inference. Bind all ten gradients and each
   rounding/intermediate convention. L1/001's native training-op composition
   is a contract example, not proof for this task. A minimal composition of
   optimized community primitives must be labeled as such; a copied manual
   gradient formula is not a strong shipped denominator.

For each task first inspect the exact original definition and 16-workload
grid, then the actually installed community source/version. Pin the upstream
callable, source hash/license and adapter scope. Search primary upstream source
or documentation if necessary; do not install a large new GPU stack during
this bounded phase. Only make minimal ABI/layout/parameter adapters. If a
kernel itself must change for Hygon, record it as a derived port and do not
silently classify it as the original community implementation.

Prepare source/probes under `campaign/baselines/` and tools under
`campaign/tools/`. Run appropriate CPU semantic probes first, then canonical
`bwbench.py check` smoke and all 16 original workloads x 10 rounds on gfx938.
Use the original `gen_inputs`, effective upstream tolerances and exact discrete
gates. Keep the reference a correctness oracle; never weaken shapes, precision,
tolerance or workload count. Every attempt has a fresh receipt/output path.
Only a full gate plus a completed/released terminal qualifies a new baseline.
If an alternative community arm exists, screen paired absolute callable latency
with original inputs, forward/reverse order and A/A controls to avoid selecting
a weaker denominator. Include necessary adaptation and public dispatch cost.
No candidate speedup or whole-model claim belongs to this qualification phase.

After full correctness, collect a bounded target-callable diagnostic through
`scripts/rocprof.sh` if a GPU bottleneck/strength explanation is claimed.
Bind original workload UUIDs, expected observed kernel regex, CSV/metric
validation, image/source hashes and terminal receipt. First-call JIT must be
prewarmed outside rocprof in persistent ignored cache. Never infer a mechanism
from missing counters or an empty filtered torch.profiler list. Explicit
`profile_unavailable` or `not_decision_relevant` causes are allowed; failing
semantic/correctness routes do not need a profiler run to justify rejection.

Use only HCU2 through this bench's `scripts/dtk.sh gpu` / `scripts/rocprof.sh`,
after observed idle admission. The immutable image is
`sha256:3ad0ae7192b8f9bafdf5b48fc414f8785f3c2463005e6b25290b7f75146ff260`.
Use a bounded `BWBENCH_TIMEOUT` up to 600s per attempt, and do not start if the
deadline leaves less than the timeout plus release/handoff allowance. Preserve
faulted, timed-out and unknown states. Observe exact container/HCU/receipts
before considering any retry. Never stop another container, use HCU0, touch
node2 serving, run on B300, or bypass the owning admission with raw Docker.

At each task's checkpoint update STATUS/JOURNAL with actual callable/version,
semantic finding, adapter classification, completed cases, receipt/source
hashes, decision and next action. Promote a new entry and its inventory row on
this branch only after the accepted full gate. Keep raw data/results ignored,
commit accepted sources and factual negative checkpoints, and preserve all
historical failures. Finish with a concise five-row handoff. Successful source
integration is useful; honest unsupported rows remain gaps, not fake baseline
or optimization successes.
