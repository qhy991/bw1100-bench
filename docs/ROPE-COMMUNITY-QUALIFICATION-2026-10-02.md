# Community RoPE qualification, 2026-10-02

Root `baselines/README.md` owns accepted denominators. Keep qualification outputs ignored
under `results/` and cite the canonical original-task verifier.

`baselines/l1_011_transformers_rope.py` calls the installed Transformers 5.16.1
`LlamaRotaryEmbedding.forward` with the original supplied inverse-frequency
buffer. `run_eager` is its eager arm; `run_compiled` compiles the same callable with
PyTorch Inductor. Source/version checks run at import, outside steady-state
timing. The only return adapter stacks the shipped cos/sin outputs.

Preserve the supplied frequency values; recomputing
standard Llama3 scaling changes this task's unusual input factory. If the
compiled arm fails, retain that failure and do not label eager as compiled or
as the fastest available Hygon path.

The first static-shape compiled attempt passed smoke, then stopped at the
default Dynamo eight-recompile limit during the full original shape grid
(job `bw-289e4da80113`, exit 1, released HCU2). The next revision uses dynamic
shape compilation of the same shipped forward; it changes no numeric
operation or tolerance.

The dynamic revision passed all 16 original workloads x 10 rounds on HCU2
(job `bw-0fec3eaa11ef`, 160/160, completed/exit 0). The eager arm also passed
the full device gate at its first revision (job `bw-1da3b525995a`).
The unchanged eager callable at the dynamic revision was rebound by a full
gate (`bw-a6486b440c3b`, 160/160). A paired no-profiler wall-latency screen on
HCU2 (`bw-26ebec59dc19`) generated the exact original inputs with seed 200,
used forward/reverse order and A/A drift, and found the compiled arm faster
in 13/16 cells. Eager was faster in the three larger cells with 14704, 16384
and 34624 positions; using compiled unconditionally would weaken their
denominator. Raw evidence is `results/l1-011-transformers-strength-001.json`.

The final `run` uses only public position-count metadata: compiled at <=8192
positions, eager above that. It is calibrated to this frozen 16-cell grid;
there is no universal best-route claim. The source is promoted from the probe
to the top-level baseline entry. The final actual dispatcher passed a fresh
full original gate (job `bw-036e2d281a44`, 160/160) and a three-arm timing
screen including dispatch cost (`bw-b772f2322074`). A direct dispatcher versus
selected-arm forward/reverse pair, with both A/A controls, isolated the
otherwise systematic three-arm order offsets. The measured dispatcher ratio
ranged 0.9984–1.0118 with median 1.0024 after a redundant outer no_grad wrapper
was removed; the necessary no_grad scope remains on both shipped-callable arms.
Source/report hashes and terminal receipts are indexed by the baseline inventory.
Neither compilation nor community popularity alone establishes a strong
performance denominator.
