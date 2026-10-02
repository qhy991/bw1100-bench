# Pending baseline qualification

Root `baselines/README.md` owns accepted denominators. Files here are probes,
not qualified performance baselines. Keep all qualification outputs ignored
under `results/` and cite the canonical original-task verifier.

`l1_011_transformers_rope.py` calls the installed Transformers 5.16.1
`LlamaRotaryEmbedding.forward` with the original supplied inverse-frequency
buffer. `run` is its eager arm; `run_compiled` compiles the same callable with
PyTorch Inductor. Source/version checks run at import, outside steady-state
timing. The only return adapter stacks the shipped cos/sin outputs.

Next gate: original CPU smoke for adapter semantics, then a separately admitted
gfx938 smoke/full gate for each arm, all 16 workloads x 10 rounds. A CPU pass
does not qualify either arm. Preserve the supplied frequency values; recomputing
standard Llama3 scaling changes this task's unusual input factory. If the
compiled arm fails, retain that failure and do not label eager as compiled or
as the fastest available Hygon path.

The first static-shape compiled attempt passed smoke, then stopped at the
default Dynamo eight-recompile limit during the full original shape grid
(job `bw-289e4da80113`, exit 1, released HCU2). The next revision uses dynamic
shape compilation of the same shipped forward; it changes no numeric
operation or tolerance. Its full gate remains pending.

The dynamic revision passed all 16 original workloads x 10 rounds on HCU2
(job `bw-0fec3eaa11ef`, 160/160, completed/exit 0). The eager arm also passed
the full device gate at its first revision (job `bw-1da3b525995a`). Next:
rebind the current eager source, then screen eager versus compiled with the
original generated inputs and paired no-profiler wall latency before choosing
the performance denominator. Do not treat correctness alone as baseline
strength or assume compilation always makes every shape faster.
