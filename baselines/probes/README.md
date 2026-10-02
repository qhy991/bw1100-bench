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
