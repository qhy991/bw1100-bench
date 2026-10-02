# BW1100 wave 2

Six independent fresh-session Claude/GLM-5.3 Ralph processes, maximum three hours
per group, on observed-idle HCU1–6. HCU0 remains outside this batch.

| Group | HCU | Original tasks | Search timeboxes |
|---|---|---|---|
| a | 1 | L1/058 stable sorting; L1/011 RoPE | 75 + 90 minutes |
| b | 2 | L1/001 GQA training backward | 165 minutes |
| c | 3 | L2/056 decoder backward | 165 minutes |
| d | 4 | L2/018 ragged vision attention | 165 minutes |
| e | 5 | L2/024 FP32 MoE | 165 minutes |
| f | 6 | L2/060 gated delta-rule | 165 minutes |

All groups reserve final15 minutes for handoff; stop early after evidence-backed
completion. Group plans own baseline bindings. Runtime intake/deadline own actual
start and assigned group. The installed official flow is unchanged; the local
ralph_flow.py derives from its frozen hash and checks DONE between fresh sessions.
No unrelated project dependency. Candidate kernels are implemented by Ralph.

Remote roots: /data3/testuser01/experiments/bw1100-bench-ralph-wave2-GROUP-20261002
on bw1100-1 (node4), each on codex/ralph-wave2-GROUP-20261002. Start with
bash campaign/launch.sh GROUP in the matching root. Each root has logs at
campaign/logs/ralph.log, task decisions at STATUS.md/JOURNAL.md in campaign,
and ignored receipts/results in campaign/results and .local/profile.
A process/log alone is not accepted optimization evidence. Completion requires
original160-case candidate gate, per-workload paired timing source bindings,
profiling evidence or specific skip reason, and released owned GPU containers.

Skills readable by all six agent processes:
- /data3/testuser01/.agents/skills/dcu-rocprof-report-skill/SKILL.md
- /data3/testuser01/.agents/skills/rocm-kernelwiki/SKILL.md

Private bench: https://github.com/qhy991/bw1100-bench
Canonical deployment: /data3/testuser01/bw1100-bench on bw1100-1;
/home/testuser01/bw1100-bench on bw1100. Main remains the qualified baseline suite.
