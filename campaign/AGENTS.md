# Campaign boundaries

Root `AGENTS.md`, `docs/RALPH-PROFILING.md`, source locks and the original suite
own correctness, profiling and GPU admission. Read `TASK.md`, `STATUS.md`,
`JOURNAL.md` and `deadline.json` in each fresh Ralph round; infer progress from
accepted files and receipts, not from a prior model's memory.

Only edit and commit source/documentation under `campaign/`. Store outputs in
ignored create-only paths. Use node4 HCU1 through the bench-owned admission;
do not use Cake, B300, raw Docker, or mutate existing containers/services.
Do not spawn another agent, server or allocator. Retain the last accepted
candidate and all failed receipts. Profile evidence, full device correctness,
paired timing, and serving claims have separate acceptance conditions.

Read these installed skills explicitly before a kernel change:
`/data3/testuser01/.agents/skills/dcu-rocprof-report-skill/SKILL.md` and
`/data3/testuser01/.agents/skills/rocm-kernelwiki/SKILL.md`.
The launch intake supplies the exact profiler skill hash. Use the wiki for
target-specific source/toolchain knowledge and rocprof for observed counters.
Do not report a bottleneck from an empty trace or uncollected counters.
