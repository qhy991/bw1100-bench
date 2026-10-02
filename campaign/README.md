# Profiled community-baseline campaign

Prepared three-hour official humanize2 Ralph loop, Claude/GLM-5.3, node4 HCU1.
The launch is pending live SSH and resource verification. `TASK.md` owns this
campaign's frozen task order and acceptance conditions. `STATUS.md` and
`JOURNAL.md` will cite create-only correctness/profile/timing receipts.

Run `bash campaign/launch.sh` from a fresh isolated checkout with the pinned
ignored `.deps/` and `.data/` materialized. The launch checks the official flow,
skill, node/image/GPU readiness and CPU audit, then prepends the profiler
skill instruction to the real Ralph prompt. It records a three-hour deadline
and exit code. Existing intake/deadline/output files must never be overwritten.
