# BW1100 continuation: another3h from each arm's previous best

Six independent fresh-session Claude/GLM-5.3:high Ralph processes, three task pairs,
fixed3h budget each, no first-win early stop, and common complete-call wall timer.

| Group | HCU | Task | Arm |
|---|---|---|---|
| a | 1 | L1/069 residual RMSNorm | direct Triton |
| b | 2 | L1/069 residual RMSNorm | Cake IR |
| c | 3 | L1/048 gate/up GELU-tanh | Cake IR |
| d | 4 | L1/048 gate/up GELU-tanh | direct Triton |
| e | 5 | L1/001 GQA backward | direct Triton |
| f | 6 | L1/001 GQA backward | Cake IR |

Original task/ABI,16workloads x10correctness rounds and frozen community
baselines are unchanged. Both arms use original gen_inputs, matched warmup,
30samples per measurement direction, A/A and input-mutation guards. Each arm starts from its own prior frozen best and negative history.
No other arm's custom winner is allowed. Both arms retain existing library calls and record
coverage; all new custom GPU kernels in the Cake arm are Compiler emissions.

Compiler is pinned at f09a3e859859e848f33a21b63412a306eea8738f, independently
usable public assess/lower API on exact gfx938. Its optional .deps copy is not
part of bw1100-bench's normal import/runtime path. The original bench main and
previous experiments retain their independence. Neither Compiler nor installed
Ralph flow is modified. This is not a canonical Open-Cake Lab Campaign.

Primary endpoint: best qualified conservative geomean speedup found within3h.
Both arms may improve an incumbent throughout the same fixed search window.
Time-to-first-correct/first-win and the30/60/120/165/180minute best-performance
trajectory are secondary. protocol.json owns endpoint/ranking/stopping policy.
Only evaluate.py accepted receipts enter the ledger; finish.py selects the best.
This is a continuation of the same runs, not a second independent replicate.
Report additional3h gains and cumulative6h performance separately. After source freeze, a
common same-HCU confirmation remeasures both endpoints with no candidate edits.
Final15minutes are reserved equally for handoff, with no new GPU admissions.

Actual arm/start/deadline are owned by campaign/intake.json and deadline.json.
Results/logs/profiles remain in each remote root:
/data3/testuser01/experiments/bw1100-bench-cake-continued3h-GROUP-20261003
on bw1100-1, branches codex/cake-continued3h-GROUP-20261003.
Read campaign/DONE.json, JOURNAL.md, STATUS.md, results/ and .local/profile/.
Source and Schedule artifacts remain committed; raw dataset stays ignored.

Parent manifests and prior context live in campaign/prior. The first canonical
evaluation is seed00. The community denominator is unchanged. Same Compiler
f09a3e8, same model, dtype/tolerances, timer and skill paths. Launch now has an
outer wall timeout plus narrowly scoped owned model cleanup to prevent late
authoring; completed-with-budget-timeout is distinct from a GPU/provider fault.
