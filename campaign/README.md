# Cake Compiler versus direct Triton: BW1100 engineering pilot

Six independent fresh-session Claude/GLM-5.3:high Ralph processes, three task pairs,
maximum3h each with identical stopping policy and common complete-call wall timer.

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
30samples per measurement direction, A/A and input-mutation guards. No prior
custom winners are allowed. Both arms retain existing library calls and record
coverage; all new custom GPU kernels in the Cake arm are Compiler emissions.

Compiler is pinned at f09a3e859859e848f33a21b63412a306eea8738f, independently
usable public assess/lower API on exact gfx938. Its optional .deps copy is not
part of bw1100-bench's normal import/runtime path. The original bench main and
previous experiments retain their independence. Neither Compiler nor installed
Ralph flow is modified. This is not a canonical Open-Cake Lab Campaign.

Claim: engineering feasibility, valid artifacts, attained task latency,
authoring refusals/effort and time-to-first-valid-candidate. One run per arm/task,
prompt-scoped reference access, no physical GPU-exclusivity witness: no causal
agent-improvement or scientific arm-effect estimate. Future replication should
freeze a stronger authoring-custody boundary and independent matched seeds.

Actual arm/start/deadline are owned by campaign/intake.json and deadline.json.
Results/logs/profiles remain in each remote root:
/data3/testuser01/experiments/bw1100-bench-cake-control-GROUP-20261002
on bw1100-1, branches codex/cake-control-GROUP-20261002.
Read campaign/DONE.json, JOURNAL.md, STATUS.md, results/ and .local/profile/.
Source and Schedule artifacts remain committed; raw dataset stays ignored.
