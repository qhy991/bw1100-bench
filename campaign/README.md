# Gate/up independent3h replication and targeted GQA continuation

Two fresh Gate/up pairs (a direct/b Cake, c Cake/d direct) from the same qualified
community baseline and high-level reference. No previous custom winners or
scores are exposed. These test the original3h endpoint, not the6h continuation
endpoint; no pooling with continuation results. One GQA continuation pair
(e direct/f Cake) starts from each arm's6h frozen best and records additional3h
parent-relative performance (cumulative9h). Compiler f09a3e8 is unchanged.

All six use GLM-5.3:high, original16x10 gates, profiling skill, common paired wall
timer,3h outer timeout and the same final15min handoff reserve. Returned results
remain exploratory because authoring isolation is prompt-scoped and there are
only two new independent Gate/up pairs. GQA is a continuation, not replication.

Roots: /data3/testuser01/experiments/bw1100-bench-cake-targeted-GROUP-20261003
on bw1100-1; branches codex/cake-targeted-GROUP-20261003. Read intake, group plan,
protocol, JOURNAL, evaluations and ENDPOINT. Original experiments are preserved.

Shared GQA hypotheses in ANALYSIS-GQA.json are marked untested; candidate code
is authored by Ralph. Existing constraints make bool-mask and mixed-int/float
fusion uncertain; CPU assessment must precede GPU work. A pinned-library
BF16-input/FP32-output GEMM probe is legitimate only if no FP32 intermediate is
rounded away and original device correctness passes.
