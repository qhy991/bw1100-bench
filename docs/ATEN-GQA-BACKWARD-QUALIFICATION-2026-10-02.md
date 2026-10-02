# Native community training backward, 2026-10-02

L1/001 starts from supplied BF16 softmax weights, supplied dropped weights
and a supplied dropout mask. A whole q/k/v attention forward/backward changes
that ABI. The adapter calls PyTorch 2.11's shipped ATen
`native_dropout_backward` and `_softmax_backward_data` on those inputs, with
vendor matmuls and the original GQA expansion/reduction and output casts.
This is an optimized community-primitive composition; it is not a shipped
whole vLLM training layer or a claim of the strongest possible implementation.

These are the primitives used by the community framework's native training
derivatives; see [PyTorch autograd definitions](https://github.com/pytorch/pytorch/blob/main/tools/autograd/derivatives.yaml)
and [native softmax](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/SoftMax.cpp).
The execution is pinned to the existing DTK image recorded in the inventory,
PyTorch 2.11.0 / HIP 6.3.26113. No copied handwritten softmax-gradient formula
is used as the denominator. The selected original maximum input size was
6.84 GiB, inspected before the full HCU2 gate.

Original CPU smoke passed 2/2, then original HCU2 smoke passed 2/2 as job
`bw-5a1fbbec4d28`. The first full numerical gate passed 160/160 but its admission
terminal (`bw-02e8482a8963`) remained not-qualified because the post-exit
snapshot still showed 32% VRAM. Subsequent live observation found HCU2 at 0%
with no visible KFD PID and no live owned container. The historical terminal
was preserved. The admission was repaired to observe release for at most a
30-second window after a zero exit, with no signal or forced memory cleanup.
Occupied/nonzero/live-container observations still cannot issue `completed`.

The fresh full canonical run `bw-0d7d3de305b8` passed 160/160 and issued
`completed`: its preserved release sequence was 32% -> 7% -> 0% in 2.78 s,
with no visible KFD process or live container. The exact adapter and raw
result/terminal hashes are indexed in `baselines/README.md`; receipts remain
in the isolated qualification worktree, excluded from Git. No candidate
speedup against this denominator has yet been measured. The 18 CPU regression
checks, including delayed release and persistent-busy rejection, passed in
the pinned DTK CPU route.
