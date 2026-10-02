# Follow-up qualification queue

Root `baselines/README.md` is the authority for the ten baseline classifications
and accepted source/receipt bindings. Keep the ten original tasks. This file
specifies next investigations, not additional qualified denominators.

After the qualified three-task optimization campaign starts, inspect community
source and prepare exact-ABI adapters for the remaining tasks. Do source and
semantic analysis without GPU first. Device qualification must use its own
bounded admission, original smoke then all 16 workloads x 10 rounds, and must
wait for an available selected HCU. An installed library or CPU comparison is
not device qualification. Do not turn a copied reference into a baseline.

1. L1/011: inspect Transformers' shipped LlamaRotaryEmbedding callable with
   supplied `inv_freq` and attention scaling bound as original inputs, rather
   than recomputing the unusual input factory's scaling. Compare its actual
   cos/sin return ABI and precision; investigate the exact community callable
   under the supported torch.compile path as a separate stronger arm. Label
   eager/compiled arms explicitly; the old handwritten vLLM cache adapter
   remains insufficient for a strong denominator.
2. L1/058: identify the specific Hygon stable argsort hang from preserved logs
   and source. Screen shipped stable Torch sort/bincount/cumsum and FlagGems
   sort variants on the exact discrete contract. Do not retry the old hanging
   route or HCU0 without a concrete changed hypothesis and observed cleanup.
3. L2/018: inspect optimized community attention paths that preserve the
   original BF16 softmax-probability materialization. The failed FlashAttention
   path changed that precision point; changing tolerance is not a repair.
4. L2/060: derive the earliest semantic divergence between the reference's
   per-token versus cumulative gate factors and FLA's shipped implementation.
   Only adopt a community parameter/layout adapter if it is algebraically
   equivalent; otherwise retain the semantic gap. Never rename an edited FLA
   kernel or the reference-semantics candidate as a shipped strong baseline.
5. L1/001 and L2/056: map the specified intermediate tensors, rounding points
   and gradients to community training primitives; a vLLM inference callable
   cannot stand in for these backward contracts. Check shipped training APIs
   before constructing a minimal composition of optimized library primitives.
6. L2/024: diagnose the preserved VMFault against source/layout/shape before
   attempting another GPU run. Screen safe shipped expert execution paths and
   retain exact top-8, combine and weight semantics. A faulting fused route has
   no speed denominator.

For every attempt record upstream revision/license, shipped callable, adapter
scope, semantic analysis and exact qualification outcome. Prioritize routes
that can reach the original ABI without editing the community kernel.
