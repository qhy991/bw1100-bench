"""Community HIP stable-sort + binary-search offsets for original L1/058.

The optimized sort and search kernels are shipped PyTorch/ATen operators.
The sorted expert values already produced by stable sort allow 257 lower
bounds to replace the full histogram/prefix-sum route. No sorting kernel or
counting implementation is reconstructed here. Pin to the qualified DTK image.
"""

import torch


def run(topk_idx):
    flat = topk_idx.reshape(-1)
    experts, indices = torch.ops.aten.sort.stable(flat, stable=True)
    boundaries = torch.arange(257, dtype=flat.dtype, device=flat.device)
    offsets = torch.ops.aten.searchsorted.Tensor(
        experts, boundaries, out_int32=True, right=False)
    return indices.to(torch.int32), offsets


def run_histogram(topk_idx):
    """Independent native-library comparison arm, not reference.py's run."""
    flat = topk_idx.reshape(-1)
    _, indices = torch.ops.aten.sort.stable(flat, stable=True)
    counts = torch.bincount(flat.long(), minlength=256)
    prefix = torch.cumsum(counts, 0, dtype=torch.int32)
    offsets = torch.cat((prefix.new_zeros(1), prefix))
    return indices.to(torch.int32), offsets
