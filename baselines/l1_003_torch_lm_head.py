"""PyTorch/HIP vendor-GEMM baseline candidate for L1/003; GPU qualification pending."""
import torch


@torch.no_grad()
def run(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    # All original workloads keep the full sequence; there is no slicing input.
    return torch.matmul(hidden_states, weight.t())
