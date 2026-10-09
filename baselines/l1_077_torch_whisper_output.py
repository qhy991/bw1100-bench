"""PyTorch/HIP vendor-GEMM baseline candidate for L1/077; GPU qualification pending."""
import torch


@torch.no_grad()
def run(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.matmul(hidden_states, weight.t())
