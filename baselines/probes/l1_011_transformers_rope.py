"""Pending L1/011 baseline: shipped Transformers RoPE, eager/Inductor arms.

Bind the original supplied inverse-frequency buffer and attention scalar to
LlamaRotaryEmbedding. Its shipped forward computes cos/sin; the adapter only
stacks the two outputs into the task ABI. Neither arm is device-qualified
until the original full gate accepts it. The compiled arm compiles that same
community callable, not a replacement handwritten formula.
"""

from functools import lru_cache
import hashlib
import inspect
from pathlib import Path

import torch
import transformers
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding


if transformers.__version__ != '5.16.1':
    raise RuntimeError('Expected installed Transformers 5.16.1')
_source = Path(inspect.getfile(LlamaRotaryEmbedding))
if hashlib.sha256(_source.read_bytes()).hexdigest() != (
        '13e65b752a9c9d8a5c22b83df73009a8940c0eefdc58c101df3eb910e3efc2f9'):
    raise RuntimeError('Transformers Llama source differs from the pinned image')


@lru_cache(maxsize=32)
def _callable(dim, attention_scaling, device):
    config = LlamaConfig(
        hidden_size=dim, num_attention_heads=1, num_key_value_heads=1,
        head_dim=dim,
        rope_parameters={'rope_type': 'default', 'rope_theta': 500000.0},
    )
    rotary = LlamaRotaryEmbedding(config).eval()
    rotary.attention_scaling = attention_scaling
    # The shipped forward uses x only for dtype/device. No data allocation is
    # needed for this metadata carrier, and supplied inv_freq is bound directly.
    carrier = torch.empty(0, dtype=torch.bfloat16, device=device)

    def call(position_ids, inv_freq):
        cos, sin = torch.func.functional_call(
            rotary, {'inv_freq': inv_freq}, (carrier, position_ids))
        return torch.stack((cos, sin), dim=-1)

    return call


@lru_cache(maxsize=32)
def _compiled(dim, attention_scaling, device):
    return torch.compile(_callable(dim, attention_scaling, device),
                         fullgraph=True, dynamic=False)


@torch.no_grad()
def run(position_ids, inv_freq, attention_scaling):
    return _callable(inv_freq.numel() * 2, float(attention_scaling),
                     str(inv_freq.device))(position_ids, inv_freq)


@torch.no_grad()
def run_compiled(position_ids, inv_freq, attention_scaling):
    return _compiled(inv_freq.numel() * 2, float(attention_scaling),
                     str(inv_freq.device))(position_ids, inv_freq)
