"""Pinned timm ConvNeXtV2 block adapted to the original L2/035 input ABI.

The block computation is timm's installed ConvNeXtBlock with GRN enabled.
``torch.func.functional_call`` binds the task's supplied weights directly to
that module, without copying parameters on each call or replacing its forward.
This adapter is a baseline candidate until the full original workload gate
passes on gfx938; its presence alone is not a performance qualification.
"""

from functools import lru_cache, partial
import hashlib
from pathlib import Path

import timm
from timm.models.convnext import ConvNeXtBlock
import torch


_SOURCE_HASHES = {
    "models/convnext.py": "f79d244b01c1f79c38526b543a715666f768e96f6167afa9caf61ac4211cb865",
    "layers/grn.py": "7a357b2a2f53c549cf8fc5ba4829934f7eb416456d0f4964cca825f0526ee95d",
}


def _check_source():
    if timm.__version__ != "1.0.28":
        raise RuntimeError("Expected timm 1.0.28, got " + timm.__version__)
    root = Path(timm.__file__).resolve().parent
    for name, expected in _SOURCE_HASHES.items():
        observed = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if observed != expected:
            raise RuntimeError("timm source differs: " + name)


_check_source()


@lru_cache(maxsize=16)
def _block(dim, grn_eps, norm_eps):
    block = ConvNeXtBlock(
        dim,
        kernel_size=7,
        mlp_ratio=4,
        conv_mlp=False,
        use_grn=True,
        ls_init_value=None,
        norm_layer=partial(torch.nn.LayerNorm, eps=norm_eps),
        drop_path=0.0,
        device="meta",
        dtype=torch.float32,
    ).eval()
    block.mlp.grn.eps = grn_eps
    return block


@torch.no_grad()
def run(
    x, dwconv_weight, dwconv_bias, layernorm_weight, layernorm_bias,
    pwconv1_weight, pwconv1_bias, grn_weight, grn_bias,
    pwconv2_weight, pwconv2_bias, eps, layer_norm_eps,
):
    dim = x.shape[1]
    block = _block(dim, float(eps), float(layer_norm_eps))
    parameters = {
        "conv_dw.weight": dwconv_weight,
        "conv_dw.bias": dwconv_bias,
        "norm.weight": layernorm_weight,
        "norm.bias": layernorm_bias,
        "mlp.fc1.weight": pwconv1_weight,
        "mlp.fc1.bias": pwconv1_bias,
        "mlp.grn.weight": grn_weight.reshape(-1),
        "mlp.grn.bias": grn_bias.reshape(-1),
        "mlp.fc2.weight": pwconv2_weight,
        "mlp.fc2.bias": pwconv2_bias,
    }
    return torch.func.functional_call(block, parameters, (x,), strict=True)
