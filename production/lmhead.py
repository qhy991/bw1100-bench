"""Real checkpoint weight, explicit synthetic activations and a complete-logit gate."""
from __future__ import annotations

import json
from pathlib import Path

from .case import Case


def load_weight(model, device):
    import torch
    from safetensors import safe_open
    model = Path(model)
    weights = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    name = next(name for name in weights if name.endswith("lm_head.weight"))
    with safe_open(str(model / weights[name]), framework="pt", device="cpu") as handle:
        weight = handle.get_tensor(name)
    if tuple(weight.shape) != (154880, 4096) or weight.dtype != torch.bfloat16:
        raise ValueError("LM-head requires the real [154880,4096] BF16 checkpoint weight")
    return weight.to(device)


def numerical_verdict(actual, native, oracle):
    import torch
    if actual.shape != native.shape or actual.dtype != native.dtype:
        raise ValueError("LM-head output shape or dtype differs")
    if not bool(torch.isfinite(actual).all()):
        raise ValueError("LM-head output contains nonfinite values")
    denominator = oracle.norm()
    if denominator == 0:
        native_error = native.double().norm()
        error = actual.double().norm()
    else:
        native_error = (native.double() - oracle).norm() / denominator
        error = (actual.double() - oracle).norm() / denominator
    if not bool(error <= 1.05 * native_error + 2 ** -9):
        raise ValueError("LM-head output exceeds its native-relative FP64 error bound")
    target = oracle.argmax(-1)
    native_right = native.argmax(-1) == target
    if not bool((actual.argmax(-1)[native_right] == target[native_right]).all()):
        raise ValueError("LM-head changes an argmax that native resolves correctly")
    return {"relative_error": float(error), "native_relative_error": float(native_error),
            "rows_excused_native_ties": int((~native_right).sum())}


def cases(model, device):
    import torch
    weight = load_weight(model, device)
    result = []
    weight_snapshot = weight.clone()
    for m in (4, 8, 16, 32):
        for seed in (200, 201):
            generator = torch.Generator(device=device).manual_seed(seed + m)
            x = torch.randn(m, 4096, device=device, dtype=torch.bfloat16, generator=generator)
            with torch.no_grad():
                native = torch.nn.functional.linear(x, weight)
                oracle = x.double() @ weight.double().t()
            public = {"inputs": {"x": x}, "weights": {"weight": weight},
                      "metadata": {"M": m, "seed": seed, "input_tier": "synthetic_activations_real_checkpoint_weight"},
                      "state": {}, "output": torch.full_like(native, float("nan"))}
            case = Case(f"M{m}-seed{seed}", public, {"native": native, "oracle": oracle},
                        {"output": public["output"]}, {"output": None}, ("lmhead", m),
                        {".weights.weight": weight_snapshot})
            result.append(case)
    return result


def validate(case):
    return numerical_verdict(case.public["output"], case.expected["native"], case.expected["oracle"])
