"""The captured data2 KDA verify seam, including caller-owned recurrent state."""
from __future__ import annotations

import itertools
import os
from pathlib import Path

from . import bundle, catalog
from .case import Case, assert_tensor

_RECEIVERS = itertools.count()
_CONFIG = None


def serving_config(model):
    global _CONFIG
    if _CONFIG is None:
        import torch
        from vllm.config import set_current_vllm_config
        from vllm.engine.arg_utils import EngineArgs
        from vllm.distributed import init_distributed_environment, initialize_model_parallel
        from vllm_hcu.patch.worker import apply_worker_patches
        os.environ.setdefault("VLLM_HCU_INDEXER_KCACHE_LAYOUT", "normal")
        os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "1")
        config = EngineArgs(
            model=str(model), max_model_len=32768, kv_cache_dtype="fp8_ds_mla",
            attention_backend="FLASHMLA_SPARSE", moe_backend="deep_gemm",
            enable_prefix_caching=True, max_num_batched_tokens=128, max_num_seqs=32,
            language_model_only=True, compilation_config={"cudagraph_mode": "FULL_AND_PIECEWISE"},
            speculative_config={"method": "mtp", "num_speculative_tokens": 3,
                                "use_local_argmax_reduction": True}).create_engine_config()
        apply_worker_patches(config)
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(29500 + os.getpid() % 500))
        with set_current_vllm_config(config):
            init_distributed_environment(1, 0, "env://", 0, "nccl")
            initialize_model_parallel(1, 1)
        _CONFIG = config
    return _CONFIG


def native_call(case, model):
    import torch
    from vllm.config import set_current_vllm_config
    from vllm.forward_context import set_forward_context
    from vllm.utils.torch_utils import set_default_torch_dtype
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
    config = serving_config(model)
    import vllm.models.glm5next.nvidia.kda as module
    prefix = f"model.layers.{case.public['metadata']['layer']}.self_attn.production_{next(_RECEIVERS)}"
    with set_current_vllm_config(config), set_default_torch_dtype(config.model_config.dtype), torch.device("cuda"):
        receiver = module.Glm5NextLinearAttention(config.model_config.hf_text_config, config, prefix=prefix)
    if not (getattr(type(receiver), "_vllm_hcu_glm5next_kda_conv_weight_applied", False)
            and getattr(module.fused_recurrent_kda, "_vllm_hcu_glm5next_kda_strided_recurrent_wrapper", False)):
        raise catalog.PrerequisiteError("Native KDA lacks the served conv-weight or strided-recurrent patch")
    public = case.public
    inputs, weights, metadata = public["inputs"], public["weights"], public["metadata"]
    receiver.kv_cache = (public["state"]["conv"], public["state"]["ssm"])
    receiver.A_log.data.copy_(weights["A_log"])
    receiver.dt_bias.data.copy_(weights["dt_bias"])
    receiver._merged_conv_weight = weights["merged_conv_weight"]
    md = {key[3:]: value for key, value in inputs.items() if key.startswith("md_")}
    attention = GDNAttentionMetadata(
        num_prefills=0, num_prefill_tokens=0, num_decodes=0, num_decode_tokens=0,
        num_spec_decodes=metadata["reqs"], num_spec_decode_tokens=metadata["num_actual_tokens"],
        num_actual_tokens=metadata["num_actual_tokens"],
        non_spec_token_indx=torch.empty(0, dtype=torch.int32, device="cuda"), **md)

    def call():
        with set_current_vllm_config(config), set_forward_context({receiver.prefix: attention}, config):
            receiver._forward(inputs["qkv_proj_states"], inputs["g1"], inputs["beta"], public["output"])
    return call


def make_case(record, row, device):
    import torch
    inputs = bundle.decode(record["inputs"], device)
    weights = bundle.decode(record["weight_values"], device)
    expected = bundle.decode(record["outputs"], "cpu")
    sc = record["scalars"]
    if (sc.get("kind") != "verify" or sc.get("reqs") != row["requests"]
            or sc.get("num_spec_decodes") != row["requests"] or sc.get("num_decodes") != 0
            or sc.get("local_num_heads") != 64 or sc.get("head_dim") != 128
            or sc.get("num_spec") != 3 or sc.get("lower_bound") != -5.0 or not sc.get("safe_gate")):
        raise ValueError("Captured KDA record differs from the data2 verify contract")
    qkv, g1, beta = (inputs[key] for key in ("qkv_proj_states", "g1", "beta"))
    if (qkv.dtype != torch.bfloat16 or qkv.ndim != 2 or qkv.shape[1] != 24576
            or qkv.stride() != (24896, 1) or g1.dtype != torch.bfloat16 or beta.dtype != torch.bfloat16):
        raise ValueError("Captured KDA input dtype/strided projection ABI differs")
    reqs, m = row["requests"], row["M"]
    qsl = inputs["md_spec_query_start_loc"][:reqs + 1].tolist()
    if qsl != list(range(0, m + 1, 4)):
        raise ValueError("KDA verify requires four consecutive tokens per request")
    accepted = inputs["md_num_accepted_tokens"][:reqs]
    if not bool(((accepted >= 1) & (accepted <= 4)).all()):
        raise ValueError("KDA accepted-token window is outside 1..4")
    slots = [int(value) for value in inputs["state_slots"].tolist()]
    if not slots or any(slot <= 0 for slot in slots):
        raise ValueError("Captured state_slots must contain the live positive slots")
    lut = {slot: i + 1 for i, slot in enumerate(dict.fromkeys(slots))}
    pre_conv, pre_ssm = inputs["pre_conv_rows"], inputs["pre_ssm_rows"]
    if (tuple(pre_conv.shape[1:]) != (6, 24576) or pre_conv.dtype != torch.bfloat16
            or tuple(pre_ssm.shape[1:]) != (64, 128, 128) or pre_ssm.dtype != torch.float32
            or pre_conv.shape[0] != len(slots) or pre_ssm.shape[0] != len(slots)):
        raise ValueError("Captured conv/SSM layout differs from the SD verify contract")
    original_indices = inputs["md_spec_state_indices_tensor"]
    referenced = {int(value) for value in original_indices[:reqs].reshape(-1).tolist() if int(value) > 0}
    if referenced != set(slots):
        raise ValueError("Captured state region does not cover the verify slot mapping")
    inputs["md_spec_state_indices_tensor"] = torch.tensor(
        [lut[int(value)] if int(value) > 0 else int(value) for value in original_indices.reshape(-1).tolist()],
        dtype=original_indices.dtype, device=device).reshape(original_indices.shape)
    local = torch.tensor([lut[slot] for slot in slots], device=device)
    size = len(lut) + 3
    conv = torch.full((size, 6, 24576), 7, dtype=torch.bfloat16, device=device)
    ssm = torch.full((size, 64, 128, 128), 7, dtype=torch.float32, device=device)
    conv[local], ssm[local] = pre_conv, pre_ssm
    output = torch.full((1, sc["num_actual_tokens"], 64, 128), 7, dtype=torch.bfloat16, device=device)
    if output.shape[1] < m:
        raise ValueError("Captured output capacity is smaller than its real-token range")
    pool_mask = torch.zeros(size, dtype=torch.bool, device=device)
    pool_mask[local] = True
    out_mask = torch.zeros_like(output, dtype=torch.bool)
    out_mask[:, :m] = True
    expected["core_attn_out"] = expected["core_attn_out"][:, :m]
    public = {"inputs": inputs, "weights": weights,
              "metadata": {**sc, "layer": row["layer"], "real_tokens": m},
              "state": {"conv": conv, "ssm": ssm}, "output": output}
    case = Case(f"{row['rank']}/{row['path'].name}", public, expected,
                {"conv": conv, "ssm": ssm, "output": output},
                {"conv": pool_mask, "ssm": pool_mask, "output": out_mask},
                ("kda", row["layer"], reqs, tuple(qkv.shape), tuple(g1.shape), tuple(beta.shape), size))
    case.local = local
    return case


def validate(case):
    public = case.public
    m = public["metadata"]["real_tokens"]
    assert_tensor(public["output"][:, :m].cpu(), case.expected["core_attn_out"], case.name + ":output")
    for key in ("conv", "ssm"):
        actual = public["state"][key][case.local].cpu()
        assert_tensor(actual, case.expected[f"post_{key}_rows"], case.name + ":" + key)


def cases(root, model, device):
    import torch
    if model is None or not Path(model).is_dir():
        raise catalog.PrerequisiteError("KDA native replay requires --model with the data2 checkpoint config")
    result = []
    for row in catalog.captured_records(Path(root)):
        case = make_case(bundle.load(row), row, device)
        call = native_call(case, model)
        for _ in range(2):
            case.restore()
            call()
            torch.cuda.synchronize()
            case.integrity()
            validate(case)
        case.restore()
        result.append(case)
    return result
