from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORD_NAME = re.compile(r".*layers_(4|24)_self_attn_-verify-r(1|2|4|8)-(\d+)\.pt")


class PrerequisiteError(ValueError):
    pass


def tasks():
    return json.loads((ROOT / "suite.json").read_text())["production_tasks"]


def task_record(task_id):
    return next((task for task in tasks() if task["id"] == task_id), None) or _unknown(task_id)


def _unknown(task_id):
    raise PrerequisiteError("Unknown production task: " + task_id)


def captured_records(root: Path):
    rows = []
    required = {"rank0": {2, 4, 8}, "rank5": {1, 2, 4}}
    for rank, requests in required.items():
        folder = root / rank
        manifest_path = folder / "manifest.json"
        if not manifest_path.is_file():
            raise PrerequisiteError(f"Missing guard/capture shard: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema") != "glm53.capture.manifest.v1":
            raise PrerequisiteError(f"Unsupported captured manifest: {manifest_path}")
        entries = manifest.get("records")
        if not isinstance(entries, dict) or len(entries) != 24:
            raise PrerequisiteError(f"{rank} requires all 24 admitted original records")
        coverage = set()
        for name, entry in sorted(entries.items()):
            match = RECORD_NAME.fullmatch(name)
            if not match or Path(name).name != name:
                raise PrerequisiteError(f"Unexpected phase/layer/record name: {name}")
            layer, reqs, call = map(int, match.groups())
            if reqs not in requests or call not in range(4):
                raise PrerequisiteError(f"Record outside original shard coverage: {name}")
            key = (layer, reqs, call)
            if key in coverage:
                raise PrerequisiteError(f"Duplicate structural record: {name}")
            coverage.add(key)
            if entry.get("status") != "real" or entry.get("rule") != "bitwise x2 (rule A)":
                raise PrerequisiteError(f"Record lacks original native replay admission: {name}")
            bound_layer = re.search(r"layers\.(4|24)\.", entry.get("layer_prefix", ""))
            if (not bound_layer or int(bound_layer.group(1)) != layer
                    or entry.get("site") != f"verify-r{reqs}" or entry.get("call_index") != call):
                raise PrerequisiteError(f"Manifest identity disagrees with its record name: {name}")
            digest = entry.get("sha256", "")
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise PrerequisiteError(f"Record lacks its original manifest identity: {name}")
            path = folder / name
            if not path.is_file():
                raise PrerequisiteError(f"Missing tensor record: {path}")
            rows.append({"path": path, "entry": entry, "rank": rank, "layer": layer,
                         "requests": reqs, "call": call, "M": 4 * reqs})
        expected = {(layer, req, call) for layer in (4, 24) for req in requests for call in range(4)}
        if coverage != expected:
            raise PrerequisiteError(f"{rank} lacks layer, shape or consecutive-call coverage")
    return rows


def audit(task, inputs=None, model=None):
    if task["id"] == "glm53-dsa-decode":
        raise PrerequisiteError(task["prerequisite"])
    if task["id"] == "glm53-kda-verify":
        if inputs is None:
            raise PrerequisiteError("--inputs must name kda-data2 with rank0 and rank5")
        rows = captured_records(Path(inputs))
        return {"input_tier": task["input_tier"], "records": len(rows),
                "shapes": sorted({row["M"] for row in rows}), "layers": [4, 24],
                "shards": ["rank0", "rank5"], "tensor_validation": "deferred_to_check"}
    if model is None:
        raise PrerequisiteError("--model must name the external Channel-FP8 checkpoint")
    index_path = Path(model) / "model.safetensors.index.json"
    if not index_path.is_file():
        raise PrerequisiteError(f"Missing external checkpoint index: {index_path}")
    weight_map = json.loads(index_path.read_text()).get("weight_map", {})
    names = [name for name in weight_map if name.endswith("lm_head.weight")]
    if len(names) != 1:
        raise PrerequisiteError("Checkpoint must contain exactly one lm_head.weight")
    shard = (Path(model) / weight_map[names[0]]).resolve()
    if Path(model).resolve() not in shard.parents or not shard.is_file():
        raise PrerequisiteError("LM-head shard is absent or outside the checkpoint")
    return {"input_tier": task["input_tier"], "shapes": [4, 8, 16, 32],
            "weight_name": names[0], "tensor_validation": "deferred_to_check"}
