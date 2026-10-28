"""Read the original admitted tensor archive at the external handoff boundary."""
from __future__ import annotations

import hashlib


def load(row):
    import torch
    path, entry = row["path"], row["entry"]
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != entry["sha256"]:
        raise ValueError(f"Record differs from its admitted manifest: {path.name}")
    import io
    record = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    if (record.get("schema") != "glm53.capture.record.v1" or record.get("seam") != "kda-data2"
            or record.get("site") != f"verify-r{row['requests']}"):
        raise ValueError(f"Record has another seam or phase: {path.name}")
    import re
    layer = re.search(r"layers\.(4|24)\.", record.get("layer_prefix", ""))
    if (not layer or int(layer.group(1)) != row["layer"]
            or record.get("rank") != int(row["rank"][4:]) or record.get("call_index") != row["call"]):
        raise ValueError(f"Record identity disagrees with admission: {path.name}")
    if record.get("weight_values") is None:
        raise ValueError("KDA record omits its prepared weights")
    return record


def decode(record, device):
    import torch
    buffers = {group: value["bytes"].to(device) for group, value in record["groups"].items()}
    out = {}
    for name, desc in record["tensors"].items():
        dtype = getattr(torch, desc["dtype"])
        size = torch.empty((), dtype=dtype).element_size()
        storage = buffers[desc["group"]]
        offset = desc["byte_offset_in_group"]
        if offset % size or storage.numel() % size:
            raise ValueError(f"Misaligned captured tensor: {name}")
        out[name] = storage.view(dtype).as_strided(desc["shape"], desc["stride"], offset // size)
    return out
