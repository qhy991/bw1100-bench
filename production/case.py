from __future__ import annotations

from dataclasses import dataclass, field


def tensors(tree, prefix=""):
    import torch
    if isinstance(tree, torch.Tensor):
        return {prefix: tree}
    if isinstance(tree, dict):
        result = {}
        for name, value in tree.items():
            result.update(tensors(value, f"{prefix}.{name}"))
        return result
    return {}


def same_bits(left, right):
    import torch
    return (left.dtype == right.dtype and left.shape == right.shape
            and torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8)))


def structure(tree):
    import torch
    if isinstance(tree, torch.Tensor):
        return "tensor"
    if isinstance(tree, dict):
        return {key: structure(value) for key, value in tree.items()}
    if isinstance(tree, (list, tuple)):
        return [structure(value) for value in tree]
    return tree


def layout(tensor):
    return (tensor.dtype, tuple(tensor.shape), tuple(tensor.stride()), tensor.storage_offset(),
            tensor.untyped_storage().data_ptr(), tensor.untyped_storage().nbytes())


@dataclass
class Case:
    name: str
    public: dict
    expected: dict
    mutable: dict
    masks: dict
    group: tuple
    shared_snapshots: dict = field(default_factory=dict)
    output_view: object = None

    def __post_init__(self):
        if self.output_view is None:
            self.output_view = self.public["output"]
        self.owned = tensors(self.public)
        self.layouts = {name: layout(value) for name, value in self.owned.items()}
        self.before = {name: self.shared_snapshots[name] if name in self.shared_snapshots else value.clone()
                       for name, value in self.owned.items()}
        self.schema = structure(self.public)
        self.mutable_ids = {id(value) for value in self.mutable.values()}
        self.immutable_storage = {}
        for name, value in self.owned.items():
            pointer = value.untyped_storage().data_ptr()
            if id(value) in self.mutable_ids or pointer in self.immutable_storage:
                continue
            import torch
            count = value.untyped_storage().nbytes()
            raw = torch.empty(0, dtype=torch.uint8, device=value.device).set_(
                value.untyped_storage(), 0, (count,), (1,))
            snapshot = (self.before[name].view(torch.uint8).reshape(-1)
                        if value.is_contiguous() and value.storage_offset() == 0
                        and value.numel() * value.element_size() == count else raw.clone())
            self.immutable_storage[pointer] = (raw, snapshot)

    def restore(self):
        for name, value in self.owned.items():
            if id(value) in self.mutable_ids:
                value.copy_(self.before[name])

    def integrity(self):
        import torch
        current = tensors(self.public)
        if set(current) != set(self.owned) or structure(self.public) != self.schema:
            raise ValueError(f"{self.name}: candidate replaced the tensor interface")
        for name, value in self.owned.items():
            if current[name] is not value or layout(value) != self.layouts[name]:
                raise ValueError(f"{self.name}: candidate replaced or resized {name}")
        for raw, snapshot in self.immutable_storage.values():
            if not torch.equal(raw, snapshot):
                raise ValueError(f"{self.name}: candidate changed input/weight storage")
        for name, value in self.mutable.items():
            mask = self.masks[name]
            if mask is not None:
                before = next(self.before[key] for key, owned in self.owned.items() if owned is value)
                if not same_bits(value[~mask], before[~mask]):
                    raise ValueError(f"{self.name}: write outside admitted {name} region")

    def substitute(self, other):
        import torch
        if self.group != other.group or set(self.owned) != set(other.owned):
            raise ValueError("Graph replay requires the same structural domain")
        for name, dst in self.owned.items():
            src = other.owned[name]
            if tuple(dst.shape) != tuple(src.shape) or dst.dtype != src.dtype:
                raise ValueError("Graph replay tensor contract differs")
            if name.startswith(".weights"):
                if not same_bits(dst, src):
                    raise ValueError("Graph replay cannot change prepared weights")
                continue
            dst.copy_(other.before[name])
            self.before[name] = other.before[name].clone()
        for raw, snapshot in self.immutable_storage.values():
            snapshot.copy_(raw)
        self.expected = other.expected
        self.integrity()


def assert_tensor(actual, expected, name):
    import torch
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError(f"{name}: output shape or dtype differs")
    if not torch.isfinite(actual).all() or not same_bits(actual, expected):
        raise ValueError(f"{name}: output differs from strict captured/native equality")
