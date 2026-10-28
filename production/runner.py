"""Correctness-only entry for production tasks; device admission stays with dtk.sh."""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
from pathlib import Path

from . import catalog


def candidate_module(path):
    tree = ast.parse(path.read_text(), str(path))
    forbidden = {"vllm", "vllm_hcu", "production", "native", "native_data2", "reference"}
    for node in ast.walk(tree):
        modules = ([item.name for item in node.names] if isinstance(node, ast.Import)
                   else [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
        if any(module.split(".")[0] in forbidden for module in modules):
            raise ValueError("Candidate imports a native/reference/evaluator module")
    spec = importlib.util.spec_from_file_location("production_candidate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "prepare", None)):
        raise ValueError("Candidate must export prepare(case) -> callable")
    return module


def runtime(device):
    import importlib.metadata
    import torch
    if not torch.version.hip or not torch.cuda.is_available():
        raise catalog.PrerequisiteError("Production check requires the admitted Hygon HIP runtime")
    selected = torch.device(device)
    if selected.type != "cuda":
        raise catalog.PrerequisiteError("Production device checks require an admitted HIP device")
    torch.cuda.set_device(selected)
    arch = torch.cuda.get_device_properties(selected).gcnArchName.split(":", 1)[0]
    if arch != "gfx938":
        raise catalog.PrerequisiteError(f"Expected gfx938, got {arch}")
    version = importlib.metadata.version("vllm")
    if not version.startswith("0.28.1+dtk2610"):
        raise catalog.PrerequisiteError("Expected the data2 vLLM 0.28.1+dtk2610 runtime")
    return {"arch": arch, "torch": torch.__version__, "hip": torch.version.hip, "vllm": version}


def eager_calls(cases, candidate, validate, synchronize):
    """Check the caller-owned ABI and retain every earlier output across calls."""
    import torch
    prepared = []
    retained = []
    for case in cases:
        call = candidate.prepare(case.public)
        if not callable(call):
            raise ValueError("prepare(case) did not return a callable")
        case.integrity()
        case.restore()
        call()
        synchronize()
        case.integrity()
        validate(case)
        for old, snapshot in retained:
            if not torch.equal(old, snapshot):
                raise ValueError("Candidate overwrote a retained earlier output")
        retained.append((case.public["output"], case.public["output"].clone()))
        prepared.append(call)
    return prepared, len(retained)


def verify_calls(cases, candidate, validate):
    """Eager retention, then a changed-content graph and unsynchronized-stream chain."""
    import torch
    prepared, retained_count = eager_calls(cases, candidate, validate, torch.cuda.synchronize)
    by_group = {}
    for i, case in enumerate(cases):
        by_group.setdefault(case.group, []).append(i)
    graph_domains = 0
    for indices in by_group.values():
        if len(indices) < 2:
            raise ValueError("Changed-content graph check requires two cases per structural domain")
        i, j = indices[:2]
        case, other = cases[i], cases[j]
        case.restore()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            prepared[i]()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        case.substitute(other)
        graph.replay()
        torch.cuda.synchronize()
        case.integrity()
        validate(case)
        graph_domains += 1
    for start in range(0, len(cases), 16):
        batch = cases[start:start + 16]
        calls = prepared[start:start + 16]
        for case in batch:
            case.restore()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        delay = torch.empty(1 << 20, device="cuda")
        with torch.cuda.stream(stream):
            delay.fill_(1)
            for _ in range(8):
                delay.mul_(1.000001)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for call in calls:
                call()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        for case in batch:
            case.restore()
        graph.replay()
        torch.cuda.synchronize()
        for case in batch:
            case.integrity()
            validate(case)
    return {"cases": len(cases), "retained_outputs": retained_count,
            "changed_content_graph_domains": graph_domains, "graph_chain_length": 16,
            "stream_order": "queued_work_before_capture"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    for name in ("audit", "check"):
        command = sub.add_parser(name)
        command.add_argument("--task", required=True)
        command.add_argument("--inputs", type=Path)
        command.add_argument("--model", type=Path)
        command.add_argument("--output", type=Path, required=name == "check")
        if name == "check":
            command.add_argument("--candidate", type=Path, required=True)
            command.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if args.command == "list":
        print(json.dumps(catalog.tasks(), ensure_ascii=False, indent=2))
        return 0
    report = {"task": args.task, "status": "blocked", "performance": "not_measured",
              "device_check_executed": False, "numeric_tolerance_calibrated": False}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as out:
            json.dump(report, out)
    try:
        task = catalog.task_record(args.task)
        report["revision"] = task["revision"]
        report["audit"] = catalog.audit(task, args.inputs, args.model)
        if args.command == "audit":
            report["status"] = "audit_passed"
        else:
            if args.task == "glm53-kda-verify" and (args.model is None or not args.model.is_dir()):
                raise catalog.PrerequisiteError("KDA check requires --model with the data2 checkpoint config")
            report["runtime"] = runtime(args.device)
            module = candidate_module(args.candidate)
            if args.task == "glm53-kda-verify":
                from . import kda
                cases = kda.cases(args.inputs, args.model, args.device)
                report["checks"] = verify_calls(cases, module, kda.validate)
            else:
                from . import lmhead
                cases = lmhead.cases(args.model, args.device)
                report["checks"] = verify_calls(cases, module, lmhead.validate)
            report["status"] = "passed"
            report["device_check_executed"] = True
    except Exception as exc:
        report["reason"] = f"{type(exc).__name__}: {exc}"
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] in {"passed", "audit_passed"} else 2
