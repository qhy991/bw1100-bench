"""Capture the original five cases through Cake emission or a separate native probe."""
from pathlib import Path
import argparse
import importlib.util
import json
import sys
import time

import torch

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
sys.path.insert(0, str(R / '.deps/cake-ir/src'))
DTYPES = {'fp32': torch.float32, 'bf16': torch.bfloat16,
          'fp16': torch.float16, 'int32': torch.int32}


def native_output_contract(workload, case_id, device='cuda:0'):
    return {arg.name: dict(shape=list(arg.shape), dtype=str(DTYPES[arg.dtype]),
                           contiguous=True, device=device)
            for arg in workload.tensor_abi(case_id) if arg.mode == 'output'}


def native_outputs(workload, case_id, output, device='cuda:0'):
    expected = native_output_contract(workload, case_id, device)
    values = (output,) if isinstance(output, torch.Tensor) else output
    if not isinstance(values, (list, tuple)) or len(values) != len(expected):
        raise ValueError('Native return count does not match the original Workload ABI')
    observed, metadata = {}, {}
    for (name, contract), value in zip(expected.items(), values, strict=True):
        if not isinstance(value, torch.Tensor):
            raise ValueError('Native output must be an original ABI tensor: ' + name)
        metadata[name] = dict(shape=list(value.shape), dtype=str(value.dtype),
                              contiguous=value.is_contiguous(), device=str(value.device))
        if metadata[name] != contract:
            raise ValueError('Native tensor shape, dtype, layout or logical device differs: ' + name)
        observed[name] = value.reshape(-1).cpu().tolist()
    return observed, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    route = parser.add_mutually_exclusive_group(required=True)
    route.add_argument('--emission')
    route.add_argument('--native-manifest')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output_path = R / args.output
    if output_path.exists():
        raise FileExistsError(output_path)
    workload = None
    if args.native_manifest:
        from campaign.hip_probe import load_entry, own_path, read, verify_files
        from open_cake_ir.tasks.workloads import load_workload
        manifest = read(own_path(R, args.native_manifest))
        if manifest.get('implementation_kind') != 'native_hip' or manifest.get('entry_symbol') != 'run':
            raise ValueError('Expected a separate native probe manifest')
        build = read(own_path(R, args.native_manifest).with_name('build.json'))
        verify_files(R, manifest, build)
        fn = load_entry(R, manifest).run
        workload = load_workload(R / 'campaign/workload.json')
    else:
        record = json.loads((R / args.emission).read_text())
        spec = importlib.util.spec_from_file_location('own_generated', R / record['source'])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        fn = getattr(module, record['entry'])
    prepared = torch.load(R / 'campaign/development-inputs.pt', weights_only=False)
    rows = []
    if workload is not None and [case['case_id'] for case in prepared] != list(workload.case_ids):
        raise ValueError('Native probe must preserve all original cases')
    for case in prepared:
        values = {name: value.cuda() for name, value in case['tensors'].items()}
        out = fn(**values)
        torch.cuda.synchronize()
        if workload is not None:
            observed, metadata = native_outputs(workload, case['case_id'], out)
        else:
            outputs = (out,) if isinstance(out, torch.Tensor) else tuple(out)
            observed = {name: value.reshape(-1).cpu().tolist() for name, value in zip(record['outputs'], outputs)}
        samples = []
        if case['case_id'] == 'primary':
            for _ in range(10):
                fn(**values)
            torch.cuda.synchronize()
            for _ in range(30):
                start = time.perf_counter()
                out = fn(**values)
                torch.cuda.synchronize()
                samples.append((time.perf_counter() - start) * 1e6)
        row = dict(case_id=case['case_id'], outputs=observed,
                   after_inputs={name: value.reshape(-1).cpu().tolist() for name, value in values.items()},
                   samples_us=samples)
        if workload is not None:
            row.update(implementation_kind='native_hip', output_metadata=metadata)
        rows.append(row)
        del values, out
        torch.cuda.synchronize()
    if workload is not None:
        verify_files(R, manifest, build)
    with output_path.open('xb') as stream:
        torch.save(rows, stream)
    print('device snapshots retained', len(rows))


if __name__ == '__main__':
    main()
