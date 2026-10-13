"""Capture the original task-owned cases through Cake emission or a separate native probe."""
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


def input_tensor(value, arg):
    tensor = (value.detach().cpu().clone() if isinstance(value, torch.Tensor)
              else torch.tensor(value, dtype=DTYPES[arg.dtype]))
    if tensor.dtype != DTYPES[arg.dtype]:
        raise ValueError('Materialized input dtype differs from its Workload ABI: ' + arg.name)
    return tensor.reshape(arg.shape)


def input_snapshots(original, values):
    return {name: (value.detach().cpu().clone() if isinstance(original[name], torch.Tensor)
                   else value.reshape(-1).cpu().tolist()) for name, value in values.items()}


def native_output_contract(workload, case_id, device='cuda:0'):
    return {arg.name: dict(shape=list(arg.shape), dtype=str(DTYPES[arg.dtype]),
                           contiguous=True, device=device)
            for arg in workload.tensor_abi(case_id) if arg.mode == 'output'}


def native_outputs(workload, case_id, output, device='cuda:0', inputs=(), names=None):
    expected = native_output_contract(workload, case_id, device)
    values = (output,) if isinstance(output, torch.Tensor) else output
    if not isinstance(values, (list, tuple)) or len(values) != len(expected):
        raise ValueError('Return count does not match the original Workload ABI')
    if names is not None:
        if len(names) != len(expected) or set(names) != set(expected):
            raise ValueError('Cake emission outputs differ from the Workload ABI')
        named = dict(zip(names, values, strict=True))
        values = tuple(named[name] for name in expected)
    observed, metadata = {}, {}
    storages = [value.untyped_storage().data_ptr() for value in inputs if value.numel()]
    for (name, contract), value in zip(expected.items(), values, strict=True):
        if not isinstance(value, torch.Tensor):
            raise ValueError('Output must be an original ABI tensor: ' + name)
        metadata[name] = dict(shape=list(value.shape), dtype=str(value.dtype),
                              contiguous=value.is_contiguous(), device=str(value.device))
        if metadata[name] != contract:
            raise ValueError('Tensor shape, dtype, layout or logical device differs: ' + name)
        pointer = value.untyped_storage().data_ptr()
        if (workload.document.get('semantics', {}).get('output_storage', '').startswith('fresh')
                and value.numel() and pointer in storages):
            raise ValueError('Output storage overlaps an input or another output: ' + name)
        if value.numel():
            storages.append(pointer)
        observed[name] = value.detach().reshape(-1).cpu().tolist()
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
    from open_cake_ir.tasks.workloads import load_workload
    workload = load_workload(R / 'campaign/workload.json')
    if args.native_manifest:
        from campaign.hip_probe import load_entry, own_path, read, verify_files
        manifest = read(own_path(R, args.native_manifest))
        if manifest.get('implementation_kind') != 'native_hip' or manifest.get('entry_symbol') != 'run':
            raise ValueError('Expected a separate native probe manifest')
        build = read(own_path(R, args.native_manifest).with_name('build.json'))
        verify_files(R, manifest, build)
        fn = load_entry(R, manifest).run
    else:
        record = json.loads((R / args.emission).read_text())
        spec = importlib.util.spec_from_file_location('own_generated', R / record['source'])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        fn = getattr(module, record['entry'])
    prepared = torch.load(R / 'campaign/development-inputs.pt', weights_only=False)
    rows = []
    if [case['case_id'] for case in prepared] != list(workload.case_ids):
        raise ValueError('Device snapshot must preserve all original cases')
    for case in prepared:
        values = {name: value.cuda() for name, value in case['tensors'].items()}
        out = fn(**values)
        torch.cuda.synchronize()
        observed, metadata = native_outputs(workload, case['case_id'], out, inputs=values.values(),
                                            names=None if args.native_manifest else record['outputs'])
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
                   after_inputs=input_snapshots(case['flat_inputs'], values),
                   samples_us=samples)
        if args.native_manifest:
            row.update(implementation_kind='native_hip', output_metadata=metadata)
        rows.append(row)
        del values, out
        torch.cuda.synchronize()
    if args.native_manifest:
        verify_files(R, manifest, build)
    with output_path.open('xb') as stream:
        torch.save(rows, stream)
    print('device snapshots retained', len(rows))


if __name__ == '__main__':
    main()
