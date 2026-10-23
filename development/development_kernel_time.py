"""Device kernel durations for one development candidate, separate from its callable samples.

A callable sample brackets a Python call and a synchronize. On gfx938 that floor is about
50 us, so it hides most schedule changes (bw1100-bench#34). rocprof timestamps give the
generated kernel's own BeginNs/EndNs. They come from a second, profiled admission after
correctness, so the unprofiled samples and the correctness result stay as they were.
"""
from pathlib import Path
import argparse
import csv
import importlib.util
import json
import sys

R = Path(__file__).resolve().parents[1]
REPETITIONS = 30
WARMUP = 10
PMC = 'pmc: GPUBusy\n'


def kernel_names(source):
    """The @triton.jit functions the generated module defines; only these are the candidate."""
    import ast
    names = set()
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and any(
                (isinstance(d, ast.Attribute) and d.attr == 'jit')
                or (isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute) and d.func.attr == 'jit')
                for d in node.decorator_list):
            names.add(node.name)
    return names


def summarize(rows, names, launches):
    """Sum each launch's candidate kernels, then take the median over launches.

    rocprof reports the kernel symbol with a `.kd` suffix. Warmup launches precede the
    timed ones in dispatch order, so the last `launches` groups are the timed ones.
    """
    durations = []
    for row in rows:
        kernel = row.get('KernelName', '').strip('"')
        if kernel.endswith('.kd'):
            kernel = kernel[:-3]
        if kernel in names:
            durations.append((int(row['DispatchNs']), int(row['EndNs']) - int(row['BeginNs'])))
    durations.sort()
    if not durations or len(durations) % (WARMUP + launches):
        raise ValueError('kernel records %d do not split into %d launches' % (len(durations), WARMUP + launches))
    per_launch = len(durations) // (WARMUP + launches)
    timed = [sum(ns for _, ns in durations[i:i + per_launch]) / 1e3
             for i in range(per_launch * WARMUP, len(durations), per_launch)]
    timed.sort()
    return dict(kernels_per_launch=per_launch, launches=len(timed),
                median_us=round(timed[len(timed) // 2], 3), min_us=round(timed[0], 3),
                max_us=round(timed[-1], 3),
                scope='rocprof BeginNs..EndNs of the generated kernels on the primary case, '
                      'summed per launch; excludes host launch and synchronize; profiled '
                      'admission separate from correctness and callable samples')


def run(emission):
    """Inside the admitted container: launch the primary case the timed number of times."""
    import torch
    from development_device import input_tensor
    sys.path.insert(0, str(R / '.deps/cake-ir/src'))
    record = json.loads((R / emission).read_text())
    spec = importlib.util.spec_from_file_location('own_generated', R / record['source'])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, record['entry'])
    prepared = torch.load(R / 'campaign/development-inputs.pt', weights_only=False)
    case = next(case for case in prepared if case['case_id'] == 'primary')
    values = {name: value.cuda() for name, value in case['tensors'].items()}
    for _ in range(WARMUP + REPETITIONS):
        fn(**values)
    torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--emission', required=True)
    parser.add_argument('--summarize')
    parser.add_argument('--output')
    args = parser.parse_args()
    if args.summarize is None:
        run(args.emission)
        return
    record = json.loads((R / args.emission).read_text())
    names = kernel_names((R / record['source']).read_text())
    with (R / args.summarize).open() as stream:
        rows = list(csv.DictReader(stream))
    with (R / args.output).open('x') as stream:
        json.dump(summarize(rows, names, REPETITIONS), stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
