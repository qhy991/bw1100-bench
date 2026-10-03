"""Common original-workload timing for both engineering arms."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def measure(fn, inputs, samples):
    import torch
    values = []
    for _ in range(samples):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = fn(*inputs)
        torch.cuda.synchronize()
        values.append((time.perf_counter() - start) * 1e6)
        del result
    return values


def main():
    import torch
    import bwbench
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', required=True, type=Path)
    parser.add_argument('--gate', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--samples', type=int, default=30)
    parser.add_argument('--baseline', type=Path, help='explicit parent comparison; default remains community denominator')
    args = parser.parse_args()
    intake = json.loads((ROOT / 'campaign/intake.json').read_text())
    plan = json.loads((ROOT / intake['plan']).read_text())
    task_id = plan['tasks'][0]['id']
    baseline_path = ROOT / plan['tasks'][0]['baseline']
    if args.baseline:
        baseline_path = args.baseline.resolve()
        baseline_path.relative_to(ROOT)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    gate = json.loads(args.gate.read_text())
    if not (gate.get('full_device_correctness') and gate['status'] == 'passed' and gate['task'] == task_id):
        raise ValueError('original full candidate gate is required')
    if Path(gate['candidate']['path']).resolve() != args.candidate.resolve():
        raise ValueError('gate names another candidate')
    if args.output.exists():
        raise FileExistsError(args.output)
    bwbench.upstream()
    from sol_execbench.core.bench.correctness import set_seed
    from sol_execbench.core.bench.io import gen_inputs
    definition, workloads, _, folder = bwbench.load_problem(bwbench.task_record(task_id))
    factory = bwbench.load_module(folder / 'reference.py', 'common_input_factory')
    custom_name = getattr(definition, 'custom_inputs_entrypoint', None)
    custom = getattr(factory, custom_name) if custom_name else None
    baseline = bwbench.load_module(baseline_path, 'common_baseline').run
    candidate = bwbench.load_module(args.candidate, 'common_candidate').run
    torch.set_num_threads(4)
    rows = []
    for workload in workloads:
        set_seed(200)
        inputs = gen_inputs(definition, workload, 'cuda:0', custom_inputs_fn=custom)
        snapshots = [v.clone() if isinstance(v, torch.Tensor) else v for v in inputs]
        for _ in range(10):
            baseline(*inputs)
            candidate(*inputs)
        streams = {k: [] for k in ('bf', 'cf', 'br', 'cr', 'a1', 'a2')}
        for _ in range(args.samples):
            streams['bf'] += measure(baseline, inputs, 1)
            streams['cf'] += measure(candidate, inputs, 1)
        for _ in range(args.samples):
            streams['cr'] += measure(candidate, inputs, 1)
            streams['br'] += measure(baseline, inputs, 1)
        for _ in range(args.samples):
            streams['a1'] += measure(baseline, inputs, 1)
            streams['a2'] += measure(baseline, inputs, 1)
        med = {k: statistics.median(v) for k, v in streams.items()}
        unchanged = all(torch.equal(v, s) if isinstance(v, torch.Tensor) else v == s
                        for v, s in zip(inputs, snapshots))
        row = {'workload_uuid': workload.uuid, 'axes': workload.axes,
               'inputs_unmutated_after_measurement': unchanged,
               'baseline_us': {'median': statistics.median(streams['bf'] + streams['br']),
                               'forward_median': med['bf'], 'reverse_median': med['br']},
               'candidate_us': {'median': statistics.median(streams['cf'] + streams['cr']),
                                'forward_median': med['cf'], 'reverse_median': med['cr']},
               'aa_control_baseline_us': {'abs_drift': abs(med['a1'] - med['a2'])},
               'samples_us': streams}
        rows.append(row)
        print(workload.uuid, row['baseline_us']['median'], row['candidate_us']['median'], flush=True)
        del inputs, snapshots
        torch.cuda.empty_cache()
    result = {'task': task_id, 'arm': plan['arm'], 'method': 'paired_complete_callable_wall',
              'seed': 200, 'warmup_calls': 10, 'samples_per_direction': args.samples,
              'candidate_source_sha256': sha(args.candidate),
              'baseline_source_sha256': sha(baseline_path), 'rows': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
