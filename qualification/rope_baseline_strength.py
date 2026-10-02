"""Screen the public dispatcher and shipped RoPE arms with paired wall latency."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import bwbench


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def timed(fn, inputs):
    torch.cuda.synchronize()
    started = time.perf_counter()
    fn(*inputs)
    torch.cuda.synchronize()
    return (time.perf_counter() - started) * 1e6


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--gate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = json.loads(args.gate.read_text())
    if not (report.get('full_device_correctness') and report['status'] == 'passed'
            and report['task'] == 'L1/011_rotary_position_embedding'
            and report['candidate']['symbol'] == 'run'
            and Path(report['candidate']['path']).resolve() == args.source.resolve()):
        raise ValueError('missing full original device gate: ' + str(args.gate))
    bwbench.upstream()
    task = bwbench.task_record('L1/011_rotary_position_embedding')
    definition, workloads, _, path = bwbench.load_problem(task)
    reference = bwbench.load_module(path / 'reference.py', 'input_factory')
    custom = getattr(reference, definition.custom_inputs_entrypoint)
    module = bwbench.load_module(args.source, 'community_rope')
    from sol_execbench.core.bench.correctness import set_seed
    from sol_execbench.core.bench.io import gen_inputs
    set_seed(200)
    torch.set_num_threads(4)
    rows = []
    for workload in workloads:
        inputs = gen_inputs(definition, workload, 'cuda:0', custom_inputs_fn=custom)
        snapshots = [item.clone() for item in inputs if isinstance(item, torch.Tensor)]
        for _ in range(10):
            module.run_eager(*inputs)
            module.run_compiled(*inputs)
            module.run(*inputs)
        samples = {name: [] for name in ('df', 'ef', 'cf', 'cr', 'er', 'dr', 'a1', 'a2')}
        for _ in range(30):
            samples['df'].append(timed(module.run, inputs))
            samples['ef'].append(timed(module.run_eager, inputs))
            samples['cf'].append(timed(module.run_compiled, inputs))
        for _ in range(30):
            samples['cr'].append(timed(module.run_compiled, inputs))
            samples['er'].append(timed(module.run_eager, inputs))
            samples['dr'].append(timed(module.run, inputs))
        for _ in range(30):
            samples['a1'].append(timed(module.run, inputs))
            samples['a2'].append(timed(module.run, inputs))
        med = {key: statistics.median(values) for key, values in samples.items()}
        drift = abs(med['a1'] - med['a2']) / min(med['a1'], med['a2'])
        forward, reverse = med['ef'] / med['cf'], med['er'] / med['cr']
        selected = module.run_eager if inputs[0].numel() > 8192 else module.run_compiled
        pairs = {name: [] for name in ('df', 'sf', 'sr', 'dr', 'sa1', 'sa2', 'da1', 'da2')}
        # Isolate dispatch cost from the three-arm order effect: the eager
        # arm between two compiled calls can change the next call's timing.
        for _ in range(30):
            pairs['df'].append(timed(module.run, inputs))
            pairs['sf'].append(timed(selected, inputs))
        for _ in range(30):
            pairs['sr'].append(timed(selected, inputs))
            pairs['dr'].append(timed(module.run, inputs))
        for _ in range(30):
            pairs['sa1'].append(timed(selected, inputs))
            pairs['sa2'].append(timed(selected, inputs))
            pairs['da1'].append(timed(module.run, inputs))
            pairs['da2'].append(timed(module.run, inputs))
        pair_med = {key: statistics.median(values) for key, values in pairs.items()}
        unchanged = all(torch.equal(item, snap) for item, snap in zip(
            (item for item in inputs if isinstance(item, torch.Tensor)), snapshots))
        row = {'workload_uuid': workload.uuid, 'axes': workload.axes,
               'wall_median_us': med, 'forward_eager_over_compiled': forward,
               'reverse_eager_over_compiled': reverse, 'aa_relative_drift': drift,
               'compiled_win_in_both_directions': unchanged and
                   min(forward, reverse) - 1 > max(drift, 0.01),
               'public_route': 'eager' if inputs[0].numel() > 8192 else 'compiled',
               'dispatch_over_selected_forward': med['df'] / (
                   med['ef'] if inputs[0].numel() > 8192 else med['cf']),
               'dispatch_over_selected_reverse': med['dr'] / (
                   med['er'] if inputs[0].numel() > 8192 else med['cr']),
               'dispatch_pair_wall_us': pair_med,
               'dispatch_pair_over_selected_forward': pair_med['df'] / pair_med['sf'],
               'dispatch_pair_over_selected_reverse': pair_med['dr'] / pair_med['sr'],
               'dispatch_pair_samples_us': pairs,
               'inputs_unmutated': unchanged, 'samples_us': samples}
        rows.append(row)
        print(json.dumps({key: value for key, value in row.items()
                          if key not in ('samples_us', 'dispatch_pair_samples_us')}), flush=True)
    result = {'schema': 'bw1100-bench.community-rope-strength.v2',
              'source_sha256': digest(args.source),
              'full_gate_sha256': digest(args.gate),
              'sources': bwbench.document('sources.lock.json'),
              'mode': 'paired_callable_wall_no_profiler', 'seed': 200,
              'control': 'dispatcher A/A, forward and reverse three-arm order', 'rows': rows}
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
