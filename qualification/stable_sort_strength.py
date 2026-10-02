"""Paired, original-input screen of two shipped native-library offset routes."""
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


def timed(fn, inputs):
    torch.cuda.synchronize()
    start = time.perf_counter()
    fn(*inputs)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--search-gate', type=Path, required=True)
    parser.add_argument('--histogram-gate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    for gate in (args.search_gate, args.histogram_gate):
        d = json.loads(gate.read_text())
        if not d.get('full_device_correctness') or d['status'] != 'passed':
            raise ValueError('missing full gate: ' + str(gate))
    if args.output.exists():
        raise FileExistsError(args.output)
    bwbench.upstream()
    task = bwbench.task_record('L1/058_moe_expert_token_radix_sort_with_prefix_sum')
    definition, workloads, _, path = bwbench.load_problem(task)
    reference = bwbench.load_module(path / 'reference.py', 'sort_input_factory')
    custom = getattr(reference, definition.custom_inputs_entrypoint)
    source = ROOT / 'baselines/l1_058_aten_stable_search.py'
    mod = bwbench.load_module(source, 'community_sort')
    from sol_execbench.core.bench.io import gen_inputs
    from sol_execbench.core.bench.correctness import set_seed
    set_seed(200)
    torch.set_num_threads(4)
    rows = []
    for w in workloads:
        inputs = gen_inputs(definition, w, 'cuda:0', custom_inputs_fn=custom)
        snapshot = inputs[0].clone()
        for _ in range(10):
            mod.run(*inputs)
            mod.run_histogram(*inputs)
        samples = {k: [] for k in ('sf', 'hf', 'hr', 'sr', 'aa1', 'aa2')}
        for _ in range(30):
            samples['sf'].append(timed(mod.run, inputs))
            samples['hf'].append(timed(mod.run_histogram, inputs))
        for _ in range(30):
            samples['hr'].append(timed(mod.run_histogram, inputs))
            samples['sr'].append(timed(mod.run, inputs))
        for _ in range(30):
            samples['aa1'].append(timed(mod.run_histogram, inputs))
            samples['aa2'].append(timed(mod.run_histogram, inputs))
        med = {k: statistics.median(v) for k, v in samples.items()}
        drift = abs(med['aa1'] - med['aa2']) / min(med['aa1'], med['aa2'])
        row = {'uuid': w.uuid, 'axes': w.axes, 'elements': inputs[0].numel(),
               'wall_median_us': med, 'forward_hist_over_search': med['hf'] / med['sf'],
               'reverse_hist_over_search': med['hr'] / med['sr'],
               'aa_relative_drift': drift, 'inputs_unmutated': torch.equal(inputs[0], snapshot),
               'samples_us': samples}
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != 'samples_us'}), flush=True)
    result = {'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'gate_sha256': [hashlib.sha256(g.read_bytes()).hexdigest()
                              for g in (args.search_gate, args.histogram_gate)],
              'schema': 'bw1100-bench.stable-sort-community-strength.v1',
              'mode': 'paired_callable_wall_no_profiler', 'sources': bwbench.document('sources.lock.json'),
              'rows': rows}
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
