"""CPU check of the complete original task contracts against a selected Compiler."""
from pathlib import Path
import argparse
import json
import sys


def catalog(compiler_source, commit, contracts, inherited=()):
    sys.path.insert(0, str(compiler_source / 'src'))
    from open_cake_ir.compiler import Compiler, frontend
    from open_cake_ir.tasks.workloads import create_task, WorkloadContract
    compiler = Compiler.load(compiler_source, compiler_source / 'compiler/revision.json')
    if compiler.commit != commit:
        raise ValueError('CPU catalog Compiler differs from the selected commit')
    for task, expected in contracts['tasks'].items():
        document, starter = create_task(task, backend='triton-dcu', **{
            key: expected[key] for key in ('rows', 'columns', 'depth')})
        if document != expected['workload']:
            raise ValueError(task + ': original whole Workload changed')
        workload = WorkloadContract(document)
        if list(workload.case_ids) != expected['case_ids'] or len(workload.case_ids) != 5:
            raise ValueError(task + ': original five-case set changed')
        for case in workload.case_ids:
            abi = [dict(name=x.name, dtype=x.dtype, shape=list(x.shape), mode=x.mode)
                   for x in workload.tensor_abi(case)]
            if abi != expected['abi']:
                raise ValueError(task + ': original case ABI changed')
        assessment = compiler.assess(frontend.parse(starter).document)
        if not assessment.lowering_eligible:
            raise ValueError(task + ': starter is not lowering eligible')
        compiler.lower(assessment)
    for row in inherited:
        task = row['task']
        assessment = compiler.assess(frontend.read_schedule(row['inherited']['source']).document)
        if not assessment.lowering_eligible or assessment.target != 'gfx938':
            raise ValueError(task + ': inherited candidate is not eligible on the original Target')
        actual = {buffer.name: (list(buffer.shape), buffer.dtype.value, buffer.mode.value)
                  for buffer in assessment.typed_schedule.buffers if buffer.space.value == 'global'}
        expected = {item['name']: (item['shape'], item['dtype'], item['mode'])
                    for item in contracts['tasks'][task]['abi']}
        if actual != expected:
            raise ValueError(task + ': inherited candidate changes the original ABI')
        compiler.lower(assessment)
    return {'tasks': len(contracts['tasks']), 'inherited_candidates': len(inherited), 'compiler': commit,
            'scope': 'CPU contract, starter and inherited emission only; no device qualification'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler-source', required=True, type=Path)
    parser.add_argument('--compiler', required=True)
    parser.add_argument('--contracts', required=True, type=Path)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--host', required=True)
    args = parser.parse_args()
    print(json.dumps(catalog(args.compiler_source, args.compiler,
                            json.loads(args.contracts.read_text()),
                            [row for row in json.loads(args.plan.read_text())['assignments']
                             if row['host'] == args.host])))
