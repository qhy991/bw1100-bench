"""Prepare the canonical starter and original inputs inside the Run's CPU budget."""
from pathlib import Path
import json
import sys
import torch

R = Path('/work')
C = R / '.deps/cake-ir'
sys.path.insert(0, str(C / 'src'))
sys.path.insert(0, str(R))
from open_cake_ir.compiler import Compiler, frontend
from open_cake_ir.tasks.workloads import load_workload, materialize_evaluation_case
from development_binding import reconcile
from development_device import input_tensor
from development_task_source import task_source

b = reconcile(R, mounted=True)
c = Compiler.load(C, C / 'compiler/revision.json')
assert c.commit == b['compiler']
contract = dict(rows=b['rows'], columns=b['columns'], depth=b['depth'],
                workload=json.loads((R / 'campaign/original-workload.json').read_text()))
doc, source = task_source(b['task'], contract)
with (R / 'campaign/workload.json').open('x') as stream:
    json.dump(doc, stream, indent=2)
with (R / 'campaign/candidates/starter.py').open('x') as stream:
    stream.write(source)
w = load_workload(R / 'campaign/workload.json')
assert list(w.case_ids) == b['case_ids']
a = c.assess(frontend.parse(source).document)
with (R / 'campaign/starter-assessment.json').open('x') as stream:
    json.dump(dict(accepted=a.accepted, lowering_eligible=a.lowering_eligible,
                   findings=[finding.to_dict() for finding in a.findings]), stream, indent=2)
assert a.lowering_eligible
cases = []
for cid in w.case_ids:
    abi = [dict(name=arg.name, dtype=arg.dtype, shape=list(arg.shape), mode=arg.mode)
           for arg in w.tensor_abi(cid)]
    assert abi == b['abi'], 'original case ABI changed'
    inputs, expected = materialize_evaluation_case(w, cid)
    tensors = {}
    for arg in w.tensor_abi(cid):
        if arg.mode == 'input':
            tensors[arg.name] = input_tensor(inputs[arg.name], arg)
    cases.append(dict(case_id=cid, tensors=tensors, flat_inputs=inputs, expected=expected))
torch.save(cases, R / 'campaign/development-inputs.pt')
print('prepared canonical Task', b['task'], len(cases), 'cases, no GPU')
