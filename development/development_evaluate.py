"""Canonical Task/Compiler/oracle adapter; callable samples are diagnostic only."""
from pathlib import Path
import argparse
import json
import os
import re
import sys

R = Path(__file__).resolve().parents[1]
C = R / '.deps/cake-ir'
sys.path.insert(0, str(C / 'src'))
sys.path.insert(0, str(R))
from open_cake_ir.compiler import Compiler, frontend
from open_cake_ir.tasks.workloads import load_workload
from open_cake_ir.evaluation.core import compare_tile_outputs
from development_binding import reconcile

p = argparse.ArgumentParser()
p.add_argument('--candidate', required=True)
p.add_argument('--id', required=True)
p.add_argument('--phase', choices=['emit', 'compare', 'validate'], required=True)
p.add_argument('--result-phase', choices=['search', 'confirmation'], default='search')
a = p.parse_args()
assert re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,63}', a.id)
os.chdir(R)
b = reconcile(R, mounted=True)
w = load_workload(R / 'campaign/workload.json')
c = Compiler.load(C, C / 'compiler/revision.json')
assert c.commit == b['compiler']
assert list(w.case_ids) == b['case_ids'] and len(w.case_ids) == 5
folder = R / 'campaign/evaluations' / a.id

if a.phase in ('emit', 'validate'):
    source = (R / a.candidate).resolve()
    source.relative_to(R)
    if a.phase == 'emit':
        folder.mkdir(parents=True, exist_ok=False)
    else:
        assert source == (folder / 'candidate.py').resolve(), 'nominee must use its retained source snapshot'
    text = source.read_text()
    parsed = frontend.parse(text)
    assessment = c.assess(parsed.document)
    if a.phase == 'emit':
        (folder / 'candidate.py').write_text(text)
        (folder / 'assessment.json').write_text(json.dumps(dict(
            findings=[f.to_dict() for f in assessment.findings], accepted=assessment.accepted,
            lowering_eligible=assessment.lowering_eligible), indent=2))
    if not assessment.lowering_eligible:
        if a.phase == 'validate':
            raise ValueError('retained nominee is no longer lowering-eligible')
        (folder / 'outcome.json').write_text(json.dumps(dict(id=a.id, status='refused',
            phase=a.result_phase, scope='canonical Task construction/verifier refusal; no GPU'), indent=2))
        print('refused; diagnostics retained')
        raise SystemExit(0)
    assert assessment.target == 'gfx938', 'candidate changes the exact Target'
    schedule = assessment.typed_schedule
    abi = {x.name: (tuple(x.shape), x.dtype, x.mode) for x in w.tensor_abi(w.case_ids[0])}
    actual = {x.name: (x.shape, x.dtype.value, x.mode.value) for x in schedule.buffers if x.space.value == 'global'}
    assert actual == abi, 'candidate changes Task ABI'
    lowering = c.lower(assessment)
    emission = dict(source=str((folder / 'source.py').relative_to(R)),
                    entry=lowering.route.entry_point, outputs=list(schedule.outputs), compiler=c.commit)
    if a.phase == 'validate':
        outcome = json.loads((folder / 'outcome.json').read_text())
        assert outcome['id'] == a.id and outcome['compiler'] == c.commit
        assert outcome['status'] == 'accepted' and outcome['phase'] == 'search'
        assert outcome['candidate'] == str((folder / 'candidate.py').relative_to(R))
        assert outcome['source'] == emission['source']
        assert [x['case_id'] for x in outcome['checks']] == list(w.case_ids)
        assert all(x['passed'] is True for x in outcome['checks'])
        assert json.loads((folder / 'schedule.json').read_text()) == parsed.document
        assert json.loads((folder / 'emission.json').read_text()) == emission
        assert (folder / 'source.py').read_text() == lowering.source, 'retained emission differs from pinned Compiler'
        print('validated fixed-source five-case nominee; CPU only')
    else:
        (folder / 'source.py').write_text(lowering.source)
        (folder / 'schedule.json').write_text(json.dumps(parsed.document, indent=2))
        (folder / 'emission.json').write_text(json.dumps(emission, indent=2))
        print('generated own source; no GPU')
else:
    import torch
    prepared = torch.load(R / 'campaign/development-inputs.pt', weights_only=False)
    observed = torch.load(folder / 'observed.pt', weights_only=False)
    assert [x['case_id'] for x in prepared] == list(w.case_ids)
    assert [x['case_id'] for x in observed] == list(w.case_ids)
    checks = []
    for case, seen in zip(prepared, observed, strict=True):
        ok, metrics = compare_tile_outputs(w, case['flat_inputs'], case['expected'], seen['outputs'], seen['after_inputs'])
        checks.append(dict(case_id=case['case_id'], passed=ok, metrics=metrics))
    result = dict(id=a.id, phase=a.result_phase,
                  status='accepted' if all(x['passed'] for x in checks) else 'numerically_failed',
                  compiler=c.commit, scope='registered Task development; diagnostic callable samples, no performance win',
                  checks=checks, primary_us=next(x['samples_us'] for x in observed if x['case_id'] == 'primary'),
                  candidate=str((folder / 'candidate.py').relative_to(R)), source=str((folder / 'source.py').relative_to(R)),
                  diagnosis_owner='inspect candidate/contract before promoting Compiler Finding')
    (folder / 'outcome.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
