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
from open_cake_ir import compiler as compiler_api
from open_cake_ir.tasks.workloads import load_workload
from open_cake_ir.evaluation.core import compare_tile_outputs
from development_binding import reconcile


class CandidateRefusal(ValueError):
    """A known source-input or adapter Task-contract refusal."""


def write_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def write_bytes_new(path, value):
    with path.open('xb') as stream:
        stream.write(value)


def compare_cases(workload, prepared, observed):
    assert [x['case_id'] for x in prepared] == list(workload.case_ids)
    assert [x['case_id'] for x in observed] == list(workload.case_ids)
    checks = []
    for case, seen in zip(prepared, observed, strict=True):
        ok, metrics = compare_tile_outputs(workload, case['flat_inputs'], case['expected'],
                                          seen['outputs'], seen['after_inputs'])
        checks.append(dict(case_id=case['case_id'], passed=ok, metrics=metrics))
    return checks


p = argparse.ArgumentParser()
p.add_argument('--candidate')
p.add_argument('--native-manifest')
p.add_argument('--id', required=True)
p.add_argument('--phase', choices=['emit', 'compare', 'validate', 'compare-native'], required=True)
p.add_argument('--result-phase', choices=['search', 'confirmation'], default='search')
a = p.parse_args()
assert re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,63}', a.id)
os.chdir(R)
b = reconcile(R, mounted=True)
w = load_workload(R / 'campaign/workload.json')
c = Compiler.load(C, C / 'compiler/revision.json')
assert c.commit == b['compiler']
assert list(w.case_ids) == b['case_ids'] and len(w.case_ids) == 5
if a.phase == 'compare-native':
    if a.candidate or not a.native_manifest or a.result_phase != 'search':
        raise ValueError('Native comparison needs its manifest and stays in search engineering evidence')
    import torch
    from campaign.hip_probe import own_path, verify_files
    from development_device import native_output_contract
    folder = own_path(R, 'campaign/engineering/' + a.id)
    manifest_path = own_path(R, a.native_manifest)
    if manifest_path != folder / 'manifest.json':
        raise ValueError('Native manifest must belong to this engineering probe ID')
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get('implementation_kind') != 'native_hip' or manifest.get('id') != a.id
            or manifest.get('compiler') != c.commit or manifest.get('task') != b['task']):
        raise ValueError('Native manifest differs from the original Task or Compiler')
    verify_files(R, manifest, json.loads((folder / 'build.json').read_text()))
    prepared = torch.load(R / 'campaign/development-inputs.pt', weights_only=False)
    observed = torch.load(folder / 'observed.pt', weights_only=False)
    for seen in observed:
        if (seen.get('implementation_kind') != 'native_hip'
                or seen.get('output_metadata') != native_output_contract(w, seen['case_id'])):
            raise ValueError('Native output metadata differs from the original Workload ABI')
    checks = compare_cases(w, prepared, observed)
    result = dict(id=a.id, implementation_kind='native_hip', compiler=c.commit,
                  numerically_passed=all(row['passed'] for row in checks), checks=checks,
                  primary_us=next(row['samples_us'] for row in observed if row['case_id'] == 'primary'),
                  scope='engineering numeric observations; precision and execution-source review pending')
    write_new(folder / 'comparison.json', result)
    print(json.dumps(result))
    raise SystemExit(0 if result['numerically_passed'] else 1)
if not a.candidate or a.native_manifest:
    raise ValueError('Cake evaluation needs its candidate and no native manifest')
folder = R / 'campaign/evaluations' / a.id
if Path(a.candidate).suffix not in ('.py', '.json'):
    raise ValueError('Candidate must be Cake Python or a complete Schedule JSON')
candidate_name = 'candidate' + Path(a.candidate).suffix

if a.phase in ('emit', 'validate'):
    source = (R / a.candidate).resolve()
    source.relative_to(R)
    if a.phase == 'emit':
        folder.mkdir(parents=True, exist_ok=False)
    else:
        assert source == (folder / candidate_name).resolve(), 'nominee must use its retained source snapshot'
    stage = 'source_read'
    retained = False
    try:
        raw = source.read_bytes()
        if a.phase == 'emit':
            stage = 'source_snapshot'
            write_bytes_new(folder / 'submitted.source', raw)
            retained = True
            write_new(folder / 'submission.json', dict(source=str(source.relative_to(R)),
                suffix=source.suffix, snapshot=str((folder / 'submitted.source').relative_to(R))))
        stage = 'decode'
        text = raw.decode('utf-8')
        stage = 'frontend' if source.suffix == '.py' else 'json'
        document = frontend.parse(text, filename=str(source)).document if source.suffix == '.py' else json.loads(text)
        if not isinstance(document, dict):
            raise CandidateRefusal('Schedule JSON root must be an object')
        stage = 'origin'
        from campaign.compiler_tools import stage_origin
        if a.phase == 'emit':
            action = stage_origin(R, source, document)
        else:
            prior = json.loads((folder / 'emission.json').read_text())
            action = prior.get('author_action_stage')
            if action is not None and stage_origin(R, action, document) is None:
                raise ValueError('Declared transform stage is outside the tool records')
        stage = 'assessment'
        assessment = c.assess(document)
        if a.phase == 'emit':
            stage = 'assessment_write'
            write_bytes_new(folder / candidate_name, raw)
            write_new(folder / 'assessment.json', dict(
                findings=[f.to_dict() for f in assessment.findings], accepted=assessment.accepted,
                lowering_eligible=assessment.lowering_eligible))
        if not assessment.lowering_eligible:
            if a.phase == 'validate':
                raise ValueError('retained nominee is no longer lowering-eligible')
            write_new(folder / 'outcome.json', dict(id=a.id, status='refused',
                phase=a.result_phase, scope='canonical Task construction/verifier refusal; no GPU'))
            print('refused; diagnostics retained')
            raise SystemExit(0)
        stage = 'target'
        if assessment.target != 'gfx938':
            raise CandidateRefusal('candidate changes the exact Target')
        stage = 'abi'
        schedule = assessment.typed_schedule
        abi = {x.name: (tuple(x.shape), x.dtype, x.mode) for x in w.tensor_abi(w.case_ids[0])}
        actual = {x.name: (x.shape, x.dtype.value, x.mode.value) for x in schedule.buffers if x.space.value == 'global'}
        if actual != abi:
            raise CandidateRefusal('candidate changes Task ABI')
        stage = 'lowering'
        lowering = c.lower(assessment)
        emission = dict(source=str((folder / 'source.py').relative_to(R)),
                        entry=lowering.route.entry_point, outputs=list(schedule.outputs), compiler=c.commit)
        if action is not None:
            emission['author_action_stage'] = action
        if a.phase == 'validate':
            outcome = json.loads((folder / 'outcome.json').read_text())
            assert outcome['id'] == a.id and outcome['compiler'] == c.commit
            assert outcome['status'] == 'accepted' and outcome['phase'] == 'search'
            assert outcome['candidate'] == str((folder / candidate_name).relative_to(R))
            assert outcome['source'] == emission['source']
            assert [x['case_id'] for x in outcome['checks']] == list(w.case_ids)
            assert all(x['passed'] is True for x in outcome['checks'])
            assert json.loads((folder / 'schedule.json').read_text()) == document
            assert json.loads((folder / 'emission.json').read_text()) == emission
            assert (folder / 'source.py').read_text() == lowering.source, 'retained emission differs from pinned Compiler'
            print('validated fixed-source five-case nominee; CPU only')
        else:
            stage = 'emission_write'
            write_bytes_new(folder / 'source.py', lowering.source.encode('utf-8'))
            write_new(folder / 'schedule.json', document)
            write_new(folder / 'emission.json', emission)
            print('generated own source; no GPU')
    except Exception as error:
        if a.phase == 'emit' and not (folder / 'outcome.json').exists():
            refused = (isinstance(error, CandidateRefusal)
                or stage == 'assessment' and isinstance(error, getattr(compiler_api, 'ScheduleEnvelopeError', ()))
                or stage == 'frontend' and isinstance(error, frontend.FrontendError)
                or stage == 'json' and isinstance(error, json.JSONDecodeError)
                or stage == 'decode' and isinstance(error, UnicodeDecodeError))
            diagnostic = dict(type=type(error).__name__, message=str(error))
            if isinstance(error, frontend.FrontendError):
                diagnostic.update(code=error.code, canonical_path=error.canonical_path,
                                  location=vars(error.location))
            write_new(folder / 'outcome.json', dict(id=a.id, phase=a.result_phase,
                status='refused' if refused else 'evaluation_error', compiler=c.commit,
                stage=stage, error=diagnostic, source=str(source.relative_to(R)),
                source_suffix=source.suffix, raw_source_complete=retained,
                submitted_source=str((folder / 'submitted.source').relative_to(R)) if retained else None,
                scope='canonical development input refusal; no GPU' if refused
                      else 'development evaluation error; ownership unknown; no GPU'))
            if refused:
                print('refused; submitted source and diagnostics retained')
                raise SystemExit(0)
        raise

else:
    import torch
    prepared = torch.load(R / 'campaign/development-inputs.pt', weights_only=False)
    observed = torch.load(folder / 'observed.pt', weights_only=False)
    checks = compare_cases(w, prepared, observed)
    result = dict(id=a.id, phase=a.result_phase,
                  status='accepted' if all(x['passed'] for x in checks) else 'numerically_failed',
                  compiler=c.commit, scope='registered Task development; diagnostic callable samples, no performance win',
                  checks=checks, primary_us=next(x['samples_us'] for x in observed if x['case_id'] == 'primary'),
                  candidate=str((folder / candidate_name).relative_to(R)), source=str((folder / 'source.py').relative_to(R)),
                  diagnosis_owner='inspect candidate/contract before promoting Compiler Finding')
    write_new(folder / 'outcome.json', result)
    print(json.dumps(result))
