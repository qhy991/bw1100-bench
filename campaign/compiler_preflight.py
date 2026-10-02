"""No-GPU public-API admission on the exact frozen Compiler."""
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from campaign.cake_bridge import CAKE, PIN, compiler

engine = compiler()
results = []
for example in ('gfx938-rmsnorm-b8-smoke', 'gfx938-swiglu-b8-smoke', 'gfx938-gemm-bias-b1-smoke'):
    document = json.loads((CAKE / 'corpus/schedules' / (example + '.json')).read_text())
    assessment = engine.assess(document)
    if not assessment.lowering_eligible:
        raise ValueError(example + ' cannot lower')
    lowered = engine.lower(assessment)
    if example == 'gfx938-rmsnorm-b8-smoke':
        destination = ROOT / 'campaign/results/compiler-preflight-rmsnorm.py'
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('x') as stream:
            stream.write(lowered.source)
    results.append({'schedule': example, 'generated': lowered.generated, 'target': lowered.target,
                    'entry_point': lowered.route.entry_point})
bad = copy.deepcopy(document)
bad['target'] = 'gfx938-undeclared'
refused = engine.assess(bad)
if refused.lowering_eligible:
    raise ValueError('exact-target refusal did not hold')
out = ROOT / 'campaign/results/compiler-preflight.json'
out.parent.mkdir(parents=True, exist_ok=True)
with out.open('x') as stream:
    json.dump({'compiler_commit': PIN, 'accepted_examples': results,
               'rejected_target_findings': [f.to_dict() for f in refused.findings],
               'device_qualification': False}, stream, indent=2)
print('Standalone Compiler public API passed; device qualification remains pending')
