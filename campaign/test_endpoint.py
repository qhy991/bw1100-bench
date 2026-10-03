"""The budget selector chooses the best valid receipt, not the first success."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest


class EndpointSelection(unittest.TestCase):
    def test_continuation_uses_parent_gain_instead_of_moving_community_score(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            campaign = root / 'campaign'
            campaign.mkdir()
            shutil.copy(Path(__file__).with_name('finish.py'), campaign / 'finish.py')
            stop = time.time() - 1
            start = stop - 10800
            documents = {
                'intake.json': {'round': 2, 'parent': {'root': 'parent'},
                               'plan': 'campaign/plan.json', 'protocol': 'campaign/protocol.json'},
                'plan.json': {'arm': 'direct_triton', 'tasks': [{'id': 'example'}]},
                'protocol.json': {'budget_hours': 3, 'checkpoint_minutes': [30, 180]},
                'deadline.json': {'started_at_epoch': start, 'stop_at_epoch': stop},
            }
            for name, document in documents.items():
                (campaign / name).write_text(json.dumps(document))
            for name, community, parent in [('seed00', 2.0, 1.0),
                                             ('noisy_baseline', 3.0, 1.05),
                                             ('true_gain', 2.8, 1.20)]:
                path = campaign / (name + '.py')
                path.write_text('def run(): pass\n')
                outcome = {'id': name, 'status': 'accepted', 'conservative_geomean': community,
                           'incremental_conservative_geomean': parent,
                           'candidate_source': 'campaign/' + name + '.py',
                           'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                           'completed_at_epoch': start + 1200,
                           'handoff_row': {'task': 'example', 'reason': name}}
                target = campaign / 'evaluations' / name
                target.mkdir(parents=True)
                (target / 'outcome.json').write_text(json.dumps(outcome))
            subprocess.run([sys.executable, str(campaign / 'finish.py')], check=True,
                           stdout=subprocess.PIPE, universal_newlines=True)
            endpoint = json.loads((campaign / 'ENDPOINT.json').read_text())
            self.assertEqual(endpoint['best_id'], 'true_gain')
            self.assertEqual(endpoint['best_parent_relative_geomean'], 1.2)

    def test_best_within_budget_excludes_late_and_changed_candidates(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            campaign = root / 'campaign'
            campaign.mkdir()
            shutil.copy(Path(__file__).with_name('finish.py'), campaign / 'finish.py')
            stop = time.time() - 1
            start = stop - 10800
            documents = {
                'intake.json': {'plan': 'campaign/plan.json', 'protocol': 'campaign/protocol.json'},
                'plan.json': {'arm': 'direct_triton', 'tasks': [{'id': 'example'}]},
                'protocol.json': {'budget_hours': 3, 'checkpoint_minutes': [30, 60, 120, 180]},
                'deadline.json': {'started_at_epoch': start, 'stop_at_epoch': stop},
            }
            for name, document in documents.items():
                (campaign / name).write_text(json.dumps(document))
            for name, score, minute in [('first', 1.2, 20), ('best', 1.8, 80),
                                        ('late', 3.0, 181), ('changed', 4.0, 90)]:
                path = campaign / (name + '.py')
                path.write_text('def run(): pass\n')
                outcome = {'id': name, 'status': 'accepted', 'conservative_geomean': score,
                           'candidate_source': 'campaign/' + name + '.py',
                           'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                           'completed_at_epoch': start + minute*60,
                           'handoff_row': {'task': 'example', 'reason': name}}
                target = campaign / 'evaluations' / name
                target.mkdir(parents=True)
                (target / 'outcome.json').write_text(json.dumps(outcome))
                if name == 'changed':
                    path.write_text('modified\n')
            subprocess.run([sys.executable, str(campaign / 'finish.py')], check=True,
                           stdout=subprocess.PIPE, universal_newlines=True)
            endpoint = json.loads((campaign / 'ENDPOINT.json').read_text())
            self.assertEqual(endpoint['best_id'], 'best')
            self.assertEqual(endpoint['trajectory'][0]['best_qualified_id'], 'first')
            self.assertEqual(endpoint['trajectory'][-1]['best_qualified_id'], 'best')
            self.assertIn('changed', endpoint['stale_artifacts_excluded'])


if __name__ == '__main__':
    unittest.main()
