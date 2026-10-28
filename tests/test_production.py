import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from production import catalog
from production.runner import candidate_module, eager_calls

try:
    import torch
except ImportError:
    torch = None


class CatalogTest(unittest.TestCase):
    def cli(self, *args):
        return subprocess.run([sys.executable, '-S', str(ROOT / 'bwbench.py'), *args],
                              text=True, capture_output=True)

    def manifests(self, root):
        for rank, requests in {'rank0': (2, 4, 8), 'rank5': (1, 2, 4)}.items():
            folder = root / rank
            folder.mkdir()
            entries = {}
            for layer in (4, 24):
                for request in requests:
                    for call in range(4):
                        name = f'language_model_model_layers_{layer}_self_attn_-verify-r{request}-{call:03d}.pt'
                        (folder / name).touch()
                        entries[name] = {'status': 'real', 'rule': 'bitwise x2 (rule A)',
                                         'sha256': 'a' * 64, 'site': f'verify-r{request}',
                                         'layer_prefix': f'language_model.model.layers.{layer}.self_attn.',
                                         'call_index': call}
            (folder / 'manifest.json').write_text(json.dumps({'schema': 'glm53.capture.manifest.v1',
                                                             'records': entries}))

    def test_sol_selection_and_help_remain_discoverable(self):
        result = self.cli('list')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(result.stdout.splitlines()), 12)
        self.assertIn('production', self.cli('--help').stdout)
        new = self.cli('production', 'list')
        self.assertEqual(new.returncode, 0, new.stderr)
        self.assertEqual({task['id'] for task in json.loads(new.stdout)},
                         {'glm53-kda-verify', 'glm53-dsa-decode', 'glm53-lmhead'})

    def test_audit_does_not_need_torch_and_requires_guard_shard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.manifests(root)
            result = self.cli('production', 'audit', '--task', 'glm53-kda-verify', '--inputs', str(root))
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(json.loads(result.stdout)['audit']['records'], 48)
            (root / 'rank0' / 'manifest.json').unlink()
            with self.assertRaisesRegex(catalog.PrerequisiteError, 'guard/capture shard'):
                catalog.captured_records(root)

    def test_blocked_or_incomplete_identity_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.manifests(root)
            path = root / 'rank5' / 'manifest.json'
            doc = json.loads(path.read_text())
            first = next(iter(doc['records'].values()))
            first['status'] = 'blocked'
            path.write_text(json.dumps(doc))
            with self.assertRaisesRegex(catalog.PrerequisiteError, 'admission'):
                catalog.captured_records(root)
            first['status'] = 'real'
            first['site'] = 'decode-r1'
            path.write_text(json.dumps(doc))
            with self.assertRaisesRegex(catalog.PrerequisiteError, 'identity'):
                catalog.captured_records(root)

    def test_dsa_refuses_before_candidate_import_and_preserves_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker, candidate, output = (root / name for name in ('imported', 'candidate.py', 'result.json'))
            candidate.write_text(f'from pathlib import Path\nPath({str(marker)!r}).touch()\n')
            args = ('production', 'check', '--task', 'glm53-dsa-decode', '--candidate', str(candidate),
                    '--inputs', str(root), '--output', str(output))
            result = self.cli(*args)
            self.assertEqual(result.returncode, 2)
            self.assertFalse(marker.exists())
            self.assertIn('DSA requires', json.loads(output.read_text())['reason'])
            saved = output.read_bytes()
            self.assertNotEqual(self.cli(*args).returncode, 0)
            self.assertEqual(output.read_bytes(), saved)

    def test_candidate_cannot_import_native(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'candidate.py'
            path.write_text('from production.kda import native_call\n')
            with self.assertRaisesRegex(ValueError, 'native/reference'):
                candidate_module(path)

    def test_missing_torch_is_a_blocked_prerequisite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'weight.safetensors').touch()
            (root / 'model.safetensors.index.json').write_text(json.dumps(
                {'weight_map': {'lm_head.weight': 'weight.safetensors'}}))
            output = root / 'result.json'
            result = self.cli('production', 'check', '--task', 'glm53-lmhead', '--model', str(root),
                              '--candidate', str(root / 'not-imported.py'), '--output', str(output))
            self.assertEqual(result.returncode, 2, result.stderr)
            report = json.loads(output.read_text())
            self.assertEqual(report['status'], 'blocked')
            self.assertEqual(report['stage'], 'runtime_admission')
            self.assertFalse(report['device_check_started'])

    def test_failed_checks_are_distinct_from_blocked_prerequisites(self):
        from production import runner, lmhead
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, stage, error, expected, started in (
                ('candidate', 'candidate_load', ValueError('invalid candidate'), 'failed', False),
                ('native', 'native_replay', catalog.PrerequisiteError('missing patch'), 'blocked', True),
                ('output', 'candidate_checks', ValueError('output differs'), 'failed', True),
            ):
                output = root / (name + '.json')
                args = ['check', '--task', 'glm53-lmhead', '--candidate', str(root / 'candidate.py'),
                        '--model', str(root), '--output', str(output)]
                with self.subTest(name=name), contextlib.redirect_stdout(io.StringIO()), \
                        patch.object(catalog, 'audit', return_value={}), \
                        patch.object(runner, 'runtime', return_value={}), \
                        patch.object(runner, 'candidate_module', side_effect=error if name == 'candidate' else None,
                                     return_value=SimpleNamespace()), \
                        patch.object(lmhead, 'cases', side_effect=error if name == 'native' else None,
                                     return_value=[]), \
                        patch.object(runner, 'verify_calls', side_effect=error if name == 'output' else None):
                    self.assertEqual(runner.main(args), 2)
                report = json.loads(output.read_text())
                self.assertEqual(report['status'], expected)
                self.assertEqual(report['stage'], stage)
                self.assertEqual(report['device_check_started'], started)


@unittest.skipIf(torch is None, 'Torch CPU dependency is unavailable')
class TensorContractTest(unittest.TestCase):
    def case(self, name='case', value=1):
        from production.case import Case
        x = torch.tensor([float(value)])
        output = torch.zeros(1)
        state = torch.full((3, 2), 7.0)
        public = {'inputs': {'x': x}, 'weights': {}, 'metadata': {'M': 1},
                  'state': {'ssm': state}, 'output': output}
        return Case(name, public, {}, {'ssm': state, 'output': output},
                    {'ssm': torch.tensor([False, True, False]), 'output': None}, ('test', 1))

    def test_evaluator_checks_state_and_immutable_storage(self):
        case = self.case()
        case.public['state']['ssm'][0, 0] = 0
        with self.assertRaisesRegex(ValueError, 'outside admitted'):
            case.integrity()
        case.restore()
        case.public['inputs']['x'].fill_(2)
        with self.assertRaisesRegex(ValueError, 'input/weight storage'):
            case.integrity()

    def test_shape_scalar_or_output_replacement_is_rejected(self):
        case = self.case()
        case.public['metadata']['M'] = 2
        with self.assertRaisesRegex(ValueError, 'interface'):
            case.integrity()
        case = self.case()
        case.public['output'] = torch.zeros(1)
        with self.assertRaisesRegex(ValueError, 'replaced'):
            case.integrity()

    def test_holes_in_strided_input_storage_are_protected(self):
        from production.case import Case
        backing = torch.arange(6.0)
        view = backing[::2]
        out = torch.zeros(1)
        case = Case('strided', {'inputs': {'x': view}, 'output': out}, {},
                    {'output': out}, {'output': None}, ('strided',))
        backing[1] = -1
        with self.assertRaisesRegex(ValueError, 'storage'):
            case.integrity()

    def test_cross_call_rejects_overwriting_prior_output(self):
        first, second = self.case('first'), self.case('second', 2)
        old = []
        def prepare(public):
            def call():
                for output in old:
                    output.fill_(-1)
                public['output'].copy_(public['inputs']['x'])
                old.append(public['output'])
            return call
        def validate(case):
            self.assertTrue(torch.equal(case.public['output'], case.public['inputs']['x']))
        with self.assertRaisesRegex(ValueError, 'retained earlier output'):
            eager_calls([first, second], SimpleNamespace(prepare=prepare), validate, lambda: None)

    def test_later_call_cannot_corrupt_a_retained_input(self):
        first, second = self.case('first'), self.case('second', 2)
        def prepare(public):
            def call():
                public['output'].copy_(public['inputs']['x'])
                if public is second.public:
                    first.public['inputs']['x'].fill_(999)
            return call
        def validate(case):
            self.assertTrue(torch.equal(case.public['output'], case.public['inputs']['x']))
        with self.assertRaisesRegex(ValueError, 'input/weight storage'):
            eager_calls([first, second], SimpleNamespace(prepare=prepare), validate, lambda: None)

    def test_later_call_cannot_corrupt_a_retained_written_state(self):
        first, second = self.case('first'), self.case('second', 2)
        def prepare(public):
            def call():
                public['output'].copy_(public['inputs']['x'])
                if public is second.public:
                    first.public['state']['ssm'][1].fill_(999)
            return call
        def validate(case):
            self.assertTrue(bool((case.public['state']['ssm'][1] == 7).all()))
        with self.assertRaises(AssertionError):
            eager_calls([first, second], SimpleNamespace(prepare=prepare), validate, lambda: None)

    def test_retained_output_requires_byte_equality(self):
        first, second = self.case('first', 0), self.case('second', 0)
        def prepare(public):
            def call():
                public['output'].zero_()
                if public is second.public:
                    first.public['output'].neg_()
            return call
        with self.assertRaisesRegex(ValueError, 'retained earlier output'):
            eager_calls([first, second], SimpleNamespace(prepare=prepare), lambda case: None, lambda: None)

    def test_captured_aliases_keep_absolute_offsets_and_storage_extent(self):
        from production.bundle import decode
        raw = torch.arange(8, dtype=torch.float32)
        record = {'groups': {'0': {'base_byte': 8, 'bytes': raw.view(torch.uint8)[8:24].clone()}},
                  'tensors': {name: {'dtype': 'float32', 'shape': [count], 'stride': [1],
                                     'group': '0', 'byte_offset_in_group': relative,
                                     'storage_offset': offset, 'storage_nbytes': 32}
                              for name, count, relative, offset in (('x', 4, 0, 2), ('y', 2, 4, 3))}}
        tensors = decode(record, 'cpu')
        self.assertEqual(tensors['x'].storage_offset(), 2)
        self.assertEqual(tensors['y'].storage_offset(), 3)
        self.assertEqual(tensors['x'].untyped_storage().nbytes(), 32)
        self.assertTrue(torch.equal(tensors['x'], raw[2:6]))
        tensors['x'][1] = -1
        self.assertEqual(tensors['y'][0], -1)

    def test_missing_output_write_fails_numerical_gate(self):
        from production.case import assert_tensor
        with self.assertRaisesRegex(ValueError, 'output differs'):
            assert_tensor(torch.zeros(2), torch.ones(2), 'output')

    def test_lmhead_rejects_nonfinite_wrong_shape_and_argmax(self):
        from production.lmhead import numerical_verdict
        native = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)
        oracle = native.double()
        self.assertEqual(numerical_verdict(native, native, oracle)['relative_error'], 0)
        for actual in (torch.tensor([[float('nan'), 2.0]], dtype=torch.bfloat16),
                       torch.zeros(3, dtype=torch.bfloat16), native.flip(-1)):
            with self.assertRaises(ValueError):
                numerical_verdict(actual, native, oracle)

    def test_kda_recurrence_state_orientation_against_hand_result(self):
        from production.kda_reference import recurrence
        q = torch.tensor([[[1., 0.]]], dtype=torch.float64)
        k = torch.tensor([[[0., 1.]]], dtype=torch.float64)
        v = torch.tensor([[[5., 6.]]], dtype=torch.float64)
        state = torch.tensor([[[1., 2.], [3., 4.]]], dtype=torch.float64)
        output, states = recurrence(q, k, v, torch.zeros_like(q), torch.ones(1, 1), state)
        self.assertTrue(torch.equal(states[0], torch.tensor([[[1., 5.], [3., 6.]]], dtype=torch.float64)))
        self.assertTrue(torch.equal(output, torch.tensor([[[1., 3.]]], dtype=torch.float64)))


if __name__ == '__main__':
    unittest.main()
