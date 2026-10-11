"""Complete diagnostic streams must not hide missing or partial device releases."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from development.development_control import released
from scripts.bench_queue import release_audit
from scripts.json_stream import read_json_stream


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def admission(job='owned'):
    return {'schema': 'bw1100-bench.hcu-admission.v1', 'job_id': job}


def terminal(job='owned', **changes):
    return {**admission(job), 'completed_at': 1, 'exit_code': 0,
            'container_still_running': False, 'after_vram': '0%', 'after_kfd_visible': False,
            **changes}


def diagnostic_stream():
    return '\n'.join([json.dumps({'warning': str(index)}) for index in range(6)] + ['[{"resource": 1}]'])


class StreamReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        (self.root / 'campaign').mkdir()
        self.path = self.root / 'campaign/resources.json'

    def tearDown(self):
        self.temp.cleanup()

    def gates(self, expected):
        self.assertEqual(released(self.root), expected)
        self.assertEqual(release_audit(self.root)['release_records_clean'], expected)

    def test_complete_values_preserve_single_json_and_diagnostic_streams(self):
        write(self.root / 'campaign/admission.json', admission())
        write(self.root / 'campaign/admission-terminal.json', terminal())
        for content in ('{}', '[]', diagnostic_stream(), ' {}\n[1,2]\r\nnull\t'):
            with self.subTest(content=content):
                self.path.write_text(content)
                self.assertTrue(read_json_stream(self.path))
                self.gates(True)

    def test_empty_partial_garbage_unreadable_and_symlink_stay_unclean(self):
        for suffix in ('', ' \n\t', '{}\n{', '{}\n[1,', '{}\nnot-json'):
            with self.subTest(suffix=suffix):
                self.path.write_text(suffix)
                with self.assertRaises(ValueError): read_json_stream(self.path)
                self.gates(False)
        self.path.write_text('{}')
        original = Path.read_text
        def unreadable(path, *args, **kwargs):
            if path == self.path: raise OSError('fixture read failure')
            return original(path, *args, **kwargs)
        with patch.object(Path, 'read_text', unreadable): self.gates(False)
        self.path.unlink()
        target = self.root / 'other.txt'
        target.write_text('{}')
        self.path.symlink_to(target)
        self.gates(False)

    def test_later_admission_and_declared_receipt_cannot_be_skipped(self):
        self.path.write_text('{"warning":"prefix"}\n' + json.dumps(admission()))
        self.gates(False)
        write(self.root / 'campaign/resources-terminal.json', terminal())
        self.gates(True)
        self.path.write_text(self.path.read_text() + '\n' + json.dumps(admission('unreleased')))
        self.gates(False)
        (self.root / 'campaign/resources-terminal.json').unlink()
        self.path.write_text('{}\n' + json.dumps({'admission_receipts': ['campaign/declared.receipt']}))
        self.gates(False)
        write(self.root / 'campaign/declared.receipt', admission())
        self.gates(False)
        write(self.root / 'campaign/declared-terminal.json', terminal())
        self.gates(True)
        self.path.write_text('{}\n' + json.dumps({'admission_receipts': ['../outside.json']}))
        self.gates(False)

    def test_terminals_remain_strict_matching_and_released(self):
        write(self.root / 'campaign/admission.json', admission())
        after = self.root / 'campaign/admission-terminal.json'
        for value in (terminal('other'), terminal(container_still_running=True),
                      terminal(after_vram='2%'), terminal(after_kfd_visible=True), {}):
            with self.subTest(value=value):
                write(after, value)
                self.gates(False)
        after.write_text(json.dumps(terminal()) + '\n{}')
        self.gates(False)
        write(after, terminal())
        self.gates(True)
        self.path.write_text('{}\n' + json.dumps(terminal(after_vram='2%')))
        self.gates(False)

    def test_terminal_preserves_every_initial_identity_and_request_value(self):
        initial = dict(admission(), selected_hcu=2, runtime_device_binding={'pci_bus_id': 'fixture-pci'},
                       image_id='fixture-image', started_at=123, command=['python', 'candidate.py'], timeout_s=600)
        write(self.root / 'campaign/admission.json', initial)
        original = dict(initial, **{key: value for key, value in terminal().items() if key not in initial})
        after = self.root / 'campaign/admission-terminal.json'
        write(after, original)
        self.gates(True)
        for key, value in (('selected_hcu', 3), ('runtime_device_binding', {'pci_bus_id': 'other-pci'}),
                           ('image_id', 'other-image'), ('started_at', 124),
                           ('command', ['different.py']), ('timeout_s', 900)):
            with self.subTest(key=key):
                write(after, dict(original, **{key: value}))
                self.gates(False)

    def test_partial_canonical_terminal_cannot_hide_beside_complete_terminal(self):
        write(self.root / 'campaign/admission.json', admission())
        write(self.root / 'campaign/admission-terminal.json', terminal())
        extra = self.root / 'campaign/other-terminal.json'
        for value in ({key: value for key, value in terminal().items() if key != 'exit_code'},
                      dict(admission(), status='completed'), admission()):
            with self.subTest(value=value):
                write(extra, value)
                self.gates(False)
                self.assertIn('partial_terminal', {issue['reason'] for issue in release_audit(self.root)['issues']})

    def test_newly_declared_extensionless_references_are_exhausted(self):
        self.path.write_text('{}\n' + json.dumps({'admission_receipts': ['campaign/first']}))
        write(self.root / 'campaign/first', {'admission_receipts': ['campaign/second']})
        self.gates(False)
        write(self.root / 'campaign/second', admission())
        self.gates(False)
        write(self.root / 'campaign/second-terminal.json', terminal())
        self.gates(True)

    def test_owner_closure_confirms_complete_stream_and_stops_on_partial(self):
        sys.path.insert(0, str(ROOT / 'development'))
        spec = importlib.util.spec_from_file_location('stream_owner_fixture', ROOT / 'development/development_launch.py')
        launch = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launch)
        for partial in (False, True):
            root = self.root / ('partial' if partial else 'complete')
            (root / 'campaign/logs').mkdir(parents=True)
            binding = dict(task='fixture', compiler='compiler', gateway='gateway', image='image',
                           batch_stop_file=str(root / 'STOP.json'))
            (root / 'campaign/resources.json').write_text(diagnostic_stream() + ('\n{' if partial else ''))
            write(root / 'campaign/admission.json', admission())
            write(root / 'campaign/admission-terminal.json', terminal())
            write(root / 'campaign/evaluations/nominee/outcome.json', dict(id='nominee', status='accepted',
                  phase='search', compiler='compiler', primary_us=[1.0], candidate='campaign/candidates/a.py'))
            def confirm(*args, **kwargs):
                write(root / 'campaign/evaluations/final-confirmation/outcome.json', {'status': 'accepted'})
                return 0
            previous = Path.cwd()
            try:
                with patch.object(launch, 'reconcile', return_value=binding), \
                     patch.object(launch, 'catalog', return_value={}), \
                     patch.object(launch, 'author_entry', return_value=(Path('/fixture/hmz'), {})), \
                     patch.object(launch, 'git', return_value='prepared'), \
                     patch.object(launch.subprocess, 'check_output', return_value='profile'), \
                     patch.object(launch.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0)), \
                     patch.object(launch, 'run_author', return_value=(0, None)), \
                     patch.object(launch, 'validate_nominee'), \
                     patch.object(launch.subprocess, 'call', side_effect=confirm) as confirmation:
                    self.assertEqual(launch.main(root), int(partial))
                    self.assertEqual(confirmation.call_count, 0 if partial else 1)
            finally:
                import os
                os.chdir(previous)
            self.assertEqual((root / 'STOP.json').exists(), partial)
            done = json.loads((root / 'campaign/DONE.json').read_text())
            self.assertEqual(done['device_release_verified'], not partial)
            self.assertEqual(done['status'], 'failed' if partial else 'completed')


if __name__ == '__main__':
    unittest.main()
