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

    def test_captured_author_stdout_is_not_an_unreleased_admission(self):
        """#42, r4b fib_fused_add_rmsnorm_h4096: a probe's whole stdout redirected into a .json
        file stopped HCU 5 although both gateway terminals certified the release."""
        write(self.root / 'campaign/admission.json', admission())
        write(self.root / 'campaign/admission-terminal.json', terminal())
        captured = self.captured_output()
        for name in ('.local/profile/probe/timing.json', 'campaign/logs/probe-timing.json'):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(captured)
        self.gates(True)
        self.assertEqual(sorted(release_audit(self.root)['author_diagnostic_files']),
                         ['.local/profile/probe/timing.json', 'campaign/logs/probe-timing.json'])

    def test_text_that_could_be_or_hide_a_gateway_record_still_withholds_release(self):
        write(self.root / 'campaign/admission.json', admission())
        write(self.root / 'campaign/admission-terminal.json', terminal())
        for content in (
                'warning\n' + json.dumps(admission('unreleased')) + '\n{',      # names the schema
                'warning\n{"schema": "bw1100-bench.hcu-admission.v1", "job_',    # torn after a prefix
                '{"schema": "bw1100-bench.hc',                                   # torn gateway write
                '{"warning": 1}\n{',                                            # starts like a record
                'warning\n{"admission_receipts": ["campaign/x"]',                # hides a declaration
                '', ' \n\t'):                                                  # empty
            with self.subTest(content=content):
                self.path.write_text(content)
                self.gates(False)
        self.path.write_text('{}')
        (self.root / 'campaign/other-terminal.json').write_text('not json at all')
        self.gates(False)

    def captured_output(self, job='owned'):
        return ('Ignoring [] DCUs as they are invalid\n'
                '{"level":"warning","msg":"mount dir: /data3"}\n'
                '{"runtime_device_identity": "passed"}\n'
                '{\n "scope": "HIP-event bracketed GPU time"\n}\n'
                + json.dumps({'job_id': job, 'status': 'completed', 'exit_code': 0}) + '\n')

    def test_captured_output_counts_only_for_jobs_released_in_scope(self):
        """Review of #43: the exemption must bind every named job to its own released terminal."""
        write(self.root / 'campaign/admission.json', admission())
        write(self.root / 'campaign/admission-terminal.json', terminal())
        probe = self.root / '.local/profile/probe/timing.json'
        probe.parent.mkdir(parents=True, exist_ok=True)
        for job, expected in (('owned', True), ('other', False)):
            with self.subTest(job=job):
                probe.write_text(self.captured_output(job))
                self.gates(expected)
        probe.write_text(self.captured_output('owned') + self.captured_output('other'))
        self.gates(False)
        self.assertEqual(
            [i for i in release_audit(self.root)['issues'] if i['reason'] == 'author_output_names_unreleased_job'],
            [{'file': '.local/profile/probe/timing.json', 'reason': 'author_output_names_unreleased_job',
              'job_ids': ['other']}])
        probe.write_text(self.captured_output())
        write(self.root / 'campaign/admission-terminal.json', terminal(container_still_running=True))
        self.gates(False)

    def test_captured_output_with_its_receipt_outside_the_walk_is_refused(self):
        """The gateway accepts any --receipt path in the repo; .data is outside both walks."""
        write(self.root / '.data/r/admission.json', admission())
        probe = self.root / '.local/profile/probe/timing.json'
        probe.parent.mkdir(parents=True, exist_ok=True)
        probe.write_text(self.captured_output())
        self.gates(False)
        write(self.root / '.data/r/admission-terminal.json', terminal())
        self.gates(False)

    def test_undecodable_or_jobless_text_and_declared_receipts_are_refused_without_crashing(self):
        write(self.root / 'campaign/admission.json', admission())
        write(self.root / 'campaign/admission-terminal.json', terminal())
        self.path.write_bytes(b'warning \xff\xfe\n{"job_id": "owned"}\n')
        self.gates(False)
        self.path.write_text('plain warning with no job\n')
        self.gates(False)
        self.path.write_text('[1, 2\n' + self.captured_output())
        self.gates(False)
        self.path.write_text('{}\n' + json.dumps({'admission_receipts': ['campaign/declared.log']}))
        (self.root / 'campaign/declared.log').write_text(self.captured_output())
        self.gates(False)
        self.path.write_text(json.dumps({'admission_receipts': ['campaign/declared.json']}))
        (self.root / 'campaign/declared.json').write_text(self.captured_output())
        self.gates(False)

    def test_each_guard_alone_withholds_an_otherwise_released_capture(self):
        """Every case below names only the released job 'owned', so one guard alone refuses it."""
        from scripts.json_stream import author_diagnostic
        write(self.root / 'campaign/admission.json', admission())
        write(self.root / 'campaign/admission-terminal.json', terminal())
        released_line = json.dumps({'job_id': 'owned', 'status': 'completed', 'exit_code': 0})
        cases = {
            'names the schema': 'warning\n' + released_line + ' bw1100-bench.hcu-admission.v1\n{',
            'names admission_receipts': 'warning\n' + released_line + ' admission_receipts\n{',
            'names no job': 'warning only, no job record\n{',
        }
        for name, content in cases.items():
            with self.subTest(case=name):
                self.path.write_text(content)
                self.assertIsNone(author_diagnostic(self.path))
                self.gates(False)
        # A -terminal.json never becomes author output, even holding only a released job line.
        self.path.write_text('{}')
        (self.root / 'campaign/other-terminal.json').write_text('warning\n' + released_line + '\n')
        self.gates(False)
        (self.root / 'campaign/other-terminal.json').unlink()
        # A receipt reached only through admission_receipts is never author output either.
        self.path.write_text(json.dumps({'admission_receipts': ['campaign/declared.receipt']}))
        (self.root / 'campaign/declared.receipt').write_text('warning\n' + released_line + '\n')
        self.assertFalse(release_audit(self.root)['release_records_clean'])
        self.assertFalse(released(self.root))

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
                write(root / 'campaign/evaluations/final-confirmation/outcome.json', {'status': 'accepted', 'phase': 'confirmation'})
                return 0
            previous = Path.cwd()
            try:
                with patch.object(launch, 'reconcile', return_value=binding), \
                     patch.object(launch, 'catalog', return_value={}), \
                     patch.object(launch, 'author_entry', return_value=(Path('/fixture/hmz'), {})), \
                     patch.object(launch, 'check_exec_line'), \
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
