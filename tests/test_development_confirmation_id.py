"""The owner confirmation id is reserved, and a confirmation must be one (C500 r1b review)."""
import json
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ReservedConfirmationId(unittest.TestCase):
    def test_an_author_search_named_final_confirmation_is_refused_before_any_work(self):
        temp = Path(tempfile.mkdtemp()); self.addCleanup(shutil.rmtree, temp)
        (temp / 'campaign').mkdir()
        for name in ('development_evaluate_owner.py', 'development_stop.py'):
            shutil.copyfile(ROOT / 'development' / name, temp / 'campaign' / name)
        (temp / 'campaign/development-binding.json').write_text(json.dumps(dict(
            image='img', hcu=5, batch_stop_file=str(temp / 'STOP.json'))))
        (temp / 'campaign/deadline.json').write_text(json.dumps(dict(search_stop_at_epoch=10**12, stop_at_epoch=10**12)))
        calls = []
        old = (subprocess.call, sys.argv[:], sys.path[:], Path.cwd())
        subprocess.call = lambda argv, **kwargs: calls.append(argv) or 0
        sys.argv = ['owner', '--candidate', 'c.py', '--id', 'final-confirmation']
        sys.path.insert(0, str(temp / 'campaign'))
        try:
            with self.assertRaises(SystemExit) as stop:
                runpy.run_path(str(temp / 'campaign/development_evaluate_owner.py'), run_name='__main__')
        finally:
            subprocess.call, sys.argv, sys.path = old[0], old[1], old[2]
            import os; os.chdir(old[3])
            sys.modules.pop('development_stop', None)
        self.assertIn('owner confirmation id', str(stop.exception.code))
        self.assertEqual(calls, [])

    def test_a_confirmation_folder_the_owner_did_not_create_is_never_read_as_one(self):
        source = (ROOT / 'development/development_launch.py').read_text()
        block = source[source.index("folder = root / 'campaign/evaluations/final-confirmation'"):]
        block = block[:block.index('except Exception as error:')]
        self.assertLess(block.index('if folder.exists() or folder.is_symlink():'), block.index('subprocess.call(command'))
        self.assertIn("confirmation.get('phase') != 'confirmation'", block)

    def test_a_mismatched_confirmation_terminal_is_no_qualified_result(self):
        sys.path.insert(0, str(ROOT / 'development')); self.addCleanup(sys.path.remove, str(ROOT / 'development'))
        import development_launch
        temp = Path(tempfile.mkdtemp()); self.addCleanup(shutil.rmtree, temp)
        (temp / 'campaign').mkdir()
        binding = dict(task='t', compiler='c', gateway={})
        old = development_launch.released
        development_launch.released = lambda root: True
        try:
            development_launch.write_terminal(temp, binding, 0, {'id': 'x'},
                {'status': 'confirmation_identity_mismatch'}, None, None)
        finally:
            development_launch.released = old
        self.assertEqual(json.loads((temp / 'campaign/DONE.json').read_text())['status'], 'no_qualified_result')


if __name__ == '__main__':
    unittest.main()
