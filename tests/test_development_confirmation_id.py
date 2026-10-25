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

    def test_the_launcher_checks_that_a_confirmation_has_the_confirmation_phase(self):
        source = (ROOT / 'development/development_launch.py').read_text()
        self.assertIn("confirmation.get('phase') not in (None, 'confirmation')", source)


if __name__ == '__main__':
    unittest.main()
