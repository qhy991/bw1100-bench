import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.ralph_profile_intake import create_intake


class RalphProfileIntakeTests(unittest.TestCase):
    def test_records_exact_skill_and_requires_explicit_read(self):
        with TemporaryDirectory() as temp:
            base = Path(temp)
            root = base / 'bench'
            home = base / 'home'
            skill = home / '.agents/skills/dcu-rocprof-report-skill/SKILL.md'
            guide = root / 'docs/RALPH-PROFILING.md'
            skill.parent.mkdir(parents=True)
            guide.parent.mkdir(parents=True)
            skill.write_text('profile through the owner\n')
            guide.write_text('collect actual kernel rows\n')
            prompt = create_intake(root, home, Path('campaign/profile-intake.json'))
            record = json.loads((root / 'campaign/profile-intake.json').read_text())
            self.assertEqual(record['agent_read_status'], 'required_unverified')
            self.assertEqual(record['skill_path'], str(skill.resolve()))
            self.assertIn(str(skill.resolve()), prompt)
            self.assertIn('use the Read tool', prompt)
            with self.assertRaises(FileExistsError):
                create_intake(root, home, Path('campaign/profile-intake.json'))
            with self.assertRaises(ValueError):
                create_intake(root, home, Path('../escape.json'))

    def test_missing_skill_fails_before_creating_receipt(self):
        with TemporaryDirectory() as temp:
            base = Path(temp)
            root = base / 'bench'
            guide = root / 'docs/RALPH-PROFILING.md'
            guide.parent.mkdir(parents=True)
            guide.write_text('guide\n')
            with self.assertRaises(FileNotFoundError):
                create_intake(root, base / 'home', Path('campaign/profile-intake.json'))
            self.assertFalse((root / 'campaign').exists())
