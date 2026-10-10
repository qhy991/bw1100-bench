"""Update or check the exact shared Compiler tool projection from a local Git source."""
import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'campaign/compiler-tools-source.json'
DESTINATION = ROOT / 'campaign/_compiler_tools.py'
SOURCE_PATH = 'tools/hmz_compiler_tools.py'
REPOSITORY = 'https://github.com/qhy991/open-cake-ir'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.check:
        source = json.loads(MANIFEST.read_text())
        if source['repository'] != REPOSITORY or source['path'] != SOURCE_PATH:
            raise ValueError('Unexpected shared Compiler tool owner')
        commit = source['commit']
    else:
        commit = subprocess.check_output(['git', '-C', str(args.source), 'rev-parse', 'HEAD'], text=True).strip()
    content = subprocess.check_output(['git', '-C', str(args.source), 'show', commit + ':' + SOURCE_PATH])
    if args.check:
        if DESTINATION.read_bytes() != content:
            raise ValueError('Compiler tool projection differs from its pinned source; rerun the updater')
    else:
        DESTINATION.write_bytes(content)
        MANIFEST.write_text(json.dumps(dict(repository=REPOSITORY, commit=commit, path=SOURCE_PATH), indent=2) + '\n')
    print(('Checked' if args.check else 'Updated') + ' Compiler tools from ' + commit)


if __name__ == '__main__':
    main()
