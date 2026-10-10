"""Workspace entrypoint for the source-pinned Open-Cake author tool projection."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from campaign._compiler_tools import (  # noqa: E402,F401
    ACTIONS, CATALOG, action_directory, author_context, catalog, freeze, history,
    inspect_parent, main, parent_source, stage_origin, transform, verify,
)


if __name__ == '__main__':
    main(ROOT)
