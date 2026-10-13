"""Workspace entrypoint for the source-pinned Open-Cake author tool projection."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from campaign._compiler_tools import (  # noqa: E402,F401
    ACTIONS, CATALOG, action_directory, catalog, freeze, history,
    inspect_parent, main, parent_source, stage_origin, transform, verify,
)


def author_context(root):
    from campaign._compiler_tools import author_context as compiler_context
    from campaign.hip_probe import author_context as native_context
    return compiler_context(root) + native_context(root)


if __name__ == '__main__':
    main(ROOT)
