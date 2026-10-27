"""Read every complete top-level JSON value; never accept a partial prefix."""
import ast
import json
import os
from pathlib import Path
import re


def read_json_stream(path):
    text = path.read_text()
    decoder = json.JSONDecoder()
    values = []
    offset = 0
    while offset < len(text):
        while offset < len(text) and text[offset] in ' \t\r\n':
            offset += 1
        if offset == len(text):
            break
        value, offset = decoder.raw_decode(text, offset)
        values.append(value)
    if not values:
        raise ValueError('Empty JSON stream')
    return values


GATEWAY_SCHEMA = 'bw1100-bench.hcu-admission.v1'
# The release checks' copy of where the gateway admits receipts (scripts/hcu_run.py
# RECEIPT_ROOTS / RECEIPT_UNWALKED_DIRS, #44). Both checks read every .json file under
# RELEASE_ROOTS, including inside RELEASE_UNWALKED_DIRS there: a gateway pinned before #44
# could write a receipt into campaign/__pycache__ (#45 review). They prune
# RELEASE_UNWALKED_DIRS only outside the scope, where no gateway since #44 writes.
RELEASE_ROOTS = ('campaign', '.local')
RELEASE_UNWALKED_DIRS = frozenset(('.git', '.deps', '.data', 'triton-cache', 'aiter-jit-cache', '__pycache__'))


def release_json_files(root, issue, whole_root=False):
    """Yield every .json file a release check reads, reporting what it could not walk.

    Inside RELEASE_ROOTS nothing is skipped: a symlinked scope root or directory and an
    unreadable directory are passed to issue(relative_path, reason), and the caller must
    withhold the release. With whole_root the rest of the tree is walked as before, pruning
    RELEASE_UNWALKED_DIRS and reporting unreadable directories.
    """
    def relative(path):
        return os.path.relpath(str(path), str(root))

    def unreadable(error):
        issue(relative(error.filename), 'unreadable_directory')

    tops = [root]
    if not whole_root:
        tops = []
        for name in RELEASE_ROOTS:
            if (root / name).is_symlink():
                issue(name, 'symlink_directory_not_followed')
            elif os.path.lexists(str(root / name)):
                tops.append(root / name)
    for top in tops:
        for directory, dirs, files in os.walk(str(top), onerror=unreadable):
            here = Path(directory)
            inside = here.relative_to(root).parts[:1]
            kept = []
            for name in dirs:
                if (inside or (name,))[0] not in RELEASE_ROOTS:
                    if name not in RELEASE_UNWALKED_DIRS:
                        kept.append(name)
                elif (here / name).is_symlink():
                    issue(relative(here / name), 'symlink_directory_not_followed')
                else:
                    kept.append(name)
            dirs[:] = kept
            for name in files:
                if name.endswith('.json'):
                    yield here / name


def declared_receipt_scope(source):
    """(RECEIPT_ROOTS, RECEIPT_UNWALKED_DIRS) a gateway's hcu_run.py source declares.

    None when the gateway has no receipt-scope guard (any gateway before #44). Preparation
    refuses a pinned gateway whose scope is not this module's RELEASE_* pair.
    """
    values = {}
    for node in ast.parse(source).body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in ('RECEIPT_ROOTS', 'RECEIPT_UNWALKED_DIRS')):
            value = node.value
            try:
                if (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
                        and value.func.id == 'frozenset' and len(value.args) == 1 and not value.keywords):
                    values[node.targets[0].id] = frozenset(ast.literal_eval(value.args[0]))
                else:
                    values[node.targets[0].id] = ast.literal_eval(value)
            except ValueError:
                return None
    if set(values) != {'RECEIPT_ROOTS', 'RECEIPT_UNWALKED_DIRS'}:
        return None
    return values['RECEIPT_ROOTS'], values['RECEIPT_UNWALKED_DIRS']


_JOB_ID = re.compile(r'"job_id"\s*:\s*"([^"]+)"')


def author_diagnostic(path):
    """Job ids an author's captured output names, or None when it may be a gateway record.

    The r4b case (issue #42): a probe's whole stdout, including the gateway's final
    {"job_id", "status", "exit_code"} line, redirected into a .json file. Such text is
    admissible only if it decodes as UTF-8, does not start like a JSON value (the gateway
    writes each record whole, schema first, so a torn record still starts with '{'), names
    neither the admission schema nor an admission_receipts declaration, and names at least
    one job. The caller must still show every named job released by its own terminal; text
    that names no job, or a job without one, withholds the release as before.
    """
    try:
        text = path.read_bytes().decode('utf-8')
    except (OSError, UnicodeDecodeError):
        return None
    stripped = text.lstrip(' \t\r\n')
    if (not stripped or stripped[0] in '{[' or GATEWAY_SCHEMA in text
            or 'admission_receipts' in text):
        return None
    jobs = set(_JOB_ID.findall(text))
    return jobs or None
