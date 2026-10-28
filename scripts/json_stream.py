"""Read every complete top-level JSON value; never accept a partial prefix."""
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
# RECEIPT_ROOTS / RECEIPT_UNWALKED_DIRS, #44). Outside RELEASE_ROOTS, release_audit prunes
# RELEASE_UNWALKED_DIRS. Inside, both checks fail closed without reading cache bytes as
# records: a gateway pinned before #44 could write a receipt into campaign/__pycache__,
# while Triton and AITER leave partial or binary .json files under .local (#45 review).
RELEASE_ROOTS = ('campaign', '.local')
RELEASE_UNWALKED_DIRS = frozenset(('.git', '.deps', '.data', 'triton-cache', 'aiter-jit-cache', '__pycache__'))
# An author's notes directory (r6 STOP-hcu5: analysis prose in a .json inside the scope,
# unreadable as a record, halted an HCU no job was holding). Unlike a cache it is created by
# hand, so only a real directory gets the cache rule: its .json files are read only when
# receipt_shaped admits them. A symlink named notes is judged like any symlinked directory,
# and the gateway refuses a receipt below it (RECEIPT_UNWALKED_DIRS).
AUTHOR_NOTES_DIR = 'notes'
# Directories the gateway must refuse a receipt in: everything the walk does not read whole.
RECEIPT_REFUSED_DIRS = RELEASE_UNWALKED_DIRS | {AUTHOR_NOTES_DIR}
_RECEIPT_MARKERS = (GATEWAY_SCHEMA.encode(), b'admission_receipts')


def receipt_shaped(path):
    """True when a file's bytes could be or hide a gateway record (as author_diagnostic
    reads them); an unreadable file counts as one, since its bytes cannot be ruled out."""
    try:
        data = path.read_bytes()
    except OSError:
        return True
    return any(marker in data for marker in _RECEIPT_MARKERS)


def release_json_files(root, issue, note=lambda path, reason: None, whole_root=False):
    """Yield every .json file a release check must read as a record.

    Inside RELEASE_ROOTS every directory is walked. Below a RELEASE_UNWALKED_DIRS component
    there, only a regular file that receipt_shaped admits is yielded; any other cache file
    is ignored whatever its bytes. An unreadable directory is issue(relative, reason): the
    caller withholds the release. A symlinked directory is never followed: see
    _symlinked_directory. With whole_root the rest of the tree is walked as before, pruning
    RELEASE_UNWALKED_DIRS and reporting unreadable directories.
    """
    def relative(path):
        return os.path.relpath(str(path), str(root))

    def unreadable(error):
        issue(relative(error.filename), 'unreadable_directory')

    tops = [root] if whole_root else []
    for name in () if whole_root else RELEASE_ROOTS:
        if (root / name).is_symlink():
            _symlinked_directory(root, root / name, issue, note)
        elif os.path.isdir(str(root / name)):
            tops.append(root / name)
    for top in tops:
        for directory, dirs, files in os.walk(str(top), onerror=unreadable):
            here = Path(directory)
            parts = here.relative_to(root).parts
            in_scope = bool(parts) and parts[0] in RELEASE_ROOTS
            in_cache = in_scope and not RELEASE_UNWALKED_DIRS.isdisjoint(parts[1:])
            in_notes = in_scope and AUTHOR_NOTES_DIR in parts[1:]
            kept = []
            for name in dirs:
                child = here / name
                if not (in_scope or (not parts and name in RELEASE_ROOTS)):
                    if name not in RELEASE_UNWALKED_DIRS:
                        kept.append(name)  # outside the scope: unchanged since #44
                elif not child.is_symlink():
                    kept.append(name)
                elif not (in_cache or name in RELEASE_UNWALKED_DIRS):
                    _symlinked_directory(root, child, issue, note)
                # A symlink at or below a cache directory is ignored, like the cache.
            dirs[:] = kept
            for name in files:
                path = here / name
                if not name.endswith('.json'):
                    continue
                if not (in_cache or in_notes):
                    yield path
                elif not path.is_symlink() and receipt_shaped(path):
                    yield path


def _symlinked_directory(root, link, issue, note):
    """Judge a symlinked directory under campaign/ or .local/ outside the caches, unfollowed.

    A gateway since #44 resolves the receipt path and refuses one outside the scope, and
    creates it without following a symlink; preparation refuses an older gateway. A receipt
    could hide behind the link only where its target is inside the Run root yet outside
    what the walk already reads. So the link withholds the release only if that part of its
    target holds a receipt-shaped .json file (or cannot be read); otherwise, for example a
    venv lib64 -> lib or an author's .local/latest, it is only a note.
    """
    relative = os.path.relpath(str(link), str(root))
    root_real = root.resolve()
    pending, visited = [link], set()
    while pending:
        try:
            target = pending.pop().resolve(strict=True)
            parts = target.relative_to(root_real).parts
        except (OSError, RuntimeError, ValueError):
            continue  # dangling, looping or outside the Run root: no receipt is written there
        if (parts and parts[0] in RELEASE_ROOTS) or target in visited or not target.is_dir():
            continue  # already read directly by the walk
        visited.add(target)
        failed = []
        for directory, dirs, files in os.walk(str(target), onerror=failed.append):
            here = Path(directory)
            pending.extend(here / name for name in dirs if (here / name).is_symlink())
            dirs[:] = [name for name in dirs if not (here / name).is_symlink()
                       and not (here == root_real and name in RELEASE_ROOTS)]
            if any(name.endswith('.json') and not (here / name).is_symlink()
                   and receipt_shaped(here / name) for name in files):
                failed.append(None)
        if failed:
            issue(relative, 'receipt_behind_symlinked_directory')
            return
    note(relative, 'symlink_directory_not_followed')


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
