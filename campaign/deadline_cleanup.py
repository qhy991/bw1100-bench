"""Stop only owned model/controller stragglers in this continuation root."""
import json
import os
from pathlib import Path
import signal
import time

ROOT = Path(__file__).resolve().parents[1]
deadline = json.loads((ROOT / 'campaign/deadline.json').read_text())
if time.time() < deadline['stop_at_epoch']:
    raise SystemExit('No cleanup before the authorized wall-clock deadline')
targets = []
for p in Path('/proc').iterdir():
    if not p.name.isdigit():
        continue
    try:
        if p.stat().st_uid != os.getuid() or (p / 'cwd').resolve() != ROOT:
            continue
        args = [a.decode(errors='replace') for a in (p / 'cmdline').read_bytes().split(b'\0') if a]
        controller = any(a == 'hmz' or a.endswith('/hmz') for a in args) and 'exec' in args
        agent = args and (args[0] == 'claude' or args[0].endswith('/claude')) and '--print' in args
        if not (controller or agent):
            continue
        stat = (p / 'stat').read_text()
        start = stat[stat.rfind(')') + 2:].split()[19]
        targets.append((p, start))
    except (OSError, ValueError):
        continue
for sig in (signal.SIGTERM, signal.SIGKILL):
    for p, expected in targets:
        try:
            stat = (p / 'stat').read_text()
            start = stat[stat.rfind(')') + 2:].split()[19]
            if p.stat().st_uid == os.getuid() and start == expected and (p / 'cwd').resolve() == ROOT:
                os.kill(int(p.name), sig)
                print(sig.name, p.name, flush=True)
        except (OSError, ValueError):
            pass
    if sig == signal.SIGTERM and targets:
        time.sleep(5)
