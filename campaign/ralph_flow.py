"""Fresh-session Ralph with completion checks, derived from official ralph_loop.

Official source SHA256: dff76869f6c823cb6136e0647d4a52777d1580bb05eff36806211182c946a852.
The installed official flow is unchanged. This local variant checks the
durable handoff after each returned agent turn, so it needs no process signal.
"""
import sys
import time
from pathlib import Path
from typing import Any
from hmz.flows import Agent, Allowance, flow

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from campaign.completion import completed


@flow(budget=Allowance(tokens=10.0), resumable=True)
def run(agents: tuple[Agent], task: str, state: dict[str, Any] | None = None) -> None:
    (agent,) = agents
    kept = state if state is not None else {}
    stalled = 0
    while True:
        deadline = Path('campaign/deadline.json')
        if deadline.exists():
            import json
            stop_at = json.loads(deadline.read_text())['stop_at_epoch']
            exhaustion = Path('campaign/SEARCH-EXHAUSTED.json')
            if exhaustion.exists() and time.time() < stop_at - 120:
                note = json.loads(exhaustion.read_text())
                outcomes = [json.loads(p.read_text()) for p in Path('campaign/evaluations').glob('*/outcome.json')]
                intake = json.loads(Path('campaign/intake.json').read_text())
                qualified = any(d.get('status') == 'accepted' and
                                (not intake.get('seed') or d.get('is_parent_initializer')) for d in outcomes)
                proofs = note.get('new_hypothesis_evidence', [])
                if (qualified and note.get('reason') and len(proofs) >= 2
                        and all(Path(p).is_file() for p in proofs)):
                    print('agent-reported exhaustion with new evidence; controller holds incumbent without more model calls', flush=True)
                    time.sleep(min(20, stop_at - 120 - time.time()))
                    continue
            if time.time() >= stop_at - 900 and time.time() < stop_at - 120:
                print('search window closed; controller waits for endpoint without another model call', flush=True)
                time.sleep(min(20, stop_at - 120 - time.time()))
                continue
            if time.time() >= stop_at - 120 and not Path('campaign/DONE.json').exists():
                import subprocess
                subprocess.run([sys.executable, 'campaign/finish.py'], check=True)
        if completed(Path.cwd()):
            print('stopping: validated campaign/DONE.json; no owned live container')
            return
        kept['rounds'] = kept.get('rounds', 0) + 1
        print(f"round {kept['rounds']}")
        answered = agent(task, suppress=True)
        if completed(Path.cwd()):
            print('stopping: validated campaign/DONE.json; no owned live container')
            return
        stalled = 0 if answered else stalled + 1
        if stalled >= 3:
            print('stopping: three rounds answered with nothing; inspect provider/state')
            return
        time.sleep(5)
