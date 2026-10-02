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
