"""One bounded external HMZ 0.2.1 Ralph loop with frozen Cake tools and the HIP probe."""
import asyncio
import json
import sys
import time
from pathlib import Path

from hmz.flows import (
    Agent,
    AgentCollection,
    EnvCollection,
    FlowContext,
    FlowParams,
    HarnessError,
    HarnessNotInstalled,
    HarnessSandboxed,
    LocalEnv,
    flow,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from development_stop import stopped
from campaign.compiler_tools import author_context


class Agents(AgentCollection):
    agent: Agent


class Envs(EnvCollection):
    workspace: LocalEnv


class Params(FlowParams):
    max_rounds: int | None = None


def await_close_decision(root, binding, deadline, text):
    """Hold further turns while the owner validates a search-close request."""
    ready = root / 'campaign/author-round-terminal.json'
    temporary = ready.with_suffix('.json.tmp')
    temporary.write_text(json.dumps({'request': text, 'node_epoch': time.time()}))
    temporary.replace(ready)
    while time.time() < deadline['search_stop_at_epoch']:
        if stopped(binding):
            return 'stopped'
        try:
            decision = json.loads((root / 'campaign/search-close-owner.json').read_text())
        except (OSError, ValueError):
            decision = {}
        if decision.get('request') == text and decision.get('decision') in ('approved', 'refused'):
            return decision['decision']
        time.sleep(1)
    return 'expired'


@flow(agents=Agents, envs=Envs, params=Params, resumable=True)
async def run(task: str, *, agents: Agents, envs: Envs, params: Params, ctx: FlowContext) -> None:
    """Fresh author sessions until the search cutoff, an approved close, a STOP or three stalls."""
    root = Path.cwd()
    agent = agents['agent']
    state = ctx.state
    assert state is not None
    binding = json.loads((root / 'campaign/development-binding.json').read_text())
    stalled = int(state.get('stalled', 0))
    rounds_this_call = 0
    while True:
        if stopped(binding):
            return
        deadline = json.loads((root / 'campaign/deadline.json').read_text())
        if time.time() >= deadline['search_stop_at_epoch']:
            return
        prompt = (task + author_context(root)
                  + '\nRead your own campaign/evaluations outcomes and generated source. '
                    'Token usage is accounting only.')
        session = await agent.spawn()
        try:
            answer = await agent.run(prompt, session=session, env=envs['workspace'])
        except (HarnessNotInstalled, HarnessSandboxed):
            raise
        except HarnessError as error:
            # 0.2.x ends the whole run on one failed turn; retain it and keep searching.
            print(f'author turn failed: {error}')
            answer = ''
        if stopped(binding):
            return
        stalled = 0 if answer else stalled + 1
        state['stalled'] = stalled
        state['rounds'] = int(state.get('rounds', 0)) + 1
        rounds_this_call += 1
        if stalled >= 3:
            print('stopping: three author turns in a row returned no answer')
            return
        request = root / 'campaign/SEARCH_CLOSE.json'
        if request.exists():
            if await_close_decision(root, binding, deadline, request.read_text()) != 'refused':
                return
        if params.max_rounds is not None and rounds_this_call >= params.max_rounds:
            return
        await asyncio.sleep(2)
