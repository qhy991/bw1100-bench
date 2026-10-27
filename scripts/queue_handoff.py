"""Adopt existing controllers without restarting them or rewriting Run clocks."""
from pathlib import Path
import json

from development.development_binding import MAX_AUTHORS_PER_HCU, author_capacity_ok

UNKNOWN_EXIT = 'unobserved_after_coordinator_handoff'


def process_identity(pid):
    root = Path('/proc') / str(pid)
    try:
        fields = (root / 'stat').read_text().rsplit(')', 1)[1].split()
        return {'pid': pid, 'start_ticks': fields[19], 'state': fields[0],
                'cwd': str((root / 'cwd').resolve())}
    except FileNotFoundError:
        return None


class AdoptedController:
    def __init__(self, identity):
        self.identity = identity
        self.pid = identity['pid']

    def poll(self):
        current = process_identity(self.pid)
        if current is None or current['start_ticks'] != self.identity['start_ticks']:
            return UNKNOWN_EXIT
        if current['state'] == 'Z':
            return UNKNOWN_EXIT
        if current['cwd'] != self.identity['cwd']:
            raise RuntimeError('Adopted controller working directory changed')
        return None


def validate_handoff(value, spec, allocation):
    if value.get('schema') != 'bw1100-bench.queue-handoff.v1':
        raise ValueError('Unsupported coordinator handoff')
    if value.get('allocation') != spec or Path(value['allocation_path']).resolve() != allocation.resolve():
        raise ValueError('Handoff allocation differs from the original allocation')
    old = Path(value['previous_owner_root']).resolve()
    if Path(value['scheduler_lock']).resolve() != old / 'owner.lock':
        raise ValueError('Handoff must retain the original scheduler lock')
    capacity = value.get('capacity_per_hcu')
    if not author_capacity_ok(capacity):
        raise ValueError(f'Declare one to {MAX_AUTHORS_PER_HCU} author slots per HCU')
    if not value.get('authorization'):
        raise ValueError('Handoff needs an explicit execution-policy amendment')
    previous = process_identity(value['previous_owner_pid'])
    if previous and previous['start_ticks'] == value['previous_owner_start_ticks']:
        raise ValueError('Previous coordinator is still present')
    status = value['status']
    if status['host'] != spec['host'] or status['owner_pid'] != value['previous_owner_pid']:
        raise ValueError('Previous coordinator identity differs')
    if status['stop_new_at_node_epoch'] != status['started_node_epoch'] + spec['start_window_seconds']:
        raise ValueError('Handoff must preserve the original intake window')
    identity_field = 'task' if spec.get('kind') == 'registered_task_development' else 'frozen_workspace'
    original = {row[identity_field]: row for row in spec['runs']}
    members = status['pending'] + [row[identity_field] for phase in ('running', 'finished') for row in status[phase]]
    if len(members) != len(original) or len(set(members)) != len(members) or set(members) != set(original):
        raise ValueError('Handoff must partition every original Run exactly once')
    for phase in ('running', 'finished'):
        for row in status[phase]:
            expected = original[row[identity_field]]
            if any(row.get(key) != item for key, item in expected.items()):
                raise ValueError('Handoff changed a Run binding')
            relative = Path(row['source_json_inventory'])
            if relative.is_absolute() or not (old / relative).resolve().is_relative_to(old):
                raise ValueError('Foreign source inventory in handoff')
    for row in status['running']:
        name = Path(row[identity_field]).name
        expected = value['controller_identities'][name]
        if expected['pid'] != row['pid'] or expected['cwd'] != row['root']:
            raise ValueError('Handoff controller differs from its Run')
        current = process_identity(row['pid'])
        if current and current['start_ticks'] == expected['start_ticks'] and current['state'] != 'Z':
            if current['cwd'] != row['root']:
                raise ValueError('Live controller has a foreign working directory')
        elif not (Path(row['root']) / 'campaign/controller-status.json').is_file():
            raise ValueError('Missing controller without a retained terminal observation')
    return value


def restore(value, spec, destination):
    status = value['status']
    identity_field = 'task' if spec.get('kind') == 'registered_task_development' else 'frozen_workspace'
    original = {row[identity_field]: row for row in spec['runs']}
    pending = [original[key] for key in status['pending']]
    old = Path(value['previous_owner_root'])
    for phase in ('running', 'finished'):
        for row in status[phase]:
            name = Path(row[identity_field]).name
            record = old / ('launch-' + name + '.json')
            if record.exists():
                with (destination / record.name).open('xb') as stream:
                    stream.write(record.read_bytes())
            relative = row['source_json_inventory']
            path = destination / relative
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('xb') as stream:
                    stream.write((old / relative).read_bytes())
    children = {}
    for row in status['running']:
        name = Path(row[identity_field]).name
        children[name] = (dict(row), AdoptedController(value['controller_identities'][name]))
    return pending, children, list(status['finished']), list(status['release_attention'])
