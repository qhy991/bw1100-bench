"""Coordinator adoption preserves Run membership and never invents an exit code."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from scripts.queue_handoff import AdoptedController, UNKNOWN_EXIT, restore, validate_handoff


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.old = self.root / 'old'
        (self.old / 'intake').mkdir(parents=True)
        (self.old / 'intake/a.json').write_text('["source.json"]')
        self.rows = [dict(frozen_workspace=name, root=str(self.root / name), host='h', hcu=4)
                     for name in ('a', 'b', 'c')]
        self.spec = dict(host='h', start_window_seconds=100, capacity_per_hcu=1, runs=self.rows)
        self.path = self.root / 'allocation.json'
        self.path.write_text(json.dumps(self.spec))
        self.identity = dict(pid=11, start_ticks='42', state='S', cwd=self.rows[0]['root'])
        status = dict(host='h', owner_pid=10, started_node_epoch=50, stop_new_at_node_epoch=150,
                      pending=['b', 'c'], running=[dict(self.rows[0], pid=11, source_json_inventory='intake/a.json')],
                      finished=[], release_attention=[dict(hcu=4, release_records_clean=False)])
        self.value = dict(schema='bw1100-bench.queue-handoff.v1', allocation=self.spec,
                          allocation_path=str(self.path), previous_owner_root=str(self.old),
                          scheduler_lock=str(self.old / 'owner.lock'), capacity_per_hcu=2,
                          authorization='Direct user concurrency amendment', previous_owner_pid=10,
                          previous_owner_start_ticks='1', status=status, controller_identities={'a':self.identity})

    def identity_for(self, pid):
        return self.identity if pid == 11 else None

    def test_adoption_keeps_pending_order_pid_inventory_and_release_block(self):
        with patch('scripts.queue_handoff.process_identity', side_effect=self.identity_for):
            validate_handoff(self.value, self.spec, self.path)
            destination = self.root / 'new'
            destination.mkdir()
            pending, children, finished, attention = restore(self.value, self.spec, destination)
            self.assertEqual([r['frozen_workspace'] for r in pending], ['b', 'c'])
            self.assertEqual(children['a'][1].pid, 11)
            self.assertIsNone(children['a'][1].poll())
            self.assertEqual((destination / 'intake/a.json').read_bytes(), (self.old / 'intake/a.json').read_bytes())
            self.assertEqual(attention, self.value['status']['release_attention'])
            self.assertEqual(finished, [])
            self.assertFalse((self.root / 'a/campaign/deadline.json').exists())

    def test_rejects_membership_clock_binding_and_lock_changes(self):
        for change in ('duplicate', 'clock', 'binding', 'lock', 'live_owner'):
            bad = copy.deepcopy(self.value)
            if change == 'duplicate':bad['status']['pending']=['a','c']
            if change == 'clock':bad['status']['stop_new_at_node_epoch']=151
            if change == 'binding':bad['status']['running'][0]['hcu']=5
            if change == 'lock':bad['scheduler_lock']=str(self.root / 'foreign.lock')
            current = lambda pid: dict(pid=10,start_ticks='1') if change=='live_owner' and pid==10 else self.identity_for(pid)
            with self.subTest(change=change), patch('scripts.queue_handoff.process_identity',side_effect=current), self.assertRaises(ValueError):
                validate_handoff(bad,self.spec,self.path)

    def test_capacity_is_bounded_by_the_named_author_limit(self):
        from scripts.queue_handoff import MAX_AUTHORS_PER_HCU
        for capacity, accepted in ((MAX_AUTHORS_PER_HCU, True), (MAX_AUTHORS_PER_HCU + 1, False), (0, False)):
            value = dict(copy.deepcopy(self.value), capacity_per_hcu=capacity)
            with self.subTest(capacity=capacity), patch('scripts.queue_handoff.process_identity', side_effect=self.identity_for):
                if accepted:
                    validate_handoff(value, self.spec, self.path)
                else:
                    with self.assertRaisesRegex(ValueError, 'author slots per HCU'):
                        validate_handoff(value, self.spec, self.path)

    def test_missing_controller_requires_terminal_observation(self):
        with patch('scripts.queue_handoff.process_identity',return_value=None), self.assertRaises(ValueError):
            validate_handoff(self.value,self.spec,self.path)
        terminal = self.root / 'a/campaign/controller-status.json'
        terminal.parent.mkdir(parents=True)
        terminal.write_text('{}')
        with patch('scripts.queue_handoff.process_identity',return_value=None):
            validate_handoff(self.value,self.spec,self.path)
            self.assertEqual(AdoptedController(self.identity).poll(),UNKNOWN_EXIT)

    def test_pid_reuse_is_unknown_and_changed_cwd_is_refused(self):
        for current in (None,dict(self.identity,start_ticks='43'),dict(self.identity,state='Z')):
            with patch('scripts.queue_handoff.process_identity',return_value=current):
                self.assertEqual(AdoptedController(self.identity).poll(),UNKNOWN_EXIT)
        with patch('scripts.queue_handoff.process_identity',return_value=dict(self.identity,cwd='/foreign')), self.assertRaises(RuntimeError):
            AdoptedController(self.identity).poll()
