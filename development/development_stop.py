"""One monotonic halt record shared by the existing queue and its admissions."""
from pathlib import Path
import json
import os
import re
import time


def stop_path(binding):
    return Path(binding['batch_stop_file'])


def stopped(binding):
    return stop_path(binding).exists()


def halt(binding, reason, source):
    path = stop_path(binding)
    try:
        with path.open('x') as stream:
            json.dump(dict(reason=reason, source=source, task=binding.get('task'),
                           node_epoch=time.time(), owner_pid=os.getpid(),
                           action='stop new tasks and device admissions; drain active authors'), stream, indent=2)
    except FileExistsError:
        pass


LOCK_WAIT_TIMEOUT = 'timed out waiting for this HCU lock; no GPU admitted'


def lock_wait_timed_out(folder):
    """A bounded wait on a busy sibling's lock: nothing was admitted or touched.

    hcu_run.py raises this only from _acquire_hcu_lock, before it inspects the image,
    observes the device or writes admission.json. It is the expected outcome when
    several authors share one HCU, so it is a resource wait for this evaluation alone,
    not a fault that should stop every other Run on the card.
    """
    if (folder / 'admission.json').exists():
        return False
    try:
        log = (folder / 'device.log').read_text(errors='replace')
    except OSError:
        return False
    return LOCK_WAIT_TIMEOUT in log


def device_failure_is_infrastructure(folder):
    """Candidate compilation/numerics failures remain candidate evidence."""
    if lock_wait_timed_out(folder):
        return None
    admission = folder / 'admission.json'
    terminal = folder / 'admission-terminal.json'
    if admission.exists():
        try:
            after = json.loads(terminal.read_text())
        except (OSError, ValueError):
            return 'device admission has no readable terminal release observation'
        if (after.get('container_still_running') is not False or
                after.get('after_vram') != '0%' or after.get('after_kfd_visible') is not False):
            return 'device release is unknown or incomplete'
    try:
        log = (folder / 'device.log').read_text(errors='replace')
    except OSError:
        return 'device failure log is unavailable'
    signatures = (
        'No CUDA GPUs are available', 'no valid DCUs', '0 active drivers',
        'HSA_STATUS_ERROR_OUT_OF_RESOURCES', 'HCU_DEVICE_IDENTITY_FAILED',
        'permission denied while trying to connect to the Docker',
        'Cannot connect to the Docker daemon', 'could not select device driver',
    )
    if (any(signature.lower() in log.lower() for signature in signatures)
            or re.search(r'\bdevice with id=[0-9]+ is unavailable\b', log, re.IGNORECASE)):
        return 'device runtime or admission service failed; see retained device.log'
    return None
