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


def lock_wait_timed_out(folder, receipt='admission.json', log_name='device.log'):
    """A bounded wait on a busy sibling's lock: nothing was admitted or touched.

    hcu_run.py raises this only from _acquire_hcu_lock, before it inspects the image,
    observes the device or writes admission.json. It is the expected outcome when
    several authors share one HCU, so it is a resource wait for this evaluation alone,
    not a fault that should stop every other Run on the card.
    """
    if (folder / receipt).exists():
        return False
    try:
        log = (folder / log_name).read_text(errors='replace')
    except OSError:
        return False
    return LOCK_WAIT_TIMEOUT in log


# Refusals scripts/hcu_run.py raises before it writes the admission receipt and without writing
# a STOP of its own: nothing was admitted, so there is no release to observe.
PRE_ADMISSION_REFUSALS = (
    LOCK_WAIT_TIMEOUT,
    'another local bw1100-bench job holds this HCU lock',
    'experiment owner stopped new GPU admissions',
    'GPU admission window expired before device execution',
    'selected HCU is not idle by observed VRAM/KFD check',
)


# The gateway's last line when --device resolves to another physical card than the plan's
# hcu_pci (node4, 2026-10-28). Nothing was admitted, but every later admission on this card
# would be refused the same way, so it stops the card with this reason rather than failing
# the one evaluation or reading the missing receipt as an unknown release.
HCU_PCI_MISMATCH = 'selected HCU is not the expected physical PCI device; no GPU admitted'
HCU_PCI_MISMATCH_STOP = 'HCU renumbered: the assigned index is no longer the planned physical card'


def _last_gateway_line(folder, receipt, log_name):
    terminal = folder / (Path(receipt).stem + '-terminal.json')
    if (folder / receipt).exists() or terminal.exists():
        return None
    try:
        log = (folder / log_name).read_text(errors='replace')
    except OSError:
        return None
    return next((line.strip() for line in reversed(log.splitlines()) if line.strip()), '')


def hcu_pci_mismatch(folder, receipt='admission.json', log_name='device.log'):
    return _last_gateway_line(folder, receipt, log_name) == 'RuntimeError: ' + HCU_PCI_MISMATCH


def refused_before_admission(folder, receipt='admission.json', log_name='device.log'):
    """The gateway refusal in log_name when neither receipt nor its terminal exists, else None."""
    terminal = folder / (Path(receipt).stem + '-terminal.json')
    if (folder / receipt).exists() or terminal.exists():
        return None
    try:
        log = (folder / log_name).read_text(errors='replace')
    except OSError:
        return None
    # The refusal is the gateway's own last word. Container output is printed before the
    # terminal receipt, so a string an author's command echoed cannot be the final line.
    last = next((line.strip() for line in reversed(log.splitlines()) if line.strip()), '')
    return next((reason for reason in PRE_ADMISSION_REFUSALS
                 if last == 'RuntimeError: ' + reason), None)


def device_failure_is_infrastructure(folder, receipt='admission.json', log_name='device.log'):
    """Candidate compilation/numerics failures remain candidate evidence."""
    if hcu_pci_mismatch(folder, receipt, log_name):
        return HCU_PCI_MISMATCH_STOP
    if lock_wait_timed_out(folder, receipt, log_name):
        return None
    admission = folder / receipt
    terminal = folder / (Path(receipt).stem + '-terminal.json')
    if admission.exists():
        try:
            after = json.loads(terminal.read_text())
        except (OSError, ValueError):
            return 'device admission has no readable terminal release observation'
        if (after.get('container_still_running') is not False or
                after.get('after_vram') != '0%' or after.get('after_kfd_visible') is not False):
            return 'device release is unknown or incomplete'
    try:
        log = (folder / log_name).read_text(errors='replace')
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
