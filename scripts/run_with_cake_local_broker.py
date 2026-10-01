"""Run one bounded correctness command under Cake's Hygon local admission."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

from open_cake_ir.evaluation.local_broker import _acquire, _lock_path, observe_local_job


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not command or args.timeout < 1:
        parser.error("a command and positive timeout are required")
    if args.receipt.exists():
        parser.error("receipt path must be new")
    job = "hip-" + uuid.uuid4().hex[:12]
    fd = _acquire(_lock_path("hip"))
    os.set_inheritable(fd, True)
    os.environ["METAL_JOB_ID"] = job
    os.environ["METAL_BROKER_LOCK_FD"] = str(fd)
    observe_local_job("hip")
    receipt = {
        "cake_broker_job_id": job,
        "mode": "local_serialized",
        "external_gpu_activity": "not_excluded",
        "allocation_scope": "this transient container and its child process",
        "command": command,
        "status": "admitted",
    }
    with args.receipt.open("x") as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps({"cake_broker_job_id": job, "status": "admitted"}), flush=True)
    try:
        completed = subprocess.run(command, pass_fds=(fd,), timeout=args.timeout)
        receipt["status"] = "completed" if completed.returncode == 0 else "child_failed"
        receipt["exit_code"] = completed.returncode
        return completed.returncode
    except subprocess.TimeoutExpired:
        receipt["status"] = "child_timeout"
        receipt["exit_code"] = 124
        return 124
    finally:
        with args.receipt.with_name(args.receipt.stem + "-terminal.json").open("x") as stream:
            json.dump(receipt, stream, indent=2)
        os.close(fd)
        print(json.dumps({"cake_broker_job_id": job, "status": receipt["status"]}), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
