#!/usr/bin/env python3
"""Verify and materialize the pinned FlagGems source from a local archive.

The Apache-2.0 third-party source is not committed to this repository. The
archive came from the existing FlagRelease Hygon image and is accepted only at
its exact SHA-256; this script never downloads or overwrites source.
"""

import argparse
import hashlib
import os
from pathlib import Path, PurePosixPath
import tarfile


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / ".deps/flag_gems_src_540_node2"
ARCHIVE_SHA256 = "b06d741b4d1f2978a539c38ced0ec41d1621455e5dec38b6350e5c9943dcda39"
SOURCE_HASHES = {
    "flag_gems/fused/gelu_and_mul.py": "5a21e5f2994c8a0b88d4676b99b1f1621891e55b1527ebe64d90f6ab7857a532",
    "flag_gems/ops/argsort.py": "6f3aba9e7e58b770edcac1879525500a8b2991c960e4baddffd48a5f1c81361c",
    "flag_gems/ops/sort.py": "ce08520220ddd5f64c612a6ee4bce2393eb9747a836485e7bb1ec1de7029de6d",
    "flag_gems/ops/bincount.py": "26fb80c2dd9edaf9588464c33aa2a14ec6a9173bcd2ddef4ddc1be52d9d35be2",
    "flag_gems/ops/cumsum.py": "1c692bcc2a7048b893a5ae66a59ddc2a5ef7d4c99193b77a82e8d2837a4aed77",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source(root):
    for name, expected in SOURCE_HASHES.items():
        path = root / name
        if sha256(path) != expected:
            raise RuntimeError("FlagGems source differs: " + name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    args = parser.parse_args()
    if sha256(args.archive) != ARCHIVE_SHA256:
        raise RuntimeError("FlagGems archive differs from pinned FlagRelease source")
    if TARGET.exists():
        verify_source(TARGET)
        print("Verified existing", TARGET)
        return
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    staging = TARGET.with_name(TARGET.name + ".staging-" + str(os.getpid()))
    staging.mkdir()
    with tarfile.open(str(args.archive), "r:gz") as archive:
        for member in archive.getmembers():
            parts = PurePosixPath(member.name).parts
            if (member.name.startswith("/") or ".." in parts or
                    not (member.isfile() or member.isdir())):
                raise RuntimeError("Unsafe or unsupported archive member: " + member.name)
        archive.extractall(str(staging))
    verify_source(staging)
    staging.rename(TARGET)
    print("Materialized", TARGET)


if __name__ == "__main__":
    main()
