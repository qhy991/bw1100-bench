#!/usr/bin/env python3
"""Accept a rocprof CSV only when the requested counters and kernel are present."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def requested_metrics(path):
    lines = [line.split(':', 1)[1].split() for line in path.read_text().splitlines()
             if line.startswith('pmc:')]
    if len(lines) != 1 or not 1 <= len(lines[0]) <= 6:
        raise ValueError('expected one pmc line with 1..6 counters')
    return lines[0]


def verify(pmc, report, kernel_pattern):
    metrics = requested_metrics(pmc)
    pattern = re.compile(kernel_pattern)
    with report.open(newline='') as stream:
        reader = csv.DictReader(stream)
        required = {'KernelName'} | set(metrics)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError('rocprof CSV is missing columns: ' + ', '.join(sorted(missing)))
        rows = list(reader)
    matched = [row for row in rows if pattern.search(row['KernelName'])]
    if not matched:
        raise ValueError('rocprof CSV has no matching kernel row: ' + kernel_pattern)
    return {
        'schema': 'bw1100-bench.rocprof-validation.v1',
        'status': 'passed',
        'report_sha256': sha256(report),
        'pmc_sha256': sha256(pmc),
        'expected_kernel_regex': kernel_pattern,
        'requested_metrics': metrics,
        'total_rows': len(rows),
        'matched_rows': len(matched),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pmc', type=Path)
    parser.add_argument('report', type=Path)
    parser.add_argument('kernel_regex')
    args = parser.parse_args()
    result = verify(args.pmc, args.report, args.kernel_regex)
    output = args.report.with_name(args.report.stem + '-validation.json')
    with output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'status': result['status'], 'matched_rows': result['matched_rows'],
                      'validation': str(output)}))


if __name__ == '__main__':
    main()
