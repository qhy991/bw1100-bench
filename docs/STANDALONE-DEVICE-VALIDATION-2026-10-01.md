# Standalone BW1100 benchmark device validation, 2026-10-01

The independent entry at `scripts/dtk.sh gpu` was exercised on `bw1100-1`
(node4), HCU 0 (`BW1101/gfx938`). The executed source was commit
`179f5789ca00371984dfc246e4ef3fc485f8253f`; the pre-existing local DTK
image resolved to
`sha256:3ad0ae7192b8f9bafdf5b48fc414f8785f3c2463005e6b25290b7f75146ff260`.
The upstream SOL-ExecBench and dataset revisions remain the locks in
`suite.json`. Raw JSON receipts and benchmark reports are retained in the
node's ignored `$HOME/bw1100-bench-standalone-20261001/results/` directory.

| Check | Benchmark report | Admission terminal | Outcome |
| --- | --- | --- | --- |
| L1/069 RMSNorm, original smoke workload, 2 rounds | `l1-069-standalone-smoke-001.json` | `l1-069-standalone-admission-001-terminal.json`, job `bw-8ff521edf0dc` | 2/2 correct; partial only |
| L1/069 RMSNorm, all 16 original workloads, 10 rounds each | `l1-069-standalone-full-001.json` | `l1-069-standalone-full-admission-001-terminal.json`, job `bw-a55886f976fe` | 160/160 correct; `full_device_correctness=true` for L1/069 |

Both terminal receipts say `completed`, exit 0, before/after HCU VRAM 0%,
no visible KFD process afterward, and no remaining named container. A separate
post-run `hy-smi` snapshot showed all seven cards at 0% use and 0% memory;
`fuser /dev/kfd` showed no process. The full report's SHA-256 is
`276dedc044c0d28f616e2d43e2082c3cb15a6c19935f65e1ef37c2e970ea08c3`;
its terminal receipt's SHA-256 is
`c5dc6372372f9c0b933c8157f7253efbb4dc968b5b67b6293cf1845c81a79f96`.

This qualifies the independent entry and the exact L1/069 candidate's complete
correctness workload on that device. It does not qualify the other nine tasks,
physical exclusivity, or performance. The source in this record ran the image
by its inspected tag; a subsequent change pins `docker run` to the inspected
image ID to close the inspect/run identity gap and needs its own device smoke.
