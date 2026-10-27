# Author concurrency and coordinator handoff

`capacity_per_hcu` permits one to six author processes on each explicitly allocated HCU (`MAX_AUTHORS_PER_HCU`). The qualified HCU gateway continues to serialize device work, so authors beyond the card's device-job throughput wait on its lock while their search clock runs; choose the value from the measured device-busy share and queue wait of the previous round. Author capacity does not establish physical device exclusivity.

A live cohort needs an explicit execution-policy amendment. Keep its original allocation and Run bindings. At a published scheduler checkpoint, stop only the exact coordinator PID after checking its start ticks. Preserve its status, launch records and active controller identities. Confirm that all reservations appear exactly once in the status. If this check fails, resume the coordinator and abort the handoff.

After the old coordinator exits, start `scripts/bench_queue.py --allocation ORIGINAL --state-directory NEW_OWNER --handoff MANIFEST` from the reviewed successor source. The manifest uses schema `bw1100-bench.queue-handoff.v1`; it records the original allocation, owner root/PID/start ticks, scheduler lock, complete status, controller identities, new capacity and authorization. The successor uses the original scheduler lock. It copies source inventories into its fresh evidence directory and adopts existing controllers without starting them again. Pending Runs receive their first intake normally. The original intake window stays fixed.

The new coordinator cannot recover the operating-system exit code of an adopted process. It records a null exit code with `unobserved_after_coordinator_handoff`, then reads the retained DONE and release records. Missing or contradictory release records continue to block that HCU.

Preserve the old owner directory. Record which Runs overlap the amendment and retain both scheduling regimes in the report. A capacity change is an execution-policy change and limits attribution to the Compiler alone.
