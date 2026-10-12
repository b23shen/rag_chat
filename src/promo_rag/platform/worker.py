"""The worker: takes queued workflow runs and executes them.

uv run worker            keep polling for queued runs
uv run worker --once     run everything that's queued now, then exit
"""

import argparse
import sys
import time

from promo_rag.agents.tracing import flush_traces
from promo_rag.db.session import new_session
from promo_rag.platform.runbook import sync_workflows
from promo_rag.platform.runner import claim_next_run, execute_run


def main() -> int:
    parser = argparse.ArgumentParser(prog="worker")
    parser.add_argument("--once", action="store_true", help="exit when the queue is empty")
    parser.add_argument("--poll", type=float, default=5.0, help="seconds between checks when idle")
    args = parser.parse_args()

    with new_session() as session:
        sync_workflows(session)
    print("worker started; waiting for queued runs" + (" (once)" if args.once else ""))
    while True:
        with new_session() as session:
            run = claim_next_run(session)
            if run is not None:
                print(f"run {run.id}: {run.workflow_name} (dataset {run.dataset_id})")
                run = execute_run(session, run)
                print(f"run {run.id}: {run.status}")
                flush_traces()
                continue
        if args.once:
            return 0
        time.sleep(args.poll)


if __name__ == "__main__":
    sys.exit(main())
