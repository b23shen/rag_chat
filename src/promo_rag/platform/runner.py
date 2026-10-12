"""Queueing, claiming and running workflow runs.

enqueue_run      POST /workflows/{name}/start: insert a `queued` workflow_runs row
claim_next_run   a worker takes the oldest queued run (FOR UPDATE SKIP LOCKED: two
                 workers never get the same run, and a restart loses nothing)
execute_run      run the runbook's actions in order, recording each in action_runs;
                 the first failure stops the run and marks the rest `skipped`
"""

import os
import socket
import traceback
import uuid
from collections.abc import Callable
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from promo_rag.db.models import ActionRun, Dataset, Workflow, WorkflowRun
from promo_rag.platform.runbook import Action, ActionContext, Runbook, workflow_actions


class RunError(ValueError):
    """The run can't be started (unknown workflow, missing dataset, ...)."""


def enqueue_run(
    session: Session, workflow_name: str, dataset_id: uuid.UUID | None, params: dict | None = None
) -> WorkflowRun:
    workflow = session.get(Workflow, workflow_name)
    if workflow is None:
        raise RunError(f"no workflow {workflow_name!r}")
    runbook = Runbook.model_validate(workflow.definition)
    if runbook.input.get("dataset") == "required" and dataset_id is None:
        raise RunError(f"{workflow_name} needs a dataset_id")
    if dataset_id is not None and session.get(Dataset, dataset_id) is None:
        raise RunError(f"no dataset {dataset_id}")
    unknown = set(params or {}) - set(runbook.params)
    if unknown:
        raise RunError(f"unknown params {sorted(unknown)}; allowed: {sorted(runbook.params)}")
    run = WorkflowRun(
        workflow_name=workflow_name,
        workflow_version=workflow.version,
        dataset_id=dataset_id,
        params=runbook.params | (params or {}),
    )
    session.add(run)
    session.commit()
    return run


def claim_next_run(session: Session, worker: str | None = None) -> WorkflowRun | None:
    run = session.scalars(
        select(WorkflowRun)
        .where(WorkflowRun.status == "queued")
        .order_by(WorkflowRun.created_at)
        .limit(1)
        .with_for_update(skip_locked=True)
    ).first()
    if run is None:
        session.rollback()
        return None
    run.status = "running"
    run.started_at = datetime.now(UTC)
    run.worker = worker or f"{socket.gethostname()}:{os.getpid()}"
    session.commit()
    return run


Recorder = Callable[[str, int, str, dict | None, str | None], None]


def run_actions(steps: list[tuple[str, Action]], ctx: ActionContext, record: Recorder) -> bool:
    """Run the steps in order. `record(name, position, status, output, error)` is called on
    every change. Returns True if every step succeeded."""
    for position, (name, action) in enumerate(steps):
        record(name, position, "running", None, None)
        try:
            output = action(ctx)
        except Exception:  # noqa: BLE001 - any failure is recorded and stops the run
            record(name, position, "failed", None, traceback.format_exc(limit=5))
            for later, (skipped, _) in enumerate(steps[position + 1 :], start=position + 1):
                record(skipped, later, "skipped", None, None)
            return False
        ctx.outputs[name] = output
        record(name, position, "succeeded", output, None)
    return True


def execute_run(session: Session, run: WorkflowRun, log=print) -> WorkflowRun:
    workflow = session.get(Workflow, run.workflow_name)
    runbook = Runbook.model_validate(workflow.definition)
    actions = workflow_actions(run.workflow_name)
    steps = [(a.name, actions[a.name]) for a in runbook.actions]

    rows = {}
    for position, (name, _) in enumerate(steps):
        rows[position] = ActionRun(run_id=run.id, position=position, name=name)
        session.add(rows[position])
    session.commit()

    def record(name, position, status, output, error):
        if status == "failed":
            session.rollback()  # a database error leaves the transaction unusable until then
        row = rows[position]
        row.status, row.error = status, error
        now = datetime.now(UTC)
        if status == "running":
            row.started_at = now
        elif status in ("succeeded", "failed"):
            row.finished_at, row.output = now, output
        session.commit()
        log(f"  [{run.workflow_name}] {name}: {status}")

    ctx = ActionContext(
        session=session,
        run_id=str(run.id),
        dataset_id=str(run.dataset_id) if run.dataset_id else None,
        params=run.params,
        log=log,
    )
    ok = run_actions(steps, ctx, record)
    session.rollback()  # drop anything a failed action left half-done
    run = session.get(WorkflowRun, run.id)
    run.status = "succeeded" if ok else "failed"
    run.output = ctx.outputs
    run.error = None if ok else next(r.error for r in rows.values() if r.status == "failed")
    run.finished_at = datetime.now(UTC)
    session.commit()
    return run
