"""The workflow HTTP API.

    POST /datasets                       {name, description} → dataset_id
    POST /datasets/{id}/artifacts        upload one fetched page (file + source_url) → artifact_id
    GET  /datasets/{id}                  the dataset and how many artifacts it has
    GET  /workflows                      the workflows synced from their runbooks
    POST /workflows/{name}/start         {dataset_id, params} → run_id (202: queued for a worker)
    GET  /runs/{id}/status               queued | running | succeeded | failed, with each action
    GET  /runs/{id}/content              the run's output, once it has finished
    GET  /health

Starting a run only queues it; `uv run worker` executes it. Run the server with `uv run serve`.
"""

import uuid
from collections.abc import Iterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from promo_rag.db.models import ActionRun, Dataset, Workflow, WorkflowRun
from promo_rag.db.session import new_session
from promo_rag.platform.datasets import add_artifact, create_dataset, dataset_size, link_artifacts
from promo_rag.platform.runbook import sync_workflows
from promo_rag.platform.runner import RunError, enqueue_run
from promo_rag.workflows.ingestion import deps


@asynccontextmanager
async def lifespan(app: FastAPI):
    with new_session() as session:
        app.state.workflows = sync_workflows(session)
    yield


app = FastAPI(title="promo-rag workflows", lifespan=lifespan)


def db() -> Iterator[Session]:
    with new_session() as session:
        yield session


DB = Annotated[Session, Depends(db)]


class DatasetIn(BaseModel):
    name: str
    description: str | None = None


class StartIn(BaseModel):
    dataset_id: uuid.UUID | None = None
    params: dict[str, Any] = {}


@app.get("/health")
def health(session: DB):
    session.execute(text("SELECT 1"))
    return {"status": "ok", "workflows": app.state.workflows}


@app.post("/datasets", status_code=201)
def post_dataset(body: DatasetIn, session: DB):
    dataset = create_dataset(session, body.name, body.description)
    session.commit()
    return {"dataset_id": dataset.id}


@app.post("/datasets/{dataset_id}/artifacts", status_code=201)
def post_artifact(
    dataset_id: uuid.UUID,
    session: DB,
    file: Annotated[UploadFile, File(description="the page's bytes, exactly as fetched")],
    source_url: Annotated[str, Form()],
    fetched_at: Annotated[datetime | None, Form()] = None,
    http_status: Annotated[int, Form()] = 200,
):
    if session.get(Dataset, dataset_id) is None:
        raise HTTPException(404, f"no dataset {dataset_id}")
    artifact_id, created = add_artifact(
        session,
        deps.blob_store(),
        file.file.read(),
        source_url=source_url,
        fetched_at=fetched_at or datetime.now(UTC),
        http_status=http_status,
        mime_type=file.content_type or "text/html",
    )
    link_artifacts(session, dataset_id, [artifact_id])
    session.commit()
    return {"artifact_id": artifact_id, "new": created}


@app.get("/datasets/{dataset_id}")
def get_dataset(dataset_id: uuid.UUID, session: DB):
    dataset = session.get(Dataset, dataset_id)
    if dataset is None:
        raise HTTPException(404, f"no dataset {dataset_id}")
    return {
        "dataset_id": dataset.id,
        "name": dataset.name,
        "description": dataset.description,
        "artifacts": dataset_size(session, dataset_id),
        "created_at": dataset.created_at,
    }


@app.get("/workflows")
def get_workflows(session: DB):
    return [
        {
            "name": w.name,
            "version": w.version,
            "description": w.description,
            "actions": [a["name"] for a in w.definition["actions"]],
            "params": w.definition["params"],
        }
        for w in session.scalars(select(Workflow).order_by(Workflow.name))
    ]


@app.post("/workflows/{name}/start", status_code=202)
def start_workflow(name: str, body: StartIn, session: DB):
    try:
        run = enqueue_run(session, name, body.dataset_id, body.params)
    except RunError as e:
        raise HTTPException(400, str(e)) from None
    return {
        "run_id": run.id,
        "status": run.status,
        "status_url": f"/runs/{run.id}/status",
        "content_url": f"/runs/{run.id}/content",
    }


@app.get("/runs/{run_id}/status")
def run_status(run_id: uuid.UUID, session: DB):
    run = _run(session, run_id)
    actions = session.scalars(
        select(ActionRun).where(ActionRun.run_id == run_id).order_by(ActionRun.position)
    ).all()
    return {
        "run_id": run.id,
        "workflow": run.workflow_name,
        "dataset_id": run.dataset_id,
        "status": run.status,
        "created_at": run.created_at,
        "started_at": run.started_at,
        "finished_at": run.finished_at,
        "error": run.error,
        "actions": [
            {
                "name": a.name,
                "status": a.status,
                "seconds": _seconds(a.started_at, a.finished_at),
                "error": a.error,
            }
            for a in actions
        ],
    }


@app.get("/runs/{run_id}/content")
def run_content(run_id: uuid.UUID, session: DB):
    run = _run(session, run_id)
    if run.status in ("queued", "running"):
        raise HTTPException(409, f"run is {run.status}; check /runs/{run_id}/status")
    return {"run_id": run.id, "status": run.status, "error": run.error, "output": run.output}


def _run(session: Session, run_id: uuid.UUID) -> WorkflowRun:
    run = session.get(WorkflowRun, run_id)
    if run is None:
        raise HTTPException(404, f"no run {run_id}")
    return run


def _seconds(start: datetime | None, end: datetime | None) -> float | None:
    return round((end - start).total_seconds(), 1) if start and end else None


def serve() -> None:
    """`uv run serve`: the API on http://127.0.0.1:8000 (docs at /docs)."""
    import uvicorn

    uvicorn.run("promo_rag.platform.api:app", host="127.0.0.1", port=8000)
