"""Runbooks: each workflow's name, input and actions in order.

`workflows/<name>/runbook.yaml` is the source of truth. When the app starts, every runbook is
checked (each action must exist in that workflow's ACTIONS) and synced into
platform.workflows, so the database always knows which workflows exist and what they run.
"""

import hashlib
import importlib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from promo_rag.db.models import Workflow

WORKFLOWS_DIR = Path(__file__).resolve().parents[1] / "workflows"


class RunbookError(ValueError):
    pass


class ActionFailed(RuntimeError):
    """An action decided it can't succeed; its message becomes the run's error."""


class RunbookAction(BaseModel):
    name: str
    description: str = ""


class Runbook(BaseModel):
    name: str
    version: int
    description: str = ""
    input: dict[str, str] = {}  # e.g. {"dataset": "required"}
    params: dict[str, Any] = {}  # defaults; a run can override them
    actions: list[RunbookAction]


@dataclass
class ActionContext:
    """What an action gets: the database, the run's input, and earlier actions' outputs."""

    session: Session
    run_id: str
    dataset_id: str | None
    params: dict[str, Any]
    outputs: dict[str, dict] = field(default_factory=dict)  # action name → its output
    log: Callable[[str], None] = print


Action = Callable[[ActionContext], dict]


def load_runbook(name: str) -> tuple[Runbook, str]:
    """The workflow's runbook and the sha256 of its file."""
    path = WORKFLOWS_DIR / name / "runbook.yaml"
    raw = path.read_bytes()
    runbook = Runbook.model_validate(yaml.safe_load(raw))
    if runbook.name != name:
        raise RunbookError(f"{path}: name is {runbook.name!r}, but the folder is {name!r}")
    missing = [a.name for a in runbook.actions if a.name not in workflow_actions(name)]
    if missing:
        raise RunbookError(f"{name}: the runbook lists actions that don't exist: {missing}")
    return runbook, hashlib.sha256(raw).hexdigest()


def workflow_actions(name: str) -> dict[str, Action]:
    """The workflow's ACTIONS, from workflows/<name>/actions/__init__.py."""
    module = importlib.import_module(f"promo_rag.workflows.{name}.actions")
    return module.ACTIONS


def workflow_names() -> list[str]:
    return sorted(p.parent.name for p in WORKFLOWS_DIR.glob("*/runbook.yaml"))


def sync_workflows(session: Session) -> list[str]:
    """Upsert every runbook into platform.workflows. Returns the names synced."""
    names = workflow_names()
    for name in names:
        runbook, digest = load_runbook(name)
        values = {
            "name": name,
            "version": runbook.version,
            "description": runbook.description,
            "definition": runbook.model_dump(),
            "runbook_hash": digest,
        }
        session.execute(
            insert(Workflow)
            .values(**values)
            .on_conflict_do_update(index_elements=["name"], set_=values | {"synced_at": func.now()})
        )
    session.commit()
    return names
