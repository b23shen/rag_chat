"""The workflow engine: runbooks and the action runner (no database)."""

import pytest
from pydantic import ValidationError

from promo_rag.platform import runbook as runbook_module
from promo_rag.platform.runbook import (
    ActionContext,
    ActionFailed,
    RunbookError,
    load_runbook,
    workflow_actions,
    workflow_names,
)
from promo_rag.platform.runner import run_actions
from promo_rag.workflows.ingestion.actions import INDEX_PARAMS
from promo_rag.workflows.ingestion.index_config import IndexConfig

# ---------------------------------------------------------------- runbooks


def test_every_workflow_runbook_matches_its_actions():
    assert "ingestion" in workflow_names()
    for name in workflow_names():
        runbook, digest = load_runbook(name)  # raises if an action is missing
        assert runbook.name == name and len(digest) == 64


def test_ingestion_runbook_order_and_params():
    runbook, _ = load_runbook("ingestion")
    assert [a.name for a in runbook.actions] == ["parse", "extract", "chunk", "embed", "activate"]
    assert runbook.input == {"dataset": "required"}
    # every runbook param is an index setting the chunk action knows how to apply
    assert set(runbook.params) <= set(INDEX_PARAMS) <= set(IndexConfig.model_fields)


def test_a_runbook_naming_a_missing_action_is_rejected(monkeypatch):
    real = workflow_actions("ingestion")
    monkeypatch.setattr(
        runbook_module,
        "workflow_actions",
        lambda name: {k: v for k, v in real.items() if k != "embed"},
    )
    with pytest.raises(RunbookError, match="embed"):
        load_runbook("ingestion")


def test_a_malformed_runbook_is_rejected():
    with pytest.raises(ValidationError):
        runbook_module.Runbook.model_validate({"name": "x", "version": "one", "actions": []})


# ---------------------------------------------------------------- the runner


class Recorder:
    def __init__(self):
        self.events = []

    def __call__(self, name, position, status, output, error):
        self.events.append((name, status, error))


def _ctx():
    return ActionContext(session=None, run_id="r1", dataset_id="d1", params={"chunk_tokens": 256})


def test_actions_run_in_order_and_see_earlier_outputs():
    def chunk(ctx):
        return {"corpus_version_id": "v1", "chunk_tokens": ctx.params["chunk_tokens"]}

    def embed(ctx):
        return {"embedded_version": ctx.outputs["chunk"]["corpus_version_id"]}

    record, ctx = Recorder(), _ctx()
    ok = run_actions([("chunk", chunk), ("embed", embed)], ctx, record)
    assert ok
    assert [(n, s) for n, s, _ in record.events] == [
        ("chunk", "running"),
        ("chunk", "succeeded"),
        ("embed", "running"),
        ("embed", "succeeded"),
    ]
    assert ctx.outputs == {
        "chunk": {"corpus_version_id": "v1", "chunk_tokens": 256},
        "embed": {"embedded_version": "v1"},
    }


def test_the_first_failure_stops_the_run_and_skips_the_rest():
    called = []

    def parse(ctx):
        called.append("parse")
        return {}

    def embed(ctx):
        raise ActionFailed("not activated: 3 chunks have no embedding")

    def activate(ctx):
        called.append("activate")
        return {}

    record = Recorder()
    ok = run_actions([("parse", parse), ("embed", embed), ("activate", activate)], _ctx(), record)
    assert not ok
    assert called == ["parse"]  # activate never ran
    assert [(n, s) for n, s, _ in record.events][-2:] == [
        ("embed", "failed"),
        ("activate", "skipped"),
    ]
    error = next(e for n, s, e in record.events if s == "failed")
    assert "3 chunks have no embedding" in error
