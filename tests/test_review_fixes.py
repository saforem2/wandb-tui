"""Regression tests for the PR review comments.

Three findings from the automated reviewers survived the Textual rewrite:
a misleading fetch error, SystemExit being wrapped by a bare `except
Exception`, and the startup picker giving no explanation when there is
nothing to pick.
"""

from __future__ import annotations

import pytest

import wandb_tui as w


# --- misleading "no project" error -------------------------------------------


def test_missing_project_error_names_the_likely_causes(monkeypatch):
    """The old text blamed a missing `requests` install for every case.

    A null project is far more often a typo or a private project you cannot
    see, so the message should say that rather than send people to pip.
    """

    def fake_graphql(query, variables, **kw):
        return {"project": None}

    monkeypatch.setattr(w, "graphql", fake_graphql)
    with pytest.raises(RuntimeError) as excinfo:
        w.fetch_run("ent", "proj", "rid")
    msg = str(excinfo.value)
    assert "ent/proj" in msg
    low = msg.lower()
    assert "spelling" in low or "not exist" in low, msg
    assert "wandb_api_key" in low, msg
    # The old text sent people to `pip install requests`, which is almost
    # never the actual cause.
    assert "install" not in low, msg


# --- SystemExit must not be swallowed ----------------------------------------


def test_picker_systemexit_propagates_unchanged(monkeypatch):
    """require_textual() raises SystemExit with install instructions.

    Wrapping it in "Could not open picker: ..." buried the actionable part.
    SystemExit is not an Exception subclass, so a bare `except Exception`
    would not have caught it anyway -- but the handler must stay explicit.
    """
    original = SystemExit("Textual is required for interactive mode.")

    def boom():
        raise original

    monkeypatch.setattr(w, "startup_picker_textual", boom)
    monkeypatch.setattr("sys.argv", ["wandb-tui"])
    with pytest.raises(SystemExit) as excinfo:
        w.main()
    # The original guidance survives rather than being re-wrapped.
    assert "Textual is required" in str(excinfo.value)


def test_picker_other_errors_still_explain_themselves(monkeypatch):
    """A network failure should still be reported, not raised raw."""

    def boom():
        raise RuntimeError("connection refused")

    monkeypatch.setattr(w, "startup_picker_textual", boom)
    monkeypatch.setattr("sys.argv", ["wandb-tui"])
    with pytest.raises(SystemExit) as excinfo:
        w.main()
    assert "connection refused" in str(excinfo.value)


# --- empty picker ------------------------------------------------------------


def test_picker_explains_when_there_are_no_entities(monkeypatch):
    """An empty table with no message looks like the app is broken."""
    monkeypatch.setattr(w, "fetch_viewer_entities", lambda: [])
    with pytest.raises(SystemExit) as excinfo:
        w.startup_picker_textual()
    msg = str(excinfo.value).lower()
    assert "wandb_api_key" in msg or "no " in msg, excinfo.value


def test_picker_explains_when_a_project_list_is_empty(monkeypatch):
    monkeypatch.setattr(w, "fetch_viewer_entities", lambda: [{"name": "ent"}])
    monkeypatch.setattr(w, "fetch_entity_projects", lambda e: [])
    monkeypatch.setattr(
        w, "choose_from_table", lambda *a, **k: {"name": "ent"}
    )
    with pytest.raises(SystemExit) as excinfo:
        w.startup_picker_textual()
    assert "ent" in str(excinfo.value)
