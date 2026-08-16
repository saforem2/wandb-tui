"""Tests for selectable chart x-axes.

W&B lets you plot against Step, Wall Time, Relative Time (Wall), Relative
Time (Process), or a metric like n_tokens_seen. Charts here previously always
used the sample index, which silently misrepresents runs that log at uneven
intervals or start mid-experiment.
"""

from __future__ import annotations

import asyncio
import json

import pytest

import wandb_tui as w


def mkrun(i: int = 0, rows: list[dict] | None = None) -> dict:
    if rows is None:
        rows = [
            {
                "_step": s,
                "_runtime": 10.0 * s,
                "_timestamp": 1_700_000_000 + 10 * s,
                "train/tokens_seen": 1000.0 * s,
                "loss": 1.0 / (s + 1),
            }
            for s in range(5)
        ]
    return {
        "name": f"id{i}",
        "displayName": f"run-{i}",
        "state": "finished",
        "createdAt": "2026-08-01T00:00:00Z",
        "updatedAt": "2026-08-01T01:00:00Z",
        "history": [json.dumps(r) for r in rows],
        "config": "{}",
        "summaryMetrics": "{}",
    }


# --- axis catalogue ----------------------------------------------------------


def test_the_five_requested_axes_exist():
    ids = {a.id for a in w.X_AXES}
    assert {"step", "wall", "relative_wall", "relative_process", "tokens"} <= ids


def test_step_is_the_default_axis():
    assert w.X_AXES[0].id == "step"


# --- per-point capture -------------------------------------------------------


def test_metric_carries_axis_values():
    m = next(m for m in w.build_metrics(mkrun()) if m["name"] == "loss")
    assert m["axes"]["step"] == [0, 1, 2, 3, 4]
    assert m["axes"]["relative_process"] == [0.0, 10.0, 20.0, 30.0, 40.0]


def test_axis_values_are_index_aligned_with_values():
    """The trap: `values` drops rows missing the key AND non-numeric ones.

    An x list built by a separate pass over all rows would be longer than y,
    so point k would pair the wrong x with the wrong y.
    """
    rows = [
        {"_step": 0, "_runtime": 0.0, "loss": 1.0},
        {"_step": 1, "_runtime": 10.0},                 # no loss at all
        {"_step": 2, "_runtime": 20.0, "loss": "NaN-ish"},  # non-numeric
        {"_step": 3, "_runtime": 30.0, "loss": 0.5},
    ]
    m = next(m for m in w.build_metrics(mkrun(rows=rows)) if m["name"] == "loss")
    assert m["values"] == [1.0, 0.5]
    assert m["axes"]["step"] == [0, 3]
    assert m["axes"]["relative_process"] == [0.0, 30.0]


def test_relative_wall_starts_at_zero():
    m = next(m for m in w.build_metrics(mkrun()) if m["name"] == "loss")
    rel = m["axes"]["relative_wall"]
    assert rel[0] == 0
    assert rel[-1] == 40  # 5 points, 10s apart


def test_wall_axis_keeps_absolute_timestamps():
    m = next(m for m in w.build_metrics(mkrun()) if m["name"] == "loss")
    assert m["axes"]["wall"][0] == 1_700_000_000


def test_token_axis_is_captured_and_monotonic():
    m = next(m for m in w.build_metrics(mkrun()) if m["name"] == "loss")
    tokens = m["axes"]["tokens"]
    assert tokens == sorted(tokens)
    assert tokens[-1] == 4000.0


def test_missing_axis_is_absent_rather_than_zeros():
    """A run with no token metric must not claim a token axis of zeros."""
    rows = [{"_step": s, "_runtime": float(s), "loss": float(s)} for s in range(3)]
    m = next(m for m in w.build_metrics(mkrun(rows=rows)) if m["name"] == "loss")
    assert "tokens" not in m["axes"]
    assert m["axes"]["step"] == [0, 1, 2]


# --- resolution --------------------------------------------------------------


def test_axis_series_falls_back_to_sample_index():
    m = {"values": [1.0, 2.0, 3.0], "axes": {}}
    assert w.axis_series(m, "tokens", 3) == [0, 1, 2]


def test_axis_series_returns_the_axis_when_present():
    m = {"values": [1.0, 2.0], "axes": {"step": [7, 9]}}
    assert w.axis_series(m, "step", 2) == [7, 9]


def test_axis_series_falls_back_when_length_disagrees():
    """A short/long axis would pair x and y wrongly -- prefer the index."""
    m = {"values": [1.0, 2.0, 3.0], "axes": {"step": [0, 1]}}
    assert w.axis_series(m, "step", 3) == [0, 1, 2]


# --- app integration ---------------------------------------------------------


async def loaded(app, pilot, tries: int = 60):
    for _ in range(tries):
        await pilot.pause()
        if app.runs:
            return True
    raise AssertionError(f"runs never loaded (status={app.status!r})")


@pytest.fixture
def patched(monkeypatch):
    runs = [mkrun(i) for i in range(2)]
    monkeypatch.setattr(w, "fetch_project_run_names", lambda e, p, limit: list(runs))
    monkeypatch.setattr(w, "fetch_histories", lambda kept, **kw: list(kept))
    return runs


def test_X_cycles_the_axis(patched):
    async def main():
        app = w.make_project_app("e/p", 2, 0)
        async with app.run_test(size=(150, 40)) as pilot:
            await loaded(app, pilot)
            assert app.x_axis().id == "step"
            await pilot.press("X")
            await pilot.pause()
            assert app.x_axis().id != "step"
            app.exit()

    asyncio.run(main())


def test_axis_choice_survives_chart_mode_toggle(patched):
    async def main():
        app = w.make_project_app("e/p", 2, 0)
        async with app.run_test(size=(150, 40)) as pilot:
            await loaded(app, pilot)
            await pilot.press("X")
            await pilot.pause()
            chosen = app.x_axis().id
            await pilot.press("m")
            await pilot.pause()
            await pilot.press("m")
            await pilot.pause()
            assert app.x_axis().id == chosen
            app.exit()

    asyncio.run(main())


def test_X_typed_in_search_stays_literal(patched):
    async def main():
        app = w.make_project_app("e/p", 2, 0)
        async with app.run_test(size=(150, 40)) as pilot:
            await loaded(app, pilot)
            await pilot.press("slash")
            await pilot.pause()
            await pilot.press("X")
            await pilot.pause()
            assert app.query_one("#search_input").value == "X"
            assert app.x_axis().id == "step"
            app.exit()

    asyncio.run(main())


def test_axis_availability_sees_per_run_slots(patched):
    """Live-data bug: every axis reported "not logged" in the project view.

    Multi-run metrics keep axes inside per-run slots, not on the metric dict,
    so a top-level-only check warned about axes the charts were visibly using.
    """

    async def main():
        app = w.make_project_app("e/p", 2, 0)
        async with app.run_test(size=(150, 40)) as pilot:
            await loaded(app, pilot)
            assert app.axis_available("step")
            assert app.axis_available("relative_process")
            assert not app.axis_available("definitely_not_an_axis")
            app.exit()

    asyncio.run(main())


def test_chart_mode_renders_with_a_non_index_axis(patched):
    """Switching axis must not crash the chart path."""

    async def main():
        app = w.make_project_app("e/p", 2, 0)
        async with app.run_test(size=(150, 40)) as pilot:
            await loaded(app, pilot)
            await pilot.press("m")
            await pilot.pause()
            for _ in range(len(w.X_AXES)):
                await pilot.press("X")
                await pilot.pause()
                assert app.is_running
            app.exit()

    asyncio.run(main())
