"""Tests for per-run chart visibility toggled from the group tree.

A leading gutter column shows an eye marker per row, like the W&B workspace.
Toggling hides that run's series from the charts ONLY -- the table keeps every
run so its numbers stay readable while it is out of the plot.
"""

from __future__ import annotations

import asyncio
import json

import pytest

import wandb_tui as w


def mkrun(i: int, config: dict | None = None) -> dict:
    return {
        "name": f"id{i}",
        "displayName": f"run-{i}",
        "state": "finished",
        "createdAt": "2026-08-01T00:00:00Z",
        "updatedAt": "2026-08-01T01:00:00Z",
        "history": [json.dumps({"_step": s, "loss": float(i) + s}) for s in range(4)],
        "config": json.dumps({k: {"value": v} for k, v in (config or {}).items()}),
        "summaryMetrics": "{}",
    }


async def loaded(app, pilot, tries: int = 60):
    for _ in range(tries):
        await pilot.pause()
        if app.runs:
            return True
    raise AssertionError(f"runs never loaded (status={app.status!r})")


@pytest.fixture
def patched(monkeypatch):
    runs = [mkrun(i, {"ws": 3072 if i < 2 else 6144}) for i in range(4)]
    monkeypatch.setattr(w, "fetch_project_run_names", lambda e, p, limit: list(runs))
    monkeypatch.setattr(w, "fetch_histories", lambda kept, **kw: list(kept))
    return runs


# --- state -------------------------------------------------------------------


def test_all_runs_visible_by_default(patched):
    async def main():
        app = w.make_project_app("e/p", 4, 0)
        async with app.run_test(size=(160, 40)) as pilot:
            await loaded(app, pilot)
            assert app.hidden_runs == set()
            assert app.run_visible(0)
            app.exit()

    asyncio.run(main())


def test_space_toggles_the_run_under_the_cursor(patched):
    async def main():
        app = w.make_project_app("e/p", 4, 0)
        async with app.run_test(size=(160, 40)) as pilot:
            await loaded(app, pilot)
            app.set_group_keys("ws")
            await pilot.pause()
            table = app.query_one("#table")
            table.focus()
            # Row 0 is a group header; find the first leaf.
            leaf = next(i for i, r in enumerate(app.tree_rows) if not r.is_group)
            table.move_cursor(row=leaf)
            idx = app.tree_rows[leaf].run_index
            await pilot.press("space")
            await pilot.pause()
            assert not app.run_visible(idx)
            await pilot.press("space")
            await pilot.pause()
            assert app.run_visible(idx)
            app.exit()

    asyncio.run(main())


def test_toggling_a_group_hides_every_run_beneath_it(patched):
    """The 100-run case: one keypress on a group, not 95 keypresses."""

    async def main():
        app = w.make_project_app("e/p", 4, 0)
        async with app.run_test(size=(160, 40)) as pilot:
            await loaded(app, pilot)
            app.set_group_keys("ws")
            await pilot.pause()
            table = app.query_one("#table")
            table.focus()
            table.move_cursor(row=0)  # first group header
            assert app.tree_rows[0].is_group
            await pilot.press("space")
            await pilot.pause()
            # Both ws=3072 runs (indices 0 and 1) are hidden.
            assert not app.run_visible(0)
            assert not app.run_visible(1)
            # The other group is untouched.
            assert app.run_visible(2)
            app.exit()

    asyncio.run(main())


def test_toggling_a_group_again_shows_them_all(patched):
    async def main():
        app = w.make_project_app("e/p", 4, 0)
        async with app.run_test(size=(160, 40)) as pilot:
            await loaded(app, pilot)
            app.set_group_keys("ws")
            await pilot.pause()
            table = app.query_one("#table")
            table.focus()
            table.move_cursor(row=0)
            await pilot.press("space")
            await pilot.pause()
            await pilot.press("space")
            await pilot.pause()
            assert app.hidden_runs == set()
            app.exit()

    asyncio.run(main())


# --- gutter rendering --------------------------------------------------------


def test_gutter_column_is_first(patched):
    async def main():
        app = w.make_project_app("e/p", 4, 0)
        async with app.run_test(size=(160, 40)) as pilot:
            await loaded(app, pilot)
            app.set_group_keys("ws")
            await pilot.pause()
            table = app.query_one("#table")
            first = str(list(table.columns.values())[0].label)
            assert first.strip() in ("", "◉"), f"unexpected first column {first!r}"
            app.exit()

    asyncio.run(main())


def test_gutter_marker_reflects_hidden_state(patched):
    async def main():
        app = w.make_project_app("e/p", 4, 0)
        async with app.run_test(size=(160, 40)) as pilot:
            await loaded(app, pilot)
            app.set_group_keys("ws")
            await pilot.pause()
            table = app.query_one("#table")
            table.focus()
            leaf = next(i for i, r in enumerate(app.tree_rows) if not r.is_group)
            table.move_cursor(row=leaf)
            before = str(table.get_row_at(leaf)[0])
            await pilot.press("space")
            await pilot.pause()
            after = str(table.get_row_at(leaf)[0])
            assert before != after, "gutter marker must change when toggled"
            app.exit()

    asyncio.run(main())


# --- effect on charts vs table ----------------------------------------------


def test_hidden_runs_are_dropped_from_charts(patched):
    async def main():
        app = w.make_project_app("e/p", 4, 0)
        async with app.run_test(size=(160, 40)) as pilot:
            await loaded(app, pilot)
            app.hidden_runs = {1}
            visible = app.visible_run_indices()
            assert 1 not in visible
            assert visible == [0, 2, 3]
            app.exit()

    asyncio.run(main())


def test_hidden_runs_stay_in_the_table(patched):
    """Charts-only scope: the row remains so its numbers stay readable."""

    async def main():
        app = w.make_project_app("e/p", 4, 0)
        async with app.run_test(size=(160, 40)) as pilot:
            await loaded(app, pilot)
            app.set_group_keys("ws")
            await pilot.pause()
            before = app.query_one("#table").row_count
            app.hidden_runs = {0}
            app.render_table()
            await pilot.pause()
            assert app.query_one("#table").row_count == before
            app.exit()

    asyncio.run(main())


def test_hiding_every_run_does_not_crash_chart_mode(patched):
    async def main():
        app = w.make_project_app("e/p", 4, 0)
        async with app.run_test(size=(160, 40)) as pilot:
            await loaded(app, pilot)
            app.hidden_runs = {0, 1, 2, 3}
            await pilot.press("m")
            await pilot.pause()
            assert app.is_running
            app.exit()

    asyncio.run(main())


def test_hidden_runs_are_not_drawn(monkeypatch, patched):
    """The end-to-end claim: a hidden run produces no series in the figure."""
    metric = {
        "name": "loss",
        "runs": [
            {"values": [1.0, 2.0, 3.0], "axes": {}},
            {"values": [4.0, 5.0, 6.0], "axes": {}},
        ],
    }

    class FakePlt:
        def __init__(self):
            self.series = []

        def plot(self, xs, ys, **kw):
            self.series.append(kw.get("label"))

        def __getattr__(self, name):
            # plotext's figure API is broad; only plot() matters here.
            return lambda *a, **k: None

        def clear_data(self, *a, **k):
            pass

        def title(self, *a, **k):
            pass

        def ylabel(self, *a, **k):
            pass

    plt = FakePlt()
    drawn = w.draw_metric_plot(plt, metric, 2, ["R1", "R2"], hidden={1})
    assert drawn == 1
    assert plt.series == ["R1"], plt.series


def test_visibility_survives_a_refresh(patched):
    async def main():
        app = w.make_project_app("e/p", 4, 0)
        async with app.run_test(size=(160, 40)) as pilot:
            await loaded(app, pilot)
            app.hidden_runs = {2}
            app.action_refresh_data()
            for _ in range(60):
                await pilot.pause()
                if not app.refresh_in_flight:
                    break
            assert 2 in app.hidden_runs
            app.exit()

    asyncio.run(main())


def test_space_in_a_text_box_stays_literal(patched):
    async def main():
        app = w.make_project_app("e/p", 4, 0)
        async with app.run_test(size=(160, 40)) as pilot:
            await loaded(app, pilot)
            await pilot.press("slash")
            await pilot.pause()
            await pilot.press("a")
            await pilot.press("space")
            await pilot.press("b")
            await pilot.pause()
            assert app.query_one("#search_input").value == "a b"
            assert app.hidden_runs == set()
            app.exit()

    asyncio.run(main())
