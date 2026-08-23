"""Tests for cycling the plotext marker at runtime.

The marker was fixed at import time from WANDB_TUI_MARKER, so comparing
"which of these renders my data best" meant restarting with a different env
var. `M` cycles it live.
"""

from __future__ import annotations

import asyncio
import json

import pytest

import wandb_tui as w


def mkrun(i: int) -> dict:
    return {
        "name": f"id{i}",
        "displayName": f"run-{i}",
        "state": "finished",
        "createdAt": "2026-08-01T00:00:00Z",
        "updatedAt": "2026-08-01T01:00:00Z",
        "history": [json.dumps({"_step": s, "loss": float(i) + s}) for s in range(20)],
        "config": "{}",
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
    runs = [mkrun(i) for i in range(3)]
    monkeypatch.setattr(w, "fetch_project_run_names", lambda e, p, limit: list(runs))
    monkeypatch.setattr(w, "fetch_histories", lambda kept, **kw: list(kept))
    return runs


# --- the catalogue -----------------------------------------------------------


def test_hd_is_the_default_and_comes_first():
    assert w.CHART_MARKERS[0] == "hd"


def test_catalogue_holds_only_markers_plotext_accepts():
    import plotext as plt

    for marker in w.CHART_MARKERS:
        plt.clf()
        plt.plotsize(40, 10)
        # Raises on an unknown marker name.
        plt.plot([0, 1, 2], [0.0, 1.0, 0.5], marker=marker)
        plt.build()


def test_env_var_still_selects_the_starting_marker(monkeypatch):
    monkeypatch.setenv("WANDB_TUI_MARKER", "braille")
    assert w.initial_marker() == "braille"


def test_unknown_env_marker_falls_back_to_default(monkeypatch):
    """A typo must not crash plotext deep inside a render."""
    monkeypatch.setenv("WANDB_TUI_MARKER", "nonsense")
    assert w.initial_marker() == w.CHART_MARKERS[0]


# --- cycling -----------------------------------------------------------------


def test_M_cycles_the_marker(patched):
    async def main():
        app = w.make_project_app("e/p", 3, 0)
        async with app.run_test(size=(140, 40)) as pilot:
            await loaded(app, pilot)
            first = app.chart_marker()
            await pilot.press("M")
            await pilot.pause()
            assert app.chart_marker() != first
            app.exit()

    asyncio.run(main())


def test_cycling_wraps_back_around(patched):
    async def main():
        app = w.make_project_app("e/p", 3, 0)
        async with app.run_test(size=(140, 40)) as pilot:
            await loaded(app, pilot)
            first = app.chart_marker()
            for _ in range(len(w.CHART_MARKERS)):
                await pilot.press("M")
                await pilot.pause()
            assert app.chart_marker() == first
            app.exit()

    asyncio.run(main())


def test_every_marker_renders_without_crashing(patched):
    async def main():
        app = w.make_project_app("e/p", 3, 0)
        async with app.run_test(size=(140, 40)) as pilot:
            await loaded(app, pilot)
            await pilot.press("m")  # chart mode
            await pilot.pause()
            for _ in range(len(w.CHART_MARKERS)):
                await pilot.press("M")
                await pilot.pause()
                assert app.is_running
            app.exit()

    asyncio.run(main())


def test_the_marker_reaches_the_plot(patched):
    """State changing is not enough -- the draw call must use it."""
    seen: list[str] = []
    real = w.draw_metric_plot

    async def main():
        import wandb_tui

        app = w.make_project_app("e/p", 3, 0)
        async with app.run_test(size=(140, 40)) as pilot:
            await loaded(app, pilot)
            await pilot.press("m")
            for _ in range(10):
                await pilot.pause()
            await pilot.press("M")
            await pilot.pause()
            chosen = app.chart_marker()

            def spy(*args, **kwargs):
                seen.append(kwargs.get("marker"))
                return real(*args, **kwargs)

            wandb_tui.draw_metric_plot = spy
            try:
                app.open_chart_fullscreen("loss")
                for _ in range(20):
                    await pilot.pause()
            finally:
                wandb_tui.draw_metric_plot = real
            assert seen, "nothing drew"
            assert seen[0] == chosen, f"drew with {seen[0]!r}, expected {chosen!r}"
            app.exit()

    asyncio.run(main())


def test_M_typed_in_search_stays_literal(patched):
    async def main():
        app = w.make_project_app("e/p", 3, 0)
        async with app.run_test(size=(140, 40)) as pilot:
            await loaded(app, pilot)
            before = app.chart_marker()
            await pilot.press("slash")
            await pilot.pause()
            await pilot.press("M")
            await pilot.pause()
            assert app.query_one("#search_input").value == "M"
            assert app.chart_marker() == before
            app.exit()

    asyncio.run(main())
