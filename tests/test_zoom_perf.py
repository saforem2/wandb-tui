"""Tests for the full-screen zoom view's redraw cost.

The zoom view drew at full resolution while the chart tiles downsampled to the
canvas width. That was a safe exemption at HISTORY_RUN_LIMIT=24, but the cap
was later raised to 100 without revisiting it: every pan/zoom/focus keypress
triggers a synchronous redraw, so the cost stacked into a visible freeze.
"""

from __future__ import annotations

import asyncio
import json
import time

import pytest

import wandb_tui as w


class CountingPlt:
    """Records how many points each series was drawn with."""

    def __init__(self):
        self.point_counts: list[int] = []

    def __getattr__(self, name):
        return lambda *a, **k: None

    def plot(self, xs, ys, **kw):
        self.point_counts.append(len(ys))


def dense_metric(runs: int = 40, points: int = 5000) -> dict:
    return {
        "name": "loss",
        "runs": [
            {"values": [float(i) + (j % 97) * 0.01 for j in range(points)], "axes": {}}
            for i in range(runs)
        ],
    }


def test_draw_respects_the_point_budget():
    plt = CountingPlt()
    metric = dense_metric(runs=4, points=5000)
    w.draw_metric_plot(plt, metric, 4, [f"R{i+1}" for i in range(4)], max_points=320)
    assert plt.point_counts, "nothing was drawn"
    assert max(plt.point_counts) <= 320, plt.point_counts


def test_short_series_skip_downsampling(patched):
    """Re-binning a series that already fits costs more than plotting it."""
    seen: list[int | None] = []
    real = w.draw_metric_plot

    async def main():
        import wandb_tui

        app = w.make_project_app("e/p", 6, 0)
        async with app.run_test(size=(160, 45)) as pilot:
            await loaded(app, pilot)
            await pilot.press("m")
            for _ in range(20):
                await pilot.pause()

            def spy(*args, **kwargs):
                seen.append(kwargs.get("max_points"))
                return real(*args, **kwargs)

            wandb_tui.draw_metric_plot = spy
            try:
                # Rebind the metric to a short series and redraw.
                metric = app.metric_by_name("loss")
                for slot in metric["runs"]:
                    if slot:
                        slot["values"] = slot["values"][:50]
                app.open_chart_fullscreen("loss")
                for _ in range(20):
                    await pilot.pause()
            finally:
                wandb_tui.draw_metric_plot = real
            assert seen, "zoom never drew"
            assert seen[0] is None, f"short series should skip the budget: {seen}"
            app.exit()

    asyncio.run(main())


def test_uncapped_draw_keeps_every_point():
    """Guards the comparison: without a budget, all 5000 points are drawn."""
    plt = CountingPlt()
    metric = dense_metric(runs=2, points=5000)
    w.draw_metric_plot(plt, metric, 2, ["R1", "R2"])
    assert max(plt.point_counts) == 5000


def test_downsampling_is_substantially_faster():
    """The actual complaint: redraw latency at the 100-run cap."""
    import plotext as plt

    metric = dense_metric(runs=w.HISTORY_RUN_LIMIT, points=5000)
    labels = [f"R{i+1}" for i in range(w.HISTORY_RUN_LIMIT)]

    def timed(max_points):
        plt.clf()
        plt.plotsize(160, 40)
        start = time.perf_counter()
        w.draw_metric_plot(plt, metric, len(metric["runs"]), labels, max_points=max_points)
        plt.build()
        return time.perf_counter() - start

    capped = timed(320)
    full = timed(None)
    assert capped < full, f"capped {capped:.2f}s not faster than full {full:.2f}s"
    # A redraw fires per keypress, so it has to stay interactive.
    assert capped < 1.0, f"capped redraw still slow: {capped:.2f}s"


# --- the zoom screen actually passes a budget --------------------------------


def mkrun(i: int, points: int = 400) -> dict:
    return {
        "name": f"id{i}",
        "displayName": f"run-{i}",
        "state": "finished",
        "createdAt": "2026-08-01T00:00:00Z",
        "updatedAt": "2026-08-01T01:00:00Z",
        "history": [json.dumps({"_step": s, "loss": float(i) + s}) for s in range(points)],
        "config": "{}",
        "summaryMetrics": "{}",
    }


@pytest.fixture
def patched(monkeypatch):
    # Dense enough to exceed the budget: the zoom view deliberately skips
    # downsampling when a series already fits, since re-binning it costs more
    # than plotting it.
    runs = [mkrun(i, points=3000) for i in range(6)]
    monkeypatch.setattr(w, "fetch_project_run_names", lambda e, p, limit: list(runs))
    monkeypatch.setattr(w, "fetch_histories", lambda kept, **kw: list(kept))
    return runs


async def loaded(app, pilot, tries: int = 60):
    for _ in range(tries):
        await pilot.pause()
        if app.runs:
            return True
    raise AssertionError(f"runs never loaded (status={app.status!r})")


def test_zoom_screen_downsamples(patched, monkeypatch):
    """End-to-end: opening zoom must not draw at full resolution."""
    seen: list[int | None] = []
    real = w.draw_metric_plot

    def spy(*args, **kwargs):
        seen.append(kwargs.get("max_points"))
        return real(*args, **kwargs)

    monkeypatch.setattr(w, "draw_metric_plot", spy)

    async def main():
        app = w.make_project_app("e/p", 6, 0)
        async with app.run_test(size=(160, 45)) as pilot:
            await loaded(app, pilot)
            await pilot.press("m")  # chart mode
            for _ in range(20):
                await pilot.pause()
            seen.clear()
            app.open_chart_fullscreen("loss")
            for _ in range(20):
                await pilot.pause()
            assert seen, "zoom never drew"
            assert all(v is not None for v in seen), (
                f"zoom drew dense series without a point budget: {seen}"
            )
            app.exit()

    asyncio.run(main())


def test_zoom_keypress_stays_responsive(patched):
    """A pan keypress must not block the loop for a noticeable time."""

    async def main():
        app = w.make_project_app("e/p", 6, 0)
        async with app.run_test(size=(160, 45)) as pilot:
            await loaded(app, pilot)
            await pilot.press("m")
            for _ in range(20):
                await pilot.pause()
            app.open_chart_fullscreen("loss")
            for _ in range(20):
                await pilot.pause()
            start = time.perf_counter()
            await pilot.press("h")  # pan left
            await pilot.pause()
            elapsed = time.perf_counter() - start
            assert elapsed < 1.0, f"pan keypress took {elapsed:.2f}s"
            app.exit()

    asyncio.run(main())
