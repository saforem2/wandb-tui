"""Tests for chart axis limits, log scales, outlier hiding, and focus cycling.

Charts were autoscaled-linear over every point, so a single loss spike flattened
the interesting range and a metric spanning orders of magnitude was unreadable.
These pin the controls that fix that, plus two pre-existing `z` bugs that live
in the same code.
"""

from __future__ import annotations

import asyncio
import json

import pytest

import wandb_tui as w


def mkrun(i: int, points: int = 200, spike: bool = False) -> dict:
    rows = []
    for s in range(points):
        loss = 10.0 - s * 0.01
        if spike and s == points // 2:
            loss = 500.0  # the outlier that squashes the chart
        rows.append({
            "_step": s,
            "train/tokens_seen": s * 1_000_000,  # a NON-index x axis
            "loss": loss,
        })
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


class CountingPlt:
    """Captures what actually reached plotext."""

    def __init__(self):
        self.series: list[tuple[list, list]] = []
        self.xlabel_text = None
        self.ylabel_text = None

    def __getattr__(self, name):
        return lambda *a, **k: None

    def plot(self, xs, ys, **kw):
        self.series.append((list(xs), list(ys)))

    def xlabel(self, text):
        self.xlabel_text = text

    def ylabel(self, text):
        self.ylabel_text = text


def metric_from(runs: list[dict], name: str = "loss") -> dict:
    return next(m for m in w.build_multi_metrics(runs) if m["name"] == name)


# --- axis-aware extents: the `z` bug ----------------------------------------


def test_metric_extent_uses_the_selected_axis():
    """The core `z` bug: extent returned sample-index x on every axis.

    A run spanning 0..199,000,000 tokens got xlim=(0, 199), so the clip in
    draw_metric_plot dropped every point and the plot went blank.
    """
    m = metric_from([mkrun(0)])
    xmn, xmx, _, _ = w.metric_extent(m, 0, x_axis="tokens")
    assert xmx > 1_000_000, f"expected token units, got {xmx}"
    # The index axis still works.
    ixmn, ixmx, _, _ = w.metric_extent(m, 0, x_axis="step")
    assert ixmx == 199


def test_focus_limits_actually_draw_something():
    """End to end: focusing a run on a token axis must not blank the plot."""
    m = metric_from([mkrun(0)])
    xmn, xmx, ymn, ymx = w.metric_extent(m, 0, x_axis="tokens")
    plt = CountingPlt()
    drawn = w.draw_metric_plot(
        plt, m, 1, ["R1"], xlim=(xmn, xmx), ylim=(ymn, ymx), x_axis="tokens"
    )
    assert drawn == 1, "focus limits dropped every point"
    assert plt.series and len(plt.series[0][1]) > 1


def test_metric_span_skips_hidden_runs():
    """Pan/zoom bounds must match what is drawn."""
    runs = [mkrun(0), mkrun(1)]
    m = metric_from(runs)
    # Give run 1 a much wider y range, then hide it.
    for slot in m["runs"][1:]:
        slot["values"] = [v * 100 for v in slot["values"]]
    _, yspan_all = w.metric_span(m, 2, x_axis="step")
    _, yspan_vis = w.metric_span(m, 2, x_axis="step", hidden={1})
    assert yspan_vis[1] < yspan_all[1], "hidden run still widened the span"


# --- outlier hiding ----------------------------------------------------------


def test_outlier_mask_drops_the_spike():
    m = metric_from([mkrun(0, spike=True)])
    raw = w.metric_series(m, 0)
    trimmed = w.metric_series(m, 0, outliers=True)
    assert max(raw) == 500.0
    assert max(trimmed) < 500.0, "the spike survived the clip"
    assert len(trimmed) < len(raw)


def test_outlier_mask_keeps_x_and_y_paired():
    """The trap: y filtered without x desyncs every later point."""
    m = metric_from([mkrun(0, spike=True)])
    ys = w.metric_series(m, 0, outliers=True)
    axes = w.metric_axes(m, 0, outliers=True)
    assert len(axes["values"]) == len(ys)
    for col in axes["axes"].values():
        assert len(col) == len(ys), "an axis column drifted out of step with y"


def test_outlier_mask_leaves_short_series_alone():
    """Clipping a 3-point series to nothing is worse than showing the spike."""
    m = metric_from([mkrun(0, points=3)])
    assert len(w.metric_series(m, 0, outliers=True)) == 3


def test_draw_honours_the_outlier_flag():
    m = metric_from([mkrun(0, spike=True)])
    plt = CountingPlt()
    w.draw_metric_plot(plt, m, 1, ["R1"], outliers=True)
    assert max(plt.series[0][1]) < 500.0


# --- log scales --------------------------------------------------------------


def test_log_modes_cover_all_four_states():
    assert len(w.LOG_MODES) == 4
    assert w.LOG_MODES[0] == (False, False)
    assert (True, False) in w.LOG_MODES   # y only
    assert (False, True) in w.LOG_MODES   # x only
    assert (True, True) in w.LOG_MODES    # log-log


def test_xlog_transforms_the_x_axis():
    m = metric_from([mkrun(0)])
    plain = CountingPlt()
    w.draw_metric_plot(plain, m, 1, ["R1"], x_axis="tokens")
    logged = CountingPlt()
    w.draw_metric_plot(logged, m, 1, ["R1"], x_axis="tokens", xlog=True)
    assert max(plain.series[0][0]) > 1_000_000
    assert max(logged.series[0][0]) < 20, "x was not log-transformed"
    assert logged.xlabel_text and "log10" in logged.xlabel_text


def test_xlog_drops_nonpositive_x_without_crashing():
    """A step axis starts at 0; log10(0) must not raise."""
    m = metric_from([mkrun(0)])
    plt = CountingPlt()
    drawn = w.draw_metric_plot(plt, m, 1, ["R1"], x_axis="step", xlog=True)
    assert drawn == 1
    assert all(x == x for x in plt.series[0][0])  # no NaNs


# --- axis limit parsing ------------------------------------------------------


def test_parse_axis_limits_both_axes():
    x, y = w.parse_axis_limits("x=0:5000 y=2.5:13")
    assert x == (0.0, 5000.0)
    assert y == (2.5, 13.0)


def test_parse_axis_limits_single_axis():
    x, y = w.parse_axis_limits("y=1:2")
    assert x == (None, None)
    assert y == (1.0, 2.0)


def test_parse_axis_limits_open_ended():
    x, _ = w.parse_axis_limits("x=100:")
    assert x == (100.0, None)
    x2, _ = w.parse_axis_limits("x=:100")
    assert x2 == (None, 100.0)


def test_parse_axis_limits_blank_is_auto():
    assert w.parse_axis_limits("") == ((None, None), (None, None))
    assert w.parse_axis_limits("   ") == ((None, None), (None, None))


def test_parse_axis_limits_rejects_garbage():
    for bad in ("x=abc:1", "z=1:2", "x=1:2:3", "x1:2"):
        with pytest.raises(ValueError):
            w.parse_axis_limits(bad)


def test_axis_limits_error_reports_rather_than_raising():
    assert w.axis_limits_error("x=0:5000") == ""
    assert w.axis_limits_error("x=abc:1") != ""


def test_parse_axis_limits_rejects_inverted_range():
    with pytest.raises(ValueError):
        w.parse_axis_limits("x=100:1")


# --- app integration ---------------------------------------------------------


def test_g_cycles_log_state(patched):
    async def main():
        app = w.make_project_app("e/p", 3, 0)
        async with app.run_test(size=(150, 40)) as pilot:
            await loaded(app, pilot)
            seen = {app.log_mode}
            for _ in range(len(w.LOG_MODES) - 1):
                await pilot.press("g")
                await pilot.pause()
                seen.add(app.log_mode)
            assert len(seen) == len(w.LOG_MODES)
            await pilot.press("g")
            await pilot.pause()
            assert app.log_mode == 0, "cycle did not wrap"
            app.exit()

    asyncio.run(main())


def test_o_toggles_outliers(patched):
    async def main():
        app = w.make_project_app("e/p", 3, 0)
        async with app.run_test(size=(150, 40)) as pilot:
            await loaded(app, pilot)
            assert app.hide_outliers is False
            await pilot.press("o")
            await pilot.pause()
            assert app.hide_outliers is True
            await pilot.press("o")
            await pilot.pause()
            assert app.hide_outliers is False
            app.exit()

    asyncio.run(main())


def test_L_opens_the_limits_box(patched):
    async def main():
        app = w.make_project_app("e/p", 3, 0)
        async with app.run_test(size=(150, 40)) as pilot:
            await loaded(app, pilot)
            box = app.query_one("#limits_input")
            assert not box.display
            await pilot.press("L")
            await pilot.pause()
            assert box.display
            assert app.focused is box
            app.exit()

    asyncio.run(main())


def test_typing_limits_applies_them(patched):
    async def main():
        app = w.make_project_app("e/p", 3, 0)
        async with app.run_test(size=(150, 40)) as pilot:
            await loaded(app, pilot)
            await pilot.press("L")
            await pilot.pause()
            for ch in "y=1:5":
                await pilot.press(ch)
            for _ in range(30):
                await pilot.pause()
                if app.ylim != (None, None):
                    break
            assert app.ylim == (1.0, 5.0), app.ylim
            app.exit()

    asyncio.run(main())


def test_limits_keys_stay_literal_in_the_search_box(patched):
    """L/g/o must type as text while an Input has focus."""

    async def main():
        app = w.make_project_app("e/p", 3, 0)
        async with app.run_test(size=(150, 40)) as pilot:
            await loaded(app, pilot)
            before_log, before_out = app.log_mode, app.hide_outliers
            await pilot.press("slash")
            await pilot.pause()
            for ch in ("L", "g", "o"):
                await pilot.press(ch)
            await pilot.pause()
            assert app.query_one("#search_input").value == "Lgo"
            assert app.log_mode == before_log
            assert app.hide_outliers == before_out
            app.exit()

    asyncio.run(main())


# --- uncapped tiles ----------------------------------------------------------


def test_every_chartable_metric_gets_a_tile(monkeypatch):
    """The cap hid metrics behind a "N more" footer."""
    def many(i):
        rows = []
        for s in range(20):
            row = {"_step": s}
            for k in range(30):
                row[f"grp{k % 4}/m{k}"] = float(k) + s
            rows.append(row)
        return {
            "name": f"id{i}", "displayName": f"run-{i}", "state": "finished",
            "history": [json.dumps(r) for r in rows],
            "config": "{}", "summaryMetrics": "{}",
        }

    runs = [many(0), many(1)]
    monkeypatch.setattr(w, "fetch_project_run_names", lambda e, p, limit: list(runs))
    monkeypatch.setattr(w, "fetch_histories", lambda kept, **kw: list(kept))

    async def main():
        app = w.make_project_app("e/p", 2, 0)
        async with app.run_test(size=(150, 40)) as pilot:
            await loaded(app, pilot)
            await pilot.press("m")
            for _ in range(40):
                await pilot.pause()
            chartable = [m for m in app.metrics if w.chartable(m)]
            tiles = app.query(".chart-tile")
            assert len(tiles) == len(chartable), (
                f"{len(tiles)} tiles for {len(chartable)} chartable metrics"
            )
            app.exit()

    asyncio.run(main())


# --- the single-run app ------------------------------------------------------


def run_app_fixture(monkeypatch, points: int = 200):
    monkeypatch.setattr(
        w, "fetch_run",
        lambda e, p, r: mkrun(0, points=points) | {"historyLineCount": points},
    )


def test_run_app_composes(monkeypatch):
    """The whole single-run view had no coverage.

    A stray `Tabs(...)` yield added for the metric-group tab bar referenced a
    name the run app never imported, so `wandb-tui <run-url>` died with a
    NameError while all 140 project-view tests stayed green.
    """
    run_app_fixture(monkeypatch)

    async def main():
        app = w.make_run_app("e/p/r", 0)
        async with app.run_test(size=(140, 40)) as pilot:
            for _ in range(60):
                await pilot.pause()
                if app.metrics:
                    break
            assert app.is_running
            assert app.metrics, "run never loaded"
            app.exit()

    asyncio.run(main())


def test_run_app_has_the_chart_controls(monkeypatch):
    """g/o/L must work in the run view too, not just the project view."""
    run_app_fixture(monkeypatch)

    async def main():
        app = w.make_run_app("e/p/r", 0)
        async with app.run_test(size=(140, 40)) as pilot:
            for _ in range(60):
                await pilot.pause()
                if app.metrics:
                    break
            await pilot.press("g")
            await pilot.pause()
            assert app.log_mode == 1
            await pilot.press("o")
            await pilot.pause()
            assert app.hide_outliers is True
            await pilot.press("L")
            await pilot.pause()
            assert app.query_one("#limits_input").display
            app.exit()

    asyncio.run(main())


def test_g_no_longer_cycles_metric_groups(patched):
    """`g` was the metric-group cycle; the tab bar owns that now."""

    async def main():
        app = w.make_project_app("e/p", 3, 0)
        async with app.run_test(size=(150, 40)) as pilot:
            await loaded(app, pilot)
            before = app.group_idx
            await pilot.press("g")
            await pilot.pause()
            assert app.group_idx == before, "g still moved the metric group"
            assert app.log_mode == 1
            app.exit()

    asyncio.run(main())


# --- lazy tile drawing -------------------------------------------------------


def many_metric_runs(n: int = 60):
    def mk(i):
        rows = [
            {"_step": s, **{f"g{k % 4}/m{k}": float(k) + s for k in range(n)}}
            for s in range(20)
        ]
        return {
            "name": f"id{i}", "displayName": f"r{i}", "state": "finished",
            "history": [json.dumps(r) for r in rows],
            "config": "{}", "summaryMetrics": "{}",
        }
    return [mk(0), mk(1)]


def test_offscreen_tiles_are_deferred(monkeypatch):
    """Uncapped tiles are only affordable if off-screen ones skip drawing."""
    runs = many_metric_runs()
    monkeypatch.setattr(w, "fetch_project_run_names", lambda e, p, limit: list(runs))
    monkeypatch.setattr(w, "fetch_histories", lambda kept, **kw: list(kept))

    async def main():
        app = w.make_project_app("e/p", 2, 0)
        async with app.run_test(size=(160, 45)) as pilot:
            await loaded(app, pilot)
            await pilot.press("m")
            for _ in range(40):
                await pilot.pause()
            tiles = list(app.query(".chart-tile"))
            deferred = [t for t in tiles if getattr(t, "_dirty", False)]
            assert len(tiles) > 20, "need enough tiles to overflow the viewport"
            assert deferred, "every tile drew at mount; lazy drawing is not working"
            app.exit()

    asyncio.run(main())


def test_scrolling_draws_the_tiles_it_reveals(monkeypatch):
    """A visible tile must never stay blank.

    Handling key/mouse events on the app missed keyboard scrolling: ProjectApp
    defines its own on_key for tab-completion, which silently shadowed the
    mixin's. Measured 3 blank-but-visible tiles after pagedown.
    """
    runs = many_metric_runs()
    monkeypatch.setattr(w, "fetch_project_run_names", lambda e, p, limit: list(runs))
    monkeypatch.setattr(w, "fetch_histories", lambda kept, **kw: list(kept))

    async def main():
        app = w.make_project_app("e/p", 2, 0)
        async with app.run_test(size=(160, 45)) as pilot:
            await loaded(app, pilot)
            await pilot.press("m")
            for _ in range(40):
                await pilot.pause()
            pane = app.query_one("#charts")
            tiles = list(app.query(".chart-tile"))

            pane.focus()
            for _ in range(6):
                await pilot.press("pagedown")
            for _ in range(25):
                await pilot.pause()
            blank = [t.metric_name for t in tiles
                     if getattr(t, "_dirty", False) and t.on_screen()]
            assert not blank, f"blank tiles after pagedown: {blank}"

            pane.scroll_to(y=400, animate=False)
            for _ in range(25):
                await pilot.pause()
            blank = [t.metric_name for t in tiles
                     if getattr(t, "_dirty", False) and t.on_screen()]
            assert not blank, f"blank tiles after scroll_to: {blank}"
            app.exit()

    asyncio.run(main())


# --- the zoom screen ---------------------------------------------------------


def open_zoom(app):
    app.open_chart_fullscreen("loss")
    return app.screen


def test_z_skips_hidden_runs(patched):
    """Focusing a hidden run fitted the view to a series that is not drawn."""

    async def main():
        app = w.make_project_app("e/p", 3, 0)
        async with app.run_test(size=(160, 45)) as pilot:
            await loaded(app, pilot)
            app.hidden_runs = {1}
            await pilot.press("m")
            for _ in range(20):
                await pilot.pause()
            open_zoom(app)
            for _ in range(20):
                await pilot.pause()
            seen = []
            for _ in range(4):
                await pilot.press("z")
                await pilot.pause()
                seen.append(app.screen.focus_run)
            assert 1 not in seen, f"focused a hidden run: {seen}"
            app.exit()

    asyncio.run(main())


def test_z_on_a_token_axis_still_draws(patched, monkeypatch):
    """The money test: focus must not blank the chart on a non-index axis.

    metric_extent returned sample-index x while the plot drew token values, so
    focus set xlim=(0, 199) against data spanning 0..199,000,000 and the clip
    dropped every point -- draw_metric_plot returned 0.
    """
    seen: list[int] = []
    real = w.draw_metric_plot

    def spy(*a, **k):
        n = real(*a, **k)
        seen.append(n)
        return n

    async def main():
        import wandb_tui

        app = w.make_project_app("e/p", 3, 0)
        async with app.run_test(size=(160, 45)) as pilot:
            await loaded(app, pilot)
            # Cycle X onto the tokens axis.
            for _ in range(len(w.X_AXES)):
                if app.x_axis().id == "tokens":
                    break
                await pilot.press("X")
                await pilot.pause()
            assert app.x_axis().id == "tokens", app.x_axis().id
            await pilot.press("m")
            for _ in range(20):
                await pilot.pause()
            open_zoom(app)
            for _ in range(20):
                await pilot.pause()
            wandb_tui.draw_metric_plot = spy
            try:
                await pilot.press("z")
                for _ in range(10):
                    await pilot.pause()
            finally:
                wandb_tui.draw_metric_plot = real
            assert seen, "zoom never redrew"
            assert seen[-1] > 0, "focus on a token axis drew nothing"
            app.exit()

    asyncio.run(main())


def test_zoom_reads_log_state_live(patched):
    """State is shared: `g` in the grid reaches an open zoom."""

    async def main():
        app = w.make_project_app("e/p", 3, 0)
        async with app.run_test(size=(160, 45)) as pilot:
            await loaded(app, pilot)
            await pilot.press("m")
            for _ in range(20):
                await pilot.pause()
            screen = open_zoom(app)
            for _ in range(20):
                await pilot.pause()
            assert screen.state().get("log") == (False, False)
            await pilot.press("g")
            await pilot.pause()
            assert screen.state().get("log") == w.LOG_MODES[1]
            app.exit()

    asyncio.run(main())


def test_zoom_pan_does_not_leak_to_the_grid(patched):
    """Pan/zoom is view-local; the `L` window is global."""

    async def main():
        app = w.make_project_app("e/p", 3, 0)
        async with app.run_test(size=(160, 45)) as pilot:
            await loaded(app, pilot)
            await pilot.press("m")
            for _ in range(20):
                await pilot.pause()
            open_zoom(app)
            for _ in range(20):
                await pilot.pause()
            await pilot.press("l")  # pan right
            await pilot.pause()
            assert app.screen.local_xlim != (None, None)
            assert app.xlim == (None, None), "a local pan leaked into the app"
            app.exit()

    asyncio.run(main())


def test_zoom_reset_falls_back_to_the_L_window(patched):
    """Z drops the local override, not the globally-set limits."""

    async def main():
        app = w.make_project_app("e/p", 3, 0)
        async with app.run_test(size=(160, 45)) as pilot:
            await loaded(app, pilot)
            app.apply_axis_limits("x=0:50")
            await pilot.press("m")
            for _ in range(20):
                await pilot.pause()
            screen = open_zoom(app)
            for _ in range(20):
                await pilot.pause()
            await pilot.press("l")
            await pilot.pause()
            await pilot.press("Z")
            await pilot.pause()
            xlim, _ = screen.effective_limits()
            assert xlim == (0.0, 50.0), xlim
            app.exit()

    asyncio.run(main())


def test_bad_limits_are_reported_not_swallowed(patched):
    """A typo'd expression must say so rather than silently doing nothing."""

    async def main():
        app = w.make_project_app("e/p", 3, 0)
        async with app.run_test(size=(150, 40)) as pilot:
            await loaded(app, pilot)
            app.apply_axis_limits("x=abc:1")
            await pilot.pause()
            assert app.limits_error, "no error recorded"
            meta = str(app.query_one("#meta").content)
            assert "limits error" in meta, meta[:200]
            # The last good window survives rather than blanking the chart.
            assert app.xlim == (None, None)
            app.exit()

    asyncio.run(main())


def test_good_limits_clear_the_error(patched):
    async def main():
        app = w.make_project_app("e/p", 3, 0)
        async with app.run_test(size=(150, 40)) as pilot:
            await loaded(app, pilot)
            app.apply_axis_limits("x=abc:1")
            await pilot.pause()
            assert app.limits_error
            app.apply_axis_limits("x=0:100")
            await pilot.pause()
            assert app.limits_error == ""
            assert app.xlim == (0.0, 100.0)
            app.exit()

    asyncio.run(main())


# --- run colours -------------------------------------------------------------


def test_legend_and_plot_colours_come_from_one_palette():
    """They used to be two lists: R2 read green in text but drew orange."""
    for i in range(15):
        r, g, b = w.rgb_for_run(i)
        assert w.run_style(i) == f"#{r:02x}{g:02x}{b:02x}"


def test_run_style_wraps_with_the_plot_palette():
    n = len(w.PLOT_PALETTE)
    assert w.run_style(0) == w.run_style(n)
    assert w.run_style(1) != w.run_style(0)
