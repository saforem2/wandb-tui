"""Tests for the nested run-group tree (W&B workspace "Group runs by...").

The W&B workspace nests runs under an ordered list of config keys, each level
collapsible and carrying a run count:

    world_size: 6144                    89
      model_spec.flavor: 20b            45
        optimizer.name: SophiaG         42
            fallen-paper-3378

These pin the builder that produces that structure.
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
        "state": "running",
        "createdAt": "2026-08-01T00:00:00Z",
        "updatedAt": "2026-08-01T01:00:00Z",
        "history": [{"loss": float(i)} for _ in range(3)],
        "config": json.dumps({k: {"value": v} for k, v in (config or {}).items()}),
        "summaryMetrics": "{}",
    }


# --- key list parsing --------------------------------------------------------


def test_parse_group_keys_splits_on_commas():
    assert w.parse_group_keys("a,b,c") == ["a", "b", "c"]


def test_parse_group_keys_trims_and_drops_blanks():
    assert w.parse_group_keys(" a , , b ,") == ["a", "b"]


def test_parse_group_keys_empty_is_no_grouping():
    assert w.parse_group_keys("") == []
    assert w.parse_group_keys("   ") == []


def test_parse_group_keys_dedupes_preserving_order():
    """Grouping by the same key twice would nest a level inside itself."""
    assert w.parse_group_keys("a,b,a") == ["a", "b"]


# --- tree construction -------------------------------------------------------


def test_group_tree_nests_in_key_order():
    runs = [
        mkrun(0, {"ws": 6144, "flavor": "20b"}),
        mkrun(1, {"ws": 6144, "flavor": "2b"}),
        mkrun(2, {"ws": 48, "flavor": "20b"}),
    ]
    rows = w.group_tree_rows(runs, ["ws", "flavor"])
    # Depth-0 nodes are ws, depth-1 are flavor, leaves are runs.
    depths = {r.depth for r in rows if r.is_group}
    assert depths == {0, 1}
    top = [r for r in rows if r.is_group and r.depth == 0]
    assert [r.label for r in top] == ["ws: 48", "ws: 6144"]


def test_group_tree_counts_runs_per_node():
    runs = [mkrun(i, {"ws": 6144 if i < 3 else 48}) for i in range(5)]
    rows = w.group_tree_rows(runs, ["ws"])
    counts = {r.label: r.count for r in rows if r.is_group}
    assert counts == {"ws: 6144": 3, "ws: 48": 2}


def test_group_tree_leaves_are_runs_under_their_group():
    runs = [mkrun(0, {"ws": 1}), mkrun(1, {"ws": 2})]
    rows = w.group_tree_rows(runs, ["ws"])
    leaves = [r for r in rows if not r.is_group]
    assert len(leaves) == 2
    # Each leaf carries the run's index into the original list, so metric
    # slots can be looked up without re-deriving order.
    assert {r.run_index for r in leaves} == {0, 1}


def test_group_tree_no_keys_is_flat_run_list():
    runs = [mkrun(i) for i in range(3)]
    rows = w.group_tree_rows(runs, [])
    assert all(not r.is_group for r in rows)
    assert [r.run_index for r in rows] == [0, 1, 2]


def test_group_tree_preserves_every_run():
    runs = [mkrun(i, {"a": i % 2, "b": i % 3}) for i in range(12)]
    rows = w.group_tree_rows(runs, ["a", "b"])
    assert sorted(r.run_index for r in rows if not r.is_group) == list(range(12))


def test_group_tree_buckets_missing_values():
    runs = [mkrun(0, {"ws": 1}), mkrun(1, {})]
    rows = w.group_tree_rows(runs, ["ws"])
    labels = [r.label for r in rows if r.is_group]
    assert any(w.GROUP_UNSET in lab for lab in labels)


def test_group_tree_sorts_numeric_groups_numerically():
    runs = [mkrun(i, {"bs": v}) for i, v in enumerate([2048, 128, 512])]
    labels = [r.label for r in w.group_tree_rows(runs, ["bs"]) if r.is_group]
    assert labels == ["bs: 128", "bs: 512", "bs: 2048"]


def test_group_tree_unknown_key_groups_everything_as_unset():
    """A typo'd key must not crash or silently drop runs."""
    runs = [mkrun(i) for i in range(3)]
    rows = w.group_tree_rows(runs, ["nope"])
    assert len([r for r in rows if not r.is_group]) == 3


# --- collapse ----------------------------------------------------------------


def test_collapsed_node_hides_its_subtree():
    runs = [mkrun(i, {"ws": 1 if i < 2 else 2}) for i in range(4)]
    rows = w.group_tree_rows(runs, ["ws"], collapsed={"ws: 1"})
    visible = [r for r in rows if not r.is_group]
    # The two runs under the collapsed node are hidden; the other two remain.
    assert len(visible) == 2
    # The collapsed group row itself stays visible so it can be re-expanded.
    assert any(r.label == "ws: 1" for r in rows if r.is_group)


def test_collapsing_parent_hides_nested_children():
    runs = [mkrun(i, {"a": 1, "b": i}) for i in range(3)]
    rows = w.group_tree_rows(runs, ["a", "b"], collapsed={"a: 1"})
    assert [r.label for r in rows if r.is_group] == ["a: 1"]
    assert not [r for r in rows if not r.is_group]


def test_meta_summary_uses_the_outer_key_not_the_whole_expression():
    """Live-data bug: the summary read "(unset)(10)" under a populated tree.

    It passed the comma-joined expression to group_value as if it were one
    key, so every run missed and fell into (unset) -- the header contradicted
    the tree directly below it.
    """
    runs = [mkrun(i, {"ws": 1 if i < 2 else 2, "flavor": "20b"}) for i in range(4)]
    text = str(
        w.format_project_meta(
            "e", "p", "u", 4, runs, [], [], "ALL", "", "metric asc", False, "ok",
            group_by="ws,flavor",
        )
    )
    line = next(ln for ln in text.splitlines() if ln.startswith("grouped by"))
    assert w.GROUP_UNSET not in line
    assert "ws: " not in line  # summarises values, not the label prefix
    assert "1(2)" in line and "2(2)" in line
    assert "nested" in line  # signals the second level exists


# --- app integration ---------------------------------------------------------


async def settled(app, pilot, expect, tries: int = 40):
    """Wait for the debounced group-key application to land."""
    for _ in range(tries):
        await pilot.pause()
        if app.group_keys == expect:
            return True
    raise AssertionError(f"group_keys never became {expect} (got {app.group_keys})")


async def loaded(app, pilot, tries: int = 60):
    for _ in range(tries):
        await pilot.pause()
        if app.runs:
            return True
    raise AssertionError(f"runs never loaded (status={app.status!r})")


@pytest.fixture
def patched(monkeypatch):
    runs = [
        mkrun(i, {"ws": 6144 if i < 2 else 48, "flavor": "20b" if i % 2 else "2b"})
        for i in range(4)
    ]
    monkeypatch.setattr(w, "fetch_project_run_names", lambda e, p, limit: list(runs))
    monkeypatch.setattr(w, "fetch_histories", lambda kept, **kw: list(kept))
    return runs


def test_G_opens_the_group_box(patched):
    async def main():
        app = w.make_project_app("e/p", 4, 0)
        async with app.run_test(size=(150, 30)) as pilot:
            await loaded(app, pilot)
            box = app.query_one("#group_input")
            assert not box.display
            await pilot.press("G")
            await pilot.pause()
            assert box.display
            assert app.focused is box
            app.exit()

    asyncio.run(main())


def test_typing_keys_groups_the_table(patched):
    async def main():
        app = w.make_project_app("e/p", 4, 0)
        async with app.run_test(size=(150, 30)) as pilot:
            await loaded(app, pilot)
            await pilot.press("G")
            await pilot.pause()
            for ch in "ws":
                await pilot.press(ch)
            await settled(app, pilot, ["ws"])
            table = app.query_one("#table")
            first = [str(table.get_row(rk)[0]) for rk in table.rows]
            assert any("ws: " in cell for cell in first)
            app.exit()

    asyncio.run(main())


def test_metric_values_stay_with_their_run_in_tree(patched):
    """Leaf rows must show their own run's numbers, not a neighbour's."""

    async def main():
        app = w.make_project_app("e/p", 4, 0)
        async with app.run_test(size=(150, 30)) as pilot:
            await loaded(app, pilot)
            app.set_group_keys("ws")
            await pilot.pause()
            rows = w.group_tree_rows(app.runs, app.group_keys)
            for row in rows:
                if row.is_group:
                    continue
                run = app.runs[row.run_index]
                assert row.label.strip() == run_label_of(run)
            app.exit()

    asyncio.run(main())


def run_label_of(run: dict) -> str:
    return str(run.get("displayName") or run.get("name") or "")


def test_bad_key_does_not_crash_the_app(patched):
    async def main():
        app = w.make_project_app("e/p", 4, 0)
        async with app.run_test(size=(150, 30)) as pilot:
            await loaded(app, pilot)
            app.set_group_keys("definitely.not.a.key")
            await pilot.pause()
            assert app.is_running
            app.exit()

    asyncio.run(main())


def test_enter_toggles_a_group_row(patched):
    async def main():
        app = w.make_project_app("e/p", 4, 0)
        async with app.run_test(size=(150, 30)) as pilot:
            await loaded(app, pilot)
            app.set_group_keys("ws")
            await pilot.pause()
            table = app.query_one("#table")
            table.focus()
            table.move_cursor(row=0)
            before = table.row_count
            await pilot.press("enter")
            await pilot.pause()
            collapsed = table.row_count
            assert collapsed < before, "collapsing should hide child rows"
            await pilot.press("enter")
            await pilot.pause()
            assert table.row_count == before, "re-expanding should restore them"
            app.exit()

    asyncio.run(main())


def test_collapse_state_survives_a_rerender(patched):
    async def main():
        app = w.make_project_app("e/p", 4, 0)
        async with app.run_test(size=(150, 30)) as pilot:
            await loaded(app, pilot)
            app.set_group_keys("ws")
            await pilot.pause()
            table = app.query_one("#table")
            table.focus()
            table.move_cursor(row=0)
            await pilot.press("enter")
            await pilot.pause()
            collapsed = table.row_count
            app.render_table()
            await pilot.pause()
            assert table.row_count == collapsed
            app.exit()

    asyncio.run(main())


def test_changing_keys_resets_collapse_state(patched):
    """Collapse is keyed on labels, which are meaningless after a key change."""

    async def main():
        app = w.make_project_app("e/p", 4, 0)
        async with app.run_test(size=(150, 30)) as pilot:
            await loaded(app, pilot)
            app.set_group_keys("ws")
            await pilot.pause()
            table = app.query_one("#table")
            table.focus()
            table.move_cursor(row=0)
            await pilot.press("enter")
            await pilot.pause()
            assert app.collapsed_groups
            app.set_group_keys("flavor")
            await pilot.pause()
            assert app.collapsed_groups == set()
            app.exit()

    asyncio.run(main())


def test_nested_keys_produce_nested_rows(patched):
    async def main():
        app = w.make_project_app("e/p", 4, 0)
        async with app.run_test(size=(160, 30)) as pilot:
            await loaded(app, pilot)
            app.set_group_keys("ws,flavor")
            await pilot.pause()
            rows = app.tree_rows
            assert {r.depth for r in rows if r.is_group} == {0, 1}
            # Every run still appears exactly once as a leaf.
            leaves = sorted(r.run_index for r in rows if not r.is_group)
            assert leaves == list(range(len(app.runs)))
            app.exit()

    asyncio.run(main())


def test_escape_keeps_the_grouping_and_frees_the_cursor(patched):
    """Esc must not discard the grouping.

    Esc is the obvious way out of a text box, and the tree's collapse
    interaction needs table focus -- if Esc also ungrouped, collapse would be
    unreachable without discovering Tab.
    """

    async def main():
        app = w.make_project_app("e/p", 4, 0)
        async with app.run_test(size=(150, 30)) as pilot:
            await loaded(app, pilot)
            await pilot.press("G")
            await pilot.pause()
            for ch in "ws":
                await pilot.press(ch)
            await settled(app, pilot, ["ws"])
            grouped_rows = app.query_one("#table").row_count
            await pilot.press("escape")
            await pilot.pause()
            assert app.group_keys == ["ws"], "Esc must not drop the grouping"
            assert app.query_one("#table").row_count == grouped_rows
            # And the cursor is back on the results, ready to collapse.
            assert getattr(app.focused, "id", None) != "group_input"
            app.exit()

    asyncio.run(main())


def test_collapse_is_reachable_after_escape(patched):
    """End-to-end: type keys, Esc, then collapse a node with Enter."""

    async def main():
        app = w.make_project_app("e/p", 4, 0)
        async with app.run_test(size=(150, 30)) as pilot:
            await loaded(app, pilot)
            await pilot.press("G")
            await pilot.pause()
            for ch in "ws":
                await pilot.press(ch)
            await settled(app, pilot, ["ws"])
            await pilot.press("escape")
            await pilot.pause()
            table = app.query_one("#table")
            table.focus()
            table.move_cursor(row=0)
            before = table.row_count
            await pilot.press("enter")
            await pilot.pause()
            assert table.row_count < before
            app.exit()

    asyncio.run(main())


def test_completion_universe_is_cached_between_keystrokes(monkeypatch):
    """Typing a key name must not re-derive the candidate set per character.

    group_run_keys walks every config key against every run (~0.1s for 166
    keys x 10 runs here). Calling it per keystroke made the group box visibly
    laggy while typing, even though its answer cannot change until a refetch.
    """
    runs = [
        mkrun(i, {f"k{j}": (i % 2 if j % 3 == 0 else j) for j in range(60)})
        for i in range(8)
    ]
    monkeypatch.setattr(w, "fetch_project_run_names", lambda e, p, limit: list(runs))
    monkeypatch.setattr(w, "fetch_histories", lambda kept, **kw: list(kept))

    calls = {"n": 0}
    real = w.group_run_keys

    def counted(rs, **kw):
        calls["n"] += 1
        return real(rs, **kw)

    monkeypatch.setattr(w, "group_run_keys", counted)

    async def main():
        app = w.make_project_app("e/p", 8, 0)
        async with app.run_test(size=(150, 30)) as pilot:
            await loaded(app, pilot)
            await pilot.press("G")
            await pilot.pause()
            calls["n"] = 0
            for ch in "k0":
                await pilot.press(ch)
                await pilot.pause()
            # Once for the first keystroke's cache fill, never again.
            assert calls["n"] <= 1, f"recomputed {calls['n']}x while typing"
            app.exit()

    asyncio.run(main())


def test_typing_does_not_regroup_per_keystroke(patched):
    """Grouping must be debounced like the other boxes.

    Applying on every keystroke rebuilt the columns and re-rendered the whole
    tree per character, and every intermediate prefix ("w", "wo", "wor", ...)
    is itself a valid grouping that gets built in full and thrown away.
    """
    calls = {"n": 0}

    async def main():
        app = w.make_project_app("e/p", 4, 0)
        async with app.run_test(size=(150, 30)) as pilot:
            await loaded(app, pilot)
            real = app.set_group_keys

            def counted(expr):
                calls["n"] += 1
                return real(expr)

            app.set_group_keys = counted
            await pilot.press("G")
            await pilot.pause()
            for ch in "flavor":
                await pilot.press(ch)
            await settled(app, pilot, ["flavor"])
            # One application for the whole burst, not one per character.
            assert calls["n"] <= 2, f"regrouped {calls['n']}x for 6 keystrokes"
            app.exit()

    asyncio.run(main())


def test_G_typed_in_search_stays_literal(patched):
    async def main():
        app = w.make_project_app("e/p", 4, 0)
        async with app.run_test(size=(150, 30)) as pilot:
            await loaded(app, pilot)
            await pilot.press("slash")
            await pilot.pause()
            await pilot.press("G")
            await pilot.pause()
            assert app.query_one("#search_input").value == "G"
            assert app.group_keys == []
            app.exit()

    asyncio.run(main())
