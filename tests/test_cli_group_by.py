"""Tests for --group-by and the --group -> --metric-group deprecation.

`--group` has always filtered *metric* groups (train, grad, config) in
--once/--json output. Once `G` added run grouping the name became actively
misleading, so it is renamed to --metric-group and --group-by is the new flag
for grouping runs.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

import wandb_tui as w


def mkrun(i: int, config: dict, loss: float) -> dict:
    return {
        "name": f"id{i}",
        "displayName": f"run-{i}",
        "state": "finished",
        "createdAt": "2026-08-01T00:00:00Z",
        "updatedAt": "2026-08-01T01:00:00Z",
        "history": [json.dumps({"_step": s, "loss": loss + s}) for s in range(3)],
        "config": json.dumps({k: {"value": v} for k, v in config.items()}),
        "summaryMetrics": "{}",
    }


@pytest.fixture
def runs():
    return [
        mkrun(0, {"ws": 3072, "flavor": "20b"}, 1.0),
        mkrun(1, {"ws": 3072, "flavor": "2b"}, 2.0),
        mkrun(2, {"ws": 6144, "flavor": "20b"}, 3.0),
    ]


# --- argument parsing --------------------------------------------------------


def test_group_by_flag_exists():
    parser = w.build_parser()
    args = parser.parse_args(["e/p", "--group-by", "ws,flavor"])
    assert args.group_by == "ws,flavor"


def test_metric_group_is_the_new_spelling():
    parser = w.build_parser()
    args = parser.parse_args(["e/p", "--metric-group", "train"])
    assert args.metric_group == "train"


def test_group_still_parses_for_compatibility():
    parser = w.build_parser()
    args = parser.parse_args(["e/p", "--group", "train"])
    assert args.group == "train"


def test_group_and_metric_group_resolve_to_one_value():
    parser = w.build_parser()
    assert w.resolve_metric_group(parser.parse_args(["e/p", "--group", "grad"])) == "grad"
    assert w.resolve_metric_group(parser.parse_args(["e/p", "--metric-group", "grad"])) == "grad"
    # Default when neither is given.
    assert w.resolve_metric_group(parser.parse_args(["e/p"])) == "ALL"


def test_metric_group_wins_when_both_given():
    """The new spelling is authoritative; the old one is a fallback."""
    parser = w.build_parser()
    args = parser.parse_args(["e/p", "--group", "old", "--metric-group", "new"])
    assert w.resolve_metric_group(args) == "new"


# --- deprecation warning -----------------------------------------------------


def run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", "import wandb_tui; wandb_tui.main()", *args],
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_group_warns_on_stderr_not_stdout():
    """The warning must not corrupt piped stdout."""
    proc = run_cli("--help")
    assert proc.returncode == 0
    # --help alone must never warn.
    assert "deprecated" not in proc.stderr.lower()


def test_help_documents_both_flags():
    proc = run_cli("--help")
    assert "--group-by" in proc.stdout
    assert "--metric-group" in proc.stdout


# --- non-interactive grouping ------------------------------------------------


def test_once_prints_a_tree_when_grouped(runs, monkeypatch, capsys):
    monkeypatch.setattr(w, "fetch_project_runs", lambda e, p, limit=8, **k: list(runs))
    w.print_once("e/p", runs_limit=3, group_by="ws")
    out = capsys.readouterr().out
    assert "Group / Run" in out
    assert "ws: 3072" in out
    assert "ws: 6144" in out
    # Every run still appears as a leaf.
    for r in runs:
        assert r["displayName"] in out


def test_once_without_group_by_is_the_flat_table(runs, monkeypatch, capsys):
    monkeypatch.setattr(w, "fetch_project_runs", lambda e, p, limit=8, **k: list(runs))
    w.print_once("e/p", runs_limit=3)
    out = capsys.readouterr().out
    assert "Group / Run" not in out
    assert "metric" in out


def test_once_tree_counts_match_group_sizes(runs, monkeypatch, capsys):
    monkeypatch.setattr(w, "fetch_project_runs", lambda e, p, limit=8, **k: list(runs))
    w.print_once("e/p", runs_limit=3, group_by="ws")
    out = capsys.readouterr().out
    line = next(ln for ln in out.splitlines() if "ws: 3072" in ln)
    assert "2" in line  # two runs at ws=3072


def test_json_nests_groups(runs, monkeypatch, tmp_path):
    monkeypatch.setattr(w, "fetch_project_runs", lambda e, p, limit=8, **k: list(runs))
    path = tmp_path / "out.json"
    w.dump_json("e/p", str(path), runs_limit=3, group_by="ws,flavor")
    data = json.loads(path.read_text())
    assert data["group_by"] == ["ws", "flavor"]
    assert "groups" in data
    labels = [g["label"] for g in data["groups"]]
    assert labels == ["ws: 3072", "ws: 6144"]
    # Nested one level deeper, and every run is present exactly once.
    names = []

    def walk(nodes):
        for node in nodes:
            names.extend(node.get("runs", []))
            walk(node.get("groups", []))

    walk(data["groups"])
    assert sorted(names) == ["run-0", "run-1", "run-2"]


def test_json_without_group_by_has_no_groups_key(runs, monkeypatch, tmp_path):
    monkeypatch.setattr(w, "fetch_project_runs", lambda e, p, limit=8, **k: list(runs))
    path = tmp_path / "out.json"
    w.dump_json("e/p", str(path), runs_limit=3)
    data = json.loads(path.read_text())
    assert "groups" not in data
    assert data.get("group_by") in (None, [])


def test_bad_group_by_key_does_not_crash_once(runs, monkeypatch, capsys):
    """A typo'd key buckets everything under (unset) rather than exploding."""
    monkeypatch.setattr(w, "fetch_project_runs", lambda e, p, limit=8, **k: list(runs))
    w.print_once("e/p", runs_limit=3, group_by="nope")
    out = capsys.readouterr().out
    assert w.GROUP_UNSET in out
    for r in runs:
        assert r["displayName"] in out
