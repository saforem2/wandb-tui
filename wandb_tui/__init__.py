#!/usr/bin/env python3
"""Terminal dashboard for one or more Weights & Biases runs.

Default run:
  https://wandb.ai/aurora_gpt/ezpz.examples.fsdp_tp/runs/vrxuo55p

Usage:
  python ~/wandb_run_tui.py
  python ~/wandb_run_tui.py https://wandb.ai/ENTITY/PROJECT/runs/RUN_ID
  python ~/wandb_run_tui.py --once
  python ~/wandb_run_tui.py --json /tmp/run.json
  python ~/wandb_run_tui.py https://wandb.ai/ENTITY/PROJECT --runs 8

Controls in TUI:
  q              quit
  ↑/↓ or j/k      scroll
  PgUp/PgDn      page scroll
  Home/End       jump
  /              search metric names
  Esc            clear search
  g              cycle metric group filter
  m              toggle table/chart mode in project view
  s              cycle sort mode
  r              refresh from W&B
  ?              help
"""
from __future__ import annotations

import argparse
import curses
import datetime as _dt
import json
import math
import os
import re
import signal
import sys
import time
from statistics import mean, pstdev
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_URL = "https://wandb.ai/aurora_gpt/ezpz.examples.fsdp_tp/runs/vrxuo55p"
GRAPHQL_URL = "https://api.wandb.ai/graphql"
SPARKS = "▁▂▃▄▅▆▇█"
SORT_MODES = ("name", "group", "latest", "count", "min", "max", "mean")


def parse_run_ref(ref: str) -> tuple[str, str, str, str]:
    """Return entity, project, run_id, canonical URL from a W&B URL or path."""
    ref = ref.strip()
    m = re.search(r"wandb\.ai/([^/]+)/(.+?)/runs/([^/?#]+)", ref)
    if m:
        entity, project, run_id = m.group(1), m.group(2), m.group(3)
    else:
        parts = ref.strip("/").split("/")
        if len(parts) == 3:
            entity, project, run_id = parts
        else:
            raise SystemExit(
                "Run must be a W&B URL or ENTITY/PROJECT/RUN_ID, e.g. " + DEFAULT_URL
            )
    return entity, project, run_id, f"https://wandb.ai/{entity}/{project}/runs/{run_id}"


def parse_project_ref(ref: str) -> tuple[str, str, str]:
    """Return entity, project, canonical URL from a W&B project URL or path."""
    ref = ref.strip()
    m = re.search(r"wandb\.ai/([^/]+)/([^/?#]+)", ref)
    if m and "/runs/" not in ref:
        entity, project = m.group(1), m.group(2)
    else:
        parts = ref.split("?", 1)[0].strip("/").split("/")
        if len(parts) == 2:
            entity, project = parts
        else:
            raise SystemExit("Project must be a W&B URL or ENTITY/PROJECT")
    return entity, project, f"https://wandb.ai/{entity}/{project}"


def ref_kind(ref: str) -> str:
    ref0 = ref.split("?", 1)[0].strip("/")
    if "/runs/" in ref or len(ref0.split("/")) == 3:
        return "run"
    return "project"


def graphql(query: str, variables: dict[str, Any], timeout: int = 90) -> dict[str, Any]:
    payload_obj = {"query": query, "variables": variables}
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "wandb-run-tui/1.0",
    }
    # If present, use WANDB_API_KEY for private runs; public runs work without it.
    api_key = os.environ.get("WANDB_API_KEY")
    if api_key:
        import base64

        token = base64.b64encode(("api:" + api_key).encode()).decode()
        headers["Authorization"] = f"Basic {token}"

    # Prefer requests when available. W&B's public GraphQL edge can behave
    # differently for urllib-only clients and return `project: null` for public
    # runs that are otherwise readable.
    try:
        import requests  # type: ignore

        resp = requests.post(GRAPHQL_URL, json=payload_obj, headers=headers, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
    except ImportError:
        payload = json.dumps(payload_obj).encode("utf-8")
        req = Request(GRAPHQL_URL, data=payload, headers=headers, method="POST")
        try:
            with urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except HTTPError as e:
            body = e.read().decode("utf-8", "replace")
            raise RuntimeError(f"W&B GraphQL HTTP {e.code}: {body[:1000]}") from e
        except URLError as e:
            raise RuntimeError(f"Could not reach W&B GraphQL: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Could not query W&B GraphQL: {e}") from e

    if data.get("errors"):
        raise RuntimeError("W&B GraphQL errors: " + json.dumps(data["errors"], indent=2)[:2000])
    return data["data"]


def fetch_run(entity: str, project: str, run_id: str, samples: int = 10000) -> dict[str, Any]:
    query = """
    query Run($entity:String!, $project:String!, $name:String!, $samples:Int!, $maxKeyLimit:Int!) {
      project(name:$project, entityName:$entity) {
        run(name:$name) {
          id
          name
          displayName
          state
          createdAt
          updatedAt
          heartbeatAt
          description
          notes
          historyLineCount
          historyKeys
          history(samples:$samples, maxKeyLimit:$maxKeyLimit)
          summaryMetrics
          config
          systemMetrics
        }
      }
    }
    """
    data = graphql(
        query,
        {
            "entity": entity,
            "project": project,
            "name": run_id,
            "samples": samples,
            "maxKeyLimit": 10000,
        },
    )
    project_obj = data.get("project")
    if project_obj is None:
        raise RuntimeError(
            f"W&B returned no project for {entity}/{project}. "
            "For this public run, the W&B edge currently requires the Python "
            "`requests` client/TLS stack; install it with `python -m pip install requests` "
            "or run the script with the conda/base Python where requests is installed."
        )
    run = project_obj.get("run")
    if not run:
        raise RuntimeError(f"Run not found: {entity}/{project}/{run_id}")
    run["entity"] = entity
    run["project"] = project
    return run


def fetch_viewer_entities(limit: int = 100) -> list[dict[str, Any]]:
    """Fetch entities/teams visible to the authenticated W&B viewer."""
    query = """
    query ViewerEntities($first:Int!) {
      viewer {
        username
        name
        entity
        defaultEntity { name entityType projectCount }
        userEntity { name entityType projectCount }
        teams(first:$first) {
          edges { node { name entityType projectCount } }
        }
      }
    }
    """
    data = graphql(query, {"first": limit})
    viewer = data.get("viewer") or {}
    entities: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(node: dict[str, Any] | None, source: str = "") -> None:
        if not node or not node.get("name") or node["name"] in seen:
            return
        item = dict(node)
        item["source"] = source
        entities.append(item)
        seen.add(item["name"])

    add(viewer.get("defaultEntity"), "default")
    add(viewer.get("userEntity"), "personal")
    for edge in ((viewer.get("teams") or {}).get("edges") or []):
        add(edge.get("node"), "team")
    # Some W&B accounts expose `viewer.entity` even when defaultEntity is sparse.
    if viewer.get("entity") and viewer["entity"] not in seen:
        add({"name": viewer["entity"], "entityType": "unknown", "projectCount": None}, "viewer")
    return entities


def fetch_entity_projects(entity: str, limit: int = 100) -> list[dict[str, Any]]:
    """Fetch recent projects for an entity/team."""
    query = """
    query EntityProjects($entity:String!, $first:Int!) {
      entity(name:$entity) {
        name
        projects(first:$first, order:"-updated_at") {
          edges {
            node { name entityName lastActive totalRuns }
          }
        }
      }
    }
    """
    data = graphql(query, {"entity": entity, "first": limit})
    ent = data.get("entity")
    if not ent:
        raise RuntimeError(f"Could not load W&B entity: {entity}")
    edges = (((ent.get("projects") or {}).get("edges")) or [])
    return [e["node"] for e in edges if e.get("node")]


def fetch_project_run_names(entity: str, project: str, limit: int = 8) -> list[dict[str, Any]]:
    """Fetch recent runs from a W&B project."""
    query = """
    query Runs($entity:String!, $project:String!, $first:Int!) {
      project(name:$project, entityName:$entity) {
        name
        totalRuns
        runCount
        runs(first:$first, order:"-created_at") {
          edges {
            node { id name displayName state createdAt updatedAt historyLineCount }
          }
        }
      }
    }
    """
    data = graphql(query, {"entity": entity, "project": project, "first": limit})
    project_obj = data.get("project")
    if project_obj is None:
        raise RuntimeError(f"W&B returned no project for {entity}/{project}")
    edges = (((project_obj.get("runs") or {}).get("edges")) or [])
    return [e["node"] for e in edges if e.get("node")]


def fetch_project_runs(entity: str, project: str, limit: int = 8, samples: int = 10000) -> list[dict[str, Any]]:
    """Fetch full histories for the most recent project runs."""
    runs = []
    for node in fetch_project_run_names(entity, project, limit):
        try:
            runs.append(fetch_run(entity, project, node["name"], samples=samples))
        except Exception as e:
            node = dict(node)
            node["entity"] = entity
            node["project"] = project
            node["load_error"] = str(e)
            runs.append(node)
    return runs


def as_number(v: Any) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)) and math.isfinite(float(v)):
        return float(v)
    return None


def compact(v: Any, width: int = 12) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if not math.isfinite(v):
            return str(v)
        av = abs(v)
        if av == 0:
            s = "0"
        elif av >= 1e5 or av < 1e-3:
            s = f"{v:.3e}"
        elif av >= 100:
            s = f"{v:.1f}"
        elif av >= 10:
            s = f"{v:.3f}"
        else:
            s = f"{v:.5f}"
        return s[:width]
    if isinstance(v, (list, tuple)):
        if len(v) <= 3:
            return str(list(v))[:width]
        return f"[{len(v)} items]"
    if isinstance(v, dict):
        return f"{{{len(v)} keys}}"
    return str(v).replace("\n", " ")[:width]


def sparkline(values: list[float], width: int) -> str:
    if width <= 0 or not values:
        return ""
    if len(values) > width:
        # Downsample by bucket mean.
        out = []
        for i in range(width):
            a = int(i * len(values) / width)
            b = int((i + 1) * len(values) / width)
            chunk = values[a : max(a + 1, b)]
            out.append(mean(chunk))
        values = out
    lo, hi = min(values), max(values)
    if hi == lo:
        return "─" * len(values)
    chars = []
    for v in values:
        idx = int((v - lo) / (hi - lo) * (len(SPARKS) - 1))
        chars.append(SPARKS[max(0, min(len(SPARKS) - 1, idx))])
    return "".join(chars)


def parse_history(run: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for raw in run.get("history") or []:
        if isinstance(raw, str):
            try:
                rows.append(json.loads(raw))
            except json.JSONDecodeError:
                continue
        elif isinstance(raw, dict):
            rows.append(raw)
    return rows


def build_metrics(run: dict[str, Any]) -> list[dict[str, Any]]:
    rows = parse_history(run)
    summary_raw = run.get("summaryMetrics") or "{}"
    config_raw = run.get("config") or "{}"
    try:
        summary = json.loads(summary_raw) if isinstance(summary_raw, str) else summary_raw
    except Exception:
        summary = {}
    try:
        config = json.loads(config_raw) if isinstance(config_raw, str) else config_raw
    except Exception:
        config = {}

    keys = set()
    hk = run.get("historyKeys") or {}
    if isinstance(hk, dict) and isinstance(hk.get("keys"), dict):
        keys.update(hk["keys"].keys())
    for row in rows:
        keys.update(row.keys())
    if isinstance(summary, dict):
        keys.update(k for k in summary.keys() if not str(k).startswith("_wandb"))

    metrics = []
    for key in sorted(keys):
        vals = [row.get(key) for row in rows if key in row and row.get(key) is not None]
        nums = [as_number(v) for v in vals]
        nums = [v for v in nums if v is not None]
        latest = None
        for row in reversed(rows):
            if key in row and row.get(key) is not None:
                latest = row.get(key)
                break
        if latest is None and isinstance(summary, dict):
            latest = summary.get(key)
        group = key.split("/", 1)[0] if "/" in key else "_system" if key.startswith("_") else "other"
        info = {
            "name": key,
            "group": group,
            "latest": latest,
            "count": len(vals),
            "numeric_count": len(nums),
            "type": "number" if nums else type(latest).__name__ if latest is not None else "unknown",
            "values": nums,
            "min": min(nums) if nums else None,
            "max": max(nums) if nums else None,
            "mean": mean(nums) if nums else None,
            "std": pstdev(nums) if len(nums) > 1 else 0.0 if len(nums) == 1 else None,
        }
        metrics.append(info)

    # Also include top-level config keys as a separate group so the dashboard captures run context.
    if isinstance(config, dict):
        for key, val in sorted(config.items()):
            if str(key).startswith("_"):
                continue
            latest_val = val.get("value") if isinstance(val, dict) and "value" in val else val
            latest_num = as_number(latest_val)
            metrics.append(
                {
                    "name": f"config/{key}",
                    "group": "config",
                    "latest": latest_val,
                    "count": 1,
                    "numeric_count": 1 if latest_num is not None else 0,
                    "type": type(latest_val).__name__,
                    "values": [latest_num] if latest_num is not None else [],
                    "min": latest_num,
                    "max": latest_num,
                    "mean": latest_num,
                    "std": 0.0 if latest_num is not None else None,
                }
            )
    return metrics


def run_label(run: dict[str, Any]) -> str:
    return str(run.get("displayName") or run.get("name") or "?")


def build_multi_metrics(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Union metrics across runs, preserving per-run latest/stats/series."""
    per_run = [build_metrics(r) if not r.get("load_error") else [] for r in runs]
    names = sorted({m["name"] for metrics in per_run for m in metrics})
    out = []
    for name in names:
        slots = []
        group = name.split("/", 1)[0] if "/" in name else "_system" if name.startswith("_") else "other"
        total_count = 0
        latest_nums = []
        for metrics in per_run:
            found = next((m for m in metrics if m["name"] == name), None)
            slots.append(found)
            if found:
                total_count += found.get("count", 0) or 0
                v = as_number(found.get("latest"))
                if v is not None:
                    latest_nums.append(v)
        out.append({
            "name": name, "group": group, "runs": slots, "count": total_count,
            "latest": latest_nums[-1] if latest_nums else None,
            "min": min(latest_nums) if latest_nums else None,
            "max": max(latest_nums) if latest_nums else None,
            "mean": mean(latest_nums) if latest_nums else None,
        })
    return out


def filtered_multi_metrics(metrics: list[dict[str, Any]], search: str, group: str, sort_mode: str) -> list[dict[str, Any]]:
    out = metrics
    if group != "ALL":
        out = [m for m in out if m["group"] == group]
    if search:
        q = search.lower()
        out = [m for m in out if q in m["name"].lower()]
    def val_for_sort(m: dict[str, Any], key: str) -> Any:
        if key == "name": return m["name"]
        if key == "group": return (m["group"], m["name"])
        if key in ("latest", "min", "max", "mean"):
            v = m.get(key)
            return (v is None, -(v or 0), m["name"])
        if key == "count": return (-m["count"], m["name"])
        return m["name"]
    return sorted(out, key=lambda m: val_for_sort(m, sort_mode))


def filtered_metrics(metrics: list[dict[str, Any]], search: str, group: str, sort_mode: str) -> list[dict[str, Any]]:
    out = metrics
    if group != "ALL":
        out = [m for m in out if m["group"] == group]
    if search:
        q = search.lower()
        out = [m for m in out if q in m["name"].lower()]

    def val_for_sort(m: dict[str, Any], key: str) -> Any:
        if key == "name":
            return m["name"]
        if key == "group":
            return (m["group"], m["name"])
        if key in ("latest", "min", "max", "mean"):
            v = as_number(m.get(key)) if key == "latest" else m.get(key)
            return (v is None, -(v or 0), m["name"])
        if key == "count":
            return (-m["count"], m["name"])
        return m["name"]

    return sorted(out, key=lambda m: val_for_sort(m, sort_mode))


def addstr(win: Any, y: int, x: int, s: str, attr: int = 0) -> None:
    h, w = win.getmaxyx()
    if y < 0 or y >= h or x >= w:
        return
    try:
        win.addnstr(y, x, s, max(0, w - x - 1), attr)
    except curses.error:
        pass


def prompt(stdscr: Any, label: str, initial: str = "") -> str:
    h, w = stdscr.getmaxyx()
    curses.echo()
    curses.curs_set(1)
    addstr(stdscr, h - 1, 0, " " * (w - 1))
    addstr(stdscr, h - 1, 0, label)
    stdscr.refresh()
    try:
        s = stdscr.getstr(h - 1, len(label), max(1, w - len(label) - 1)).decode("utf-8", "replace")
    finally:
        curses.noecho()
        curses.curs_set(0)
    return s if s else initial


def draw_help(stdscr: Any) -> None:
    stdscr.clear()
    lines = __doc__.strip().splitlines()
    addstr(stdscr, 0, 0, "W&B Run Metrics TUI Help", curses.A_BOLD)
    for i, line in enumerate(lines[: curses.LINES - 3], 2):
        addstr(stdscr, i, 0, line)
    addstr(stdscr, curses.LINES - 1, 0, "Press any key to return", curses.A_REVERSE)
    stdscr.refresh()
    stdscr.getch()


def tui(stdscr: Any, run_ref: str, refresh_seconds: int = 60) -> None:
    curses.curs_set(0)
    stdscr.nodelay(True)
    if curses.has_colors():
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_CYAN, -1)
        curses.init_pair(2, curses.COLOR_GREEN, -1)
        curses.init_pair(3, curses.COLOR_YELLOW, -1)
        curses.init_pair(4, curses.COLOR_RED, -1)
        curses.init_pair(5, curses.COLOR_MAGENTA, -1)
    cyan = curses.color_pair(1) if curses.has_colors() else curses.A_BOLD
    green = curses.color_pair(2) if curses.has_colors() else curses.A_BOLD
    yellow = curses.color_pair(3) if curses.has_colors() else curses.A_BOLD
    red = curses.color_pair(4) if curses.has_colors() else curses.A_BOLD
    magenta = curses.color_pair(5) if curses.has_colors() else curses.A_BOLD

    entity, project, run_id, url = parse_run_ref(run_ref)
    offset = 0
    search = ""
    sort_idx = 0
    group_idx = 0
    status = "loading…"
    run: dict[str, Any] = {}
    metrics: list[dict[str, Any]] = []
    groups = ["ALL"]
    last_fetch = 0.0

    def refresh() -> None:
        nonlocal run, metrics, groups, status, last_fetch, group_idx
        run = fetch_run(entity, project, run_id)
        metrics = build_metrics(run)
        groups = ["ALL"] + sorted({m["group"] for m in metrics})
        group_idx = min(group_idx, len(groups) - 1)
        last_fetch = time.time()
        status = f"loaded {len(metrics)} metrics from {run.get('historyLineCount')} history rows"

    while True:
        now = time.time()
        if not run or (refresh_seconds and now - last_fetch > refresh_seconds and run.get("state") != "finished"):
            try:
                refresh()
            except Exception as e:
                status = f"ERROR: {e}"

        stdscr.erase()
        h, w = stdscr.getmaxyx()
        sort_mode = SORT_MODES[sort_idx]
        group = groups[group_idx] if groups else "ALL"
        shown = filtered_metrics(metrics, search, group, sort_mode)
        visible_rows = max(1, h - 8)
        offset = max(0, min(offset, max(0, len(shown) - visible_rows)))

        title = f" W&B Run: {run.get('displayName') or run_id} ({entity}/{project}/{run_id}) "
        addstr(stdscr, 0, 0, title[: w - 1], curses.A_REVERSE)
        addstr(stdscr, 1, 0, f"URL: {url}", cyan)
        addstr(
            stdscr,
            2,
            0,
            f"state={run.get('state','?')}  created={run.get('createdAt','?')}  updated={run.get('updatedAt','?')}  rows={run.get('historyLineCount','?')}",
            green if run.get("state") == "finished" else yellow,
        )
        addstr(
            stdscr,
            3,
            0,
            f"metrics={len(metrics)}  shown={len(shown)}  group={group}  search='{search}'  sort={sort_mode}  {status}",
            yellow if status.startswith("ERROR") else 0,
        )
        addstr(stdscr, 4, 0, "Keys: q quit | ↑↓/j/k scroll | / search | Esc clear | g group | m mode | s sort | r refresh | ? help", magenta)
        header = f"{'Metric':<34} {'Latest':>12} {'Min':>11} {'Mean':>11} {'Max':>11} {'N':>5}  Sparkline"
        addstr(stdscr, 5, 0, header[: w - 1], curses.A_BOLD | curses.A_UNDERLINE)

        name_w = min(42, max(22, w - 76))
        spark_w = max(0, w - (name_w + 58))
        for row_i, m in enumerate(shown[offset : offset + visible_rows], start=6):
            name = m["name"]
            if len(name) > name_w:
                name = "…" + name[-(name_w - 1) :]
            latest = compact(m["latest"])
            mn = compact(m["min"])
            avg = compact(m["mean"])
            mx = compact(m["max"])
            n = str(m["count"])
            line = f"{name:<{name_w}} {latest:>12} {mn:>11} {avg:>11} {mx:>11} {n:>5}  "
            attr = cyan if m["name"].startswith("_") else 0
            if m["group"] == "config":
                attr = magenta
            addstr(stdscr, row_i, 0, line[: w - 1], attr)
            addstr(stdscr, row_i, len(line), sparkline(m["values"], spark_w), green)

        if shown:
            pos = f" {offset + 1}-{min(offset + visible_rows, len(shown))}/{len(shown)} "
        else:
            pos = " no metrics "
        addstr(stdscr, h - 2, 0, "─" * (w - 1))
        addstr(stdscr, h - 1, 0, pos + "  " + _dt.datetime.now().strftime("%H:%M:%S"), curses.A_REVERSE)
        stdscr.refresh()

        ch = stdscr.getch()
        if ch == -1:
            time.sleep(0.08)
            continue
        if ch in (ord("q"), ord("Q")):
            return
        if ch in (curses.KEY_DOWN, ord("j")):
            offset += 1
        elif ch in (curses.KEY_UP, ord("k")):
            offset -= 1
        elif ch in (curses.KEY_NPAGE, ord(" ")):
            offset += visible_rows
        elif ch == curses.KEY_PPAGE:
            offset -= visible_rows
        elif ch == curses.KEY_HOME:
            offset = 0
        elif ch == curses.KEY_END:
            offset = max(0, len(shown) - visible_rows)
        elif ch == ord("/"):
            stdscr.nodelay(False)
            search = prompt(stdscr, "Search metric: ", search)
            offset = 0
            stdscr.nodelay(True)
        elif ch == 27:  # Esc
            search = ""
            offset = 0
        elif ch in (ord("g"), ord("G")):
            group_idx = (group_idx + 1) % max(1, len(groups))
            offset = 0
        elif ch in (ord("s"), ord("S")):
            sort_idx = (sort_idx + 1) % len(SORT_MODES)
            offset = 0
        elif ch in (ord("r"), ord("R")):
            try:
                refresh()
            except Exception as e:
                status = f"ERROR: {e}"
        elif ch == ord("?"):
            stdscr.nodelay(False)
            draw_help(stdscr)
            stdscr.nodelay(True)


def downsample_series(values: list[float], width: int) -> list[float]:
    """Downsample numeric series to at most width bucket means."""
    if width <= 0 or not values:
        return []
    if len(values) <= width:
        return values[:]
    out = []
    for i in range(width):
        a = int(i * len(values) / width)
        b = int((i + 1) * len(values) / width)
        chunk = values[a:max(a + 1, b)]
        out.append(mean(chunk))
    return out


ANSI_RE = re.compile(r"\x1b\[([0-9;]*)m")


def add_ansi(stdscr: Any, y: int, x: int, text: str, ansi_color_attrs: dict[int, int]) -> None:
    """Draw a string containing simple ANSI SGR color codes into curses."""
    attr = 0
    pos = 0
    cx = x
    for m in ANSI_RE.finditer(text):
        if m.start() > pos:
            chunk = text[pos:m.start()]
            addstr(stdscr, y, cx, chunk, attr)
            cx += len(chunk)
        codes = [int(c) if c else 0 for c in m.group(1).split(";")]
        if not codes or 0 in codes:
            attr = 0
        # plotext generally emits 38;5;<idx> foreground color codes.
        for i in range(len(codes) - 2):
            if codes[i] == 38 and codes[i + 1] == 5:
                attr = ansi_color_attrs.get(codes[i + 2], attr)
        pos = m.end()
    if pos < len(text):
        addstr(stdscr, y, cx, text[pos:], attr)


def render_plotext_chart(
    slots: list[dict[str, Any] | None],
    width: int,
    height: int,
    labels: list[str],
    colors: list[str],
) -> str | None:
    """Return a plotext multi-line chart string, or None if plotext is unavailable."""
    try:
        import plotext as plt  # type: ignore
    except Exception:
        return None
    try:
        plt.clt()
        plt.cld()
        plt.plotsize(max(20, width), max(6, height))
        plt.theme("clear")
        plotted = 0
        for i, slot in enumerate(slots):
            vals = [float(v) for v in (slot or {}).get("values", []) if isinstance(v, (int, float)) and math.isfinite(float(v))]
            if not vals:
                continue
            # Keep plots readable/fast in small terminals.
            vals = downsample_series(vals, max(20, width - 12))
            xs = list(range(len(vals)))
            plt.plot(xs, vals, label=labels[i] if i < len(labels) else f"R{i+1}", color=colors[i % len(colors)])
            plotted += 1
        if not plotted:
            return None
        plt.grid(True, True)
        return plt.build()
    except Exception:
        return None


def draw_overlay_chart(
    stdscr: Any,
    y: int,
    x: int,
    width: int,
    height: int,
    slots: list[dict[str, Any] | None],
    color_attrs: list[int],
) -> None:
    """Draw a shared-scale colored overlay chart for one metric across runs."""
    if width <= 2 or height <= 1:
        return
    series = []
    all_vals = []
    for slot in slots:
        vals = [v for v in (slot or {}).get("values", []) if isinstance(v, (int, float)) and math.isfinite(float(v))]
        vals = [float(v) for v in vals]
        ds = downsample_series(vals, width)
        series.append(ds)
        all_vals.extend(ds)
    if not all_vals:
        addstr(stdscr, y, x, "·" * min(width, 8))
        return
    lo, hi = min(all_vals), max(all_vals)
    if hi == lo:
        hi = lo + 1.0
    # faint horizontal bounds/zero-ish guides
    for row in range(height):
        addstr(stdscr, y + row, x, "·" * width)
    occupied: dict[tuple[int, int], int] = {}
    for run_i, vals in enumerate(series):
        if not vals:
            continue
        attr = color_attrs[run_i % len(color_attrs)] | curses.A_BOLD
        for col, v in enumerate(vals[:width]):
            yy = int(round((hi - v) / (hi - lo) * (height - 1)))
            yy = max(0, min(height - 1, yy))
            pos = (yy, col)
            if pos in occupied and occupied[pos] != run_i:
                addstr(stdscr, y + yy, x + col, "✕", curses.A_BOLD)
            else:
                occupied[pos] = run_i
                addstr(stdscr, y + yy, x + col, "●", attr)


def choose_list_tui(
    stdscr: Any,
    title: str,
    items: list[dict[str, Any]],
    render_item: Any,
    subtitle: str = "",
) -> dict[str, Any] | None:
    """Small keyboard picker used by the startup flow."""
    curses.curs_set(0)
    stdscr.nodelay(False)
    selected = 0
    offset = 0
    search = ""
    while True:
        h, w = stdscr.getmaxyx()
        filtered = [it for it in items if not search or search.lower() in render_item(it).lower()]
        if selected >= len(filtered):
            selected = max(0, len(filtered) - 1)
        visible = max(1, h - 6)
        offset = max(0, min(offset, max(0, len(filtered) - visible)))
        if selected < offset:
            offset = selected
        if selected >= offset + visible:
            offset = selected - visible + 1

        stdscr.erase()
        addstr(stdscr, 0, 0, f" {title} ", curses.A_REVERSE)
        if subtitle:
            addstr(stdscr, 1, 0, subtitle[: w - 1])
        addstr(stdscr, 2, 0, "↑/↓ j/k move | Enter select | / search | Esc clear | q quit", curses.A_BOLD)
        addstr(stdscr, 3, 0, f"search='{search}'  showing={len(filtered)}/{len(items)}")
        for row, item in enumerate(filtered[offset: offset + visible], start=4):
            idx = offset + row - 4
            prefix = "➜ " if idx == selected else "  "
            attr = curses.A_REVERSE if idx == selected else 0
            addstr(stdscr, row, 0, (prefix + render_item(item))[: w - 1], attr)
        pos = f" {selected + 1 if filtered else 0}-{min(offset + visible, len(filtered))}/{len(filtered)} "
        addstr(stdscr, h - 1, 0, pos, curses.A_REVERSE)
        stdscr.refresh()
        ch = stdscr.getch()
        if ch in (ord("q"), ord("Q")):
            return None
        if ch in (curses.KEY_DOWN, ord("j")):
            selected = min(selected + 1, max(0, len(filtered) - 1))
        elif ch in (curses.KEY_UP, ord("k")):
            selected = max(0, selected - 1)
        elif ch in (curses.KEY_NPAGE, ord(" ")):
            selected = min(selected + visible, max(0, len(filtered) - 1))
        elif ch == curses.KEY_PPAGE:
            selected = max(0, selected - visible)
        elif ch == curses.KEY_HOME:
            selected = 0
        elif ch == curses.KEY_END:
            selected = max(0, len(filtered) - 1)
        elif ch in (10, 13, curses.KEY_ENTER):
            return filtered[selected] if filtered else None
        elif ch == ord("/"):
            search = prompt(stdscr, "Search: ", search)
            selected = 0
            offset = 0
        elif ch == 27:
            search = ""
            selected = 0
            offset = 0


def startup_picker_tui(stdscr: Any) -> str | None:
    """Let the user choose owner/entity and project when no ref is provided."""
    stdscr.erase()
    addstr(stdscr, 0, 0, " Loading W&B entities… ", curses.A_REVERSE)
    addstr(stdscr, 2, 0, "Tip: pass ENTITY/PROJECT or a W&B URL to skip this picker.")
    stdscr.refresh()
    try:
        entities = fetch_viewer_entities()
    except Exception as e:
        stdscr.erase()
        addstr(stdscr, 0, 0, " Could not load W&B entities ", curses.A_REVERSE)
        addstr(stdscr, 2, 0, str(e)[: max(10, stdscr.getmaxyx()[1] - 1)])
        addstr(stdscr, 4, 0, "Set WANDB_API_KEY or pass a project/run URL explicitly. Press any key to exit.")
        stdscr.refresh()
        stdscr.getch()
        return None
    if not entities:
        return None

    entity = choose_list_tui(
        stdscr,
        "Choose W&B owner / entity",
        entities,
        lambda e: f"{e.get('name')}  type={e.get('entityType') or '?'}  projects={e.get('projectCount') if e.get('projectCount') is not None else '?'}  {e.get('source','')}",
    )
    if not entity:
        return None
    entity_name = entity["name"]

    stdscr.erase()
    addstr(stdscr, 0, 0, f" Loading projects for {entity_name}… ", curses.A_REVERSE)
    stdscr.refresh()
    try:
        projects = fetch_entity_projects(entity_name)
    except Exception as e:
        stdscr.erase()
        addstr(stdscr, 0, 0, " Could not load projects ", curses.A_REVERSE)
        addstr(stdscr, 2, 0, str(e)[: max(10, stdscr.getmaxyx()[1] - 1)])
        addstr(stdscr, 4, 0, "Press any key to exit.")
        stdscr.refresh()
        stdscr.getch()
        return None
    if not projects:
        stdscr.erase()
        addstr(stdscr, 0, 0, f" No projects found for {entity_name}. Press any key to exit. ", curses.A_REVERSE)
        stdscr.refresh()
        stdscr.getch()
        return None

    project = choose_list_tui(
        stdscr,
        f"Choose project in {entity_name}",
        projects,
        lambda p: f"{p.get('name')}  runs={p.get('totalRuns') if p.get('totalRuns') is not None else '?'}  lastActive={p.get('lastActive') or '?'}",
        subtitle="Enter opens the project comparison view for recent runs.",
    )
    if not project:
        return None
    return f"{entity_name}/{project['name']}"


def multi_tui(stdscr: Any, project_ref: str, limit: int = 8, refresh_seconds: int = 60) -> None:
    """Project dashboard: compare same metrics across recent runs with colored columns."""
    curses.curs_set(0)
    stdscr.nodelay(True)
    if curses.has_colors():
        curses.start_color(); curses.use_default_colors()
        cols = [curses.COLOR_CYAN, curses.COLOR_GREEN, curses.COLOR_YELLOW, curses.COLOR_MAGENTA, curses.COLOR_BLUE, curses.COLOR_RED, curses.COLOR_WHITE]
        for i, c in enumerate(cols, start=1): curses.init_pair(i, c, -1)
    color_attrs = [(curses.color_pair(i) if curses.has_colors() else curses.A_BOLD) for i in range(1, 8)]
    cyan, yellow, magenta = color_attrs[0], color_attrs[2], color_attrs[3]
    entity, project, url = parse_project_ref(project_ref)
    offset = 0; search = ""; sort_idx = 0; group_idx = 0; status = "loading…"; chart_mode = False
    runs: list[dict[str, Any]] = []; metrics: list[dict[str, Any]] = []; groups = ["ALL"]; last_fetch = 0.0

    def refresh() -> None:
        nonlocal runs, metrics, groups, status, last_fetch, group_idx
        runs = fetch_project_runs(entity, project, limit=limit)
        metrics = build_multi_metrics(runs)
        groups = ["ALL"] + sorted({m["group"] for m in metrics})
        group_idx = min(group_idx, len(groups) - 1)
        loaded = sum(1 for r in runs if not r.get("load_error"))
        status = f"loaded {loaded}/{len(runs)} runs, {len(metrics)} union metrics"
        last_fetch = time.time()

    while True:
        now = time.time()
        if not runs or (refresh_seconds and now - last_fetch > refresh_seconds and any(r.get("state") != "finished" for r in runs)):
            try: refresh()
            except Exception as e: status = f"ERROR: {e}"
        stdscr.erase(); h, w = stdscr.getmaxyx()
        sort_mode = SORT_MODES[sort_idx]
        group = groups[group_idx] if groups else "ALL"
        shown = filtered_multi_metrics(metrics, search, group, sort_mode)
        visible_rows = max(1, h - 9)
        offset = max(0, min(offset, max(0, len(shown) - visible_rows)))
        addstr(stdscr, 0, 0, f" W&B Project: {entity}/{project}  recent runs={limit} ", curses.A_REVERSE)
        addstr(stdscr, 1, 0, f"URL: {url}", cyan)
        addstr(stdscr, 2, 0, f"mode={'chart' if chart_mode else 'table'} metrics={len(metrics)} shown={len(shown)} group={group} search='{search}' sort={sort_mode}  {status}", yellow if status.startswith("ERROR") else 0)
        addstr(stdscr, 3, 0, "Keys: q quit | ↑↓/j/k scroll | / search | Esc clear | g group | m mode | s sort | r refresh | ? help", magenta)
        x = 0
        for i, run in enumerate(runs):
            lab = f"[{i+1}] {run_label(run)} "
            if x + len(lab) >= w: break
            addstr(stdscr, 4, x, lab, color_attrs[i % len(color_attrs)] | curses.A_BOLD)
            x += len(lab) + 1
        if chart_mode:
            numeric_shown = [m for m in shown if any((slot and slot.get("values")) for slot in (m.get("runs") or []))]
            # Fill the available vertical space instead of using a tiny fixed-height plot.
            # Rows 0-5 are header/legend/mode, rows h-2:h are status, so rows 6..h-3
            # are available for chart panels. Each panel uses 1 title row + chart rows + gap.
            available_h = max(8, h - 8)
            min_panel_h = 10
            visible_panels = min(max(1, len(numeric_shown)), max(1, available_h // min_panel_h))
            panel_h = max(min_panel_h, available_h // max(1, visible_panels))
            chart_h = max(6, panel_h - 2)
            offset = max(0, min(offset, max(0, len(numeric_shown) - visible_panels)))
            chart_w = max(10, w - 2)
            addstr(stdscr, 5, 0, "Chart mode: plotext line charts fill available space; shared Y-scale per metric", curses.A_BOLD | curses.A_UNDERLINE)
            for pidx, m in enumerate(numeric_shown[offset: offset + visible_panels]):
                yy = 6 + pidx * panel_h
                title = f"{m['name']}  latest: " + " ".join(
                    f"R{i+1}={compact(slot.get('latest') if slot else None, 10)}"
                    for i, slot in enumerate((m.get("runs") or [])[:len(runs)])
                    if slot
                )
                addstr(stdscr, yy, 0, title[:w-1], curses.A_BOLD)
                slots = (m.get("runs") or [])[:len(runs)]
                labels = [f"R{i+1}" for i in range(len(slots))]
                color_names = ["cyan", "green", "yellow", "magenta", "blue", "red", "white"]
                built = render_plotext_chart(slots, chart_w, chart_h, labels, color_names)
                if built:
                    ansi_map = {6: color_attrs[0], 2: color_attrs[1], 3: color_attrs[2], 5: color_attrs[3], 4: color_attrs[4], 1: color_attrs[5], 7: color_attrs[6], 15: color_attrs[6]}
                    for li, line in enumerate(built.splitlines()[:chart_h]):
                        add_ansi(stdscr, yy + 1 + li, 0, line[:max(0, w * 20)], ansi_map)
                else:
                    draw_overlay_chart(stdscr, yy + 1, 1, chart_w, chart_h, slots, color_attrs)
            pos = f" {offset + 1}-{min(offset + visible_panels, len(numeric_shown))}/{len(numeric_shown)} charts " if numeric_shown else " no numeric charts "
        else:
            name_w = min(36, max(18, w // 4))
            col_w = max(9, min(14, (w - name_w - 4) // max(1, len(runs))))
            max_cols = max(1, (w - name_w - 4) // col_w)
            header = f"{'Metric':<{name_w}} " + " ".join(f"R{i+1:02d}".rjust(col_w) for i in range(min(len(runs), max_cols)))
            addstr(stdscr, 5, 0, header[: w - 1], curses.A_BOLD | curses.A_UNDERLINE)
            for row_i, m in enumerate(shown[offset: offset + visible_rows], start=6):
                name = m["name"]
                if len(name) > name_w: name = "…" + name[-(name_w - 1):]
                addstr(stdscr, row_i, 0, f"{name:<{name_w}} ", magenta if m["group"] == "config" else 0)
                x = name_w + 1
                for i, slot in enumerate((m.get("runs") or [])[:max_cols]):
                    txt = compact(slot.get("latest") if slot else None, col_w - 1) if slot else "·"
                    addstr(stdscr, row_i, x, txt.rjust(col_w), color_attrs[i % len(color_attrs)] if slot else 0)
                    x += col_w + 1
            pos = f" {offset + 1}-{min(offset + visible_rows, len(shown))}/{len(shown)} " if shown else " no metrics "
        addstr(stdscr, h - 2, 0, "─" * (w - 1))
        addstr(stdscr, h - 1, 0, pos + "  " + _dt.datetime.now().strftime("%H:%M:%S"), curses.A_REVERSE)
        stdscr.refresh(); ch = stdscr.getch()
        if ch == -1: time.sleep(0.08); continue
        if ch in (ord("q"), ord("Q")): return
        if ch in (curses.KEY_DOWN, ord("j")): offset += 1
        elif ch in (curses.KEY_UP, ord("k")): offset -= 1
        elif ch in (curses.KEY_NPAGE, ord(" ")): offset += (visible_panels if chart_mode else visible_rows)
        elif ch == curses.KEY_PPAGE: offset -= (visible_panels if chart_mode else visible_rows)
        elif ch == curses.KEY_HOME: offset = 0
        elif ch == curses.KEY_END: offset = max(0, len(shown) - visible_rows)
        elif ch == ord("/"):
            stdscr.nodelay(False); search = prompt(stdscr, "Search metric: ", search); offset = 0; stdscr.nodelay(True)
        elif ch == 27: search = ""; offset = 0
        elif ch in (ord("g"), ord("G")): group_idx = (group_idx + 1) % max(1, len(groups)); offset = 0
        elif ch in (ord("m"), ord("M")): chart_mode = not chart_mode; offset = 0
        elif ch in (ord("s"), ord("S")): sort_idx = (sort_idx + 1) % len(SORT_MODES); offset = 0
        elif ch in (ord("r"), ord("R")):
            try: refresh()
            except Exception as e: status = f"ERROR: {e}"
        elif ch == ord("?"):
            stdscr.nodelay(False); draw_help(stdscr); stdscr.nodelay(True)


def apply_row_limit(metrics: list[dict[str, Any]], top: int) -> list[dict[str, Any]]:
    return metrics[:top] if top and top > 0 else metrics


def print_once(ref: str, runs_limit: int = 8, search: str = "", group: str = "ALL", sort_mode: str = "group", top: int = 0) -> None:
    """Print a pipe-friendly snapshot. Filters mirror the interactive TUI controls."""
    if sort_mode not in SORT_MODES:
        raise SystemExit(f"--sort must be one of: {', '.join(SORT_MODES)}")
    if ref_kind(ref) == "project":
        entity, project, url = parse_project_ref(ref)
        runs = fetch_project_runs(entity, project, limit=runs_limit)
        metrics = apply_row_limit(filtered_multi_metrics(build_multi_metrics(runs), search, group, sort_mode), top)
        print(f"W&B project: {entity}/{project}")
        print(f"URL: {url}")
        print(f"runs={len(runs)} metrics_shown={len(metrics)} search='{search}' group={group} sort={sort_mode}")
        print("runs: " + " | ".join(f"R{i+1}={run_label(r)}" for i, r in enumerate(runs)))
        print(f"{'metric':44} " + " ".join(f"R{i+1:02d}".rjust(13) for i in range(len(runs))))
        print("-" * max(108, 45 + 14 * len(runs)))
        for m in metrics:
            vals = [compact(slot.get("latest") if slot else None, 13).rjust(13) if slot else "·".rjust(13) for slot in m.get("runs", [])]
            print(f"{m['name'][:44]:44} " + " ".join(vals))
        return
    entity, project, run_id, url = parse_run_ref(ref)
    run = fetch_run(entity, project, run_id)
    metrics = apply_row_limit(filtered_metrics(build_metrics(run), search, group, sort_mode), top)
    print(f"W&B run: {run.get('displayName')} ({entity}/{project}/{run_id})")
    print(f"URL: {url}")
    print(f"state={run.get('state')} rows={run.get('historyLineCount')} metrics_shown={len(metrics)} search='{search}' group={group} sort={sort_mode}")
    print(f"{'metric':44} {'latest':>13} {'min':>13} {'mean':>13} {'max':>13} {'n':>5}")
    print("-" * 108)
    for m in metrics:
        print(f"{m['name'][:44]:44} {compact(m['latest'],13):>13} {compact(m['min'],13):>13} {compact(m['mean'],13):>13} {compact(m['max'],13):>13} {m['count']:>5}")


def dump_json(ref: str, path: str, runs_limit: int = 8, search: str = "", group: str = "ALL", sort_mode: str = "group", top: int = 0) -> None:
    """Write parsed run/project stats to JSON, with optional metric filtering."""
    if sort_mode not in SORT_MODES:
        raise SystemExit(f"--sort must be one of: {', '.join(SORT_MODES)}")
    if ref_kind(ref) == "project":
        entity, project, url = parse_project_ref(ref)
        runs = fetch_project_runs(entity, project, limit=runs_limit)
        metrics = apply_row_limit(filtered_multi_metrics(build_multi_metrics(runs), search, group, sort_mode), top)
        serializable = {
            "entity": entity, "project": project, "url": url,
            "filters": {"search": search, "group": group, "sort": sort_mode, "top": top or None},
            "runs": [{k: v for k, v in r.items() if k != "history"} for r in runs],
            "metrics": [{k: v for k, v in m.items() if k != "runs"} | {"runs": [{kk: vv for kk, vv in slot.items() if kk != "values"} if slot else None for slot in m.get("runs", [])]} for m in metrics],
        }
        with open(path, "w", encoding="utf-8") as f: json.dump(serializable, f, indent=2, sort_keys=True)
        print(f"Wrote {len(metrics)} metrics across {len(runs)} runs to {path}")
        return
    entity, project, run_id, url = parse_run_ref(ref)
    run = fetch_run(entity, project, run_id)
    metrics = apply_row_limit(filtered_metrics(build_metrics(run), search, group, sort_mode), top)
    serializable = {k: v for k, v in run.items() if k != "history"}
    serializable["url"] = url
    serializable["filters"] = {"search": search, "group": group, "sort": sort_mode, "top": top or None}
    serializable["metrics"] = [{k: v for k, v in m.items() if k != "values"} for m in metrics]
    with open(path, "w", encoding="utf-8") as f: json.dump(serializable, f, indent=2, sort_keys=True)
    print(f"Wrote {len(metrics)} metrics to {path}")


def main() -> None:
    # Make `--once | head` and other pipelines exit cleanly instead of printing BrokenPipeError.
    if hasattr(signal, "SIGPIPE"):
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    p = argparse.ArgumentParser(description="TUI dashboard for all metrics in W&B run(s)")
    p.add_argument("ref", nargs="?", default=None, help="W&B run URL, project URL, ENTITY/PROJECT/RUN_ID, or ENTITY/PROJECT. If omitted, open an entity/project picker.")
    p.add_argument("--runs", type=int, default=8, help="Project mode: number of recent runs to compare")
    p.add_argument("--once", action="store_true", help="Print a one-shot table instead of launching curses")
    p.add_argument("--json", metavar="PATH", help="Write parsed run metadata/metric stats to JSON and exit")
    p.add_argument("--refresh", type=int, default=60, help="Refresh interval for non-finished runs")
    p.add_argument("--search", default="", help="Filter metric names in --once/--json output")
    p.add_argument("--group", default="ALL", help="Filter metric group in --once/--json output, e.g. train, grad, config, ALL")
    p.add_argument("--sort", choices=SORT_MODES, default="group", help="Sort mode for --once/--json output")
    p.add_argument("--top", type=int, default=0, help="Limit --once/--json to the first N metrics after filtering/sorting")
    args = p.parse_args()
    ref = args.ref
    if ref is None:
        if args.once or args.json:
            raise SystemExit("A W&B ref is required for --once/--json. Omit those flags for the interactive picker.")
        ref = curses.wrapper(startup_picker_tui)
        if not ref:
            raise SystemExit(1)

    if args.json:
        dump_json(ref, args.json, args.runs, args.search, args.group, args.sort, args.top)
    elif args.once:
        print_once(ref, args.runs, args.search, args.group, args.sort, args.top)
    else:
        if ref_kind(ref) == "project": curses.wrapper(multi_tui, ref, args.runs, args.refresh)
        else: curses.wrapper(tui, ref, args.refresh)


if __name__ == "__main__":
    main()
