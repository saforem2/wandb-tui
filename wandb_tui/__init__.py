#!/usr/bin/env python3
"""Terminal dashboard for one or more Weights & Biases runs."""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import re
import signal
import sys
from statistics import mean, pstdev
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_URL = "https://wandb.ai/aurora_gpt/ezpz.examples.fsdp_tp/runs/vrxuo55p"
GRAPHQL_URL = "https://api.wandb.ai/graphql"
SPARKS = "▁▂▃▄▅▆▇█"
SORT_MODES = ("name", "group", "latest", "count", "min", "max", "mean")


def parse_run_ref(ref: str) -> tuple[str, str, str, str]:
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
        "User-Agent": "wandb-tui/1.0",
    }
    api_key = os.environ.get("WANDB_API_KEY")
    if api_key:
        import base64

        token = base64.b64encode(("api:" + api_key).encode()).decode()
        headers["Authorization"] = f"Basic {token}"

    try:
        import requests

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
            f"W&B returned no project for {entity}/{project}. Install `requests` or set WANDB_API_KEY for private projects."
        )
    run = project_obj.get("run")
    if not run:
        raise RuntimeError(f"Run not found: {entity}/{project}/{run_id}")
    run["entity"] = entity
    run["project"] = project
    return run


def fetch_viewer_entities(limit: int = 100) -> list[dict[str, Any]]:
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
    if viewer.get("entity") and viewer["entity"] not in seen:
        add({"name": viewer["entity"], "entityType": "unknown", "projectCount": None}, "viewer")
    return entities


def fetch_entity_projects(entity: str, limit: int = 100) -> list[dict[str, Any]]:
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
        return str(v)[:width]
    if isinstance(v, int):
        return str(v)[:width]
    if isinstance(v, float):
        if not math.isfinite(v):
            return str(v)[:width]
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
        return f"[{len(v)} items]"[:width]
    if isinstance(v, dict):
        return f"{{{len(v)} keys}}"[:width]
    return str(v).replace("\n", " ")[:width]


def sparkline(values: list[float], width: int) -> str:
    if width <= 0 or not values:
        return ""
    if len(values) > width:
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
        metrics.append({
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
        })

    if isinstance(config, dict):
        for key, val in sorted(config.items()):
            if str(key).startswith("_"):
                continue
            latest_val = val.get("value") if isinstance(val, dict) and "value" in val else val
            latest_num = as_number(latest_val)
            metrics.append({
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
            })
    return metrics


def run_label(run: dict[str, Any]) -> str:
    return str(run.get("displayName") or run.get("name") or "?")


def build_multi_metrics(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
            "name": name,
            "group": group,
            "runs": slots,
            "count": total_count,
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
        if key == "name":
            return m["name"]
        if key == "group":
            return (m["group"], m["name"])
        if key in ("latest", "min", "max", "mean"):
            v = m.get(key)
            return (v is None, -(v or 0), m["name"])
        if key == "count":
            return (-m["count"], m["name"])
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


def downsample_series(values: list[float], width: int) -> list[float]:
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


def textual_css() -> str:
    return """
    Screen { layout: vertical; background: #111111; color: #eeeeee; }
    Header, Footer { background: #0f172a; color: #e5e7eb; }
    #meta { dock: top; height: 5; padding: 0 1; color: #d1d5db; background: #111827; }
    #search_input { dock: top; height: 3; margin: 0 1; background: #1f2937; color: #e5e7eb; border: tall #374151; }
    #table { height: 1fr; background: #111111; color: #e5e7eb; }
    #status { dock: bottom; height: 1; color: #d1d5db; background: #111827; }
    DataTable { background: #111111; color: #e5e7eb; }
    DataTable > .datatable--header { background: #1f2937; color: #facc15; text-style: bold; }
    DataTable > .datatable--cursor { background: #1d4ed8; color: #ffffff; text-style: bold; }
    DataTable > .datatable--hover { background: #334155; }
    """


def require_textual() -> None:
    import importlib.util

    if importlib.util.find_spec("textual") is None:
        raise SystemExit("Textual is required for interactive mode. Install with `pip install textual`.")


RUN_COLORS = ("cyan", "green", "yellow", "magenta", "blue", "red", "white")


def rich_cell(value: Any, style: str = "") -> Any:
    from rich.text import Text

    text = compact(value) if not isinstance(value, str) else value
    return Text(str(text), style=style)


def metric_style(metric: dict[str, Any]) -> str:
    if metric["group"] == "config":
        return "magenta"
    if str(metric["name"]).startswith("_"):
        return "cyan"
    return ""


def format_run_meta(run: dict[str, Any], entity: str, project: str, run_id: str, url: str, metrics: list[dict[str, Any]], shown: list[dict[str, Any]], group: str, search: str, sort_mode: str, status: str) -> str:
    title = f"W&B Run: {run.get('displayName') or run_id} ({entity}/{project}/{run_id})"
    state = f"state={run.get('state','?')} created={run.get('createdAt','?')} updated={run.get('updatedAt','?')} rows={run.get('historyLineCount','?')}"
    filters = f"metrics={len(metrics)} shown={len(shown)} group={group} search='{search}' sort={sort_mode} {status}"
    keys = "Keys: q quit | r refresh | / search | Esc clear | g group | s sort column | x reverse"
    return f"{title}\nURL: {url}\n{state}\n{filters}\n{keys}"


def format_project_meta(entity: str, project: str, url: str, limit: int, runs: list[dict[str, Any]], metrics: list[dict[str, Any]], shown: list[dict[str, Any]], group: str, search: str, sort_mode: str, chart_mode: bool, status: str) -> str:
    title = f"W&B Project: {entity}/{project} recent runs={limit}"
    filters = f"mode={'chart' if chart_mode else 'table'} runs={len(runs)} metrics={len(metrics)} shown={len(shown)} group={group} search='{search}' sort={sort_mode} {status}"
    labels = " | ".join(f"R{i+1}={run_label(r)}" for i, r in enumerate(runs))
    keys = "Keys: q quit | r refresh | / search | Esc clear | g group | m mode | s sort column | x reverse"
    return f"{title}\nURL: {url}\n{filters}\n{labels}\n{keys}"


class RunTextualAppMixin:
    CSS = textual_css()
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "refresh_data", "Refresh"),
        ("g", "cycle_group", "Group"),
        ("s", "cycle_sort", "Sort"),
        ("x", "reverse_sort", "Reverse"),
        ("slash", "focus_search", "Search"),
        ("escape", "clear_search", "Clear"),
    ]

    def current_group(self) -> str:
        return self.groups[self.group_idx] if self.groups else "ALL"

    def current_sort(self) -> str:
        return self.sort_columns[self.sort_idx][0]

    def sort_label(self) -> str:
        direction = "desc" if self.sort_reverse else "asc"
        return f"{self.current_sort()} {direction}"

    def action_reverse_sort(self) -> None:
        self.sort_reverse = not self.sort_reverse
        self.render_table()

    def action_focus_search(self) -> None:
        self.query_one("#search_input").focus()

    def action_clear_search(self) -> None:
        self.search = ""
        search_input = self.query_one("#search_input")
        search_input.value = ""
        self.render_table()

    def action_cycle_group(self) -> None:
        self.group_idx = (self.group_idx + 1) % max(1, len(self.groups))
        self.render_table()

    def action_cycle_sort(self) -> None:
        self.sort_idx = (self.sort_idx + 1) % max(1, len(self.sort_columns))
        self.render_table()


def make_run_app(run_ref: str, refresh_seconds: int):
    require_textual()
    from textual.app import App, ComposeResult
    from textual.widgets import DataTable, Footer, Header, Input, Static

    class RunApp(RunTextualAppMixin, App[None]):
        TITLE = "wandb-tui"

        def __init__(self, run_ref: str, refresh_seconds: int) -> None:
            super().__init__()
            self.run_ref = run_ref
            self.refresh_seconds = refresh_seconds
            self.entity, self.project, self.run_id, self.url = parse_run_ref(run_ref)
            self.run: dict[str, Any] = {}
            self.metrics: list[dict[str, Any]] = []
            self.groups = ["ALL"]
            self.group_idx = 0
            self.sort_columns = [("metric", "name"), ("latest", "latest"), ("min", "min"), ("mean", "mean"), ("max", "max"), ("n", "count")]
            self.sort_idx = 0
            self.sort_reverse = False
            self.search = ""
            self.status = "loading…"

        def compose(self) -> ComposeResult:
            yield Header()
            yield Static("loading…", id="meta")
            yield Input(placeholder="Search metrics", id="search_input")
            yield DataTable(id="table", cursor_type="row", zebra_stripes=True)
            yield Static("", id="status")
            yield Footer()

        def on_mount(self) -> None:
            self.query_one("#table", DataTable).add_columns("Metric", "Latest", "Min", "Mean", "Max", "N", "Sparkline")
            self.action_refresh_data()
            if self.refresh_seconds:
                self.set_interval(self.refresh_seconds, self.refresh_if_live)

        def on_input_changed(self, event: Input.Changed) -> None:
            self.search = event.value
            self.render_table()

        def refresh_if_live(self) -> None:
            if self.run and self.run.get("state") == "finished":
                return
            self.action_refresh_data()

        def action_refresh_data(self) -> None:
            try:
                self.status = "refreshing…"
                self.query_one("#status", Static).update(self.status)
                self.run = fetch_run(self.entity, self.project, self.run_id)
                self.metrics = build_metrics(self.run)
                self.groups = ["ALL"] + sorted({m["group"] for m in self.metrics})
                self.group_idx = min(self.group_idx, len(self.groups) - 1)
                self.status = f"loaded {len(self.metrics)} metrics at {_dt.datetime.now().strftime('%H:%M:%S')}"
            except Exception as e:
                self.status = f"ERROR: {e}"
            self.render_table()

        def sort_value(self, metric: dict[str, Any], key: str) -> Any:
            if key == "name":
                return str(metric["name"]).lower()
            raw = metric.get(key)
            value = as_number(raw) if key == "latest" else raw
            if isinstance(value, (int, float)):
                return (0, value, str(metric["name"]).lower())
            return (1, compact(raw), str(metric["name"]).lower())

        def render_table(self) -> None:
            table = self.query_one("#table", DataTable)
            table.clear()
            shown = filtered_metrics(self.metrics, self.search, self.current_group(), "group")
            key = self.sort_columns[self.sort_idx][1]
            shown = sorted(shown, key=lambda m: self.sort_value(m, key), reverse=self.sort_reverse)
            for m in shown:
                style = metric_style(m)
                table.add_row(
                    rich_cell(m["name"], style),
                    rich_cell(m["latest"], "green"),
                    rich_cell(m["min"], "yellow"),
                    rich_cell(m["mean"], "cyan"),
                    rich_cell(m["max"], "red"),
                    rich_cell(str(m["count"]), "white"),
                    rich_cell(sparkline(m["values"], 36), "green"),
                )
            self.query_one("#meta", Static).update(format_run_meta(self.run, self.entity, self.project, self.run_id, self.url, self.metrics, shown, self.current_group(), self.search, self.sort_label(), self.status))
            self.query_one("#status", Static).update("q quit | r refresh | / search | Esc clear | g group | s sort column | x reverse")

    return RunApp(run_ref, refresh_seconds)


def make_project_app(project_ref: str, limit: int, refresh_seconds: int):
    require_textual()
    from textual.app import App, ComposeResult
    from textual.widgets import DataTable, Footer, Header, Input, Static

    class ProjectApp(RunTextualAppMixin, App[None]):
        TITLE = "wandb-tui"
        BINDINGS = RunTextualAppMixin.BINDINGS + [("m", "toggle_mode", "Mode")]

        def __init__(self, project_ref: str, limit: int, refresh_seconds: int) -> None:
            super().__init__()
            self.project_ref = project_ref
            self.limit = limit
            self.refresh_seconds = refresh_seconds
            self.entity, self.project, self.url = parse_project_ref(project_ref)
            self.runs: list[dict[str, Any]] = []
            self.metrics: list[dict[str, Any]] = []
            self.groups = ["ALL"]
            self.group_idx = 0
            self.sort_columns = [("metric", "name")]
            self.sort_idx = 0
            self.sort_reverse = False
            self.search = ""
            self.chart_mode = False
            self.status = "loading…"

        def compose(self) -> ComposeResult:
            yield Header()
            yield Static("loading…", id="meta")
            yield Input(placeholder="Search metrics", id="search_input")
            yield DataTable(id="table", cursor_type="row", zebra_stripes=True)
            yield Static("", id="status")
            yield Footer()

        def on_mount(self) -> None:
            self.rebuild_columns()
            self.action_refresh_data()
            if self.refresh_seconds:
                self.set_interval(self.refresh_seconds, self.refresh_if_live)

        def on_input_changed(self, event: Input.Changed) -> None:
            self.search = event.value
            self.render_table()

        def action_toggle_mode(self) -> None:
            self.chart_mode = not self.chart_mode
            self.rebuild_columns()
            self.render_table()

        def refresh_if_live(self) -> None:
            if self.runs and all(r.get("state") == "finished" for r in self.runs):
                return
            self.action_refresh_data()

        def action_refresh_data(self) -> None:
            try:
                self.status = "refreshing…"
                self.query_one("#status", Static).update(self.status)
                self.runs = fetch_project_runs(self.entity, self.project, limit=self.limit)
                self.metrics = build_multi_metrics(self.runs)
                self.sort_columns = [("metric", "name")] + [(f"R{i+1:02d}", f"run:{i}") for i in range(len(self.runs))]
                self.sort_idx = min(self.sort_idx, len(self.sort_columns) - 1)
                self.groups = ["ALL"] + sorted({m["group"] for m in self.metrics})
                self.group_idx = min(self.group_idx, len(self.groups) - 1)
                loaded = sum(1 for r in self.runs if not r.get("load_error"))
                self.status = f"loaded {loaded}/{len(self.runs)} runs at {_dt.datetime.now().strftime('%H:%M:%S')}"
            except Exception as e:
                self.status = f"ERROR: {e}"
            self.rebuild_columns()
            self.render_table()

        def sort_value(self, metric: dict[str, Any], key: str) -> Any:
            if key == "name":
                return str(metric["name"]).lower()
            if key.startswith("run:"):
                idx = int(key.split(":", 1)[1])
                slots = metric.get("runs") or []
                slot = slots[idx] if idx < len(slots) else None
                raw = slot.get("latest") if slot else None
                value = as_number(raw)
                if isinstance(value, (int, float)):
                    return (0, value, str(metric["name"]).lower())
                return (1, compact(raw), str(metric["name"]).lower())
            return str(metric["name"]).lower()

        def rebuild_columns(self) -> None:
            table = self.query_one("#table", DataTable)
            table.clear(columns=True)
            if self.chart_mode:
                table.add_columns("Metric", "Latest", "Chart")
            else:
                labels = [f"R{i+1:02d}" for i in range(max(1, len(self.runs)))]
                table.add_columns("Metric", *labels)

        def render_table(self) -> None:
            table = self.query_one("#table", DataTable)
            table.clear()
            shown = filtered_multi_metrics(self.metrics, self.search, self.current_group(), "group")
            key = self.sort_columns[self.sort_idx][1]
            shown = sorted(shown, key=lambda m: self.sort_value(m, key), reverse=self.sort_reverse)
            if self.chart_mode:
                for m in shown:
                    slots = m.get("runs") or []
                    values = [v for slot in slots for v in ((slot or {}).get("values") or []) if isinstance(v, (int, float))]
                    latest = " ".join(f"R{i+1}={compact(slot.get('latest') if slot else None, 10)}" for i, slot in enumerate(slots) if slot)
                    table.add_row(rich_cell(m["name"], metric_style(m)), rich_cell(latest, "yellow"), rich_cell(sparkline([float(v) for v in values], 64), "green"))
            else:
                for m in shown:
                    vals = [rich_cell(compact(slot.get("latest") if slot else None, 12), RUN_COLORS[i % len(RUN_COLORS)]) if slot else rich_cell("·", "bright_black") for i, slot in enumerate(m.get("runs") or [])]
                    table.add_row(rich_cell(m["name"], metric_style(m)), *vals)
            self.query_one("#meta", Static).update(format_project_meta(self.entity, self.project, self.url, self.limit, self.runs, self.metrics, shown, self.current_group(), self.search, self.sort_label(), self.chart_mode, self.status))
            self.query_one("#status", Static).update("q quit | r refresh | / search | Esc clear | g group | m mode | s sort column | x reverse")

    return ProjectApp(project_ref, limit, refresh_seconds)


def choose_from_table(title: str, rows: list[dict[str, Any]], columns: list[str], values: Any) -> dict[str, Any] | None:
    require_textual()
    from textual.app import App, ComposeResult
    from textual.widgets import DataTable, Footer, Header, Static

    class PickerApp(App[dict[str, Any] | None]):
        CSS = textual_css()
        BINDINGS = [("q", "quit_none", "Quit"), ("enter", "select", "Select"), ("s", "cycle_sort", "Sort"), ("r", "reverse_sort", "Reverse")]
        TITLE = title

        def __init__(self) -> None:
            super().__init__()
            self.sort_column: int | None = None
            self.sort_reverse = False
            self.visible_rows: list[tuple[int, dict[str, Any]]] = list(enumerate(rows))

        def compose(self) -> ComposeResult:
            yield Header()
            yield Static(title, id="meta")
            yield DataTable(id="table", cursor_type="row", zebra_stripes=True)
            yield Static("", id="status")
            yield Footer()

        def on_mount(self) -> None:
            table = self.query_one("#table", DataTable)
            table.add_columns(*columns)
            self.render_rows()

        def sort_value(self, item: tuple[int, dict[str, Any]]) -> Any:
            if self.sort_column is None:
                return item[0]
            raw = values(item[1])[self.sort_column]
            try:
                return (0, float(str(raw)))
            except ValueError:
                return (1, str(raw).lower())

        def render_rows(self) -> None:
            table = self.query_one("#table", DataTable)
            table.clear()
            self.visible_rows = sorted(enumerate(rows), key=self.sort_value, reverse=self.sort_reverse)
            for index, row in self.visible_rows:
                table.add_row(*(rich_cell(v, RUN_COLORS[i % len(RUN_COLORS)]) for i, v in enumerate(values(row))), key=str(index))
            sort_label = "source order" if self.sort_column is None else f"{columns[self.sort_column]} {'desc' if self.sort_reverse else 'asc'}"
            self.query_one("#status", Static).update(f"Enter select | s sort column | r reverse | q quit | sort={sort_label}")

        def selected_row(self) -> dict[str, Any] | None:
            table = self.query_one("#table", DataTable)
            if table.cursor_row is None or table.cursor_row >= len(self.visible_rows):
                return None
            return self.visible_rows[table.cursor_row][1]

        def action_select(self) -> None:
            self.exit(self.selected_row())

        def action_cycle_sort(self) -> None:
            self.sort_column = 0 if self.sort_column is None else (self.sort_column + 1) % len(columns)
            self.render_rows()

        def action_reverse_sort(self) -> None:
            self.sort_reverse = not self.sort_reverse
            self.render_rows()

        def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
            self.exit(rows[int(str(event.row_key.value))])

        def action_quit_none(self) -> None:
            self.exit(None)

    return PickerApp().run()


def startup_picker_textual() -> str | None:
    entities = fetch_viewer_entities()
    entity = choose_from_table(
        "Choose W&B owner / entity",
        entities,
        ["Name", "Type", "Projects", "Source"],
        lambda e: [str(e.get("name") or ""), str(e.get("entityType") or "?"), str(e.get("projectCount") if e.get("projectCount") is not None else "?"), str(e.get("source") or "")],
    )
    if not entity:
        return None
    entity_name = entity["name"]
    projects = fetch_entity_projects(entity_name)
    project = choose_from_table(
        f"Choose project in {entity_name}",
        projects,
        ["Name", "Runs", "Last Active"],
        lambda p: [str(p.get("name") or ""), str(p.get("totalRuns") if p.get("totalRuns") is not None else "?"), str(p.get("lastActive") or "?")],
    )
    if not project:
        return None
    return f"{entity_name}/{project['name']}"


def apply_row_limit(metrics: list[dict[str, Any]], top: int) -> list[dict[str, Any]]:
    return metrics[:top] if top and top > 0 else metrics


def print_once(ref: str, runs_limit: int = 8, search: str = "", group: str = "ALL", sort_mode: str = "group", top: int = 0) -> None:
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
    if sort_mode not in SORT_MODES:
        raise SystemExit(f"--sort must be one of: {', '.join(SORT_MODES)}")
    if ref_kind(ref) == "project":
        entity, project, url = parse_project_ref(ref)
        runs = fetch_project_runs(entity, project, limit=runs_limit)
        metrics = apply_row_limit(filtered_multi_metrics(build_multi_metrics(runs), search, group, sort_mode), top)
        serializable = {
            "entity": entity,
            "project": project,
            "url": url,
            "filters": {"search": search, "group": group, "sort": sort_mode, "top": top or None},
            "runs": [{k: v for k, v in r.items() if k != "history"} for r in runs],
            "metrics": [{k: v for k, v in m.items() if k != "runs"} | {"runs": [{kk: vv for kk, vv in slot.items() if kk != "values"} if slot else None for slot in m.get("runs", [])]} for m in metrics],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, sort_keys=True)
        print(f"Wrote {len(metrics)} metrics across {len(runs)} runs to {path}")
        return
    entity, project, run_id, url = parse_run_ref(ref)
    run = fetch_run(entity, project, run_id)
    metrics = apply_row_limit(filtered_metrics(build_metrics(run), search, group, sort_mode), top)
    serializable = {k: v for k, v in run.items() if k != "history"}
    serializable["url"] = url
    serializable["filters"] = {"search": search, "group": group, "sort": sort_mode, "top": top or None}
    serializable["metrics"] = [{k: v for k, v in m.items() if k != "values"} for m in metrics]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, sort_keys=True)
    print(f"Wrote {len(metrics)} metrics to {path}")


def main() -> None:
    if hasattr(signal, "SIGPIPE"):
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    p = argparse.ArgumentParser(description="TUI dashboard for all metrics in W&B run(s)")
    p.add_argument("ref", nargs="?", default=None, help="W&B run URL, project URL, ENTITY/PROJECT/RUN_ID, or ENTITY/PROJECT. If omitted, open an entity/project picker.")
    p.add_argument("--runs", type=int, default=8, help="Project mode: number of recent runs to compare")
    p.add_argument("--once", action="store_true", help="Print a one-shot table instead of launching the TUI")
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
        try:
            ref = startup_picker_textual()
        except Exception as e:
            raise SystemExit(f"Could not open picker: {e}") from e
        if not ref:
            raise SystemExit(1)

    if args.json:
        dump_json(ref, args.json, args.runs, args.search, args.group, args.sort, args.top)
    elif args.once:
        print_once(ref, args.runs, args.search, args.group, args.sort, args.top)
    else:
        if ref_kind(ref) == "project":
            make_project_app(ref, args.runs, args.refresh).run()
        else:
            make_run_app(ref, args.refresh).run()


if __name__ == "__main__":
    main()
