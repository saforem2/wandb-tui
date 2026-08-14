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
        # Never slice: "Tru" is a lie, "True" that overflows by one is not.
        return str(v)
    if isinstance(v, int):
        s = str(v)
        if len(s) <= width:
            return s
        # Slicing digits off the right produces a different, plausible-looking
        # number. Fall back to scientific notation instead, shedding mantissa
        # precision (which is lossy but honest) until it fits.
        for prec in range(max(0, width - 6), -1, -1):
            s = f"{v:.{prec}e}"
            if len(s) <= width:
                return s
        return s
    if isinstance(v, float):
        if not math.isfinite(v):
            return str(v)
        av = abs(v)
        if av == 0:
            return "0"
        if av >= 1e5 or av < 1e-3:
            # Shed mantissa precision until it fits. Slicing instead would cut
            # the exponent off ("9.912e+05" -> "9.912e+"), which reads as a
            # different number entirely.
            for prec in range(3, -1, -1):
                s = f"{v:.{prec}e}"
                if len(s) <= width:
                    return s
            return s
        if av >= 100:
            s = f"{v:.1f}"
        elif av >= 10:
            s = f"{v:.3f}"
        else:
            s = f"{v:.5f}"
        if len(s) <= width:
            return s
        # Trim fractional digits rather than truncating mid-number.
        whole = s.split(".", 1)[0]
        keep = width - len(whole) - 1
        return f"{v:.{keep}f}" if keep > 0 else whole
    if isinstance(v, (list, tuple)):
        if len(v) <= 3:
            return str(list(v))[:width]
        return f"[{len(v)} items]"[:width]
    if isinstance(v, dict):
        return f"{{{len(v)} keys}}"[:width]
    s = str(v).replace("\n", " ").replace("\r", " ")
    if len(s) > width:
        return s[: max(0, width - 1)] + "…" if width >= 1 else ""
    return s


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


def run_config(run: dict[str, Any]) -> dict[str, Any]:
    """Flatten a run's config into {key: value}, unwrapping W&B's {"value": x}."""
    raw = run.get("config") or "{}"
    try:
        cfg = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        return {}
    if not isinstance(cfg, dict):
        return {}
    out: dict[str, Any] = {}
    for key, val in cfg.items():
        if str(key).startswith("_"):
            continue
        out[str(key)] = val.get("value") if isinstance(val, dict) and "value" in val else val
    return out


def run_filter_field(run: dict[str, Any], key: str) -> Any:
    """Resolve a filter key against a run.

    Bare keys and `config.x` / `config/x` read from the run config; a few
    run-level attributes are exposed under the reserved `run.` prefix so that
    `state`, `name`, and friends are filterable even when a config key shadows
    them.
    """
    for prefix in ("config.", "config/"):
        if key.startswith(prefix):
            return run_config(run).get(key[len(prefix):])
    if key.startswith("run."):
        attr = key[4:]
        if attr in ("name", "displayName", "label"):
            return run_label(run)
        return run.get(attr)
    cfg = run_config(run)
    if key in cfg:
        return cfg[key]
    # Fall back to run-level attributes so `state=finished` works unprefixed.
    if key in ("name", "displayName", "label"):
        return run_label(run)
    return run.get(key)


# Longest-first so that ">=" is matched before ">", and "!=" before "=".
FILTER_OPS = ("!~", ">=", "<=", "!=", "~", "=", ">", "<")
_FILTER_SPLIT = re.compile(r"\s+(?=[^\s]+\s*(?:" + "|".join(re.escape(o) for o in FILTER_OPS) + "))")


def parse_run_filters(expr: str) -> list[tuple[str, str, str]]:
    """Parse `lr>=0.001 model~llama state=finished` into (key, op, value) triples.

    Space-separated terms are AND-ed, matching how W&B workspace filters
    compose. Quoted values may contain spaces.
    """
    terms: list[tuple[str, str, str]] = []
    for raw in _split_filter_terms(expr):
        raw = raw.strip()
        if not raw:
            continue
        for op in FILTER_OPS:
            idx = raw.find(op)
            if idx > 0:
                key = raw[:idx].strip()
                value = raw[idx + len(op):].strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                    value = value[1:-1]
                if not value and op not in ("=", "!="):
                    # `lr>` mid-typing. Only = and != are meaningful against an
                    # empty value (match blank/missing); the rest are unfinished.
                    raise ValueError(f"filter term {raw!r} is missing a value")
                if key:
                    terms.append((key, op, value))
                break
        else:
            raise ValueError(f"filter term {raw!r} needs one of: {', '.join(FILTER_OPS)}")
    return terms


def _split_filter_terms(expr: str) -> list[str]:
    """Split on whitespace, but keep quoted runs of text together."""
    out, buf, quote = [], [], ""
    for ch in expr:
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = ""
        elif ch in "\"'":
            quote = ch
            buf.append(ch)
        elif ch.isspace():
            if buf:
                out.append("".join(buf))
                buf = []
        else:
            buf.append(ch)
    if buf:
        out.append("".join(buf))
    return out


def match_filter_term(actual: Any, op: str, expected: str) -> bool:
    """Apply one comparison. Numeric when both sides parse as numbers."""
    if op in ("~", "!~"):
        hit = expected.lower() in str("" if actual is None else actual).lower()
        return hit if op == "~" else not hit

    a_num = as_number(actual)
    try:
        e_num: float | None = float(expected)
    except (TypeError, ValueError):
        e_num = None

    if a_num is not None and e_num is not None:
        if op == "=":
            return a_num == e_num
        if op == "!=":
            return a_num != e_num
        if op == ">":
            return a_num > e_num
        if op == "<":
            return a_num < e_num
        if op == ">=":
            return a_num >= e_num
        return a_num <= e_num

    # Non-numeric: equality is a case-insensitive string compare; ordering
    # comparisons are meaningless, so they exclude the run rather than raise.
    a_str = str("" if actual is None else actual).strip().lower()
    e_str = expected.strip().lower()
    if op == "=":
        return a_str == e_str
    if op == "!=":
        return a_str != e_str
    return False


def filter_runs(runs: list[dict[str, Any]], expr: str) -> list[dict[str, Any]]:
    """Keep runs matching every term in `expr` (empty expr keeps everything)."""
    if not expr or not expr.strip():
        return list(runs)
    terms = parse_run_filters(expr)
    if not terms:
        return list(runs)
    return [
        run
        for run in runs
        if all(match_filter_term(run_filter_field(run, key), op, value) for key, op, value in terms)
    ]


def config_filter_keys(runs: list[dict[str, Any]]) -> list[str]:
    """Config keys present across the loaded runs, for hints/completion."""
    keys: set[str] = set()
    for run in runs:
        keys.update(run_config(run).keys())
    return sorted(keys)


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
    /* Neither #meta nor #search_input may dock: two widgets docked to the same
       edge overlap, and the 3-row input was covering the top 3 of the meta
       panel's 5 lines (title, URL, state). Let the vertical layout stack them. */
    #meta { height: auto; padding: 0 1; color: #d1d5db; background: #111827; }
    #search_input, #filter_input { height: 3; margin: 0 1; background: #1f2937; color: #e5e7eb; border: tall #374151; }
    #filter_input { border: tall #4b5563; }
    #table { height: 1fr; background: #111111; color: #e5e7eb; }
    #charts { height: 1fr; background: #111111; color: #e5e7eb; display: none; }
    #chart_text { padding: 0 1; }
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

KEYS_HINT_RUN = "Keys: q quit | r refresh | / search | Esc clear | g group | s sort column | x reverse"
KEYS_HINT_PROJECT = "Keys: q quit | r refresh | / search | f filter runs | Esc clear | g group | m mode | s sort column | x reverse"


CELL_WIDTH = 12
NAME_CELL_WIDTH = 60
PICKER_CELL_WIDTH = 120

# Column budgeting for the data tables. DataTable sizes each column to its
# widest cell, so emitting fixed-width cells overflows narrow terminals and
# silently drops the rightmost columns. Instead we fit to the available width.
# DataTable pads cell_padding (default 1) on BOTH sides of every column, so a
# column costs content + 2 on screen.
COL_PAD = 2
STAT_COL_WIDTH = 9        # min/mean/max
LATEST_COL_WIDTH = 12
COUNT_COL_WIDTH = 4
MIN_NAME_WIDTH = 14
MAX_NAME_WIDTH = 44
MIN_SPARK_WIDTH = 8
MAX_SPARK_WIDTH = 36
# "9.912e+05" is 9 chars, "-9.912e+05" is 10 -- narrower than this and
# scientific-notation values lose their exponent.
MIN_RUN_COL_WIDTH = 9
MAX_RUN_COL_WIDTH = 12


def fit_run_metric_widths(total: int) -> tuple[int, int, bool]:
    """Pick (name_width, sparkline_width, show_stats) for the metrics table.

    Columns are Metric, Latest, [Min, Mean, Max,] N, [Sparkline]. The name and
    sparkline absorb slack; when the terminal is too narrow even for the fixed
    stat columns we drop Min/Mean/Max rather than let DataTable clip them off
    the right edge, keeping Metric/Latest/N always readable.
    """
    # Metric + Latest + N are always present; each column costs COL_PAD extra.
    base = (LATEST_COL_WIDTH + COL_PAD) + (COUNT_COL_WIDTH + COL_PAD) + COL_PAD
    stats_cost = 3 * (STAT_COL_WIDTH + COL_PAD)
    show_stats = total >= MIN_NAME_WIDTH + base + stats_cost
    fixed = base + (stats_cost if show_stats else 0)
    slack = max(0, total - fixed)
    spark_cost = MIN_SPARK_WIDTH + COL_PAD
    name = max(MIN_NAME_WIDTH, min(MAX_NAME_WIDTH, slack - spark_cost))
    spark = max(0, min(MAX_SPARK_WIDTH, slack - name - COL_PAD))
    if spark < MIN_SPARK_WIDTH:
        # Not enough room for a legible sparkline: give the space to the name.
        name = max(MIN_NAME_WIDTH, min(MAX_NAME_WIDTH, slack))
        spark = 0
    return name, spark, show_stats


def fit_project_widths(total: int, run_count: int) -> tuple[int, int, int]:
    """Pick (name_width, run_col_width, visible_runs) for the project table.

    Returns how many run columns actually fit, so the caller can add only
    those rather than emitting columns that get clipped off the right edge.
    """
    run_count = max(1, run_count)
    # Start from a modest name column so run columns get first claim on the
    # width, then hand leftover space back to the name at the end.
    name = max(MIN_NAME_WIDTH, min(MAX_NAME_WIDTH, total // 3))
    avail = max(0, total - (name + COL_PAD))
    per = MIN_RUN_COL_WIDTH + COL_PAD
    visible = max(1, min(run_count, avail // per if per else 1))

    if visible < run_count:
        # Not all runs fit at the comfortable name width. Shrink the name
        # column toward its minimum to buy more run columns before giving up
        # on showing them.
        min_avail = max(0, total - (MIN_NAME_WIDTH + COL_PAD))
        wider = max(1, min(run_count, min_avail // per if per else 1))
        if wider > visible:
            visible = wider
            name = MIN_NAME_WIDTH
            avail = min_avail

    # Spend whatever is left widening the run columns (up to the cap) so
    # scientific-notation values are not needlessly cramped.
    col = max(MIN_RUN_COL_WIDTH, min(MAX_RUN_COL_WIDTH, (avail // visible) - COL_PAD))
    # Any remainder goes back to the name column.
    used = visible * (col + COL_PAD)
    name = max(name, min(MAX_NAME_WIDTH, name + max(0, avail - used)))
    return name, col, visible


def rich_cell(value: Any, style: str = "", width: int = CELL_WIDTH) -> Any:
    """Render one table cell. Strings are clamped and newline-stripped too --
    an unbounded config value would otherwise blow every other column
    off-screen, and an embedded newline would break the row."""
    from rich.text import Text

    return Text(compact(value, width), style=style)


def metric_style(metric: dict[str, Any]) -> str:
    if metric["group"] == "config":
        return "magenta"
    if str(metric["name"]).startswith("_"):
        return "cyan"
    return ""


ANSI_RE = re.compile(r"\x1b\[([0-9;]*)m")
ANSI_256_TO_RICH = {1: "red", 2: "green", 3: "yellow", 4: "blue", 5: "magenta", 6: "cyan", 7: "white", 15: "white"}


def ansi_to_text(text: str) -> Any:
    from rich.text import Text

    out = Text()
    style = ""
    pos = 0
    for match in ANSI_RE.finditer(text):
        if match.start() > pos:
            out.append(text[pos:match.start()], style=style)
        codes = [int(c) if c else 0 for c in match.group(1).split(";")]
        if not codes or 0 in codes:
            style = ""
        if 1 in codes and "bold" not in style.split():
            # Guard against repeated \x1b[1m producing "bold bold bold ...".
            style = (style + " bold").strip()
        for i in range(len(codes) - 2):
            if codes[i] == 38 and codes[i + 1] == 5:
                style = ANSI_256_TO_RICH.get(codes[i + 2], style)
        pos = match.end()
    if pos < len(text):
        out.append(text[pos:], style=style)
    return out


def render_plotext_chart(slots: list[dict[str, Any] | None], width: int, height: int, labels: list[str]) -> str | None:
    try:
        import contextlib
        import io
        import plotext as plt
    except Exception:
        return None
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            plt.clt()
            plt.cld()
            plt.plotsize(max(24, width), max(6, height))
            plt.theme("clear")
            plotted = 0
            for i, slot in enumerate(slots):
                vals = [float(v) for v in (slot or {}).get("values", []) if isinstance(v, (int, float)) and math.isfinite(float(v))]
                if not vals:
                    continue
                # Braille packs 2x4 subpixels per cell, so keep 2 samples per
                # column instead of 1 -- otherwise the extra resolution is
                # wasted on a series we already flattened.
                vals = downsample_series(vals, max(40, (width - 12) * 2))
                plt.plot(
                    list(range(len(vals))),
                    vals,
                    label=labels[i] if i < len(labels) else f"R{i+1}",
                    color=RUN_COLORS[i % len(RUN_COLORS)],
                    marker="braille",
                )
                plotted += 1
            if not plotted:
                return None
            plt.grid(True, True)
            return plt.build()
    except Exception:
        return None


# A braille cell packs a 2-wide x 4-tall dot matrix. Bit values are laid out
# in two columns of four, with the low nibble ordered top-to-bottom on the left
# (0x01/0x02/0x04 then 0x40 at the bottom) and mirrored on the right.
BRAILLE_BASE = 0x2800
BRAILLE_DOTS = ((0x01, 0x08), (0x02, 0x10), (0x04, 0x20), (0x40, 0x80))
BRAILLE_COLS = 2
BRAILLE_ROWS = 4


def overlay_chart_text(slots: list[dict[str, Any] | None], width: int, height: int) -> Any:
    """Braille fallback chart, used when plotext is unavailable.

    Each terminal cell carries a 2x4 braille matrix, so the effective plotting
    surface is 8x the cell grid. Dots belonging to one run are OR-ed into a
    shared cell; where two runs land in the same cell the dots merge and the
    cell is drawn in a neutral colour, since a single glyph can only carry one.
    """
    from rich.text import Text

    width = max(8, width)
    height = max(3, height)
    px_w = width * BRAILLE_COLS
    px_h = height * BRAILLE_ROWS

    series = []
    all_vals = []
    for slot in slots:
        vals = [float(v) for v in (slot or {}).get("values", []) if isinstance(v, (int, float)) and math.isfinite(float(v))]
        ds = downsample_series(vals, px_w)
        series.append(ds)
        all_vals.extend(ds)
    if not all_vals:
        return Text("·" * min(width, 16), style="bright_black")
    lo, hi = min(all_vals), max(all_vals)
    if hi == lo:
        hi = lo + 1.0

    # bits[(cell_row, cell_col)] -> braille bitmask; owner tracks the run index
    # so a collision can be detected by index rather than by colour (RUN_COLORS
    # wraps at 7, so runs 0 and 7 would otherwise alias).
    bits: dict[tuple[int, int], int] = {}
    owner: dict[tuple[int, int], int] = {}

    for run_i, vals in enumerate(series):
        prev_py = None
        for px, value in enumerate(vals[:px_w]):
            py = int(round((hi - value) / (hi - lo) * (px_h - 1)))
            py = max(0, min(px_h - 1, py))
            # Join consecutive samples vertically so steep segments read as a
            # continuous line instead of disconnected dots.
            span = range(py, py + 1) if prev_py is None else range(min(prev_py, py), max(prev_py, py) + 1)
            for y in span:
                key = (y // BRAILLE_ROWS, px // BRAILLE_COLS)
                bits[key] = bits.get(key, 0) | BRAILLE_DOTS[y % BRAILLE_ROWS][px % BRAILLE_COLS]
                if owner.setdefault(key, run_i) != run_i:
                    owner[key] = -1  # shared cell
            prev_py = py

    out = Text()
    for row in range(height):
        for col in range(width):
            mask = bits.get((row, col))
            if not mask:
                out.append("·", style="bright_black")
                continue
            who = owner.get((row, col), -1)
            style = "bold white" if who < 0 else RUN_COLORS[who % len(RUN_COLORS)]
            out.append(chr(BRAILLE_BASE + mask), style=style)
        out.append("\n")
    return out


def format_run_legend(runs: list[dict[str, Any]]) -> Any:
    from rich.text import Text

    text = Text()
    for i, run in enumerate(runs):
        if i:
            text.append("  ")
        text.append(f"R{i+1}={run_label(run)}", style=f"bold {RUN_COLORS[i % len(RUN_COLORS)]}")
    return text


def format_run_meta(run: dict[str, Any], entity: str, project: str, run_id: str, url: str, metrics: list[dict[str, Any]], shown: list[dict[str, Any]], group: str, search: str, sort_mode: str, status: str) -> Any:
    # Must return a Text, not a str: Static.update() parses str content as
    # Textual markup, so a run named "sweep[lr=1e-3]" -- or an error message
    # containing brackets -- would raise MarkupError and kill the app.
    from rich.text import Text

    text = Text()
    text.append(f"W&B Run: {run.get('displayName') or run_id} ({entity}/{project}/{run_id})\n", style="bold white")
    text.append(f"URL: {url}\n", style="cyan")
    text.append(
        f"state={run.get('state', '?')}  created={run.get('createdAt', '?')}  updated={run.get('updatedAt', '?')}  rows={run.get('historyLineCount', '?')}\n",
        style="green" if run.get("state") == "finished" else "yellow",
    )
    text.append(
        f"metrics={len(metrics)}  shown={len(shown)}  group={group}  search='{search}'  sort={sort_mode}  {status}\n",
        style="yellow" if status.startswith("ERROR") else "white",
    )
    text.append(KEYS_HINT_RUN, style="magenta")
    return text


def format_project_meta(entity: str, project: str, url: str, limit: int, runs: list[dict[str, Any]], metrics: list[dict[str, Any]], shown: list[dict[str, Any]], group: str, search: str, sort_mode: str, chart_mode: bool, status: str, run_filter: str = "", total_runs: int | None = None, filter_error: str = "", visible_runs: int | None = None) -> Any:
    from rich.text import Text

    text = Text()
    text.append(f"W&B Project: {entity}/{project}  recent runs={limit}\n", style="bold white")
    text.append(f"URL: {url}\n", style="cyan")
    text.append(f"mode={'chart' if chart_mode else 'table'}  metrics={len(metrics)}  shown={len(shown)}  group={group}  search='{search}'  sort={sort_mode}  {status}\n", style="yellow" if status.startswith("ERROR") else "white")
    if filter_error:
        text.append(f"filter error: {filter_error}\n", style="bold red")
    elif run_filter:
        total = len(runs) if total_runs is None else total_runs
        text.append(f"runs: {len(runs)}/{total} matching  filter='{run_filter}'\n", style="bold green")
    if visible_runs is not None and 0 < visible_runs < len(runs):
        # Say so rather than silently clipping columns off the right edge.
        text.append(
            f"showing R01-R{visible_runs:02d} of {len(runs)} runs (widen the terminal for more)\n",
            style="bold yellow",
        )
    text.append_text(format_run_legend(runs))
    text.append("\n")
    text.append(KEYS_HINT_PROJECT, style="magenta")
    return text


# Textual's DOMNode._merge_bindings() only collects BINDINGS from bases that
# are themselves DOMNode subclasses, so a plain mixin's BINDINGS are silently
# dropped. Keep them in a module constant and assign them into each App
# subclass's own class body, where the merge will actually see them.
BASE_BINDINGS = [
    ("q", "quit", "Quit"),
    ("r", "refresh_data", "Refresh"),
    ("g", "cycle_group", "Group"),
    ("s", "cycle_sort", "Sort"),
    ("x", "reverse_sort", "Reverse"),
    ("slash", "focus_search", "Search"),
    ("escape", "clear_search", "Clear"),
]


class RunTextualAppMixin:
    CSS = textual_css()
    # Don't auto-focus the search Input: it would swallow every single-letter
    # binding (q/r/g/s/x/m) as literal text before the action could fire.
    AUTO_FOCUS = "#table"

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

    def schedule_render(self, callback: Any = None) -> None:
        """Debounce a re-render (or another local recompute) while typing."""
        existing = getattr(self, "render_timer", None)
        if existing is not None:
            existing.stop()
        self.render_timer = self.set_timer(0.18, callback or self.render_table)

    def action_focus_search(self) -> None:
        self.query_one("#search_input").focus()

    def action_clear_search(self) -> None:
        # Clear whichever box has focus; if focus is elsewhere (e.g. the
        # results table), clear both -- Esc from the results means "drop all
        # filtering", not "do nothing".
        focused_id = getattr(self.focused, "id", None)
        targets = ("search_input", "filter_input")
        selective = focused_id in targets
        cleared_filter = False
        for selector in ("#search_input", "#filter_input"):
            if selective and focused_id != selector.lstrip("#"):
                continue
            try:
                self.query_one(selector).value = ""
            except Exception:
                continue
            if selector == "#search_input":
                self.search = ""
            else:
                self.run_filter = ""
                cleared_filter = True
        # Hand focus back to the table so the single-letter bindings work again
        # instead of typing into the box the user just cleared.
        self.focus_results_pane()
        if cleared_filter and hasattr(self, "apply_run_filter"):
            self.apply_run_filter()
        else:
            self.render_table()

    def focus_results_pane(self) -> None:
        """Focus whichever results widget is currently visible."""
        for selector in ("#charts", "#table"):
            try:
                widget = self.query_one(selector)
            except Exception:
                continue
            if widget.display:
                widget.focus()
                return

    def action_cycle_group(self) -> None:
        self.group_idx = (self.group_idx + 1) % max(1, len(self.groups))
        self.render_table()

    def action_cycle_sort(self) -> None:
        self.sort_idx = (self.sort_idx + 1) % max(1, len(self.sort_columns))
        self.render_table()


def fitted_data_table():
    """DataTable that reports its own width so the app can re-budget columns.

    The App-level on_resize fires before child layout is recomputed (and even
    call_after_refresh still sees the stale size), so the table has to be the
    one to announce its new width.
    """
    from textual.widgets import DataTable

    class FittedDataTable(DataTable):
        def on_resize(self, event: Any) -> None:
            # DataTable defines no on_resize of its own, so there is nothing to
            # delegate to; the event still propagates normally afterwards.
            refit = getattr(self.app, "refit_columns", None)
            if refit is not None:
                refit(event.size.width)

    return FittedDataTable


def make_run_app(run_ref: str, refresh_seconds: int):
    require_textual()
    from textual.app import App, ComposeResult
    from textual.widgets import DataTable, Footer, Header, Input, Static

    FittedDataTable = fitted_data_table()

    class RunApp(RunTextualAppMixin, App[None]):
        TITLE = "wandb-tui"
        BINDINGS = list(BASE_BINDINGS)

        def __init__(self, run_ref: str, refresh_seconds: int) -> None:
            super().__init__()
            self.run_ref = run_ref
            self.refresh_seconds = refresh_seconds
            self.entity, self.project, self.run_id, self.url = parse_run_ref(run_ref)
            self.run_data: dict[str, Any] = {}
            self.metrics: list[dict[str, Any]] = []
            self.groups = ["ALL"]
            self.group_idx = 0
            self.sort_columns = [("metric", "name"), ("latest", "latest"), ("min", "min"), ("mean", "mean"), ("max", "max"), ("n", "count")]
            self.sort_idx = 0
            self.sort_reverse = False
            self.search = ""
            self.status = "loading…"
            self.render_timer = None
            self.refresh_in_flight = False

        def compose(self) -> ComposeResult:
            yield Header()
            yield Static("loading…", id="meta")
            yield Input(placeholder="Search metrics", id="search_input")
            yield FittedDataTable(id="table", cursor_type="row", zebra_stripes=True)
            yield Static("", id="status")
            yield Footer()

        def on_mount(self) -> None:
            self.rebuild_columns()
            self.action_refresh_data()
            if self.refresh_seconds:
                self.set_interval(self.refresh_seconds, self.refresh_if_live)

        def on_input_changed(self, event: Input.Changed) -> None:
            self.search = event.value
            self.schedule_render()

        def refresh_if_live(self) -> None:
            if self.run_data and self.run_data.get("state") == "finished":
                return
            self.action_refresh_data()

        def action_refresh_data(self) -> None:
            if self.refresh_in_flight:
                return
            self.refresh_in_flight = True
            self.status = "refreshing…"
            self.render_table()
            # fetch_run is blocking HTTP with a 90s timeout; running it inline
            # would freeze the whole UI (no repaint, no keys) until it returns.
            self.run_worker(self.fetch_in_thread, thread=True, exclusive=True)

        def fetch_in_thread(self) -> None:
            try:
                run_data = fetch_run(self.entity, self.project, self.run_id)
                metrics = build_metrics(run_data)
                groups = ["ALL"] + sorted({m["group"] for m in metrics})
                status = f"loaded {len(metrics)} metrics at {_dt.datetime.now().strftime('%H:%M:%S')}"
            except Exception as e:
                self.call_from_thread(self.apply_refresh_error, str(e))
                return
            self.call_from_thread(self.apply_refresh_result, run_data, metrics, groups, status)

        def apply_refresh_result(self, run_data, metrics, groups, status) -> None:
            # Assign only once every piece succeeded, so a failure can never
            # leave state half-updated.
            self.run_data = run_data
            self.metrics = metrics
            self.groups = groups
            self.group_idx = min(self.group_idx, len(self.groups) - 1)
            self.status = status
            self.refresh_in_flight = False
            self.render_table()

        def apply_refresh_error(self, message: str) -> None:
            self.status = f"ERROR: {message}"
            self.refresh_in_flight = False
            self.render_table()

        def sort_value(self, metric: dict[str, Any], key: str) -> Any:
            if key == "name":
                return str(metric["name"]).lower()
            raw = metric.get(key)
            value = as_number(raw) if key == "latest" else raw
            if isinstance(value, (int, float)):
                return (0, value, str(metric["name"]).lower())
            return (1, compact(raw), str(metric["name"]).lower())

        def table_width(self) -> int:
            table = self.query_one("#table", DataTable)
            return table.size.width or self.size.width or 100

        def rebuild_columns(self, width: int | None = None) -> None:
            """(Re)declare columns for the current terminal width."""
            table = self.query_one("#table", DataTable)
            name_w, spark_w, show_stats = fit_run_metric_widths(width or self.table_width())
            self.col_widths = (name_w, spark_w, show_stats)
            table.clear(columns=True)
            cols = ["Metric", "Latest"]
            if show_stats:
                cols += ["Min", "Mean", "Max"]
            cols.append("N")
            if spark_w:
                cols.append("Sparkline")
            table.add_columns(*cols)

        def refit_columns(self, width: int) -> None:
            """Re-budget columns for a newly-known table width."""
            if width and width != getattr(self, "fitted_width", None):
                self.fitted_width = width
                self.rebuild_columns(width)
                self.render_table()

        def on_resize(self, event: Any = None) -> None:
            # Column budget depends on the width, so re-fit when it changes.
            self.rebuild_columns()
            self.render_table()

        def render_table(self) -> None:
            table = self.query_one("#table", DataTable)
            name_w, spark_w, show_stats = getattr(self, "col_widths", (NAME_CELL_WIDTH, MAX_SPARK_WIDTH, True))
            if len(table.columns) != 2 + (3 if show_stats else 0) + 1 + (1 if spark_w else 0):
                self.rebuild_columns()
                name_w, spark_w, show_stats = self.col_widths
            table.clear()
            shown = filtered_metrics(self.metrics, self.search, self.current_group(), "group")
            key = self.sort_columns[self.sort_idx][1]
            shown = sorted(shown, key=lambda m: self.sort_value(m, key), reverse=self.sort_reverse)
            for m in shown:
                style = metric_style(m)
                cells = [
                    rich_cell(m["name"], style, name_w),
                    rich_cell(m["latest"], "green", LATEST_COL_WIDTH),
                ]
                if show_stats:
                    cells += [
                        rich_cell(m["min"], "yellow", STAT_COL_WIDTH),
                        rich_cell(m["mean"], "cyan", STAT_COL_WIDTH),
                        rich_cell(m["max"], "red", STAT_COL_WIDTH),
                    ]
                cells.append(rich_cell(str(m["count"]), "white", COUNT_COL_WIDTH))
                if spark_w:
                    cells.append(rich_cell(sparkline(m["values"], spark_w), "green", spark_w))
                table.add_row(*cells)
            self.query_one("#meta", Static).update(format_run_meta(self.run_data, self.entity, self.project, self.run_id, self.url, self.metrics, shown, self.current_group(), self.search, self.sort_label(), self.status))
            self.query_one("#status", Static).update(KEYS_HINT_RUN)

    return RunApp(run_ref, refresh_seconds)


def make_project_app(project_ref: str, limit: int, refresh_seconds: int, run_filter: str = ""):
    require_textual()
    from textual.app import App, ComposeResult
    from textual.containers import VerticalScroll
    from textual.widgets import DataTable, Footer, Header, Input, Static

    FittedDataTable = fitted_data_table()

    class ProjectApp(RunTextualAppMixin, App[None]):
        TITLE = "wandb-tui"
        BINDINGS = BASE_BINDINGS + [
            ("m", "toggle_mode", "Mode"),
            ("f", "focus_filter", "Filter runs"),
        ]

        def __init__(self, project_ref: str, limit: int, refresh_seconds: int, run_filter: str = "") -> None:
            super().__init__()
            self.project_ref = project_ref
            self.limit = limit
            self.refresh_seconds = refresh_seconds
            self.entity, self.project, self.url = parse_project_ref(project_ref)
            self.all_runs: list[dict[str, Any]] = []
            self.runs: list[dict[str, Any]] = []
            self.metrics: list[dict[str, Any]] = []
            self.groups = ["ALL"]
            self.group_idx = 0
            self.sort_columns = [("metric", "name")]
            self.sort_idx = 0
            self.sort_reverse = False
            self.search = ""
            self.run_filter = run_filter
            self.filter_error = ""
            self.chart_mode = False
            self.status = "loading…"
            self.render_timer = None
            self.refresh_in_flight = False

        def compose(self) -> ComposeResult:
            yield Header()
            yield Static("loading…", id="meta")
            yield Input(placeholder="Search metrics", id="search_input")
            yield Input(placeholder="Filter runs by config, e.g. lr>=0.001 model~llama", id="filter_input")
            yield FittedDataTable(id="table", cursor_type="row", zebra_stripes=True)
            with VerticalScroll(id="charts"):
                yield Static("", id="chart_text")
            yield Static("", id="status")
            yield Footer()

        def on_mount(self) -> None:
            if self.run_filter:
                self.query_one("#filter_input", Input).value = self.run_filter
            self.rebuild_columns()
            self.action_refresh_data()
            if self.refresh_seconds:
                self.set_interval(self.refresh_seconds, self.refresh_if_live)

        def action_focus_filter(self) -> None:
            self.query_one("#filter_input").focus()

        def on_input_changed(self, event: Input.Changed) -> None:
            if event.input.id == "filter_input":
                self.run_filter = event.value
                # Re-filtering only re-slices already-fetched runs, so this is
                # local work -- no refetch needed as the user types.
                self.schedule_render(self.apply_run_filter)
            else:
                self.search = event.value
                self.schedule_render()

        def apply_run_filter(self) -> None:
            """Re-derive self.runs/metrics from self.all_runs for the filter."""
            try:
                self.runs = filter_runs(self.all_runs, self.run_filter)
                self.filter_error = ""
            except ValueError as e:
                # Incomplete expression while typing ("lr>" etc.): keep the last
                # good run set and surface the reason rather than emptying out.
                self.filter_error = str(e)
                self.render_table()
                return
            self.metrics = build_multi_metrics(self.runs)
            self.sort_columns = [("metric", "name")] + [(f"R{i+1:02d}", f"run:{i}") for i in range(len(self.runs))]
            self.sort_idx = min(self.sort_idx, len(self.sort_columns) - 1)
            self.groups = ["ALL"] + sorted({m["group"] for m in self.metrics})
            self.group_idx = min(self.group_idx, len(self.groups) - 1)
            self.rebuild_columns()
            self.render_table()

        def action_toggle_mode(self) -> None:
            self.chart_mode = not self.chart_mode
            self.render_table()
            # Keep focus on whichever pane is now visible, so a second `m`
            # toggles back instead of being typed into the search box.
            self.focus_results_pane()

        def refresh_if_live(self) -> None:
            # Check every fetched run, not just the filtered subset: a filter
            # that currently hides the only live run must not stop polling.
            if self.all_runs and all(r.get("state") == "finished" for r in self.all_runs):
                return
            self.action_refresh_data()

        def action_refresh_data(self) -> None:
            if self.refresh_in_flight:
                return
            self.refresh_in_flight = True
            self.status = "refreshing…"
            self.render_table()
            # One blocking fetch_run per run (default 8) at a 90s timeout each:
            # inline this would freeze the UI for minutes.
            self.run_worker(self.fetch_in_thread, thread=True, exclusive=True)

        def fetch_in_thread(self) -> None:
            try:
                all_runs = fetch_project_runs(self.entity, self.project, limit=self.limit)
                # Filter here too, so the expensive metric union is built only
                # over the runs that survive.
                try:
                    runs = filter_runs(all_runs, self.run_filter)
                    filter_error = ""
                except ValueError as e:
                    runs, filter_error = all_runs, str(e)
                metrics = build_multi_metrics(runs)
                sort_columns = [("metric", "name")] + [(f"R{i+1:02d}", f"run:{i}") for i in range(len(runs))]
                groups = ["ALL"] + sorted({m["group"] for m in metrics})
                loaded = sum(1 for r in runs if not r.get("load_error"))
                status = f"loaded {loaded}/{len(runs)} runs at {_dt.datetime.now().strftime('%H:%M:%S')}"
            except Exception as e:
                self.call_from_thread(self.apply_refresh_error, str(e))
                return
            self.call_from_thread(self.apply_refresh_result, all_runs, runs, metrics, sort_columns, groups, status, filter_error)

        def apply_refresh_result(self, all_runs, runs, metrics, sort_columns, groups, status, filter_error) -> None:
            # Commit all of it at once. Assigning self.runs before metrics were
            # built would desync the column count (from len(self.runs)) against
            # the row width (from len(m["runs"]) in the stale self.metrics),
            # raising "More values provided than there are columns".
            self.all_runs = all_runs
            self.runs = runs
            self.metrics = metrics
            self.sort_columns = sort_columns
            self.sort_idx = min(self.sort_idx, len(self.sort_columns) - 1)
            self.groups = groups
            self.group_idx = min(self.group_idx, len(self.groups) - 1)
            self.status = status
            self.filter_error = filter_error
            self.refresh_in_flight = False
            self.rebuild_columns()
            self.render_table()

        def apply_refresh_error(self, message: str) -> None:
            self.status = f"ERROR: {message}"
            self.refresh_in_flight = False
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

        def table_width(self) -> int:
            table = self.query_one("#table", DataTable)
            return table.size.width or self.size.width or 100

        def rebuild_columns(self, width: int | None = None) -> None:
            # Only ever describes the table (the run-comparison grid). Chart
            # mode hides the table entirely and renders into #chart_text, so
            # there is nothing to reshape for it -- previously this installed
            # bogus "Latest"/"Chart" columns that nothing ever populated.
            table = self.query_one("#table", DataTable)
            name_w, col_w, visible = fit_project_widths(width or self.table_width(), len(self.runs))
            self.name_w, self.col_w, self.visible_runs = name_w, col_w, visible
            table.clear(columns=True)
            # Only declare the run columns that actually fit; the rest would be
            # clipped off the right edge with no indication they exist.
            labels = [f"R{i+1:02d}" for i in range(max(1, min(len(self.runs), visible)))]
            table.add_columns("Metric", *labels)

        def refit_columns(self, width: int) -> None:
            """Re-budget columns for a newly-known table width."""
            if width and width != getattr(self, "fitted_width", None):
                self.fitted_width = width
                self.rebuild_columns(width)
                self.render_table()

        def render_charts(self, shown: list[dict[str, Any]]) -> None:
            from rich.text import Text

            # Require an actual series (2+ points) in at least one run. Config
            # scalars carry a single value, which charts as a flat line and
            # would crowd real curves out of the visible pane.
            numeric = [
                m
                for m in shown
                if any(len((slot or {}).get("values") or ()) > 1 for slot in (m.get("runs") or []))
            ]
            out = Text()
            labels = [f"R{i+1}" for i in range(len(self.runs))]
            # Size charts to the actual pane. A hard-coded width wraps every
            # plotext line in two on a narrow terminal, destroying the plot.
            pane = self.query_one("#charts", VerticalScroll)
            chart_w = max(24, (pane.size.width or 100) - 2)
            for m in numeric[:12]:
                slots = (m.get("runs") or [])[:len(self.runs)]
                title = Text()
                title.append(f"{m['name']}  ", style=f"bold {metric_style(m) or 'white'}")
                for i, slot in enumerate(slots):
                    if slot:
                        title.append(f"R{i+1}={compact(slot.get('latest'), 10)} ", style=RUN_COLORS[i % len(RUN_COLORS)])
                out.append_text(title)
                out.append("\n")
                built = render_plotext_chart(slots, chart_w, 14, labels)
                if built:
                    out.append_text(ansi_to_text(built))
                else:
                    out.append_text(overlay_chart_text(slots, chart_w, 12))
                out.append("\n\n")
            if not numeric:
                out.append("No numeric metrics with history to chart.", style="yellow")
            self.query_one("#chart_text", Static).update(out)

        def render_table(self) -> None:
            table = self.query_one("#table", DataTable)
            table.clear()
            shown = filtered_multi_metrics(self.metrics, self.search, self.current_group(), "group")
            key = self.sort_columns[self.sort_idx][1]
            shown = sorted(shown, key=lambda m: self.sort_value(m, key), reverse=self.sort_reverse)
            table.display = not self.chart_mode
            charts = self.query_one("#charts", VerticalScroll)
            charts.display = self.chart_mode
            if self.chart_mode:
                self.render_charts(shown)
            else:
                # Pad/trim each row to the column count. self.metrics and
                # self.runs are committed together, but a metric union built
                # from a partially-failed fetch can still be short -- and only
                # the run columns that fit on screen were declared.
                name_w = getattr(self, "name_w", NAME_CELL_WIDTH)
                col_w = getattr(self, "col_w", CELL_WIDTH)
                width = max(1, min(len(self.runs), getattr(self, "visible_runs", len(self.runs)) or 1))
                if len(table.columns) != width + 1:
                    self.rebuild_columns()
                    name_w, col_w = self.name_w, self.col_w
                    width = max(1, min(len(self.runs), self.visible_runs))
                for m in shown:
                    slots = (m.get("runs") or [])[:width]
                    vals = [
                        rich_cell(slot.get("latest") if slot else None, RUN_COLORS[i % len(RUN_COLORS)], col_w)
                        if slot
                        else rich_cell("·", "bright_black", col_w)
                        for i, slot in enumerate(slots)
                    ]
                    vals += [rich_cell("·", "bright_black", col_w)] * (width - len(vals))
                    table.add_row(rich_cell(m["name"], metric_style(m), name_w), *vals)
            self.query_one("#meta", Static).update(format_project_meta(self.entity, self.project, self.url, self.limit, self.runs, self.metrics, shown, self.current_group(), self.search, self.sort_label(), self.chart_mode, self.status, self.run_filter, len(self.all_runs), self.filter_error, None if self.chart_mode else getattr(self, "visible_runs", None)))
            self.query_one("#status", Static).update(KEYS_HINT_PROJECT)

    return ProjectApp(project_ref, limit, refresh_seconds, run_filter)


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
            # The title is already in the Header; repeating it in #meta just
            # burned a row. Show the row count there instead.
            yield Header()
            yield Static(f"{len(rows)} to choose from", id="meta")
            yield DataTable(id="table", cursor_type="row", zebra_stripes=True)
            yield Static("", id="status")
            yield Footer()

        def on_mount(self) -> None:
            table = self.query_one("#table", DataTable)
            table.add_columns(*columns)
            self.render_rows()

        def column_is_numeric(self, index: int) -> bool:
            """True when every non-empty value in a column parses as a number."""
            seen = False
            for row in rows:
                raw = str(values(row)[index]).strip()
                if not raw or raw == "?":
                    continue
                try:
                    float(raw)
                except ValueError:
                    return False
                seen = True
            return seen

        def sort_value(self, item: tuple[int, dict[str, Any]]) -> Any:
            if self.sort_column is None:
                return item[0]
            raw = values(item[1])[self.sort_column]
            try:
                return (0, float(str(raw)))
            except ValueError:
                return (1, str(raw).lower())

        def render_rows(self) -> None:
            from rich.text import Text

            table = self.query_one("#table", DataTable)
            table.clear()
            numeric_cols = {i for i in range(len(columns)) if self.column_is_numeric(i)}
            widths = [
                max([len(columns[i])] + [len(str(values(r)[i])) for r in rows] or [0])
                for i in range(len(columns))
            ]
            self.visible_rows = sorted(enumerate(rows), key=self.sort_value, reverse=self.sort_reverse)
            for index, row in self.visible_rows:
                cells = []
                for i, v in enumerate(values(row)):
                    # Picker values are identifiers the user has to read in full
                    # (project names, timestamps) -- not numeric metric cells,
                    # so the narrow CELL_WIDTH default would mangle them.
                    text = compact(v, PICKER_CELL_WIDTH)
                    # Right-align numeric columns so run counts line up on the
                    # ones digit instead of ragged against the label.
                    text = text.rjust(widths[i]) if i in numeric_cols else text
                    cells.append(Text(text, style=RUN_COLORS[i % len(RUN_COLORS)]))
                table.add_row(*cells, key=str(index))
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
            # Route through the same cursor->visible_rows lookup action_select
            # uses. Parsing event.row_key was a second, divergent path that
            # indexed the *unsorted* list and would raise inside a message
            # handler if a row ever lacked an integer key.
            self.exit(self.selected_row())

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


def apply_run_filter_cli(runs: list[dict[str, Any]], run_filter: str) -> list[dict[str, Any]]:
    """filter_runs for the non-interactive paths: a bad expression is a usage
    error worth failing on, rather than something to surface in a UI panel."""
    try:
        return filter_runs(runs, run_filter)
    except ValueError as e:
        raise SystemExit(f"--filter: {e}") from e


def print_once(ref: str, runs_limit: int = 8, search: str = "", group: str = "ALL", sort_mode: str = "group", top: int = 0, run_filter: str = "") -> None:
    if sort_mode not in SORT_MODES:
        raise SystemExit(f"--sort must be one of: {', '.join(SORT_MODES)}")
    if ref_kind(ref) == "project":
        entity, project, url = parse_project_ref(ref)
        all_runs = fetch_project_runs(entity, project, limit=runs_limit)
        runs = apply_run_filter_cli(all_runs, run_filter)
        metrics = apply_row_limit(filtered_multi_metrics(build_multi_metrics(runs), search, group, sort_mode), top)
        print(f"W&B project: {entity}/{project}")
        print(f"URL: {url}")
        print(f"runs={len(runs)}/{len(all_runs)} metrics_shown={len(metrics)} search='{search}' group={group} sort={sort_mode}" + (f" filter='{run_filter}'" if run_filter else ""))
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


def dump_json(ref: str, path: str, runs_limit: int = 8, search: str = "", group: str = "ALL", sort_mode: str = "group", top: int = 0, run_filter: str = "") -> None:
    if sort_mode not in SORT_MODES:
        raise SystemExit(f"--sort must be one of: {', '.join(SORT_MODES)}")
    if ref_kind(ref) == "project":
        entity, project, url = parse_project_ref(ref)
        all_runs = fetch_project_runs(entity, project, limit=runs_limit)
        runs = apply_run_filter_cli(all_runs, run_filter)
        metrics = apply_row_limit(filtered_multi_metrics(build_multi_metrics(runs), search, group, sort_mode), top)
        serializable = {
            "entity": entity,
            "project": project,
            "url": url,
            "filters": {"search": search, "group": group, "sort": sort_mode, "top": top or None, "run_filter": run_filter or None},
            "runs_total": len(all_runs),
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
    p.add_argument(
        "--filter",
        default="",
        metavar="EXPR",
        help=(
            "Project mode: keep only runs matching a config expression. "
            "Space-separated terms are AND-ed, e.g. \"lr>=0.001 model~llama state=finished\". "
            "Operators: = != > < >= <= ~ (contains) !~ (not contains). "
            "Bare keys read the run config; use run.<attr> for run attributes."
        ),
    )
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

    if args.filter and ref_kind(ref) != "project":
        raise SystemExit("--filter selects among a project's runs; pass ENTITY/PROJECT rather than a single run.")

    if args.json:
        dump_json(ref, args.json, args.runs, args.search, args.group, args.sort, args.top, args.filter)
    elif args.once:
        print_once(ref, args.runs, args.search, args.group, args.sort, args.top, args.filter)
    else:
        if ref_kind(ref) == "project":
            make_project_app(ref, args.runs, args.refresh, args.filter).run()
        else:
            make_run_app(ref, args.refresh).run()


if __name__ == "__main__":
    main()
