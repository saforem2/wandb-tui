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

# Metadata for this many runs costs one ~1s query; per-run history is fetched
# in parallel and cached on disk, so a large default is affordable.
DEFAULT_RUN_LIMIT = 100
# Cap on how many runs we pull full history for at once. Metadata for all
# DEFAULT_RUN_LIMIT runs is cheap; history is ~0.3s/run even threaded, and more
# than this many overlaid series is unreadable. Narrow with a filter to choose
# WHICH runs land inside the cap.
HISTORY_RUN_LIMIT = 24

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
    """Cheap metadata for the most recent runs -- no history.

    `config` and `summaryMetrics` come back in this single query (measured:
    ~0.6s for 100 runs), which is what makes config filtering work over a large
    run set without paying for per-run history fetches.
    """
    query = """
    query Runs($entity:String!, $project:String!, $first:Int!) {
      project(name:$project, entityName:$entity) {
        name
        totalRuns
        runCount
        runs(first:$first, order:"-created_at") {
          edges {
            node {
              id name displayName state createdAt updatedAt historyLineCount
              config summaryMetrics
            }
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
    out = []
    for edge in edges:
        node = edge.get("node")
        if not node:
            continue
        node = dict(node)
        node["entity"] = entity
        node["project"] = project
        out.append(node)
    return out


def _cache_dir() -> str:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "wandb-tui")


def _cache_path(entity: str, project: str, run_id: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z._-]", "_", f"{entity}__{project}__{run_id}")
    return os.path.join(_cache_dir(), f"{safe}.json")


def load_cached_run(entity: str, project: str, run_id: str, updated_at: str | None) -> dict[str, Any] | None:
    """Return a cached run history, or None on any miss.

    `updated_at` is the freshness key: a run whose W&B updatedAt has moved on
    since we cached it must be refetched. A finished run never moves again, so
    it is served from disk forever.
    """
    path = _cache_path(entity, project, run_id)
    try:
        with open(path, encoding="utf-8") as f:
            blob = json.load(f)
    except Exception:
        return None
    if not isinstance(blob, dict):
        return None
    if updated_at is not None and blob.get("_cached_updated_at") != updated_at:
        return None
    run = blob.get("run")
    return run if isinstance(run, dict) else None


def store_cached_run(entity: str, project: str, run_id: str, run: dict[str, Any]) -> None:
    """Best-effort write-through cache. A cache failure must never break a fetch."""
    try:
        os.makedirs(_cache_dir(), exist_ok=True)
        path = _cache_path(entity, project, run_id)
        tmp = f"{path}.tmp{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"_cached_updated_at": run.get("updatedAt"), "run": run}, f)
        os.replace(tmp, path)  # atomic: a crash mid-write can't leave a torn file
    except Exception:
        pass


def fetch_run_cached(entity: str, project: str, run_id: str, updated_at: str | None = None, samples: int = 10000) -> dict[str, Any]:
    cached = load_cached_run(entity, project, run_id, updated_at)
    if cached is not None:
        return cached
    run = fetch_run(entity, project, run_id, samples=samples)
    store_cached_run(entity, project, run_id, run)
    return run


def fetch_histories(nodes: list[dict[str, Any]], samples: int = 10000, workers: int = 8) -> list[dict[str, Any]]:
    """Fetch full history for each metadata node, in parallel, cache-aware.

    Sequential fetching measured ~0.30s/run; 8 threads cuts a batch of 8 from
    2.4s to 0.44s (~5.5x), which is what makes a large run limit usable. Order
    of the input list is preserved. A per-run failure becomes `load_error` on
    that run rather than failing the whole batch.
    """
    if not nodes:
        return []

    def one(node: dict[str, Any]) -> dict[str, Any]:
        entity = node.get("entity") or ""
        project = node.get("project") or ""
        try:
            return fetch_run_cached(entity, project, node["name"], node.get("updatedAt"), samples)
        except Exception as e:
            out = dict(node)
            out["load_error"] = str(e)
            return out

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(nodes)))) as pool:
        return list(pool.map(one, nodes))


def fetch_project_runs(entity: str, project: str, limit: int = 8, samples: int = 10000) -> list[dict[str, Any]]:
    return fetch_histories(fetch_project_run_names(entity, project, limit), samples=samples)


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
# Set membership is expressed as a comma list on = / != rather than a
# dedicated ":" operator: ":" is legal inside config keys (optim:lr), and
# making it an operator would make such keys unaddressable.
FILTER_OPS = ("!~", ">=", "<=", "!=", "~", "=", ">", "<")
SET_OPS = ("=", "!=")
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
                if "," in value and not [a for a in value.split(",") if a.strip()]:
                    # `world_size=,` -- no usable alternatives.
                    raise ValueError(f"filter term {raw!r} needs at least one value")
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

    if op in SET_OPS and "," in expected:
        # Set membership: `world_size=3072,6144` keeps runs matching ANY
        # alternative; `!=` keeps runs matching none. Each alternative uses the
        # same numeric-then-string rules as a bare `=`, so 3072 matches 3072.0.
        alts = [a.strip() for a in expected.split(",") if a.strip() != ""]
        hit = any(match_filter_term(actual, "=", alt) for alt in alts)
        return hit if op == "=" else not hit

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
    /* Fixed, not auto: the meta panel wraps its legend/keys lines, and with
       many runs an auto height grew without bound and squeezed the results
       pane to nothing. Overflow is clipped rather than allowed to push. */
    #meta { height: 6; overflow: hidden; padding: 0 1; color: #d1d5db; background: #111827; }
    #search_input, #filter_input { height: 3; margin: 0 1; background: #1f2937; color: #e5e7eb; border: tall #374151; }
    #filter_input { border: tall #4b5563; }
    #table { height: 1fr; background: #111111; color: #e5e7eb; }
    #charts { height: 1fr; background: #111111; color: #e5e7eb; display: none; }
    /* Tabs must be pinned to its real height: `height: auto` let it expand to
       fill the container, which pushed the chart pane off-screen entirely. */
    #group_tabs { height: 2; display: none; }
    /* Each tile gets a FIXED height and the full pane width, so charts stay a
       readable size and the pane scrolls instead of tiles expanding to consume
       whatever space the run count happens to leave. */
    .chart-tile {
        width: 1fr;
        height: 18;
        margin: 0 1 1 1;
        border: round #374151;
    }
    .chart-tile:focus { border: round #facc15; }
    .chart-empty { padding: 1; color: #facc15; }
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

# RGB palette for plots, ordered so adjacent run indices get maximally
# different hues (the named-color set above wraps at 7 and aliases quickly).
PLOT_PALETTE = (
    (66, 135, 245),   # blue
    (245, 133, 24),   # orange
    (84, 196, 75),    # green
    (228, 87, 86),    # red
    (162, 122, 255),  # purple
    (0, 199, 190),    # teal
    (255, 105, 180),  # pink
    (214, 197, 45),   # gold
    (150, 100, 60),   # brown
    (120, 220, 150),  # mint
    (140, 160, 175),  # slate
    (255, 160, 90),   # apricot
)

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


def metric_series(metric: dict[str, Any], run_index: int) -> list[float]:
    """The numeric series one run contributes to a metric, or []."""
    slots = metric.get("runs") or []
    slot = slots[run_index] if run_index < len(slots) else None
    if not slot:
        return []
    return [float(v) for v in (slot.get("values") or []) if isinstance(v, (int, float)) and math.isfinite(float(v))]


def chartable(metric: dict[str, Any]) -> bool:
    """True when at least one run has a real series (2+ points) for this metric.

    Config scalars carry a single value; charting them yields a flat line that
    crowds real curves out of the view.
    """
    return any(len((slot or {}).get("values") or ()) > 1 for slot in (metric.get("runs") or []))


def rgb_for_run(index: int) -> tuple[int, int, int]:
    """High-contrast categorical color for a run, by stable index."""
    return PLOT_PALETTE[index % len(PLOT_PALETTE)]


def dim_rgb(rgb: tuple[int, int, int], factor: float = 0.45, bg: tuple[int, int, int] = (0, 0, 0)) -> tuple[int, int, int]:
    """Fade a color toward the background so unfocused runs recede.

    Blending toward the actual background (rather than multiplying toward
    black) dims correctly on light themes too, where darkening would instead
    *raise* contrast.
    """
    return tuple(int(c * factor + b * (1.0 - factor)) for c, b in zip(rgb, bg))


def draw_metric_plot(
    plt: Any,
    metric: dict[str, Any],
    run_count: int,
    labels: list[str],
    xlim: tuple[float | None, float | None] = (None, None),
    ylim: tuple[float | None, float | None] = (None, None),
    focus_run: int | None = None,
    ylog: bool = False,
    title: str | None = None,
    bg: tuple[int, int, int] = (0, 0, 0),
) -> int:
    """Draw one metric's runs onto a plotext figure. Returns series drawn.

    Axis limits are applied by CLIPPING points in python rather than via
    plt.xlim/plt.ylim: plotext raises IndexError inside its legend loop when a
    plotted series has zero points inside the limit window. Clipping and
    skipping now-empty series sidesteps that; the axes then autoscale to the
    surviving points, which looks the same as the requested window.
    """
    plt.clear_data()
    plt.clear_figure()
    xlo, xhi = xlim
    ylo, yhi = ylim
    xL = xlo if xlo is not None else float("-inf")
    xH = xhi if xhi is not None else float("inf")
    yL = ylo if ylo is not None else float("-inf")
    yH = yhi if yhi is not None else float("inf")
    clip = any(v is not None for v in (xlo, xhi, ylo, yhi))

    drawn = 0
    for i in range(run_count):
        ys = metric_series(metric, i)
        if len(ys) < 2:
            continue
        xs = list(range(len(ys)))
        if clip:
            pts = [(x, y) for x, y in zip(xs, ys) if xL <= x <= xH and yL <= y <= yH]
            if len(pts) < 2:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
        if ylog:
            # Transform here and plot on a linear axis: plotext's own log path
            # runs log10 over synthesized ticks and raises math-domain errors.
            lp = [(x, math.log10(y)) for x, y in zip(xs, ys) if y > 0]
            if len(lp) < 2:
                continue
            xs = [p[0] for p in lp]
            ys = [p[1] for p in lp]
        color = rgb_for_run(i)
        if focus_run is not None and i != focus_run:
            color = dim_rgb(color, bg=bg)
        plt.plot(xs, ys, color=color, marker="braille",
                 label=labels[i] if i < len(labels) else f"R{i+1}")
        drawn += 1

    name = str(metric.get("name", ""))
    plt.title(title if title is not None else name)
    plt.ylabel("log10(value)" if ylog else "value")
    return drawn


def metric_extent(metric: dict[str, Any], run_index: int) -> tuple[float, float, float, float] | None:
    """(xmin, xmax, ymin, ymax) for one run's series on a metric."""
    ys = metric_series(metric, run_index)
    if len(ys) < 2:
        return None
    return 0.0, float(len(ys) - 1), min(ys), max(ys)


def metric_span(metric: dict[str, Any], run_count: int) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    """Overall (x span, y span) across every run drawn for a metric."""
    xs_all: list[float] = []
    ys_all: list[float] = []
    for i in range(run_count):
        ys = metric_series(metric, i)
        if len(ys) < 2:
            continue
        xs_all += [0.0, float(len(ys) - 1)]
        ys_all += [min(ys), max(ys)]
    if not xs_all:
        return None, None
    return (min(xs_all), max(xs_all)), (min(ys_all), max(ys_all))


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


LEGEND_MAX_RUNS = 8


def format_run_legend(runs: list[dict[str, Any]], limit: int = LEGEND_MAX_RUNS) -> Any:
    """One-line legend. Truncates rather than wrapping over many rows.

    With 20+ runs an untruncated legend wrapped to six or more rows and pushed
    the charts off the bottom of the screen, so the panel it lives in must stay
    a predictable height.
    """
    from rich.text import Text

    text = Text()
    for i, run in enumerate(runs[:limit]):
        if i:
            text.append("  ")
        text.append(f"R{i+1}={run_label(run)}", style=f"bold {RUN_COLORS[i % len(RUN_COLORS)]}")
    if len(runs) > limit:
        text.append(f"  (+{len(runs) - limit} more)", style="dim")
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


# Cap on chart tiles mounted at once. Without a cap a project with hundreds of
# metrics mounts hundreds of plot widgets, each of which renders on every
# resize -- the pane grows unboundedly and the app crawls.
MAX_CHART_TILES = 12

# Textual Tab ids must be identifiers, but metric group names can contain "/",
# "." and "-". Encode, and keep the reverse map so an activation resolves back.
_GROUP_TAB_IDS: dict[str, str] = {}


def _group_tab_id(group: str) -> str:
    tid = "grp_" + re.sub(r"[^0-9A-Za-z]", "_", str(group))
    _GROUP_TAB_IDS[tid] = group
    return tid


def require_plotext_widget() -> None:
    import importlib.util

    if importlib.util.find_spec("textual_plotext") is None:
        raise SystemExit(
            "textual-plotext is required for chart mode. Install with `pip install textual-plotext`."
        )


def metric_chart_widget():
    """A focusable PlotextPlot tile for one metric in the chart grid.

    PlotextPlot re-renders against its own allotted size, so a tile laid out
    with fr units reflows on terminal resize with no manual width math.
    """
    require_plotext_widget()
    from textual_plotext import PlotextPlot

    class MetricChart(PlotextPlot):
        can_focus = True

        def __init__(self, metric: dict[str, Any], run_count: int, labels: list[str], **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.metric = metric
            self.run_count = run_count
            self.labels = labels

        def on_mount(self) -> None:
            # Transparent canvas so the plot inherits the app theme instead of
            # painting its own black background under a light theme.
            try:
                self.theme = "textual-clear"
            except Exception:
                pass
            self.border_title = str(self.metric.get("name", ""))
            self.replot()

        def replot(self) -> None:
            draw_metric_plot(self.plt, self.metric, self.run_count, self.labels, title="")
            self.refresh()

        def on_click(self) -> None:
            self.focus()
            open_full = getattr(self.app, "open_chart_fullscreen", None)
            if open_full is not None:
                open_full(self.metric)

    return MetricChart


def chart_zoom_screen():
    """Full-screen single-chart view with pan/zoom/focus, mirroring prod_dash.

    Interaction: z cycles run focus (fit to that run, dim others), Z/0 reset,
    +/- zoom, h/l pan x, j/k pan y, L log/linear, esc back to the grid.
    """
    require_plotext_widget()
    from textual.app import ComposeResult
    from textual.screen import Screen
    from textual.widgets import Footer, Header, Static
    from textual_plotext import PlotextPlot

    class ChartZoomScreen(Screen):
        BINDINGS = [
            ("escape", "close", "Back"),
            ("q", "close", "Back"),
            ("z", "cycle_focus", "Focus run"),
            ("Z", "reset_view", "Reset"),
            ("0", "reset_view", "Reset"),
            ("plus", "zoom_in", "Zoom in"),
            ("equals_sign", "zoom_in", "Zoom in"),
            ("minus", "zoom_out", "Zoom out"),
            ("h", "pan_left", "Pan left"),
            ("l", "pan_right", "Pan right"),
            ("j", "pan_down", "Pan down"),
            ("k", "pan_up", "Pan up"),
            ("L", "toggle_ylog", "Log/linear"),
        ]
        CSS = """
        #zoom_plot { width: 1fr; height: 1fr; }
        #zoom_meta { dock: top; height: auto; padding: 0 1; background: #111827; color: #d1d5db; }
        #zoom_status { dock: bottom; height: 1; padding: 0 1; background: #111827; color: #d1d5db; }
        """

        def __init__(self, metric: dict[str, Any], run_count: int, labels: list[str]) -> None:
            super().__init__()
            self.metric = metric
            self.run_count = run_count
            self.labels = labels
            self.xlim: tuple[float | None, float | None] = (None, None)
            self.ylim: tuple[float | None, float | None] = (None, None)
            self.focus_run: int | None = None
            self.focus_idx = -1
            self.ylog = False

        def compose(self) -> ComposeResult:
            yield Header()
            yield Static("", id="zoom_meta")
            yield PlotextPlot(id="zoom_plot")
            yield Static("", id="zoom_status")
            yield Footer()

        def on_mount(self) -> None:
            plot = self.query_one("#zoom_plot", PlotextPlot)
            try:
                plot.theme = "textual-clear"
            except Exception:
                pass
            plot.focus()
            self.redraw()

        def redraw(self) -> None:
            plot = self.query_one("#zoom_plot", PlotextPlot)
            drawn = draw_metric_plot(
                plot.plt, self.metric, self.run_count, self.labels,
                xlim=self.xlim, ylim=self.ylim, focus_run=self.focus_run,
                ylog=self.ylog, title="",
            )
            plot.refresh()
            tags = []
            if self.ylog:
                tags.append("y:log")
            if self.focus_run is not None:
                who = self.labels[self.focus_run] if self.focus_run < len(self.labels) else f"R{self.focus_run+1}"
                tags.append(f"focus:{who}")
            if any(v is not None for v in self.xlim):
                tags.append(f"x[{_fmt_lim(self.xlim[0])},{_fmt_lim(self.xlim[1])}]")
            if any(v is not None for v in self.ylim):
                tags.append(f"y[{_fmt_lim(self.ylim[0])},{_fmt_lim(self.ylim[1])}]")
            if not drawn:
                tags.append("no data in view")
            from rich.text import Text

            head = Text()
            head.append(f"{self.metric.get('name','')}\n", style="bold white")
            head.append("  ".join(tags) or "full view", style="yellow" if not drawn else "cyan")
            self.query_one("#zoom_meta", Static).update(head)
            self.query_one("#zoom_status", Static).update(
                "esc back | z focus run | Z reset | +/- zoom | h/l pan x | j/k pan y | L log"
            )

        def action_close(self) -> None:
            self.dismiss(None)

        def action_toggle_ylog(self) -> None:
            self.ylog = not self.ylog
            self.redraw()

        def action_reset_view(self) -> None:
            self.xlim = (None, None)
            self.ylim = (None, None)
            self.focus_run = None
            self.focus_idx = -1
            self.redraw()

        def action_cycle_focus(self) -> None:
            drawable = [i for i in range(self.run_count) if len(metric_series(self.metric, i)) > 1]
            if not drawable:
                return
            self.focus_idx += 1
            if self.focus_idx >= len(drawable):
                # Wrapped past the end: back to showing everything.
                self.action_reset_view()
                return
            idx = drawable[self.focus_idx]
            ext = metric_extent(self.metric, idx)
            if ext is None:
                return
            xmn, xmx, ymn, ymx = ext
            xpad = (xmx - xmn) * 0.02 or 1.0
            ypad = (ymx - ymn) * 0.05 or 0.01
            self.xlim = (xmn - xpad, xmx + xpad)
            self.ylim = (ymn - ypad, ymx + ypad)
            self.focus_run = idx
            self.redraw()

        def _xwin(self) -> tuple[float, float]:
            xspan, _ = metric_span(self.metric, self.run_count)
            span = xspan or (0.0, 1.0)
            lo = self.xlim[0] if self.xlim[0] is not None else span[0]
            hi = self.xlim[1] if self.xlim[1] is not None else span[1]
            return lo, hi

        def _ywin(self) -> tuple[float, float]:
            _, yspan = metric_span(self.metric, self.run_count)
            span = yspan or (0.0, 1.0)
            lo = self.ylim[0] if self.ylim[0] is not None else span[0]
            hi = self.ylim[1] if self.ylim[1] is not None else span[1]
            return lo, hi

        def _zoom(self, factor: float) -> None:
            lo, hi = self._xwin()
            mid = (lo + hi) / 2.0
            half = (hi - lo) / 2.0 * factor
            self.xlim = (mid - half, mid + half)
            self.focus_run = None
            self.redraw()

        def action_zoom_in(self) -> None:
            self._zoom(0.7)

        def action_zoom_out(self) -> None:
            self._zoom(1 / 0.7)

        def _pan_x(self, frac: float) -> None:
            lo, hi = self._xwin()
            d = (hi - lo) * frac
            self.xlim = (lo + d, hi + d)
            self.focus_run = None
            self.redraw()

        def _pan_y(self, frac: float) -> None:
            lo, hi = self._ywin()
            d = (hi - lo) * frac
            self.ylim = (lo + d, hi + d)
            self.focus_run = None
            self.redraw()

        def action_pan_left(self) -> None:
            self._pan_x(-0.25)

        def action_pan_right(self) -> None:
            self._pan_x(0.25)

        def action_pan_up(self) -> None:
            self._pan_y(0.25)

        def action_pan_down(self) -> None:
            self._pan_y(-0.25)

    return ChartZoomScreen


def _fmt_lim(v: float | None) -> str:
    return "auto" if v is None else f"{v:g}"


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
    from textual.widgets import DataTable, Footer, Header, Input, Static, Tab, Tabs

    FittedDataTable = fitted_data_table()
    MetricChart = metric_chart_widget()
    ChartZoomScreen = chart_zoom_screen()

    class ProjectApp(RunTextualAppMixin, App[None]):
        TITLE = "wandb-tui"
        BINDINGS = BASE_BINDINGS + [
            ("m", "toggle_mode", "Mode"),
            ("f", "focus_filter", "Filter runs"),
            ("enter", "open_chart", "Open chart"),
        ]

        def __init__(self, project_ref: str, limit: int, refresh_seconds: int, run_filter: str = "") -> None:
            super().__init__()
            self.project_ref = project_ref
            self.limit = limit
            # How many runs we will pull full history for. Metadata is cheap
            # for all `limit` runs; history is not, and more than this many
            # overlaid series is unreadable anyway.
            self.history_limit = HISTORY_RUN_LIMIT
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
            self.pending_nodes: list[dict[str, Any]] = []
            self.pending_truncated = False

        def compose(self) -> ComposeResult:
            yield Header()
            yield Static("loading…", id="meta")
            yield Input(placeholder="Search metrics", id="search_input")
            yield Input(placeholder="Filter runs by config, e.g. lr>=0.001 model~llama", id="filter_input")
            yield Tabs(Tab("ALL", id="grp_ALL"), id="group_tabs")
            yield FittedDataTable(id="table", cursor_type="row", zebra_stripes=True)
            yield VerticalScroll(id="charts")
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
            """Re-apply the filter to the cached metadata and reload history.

            all_runs holds METADATA ONLY (no history), so this cannot just
            re-slice it -- the surviving runs need their history fetched before
            they can be charted. Cached runs come back from disk instantly, so
            in practice this is fast, but it still has to go through the worker.
            """
            try:
                kept = filter_runs(self.all_runs, self.run_filter)
                self.filter_error = ""
            except ValueError as e:
                # Incomplete expression while typing ("lr>" etc.): keep the last
                # good run set and surface the reason rather than emptying out.
                self.filter_error = str(e)
                self.render_table()
                return
            budget = max(1, self.history_limit)
            self.pending_nodes = kept[:budget]
            self.pending_truncated = len(kept) > budget
            if self.refresh_in_flight:
                return  # a fetch is already running; it will pick up the filter
            self.refresh_in_flight = True
            self.status = "filtering…"
            self.render_table()
            self.run_worker(self.filter_in_thread, thread=True, exclusive=True)

        def filter_in_thread(self) -> None:
            """Fetch history for the filtered node set (mostly disk-cached)."""
            nodes = list(self.pending_nodes)
            try:
                runs = fetch_histories(nodes)
                metrics = build_multi_metrics(runs)
                sort_columns = [("metric", "name")] + [(f"R{i+1:02d}", f"run:{i}") for i in range(len(runs))]
                groups = ["ALL"] + sorted({m["group"] for m in metrics})
                loaded = sum(1 for r in runs if not r.get("load_error"))
                status = f"loaded {loaded}/{len(runs)} runs of {len(self.all_runs)} at {_dt.datetime.now().strftime('%H:%M:%S')}"
                if self.pending_truncated:
                    status += f" (history capped at {max(1, self.history_limit)}; narrow the filter)"
            except Exception as e:
                self.call_from_thread(self.apply_refresh_error, str(e))
                return
            # all_runs (the metadata set) is unchanged by a filter.
            self.call_from_thread(
                self.apply_refresh_result, self.all_runs, runs, metrics, sort_columns, groups, status, self.filter_error
            )

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
                # Two-stage load. Stage 1 is one cheap query returning metadata
                # + config for every run (~1s for 100). Stage 2 fetches full
                # history ONLY for runs that survive the filter, in parallel and
                # served from disk cache when the run hasn't changed. Filtering
                # on config before paying for history is what makes a 100-run
                # default affordable.
                nodes = fetch_project_run_names(self.entity, self.project, self.limit)
                try:
                    kept = filter_runs(nodes, self.run_filter)
                    filter_error = ""
                except ValueError as e:
                    kept, filter_error = nodes, str(e)
                budget = max(1, self.history_limit)
                if len(kept) > budget:
                    # Guard the pathological case: an unfiltered 100-run project
                    # would otherwise chart 100 overlaid series illegibly.
                    kept = kept[:budget]
                runs = fetch_histories(kept)
                all_runs = nodes
                metrics = build_multi_metrics(runs)
                sort_columns = [("metric", "name")] + [(f"R{i+1:02d}", f"run:{i}") for i in range(len(runs))]
                groups = ["ALL"] + sorted({m["group"] for m in metrics})
                loaded = sum(1 for r in runs if not r.get("load_error"))
                status = f"loaded {loaded}/{len(runs)} runs of {len(nodes)} at {_dt.datetime.now().strftime('%H:%M:%S')}"
                if len(nodes) > len(runs):
                    status += f" (history capped at {budget}; filter to pick which)"
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

        def run_labels(self) -> list[str]:
            return [f"R{i+1}" for i in range(len(self.runs))]

        def render_charts(self, shown: list[dict[str, Any]]) -> None:
            """Mount one focusable PlotextPlot per chartable metric.

            Each tile is a real widget sized by CSS (fr units + a fixed row
            height), so tiles reflow on resize instead of being drawn at a
            hard-coded size, and each can be focused and opened full-screen.
            """
            pane = self.query_one("#charts", VerticalScroll)
            numeric = [m for m in shown if chartable(m)]
            wanted = [str(m["name"]) for m in numeric[:MAX_CHART_TILES]]
            signature = (tuple(wanted), len(self.runs))
            if signature == getattr(self, "_chart_signature", None):
                # Same metrics and run count: just refresh the existing tiles
                # rather than tearing down and remounting the whole grid.
                for tile in pane.query(MetricChart):
                    tile.replot()
                return
            self._chart_signature = signature
            pane.remove_children()
            if not numeric:
                pane.mount(Static("No numeric metrics with history to chart.", classes="chart-empty"))
                return
            labels = self.run_labels()
            by_name = {str(m["name"]): m for m in numeric}
            for name in wanted:
                tile = MetricChart(by_name[name], len(self.runs), labels, classes="chart-tile")
                pane.mount(tile)
            if len(numeric) > len(wanted):
                pane.mount(
                    Static(
                        f"… {len(numeric) - len(wanted)} more metrics; search to narrow.",
                        classes="chart-empty",
                    )
                )

        def open_chart_fullscreen(self, metric: dict[str, Any]) -> None:
            self.push_screen(ChartZoomScreen(metric, len(self.runs), self.run_labels()))

        def action_open_chart(self) -> None:
            """Enter on a focused chart tile opens it full-screen."""
            focused = self.focused
            if isinstance(focused, MetricChart):
                self.open_chart_fullscreen(focused.metric)

        def sync_group_tabs(self) -> None:
            """Mirror the discovered metric groups into the tab bar.

            Tabs are DIFFED rather than cleared and re-added: Tabs.clear() is
            async (removal lands on the next pump), so clear+add in one call
            races and a later rebuild can re-add an id whose old tab is still
            mounted, raising DuplicateIds.
            """
            try:
                tabs = self.query_one("#group_tabs", Tabs)
            except Exception:
                return
            want = {_group_tab_id(g): g for g in self.groups}
            have = {t.id for t in tabs.query(Tab)}
            for tid in have - set(want):
                try:
                    tabs.remove_tab(tid)
                except Exception:
                    pass
            for tid, g in want.items():
                if tid not in have:
                    try:
                        tabs.add_tab(Tab(g, id=tid))
                    except Exception:
                        pass

        def on_tabs_tab_activated(self, event: Tabs.TabActivated) -> None:
            if getattr(event.tabs, "id", None) != "group_tabs" or event.tab is None:
                return
            group = _GROUP_TAB_IDS.get(event.tab.id)
            if group is None or group not in self.groups:
                return
            idx = self.groups.index(group)
            if idx != self.group_idx:
                self.group_idx = idx
                self.render_table()

        def render_table(self) -> None:
            table = self.query_one("#table", DataTable)
            table.clear()
            shown = filtered_multi_metrics(self.metrics, self.search, self.current_group(), "group")
            key = self.sort_columns[self.sort_idx][1]
            shown = sorted(shown, key=lambda m: self.sort_value(m, key), reverse=self.sort_reverse)
            table.display = not self.chart_mode
            charts = self.query_one("#charts", VerticalScroll)
            charts.display = self.chart_mode
            # Group tabs replace the `g` cycle once there is more than one
            # group to choose between.
            self.sync_group_tabs()
            try:
                self.query_one("#group_tabs", Tabs).display = len(self.groups) > 2
            except Exception:
                pass
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
    from textual.widgets import DataTable, Footer, Header, Input, Static

    class PickerApp(App[dict[str, Any] | None]):
        CSS = textual_css()
        BINDINGS = [
            ("q", "quit_none", "Quit"),
            ("enter", "select", "Select"),
            ("s", "cycle_sort", "Sort"),
            ("r", "reverse_sort", "Reverse"),
            ("slash", "focus_search", "Search"),
            ("escape", "clear_search", "Clear"),
        ]
        TITLE = title
        # Focus the table, not the search box: otherwise every single-letter
        # binding (q/s/r) would be typed as text instead of firing.
        AUTO_FOCUS = "#table"

        def __init__(self) -> None:
            super().__init__()
            self.sort_column: int | None = None
            self.sort_reverse = False
            self.search = ""
            self.visible_rows: list[tuple[int, dict[str, Any]]] = list(enumerate(rows))

        def compose(self) -> ComposeResult:
            # The title is already in the Header; repeating it in #meta just
            # burned a row. Show the row count there instead.
            yield Header()
            yield Static(f"{len(rows)} to choose from", id="meta")
            yield Input(placeholder="Search (/ to focus, Esc to clear)", id="search_input")
            yield DataTable(id="table", cursor_type="row", zebra_stripes=True)
            yield Static("", id="status")
            yield Footer()

        def on_mount(self) -> None:
            table = self.query_one("#table", DataTable)
            table.add_columns(*columns)
            self.render_rows()

        def on_input_changed(self, event: Input.Changed) -> None:
            self.search = event.value
            self.render_rows()

        def action_focus_search(self) -> None:
            self.query_one("#search_input").focus()

        def action_clear_search(self) -> None:
            self.search = ""
            self.query_one("#search_input", Input).value = ""
            # Hand focus back so the letter bindings work again.
            self.query_one("#table").focus()
            self.render_rows()

        def matches_search(self, row: dict[str, Any]) -> bool:
            """Case-insensitive substring match across every visible column."""
            if not self.search:
                return True
            needle = self.search.lower()
            return any(needle in str(v).lower() for v in values(row))

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
            # Filter first, then sort, so visible_rows stays the single source
            # of truth for the cursor -> row lookup in selected_row().
            matching = [(i, r) for i, r in enumerate(rows) if self.matches_search(r)]
            self.visible_rows = sorted(matching, key=self.sort_value, reverse=self.sort_reverse)
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
            shown = f"{len(self.visible_rows)}/{len(rows)}" if self.search else str(len(rows))
            self.query_one("#meta", Static).update(
                f"{shown} to choose from" + (f"  search='{self.search}'" if self.search else "")
            )
            self.query_one("#status", Static).update(
                f"Enter select | / search | Esc clear | s sort column | r reverse | q quit | sort={sort_label}"
            )

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
    p.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_RUN_LIMIT,
        help=(
            f"Project mode: number of recent runs to load (default {DEFAULT_RUN_LIMIT}). "
            "Metadata and config come from one cheap query; per-run history is "
            "fetched in parallel and cached on disk."
        ),
    )
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
            "A comma-separated value means any-of, e.g. \"world_size=3072,6144\". "
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
