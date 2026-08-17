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
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# Metadata for this many runs costs one ~1s query; per-run history is fetched
# in parallel and cached on disk, so a large default is affordable.
DEFAULT_RUN_LIMIT = 100
# Cap on how many runs we pull full history for at once. Measured on a 100-run
# project: cold 1.1s @24, 3.6s @50, 10.1s @100 -- but warm (disk cache) is
# 0.01-0.08s at every size, and the metric union costs <1s even at 100. So the
# cap exists only to bound the FIRST load on a large project; it is not a
# rendering limit. Charts stay legible because a filter narrows what's drawn.
HISTORY_RUN_LIMIT = 100

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

    # Resolve the token axis once: which column (if any) this run uses.
    token_source = next(
        (src for src in TOKEN_SOURCES if any(src in row for row in rows)), None
    )

    metrics = []
    for key in sorted(keys):
        # Collect x alongside y in ONE pass, appending to both only for rows
        # that survive. `values` is filtered twice (row must have the key, and
        # the value must be numeric), so an x list built by a separate pass
        # over all rows would be longer and pair point k with the wrong x.
        nums: list[Any] = []
        axis_cols: dict[str, list[Any]] = {}
        present = 0  # rows holding the key at all, numeric or not
        for row in rows:
            if key not in row or row.get(key) is None:
                continue
            present += 1
            num = as_number(row.get(key))
            if num is None:
                continue
            nums.append(num)
            for axis in X_AXES:
                source = token_source if axis.id == "tokens" else axis.source
                if source is None:
                    continue
                raw = as_number(row.get(source))
                axis_cols.setdefault(axis.id, []).append(raw)
        axes: dict[str, list[Any]] = {}
        for axis in X_AXES:
            col = axis_cols.get(axis.id) or []
            # A column present in only some rows would misalign; require all.
            if not col or any(v is None for v in col):
                continue
            if axis.relative:
                base = col[0]
                col = [v - base for v in col]
            axes[axis.id] = col
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
            "count": present,
            "numeric_count": len(nums),
            "type": "number" if nums else type(latest).__name__ if latest is not None else "unknown",
            "values": nums,
            "axes": axes,
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


def run_url(run: dict[str, Any]) -> str:
    """Canonical W&B URL for a run, or "" when we can't build one.

    The run's own `name` is the W&B id used in the URL path -- `displayName`
    is the human label ("cosmic-glitter-3343") and does NOT resolve.
    """
    entity = run.get("entity")
    project = run.get("project")
    run_id = run.get("name")
    if not (entity and project and run_id):
        return ""
    return f"https://wandb.ai/{entity}/{project}/runs/{run_id}"


# How deep to walk nested config dicts when flattening to dotted paths. W&B
# configs are usually 1-2 levels (`args`, `training`, `precision`); a bound
# keeps a pathological config from exploding the key space.
CONFIG_MAX_DEPTH = 4


def _flatten_config(node: Any, prefix: str, out: dict[str, Any], depth: int) -> None:
    """Record `node` at `prefix`, recursing into dicts as dotted sub-keys."""
    if prefix:
        # Keep the container itself addressable as well as its children, so
        # `args` still resolves (to the dict) alongside `args.lr`.
        out[prefix] = node
    if not isinstance(node, dict) or depth >= CONFIG_MAX_DEPTH:
        return
    for key, val in node.items():
        name = str(key)
        if name.startswith("_"):
            continue
        # Unwrap W&B's {"value": x} envelope at every level, not just the top.
        if isinstance(val, dict) and "value" in val and len(val) <= 2:
            val = val["value"]
        _flatten_config(val, f"{prefix}.{name}" if prefix else name, out, depth + 1)


def run_config(run: dict[str, Any]) -> dict[str, Any]:
    """Flatten a run's config to {dotted.key: value}.

    W&B configs nest -- a single `args` key can hold 59 children -- so a
    top-level-only flatten left `args.lr` unresolvable. That broke filtering
    silently: `args.lr=0.003` matched zero runs even though the value was
    right there, which is worse than the missing completion it also caused.
    """
    raw = run.get("config") or "{}"
    try:
        cfg = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        return {}
    if not isinstance(cfg, dict):
        return {}
    out: dict[str, Any] = {}
    _flatten_config(cfg, "", out, 0)
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


# Sentinels for run grouping. Distinct from the *metric* groups behind `g`
# (ALL/config/grad/...): this axis clusters runs by a shared config value, the
# way the W&B workspace "Group runs by..." control does.
GROUP_BY_NONE = "(none)"
GROUP_UNSET = "(unset)"

# A key is only worth offering if it actually partitions. One group per run is
# the ungrouped view with extra ceremony; one group for everything is noise. In
# a real project (166 config keys) this cut the menu to a handful.
GROUP_MIN_VALUES = 2
GROUP_MAX_VALUES = 8
# Group labels head a narrow column and appear in the meta summary. A value
# longer than this (an argv list, a hostfile path) truncates to an identical
# prefix in every column -- visually grouped, but unreadable. Real grouping
# keys are short scalars: tp=1, batch_size=2, model=llama.
GROUP_MAX_LABEL = 20


def group_run_keys(runs: list[dict[str, Any]], max_values: int = GROUP_MAX_VALUES) -> list[str]:
    """Config keys that meaningfully partition `runs`, for the group-by menu."""
    keys = [GROUP_BY_NONE]
    if not runs:
        return keys
    # One pass over the runs, accumulating distinct values per key. The
    # key-major version re-parsed every run's JSON config once per key
    # (166 keys x 100 runs = 16.6k parses), which measured 0.93s and froze
    # the UI on the first `G`. Run-major parses each config exactly once.
    values: dict[str, set[str]] = {}
    for run in runs:
        for key, raw in run_config(run).items():
            if raw is None:
                continue
            label = raw if isinstance(raw, str) else compact(raw, 24)
            bucket = values.setdefault(key, set())
            # Cap the set: a key with a distinct value per run is rejected
            # below anyway, so there is no reason to accumulate hundreds.
            if len(bucket) <= max_values + 1:
                bucket.add(label)
    for key in sorted(values):
        seen = values[key]
        if not GROUP_MIN_VALUES <= len(seen) <= max_values:
            continue
        # Values that only differ past the truncation point look identical in
        # every column header, so the grouping is invisible to the reader.
        if any(len(label) > GROUP_MAX_LABEL for label in seen):
            continue
        # One distinct value per run (run ids, timestamps, hostnames) is the
        # ungrouped view with extra ceremony. Only judge this with enough runs
        # to tell it apart from a genuine 2-of-2 split.
        if len(runs) >= 3 and len(seen) == len(runs):
            continue
        keys.append(key)
    return keys


def group_value(run: dict[str, Any], key: str) -> str:
    """The group a run falls into for `key`.

    Runs missing the key get their own bucket rather than being dropped: a
    hidden run reads as "this run doesn't exist", which is worse than an
    honestly-labelled `(unset)` column.
    """
    raw = run_filter_field(run, key)
    if raw is None:
        return GROUP_UNSET
    return compact(raw, 24) if not isinstance(raw, str) else raw


def _group_sort_key(label: str) -> tuple[int, float, str]:
    """Order groups numerically when possible so 128 precedes 2048."""
    if label == GROUP_UNSET:
        return (2, 0.0, "")  # always last
    # Group labels are always strings by this point, so parse rather than
    # calling as_number (which only narrows already-numeric types).
    try:
        return (0, float(label), "")
    except (TypeError, ValueError):
        return (1, 0.0, label.lower())


@dataclass(frozen=True)
class XAxis:
    """One selectable chart x-axis.

    `source` is the history column the values come from; `relative` subtracts
    the run's first value so runs starting at different times overlay.
    """

    id: str
    label: str
    source: str
    relative: bool = False


# Mirrors the W&B chart x-axis menu. Step first: it is the only axis every run
# always has, so it is the safe default.
X_AXES: tuple[XAxis, ...] = (
    XAxis("step", "Step", "_step"),
    XAxis("relative_process", "Relative Time (Process)", "_runtime"),
    XAxis("relative_wall", "Relative Time (Wall)", "_timestamp", relative=True),
    XAxis("wall", "Wall Time", "_timestamp"),
    XAxis("tokens", "n_tokens_seen", "train/tokens_seen"),
)

# Alternative spellings for the token axis: the column name varies by training
# harness, so accept the common ones rather than forcing one convention.
TOKEN_SOURCES = (
    "train/tokens_seen",
    "n_tokens_seen",
    "train/n_tokens_seen",
    "tokens_seen",
    "train/tokens",
)


def axis_series(metric: dict[str, Any], axis_id: str, count: int) -> list[Any]:
    """X values for `metric` on `axis_id`, or the sample index as a fallback.

    Falls back when the axis is absent OR its length disagrees with the y
    series: a mismatched pair would silently plot point k's y against some
    other point's x, which is worse than an honest index axis.
    """
    xs = (metric.get("axes") or {}).get(axis_id)
    if isinstance(xs, list) and len(xs) == count:
        return xs
    return list(range(count))


def parse_group_keys(expr: str) -> list[str]:
    """Parse the comma-separated group-by expression into ordered keys.

    Order is nesting order: "ws,flavor" nests flavor inside ws. Duplicates are
    dropped -- grouping by a key twice would nest a level inside itself, which
    always yields single-child nodes.
    """
    out: list[str] = []
    for part in (expr or "").split(","):
        key = part.strip()
        if key and key not in out:
            out.append(key)
    return out


@dataclass
class GroupRow:
    """One rendered line: either a group header or a run leaf.

    `run_index` indexes the ORIGINAL runs list, so metric slots (which are
    positional against that list) can be read without permuting anything --
    the mistake that made the previous column-clustering show values under
    the wrong headers.
    """

    label: str
    depth: int
    is_group: bool
    count: int = 0
    run_index: int = -1
    path: str = ""
    collapsed: bool = False


def group_tree_rows(
    runs: list[dict[str, Any]],
    keys: list[str],
    collapsed: frozenset[str] | set[str] | None = None,
) -> list[GroupRow]:
    """Flatten runs into an ordered, indented tree of group headers and leaves.

    Mirrors the W&B workspace "Group runs by..." panel: an ordered key list
    produces nested collapsible groups, each carrying the number of runs
    beneath it, with the runs themselves as leaves.
    """
    hidden = set(collapsed or ())

    def build(items: list[tuple[int, dict[str, Any]]], depth: int, prefix: str) -> list[GroupRow]:
        if depth >= len(keys):
            return [
                GroupRow(label=run_label(run), depth=depth, is_group=False, run_index=idx)
                for idx, run in items
            ]
        key = keys[depth]
        buckets: dict[str, list[tuple[int, dict[str, Any]]]] = {}
        for idx, run in items:
            buckets.setdefault(group_value(run, key), []).append((idx, run))
        rows: list[GroupRow] = []
        for value in sorted(buckets, key=_group_sort_key):
            members = buckets[value]
            label = f"{key}: {value}"
            path = f"{prefix}/{label}" if prefix else label
            is_collapsed = label in hidden or path in hidden
            rows.append(
                GroupRow(
                    label=label,
                    depth=depth,
                    is_group=True,
                    count=len(members),
                    path=path,
                    collapsed=is_collapsed,
                )
            )
            if not is_collapsed:
                rows.extend(build(members, depth + 1, path))
        return rows

    return build(list(enumerate(runs)), 0, "")


def config_filter_values(runs: list[dict[str, Any]], key: str, limit: int = 40) -> list[str]:
    """Distinct values a config key takes across the runs, most common first.

    Answers "what can I even filter on?" -- without this you have to guess
    values. Numeric-looking values sort numerically so 128 precedes 2048.
    """
    from collections import Counter

    counts: Counter[str] = Counter()
    for run in runs:
        raw = run_filter_field(run, key)
        if raw is None:
            continue
        if isinstance(raw, str):
            text = raw
        elif isinstance(raw, bool) or isinstance(raw, int):
            text = str(raw)
        elif isinstance(raw, float):
            # repr, not compact(): a suggestion should read back as what the
            # user would type. compact() renders 0.003 as "0.00300", which
            # still matches numerically but looks like a different value.
            text = repr(raw)
        else:
            text = compact(raw, 40)
        if text != "":
            counts[text] += 1
    if not counts:
        return []

    def order(item: tuple[str, int]) -> Any:
        text, n = item
        num = as_number_str(text)
        return (0, num, "") if num is not None else (1, 0.0, text.lower())

    return [t for t, _ in sorted(counts.items(), key=order)][:limit]


def as_number_str(text: str) -> float | None:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def split_filter_tail(expr: str) -> tuple[str, str]:
    """Split a filter expression into (already-complete prefix, tail being typed).

    Completion only ever rewrites the final whitespace-separated term, so the
    earlier terms of a compound filter are preserved verbatim.
    """
    if not expr or expr[-1].isspace():
        return expr, ""
    parts = expr.rsplit(" ", 1)
    if len(parts) == 1:
        return "", parts[0]
    return parts[0] + " ", parts[1]


def complete_run_filter(expr: str, runs: list[dict[str, Any]], limit: int = 12) -> list[str]:
    """Candidate completions for the filter expression's trailing term.

    Completes the KEY while typing a bare word (`precision/` -> every key under
    that prefix), and the VALUE once an operator has been typed
    (`dim=` -> the values dim actually takes). Each candidate is a full
    replacement expression, so the caller can set the input directly.
    """
    head, tail = split_filter_tail(expr)
    if not tail:
        return []
    # Does the tail already carry an operator? If so we are completing a value.
    for op in FILTER_OPS:
        idx = tail.find(op)
        if idx > 0:
            key = tail[:idx]
            partial = tail[idx + len(op):]
            # Only complete the last alternative of a comma list.
            before, _, frag = partial.rpartition(",")
            lead = f"{before}," if before or partial.endswith(",") else ""
            vals = config_filter_values(runs, key)
            hits = [v for v in vals if v.lower().startswith(frag.strip().lower())]
            return [f"{head}{key}{op}{lead}{v}" for v in hits[:limit]]
    # No operator yet: complete the key.
    keys = config_filter_keys(runs)
    low = tail.lower()
    hits = [k for k in keys if k.lower().startswith(low)]
    if not hits:
        hits = [k for k in keys if low in k.lower()]
    return [f"{head}{k}" for k in hits[:limit]]


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


def metric_name_matcher(search: str) -> Any:
    """Build a predicate over metric names from a search string.

    Plain text is a case-insensitive SUBSTRING match. Regex is opt-in by
    wrapping the pattern in slashes (`/^train/`), because metric names are full
    of regex metacharacters -- `mfu(%)` as a regex matches nothing (the parens
    become a group) and `loss(` is an outright error -- so treating every query
    as a pattern would silently break ordinary searches.

    Returns None when the search is empty (caller keeps everything). Raises
    re.error for a malformed pattern so the caller can surface it.
    """
    if not search or not search.strip():
        return None
    raw = search.strip()
    if len(raw) >= 2 and raw.startswith("/") and raw.endswith("/"):
        body = raw[1:-1]
        if body:
            rx = re.compile(body, re.IGNORECASE)
            return lambda name: rx.search(name) is not None
    q = raw.lower()
    return lambda name: q in name.lower()


def filter_metrics_by_search(metrics: list[dict[str, Any]], search: str) -> list[dict[str, Any]]:
    """Apply a metric-name search, ignoring a malformed regex.

    A half-typed pattern (`/train(`) must not empty the view while the user is
    still typing, so an invalid regex keeps everything rather than raising.
    """
    try:
        match = metric_name_matcher(search)
    except re.error:
        return metrics
    if match is None:
        return metrics
    return [m for m in metrics if match(str(m["name"]))]


def search_error(search: str) -> str:
    """Human-readable reason a search string is not usable, else ''."""
    try:
        metric_name_matcher(search)
    except re.error as e:
        return f"bad regex: {e}"
    return ""


def filtered_multi_metrics(metrics: list[dict[str, Any]], search: str, group: str, sort_mode: str) -> list[dict[str, Any]]:
    out = metrics
    if group != "ALL":
        out = [m for m in out if m["group"] == group]
    out = filter_metrics_by_search(out, search)

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
    out = filter_metrics_by_search(out, search)

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
    """App stylesheet, expressed in THEME VARIABLES rather than literal hex.

    Every color here must come from the active theme ($surface, $panel, $text,
    ...). Hardcoded hex does not follow the theme, so on a light theme the
    inner panes stayed dark while the chrome Textual styles itself (header,
    footer, key hints) correctly flipped -- which reads as the panes being
    "broken" rather than as a deliberate dark-on-light design.
    """
    return """
    Screen { layout: vertical; background: $background; color: $text; }
    /* Neither #meta nor #search_input may dock: two widgets docked to the same
       edge overlap, and the 3-row input was covering the top 3 of the meta
       panel's 5 lines (title, URL, state). Let the vertical layout stack them. */
    /* Fixed, not auto: the meta panel wraps its legend/keys lines, and with
       many runs an auto height grew without bound and squeezed the results
       pane to nothing. Overflow is clipped rather than allowed to push. */
    #meta { height: 6; overflow: hidden; padding: 0 1; color: $text; background: $panel; }
    /* Hidden until summoned with `/` or `f`: two always-on boxes cost 6 fixed
       rows of results even when empty. Revealed on focus, hidden again on
       blur/Esc -- see reveal_input/hide_input. */
    #search_input, #filter_input, #group_input {
        height: 3;
        margin: 0 1;
        background: $surface;
        color: $text;
        border: tall $panel;
        display: none;
    }
    #search_input:focus, #filter_input:focus, #group_input:focus { border: tall $accent; }
    #table { height: 1fr; background: $surface; color: $text; }
    #charts { height: 1fr; background: $surface; color: $text; display: none; }
    /* Tabs must be pinned to its real height: `height: auto` let it expand to
       fill the container, which pushed the chart pane off-screen entirely. */
    #group_tabs { height: 2; display: none; background: $panel; }
    /* Each tile gets a FIXED height and the full pane width, so charts stay a
       readable size and the pane scrolls instead of tiles expanding to consume
       whatever space the run count happens to leave. */
    .chart-tile {
        width: 1fr;
        height: 18;
        margin: 0 1 1 1;
        background: $surface;
        border: round $panel;
    }
    .chart-tile:focus { border: round $accent; }
    .chart-empty { padding: 1; color: $warning; background: $surface; }
    #filter_hint {
        height: 1;
        padding: 0 2;
        background: $panel;
        color: $text-muted;
        display: none;
    }
    #status { dock: bottom; height: 1; color: $text-muted; background: $panel; }
    DataTable { background: $surface; color: $text; }
    DataTable > .datatable--header { background: $panel; color: $accent; text-style: bold; }
    DataTable > .datatable--cursor { background: $primary; color: $text; text-style: bold; }
    DataTable > .datatable--hover { background: $boost; }
    """


def require_textual() -> None:
    import importlib.util

    if importlib.util.find_spec("textual") is None:
        raise SystemExit("Textual is required for interactive mode. Install with `pip install textual`.")


# Named colors for run columns/legend. Deliberately no "white": it disappears
# on a light theme. These seven all keep contrast against either polarity
# because the terminal maps them to its own palette.
RUN_COLORS = ("cyan", "green", "yellow", "magenta", "blue", "red", "bright_blue")

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

KEYS_HINT_RUN = "Keys: q quit | r refresh | / search | Esc clear | g group | s sort column | x reverse | h header | X x-axis"
KEYS_HINT_PROJECT = "Keys: q quit | r refresh | / search | f filter runs | Esc clear | g metric group | G group runs | m mode | s sort column | x reverse | h header | X x-axis"


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


def _finite_mask(values: list[Any]) -> list[int]:
    return [
        i
        for i, v in enumerate(values)
        if isinstance(v, (int, float)) and math.isfinite(float(v))
    ]


def metric_series(metric: dict[str, Any], run_index: int) -> list[float]:
    """The numeric series one run contributes to a metric, or []."""
    slots = metric.get("runs") or []
    slot = slots[run_index] if run_index < len(slots) else None
    if not slot:
        return []
    values = slot.get("values") or []
    return [float(values[i]) for i in _finite_mask(values)]


def metric_axes(metric: dict[str, Any], run_index: int) -> dict[str, Any]:
    """One run's slot, with its axes filtered to match metric_series().

    metric_series drops non-finite points, so the raw stored axis would be
    longer than the plotted y series and pair every later point with the
    wrong x. Applying the same mask keeps them aligned.
    """
    slots = metric.get("runs") or []
    slot = slots[run_index] if run_index < len(slots) else None
    if not slot:
        return {"values": [], "axes": {}}
    values = slot.get("values") or []
    mask = _finite_mask(values)
    axes = {
        axis_id: [col[i] for i in mask]
        for axis_id, col in (slot.get("axes") or {}).items()
        if len(col) == len(values)
    }
    return {"values": [float(values[i]) for i in mask], "axes": axes}


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


# plotext marker for chart series. "hd" packs a 2x2 block per cell; "braille"
# packs 2x4 and so carries more resolution, but its dots are visibly fainter --
# on a real loss curve braille reads as a dotted trace where hd reads as a
# continuous line. Both are 2 subpixels wide, so the width*2 downsample budget
# holds for either. Override with WANDB_TUI_MARKER (braille, hd, fhd, dot, sd).
CHART_MARKER = os.environ.get("WANDB_TUI_MARKER", "hd")


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
    max_points: int | None = None,
    x_axis: str = "step",
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
        # Per-run x values for the selected axis. Each run resolves its own
        # (falling back to sample index independently), so a run missing the
        # axis still plots rather than dropping out of the comparison.
        xs = axis_series(metric_axes(metric, i), x_axis, len(ys))
        if clip:
            pts = [(x, y) for x, y in zip(xs, ys) if xL <= x <= xH and yL <= y <= yH]
            if len(pts) < 2:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
        if max_points and len(ys) > max_points:
            # plotext's build() dominates render cost and scales with point
            # count (measured 0.21s for 24 runs x 5000 pts, 0.01s downsampled).
            # Downsample BOTH axes so x stays in original index units --
            # otherwise a zoom window computed from metric_extent (which uses
            # full-resolution indices) would be in a different scale.
            ys = downsample_series(ys, max_points)
            xs = downsample_series(xs, max_points)
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
        plt.plot(xs, ys, color=color, marker=CHART_MARKER,
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
        color = RUN_COLORS[i % len(RUN_COLORS)]
        text.append(f"R{i+1}=", style=f"bold {color}")
        # The name itself carries an OSC 8 hyperlink, so it is clickable in
        # terminals that support them (kitty, iTerm2, WezTerm, modern VTE) and
        # renders as plain underlined text everywhere else. Only the name is
        # linked -- the "R1=" prefix stays inert so the link target reads
        # cleanly on hover.
        url = run_url(run)
        style = f"bold {color} underline link {url}" if url else f"bold {color}"
        text.append(run_label(run), style=style)
    if len(runs) > limit:
        text.append(f"  (+{len(runs) - limit} more)", style="dim")
    return text


def format_run_meta(run: dict[str, Any], entity: str, project: str, run_id: str, url: str, metrics: list[dict[str, Any]], shown: list[dict[str, Any]], group: str, search: str, sort_mode: str, status: str) -> Any:
    # Must return a Text, not a str: Static.update() parses str content as
    # Textual markup, so a run named "sweep[lr=1e-3]" -- or an error message
    # containing brackets -- would raise MarkupError and kill the app.
    from rich.text import Text

    text = Text()
    text.append(f"W&B Run: {run.get('displayName') or run_id} ({entity}/{project}/{run_id})\n", style="bold")
    text.append(f"URL: {url}\n", style="cyan")
    text.append(
        f"state={run.get('state', '?')}  created={run.get('createdAt', '?')}  updated={run.get('updatedAt', '?')}  rows={run.get('historyLineCount', '?')}\n",
        style="green" if run.get("state") == "finished" else "yellow",
    )
    text.append(
        f"metrics={len(metrics)}  shown={len(shown)}  group={group}  search='{search}'  sort={sort_mode}  {status}\n",
        style="yellow" if status.startswith("ERROR") else "",
    )
    serr = search_error(search)
    if serr:
        text.append(f"search {serr}\n", style="bold red")
    text.append(KEYS_HINT_RUN, style="magenta")
    return text


def format_project_meta(entity: str, project: str, url: str, limit: int, runs: list[dict[str, Any]], metrics: list[dict[str, Any]], shown: list[dict[str, Any]], group: str, search: str, sort_mode: str, chart_mode: bool, status: str, run_filter: str = "", total_runs: int | None = None, filter_error: str = "", visible_runs: int | None = None, group_by: str = GROUP_BY_NONE, x_axis_label: str = "") -> Any:
    from rich.text import Text

    text = Text()
    text.append(f"W&B Project: {entity}/{project}  recent runs={limit}\n", style="bold")
    text.append(f"URL: {url}\n", style="cyan")
    text.append(f"mode={'chart' if chart_mode else 'table'}{('  x=' + x_axis_label) if (chart_mode and x_axis_label) else ''}  metrics={len(metrics)}  shown={len(shown)}  group={group}  search='{search}'  sort={sort_mode}  {status}\n", style="yellow" if status.startswith("ERROR") else "")
    serr = search_error(search)
    if serr:
        text.append(f"search {serr}\n", style="bold red")
    if filter_error:
        text.append(f"filter error: {filter_error}\n", style="bold red")
    elif run_filter:
        total = len(runs) if total_runs is None else total_runs
        text.append(f"runs: {len(runs)}/{total} matching  filter='{run_filter}'\n", style="bold green")
    group_keys = parse_group_keys(group_by)
    if group_keys:
        # Summarise the OUTERMOST level only. Passing the whole comma-joined
        # expression to group_value looks up a key that does not exist, which
        # bucketed every run into "(unset)" while the tree below showed real
        # groups -- the summary contradicted the thing it summarised.
        counts: dict[str, int] = {}
        for run in runs:
            label = group_value(run, group_keys[0])
            counts[label] = counts.get(label, 0) + 1
        # Cap both the label and the count of groups shown: the meta panel is
        # a fixed 6 rows, and an over-long line wraps into the results pane.
        ordered = sorted(counts.items(), key=lambda kv: _group_sort_key(kv[0]))
        shown_groups = ordered[:6]
        summary = "  ".join(
            f"{compact(label, GROUP_MAX_LABEL)}({n})" for label, n in shown_groups
        )
        if len(counts) > len(shown_groups):
            summary += f"  (+{len(counts) - len(shown_groups)} more)"
        nested = f" (+{len(group_keys) - 1} nested)" if len(group_keys) > 1 else ""
        text.append(
            f"grouped by {group_keys[0]}{nested}: {summary}\n", style="bold cyan"
        )
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
    ("h", "toggle_header", "Header"),
    ("X", "cycle_x_axis", "X axis"),
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

    def reveal_input(self, selector: str) -> None:
        """Show a hidden input and move focus into it."""
        try:
            box = self.query_one(selector)
        except Exception:
            return
        box.display = True
        box.focus()

    def hide_input(self, selector: str) -> None:
        """Hide an input again, but only while it holds no text.

        A box with a live query stays visible even unfocused: hiding it would
        leave the results filtered by something the user can no longer see.
        """
        try:
            box = self.query_one(selector)
        except Exception:
            return
        if not box.value:
            box.display = False

    def hide_idle_inputs(self) -> None:
        for selector in ("#search_input", "#filter_input", "#group_input"):
            if getattr(self.focused, "id", None) == selector.lstrip("#"):
                continue
            self.hide_input(selector)

    def on_descendant_blur(self, event: Any = None) -> None:
        # Blur fires BEFORE the new focus lands, so defer: checking now would
        # read the outgoing focus and hide a box we are tabbing into.
        self.call_after_refresh(self.hide_idle_inputs)

    def action_focus_search(self) -> None:
        self.reveal_input("#search_input")

    def action_clear_search(self) -> None:
        # Clear whichever box has focus; if focus is elsewhere (e.g. the
        # results table), clear both -- Esc from the results means "drop all
        # filtering", not "do nothing".
        focused_id = getattr(self.focused, "id", None)
        targets = ("search_input", "filter_input", "group_input")
        selective = focused_id in targets
        cleared_filter = False
        cleared_groups = False
        for selector in ("#search_input", "#filter_input", "#group_input"):
            if selective and focused_id != selector.lstrip("#"):
                continue
            try:
                box = self.query_one(selector)
            except Exception:
                continue
            if selector != "#group_input":
                box.value = ""
            if selector == "#search_input":
                self.search = ""
            elif selector == "#filter_input":
                self.run_filter = ""
                cleared_filter = True
            else:
                # Deliberately NOT cleared here. Esc from the group box means
                # "put the box away and let me drive the tree" -- wiping the
                # grouping would make collapse unreachable, since Esc is the
                # obvious way out of a text box. `G` then Esc-on-empty is the
                # path to ungrouping (handled below).
                if not self.group_expr:
                    cleared_groups = True
        # Hand focus back to the table so the single-letter bindings work again
        # instead of typing into the box the user just cleared.
        self.focus_results_pane()
        # Now empty, so these collapse and give their rows back to the results.
        for selector in ("#search_input", "#filter_input", "#group_input"):
            self.hide_input(selector)
        if cleared_groups:
            self.rebuild_columns()
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

    def x_axis(self) -> XAxis:
        return X_AXES[getattr(self, "x_axis_idx", 0) % len(X_AXES)]

    def axis_available(self, axis_id: str) -> bool:
        """Whether any loaded series actually carries this axis.

        Single-run metrics hold `axes` directly; multi-run metrics hold one
        slot per run, each with its own. Checking only the top level reported
        "not logged" for every axis in the project view, contradicting charts
        that were plainly using it.
        """
        for metric in self.metrics:
            if (metric.get("axes") or {}).get(axis_id):
                return True
            for slot in metric.get("runs") or []:
                if slot and (slot.get("axes") or {}).get(axis_id):
                    return True
        return False

    def action_cycle_x_axis(self) -> None:
        """Cycle the chart x-axis (Step -> Relative -> Wall -> tokens)."""
        self.x_axis_idx = (getattr(self, "x_axis_idx", 0) + 1) % len(X_AXES)
        axis = self.x_axis()
        # Say which axis is active AND whether the data actually supports it:
        # silently falling back to sample index would look like the key did
        # nothing.
        if self.metrics and not self.axis_available(axis.id):
            self.notify(f"x-axis: {axis.label} (not logged; using sample index)", severity="warning")
        else:
            self.notify(f"x-axis: {axis.label}")
        self.render_table()

    def action_toggle_header(self) -> None:
        """Collapse the meta panel to reclaim its 6 fixed rows for results.

        The panel is a fixed height rather than auto (see the #meta CSS), so on
        a short terminal it is a large constant cost. Hiding it also hides the
        Header bar, since the two together are what reads as "the header".
        """
        self.header_hidden = not getattr(self, "header_hidden", False)
        self.apply_header_visibility()

    def apply_header_visibility(self) -> None:
        hidden = getattr(self, "header_hidden", False)
        for selector in ("#meta", "Header"):
            for widget in self.query(selector):
                widget.display = not hidden


# Metric columns shown beside the group tree. The tree eats the horizontal
# budget with indentation and long run names, so only the first few metrics
# fit; the ungrouped view remains the way to scan many metrics at once.
MAX_TREE_METRIC_COLS = 4


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
        # A tile looks its metric up by NAME on every draw rather than owning
        # the dict it was built with. build_multi_metrics returns fresh dicts on
        # every refresh, so a captured dict silently goes stale and the chart
        # freezes -- precisely on the live runs that auto-refresh exists for.
        can_focus = True

        def __init__(self, name: str, provider: Any, run_count: int, labels: list[str], axis_provider: Any = None, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.metric_name = name
            self._provider = provider
            # Read live rather than captured at construction: tiles are reused
            # across renders, so a captured axis would go stale after `X`.
            self._axis_provider = axis_provider or (lambda: "step")
            self.run_count = run_count
            self.labels = labels

        @property
        def metric(self) -> dict[str, Any] | None:
            return self._provider(self.metric_name)

        def rebind(self, run_count: int, labels: list[str]) -> None:
            self.run_count = run_count
            self.labels = labels

        def on_mount(self) -> None:
            # Transparent canvas so the plot inherits the app theme instead of
            # painting its own black background under a light theme.
            try:
                self.theme = "textual-clear"
            except Exception:
                pass
            self.border_title = self.metric_name
            self.replot()

        def replot(self) -> None:
            metric = self.metric
            if metric is None:
                return  # the metric vanished from the latest refresh
            self._drawn_width = self.size.width
            # 2 samples per column: hd (and braille) pack 2 subpixels
            # horizontally, so 1/column would throw away half the resolution.
            budget = max(40, (self.size.width or 60) * 2)
            draw_metric_plot(self.plt, metric, self.run_count, self.labels, title="", max_points=budget, x_axis=self._axis_provider())
            self.refresh()

        def on_resize(self, event: Any = None) -> None:
            # The downsample budget is width-derived, so a resize needs a real
            # re-draw, not just plotext's re-build at the new size.
            if self.size.width != getattr(self, "_drawn_width", None):
                self.replot()

        def on_click(self) -> None:
            self.focus()
            open_full = getattr(self.app, "open_chart_fullscreen", None)
            if open_full is not None:
                open_full(self.metric_name)

    return MetricChart


def chart_zoom_screen():
    """Full-screen single-chart view with pan/zoom/focus, mirroring prod_dash.

    Interaction: z cycles run focus (fit to that run, dim others), Z/0 reset,
    +/- zoom, h/l pan x, j/k pan y, L log/linear, esc back to the grid.
    """
    require_plotext_widget()
    from textual.app import ComposeResult
    from textual.screen import ModalScreen
    from textual.widgets import Footer, Header, Static
    from textual_plotext import PlotextPlot

    # MUST be modal, not a plain Screen. Textual's binding chain only stops at
    # a screen whose is_modal is True; over a plain Screen the app's own
    # bindings still fire, so s/m/g reached ProjectApp and mutated the hidden
    # grid underneath (verified: pressing them re-sorted and toggled the chart
    # mode of the view you had zoomed out of). Modal also keeps App.query_one
    # from resolving against this screen for the app's #table/#filter_input.
    class ChartZoomScreen(ModalScreen[None]):
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
        /* Opaque: ModalScreen defaults to a 60% wash that would show the grid
           bleeding through behind the chart. */
        ChartZoomScreen { background: $background; }
        #zoom_plot { width: 1fr; height: 1fr; background: $surface; }
        #zoom_meta { dock: top; height: auto; padding: 0 1; background: $panel; color: $text; }
        #zoom_status { dock: bottom; height: 1; padding: 0 1; background: $panel; color: $text-muted; }
        """

        def __init__(self, name: str, provider: Any, run_count: int, labels: list[str], x_axis_id: str = "step") -> None:
            super().__init__()
            # Same rebind-by-name reasoning as MetricChart: an open zoom view
            # would otherwise stay frozen on the dict it was constructed with
            # for its whole lifetime.
            self.metric_name = name
            self._provider = provider
            self.x_axis_id = x_axis_id
            self.run_count = run_count
            self.labels = labels
            self.xlim: tuple[float | None, float | None] = (None, None)
            self.ylim: tuple[float | None, float | None] = (None, None)
            self.focus_run: int | None = None
            self.focus_idx = -1
            self.ylog = False

        @property
        def metric(self) -> dict[str, Any] | None:
            return self._provider(self.metric_name)

        def rebind(self, run_count: int, labels: list[str]) -> None:
            # Deliberately does NOT reset xlim/ylim/focus: a live chart that
            # threw away your zoom on every refresh would be worse than a
            # frozen one.
            self.run_count = run_count
            self.labels = labels

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
            metric = self.metric
            if metric is None:
                return  # metric gone from the latest refresh; keep last frame
            plot = self.query_one("#zoom_plot", PlotextPlot)
            drawn = draw_metric_plot(
                plot.plt, metric, self.run_count, self.labels,
                xlim=self.xlim, ylim=self.ylim, focus_run=self.focus_run,
                ylog=self.ylog, title="", x_axis=self.x_axis_id,
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
            head.append(f"{self.metric_name}\n", style="bold")
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
            metric = self.metric
            if metric is None:
                return
            drawable = [i for i in range(self.run_count) if len(metric_series(metric, i)) > 1]
            if not drawable:
                return
            self.focus_idx += 1
            if self.focus_idx >= len(drawable):
                # Wrapped past the end: back to showing everything.
                self.action_reset_view()
                return
            idx = drawable[self.focus_idx]
            ext = metric_extent(metric, idx)
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
            xspan, _ = metric_span(self.metric or {}, self.run_count)
            span = xspan or (0.0, 1.0)
            lo = self.xlim[0] if self.xlim[0] is not None else span[0]
            hi = self.xlim[1] if self.xlim[1] is not None else span[1]
            return lo, hi

        def _ywin(self) -> tuple[float, float]:
            _, yspan = metric_span(self.metric or {}, self.run_count)
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
            yield Input(placeholder="Search metrics (plain text, or /regex/)", id="search_input")
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
                cells.append(rich_cell(str(m["count"]), "", COUNT_COL_WIDTH))
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
            # Shift-G, distinct from `g` (metric groups). Two different axes:
            # `g` filters which metrics are listed, `G` clusters which runs are
            # adjacent. Sharing a key would conflate them.
            ("G", "focus_group_by", "Group runs"),
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
            # Run grouping (the `G` axis): which config key clusters the run
            # columns. Recomputed on refresh, since the candidate keys depend
            # on which runs are loaded.
            # Run grouping (`G`): an ORDERED list of config keys producing a
            # nested collapsible tree, like the W&B workspace panel. Empty ==
            # ungrouped flat run list.
            self.group_expr = ""
            self.group_keys: list[str] = []
            self.collapsed_groups: set[str] = set()
            self.run_filter = run_filter
            self.filter_error = ""
            self.chart_mode = False
            self.status = "loading…"
            self.render_timer = None
            self.refresh_in_flight = False
            self.pending_nodes: list[dict[str, Any]] = []
            self.pending_truncated = False
            self._metric_index: dict[str, dict[str, Any]] = {}

        def compose(self) -> ComposeResult:
            yield Header()
            yield Static("loading…", id="meta")
            yield Input(placeholder="Search metrics (plain text, or /regex/)", id="search_input")
            yield Input(placeholder="Filter runs by config, e.g. lr>=0.001 model~llama  (tab completes)", id="filter_input")
            yield Input(placeholder="Group runs by config keys, e.g. world_size,model_spec.flavor  (tab completes)", id="group_input")
            yield Static("", id="filter_hint")
            yield Tabs(Tab("ALL", id="grp_ALL"), id="group_tabs")
            yield FittedDataTable(id="table", cursor_type="row", zebra_stripes=True)
            yield VerticalScroll(id="charts")
            yield Static("", id="status")
            yield Footer()

        def on_mount(self) -> None:
            if self.run_filter:
                # A filter passed via --filter must be VISIBLE even though the
                # boxes default to hidden: the run list is already narrowed, and
                # an invisible cause is worse than a wasted row.
                box = self.query_one("#filter_input", Input)
                box.value = self.run_filter
                box.display = True
            self.rebuild_columns()
            self.action_refresh_data()
            if self.refresh_seconds:
                self.set_interval(self.refresh_seconds, self.refresh_if_live)

        def action_focus_group_by(self) -> None:
            self.reveal_input("#group_input")
            self.call_after_refresh(self.update_group_hint)

        def apply_group_keys(self) -> None:
            """Apply the group box's current text (debounce target)."""
            self.set_group_keys(self.group_expr)

        def set_group_keys(self, expr: str) -> None:
            """Apply a comma-separated group-by expression.

            Collapse state is keyed on group labels, which are meaningless
            once the key list changes, so it resets with the expression.
            """
            keys = parse_group_keys(expr)
            if keys != self.group_keys:
                self.collapsed_groups = set()
            self.group_expr = expr
            self.group_keys = keys
            self.rebuild_columns()
            self.render_table()

        def completion_universe(self) -> tuple[list[str], set[str]]:
            """(all config keys, keys that partition) for the loaded runs.

            Cached against the run set: group_run_keys walks every config key
            against every run, which measured ~0.1s for 166 keys x 10 runs.
            Recomputing it per keystroke made typing a key name visibly
            laggy (~0.18s per character) even though the answer cannot change
            between keystrokes -- only a refetch changes it.
            """
            token = id(self.runs), len(self.runs)
            if getattr(self, "_completion_token", None) != token:
                self._completion_token = token
                self._completion_all = config_filter_keys(self.runs)
                self._completion_split = set(group_run_keys(self.runs)) - {GROUP_BY_NONE}
            return self._completion_all, self._completion_split

        def group_candidates(self) -> list[str]:
            """Completions for the key being typed after the last comma.

            Every config key is offered -- the user may know something the
            partition heuristic does not -- but keys that actually split the
            runs sort first, since those are almost always what is wanted.
            """
            prefix = (self.group_expr or "").split(",")[-1].strip()
            all_keys, partitioning = self.completion_universe()
            hits = [k for k in all_keys if k.startswith(prefix)]
            return sorted(hits, key=lambda k: (k not in partitioning, k))

        def update_group_hint(self) -> None:
            hint = self.query_one("#group_input", Input)
            cands = self.group_candidates()[:8]
            if cands:
                hint.placeholder = "  ".join(cands)
            else:
                hint.placeholder = "no matching config keys"

        def action_focus_filter(self) -> None:
            self.reveal_input("#filter_input")
            # focus() lands on the next message-pump cycle, so the hint has to
            # be refreshed after it, not inline (it would read the old focus).
            self.call_after_refresh(self.update_filter_hint)

        def on_descendant_focus(self, event: Any = None) -> None:
            self.update_filter_hint()

        def on_descendant_blur(self, event: Any = None) -> None:
            # This overrides the mixin's handler, so it must ALSO do the
            # auto-hide; otherwise the boxes would never collapse in the
            # project app (the one that has both of them).
            def settle() -> None:
                self.hide_idle_inputs()
                self.update_filter_hint()

            self.call_after_refresh(settle)

        def filter_candidates(self) -> list[str]:
            # Complete against the full metadata set: all_runs carries every
            # run's config even when history is capped, so the suggestions
            # describe what is actually filterable.
            return complete_run_filter(self.run_filter, self.all_runs or self.runs)

        def update_filter_hint(self) -> None:
            """Show what can come next: matching config keys, or a key's values."""
            from rich.text import Text

            try:
                hint = self.query_one("#filter_hint", Static)
            except Exception:
                return
            focused = getattr(self.focused, "id", None) == "filter_input"
            if not focused:
                hint.display = False
                return
            head, tail = split_filter_tail(self.run_filter)
            cands = self.filter_candidates()
            text = Text()
            if not tail:
                keys = config_filter_keys(self.all_runs or self.runs)
                # Lead with namespaces (keys that have children). Flattening
                # nested configs pushed the key count from ~107 to ~166, so an
                # alphabetical head is mostly noise -- "args/" tells you where
                # the interesting knobs live, "DEVICE" does not.
                spaces = sorted({k.split(".", 1)[0] for k in keys if "." in k})
                flat = [k for k in keys if "." not in k]
                if spaces:
                    text.append("groups: ", style="dim")
                    text.append("  ".join(f"{s}." for s in spaces[:6]), style="bold cyan")
                    text.append("   keys: ", style="dim")
                    text.append("  ".join(flat[:6]) or "(none)", style="cyan")
                else:
                    text.append("keys: ", style="dim")
                    text.append("  ".join(flat[:10]) or "(none)", style="cyan")
                text.append(f"  ({len(keys)} total; type a prefix)", style="dim")
            elif cands:
                # Show only the part being completed, not the whole expression.
                shown = [c[len(head):] for c in cands]
                text.append("tab: ", style="dim")
                text.append("  ".join(shown[:8]), style="bold cyan")
                if len(cands) > 8:
                    text.append(f"  (+{len(cands) - 8})", style="dim")
            else:
                text.append("no matching config keys/values", style="yellow")
            hint.update(text)
            hint.display = True

        def on_key(self, event: Any) -> None:
            # Tab normally moves focus; inside the filter box it completes
            # instead. Only swallow it when there is something to complete, so
            # tab still escapes the box once the term is finished.
            if event.key == "tab" and getattr(self.focused, "id", None) == "group_input":
                cands = self.group_candidates()
                typed = (self.group_expr or "").split(",")[-1].strip()
                if cands and cands != [typed]:
                    event.prevent_default()
                    event.stop()
                    self.complete_group_key()
                    return
            if event.key == "tab" and getattr(self.focused, "id", None) == "filter_input":
                cands = self.filter_candidates()
                # A finished term still matches itself ("dim=128" -> ["dim=128"]),
                # so completing would be a no-op. Let tab fall through to focus
                # movement in that case, or the box becomes a trap.
                if cands and cands != [self.run_filter]:
                    event.prevent_default()
                    event.stop()
                    self.complete_filter()

        def on_input_submitted(self, event: Any) -> None:
            if getattr(event.input, "id", None) == "group_input":
                # Enter accepts and gets out of the way; the grouping is
                # already applied from on_input_changed.
                self.focus_results_pane()
                return
            if getattr(event.input, "id", None) == "filter_input":
                self.complete_filter()

        def complete_group_key(self) -> None:
            """Complete the key after the last comma, leaving earlier keys alone."""
            cands = self.group_candidates()
            if not cands:
                return
            if len(cands) == 1:
                value = cands[0]
            else:
                value = os.path.commonprefix(cands)
                typed = (self.group_expr or "").split(",")[-1].strip()
                if len(value) <= len(typed):
                    value = cands[0]
            parts = (self.group_expr or "").split(",")
            parts[-1] = value
            expr = ",".join(p.strip() for p in parts)
            box = self.query_one("#group_input", Input)
            box.value = expr
            box.cursor_position = len(expr)
            self.set_group_keys(expr)
            self.update_group_hint()

        def complete_filter(self) -> None:
            """Accept the first candidate, or extend to the common prefix.

            Extending to the longest common prefix (rather than jumping to the
            first hit) means tab narrows predictably when several keys share a
            namespace, the way shell completion does.
            """
            cands = self.filter_candidates()
            if not cands:
                return
            if len(cands) == 1:
                value = cands[0]
            else:
                value = os.path.commonprefix(cands)
                if len(value) <= len(self.run_filter):
                    value = cands[0]
            inp = self.query_one("#filter_input", Input)
            inp.value = value
            inp.cursor_position = len(value)
            self.run_filter = value
            self.update_filter_hint()
            self.schedule_render(self.apply_run_filter)

        def on_input_changed(self, event: Input.Changed) -> None:
            if event.input.id == "filter_input":
                self.run_filter = event.value
                self.update_filter_hint()
                self.schedule_render(self.apply_run_filter)
            elif event.input.id == "group_input":
                # Debounced like the other boxes. Regrouping needs no refetch,
                # but it does rebuild the columns and re-render the whole tree,
                # and every intermediate prefix of a key name ("w", "wo",
                # "wor", ...) is a DIFFERENT valid grouping that would be built
                # in full and immediately thrown away. Typing a two-key
                # expression fired ~27 of those.
                self.group_expr = event.value
                self.update_group_hint()
                self.schedule_render(self.apply_group_keys)
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
            # The invariant now spans six fields: all_runs, runs, metrics,
            # _metric_index, sort_columns and the derived indices must land
            # together, and nothing that READS them may run in between.
            # _metric_index is built here from the argument -- not lazily and
            # not on the worker -- so it can never be derived from a
            # half-updated self.metrics.
            self.all_runs = all_runs
            self.runs = runs
            # Grouping never reorders self.runs -- the tree carries a
            # run_index into this list instead. That keeps metric slots (which
            # are positional against it) valid by construction, rather than
            # needing to be permuted in lockstep.
            self.metrics = metrics
            self._metric_index = {str(m["name"]): m for m in metrics}
            self.sort_columns = sort_columns
            self.sort_idx = min(self.sort_idx, len(self.sort_columns) - 1)
            self.groups = groups
            self.group_idx = min(self.group_idx, len(self.groups) - 1)
            # Candidate group keys depend on which runs loaded, so re-derive
            # them here. Keep the current key selected if it still partitions;
            # otherwise fall back to ungrouped rather than silently jumping to
            # an unrelated key.
            self.status = status
            self.filter_error = filter_error
            self.refresh_in_flight = False
            self.rebuild_columns()
            self.render_table()
            # An open zoom view holds its own widget tree, so it does not go
            # through render_table; re-point it at the new data explicitly.
            for scr in self.screen_stack[1:]:
                if isinstance(scr, ChartZoomScreen):
                    scr.rebind(len(self.runs), self.run_labels())
                    scr.redraw()

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
            table.header_height = 1
            if self.group_keys:
                # Grouped view transposes the grid: rows become the run tree
                # (that is what a group hierarchy nests), so the columns have
                # to become metrics. Ungrouped keeps metrics-as-rows, which is
                # the better shape when there is no hierarchy to show.
                total = width or self.table_width()
                shown = self.shown_metrics()
                self.tree_metrics = [str(m["name"]) for m in shown[:MAX_TREE_METRIC_COLS]]
                name_w = max(28, min(52, total - 8 - 12 * len(self.tree_metrics)))
                self.name_w, self.col_w = name_w, 12
                self.visible_runs = len(self.runs)
                table.clear(columns=True)
                table.add_columns("Group / Run", "n", *self.tree_metrics)
                return
            name_w, col_w, visible = fit_project_widths(total_w := (width or self.table_width()), len(self.runs))
            self.name_w, self.col_w, self.visible_runs = name_w, col_w, visible
            self.tree_metrics = []
            table.clear(columns=True)
            # Only declare the run columns that actually fit; the rest would be
            # clipped off the right edge with no indication they exist.
            count = max(1, min(len(self.runs), visible))
            labels = [f"R{i+1:02d}" for i in range(count)]
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
            labels = self.run_labels()
            if signature == getattr(self, "_chart_signature", None):
                # Same metrics and run count: keep the widgets and re-point them
                # at the current data rather than tearing down the whole grid.
                for tile in pane.query(MetricChart):
                    tile.rebind(len(self.runs), labels)
                    tile.replot()
                return
            self._chart_signature = signature
            # Remember which tile had focus: remove_children() moves focus to
            # the container, which would yank it out of the grid on every
            # keystroke in the search box.
            focused_name = getattr(self.focused, "metric_name", None)
            pane.remove_children()
            if not numeric:
                pane.mount(Static("No numeric metrics with history to chart.", classes="chart-empty"))
                return
            for name in wanted:
                # No `id=` on tiles: remove_children() is async, so the old
                # widgets are still registered when these mount in the same
                # tick and a stable id would raise DuplicateIds.
                tile = MetricChart(name, self.metric_by_name, len(self.runs), labels, axis_provider=lambda: self.x_axis().id, classes="chart-tile")
                pane.mount(tile)
            if focused_name in wanted:
                for tile in pane.query(MetricChart):
                    if tile.metric_name == focused_name:
                        tile.focus()
                        break
            if len(numeric) > len(wanted):
                pane.mount(
                    Static(
                        f"… {len(numeric) - len(wanted)} more metrics; search to narrow.",
                        classes="chart-empty",
                    )
                )

        def metric_by_name(self, name: str) -> dict[str, Any] | None:
            """Current dict for a metric name -- the tiles' data source."""
            return getattr(self, "_metric_index", {}).get(name)

        def open_chart_fullscreen(self, name: str) -> None:
            self.push_screen(ChartZoomScreen(name, self.metric_by_name, len(self.runs), self.run_labels(), x_axis_id=self.x_axis().id))

        def action_open_chart(self) -> None:
            """Enter on a focused chart tile opens it full-screen.

            In the grouped tree, Enter on a group row expands/collapses it
            instead -- that is the primary interaction there, and chart tiles
            only exist in chart mode anyway.
            """
            focused = self.focused
            if isinstance(focused, MetricChart):
                self.open_chart_fullscreen(focused.metric_name)
                return
            if self.group_keys and not self.chart_mode:
                self.toggle_selected_group()

        def on_data_table_row_selected(self, event: Any) -> None:
            """Enter inside the table.

            DataTable binds `enter` itself and emits RowSelected, so the
            app-level `enter` binding never fires while the table has focus --
            the toggle has to hang off this message instead.
            """
            if self.group_keys and not self.chart_mode:
                self.toggle_selected_group()

        def toggle_selected_group(self) -> None:
            """Expand/collapse the group row under the table cursor."""
            table = self.query_one("#table", DataTable)
            rows = getattr(self, "tree_rows", None) or []
            idx = table.cursor_row
            if not (0 <= idx < len(rows)):
                return
            row = rows[idx]
            if not row.is_group:
                return
            # Keyed on the full path, so "flavor: 20b" under two different
            # parents collapse independently.
            if row.path in self.collapsed_groups:
                self.collapsed_groups.discard(row.path)
            else:
                self.collapsed_groups.add(row.path)
            self.render_table()
            # Keep the cursor on the row just toggled rather than letting it
            # jump to the top as rows appear/disappear beneath it.
            try:
                table.move_cursor(row=min(idx, len(self.tree_rows) - 1))
            except Exception:
                pass

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

        def shown_metrics(self) -> list[dict[str, Any]]:
            """Metrics passing the current search + metric-group filter."""
            shown = filtered_multi_metrics(self.metrics, self.search, self.current_group(), "group")
            key = self.sort_columns[self.sort_idx][1]
            return sorted(shown, key=lambda m: self.sort_value(m, key), reverse=self.sort_reverse)

        def render_group_tree(self, shown: list[dict[str, Any]]) -> None:
            """Render runs as an indented, collapsible group hierarchy.

            Rows are the tree; columns are the first few metrics. A leaf reads
            its values via row.run_index into self.runs -- the runs list is
            never reordered, so a slot cannot drift away from its run.
            """
            table = self.query_one("#table", DataTable)
            names = getattr(self, "tree_metrics", None) or []
            if len(table.columns) != len(names) + 2:
                self.rebuild_columns()
                names = getattr(self, "tree_metrics", None) or []
            by_name = {str(m["name"]): m for m in shown}
            rows = group_tree_rows(self.runs, self.group_keys, self.collapsed_groups)
            self.tree_rows = rows
            name_w = getattr(self, "name_w", NAME_CELL_WIDTH)
            for row in rows:
                indent = "  " * row.depth
                if row.is_group:
                    marker = "\u25b6" if row.collapsed else "\u25bc"
                    label = rich_cell(f"{indent}{marker} {row.label}", "bold cyan", name_w)
                    cells = [label, rich_cell(str(row.count), "cyan", 4)]
                    # Group rows summarise nothing numerically -- aggregation
                    # was explicitly out of scope -- so leave metric cells blank
                    # rather than inventing a number the user did not ask for.
                    cells += [rich_cell("", "", 12) for _ in names]
                else:
                    style = RUN_COLORS[row.run_index % len(RUN_COLORS)]
                    label = rich_cell(f"{indent}    {row.label}", style, name_w)
                    cells = [label, rich_cell("", "", 4)]
                    for metric_name in names:
                        metric = by_name.get(metric_name)
                        slots = (metric or {}).get("runs") or []
                        slot = slots[row.run_index] if row.run_index < len(slots) else None
                        value = slot.get("latest") if slot else None
                        cells.append(
                            rich_cell(compact(value, 12), style, 12)
                            if slot
                            else rich_cell("\u00b7", "bright_black", 12)
                        )
                table.add_row(*cells)

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
            elif self.group_keys:
                self.render_group_tree(shown)
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
                        else rich_cell("·", "dim", col_w)
                        for i, slot in enumerate(slots)
                    ]
                    vals += [rich_cell("·", "dim", col_w)] * (width - len(vals))
                    table.add_row(rich_cell(m["name"], metric_style(m), name_w), *vals)
            self.query_one("#meta", Static).update(format_project_meta(self.entity, self.project, self.url, self.limit, self.runs, self.metrics, shown, self.current_group(), self.search, self.sort_label(), self.chart_mode, self.status, self.run_filter, len(self.all_runs), self.filter_error, None if self.chart_mode else getattr(self, "visible_runs", None), ",".join(self.group_keys), self.x_axis().label))
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
            box = self.query_one("#search_input", Input)
            box.display = True
            box.focus()

        def action_clear_search(self) -> None:
            self.search = ""
            box = self.query_one("#search_input", Input)
            box.value = ""
            # Hand focus back so the letter bindings work again.
            self.query_one("#table").focus()
            box.display = False
            self.render_rows()

        def on_descendant_blur(self, event: Any = None) -> None:
            def settle() -> None:
                box = self.query_one("#search_input", Input)
                # Keep a non-empty query visible: the row count is filtered by
                # it, so hiding it would strand the user with no way to see why.
                if not box.value and not box.has_focus:
                    box.display = False

            self.call_after_refresh(settle)

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
