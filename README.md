# wandb-tui-compare

A lightweight terminal dashboard for comparing Weights & Biases runs directly from W&B project or run URLs.

It was built for remote/cloud W&B runs when you want a LEET-like terminal view without needing the original local `wandb/` run directories.

## Features

- Single-run metric dashboard
- Multi-run project comparison from a W&B project URL
- W&B-style per-run colored columns
- `plotext` line charts for multi-run metric overlays
- Metric search, group filtering, sorting, and refresh
- JSON export for downstream analysis
- Works against public runs without `wandb` installed; uses `WANDB_API_KEY` automatically for private runs

## Install

```bash
python -m pip install -r requirements.txt
```

## Usage

### Compare recent runs in a project

```bash
python wandb_tui_compare.py   'https://wandb.ai/aurora_gpt/ezpz.examples.fsdp_tp?nw=nwuserforemans'   --runs 8
```

Press `m` to toggle from table mode to plot mode.

### View a single run

```bash
python wandb_tui_compare.py   https://wandb.ai/aurora_gpt/ezpz.examples.fsdp_tp/runs/vrxuo55p
```

### Non-interactive table snapshot

```bash
python wandb_tui_compare.py   'https://wandb.ai/aurora_gpt/ezpz.examples.fsdp_tp?nw=nwuserforemans'   --runs 8   --once   --search train/loss   --group train
```

### Export JSON

```bash
python wandb_tui_compare.py   'https://wandb.ai/aurora_gpt/ezpz.examples.fsdp_tp?nw=nwuserforemans'   --runs 8   --json /tmp/wandb_project_metrics.json
```

## Controls

| Key | Action |
| --- | --- |
| `q` | Quit |
| `↑` / `↓`, `j` / `k` | Scroll |
| `PgUp` / `PgDn` | Page scroll |
| `Home` / `End` | Jump |
| `/` | Search metric names |
| `Esc` | Clear search |
| `g` | Cycle metric group filter |
| `m` | Toggle table/chart mode in project view |
| `s` | Cycle sort mode |
| `r` | Refresh from W&B |
| `?` | Help |

## W&B LEET comparison

W&B's official LEET TUI is excellent for local `wandb/` directories and `.wandb` files, and newer versions support remote single-run URLs. This tool focuses on remote multi-run project comparisons over W&B's GraphQL API.

## Notes

- Public W&B projects/runs can be queried without authentication.
- For private projects, set `WANDB_API_KEY` in your environment.
- `plotext` is optional at runtime; if unavailable, chart mode falls back to a crude curses renderer.
