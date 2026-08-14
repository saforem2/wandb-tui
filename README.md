# wandb-tui

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

## Install / run with uv

```bash
uv sync
```

Run the CLI entrypoint:

```bash
uv run wandb-tui --help
```

Or run it as a Python module:

```bash
uv run python -m wandb_tui --help
```

## Usage

### Interactive startup picker

If you launch without a W&B URL or `ENTITY/PROJECT`, `wandb-tui` opens a startup picker:

```bash
uvx --from wandb-tui wandb-tui
```

The picker lets you choose a W&B owner/entity, then a project, then opens the multi-run project dashboard. For private entities, set `WANDB_API_KEY` first.

### Compare recent runs in a project

```bash
uv run wandb-tui \
  'https://wandb.ai/aurora_gpt/ezpz.examples.fsdp_tp?nw=nwuserforemans' \
  --runs 8
```

Press `m` to toggle from table mode to plot mode.

### View a single run

```bash
uv run wandb-tui \
  https://wandb.ai/aurora_gpt/ezpz.examples.fsdp_tp/runs/vrxuo55p
```

### Non-interactive table snapshot

```bash
uv run wandb-tui \
  'https://wandb.ai/aurora_gpt/ezpz.examples.fsdp_tp?nw=nwuserforemans' \
  --runs 8 \
  --once \
  --search train/loss \
  --group train
```

### Export JSON

```bash
uv run wandb-tui \
  'https://wandb.ai/aurora_gpt/ezpz.examples.fsdp_tp?nw=nwuserforemans' \
  --runs 8 \
  --json /tmp/wandb_project_metrics.json
```

## Controls

| Key | Action |
| --- | --- |
| `q` | Quit |
| `↑` / `↓` | Move row cursor |
| `PgUp` / `PgDn` | Page scroll |
| `Home` / `End` | Jump to first/last row |
| `Tab` | Move focus between search box and results |
| `/` | Focus the search box (filters metric names) |
| `f` | Focus the run filter box (project view) |
| `Esc` | Clear the focused box, or both when the results have focus |
| `g` | Cycle metric group filter |
| `m` | Toggle table/chart mode in project view |
| `s` | Cycle sort column |
| `x` | Reverse sort direction |
| `r` | Refresh from W&B |

Single-letter keys act on the results pane. While a text box has focus they
are typed as text instead — press `Tab` or `Esc` to return focus to the results.

## Filtering runs by config

In project view, press `f` and type an expression to keep only the runs whose
config matches — the same idea as filtering a W&B workspace in the browser.
Space-separated terms are AND-ed:

```
lr>=0.001 model~llama state=finished
```

| Operator | Meaning |
| --- | --- |
| `=` / `!=` | Equal / not equal (numeric when both sides are numbers, else case-insensitive string) |
| `>` `<` `>=` `<=` | Numeric comparison |
| `~` / `!~` | Contains / does not contain (case-insensitive substring) |

Bare keys read the run's config. Use `config.<key>` to be explicit, and
`run.<attr>` for run attributes (`run.state`, `run.name`) when a config key
would otherwise shadow them. Quote values containing spaces: `name~"my run"`.

The same expression works non-interactively:

```bash
uv run wandb-tui ENTITY/PROJECT --runs 20 --filter 'lr>=0.001 model~llama' --once
```

## W&B LEET comparison

W&B's official LEET TUI is excellent for local `wandb/` directories and `.wandb` files, and newer versions support remote single-run URLs. This tool focuses on remote multi-run project comparisons over W&B's GraphQL API.

## Notes

- Public W&B projects/runs can be queried without authentication.
- For private projects, set `WANDB_API_KEY` in your environment.
- `plotext` is included as a dependency for chart mode.
