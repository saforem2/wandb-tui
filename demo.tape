# wandb-tui demo — vhs tape
# Shows: multi-run table -> chart mode with real sparklines on numeric metrics -> search
# NOTE: the search Input auto-focuses on launch, so press Tab to focus the table
# before the `m` (toggle mode) binding will fire.
Output "/Users/samforeman/Obsidian/notes-claude/assets/tui-demos/wandb-tui.gif"

Set Shell "zsh"
Set FontSize 14
Set Width 1500
Set Height 800
Set Padding 18
Set Theme "Catppuccin Mocha"
Set TypingSpeed 55ms

Hide
Type "cd ~/projects/saforem2/wandb-tui"
Enter
Sleep 500ms
Show

Type "uv run wandb-tui 'https://wandb.ai/aurora_gpt/ezpz.examples.fsdp_tp' --runs 6"
Sleep 700ms
Enter

# let it load runs from W&B
Sleep 6s

# --- table view: scroll through the multi-run columns ---
Tab
Sleep 400ms
Down@220ms 6
Sleep 1500ms

# --- switch to CHART mode (Metric | Latest | Chart sparklines) ---
Type "m"
Sleep 1500ms

# filter to numeric grad/* metrics so the sparklines show real trajectories
Type "/"
Sleep 300ms
Type "grad/norm"
Sleep 3s

# scroll through the populated sparklines
Escape
Sleep 500ms
Down@220ms 8
Sleep 3s

# --- back to table view ---
Type "m"
Sleep 1500ms

Type "q"
Sleep 800ms
