#!/usr/bin/env bash
# Regenerate the README screenshots, light and dark.
#
# Every non-obvious step is load-bearing:
#
#  * Capture by WINDOW ID (screencapture -l), never by region (-R). A region
#    grab takes whatever is on screen there, so a notification or an
#    authenticator popup ends up in the PNG -- that happened twice. -l reads
#    the window's own buffer: overlays cannot appear, and neither can macOS
#    inactive-window dimming, so the window need not even be focused.
#  * The window is given a unique title and everything matches on it, so no
#    command can ever touch another kitty window.
#  * Colours are set BEFORE the app starts: wandb-tui queries the terminal
#    background (OSC 11) once at startup, and Textual owns the tty after that.
#  * Tab does NOT reach the results pane; it cycles the input boxes. Enter
#    accepts and returns focus (all four boxes). Esc CLEARS search/filter but
#    only puts the group box away -- so a CLI --filter is wiped by an Esc in
#    the filter box. Sequences below are ordered around that.
#  * --filter/--group-by leave their boxes open on purpose (showing what was
#    applied); `G` then Esc collapses the group box while KEEPING the grouping.
#
# Usage: scripts/screenshots.sh [dark|light]
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$REPO/assets"
TITLE="WANDBTUI-SHOTBOX"

PROJECT="${WANDB_TUI_SHOT_PROJECT:-aurora_gpt/ezpz.examples.fsdp_tp}"
# copper-waterfall-1130 is deliberately NOT here. It is the odd one out on
# both axes -- 2226 steps against everyone else's 50, and a grad_preclip max
# of 3.7 against their 43-97 -- so including it squeezed the other six into a
# sliver on the left and flattened their spikes. The six that remain are
# directly comparable, which is the point of the screenshot.
KEEP="balmy-water-1141,charmed-feather-1139,firm-thunder-1140,generous-sunset-1137,helpful-durian-1135,super-night-1136"
FILTER="${WANDB_TUI_SHOT_FILTER:-month=8 day=24 name=$KEEP}"
GROUP="${WANDB_TUI_SHOT_GROUP:-machine,args.model}"
RUNS="${WANDB_TUI_SHOT_RUNS:-60}"
MARKER="${WANDB_TUI_MARKER:-fhd}"
SEARCH='/^(train.loss|grad.norm_preclip)$/'
# Empty by default: with copper-waterfall gone every run is 50 steps, so the
# natural range already fills the frame and a clip would only crop real data.
# Set WANDB_TUI_SHOT_XLIM to demo the feature.
XLIM="${WANDB_TUI_SHOT_XLIM:-}"
COLS="${WANDB_TUI_SHOT_COLS:-178}"
ROWS="${WANDB_TUI_SHOT_ROWS:-42}"
FONT="${WANDB_TUI_SHOT_FONT:-15}"
BOOT="${WANDB_TUI_SHOT_BOOT:-34}"

DARK_BG="#1c1c1c";  DARK_FG="#eeeeee"
LIGHT_BG="#ffffff"; LIGHT_FG="#0b0e11"

win=""

send() { kitty @ send-text --match "title:$TITLE" "$1"; sleep "${2:-1.3}"; }

# Type one character at a time. The inputs debounce on a 0.18s timer, and a
# whole string delivered in one send-text write arrives as a single burst --
# the box ends up displaying the text while the debounced apply never sees the
# final value, so "x=0:250" showed in the box with the axis still unclipped.
type_slow() { local i; for ((i=0;i<${#1};i++)); do
  kitty @ send-text --match "title:$TITLE" "${1:i:1}"; sleep 0.35; done; sleep "${2:-3}"; }

wid() { uv run --with pyobjc-framework-Quartz python "$REPO/scripts/winid.py" "$TITLE"; }

# NOTE on collapsing the input boxes: only the GROUP box tolerates it. Esc
# there merely puts the box away, whereas on the search/filter/limits boxes it
# CLEARS the value -- verified: L then Esc resets xlim to (None, None), which
# would silently undo the axis clip in the very shot meant to show it. So the
# group box is collapsed once (below) and the others stay visible; an open box
# reading "x=0:250" documents the feature anyway.
shot() {
  # MUST be focused: kitty dims inactive windows' TEXT (inactive_text_alpha),
  # and that happens at render time, so -l captures the dimmed glyphs too --
  # measured 224 -> 166 peak luminance unfocused. Note kitty.conf sets 1.0 at
  # line 857 but then includes theme.conf, which sets 0.5; last write wins.
  kitty @ focus-window --match "title:$TITLE" >/dev/null 2>&1 || true
  sleep 1.2
  local id; id=$(wid)
  [ "$id" = NOTFOUND ] && { echo "!! window '$TITLE' not found; refusing" >&2; return 1; }
  screencapture -x -o -l"$id" "$OUT/$1.png"
  echo "   assets/$1.png"
}

open_term() {  # open_term <light|dark>
  local bg fg
  if [ "$1" = dark ]; then bg=$DARK_BG; fg=$DARK_FG; else bg=$LIGHT_BG; fg=$LIGHT_FG; fi
  win=$(kitty @ launch --type=os-window --cwd="$REPO" --keep-focus)
  sleep 2
  kitty @ set-window-title --match "id:$win" "$TITLE"
  sleep 0.5
  kitty @ set-colors --match "id:$win" background="$bg" foreground="$fg" >/dev/null
  # set-font-size acts on the ACTIVE os-window, so focus ours first or it
  # would resize the font in whatever the user is working in.
  kitty @ focus-window --match "id:$win" >/dev/null
  sleep 1
  kitty @ set-font-size "$FONT" >/dev/null 2>&1 || true
  sleep 0.5
  kitty @ resize-os-window --match "title:$TITLE" --action=resize \
      --width "$COLS" --height "$ROWS" --unit cells >/dev/null 2>&1 || true
  sleep 1.5
  # Quiet prompt: the shell line is visible for a beat before the TUI paints.
  send "clear" 1
}

# NOTE: $'\r', never $'\n' -- a bare newline does not reach Textual as an
# Enter keypress, so the input box keeps focus and every following "keystroke"
# is typed into it as text ("m", "L" landing in the search string).
start() { send "clear; WANDB_TUI_MARKER=$MARKER uv run wandb-tui '$PROJECT' $1"$'\r' 1; sleep "$BOOT"; }
quit()  { send "q" 3; }

capture() {  # capture <light|dark>
  local t="$1"
  echo "[$t]"
  open_term "$t"

  # tree: grouped table. `G` + Esc collapses the group box but keeps grouping.
  local lim=""; [ -n "$XLIM" ] && lim="--limits '$XLIM'"
  start "--runs $RUNS --filter '$FILTER' --group-by '$GROUP' $lim"
  send "G" 1.5; send $'\033' 3
  shot "tree-$t"

  # charts: loss + grad/norm_preclip, x clipped so the six 50-step runs are
  # not crushed against the left edge by the single 2000-step one.
  send "/" 1.5; type_slow "$SEARCH" 3; send $'\r' 3
  send "m" 8
  shot "charts-$t"

  # zoom: one metric full-screen
  send $'\r' 5
  shot "zoom-$t"
  send "q" 3

  # filter: same selection back in table form
  send "m" 6
  shot "filter-$t"
  quit

  # singlerun: a single run's own dashboard
  start "'$PROJECT' --runs 1" ; shot "singlerun-$t" ; quit

  # picker: startup entity/project chooser
  send "clear; uv run wandb-tui"$'\r' 1; sleep 14
  shot "picker-$t"
  send "q" 2

  kitty @ close-window --match "title:$TITLE" 2>/dev/null || true
  win=""
}

for theme in "${@:-dark light}"; do capture "$theme"; done
echo "done -> $OUT"
