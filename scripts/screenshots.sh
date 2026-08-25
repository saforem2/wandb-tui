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
# A concrete finished run for the single-run dashboard shot.
RUN_REF="${WANDB_TUI_SHOT_RUN_REF:-$PROJECT/ekdj28mx}"
# Time-series metrics for the single-run shot, so the sparkline column has
# something in it.
# Anchored and explicit. A bare /^(train|grad)\// also drags in the /max,
# /mean, /min, /std roll-up of every metric, plus series that are all zero on
# this run (grad/norm, grad/max_abs, grad/nonfinite) -- so the table filled
# with flat lines and 0s where the sparkline column is the whole point.
RUN_SEARCH="${WANDB_TUI_SHOT_RUN_SEARCH:-/^(train\\/(loss|dt|dtf|tps)|grad\\/norm_preclip)$/}"
MARKER="${WANDB_TUI_MARKER:-fhd}"
SEARCH='/^(train.loss|grad.norm_preclip)$/'
# Empty by default: with copper-waterfall gone every run is 50 steps, so the
# natural range already fills the frame and a clip would only crop real data.
# Set WANDB_TUI_SHOT_XLIM to demo the feature.
XLIM="${WANDB_TUI_SHOT_XLIM:-}"
COLS="${WANDB_TUI_SHOT_COLS:-178}"
ROWS="${WANDB_TUI_SHOT_ROWS:-42}"
FONT="${WANDB_TUI_SHOT_FONT:-14}"
# Origin of the built-in 2x display. The external ultra-wide starts at x=2056
# and is 1x, so anything placed there captures at half the pixel density.
WINPOS="${WANDB_TUI_SHOT_POS:-60x60}"
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

# kitty knows its own native window id, so ask IT rather than matching a
# title through Quartz -- window titles need screen-recording permission and
# come back as None without it, which made the lookup fail for reasons that
# had nothing to do with the window.
wid() {
  kitty @ ls 2>/dev/null | python3 -c "
import json,sys
want = ${win:-0}
for osw in json.load(sys.stdin):
    for tab in osw.get('tabs', []):
        for w in tab.get('windows', []):
            if w.get('id') == want:
                print(osw.get('platform_window_id') or 'NOTFOUND')
                raise SystemExit
print('NOTFOUND')
"
}

# NOTE on collapsing the input boxes: only the GROUP box tolerates it. Esc
# there merely puts the box away, whereas on the search/filter/limits boxes it
# CLEARS the value -- verified: L then Esc resets xlim to (None, None), which
# would silently undo the axis clip in the very shot meant to show it. So the
# group box is collapsed once (below) and the others stay visible; an open box
# reading "x=0:250" documents the feature anyway.
# Pillow drives the blank-frame check below.
PY_IMG="uv run --with pillow python"

png_width() { python3 -c "import struct,sys;print(struct.unpack('>I',open(sys.argv[1],'rb').read(20)[16:20])[0])" "$1" 2>/dev/null || echo 0; }

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
  # Guard the display-scale trap. This machine has a 2x built-in and a 1x
  # external; the SAME 178x42 cells captured 2314px wide on one and 1246px on
  # the other, so a dark set shot on the external came out at half the
  # resolution of a light set shot on the laptop. Invisible until the README
  # puts a crisp shot next to a blurry one. Fail loudly instead.
  local w
  w=$(png_width "$OUT/$1.png")
  if [ "$w" -lt "${MIN_SHOT_WIDTH:-1800}" ]; then
    echo "!! assets/$1.png is only ${w}px wide -- drag the kitty window to the" >&2
    echo "   built-in Retina display and re-run (or set MIN_SHOT_WIDTH to override)." >&2
    return 1
  fi
  # A window that has not painted yet captures as a solid white rectangle --
  # right size, right window, no content. The width check above cannot see
  # that, and it is how a whole "dark" set came out pure #ffffff. Reject any
  # shot whose pixels are all one colour.
  if ! $PY_IMG - "$OUT/$1.png" "$THEME" <<'PY'
import sys
from PIL import Image
# Count distinct colours across the WHOLE image. A sparse grid can land
# entirely on background even in a good screenshot, and getcolors() returns
# None once an image exceeds maxcolors -- which for a painted TUI is the
# SUCCESS case, so treat None as "plenty of colours".
im = Image.open(sys.argv[1]).convert("RGB")
colors = im.getcolors(maxcolors=256)
if colors is not None and len(colors) < 8:
    sys.exit(1)                      # blank: one flat colour
# The DOMINANT colour is the terminal background. Checking a single pixel is
# not enough -- a captured light theme has plenty of colours and can still be
# the wrong theme, which is exactly how a light set got saved as "-dark".
top = max(im.getcolors(maxcolors=1 << 24), key=lambda c: c[0])[1]
lum = 0.2126 * top[0] + 0.7152 * top[1] + 0.0722 * top[2]
want_dark = sys.argv[2] == "dark"
sys.exit(2 if (lum < 128) != want_dark else 0)
PY
  then
    rc=$?
    if [ "$rc" = 2 ]; then
      echo "!! assets/$1.png is the WRONG THEME for $THEME" >&2
    else
      echo "!! assets/$1.png is blank -- the window had not painted yet" >&2
    fi
    return 1
  fi
  echo "   assets/$1.png (${w}px)"
}

open_term() {  # open_term <light|dark>
  local bg fg
  if [ "$1" = dark ]; then bg=$DARK_BG; fg=$DARK_FG; else bg=$LIGHT_BG; fg=$LIGHT_FG; fi
  # Kill any leftover from an aborted run FIRST. Everything here matches on
  # the title, so a survivor makes every match ambiguous -- that is how a
  # light-themed window from a previous run ended up saved as picker-dark.
  while kitty @ ls 2>/dev/null | grep -q "$TITLE"; do
    kitty @ close-window --match "title:$TITLE" 2>/dev/null || break
    sleep 1
  done
  # --os-window-position puts it on the built-in Retina display up front.
  # Doing it after the fact needs osascript + assistive access, a separate
  # permission that can be missing even when screen capture works.
  win=$(kitty @ launch --type=os-window --os-window-position "$WINPOS" \
        --cwd="$REPO" --keep-focus)
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
  # Confirm the colours really landed on OUR window before anything is shot.
  local got
  got=$(kitty @ get-colors --match "id:$win" 2>/dev/null | awk '$1=="background"{print $2}')
  if [ "$got" != "$bg" ]; then
    echo "!! theme not applied (want $bg, got ${got:-none})" >&2
    return 1
  fi
  # Quiet prompt: the shell line is visible for a beat before the TUI paints.
  send "clear" 1
}

# NOTE: $'\r', never $'\n' -- a bare newline does not reach Textual as an
# Enter keypress, so the input box keeps focus and every following "keystroke"
# is typed into it as text ("m", "L" landing in the search string).
start_ref() { # start_ref <ref> <args>
  send "clear; WANDB_TUI_MARKER=$MARKER uv run wandb-tui '$1' $2"$'\r' 1
  sleep "$BOOT"
}
start() { start_ref "$PROJECT" "$1"; }

# Quit, then PROVE the shell is back before sending anything else.
#
# Matching the prompt by pattern does not work: the TUI's own footer contains
# the same punctuation the prompt does, so a grep for ';' or '$' matched while
# the app was still running -- the next command was then typed into its search
# box ("clearuv run wandb-tui" appearing as a search term). Echoing a sentinel
# is unambiguous: only a real shell can run it and produce the text.
quit() {
  send "q" 2
  local i
  for i in $(seq 1 20); do
    # Ctrl-U first, in case the TUI left a partial line behind.
    kitty @ send-text --match "title:$TITLE" $'\025'
    kitty @ send-text --match "title:$TITLE" "echo SHELLBACK-$i"$'\r'
    sleep 1.5
    if kitty @ get-text --match "title:$TITLE" 2>/dev/null | grep -q "^SHELLBACK-$i"; then
      send "clear"$'\r' 1
      return 0
    fi
  done
  echo "!! app never exited; refusing to keep typing" >&2
  return 1
}

capture() {  # capture <light|dark>
  local t="$1"
  THEME="$t"   # shot() reads this; a local would not reach it
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

  # singlerun: one run's own dashboard. This needs a RUN ref
  # (ENTITY/PROJECT/RUN_ID) -- `--runs 1` just limits the project view, and
  # start() already prepends $PROJECT, so passing it again made argparse
  # reject the duplicate ref and the shot captured a usage error.
  start_ref "$RUN_REF" ""
  # Land on TIME SERIES, not config. Sorted alphabetically the view opens on
  # ~100 config/* rows: every one static, N=1, no sparkline -- which sells the
  # single-run dashboard as a config dump. The `train` tab is the actual
  # per-step data the sparklines are for.
  send "/" 1.5; type_slow "$RUN_SEARCH" 3; send $'\r' 4
  shot "singlerun-$t"
  quit

  # picker: startup entity/project chooser (no ref at all)
  send "uv run wandb-tui"$'\r' 1; sleep 16
  shot "picker-$t"
  send "q" 2

  kitty @ close-window --match "title:$TITLE" 2>/dev/null || true
  win=""
}

for theme in "${@:-dark light}"; do capture "$theme"; done
echo "done -> $OUT"
