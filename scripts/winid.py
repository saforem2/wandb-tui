"""Print the CoreGraphics window id of the kitty window whose title matches.

screencapture -R<rect> grabs a screen REGION, so anything floating above it
(notifications, authenticator popups) lands in the PNG. -l<windowid> captures
the window's own buffer instead: overlays and inactive-window dimming cannot
appear. This resolves the id to pass to -l.
"""
import sys

from Quartz import (
    CGWindowListCopyWindowInfo,
    kCGNullWindowID,
    kCGWindowListOptionOnScreenOnly,
)

want = sys.argv[1] if len(sys.argv) > 1 else ""
for w in CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, kCGNullWindowID):
    if w.get("kCGWindowOwnerName") == "kitty" and want and want in (w.get("kCGWindowName") or ""):
        print(w["kCGWindowNumber"])
        break
else:
    print("NOTFOUND")
    sys.exit(1)
