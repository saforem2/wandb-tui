"""Tests for following the terminal's light/dark theme.

The app previously always used Textual's dark default, so on a light terminal
it rendered dark-on-light regardless -- while the README claimed it followed
the terminal. Detection is an OSC 11 query ("what is your background?"), with
COLORFGBG as a fallback for terminals that set it.
"""

from __future__ import annotations

import wandb_tui as w


# --- parsing the OSC 11 reply ------------------------------------------------


def test_parses_16bit_rgb_reply():
    assert w.parse_osc11("\x1b]11;rgb:1c1c/1c1c/1c1c\x1b\\") == (0x1C, 0x1C, 0x1C)


def test_parses_white_reply():
    assert w.parse_osc11("\x1b]11;rgb:ffff/ffff/ffff\x1b\\") == (255, 255, 255)


def test_parses_bel_terminated_reply():
    """xterm answers with BEL rather than ST."""
    assert w.parse_osc11("\x1b]11;rgb:0000/0000/0000\x07") == (0, 0, 0)


def test_parses_8bit_components():
    assert w.parse_osc11("\x1b]11;rgb:1c/1c/1c\x1b\\") == (0x1C, 0x1C, 0x1C)


def test_garbage_reply_is_none():
    for junk in ("", "nonsense", "\x1b]11;\x1b\\", "\x1b]11;rgb:zz/zz/zz\x1b\\"):
        assert w.parse_osc11(junk) is None, junk


# --- light vs dark decision --------------------------------------------------


def test_dark_background_is_dark():
    assert w.is_dark_rgb((0x1C, 0x1C, 0x1C))
    assert w.is_dark_rgb((0, 0, 0))


def test_light_background_is_light():
    assert not w.is_dark_rgb((255, 255, 255))
    assert not w.is_dark_rgb((0xEE, 0xEE, 0xEE))


def test_uses_luminance_not_a_single_channel():
    """A saturated blue is dark; a saturated yellow is light."""
    assert w.is_dark_rgb((0, 0, 180))
    assert not w.is_dark_rgb((255, 255, 0))


# --- COLORFGBG fallback ------------------------------------------------------


def test_colorfgbg_dark():
    assert w.theme_from_colorfgbg("15;0") == "dark"
    assert w.theme_from_colorfgbg("default;0") == "dark"


def test_colorfgbg_light():
    assert w.theme_from_colorfgbg("0;15") == "light"


def test_colorfgbg_unusable():
    for junk in (None, "", "nonsense", "15"):
        assert w.theme_from_colorfgbg(junk) is None, junk


# --- resolution order --------------------------------------------------------


def test_explicit_env_wins(monkeypatch):
    """TEXTUAL_THEME is an explicit choice; never override it."""
    monkeypatch.setenv("TEXTUAL_THEME", "gruvbox")
    assert w.resolve_theme(query=lambda: (255, 255, 255)) is None


def test_detects_light_from_query(monkeypatch):
    monkeypatch.delenv("TEXTUAL_THEME", raising=False)
    monkeypatch.delenv("COLORFGBG", raising=False)
    assert w.resolve_theme(query=lambda: (255, 255, 255)) == "textual-light"


def test_detects_dark_from_query(monkeypatch):
    monkeypatch.delenv("TEXTUAL_THEME", raising=False)
    monkeypatch.delenv("COLORFGBG", raising=False)
    assert w.resolve_theme(query=lambda: (0x1C, 0x1C, 0x1C)) == "textual-dark"


def test_falls_back_to_colorfgbg(monkeypatch):
    monkeypatch.delenv("TEXTUAL_THEME", raising=False)
    monkeypatch.setenv("COLORFGBG", "0;15")
    assert w.resolve_theme(query=lambda: None) == "textual-light"


def test_no_signal_leaves_the_default(monkeypatch):
    """Undetectable terminal: keep Textual's default rather than guessing."""
    monkeypatch.delenv("TEXTUAL_THEME", raising=False)
    monkeypatch.delenv("COLORFGBG", raising=False)
    assert w.resolve_theme(query=lambda: None) is None


def test_query_failure_is_not_fatal(monkeypatch):
    """A terminal that hangs or errors must not take the app down."""
    monkeypatch.delenv("TEXTUAL_THEME", raising=False)
    monkeypatch.delenv("COLORFGBG", raising=False)

    def boom():
        raise OSError("no tty")

    assert w.resolve_theme(query=boom) is None
