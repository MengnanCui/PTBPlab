"""VSCode-ish dark theme: the single source of truth for colours & fonts.

Kept import-light (no customtkinter import at module top) so tests and the
core can read the palette without a display. `apply()` is the only function
that touches customtkinter, and it's called once at app start.
"""
from __future__ import annotations

# ---- palette (approximate VSCode "Dark+") ----
BG          = "#1e1e1e"   # editor background
SIDEBAR     = "#333333"   # activity bar
PANEL       = "#252526"   # side panel / cards
PANEL_ALT   = "#2d2d30"   # slightly raised card
BORDER      = "#3c3c3c"
ACCENT      = "#0e639c"   # button blue
ACCENT_HOVER = "#1177bb"
ACCENT_ACTIVE = "#094771" # selected list row
TEXT        = "#cccccc"
TEXT_MUTED  = "#858585"
TEXT_BRIGHT = "#ffffff"
OK          = "#4ec9b0"   # teal — success
WARN        = "#ce9178"   # orange — warning
ERR         = "#f14c4c"   # red — error

CORNER = 8                # rounded-corner radius for cards/buttons
CORNER_SM = 6

# Font stack: first available wins on the user's machine.
FONT_FAMILY = "Segoe UI"          # ctk falls back gracefully if absent
FONT_MONO = "JetBrains Mono"

FONT_H1 = (FONT_FAMILY, 20, "bold")
FONT_H2 = (FONT_FAMILY, 15, "bold")
FONT_BODY = (FONT_FAMILY, 13)
FONT_SMALL = (FONT_FAMILY, 11)
FONT_CODE = (FONT_MONO, 12)

# Activity-bar entries: (view key, glyph, tooltip/label). Order == bar order.
NAV_ITEMS = [
    ("home",       "\U0001F3E0", "Home"),          # 🏠
    ("structures", "\U0001F9EC", "Structures"),     # 🧬
    ("generate",   "⚙",      "Generate SKF"),  # ⚙
    ("optimize",   "▶",      "Optimize"),      # ▶
    ("monitor",    "\U0001F4C8", "Monitor"),        # 📈
    ("calculator", "\U0001FA7A", "Calculator"),     # 🩺
    ("logview",    "\U0001F4C4", "Logs"),           # 📄
]


def apply() -> None:
    """Set the global customtkinter appearance. Call once at startup."""
    import customtkinter as ctk

    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("dark-blue")
