"""TURT terminal splash text (plain ASCII so it prints on any console)."""

import sys

from . import __fullname__, __title__, __version__

_TURTLE = r"""
                 __
      .,-;-;-,. /'_\
    _/_/_/_|_\_\) /
  '-<_><_><_><_>=/\
    `/_/====/_/-'\_\
     ""     ""    ""
"""

_LOGO = r"""
   _____ _   _ ____ _____
  |_   _| | | |  _ \_   _|
    | | | | | | |_) || |
    | | | |_| |  _ < | |
    |_|  \___/|_| \_\|_|
"""

TAGLINES = [
    "Slow and steady wins the couplet.",
    "Draw a box. Zoom the gates. Fit the vortex.",
    "Every gate counts.",
]


def splash_text():
    rule = "=" * 62
    return "\n".join([
        rule,
        _LOGO.rstrip("\n"),
        _TURTLE.rstrip("\n"),
        "",
        f"   {__title__} v{__version__}  --  {__fullname__}",
        "   Rankine fit | DJWR circulation | Patch | arc cross-sections",
        "   fan export -> CfRadial radar files + full-sweep dummy radar",
        "",
        f"   > {TAGLINES[0]}",
        rule,
    ])


def print_splash(stream=None):
    stream = stream or sys.stdout
    try:
        print(splash_text(), file=stream, flush=True)
    except UnicodeEncodeError:  # extremely old consoles
        print(f"{__title__} v{__version__} -- {__fullname__}", file=stream, flush=True)


def print_environment(stream=None):
    """One-line-per-dependency status (call after the heavy imports)."""
    stream = stream or sys.stdout
    lines = [f"   python      {sys.version.split()[0]}"]
    for name in ("numpy", "matplotlib", "pyart", "scipy", "cmweather"):
        try:
            mod = __import__(name)
            lines.append(f"   {name:<11} {getattr(mod, '__version__', 'ok')}")
        except Exception:
            note = "  (needed for the Rankine fit)" if name == "scipy" else ""
            note = "  (needed to read/export radar files)" if name == "pyart" else note
            lines.append(f"   {name:<11} not installed{note}")
    print("\n".join(lines) + "\n", file=stream, flush=True)
