"""arm_hand_lab: arm and hand manipulation tasks for Isaac Lab 3.0.

Tasks are registered with gymnasium when :mod:`arm_hand_lab.tasks` is imported. Isaac Lab's
CLI imports it automatically through the ``isaaclab.tasks`` entry point in ``pyproject.toml``.
"""

import sys

__version__ = "0.1.0"


def _utf8_when_redirected() -> None:
    """Write UTF-8 when stdout/stderr go to a file or pipe (launcher, reward agent).

    Isaac Sim's bundled Python runs in isolated mode, so PYTHONIOENCODING is ignored and a
    redirected stdout uses the Windows code page (cp1252). Isaac Lab's start-up banner contains
    characters cp1252 cannot encode, which aborts training. Isaac Lab imports this package before
    printing the banner, so fixing the streams here avoids that. Interactive consoles are left alone.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            if stream is not None and not stream.isatty() and hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError, AttributeError):
            pass


_utf8_when_redirected()
