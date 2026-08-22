"""Centralized debug logger with global on/off toggle.
Output goes to stdout for visibility in Gradio UI."""
import sys

_debug_enabled = False


def set_debug_mode(enabled: bool) -> None:
    """Toggle debug output globally."""
    global _debug_enabled
    _debug_enabled = bool(enabled)


def debug(msg: str) -> None:
    """Print debug message only when debug mode is enabled.
    Uses [DEBUG] prefix so it's visible in the Gradio agent trace."""
    if _debug_enabled:
        print(f"[DEBUG] {msg}")


def info(msg: str) -> None:
    """Print info message always (user-facing output)."""
    print(f"[INFO] {msg}")


def error(msg: str) -> None:
    """Print error message always."""
    print(f"[ERROR] {msg}")
