"""Detect the focused application and map it to a formatting profile.

Used by the multi-pass pipeline (Pass 3 — FORMAT) to adapt output
style to the target app (code editor, chat, email, terminal, prose).
"""

import json
import subprocess
import shutil
import os
from typing import Optional
from voxd.utils.libw import verbo

# ---------------------------------------------------------------------------
# Formatting profiles — each maps to a prompt suffix for Pass 3
# ---------------------------------------------------------------------------

APP_PROFILES: dict[str, str] = {
    "code": (
        "Format for a code editor. Use proper variable naming "
        "(camelCase or snake_case as contextually appropriate). "
        "If the text describes code, output valid code. "
        "If it describes a comment, prefix with the appropriate comment marker."
    ),
    "chat": (
        "Format for casual chat messaging. Keep it concise and conversational. "
        "Contractions are fine. No formal salutations or sign-offs."
    ),
    "email": (
        "Format as professional email prose. Use proper sentences, "
        "paragraph breaks between topics. Formal but not stiff."
    ),
    "terminal": (
        "Format as a shell command or terse technical note. Be precise and minimal."
    ),
    "prose": (
        "Format as clean written prose. Proper grammar, punctuation, "
        "and paragraph structure."
    ),
}

# ---------------------------------------------------------------------------
# Window class → profile mapping
# ---------------------------------------------------------------------------

_CLASS_MAP: dict[str, str] = {
    # Code editors
    "code": "code",
    "code-oss": "code",
    "vscodium": "code",
    "cursor": "code",
    "windsurf": "code",
    "jetbrains-idea": "code",
    "jetbrains-idea-ce": "code",
    "jetbrains-pycharm": "code",
    "jetbrains-pycharm-ce": "code",
    "jetbrains-webstorm": "code",
    "jetbrains-clion": "code",
    "jetbrains-goland": "code",
    "jetbrains-rider": "code",
    "sublime_text": "code",
    "neovide": "code",
    "emacs": "code",
    "zed": "code",
    # Terminals
    "kitty": "terminal",
    "alacritty": "terminal",
    "gnome-terminal-server": "terminal",
    "gnome-terminal": "terminal",
    "konsole": "terminal",
    "wezterm": "terminal",
    "wezterm-gui": "terminal",
    "foot": "terminal",
    "xterm": "terminal",
    "urxvt": "terminal",
    "tilix": "terminal",
    "terminator": "terminal",
    "st": "terminal",
    "ghostty": "terminal",
    # Chat / messaging
    "slack": "chat",
    "discord": "chat",
    "telegram-desktop": "chat",
    "signal": "chat",
    "element": "chat",
    "whatsapp": "chat",
    "teams": "chat",
    "microsoft teams": "chat",
    # Email
    "thunderbird": "email",
    "geary": "email",
    "evolution": "email",
    "mailspring": "email",
    # Writing / notes (prose)
    "obsidian": "prose",
    "zettlr": "prose",
    "marktext": "prose",
    "typora": "prose",
    "logseq": "prose",
    "notion": "prose",
    "libreoffice": "prose",
    "soffice": "prose",
}

# Browser title keywords that hint at a sub-profile
_BROWSER_TITLE_HINTS: dict[str, str] = {
    "gmail": "email",
    "outlook": "email",
    "protonmail": "email",
    "slack": "chat",
    "discord": "chat",
    "teams": "chat",
    "github": "code",
    "gitlab": "code",
    "codepen": "code",
    "codesandbox": "code",
    "replit": "code",
    "stackoverflow": "code",
}


def detect_focused_app(cfg=None) -> dict[str, str]:
    """Return context about the currently focused application.

    Returns a dict with:
      - app_class: raw WM_CLASS / app_id string (lowercased)
      - window_title: the focused window title (if available)
      - profile: one of the APP_PROFILES keys
      - profile_prompt: the corresponding prompt text
    """
    # Check for user overrides in config
    overrides: dict[str, str] = {}
    if cfg is not None:
        overrides = cfg.data.get("app_profile_overrides", {})

    wm_class = _get_window_class()
    window_title = _get_window_title()
    class_lower = wm_class.lower()

    # 1. Check user overrides first
    profile = overrides.get(class_lower)

    # 2. Check built-in class map
    if profile is None:
        profile = _CLASS_MAP.get(class_lower)

    # 3. For browsers, try to sniff from window title
    if profile is None and class_lower in (
        "firefox", "chromium", "chromium-browser", "google-chrome",
        "brave-browser", "vivaldi-stable", "zen-browser", "librewolf",
        "microsoft-edge", "opera",
    ):
        profile = _sniff_browser_profile(window_title)

    # 4. Check custom profiles from config
    if profile is None and cfg is not None:
        custom_profiles = cfg.data.get("app_custom_profiles", {})
        for pname, pdata in custom_profiles.items():
            if isinstance(pdata, dict):
                classes = pdata.get("classes", [])
                if class_lower in [c.lower() for c in classes]:
                    return {
                        "app_class": class_lower,
                        "window_title": window_title,
                        "profile": pname,
                        "profile_prompt": pdata.get("prompt", APP_PROFILES["prose"]),
                    }

    # 5. Default to prose
    if profile is None:
        profile = "prose"

    return {
        "app_class": class_lower,
        "window_title": window_title,
        "profile": profile,
        "profile_prompt": APP_PROFILES.get(profile, APP_PROFILES["prose"]),
    }


def _sniff_browser_profile(title: str) -> str:
    """Guess a profile from browser window title keywords."""
    title_lower = title.lower()
    for keyword, profile in _BROWSER_TITLE_HINTS.items():
        if keyword in title_lower:
            return profile
    return "prose"


# ---------------------------------------------------------------------------
# Window class / title detection (Wayland-first, X11 fallback)
# ---------------------------------------------------------------------------

def _get_window_class() -> str:
    """Get the WM_CLASS / app_id of the focused window."""
    # Hyprland
    result = _try_hyprctl_class()
    if result:
        return result

    # Sway / i3 (swaymsg)
    result = _try_swaymsg_class()
    if result:
        return result

    # KDE Wayland (kdotool)
    result = _try_kdotool_class()
    if result:
        return result

    # GNOME Wayland (gdbus)
    result = _try_gnome_class()
    if result:
        return result

    # X11 fallback (xdotool)
    result = _try_xdotool_class()
    if result:
        return result

    return "unknown"


def _get_window_title() -> str:
    """Get the title of the focused window."""
    # Hyprland
    result = _try_hyprctl_title()
    if result:
        return result

    # Sway
    result = _try_swaymsg_title()
    if result:
        return result

    # X11
    result = _try_xdotool_title()
    if result:
        return result

    return ""


def _run_cmd(cmd: list[str], timeout: float = 1.0) -> Optional[str]:
    """Run a command and return stdout, or None on failure."""
    if not shutil.which(cmd[0]):
        return None
    try:
        cp = subprocess.run(
            cmd,
            capture_output=True, text=True,
            timeout=timeout,
        )
        if cp.returncode == 0 and cp.stdout.strip():
            return cp.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        pass
    return None


# -- Hyprland ---------------------------------------------------------------

def _try_hyprctl_class() -> Optional[str]:
    raw = _run_cmd(["hyprctl", "activewindow", "-j"])
    if raw:
        try:
            data = json.loads(raw)
            return data.get("class", "") or data.get("initialClass", "")
        except (json.JSONDecodeError, KeyError):
            pass
    return None


def _try_hyprctl_title() -> Optional[str]:
    raw = _run_cmd(["hyprctl", "activewindow", "-j"])
    if raw:
        try:
            data = json.loads(raw)
            return data.get("title", "")
        except (json.JSONDecodeError, KeyError):
            pass
    return None


# -- Sway / i3 --------------------------------------------------------------

def _try_swaymsg_class() -> Optional[str]:
    raw = _run_cmd(["swaymsg", "-t", "get_tree"])
    if raw:
        try:
            tree = json.loads(raw)
            focused = _find_focused_sway(tree)
            if focused:
                return focused.get("app_id", "") or focused.get("window_properties", {}).get("class", "")
        except (json.JSONDecodeError, KeyError):
            pass
    return None


def _try_swaymsg_title() -> Optional[str]:
    raw = _run_cmd(["swaymsg", "-t", "get_tree"])
    if raw:
        try:
            tree = json.loads(raw)
            focused = _find_focused_sway(tree)
            if focused:
                return focused.get("name", "")
        except (json.JSONDecodeError, KeyError):
            pass
    return None


def _find_focused_sway(node: dict) -> Optional[dict]:
    """Recursively find the focused leaf node in a sway tree."""
    if node.get("focused"):
        return node
    for child in node.get("nodes", []) + node.get("floating_nodes", []):
        result = _find_focused_sway(child)
        if result:
            return result
    return None


# -- KDE Wayland (kdotool) --------------------------------------------------

def _try_kdotool_class() -> Optional[str]:
    return _run_cmd(["kdotool", "getactivewindow", "getappid"])


# -- GNOME Wayland (gdbus) --------------------------------------------------

def _try_gnome_class() -> Optional[str]:
    raw = _run_cmd([
        "gdbus", "call", "--session",
        "--dest", "org.gnome.Shell",
        "--object-path", "/org/gnome/Shell",
        "--method", "org.gnome.Shell.Eval",
        "global.display.focus_window ? global.display.focus_window.get_wm_class() : ''"
    ])
    if raw:
        # Output looks like: (true, "'firefox'")
        try:
            if "'" in raw:
                start = raw.index("'") + 1
                end = raw.rindex("'")
                return raw[start:end]
        except (ValueError, IndexError):
            pass
    return None


# -- X11 (xdotool) ----------------------------------------------------------

def _try_xdotool_class() -> Optional[str]:
    return _run_cmd(["xdotool", "getactivewindow", "getwindowclassname"])


def _try_xdotool_title() -> Optional[str]:
    return _run_cmd(["xdotool", "getactivewindow", "getwindowname"])
