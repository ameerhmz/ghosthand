"""
DecisionEngine — The Brain
Powered by laya-mlx (System 1 single forward pass in ~7-14ms).
Evaluates typed questions (choice, noul) over candidate actions with Metal GPU acceleration.
Includes KeywordFallback for instant execution during model downloads or offline scenarios.

Advanced Computer Use Routing:
- 12 Window Tiling & Workspace Modes (left, right, top, bottom, 4 quarters, max, center, spaces)
- 8 Web Search Engines (Google, YouTube, GitHub, Reddit, Wikipedia, Amazon, Twitter, DuckDuckGo)
- Browser Tab & Navigation Controls (next/prev tab, reload, scroll, zoom, devtools, back/forward)
- High-fidelity Keystroke & Typing Injection (type, paste text, select, cut, copy, paste, undo, redo, save, line moves)
- Menu Bar & UI Button Automation (click menu item, click dialog button)
- System Perception & Inspection (battery status, active window, wifi, clipboard, time, now playing)
- Environmental Controls (brightness, exact volume %, dark mode, folders, screenshots, lock, sleep)
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

from semantic_router import PhoneticNormalizer, SemanticRouter


@dataclass(frozen=True, slots=True)
class Decision:
    action_type: str
    target_entity: str
    media_command: str
    confidence: float
    requires_confirmation: float
    latency_ms: float
    used_fallback: bool = False


DECISION_SCHEMA: dict[str, dict[str, Any]] = {
    "action_type": {
        "type": "choice",
        "instructions": "What type of macOS action is the user requesting with this voice command?",
        "criteria": {
            "launch_app": "Open, launch, switch to, start, activate an application",
            "window_mgmt": "Snap left, right, top, bottom, maximize, center, tile, fullscreen, minimize, close, spaces, windows",
            "open_url": "Open website, search google, search youtube, search github, search reddit, web search, browser tabs",
            "keystroke": "Type text, paste, copy, cut, select all, undo, redo, save, enter, escape, keystrokes",
            "system_volume": "Volume up, down, mute, unmute, set volume level, louder, quieter",
            "media_control": "Play, pause, resume, skip, next, previous track in music player",
            "system_action": "Lock screen, sleep display, screenshot, dark mode, empty trash, folders, brightness, spotlight",
            "perception": "Query battery status, active window, current time, clipboard, wifi status, now playing",
            "ui_click": "Click a button, click menu item, click cancel, click save, click submit",
            "mouse": "Move mouse cursor, glide cursor, click, double click, right click, click center, drag, mouse scroll",
            "unrecognized": "Cannot determine intent, not a valid command, casual conversation or general question",
        },
    },
    "target_entity": {
        "type": "choice",
        "instructions": "Which application or target is the user referring to in this voice command?",
        "criteria": {
            "terminal": "Terminal, iTerm, command line, shell, console, CLI",
            "chrome": "Google Chrome, Chrome, browser, web, internet",
            "safari": "Safari, Apple browser",
            "spotify": "Spotify, Spotify app, music streaming",
            "slack": "Slack, team chat, messaging",
            "vscode": "VS Code, Visual Studio Code, code editor, IDE",
            "finder": "Finder, files, file manager, desktop, documents",
            "notes": "Notes, Apple Notes, note-taking",
            "messages": "Messages, iMessage, texts, SMS",
            "calendar": "Calendar, schedule, events, appointments",
            "mail": "Mail, email, inbox, Apple Mail",
            "music": "Apple Music, iTunes, Music app",
            "system": "System settings, system preferences, settings",
            "none": "No specific application target identified",
        },
    },
    "media_command": {
        "type": "choice",
        "instructions": "If this is a media or volume control command, which specific action?",
        "criteria": {
            "play_pause": "Play, pause, resume, stop music or media",
            "next": "Next track, skip song, skip forward",
            "previous": "Previous track, go back, last song, rewind",
            "volume_up": "Volume up, louder, increase volume, turn it up",
            "volume_down": "Volume down, quieter, softer, decrease volume, lower",
            "mute": "Mute, silence, unmute, toggle mute",
            "none": "Not a media or volume command",
        },
    },
    "requires_confirmation": {
        "type": "noul",
        "instructions": "Is this a destructive, dangerous, or irreversible action that should require explicit user confirmation before executing? Examples: closing apps, shutting down, logging out, deleting files.",
    },
}


class KeywordFallback:
    """
    Weighted keyword scoring fallback parser.
    Provides sub-millisecond classification across 100+ commands.
    """

    ACTION_KEYWORDS: dict[str, dict[str, int]] = {
        "launch_app": {
            "open": 3, "launch": 4, "start": 3, "switch": 4, "activate": 4, "go": 2, "run": 2,
        },
        "media_control": {
            "play": 4, "pause": 4, "skip": 4, "next": 4, "previous": 4, "resume": 3,
            "song": 2, "track": 2, "music": 2,
        },
        "system_volume": {
            "volume": 5, "louder": 4, "quieter": 4, "mute": 5, "unmute": 5, "sound": 2,
        },
        "open_url": {
            ".com": 6, ".org": 6, ".net": 6, ".io": 6, ".ai": 6,
            "http": 6, "https": 6, "www": 6, "website": 4, "site": 3, "dot": 3,
        },
        "window_mgmt": {
            "minimize": 5, "maximize": 5, "close window": 5, "hide": 4, "fullscreen": 4, "window": 2,
            "tile": 5, "snap": 5,
        },
        "system_action": {
            "lock": 5, "sleep": 4, "screenshot": 5, "shutdown": 5, "restart": 5, "display": 2, "screen": 2,
            "brightness": 5, "trash": 5,
        },
    }

    TARGET_KEYWORDS: dict[str, dict[str, int]] = {
        "terminal": {"terminal": 5, "iterm": 5, "shell": 4, "console": 4, "cli": 3},
        "chrome": {"chrome": 5, "google chrome": 6},
        "safari": {"safari": 5},
        "spotify": {"spotify": 5},
        "slack": {"slack": 5},
        "vscode": {"vscode": 5, "code": 3, "visual studio": 5},
        "finder": {"finder": 5, "files": 3, "desktop": 2},
        "notes": {"notes": 5},
        "messages": {"messages": 5, "imessage": 5, "text": 3},
        "calendar": {"calendar": 5, "schedule": 3},
        "mail": {"mail": 5, "email": 4},
        "music": {"music": 4, "itunes": 5, "apple music": 5},
        "system": {"settings": 4, "preferences": 4, "system": 3},
    }

    MEDIA_KEYWORDS: dict[str, dict[str, int]] = {
        "play_pause": {"play": 4, "pause": 4, "resume": 4, "stop": 3},
        "next": {"next": 5, "skip": 5, "forward": 3},
        "previous": {"previous": 5, "back": 4, "last": 3, "rewind": 4},
        "volume_up": {"up": 4, "louder": 5, "increase": 4, "raise": 4},
        "volume_down": {"down": 4, "quieter": 5, "decrease": 4, "lower": 4, "softer": 4},
        "mute": {"mute": 6, "unmute": 6, "silence": 5},
    }

    UNRECOGNIZED_PATTERNS = [
        r"^(what|who|when|where|why|how)\s+(is|are|was|were|do|does|did|can|could|would|should)\b",
        r"\b(weather|temperature|forecast|meaning|define|capital of)\b",
        r"^(hello|hi|hey|good morning|tell me a joke)\b",
    ]

    @classmethod
    def parse(cls, transcript: str) -> Decision:
        t0 = time.perf_counter_ns()
        text = PhoneticNormalizer.repair(transcript.strip().lower())

        # 0. System Perception & Status Inspection Queries
        if any(w in text for w in ["battery status", "how is my battery", "battery level", "battery percent", "check battery"]):
            lat = (time.perf_counter_ns() - t0) / 1_000_000
            return Decision("perception", "battery", "none", 0.98, 0.0, lat, True)
        if any(w in text for w in ["active window", "front window", "current window", "what window", "what app is this"]):
            lat = (time.perf_counter_ns() - t0) / 1_000_000
            return Decision("perception", "window", "none", 0.98, 0.0, lat, True)
        if any(w in text for w in ["clipboard", "read clipboard", "what's on my clipboard", "clipboard content"]):
            lat = (time.perf_counter_ns() - t0) / 1_000_000
            return Decision("perception", "clipboard", "none", 0.98, 0.0, lat, True)
        if any(w in text for w in ["what time is it", "current time", "what day is it", "what date is today", "what is the date"]):
            lat = (time.perf_counter_ns() - t0) / 1_000_000
            return Decision("perception", "time", "none", 0.98, 0.0, lat, True)
        if any(w in text for w in ["wifi status", "wifi network", "what wifi", "am i on wifi", "network status"]):
            lat = (time.perf_counter_ns() - t0) / 1_000_000
            return Decision("perception", "wifi", "none", 0.98, 0.0, lat, True)
        if any(w in text for w in ["what song is playing", "what is playing", "current song", "now playing"]):
            lat = (time.perf_counter_ns() - t0) / 1_000_000
            return Decision("perception", "now_playing", "none", 0.98, 0.0, lat, True)

        # Check for explicitly unrecognized queries (e.g. conversational / QA)
        for pat in cls.UNRECOGNIZED_PATTERNS:
            if re.search(pat, text):
                lat = (time.perf_counter_ns() - t0) / 1_000_000
                return Decision("unrecognized", "none", "none", 0.9, 0.0, lat, True)

        # 1. UI Automation: Menu Bar & Dialog Buttons
        if text.startswith("click menu ") or "menu bar " in text or (text.startswith("menu ") and len(text.split()) >= 3):
            sub = text.replace("click menu ", "").replace("menu bar ", "").replace("menu ", "").strip()
            lat = (time.perf_counter_ns() - t0) / 1_000_000
            return Decision("ui_click", f"menu:{sub}", "none", 0.96, 0.0, lat, True)

        # Menu clicking heuristic: "click File Save", "click View Zoom In", "click Edit Select All"
        menu_roots = ["file", "edit", "selection", "view", "go", "run", "terminal", "window", "help", "tools"]
        words = text.split()
        if len(words) >= 3 and words[0] == "click" and words[1] in menu_roots:
            menu_target = f"{words[1].capitalize()} {' '.join(words[2:])}"
            lat = (time.perf_counter_ns() - t0) / 1_000_000
            return Decision("ui_click", f"menu:{menu_target}", "none", 0.96, 0.0, lat, True)

        # Mouse Cursor & Physical Clicking
        if text in ["double click", "double click here", "double click mouse"]:
            lat = (time.perf_counter_ns() - t0) / 1_000_000
            return Decision("mouse", "double_click", "none", 0.98, 0.0, lat, True)
        if text in ["right click", "right click here", "secondary click"]:
            lat = (time.perf_counter_ns() - t0) / 1_000_000
            return Decision("mouse", "right_click", "none", 0.98, 0.0, lat, True)
        if text in ["triple click", "triple click here"]:
            lat = (time.perf_counter_ns() - t0) / 1_000_000
            return Decision("mouse", "triple_click", "none", 0.98, 0.0, lat, True)
        if text in ["click", "click here", "click mouse", "left click", "press mouse"]:
            lat = (time.perf_counter_ns() - t0) / 1_000_000
            return Decision("mouse", "click", "none", 0.98, 0.0, lat, True)
        if any(w in text for w in ["click center", "click the center", "move to center and click"]):
            lat = (time.perf_counter_ns() - t0) / 1_000_000
            return Decision("mouse", "click_center", "none", 0.98, 0.0, lat, True)
        if any(w in text for w in ["move cursor to center", "cursor center", "move mouse to center", "mouse center", "cursor to center"]):
            lat = (time.perf_counter_ns() - t0) / 1_000_000
            return Decision("mouse", "move:center", "none", 0.98, 0.0, lat, True)
        if any(w in text for w in ["move cursor top left", "cursor top left"]):
            lat = (time.perf_counter_ns() - t0) / 1_000_000
            return Decision("mouse", "move:top_left", "none", 0.98, 0.0, lat, True)
        if any(w in text for w in ["move cursor top right", "cursor top right"]):
            lat = (time.perf_counter_ns() - t0) / 1_000_000
            return Decision("mouse", "move:top_right", "none", 0.98, 0.0, lat, True)
        if any(w in text for w in ["move cursor bottom left", "cursor bottom left"]):
            lat = (time.perf_counter_ns() - t0) / 1_000_000
            return Decision("mouse", "move:bottom_left", "none", 0.98, 0.0, lat, True)
        if any(w in text for w in ["move cursor bottom right", "cursor bottom right"]):
            lat = (time.perf_counter_ns() - t0) / 1_000_000
            return Decision("mouse", "move:bottom_right", "none", 0.98, 0.0, lat, True)

        # Relative cursor moves: "move cursor up 100", "cursor left 150", "move cursor down"
        rel_match = re.search(r"\b(?:move\s+)?(?:cursor|mouse)\s+(up|down|left|right)(?:\s+(\d+))?\b", text)
        if rel_match:
            direction = rel_match.group(1)
            dist = float(rel_match.group(2)) if rel_match.group(2) else 150.0
            dx, dy = 0.0, 0.0
            if direction == "up":
                dy = -dist
            elif direction == "down":
                dy = dist
            elif direction == "left":
                dx = -dist
            elif direction == "right":
                dx = dist
            lat = (time.perf_counter_ns() - t0) / 1_000_000
            return Decision("mouse", f"move_rel:{dx:.0f},{dy:.0f}", "none", 0.98, 0.0, lat, True)

        if text in ["mouse scroll down", "scroll down with mouse"]:
            lat = (time.perf_counter_ns() - t0) / 1_000_000
            return Decision("mouse", "scroll_down", "none", 0.98, 0.0, lat, True)
        if text in ["mouse scroll up", "scroll up with mouse"]:
            lat = (time.perf_counter_ns() - t0) / 1_000_000
            return Decision("mouse", "scroll_up", "none", 0.98, 0.0, lat, True)

        # Button clicking: "click cancel", "click save", "click don't save", "click ok", "click submit", "click continue"
        if text.startswith("click "):
            btn = text[6:].strip()
            lat = (time.perf_counter_ns() - t0) / 1_000_000
            return Decision("ui_click", f"button:{btn}", "none", 0.95, 0.0, lat, True)

        # 2. Spotlight & Fast File Search
        if text.startswith("spotlight "):
            sq = text[10:].strip()
            lat = (time.perf_counter_ns() - t0) / 1_000_000
            return Decision("system_action", f"spotlight:{sq}", "none", 0.98, 0.0, lat, True)
        if any(text.startswith(pfx) for pfx in ["find file ", "locate file ", "search file ", "find "]):
            for pfx in ["find file ", "locate file ", "search file ", "find "]:
                if text.startswith(pfx):
                    fname = text[len(pfx):].strip()
                    lat = (time.perf_counter_ns() - t0) / 1_000_000
                    return Decision("system_action", f"find_file:{fname}", "none", 0.95, 0.0, lat, True)

        # 3. Exact Volume Commands: "set volume to 40%", "volume 30"
        vol_match = re.search(r"\b(?:set\s+)?volume\s+(?:to\s+)?(\d{1,3})\b", text)
        if vol_match:
            val = min(100, int(vol_match.group(1)))
            lat = (time.perf_counter_ns() - t0) / 1_000_000
            return Decision("system_volume", f"volume_set:{val}", "none", 0.98, 0.0, lat, True)

        # 4. Search queries (Google, YouTube, GitHub, Reddit, Wikipedia, Amazon, Twitter)
        search_engines = [
            ("search google for ", "search_google:"),
            ("google ", "search_google:"),
            ("search youtube for ", "search_youtube:"),
            ("youtube ", "search_youtube:"),
            ("search github for ", "search_github:"),
            ("github ", "search_github:"),
            ("search reddit for ", "search_reddit:"),
            ("reddit ", "search_reddit:"),
            ("search wikipedia for ", "search_wikipedia:"),
            ("wikipedia ", "search_wikipedia:"),
            ("search amazon for ", "search_amazon:"),
            ("amazon ", "search_amazon:"),
            ("search twitter for ", "search_twitter:"),
            ("search duckduckgo for ", "search_duckduckgo:"),
            ("search for ", "search_google:"),
            ("search ", "search_google:"),
        ]
        for prefix, target_pfx in search_engines:
            if text.startswith(prefix):
                q = text[len(prefix):].strip()
                if q:
                    lat = (time.perf_counter_ns() - t0) / 1_000_000
                    return Decision("open_url", f"{target_pfx}{q}", "none", 0.96, 0.0, lat, True)

        # 5. Keystroke Typing & High-Fidelity Paste
        if text.startswith("type "):
            typed = text[5:].strip()
            lat = (time.perf_counter_ns() - t0) / 1_000_000
            return Decision("keystroke", f"type:{typed}", "none", 0.98, 0.0, lat, True)
        if text.startswith("paste text ") or text.startswith("paste "):
            if text.startswith("paste text "):
                ptxt = text[11:].strip()
                lat = (time.perf_counter_ns() - t0) / 1_000_000
                return Decision("keystroke", f"paste_text:{ptxt}", "none", 0.98, 0.0, lat, True)
            elif len(text.split()) > 1 and text != "paste this" and text != "paste clipboard":
                ptxt = text[6:].strip()
                lat = (time.perf_counter_ns() - t0) / 1_000_000
                return Decision("keystroke", f"paste_text:{ptxt}", "none", 0.98, 0.0, lat, True)

        # 6. Direct Shortcuts & System Keystrokes
        keystroke_map = {
            "copy": "copy", "copy this": "copy", "copy selection": "copy",
            "paste": "paste", "paste this": "paste", "paste clipboard": "paste",
            "cut": "cut", "cut this": "cut", "cut selection": "cut",
            "select all": "select", "select all text": "select",
            "undo": "undo", "redo": "redo", "save": "save", "save file": "save",
            "press enter": "enter", "enter": "enter", "return": "enter", "new line": "enter",
            "press escape": "esc", "escape": "esc", "cancel": "esc",
            "press tab": "tab", "tab": "tab",
            "press space": "space", "space": "space", "spacebar": "space",
            "backspace": "backspace", "delete": "backspace", "delete line": "delete_line",
            "beginning of line": "line_start", "start of line": "line_start",
            "end of line": "line_end", "top of document": "doc_start", "top of file": "doc_start",
            "bottom of document": "doc_end", "bottom of file": "doc_end",
        }
        if text in keystroke_map:
            lat = (time.perf_counter_ns() - t0) / 1_000_000
            return Decision("keystroke", keystroke_map[text], "none", 0.96, 0.0, lat, True)

        # 7. Browser Tab & Navigation Controls
        browser_map = {
            "new tab": "new_tab", "open new tab": "new_tab",
            "close tab": "close_tab", "reopen tab": "reopen_tab",
            "next tab": "next_tab", "previous tab": "prev_tab", "prev tab": "prev_tab",
            "reload": "reload", "reload page": "reload", "refresh": "reload",
            "hard reload": "hard_reload", "force refresh": "hard_reload",
            "scroll down": "scroll_down", "page down": "scroll_down",
            "scroll up": "scroll_up", "page up": "scroll_up",
            "zoom in": "zoom_in", "zoom out": "zoom_out", "reset zoom": "zoom_reset",
            "devtools": "devtools", "open devtools": "devtools", "inspect element": "devtools",
            "go back": "back", "browser back": "back",
            "go forward": "forward", "browser forward": "forward",
        }
        if text in browser_map:
            lat = (time.perf_counter_ns() - t0) / 1_000_000
            return Decision("open_url", browser_map[text], "none", 0.96, 0.0, lat, True)

        # Find in page
        if text.startswith("find in page ") or text.startswith("find on page "):
            fq = text.replace("find in page ", "").replace("find on page ", "").strip()
            lat = (time.perf_counter_ns() - t0) / 1_000_000
            return Decision("open_url", f"find:{fq}", "none", 0.96, 0.0, lat, True)

        # 8. Advanced Window Tiling (12 Modes + Spaces)
        if any(w in text for w in ["top left", "quarter top left"]):
            lat = (time.perf_counter_ns() - t0) / 1_000_000
            return Decision("window_mgmt", "top_left", "none", 0.96, 0.0, lat, True)
        if any(w in text for w in ["top right", "quarter top right"]):
            lat = (time.perf_counter_ns() - t0) / 1_000_000
            return Decision("window_mgmt", "top_right", "none", 0.96, 0.0, lat, True)
        if any(w in text for w in ["bottom left", "quarter bottom left"]):
            lat = (time.perf_counter_ns() - t0) / 1_000_000
            return Decision("window_mgmt", "bottom_left", "none", 0.96, 0.0, lat, True)
        if any(w in text for w in ["bottom right", "quarter bottom right"]):
            lat = (time.perf_counter_ns() - t0) / 1_000_000
            return Decision("window_mgmt", "bottom_right", "none", 0.96, 0.0, lat, True)
        if any(w in text for w in ["snap top", "tile top", "window top", "top half"]):
            lat = (time.perf_counter_ns() - t0) / 1_000_000
            return Decision("window_mgmt", "top", "none", 0.96, 0.0, lat, True)
        if any(w in text for w in ["snap bottom", "tile bottom", "window bottom", "bottom half"]):
            lat = (time.perf_counter_ns() - t0) / 1_000_000
            return Decision("window_mgmt", "bottom", "none", 0.96, 0.0, lat, True)
        if any(w in text for w in ["snap left", "tile left", "window left", "left half"]):
            lat = (time.perf_counter_ns() - t0) / 1_000_000
            return Decision("window_mgmt", "left", "none", 0.96, 0.0, lat, True)
        if any(w in text for w in ["snap right", "tile right", "window right", "right half"]):
            lat = (time.perf_counter_ns() - t0) / 1_000_000
            return Decision("window_mgmt", "right", "none", 0.96, 0.0, lat, True)
        if any(w in text for w in ["almost maximize", "nearly maximize"]):
            lat = (time.perf_counter_ns() - t0) / 1_000_000
            return Decision("window_mgmt", "almost", "none", 0.96, 0.0, lat, True)
        if any(w in text for w in ["maximize", "maximize window", "full screen", "fullscreen"]):
            lat = (time.perf_counter_ns() - t0) / 1_000_000
            return Decision("window_mgmt", "maximize", "none", 0.96, 0.0, lat, True)
        if any(w in text for w in ["center window", "center this"]):
            lat = (time.perf_counter_ns() - t0) / 1_000_000
            return Decision("window_mgmt", "center", "none", 0.96, 0.0, lat, True)
        if any(w in text for w in ["toggle fullscreen", "fullscreen mode"]):
            lat = (time.perf_counter_ns() - t0) / 1_000_000
            return Decision("window_mgmt", "fullscreen", "none", 0.96, 0.0, lat, True)
        if any(w in text for w in ["hide others", "hide other windows", "hide other apps"]):
            lat = (time.perf_counter_ns() - t0) / 1_000_000
            return Decision("window_mgmt", "hide_others", "none", 0.96, 0.0, lat, True)
        if any(w in text for w in ["next space", "switch desktop right", "next desktop"]):
            lat = (time.perf_counter_ns() - t0) / 1_000_000
            return Decision("window_mgmt", "next_space", "none", 0.96, 0.0, lat, True)
        if any(w in text for w in ["previous space", "switch desktop left", "previous desktop"]):
            lat = (time.perf_counter_ns() - t0) / 1_000_000
            return Decision("window_mgmt", "prev_space", "none", 0.96, 0.0, lat, True)
        if any(w in text for w in ["cycle window", "next window", "switch window"]):
            lat = (time.perf_counter_ns() - t0) / 1_000_000
            return Decision("window_mgmt", "cycle", "none", 0.96, 0.0, lat, True)
        if any(w in text for w in ["mission control", "all windows"]):
            lat = (time.perf_counter_ns() - t0) / 1_000_000
            return Decision("window_mgmt", "mission", "none", 0.96, 0.0, lat, True)
        if text.startswith("close ") or text.startswith("quit "):
            parts = text.split(maxsplit=1)
            target_raw = parts[1].strip() if len(parts) > 1 else "window"
            lat = (time.perf_counter_ns() - t0) / 1_000_000
            if target_raw in ["window", "this window", "active window", "the window", "this", "app"]:
                return Decision("window_mgmt", "close", "none", 0.96, 0.0, lat, True)
            elif target_raw in ["settings", "preferences", "system"]:
                return Decision("window_mgmt", "quit:system", "none", 0.96, 0.0, lat, True)
            else:
                return Decision("window_mgmt", f"quit:{target_raw}", "none", 0.96, 0.0, lat, True)
        if any(w in text for w in ["close window", "close active window", "close this window", "close the window"]):
            lat = (time.perf_counter_ns() - t0) / 1_000_000
            return Decision("window_mgmt", "close", "none", 0.96, 0.0, lat, True)
        if any(w in text for w in ["minimize window", "minimize this window", "minimize the window", "minimize"]):
            lat = (time.perf_counter_ns() - t0) / 1_000_000
            return Decision("window_mgmt", "minimize", "none", 0.96, 0.0, lat, True)

        # 9. Advanced System Preferences & Folders
        if any(w in text for w in ["dark mode", "light mode", "toggle dark", "toggle theme"]):
            lat = (time.perf_counter_ns() - t0) / 1_000_000
            return Decision("system_action", "dark_mode", "none", 0.96, 0.0, lat, True)
        if any(w in text for w in ["brightness up", "brighter", "increase brightness"]):
            lat = (time.perf_counter_ns() - t0) / 1_000_000
            return Decision("system_action", "brightness_up", "none", 0.96, 0.0, lat, True)
        if any(w in text for w in ["brightness down", "dimmer", "decrease brightness", "lower brightness"]):
            lat = (time.perf_counter_ns() - t0) / 1_000_000
            return Decision("system_action", "brightness_down", "none", 0.96, 0.0, lat, True)
        if "empty trash" in text:
            lat = (time.perf_counter_ns() - t0) / 1_000_000
            return Decision("system_action", "empty_trash", "none", 0.96, 0.0, lat, True)
        if "open trash" in text or text == "trash":
            lat = (time.perf_counter_ns() - t0) / 1_000_000
            return Decision("system_action", "open_trash", "none", 0.96, 0.0, lat, True)
        if any(w in text for w in ["downloads", "open downloads"]):
            lat = (time.perf_counter_ns() - t0) / 1_000_000
            return Decision("system_action", "downloads", "none", 0.96, 0.0, lat, True)
        if any(w in text for w in ["open desktop", "go to desktop", "desktop folder"]):
            lat = (time.perf_counter_ns() - t0) / 1_000_000
            return Decision("system_action", "desktop_folder", "none", 0.96, 0.0, lat, True)
        if any(w in text for w in ["open documents", "documents folder"]):
            lat = (time.perf_counter_ns() - t0) / 1_000_000
            return Decision("system_action", "documents", "none", 0.96, 0.0, lat, True)
        if any(w in text for w in ["open applications", "applications folder"]):
            lat = (time.perf_counter_ns() - t0) / 1_000_000
            return Decision("system_action", "applications", "none", 0.96, 0.0, lat, True)
        if any(w in text for w in ["open home", "home folder"]):
            lat = (time.perf_counter_ns() - t0) / 1_000_000
            return Decision("system_action", "home", "none", 0.96, 0.0, lat, True)
        if any(w in text for w in ["open pictures", "photos folder"]):
            lat = (time.perf_counter_ns() - t0) / 1_000_000
            return Decision("system_action", "pictures", "none", 0.96, 0.0, lat, True)
        if any(w in text for w in ["open icloud", "icloud drive"]):
            lat = (time.perf_counter_ns() - t0) / 1_000_000
            return Decision("system_action", "icloud", "none", 0.96, 0.0, lat, True)

        # Screenshots
        if "screenshot window" in text:
            lat = (time.perf_counter_ns() - t0) / 1_000_000
            return Decision("system_action", "screenshot_window", "none", 0.96, 0.0, lat, True)
        if "screenshot area" in text or "select screenshot" in text:
            lat = (time.perf_counter_ns() - t0) / 1_000_000
            return Decision("system_action", "screenshot_area", "none", 0.96, 0.0, lat, True)
        if any(w in text for w in ["screenshot", "take a screenshot", "screen capture"]):
            lat = (time.perf_counter_ns() - t0) / 1_000_000
            return Decision("system_action", "screenshot", "none", 0.96, 0.0, lat, True)

        # Lock / Sleep
        if "lock" in text:
            lat = (time.perf_counter_ns() - t0) / 1_000_000
            return Decision("system_action", "lock", "none", 0.96, 0.0, lat, True)
        if any(w in text for w in ["sleep display", "sleep screen", "turn off screen"]):
            lat = (time.perf_counter_ns() - t0) / 1_000_000
            return Decision("system_action", "sleep", "none", 0.96, 0.0, lat, True)

        # 10. URL detection
        if any(ext in text for ext in [".com", ".org", ".net", ".io", ".ai", "http", "www."]):
            tokens = text.split()
            url_target = "google.com"
            for tok in tokens:
                if any(x in tok for x in [".com", ".org", ".net", ".io", ".ai", "http"]):
                    url_target = tok.replace("http://", "").replace("https://", "")
                    break
            lat = (time.perf_counter_ns() - t0) / 1_000_000
            return Decision(
                action_type="open_url",
                target_entity=url_target,
                media_command="none",
                confidence=0.95,
                requires_confirmation=0.0,
                latency_ms=lat,
                used_fallback=True,
            )

        # Target scoring
        best_target = "none"
        best_target_score = 0
        for target, kw_dict in cls.TARGET_KEYWORDS.items():
            score = sum(weight for kw, weight in kw_dict.items() if kw in text)
            if score > best_target_score:
                best_target_score = score
                best_target = target

        # Action scoring
        action_scores: dict[str, int] = {}
        for action, kw_dict in cls.ACTION_KEYWORDS.items():
            action_scores[action] = sum(weight for kw, weight in kw_dict.items() if kw in text)

        # If a target app is mentioned with launch words, boost launch_app
        if best_target != "none" and any(w in text for w in ["open", "launch", "switch", "start", "to", "go"]):
            action_scores["launch_app"] = action_scores.get("launch_app", 0) + 6

        # Media command scoring
        best_media = "none"
        best_media_score = 0
        for media_cmd, kw_dict in cls.MEDIA_KEYWORDS.items():
            score = sum(weight for kw, weight in kw_dict.items() if kw in text)
            if score > best_media_score:
                best_media_score = score
                best_media = media_cmd

        if best_media in ("volume_up", "volume_down", "mute"):
            action_scores["system_volume"] = action_scores.get("system_volume", 0) + 6
        elif best_media in ("play_pause", "next", "previous"):
            action_scores["media_control"] = action_scores.get("media_control", 0) + 6

        sorted_actions = sorted(action_scores.items(), key=lambda kv: kv[1], reverse=True)
        best_action = "unrecognized"
        if sorted_actions and sorted_actions[0][1] > 0:
            best_action = sorted_actions[0][0]

        # Confirmation heuristic
        requires_conf = 0.9 if any(w in text for w in ["shutdown", "delete", "destroy", "close app"]) else 0.0

        lat = (time.perf_counter_ns() - t0) / 1_000_000
        return Decision(
            action_type=best_action,
            target_entity=best_target,
            media_command=best_media,
            confidence=0.85 if best_action != "unrecognized" else 0.2,
            requires_confirmation=requires_conf,
            latency_ms=lat,
            used_fallback=True,
        )


class DecisionEngine:
    """
    Main Decision Engine encapsulating laya-mlx System 1 decision model.
    """

    CONJUNCTIONS = [r"\band then\b", r"\bafter that\b", r"\band\b", r"\bthen\b"]

    def __init__(
        self,
        checkpoint: str = "aac6fef/laya-multilingual-mlx",
        confidence_threshold: float = 0.45,
        confirmation_threshold: float = 0.70,
        compile_mlx: bool = False,
    ) -> None:
        self.checkpoint = checkpoint
        self.confidence_threshold = confidence_threshold
        self.confirmation_threshold = confirmation_threshold
        self.compile_mlx = compile_mlx
        self.agent: Any = None
        self._is_initialized = False
        self.router = SemanticRouter()

    async def initialize(self) -> None:
        """Load the laya-mlx agent, downloading checkpoint weights if needed."""
        try:
            import laya_mlx as laya
            print(f"🧠 Loading laya-mlx checkpoint: {self.checkpoint}...")
            t0 = time.perf_counter_ns()
            self.agent = laya.load(self.checkpoint, compile=self.compile_mlx)
            dur_ms = (time.perf_counter_ns() - t0) / 1_000_000
            self._is_initialized = True
            print(f"✅ laya-mlx loaded in {dur_ms:.1f}ms")
        except Exception as ex:
            print(f"⚠️ laya-mlx load error ({ex}); operating in high-performance KeywordFallback mode.")
            self.agent = None
            self._is_initialized = False

    def warmup(self) -> float:
        """Execute a throwaway predict to JIT compile Metal graphs."""
        if not self.agent:
            return 0.0
        try:
            t0 = time.perf_counter_ns()
            _ = self.agent.predict("open terminal", DECISION_SCHEMA)
            return (time.perf_counter_ns() - t0) / 1_000_000
        except Exception:
            return 0.0

    @classmethod
    def split_compound(cls, transcript: str) -> list[str]:
        """Split compound multi-intent commands like 'open safari and snap left'."""
        text = transcript.strip()
        pattern = "|".join(cls.CONJUNCTIONS)
        parts = [p.strip() for p in re.split(pattern, text, flags=re.IGNORECASE) if p.strip()]
        if len(parts) <= 1:
            return [text]

        cmd_verbs = {
            "open", "launch", "switch", "play", "pause", "skip", "next", "previous",
            "mute", "volume", "lock", "close", "minimize", "maximize", "snap", "tile",
            "search", "type", "paste", "copy", "cut", "select", "undo", "save", "click"
        }
        valid_parts = []
        for part in parts:
            tokens = set(part.lower().split())
            if tokens & cmd_verbs or len(tokens) <= 3:
                valid_parts.append(part)
        return valid_parts if len(valid_parts) > 1 else [text]

    async def decide_single(self, utterance: str) -> Decision:
        """Classify a single atomic utterance with multi-layer acoustic and semantic routing."""
        clean_text = PhoneticNormalizer.repair(utterance.strip().lower())

        if not self.agent:
            return KeywordFallback.parse(clean_text)

        # Fast path for deterministic computer use queries
        if any(w in clean_text for w in [
            "battery", "active window", "front window", "clipboard", "wifi", "what time",
            "snap ", "tile ", "search google", "search youtube", "search github", "search reddit",
            "type ", "click", "spotlight ", "find file ", "close window", "minimize window",
            "cursor", "mouse", "double click", "right click", "triple click", "close ", "quit "
        ]):
            return KeywordFallback.parse(clean_text)

        t0 = time.perf_counter_ns()
        try:
            pred = self.agent.predict(clean_text, DECISION_SCHEMA)
            answers = pred.get("answers", {})

            # Action type
            act_ans = answers.get("action_type", {})
            action_type = act_ans.get("choice", "unrecognized")
            action_conf = act_ans.get("confidence", 0.0)

            # Target entity
            target_ans = answers.get("target_entity", {})
            target_entity = target_ans.get("choice", "none")

            # Media command
            media_ans = answers.get("media_command", {})
            media_cmd = media_ans.get("choice", "none")

            # Requires confirmation (noul = P(true))
            conf_ans = answers.get("requires_confirmation", {})
            req_conf = conf_ans.get("noul", 0.0)

            lat_ms = (time.perf_counter_ns() - t0) / 1_000_000

            # Cross-question consistency resolution:
            if media_cmd in ("play_pause", "next", "previous") and action_type == "unrecognized":
                action_type = "media_control"
                action_conf = max(action_conf, 0.85)

            if target_entity != "none" and any(w in clean_text for w in ["switch", "open", "launch", "start", "to", "go"]):
                if action_type == "unrecognized":
                    action_type = "launch_app"
                    action_conf = max(action_conf, 0.85)

            if any(dom in clean_text for dom in [".com", ".org", ".net", ".io", "http", "www."]):
                tokens = clean_text.split()
                for tok in tokens:
                    if any(dom in tok for dom in [".com", ".org", ".net", ".io"]):
                        target_entity = tok.replace("http://", "").replace("https://", "")
                        action_type = "open_url"
                        action_conf = max(action_conf, 0.95)
                        break

            # If model is uncertain or predicts unrecognized, route through SemanticRouter first
            if action_type == "unrecognized" or action_conf < self.confidence_threshold:
                # 1. SemanticRouter prototype vector similarity
                route_match = self.router.route(clean_text)
                if route_match and route_match.score >= 0.60:
                    return Decision(
                        action_type=route_match.name,
                        target_entity=route_match.target,
                        media_command=route_match.media,
                        confidence=route_match.score,
                        requires_confirmation=0.0,
                        latency_ms=lat_ms,
                        used_fallback=True,
                    )

                # 2. KeywordFallback parser
                fb = KeywordFallback.parse(clean_text)
                if fb.action_type != "unrecognized" and fb.confidence >= 0.50:
                    return fb

                # 3. SAFETY GATE: Reject low-confidence hallucinations cleanly
                return Decision(
                    action_type="unrecognized",
                    target_entity="none",
                    media_command="none",
                    confidence=action_conf,
                    requires_confirmation=0.0,
                    latency_ms=lat_ms,
                    used_fallback=False,
                )

            return Decision(
                action_type=action_type,
                target_entity=target_entity,
                media_command=media_cmd,
                confidence=action_conf,
                requires_confirmation=req_conf,
                latency_ms=lat_ms,
                used_fallback=False,
            )
        except Exception:
            return KeywordFallback.parse(clean_text)

    async def decide(self, transcript: str) -> list[Decision]:
        """Classify user transcript, splitting compound commands if present."""
        chunks = self.split_compound(transcript)
        decisions: list[Decision] = []
        for chunk in chunks:
            dec = await self.decide_single(chunk)
            decisions.append(dec)
        return decisions
