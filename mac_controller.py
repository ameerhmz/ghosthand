"""
MacController — The Hands (Advanced Computer Use)
Comprehensive native macOS automation via async osascript, PyObjC Cocoa, and system subprocesses.

Features:
- Dynamic discovery of 115+ installed macOS apps
- Raycast / Rectangle-grade Window Tiling (12 modes: left, right, top, bottom, 4 quarters, max, center, fullscreen, spaces)
- Keystroke injection, arbitrary typing & high-speed clipboard automation (Cmd+C/V/X/Z, line moves, selection)
- Omnipresent Web Superpowers (Google, YouTube, GitHub, Reddit, Wikipedia, Amazon, Twitter, DuckDuckGo, tab/scroll controls)
- Menu Bar & UI Automation (clicks any menu bar item or dialog button by name in the active app)
- Spotlight & Fast File Search (mdfind file locator with automatic Finder reveal)
- System Perception & Hardware Status (Battery, Active Window, Wi-Fi, Clipboard, Time, Now Playing)
- Native Speech Synthesis (Cocoa NSSpeechSynthesizer status announcements)
- System Preferences & Hardware (Dark mode, Brightness, exact Volume %, Screenshots, Trash, System Folders)
- Media Player auto-detection (Spotify / Apple Music)
"""

from __future__ import annotations

import asyncio
import datetime
import os
import re
import shutil
import time
import urllib.parse
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import AppKit
import Foundation
from mouse_controller import MouseController
from sound_fx import SoundFX

if TYPE_CHECKING:
    from decision_engine import Decision


@dataclass(frozen=True, slots=True)
class ActionResult:
    success: bool
    latency_ms: float
    error: str | None = None
    output: str | None = None


# App name normalization — baseline common mappings
APP_ALIASES: dict[str, str] = {
    "terminal": "Terminal",
    "iterm": "iTerm",
    "iterm2": "iTerm2",
    "chrome": "Google Chrome",
    "google chrome": "Google Chrome",
    "code": "Visual Studio Code",
    "vscode": "Visual Studio Code",
    "vs code": "Visual Studio Code",
    "spotify": "Spotify",
    "slack": "Slack",
    "finder": "Finder",
    "notes": "Notes",
    "messages": "Messages",
    "imessage": "Messages",
    "mail": "Mail",
    "music": "Music",
    "apple music": "Music",
    "safari": "Safari",
    "calendar": "Calendar",
    "system": "System Settings",
    "settings": "System Settings",
    "system settings": "System Settings",
    "system preferences": "System Settings",
    "calculator": "Calculator",
    "preview": "Preview",
    "reminders": "Reminders",
    "activity monitor": "Activity Monitor",
    "claude": "Claude",
    "gemini": "Gemini",
    "chatgpt": "ChatGPT Classic",
    "iina": "IINA",
}


class MacController:
    """Advanced macOS Computer Use controller using non-blocking async subprocesses and Cocoa APIs."""

    def __init__(self, timeout: float = 3.5) -> None:
        self.timeout = timeout
        self.dynamic_apps: dict[str, str] = self._index_installed_apps()
        self._speech_synth: AppKit.NSSpeechSynthesizer | None = None
        self.mouse = MouseController()

    def _index_installed_apps(self) -> dict[str, str]:
        """Automatically index all installed macOS .app bundles dynamically."""
        apps: dict[str, str] = dict(APP_ALIASES)
        search_dirs = [
            "/Applications",
            "/System/Applications",
            "/System/Applications/Utilities",
            os.path.expanduser("~/Applications"),
        ]
        for sdir in search_dirs:
            if os.path.exists(sdir):
                for item in os.listdir(sdir):
                    if item.endswith(".app"):
                        name = item[:-4]
                        clean_key = name.lower().strip()
                        apps[clean_key] = name
        return apps

    async def _run_osascript(self, script: str) -> ActionResult:
        """Execute an AppleScript string asynchronously and measure latency."""
        t0 = time.perf_counter_ns()
        try:
            proc = await asyncio.create_subprocess_exec(
                "osascript",
                "-e",
                script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self.timeout)
            latency = (time.perf_counter_ns() - t0) / 1_000_000

            if proc.returncode != 0:
                err_msg = stderr.decode().strip() or "osascript returned non-zero code"
                return ActionResult(False, latency, error=err_msg)

            out_msg = stdout.decode().strip()
            return ActionResult(True, latency, output=out_msg if out_msg else None)
        except asyncio.TimeoutError:
            latency = (time.perf_counter_ns() - t0) / 1_000_000
            return ActionResult(False, latency, error=f"osascript timed out after {self.timeout}s")
        except Exception as ex:
            latency = (time.perf_counter_ns() - t0) / 1_000_000
            return ActionResult(False, latency, error=str(ex))

    async def _run_command(self, *args: str) -> ActionResult:
        """Execute an arbitrary CLI command asynchronously and measure latency."""
        t0 = time.perf_counter_ns()
        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self.timeout)
            latency = (time.perf_counter_ns() - t0) / 1_000_000

            if proc.returncode != 0:
                err_msg = stderr.decode().strip() or f"{args[0]} returned non-zero code"
                return ActionResult(False, latency, error=err_msg)

            out_msg = stdout.decode().strip()
            return ActionResult(True, latency, output=out_msg if out_msg else None)
        except asyncio.TimeoutError:
            latency = (time.perf_counter_ns() - t0) / 1_000_000
            return ActionResult(False, latency, error=f"{args[0]} timed out after {self.timeout}s")
        except Exception as ex:
            latency = (time.perf_counter_ns() - t0) / 1_000_000
            return ActionResult(False, latency, error=str(ex))

    def _resolve_app(self, app_name: str) -> str:
        """Resolve voice representation or alias to canonical macOS application name."""
        clean = app_name.strip().lower()
        if clean in self.dynamic_apps:
            return self.dynamic_apps[clean]
        for key, canonical in self.dynamic_apps.items():
            if clean in key or key in clean:
                return canonical
        return app_name.strip()

    # =========================================================================
    # 1. Application Management (115+ Apps)
    # =========================================================================

    async def launch_app(self, app_name: str) -> ActionResult:
        """Launch or switch to the frontmost window of the specified application."""
        canonical = self._resolve_app(app_name)
        script = f'tell application "{canonical}" to activate'
        return await self._run_osascript(script)

    async def switch_to_app(self, app_name: str) -> ActionResult:
        """Activate the specified application (alias to launch_app)."""
        return await self.launch_app(app_name)

    async def quit_app(self, app_name: str) -> ActionResult:
        """Gracefully quit the specified application."""
        canonical = self._resolve_app(app_name)
        script = f'tell application "{canonical}" to quit'
        return await self._run_osascript(script)

    async def hide_app(self, app_name: str) -> ActionResult:
        """Hide the specified application."""
        canonical = self._resolve_app(app_name)
        script = f'tell application "System Events" to set visible of process "{canonical}" to false'
        return await self._run_osascript(script)

    async def hide_other_apps(self) -> ActionResult:
        """Hide all other running applications except the frontmost."""
        script = 'tell application "System Events" to keystroke "h" using {command down, option down}'
        return await self._run_osascript(script)

    # =========================================================================
    # 2. Window Tiling & Geometry (Raycast / Rectangle Pro Style - 12 Modes)
    # =========================================================================

    def _window_geom_script(self, x_expr: str, y_expr: str, w_expr: str, h_expr: str) -> str:
        return f"""
        tell application "Finder"
            set b to bounds of window of desktop
            set sw to item 3 of b
            set sh to item 4 of b
        end tell
        tell application "System Events"
            set p to first process whose frontmost is true
            tell p
                if (count of windows) > 0 then
                    set position of window 1 to {{{x_expr}, {y_expr}}}
                    set size of window 1 to {{{w_expr}, {h_expr}}}
                end if
            end tell
        end tell
        """

    async def tile_left(self) -> ActionResult:
        """Snap active window to the left half of the screen."""
        return await self._run_osascript(self._window_geom_script("0", "0", "sw / 2", "sh"))

    async def tile_right(self) -> ActionResult:
        """Snap active window to the right half of the screen."""
        return await self._run_osascript(self._window_geom_script("sw / 2", "0", "sw / 2", "sh"))

    async def tile_top(self) -> ActionResult:
        """Snap active window to the top half of the screen."""
        return await self._run_osascript(self._window_geom_script("0", "0", "sw", "sh / 2"))

    async def tile_bottom(self) -> ActionResult:
        """Snap active window to the bottom half of the screen."""
        return await self._run_osascript(self._window_geom_script("0", "sh / 2", "sw", "sh / 2"))

    async def tile_top_left(self) -> ActionResult:
        """Snap active window to top-left quarter."""
        return await self._run_osascript(self._window_geom_script("0", "0", "sw / 2", "sh / 2"))

    async def tile_top_right(self) -> ActionResult:
        """Snap active window to top-right quarter."""
        return await self._run_osascript(self._window_geom_script("sw / 2", "0", "sw / 2", "sh / 2"))

    async def tile_bottom_left(self) -> ActionResult:
        """Snap active window to bottom-left quarter."""
        return await self._run_osascript(self._window_geom_script("0", "sh / 2", "sw / 2", "sh / 2"))

    async def tile_bottom_right(self) -> ActionResult:
        """Snap active window to bottom-right quarter."""
        return await self._run_osascript(self._window_geom_script("sw / 2", "sh / 2", "sw / 2", "sh / 2"))

    async def maximize_window(self) -> ActionResult:
        """Maximize active window to fill the entire screen."""
        return await self._run_osascript(self._window_geom_script("0", "0", "sw", "sh"))

    async def almost_maximize(self) -> ActionResult:
        """Nearly maximize window (90% width and height centered)."""
        return await self._run_osascript(self._window_geom_script("sw * 0.05", "sh * 0.05", "sw * 0.90", "sh * 0.90"))

    async def center_window(self) -> ActionResult:
        """Center active window on screen (76% width, 80% height)."""
        return await self._run_osascript(self._window_geom_script("sw * 0.12", "sh * 0.10", "sw * 0.76", "sh * 0.80"))

    async def toggle_fullscreen(self) -> ActionResult:
        """Toggle native macOS fullscreen mode (Cmd+Ctrl+F)."""
        script = 'tell application "System Events" to keystroke "f" using {command down, control down}'
        return await self._run_osascript(script)

    async def minimize_frontmost(self) -> ActionResult:
        """Minimize the active frontmost window."""
        script = """
        tell application "System Events"
            tell (first process whose frontmost is true)
                if (count of windows) > 0 then
                    set value of attribute "AXMinimized" of window 1 to true
                else
                    keystroke "m" using command down
                end if
            end tell
        end tell
        """
        return await self._run_osascript(script)

    async def close_frontmost_window(self) -> ActionResult:
        """Close the active window using standard Cmd+W."""
        script = 'tell application "System Events" to keystroke "w" using command down'
        return await self._run_osascript(script)

    async def cycle_windows(self) -> ActionResult:
        """Cycle between open windows of the active application (Cmd+`)."""
        script = 'tell application "System Events" to keystroke "`" using command down'
        return await self._run_osascript(script)

    async def switch_space_left(self) -> ActionResult:
        """Switch to previous macOS Space/Desktop (Ctrl+Left)."""
        script = 'tell application "System Events" to key code 123 using control down'
        return await self._run_osascript(script)

    async def switch_space_right(self) -> ActionResult:
        """Switch to next macOS Space/Desktop (Ctrl+Right)."""
        script = 'tell application "System Events" to key code 124 using control down'
        return await self._run_osascript(script)

    async def show_mission_control(self) -> ActionResult:
        """Show macOS Mission Control."""
        return await self._run_command("open", "-a", "Mission Control")

    async def show_desktop(self) -> ActionResult:
        """Show Desktop (F11 keystroke)."""
        script = 'tell application "System Events" to key code 103'
        return await self._run_osascript(script)

    # =========================================================================
    # 3. Keystrokes, Typing & High-Speed Clipboard Automation
    # =========================================================================

    async def type_text(self, text: str) -> ActionResult:
        """Type spoken text directly into the focused application field."""
        escaped = text.replace('\\', '\\\\').replace('"', '\\"')
        script = f'tell application "System Events" to keystroke "{escaped}"'
        return await self._run_osascript(script)

    async def paste_text(self, text: str) -> ActionResult:
        """
        High-fidelity instant typing via NSPasteboard and Cmd+V.
        Guarantees 100% character accuracy, unicode/emoji support, and zero dropped keys.
        """
        pb = AppKit.NSPasteboard.generalPasteboard()
        pb.clearContents()
        pb.setString_forType_(text, AppKit.NSPasteboardTypeString)
        return await self.paste_clipboard()

    async def press_enter(self) -> ActionResult:
        """Simulate Return / Enter (key code 36)."""
        script = 'tell application "System Events" to key code 36'
        return await self._run_osascript(script)

    async def press_escape(self) -> ActionResult:
        """Simulate Escape (key code 53)."""
        script = 'tell application "System Events" to key code 53'
        return await self._run_osascript(script)

    async def press_tab(self) -> ActionResult:
        """Simulate Tab (key code 48)."""
        script = 'tell application "System Events" to key code 48'
        return await self._run_osascript(script)

    async def press_space(self) -> ActionResult:
        """Simulate Spacebar (key code 49)."""
        script = 'tell application "System Events" to key code 49'
        return await self._run_osascript(script)

    async def backspace(self) -> ActionResult:
        """Simulate Backspace / Delete (key code 51)."""
        script = 'tell application "System Events" to key code 51'
        return await self._run_osascript(script)

    async def delete_line(self) -> ActionResult:
        """Delete current line (Cmd+Delete)."""
        script = 'tell application "System Events" to key code 51 using command down'
        return await self._run_osascript(script)

    async def copy_selection(self) -> ActionResult:
        """Simulate Cmd+C copy."""
        script = 'tell application "System Events" to keystroke "c" using command down'
        return await self._run_osascript(script)

    async def paste_clipboard(self) -> ActionResult:
        """Simulate Cmd+V paste."""
        script = 'tell application "System Events" to keystroke "v" using command down'
        return await self._run_osascript(script)

    async def cut_selection(self) -> ActionResult:
        """Simulate Cmd+X cut."""
        script = 'tell application "System Events" to keystroke "x" using command down'
        return await self._run_osascript(script)

    async def select_all(self) -> ActionResult:
        """Simulate Cmd+A select all."""
        script = 'tell application "System Events" to keystroke "a" using command down'
        return await self._run_osascript(script)

    async def undo(self) -> ActionResult:
        """Simulate Cmd+Z undo."""
        script = 'tell application "System Events" to keystroke "z" using command down'
        return await self._run_osascript(script)

    async def redo(self) -> ActionResult:
        """Simulate Cmd+Shift+Z redo."""
        script = 'tell application "System Events" to keystroke "z" using {command down, shift down}'
        return await self._run_osascript(script)

    async def save_file(self) -> ActionResult:
        """Simulate Cmd+S save."""
        script = 'tell application "System Events" to keystroke "s" using command down'
        return await self._run_osascript(script)

    async def cursor_to_line_start(self) -> ActionResult:
        """Move cursor to beginning of line (Cmd+Left)."""
        script = 'tell application "System Events" to key code 123 using command down'
        return await self._run_osascript(script)

    async def cursor_to_line_end(self) -> ActionResult:
        """Move cursor to end of line (Cmd+Right)."""
        script = 'tell application "System Events" to key code 124 using command down'
        return await self._run_osascript(script)

    async def cursor_to_doc_start(self) -> ActionResult:
        """Move cursor to top of document (Cmd+Up)."""
        script = 'tell application "System Events" to key code 126 using command down'
        return await self._run_osascript(script)

    async def cursor_to_doc_end(self) -> ActionResult:
        """Move cursor to bottom of document (Cmd+Down)."""
        script = 'tell application "System Events" to key code 125 using command down'
        return await self._run_osascript(script)

    # =========================================================================
    # 4. Omnipresent Web & Browser Superpowers (8 Search Engines + Nav)
    # =========================================================================

    async def search_google(self, query: str) -> ActionResult:
        """Search Google in default browser."""
        q = urllib.parse.quote_plus(query.strip())
        return await self._run_command("open", f"https://www.google.com/search?q={q}")

    async def search_youtube(self, query: str) -> ActionResult:
        """Search YouTube in default browser."""
        q = urllib.parse.quote_plus(query.strip())
        return await self._run_command("open", f"https://www.youtube.com/results?search_query={q}")

    async def search_github(self, query: str) -> ActionResult:
        """Search GitHub repositories in default browser."""
        q = urllib.parse.quote_plus(query.strip())
        return await self._run_command("open", f"https://github.com/search?q={q}")

    async def search_reddit(self, query: str) -> ActionResult:
        """Search Reddit in default browser."""
        q = urllib.parse.quote_plus(query.strip())
        return await self._run_command("open", f"https://www.reddit.com/search/?q={q}")

    async def search_wikipedia(self, query: str) -> ActionResult:
        """Search Wikipedia in default browser."""
        q = urllib.parse.quote_plus(query.strip())
        return await self._run_command("open", f"https://en.wikipedia.org/wiki/Special:Search?search={q}")

    async def search_amazon(self, query: str) -> ActionResult:
        """Search Amazon products in default browser."""
        q = urllib.parse.quote_plus(query.strip())
        return await self._run_command("open", f"https://www.amazon.com/s?k={q}")

    async def search_twitter(self, query: str) -> ActionResult:
        """Search X / Twitter in default browser."""
        q = urllib.parse.quote_plus(query.strip())
        return await self._run_command("open", f"https://x.com/search?q={q}")

    async def search_duckduckgo(self, query: str) -> ActionResult:
        """Search DuckDuckGo in default browser."""
        q = urllib.parse.quote_plus(query.strip())
        return await self._run_command("open", f"https://duckduckgo.com/?q={q}")

    async def new_browser_tab(self) -> ActionResult:
        """Open a new browser tab (Cmd+T)."""
        script = 'tell application "System Events" to keystroke "t" using command down'
        return await self._run_osascript(script)

    async def close_browser_tab(self) -> ActionResult:
        """Close active browser tab (Cmd+W)."""
        script = 'tell application "System Events" to keystroke "w" using command down'
        return await self._run_osascript(script)

    async def reopen_browser_tab(self) -> ActionResult:
        """Reopen last closed tab (Cmd+Shift+T)."""
        script = 'tell application "System Events" to keystroke "t" using {command down, shift down}'
        return await self._run_osascript(script)

    async def browser_next_tab(self) -> ActionResult:
        """Switch to next tab (Cmd+Option+Right / Cmd+Shift+])."""
        script = 'tell application "System Events" to keystroke "]" using {command down, shift down}'
        return await self._run_osascript(script)

    async def browser_prev_tab(self) -> ActionResult:
        """Switch to previous tab (Cmd+Option+Left / Cmd+Shift+[)."""
        script = 'tell application "System Events" to keystroke "[" using {command down, shift down}'
        return await self._run_osascript(script)

    async def browser_reload(self) -> ActionResult:
        """Reload active page (Cmd+R)."""
        script = 'tell application "System Events" to keystroke "r" using command down'
        return await self._run_osascript(script)

    async def browser_hard_reload(self) -> ActionResult:
        """Hard reload ignoring cache (Cmd+Shift+R)."""
        script = 'tell application "System Events" to keystroke "r" using {command down, shift down}'
        return await self._run_osascript(script)

    async def browser_scroll_down(self) -> ActionResult:
        """Scroll down page (Page Down key code 121)."""
        script = 'tell application "System Events" to key code 121'
        return await self._run_osascript(script)

    async def browser_scroll_up(self) -> ActionResult:
        """Scroll up page (Page Up key code 116)."""
        script = 'tell application "System Events" to key code 116'
        return await self._run_osascript(script)

    async def browser_zoom_in(self) -> ActionResult:
        """Zoom in page content (Cmd+=)."""
        script = 'tell application "System Events" to keystroke "=" using command down'
        return await self._run_osascript(script)

    async def browser_zoom_out(self) -> ActionResult:
        """Zoom out page content (Cmd+-)."""
        script = 'tell application "System Events" to keystroke "-" using command down'
        return await self._run_osascript(script)

    async def browser_zoom_reset(self) -> ActionResult:
        """Reset page zoom to default 100% (Cmd+0)."""
        script = 'tell application "System Events" to keystroke "0" using command down'
        return await self._run_osascript(script)

    async def browser_open_devtools(self) -> ActionResult:
        """Open Developer Tools inspector (Cmd+Option+I)."""
        script = 'tell application "System Events" to keystroke "i" using {command down, option down}'
        return await self._run_osascript(script)

    async def browser_find_on_page(self, query: str = "") -> ActionResult:
        """Trigger find in page (Cmd+F) and optionally type search text."""
        if query:
            escaped = query.replace('\\', '\\\\').replace('"', '\\"')
            script = f'''
            tell application "System Events"
                keystroke "f" using command down
                delay 0.1
                keystroke "{escaped}"
                key code 36
            end tell
            '''
        else:
            script = 'tell application "System Events" to keystroke "f" using command down'
        return await self._run_osascript(script)

    async def browser_go_back(self) -> ActionResult:
        """Navigate back in browser history (Cmd+[)."""
        script = 'tell application "System Events" to keystroke "[" using command down'
        return await self._run_osascript(script)

    async def browser_go_forward(self) -> ActionResult:
        """Navigate forward in browser history (Cmd+])."""
        script = 'tell application "System Events" to keystroke "]" using command down'
        return await self._run_osascript(script)

    async def open_url(self, raw_url: str) -> ActionResult:
        """Open an arbitrary web address in the default browser."""
        url = raw_url.strip()
        if not url.startswith("http://") and not url.startswith("https://"):
            url = f"https://{url}"
        return await self._run_command("open", url)

    # =========================================================================
    # 5. Menu Bar Automation & UI Button Clicking
    # =========================================================================

    async def click_menu_item(self, menu_path: str) -> ActionResult:
        """
        Click any menu bar item in the active application.
        Accepts formats like: "File > New Window", "Edit > Select All", "View > Zoom In"
        or "File New Window" / "File Save".
        """
        parts = [p.strip() for p in re.split(r"[>/]+|\s{2,}", menu_path) if p.strip()]
        if len(parts) == 1:
            words = parts[0].split()
            if len(words) >= 2:
                menu_name = words[0].capitalize()
                item_name = " ".join(words[1:])
            else:
                menu_name = "File"
                item_name = words[0]
        else:
            menu_name = parts[0].capitalize()
            item_name = parts[1]

        escaped_menu = menu_name.replace('"', '\\"')
        escaped_item = item_name.replace('"', '\\"')

        script = f"""
        tell application "System Events"
            tell (first application process whose frontmost is true)
                tell menu bar 1
                    try
                        tell menu bar item "{escaped_menu}"
                            tell menu "{escaped_menu}"
                                click menu item "{escaped_item}"
                                return "clicked {escaped_menu} > {escaped_item}"
                            end tell
                        end tell
                    on error errMsg
                        -- Partial match search across all menu bar items
                        repeat with mb in menu bar items
                            repeat with m in menus of mb
                                repeat with mi in menu items of m
                                    if name of mi contains "{escaped_item}" then
                                        click mi
                                        return "clicked " & (name of mi)
                                    end if
                                end repeat
                            end repeat
                        end repeat
                        error errMsg
                    end try
                end tell
            end tell
        end tell
        """
        return await self._run_osascript(script)

    async def click_ui_button(self, button_name: str) -> ActionResult:
        """
        Click any button in frontmost window.
        Locates physical screen coordinates to smoothly glide the mouse cursor
        onto the button, flashes the visual click ripple, and performs the physical click.
        Falls back to direct Accessibility click if coordinates cannot be resolved.
        """
        escaped = button_name.strip().replace('"', '\\"')
        coord_script = f"""
        tell application "System Events"
            tell (first application process whose frontmost is true)
                repeat with w in windows
                    try
                        repeat with b in buttons of w
                            if (name of b contains "{escaped}") or (description of b contains "{escaped}") or (title of b contains "{escaped}") then
                                set p to position of b
                                set s to size of b
                                return (item 1 of p as string) & "," & (item 2 of p as string) & "," & (item 1 of s as string) & "," & (item 2 of s as string) & "," & (name of b as string)
                            end if
                        end repeat
                    end try
                end repeat
                return "not found"
            end tell
        end tell
        """
        res = await self._run_osascript(coord_script)
        if res.success and res.output and res.output != "not found":
            parts = [p.strip() for p in res.output.split(",")]
            if len(parts) >= 4:
                try:
                    bx, by = float(parts[0]), float(parts[1])
                    bw, bh = float(parts[2]), float(parts[3])
                    bname = parts[4] if len(parts) > 4 else button_name
                    target_x = bx + (bw / 2.0)
                    target_y = by + (bh / 2.0)
                    # Smoothly glide cursor to button center and click
                    self.mouse.smooth_move(target_x, target_y, duration=0.24)
                    self.mouse.click(target_x, target_y, button="left")
                    SoundFX.play_activate()
                    return ActionResult(True, res.latency_ms, output=f"Cursor glided & clicked '{bname}' at ({target_x:.0f}, {target_y:.0f})")
                except (ValueError, IndexError):
                    pass

        # Fallback to direct Accessibility click
        click_script = f"""
        tell application "System Events"
            tell (first application process whose frontmost is true)
                repeat with w in windows
                    try
                        repeat with b in buttons of w
                            if (name of b contains "{escaped}") or (description of b contains "{escaped}") or (title of b contains "{escaped}") then
                                click b
                                return "clicked button: " & (name of b)
                            end if
                        end repeat
                    end try
                end repeat
                error "Button not found: {escaped}"
            end tell
        end tell
        """
        return await self._run_osascript(click_script)

    # =========================================================================
    # Mouse Cursor Movement & Physical Clicking
    # =========================================================================

    async def mouse_click(self, button: str = "left", clicks: int = 1) -> ActionResult:
        """Physically click at current cursor position with glowing visual ripple."""
        t0 = time.perf_counter_ns()
        cx, cy = self.mouse.click(button=button, click_count=clicks)
        lat = (time.perf_counter_ns() - t0) / 1_000_000
        SoundFX.play_activate()
        return ActionResult(True, lat, output=f"Clicked at ({cx:.0f}, {cy:.0f})")

    async def mouse_double_click(self) -> ActionResult:
        """Physically double-click at current cursor position with visual ripple."""
        t0 = time.perf_counter_ns()
        cx, cy = self.mouse.double_click()
        lat = (time.perf_counter_ns() - t0) / 1_000_000
        SoundFX.play_activate()
        return ActionResult(True, lat, output=f"Double-clicked at ({cx:.0f}, {cy:.0f})")

    async def mouse_right_click(self) -> ActionResult:
        """Physically right-click at current cursor position with amber visual ripple."""
        t0 = time.perf_counter_ns()
        cx, cy = self.mouse.right_click()
        lat = (time.perf_counter_ns() - t0) / 1_000_000
        SoundFX.play_activate()
        return ActionResult(True, lat, output=f"Right-clicked at ({cx:.0f}, {cy:.0f})")

    async def mouse_triple_click(self) -> ActionResult:
        """Physically triple-click at current cursor position."""
        t0 = time.perf_counter_ns()
        cx, cy = self.mouse.triple_click()
        lat = (time.perf_counter_ns() - t0) / 1_000_000
        return ActionResult(True, lat, output=f"Triple-clicked at ({cx:.0f}, {cy:.0f})")

    async def mouse_move_landmark(self, landmark: str) -> ActionResult:
        """Smoothly glide mouse cursor across screen to landmark (center, corners)."""
        t0 = time.perf_counter_ns()
        tx, ty = self.mouse.move_to_landmark(landmark)
        lat = (time.perf_counter_ns() - t0) / 1_000_000
        return ActionResult(True, lat, output=f"Cursor moved to {landmark} ({tx:.0f}, {ty:.0f})")

    async def mouse_move_by(self, dx: float, dy: float) -> ActionResult:
        """Smoothly move mouse cursor relative to current position."""
        t0 = time.perf_counter_ns()
        tx, ty = self.mouse.move_by(dx, dy)
        lat = (time.perf_counter_ns() - t0) / 1_000_000
        return ActionResult(True, lat, output=f"Cursor moved by ({dx:+.0f}, {dy:+.0f})")

    async def mouse_click_center(self) -> ActionResult:
        """Smoothly glide cursor to screen center, flash ripple, and click."""
        t0 = time.perf_counter_ns()
        sw, sh = self.mouse.get_screen_size()
        cx, cy = sw / 2.0, sh / 2.0
        self.mouse.smooth_move(cx, cy, duration=0.25)
        self.mouse.click(cx, cy)
        lat = (time.perf_counter_ns() - t0) / 1_000_000
        SoundFX.play_activate()
        return ActionResult(True, lat, output=f"Clicked center ({cx:.0f}, {cy:.0f})")

    async def mouse_scroll(self, lines: int) -> ActionResult:
        """Scroll mouse wheel."""
        t0 = time.perf_counter_ns()
        self.mouse.scroll(lines_y=lines)
        lat = (time.perf_counter_ns() - t0) / 1_000_000
        direction = "down" if lines < 0 else "up"
        return ActionResult(True, lat, output=f"Scrolled {direction}")

    # =========================================================================
    # 6. Spotlight & Fast File Search (mdfind)
    # =========================================================================

    async def spotlight_search(self, query: str) -> ActionResult:
        """Open macOS Spotlight (Cmd+Space) and enter query."""
        escaped = query.replace('\\', '\\\\').replace('"', '\\"')
        script = f"""
        tell application "System Events"
            keystroke space using command down
            delay 0.15
            keystroke "{escaped}"
            delay 0.1
            key code 36
        end tell
        """
        return await self._run_osascript(script)

    async def find_file(self, filename: str) -> ActionResult:
        """
        Instantly locate a file on macOS using mdfind and pruned find fallback, and reveal in Finder.
        """
        clean = filename.strip()
        t0 = time.perf_counter_ns()
        try:
            # 1. Query mdfind with filename filter
            proc = await asyncio.create_subprocess_exec(
                "mdfind",
                "-name",
                clean,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=2.0)
            lines = [l.strip() for l in stdout.decode().splitlines() if l.strip()]

            if lines:
                best_match = lines[0]
                await asyncio.create_subprocess_exec("open", "-R", best_match)
                lat = (time.perf_counter_ns() - t0) / 1_000_000
                return ActionResult(True, lat, output=f"Found: {os.path.basename(best_match)}")

            # 2. Fast pruned find across home directory (skipping heavy non-source dirs)
            find_proc = await asyncio.create_subprocess_exec(
                "find", os.path.expanduser("~"),
                "-maxdepth", "5",
                "(", "-name", "node_modules", "-o", "-name", ".git", "-o", "-name", ".venv",
                "-o", "-name", "venv", "-o", "-name", "Library", ")", "-prune",
                "-o", "-iname", f"*{clean}*", "-print",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            f_out, _ = await asyncio.wait_for(find_proc.communicate(), timeout=2.5)
            f_lines = [l.strip() for l in f_out.decode().splitlines() if l.strip()]
            if f_lines:
                best_match = f_lines[0]
                await asyncio.create_subprocess_exec("open", "-R", best_match)
                lat = (time.perf_counter_ns() - t0) / 1_000_000
                return ActionResult(True, lat, output=f"Found: {os.path.basename(best_match)}")

            lat = (time.perf_counter_ns() - t0) / 1_000_000
            return ActionResult(False, lat, error=f"No file matching '{clean}' found")
        except Exception as ex:
            lat = (time.perf_counter_ns() - t0) / 1_000_000
            return ActionResult(False, lat, error=str(ex))

    # =========================================================================
    # 7. System Perception & Hardware Status (Inspect & Report)
    # =========================================================================

    async def get_battery_status(self) -> ActionResult:
        """Query battery percentage, charging state, and remaining runtime."""
        res = await self._run_command("pmset", "-g", "batt")
        if res.success and res.output:
            lines = res.output.splitlines()
            power_source = "Battery Power" if "Battery Power" in lines[0] else "AC Power"
            batt_match = re.search(r"(\d+)%;\s*([^;]+);(?:\s*(\d+:\d+))?", res.output)
            if batt_match:
                pct = batt_match.group(1)
                state = batt_match.group(2).strip()
                rem = batt_match.group(3)
                time_str = f" ({rem} remaining)" if rem else ""
                report = f"Battery: {pct}% — {state.capitalize()} on {power_source}{time_str}"
                self.speak(report)
                return ActionResult(True, res.latency_ms, output=report)
        return ActionResult(False, res.latency_ms, error="Could not read battery state")

    async def get_active_window(self) -> ActionResult:
        """Inspect the currently focused application and active window title."""
        script = """
        tell application "System Events"
            set frontApp to first application process whose frontmost is true
            set appName to name of frontApp
            set winTitle to ""
            if (count of windows of frontApp) > 0 then
                set winTitle to name of window 1 of frontApp
            end if
            return appName & " ||| " & winTitle
        end tell
        """
        res = await self._run_osascript(script)
        if res.success and res.output:
            parts = res.output.split("|||")
            app_name = parts[0].strip()
            win_name = parts[1].strip() if len(parts) > 1 and parts[1].strip() else "Main Window"
            report = f"Active App: {app_name} — {win_name}"
            return ActionResult(True, res.latency_ms, output=report)
        return ActionResult(False, res.latency_ms, error="Could not detect active window")

    async def get_wifi_status(self) -> ActionResult:
        """Query current Wi-Fi network SSID."""
        res = await self._run_command("networksetup", "-getairportnetwork", "en0")
        if res.success and res.output:
            if "Current Wi-Fi Network:" in res.output:
                ssid = res.output.split("Current Wi-Fi Network:")[1].strip()
                report = f"Connected to Wi-Fi: {ssid}"
                self.speak(report)
                return ActionResult(True, res.latency_ms, output=report)
            elif "not associated" in res.output.lower():
                report = "Wi-Fi is not connected to any network."
                self.speak(report)
                return ActionResult(True, res.latency_ms, output=report)
        return ActionResult(True, res.latency_ms, output="Wi-Fi interface en0 active")

    async def get_clipboard_preview(self) -> ActionResult:
        """Read and preview current text content on the system clipboard."""
        t0 = time.perf_counter_ns()
        pb = AppKit.NSPasteboard.generalPasteboard()
        text = pb.stringForType_(AppKit.NSPasteboardTypeString)
        lat = (time.perf_counter_ns() - t0) / 1_000_000
        if text:
            preview = text.strip()[:100]
            display = f"Clipboard: \"{preview}\""
            return ActionResult(True, lat, output=display)
        return ActionResult(True, lat, output="Clipboard is currently empty.")

    async def get_current_time(self) -> ActionResult:
        """Announce current time and date."""
        t0 = time.perf_counter_ns()
        now = datetime.datetime.now()
        time_str = now.strftime("%I:%M %p, %A, %B %d")
        lat = (time.perf_counter_ns() - t0) / 1_000_000
        report = f"Current Time: {time_str}"
        self.speak(now.strftime("It is %I:%M %p"))
        return ActionResult(True, lat, output=report)

    async def get_now_playing(self) -> ActionResult:
        """Query currently playing track & artist from Apple Music or Spotify."""
        script = """
        tell application "System Events"
            if exists (process "Music") then
                tell application "Music"
                    try
                        if player state is playing then
                            return (name of current track) & " by " & (artist of current track)
                        else
                            return "Music is paused"
                        end if
                    end try
                end tell
            end if
        end tell
        return "No music playing"
        """
        res = await self._run_osascript(script)
        if res.success and res.output:
            report = f"Now Playing: {res.output}"
            self.speak(res.output)
            return ActionResult(True, res.latency_ms, output=report)
        return ActionResult(True, res.latency_ms, output="No music player active")

    def speak(self, text: str) -> None:
        """Asynchronously announce spoken feedback without blocking."""
        try:
            if self._speech_synth is None:
                self._speech_synth = AppKit.NSSpeechSynthesizer.alloc().init()
            if self._speech_synth:
                self._speech_synth.stopSpeaking()
                self._speech_synth.startSpeakingString_(text)
        except Exception:
            pass

    # =========================================================================
    # 8. System Preferences & Environmental Controls
    # =========================================================================

    async def toggle_dark_mode(self) -> ActionResult:
        """Toggle macOS Dark Mode / Light Mode."""
        script = 'tell application "System Events" to tell appearance preferences to set dark mode to not dark mode'
        return await self._run_osascript(script)

    async def brightness_up(self) -> ActionResult:
        """Increase screen brightness."""
        script = 'tell application "System Events" to key code 144'
        return await self._run_osascript(script)

    async def brightness_down(self) -> ActionResult:
        """Decrease screen brightness."""
        script = 'tell application "System Events" to key code 145'
        return await self._run_osascript(script)

    async def empty_trash(self) -> ActionResult:
        """Empty the macOS Trash."""
        script = 'tell application "Finder" to empty trash'
        return await self._run_osascript(script)

    async def open_trash(self) -> ActionResult:
        """Open the macOS Trash folder in Finder."""
        return await self._run_command("open", os.path.expanduser("~/.Trash"))

    async def open_folder(self, folder: str) -> ActionResult:
        """Open standard user folders in Finder."""
        f = folder.lower().strip()
        path = os.path.expanduser("~/Downloads")
        if "desktop" in f:
            path = os.path.expanduser("~/Desktop")
        elif "doc" in f:
            path = os.path.expanduser("~/Documents")
        elif "app" in f:
            path = "/Applications"
        elif "picture" in f or "photo" in f:
            path = os.path.expanduser("~/Pictures")
        elif "home" in f or "user" in f:
            path = os.path.expanduser("~")
        elif "icloud" in f:
            path = os.path.expanduser("~/Library/Mobile Documents/com~apple~CloudDocs")
        elif "down" in f:
            path = os.path.expanduser("~/Downloads")
        return await self._run_command("open", path)

    # =========================================================================
    # 9. Media & Volume Controls
    # =========================================================================

    async def _detect_running_player(self) -> str:
        """Detect whether Spotify or Apple Music is currently running/available."""
        script = (
            'tell application "System Events" to return '
            '(exists (process "Spotify")) as string & "," & (exists (process "Music")) as string'
        )
        res = await self._run_osascript(script)
        if res.success and res.output:
            parts = [p.strip().lower() for p in res.output.split(",")]
            spotify_running = len(parts) > 0 and parts[0] == "true"
            music_running = len(parts) > 1 and parts[1] == "true"
            if spotify_running:
                return "Spotify"
            if music_running:
                return "Music"
        if os.path.exists("/Applications/Spotify.app"):
            return "Spotify"
        return "Music"

    async def media_play_pause(self) -> ActionResult:
        """Toggle playback on the active or primary music player."""
        player = await self._detect_running_player()
        script = f'tell application "{player}" to playpause'
        return await self._run_osascript(script)

    async def media_next(self) -> ActionResult:
        """Skip to next track on the active or primary music player."""
        player = await self._detect_running_player()
        script = f'tell application "{player}" to next track'
        return await self._run_osascript(script)

    async def media_previous(self) -> ActionResult:
        """Skip to previous track on the active or primary music player."""
        player = await self._detect_running_player()
        script = f'tell application "{player}" to previous track'
        return await self._run_osascript(script)

    async def volume_get(self) -> tuple[int, bool]:
        """Get current system volume level (0-100) and mute status."""
        script = 'get volume settings'
        res = await self._run_osascript(script)
        level = 50
        muted = False
        if res.success and res.output:
            for part in res.output.split(","):
                part = part.strip()
                if part.startswith("output volume:"):
                    try:
                        level = int(part.split(":", 1)[1].strip())
                    except ValueError:
                        pass
                elif part.startswith("output muted:"):
                    muted = part.split(":", 1)[1].strip().lower() == "true"
        return level, muted

    async def volume_set(self, level: int) -> ActionResult:
        """Set output volume level between 0 and 100."""
        clamped = max(0, min(100, level))
        script = f"set volume output volume {clamped}"
        return await self._run_osascript(script)

    async def volume_up(self, step: int = 10) -> ActionResult:
        """Increase system volume by step percentage."""
        current, _ = await self.volume_get()
        return await self.volume_set(current + step)

    async def volume_down(self, step: int = 10) -> ActionResult:
        """Decrease system volume by step percentage."""
        current, _ = await self.volume_get()
        return await self.volume_set(current - step)

    async def volume_mute(self) -> ActionResult:
        """Toggle mute status."""
        _, muted = await self.volume_get()
        new_state = "false" if muted else "true"
        script = f"set volume output muted {new_state}"
        return await self._run_osascript(script)

    async def lock_screen(self) -> ActionResult:
        """Lock the screen immediately via System Events."""
        script = 'tell application "System Events" to keystroke "q" using {command down, control down}'
        return await self._run_osascript(script)

    async def sleep_display(self) -> ActionResult:
        """Put display to sleep immediately via pmset."""
        return await self._run_command("pmset", "displaysleepnow")

    async def take_screenshot(self) -> ActionResult:
        """Take a full screen screenshot and copy to clipboard."""
        return await self._run_command("screencapture", "-c")

    async def take_screenshot_window(self) -> ActionResult:
        """Take a screenshot of the active window and copy to clipboard."""
        return await self._run_command("screencapture", "-cw")

    async def take_screenshot_area(self) -> ActionResult:
        """Take an interactive area screenshot and copy to clipboard."""
        return await self._run_command("screencapture", "-ci")

    def get_running_apps(self) -> list[str]:
        """Return list of names of running GUI applications."""
        try:
            import subprocess
            res = subprocess.run(
                [
                    "osascript",
                    "-e",
                    'tell application "System Events" to get name of every process whose visible is true',
                ],
                capture_output=True,
                text=True,
                timeout=2.0,
            )
            if res.returncode == 0 and res.stdout:
                return [app.strip() for app in res.stdout.split(",") if app.strip()]
        except Exception:
            pass
        return []

    def get_installed_apps(self) -> list[str]:
        """Return all discovered installed application names."""
        return sorted(list(self.dynamic_apps.values()))

    # =========================================================================
    # 10. High-level Dispatcher
    # =========================================================================

    async def execute(self, decision: Decision) -> ActionResult:
        """Dispatch a classified Decision to its corresponding action handler."""
        action = decision.action_type
        target = str(decision.target_entity).lower().strip()
        media = str(decision.media_command).lower().strip()

        # 1. Application Launch
        if action == "launch_app":
            app = target if target and target != "none" else "terminal"
            return await self.launch_app(app)

        # 2. Window Tiling & Management (Raycast / Rectangle 12 modes)
        if action == "window_mgmt":
            if "quit:" in target or "close:" in target:
                app = target.split(":", 1)[1]
                return await self.quit_app(app)
            if "top_left" in target or "top left" in target:
                return await self.tile_top_left()
            if "top_right" in target or "top right" in target:
                return await self.tile_top_right()
            if "bottom_left" in target or "bottom left" in target:
                return await self.tile_bottom_left()
            if "bottom_right" in target or "bottom right" in target:
                return await self.tile_bottom_right()
            if "top" in target:
                return await self.tile_top()
            if "bottom" in target:
                return await self.tile_bottom()
            if "left" in target:
                return await self.tile_left()
            if "right" in target:
                return await self.tile_right()
            if "almost" in target:
                return await self.almost_maximize()
            if "max" in target or "full" in target:
                return await self.maximize_window()
            if "center" in target:
                return await self.center_window()
            if "mission" in target or "all" in target:
                return await self.show_mission_control()
            if "desktop" in target:
                return await self.show_desktop()
            if "close" in target or "quit" in target:
                return await self.close_frontmost_window()
            if "minimize" in target:
                return await self.minimize_frontmost()
            if "fullscreen" in target:
                return await self.toggle_fullscreen()
            if "hide_others" in target:
                return await self.hide_other_apps()
            if "next_space" in target or "space_right" in target:
                return await self.switch_space_right()
            if "prev_space" in target or "space_left" in target:
                return await self.switch_space_left()
            if "cycle" in target or "next_window" in target:
                return await self.cycle_windows()
            return ActionResult(False, 0.0, error=f"Unknown window management target: {target}")

        # 3. Media Playback
        if action == "media_control":
            if media == "play_pause":
                return await self.media_play_pause()
            if media == "next":
                return await self.media_next()
            if media == "previous":
                return await self.media_previous()
            return await self.media_play_pause()

        # 4. System Volume
        if action == "system_volume":
            if target.startswith("volume_set:"):
                try:
                    lvl = int(target.split(":", 1)[1])
                    return await self.volume_set(lvl)
                except ValueError:
                    pass
            if media == "volume_up":
                return await self.volume_up()
            if media == "volume_down":
                return await self.volume_down()
            if media == "mute":
                return await self.volume_mute()
            return ActionResult(False, 0.0, error=f"Unknown volume action: media={media}, target={target}")

        # 5. URL, Web Search & Browser Navigation
        if action == "open_url":
            if target.startswith("search_google:"):
                return await self.search_google(target.split(":", 1)[1])
            if target.startswith("search_youtube:"):
                return await self.search_youtube(target.split(":", 1)[1])
            if target.startswith("search_github:"):
                return await self.search_github(target.split(":", 1)[1])
            if target.startswith("search_reddit:"):
                return await self.search_reddit(target.split(":", 1)[1])
            if target.startswith("search_wikipedia:"):
                return await self.search_wikipedia(target.split(":", 1)[1])
            if target.startswith("search_amazon:"):
                return await self.search_amazon(target.split(":", 1)[1])
            if target.startswith("search_twitter:"):
                return await self.search_twitter(target.split(":", 1)[1])
            if target.startswith("search_duckduckgo:"):
                return await self.search_duckduckgo(target.split(":", 1)[1])
            if target.startswith("find:"):
                return await self.browser_find_on_page(target.split(":", 1)[1])

            # Browser tab controls
            if "next_tab" in target:
                return await self.browser_next_tab()
            if "prev_tab" in target:
                return await self.browser_prev_tab()
            if "hard_reload" in target:
                return await self.browser_hard_reload()
            if "reload" in target:
                return await self.browser_reload()
            if "scroll_down" in target:
                return await self.browser_scroll_down()
            if "scroll_up" in target:
                return await self.browser_scroll_up()
            if "zoom_in" in target:
                return await self.browser_zoom_in()
            if "zoom_out" in target:
                return await self.browser_zoom_out()
            if "zoom_reset" in target:
                return await self.browser_zoom_reset()
            if "devtools" in target:
                return await self.browser_open_devtools()
            if "back" in target:
                return await self.browser_go_back()
            if "forward" in target:
                return await self.browser_go_forward()
            if "new_tab" in target:
                return await self.new_browser_tab()
            if "close_tab" in target:
                return await self.close_browser_tab()
            if "reopen_tab" in target:
                return await self.reopen_browser_tab()

            raw_url = target if target and target != "none" else "google.com"
            return await self.open_url(raw_url)

        # 6. Keystroke & Typing Injection
        if action == "keystroke":
            if target.startswith("type:"):
                return await self.type_text(target.split(":", 1)[1])
            if target.startswith("paste_text:"):
                return await self.paste_text(target.split(":", 1)[1])
            if "copy" in target:
                return await self.copy_selection()
            if "paste" in target:
                return await self.paste_clipboard()
            if "cut" in target:
                return await self.cut_selection()
            if "select" in target:
                return await self.select_all()
            if "undo" in target:
                return await self.undo()
            if "redo" in target:
                return await self.redo()
            if "save" in target:
                return await self.save_file()
            if "delete_line" in target:
                return await self.delete_line()
            if "backspace" in target:
                return await self.backspace()
            if "enter" in target or "return" in target:
                return await self.press_enter()
            if "esc" in target:
                return await self.press_escape()
            if "tab" in target:
                return await self.press_tab()
            if "space" in target:
                return await self.press_space()
            if "line_start" in target:
                return await self.cursor_to_line_start()
            if "line_end" in target:
                return await self.cursor_to_line_end()
            if "doc_start" in target:
                return await self.cursor_to_doc_start()
            if "doc_end" in target:
                return await self.cursor_to_doc_end()
            return await self.press_enter()

        # 7. Menu Bar & UI Button Clicking
        if action == "ui_click":
            if target.startswith("menu:"):
                return await self.click_menu_item(target.split(":", 1)[1])
            if target.startswith("button:"):
                return await self.click_ui_button(target.split(":", 1)[1])
            return await self.click_ui_button(target)

        # 8. Perception & Status Inspection (Tell Me)
        if action == "perception":
            if "battery" in target:
                return await self.get_battery_status()
            if "window" in target or "app" in target:
                return await self.get_active_window()
            if "wifi" in target or "network" in target:
                return await self.get_wifi_status()
            if "clipboard" in target:
                return await self.get_clipboard_preview()
            if "time" in target or "date" in target:
                return await self.get_current_time()
            if "now_playing" in target or "song" in target:
                return await self.get_now_playing()
            return await self.get_active_window()

        # 9. System Actions & Environmental Controls
        if action == "system_action":
            if target.startswith("spotlight:"):
                return await self.spotlight_search(target.split(":", 1)[1])
            if target.startswith("find_file:"):
                return await self.find_file(target.split(":", 1)[1])
            if "dark" in target or "light" in target or "theme" in target or "appearance" in target:
                return await self.toggle_dark_mode()
            if "brightness_up" in target:
                return await self.brightness_up()
            if "brightness_down" in target:
                return await self.brightness_down()
            if "empty_trash" in target or target == "trash":
                return await self.empty_trash()
            if "open_trash" in target:
                return await self.open_trash()
            if any(f in target for f in ["download", "desktop", "document", "application", "home", "picture", "icloud"]):
                return await self.open_folder(target)
            if "sleep" in target:
                return await self.sleep_display()
            if "screenshot_window" in target:
                return await self.take_screenshot_window()
            if "screenshot_area" in target:
                return await self.take_screenshot_area()
            if "shot" in target or "capture" in target:
                return await self.take_screenshot()
            if "lock" in target:
                return await self.lock_screen()
            return ActionResult(False, 0.0, error=f"Unknown system action target: {target}")

        # 10. Mouse Cursor Movement & Physical Clicking
        if action == "mouse":
            if target == "double_click":
                return await self.mouse_double_click()
            if target == "right_click":
                return await self.mouse_right_click()
            if target == "triple_click":
                return await self.mouse_triple_click()
            if target == "click_center":
                return await self.mouse_click_center()
            if target.startswith("move:"):
                lm = target.split(":", 1)[1]
                return await self.mouse_move_landmark(lm)
            if target.startswith("move_rel:"):
                parts = target.split(":", 1)[1].split(",")
                dx, dy = float(parts[0]), float(parts[1])
                return await self.mouse_move_by(dx, dy)
            if target == "scroll_down":
                return await self.mouse_scroll(-6)
            if target == "scroll_up":
                return await self.mouse_scroll(6)
            return await self.mouse_click()

        if action == "unrecognized":
            return ActionResult(False, 0.0, error="Unrecognized voice intent")

        return ActionResult(False, 0.0, error=f"Unknown action type: {action}")
