"""
MouseController — Physical Cursor Movement & Visual Clicking
High-performance macOS mouse cursor automation using Quartz CoreGraphics and Cocoa AppKit.

Features:
- Smooth human-like cursor gliding along natural cubic ease-out trajectory (visibly moves across screen)
- Native mouse clicking: single click, double click, right click, triple click, drag
- Floating Visual Click Ripple: glowing cyan/amber animated ring at the click coordinates
- Screen landmark navigation: center, top-left, top-right, bottom-left, bottom-right
- Relative cursor movement: move up, down, left, right by N pixels
- UI element target navigation: glides cursor to button/element center before clicking
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any

import AppKit
import Foundation
import Quartz


class ClickRipple:
    """Transient glowing circular ripple ring displayed at cursor click coordinates."""

    _active_panels: list[AppKit.NSPanel] = []

    @classmethod
    def flash(cls, x: float, y: float, color: str = "cyan") -> None:
        """Spawn an animated glowing ring at (x, y) that expands and fades out."""
        def _run():
            screen = AppKit.NSScreen.mainScreen()
            screen_h = screen.frame().size.height if screen else 900
            size = 38.0
            cocoa_x = x - (size / 2.0)
            cocoa_y = screen_h - y - (size / 2.0)

            rect = Foundation.NSMakeRect(cocoa_x, cocoa_y, size, size)
            panel = AppKit.NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
                rect,
                AppKit.NSWindowStyleMaskNonactivatingPanel | AppKit.NSWindowStyleMaskBorderless,
                AppKit.NSBackingStoreBuffered,
                False,
            )
            panel.setLevel_(AppKit.NSStatusWindowLevel + 2)
            panel.setOpaque_(False)
            panel.setBackgroundColor_(AppKit.NSColor.clearColor())
            panel.setIgnoresMouseEvents_(True)
            panel.setHasShadow_(True)

            view = AppKit.NSView.alloc().initWithFrame_(Foundation.NSMakeRect(0, 0, size, size))
            view.setWantsLayer_(True)
            view.layer().setCornerRadius_(size / 2.0)
            view.layer().setBorderWidth_(2.5)

            if color == "amber" or color == "right":
                view.layer().setBorderColor_(
                    AppKit.NSColor.colorWithRed_green_blue_alpha_(1.0, 0.65, 0.15, 0.95).CGColor()
                )
                view.layer().setBackgroundColor_(
                    AppKit.NSColor.colorWithRed_green_blue_alpha_(1.0, 0.65, 0.15, 0.25).CGColor()
                )
            else:
                view.layer().setBorderColor_(
                    AppKit.NSColor.colorWithRed_green_blue_alpha_(0.15, 0.80, 1.0, 0.95).CGColor()
                )
                view.layer().setBackgroundColor_(
                    AppKit.NSColor.colorWithRed_green_blue_alpha_(0.15, 0.80, 1.0, 0.25).CGColor()
                )

            panel.contentView().addSubview_(view)
            panel.orderFrontRegardless()
            panel.setAlphaValue_(1.0)

            # Smooth fade animation
            AppKit.NSAnimationContext.beginGrouping()
            AppKit.NSAnimationContext.currentContext().setDuration_(0.22)
            panel.animator().setAlphaValue_(0.0)
            AppKit.NSAnimationContext.endGrouping()

            # Clean up after animation finishes
            def _cleanup():
                time.sleep(0.25)
                Foundation.NSOperationQueue.mainQueue().addOperationWithBlock_(panel.close)

            threading.Thread(target=_cleanup, daemon=True).start()

        Foundation.NSOperationQueue.mainQueue().addOperationWithBlock_(_run)


class MouseController:
    """High-precision physical mouse cursor automation with smooth visual gliding."""

    def __init__(self) -> None:
        self._screen = AppKit.NSScreen.mainScreen()

    def get_position(self) -> tuple[float, float]:
        """Return current cursor position in Quartz screen coordinates (top-left origin)."""
        loc = Quartz.CGEventGetLocation(Quartz.CGEventCreate(None))
        return float(loc.x), float(loc.y)

    def get_screen_size(self) -> tuple[float, float]:
        """Return primary screen width and height in points."""
        screen = AppKit.NSScreen.mainScreen()
        if screen:
            f = screen.frame().size
            return float(f.width), float(f.height)
        return 1440.0, 900.0

    def smooth_move(
        self,
        target_x: float,
        target_y: float,
        duration: float = 0.24,
        steps: int = 24,
    ) -> None:
        """
        Smoothly move the mouse cursor from current location to (target_x, target_y).
        Uses a cubic ease-out trajectory so the cursor visibly glides across the screen.
        """
        start_x, start_y = self.get_position()
        dx = target_x - start_x
        dy = target_y - start_y

        # If already at destination, skip
        dist = math.hypot(dx, dy)
        if dist < 2.0:
            return

        # Adaptive step count based on distance
        actual_steps = max(12, min(steps, int(dist / 20.0)))
        step_delay = duration / actual_steps

        for i in range(1, actual_steps + 1):
            t = i / actual_steps
            # Cubic ease-out: 1 - (1 - t)^3
            ease = 1.0 - (1.0 - t) ** 3
            curr_x = start_x + (dx * ease)
            curr_y = start_y + (dy * ease)

            pt = Quartz.CGPoint(curr_x, curr_y)
            Quartz.CGWarpMouseCursorPosition(pt)
            event = Quartz.CGEventCreateMouseEvent(
                None, Quartz.kCGEventMouseMoved, pt, Quartz.kCGMouseButtonLeft
            )
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
            time.sleep(step_delay)

        # Ensure final exact coordinate
        final_pt = Quartz.CGPoint(target_x, target_y)
        Quartz.CGWarpMouseCursorPosition(final_pt)

    def move_by(self, dx: float, dy: float, duration: float = 0.20) -> tuple[float, float]:
        """Move cursor relative to current position by (dx, dy)."""
        x, y = self.get_position()
        target_x = max(0.0, x + dx)
        target_y = max(0.0, y + dy)
        self.smooth_move(target_x, target_y, duration=duration)
        return target_x, target_y

    def move_to_landmark(self, landmark: str) -> tuple[float, float]:
        """Glide cursor to a standard screen landmark (center, corners)."""
        sw, sh = self.get_screen_size()
        clean = landmark.lower().strip()

        if "center" in clean:
            tx, ty = sw / 2.0, sh / 2.0
        elif "top_left" in clean or "top left" in clean:
            tx, ty = sw * 0.20, sh * 0.20
        elif "top_right" in clean or "top right" in clean:
            tx, ty = sw * 0.80, sh * 0.20
        elif "bottom_left" in clean or "bottom left" in clean:
            tx, ty = sw * 0.20, sh * 0.80
        elif "bottom_right" in clean or "bottom right" in clean:
            tx, ty = sw * 0.80, sh * 0.80
        elif "top" in clean:
            tx, ty = sw / 2.0, sh * 0.20
        elif "bottom" in clean:
            tx, ty = sw / 2.0, sh * 0.80
        elif "left" in clean:
            tx, ty = sw * 0.20, sh / 2.0
        elif "right" in clean:
            tx, ty = sw * 0.80, sh / 2.0
        else:
            tx, ty = sw / 2.0, sh / 2.0

        self.smooth_move(tx, ty, duration=0.25)
        return tx, ty

    def click(
        self,
        x: float | None = None,
        y: float | None = None,
        button: str = "left",
        click_count: int = 1,
        show_ripple: bool = True,
    ) -> tuple[float, float]:
        """
        Physically click the mouse at (x, y) or at current cursor position.
        Glides cursor to target if provided, flashes glowing visual ripple, and posts CGEvent.
        """
        if x is not None and y is not None:
            self.smooth_move(x, y, duration=0.22)
            curr_x, curr_y = x, y
        else:
            curr_x, curr_y = self.get_position()

        # Trigger visual click ripple
        if show_ripple:
            color = "amber" if button == "right" else "cyan"
            ClickRipple.flash(curr_x, curr_y, color=color)

        pt = Quartz.CGPoint(curr_x, curr_y)

        if button == "right":
            down_type = Quartz.kCGEventRightMouseDown
            up_type = Quartz.kCGEventRightMouseUp
            btn = Quartz.kCGMouseButtonRight
        else:
            down_type = Quartz.kCGEventLeftMouseDown
            up_type = Quartz.kCGEventLeftMouseUp
            btn = Quartz.kCGMouseButtonLeft

        for click_idx in range(1, click_count + 1):
            down = Quartz.CGEventCreateMouseEvent(None, down_type, pt, btn)
            up = Quartz.CGEventCreateMouseEvent(None, up_type, pt, btn)

            Quartz.CGEventSetIntegerValueField(down, Quartz.kCGMouseEventClickState, click_idx)
            Quartz.CGEventSetIntegerValueField(up, Quartz.kCGMouseEventClickState, click_idx)

            Quartz.CGEventPost(Quartz.kCGHIDEventTap, down)
            time.sleep(0.015)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, up)

            if click_count > 1 and click_idx < click_count:
                time.sleep(0.04)

        return curr_x, curr_y

    def double_click(self, x: float | None = None, y: float | None = None) -> tuple[float, float]:
        """Perform a double click with visual ripple."""
        return self.click(x, y, button="left", click_count=2)

    def right_click(self, x: float | None = None, y: float | None = None) -> tuple[float, float]:
        """Perform a secondary / right click with visual ripple."""
        return self.click(x, y, button="right", click_count=1)

    def triple_click(self, x: float | None = None, y: float | None = None) -> tuple[float, float]:
        """Perform a triple click with visual ripple."""
        return self.click(x, y, button="left", click_count=3)

    def drag_to(self, target_x: float, target_y: float, duration: float = 0.30) -> None:
        """Press left mouse button down, smoothly drag to target, and release."""
        start_x, start_y = self.get_position()
        start_pt = Quartz.CGPoint(start_x, start_y)

        down = Quartz.CGEventCreateMouseEvent(
            None, Quartz.kCGEventLeftMouseDown, start_pt, Quartz.kCGMouseButtonLeft
        )
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, down)
        time.sleep(0.02)

        dx = target_x - start_x
        dy = target_y - start_y
        steps = 25
        step_delay = duration / steps

        for i in range(1, steps + 1):
            t = i / steps
            ease = 1.0 - (1.0 - t) ** 3
            curr_x = start_x + (dx * ease)
            curr_y = start_y + (dy * ease)
            pt = Quartz.CGPoint(curr_x, curr_y)
            Quartz.CGWarpMouseCursorPosition(pt)
            drag_ev = Quartz.CGEventCreateMouseEvent(
                None, Quartz.kCGEventLeftMouseDragged, pt, Quartz.kCGMouseButtonLeft
            )
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, drag_ev)
            time.sleep(step_delay)

        final_pt = Quartz.CGPoint(target_x, target_y)
        up = Quartz.CGEventCreateMouseEvent(
            None, Quartz.kCGEventLeftMouseUp, final_pt, Quartz.kCGMouseButtonLeft
        )
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, up)

    def scroll(self, lines_y: int = -5, lines_x: int = 0) -> None:
        """Generate mouse wheel scroll event."""
        scroll_ev = Quartz.CGEventCreateScrollWheelEvent(
            None, Quartz.kCGScrollEventUnitLine, 1, lines_y
        )
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, scroll_ev)
