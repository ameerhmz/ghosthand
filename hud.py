"""
HUD — Floating Native macOS Overlay (The Face)
Frosted-glass floating panel inspired by Siri and Raycast.
Built with native AppKit (Cocoa NSPanel & NSVisualEffectView).
Displays real-time voice states, transcripts, action execution, and latency metrics.
"""

from __future__ import annotations

import sys
import threading
import time
from typing import Any

import AppKit
import Foundation
from sound_fx import SoundFX


class VoiceHUD:
    """Floating native macOS HUD panel with vibrant frosted glass and micro-animations."""

    def __init__(self, width: int = 440, height: int = 80) -> None:
        self.width = width
        self.height = height
        self.app = AppKit.NSApplication.sharedApplication()
        self.app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)

        # Position at bottom center of screen
        screen = AppKit.NSScreen.mainScreen()
        screen_frame = screen.frame() if screen else Foundation.NSMakeRect(0, 0, 1440, 900)
        x = (screen_frame.size.width - self.width) / 2
        y = 120  # Floating above the Dock

        rect = Foundation.NSMakeRect(x, y, self.width, self.height)
        self.panel = AppKit.NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            rect,
            AppKit.NSWindowStyleMaskNonactivatingPanel | AppKit.NSWindowStyleMaskBorderless,
            AppKit.NSBackingStoreBuffered,
            False,
        )
        self.panel.setLevel_(AppKit.NSStatusWindowLevel)
        self.panel.setOpaque_(False)
        self.panel.setBackgroundColor_(AppKit.NSColor.clearColor())
        self.panel.setHasShadow_(True)
        self.panel.setMovableByWindowBackground_(True)
        self.panel.setAlphaValue_(0.0)

        # Visual Effect View (Frosted Glass Background)
        content_view = self.panel.contentView()
        self.effect_view = AppKit.NSVisualEffectView.alloc().initWithFrame_(
            Foundation.NSMakeRect(0, 0, self.width, self.height)
        )
        self.effect_view.setMaterial_(AppKit.NSVisualEffectMaterialHUDWindow)
        self.effect_view.setBlendingMode_(AppKit.NSVisualEffectBlendingModeBehindWindow)
        self.effect_view.setState_(AppKit.NSVisualEffectStateActive)
        self.effect_view.setWantsLayer_(True)
        self.effect_view.layer().setCornerRadius_(24.0)
        self.effect_view.layer().setMasksToBounds_(True)
        self.effect_view.layer().setBorderWidth_(1.0)
        self.effect_view.layer().setBorderColor_(
            AppKit.NSColor.colorWithWhite_alpha_(1.0, 0.15).CGColor()
        )
        content_view.addSubview_(self.effect_view)

        # Left Icon Badge
        self.icon_badge = AppKit.NSView.alloc().initWithFrame_(Foundation.NSMakeRect(18, 18, 44, 44))
        self.icon_badge.setWantsLayer_(True)
        self.icon_badge.layer().setCornerRadius_(22.0)
        self.icon_badge.layer().setBackgroundColor_(
            AppKit.NSColor.colorWithRed_green_blue_alpha_(0.15, 0.35, 0.95, 0.35).CGColor()
        )
        self.effect_view.addSubview_(self.icon_badge)

        self.icon_label = AppKit.NSTextField.alloc().initWithFrame_(
            Foundation.NSMakeRect(2, 6, 40, 32)
        )
        self.icon_label.setStringValue_("🎙️")
        self.icon_label.setFont_(AppKit.NSFont.systemFontOfSize_(20.0))
        self.icon_label.setAlignment_(AppKit.NSTextAlignmentCenter)
        self.icon_label.setBezeled_(False)
        self.icon_label.setDrawsBackground_(False)
        self.icon_label.setEditable_(False)
        self.icon_label.setSelectable_(False)
        self.icon_badge.addSubview_(self.icon_label)

        # Title Label
        self.title_label = AppKit.NSTextField.alloc().initWithFrame_(
            Foundation.NSMakeRect(74, 40, 260, 24)
        )
        self.title_label.setStringValue_("Listening...")
        self.title_label.setFont_(AppKit.NSFont.systemFontOfSize_weight_(15.0, AppKit.NSFontWeightBold))
        self.title_label.setTextColor_(AppKit.NSColor.whiteColor())
        self.title_label.setBezeled_(False)
        self.title_label.setDrawsBackground_(False)
        self.title_label.setEditable_(False)
        self.title_label.setSelectable_(False)
        self.effect_view.addSubview_(self.title_label)

        # Subtitle Label
        self.sub_label = AppKit.NSTextField.alloc().initWithFrame_(
            Foundation.NSMakeRect(74, 18, 260, 20)
        )
        self.sub_label.setStringValue_("Hold Left Control to speak")
        self.sub_label.setFont_(
            AppKit.NSFont.systemFontOfSize_weight_(12.0, AppKit.NSFontWeightMedium)
        )
        self.sub_label.setTextColor_(AppKit.NSColor.colorWithWhite_alpha_(1.0, 0.70))
        self.sub_label.setBezeled_(False)
        self.sub_label.setDrawsBackground_(False)
        self.sub_label.setEditable_(False)
        self.sub_label.setSelectable_(False)
        self.effect_view.addSubview_(self.sub_label)

        # Right Status / Latency Pill Badge
        self.pill_badge = AppKit.NSView.alloc().initWithFrame_(
            Foundation.NSMakeRect(self.width - 92, 26, 76, 28)
        )
        self.pill_badge.setWantsLayer_(True)
        self.pill_badge.layer().setCornerRadius_(14.0)
        self.pill_badge.layer().setBackgroundColor_(
            AppKit.NSColor.colorWithWhite_alpha_(1.0, 0.10).CGColor()
        )
        self.effect_view.addSubview_(self.pill_badge)

        self.pill_label = AppKit.NSTextField.alloc().initWithFrame_(
            Foundation.NSMakeRect(0, 4, 76, 20)
        )
        self.pill_label.setStringValue_("Metal ⚡")
        self.pill_label.setFont_(
            AppKit.NSFont.systemFontOfSize_weight_(11.0, AppKit.NSFontWeightBold)
        )
        self.pill_label.setTextColor_(AppKit.NSColor.colorWithWhite_alpha_(1.0, 0.90))
        self.pill_label.setAlignment_(AppKit.NSTextAlignmentCenter)
        self.pill_label.setBezeled_(False)
        self.pill_label.setDrawsBackground_(False)
        self.pill_label.setEditable_(False)
        self.pill_label.setSelectable_(False)
        self.pill_badge.addSubview_(self.pill_label)

        self._hide_timer: threading.Timer | None = None

    def _run_on_main(self, block) -> None:
        """Execute UI update on main Cocoa thread."""
        Foundation.NSOperationQueue.mainQueue().addOperationWithBlock_(block)

    def show_listening(self) -> None:
        """Triggered on PTT key-down: smooth slide-in, audio chime, and listening state."""
        def update():
            if self._hide_timer:
                self._hide_timer.cancel()
                self._hide_timer = None

            self.icon_label.setStringValue_("🎙️")
            self.icon_badge.layer().setBackgroundColor_(
                AppKit.NSColor.colorWithRed_green_blue_alpha_(0.15, 0.45, 1.0, 0.45).CGColor()
            )
            self.title_label.setStringValue_("Listening...")
            self.sub_label.setStringValue_("Hold Left Control and speak")
            self.pill_label.setStringValue_("REC 🔴")
            self.panel.orderFrontRegardless()
            AppKit.NSAnimationContext.beginGrouping()
            AppKit.NSAnimationContext.currentContext().setDuration_(0.18)
            self.panel.animator().setAlphaValue_(1.0)
            AppKit.NSAnimationContext.endGrouping()

        self._run_on_main(update)
        SoundFX.play_activate()

    def show_processing(self, transcript: str) -> None:
        """Triggered on PTT key-up: show recognized speech and laya-mlx inference state."""
        def update():
            self.icon_label.setStringValue_("🧠")
            self.icon_badge.layer().setBackgroundColor_(
                AppKit.NSColor.colorWithRed_green_blue_alpha_(0.65, 0.20, 0.95, 0.45).CGColor()
            )
            display_text = f'“{transcript}”' if transcript else "Analyzing..."
            self.title_label.setStringValue_(display_text)
            self.sub_label.setStringValue_("Evaluating System 1 decision (laya-mlx)...")
            self.pill_label.setStringValue_("MLX ⚡")

        self._run_on_main(update)
        SoundFX.play_processing()

    def show_success(self, action_type: str, target: str, latency_ms: float, output: str | None = None) -> None:
        """Triggered on action execution: show result, latency badge, output, and auto-fade out."""
        def update():
            glyph = "✅"
            if "launch" in action_type:
                glyph = "💻"
            elif "media" in action_type:
                glyph = "🎵"
            elif "volume" in action_type:
                glyph = "🔊"
            elif "url" in action_type:
                glyph = "🌐"
            elif "perception" in action_type:
                glyph = "👁️"
            elif "keystroke" in action_type:
                glyph = "⌨️"
            elif "window" in action_type:
                glyph = "🪟"
            elif "ui_click" in action_type:
                glyph = "🎯"
            elif "mouse" in action_type:
                glyph = "🖱️"

            self.icon_label.setStringValue_(glyph)
            self.icon_badge.layer().setBackgroundColor_(
                AppKit.NSColor.colorWithRed_green_blue_alpha_(0.15, 0.85, 0.45, 0.45).CGColor()
            )
            target_desc = target.capitalize() if target and target != "none" else action_type
            if output:
                if len(output) <= 35:
                    self.title_label.setStringValue_(output)
                    self.sub_label.setStringValue_(f"{target_desc} • {latency_ms:.0f}ms ⚡")
                else:
                    self.title_label.setStringValue_(f"{target_desc}")
                    self.sub_label.setStringValue_(output[:45] + ("..." if len(output) > 45 else ""))
            else:
                self.title_label.setStringValue_(f"Executed: {target_desc}")
                self.sub_label.setStringValue_(f"Pipeline: {latency_ms:.0f}ms ⚡ (laya-mlx)")
            self.pill_label.setStringValue_(f"{latency_ms:.0f}ms")

            # Schedule smooth auto-hide (longer display if there is output to read)
            delay = 2.4 if output else 1.6
            if self._hide_timer:
                self._hide_timer.cancel()
            self._hide_timer = threading.Timer(delay, self.hide)
            self._hide_timer.start()

        self._run_on_main(update)
        SoundFX.play_success()

    def show_error(self, message: str) -> None:
        """Triggered on unrecognized query or error."""
        def update():
            self.icon_label.setStringValue_("⚠️")
            self.icon_badge.layer().setBackgroundColor_(
                AppKit.NSColor.colorWithRed_green_blue_alpha_(0.95, 0.35, 0.15, 0.45).CGColor()
            )
            self.title_label.setStringValue_("Unrecognized Command")
            self.sub_label.setStringValue_(message)
            self.pill_label.setStringValue_("Skip")

            if self._hide_timer:
                self._hide_timer.cancel()
            self._hide_timer = threading.Timer(1.8, self.hide)
            self._hide_timer.start()

        self._run_on_main(update)
        SoundFX.play_error()

    def hide(self) -> None:
        """Smoothly fade out the HUD."""
        def update():
            AppKit.NSAnimationContext.beginGrouping()
            AppKit.NSAnimationContext.currentContext().setDuration_(0.25)
            self.panel.animator().setAlphaValue_(0.0)
            AppKit.NSAnimationContext.endGrouping()

        self._run_on_main(update)
