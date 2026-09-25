"""
SoundFX — Audio Sensory Feedback
Native Apple macOS audio cues for activation, listening, processing, and success.
Zero external audio players needed; uses native Cocoa NSSound.
"""

from __future__ import annotations

import AppKit


class SoundFX:
    """Provides crisp macOS audio cues for the voice assistant."""

    _sounds: dict[str, AppKit.NSSound] = {}

    @classmethod
    def _get_sound(cls, name: str) -> AppKit.NSSound | None:
        if name not in cls._sounds:
            snd = AppKit.NSSound.soundNamed_(name)
            if snd:
                cls._sounds[name] = snd
        return cls._sounds.get(name)

    @classmethod
    def play_activate(cls) -> None:
        """Triggered on PTT key-down: crisp initiation tick."""
        snd = cls._get_sound("Tink")
        if snd:
            snd.stop()
            snd.play()

    @classmethod
    def play_processing(cls) -> None:
        """Triggered on PTT key-up: subtle audio transition into decision."""
        snd = cls._get_sound("Pop")
        if snd:
            snd.stop()
            snd.play()

    @classmethod
    def play_success(cls) -> None:
        """Triggered on successful action execution: satisfying Apple glass chime."""
        snd = cls._get_sound("Glass") or cls._get_sound("Ping")
        if snd:
            snd.stop()
            snd.play()

    @classmethod
    def play_error(cls) -> None:
        """Triggered on unrecognized voice intent or safeguard block."""
        snd = cls._get_sound("Basso") or cls._get_sound("Sosumi")
        if snd:
            snd.stop()
            snd.play()
