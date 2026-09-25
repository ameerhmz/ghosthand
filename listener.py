"""
VoiceListener — The Ears (Push-to-Talk)
Zero background CPU listening. Physical key-gating using pynput and sounddevice.
Streaming offline local speech-to-text via Vosk KaldiRecognizer.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import time
from typing import Any, Awaitable, Callable

import numpy as np
import sounddevice as sd
from pynput import keyboard
from pynput.keyboard import Key, KeyCode
import vosk

# Default Push-To-Talk hotkey: Left Control (⌃)
DEFAULT_PTT_KEY = Key.ctrl_l

KEY_NAME_MAP: dict[str, Key] = {
    "ctrl_l": Key.ctrl_l,
    "ctrl_r": Key.ctrl_r,
    "alt_l": Key.alt_l,
    "alt_r": Key.alt_r,
    "cmd_l": Key.cmd_l,
    "cmd_r": Key.cmd_r,
    "shift_l": Key.shift_l,
    "shift_r": Key.shift_r,
    "caps_lock": Key.caps_lock,
    "space": Key.space,
}


COMMAND_VOCABULARY = [
    # Verbs / Action words
    "open", "launch", "start", "switch", "activate", "go", "run",
    "close", "quit", "hide", "minimize", "maximize", "fullscreen",
    "play", "pause", "resume", "stop", "skip", "next", "previous", "rewind",
    "volume", "louder", "quieter", "up", "down", "mute", "unmute", "increase", "decrease",
    "lock", "sleep", "screenshot",
    # Connectors & Fillers
    "to", "the", "and", "then", "a", "an", "this", "that", "it", "my", "is",
    "what", "whats", "what's", "please",
    # macOS Apps & Targets
    "terminal", "iterm", "console", "shell", "safari", "chrome", "google",
    "spotify", "slack", "code", "vscode", "finder", "files", "notes",
    "messages", "imessage", "mail", "email", "calendar", "settings",
    "system", "preferences", "calculator", "preview", "reminders",
    "music", "song", "track", "sound", "window", "screen", "display", "weather",
    # Web & URLs
    "google.com", "youtube.com", "github.com", "dot", "com", "org", "website",
    "[unk]",
]


def check_accessibility(prompt: bool = True) -> bool:
    """Check macOS Accessibility permission and trigger OS prompt if needed."""
    try:
        from ApplicationServices import (
            AXIsProcessTrusted,
            AXIsProcessTrustedWithOptions,
            kAXTrustedCheckOptionPrompt,
        )
        if not AXIsProcessTrusted():
            if prompt:
                AXIsProcessTrustedWithOptions({kAXTrustedCheckOptionPrompt: True})
            return False
        return True
    except Exception:
        return True


class VoiceListener:
    """
    High-performance Push-to-Talk listener.
    Opens microphone stream strictly while the designated PTT key is held down.
    Streams raw PCM frames directly to Vosk for near-zero-latency ASR upon release.
    """

    def __init__(
        self,
        on_utterance: Callable[[str, float], Awaitable[None]],
        ptt_key: str | Key = "ctrl_l",
        sample_rate: int = 16000,
        min_duration_ms: int = 200,
        max_duration_ms: int = 15000,
        vosk_model_path: str | None = None,
        use_grammar: bool = True,
        asr_backend: str = "whisper",
        whisper_model: str = "mlx-community/whisper-base.en-mlx",
        on_start_listening: Callable[[], None] | None = None,
        on_stop_listening: Callable[[str], None] | None = None,
    ) -> None:
        self.on_utterance = on_utterance
        self.on_start_listening = on_start_listening
        self.on_stop_listening = on_stop_listening
        self.sample_rate = sample_rate
        self.min_duration_ms = min_duration_ms
        self.max_duration_ms = max_duration_ms
        self.use_grammar = use_grammar
        self.asr_backend = asr_backend.lower().strip()
        self.whisper_model = whisper_model
        self._grammar_json = json.dumps(COMMAND_VOCABULARY) if use_grammar else None

        # Resolve PTT Key
        if isinstance(ptt_key, str):
            clean_name = ptt_key.strip().lower()
            self.ptt_key = KEY_NAME_MAP.get(clean_name, Key.ctrl_l)
        else:
            self.ptt_key = ptt_key

        self._is_pressed = False
        self._press_time: float = 0.0
        self._running = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._key_listener: keyboard.Listener | None = None
        self._stream: sd.InputStream | None = None
        self._stream_lock = threading.Lock()
        self._audio_frames: list[np.ndarray] = []

        # Vosk initialization
        self.model: vosk.Model | None = None
        self.recognizer: vosk.KaldiRecognizer | None = None
        if self.asr_backend == "vosk":
            self._init_vosk(vosk_model_path)

    def _create_recognizer(self) -> vosk.KaldiRecognizer | None:
        if not self.model:
            return None
        if self._grammar_json:
            rec = vosk.KaldiRecognizer(self.model, float(self.sample_rate), self._grammar_json)
        else:
            rec = vosk.KaldiRecognizer(self.model, float(self.sample_rate))
        rec.SetWords(False)
        return rec

    def _init_vosk(self, model_path: str | None) -> None:
        """Initialize Vosk speech recognition model."""
        vosk.SetLogLevel(-1)  # Silence verbose Vosk logs
        if model_path and os.path.exists(model_path):
            self.model = vosk.Model(model_path)
        else:
            try:
                self.model = vosk.Model(lang="en-us")
            except Exception as ex:
                print(f"⚠️ Vosk model download/load error: {ex}")
                self.model = None

        self.recognizer = self._create_recognizer()

    @property
    def is_recording(self) -> bool:
        """Returns True while PTT key is held and audio stream is actively capturing."""
        return self._is_pressed

    def _audio_callback(
        self, indata: np.ndarray, frames: int, time_info: Any, status: sd.CallbackFlags
    ) -> None:
        """Audio callback running on high-priority audio thread."""
        if not self._is_pressed:
            return

        # Check maximum duration timeout (prevent stuck key)
        if (time.time() - self._press_time) * 1000 > self.max_duration_ms:
            print("\n⚠️ PTT maximum duration reached (15s); auto-releasing audio stream.")
            self._handle_release()
            return

        # Buffer frames for Whisper / audio processing
        self._audio_frames.append(indata.copy())

        # Feed 16-bit PCM bytes to Vosk if using Vosk backend
        if self.asr_backend == "vosk" and self.recognizer:
            data_bytes = indata.tobytes()
            self.recognizer.AcceptWaveform(data_bytes)

    def _handle_press(self) -> None:
        """Invoked when PTT key transition from UP to DOWN occurs."""
        with self._stream_lock:
            if self._is_pressed:
                return  # Debounce: ignore repeated OS key-press events
            self._is_pressed = True
            self._press_time = time.time()
            self._audio_frames.clear()

            # Reset Vosk recognizer state with grammar if using Vosk
            if self.asr_backend == "vosk":
                self.recognizer = self._create_recognizer()

            # Start audio input stream
            try:
                self._stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype="int16",
                    callback=self._audio_callback,
                    blocksize=512,  # 32ms frame size
                )
                self._stream.start()
                if self.on_start_listening:
                    self.on_start_listening()
                backend_lbl = "Whisper" if self.asr_backend == "whisper" else "Vosk"
                print(f"\r🔴 [{backend_lbl} Recording...] Hold key & speak    ", end="", flush=True)
            except Exception as ex:
                print(f"\n❌ Microphone stream error: {ex}")
                self._is_pressed = False

    def _handle_release(self) -> None:
        """Invoked when PTT key transition from DOWN to UP occurs."""
        with self._stream_lock:
            if not self._is_pressed:
                return
            self._is_pressed = False
            duration_ms = (time.time() - self._press_time) * 1000

            # Stop and close audio stream
            if self._stream:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None

            print(f"\r⚡ [Processing...] Key released ({duration_ms:.0f}ms) ", end="", flush=True)

            # Ignore accidental taps shorter than minimum duration
            if duration_ms < self.min_duration_ms:
                print("\r[Ignored brief tap]                       ", flush=True)
                return

            transcript = ""
            if self.asr_backend == "whisper" and self._audio_frames:
                try:
                    import mlx_whisper
                    pcm_all = np.concatenate(self._audio_frames, axis=0).flatten()
                    audio_float32 = pcm_all.astype(np.float32) / 32768.0

                    # 1. Transient Click Trimming: strip mechanical key clatter (~50ms from start and end)
                    click_samples = int(self.sample_rate * 0.050)
                    if len(audio_float32) > click_samples * 3:
                        audio_float32 = audio_float32[click_samples:-click_samples]

                    # 2. Dynamic Energy Gate & Peak Normalization
                    max_amp = float(np.max(np.abs(audio_float32))) if len(audio_float32) > 0 else 0.0
                    if max_amp < 0.015:
                        print("\r[Ignored background silence/click]        ", flush=True)
                        return
                    audio_float32 = (audio_float32 / max_amp) * 0.95

                    # 3. Vocabulary Priming Prompt to anchor Whisper's decoding language model
                    whisper_prompt = (
                        "Open Terminal, Safari, Chrome, Spotify, Settings, Code, Notes, Finder, Slack. "
                        "Close window, close settings, minimize, maximize, snap left, snap right, snap top, snap bottom. "
                        "Click, double click, right click, click center, move cursor. "
                        "Volume up, volume down, mute, battery status, search Google, what time is it."
                    )

                    res = mlx_whisper.transcribe(
                        audio_float32,
                        path_or_hf_repo=self.whisper_model,
                        initial_prompt=whisper_prompt,
                        temperature=0.0,
                    )
                    transcript = res.get("text", "").strip()
                except Exception as ex:
                    print(f"\n⚠️ MLX-Whisper error ({ex}); trying Vosk fallback")
                    if self.recognizer:
                        final_json = self.recognizer.FinalResult()
                        transcript = json.loads(final_json).get("text", "").strip()
            elif self.recognizer:
                try:
                    final_json = self.recognizer.FinalResult()
                    res_dict = json.loads(final_json)
                    transcript = res_dict.get("text", "").strip()
                except Exception as ex:
                    print(f"\n⚠️ STT parsing error: {ex}")

            # Strip punctuation and normalize case (e.g. "Open Terminal." -> "open terminal")
            transcript = transcript.strip().strip(".?!,;:\"'").lower()

            # Apply phonetic and homophone repair
            try:
                from semantic_router import PhoneticNormalizer
                repaired = PhoneticNormalizer.repair(transcript)
                if repaired != transcript:
                    transcript = repaired
            except Exception:
                pass

            if self.on_stop_listening:
                self.on_stop_listening(transcript)

            if not transcript:
                print("\r[No speech detected]                      ", flush=True)
                return

            print(f'\r🎤 Heard: "{transcript}" ({duration_ms:.0f}ms)    ', flush=True)

            # Schedule utterance processing in the main event loop
            if self._loop and self._loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    self.on_utterance(transcript, duration_ms), self._loop
                )

    def _on_key_press(self, key: Key | KeyCode | None) -> None:
        """Global key press listener callback."""
        if key == self.ptt_key:
            self._handle_press()

    def _on_key_release(self, key: Key | KeyCode | None) -> None:
        """Global key release listener callback."""
        if key == self.ptt_key:
            self._handle_release()

    def start_recording(self) -> None:
        """Manually trigger recording start (for Enter-key or UI triggers)."""
        self._handle_press()

    def stop_recording(self) -> None:
        """Manually trigger recording stop and execute action."""
        self._handle_release()

    async def start(self) -> bool:
        """Start the keyboard listener. Returns True if global PTT is active, False if Accessibility missing."""
        self._loop = asyncio.get_running_loop()
        self._running = True

        if not check_accessibility(prompt=True):
            return False

        try:
            self._key_listener = keyboard.Listener(
                on_press=self._on_key_press,
                on_release=self._on_key_release,
            )
            self._key_listener.start()
            return True
        except Exception:
            return False

    async def stop(self) -> None:
        """Stop keyboard and audio listeners cleanly."""
        self._running = False
        if self._is_pressed:
            self._handle_release()
        if self._key_listener:
            try:
                self._key_listener.stop()
            except Exception:
                pass
            self._key_listener = None
