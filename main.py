"""
Voice-Controlled Mac Assistant (laya-mlx)
Native Apple Silicon "System 1" Decision Paradigm (<100ms End-to-End Latency).
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
import time
from typing import Any

import AppKit
from decision_engine import Decision, DecisionEngine, KeywordFallback
from hud import VoiceHUD
from listener import VoiceListener
from mac_controller import ActionResult, MacController
from sound_fx import SoundFX

TEST_CASES = [
    # (input_text, expected_action_type, expected_target, expected_media)
    ("open terminal", "launch_app", "terminal", "none"),
    ("switch to chrome", "launch_app", "chrome", "none"),
    ("launch spotify", "launch_app", "spotify", "none"),
    ("play music", "media_control", None, "play_pause"),
    ("pause", "media_control", None, "play_pause"),
    ("next song", "media_control", None, "next"),
    ("skip", "media_control", None, "next"),
    ("volume up", "system_volume", None, "volume_up"),
    ("mute", "system_volume", None, "mute"),
    ("lock the screen", "system_action", None, "none"),
    ("open google.com", "open_url", None, "none"),
    ("minimize this window", "window_mgmt", None, "none"),
    ("switch to spotify and play", "launch_app", "spotify", None),  # compound
    ("what's the weather", "unrecognized", None, "none"),  # reject casual / QA
    ("snap left", "window_mgmt", "left", "none"),
    ("snap top", "window_mgmt", "top", "none"),
    ("maximize window", "window_mgmt", "maximize", "none"),
    ("search google for apple silicon", "open_url", "search_google:apple silicon", "none"),
    ("search reddit for macbook air", "open_url", "search_reddit:macbook air", "none"),
    ("type git commit", "keystroke", "type:git commit", "none"),
    ("copy", "keystroke", "copy", "none"),
    ("paste", "keystroke", "paste", "none"),
    ("click cancel", "ui_click", "button:cancel", "none"),
    ("click file save", "ui_click", "menu:File save", "none"),
    ("battery status", "perception", "battery", "none"),
    ("active window", "perception", "window", "none"),
    ("toggle dark mode", "system_action", "dark_mode", "none"),
    ("empty trash", "system_action", "empty_trash", "none"),
    ("click", "mouse", "click", "none"),
    ("double click", "mouse", "double_click", "none"),
    ("right click", "mouse", "right_click", "none"),
    ("move cursor to center", "mouse", "move:center", "none"),
    ("click center", "mouse", "click_center", "none"),
    ("move cursor up 100", "mouse", "move_rel:0,-100", "none"),
    # Phonetic repair & acoustic conditioning test cases
    ("those the settings", "window_mgmt", "quit:system", "none"),
    ("hello settings", "launch_app", "system", "none"),
    ("opened mm", "launch_app", "terminal", "none"),
    ("opened a minute", "launch_app", "terminal", "none"),
    ("close window", "window_mgmt", "close", "none"),
    ("close settings", "window_mgmt", "quit:system", "none"),
]


class AssistantDaemon:
    """Orchestrates Listener, DecisionEngine, and MacController."""

    def __init__(
        self,
        checkpoint: str = "aac6fef/laya-multilingual-mlx",
        ptt_key: str = "ctrl_l",
        dry_run: bool = False,
        use_fallback_only: bool = False,
        asr_backend: str = "whisper",
        whisper_model: str = "mlx-community/whisper-base.en-mlx",
        use_hud: bool = True,
    ) -> None:
        self.dry_run = dry_run
        self.controller = MacController()
        self.engine = (
            None
            if use_fallback_only
            else DecisionEngine(checkpoint=checkpoint)
        )
        self.ptt_key = ptt_key
        self.asr_backend = asr_backend
        self.whisper_model = whisper_model
        self.use_hud = use_hud
        self.hud: VoiceHUD | None = VoiceHUD() if use_hud else None
        self.listener: VoiceListener | None = None
        self._shutdown_event = asyncio.Event()

    async def initialize(self) -> None:
        """Initialize all subsystems with startup timing metrics."""
        t_init = time.perf_counter_ns()
        print("=" * 60)
        print("⚡ VOICE-CONTROLLED MAC ASSISTANT (laya-mlx System 1)")
        print("=" * 60)

        # 1. Controller check
        t0 = time.perf_counter_ns()
        running_apps = self.controller.get_running_apps()
        c_lat = (time.perf_counter_ns() - t0) / 1_000_000
        print(f"🖥️  MacController ready ({c_lat:.1f}ms) — {len(running_apps)} running apps detected")

        # 2. Decision Engine initialization
        if self.engine:
            await self.engine.initialize()
            warm_ms = self.engine.warmup()
            print(f"🔥 Warmup inference JIT compilation: {warm_ms:.1f}ms")
        else:
            print("🚀 Running in KeywordFallback-only mode (ultra-fast sub-ms mode)")

        total_init = (time.perf_counter_ns() - t_init) / 1_000_000
        print(f"✨ Subsystems initialized in {total_init:.1f}ms\n")

    async def process_utterance(self, transcript: str, duration_ms: float = 0.0) -> list[ActionResult]:
        """Core pipeline: Transcript -> Decision -> Execution -> Telemetry."""
        t_pipeline = time.perf_counter_ns()

        # Step 1: Decision
        t_dec = time.perf_counter_ns()
        if self.engine:
            decisions = await self.engine.decide(transcript)
        else:
            chunks = DecisionEngine.split_compound(transcript)
            decisions = [KeywordFallback.parse(c) for c in chunks]
        dec_time_ms = (time.perf_counter_ns() - t_dec) / 1_000_000

        results: list[ActionResult] = []

        print(f'┌─ [🎤 "{transcript}"] (spoken: {duration_ms:.0f}ms)')
        for i, dec in enumerate(decisions):
            engine_tag = "fallback" if dec.used_fallback else "laya-mlx"
            print(
                f'│  → [🧠 {engine_tag}: {dec.latency_ms:.1f}ms | '
                f'action={dec.action_type} target={dec.target_entity} media={dec.media_command} '
                f'conf={dec.confidence:.2f} confirm={dec.requires_confirmation:.2f}]'
            )

            # Step 2: Safety confirmation check
            if dec.requires_confirmation > 0.7:
                print("│  ⚠️ [DESTRUCTIVE ACTION BLOCKED: requires manual confirmation]")
                results.append(ActionResult(False, 0.0, error="Blocked by confirmation safeguard"))
                continue

            # Step 3: Execution
            if self.dry_run:
                print("│  → [🖥️  macOS: 0.0ms | dry-run=true (action skipped)]")
                results.append(ActionResult(True, 0.0))
            else:
                t_exec = time.perf_counter_ns()
                res = await self.controller.execute(dec)
                exec_time_ms = (time.perf_counter_ns() - t_exec) / 1_000_000

                status = "✅ success" if res.success else f"❌ failed: {res.error}"
                if res.output:
                    status += f" [{res.output}]"
                print(f"│  → [🖥️  macOS: {res.latency_ms:.1f}ms | {status}]")
                results.append(res)

        total_pipe_ms = (time.perf_counter_ns() - t_pipeline) / 1_000_000
        print(f"└─ Total Latency: {total_pipe_ms:.1f}ms ⚡ (excl. speech)\n")

        # Visual HUD & Audio feedback update
        if self.hud and decisions:
            primary_dec = decisions[0]
            first_output = next((r.output for r in results if r.output), None)
            if results and any(r.success for r in results):
                self.hud.show_success(
                    primary_dec.action_type, primary_dec.target_entity, total_pipe_ms, output=first_output
                )
            elif primary_dec.action_type == "unrecognized":
                self.hud.show_error("Unrecognized Voice Command")
            elif any(r.error and "safeguard" in r.error.lower() for r in results):
                self.hud.show_error("Destructive Action Blocked")
            else:
                self.hud.show_error("Action Failed")

        return results

    async def run_enter_mode(self) -> None:
        """Fallback interactive terminal voice mode using Enter key."""
        print("🎙️  [Terminal Voice Mode Active]")
        print("   1. Press [ENTER] to START recording")
        print("   2. Speak your command")
        print("   3. Press [ENTER] to STOP and execute\n")
        loop = asyncio.get_running_loop()
        is_rec = False
        while not self._shutdown_event.is_set():
            try:
                line = await loop.run_in_executor(None, sys.stdin.readline)
                if not line and line != "\n":
                    break
            except Exception:
                break
            if self._shutdown_event.is_set():
                break

            if not is_rec:
                is_rec = True
                self.listener.start_recording()
            else:
                is_rec = False
                self.listener.stop_recording()

    async def run_live(self) -> None:
        """Run continuous Push-to-Talk daemon with automatic fallback."""
        self.listener = VoiceListener(
            on_utterance=self.process_utterance,
            ptt_key=self.ptt_key,
            asr_backend=self.asr_backend,
            whisper_model=self.whisper_model,
            on_start_listening=self.hud.show_listening if self.hud else None,
            on_stop_listening=self.hud.show_processing if self.hud else None,
        )
        started = await self.listener.start()

        if started:
            key_desc = self.ptt_key.upper() if isinstance(self.ptt_key, str) else str(self.ptt_key)
            print(f"🟢 Global Daemon active: Hold [{key_desc}] anywhere on your Mac and speak.")
            print("   Press Ctrl+C in terminal to stop.\n")
            try:
                while not self._shutdown_event.is_set():
                    if self.hud:
                        event = self.hud.app.nextEventMatchingMask_untilDate_inMode_dequeue_(
                            AppKit.NSEventMaskAny,
                            AppKit.NSDate.dateWithTimeIntervalSinceNow_(0.03),
                            AppKit.NSDefaultRunLoopMode,
                            True,
                        )
                        if event:
                            self.hud.app.sendEvent_(event)
                    await asyncio.sleep(0.01)
            finally:
                if self.listener:
                    await self.listener.stop()
                if self.hud:
                    self.hud.hide()
                print("\n👋 Assistant shut down cleanly.")
        else:
            print("⚠️  macOS Accessibility Permission Required for Global Hotkeys (pynput)")
            print("   System Settings > Privacy & Security > Accessibility")
            print("   Toggle ON the terminal/IDE application you are running.")
            print("   Opening System Settings now...")
            try:
                import subprocess
                subprocess.Popen(["open", "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"])
            except Exception:
                pass
            print("-" * 60)
            try:
                # Run terminal mode alongside Cocoa event loop
                asyncio.create_task(self.run_enter_mode())
                while not self._shutdown_event.is_set():
                    if self.hud:
                        event = self.hud.app.nextEventMatchingMask_untilDate_inMode_dequeue_(
                            AppKit.NSEventMaskAny,
                            AppKit.NSDate.dateWithTimeIntervalSinceNow_(0.03),
                            AppKit.NSDefaultRunLoopMode,
                            True,
                        )
                        if event:
                            self.hud.app.sendEvent_(event)
                    await asyncio.sleep(0.01)
            finally:
                if self.listener:
                    await self.listener.stop()
                if self.hud:
                    self.hud.hide()
                print("\n👋 Assistant shut down cleanly.")

    def stop(self) -> None:
        """Signal shutdown."""
        self._shutdown_event.set()


async def run_benchmark(daemon: AssistantDaemon, iterations: int = 100) -> None:
    """Benchmark decision and execution latency distribution."""
    print(f"\n📊 Running Benchmark ({iterations} iterations)...")
    prompt = "open terminal"
    latencies: list[float] = []

    for _ in range(iterations):
        t0 = time.perf_counter_ns()
        if daemon.engine:
            _ = await daemon.engine.decide(prompt)
        else:
            _ = KeywordFallback.parse(prompt)
        elapsed = (time.perf_counter_ns() - t0) / 1_000_000
        latencies.append(elapsed)

    latencies.sort()
    p50 = latencies[int(len(latencies) * 0.50)]
    p95 = latencies[int(len(latencies) * 0.95)]
    p99 = latencies[int(len(latencies) * 0.99)]
    avg = sum(latencies) / len(latencies)

    print("-" * 45)
    print(f"Prompt: '{prompt}'")
    print(f"Average: {avg:.2f}ms")
    print(f"P50:     {p50:.2f}ms")
    print(f"P95:     {p95:.2f}ms")
    print(f"P99:     {p99:.2f}ms")
    print("-" * 45)


async def run_test_suite(daemon: AssistantDaemon) -> None:
    """Execute the full 14-test suite and report accuracy and pass rate."""
    print(f"\n🧪 Running Test Suite ({len(TEST_CASES)} cases)...\n")
    passed = 0
    total = len(TEST_CASES)

    for prompt, exp_action, exp_target, exp_media in TEST_CASES:
        if daemon.engine:
            decisions = await daemon.engine.decide(prompt)
        else:
            decisions = [KeywordFallback.parse(prompt)]

        primary = decisions[0]
        act_ok = primary.action_type == exp_action
        target_ok = (exp_target is None) or (primary.target_entity == exp_target)
        media_ok = (exp_media is None) or (primary.media_command == exp_media)

        is_correct = act_ok and (target_ok or media_ok)
        if is_correct:
            passed += 1
            mark = "✅ PASS"
        else:
            mark = "❌ FAIL"

        engine_lbl = "fallback" if primary.used_fallback else "laya"
        print(
            f"{mark} | '{prompt:<26}' -> act={primary.action_type:<13} "
            f"target={primary.target_entity:<9} media={primary.media_command:<10} "
            f"({primary.latency_ms:.1f}ms [{engine_lbl}])"
        )

    print("\n" + "=" * 50)
    print(f"Test Suite Result: {passed}/{total} Passed ({passed/total*100:.1f}%)")
    print("=" * 50 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Voice-Controlled Mac Assistant (laya-mlx)")
    parser.add_argument("--test", type=str, help="Execute pipeline on a test text command")
    parser.add_argument("--test-suite", action="store_true", help="Run full 14-test suite")
    parser.add_argument("--benchmark", type=int, nargs="?", const=100, help="Run latency benchmark N iterations")
    parser.add_argument("--dry-run", action="store_true", help="Evaluate decisions without executing macOS actions")
    parser.add_argument("--ptt-key", type=str, default="ctrl_l", help="Push-to-talk key (default: ctrl_l)")
    parser.add_argument("--checkpoint", type=str, default="aac6fef/laya-multilingual-mlx", help="laya-mlx checkpoint")
    parser.add_argument("--no-laya", action="store_true", help="Use KeywordFallback only (skip laya weights)")
    parser.add_argument(
        "--asr",
        type=str,
        default="whisper",
        choices=["whisper", "vosk"],
        help="Speech-to-text backend: 'whisper' (Metal-accelerated Whisper, high accuracy) or 'vosk' (grammar instant)",
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="mlx-community/whisper-base.en-mlx",
        help="HuggingFace model ID for MLX-Whisper (default: mlx-community/whisper-base.en-mlx)",
    )
    parser.add_argument("--no-hud", action="store_true", help="Disable native floating macOS HUD")
    return parser.parse_args()


async def async_main() -> None:
    args = parse_args()
    daemon = AssistantDaemon(
        checkpoint=args.checkpoint,
        ptt_key=args.ptt_key,
        dry_run=args.dry_run,
        use_fallback_only=args.no_laya,
        asr_backend=args.asr,
        whisper_model=args.whisper_model,
        use_hud=not args.no_hud,
    )
    await daemon.initialize()

    # Handle OS termination signals
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, daemon.stop)
        except NotImplementedError:
            pass

    if args.test:
        await daemon.process_utterance(args.test)
    elif args.test_suite:
        await run_test_suite(daemon)
    elif args.benchmark:
        await run_benchmark(daemon, iterations=args.benchmark)
    else:
        await daemon.run_live()


def main() -> None:
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
