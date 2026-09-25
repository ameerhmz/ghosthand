# LAYA

A local, low-latency push-to-talk voice assistant for macOS built specifically for Apple Silicon. 

Instead of routing spoken voice through cloud APIs or slow token-streaming LLMs that take 3–5 seconds to respond, LAYA uses on-device MLX neural models and deterministic semantic routing to execute system actions in **under 100 milliseconds** post-speech.

```
┌─────────────────┐     ┌──────────────────────┐     ┌─────────────────────┐     ┌─────────────────┐
│ Push-to-Talk    │ ──> │ Anti-Click Trim &    │ ──> │ MLX-Whisper Base    │ ──> │ Semantic Router │
│ [Left Control]  │     │ Peak Normalization   │     │ (Metal FP16 ~70ms)  │     │ (<0.1ms Cosine) │
└─────────────────┘     └──────────────────────┘     └─────────────────────┘     └────────┬────────┘
                                                                                          │
┌─────────────────┐     ┌──────────────────────┐     ┌─────────────────────┐              │
│ Native Floating │ <── │ Physical Mouse Glid- │ <── │ MacController       │ <────────────┘
│ Frosted HUD     │     │ ing / Window Tiling  │     │ (Quartz & PyObjC)   │
└─────────────────┘     └──────────────────────┘     └─────────────────────┘
```

---

## Why This Exists

Most voice assistants on the desktop suffer from two problems:
1. **The Cloud Latency Tax**: Uploading audio, waiting for cloud speech-to-text, streaming tokens from a frontier LLM, and parsing JSON function calls adds 2,000–5,000ms of lag for basic commands like *"snap left"* or *"volume up"*.
2. **The Acoustic "Hit-or-Miss" Problem**: Offline lightweight models often fail on short desktop speech clips due to mechanical key-switch clatter (pressing and releasing physical keys) and unconditioned decoder vocabulary, yielding phonetic garbage like *"opened mm"* instead of *"open terminal"*.

LAYA approaches desktop voice control as a **System 1 classification and routing problem**:
- Audio capture is physical key-gated (zero background CPU usage).
- Mechanical click transients are trimmed out before ASR.
- Decoder decoding is vocabulary-primed via `initial_prompt`.
- Phonetic homophones are normalized deterministically.
- Intents are classified via dual-stage routing: sub-millisecond vector prototype matching combined with Apple Silicon Metal inference (`laya-mlx`).
- Actions execute natively via Quartz CGEvents, Cocoa AppKit, and AppleScript.

---

## Latency Benchmark (Apple Silicon M4)

Benchmarked on an M4 Mac running macOS Sonoma / Sequoia with unified memory:

| Subsystem | Implementation | Latency |
|---|---|---|
| Hardware Key Detection | `pynput` CGEventTap | < 1 ms |
| Microphone Capture | `sounddevice` (PortAudio 16kHz PCM) | 0 ms (streamed during keypress) |
| Speech Recognition | `mlx-whisper` (`whisper-base.en-mlx` on Metal) | ~65–75 ms |
| Acoustic & Phonetic Repair | Regex homophone normalizer | < 0.1 ms |
| Semantic Intent Routing | Subword n-gram TF-IDF cosine similarity | < 0.1 ms |
| System 1 Decision Model | `laya-mlx` (Single forward pass) | ~10–14 ms |
| macOS Execution | Cocoa AppKit / Quartz / `osascript` | 5–15 ms |
| **Total Turnaround** | **Key Release $\rightarrow$ Action Complete** | **~85–115 ms** |

---

## Capabilities

### 1. Physical Cursor & Mouse Engine (`mouse_controller.py`)
- **Smooth Human-Like Cursor Gliding**: Visible cursor movement animated along a cubic ease-out curve (`t' = 1 - (1 - t)³`).
- **Physical Clicking**: Single click, double click, right click, and triple click using native Quartz events (`CGEventCreateMouseEvent`).
- **Visual Ripple Overlay**: A floating Cocoa `NSPanel` overlay that flashes a glowing animated cyan/amber ring directly underneath the cursor whenever a click executes.
- **Button Auto-Gliding**: For dialog confirmations (`"click save"`, `"click cancel"`), queries window accessibility coordinates, smoothly glides the mouse cursor onto the physical button, and clicks.
- **Landmarks & Relative Movement**:
  - `"move cursor to center"`, `"cursor top left"`, `"cursor bottom right"`
  - `"move cursor up 100"`, `"cursor left 200"`, `"cursor down 50"`
  - `"mouse scroll down"`, `"mouse scroll up"`

### 2. Window & Space Tiling (12 Modes)
- **Halves**: `"snap left"`, `"snap right"`, `"snap top"`, `"snap bottom"`
- **Quarters**: `"top left"`, `"top right"`, `"bottom left"`, `"bottom right"`
- **Displays & Sizing**: `"maximize window"`, `"almost maximize"` (90% centered), `"center window"`
- **Virtual Desktops (Spaces)**: `"next space"`, `"previous space"`, `"cycle windows"`
- **System Views**: `"mission control"`, `"show desktop"`, `"fullscreen"`, `"hide other apps"`, `"minimize"`, `"close window"`, `"close settings"`

### 3. Keystroke & Clipboard Injection
- **Spoken Text Typing**: `"type git status"`, `"type cargo build"`
- **Zero-Loss Unicode Paste**: `"paste text <string>"` (injects via Cocoa `NSPasteboard` and synthetic `Cmd+V`, bypassing dropped character issues on long strings)
- **Shortcuts**: `"copy"`, `"paste"`, `"cut"`, `"select all"`, `"undo"`, `"redo"`, `"save"`
- **Line Navigation**: `"start of line"`, `"end of line"`, `"top of document"`, `"bottom of document"`, `"delete line"`

### 4. Omnipresent Web Search & Navigation
- **8 Search Engines**:
  - `"search google for <query>"`
  - `"search github for <query>"`
  - `"search youtube for <query>"`
  - `"search reddit for <query>"`
  - `"search wikipedia for <query>"`
  - `"search amazon for <query>"`
  - `"search twitter for <query>"`
  - `"search duckduckgo for <query>"`
- **Browser Controls**: `"new tab"`, `"close tab"`, `"reopen tab"`, `"next tab"`, `"prev tab"`, `"reload page"`, `"hard reload"`, `"zoom in"`, `"zoom out"`, `"devtools"`, `"find on page <query>"`

### 5. Menu Bar & Accessibility Automation
- **Arbitrary Menu Clicks**: `"click menu File New Window"`, `"click menu Edit Select All"`
- **Dialog Button Clicks**: Scans frontmost UI processes to click buttons matching target labels.

### 6. System Perception & Status Reports
- **Battery**: `"battery status"` (queries `pmset`, speaks state aloud, and badges HUD)
- **Active Window**: `"active window"` (inspects frontmost process and window title)
- **Wi-Fi**: `"wifi status"` (queries active network interface and SSID)
- **Clipboard**: `"read clipboard"` (reads preview of clipboard content)
- **Time**: `"what time is it"` (announces formatted time and date)
- **Now Playing**: `"what song is playing"` (inspects Apple Music / Spotify)

### 7. Native Floating Frosted-Glass HUD (`hud.py`)
- Built with pure `AppKit` (`NSPanel` and `NSVisualEffectView` with `NSVisualEffectMaterialHUDWindow`).
- Zero Electron or Chromium runtime overhead.
- Floats non-activatingly above all full-screen games, terminals, and IDEs.
- Real-time states: Listening (pulsing green), Processing (amber), Success badge with timing metrics, and Error/Safeguard warnings.

---

## Installation

### Prerequisites
- macOS 14+ on Apple Silicon (M1, M2, M3, M4)
- Python 3.12 (managed via `uv` or `venv`)
- PortAudio

```bash
# 1. Install PortAudio via Homebrew
brew install portaudio

# 2. Clone the repository
git clone https://github.com/ameerhmz/laya.git
cd laya

# 3. Create virtual environment and install dependencies
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

### macOS Permissions
Because LAYA monitors global push-to-talk key events and controls system windows, macOS requires two permissions:
1. **Accessibility**: `System Settings > Privacy & Security > Accessibility` $\rightarrow$ Enable your terminal emulator (Terminal, iTerm2, Kitty, Ghostty, or VS Code).
2. **Microphone**: `System Settings > Privacy & Security > Microphone` $\rightarrow$ Allow access when prompted on first run.

---

## Usage

### Run the Push-to-Talk Daemon
```bash
python main.py
```
Hold **Left Control (`⌃`)** anywhere on your Mac, speak your command, and release the key.

### CLI Options

| Flag | Description | Default |
|---|---|---|
| `--ptt-key <key>` | Push-to-talk key (`ctrl_l`, `ctrl_r`, `alt_l`, `alt_r`, `caps_lock`) | `ctrl_l` |
| `--asr <backend>` | Speech engine: `whisper` (Metal FP16) or `vosk` (Kaldi offline) | `whisper` |
| `--whisper-model <id>`| HuggingFace repo ID for MLX Whisper model | `mlx-community/whisper-base.en-mlx` |
| `--no-hud` | Disable floating frosted-glass HUD (run pure CLI mode) | `False` |
| `--dry-run` | Classify and log intents without executing macOS actions | `False` |
| `--test-suite` | Run automated 40-case validation test suite | `False` |
| `--benchmark <N>` | Run latency benchmark across N iterations | `False` |
| `--test "<phrase>"` | Execute a single test utterance without speaking | None |

### Examples
```bash
# Test a command directly from the shell
python main.py --test "snap left"

# Run with Right Option (Alt) as the hotkey
python main.py --ptt-key alt_r

# Benchmark intent decision latency over 100 iterations
python main.py --benchmark 100

# Run full test suite headlessly
python main.py --test-suite --no-hud
```

---

## Architecture

```
LAYA/
├── listener.py          # Push-to-talk audio capture, anti-click transient gating, MLX-Whisper
├── semantic_router.py   # PhoneticNormalizer homophone repair & subword n-gram vector router
├── decision_engine.py   # laya-mlx System 1 classifier, compound splitting, confidence gating
├── mouse_controller.py  # Quartz CGEvent mouse gliding, physical clicking & Cocoa click ripple
├── mac_controller.py    # Window tiling (12 modes), 8 search engines, keystrokes, AppleScript
├── hud.py               # Native Cocoa NSPanel frosted-glass HUD (AppKit PyObjC)
├── sound_fx.py          # Native NSSound audio cues & NSSpeechSynthesizer voice output
├── main.py              # Daemon orchestrator, Cocoa event pump, and CLI test harness
└── requirements.txt     # Pinned Python dependencies
```

---

## Safety & Rejection Design

Voice controllers often perform destructive actions when they mishear ambient speech. LAYA includes strict safety gates:
1. **Low-Confidence Rejection**: If inference confidence is below `0.45` and semantic similarity cannot be resolved, the assistant explicitly rejects the utterance instead of falling back to random argmax actions.
2. **Safeguard Block**: Irreversible actions (`"shutdown"`, `"empty trash"`, `"delete"`) are flagged by the decision model (`requires_confirmation > 0.70`) and blocked from silent execution.
3. **No Phantom Window Operations**: Window management targets require explicit direction tokens; unmapped actions safely log errors rather than minimizing the active window.

---

## License

MIT License. Free to use, modify, and build upon.
