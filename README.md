# Qwerty

**Turn the desk around your laptop into tap zones.** Qwerty listens through
your microphone, recognises *where* on the desk you tapped from the sound
alone, and fires an action you assigned to that spot — open an app, run a
command, speak text, press a hotkey, take a screenshot, and more.

No cloud. No accounts. Everything runs and stays on your machine.

---

## Requirements

- **Windows 10 or 11.**
- **A microphone** — the built-in laptop mic works; an external or USB mic
  often works better. Qwerty calibrates to whichever mic you pick.

That's it. No .NET, no Python, no Rust toolchain needed to *run* it.

## Install

Qwerty ships as a **portable folder** — there is no installer and it needs no
administrator rights.

1. Unzip the release anywhere (e.g. your Desktop, or `C:\Tools\Qwerty`).
2. Run **`qwerty-app.exe`**.
3. On first launch, Qwerty opens the **calibration wizard** (below).

To remove it, delete the folder. Your profiles and settings live under
`%APPDATA%\Qwerty` — delete that too for a completely clean removal.

## First run — calibration

Accuracy comes from teaching Qwerty *your* desk and *your* mic, so the first
thing it does is walk you through a short guided calibration:

1. **Environment check** — confirms the mic is being heard and the room is
   quiet enough.
2. **Zone layout** — you choose how many tap zones you want (e.g. four corners
   of the trackpad surround, or spots along the desk edge).
3. **Per-zone capture** — you tap each zone a handful of times so Qwerty learns
   its sound.
4. **Consistency check** — Qwerty verifies the zones actually sound distinct
   from each other, and tells you if two are too alike to separate reliably.
5. **Negative examples** — a few taps that should *not* trigger anything, so
   stray knocks and typing don't fire actions.
6. **Save** — the result is stored as a **profile** you can reuse, rename,
   duplicate, import, or export.

You can recalibrate any time, and keep **multiple profiles** for different desks
or mics and switch between them.

## The app

A left-hand rail gives one-click access to five areas:

- **Home** — start/stop listening, see the live input meter, and watch each
  zone light up as you tap. It also warns you, honestly, when something is off:
  the mic differs from the one a profile was calibrated on, the input signal is
  too low, or the system fell behind and dropped audio.
- **Zones & profiles** — assign an action (or a chain of actions) to each zone,
  with a per-action **Test** button, and manage your profiles.
- **Diagnostics** — a live waveform, the classifier's own spectrogram, the
  feature-space view of your last tap, and a readout of the raw input health
  (sample rate, applied gain, dropped samples).
- **Evaluation** — run a guided held-out test and get a confusion matrix and an
  accuracy report you can keep a history of and export as JSON or CSV.
- **Settings** — theme, start-with-Windows, the global pause/resume hotkey, and
  more.

Qwerty also lives in the **system tray**: the tray icon reflects whether it is
listening, paused, or in an error state, and a **global hotkey** pauses and
resumes listening from anywhere.

## What a tap can do

Each zone can trigger one action or a chain of them:

| Action | What it does |
|---|---|
| **Open app** | Launch a program |
| **Open file / folder** | Open a path in its default handler |
| **Open URL** | Open a link in your browser |
| **Run command** | Run a command line with arguments |
| **Copy to clipboard** | Put text on the clipboard |
| **Speak** | Read text aloud (Windows text-to-speech) |
| **Play sound** | Play a system sound |
| **Show toast** | Pop a Windows notification |
| **Send keystroke** | Press a key combination (e.g. `Ctrl+Shift+M`) |
| **Screenshot** | Capture the screen or a selected region |
| **Visual only** | Just light up the zone — useful while calibrating |

## Themes

Eight themes, including a system-following mode, a high-contrast theme, and a
see-through "glass" look: **System, Daylight, Midnight, Aurora, High Contrast,
Ember, Grape, Translucent**.

## Privacy

Qwerty is **local-only by design**. It makes **no network calls** — no
telemetry, no update check, no upload. Audio is analysed on the fly to detect
taps and is never recorded to disk or sent anywhere. Your profiles and reports
are plain, human-readable JSON files under `%APPDATA%\Qwerty`.

## Troubleshooting

Qwerty tries to tell you *why* something isn't working rather than fail quietly:

- **"Recalibrate — the mic changed."** A tap's sound is specific to the mic, so
  a profile calibrated on one mic isn't trusted on another. Recalibrate on the
  current mic, or switch to a profile made for it.
- **"Low mic signal."** Qwerty is boosting the input hard to hear the desk.
  Move the mic closer to the tap zones, or raise its input level in Windows
  sound settings.
- **"Audio samples dropped."** The system fell behind and some taps may have
  been missed. Close heavy background apps.
- **A tap does nothing.** Open **Diagnostics** and tap: the feature-space view
  shows whether the tap was accepted and which zone it landed nearest. If zones
  overlap there, recalibrate with the spots spaced further apart.

## Building from source

You need a recent stable **Rust** toolchain and the MSVC build tools.

```powershell
# Build and run in development
cargo run -p qwerty-app

# Run the whole test suite and lints
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings

# Synthetic DSP stress / acceptance test
cargo run -p qwerty-core --example soak -- --duration 1800
```

### Making a portable release

Qwerty is distributed as a portable folder (no installer, by design — it keeps
distribution simple and needs no admin rights):

```powershell
# Optimised build
cargo build --release -p qwerty-app

# The runnable binary is:
#   target\release\qwerty-app.exe
# Zip that .exe together with LICENSE into a folder and share it. The app
# creates %APPDATA%\Qwerty on first run; nothing else needs to travel with it.
```

## License

See [LICENSE](LICENSE).
