//! Windows implementation of [`PlatformActions`] (Phase 5; v2 behaviour p2 §2.6).
//!
//! Every Win32 signature here was verified against the installed
//! `windows-0.62.2` source. This is the only OS-touching module; `qwerty-core`
//! stays platform-free. Construction is infallible (it degrades with a warning
//! like the tray/hotkey setup). One instance is used from one thread (v1: the UI
//! thread or a per-tap worker; v2: the engine's dispatch thread), so the
//! `RefCell`s are fine; the `!Send` `Tts` engine lives on its own speech thread.

use std::cell::RefCell;
use std::os::windows::process::CommandExt;
use std::panic::AssertUnwindSafe;
use std::path::Path;
use std::process::Command;
use std::sync::Arc;
use std::time::Duration;

use arboard::{Clipboard, ImageData};
use crossbeam_channel::{Receiver, RecvTimeoutError, Sender, TrySendError};
use qwerty_core::profile::{KeyCombo, Modifier, ScreenshotMode, SystemSound};
use qwerty_core::veto::SelfOutputKind;
use tts::Tts;
use windows::core::{w, BOOL, HSTRING, PCWSTR};
use windows::Win32::Foundation::{
    GetLastError, ERROR_FILE_NOT_FOUND, ERROR_MORE_DATA, ERROR_PATH_NOT_FOUND, ERROR_SUCCESS,
    ERROR_UNSUPPORTED_TYPE, WIN32_ERROR,
};
use windows::Win32::Graphics::Gdi::{
    BitBlt, CreateCompatibleBitmap, CreateCompatibleDC, DeleteDC, DeleteObject, GetDC, GetDIBits,
    ReleaseDC, SelectObject, BITMAPINFO, BITMAPINFOHEADER, DIB_RGB_COLORS, SRCCOPY,
};
use windows::Win32::Media::Audio::{PlaySoundW, SND_ALIAS, SND_NODEFAULT, SND_SYNC};
use windows::Win32::System::Com::{CoInitializeEx, CoUninitialize, COINIT_APARTMENTTHREADED};
use windows::Win32::System::Registry::{
    RegCloseKey, RegDeleteValueW, RegGetValueW, RegOpenKeyExW, RegSetValueExW, HKEY,
    HKEY_CURRENT_USER, KEY_QUERY_VALUE, KEY_SET_VALUE, REG_ROUTINE_FLAGS, REG_SZ, RRF_NOEXPAND,
    RRF_RT_REG_EXPAND_SZ, RRF_RT_REG_SZ,
};
use windows::Win32::System::Threading::GetCurrentProcessId;
use windows::Win32::UI::Input::KeyboardAndMouse::{
    SendInput, INPUT, INPUT_0, INPUT_KEYBOARD, KEYBDINPUT, KEYBD_EVENT_FLAGS,
    KEYEVENTF_EXTENDEDKEY, KEYEVENTF_KEYUP, VIRTUAL_KEY,
};
use windows::Win32::UI::Shell::{
    ShellExecuteExW, SEE_MASK_FLAG_NO_UI, SEE_MASK_NOASYNC, SHELLEXECUTEINFOW,
};
use windows::Win32::UI::WindowsAndMessaging::{
    GetForegroundWindow, GetSystemMetrics, GetWindowThreadProcessId, SystemParametersInfoW,
    SM_CXVIRTUALSCREEN, SM_CYVIRTUALSCREEN, SM_XVIRTUALSCREEN, SM_YVIRTUALSCREEN,
    SPI_GETCLIENTAREAANIMATION, SW_SHOWNORMAL, SYSTEM_PARAMETERS_INFO_UPDATE_FLAGS,
};
use winrt_toast::{register, Toast, ToastManager};

use super::{ActionError, OutputObserver, PlatformActions, RunKeyChange, Screenshot, ToastSpec};
use crate::engine::protocol::SPEECH_START_TIMEOUT;

/// How long `stop_speech` waits for the speech thread's reply (p2 §2.6).
const STOP_SPEECH_WAIT: Duration = Duration::from_secs(1);

/// The per-user autostart key and this app's value in it.
const RUN_KEY: PCWSTR = w!(r"Software\Microsoft\Windows\CurrentVersion\Run");
const RUN_VALUE: PCWSTR = w!("Qwerty");

/// Run a launched command without flashing a console window.
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

/// The Application User Model ID this app registers itself under so Windows
/// will display its toasts. Windows only shows a notification for a *registered*
/// AUMID; see [`WindowsPlatform::notifier`].
const AUM_ID: &str = "Qwerty.TapZones";

/// Run `f` inside a COM apartment on the current thread, then release it.
///
/// Actions launched off the UI thread (`platform::run_action_chain`) run on a
/// fresh worker thread that has no COM apartment, but the shell open verb
/// (`ShellExecuteW`) and WinRT toast both need one initialized on the calling
/// thread. Initialize single-threaded-apartment (the right model for shell/UI
/// calls), run the work, and uninitialize — but only balance a `CoUninitialize`
/// against an init we actually performed: `CoInitializeEx` returns `S_OK`/
/// `S_FALSE` (both `is_ok()`, both increment this thread's ref count) when it
/// initializes, and an error such as `RPC_E_CHANGED_MODE` when a different
/// apartment already exists — in which case we must not uninitialize a count we
/// did not add. `f` still runs either way (COM is usable regardless).
pub(crate) fn with_com_apartment<T>(f: impl FnOnce() -> T) -> T {
    let hr = unsafe { CoInitializeEx(None, COINIT_APARTMENTTHREADED) };
    let out = f();
    if hr.is_ok() {
        unsafe { CoUninitialize() };
    }
    out
}

/// Whether the OS accessibility setting asks for reduced motion.
///
/// Reads `SPI_GETCLIENTAREAANIMATION`: Windows writes `TRUE` into the out `BOOL`
/// when client-area animations are *enabled*, so "prefers reduced motion" is its
/// negation. Signature verified against the installed `windows-0.62.2` source —
/// `SystemParametersInfoW` returns `windows_core::Result<()>` (the crate `.ok()`s
/// the underlying `BOOL` for us), and `BOOL` lives at `windows::core::BOOL`, not
/// `Win32::Foundation`. On query failure we log and report "no reduced motion",
/// matching this module's documented degrade-with-a-warning construction pattern.
pub(crate) fn os_prefers_reduced_motion() -> bool {
    let mut animations_enabled = BOOL(0);
    let queried = unsafe {
        SystemParametersInfoW(
            SPI_GETCLIENTAREAANIMATION,
            0,
            // `.cast()` infers `*mut c_void`, the same idiom as this file's
            // GetDIBits call.
            Some((&mut animations_enabled as *mut BOOL).cast()),
            SYSTEM_PARAMETERS_INFO_UPDATE_FLAGS(0), // fWinIni is ignored for a GET
        )
    };
    match queried {
        Ok(()) => !animations_enabled.as_bool(),
        Err(e) => {
            eprintln!(
                "warning: could not read the OS animation setting \
                 (SPI_GETCLIENTAREAANIMATION): {e}"
            );
            false
        }
    }
}

/// Register or remove this executable's "Start with Windows" entry under the
/// per-user Run key (`HKCU\Software\Microsoft\Windows\CurrentVersion\Run`).
///
/// Per-user (HKCU, not HKLM) needs no admin rights and matches this app's
/// local-only, single-user nature. `enabled == true` writes the value to the
/// current exe's quoted path; `false` removes it. On any failure this returns a
/// human-readable message for the UI to show — a loud error, never a silent
/// no-op (`CLAUDE.md`: fail fast). Signatures verified against the installed
/// `windows-0.62.2` source, like the rest of this module: the `Reg*` calls
/// return a `WIN32_ERROR` (compared against `ERROR_SUCCESS`), and
/// `RegSetValueExW` takes the value as an `Option<&[u8]>`.
pub(crate) fn set_start_with_windows(enabled: bool) -> Result<(), String> {
    // Resolve the value to write up front, before touching the registry, so the
    // only fallible step while the key handle is open is the registry call
    // itself — nothing early-returns and leaks the `HKEY`.
    let command = if enabled {
        let exe = std::env::current_exe()
            .map_err(|e| format!("could not resolve this app's path: {e}"))?;
        // Quote the path so a space in it can't split the launch command line.
        Some(format!("\"{}\"", exe.display()))
    } else {
        None
    };

    // The Run key exists on every Windows install, so open it (not create it)
    // for writing.
    let mut hkey = HKEY(std::ptr::null_mut());
    let opened = unsafe {
        RegOpenKeyExW(
            HKEY_CURRENT_USER,
            w!(r"Software\Microsoft\Windows\CurrentVersion\Run"),
            None,
            KEY_SET_VALUE,
            &mut hkey,
        )
    };
    if opened != ERROR_SUCCESS {
        return Err(format!(
            "could not open the startup registry key (code {})",
            opened.0
        ));
    }

    let result = match command {
        Some(cmd) => {
            // REG_SZ stores a NUL-terminated UTF-16 string; hand the API a byte
            // view of the wide buffer (`wide` must outlive the call — it does,
            // it lives to the end of this arm).
            let wide: Vec<u16> = cmd.encode_utf16().chain(std::iter::once(0)).collect();
            let bytes =
                unsafe { std::slice::from_raw_parts(wide.as_ptr().cast::<u8>(), wide.len() * 2) };
            let code = unsafe { RegSetValueExW(hkey, w!("Qwerty"), None, REG_SZ, Some(bytes)) };
            if code == ERROR_SUCCESS {
                Ok(())
            } else {
                Err(format!(
                    "could not write the startup entry (code {})",
                    code.0
                ))
            }
        }
        None => {
            let code = unsafe { RegDeleteValueW(hkey, w!("Qwerty")) };
            // Deleting a value that isn't there still yields the desired
            // end-state (no startup entry), so treat "not found" as success.
            if code == ERROR_SUCCESS || code == ERROR_FILE_NOT_FOUND {
                Ok(())
            } else {
                Err(format!(
                    "could not remove the startup entry (code {})",
                    code.0
                ))
            }
        }
    };

    unsafe {
        let _ = RegCloseKey(hkey);
    }
    result
}

/// The text of a Win32 error code, e.g. "The system cannot find the file specified."
fn win32_text(code: WIN32_ERROR) -> String {
    code.to_hresult().message()
}

/// A string registry value's UTF-16 text, without its terminator. `Ok(None)`: the key or the value
/// does not exist (ERROR_FILE_NOT_FOUND). Any other failure is the Win32 code; a value whose type
/// `flags` excludes gives ERROR_UNSUPPORTED_TYPE.
fn read_reg_text(
    root: HKEY,
    subkey: PCWSTR,
    value: PCWSTR,
    flags: REG_ROUTINE_FLAGS,
) -> Result<Option<Vec<u16>>, WIN32_ERROR> {
    let mut size: u32 = 0;
    loop {
        // SAFETY: the strings are NUL-terminated for the call; no data buffer, only the size out.
        let sized =
            unsafe { RegGetValueW(root, subkey, value, flags, None, None, Some(&mut size)) };
        if sized == ERROR_FILE_NOT_FOUND {
            return Ok(None);
        }
        if sized != ERROR_SUCCESS {
            return Err(sized);
        }
        let mut text = vec![0u16; (size as usize).div_ceil(2)];
        let mut got = (text.len() * 2) as u32;
        // SAFETY: `text` holds `got` writable bytes for the duration of the call.
        let read = unsafe {
            RegGetValueW(
                root,
                subkey,
                value,
                flags,
                None,
                Some(text.as_mut_ptr().cast()),
                Some(&mut got),
            )
        };
        if read == ERROR_MORE_DATA {
            // The value grew between the two calls: the next pass queries its new size again
            // (that query ignores `size` on input and overwrites it).
            continue;
        }
        if read == ERROR_FILE_NOT_FOUND {
            return Ok(None);
        }
        if read != ERROR_SUCCESS {
            return Err(read);
        }
        text.truncate(got as usize / 2);
        if let Some(end) = text.iter().position(|&c| c == 0) {
            text.truncate(end);
        }
        return Ok(Some(text));
    }
}

/// An open registry key, closed on drop, so no early return leaks it.
struct OpenKey(HKEY);

impl Drop for OpenKey {
    fn drop(&mut self) {
        // SAFETY: the handle came from RegOpenKeyExW and is closed only here.
        let closed = unsafe { RegCloseKey(self.0) };
        if closed != ERROR_SUCCESS {
            tracing::warn!(code = closed.0, "RegCloseKey failed");
        }
    }
}

/// The Run value "Qwerty" (p2 §2.6, §2.14): when `enabled`, `"<exe path>" <args>` is written if
/// it is missing or different; when not, the value is deleted if present; otherwise nothing
/// changes (XR-UI-18).
pub(crate) fn reconcile_start_with_windows(
    enabled: bool,
    args: &str,
) -> Result<RunKeyChange, String> {
    let desired = if enabled {
        let exe = std::env::current_exe()
            .map_err(|e| format!("could not resolve this app's path: {e}"))?;
        Some(run_command_line(&exe, args))
    } else {
        None
    };
    reconcile_run_value(RUN_KEY, RUN_VALUE, desired.as_deref())
}

/// `"<exe path>" <args>`: the path quoted so a space in it can't split the command line.
// SPEC: §2.6 writes `"<exe path>" <args>`; with empty args no trailing space is added, so the
// value equals the one v1's set_start_with_windows writes.
fn run_command_line(exe: &Path, args: &str) -> String {
    if args.is_empty() {
        format!("\"{}\"", exe.display())
    } else {
        format!("\"{}\" {args}", exe.display())
    }
}

/// Reconciles one REG_SZ value of an existing HKCU key with `desired` (None: absent). The key is
/// opened with KEY_QUERY_VALUE | KEY_SET_VALUE; a write or delete happens only on a difference.
fn reconcile_run_value(
    subkey: PCWSTR,
    value: PCWSTR,
    desired: Option<&str>,
) -> Result<RunKeyChange, String> {
    let mut hkey = HKEY(std::ptr::null_mut());
    // SAFETY: `subkey` is a NUL-terminated wide string; `hkey` is a valid out pointer.
    let opened = unsafe {
        RegOpenKeyExW(
            HKEY_CURRENT_USER,
            subkey,
            None,
            KEY_QUERY_VALUE | KEY_SET_VALUE,
            &mut hkey,
        )
    };
    if opened != ERROR_SUCCESS {
        return Err(format!(
            "could not open the startup registry key: {}",
            win32_text(opened)
        ));
    }
    let key = OpenKey(hkey);

    // A value of another type than REG_SZ (or REG_SZ that isn't ours) is present but different.
    let current = match read_reg_text(key.0, PCWSTR::null(), value, RRF_RT_REG_SZ | RRF_NOEXPAND) {
        Ok(text) => RunValue::from(text),
        Err(code) if code == ERROR_UNSUPPORTED_TYPE => RunValue::OtherType,
        Err(code) => {
            return Err(format!(
                "could not read the startup entry: {}",
                win32_text(code)
            ))
        }
    };

    match desired {
        Some(command) => {
            let wide: Vec<u16> = command.encode_utf16().collect();
            if current == RunValue::Text(wide.clone()) {
                return Ok(RunKeyChange::Unchanged);
            }
            // REG_SZ stores a NUL-terminated UTF-16 string, handed to the API as bytes.
            let bytes: Vec<u8> = wide
                .iter()
                .chain(std::iter::once(&0u16))
                .flat_map(|unit| unit.to_le_bytes())
                .collect();
            // SAFETY: the key is open with KEY_SET_VALUE; `bytes` outlives the call.
            let code = unsafe { RegSetValueExW(key.0, value, None, REG_SZ, Some(&bytes)) };
            if code == ERROR_SUCCESS {
                Ok(RunKeyChange::Written)
            } else {
                Err(format!(
                    "could not write the startup entry: {}",
                    win32_text(code)
                ))
            }
        }
        None => match current {
            RunValue::Missing => Ok(RunKeyChange::Unchanged),
            RunValue::Text(_) | RunValue::OtherType => {
                // SAFETY: the key is open with KEY_SET_VALUE.
                let code = unsafe { RegDeleteValueW(key.0, value) };
                if code == ERROR_SUCCESS {
                    Ok(RunKeyChange::Deleted)
                } else {
                    Err(format!(
                        "could not remove the startup entry: {}",
                        win32_text(code)
                    ))
                }
            }
        },
    }
}

/// What the Run value holds now.
#[derive(Debug, PartialEq, Eq)]
enum RunValue {
    Missing,
    Text(Vec<u16>),
    OtherType,
}

impl From<Option<Vec<u16>>> for RunValue {
    fn from(text: Option<Vec<u16>>) -> RunValue {
        match text {
            Some(text) => RunValue::Text(text),
            None => RunValue::Missing,
        }
    }
}

/// Whether a PlaySound step called PlaySoundW (DP3 and the PlaySound evidence report it).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SoundPlay {
    Played,
    /// The user's sound scheme silences the event (e.g. "No Sounds"): no call was made.
    SilencedByScheme,
}

/// The event key under `HKCU\AppEvents\Schemes\Apps\.Default` and the PlaySound alias, mirroring
/// v1's MessageBeep mapping (p2 §2.6). winmm resolves "SystemDefault" although its scheme key is
/// `.Default`.
fn sound_event(sound: SystemSound) -> (&'static str, PCWSTR, &'static str) {
    match sound {
        SystemSound::Default => (".Default", w!("SystemDefault"), "SystemDefault"),
        SystemSound::Success => ("SystemAsterisk", w!("SystemAsterisk"), "SystemAsterisk"),
        SystemSound::Error => ("SystemHand", w!("SystemHand"), "SystemHand"),
        SystemSound::Notification => (
            "SystemExclamation",
            w!("SystemExclamation"),
            "SystemExclamation",
        ),
    }
}

/// The event's sound in the current scheme: an empty or missing `.Current` default value means
/// the scheme silences it. Otherwise a synchronous PlaySoundW of the alias. *(deviation f)*
fn play_unless_silenced(sound: SystemSound) -> Result<SoundPlay, ActionError> {
    let (event, alias, alias_name) = sound_event(sound);
    let subkey = HSTRING::from(format!(r"AppEvents\Schemes\Apps\.Default\{event}\.Current"));
    let current = read_reg_text(
        HKEY_CURRENT_USER,
        PCWSTR(subkey.as_ptr()),
        PCWSTR::null(),
        RRF_RT_REG_SZ | RRF_RT_REG_EXPAND_SZ | RRF_NOEXPAND,
    )
    .map_err(|code| ActionError::Sound {
        detail: win32_text(code),
    })?;
    match current {
        Some(file) if !file.is_empty() => {}
        Some(_) | None => return Ok(SoundPlay::SilencedByScheme),
    }
    // SAFETY: `alias` is a static NUL-terminated wide string.
    let played = unsafe { PlaySoundW(alias, None, SND_ALIAS | SND_SYNC | SND_NODEFAULT) };
    if played.as_bool() {
        Ok(SoundPlay::Played)
    } else {
        // PlaySound documents no extended error, so GetLastError is never read.
        Err(ActionError::Sound {
            detail: format!("PlaySound({alias_name}) returned FALSE"),
        })
    }
}

/// `output_started(Sound)`, the scheme check and the synchronous play, then `output_ended`, so the
/// interval spans the whole sound (p2 §2.6).
fn play_system_sound(
    observer: Option<&dyn OutputObserver>,
    sound: SystemSound,
) -> Result<SoundPlay, ActionError> {
    match observer {
        Some(observer) => {
            let id = observer.output_started(SelfOutputKind::Sound);
            let played = play_unless_silenced(sound);
            observer.output_ended(id);
            played
        }
        None => play_unless_silenced(sound),
    }
}

/// Capture a sub-rectangle of the virtual screen — `src_x`, `src_y` in
/// virtual-screen pixels, size `w`x`h` — to the clipboard as an RGBA image.
/// Shared by both screenshot modes (full-screen passes the whole virtual
/// screen; region-select passes the user's chosen rectangle). Kept a safe fn
/// with one internal `unsafe` block, like the code it was extracted from.
fn capture_rect_to_clipboard(src_x: i32, src_y: i32, w: i32, h: i32) -> Result<(), ActionError> {
    if w <= 0 || h <= 0 {
        return Err(ActionError::Screenshot {
            detail: "capture rectangle is empty".into(),
        });
    }
    unsafe {
        let screen = GetDC(None);
        if screen.is_invalid() {
            return Err(ActionError::Screenshot {
                detail: "GetDC(NULL) returned null".into(),
            });
        }

        let grab = || -> Result<(usize, usize, Vec<u8>), ActionError> {
            let mem = CreateCompatibleDC(Some(screen));
            let bmp = CreateCompatibleBitmap(screen, w, h);
            let old = SelectObject(mem, bmp.into());
            let blt = BitBlt(mem, 0, 0, w, h, Some(screen), src_x, src_y, SRCCOPY);
            let mut bi = BITMAPINFO {
                bmiHeader: BITMAPINFOHEADER {
                    biSize: std::mem::size_of::<BITMAPINFOHEADER>() as u32,
                    biWidth: w,
                    biHeight: -h, // negative => top-down rows
                    biPlanes: 1,
                    biBitCount: 32,
                    biCompression: 0, // BI_RGB
                    ..Default::default()
                },
                ..Default::default()
            };
            let mut buf = vec![0u8; (w as usize) * (h as usize) * 4]; // BGRX
                                                                      // With a buffer supplied, GetDIBits returns the number of scan
                                                                      // lines copied; require the FULL height, not merely nonzero, so
                                                                      // a short copy is a loud error rather than a truncated image.
            let lines = GetDIBits(
                mem,
                bmp,
                0,
                h as u32,
                Some(buf.as_mut_ptr().cast()),
                &mut bi,
                DIB_RGB_COLORS,
            );
            let ok = blt.is_ok() && lines == h;
            SelectObject(mem, old);
            let _ = DeleteObject(bmp.into());
            let _ = DeleteDC(mem);
            if !ok {
                return Err(ActionError::Screenshot {
                    detail: "BitBlt/GetDIBits failed".into(),
                });
            }
            // BGRX -> RGBA (arboard wants meaningful RGBA; force opaque alpha).
            for px in buf.chunks_exact_mut(4) {
                px.swap(0, 2);
                px[3] = 255;
            }
            Ok((w as usize, h as usize, buf))
        };
        let result = grab();
        let _ = ReleaseDC(None, screen);
        let (wi, hi, rgba) = result?;
        let mut cb = Clipboard::new().map_err(|e| ActionError::Screenshot {
            detail: e.to_string(),
        })?;
        cb.set_image(ImageData {
            width: wi,
            height: hi,
            bytes: rgba.into(),
        })
        .map_err(|e| ActionError::Screenshot {
            detail: e.to_string(),
        })
    }
}

/// The Windows action backend.
///
/// It holds lazily-created, then long-lived, handles behind `RefCell`s because
/// [`PlatformActions`] takes `&self` while these need `&mut` — sound here because
/// one instance is only ever used from the thread that created it (`WindowsPlatform`
/// is `!Send`).
pub struct WindowsPlatform {
    /// The speech engine, on its own thread (see [`SpeechWorker`]). Lazily
    /// started on first `speak` so an app that never speaks pays nothing.
    speech: RefCell<Option<SpeechWorker>>,
    /// The toast notifier, created once the AUMID has been registered.
    toast: RefCell<Option<ToastManager>>,
    /// The engine's self-output observer (p2 §2.6); None in v1.
    observer: RefCell<Option<Arc<dyn OutputObserver>>>,
}

impl Default for WindowsPlatform {
    fn default() -> Self {
        Self::new()
    }
}

impl WindowsPlatform {
    pub fn new() -> Self {
        WindowsPlatform {
            speech: RefCell::new(None),
            toast: RefCell::new(None),
            observer: RefCell::new(None),
        }
    }

    fn observer(&self) -> Option<Arc<dyn OutputObserver>> {
        self.observer.borrow().clone()
    }

    /// The toast notifier, registering this app's AUMID on first use.
    ///
    /// Windows only displays a toast for a *registered* Application User Model
    /// ID. `winrt_toast::register` writes
    /// `HKCU\SOFTWARE\Classes\AppUserModelId\<id>` — no installer, no admin
    /// rights — which is the sanctioned path for an unpackaged `.exe` like this
    /// one. Skipping it would mean toasts are dropped by the platform with no
    /// error at all. `ToastManager` holds only that id and derives `Clone`, so
    /// callers take a cheap copy rather than hold a `RefCell` borrow across the
    /// call.
    fn notifier(&self) -> Result<ToastManager, ActionError> {
        let mut slot = self.toast.borrow_mut();
        if slot.is_none() {
            let icon = std::env::current_exe().ok();
            // `register` *asserts* (panics) instead of returning `Err` if the
            // Kernel Transaction Manager hands back an invalid handle. Contain
            // that: one failed notification must not take down the whole
            // session. This is not swallowing an error — the cause is still
            // surfaced as a specific, named `ActionError`.
            let outcome = std::panic::catch_unwind(AssertUnwindSafe(|| {
                register(AUM_ID, "Qwerty", icon.as_deref())
            }));
            match outcome {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    return Err(ActionError::Notification {
                        detail: format!(
                            "could not register the notification identity ({AUM_ID}): {e}"
                        ),
                    })
                }
                Err(payload) => {
                    // Keep the assertion text (e.g. "assertion failed:
                    // !transaction.is_invalid()") — it names which KTM step
                    // failed, the one detail worth having on a rare failure.
                    let detail = payload
                        .downcast_ref::<&str>()
                        .map(|s| s.to_string())
                        .or_else(|| payload.downcast_ref::<String>().cloned())
                        .unwrap_or_else(|| "unknown panic".to_string());
                    return Err(ActionError::Notification {
                        detail: format!(
                            "registering the notification identity ({AUM_ID}) panicked \
                         inside winrt-toast: {detail}"
                        ),
                    });
                }
            }
            *slot = Some(ToastManager::new(AUM_ID));
        }
        Ok(slot.as_ref().expect("notifier initialized above").clone())
    }

    /// Open a file/folder/app/URL with the shell's default "open" verb.
    ///
    /// ShellExecuteExW with SEE_MASK_NOASYNC completes any DDE conversation before returning (on
    /// the engine's pumping STA), and SEE_MASK_FLAG_NO_UI shows no error dialog, because failures
    /// are reported (p2 §2.6).
    fn shell_open(target: &str) -> Result<(), ActionError> {
        let file = HSTRING::from(target);
        let mut info = SHELLEXECUTEINFOW {
            cbSize: std::mem::size_of::<SHELLEXECUTEINFOW>() as u32,
            fMask: SEE_MASK_NOASYNC | SEE_MASK_FLAG_NO_UI,
            lpVerb: w!("open"),
            lpFile: PCWSTR(file.as_ptr()),
            nShow: SW_SHOWNORMAL.0,
            ..Default::default()
        };
        // SAFETY: `info` is a valid SHELLEXECUTEINFOW whose strings outlive the call.
        match unsafe { ShellExecuteExW(&mut info) } {
            Ok(()) => Ok(()),
            Err(e)
                if e.code() == ERROR_FILE_NOT_FOUND.to_hresult()
                    || e.code() == ERROR_PATH_NOT_FOUND.to_hresult() =>
            {
                Err(ActionError::TargetNotFound {
                    target: target.to_string(),
                })
            }
            Err(e) => Err(ActionError::LaunchFailed {
                target: target.to_string(),
                reason: e.message(),
            }),
        }
    }
}

/// Whether Qwerty's own window has focus: a keystroke would land in Qwerty itself (PARITY-16).
fn foreground_is_this_process() -> bool {
    // SAFETY: no arguments; a null HWND (no foreground window) is handled below.
    let hwnd = unsafe { GetForegroundWindow() };
    let mut pid = 0u32;
    // SAFETY: `pid` is a valid out pointer. A null or stale HWND returns 0 and leaves `pid` 0,
    // which is never this process's id.
    unsafe { GetWindowThreadProcessId(hwnd, Some(&mut pid)) };
    // SAFETY: no arguments.
    pid == unsafe { GetCurrentProcessId() }
}

/// Map a platform-neutral key name to a Windows virtual-key code + whether it is
/// an "extended" key (arrows need `KEYEVENTF_EXTENDEDKEY`).
fn vk_of(name: &str) -> Result<(VIRTUAL_KEY, bool), ActionError> {
    let n = name.trim();
    if n.len() == 1 {
        let c = n.as_bytes()[0].to_ascii_uppercase();
        if c.is_ascii_uppercase() || c.is_ascii_digit() {
            return Ok((VIRTUAL_KEY(c as u16), false));
        }
    }
    if let Some(num) = n
        .strip_prefix(['F', 'f'])
        .and_then(|d| d.parse::<u16>().ok())
    {
        if (1..=12).contains(&num) {
            return Ok((VIRTUAL_KEY(0x70 + num - 1), false)); // VK_F1 = 0x70
        }
    }
    let (vk, ext): (u16, bool) = match n.to_ascii_lowercase().as_str() {
        "space" | "spacebar" => (0x20, false),
        "enter" | "return" => (0x0D, false),
        "tab" => (0x09, false),
        "esc" | "escape" => (0x1B, false),
        "left" => (0x25, true),
        "up" => (0x26, true),
        "right" => (0x27, true),
        "down" => (0x28, true),
        other => {
            return Err(ActionError::Keystroke {
                detail: format!("unrecognized key \"{other}\""),
            })
        }
    };
    Ok((VIRTUAL_KEY(vk), ext))
}

fn modifier_vk(m: &Modifier) -> VIRTUAL_KEY {
    VIRTUAL_KEY(match m {
        Modifier::Ctrl => 0x11,
        Modifier::Shift => 0x10,
        Modifier::Alt => 0x12,
        Modifier::Win => 0x5B,
    })
}

fn key_event(vk: VIRTUAL_KEY, ext: bool, press: bool) -> INPUT {
    let mut flags = KEYBD_EVENT_FLAGS(0);
    if ext {
        flags |= KEYEVENTF_EXTENDEDKEY;
    }
    if !press {
        flags |= KEYEVENTF_KEYUP;
    }
    INPUT {
        r#type: INPUT_KEYBOARD,
        Anonymous: INPUT_0 {
            ki: KEYBDINPUT {
                wVk: vk,
                wScan: 0,
                dwFlags: flags,
                time: 0,
                dwExtraInfo: 0,
            },
        },
    }
}

impl PlatformActions for WindowsPlatform {
    fn open_uri(&self, uri: &str) -> Result<(), ActionError> {
        Self::shell_open(uri)
    }

    fn open_path(&self, path: &Path) -> Result<(), ActionError> {
        let target = path.to_str().ok_or_else(|| ActionError::TargetNotFound {
            target: path.display().to_string(),
        })?;
        Self::shell_open(target)
    }

    fn run_command(&self, cmd: &str, args: &[String]) -> Result<std::process::Child, ActionError> {
        match Command::new(cmd)
            .args(args)
            .creation_flags(CREATE_NO_WINDOW)
            .spawn()
        {
            Ok(child) => Ok(child),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                Err(ActionError::TargetNotFound {
                    target: cmd.to_string(),
                })
            }
            Err(e) => Err(ActionError::LaunchFailed {
                target: cmd.to_string(),
                reason: e.to_string(),
            }),
        }
    }

    fn copy_to_clipboard(&self, text: &str) -> Result<(), ActionError> {
        let mut cb = Clipboard::new().map_err(|e| ActionError::Clipboard {
            detail: e.to_string(),
        })?;
        cb.set_text(text).map_err(|e| ActionError::Clipboard {
            detail: e.to_string(),
        })
    }

    fn play_sound(&self, sound: SystemSound) -> Result<(), ActionError> {
        play_system_sound(self.observer().as_deref(), sound).map(|_| ())
    }

    fn send_keystroke(&self, combo: &KeyCombo) -> Result<(), ActionError> {
        // Never type into Qwerty itself (PARITY-16).
        if foreground_is_this_process() {
            return Err(ActionError::ForegroundIsQwerty);
        }
        let (key, ext) = vk_of(&combo.key)?; // resolve before touching the OS
        let mut inputs = Vec::with_capacity((combo.modifiers.len() + 1) * 2);
        for m in &combo.modifiers {
            inputs.push(key_event(modifier_vk(m), false, true));
        }
        inputs.push(key_event(key, ext, true));
        inputs.push(key_event(key, ext, false));
        for m in combo.modifiers.iter().rev() {
            inputs.push(key_event(modifier_vk(m), false, false));
        }
        let sent = unsafe { SendInput(&inputs, std::mem::size_of::<INPUT>() as i32) };
        if sent as usize != inputs.len() {
            let e = unsafe { GetLastError() };
            // A partial injection (blocked by UIPI or a low-level hook) may have
            // pressed modifiers down without their matching key-up ever landing,
            // which would leave Ctrl/Shift/Alt/Win stuck down system-wide. Emit a
            // best-effort key-up for the key and every modifier before failing.
            let mut release = vec![key_event(key, ext, false)];
            for m in &combo.modifiers {
                release.push(key_event(modifier_vk(m), false, false));
            }
            unsafe { SendInput(&release, std::mem::size_of::<INPUT>() as i32) };
            return Err(ActionError::Keystroke {
                detail: format!(
                    "SendInput injected {sent}/{} (GetLastError {:#x})",
                    inputs.len(),
                    e.0
                ),
            });
        }
        Ok(())
    }

    fn screenshot_to_clipboard(&self, mode: ScreenshotMode) -> Result<Screenshot, ActionError> {
        // The virtual screen (the bounding box of every monitor) is the shared
        // coordinate space for both the region-select overlay and the capture.
        let (vx, vy, vw, vh) = unsafe {
            (
                GetSystemMetrics(SM_XVIRTUALSCREEN),
                GetSystemMetrics(SM_YVIRTUALSCREEN),
                GetSystemMetrics(SM_CXVIRTUALSCREEN),
                GetSystemMetrics(SM_CYVIRTUALSCREEN),
            )
        };
        if vw <= 0 || vh <= 0 {
            return Err(ActionError::Screenshot {
                detail: "virtual screen size is 0".into(),
            });
        }

        let (x, y, w, h) = match mode {
            ScreenshotMode::FullScreen => (vx, vy, vw, vh),
            ScreenshotMode::RegionSelect => {
                match super::region_select::select_screen_region(vx, vy, vw, vh) {
                    Some(rect) => rect,
                    // The user dismissed the overlay (Escape) without selecting — a
                    // deliberate no-capture, not a failure.
                    // SPEC: region_select.rs (kept unchanged, p2 file table) also returns `None`
                    // after logging a Win32 setup failure, so that failure reports as Canceled
                    // too; separating it needs a spec change to select_screen_region.
                    None => return Ok(Screenshot::Canceled),
                }
            }
        };

        capture_rect_to_clipboard(x, y, w, h).map(|()| Screenshot::Copied)
    }

    fn speak(&self, text: &str) -> Result<(), ActionError> {
        let observer = self.observer();
        let mut slot = self.speech.borrow_mut();
        if slot.is_none() {
            let for_callbacks = observer.clone();
            *slot = Some(SpeechWorker::spawn(
                move || open_tts(for_callbacks),
                observer.clone(),
            )?);
        }
        match slot.as_ref() {
            // Waits ≤ SPEECH_START_TIMEOUT for the start reply (synthesis runs on the worker).
            Some(worker) => request_speech(worker, observer.as_deref(), text),
            // INVARIANT: the slot was filled just above.
            None => unreachable!("the speech worker was created above"),
        }
    }

    fn set_output_observer(&self, observer: Arc<dyn OutputObserver>) {
        *self.observer.borrow_mut() = Some(observer);
    }

    fn stop_speech(&self) {
        if let Some(worker) = self.speech.borrow().as_ref() {
            worker.stop();
        }
    }

    fn notify(&self, toast: &ToastSpec) -> Result<(), ActionError> {
        // The veto applies its toast allowance from this start; a toast has no end (p2 §2.6).
        if let Some(observer) = self.observer() {
            observer.output_started(SelfOutputKind::Toast);
        }
        let manager = self.notifier()?;
        let mut card = Toast::new();
        // Line 1 is the title, line 2 the body (`winrt-toast`'s ToastGeneric
        // template). `&str` satisfies `Into<Text>` via its blanket impl.
        card.text1(toast.title.as_str()).text2(toast.body.as_str());

        // `show_with_callbacks`, never `show`: Windows reports some toast
        // failures *asynchronously*, and `show` discards them (it passes `None`
        // for `on_failed`), so a toast that never appears would return `Ok`.
        // That is precisely the silent failure `CLAUDE.md` bans. The callback
        // runs on a WinRT thread, not the UI thread, so it only logs.
        manager
            .show_with_callbacks(
                &card,
                None,
                None,
                Some(Box::new(|e| {
                    eprintln!("warning: Windows failed to display the toast: {e}");
                })),
            )
            .map_err(|e| ActionError::Notification {
                detail: e.to_string(),
            })
    }
}

/// The speech worker's engine: `tts::Tts` in the product, a fake in DP9, because `tts::Tts` cannot
/// take an injected backend (p2 §2.6).
trait SpeechEngine {
    /// Queues `text`, interrupting what is speaking.
    fn speak(&mut self, text: &str) -> Result<(), String>;
    fn is_speaking(&self) -> Result<bool, String>;
    /// Ends every queued utterance; each gets its stop callback, in queue order.
    fn stop(&mut self) -> Result<(), String>;
}

/// `Debug`, not `Display`: `tts::Error::WinRt`'s `Display` is the bare "WinRT error" and drops the
/// HRESULT.
impl SpeechEngine for Tts {
    fn speak(&mut self, text: &str) -> Result<(), String> {
        // interrupt = true: a newly tapped zone speaks now, not queued behind the previous
        // utterance, and utterances finish in the order they began.
        Tts::speak(self, text, true)
            .map(|_| ())
            .map_err(|e| format!("{e:?}"))
    }

    fn is_speaking(&self) -> Result<bool, String> {
        Tts::is_speaking(self).map_err(|e| format!("{e:?}"))
    }

    fn stop(&mut self) -> Result<(), String> {
        Tts::stop(self).map(|_| ()).map_err(|e| format!("{e:?}"))
    }
}

/// The Speak failure detail when the speech backend has no utterance callbacks (p35 R9).
const CALLBACKS_UNSUPPORTED: &str = "utterance callbacks unsupported by the speech backend";

/// Creates the engine on the speech thread. With an observer, the two utterance callbacks are
/// registered once; each captures only the observer and calls its wait-free `speech_finished`
/// (tts 0.26.3 runs the end callback inside its MediaEnded handler with its global mutexes held,
/// and the stop callback inside `stop()`). An `Err` is every Speak's failure detail, verbatim.
fn open_tts(observer: Option<Arc<dyn OutputObserver>>) -> Result<Tts, String> {
    let unavailable = |what: String| format!("text-to-speech is unavailable: {what}");
    let tts = Tts::default().map_err(|e| unavailable(format!("{e:?}")))?;
    if let Some(observer) = observer {
        if !tts.supported_features().utterance_callbacks {
            // R9: fail fast; there is no time-estimate fallback.
            return Err(CALLBACKS_UNSUPPORTED.to_string());
        }
        let on_end = Arc::clone(&observer);
        tts.on_utterance_end(Some(Box::new(move |_| on_end.speech_finished())))
            .map_err(|e| unavailable(format!("registering the utterance end callback: {e:?}")))?;
        tts.on_utterance_stop(Some(Box::new(move |_| observer.speech_finished())))
            .map_err(|e| unavailable(format!("registering the utterance stop callback: {e:?}")))?;
    }
    Ok(tts)
}

enum SpeechRequest {
    /// reply: bounded(1)
    Speak {
        text: String,
        reply: Sender<SpeechReply>,
    },
    /// reply: bounded(1)
    Stop { reply: Sender<Result<(), String>> },
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum SpeechReply {
    Started,
    Failed { detail: String },
}

/// A dedicated thread that owns the speech engine and speaks text handed to it
/// over a channel.
///
/// Two facts about `tts` force this shape. (1) `Tts::speak` blocks the calling
/// thread until the *entire utterance has been synthesized* (it calls `.get()`
/// on a WinRT async op; only playback is fire-and-forget). (2) `Tts`'s handle is
/// an `Rc` internally, so it must be created and used on one thread and never
/// moved across threads. Both are satisfied by creating the engine *on this
/// worker thread* and feeding it requests over a channel. The caller waits only
/// for the start reply (≤ `SPEECH_START_TIMEOUT`), never for playback.
///
/// Failure accounting is the worker's (p2 §2.6): every `Began{Speech}` gets
/// exactly one `speech_finished`, whether the speech succeeds, fails or times out.
struct SpeechWorker {
    requests: Sender<SpeechRequest>,
    /// Never joined: dropping `requests` ends the worker's loop, after which it
    /// stops the engine's queue and drops the engine.
    _thread: std::thread::JoinHandle<()>,
}

impl SpeechWorker {
    fn spawn<E, F>(
        open: F,
        observer: Option<Arc<dyn OutputObserver>>,
    ) -> Result<SpeechWorker, ActionError>
    where
        E: SpeechEngine,
        F: FnOnce() -> Result<E, String> + Send + 'static,
    {
        let (requests, inbox) = crossbeam_channel::unbounded();
        let thread = std::thread::Builder::new()
            .name("qwerty-speech".to_string())
            .spawn(move || run_speech_worker(open, observer.as_deref(), &inbox))
            .map_err(|e| ActionError::Speech {
                detail: format!("could not start the speech thread: {e}"),
            })?;
        Ok(SpeechWorker {
            requests,
            _thread: thread,
        })
    }

    /// `SpeechRequest::Stop`, then a wait of ≤ 1 s for the reply. The worker calls `stop()` on its
    /// own thread (`Tts` is not `Send`), which finishes every queued utterance in order and empties
    /// tts's queue, so a late MediaEnded finds nothing.
    fn stop(&self) {
        let (reply, answer) = crossbeam_channel::bounded(1);
        if self.requests.send(SpeechRequest::Stop { reply }).is_err() {
            // The speech thread has ended, so nothing is queued on it.
            return;
        }
        match answer.recv_timeout(STOP_SPEECH_WAIT) {
            Ok(Ok(())) => {}
            Ok(Err(detail)) => tracing::warn!(%detail, "stopping speech failed"),
            Err(RecvTimeoutError::Timeout) => {
                tracing::warn!("the speech thread did not confirm the stop within 1 s");
            }
            Err(RecvTimeoutError::Disconnected) => {
                tracing::warn!("the speech thread ended before confirming the stop");
            }
        }
    }
}

/// The platform side of a Speak (p2 §2.6): `output_started(Speech)`, then the request, then a wait
/// of ≤ `SPEECH_START_TIMEOUT`: `Started` → `Ok`; `Failed` → `Speech{detail}` (the worker already
/// delivered the finish); timeout → `SpeechTimeout` (the interval stays open until the utterance
/// ends, or `close_stuck` closes it).
fn request_speech(
    worker: &SpeechWorker,
    observer: Option<&dyn OutputObserver>,
    text: &str,
) -> Result<(), ActionError> {
    if let Some(observer) = observer {
        // Speech ends are matched in order (speech_finished), not by id.
        observer.output_started(SelfOutputKind::Speech);
    }
    let (reply, answer) = crossbeam_channel::bounded(1);
    let request = SpeechRequest::Speak {
        text: text.to_string(),
        reply,
    };
    if worker.requests.send(request).is_err() {
        // SPEC: §2.6 does not cover a speech thread that has ended. The request never reached it,
        // so the utterance was never queued and nothing else will finish it: its one finish is
        // delivered here, by the worker's own rule for an utterance that was never queued.
        if let Some(observer) = observer {
            observer.speech_finished();
        }
        return Err(ActionError::Speech {
            detail: "the speech thread has ended".to_string(),
        });
    }
    let wait = Duration::from_secs_f64(SPEECH_START_TIMEOUT.get() / 1000.0);
    match answer.recv_timeout(wait) {
        Ok(SpeechReply::Started) => Ok(()),
        Ok(SpeechReply::Failed { detail }) => Err(ActionError::Speech { detail }),
        Err(RecvTimeoutError::Timeout) => Err(ActionError::SpeechTimeout),
        Err(RecvTimeoutError::Disconnected) => Err(ActionError::Speech {
            detail: "the speech thread ended before replying".to_string(),
        }),
    }
}

/// The speech thread's body. Every request is answered, even when the engine cannot be created:
/// then each Speak fails with `open`'s error as its detail, unchanged.
fn run_speech_worker<E: SpeechEngine>(
    open: impl FnOnce() -> Result<E, String>,
    observer: Option<&dyn OutputObserver>,
    inbox: &Receiver<SpeechRequest>,
) {
    let mut engine = match open() {
        Ok(engine) => engine,
        Err(detail) => {
            tracing::error!(%detail, "text-to-speech is unavailable");
            while let Ok(request) = inbox.recv() {
                match request {
                    SpeechRequest::Speak { reply, .. } => {
                        // Never queued, so its one finish is the worker's.
                        if let Some(observer) = observer {
                            observer.speech_finished();
                        }
                        send_reply(
                            &reply,
                            SpeechReply::Failed {
                                detail: detail.clone(),
                            },
                        );
                    }
                    SpeechRequest::Stop { reply } => send_reply(&reply, Ok(())),
                }
            }
            return;
        }
    };
    while let Ok(request) = inbox.recv() {
        match request {
            SpeechRequest::Speak { text, reply } => {
                let answer = speak_accounted(&mut engine, observer, &text);
                send_reply(&reply, answer);
            }
            SpeechRequest::Stop { reply } => {
                let stopped = engine.stop();
                send_reply(&reply, stopped);
            }
        }
    }
    // The platform is gone. tts 0.26.3's `Tts::drop` removes the CALLBACKS entry before
    // `WinRt::drop` removes BACKEND_TO_MEDIA_PLAYER, and a MediaEnded in that gap panics inside a
    // WinRT delegate and aborts the process; an empty queue closes that gap, and no in-flight
    // utterance is left without its finish.
    if let Err(detail) = engine.stop() {
        tracing::warn!(%detail, "stopping speech before closing the engine failed");
    }
}

/// One Speak with its failure accounting (p2 §2.6). tts 0.26.3 `WinRt::speak` queues the utterance
/// before the fallible synthesis, so a failed speak can leave it queued:
/// - `is_speaking()` is `Ok(true)`: it is still queued, so `stop()` delivers its one finish through
///   the stop callback (with interrupt = true the queue holds only that utterance). An `Err` from
///   that `stop()` (only possible at `Pause`, after the callback fired and the queue was cleared)
///   is added to the detail and changes no count;
/// - `Ok(false)`: it was never queued (the inner `stop()` failed before the push), so the worker
///   delivers its one finish itself;
/// - `Err` (impossible on WinRT, which reads its queue) is treated as `Ok(true)`: `stop()` finishes
///   only a queued utterance, so the worst case is a longer suppression, never a second finish.
fn speak_accounted<E: SpeechEngine>(
    engine: &mut E,
    observer: Option<&dyn OutputObserver>,
    text: &str,
) -> SpeechReply {
    let Err(mut detail) = engine.speak(text) else {
        return SpeechReply::Started;
    };
    let queued = match engine.is_speaking() {
        Ok(queued) => queued,
        Err(e) => {
            detail.push_str(&format!("; is_speaking failed: {e}"));
            true
        }
    };
    if queued {
        if let Err(e) = engine.stop() {
            detail.push_str(&format!("; stop failed: {e}"));
        }
    } else if let Some(observer) = observer {
        observer.speech_finished();
    }
    SpeechReply::Failed { detail }
}

/// Replies on a bounded(1) channel that carries exactly one reply.
fn send_reply<T>(reply: &Sender<T>, value: T) {
    match reply.try_send(value) {
        Ok(()) => {}
        // The platform stopped waiting (it already reported SpeechTimeout or gave up the stop).
        Err(TrySendError::Disconnected(_)) => {
            tracing::debug!("a speech reply arrived after its wait ended");
        }
        Err(TrySendError::Full(_)) => {
            tracing::error!("a speech reply channel already held its one reply");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Mutex;
    use std::time::Instant;

    use qwerty_core::veto::OutputId;
    use windows::Win32::System::Registry::{
        RegCreateKeyExW, RegDeleteTreeW, KEY_WRITE, REG_OPTION_NON_VOLATILE,
    };

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Call {
        Started(SelfOutputKind),
        Ended(OutputId),
        SpeechFinished,
    }

    /// Records every observer call with its time.
    #[derive(Default)]
    struct FakeObserver {
        calls: Mutex<Vec<(Call, Instant)>>,
        next_id: AtomicU64,
    }

    impl FakeObserver {
        fn record(&self, call: Call) {
            self.calls
                .lock()
                .expect("observer lock")
                .push((call, Instant::now()));
        }

        fn calls(&self) -> Vec<Call> {
            self.timed_calls().into_iter().map(|(c, _)| c).collect()
        }

        fn timed_calls(&self) -> Vec<(Call, Instant)> {
            self.calls.lock().expect("observer lock").clone()
        }

        fn finishes(&self) -> usize {
            self.calls()
                .iter()
                .filter(|c| **c == Call::SpeechFinished)
                .count()
        }
    }

    impl OutputObserver for FakeObserver {
        fn output_started(&self, kind: SelfOutputKind) -> OutputId {
            let id = OutputId(self.next_id.fetch_add(1, Ordering::Relaxed));
            self.record(Call::Started(kind));
            id
        }

        fn output_ended(&self, id: OutputId) {
            self.record(Call::Ended(id));
        }

        fn speech_finished(&self) {
            self.record(Call::SpeechFinished);
        }
    }

    /// Exactly one finish follows each `Began{Speech}`, matched in order (the processing FIFO).
    fn assert_each_speech_finished_once(calls: &[Call]) {
        let mut open = 0usize;
        for call in calls {
            match call {
                Call::Started(SelfOutputKind::Speech) => open += 1,
                Call::SpeechFinished => {
                    assert!(open > 0, "a finish with no open speech: {calls:?}");
                    open -= 1;
                }
                Call::Started(SelfOutputKind::Sound | SelfOutputKind::Toast) | Call::Ended(_) => {}
            }
        }
        assert_eq!(open, 0, "a speech without its finish: {calls:?}");
    }

    /// Behaves like tts 0.26.3's WinRT backend (winrt.rs 161-233): `speak` stops what is queued
    /// (interrupt = true), queues, then synthesizes, which can fail after the push; `stop` fires
    /// the stop callback once per queued entry and clears the queue, then pauses, which can fail.
    struct FakeTtsState {
        queue: VecDeque<String>,
        observer: Arc<FakeObserver>,
        fail_next_synthesis: bool,
        fail_next_pause: bool,
    }

    impl FakeTtsState {
        fn new(observer: Arc<FakeObserver>) -> FakeTtsState {
            FakeTtsState {
                queue: VecDeque::new(),
                observer,
                fail_next_synthesis: false,
                fail_next_pause: false,
            }
        }

        fn stop(&mut self) -> Result<(), String> {
            if self.queue.is_empty() {
                return Ok(());
            }
            for _ in &self.queue {
                self.observer.speech_finished();
            }
            self.queue.clear();
            if std::mem::take(&mut self.fail_next_pause) {
                return Err("MediaPlayer.Pause failed".to_string());
            }
            Ok(())
        }

        /// MediaEnded: the head utterance ends and its end callback fires.
        fn media_ended(&mut self) {
            if self.queue.pop_front().is_some() {
                self.observer.speech_finished();
            }
        }
    }

    struct FakeTts(Arc<Mutex<FakeTtsState>>);

    impl SpeechEngine for FakeTts {
        fn speak(&mut self, text: &str) -> Result<(), String> {
            let mut state = self.0.lock().expect("fake tts lock");
            state.stop()?;
            let was_empty = state.queue.is_empty();
            state.queue.push_back(text.to_string());
            if was_empty && std::mem::take(&mut state.fail_next_synthesis) {
                return Err("SynthesizeTextToStreamAsync failed".to_string());
            }
            Ok(())
        }

        fn is_speaking(&self) -> Result<bool, String> {
            Ok(!self.0.lock().expect("fake tts lock").queue.is_empty())
        }

        fn stop(&mut self) -> Result<(), String> {
            self.0.lock().expect("fake tts lock").stop()
        }
    }

    fn fake_speech(observer: &Arc<FakeObserver>) -> (SpeechWorker, Arc<Mutex<FakeTtsState>>) {
        let state = Arc::new(Mutex::new(FakeTtsState::new(Arc::clone(observer))));
        let engine = Arc::clone(&state);
        let worker = SpeechWorker::spawn(
            move || Ok(FakeTts(engine)),
            Some(Arc::clone(observer) as Arc<dyn OutputObserver>),
        )
        .expect("speech thread");
        (worker, state)
    }

    #[test]
    fn dp9_speech_failure_accounting_finishes_each_utterance_once() {
        // (a) A speak that fails after queuing, then a successful speak whose end callback fires.
        let observer = Arc::new(FakeObserver::default());
        let (worker, state) = fake_speech(&observer);
        state.lock().expect("lock").fail_next_synthesis = true;
        let failed = request_speech(&worker, Some(observer.as_ref()), "one");
        assert!(
            matches!(&failed, Err(ActionError::Speech { detail }) if detail.contains("Synthesize")),
            "{failed:?}"
        );
        assert_eq!(
            request_speech(&worker, Some(observer.as_ref()), "two"),
            Ok(())
        );
        state.lock().expect("lock").media_ended();
        let calls = observer.calls();
        assert_eq!(
            calls,
            vec![
                Call::Started(SelfOutputKind::Speech),
                Call::SpeechFinished,
                Call::Started(SelfOutputKind::Speech),
                Call::SpeechFinished,
            ]
        );
        assert_each_speech_finished_once(&calls);

        // (b) A speak whose inner stop fails before queuing: the playing utterance is finished by
        // its stop callback, the new one by the worker.
        let observer = Arc::new(FakeObserver::default());
        let (worker, state) = fake_speech(&observer);
        assert_eq!(
            request_speech(&worker, Some(observer.as_ref()), "one"),
            Ok(())
        );
        state.lock().expect("lock").fail_next_pause = true;
        let failed = request_speech(&worker, Some(observer.as_ref()), "two");
        assert!(
            matches!(&failed, Err(ActionError::Speech { detail }) if detail.contains("Pause")),
            "{failed:?}"
        );
        // Nothing is queued any more: a stop or a late MediaEnded adds no finish.
        worker.stop();
        state.lock().expect("lock").media_ended();
        let calls = observer.calls();
        assert_eq!(
            calls,
            vec![
                Call::Started(SelfOutputKind::Speech),
                Call::Started(SelfOutputKind::Speech),
                Call::SpeechFinished,
                Call::SpeechFinished,
            ]
        );
        assert_each_speech_finished_once(&calls);
    }

    #[test]
    fn stop_speech_finishes_every_queued_utterance_and_dropping_the_worker_empties_the_queue() {
        let observer = Arc::new(FakeObserver::default());
        let (worker, state) = fake_speech(&observer);
        assert_eq!(
            request_speech(&worker, Some(observer.as_ref()), "one"),
            Ok(())
        );
        worker.stop();
        assert_eq!(observer.finishes(), 1);
        assert!(state.lock().expect("lock").queue.is_empty());

        assert_eq!(
            request_speech(&worker, Some(observer.as_ref()), "two"),
            Ok(())
        );
        drop(worker);
        let deadline = Instant::now() + Duration::from_secs(5);
        while observer.finishes() < 2 {
            assert!(
                Instant::now() < deadline,
                "the worker did not stop its queue"
            );
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_each_speech_finished_once(&observer.calls());
    }

    #[test]
    fn an_unavailable_speech_engine_still_finishes_each_utterance() {
        let observer = Arc::new(FakeObserver::default());
        let worker = SpeechWorker::spawn(
            || Err::<FakeTts, String>("no voices".to_string()),
            Some(Arc::clone(&observer) as Arc<dyn OutputObserver>),
        )
        .expect("speech thread");
        let failed = request_speech(&worker, Some(observer.as_ref()), "one");
        assert!(
            matches!(&failed, Err(ActionError::Speech { detail }) if detail.contains("no voices")),
            "{failed:?}"
        );
        worker.stop();
        assert_each_speech_finished_once(&observer.calls());
        assert_eq!(observer.finishes(), 1);
    }

    /// p35 R9: with no utterance callbacks, each Speak fails fast with exactly the spec's detail.
    #[test]
    fn a_backend_without_utterance_callbacks_fails_each_speak_with_the_r9_detail() {
        let observer = Arc::new(FakeObserver::default());
        let worker = SpeechWorker::spawn(
            || Err::<FakeTts, String>(CALLBACKS_UNSUPPORTED.to_string()),
            Some(Arc::clone(&observer) as Arc<dyn OutputObserver>),
        )
        .expect("speech thread");
        assert_eq!(
            request_speech(&worker, Some(observer.as_ref()), "one"),
            Err(ActionError::Speech {
                detail: "utterance callbacks unsupported by the speech backend".to_string()
            })
        );
        assert_each_speech_finished_once(&observer.calls());
    }

    #[test]
    fn the_run_value_is_quoted_and_carries_its_args() {
        let exe = Path::new(r"C:\Program Files\Qwerty\qwerty.exe");
        assert_eq!(
            run_command_line(exe, "--background"),
            r#""C:\Program Files\Qwerty\qwerty.exe" --background"#
        );
        assert_eq!(
            run_command_line(exe, ""),
            r#""C:\Program Files\Qwerty\qwerty.exe""#
        );
    }

    /// A test-local key, HKCU\Software\Qwerty-test-<unique>, deleted on drop. Never the real Run key.
    struct TestKey {
        path: HSTRING,
    }

    impl TestKey {
        fn create() -> TestKey {
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock after 1970")
                .as_nanos();
            let path = HSTRING::from(format!(
                r"Software\Qwerty-test-{}-{nanos}",
                std::process::id()
            ));
            let mut hkey = HKEY(std::ptr::null_mut());
            // SAFETY: `path` is NUL-terminated; `hkey` is a valid out pointer.
            let created = unsafe {
                RegCreateKeyExW(
                    HKEY_CURRENT_USER,
                    PCWSTR(path.as_ptr()),
                    None,
                    PCWSTR::null(),
                    REG_OPTION_NON_VOLATILE,
                    KEY_WRITE,
                    None,
                    &mut hkey,
                    None,
                )
            };
            assert_eq!(created, ERROR_SUCCESS, "create the test key");
            drop(OpenKey(hkey));
            TestKey { path }
        }

        fn subkey(&self) -> PCWSTR {
            PCWSTR(self.path.as_ptr())
        }

        fn value(&self, name: PCWSTR) -> Option<String> {
            read_reg_text(HKEY_CURRENT_USER, self.subkey(), name, RRF_RT_REG_SZ)
                .expect("read the test value")
                .map(|text| String::from_utf16(&text).expect("UTF-16"))
        }
    }

    impl Drop for TestKey {
        fn drop(&mut self) {
            // SAFETY: `path` is NUL-terminated; it names only this test's own key.
            let deleted = unsafe { RegDeleteTreeW(HKEY_CURRENT_USER, self.subkey()) };
            assert_eq!(deleted, ERROR_SUCCESS, "delete the test key");
        }
    }

    #[test]
    fn the_run_value_is_written_or_deleted_only_on_a_difference() {
        let key = TestKey::create();
        let value = w!("Qwerty");
        let launch = r#""C:\Program Files\Qwerty\qwerty.exe" --background"#;

        assert_eq!(
            reconcile_run_value(key.subkey(), value, None),
            Ok(RunKeyChange::Unchanged)
        );
        assert_eq!(
            reconcile_run_value(key.subkey(), value, Some(launch)),
            Ok(RunKeyChange::Written)
        );
        assert_eq!(key.value(value).as_deref(), Some(launch));
        assert_eq!(
            reconcile_run_value(key.subkey(), value, Some(launch)),
            Ok(RunKeyChange::Unchanged)
        );
        assert_eq!(
            reconcile_run_value(key.subkey(), value, Some(r#""C:\q.exe""#)),
            Ok(RunKeyChange::Written)
        );
        assert_eq!(key.value(value).as_deref(), Some(r#""C:\q.exe""#));
        assert_eq!(
            reconcile_run_value(key.subkey(), value, None),
            Ok(RunKeyChange::Deleted)
        );
        assert_eq!(key.value(value), None);
        assert_eq!(
            reconcile_run_value(key.subkey(), value, None),
            Ok(RunKeyChange::Unchanged)
        );

        let path = key.path.clone();
        drop(key);
        let mut hkey = HKEY(std::ptr::null_mut());
        // SAFETY: `path` is NUL-terminated; `hkey` is a valid out pointer.
        let reopened = unsafe {
            RegOpenKeyExW(
                HKEY_CURRENT_USER,
                PCWSTR(path.as_ptr()),
                None,
                KEY_QUERY_VALUE,
                &mut hkey,
            )
        };
        assert_eq!(reopened, ERROR_FILE_NOT_FOUND, "the test key is gone");
    }

    #[test]
    fn a_missing_key_reads_as_absent() {
        let missing = read_reg_text(
            HKEY_CURRENT_USER,
            w!(r"Software\Qwerty-test-never-created\.Current"),
            PCWSTR::null(),
            RRF_RT_REG_SZ | RRF_RT_REG_EXPAND_SZ | RRF_NOEXPAND,
        );
        assert_eq!(missing, Ok(None));
    }

    /// `%VAR%` expansion for a scheme's `.Current` path (test support only).
    fn expand_env(path: &str) -> String {
        let mut out = String::new();
        let mut rest = path;
        while let Some(start) = rest.find('%') {
            out.push_str(&rest[..start]);
            let after = &rest[start + 1..];
            let Some(end) = after.find('%') else {
                out.push_str(&rest[start..]);
                return out;
            };
            let name = &after[..end];
            match std::env::var(name) {
                Ok(value) => out.push_str(&value),
                Err(_) => {
                    out.push('%');
                    out.push_str(name);
                    out.push('%');
                }
            }
            rest = &after[end + 1..];
        }
        out.push_str(rest);
        out
    }

    /// A RIFF/WAVE file's playing time from its `fmt ` byte rate and `data` size.
    fn wav_length(bytes: &[u8]) -> Duration {
        assert!(bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WAVE");
        let mut at = 12;
        let mut byte_rate = None;
        let mut data = None;
        while at + 8 <= bytes.len() {
            let id = &bytes[at..at + 4];
            let size =
                u32::from_le_bytes([bytes[at + 4], bytes[at + 5], bytes[at + 6], bytes[at + 7]])
                    as usize;
            let body = at + 8;
            if id == b"fmt " {
                byte_rate = Some(u32::from_le_bytes([
                    bytes[body + 8],
                    bytes[body + 9],
                    bytes[body + 10],
                    bytes[body + 11],
                ]));
            } else if id == b"data" {
                data = Some(size);
            }
            at = body + size + (size & 1);
        }
        let byte_rate = byte_rate.expect("a fmt chunk");
        let data = data.expect("a data chunk");
        Duration::from_secs_f64(data as f64 / f64::from(byte_rate))
    }

    /// DP3 (hardware evidence, audible): each of the four aliases plays synchronously between Began
    /// and Ended. Run by the user at the desk, never by an agent:
    /// `cargo test -p qwerty-app --lib dp3_ -- --ignored --nocapture`.
    #[test]
    #[ignore = "audible hardware evidence: plays the four system sounds"]
    fn dp3_playsound_spans_the_synchronous_sound() {
        for sound in [
            SystemSound::Default,
            SystemSound::Success,
            SystemSound::Error,
            SystemSound::Notification,
        ] {
            let (event, _, alias) = sound_event(sound);
            let observer = FakeObserver::default();
            let played =
                play_system_sound(Some(&observer), sound).expect("PlaySound returned TRUE");
            let calls = observer.timed_calls();
            assert_eq!(calls.len(), 2, "{alias}: Began then Ended");
            assert_eq!(calls[0].0, Call::Started(SelfOutputKind::Sound));
            assert_eq!(calls[1].0, Call::Ended(OutputId(0)));
            let span = calls[1].1 - calls[0].1;
            match played {
                SoundPlay::Played => {
                    let subkey =
                        HSTRING::from(format!(r"AppEvents\Schemes\Apps\.Default\{event}\.Current"));
                    let file = read_reg_text(
                        HKEY_CURRENT_USER,
                        PCWSTR(subkey.as_ptr()),
                        PCWSTR::null(),
                        RRF_RT_REG_SZ | RRF_RT_REG_EXPAND_SZ | RRF_NOEXPAND,
                    )
                    .expect("scheme read")
                    .expect("a scheme file");
                    let path = expand_env(&String::from_utf16(&file).expect("UTF-16"));
                    let length = wav_length(&std::fs::read(&path).expect("the scheme's WAV"));
                    println!("{alias}: PlaySoundW TRUE; Ended - Began = {span:?}; sound = {length:?} ({path})");
                    assert!(
                        span + Duration::from_millis(10) >= length,
                        "{alias}: {span:?} < {length:?} - 10 ms"
                    );
                }
                SoundPlay::SilencedByScheme => {
                    println!(
                        "{alias}: silenced by the sound scheme: Done without PlaySoundW ({span:?})"
                    );
                }
            }
        }
    }

    /// DP4 (hardware evidence, audible, R9): a Speak reports its start, then exactly one finish from
    /// the tts end callback; a second Speak during the first gives two finishes in order. Run by
    /// the user at the desk: `cargo test -p qwerty-app --lib dp4_ -- --ignored --nocapture`.
    #[test]
    #[ignore = "audible hardware evidence: speaks through the real tts backend"]
    fn dp4_speak_reports_start_then_one_finish_per_utterance() {
        let observer = Arc::new(FakeObserver::default());
        let platform = WindowsPlatform::new();
        platform.set_output_observer(Arc::clone(&observer) as Arc<dyn OutputObserver>);

        let wait_for = |finishes: usize| {
            let deadline = Instant::now() + Duration::from_secs(20);
            while observer.finishes() < finishes {
                assert!(Instant::now() < deadline, "waiting for {finishes} finishes");
                std::thread::sleep(Duration::from_millis(10));
            }
        };

        let began = Instant::now();
        platform.speak("Qwerty speech check.").expect("start reply");
        println!("start reply after {:?}", began.elapsed());
        wait_for(1);
        std::thread::sleep(Duration::from_secs(1));
        assert_eq!(observer.finishes(), 1, "exactly one finish");

        platform
            .speak("This is the first of two sentences, long enough to be interrupted.")
            .expect("start reply");
        std::thread::sleep(Duration::from_millis(500));
        platform.speak("Second.").expect("start reply");
        wait_for(3);
        std::thread::sleep(Duration::from_secs(1));
        let calls = observer.calls();
        println!("{calls:?}");
        assert_eq!(observer.finishes(), 3);
        assert_each_speech_finished_once(&calls);
        platform.stop_speech();
    }

    /// Evidence (R9): the installed speech backend reports utterance callbacks. It creates the
    /// engine and queries it; it speaks nothing.
    #[test]
    #[ignore = "evidence (R9): queries the installed speech backend"]
    fn tts_backend_reports_utterance_callbacks() {
        let tts = Tts::default().expect("the speech backend");
        let features = tts.supported_features();
        println!("tts 0.26 supported_features: {features:?}");
        assert!(features.utterance_callbacks);
    }
}
