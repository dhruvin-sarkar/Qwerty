//! Windows implementation of [`PlatformActions`] (Phase 5).
//!
//! Every Win32 signature here was verified against the installed
//! `windows-0.62.2` source. This is the only OS-touching module; `qwerty-core`
//! stays platform-free. Construction is infallible (it degrades with a warning
//! like the tray/hotkey setup) and everything runs on the UI thread, so the
//! `!Send` `Tts` engine in a `RefCell` is fine.

use std::cell::RefCell;
use std::os::windows::process::CommandExt;
use std::panic::AssertUnwindSafe;
use std::path::Path;
use std::process::Command;

use arboard::{Clipboard, ImageData};
use tts::Tts;
use winrt_toast::{register, Toast, ToastManager};
use windows::core::{w, HSTRING, PCWSTR};
use windows::Win32::Foundation::GetLastError;
use windows::Win32::Graphics::Gdi::{
    BitBlt, CreateCompatibleBitmap, CreateCompatibleDC, DeleteDC, DeleteObject, GetDC, GetDIBits,
    ReleaseDC, SelectObject, BITMAPINFO, BITMAPINFOHEADER, DIB_RGB_COLORS, SRCCOPY,
};
use windows::Win32::System::Diagnostics::Debug::MessageBeep;
use windows::Win32::UI::Input::KeyboardAndMouse::{
    SendInput, INPUT, INPUT_0, INPUT_KEYBOARD, KEYBDINPUT, KEYBD_EVENT_FLAGS, KEYEVENTF_EXTENDEDKEY,
    KEYEVENTF_KEYUP, VIRTUAL_KEY,
};
use windows::Win32::UI::Shell::ShellExecuteW;
use windows::Win32::UI::WindowsAndMessaging::{
    GetSystemMetrics, MB_ICONASTERISK, MB_ICONEXCLAMATION, MB_ICONHAND, MB_OK, SM_CXVIRTUALSCREEN,
    SM_CYVIRTUALSCREEN, SM_XVIRTUALSCREEN, SM_YVIRTUALSCREEN, SW_SHOWNORMAL,
};
use qwerty_core::profile::{KeyCombo, Modifier, ScreenshotMode, SystemSound};

use super::{ActionError, PlatformActions, ToastSpec};

/// Run a launched command without flashing a console window.
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

/// The Application User Model ID this app registers itself under so Windows
/// will display its toasts. Windows only shows a notification for a *registered*
/// AUMID; see [`WindowsPlatform::notifier`].
const AUM_ID: &str = "Qwerty.TapZones";

/// The Windows action backend.
///
/// It holds two lazily-created, then long-lived, handles. Both live behind a
/// `RefCell` because [`PlatformActions`] takes `&self` while these need `&mut`
/// — sound here because every action is dispatched from the single UI thread
/// (`ARCHITECTURE.md` → Threading model).
pub struct WindowsPlatform {
    /// The speech engine. It must be *kept alive*, not created per call:
    /// dropping it releases the WinRT `MediaPlayer` that is still playing, which
    /// cuts the utterance off mid-word.
    tts: RefCell<Option<Tts>>,
    /// The toast notifier, created once the AUMID has been registered.
    toast: RefCell<Option<ToastManager>>,
}

impl Default for WindowsPlatform {
    fn default() -> Self {
        Self::new()
    }
}

impl WindowsPlatform {
    pub fn new() -> Self {
        WindowsPlatform {
            tts: RefCell::new(None),
            toast: RefCell::new(None),
        }
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
                    return Err(ActionError::Notification(format!(
                        "could not register the notification identity ({AUM_ID}): {e}"
                    )))
                }
                Err(_) => {
                    return Err(ActionError::Notification(format!(
                        "registering the notification identity ({AUM_ID}) panicked \
                         inside winrt-toast"
                    )))
                }
            }
            *slot = Some(ToastManager::new(AUM_ID));
        }
        Ok(slot.as_ref().expect("notifier initialized above").clone())
    }

    /// Open a file/folder/app/URL with the shell's default "open" verb.
    fn shell_open(target: &str) -> Result<(), ActionError> {
        let wide = HSTRING::from(target);
        let r = unsafe {
            ShellExecuteW(
                None,
                w!("open"),
                PCWSTR(wide.as_ptr()),
                PCWSTR::null(),
                PCWSTR::null(),
                SW_SHOWNORMAL,
            )
        };
        // ShellExecuteW returns an HINSTANCE whose value is > 32 on success and a
        // SE_ERR_* code otherwise (Win32 contract).
        let code = r.0 as isize;
        if code > 32 {
            return Ok(());
        }
        match code {
            2 | 3 => Err(ActionError::TargetNotFound(target.to_string())),
            other => Err(ActionError::LaunchFailed {
                target: target.to_string(),
                reason: format!("shell refused to open target (SE_ERR {other})"),
            }),
        }
    }
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
    if let Some(num) = n.strip_prefix(['F', 'f']).and_then(|d| d.parse::<u16>().ok()) {
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
        other => return Err(ActionError::Keystroke(format!("unrecognized key \"{other}\""))),
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
        let target = path
            .to_str()
            .ok_or_else(|| ActionError::TargetNotFound(path.display().to_string()))?;
        Self::shell_open(target)
    }

    fn run_command(&self, cmd: &str, args: &[String]) -> Result<(), ActionError> {
        match Command::new(cmd)
            .args(args)
            .creation_flags(CREATE_NO_WINDOW)
            .spawn()
        {
            Ok(_child) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                Err(ActionError::TargetNotFound(cmd.to_string()))
            }
            Err(e) => Err(ActionError::LaunchFailed {
                target: cmd.to_string(),
                reason: e.to_string(),
            }),
        }
    }

    fn copy_to_clipboard(&self, text: &str) -> Result<(), ActionError> {
        let mut cb = Clipboard::new().map_err(|e| ActionError::Clipboard(e.to_string()))?;
        cb.set_text(text)
            .map_err(|e| ActionError::Clipboard(e.to_string()))
    }

    fn play_sound(&self, sound: SystemSound) -> Result<(), ActionError> {
        let flag = match sound {
            SystemSound::Default => MB_OK,
            SystemSound::Success => MB_ICONASTERISK,
            SystemSound::Error => MB_ICONHAND,
            SystemSound::Notification => MB_ICONEXCLAMATION,
        };
        unsafe { MessageBeep(flag) }.map_err(|e| ActionError::Sound(e.to_string()))
    }

    fn send_keystroke(&self, combo: &KeyCombo) -> Result<(), ActionError> {
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
            return Err(ActionError::Keystroke(format!(
                "SendInput injected {sent}/{} (GetLastError {:#x})",
                inputs.len(),
                e.0
            )));
        }
        Ok(())
    }

    fn screenshot_to_clipboard(&self, mode: ScreenshotMode) -> Result<(), ActionError> {
        match mode {
            ScreenshotMode::RegionSelect => {
                return Err(ActionError::Unsupported("region-select screenshot"))
            }
            ScreenshotMode::FullScreen => {}
        }
        unsafe {
            let (x, y) = (
                GetSystemMetrics(SM_XVIRTUALSCREEN),
                GetSystemMetrics(SM_YVIRTUALSCREEN),
            );
            let (w, h) = (
                GetSystemMetrics(SM_CXVIRTUALSCREEN),
                GetSystemMetrics(SM_CYVIRTUALSCREEN),
            );
            if w <= 0 || h <= 0 {
                return Err(ActionError::Screenshot("virtual screen size is 0".into()));
            }
            let screen = GetDC(None);
            if screen.is_invalid() {
                return Err(ActionError::Screenshot("GetDC(NULL) returned null".into()));
            }

            let grab = || -> Result<(usize, usize, Vec<u8>), ActionError> {
                let mem = CreateCompatibleDC(Some(screen));
                let bmp = CreateCompatibleBitmap(screen, w, h);
                let old = SelectObject(mem, bmp.into());
                let blt = BitBlt(mem, 0, 0, w, h, Some(screen), x, y, SRCCOPY);
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
                    return Err(ActionError::Screenshot("BitBlt/GetDIBits failed".into()));
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
            let mut cb = Clipboard::new().map_err(|e| ActionError::Screenshot(e.to_string()))?;
            cb.set_image(ImageData {
                width: wi,
                height: hi,
                bytes: rgba.into(),
            })
            .map_err(|e| ActionError::Screenshot(e.to_string()))
        }
    }

    fn speak(&self, text: &str) -> Result<(), ActionError> {
        let mut slot = self.tts.borrow_mut();
        if slot.is_none() {
            *slot = Some(Tts::default().map_err(speech_err)?);
        }
        // `expect` on the line after the assignment above: genuinely unreachable.
        let engine = slot.as_mut().expect("speech engine initialized above");
        // interrupt = true: a newly tapped zone should speak now, not queue
        // behind whatever the previous tap said.
        engine.speak(text, true).map(|_utterance| ()).map_err(speech_err)
    }

    fn notify(&self, toast: &ToastSpec) -> Result<(), ActionError> {
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
            .map_err(|e| ActionError::Notification(e.to_string()))
    }
}

/// Map a `tts` error into ours using `Debug`, not `Display`, deliberately:
/// `tts::Error::WinRt`'s `Display` is the bare string "WinRT error" and drops
/// the underlying HRESULT, which would leave the user (and the detective method)
/// with an unactionable message.
fn speech_err(e: tts::Error) -> ActionError {
    ActionError::Speech(format!("{e:?}"))
}
