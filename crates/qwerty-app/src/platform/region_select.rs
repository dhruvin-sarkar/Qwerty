//! Interactive region selector for the "region-select screenshot" action.
//!
//! Presents a translucent, top-most overlay window spanning the whole virtual
//! screen and lets the user drag a rectangle; the chosen rectangle (in
//! virtual-screen pixel coordinates) is handed back to `windows.rs`, which
//! BitBlts exactly that sub-rectangle. Cancelling (Esc or right-click) returns
//! `None` — a deliberate no-capture, not an error.
//!
//! This is the only place in the app that spins its own Win32 message loop. It
//! is modal on the UI thread: the nested `GetMessageW` pump runs until the
//! overlay is destroyed. The real-time audio thread is unaffected (it lives on
//! its own thread); only the egui UI pauses while the user selects, which is
//! the expected behaviour for a modal capture.
//!
//! Every Win32 signature here was verified against the installed
//! `windows-0.62.2` source, matching `windows.rs`'s standard.

use std::cell::Cell;

use windows::core::{w, PCWSTR};
use windows::Win32::Foundation::{
    GetLastError, COLORREF, HINSTANCE, HWND, LPARAM, LRESULT, RECT, WPARAM,
};
use windows::Win32::Graphics::Gdi::{
    BeginPaint, CreatePen, CreateSolidBrush, DeleteObject, EndPaint, FillRect, GetStockObject,
    InvalidateRect, Rectangle, SelectObject, NULL_BRUSH, PAINTSTRUCT, PS_SOLID,
};
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::UI::Input::KeyboardAndMouse::{ReleaseCapture, SetCapture};
use windows::Win32::UI::WindowsAndMessaging::{
    CreateWindowExW, DefWindowProcW, DestroyWindow, DispatchMessageW, GetClientRect, GetMessageW,
    LoadCursorW, PostQuitMessage, RegisterClassW, SetForegroundWindow, SetLayeredWindowAttributes,
    ShowWindow, TranslateMessage, IDC_CROSS, LWA_ALPHA, MSG, SW_SHOW, WM_DESTROY, WM_KEYDOWN,
    WM_LBUTTONDOWN, WM_LBUTTONUP, WM_MOUSEMOVE, WM_PAINT, WM_RBUTTONDOWN, WNDCLASSW, WS_EX_LAYERED,
    WS_EX_TOOLWINDOW, WS_EX_TOPMOST, WS_POPUP,
};

/// Virtual-key code for Esc (avoids pulling in the whole `VIRTUAL_KEY` set for
/// one constant; `WPARAM` carries the code as a `usize`).
const VK_ESCAPE: usize = 0x1B;

/// `RegisterClassW` sets this in `GetLastError` when the class is already
/// registered — expected on the second and later invocations, since the class
/// persists (we `DestroyWindow`, never `UnregisterClass`).
const ERROR_CLASS_ALREADY_EXISTS: u32 = 1410;

/// Drag state shared with the window procedure. `WndProc` is a bare
/// `extern "system"` function with no environment, so the state lives in a
/// thread-local `Cell` — sound because the whole capture is single-threaded and
/// modal. All coordinates are window-client pixels.
#[derive(Clone, Copy)]
struct DragState {
    /// The left mouse button is currently held and being dragged.
    dragging: bool,
    /// A drag has started (so a selection rectangle should be drawn).
    started: bool,
    start: (i32, i32),
    cur: (i32, i32),
    /// The drag finished with a button-up — a rectangle is available.
    committed: bool,
    /// The user pressed Esc or right-clicked — no capture.
    cancelled: bool,
}

impl DragState {
    const fn new() -> Self {
        DragState {
            dragging: false,
            started: false,
            start: (0, 0),
            cur: (0, 0),
            committed: false,
            cancelled: false,
        }
    }
}

thread_local! {
    static DRAG: Cell<DragState> = const { Cell::new(DragState::new()) };
}

/// Compose a `COLORREF` from RGB bytes. `COLORREF` is `0x00BBGGRR`, so red is
/// the low byte — the reverse of a hex literal, hence this helper.
const fn rgb(r: u8, g: u8, b: u8) -> COLORREF {
    COLORREF((r as u32) | ((g as u32) << 8) | ((b as u32) << 16))
}

/// Unpack the signed x/y a mouse message packs into `LPARAM` (low word = x,
/// high word = y). The `as i16` step preserves negative coordinates on
/// multi-monitor setups where a monitor sits left of / above the primary.
fn lparam_xy(lp: LPARAM) -> (i32, i32) {
    let x = (lp.0 & 0xFFFF) as i16 as i32;
    let y = ((lp.0 >> 16) & 0xFFFF) as i16 as i32;
    (x, y)
}

/// Turn two dragged corner points (virtual-screen pixels) into a top-left-origin
/// rectangle `(x, y, w, h)`, clamped to `clip = (vx, vy, vw, vh)` and rejected
/// (returns `None`) if it collapses to no area — i.e. a click with no drag.
///
/// Pure: the whole point of splitting it out is that the geometry is testable
/// without a display.
fn normalize_selection(
    a: (i32, i32),
    b: (i32, i32),
    clip: (i32, i32, i32, i32),
) -> Option<(i32, i32, i32, i32)> {
    let (vx, vy, vw, vh) = clip;
    let left = a.0.min(b.0).max(vx);
    let top = a.1.min(b.1).max(vy);
    let right = a.0.max(b.0).min(vx + vw);
    let bottom = a.1.max(b.1).min(vy + vh);
    let w = right - left;
    let h = bottom - top;
    if w >= 1 && h >= 1 {
        Some((left, top, w, h))
    } else {
        None
    }
}

/// Paint the translucent scrim plus the current selection outline.
unsafe fn paint(hwnd: HWND) {
    // PAINTSTRUCT does not derive Default in this crate; it is plain POD, so a
    // zeroed value is a valid starting point for BeginPaint to fill.
    let mut ps: PAINTSTRUCT = std::mem::zeroed();
    let hdc = BeginPaint(hwnd, &mut ps);

    let mut client = RECT::default();
    let _ = GetClientRect(hwnd, &mut client);

    // Uniform dark fill; the window's global layered alpha turns it into a
    // translucent scrim over the live desktop.
    let scrim = CreateSolidBrush(rgb(14, 17, 22));
    let _ = FillRect(hdc, &client, scrim);
    let _ = DeleteObject(scrim.into());

    let s = DRAG.with(|d| d.get());
    if s.started {
        let l = s.start.0.min(s.cur.0);
        let t = s.start.1.min(s.cur.1);
        let r = s.start.0.max(s.cur.0);
        let b = s.start.1.max(s.cur.1);
        // A bright amber outline; NULL_BRUSH so the interior isn't filled (it
        // stays scrim, since a per-pixel "cut-out" would need a full layered
        // DIB — deliberately kept simple, a visible outline is enough).
        let pen = CreatePen(PS_SOLID, 2, rgb(230, 190, 90));
        let old_pen = SelectObject(hdc, pen.into());
        let old_brush = SelectObject(hdc, GetStockObject(NULL_BRUSH));
        let _ = Rectangle(hdc, l, t, r, b);
        SelectObject(hdc, old_pen);
        SelectObject(hdc, old_brush);
        let _ = DeleteObject(pen.into());
    }

    let _ = EndPaint(hwnd, &ps);
}

/// The overlay's window procedure. Translates raw input into `DragState`
/// transitions and tears the window down on completion/cancel.
unsafe extern "system" fn wndproc(hwnd: HWND, msg: u32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
    match msg {
        WM_LBUTTONDOWN => {
            let p = lparam_xy(lparam);
            DRAG.with(|d| {
                let mut s = d.get();
                s.dragging = true;
                s.started = true;
                s.start = p;
                s.cur = p;
                d.set(s);
            });
            let _ = SetCapture(hwnd);
            let _ = InvalidateRect(Some(hwnd), None, false);
            LRESULT(0)
        }
        WM_MOUSEMOVE => {
            let p = lparam_xy(lparam);
            let redraw = DRAG.with(|d| {
                let mut s = d.get();
                if s.dragging {
                    s.cur = p;
                    d.set(s);
                    true
                } else {
                    false
                }
            });
            if redraw {
                let _ = InvalidateRect(Some(hwnd), None, false);
            }
            LRESULT(0)
        }
        WM_LBUTTONUP => {
            let p = lparam_xy(lparam);
            DRAG.with(|d| {
                let mut s = d.get();
                if s.dragging {
                    s.dragging = false;
                    s.cur = p;
                    s.committed = true;
                    d.set(s);
                }
            });
            let _ = ReleaseCapture();
            let _ = DestroyWindow(hwnd);
            LRESULT(0)
        }
        WM_RBUTTONDOWN => {
            DRAG.with(|d| {
                let mut s = d.get();
                s.cancelled = true;
                d.set(s);
            });
            let _ = DestroyWindow(hwnd);
            LRESULT(0)
        }
        WM_KEYDOWN if wparam.0 == VK_ESCAPE => {
            DRAG.with(|d| {
                let mut s = d.get();
                s.cancelled = true;
                d.set(s);
            });
            let _ = DestroyWindow(hwnd);
            LRESULT(0)
        }
        WM_PAINT => {
            paint(hwnd);
            LRESULT(0)
        }
        WM_DESTROY => {
            PostQuitMessage(0);
            LRESULT(0)
        }
        _ => DefWindowProcW(hwnd, msg, wparam, lparam),
    }
}

/// Show the selection overlay across the virtual screen `(vx, vy, vw, vh)` and
/// block until the user commits or cancels. Returns the chosen rectangle in
/// virtual-screen pixels, or `None` if cancelled or the drag had no area.
///
/// On a Win32 setup failure this logs and returns `None` (the action then does
/// nothing rather than crashing) — the honest degrade-with-a-warning pattern
/// this module family uses for non-fatal OS hiccups.
pub(crate) fn select_screen_region(
    vx: i32,
    vy: i32,
    vw: i32,
    vh: i32,
) -> Option<(i32, i32, i32, i32)> {
    // Fresh state for this capture (the class/thread-local persist across calls).
    DRAG.with(|d| d.set(DragState::new()));

    unsafe {
        let hinstance: HINSTANCE = GetModuleHandleW(PCWSTR::null()).ok()?.into();
        let class_name = w!("QwertyRegionSelect");

        let wc = WNDCLASSW {
            lpfnWndProc: Some(wndproc),
            hInstance: hinstance,
            lpszClassName: class_name,
            // Crosshair cursor signals "drag to select"; default (null) if the
            // system cursor can't be loaded — a missing cursor is cosmetic.
            hCursor: LoadCursorW(None, IDC_CROSS).unwrap_or_default(),
            ..Default::default()
        };

        // Registering an already-registered class fails with a specific code;
        // that is expected on repeat invocations and fine to proceed past.
        if RegisterClassW(&wc) == 0 {
            let err = GetLastError();
            if err.0 != ERROR_CLASS_ALREADY_EXISTS {
                eprintln!("warning: region-select could not register its window class: {err:?}");
                return None;
            }
        }

        let hwnd = CreateWindowExW(
            WS_EX_LAYERED | WS_EX_TOPMOST | WS_EX_TOOLWINDOW,
            class_name,
            w!(""),
            WS_POPUP,
            vx,
            vy,
            vw,
            vh,
            None,
            None,
            Some(hinstance),
            None,
        )
        .ok()?;

        // Uniform translucent scrim (~43% opaque) so the desktop shows through
        // while the overlay is clearly an overlay. crkey is ignored for LWA_ALPHA.
        let _ = SetLayeredWindowAttributes(hwnd, COLORREF(0), 110, LWA_ALPHA);
        let _ = ShowWindow(hwnd, SW_SHOW);
        let _ = SetForegroundWindow(hwnd);

        // Modal pump: DestroyWindow -> WM_DESTROY -> PostQuitMessage(0) makes
        // GetMessageW return 0 and ends the loop.
        let mut msg = MSG::default();
        while GetMessageW(&mut msg, None, 0, 0).as_bool() {
            let _ = TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
    }

    let s = DRAG.with(|d| d.get());
    if s.cancelled || !s.committed {
        return None;
    }
    // Client coords -> virtual-screen coords (the window's origin is (vx, vy)).
    let a = (vx + s.start.0, vy + s.start.1);
    let b = (vx + s.cur.0, vy + s.cur.1);
    normalize_selection(a, b, (vx, vy, vw, vh))
}

#[cfg(test)]
mod tests {
    use super::normalize_selection;

    const CLIP: (i32, i32, i32, i32) = (0, 0, 1920, 1080);

    #[test]
    fn orders_corners_into_top_left_origin_rect() {
        // Dragged bottom-right -> top-left still yields the same normalized rect.
        let a = normalize_selection((100, 200), (400, 500), CLIP);
        let b = normalize_selection((400, 500), (100, 200), CLIP);
        assert_eq!(a, Some((100, 200, 300, 300)));
        assert_eq!(a, b);
    }

    #[test]
    fn a_click_without_a_drag_is_rejected() {
        assert_eq!(normalize_selection((300, 300), (300, 300), CLIP), None);
    }

    #[test]
    fn clamps_to_the_virtual_screen() {
        // A drag that spills past both edges is clamped to the clip rectangle.
        let r = normalize_selection((-50, -80), (5000, 5000), CLIP);
        assert_eq!(r, Some((0, 0, 1920, 1080)));
    }

    #[test]
    fn honours_a_non_zero_virtual_origin() {
        // Multi-monitor: the virtual screen can start at a negative origin.
        let clip = (-1920, 0, 3840, 1080);
        let r = normalize_selection((-1900, 100), (-1500, 400), clip);
        assert_eq!(r, Some((-1900, 100, 400, 300)));
    }

    #[test]
    fn a_selection_fully_outside_the_clip_has_no_area() {
        // Both corners past the right edge -> clamped width collapses to 0.
        assert_eq!(normalize_selection((3000, 100), (4000, 400), CLIP), None);
    }
}
