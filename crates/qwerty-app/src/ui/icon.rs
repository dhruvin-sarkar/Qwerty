//! The bundled application icon (a keycap), decoded once into raw RGBA and
//! reused everywhere the app presents an icon: the window/taskbar icon (eframe
//! viewport) and the system-tray icon.
//!
//! The PNG is embedded with `include_bytes!`, so there is no runtime file
//! dependency and nothing to install alongside the exe. This is the one
//! committed binary asset — a legitimate product icon, not test data — added at
//! the user's request to brand the app.
//!
//! The tray icon composites the keycap with a small state-colored status dot
//! (listening/paused/error), so the tray still "reflects state at a glance"
//! (`DESIGN.md`) while using the product mark.

use std::sync::LazyLock;

use crate::app_state::ListeningState;

/// The keycap icon, embedded at build time. Known to be 360×360 8-bit RGBA.
const ICON_PNG: &[u8] = include_bytes!("../../assets/icon.png");

/// Tray icons are rendered small; 32×32 is the standard notification-area size.
const TRAY: u32 = 32;

/// Decoded once into `(rgba, width, height)`. A bundled, known-good asset, so a
/// decode failure is a build/programmer error and is surfaced loudly — the same
/// "programmer error, not runtime failure" policy the feature/tray code uses.
static DECODED: LazyLock<(Vec<u8>, u32, u32)> = LazyLock::new(|| {
    // png 0.18's Decoder needs BufRead + Seek; a Cursor over the embedded bytes
    // provides both without touching the filesystem.
    let mut reader = png::Decoder::new(std::io::Cursor::new(ICON_PNG))
        .read_info()
        .expect("bundled icon.png is a valid PNG");
    let size = reader
        .output_buffer_size()
        .expect("bundled icon.png has known dimensions");
    let mut buf = vec![0u8; size];
    let info = reader.next_frame(&mut buf).expect("decode bundled icon.png");
    assert!(
        info.color_type == png::ColorType::Rgba && info.bit_depth == png::BitDepth::Eight,
        "bundled icon.png must be 8-bit RGBA",
    );
    buf.truncate(info.buffer_size());
    (buf, info.width, info.height)
});

/// The keycap box-downsampled to [`TRAY`]×[`TRAY`], decoded/scaled once.
static TRAY_BASE: LazyLock<Vec<u8>> = LazyLock::new(|| {
    let (rgba, w, h) = &*DECODED;
    downscale_rgba(rgba, *w, *h, TRAY, TRAY)
});

/// The window/taskbar icon (full-resolution keycap). Wrapped by the shell into
/// the eframe viewport at startup.
pub fn app_icon() -> egui::IconData {
    let (rgba, w, h) = &*DECODED;
    egui::IconData {
        rgba: rgba.clone(),
        width: *w,
        height: *h,
    }
}

/// The tray icon RGBA for a listening `state`: the keycap plus a state-colored
/// status dot in the lower-right corner. Returns `(rgba, width, height)`.
pub fn tray_rgba(state: ListeningState) -> (Vec<u8>, u32, u32) {
    let mut rgba = TRAY_BASE.clone();
    let (r, g, b) = match state {
        ListeningState::Listening => (0x35, 0xC4, 0x6B), // success green
        ListeningState::Paused => (0x9A, 0xA6, 0xB2),    // neutral gray
        ListeningState::Error => (0xF0, 0x6A, 0x6A),     // danger red
    };
    stamp_status_dot(&mut rgba, TRAY, r, g, b);
    (rgba, TRAY, TRAY)
}

/// Box-average downscale of a straight-alpha RGBA image. Each destination pixel
/// averages the source pixels in its footprint — adequate quality for a small
/// icon and dependency-free (no image-resize crate needed).
fn downscale_rgba(src: &[u8], sw: u32, sh: u32, dw: u32, dh: u32) -> Vec<u8> {
    let mut out = vec![0u8; (dw * dh * 4) as usize];
    for dy in 0..dh {
        let y0 = dy * sh / dh;
        let y1 = ((dy + 1) * sh / dh).max(y0 + 1).min(sh);
        for dx in 0..dw {
            let x0 = dx * sw / dw;
            let x1 = ((dx + 1) * sw / dw).max(x0 + 1).min(sw);
            let (mut sr, mut sg, mut sb, mut sa, mut n) = (0u32, 0u32, 0u32, 0u32, 0u32);
            for sy in y0..y1 {
                for sx in x0..x1 {
                    let i = ((sy * sw + sx) * 4) as usize;
                    sr += src[i] as u32;
                    sg += src[i + 1] as u32;
                    sb += src[i + 2] as u32;
                    sa += src[i + 3] as u32;
                    n += 1;
                }
            }
            let n = n.max(1);
            let o = ((dy * dw + dx) * 4) as usize;
            out[o] = (sr / n) as u8;
            out[o + 1] = (sg / n) as u8;
            out[o + 2] = (sb / n) as u8;
            out[o + 3] = (sa / n) as u8;
        }
    }
    out
}

/// Draw a filled status disc (with a thin dark outline for contrast on any
/// background) into the lower-right corner of a `size`×`size` RGBA buffer.
fn stamp_status_dot(rgba: &mut [u8], size: u32, r: u8, g: u8, b: u8) {
    let radius = 7.0f32;
    let cx = size as f32 - radius - 1.0;
    let cy = size as f32 - radius - 1.0;
    for y in 0..size {
        for x in 0..size {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let d = (dx * dx + dy * dy).sqrt();
            let i = ((y * size + x) * 4) as usize;
            if d <= radius {
                rgba[i] = r;
                rgba[i + 1] = g;
                rgba[i + 2] = b;
                rgba[i + 3] = 255;
            } else if d <= radius + 1.3 {
                // A thin dark ring so the dot reads on a light OR dark taskbar.
                rgba[i] = 20;
                rgba[i + 1] = 24;
                rgba[i + 2] = 28;
                rgba[i + 3] = 255;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundled_icon_decodes_to_rgba() {
        let (rgba, w, h) = &*DECODED;
        assert!(*w > 0 && *h > 0);
        assert_eq!(rgba.len(), (w * h * 4) as usize);
    }

    #[test]
    fn tray_icon_is_correct_size_for_each_state() {
        for state in [
            ListeningState::Listening,
            ListeningState::Paused,
            ListeningState::Error,
        ] {
            let (rgba, w, h) = tray_rgba(state);
            assert_eq!((w, h), (TRAY, TRAY));
            assert_eq!(rgba.len(), (TRAY * TRAY * 4) as usize);
        }
    }

    #[test]
    fn status_dot_paints_its_state_color_at_its_center() {
        // The status dot is centered at (size - radius - 1) in both axes; that
        // pixel must carry the state color (listening = green), confirming the
        // dot lands in the lower-right for DESIGN.md's at-a-glance state.
        let (rgba, w, _h) = tray_rgba(ListeningState::Listening);
        let (cx, cy) = (w - 8, w - 8); // radius 7 + 1px inset from the edge
        let i = ((cy * w + cx) * 4) as usize;
        assert_eq!(
            (rgba[i], rgba[i + 1], rgba[i + 2]),
            (0x35, 0xC4, 0x6B),
            "status dot center should be the listening color",
        );
    }
}
