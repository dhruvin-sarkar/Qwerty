//! Custom `epaint` painting primitives — the CPU-side visual-quality lift.
//!
//! # Render-stack research (why these live here and not in a wgpu shader)
//!
//! The "visual ceiling" for this app was scoped against egui 0.31's real
//! capability layers:
//!
//! - **Layer 1 — `epaint::Painter` / `Mesh`** (this module). Colored-vertex
//!   meshes get free Gouraud interpolation on the GPU, so smooth gradients and
//!   radial glows cost only a handful of triangles and zero shader code. This
//!   is the highest-ROI layer and works on *any* backend.
//! - **Layer 2 — `TextureHandle`** (`ctx.load_texture` + `painter.image`).
//!   Also backend-agnostic; the path a live spectrogram frame buffer would use.
//! - **Layer 3 — `egui_wgpu::CallbackTrait` (raw WGSL)**. The theoretical
//!   ceiling (SDF tiles, backdrop blur, animated aurora field) **but it is
//!   unreachable here**: it needs `eframe`'s `wgpu` feature, and the workspace
//!   manifest documents that the wgpu backend fails to compile against the
//!   `windows` 0.62 crate that `tray-icon`/`global-hotkey` pin (a wgpu-hal
//!   conflict). glow is also the lighter backend for an always-on app
//!   (`PERFORMANCE.md`). Switching backends to chase shader effects would
//!   destabilise the tray/hotkey foundation, so Layers 1–2 are the real
//!   ceiling for Qwerty.
//!
//! API note: epaint 0.31's `Shadow` uses integer fields (`offset:[i8;2]`,
//! `blur:u8`, `spread:u8`) — not the f32/`Vec2` form seen in newer releases.
//!
//! Everything here is pure painting: it allocates nothing that outlives the
//! frame and requests no repaints, so it stays within `PERFORMANCE.md`'s
//! redraw discipline — the *caller* decides when a frame is drawn.

use egui::{Color32, Mesh, Painter, Pos2, Rect, Shape};

/// Paint a soft radial glow: a triangle fan from `center` (full `color`) out to
/// a ring at `radius` (same rgb, alpha 0), so the GPU interpolates the falloff
/// for free. `segments` sets the ring smoothness (16 is plenty for small radii,
/// 32 for large). Drawn in the caller's current draw order, so paint it *before*
/// the thing it should glow behind.
pub(crate) fn radial_glow(
    painter: &Painter,
    center: Pos2,
    radius: f32,
    color: Color32,
    segments: u32,
) {
    if radius <= 0.0 || color.a() == 0 || segments < 3 {
        return;
    }
    let edge = Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), 0);
    let mut mesh = Mesh::default();
    mesh.colored_vertex(center, color); // index 0 — the bright core
    for i in 0..segments {
        let a = (i as f32 / segments as f32) * std::f32::consts::TAU;
        mesh.colored_vertex(center + egui::vec2(a.cos(), a.sin()) * radius, edge);
    }
    for i in 0..segments {
        // center, this ring vertex, next ring vertex (wrapping).
        mesh.add_triangle(0, 1 + i, 1 + (i + 1) % segments);
    }
    painter.add(Shape::mesh(mesh));
}

/// Stack three [`radial_glow`] passes at 1.0×/1.6×/2.4× the base radius with
/// decreasing alpha, for a layered bloom that reads softer than a single ring.
/// `intensity` in `[0,1]` scales the whole thing (0 paints nothing), so a
/// caller can drive it straight from an `AnimState::t()`.
pub(crate) fn radial_bloom(
    painter: &Painter,
    center: Pos2,
    radius: f32,
    color: Color32,
    intensity: f32,
) {
    let k = intensity.clamp(0.0, 1.0);
    if k <= 0.0 {
        return;
    }
    let a = |frac: f32| {
        let alpha = (color.a() as f32 * frac * k) as u8;
        Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), alpha)
    };
    // Widest and faintest first so the tighter, brighter cores layer on top.
    radial_glow(painter, center, radius * 2.4, a(0.30), 40);
    radial_glow(painter, center, radius * 1.6, a(0.55), 32);
    radial_glow(painter, center, radius * 1.0, a(1.00), 28);
}

/// Fill `rect` with a vertical two-stop gradient (top → bottom) via a 4-vertex
/// colored mesh. Free GPU interpolation, no shader.
pub(crate) fn vertical_gradient(painter: &Painter, rect: Rect, top: Color32, bottom: Color32) {
    let mut mesh = Mesh::default();
    mesh.colored_vertex(rect.left_top(), top);
    mesh.colored_vertex(rect.right_top(), top);
    mesh.colored_vertex(rect.left_bottom(), bottom);
    mesh.colored_vertex(rect.right_bottom(), bottom);
    mesh.add_triangle(0, 1, 2);
    mesh.add_triangle(1, 3, 2);
    painter.add(Shape::mesh(mesh));
}

/// Fill `rect` with a horizontal multi-stop gradient defined by `stops`
/// (`(t, color)` with `t` in `[0,1]`, ascending). Builds one triangle strip so
/// the GPU blends continuously between stops — used by the input level meter to
/// shade healthy → hot → clipping across its width. Needs ≥ 2 stops.
pub(crate) fn horizontal_gradient(painter: &Painter, rect: Rect, stops: &[(f32, Color32)]) {
    if stops.len() < 2 || rect.width() <= 0.0 {
        return;
    }
    let mut mesh = Mesh::default();
    for (i, &(t, color)) in stops.iter().enumerate() {
        let x = rect.left() + rect.width() * t.clamp(0.0, 1.0);
        mesh.colored_vertex(Pos2::new(x, rect.top()), color);
        mesh.colored_vertex(Pos2::new(x, rect.bottom()), color);
        if i > 0 {
            let base = (i as u32 - 1) * 2;
            // Two triangles bridging the previous column pair to this one.
            mesh.add_triangle(base, base + 1, base + 2);
            mesh.add_triangle(base + 1, base + 3, base + 2);
        }
    }
    painter.add(Shape::mesh(mesh));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(f: impl FnOnce(&Painter)) {
        let ctx = egui::Context::default();
        let mut f = Some(f);
        let _ = ctx.run(Default::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                if let Some(f) = f.take() {
                    f(ui.painter());
                }
            });
        });
    }

    #[test]
    fn helpers_handle_normal_and_edge_inputs_without_panic() {
        run(|p| {
            let r = Rect::from_min_size(Pos2::new(10.0, 10.0), egui::vec2(120.0, 20.0));
            radial_glow(p, r.center(), 30.0, Color32::RED, 24);
            radial_glow(p, r.center(), 0.0, Color32::RED, 24); // radius 0 → no-op
            radial_glow(p, r.center(), 30.0, Color32::TRANSPARENT, 24); // alpha 0 → no-op
            radial_bloom(p, r.center(), 25.0, Color32::GREEN, 0.7);
            radial_bloom(p, r.center(), 25.0, Color32::GREEN, 0.0); // intensity 0 → no-op
            vertical_gradient(p, r, Color32::RED, Color32::BLUE);
            horizontal_gradient(
                p,
                r,
                &[(0.0, Color32::GREEN), (0.7, Color32::YELLOW), (1.0, Color32::RED)],
            );
            horizontal_gradient(p, r, &[(0.0, Color32::GREEN)]); // 1 stop → no-op
        });
    }
}
