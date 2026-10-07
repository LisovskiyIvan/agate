//! UI draw primitives: solid quads, rects, outlines, panels, thick lines,
//! progress bars, plus the pure batch math shared with the GPU upload path.
//!
//! Split out of `ui.zig` (facade). Functions take a generic `canvas: anytype`
//! (a `*UICanvas` in practice) so this module never imports `../ui.zig` —
//! following the `scene/` precedent, subsystems never import the facade;
//! the canvas passes itself. `ui.zig` owns the `UICanvas` type and provides
//! thin forwarders (`canvas.drawRect(...)` keeps working exactly as before).
//!
//! `UIVertex` lives here (the vertex layout belongs to the batch); `ui.zig`
//! re-exports it under its historical path.

const std = @import("std");

const math = @import("math");
const Color4 = math.Color4;

pub const UIVertex = extern struct {
    position: [2]f32,
    uv: [2]f32,
    color: [4]f32,
    mode_params: [4]f32, // x: mode (0=solid, 1=sdf_text, 2=sdf_outline), y: outline_width, z: softness, w: extra
};

// Shared solid-quad constants (same values as the previous per-call literals).
const solid_mode: [4]f32 = .{ 0.0, 0.0, 0.0, 0.0 };
const solid_uv: f32 = 1.0 / 512.0;

// Corner math using the already-computed segment length (single sqrt per line).
fn lineCornersWithLen(x0: f32, y0: f32, x1: f32, y1: f32, dx: f32, dy: f32, len: f32, thickness: f32) [4][2]f32 {
    if (len < 1e-6 or thickness <= 0.0) {
        return .{
            .{ x0, y0 },
            .{ x0, y0 },
            .{ x1, y1 },
            .{ x1, y1 },
        };
    }
    const nx = -dy / len;
    const ny = dx / len;
    const hx = nx * thickness * 0.5;
    const hy = ny * thickness * 0.5;
    return .{
        .{ x0 - hx, y0 - hy },
        .{ x0 + hx, y0 + hy },
        .{ x1 + hx, y1 + hy },
        .{ x1 - hx, y1 - hy },
    };
}

/// Helper to add a textured / colored quad
pub fn addQuad(
    canvas: anytype,
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    u_min: f32,
    v_min: f32,
    u_max: f32,
    v_max: f32,
    color: Color4,
    mode_params: [4]f32,
) void {
    if (canvas.vertices.items.len + 4 > canvas.capacity_vertices) return;
    if (canvas.indices.items.len + 6 > canvas.capacity_indices) return;

    const base_idx: u16 = @intCast(canvas.vertices.items.len);
    const col_arr = color.toArray();

    canvas.vertices.appendSlice(canvas.allocator, &[_]UIVertex{
        .{ .position = .{ x, y }, .uv = .{ u_min, v_min }, .color = col_arr, .mode_params = mode_params },
        .{ .position = .{ x + w, y }, .uv = .{ u_max, v_min }, .color = col_arr, .mode_params = mode_params },
        .{ .position = .{ x + w, y + h }, .uv = .{ u_max, v_max }, .color = col_arr, .mode_params = mode_params },
        .{ .position = .{ x, y + h }, .uv = .{ u_min, v_max }, .color = col_arr, .mode_params = mode_params },
    }) catch return;

    canvas.indices.appendSlice(canvas.allocator, &[_]u16{
        base_idx + 0, base_idx + 1, base_idx + 2,
        base_idx + 0, base_idx + 2, base_idx + 3,
    }) catch return;
}

/// Draws a solid rectangle in screen pixel coordinates
pub fn drawRect(canvas: anytype, x: f32, y: f32, w: f32, h: f32, color: Color4) void {
    addQuad(canvas, x, y, w, h, solid_uv, solid_uv, solid_uv, solid_uv, color, solid_mode);
}

/// Draws a rectangle outline with specified border thickness
pub fn drawRectOutline(canvas: anytype, x: f32, y: f32, w: f32, h: f32, thickness: f32, color: Color4) void {
    const t = @min(thickness, @min(w * 0.5, h * 0.5));
    drawRect(canvas, x, y, w, t, color); // Top
    drawRect(canvas, x, y + h - t, w, t, color); // Bottom
    drawRect(canvas, x, y + t, t, h - 2.0 * t, color); // Left
    drawRect(canvas, x + w - t, y + t, t, h - 2.0 * t, color); // Right
}

/// Draws a styled UI panel (filled rectangle + border)
pub fn drawPanel(canvas: anytype, x: f32, y: f32, w: f32, h: f32, bg_col: Color4, border_col: Color4, border_width: f32) void {
    drawRect(canvas, x, y, w, h, bg_col);
    if (border_width > 0.0) {
        drawRectOutline(canvas, x, y, w, h, border_width, border_col);
    }
}

/// Pure corner math for drawLine: returns the 4 quad corners
/// (p0-left, p0-right, p1-right, p1-left) offset perpendicular
/// to the segment by half the thickness. Zero-area on degenerate input.
pub fn lineCorners(x0: f32, y0: f32, x1: f32, y1: f32, thickness: f32) [4][2]f32 {
    const dx = x1 - x0;
    const dy = y1 - y0;
    const len = @sqrt(dx * dx + dy * dy);
    return lineCornersWithLen(x0, y0, x1, y1, dx, dy, len, thickness);
}

/// Draws a solid thick line in screen pixel coordinates (no depth test).
pub fn drawLine(canvas: anytype, x0: f32, y0: f32, x1: f32, y1: f32, thickness: f32, color: Color4) void {
    const dx = x1 - x0;
    const dy = y1 - y0;
    const len = @sqrt(dx * dx + dy * dy);
    if (len < 1e-6 or thickness <= 0.0) return;
    if (canvas.vertices.items.len + 4 > canvas.capacity_vertices) return;
    if (canvas.indices.items.len + 6 > canvas.capacity_indices) return;

    const corners = lineCornersWithLen(x0, y0, x1, y1, dx, dy, len, thickness);
    const base_idx: u16 = @intCast(canvas.vertices.items.len);
    const col_arr = color.toArray();

    canvas.vertices.appendSlice(canvas.allocator, &[_]UIVertex{
        .{ .position = corners[0], .uv = .{ solid_uv, solid_uv }, .color = col_arr, .mode_params = solid_mode },
        .{ .position = corners[1], .uv = .{ solid_uv, solid_uv }, .color = col_arr, .mode_params = solid_mode },
        .{ .position = corners[2], .uv = .{ solid_uv, solid_uv }, .color = col_arr, .mode_params = solid_mode },
        .{ .position = corners[3], .uv = .{ solid_uv, solid_uv }, .color = col_arr, .mode_params = solid_mode },
    }) catch return;

    canvas.indices.appendSlice(canvas.allocator, &[_]u16{
        base_idx + 0, base_idx + 1, base_idx + 2,
        base_idx + 0, base_idx + 2, base_idx + 3,
    }) catch return;
}

/// Draws a smooth horizontal progress / health bar
pub fn drawProgressBar(canvas: anytype, x: f32, y: f32, w: f32, h: f32, progress: f32, bg_col: Color4, fill_col: Color4) void {
    drawRect(canvas, x, y, w, h, bg_col);
    const clamped_p = std.math.clamp(progress, 0.0, 1.0);
    if (clamped_p > 0.001) {
        drawRect(canvas, x, y, w * clamped_p, h, fill_col);
    }
    drawRectOutline(canvas, x, y, w, h, 1.0, Color4.new(0.45, 0.5, 0.6, 0.7));
}

/// Bytes handed to sg by one rendered UI batch (clamped vertex prefix +
/// full index list). Must stay in usize: the vertex cap times
/// @sizeOf(UIVertex) overflows u16 arithmetic, so this must never run in
/// the clamped vertex type. Shared with the P6 frame path (same bytes
/// whether the upload runs in prepare or in the legacy render).
pub fn batchUploadBytes(vert_count: usize, index_count: usize) usize {
    return vert_count * @sizeOf(UIVertex) + index_count * @sizeOf(u16);
}

/// u16-clamped drawable vertex prefix (indices address vertices as
/// u16). Pure for tests; shared by the legacy render and the P6 frame.
pub fn clampedVertCount(len: usize) usize {
    return @min(len, @as(usize, std.math.maxInt(u16)));
}

/// Replacement capacity on growth (same formula as the legacy path).
/// Pure for tests.
pub fn grownCapacity(current: usize, need: usize) usize {
    return @max(current * 2, need);
}

test "UICanvas lineCorners" {
    // Horizontal segment: thickness extends along +/-Y.
    const h = lineCorners(0, 0, 10, 0, 2.0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), h[0][0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), h[0][1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), h[1][0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), h[1][1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), h[2][0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), h[2][1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), h[3][0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), h[3][1], 1e-5);

    // Vertical segment: thickness extends along +/-X.
    const v = lineCorners(0, 0, 0, 8, 4.0);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), v[0][0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), v[0][1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -2.0), v[1][0], 1e-5);
    // p1-right = end + half-normal; the normal is (-1, 0) here, mirroring
    // the horizontal case (p1-right keeps the +Y side there).
    try std.testing.expectApproxEqAbs(@as(f32, -2.0), v[2][0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 8.0), v[2][1], 1e-5);

    // Quad width matches thickness (diagonal case).
    const d = lineCorners(0, 0, 3, 4, 2.0);
    const w0x = d[1][0] - d[0][0];
    const w0y = d[1][1] - d[0][1];
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), @sqrt(w0x * w0x + w0y * w0y), 1e-5);

    // Degenerate segment collapses to the endpoints.
    const z = lineCorners(5, 5, 5, 5, 2.0);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), z[0][0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), z[2][0], 1e-5);
}

test "batchUploadBytes keeps the u16 vertex cap in usize arithmetic" {
    // 65535 vertices x 48 B ~= 3.1 MB: u16 arithmetic would already trap at
    // 1366 vertices, so the helper must compute in usize in every build mode.
    const verts: usize = std.math.maxInt(u16);
    const idx: usize = std.math.maxInt(u16);
    const expected = verts * @sizeOf(UIVertex) + idx * @sizeOf(u16);
    try std.testing.expectEqual(expected, batchUploadBytes(verts, idx));
}
