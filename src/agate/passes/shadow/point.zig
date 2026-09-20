//! Point-light shadow atlas rendering. Split out of `shadow_pass.zig`
//! (facade).
//!
//! `renderPoint` renders the prepared buckets into the point-atlas tiles
//! (one 256px tile per cube face per shadow slot). Takes the pass and the
//! prepared payload as `anytype` so this module never imports `core.zig` or
//! the facade back (same discipline as `particles/*`, `profiler/*`);
//! bucket iteration + uniform packing live in `buckets.zig`.
const sokol = @import("sokol");
const sg = sokol.gfx;
const math = @import("math");
const Mat4 = math.Mat4;

const types = @import("types.zig");
const buckets = @import("buckets.zig");

pub fn renderPoint(
    self: anytype,
    prepared: anytype,
    point_shadows: []const types.PointShadowRenderInfo,
    draw_calls: *u32,
) void {
    // 3. Point light shadow pass: each entry is one cube-face tile of
    // the point atlas (regular/instanced/skinned bins via renderBuckets,
    // per-face frustum culling, no cascade size policy).
    if (point_shadows.len > 0 or self.point_needs_clear) {
        var point_action = sg.PassAction{};
        point_action.depth = .{
            .load_action = .CLEAR,
            .clear_value = 1.0,
            .store_action = .STORE,
        };
        var point_pass = sg.Pass{
            .action = point_action,
        };
        point_pass.attachments.depth_stencil = self.point_attachment_view;
        sg.beginPass(point_pass);
        var point_last_pipeline_id: u32 = 0;

        for (point_shadows) |point_info| {
            sg.applyViewport(point_info.tile_x, point_info.tile_y, types.POINT_SHADOW_RES, types.POINT_SHADOW_RES, false);
            sg.applyScissorRect(point_info.tile_x, point_info.tile_y, types.POINT_SHADOW_RES, types.POINT_SHADOW_RES, false);

            const point_frustum = math.Frustum.fromViewProjection(point_info.view_proj);
            buckets.renderBuckets(self, prepared, point_info.view_proj, point_frustum, null, &point_last_pipeline_id, draw_calls);
        }

        sg.endPass();
        self.point_needs_clear = false;
    }
}
