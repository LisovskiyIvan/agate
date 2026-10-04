//! Spot-light shadow depth rendering. Split out of `shadow_pass.zig`
//! (facade).
//!
//! `renderSpot` renders the prepared buckets into the spot atlas tiles (one
//! 512px tile per spot, side by side). Takes the pass and the prepared
//! payload as `anytype` so this module never imports `core.zig` or the
//! facade back (same discipline as `particles/*`, `profiler/*`); bucket
//! iteration + uniform packing live in `buckets.zig`.
const sokol = @import("sokol");
const sg = sokol.gfx;
const math = @import("math");
const Mat4 = math.Mat4;

const types = @import("types.zig");
const buckets = @import("buckets.zig");

pub fn renderSpot(
    self: anytype,
    prepared: anytype,
    spot_shadows: []const types.SpotShadowRenderInfo,
    draw_calls: *u32,
) void {
    // 2. Spot light shadow pass
    if (spot_shadows.len > 0 or self.spot_needs_clear) {
        var spot_action = sg.PassAction{};
        spot_action.depth = .{
            .load_action = .CLEAR,
            .clear_value = 1.0,
            .store_action = .STORE,
        };
        var spot_pass = sg.Pass{
            .action = spot_action,
        };
        spot_pass.attachments.depth_stencil = self.spot_attachment_view;
        sg.beginPass(spot_pass);
        var spot_last_pipeline_id: u32 = 0;

        for (spot_shadows) |spot_info| {
            const origin = types.spotTileOrigin(spot_info.spot_index);
            const vx: i32 = if (spot_info.tile_x != 0 or spot_info.tile_y != 0) spot_info.tile_x else origin.x;
            const vy: i32 = if (spot_info.tile_x != 0 or spot_info.tile_y != 0) spot_info.tile_y else origin.y;
            sg.applyViewport(vx, vy, types.SPOT_SHADOW_RES, types.SPOT_SHADOW_RES, false);
            sg.applyScissorRect(vx, vy, types.SPOT_SHADOW_RES, types.SPOT_SHADOW_RES, false);

            const spot_frustum = math.Frustum.fromViewProjection(spot_info.view_proj);
            buckets.renderBuckets(self, prepared, spot_info.view_proj, spot_frustum, null, &spot_last_pipeline_id, draw_calls);
        }

        sg.endPass();
        self.spot_needs_clear = false;
    }
}
