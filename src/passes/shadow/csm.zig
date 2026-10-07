//! CSM cascade depth rendering. Split out of `shadow_pass.zig` (facade).
//!
//! `renderCsm` renders the prepared buckets into the 4 cascade tiles of the
//! CSM atlas (2x2 grid of `SHADOW_ATLAS_SIZE`/2 tiles). Takes the pass and
//! the prepared payload as `anytype` so this module never imports `core.zig`
//! or the facade back (same discipline as `particles/*`, `profiler/*`);
//! bucket iteration + uniform packing live in `buckets.zig`.
const sokol = @import("sokol");
const sg = sokol.gfx;
const math = @import("math");
const Mat4 = math.Mat4;

const types = @import("types.zig");
const buckets = @import("buckets.zig");

pub fn renderCsm(
    self: anytype,
    prepared: anytype,
    cascades: [4]Mat4,
    draw_calls: *u32,
) void {
    var shadow_action = sg.PassAction{};
    shadow_action.depth = .{
        .load_action = .CLEAR,
        .clear_value = 1.0,
        .store_action = .STORE,
    };
    var shadow_pass = sg.Pass{
        .action = shadow_action,
    };
    shadow_pass.attachments.depth_stencil = self.attachment_view;
    sg.beginPass(shadow_pass);

    const CASCADE_RES: i32 = @intCast(types.SHADOW_ATLAS_SIZE / 2);
    // Kept across cascades: identical re-applies are skipped.
    var last_pipeline_id: u32 = 0;

    for (0..4) |c_idx| {
        const light_view_proj = cascades[c_idx];
        const vx: i32 = if (c_idx % 2 == 1) CASCADE_RES else 0;
        const vy: i32 = if (c_idx >= 2) CASCADE_RES else 0;

        sg.applyViewport(vx, vy, CASCADE_RES, CASCADE_RES, true);
        sg.applyScissorRect(vx, vy, CASCADE_RES, CASCADE_RES, true);

        const c_frustum = math.Frustum.fromViewProjection(light_view_proj);
        buckets.renderBuckets(self, prepared, light_view_proj, c_frustum, c_idx, &last_pipeline_id, draw_calls);
    }

    sg.endPass();
}
