//! Bucket rendering: per-item culling plus pipeline-grouped depth draws.
//! Split out of `shadow_pass.zig` (facade).
//!
//! `renderBuckets` is the shared draw core behind the CSM/spot/point paths
//! (`csm.zig`, `spot.zig`, `point.zig`): it walks the prepared bins in
//! `bucket_order`, skips culled/`gpu_pending` items and packs the
//! per-item/per-bucket uniforms. Takes the pass and the prepared payload as
//! `anytype` so this module never imports `core.zig` or the facade back
//! (same discipline as `particles/*`, `profiler/*`).
//!
//! Moved tests reach `core.ShadowPass` through a block-scoped import that
//! exists only in test builds.
const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const shadow_shd = @import("shadow_shader");
const math = @import("math");
const Mat4 = math.Mat4;
const scene_render_queue = @import("../../scene/render_queue.zig");

const types = @import("types.zig");

fn pipelineFor(self: anytype, bucket: types.Bucket) u32 {
    return switch (bucket) {
        .regular_u16 => self.pipeline_u16.id,
        .regular_u32 => self.pipeline_u32.id,
        .inst_u16 => self.inst_pipeline_u16.id,
        .inst_u32 => self.inst_pipeline_u32.id,
        .skinned_u16 => self.skinned_pipeline_u16.id,
        .skinned_u32 => self.skinned_pipeline_u32.id,
    };
}

// Union of the regular and instanced shadow culling checks, applied to
// both paths. Batch-AABB semantics for instanced items: world_aabb is
// the fresh combined AABB of the whole batch for the frame, so the
// frustum test is conservative — the whole batch is skipped only when
// it lies fully outside the light frustum. max_dim is that batch
// AABB's max extent, and far-cascade small-batch culling uses the same
// 0.12/0.35/0.75 policy as regular items.
fn shadowItemCulled(item: anytype, frustum: math.Frustum, cascade_idx: ?usize) bool {
    if (!item.is_visible) return true;
    if (!frustum.intersectsAABB(item.world_aabb)) return true;
    if (item.is_instanced) {
        if (item.visible_instance_count == 0 or item.instance_buffer.id == 0) return true;
    }
    if (cascade_idx) |c_idx| {
        if (c_idx == 1 and item.max_dim < 0.12) return true;
        if (c_idx == 2 and item.max_dim < 0.35) return true;
        if (c_idx == 3 and item.max_dim < 0.75) return true;
    }
    return false;
}

pub fn renderBuckets(
    self: anytype,
    prepared: anytype,
    light_view_proj: Mat4,
    frustum: math.Frustum,
    cascade_idx: ?usize,
    last_pipeline_id: *u32,
    draw_calls: *u32,
) void {
    var last_vb_id: u32 = 0;
    var last_ib_id: u32 = 0;

    for (types.bucket_order) |bucket| {
        const b_idx = @intFromEnum(bucket);
        const count = prepared.bin.counts[b_idx];
        if (count == 0) continue;

        const pip_id = pipelineFor(self, bucket);
        if (pip_id == 0) continue;

        const bucket_items = prepared.items.items[prepared.bin.offsets[b_idx] .. prepared.bin.offsets[b_idx] + count];
        for (bucket_items) |item| {
            if (item.gpu_pending) continue;
            if (shadowItemCulled(item, frustum, cascade_idx)) continue;
            if (item.is_instanced) {
                if (pip_id != last_pipeline_id.*) {
                    sg.applyPipeline(.{ .id = pip_id });
                    last_pipeline_id.* = pip_id;
                    last_vb_id = 0;
                    last_ib_id = 0;
                }

                var bind = sg.Bindings{};
                bind.vertex_buffers[0] = item.vertex_buffer;
                bind.vertex_buffers[1] = item.instance_buffer;
                bind.index_buffer = item.index_buffer;
                sg.applyBindings(bind);
                last_vb_id = 0;
                last_ib_id = 0;

                const inst_vs = shadow_shd.VsInstParams{
                    .light_view_proj = light_view_proj,
                };
                sg.applyUniforms(shadow_shd.UB_vs_inst_params, sg.asRange(&inst_vs));
                sg.draw(0, item.index_count, item.visible_instance_count);
                draw_calls.* += 1;
            } else {
                if (pip_id != last_pipeline_id.*) {
                    sg.applyPipeline(.{ .id = pip_id });
                    last_pipeline_id.* = pip_id;
                    last_vb_id = 0;
                    last_ib_id = 0;
                }

                if (item.vertex_buffer.id != last_vb_id or item.index_buffer.id != last_ib_id) {
                    var bind = sg.Bindings{};
                    bind.vertex_buffers[0] = item.vertex_buffer;
                    bind.index_buffer = item.index_buffer;
                    sg.applyBindings(bind);
                    last_vb_id = item.vertex_buffer.id;
                    last_ib_id = item.index_buffer.id;
                }

                const shadow_vs = shadow_shd.VsParams{
                    .mvp = Mat4.mul(light_view_proj, item.model),
                };
                sg.applyUniforms(shadow_shd.UB_vs_params, sg.asRange(&shadow_vs));

                // Skinned-бакет без валидной копии (билдером недостижимо):
                // пропуск draw вместо stale-униформы чужого draw.
                if (item.bucket == .skinned_u16 or item.bucket == .skinned_u32) {
                    const bones = scene_render_queue.skinAt(prepared.skins.items, item.skin_index) orelse continue;
                    const vs_skin = shadow_shd.VsSkin{
                        .bones = bones.*,
                    };
                    sg.applyUniforms(shadow_shd.UB_vs_skin, sg.asRange(&vs_skin));
                }

                sg.draw(0, item.index_count, 1);
                draw_calls.* += 1;
            }
        }
    }
}

test "shadow item culling skips regular items outside the light frustum" {
    const ShadowPass = @import("core.zig").ShadowPass;
    const frustum = math.Frustum.fromViewProjection(Mat4.identity);
    const inside_aabb = math.BoundingBox.init(
        math.Vec3.new(-0.5, -0.5, 0.2),
        math.Vec3.new(0.5, 0.5, 0.8),
    );
    const outside_aabb = math.BoundingBox.init(
        math.Vec3.new(5.0, 5.0, 5.0),
        math.Vec3.new(6.0, 6.0, 6.0),
    );

    var inside = ShadowPass.ShadowDrawItem{
        .world_aabb = inside_aabb,
        .max_dim = 1.0,
    };
    try std.testing.expect(!shadowItemCulled(inside, frustum, 0));

    inside.world_aabb = outside_aabb;
    try std.testing.expect(shadowItemCulled(inside, frustum, 0));
}

test "shadow item culling applies batch AABB and instance checks to instanced items" {
    const ShadowPass = @import("core.zig").ShadowPass;
    const frustum = math.Frustum.fromViewProjection(Mat4.identity);
    const inside_aabb = math.BoundingBox.init(
        math.Vec3.new(-0.5, -0.5, 0.2),
        math.Vec3.new(0.5, 0.5, 0.8),
    );
    const outside_aabb = math.BoundingBox.init(
        math.Vec3.new(5.0, 5.0, 5.0),
        math.Vec3.new(6.0, 6.0, 6.0),
    );

    var item = ShadowPass.ShadowDrawItem{
        .world_aabb = outside_aabb,
        .max_dim = 1.0,
        .is_instanced = true,
        .visible_instance_count = 4,
        .instance_buffer = .{ .id = 1 },
    };
    try std.testing.expect(shadowItemCulled(item, frustum, 0));

    item.world_aabb = inside_aabb;
    try std.testing.expect(!shadowItemCulled(item, frustum, 0));

    item.visible_instance_count = 0;
    try std.testing.expect(shadowItemCulled(item, frustum, 0));

    item.visible_instance_count = 4;
    item.instance_buffer = .{};
    try std.testing.expect(shadowItemCulled(item, frustum, 0));
}

test "shadow item culling applies far-cascade max_dim policy to all items" {
    const ShadowPass = @import("core.zig").ShadowPass;
    const frustum = math.Frustum.fromViewProjection(Mat4.identity);
    const inside_aabb = math.BoundingBox.init(
        math.Vec3.new(-0.5, -0.5, 0.2),
        math.Vec3.new(0.5, 0.5, 0.8),
    );

    const item = ShadowPass.ShadowDrawItem{
        .world_aabb = inside_aabb,
        .max_dim = 0.5,
    };
    try std.testing.expect(shadowItemCulled(item, frustum, 3));
    try std.testing.expect(!shadowItemCulled(item, frustum, 0));

    const inst = ShadowPass.ShadowDrawItem{
        .world_aabb = inside_aabb,
        .max_dim = 0.5,
        .is_instanced = true,
        .visible_instance_count = 4,
        .instance_buffer = .{ .id = 1 },
    };
    try std.testing.expect(shadowItemCulled(inst, frustum, 3));
    try std.testing.expect(!shadowItemCulled(inst, frustum, 0));
}

test "shadow item culling skips invisible items" {
    const ShadowPass = @import("core.zig").ShadowPass;
    const frustum = math.Frustum.fromViewProjection(Mat4.identity);
    const inside_aabb = math.BoundingBox.init(
        math.Vec3.new(-0.5, -0.5, 0.2),
        math.Vec3.new(0.5, 0.5, 0.8),
    );

    const hidden = ShadowPass.ShadowDrawItem{
        .world_aabb = inside_aabb,
        .max_dim = 1.0,
        .is_visible = false,
    };
    try std.testing.expect(shadowItemCulled(hidden, frustum, 0));

    const hidden_inst = ShadowPass.ShadowDrawItem{
        .world_aabb = inside_aabb,
        .max_dim = 1.0,
        .is_visible = false,
        .is_instanced = true,
        .visible_instance_count = 4,
        .instance_buffer = .{ .id = 1 },
    };
    try std.testing.expect(shadowItemCulled(hidden_inst, frustum, 0));
}
