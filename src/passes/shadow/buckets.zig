//! Bucket rendering: per-item culling plus pipeline-grouped depth draws.
//! Split out of `shadow_pass.zig` (facade).
//!
//! `renderBuckets` is the shared draw core behind the CSM/spot/point paths
//! (`csm.zig`, `spot.zig`, `point.zig`): it walks the prepared bins in
//! `bucket_order`, skips culled/`gpu_pending` items and packs the
//! per-item/per-bucket uniforms. Per cascade it applies two distant-cascade
//! optimizations from `types.zig`: size culling of small AABBs
//! (`CASCADE_MIN_DIM`, near cascade and spot/point paths exempt) and the
//! low-poly shadow LOD stand-in (`shadowLodActive`, distant CSM cascades
//! only). Takes the pass and the prepared payload as
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
pub fn shadowItemCulled(item: anytype, frustum: math.Frustum, cascade_idx: ?usize) bool {
    if (!item.is_visible) return true;
    if (item.is_instanced) {
        if (item.visible_instance_count == 0 or item.instance_buffer.id == 0) return true;
    }
    const use_lod = types.shadowLodActive(item.has_shadow_lod, cascade_idx);
    const geo_vb = if (use_lod) item.lod_vertex_buffer else item.vertex_buffer;
    const geo_ib = if (use_lod) item.lod_index_buffer else item.index_buffer;
    if (sg.isvalid()) {
        // Validate the geometry actually selected for this cascade. The
        // snapshotted LOD child may have been retired after prepare; checking
        // only the source mesh handles would miss that case.
        if (geo_vb.id == 0 or sg.queryBufferState(geo_vb) != .VALID) return true;
        if (geo_ib.id == 0 or sg.queryBufferState(geo_ib) != .VALID) return true;
        if (item.is_instanced and item.instance_buffer.id != 0 and sg.queryBufferState(item.instance_buffer) != .VALID) return true;
    }
    if (!frustum.intersectsAABB(item.world_aabb)) return true;
    // Cascade-aware size culling: distant CSM cascades drop sub-texel
    // casters (named `types.CASCADE_MIN_DIM` policy; cascade 0 and the
    // spot/point paths never cull by size).
    if (types.cascadeSizeCulled(cascade_idx, item.max_dim)) return true;
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
            // Distant CSM cascades draw the snapshotted QEM stand-in when
            // one exists; near cascades and spot/point paths stay full-res.
            // Skinned items never carry a stand-in (see prepareInto), so the
            // bone-uniform path below always matches full-res geometry.
            const use_lod = types.shadowLodActive(item.has_shadow_lod, cascade_idx);
            const geo_vb = if (use_lod) item.lod_vertex_buffer else item.vertex_buffer;
            const geo_ib = if (use_lod) item.lod_index_buffer else item.index_buffer;
            const geo_ic = if (use_lod) item.lod_index_count else item.index_count;
            if (item.is_instanced) {
                if (item.visible_instance_count == 0 or item.instance_buffer.id == 0) continue;
                if (pip_id != last_pipeline_id.*) {
                    sg.applyPipeline(.{ .id = pip_id });
                    last_pipeline_id.* = pip_id;
                    last_vb_id = 0;
                    last_ib_id = 0;
                }

                var bind = sg.Bindings{};
                bind.vertex_buffers[0] = geo_vb;
                bind.vertex_buffers[1] = item.instance_buffer;
                bind.index_buffer = geo_ib;
                sg.applyBindings(bind);
                last_vb_id = 0;
                last_ib_id = 0;

                const inst_vs = shadow_shd.VsInstParams{
                    .light_view_proj = light_view_proj,
                };
                sg.applyUniforms(shadow_shd.UB_vs_inst_params, sg.asRange(&inst_vs));
                sg.draw(0, geo_ic, item.visible_instance_count);
                draw_calls.* += 1;
            } else {
                if (pip_id != last_pipeline_id.*) {
                    sg.applyPipeline(.{ .id = pip_id });
                    last_pipeline_id.* = pip_id;
                    last_vb_id = 0;
                    last_ib_id = 0;
                }

                if (geo_vb.id != last_vb_id or geo_ib.id != last_ib_id) {
                    var bind = sg.Bindings{};
                    bind.vertex_buffers[0] = geo_vb;
                    bind.index_buffer = geo_ib;
                    sg.applyBindings(bind);
                    last_vb_id = geo_vb.id;
                    last_ib_id = geo_ib.id;
                }

                const shadow_vs = shadow_shd.VsParams{
                    .mvp = Mat4.mul(light_view_proj, item.model),
                };
                sg.applyUniforms(shadow_shd.UB_vs_params, sg.asRange(&shadow_vs));

                // Skinned bucket missing valid bone copy:
                // skip draw rather than upload stale bone uniform.
                if (item.bucket == .skinned_u16 or item.bucket == .skinned_u32) {
                    const bones = scene_render_queue.skinAt(prepared.skins.items, item.skin_index) orelse continue;
                    const vs_skin = shadow_shd.VsSkin{
                        .bones = bones.*,
                    };
                    sg.applyUniforms(shadow_shd.UB_vs_skin, sg.asRange(&vs_skin));
                }

                sg.draw(0, geo_ic, 1);
                draw_calls.* += 1;
            }
        }
    }
}
