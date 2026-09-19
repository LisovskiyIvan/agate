const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const shadow_shd = @import("shadow_shader");
const math = @import("math");
const Mat4 = math.Mat4;
const mesh_mod = @import("../mesh.zig");
const Mesh = mesh_mod.Mesh;
const InstancedMeshForP5 = mesh_mod.InstancedMesh;
const scene_render_queue = @import("../scene/render_queue.zig");
const instance_staging = @import("../scene/instance_staging.zig");
const outline_pass = @import("outline_pass.zig");
const Vertex = mesh_mod.Vertex;
const jobs = @import("../jobs.zig");

// Resolution of the shadow atlas texture (square). Must match the
// SHADOW_ATLAS_SIZE fallback in shaders/{standard,pbr,instanced,skinned_pbr}.glsl:
// sokol-shdc --defines only supports valueless macros, so the value cannot be
// injected from the build and lives in these two places by convention.
pub const SHADOW_ATLAS_SIZE: u32 = 2048;
pub const SPOT_SHADOW_MAP_WIDTH: u32 = 1024;
pub const SPOT_SHADOW_MAP_HEIGHT: u32 = 512;
pub const SPOT_SHADOW_RES: i32 = 512;

// Point-light shadow atlas: 2 shadow slots x 6 cube faces as 256px tiles in
// one 2D depth texture, sampled with the same 2D compare + PCF path as the
// spot atlas (no cube textures anywhere).
//
// Final atlas budget (all atlases are separate depth textures):
//   CSM atlas   2048x2048: 4 cascades of 1024x1024 (2x2 grid, unchanged).
//   Spot atlas  1024x512:  2 tiles of 512x512 side by side (unchanged).
//   Point atlas 1536x512:  12 tiles of 256x256 — 6 faces per row, one row
//     per shadow slot (slot 0: y 0..256, slot 1: y 256..512).
// SHADOW_ATLAS_SIZE stays 2048: the CSM layout is untouched, so no shader
// fallback needs updating.
pub const POINT_SHADOW_SLOTS: usize = 2;
pub const POINT_SHADOW_FACES: usize = 6;
pub const POINT_SHADOW_RES: i32 = 256;
pub const POINT_SHADOW_MAP_WIDTH: u32 = 1536;
pub const POINT_SHADOW_MAP_HEIGHT: u32 = 512;

pub const SpotShadowRenderInfo = struct {
    spot_index: usize = 0,
    view_proj: Mat4 = Mat4.identity,
};

pub const PointShadowRenderInfo = struct {
    tile_x: i32 = 0,
    tile_y: i32 = 0,
    view_proj: Mat4 = Mat4.identity,
};

/// Cube-face index for a light-space direction, mirroring the GLSL
/// pointFaceIndex in the forward shaders: major axis wins, ties prefer
/// X over Y over Z, the sign picks the positive/negative face
/// (order +X, -X, +Y, -Y, +Z, -Z, matching PointLight face order).
pub fn pointFaceForDir(d: math.Vec3) usize {
    const ax = @abs(d.x);
    const ay = @abs(d.y);
    const az = @abs(d.z);
    if (ax >= ay and ax >= az) return if (d.x >= 0.0) 0 else 1;
    if (ay >= ax and ay >= az) return if (d.y >= 0.0) 2 else 3;
    return if (d.z >= 0.0) 4 else 5;
}

/// Pixel origin of a point-shadow tile: faces run left to right, one row
/// per shadow slot.
pub fn pointTileOrigin(slot: usize, face: usize) struct { x: i32, y: i32 } {
    return .{
        .x = @as(i32, @intCast(face % POINT_SHADOW_FACES)) * POINT_SHADOW_RES,
        .y = @as(i32, @intCast(slot % POINT_SHADOW_SLOTS)) * POINT_SHADOW_RES,
    };
}

pub const ShadowPass = struct {
    image: sg.Image,
    attachment_view: sg.View,
    texture_view: sg.View,
    sampler: sg.Sampler,
    /// Nonfiltering sampler for raw depth reads (PCSS blocker search); the
    /// comparison sampler above is only valid for depth2d shadow lookups.
    depth_sampler: sg.Sampler,
    spot_image: sg.Image,
    spot_attachment_view: sg.View,
    spot_texture_view: sg.View,
    spot_needs_clear: bool = true,
    point_image: sg.Image,
    point_attachment_view: sg.View,
    point_texture_view: sg.View,
    point_needs_clear: bool = true,
    pipeline_u16: sg.Pipeline,
    pipeline_u32: sg.Pipeline,
    inst_pipeline_u16: sg.Pipeline,
    inst_pipeline_u32: sg.Pipeline,
    skinned_pipeline_u16: sg.Pipeline,
    skinned_pipeline_u32: sg.Pipeline,
    shadow_shader: sg.Shader = .{},
    inst_shader: sg.Shader = .{},
    skinned_shader: sg.Shader = .{},
    allocator: std.mem.Allocator,
    binned_meshes: std.ArrayListUnmanaged(*Mesh) = .empty,
    /// Mesh-list indices parallel to `binned_meshes` (stage-2 increment A):
    /// `binned_meshes[i]` came from input `meshes[binned_source[i]]`.
    /// Retained across frames like `binned_meshes`; cleared together on OOM
    /// so the two lists stay coherent (never stale ranges).
    binned_source: std.ArrayListUnmanaged(u32) = .empty,
    /// Standalone owned prepared payload (P4 render-owned snapshot contract):
    /// `prepare` publishes here, `renderPrepared` consumes it. Scene drives
    /// per-slot payloads through `prepareInto`/`renderPreparedFrom` instead,
    /// sharing the same algorithm — no duplication.
    prepared: PreparedShadowDraws = .{},

    // Pipeline buckets in fixed order; the grouping key is defined once per
    // render so each cascade issues at most one applyPipeline per bucket.
    pub const Bucket = enum { regular_u16, regular_u32, inst_u16, inst_u32, skinned_u16, skinned_u32 };
    pub const bucket_order: [6]Bucket = .{ .regular_u16, .regular_u32, .inst_u16, .inst_u32, .skinned_u16, .skinned_u32 };

    /// Self-contained per-item payload for shadow rendering. Хранит только
    /// render-owned снимки (модель, AABB, хендлы, индекс копии скина):
    /// живых указателей на Mesh/Skeleton здесь нет.
    ///
    /// Identity (stage-2 increment A, refactor-only): `source_uid` is the
    /// source mesh's `Mesh.uid` (nonzero, stable for lifetime), `source_mesh`
    /// is the mesh-list index at build time. `instance_buffer`/
    /// `visible_instance_count` are provisional for game-built (`.build_view`)
    /// payloads until the latch `patchInstanceRefs` finalizes them; identity
    /// is validated by uid at patch time.
    pub const ShadowDrawItem = struct {
        vertex_buffer: sg.Buffer = .{},
        index_buffer: sg.Buffer = .{},
        index_count: u32 = 0,
        instance_buffer: sg.Buffer = .{},
        visible_instance_count: u32 = 0,
        model: Mat4 = Mat4.identity,
        world_aabb: math.BoundingBox = math.BoundingBox.zero,
        max_dim: f32 = 0.0,
        /// Индекс копии скин-матриц в PreparedShadowDraws.skins (null = не скин).
        skin_index: ?u32 = null,
        bucket: Bucket = .regular_u16,
        is_instanced: bool = false,
        gpu_pending: bool = false,
        is_visible: bool = true,
        source_uid: u64 = 0,
        source_mesh: u32 = 0,
    };

    // Matches Scene.render's pipeline pick: instanced wins over skinned.
    fn bucketFor(mesh: *const Mesh) Bucket {
        const is_32 = mesh.index_type == .UINT32;
        if (mesh.instances.items.len > 0) return if (is_32) .inst_u32 else .inst_u16;
        if (mesh.skeleton != null) return if (is_32) .skinned_u32 else .skinned_u16;
        return if (is_32) .regular_u32 else .regular_u16;
    }

    fn pipelineFor(self: *const ShadowPass, bucket: Bucket) u32 {
        return switch (bucket) {
            .regular_u16 => self.pipeline_u16.id,
            .regular_u32 => self.pipeline_u32.id,
            .inst_u16 => self.inst_pipeline_u16.id,
            .inst_u32 => self.inst_pipeline_u32.id,
            .skinned_u16 => self.skinned_pipeline_u16.id,
            .skinned_u32 => self.skinned_pipeline_u32.id,
        };
    }

    pub fn init(allocator: std.mem.Allocator) ShadowPass {
        // SHADOW_ATLAS_SIZE atlas holding 4x (SHADOW_ATLAS_SIZE/2) cascades (2x2).
        // Was 4096/2048: same look for near geometry, 4x fewer depth texels rasterized per frame.
        const depth_img = sg.makeImage(.{
            .usage = .{ .depth_stencil_attachment = true },
            .pixel_format = .DEPTH,
            .width = SHADOW_ATLAS_SIZE,
            .height = SHADOW_ATLAS_SIZE,
            .sample_count = 1,
        });
        const att_view = sg.makeView(.{
            .depth_stencil_attachment = .{ .image = depth_img },
        });

        const tex_view = sg.makeView(.{
            .texture = .{ .image = depth_img },
        });

        const smp = sg.makeSampler(.{
            .min_filter = .LINEAR,
            .mag_filter = .LINEAR,
            .wrap_u = .CLAMP_TO_EDGE,
            .wrap_v = .CLAMP_TO_EDGE,
            .compare = .LESS_EQUAL,
        });
        const raw_depth_smp = sg.makeSampler(.{
            .min_filter = .NEAREST,
            .mag_filter = .NEAREST,
            .wrap_u = .CLAMP_TO_EDGE,
            .wrap_v = .CLAMP_TO_EDGE,
        });

        // 1. Shadow Depth pipelines (regular meshes)
        const shadow_shd_handle = sg.makeShader(shadow_shd.shadowShaderDesc(sg.queryBackend()));
        var shadow_pip_desc = sg.PipelineDesc{
            .shader = shadow_shd_handle,
            .index_type = .UINT16,
            .sample_count = 1,
            .depth = .{
                .pixel_format = .DEPTH,
                .compare = .LESS_EQUAL,
                .write_enabled = true,
                .bias = 1.0,
                .bias_slope_scale = 1.0,
            },
            .cull_mode = .BACK,
            .face_winding = .CCW,
        };
        shadow_pip_desc.colors[0].pixel_format = .NONE;
        shadow_pip_desc.layout.buffers[0] = .{ .stride = @sizeOf(Vertex) };
        shadow_pip_desc.layout.attrs[shadow_shd.ATTR_shadow_position] = .{
            .format = .FLOAT3,
            .offset = @offsetOf(Vertex, "position"),
        };

        const pip_u16 = sg.makePipeline(shadow_pip_desc);
        shadow_pip_desc.index_type = .UINT32;
        const pip_u32 = sg.makePipeline(shadow_pip_desc);

        // 2. Shadow Depth pipelines (instanced meshes)
        const inst_shd_handle = sg.makeShader(shadow_shd.shadowInstancedShaderDesc(sg.queryBackend()));
        var shadow_inst_desc = sg.PipelineDesc{
            .shader = inst_shd_handle,
            .index_type = .UINT16,
            .sample_count = 1,
            .depth = .{
                .pixel_format = .DEPTH,
                .compare = .LESS_EQUAL,
                .write_enabled = true,
                .bias = 1.0,
                .bias_slope_scale = 1.0,
            },
            .cull_mode = .BACK,
            .face_winding = .CCW,
        };
        shadow_inst_desc.colors[0].pixel_format = .NONE;
        shadow_inst_desc.layout.buffers[0] = .{ .stride = @sizeOf(Vertex) };
        shadow_inst_desc.layout.attrs[shadow_shd.ATTR_shadow_instanced_position] = .{
            .buffer_index = 0,
            .format = .FLOAT3,
            .offset = @offsetOf(Vertex, "position"),
        };
        shadow_inst_desc.layout.buffers[1] = .{
            .step_func = .PER_INSTANCE,
            .step_rate = 1,
            .stride = @sizeOf(Mat4),
        };
        shadow_inst_desc.layout.attrs[shadow_shd.ATTR_shadow_instanced_inst_mat0] = .{ .buffer_index = 1, .offset = 0, .format = .FLOAT4 };
        shadow_inst_desc.layout.attrs[shadow_shd.ATTR_shadow_instanced_inst_mat1] = .{ .buffer_index = 1, .offset = 16, .format = .FLOAT4 };
        shadow_inst_desc.layout.attrs[shadow_shd.ATTR_shadow_instanced_inst_mat2] = .{ .buffer_index = 1, .offset = 32, .format = .FLOAT4 };
        shadow_inst_desc.layout.attrs[shadow_shd.ATTR_shadow_instanced_inst_mat3] = .{ .buffer_index = 1, .offset = 48, .format = .FLOAT4 };

        const inst_pip_u16 = sg.makePipeline(shadow_inst_desc);
        shadow_inst_desc.index_type = .UINT32;
        const inst_pip_u32 = sg.makePipeline(shadow_inst_desc);

        // 3. Skinned Shadow pipelines
        const skinned_shd_handle = sg.makeShader(shadow_shd.shadowSkinnedShaderDesc(sg.queryBackend()));
        var shadow_skinned_desc = sg.PipelineDesc{
            .shader = skinned_shd_handle,
            .index_type = .UINT16,
            .sample_count = 1,
            .depth = .{
                .pixel_format = .DEPTH,
                .compare = .LESS_EQUAL,
                .write_enabled = true,
                .bias = 1.0,
                .bias_slope_scale = 1.0,
            },
            .cull_mode = .BACK,
            .face_winding = .CCW,
        };
        shadow_skinned_desc.colors[0].pixel_format = .NONE;
        shadow_skinned_desc.layout.buffers[0] = .{ .stride = @sizeOf(Vertex) };
        shadow_skinned_desc.layout.attrs[shadow_shd.ATTR_shadow_skinned_position] = .{
            .format = .FLOAT3,
            .offset = @offsetOf(Vertex, "position"),
        };
        shadow_skinned_desc.layout.attrs[shadow_shd.ATTR_shadow_skinned_joints] = .{
            .format = .FLOAT4,
            .offset = @offsetOf(Vertex, "joints"),
        };
        shadow_skinned_desc.layout.attrs[shadow_shd.ATTR_shadow_skinned_weights] = .{
            .format = .FLOAT4,
            .offset = @offsetOf(Vertex, "weights"),
        };
        const skinned_pip_u16 = sg.makePipeline(shadow_skinned_desc);
        shadow_skinned_desc.index_type = .UINT32;
        const skinned_pip_u32 = sg.makePipeline(shadow_skinned_desc);

        const spot_depth_img = sg.makeImage(.{
            .usage = .{ .depth_stencil_attachment = true },
            .pixel_format = .DEPTH,
            .width = SPOT_SHADOW_MAP_WIDTH,
            .height = SPOT_SHADOW_MAP_HEIGHT,
            .sample_count = 1,
        });
        const spot_att_view = sg.makeView(.{
            .depth_stencil_attachment = .{ .image = spot_depth_img },
        });
        const spot_tex_view = sg.makeView(.{
            .texture = .{ .image = spot_depth_img },
        });

        // Point-light shadow atlas: 6 cube faces x 2 slots as 256px tiles
        // (see the layout note on the POINT_SHADOW_* constants above).
        const point_depth_img = sg.makeImage(.{
            .usage = .{ .depth_stencil_attachment = true },
            .pixel_format = .DEPTH,
            .width = POINT_SHADOW_MAP_WIDTH,
            .height = POINT_SHADOW_MAP_HEIGHT,
            .sample_count = 1,
        });
        const point_att_view = sg.makeView(.{
            .depth_stencil_attachment = .{ .image = point_depth_img },
        });
        const point_tex_view = sg.makeView(.{
            .texture = .{ .image = point_depth_img },
        });

        return .{
            .allocator = allocator,
            .binned_meshes = .empty,
            .image = depth_img,
            .attachment_view = att_view,
            .texture_view = tex_view,
            .sampler = smp,
            .depth_sampler = raw_depth_smp,
            .spot_image = spot_depth_img,
            .spot_attachment_view = spot_att_view,
            .spot_texture_view = spot_tex_view,
            .spot_needs_clear = true,
            .point_image = point_depth_img,
            .point_attachment_view = point_att_view,
            .point_texture_view = point_tex_view,
            .point_needs_clear = true,
            .pipeline_u16 = pip_u16,
            .pipeline_u32 = pip_u32,
            .inst_pipeline_u16 = inst_pip_u16,
            .inst_pipeline_u32 = inst_pip_u32,
            .skinned_pipeline_u16 = skinned_pip_u16,
            .skinned_pipeline_u32 = skinned_pip_u32,
            .shadow_shader = shadow_shd_handle,
            .inst_shader = inst_shd_handle,
            .skinned_shader = skinned_shd_handle,
        };
    }

    // Union of the regular and instanced shadow culling checks, applied to
    // both paths. Batch-AABB semantics for instanced items: world_aabb is
    // the fresh combined AABB of the whole batch for the frame, so the
    // frustum test is conservative — the whole batch is skipped only when
    // it lies fully outside the light frustum. max_dim is that batch
    // AABB's max extent, and far-cascade small-batch culling uses the same
    // 0.12/0.35/0.75 policy as regular items.
    fn shadowItemCulled(item: ShadowDrawItem, frustum: math.Frustum, cascade_idx: ?usize) bool {
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

    fn renderBuckets(
        self: *ShadowPass,
        prepared: *const PreparedShadowDraws,
        light_view_proj: Mat4,
        frustum: math.Frustum,
        cascade_idx: ?usize,
        last_pipeline_id: *u32,
        draw_calls: *u32,
    ) void {
        var last_vb_id: u32 = 0;
        var last_ib_id: u32 = 0;

        for (bucket_order) |bucket| {
            const b_idx = @intFromEnum(bucket);
            const count = prepared.bin.counts[b_idx];
            if (count == 0) continue;

            const pip_id = self.pipelineFor(bucket);
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

    pub const BinResult = struct {
        counts: [6]usize,
        offsets: [6]usize,
    };

    /// Small owning prepared payload: shadow items + skin copies + bin
    /// ranges. Render-owned CPU snapshots (model/AABB/handles/skin index);
    /// GPU handles inside are BORROWED (phase mutex / P3 epochs), never
    /// destroyed or duplicated here. Counts/offsets always stay inside
    /// items (coherent-empty on OOM, never stale ranges).
    pub const PreparedShadowDraws = struct {
        items: std.ArrayListUnmanaged(ShadowDrawItem) = .empty,
        /// Render-owned копии скин-матриц shadow-draws (резолв по skin_index).
        /// Сбрасывается в prepareInto, переживает кадры (без per-frame churn).
        skins: scene_render_queue.SkinStorage = .empty,
        bin: BinResult = .{
            .counts = [_]usize{0} ** 6,
            .offsets = [_]usize{0} ** 6,
        },

        /// Clear lengths for reuse, retaining capacity (coherent-empty).
        pub fn reset(self: *PreparedShadowDraws) void {
            self.items.clearRetainingCapacity();
            self.skins.clearRetainingCapacity();
            self.bin = .{
                .counts = [_]usize{0} ** 6,
                .offsets = [_]usize{0} ** 6,
            };
        }

        pub fn deinit(self: *PreparedShadowDraws, allocator: std.mem.Allocator) void {
            self.items.deinit(allocator);
            self.skins.deinit(allocator);
        }
    };

    const ParallelShadowBinning = struct {
        meshes: []const *Mesh,
        span: usize,
        chunk_counts: [][6]usize,
        chunk_offsets: [][6]usize,
        out_binned: []*Mesh,
        out_source: []u32,

        fn countChunkRange(pass: *ParallelShadowBinning, start: usize, end: usize) void {
            for (start..end) |chunk_id| {
                const lo = chunk_id * pass.span;
                if (lo >= pass.meshes.len) {
                    pass.chunk_counts[chunk_id] = .{ 0, 0, 0, 0, 0, 0 };
                    continue;
                }
                const hi = @min(lo + pass.span, pass.meshes.len);
                var local: [6]usize = .{ 0, 0, 0, 0, 0, 0 };
                for (pass.meshes[lo..hi]) |mesh| {
                    if (!mesh.cast_shadows or mesh.is_lod_child or mesh.is_decal) continue;
                    local[@intFromEnum(bucketFor(mesh))] += 1;
                }
                pass.chunk_counts[chunk_id] = local;
            }
        }

        fn scatterChunkRange(pass: *ParallelShadowBinning, start: usize, end: usize) void {
            for (start..end) |chunk_id| {
                const lo = chunk_id * pass.span;
                if (lo >= pass.meshes.len) continue;
                const hi = @min(lo + pass.span, pass.meshes.len);
                var cursors = pass.chunk_offsets[chunk_id];
                for (pass.meshes[lo..hi], lo..) |mesh, src_idx| {
                    if (!mesh.cast_shadows or mesh.is_lod_child or mesh.is_decal) continue;
                    const b = @intFromEnum(bucketFor(mesh));
                    pass.out_binned[cursors[b]] = mesh;
                    pass.out_source[cursors[b]] = @intCast(src_idx);
                    cursors[b] += 1;
                }
            }
        }
    };

    pub fn binMeshes(
        self: *ShadowPass,
        meshes: []const *Mesh,
        pool: ?*jobs.Pool,
    ) BinResult {
        var counts: [6]usize = .{ 0, 0, 0, 0, 0, 0 };
        var offsets: [6]usize = undefined;

        const min_meshes_for_parallel: usize = 128;
        if (pool != null and pool.?.workerCount() > 0 and meshes.len >= min_meshes_for_parallel) {
            const p = pool.?;
            const chunk_count = @min((p.workerCount() + 1) * 2, 32);
            const span = (meshes.len + chunk_count - 1) / chunk_count;

            var chunk_counts_buf: [32][6]usize = undefined;
            var chunk_offsets_buf: [32][6]usize = undefined;
            const chunk_counts = chunk_counts_buf[0..chunk_count];
            const chunk_offsets = chunk_offsets_buf[0..chunk_count];

            var pass = ParallelShadowBinning{
                .meshes = meshes,
                .span = span,
                .chunk_counts = chunk_counts,
                .chunk_offsets = chunk_offsets,
                .out_binned = &.{},
                .out_source = &.{},
            };

            p.forkJoin(ParallelShadowBinning, &pass, ParallelShadowBinning.countChunkRange, chunk_count);

            for (0..chunk_count) |c| {
                for (0..6) |b| {
                    counts[b] += chunk_counts[c][b];
                }
            }

            var total: usize = 0;
            for (0..6) |b| {
                offsets[b] = total;
                total += counts[b];
            }

            if (total > self.binned_meshes.items.len) {
                // Атомарность публикации: рост не удался — список заимствованных
                // указателей очищается (меши могли быть destroyMesh между кадрами,
                // prepare НЕ ДОЛЖЕН их читать), возвращается пустой coherent-снимок.
                self.binned_meshes.resize(self.allocator, total) catch {
                    self.binned_meshes.clearRetainingCapacity();
                    self.binned_source.clearRetainingCapacity();
                    return .{
                        .counts = [_]usize{0} ** 6,
                        .offsets = [_]usize{0} ** 6,
                    };
                };
                self.binned_source.resize(self.allocator, total) catch {
                    self.binned_meshes.clearRetainingCapacity();
                    self.binned_source.clearRetainingCapacity();
                    return .{
                        .counts = [_]usize{0} ** 6,
                        .offsets = [_]usize{0} ** 6,
                    };
                };
            } else {
                self.binned_meshes.shrinkRetainingCapacity(total);
                self.binned_source.shrinkRetainingCapacity(total);
            }

            for (0..6) |b| {
                var cur = offsets[b];
                for (0..chunk_count) |c| {
                    chunk_offsets[c][b] = cur;
                    cur += chunk_counts[c][b];
                }
            }

            pass.out_binned = self.binned_meshes.items;
            pass.out_source = self.binned_source.items;
            p.forkJoin(ParallelShadowBinning, &pass, ParallelShadowBinning.scatterChunkRange, chunk_count);

            return .{ .counts = counts, .offsets = offsets };
        }

        // Serial fallback
        for (meshes) |mesh| {
            if (!mesh.cast_shadows or mesh.is_lod_child or mesh.is_decal) continue;
            counts[@intFromEnum(bucketFor(mesh))] += 1;
        }

        var total: usize = 0;
        var cursors: [6]usize = undefined;
        for (0..6) |b| {
            offsets[b] = total;
            cursors[b] = total;
            total += counts[b];
        }

        if (total > self.binned_meshes.items.len) {
            // Атомарность публикации: см. выше — старые указатели не
            // удерживаются, prepare ничего из них не читает.
            self.binned_meshes.resize(self.allocator, total) catch {
                self.binned_meshes.clearRetainingCapacity();
                self.binned_source.clearRetainingCapacity();
                return .{
                    .counts = [_]usize{0} ** 6,
                    .offsets = [_]usize{0} ** 6,
                };
            };
            self.binned_source.resize(self.allocator, total) catch {
                self.binned_meshes.clearRetainingCapacity();
                self.binned_source.clearRetainingCapacity();
                return .{
                    .counts = [_]usize{0} ** 6,
                    .offsets = [_]usize{0} ** 6,
                };
            };
        } else {
            self.binned_meshes.shrinkRetainingCapacity(total);
            self.binned_source.shrinkRetainingCapacity(total);
        }

        for (meshes, 0..) |mesh, src_idx| {
            if (!mesh.cast_shadows or mesh.is_lod_child or mesh.is_decal) continue;
            const b = @intFromEnum(bucketFor(mesh));
            self.binned_meshes.items[cursors[b]] = mesh;
            self.binned_source.items[cursors[b]] = @intCast(src_idx);
            cursors[b] += 1;
        }

        return .{ .counts = counts, .offsets = offsets };
    }

    /// Core prepare algorithm, parameterized by the destination payload: bins
    /// meshes into the pass-owned `binned_meshes` scratch, then snapshots
    /// items + skin copies into `out`. `out` may be the standalone
    /// `prepared` (via `prepare`) or a Scene double-buffer slot (P7) — the
    /// algorithm runs once here, never duplicated. OOM semantics preserved:
    /// growth failure publishes a coherent-empty snapshot (items + skins +
    /// zero counts, never stale), skin-copy failure flags the single item
    /// gpu_pending without shifting bucket layout.
    ///
    /// `cache_key` tags the world-matrix/AABB cache (fallback: `Scene.frame_id`,
    /// game build: build-unique `(build_seq | (1<<63))`); `instance_source`
    /// selects the instanced state (`.published` = fallback `instance_render`,
    /// `.build_view` = game-frozen provisional). No new caches, no
    /// invalidation change.
    pub fn prepareInto(
        self: *ShadowPass,
        out: *PreparedShadowDraws,
        meshes: []const *Mesh,
        cache_key: u64,
        instance_source: @import("../mesh.zig").InstanceSource,
        pool: ?*jobs.Pool,
    ) BinResult {
        for (meshes) |m| _ = m.ensureUid();
        const binned = self.binMeshes(meshes, pool);
        const total = self.binned_meshes.items.len;
        if (total > out.items.items.len) {
            out.items.resize(self.allocator, total) catch {
                // Атомарность публикации (P4): рост не удался — пустой
                // coherent-снимок (предметы + скины + counts), а не stale-items
                // при очищенных скинах. Следующий кадр строится заново.
                out.items.clearRetainingCapacity();
                out.skins.clearRetainingCapacity();
                out.bin = .{
                    .counts = [_]usize{0} ** 6,
                    .offsets = [_]usize{0} ** 6,
                };
                return out.bin;
            };
        } else {
            out.items.shrinkRetainingCapacity(total);
        }
        out.skins.clearRetainingCapacity();

        for (self.binned_meshes.items, 0..) |mesh, idx| {
            const is_inst = mesh.instances.items.len > 0;
            // P5: instanced meshes read the frame's staged render state
            // (staged before this prepare); regular meshes use the fresh
            // world cache. No live instance/game-cache reads here.
            // Stage-2B: staged state resolves via instance_source
            // (fallback `.published`, game build `.build_view` provisional).
            const staged = mesh.instanceRenderSource(instance_source).*;
            const aabb_w = if (!is_inst) scene_render_queue.worldAABBCached(cache_key, mesh) else staged.bounds;
            const model = if (!is_inst) scene_render_queue.worldMatrixCached(cache_key, mesh) else Mat4.identity;
            // Копия скина в render-owned хранилище (prepare-фаза). Раскладка
            // items обязана оставаться 1:1 с binned_meshes (бакеты
            // режутся по counts/offsets), поэтому OOM помечает item флагом
            // gpu_pending — renderBuckets его пропускает, но слайсы бакетов
            // не съезжают. Живые матрицы и неверная поза исключены.
            var skin_index: ?u32 = null;
            var skin_oom = false;
            if (mesh.skeleton) |skel| {
                const src = skel.getRenderSkinMatrices();
                out.skins.ensureUnusedCapacity(self.allocator, 1) catch {
                    skin_oom = true;
                };
                if (!skin_oom) {
                    skin_index = @intCast(out.skins.items.len);
                    out.skins.appendAssumeCapacity(src.*);
                }
            }
            const ext = aabb_w.extents();
            const max_dim = @max(ext.x, @max(ext.y, ext.z));

            out.items.items[idx] = ShadowDrawItem{
                .vertex_buffer = mesh.vertex_buffer,
                .index_buffer = mesh.index_buffer,
                .index_count = mesh.index_count,
                .instance_buffer = staged.buffer,
                .visible_instance_count = staged.count,
                .model = model,
                .world_aabb = aabb_w,
                .max_dim = max_dim,
                .skin_index = skin_index,
                .bucket = bucketFor(mesh),
                .is_instanced = is_inst,
                .gpu_pending = mesh.gpu_pending or skin_oom,
                .is_visible = mesh.is_visible,
                .source_uid = mesh.uid,
                .source_mesh = self.binned_source.items[idx],
            };
        }

        out.bin = binned;
        return binned;
    }

    pub fn prepare(
        self: *ShadowPass,
        meshes: []const *Mesh,
        cache_key: u64,
        instance_source: @import("../mesh.zig").InstanceSource,
        pool: ?*jobs.Pool,
    ) BinResult {
        return self.prepareInto(&self.prepared, meshes, cache_key, instance_source, pool);
    }

    /// Core shadow render, parameterized by a prepared payload (standalone
    /// `prepared` or a Scene P7 slot). Reads only the payload's const
    /// snapshot plus the pass pipelines.
    pub fn renderPreparedFrom(
        self: *ShadowPass,
        prepared: *const PreparedShadowDraws,
        cascades: [4]Mat4,
        spot_shadows: []const SpotShadowRenderInfo,
        point_shadows: []const PointShadowRenderInfo,
    ) u32 {
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

        const CASCADE_RES: i32 = @intCast(SHADOW_ATLAS_SIZE / 2);
        var draw_calls: u32 = 0;
        // Kept across cascades: identical re-applies are skipped.
        var last_pipeline_id: u32 = 0;

        for (0..4) |c_idx| {
            const light_view_proj = cascades[c_idx];
            const vx: i32 = if (c_idx % 2 == 1) CASCADE_RES else 0;
            const vy: i32 = if (c_idx >= 2) CASCADE_RES else 0;

            sg.applyViewport(vx, vy, CASCADE_RES, CASCADE_RES, false);
            sg.applyScissorRect(vx, vy, CASCADE_RES, CASCADE_RES, false);

            const c_frustum = math.Frustum.fromViewProjection(light_view_proj);
            self.renderBuckets(prepared, light_view_proj, c_frustum, c_idx, &last_pipeline_id, &draw_calls);
        }

        sg.endPass();

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
                const vx: i32 = if (spot_info.spot_index == 0) 0 else SPOT_SHADOW_RES;
                sg.applyViewport(vx, 0, SPOT_SHADOW_RES, SPOT_SHADOW_RES, false);
                sg.applyScissorRect(vx, 0, SPOT_SHADOW_RES, SPOT_SHADOW_RES, false);

                const spot_frustum = math.Frustum.fromViewProjection(spot_info.view_proj);
                self.renderBuckets(prepared, spot_info.view_proj, spot_frustum, null, &spot_last_pipeline_id, &draw_calls);
            }

            sg.endPass();
            self.spot_needs_clear = false;
        }

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
                sg.applyViewport(point_info.tile_x, point_info.tile_y, POINT_SHADOW_RES, POINT_SHADOW_RES, false);
                sg.applyScissorRect(point_info.tile_x, point_info.tile_y, POINT_SHADOW_RES, POINT_SHADOW_RES, false);

                const point_frustum = math.Frustum.fromViewProjection(point_info.view_proj);
                self.renderBuckets(prepared, point_info.view_proj, point_frustum, null, &point_last_pipeline_id, &draw_calls);
            }

            sg.endPass();
            self.point_needs_clear = false;
        }

        return draw_calls;
    }

    pub fn renderPrepared(
        self: *ShadowPass,
        cascades: [4]Mat4,
        spot_shadows: []const SpotShadowRenderInfo,
        point_shadows: []const PointShadowRenderInfo,
    ) u32 {
        return self.renderPreparedFrom(&self.prepared, cascades, spot_shadows, point_shadows);
    }

    pub fn render(
        self: *ShadowPass,
        meshes: []const *Mesh,
        cache_key: u64,
        instance_source: @import("../mesh.zig").InstanceSource,
        cascades: [4]Mat4,
        spot_shadows: []const SpotShadowRenderInfo,
        pool: ?*jobs.Pool,
    ) u32 {
        _ = self.prepare(meshes, cache_key, instance_source, pool);
        return self.renderPrepared(cascades, spot_shadows, &.{});
    }

    pub fn deinit(self: *ShadowPass) void {
        self.binned_meshes.deinit(self.allocator);
        self.binned_source.deinit(self.allocator);
        self.prepared.deinit(self.allocator);
        sg.destroyPipeline(self.pipeline_u16);
        sg.destroyPipeline(self.pipeline_u32);
        sg.destroyPipeline(self.inst_pipeline_u16);
        sg.destroyPipeline(self.inst_pipeline_u32);
        sg.destroyPipeline(self.skinned_pipeline_u16);
        sg.destroyPipeline(self.skinned_pipeline_u32);
        if (self.shadow_shader.id != 0) sg.destroyShader(self.shadow_shader);
        if (self.inst_shader.id != 0) sg.destroyShader(self.inst_shader);
        if (self.skinned_shader.id != 0) sg.destroyShader(self.skinned_shader);
        self.shadow_shader = .{};
        self.inst_shader = .{};
        self.skinned_shader = .{};
        sg.destroyView(self.attachment_view);
        sg.destroyView(self.texture_view);
        sg.destroyView(self.spot_attachment_view);
        sg.destroyView(self.spot_texture_view);
        sg.destroyView(self.point_attachment_view);
        sg.destroyView(self.point_texture_view);
        sg.destroySampler(self.sampler);
        sg.destroySampler(self.depth_sampler);
        sg.destroyImage(self.image);
        sg.destroyImage(self.spot_image);
        sg.destroyImage(self.point_image);
    }
};

test "parallel shadow binning produces serial-identical results" {
    const ally = std.testing.allocator;
    const count = 300;
    const meshes = try ally.alloc(Mesh, count);
    defer ally.free(meshes);
    const ptrs = try ally.alloc(*Mesh, count);
    defer ally.free(ptrs);

    for (0..count) |i| {
        meshes[i] = Mesh{
            .name = "m",
            .vertex_buffer = .{},
            .index_buffer = .{},
            .index_count = 3,
            .index_type = if (i % 3 == 0) .UINT32 else .UINT16,
            .cast_shadows = (i % 5 != 0),
            .is_lod_child = (i % 11 == 0),
            .is_decal = (i % 13 == 0),
        };
        ptrs[i] = &meshes[i];
    }

    var pass_s = ShadowPass{
        .allocator = ally,
        .binned_meshes = .empty,
        .image = .{},
        .attachment_view = .{},
        .texture_view = .{},
        .sampler = .{},
        .depth_sampler = .{},
        .spot_image = .{},
        .spot_attachment_view = .{},
        .spot_texture_view = .{},
        .spot_needs_clear = false,
        .point_image = .{},
        .point_attachment_view = .{},
        .point_texture_view = .{},
        .point_needs_clear = false,
        .pipeline_u16 = .{},
        .pipeline_u32 = .{},
        .inst_pipeline_u16 = .{},
        .inst_pipeline_u32 = .{},
        .skinned_pipeline_u16 = .{},
        .skinned_pipeline_u32 = .{},
        .shadow_shader = .{},
        .inst_shader = .{},
        .skinned_shader = .{},
    };
    defer pass_s.binned_meshes.deinit(ally);
    defer pass_s.binned_source.deinit(ally);

    var pass_p = ShadowPass{
        .allocator = ally,
        .binned_meshes = .empty,
        .image = .{},
        .attachment_view = .{},
        .texture_view = .{},
        .sampler = .{},
        .depth_sampler = .{},
        .spot_image = .{},
        .spot_attachment_view = .{},
        .spot_texture_view = .{},
        .spot_needs_clear = false,
        .point_image = .{},
        .point_attachment_view = .{},
        .point_texture_view = .{},
        .point_needs_clear = false,
        .pipeline_u16 = .{},
        .pipeline_u32 = .{},
        .inst_pipeline_u16 = .{},
        .inst_pipeline_u32 = .{},
        .skinned_pipeline_u16 = .{},
        .skinned_pipeline_u32 = .{},
        .shadow_shader = .{},
        .inst_shader = .{},
        .skinned_shader = .{},
    };
    defer pass_p.binned_meshes.deinit(ally);
    defer pass_p.binned_source.deinit(ally);

    // Serial
    const res_s = pass_s.binMeshes(ptrs, null);

    // Parallel
    const pool = try jobs.Pool.init(ally, 2);
    defer pool.deinit();
    const res_p = pass_p.binMeshes(ptrs, pool);

    // Verify
    try std.testing.expectEqual(res_s.counts, res_p.counts);
    try std.testing.expectEqual(res_s.offsets, res_p.offsets);
    try std.testing.expectEqual(pass_s.binned_meshes.items.len, pass_p.binned_meshes.items.len);
    try std.testing.expect(pass_s.binned_meshes.items.len > 0);

    for (pass_s.binned_meshes.items, pass_p.binned_meshes.items) |m_s, m_p| {
        try std.testing.expectEqual(m_s, m_p);
    }
}

test "shadow item culling skips regular items outside the light frustum" {
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
    try std.testing.expect(!ShadowPass.shadowItemCulled(inside, frustum, 0));

    inside.world_aabb = outside_aabb;
    try std.testing.expect(ShadowPass.shadowItemCulled(inside, frustum, 0));
}

test "shadow item culling applies batch AABB and instance checks to instanced items" {
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
    try std.testing.expect(ShadowPass.shadowItemCulled(item, frustum, 0));

    item.world_aabb = inside_aabb;
    try std.testing.expect(!ShadowPass.shadowItemCulled(item, frustum, 0));

    item.visible_instance_count = 0;
    try std.testing.expect(ShadowPass.shadowItemCulled(item, frustum, 0));

    item.visible_instance_count = 4;
    item.instance_buffer = .{};
    try std.testing.expect(ShadowPass.shadowItemCulled(item, frustum, 0));
}

test "shadow item culling applies far-cascade max_dim policy to all items" {
    const frustum = math.Frustum.fromViewProjection(Mat4.identity);
    const inside_aabb = math.BoundingBox.init(
        math.Vec3.new(-0.5, -0.5, 0.2),
        math.Vec3.new(0.5, 0.5, 0.8),
    );

    const item = ShadowPass.ShadowDrawItem{
        .world_aabb = inside_aabb,
        .max_dim = 0.5,
    };
    try std.testing.expect(ShadowPass.shadowItemCulled(item, frustum, 3));
    try std.testing.expect(!ShadowPass.shadowItemCulled(item, frustum, 0));

    const inst = ShadowPass.ShadowDrawItem{
        .world_aabb = inside_aabb,
        .max_dim = 0.5,
        .is_instanced = true,
        .visible_instance_count = 4,
        .instance_buffer = .{ .id = 1 },
    };
    try std.testing.expect(ShadowPass.shadowItemCulled(inst, frustum, 3));
    try std.testing.expect(!ShadowPass.shadowItemCulled(inst, frustum, 0));
}

test "shadow item culling skips invisible items" {
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
    try std.testing.expect(ShadowPass.shadowItemCulled(hidden, frustum, 0));

    const hidden_inst = ShadowPass.ShadowDrawItem{
        .world_aabb = inside_aabb,
        .max_dim = 1.0,
        .is_visible = false,
        .is_instanced = true,
        .visible_instance_count = 4,
        .instance_buffer = .{ .id = 1 },
    };
    try std.testing.expect(ShadowPass.shadowItemCulled(hidden_inst, frustum, 0));
}

// ---- P4 render-owned draw snapshot: регрессия владения. ----

const SkeletonForP4 = @import("../animation/skeleton.zig").Skeleton;

fn testShadowPass(ally: std.mem.Allocator) ShadowPass {
    return .{
        .allocator = ally,
        .binned_meshes = .empty,
        .image = .{},
        .attachment_view = .{},
        .texture_view = .{},
        .sampler = .{},
        .depth_sampler = .{},
        .spot_image = .{},
        .spot_attachment_view = .{},
        .spot_texture_view = .{},
        .spot_needs_clear = false,
        .point_image = .{},
        .point_attachment_view = .{},
        .point_texture_view = .{},
        .point_needs_clear = false,
        .pipeline_u16 = .{},
        .pipeline_u32 = .{},
        .inst_pipeline_u16 = .{},
        .inst_pipeline_u32 = .{},
        .skinned_pipeline_u16 = .{},
        .skinned_pipeline_u32 = .{},
        .shadow_shader = .{},
        .inst_shader = .{},
        .skinned_shader = .{},
    };
}

// Подготовленный shadow-item не ссылается на живые данные: модель и копия
// скина пережили мутацию TRS и две публикации скелета.
test "P4: shadow item owns model and skin snapshots" {
    const ally = std.testing.allocator;
    const skel = try SkeletonForP4.init(ally, 1);
    defer skel.deinit();
    skel.bones[0].local_position = math.Vec3.new(1, 0, 0);
    skel.update();

    const unit_box = math.BoundingBox.init(math.Vec3.new(-0.5, -0.5, -0.5), math.Vec3.new(0.5, 0.5, 0.5));
    var mesh = Mesh{
        .name = "shadow_skinned",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .position = math.Vec3.new(3, 0, 0),
        .local_bounding_box = unit_box,
        .skeleton = skel,
    };
    var pass = testShadowPass(ally);
    defer pass.binned_meshes.deinit(ally);
    defer pass.binned_source.deinit(ally);
    defer pass.prepared.deinit(ally);

    const meshes = [_]*Mesh{&mesh};
    _ = pass.prepare(&meshes, 5, .published, null);
    try std.testing.expectEqual(@as(usize, 1), pass.prepared.items.items.len);
    try std.testing.expectEqual(@as(usize, 1), pass.prepared.skins.items.len);
    const it = pass.prepared.items.items[0];
    try std.testing.expect(it.skin_index != null);

    mesh.position = math.Vec3.new(99, 99, 99);
    skel.bones[0].local_position = math.Vec3.new(5, 0, 0);
    skel.update();
    skel.update();
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), skel.getRenderSkinMatrices()[0].m[12], 1e-4);

    try std.testing.expectApproxEqAbs(@as(f32, 3.0), pass.prepared.items.items[0].model.m[12], 1e-4);
    const bones = pass.prepared.skins.items[pass.prepared.items.items[0].skin_index.?];
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), bones[0].m[12], 1e-4);
}

// Точечный OOM-аллокатор: падает ровно на n-й операции роста. Рост через
// remap и его fallback alloc+copy считаются ОДНОЙ операцией (флаги armed и
// coalesce): иначе число vtable-вызовов на рост зависит от того, смог ли
// backing-аллокатор расширить in-place, и стадия OOM неупорядочена.
// From-empty рост std сводит к alloc и ловится счётчиком напрямую.
const FailNthP4 = struct {
    backing: std.mem.Allocator,
    fail_on: usize,
    count: usize = 0,
    armed: bool = false,
    coalesce: bool = false,

    fn allocator(self: *FailNthP4) std.mem.Allocator {
        return .{
            .ptr = self,
            .vtable = &.{
                .alloc = allocFn,
                .resize = resizeFn,
                .remap = remapFn,
                .free = freeFn,
            },
        };
    }

    fn allocFn(ctx: *anyopaque, len: usize, alignment: std.mem.Alignment, ra: usize) ?[*]u8 {
        const self: *FailNthP4 = @ptrCast(@alignCast(ctx));
        if (self.armed) {
            self.armed = false;
            return null;
        }
        if (self.coalesce) {
            // Fallback после естественной неудачи remap: часть той же
            // операции роста, счётчик не тратится.
            self.coalesce = false;
            return self.backing.rawAlloc(len, alignment, ra);
        }
        self.count += 1;
        if (self.count == self.fail_on) return null;
        return self.backing.rawAlloc(len, alignment, ra);
    }

    fn resizeFn(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ra: usize) bool {
        const self: *FailNthP4 = @ptrCast(@alignCast(ctx));
        return self.backing.rawResize(memory, alignment, new_len, ra);
    }

    fn remapFn(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ra: usize) ?[*]u8 {
        const self: *FailNthP4 = @ptrCast(@alignCast(ctx));
        // Рост существующих буферов (ArrayList идёт через remap) — операция
        // роста; ужатие (shrink) пропускается без счёта.
        // From-empty рост std сводит к alloc и ловится выше.
        if (new_len > memory.len) {
            self.count += 1;
            if (self.count == self.fail_on) {
                self.armed = true;
                return null;
            }
            const res = self.backing.rawRemap(memory, alignment, new_len, ra);
            if (res == null) self.coalesce = true;
            return res;
        }
        return self.backing.rawRemap(memory, alignment, new_len, ra);
    }

    fn freeFn(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, ra: usize) void {
        const self: *FailNthP4 = @ptrCast(@alignCast(ctx));
        return self.backing.rawFree(memory, alignment, ra);
    }
};

// OOM копии скина: раскладка бакетов остаётся 1:1 (item на месте, помечен
// gpu_pending и пропускается рендером), нескinned-сосед рисуется как обычно.
test "P4: shadow skin OOM keeps bucket layout and skips the item" {
    const ally = std.testing.allocator;
    const skel = try SkeletonForP4.init(ally, 1);
    defer skel.deinit();
    skel.bones[0].local_position = math.Vec3.new(1, 0, 0);
    skel.update();

    const unit_box = math.BoundingBox.init(math.Vec3.new(-0.5, -0.5, -0.5), math.Vec3.new(0.5, 0.5, 0.5));
    var skinned = Mesh{
        .name = "oom_skinned",
        .vertex_buffer = .{},
        .index_buffer = .{},
        // Маркер skinned-бакета: очереди не несут живых указателей.
        .index_count = 9,
        .local_bounding_box = unit_box,
        .skeleton = skel,
    };
    var plain = Mesh{
        .name = "oom_plain",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = unit_box,
    };
    const meshes = [_]*Mesh{ &skinned, &plain };

    // Аллокации свежего prepare: 1) binned_meshes.resize, 2) binned_source.resize,
    // 3) prepared.items.resize, 4) prepared.skins.ensure для skinned-меша.
    // Роняем четвёртую (stage-2A добавил binned_source как вторую).
    var limited = FailNthP4{ .backing = ally, .fail_on = 4 };
    var pass = testShadowPass(limited.allocator());
    defer pass.binned_meshes.deinit(limited.allocator());
    defer pass.binned_source.deinit(limited.allocator());
    defer pass.prepared.deinit(limited.allocator());

    const res = pass.prepare(&meshes, 9, .published, null);
    try std.testing.expectEqual(pass.binned_meshes.items.len, pass.prepared.items.items.len);
    try std.testing.expectEqual(@as(usize, 2), pass.prepared.items.items.len);
    try std.testing.expectEqual(@as(usize, 0), pass.prepared.skins.items.len);
    // Counts бакетов обязаны оставаться внутри длин (иначе renderBuckets
    // срежет за границу).
    var total: usize = 0;
    for (res.counts) |c| total += c;
    try std.testing.expectEqual(@as(usize, 2), total);
    for (res.counts, res.offsets) |c, o| try std.testing.expect(o + c <= pass.prepared.items.items.len);

    // Явная побакетная проверка: skinned-item помечен gpu_pending (рендер его
    // пропустит, в нескinned-позе он НЕ рисуется), plain-item чист.
    var skinned_seen = false;
    var plain_seen = false;
    for (pass.prepared.items.items) |it| {
        if (it.index_count == 9) {
            skinned_seen = true;
            try std.testing.expect(it.gpu_pending);
            try std.testing.expect(it.skin_index == null);
        } else {
            plain_seen = true;
            try std.testing.expect(!it.gpu_pending);
            try std.testing.expect(it.skin_index == null);
            try std.testing.expect(it.is_visible);
        }
    }
    try std.testing.expect(skinned_seen and plain_seen);
}

// Атомарность публикации при OOM роста: успешный skinned-prepare, затем
// вынужденный рост на каждой fallible-стадии. Возвращённые/last counts —
// пустой coherent-снимок БЕЗ чтения старых источников (один из них —
// freed-heap-меш, модель P3-destroyMesh между кадрами); binned-указатели,
// предметы, скины — всё пусто. Следующий кадр восстанавливается полностью.
test "P4: shadow prepare growth OOM publishes empty snapshot and recovers" {
    const ally = std.testing.allocator;
    const skel = try SkeletonForP4.init(ally, 1);
    defer skel.deinit();
    skel.bones[0].local_position = math.Vec3.new(1, 0, 0);
    skel.update();

    const unit_box = math.BoundingBox.init(math.Vec3.new(-0.5, -0.5, -0.5), math.Vec3.new(0.5, 0.5, 0.5));
    var skinned = [8]Mesh{ undefined, undefined, undefined, undefined, undefined, undefined, undefined, undefined };
    for (&skinned, 0..) |*m, i| {
        m.* = Mesh{
            .name = "grow_skinned",
            .vertex_buffer = .{},
            .index_buffer = .{},
            .index_count = 9 + @as(u32, @intCast(i)),
            .local_bounding_box = unit_box,
            .skeleton = skel,
        };
    }
    var plains: [88]Mesh = undefined;
    for (&plains) |*m| {
        m.* = Mesh{
            .name = "grow_plain",
            .vertex_buffer = .{},
            .index_buffer = .{},
            .index_count = 3,
            .local_bounding_box = unit_box,
        };
    }
    // 96 мешей (8 skinned + 88 plains): заведомо больше любой стартовой
    // ёмкости кадра N (формула роста зависит от cache_line платформы —
    // stage-2A: u32-binned_source стартует с 19..35 слотов, поэтому 32 мешей
    // уже недостаточно), рост всех трёх таблиц (binned/binned_source/items)
    // гарантирован.
    var many: [96]*Mesh = undefined;
    for (&skinned, 0..) |*m, i| many[i] = m;
    for (&plains, 0..) |*m, i| many[8 + i] = m;

    var pass = testShadowPass(ally);
    defer pass.binned_meshes.deinit(ally);
    defer pass.binned_source.deinit(ally);
    defer pass.prepared.deinit(ally);

    // Успешный SKINNED-кадр N: heap-меш + skinned-меш. Очередь и skin storage
    // непусты, skinned-item валиден. Heap-меш затем уничтожается (модель
    // P3-destroyMesh между кадрами) — следующий prepare не должен его читать.
    // Кадр N+1 из 32 мешей гарантированно требует роста обеих таблиц.
    const heap_mesh = try ally.create(Mesh);
    heap_mesh.* = Mesh{
        .name = "grow_heap_plain",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = unit_box,
    };
    const frame_n = [_]*Mesh{ heap_mesh, &skinned[0] };
    const ok = pass.prepare(&frame_n, 1, .published, null);
    var ok_total: usize = 0;
    for (ok.counts) |c| ok_total += c;
    try std.testing.expectEqual(@as(usize, 2), ok_total);
    try std.testing.expectEqual(@as(usize, 2), pass.prepared.items.items.len);
    try std.testing.expectEqual(@as(usize, 1), pass.prepared.skins.items.len);

    // destroyMesh между кадрами: указатель в binned_meshes висячий до
    // следующего binMeshes; prepare при OOM не должен его разыменовывать.
    ally.destroy(heap_mesh);

    // Стадия A: падает рост binned_meshes — пустые counts И пустой binned-список:
    // prepare не читает ни freed-heap-меш, ни остальные старые источники.
    var lim_a = FailNthP4{ .backing = ally, .fail_on = 1 };
    pass.allocator = lim_a.allocator();
    const ra = pass.prepare(&many, 2, .published, null);
    for (ra.counts) |c| try std.testing.expectEqual(@as(usize, 0), c);
    for (pass.prepared.bin.counts) |c| try std.testing.expectEqual(@as(usize, 0), c);
    try std.testing.expectEqual(@as(usize, 0), pass.binned_meshes.items.len);
    try std.testing.expectEqual(@as(usize, 0), pass.binned_source.items.len);
    try std.testing.expectEqual(@as(usize, 0), pass.prepared.items.items.len);
    try std.testing.expectEqual(@as(usize, 0), pass.prepared.skins.items.len);

    // Стадия B: binned вырос (оба списка), падает рост prepared.items — пустой
    // снимок: ни предметов, ни скинов, counts нулевые. (Stage-2A: binned_source
    // как вторая аллокация, поэтому fail_on=3.)
    var lim_b = FailNthP4{ .backing = ally, .fail_on = 3 };
    pass.allocator = lim_b.allocator();
    const rb = pass.prepare(&many, 3, .published, null);
    for (rb.counts) |c| try std.testing.expectEqual(@as(usize, 0), c);
    for (pass.prepared.bin.counts) |c| try std.testing.expectEqual(@as(usize, 0), c);
    try std.testing.expectEqual(@as(usize, 0), pass.prepared.items.items.len);
    try std.testing.expectEqual(@as(usize, 0), pass.prepared.skins.items.len);

    // Восстановление следующим кадром с рабочим аллокатором.
    pass.allocator = ally;
    const rc = pass.prepare(&many, 4, .published, null);
    var rc_total: usize = 0;
    for (rc.counts) |c| rc_total += c;
    try std.testing.expectEqual(@as(usize, 96), rc_total);
    try std.testing.expectEqual(@as(usize, 96), pass.prepared.items.items.len);
    try std.testing.expectEqual(@as(usize, 8), pass.prepared.skins.items.len);
    for (rc.counts, rc.offsets) |c, o| try std.testing.expect(o + c <= pass.prepared.items.items.len);
    for (pass.prepared.items.items) |it| {
        if (it.skin_index) |s| {
            try std.testing.expect(s < pass.prepared.skins.items.len);
            try std.testing.expect(!it.gpu_pending);
        }
    }

    // Стадия C: та же атомарность через parallel-ветку binMeshes (порог 128;
    // stage-2A: 200 мешей, 20 skinned — заведомо больше retained-ёмкостей
    // 96-кадра выше, рост binned_meshes гарантирован). OOM роста
    // binned_meshes — пустые counts И пустой binned-список: старые указатели
    // (включая freed-heap кадра N) не читаются.
    const pool = try jobs.Pool.init(ally, 2);
    defer pool.deinit();
    const big_meshes = try ally.alloc(Mesh, 200);
    defer ally.free(big_meshes);
    const big_ptrs = try ally.alloc(*Mesh, 200);
    defer ally.free(big_ptrs);
    for (big_meshes, 0..) |*m, i| {
        m.* = if (i < 20) skinned[i % 8] else plains[0];
        big_ptrs[i] = m;
    }
    var lim_c = FailNthP4{ .backing = ally, .fail_on = 1 };
    pass.allocator = lim_c.allocator();
    const rd = pass.prepare(big_ptrs, 5, .published, pool);
    for (rd.counts) |c| try std.testing.expectEqual(@as(usize, 0), c);
    for (pass.prepared.bin.counts) |c| try std.testing.expectEqual(@as(usize, 0), c);
    try std.testing.expectEqual(@as(usize, 0), pass.binned_meshes.items.len);
    try std.testing.expectEqual(@as(usize, 0), pass.binned_source.items.len);
    try std.testing.expectEqual(@as(usize, 0), pass.prepared.items.items.len);
    try std.testing.expectEqual(@as(usize, 0), pass.prepared.skins.items.len);

    // Финальное восстановление parallel-кадром: все 200 на месте.
    pass.allocator = ally;
    const re = pass.prepare(big_ptrs, 6, .published, pool);
    var re_total: usize = 0;
    for (re.counts) |c| re_total += c;
    try std.testing.expectEqual(@as(usize, 200), re_total);
    try std.testing.expectEqual(@as(usize, 200), pass.prepared.items.items.len);
    try std.testing.expectEqual(@as(usize, 20), pass.prepared.skins.items.len);
    for (re.counts, re.offsets) |c, o| try std.testing.expect(o + c <= pass.prepared.items.items.len);
}

// ---- P5 instance staging ownership: читатели одного published state. ----

// Shadow/main/outline обязаны читать одно и то же опубликованное состояние
// кадра (bounds/count/handle) — ни прошлокадровых значений, ни живых
// instance/game-кешей. Без GPU-контекста хендлы пустые, но равенство
// источников и точные count/bounds ловят рассинхрон читателей.
test "P5: shadow, main batch and outline read identical published state" {
    const ally = std.testing.allocator;
    const Vec3 = math.Vec3;
    const BoundingBox = math.BoundingBox;

    var src = Mesh{
        .name = "shared_src",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    var inst0 = InstancedMeshForP5{ .name = "s0", .source_mesh = &src, .position = Vec3.new(0, 0, 0) };
    var inst1 = InstancedMeshForP5{ .name = "s1", .source_mesh = &src, .position = Vec3.new(6, 0, 0) };
    var inst2 = InstancedMeshForP5{ .name = "s2", .source_mesh = &src, .position = Vec3.new(12, 0, 0), .is_visible = false };
    var ptrs = [_]*InstancedMeshForP5{ &inst0, &inst1, &inst2 };
    var parent = Mesh{
        .name = "shared_parent",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .instances = std.ArrayListUnmanaged(*InstancedMeshForP5){ .items = &ptrs, .capacity = 3 },
    };
    const meshes = [_]*Mesh{&parent};

    // Pre-stage кадра (как Scene.prepareFrame до shadow prepare).
    var stage_queues = scene_render_queue.RenderQueues{};
    defer stage_queues.deinit(ally);
    instance_staging.stageInstances(.{
        .allocator = ally,
        .instance_matrices = &stage_queues.instance_matrices,
        .thread_pool = null,
        .frame_id = 61,
        .eye = Vec3.zero,
    }, &meshes);
    try std.testing.expectEqual(@as(u32, 2), parent.instance_render.count);
    try std.testing.expectEqual(@as(u64, 61), parent.instance_render.staged_frame);

    // Shadow-читатель.
    var pass = testShadowPass(ally);
    defer pass.binned_meshes.deinit(ally);
    defer pass.binned_source.deinit(ally);
    defer pass.prepared.deinit(ally);
    _ = pass.prepare(&meshes, 61, .published, null);
    try std.testing.expectEqual(@as(usize, 1), pass.prepared.items.items.len);
    const shadow_item = pass.prepared.items.items[0];
    try std.testing.expect(shadow_item.is_instanced);

    // Main-читатель (view queue batch).
    var queues = scene_render_queue.RenderQueues{};
    defer queues.deinit(ally);
    var stats = @import("../scene/stats.zig").SceneStats{};
    var culler = @import("../visibility/mod.zig").OcclusionCuller.init();
    scene_render_queue.buildFrameQueues(.{
        .allocator = ally,
        .meshes = &meshes,
        .cache_key = 61,
        .view_proj = Mat4.identity,
        .eye = Vec3.zero,
        .cull_frustum = false,
        .cull_occlusion = false,
        .occlusion_culler = &culler,
        .stats = &stats,
        .queues = &queues,
        .default_white_id = 1,
    });
    try std.testing.expectEqual(@as(usize, 1), queues.opaque_instanced.items.len);
    const batch = queues.opaque_instanced.items[0];

    // Outline-читатель.
    var skins = scene_render_queue.SkinStorage.empty;
    defer skins.deinit(ally);
    const outline_item = outline_pass.makeOutlineDrawItem(ally, &skins, &parent, 61, 0, .published) orelse
        return error.TestUnexpectedResult;

    // Все трое — один count (2 видимых, не 3 всего: count-only по
    // instances.len провалился бы), один хендл, одни границы.
    try std.testing.expectEqual(parent.instance_render.count, shadow_item.visible_instance_count);
    try std.testing.expectEqual(parent.instance_render.count, batch.visible_instance_count);
    try std.testing.expectEqual(parent.instance_render.count, outline_item.visible_instance_count);
    try std.testing.expectEqual(parent.instance_render.buffer.id, shadow_item.instance_buffer.id);
    try std.testing.expectEqual(parent.instance_render.buffer.id, batch.instance_buffer.id);
    try std.testing.expectEqual(parent.instance_render.buffer.id, outline_item.instance_buffer.id);
    try std.testing.expectEqual(parent.instance_render.bounds, shadow_item.world_aabb);
    try std.testing.expectEqual(parent.instance_render.bounds.center(), outline_item.world_center);
    // Точное значение: inst0 [-1,1] + inst1 [5,7] → центр staged границ x=3.
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), outline_item.world_center.x, 1e-4);
}

// ---- P5 revision: сбойный пре-стейдж дефинитивен для кадра. ----

// Transient pre-stage failure (scratch OOM) keeps the previous complete
// publish; the shadow snapshot taken from it must stay coherent with every
// main view even though a retry with a working allocator would succeed —
// Scene view builds (instances_prepared) consume the old state and never
// retry mid-frame. Next frame's successful pre-stage updates all readers.
// The transparent parent additionally pins primary-eye sort order: extra
// views with other eyes must not re-sort.
test "P5: failed pre-stage is definitive — views consume the old snapshot" {
    const ally = std.testing.allocator;
    const Vec3 = math.Vec3;
    const BoundingBox = math.BoundingBox;
    const material = @import("../material.zig");
    var blend_mat = material.StandardMaterial.init("blend");
    blend_mat.alpha_mode = .blend;

    var src = Mesh{
        .name = "defin_src",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    var inst0 = InstancedMeshForP5{ .name = "d0", .source_mesh = &src, .position = Vec3.new(0, 0, 0) };
    var inst1 = InstancedMeshForP5{ .name = "d1", .source_mesh = &src, .position = Vec3.new(6, 0, 0) };
    var ptrs = [_]*InstancedMeshForP5{ &inst0, &inst1 };
    var parent = Mesh{
        .name = "defin_parent",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .material = .{ .standard = &blend_mat },
        .instances = std.ArrayListUnmanaged(*InstancedMeshForP5){ .items = &ptrs, .capacity = 2 },
    };
    const meshes = [_]*Mesh{&parent};

    // Frame 71: successful pre-stage with the primary eye — count 2,
    // transparent sort back-to-front puts the farther inst1 (x=6) first.
    var stage_queues = scene_render_queue.RenderQueues{};
    defer stage_queues.deinit(ally);
    const primary_eye = Vec3.new(-50, 0, 0);
    instance_staging.stageInstances(.{
        .allocator = ally,
        .instance_matrices = &stage_queues.instance_matrices,
        .thread_pool = null,
        .frame_id = 71,
        .eye = primary_eye,
    }, &meshes);
    try std.testing.expectEqual(@as(u32, 2), parent.instance_render.count);
    try std.testing.expectEqual(@as(u64, 71), parent.instance_render.staged_frame);
    try std.testing.expectEqual(@as(usize, 2), stage_queues.instance_matrices.items.len);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), stage_queues.instance_matrices.items[0].m[12], 1e-4);
    const old_bounds = parent.instance_render.bounds;

    // Frame 72: hide inst1, then FAIL the pre-stage (fresh scratch +
    // failing allocator) — the frame-71 publish stays intact.
    inst1.is_visible = false;
    var bare = scene_render_queue.RenderQueues{};
    defer bare.deinit(ally);
    var failing = std.testing.FailingAllocator.init(ally, .{ .fail_index = 0 });
    instance_staging.stageInstances(.{
        .allocator = failing.allocator(),
        .instance_matrices = &bare.instance_matrices,
        .thread_pool = null,
        .frame_id = 72,
        .eye = primary_eye,
    }, &meshes);
    try std.testing.expectEqual(@as(u32, 2), parent.instance_render.count);
    try std.testing.expectEqual(@as(u64, 71), parent.instance_render.staged_frame);

    // Shadow snapshot captures the OLD state (count 2, old bounds).
    var pass = testShadowPass(ally);
    defer pass.binned_meshes.deinit(ally);
    defer pass.binned_source.deinit(ally);
    defer pass.prepared.deinit(ally);
    _ = pass.prepare(&meshes, 72, .published, null);
    try std.testing.expectEqual(@as(usize, 1), pass.prepared.items.items.len);
    const shadow_item = pass.prepared.items.items[0];
    try std.testing.expectEqual(@as(u32, 2), shadow_item.visible_instance_count);
    try std.testing.expectEqual(old_bounds, shadow_item.world_aabb);

    // Main build with a WORKING allocator: instances_prepared (as Scene
    // sets) forbids the mid-frame retry — the batch consumes the same old
    // state the shadow saw, and the view build stages nothing itself.
    var queues = scene_render_queue.RenderQueues{};
    defer queues.deinit(ally);
    var stats = @import("../scene/stats.zig").SceneStats{};
    var culler = @import("../visibility/mod.zig").OcclusionCuller.init();
    scene_render_queue.buildFrameQueues(.{
        .allocator = ally,
        .meshes = &meshes,
        .cache_key = 72,
        .view_proj = Mat4.identity,
        .eye = Vec3.zero,
        .cull_frustum = false,
        .cull_occlusion = false,
        .occlusion_culler = &culler,
        .stats = &stats,
        .queues = &queues,
        .default_white_id = 1,
        .instances_prepared = true,
    });
    try std.testing.expectEqual(@as(usize, 1), queues.transparent_instanced.items.len);
    try std.testing.expectEqual(shadow_item.visible_instance_count, queues.transparent_instanced.items[0].visible_instance_count);
    try std.testing.expectEqual(@as(u64, 71), parent.instance_render.staged_frame);
    try std.testing.expectEqual(@as(usize, 0), queues.instance_matrices.items.len);

    // Extra view with the opposite eye: still no retry, and the pre-stage
    // scratch keeps primary-eye order (inst1 first).
    var queues2 = scene_render_queue.RenderQueues{};
    defer queues2.deinit(ally);
    var stats2 = @import("../scene/stats.zig").SceneStats{};
    scene_render_queue.buildFrameQueues(.{
        .allocator = ally,
        .meshes = &meshes,
        .cache_key = 72,
        .view_proj = Mat4.identity,
        .eye = Vec3.new(50, 0, 0),
        .cull_frustum = false,
        .cull_occlusion = false,
        .occlusion_culler = &culler,
        .stats = &stats2,
        .queues = &queues2,
        .default_white_id = 1,
        .instances_prepared = true,
    });
    try std.testing.expectEqual(@as(u64, 71), parent.instance_render.staged_frame);
    try std.testing.expectEqual(@as(u32, 2), parent.instance_render.count);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), stage_queues.instance_matrices.items[0].m[12], 1e-4);

    // Frame 73: successful pre-stage updates every reader to count 1.
    instance_staging.stageInstances(.{
        .allocator = ally,
        .instance_matrices = &stage_queues.instance_matrices,
        .thread_pool = null,
        .frame_id = 73,
        .eye = primary_eye,
    }, &meshes);
    try std.testing.expectEqual(@as(u32, 1), parent.instance_render.count);
    try std.testing.expectEqual(@as(u64, 73), parent.instance_render.staged_frame);
    _ = pass.prepare(&meshes, 73, .published, null);
    try std.testing.expectEqual(@as(u32, 1), pass.prepared.items.items[0].visible_instance_count);
    try std.testing.expectEqual(parent.instance_render.bounds, pass.prepared.items.items[0].world_aabb);
}

test "stage-2A: shadow items carry source uid and list index" {
    const ally = std.testing.allocator;
    var skipped = Mesh{
        .name = "skip_no_shadow",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .cast_shadows = false,
    };
    var a = Mesh{
        .name = "shadow_a",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
    };
    var b = Mesh{
        .name = "shadow_b",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
    };
    const meshes = [_]*Mesh{ &skipped, &a, &b };
    var pass = testShadowPass(ally);
    defer pass.binned_meshes.deinit(ally);
    defer pass.binned_source.deinit(ally);
    defer pass.prepared.deinit(ally);
    _ = pass.prepare(&meshes, 17, .published, null);
    try std.testing.expectEqual(@as(usize, 2), pass.prepared.items.items.len);
    try std.testing.expectEqual(@as(usize, 2), pass.binned_source.items.len);
    // Binned order is bucket-grouped but source indices map back to the
    // input list (1 and 2; skipped mesh 0 never appears).
    for (pass.prepared.items.items) |it| {
        try std.testing.expect(it.source_uid != 0);
        try std.testing.expect(it.source_mesh == 1 or it.source_mesh == 2);
        if (it.source_mesh == 1) try std.testing.expectEqual(a.uid, it.source_uid);
        if (it.source_mesh == 2) try std.testing.expectEqual(b.uid, it.source_uid);
    }
    try std.testing.expect(a.uid != 0 and b.uid != 0 and a.uid != b.uid);
}

// ---- Point-light shadow atlas: face math + tile bookkeeping. ----

test "pointFaceForDir selects the major-axis face with X>Y>Z tie-break" {
    const V = math.Vec3.new;
    try std.testing.expectEqual(@as(usize, 0), pointFaceForDir(V(1, 0, 0)));
    try std.testing.expectEqual(@as(usize, 1), pointFaceForDir(V(-2, 0.5, 0.5)));
    try std.testing.expectEqual(@as(usize, 2), pointFaceForDir(V(0.1, 3, 0.1)));
    try std.testing.expectEqual(@as(usize, 3), pointFaceForDir(V(0, -1, 0)));
    try std.testing.expectEqual(@as(usize, 4), pointFaceForDir(V(0, 0, 5)));
    try std.testing.expectEqual(@as(usize, 5), pointFaceForDir(V(0.2, 0.1, -4)));
    // Ties prefer X, then Y, then Z (mirrors the GLSL pointFaceIndex).
    try std.testing.expectEqual(@as(usize, 0), pointFaceForDir(V(1, 1, 0)));
    try std.testing.expectEqual(@as(usize, 1), pointFaceForDir(V(-1, 1, 1)));
    try std.testing.expectEqual(@as(usize, 2), pointFaceForDir(V(0, 1, 1)));
    try std.testing.expectEqual(@as(usize, 3), pointFaceForDir(V(0, -1, -1)));
    try std.testing.expectEqual(@as(usize, 4), pointFaceForDir(V(0, 0, 1)));
}

test "pointTileOrigin tiles 2 slots x 6 faces inside the atlas without overlap" {
    try std.testing.expectEqual(@as(u32, 6 * POINT_SHADOW_RES), POINT_SHADOW_MAP_WIDTH);
    try std.testing.expectEqual(@as(u32, 2 * POINT_SHADOW_RES), POINT_SHADOW_MAP_HEIGHT);
    var seen: [POINT_SHADOW_SLOTS * POINT_SHADOW_FACES]@TypeOf(pointTileOrigin(0, 0)) = undefined;
    var n: usize = 0;
    for (0..POINT_SHADOW_SLOTS) |slot| {
        for (0..POINT_SHADOW_FACES) |face| {
            const o = pointTileOrigin(slot, face);
            try std.testing.expectEqual(@as(i32, @intCast(face)) * POINT_SHADOW_RES, o.x);
            try std.testing.expectEqual(@as(i32, @intCast(slot)) * POINT_SHADOW_RES, o.y);
            // Tile stays inside the atlas.
            try std.testing.expect(o.x >= 0 and o.x + POINT_SHADOW_RES <= POINT_SHADOW_MAP_WIDTH);
            try std.testing.expect(o.y >= 0 and o.y + POINT_SHADOW_RES <= POINT_SHADOW_MAP_HEIGHT);
            for (seen[0..n]) |prev| try std.testing.expect(prev.x != o.x or prev.y != o.y);
            seen[n] = o;
            n += 1;
        }
    }
    try std.testing.expectEqual(@as(usize, 12), n);
}
