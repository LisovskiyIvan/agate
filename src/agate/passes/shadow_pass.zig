const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const shadow_shd = @import("shadow_shader");
const math = @import("math");
const Mat4 = math.Mat4;
const mesh_mod = @import("../mesh.zig");
const Mesh = mesh_mod.Mesh;
const scene_render_queue = @import("../scene/render_queue.zig");
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

pub const SpotShadowRenderInfo = struct {
    spot_index: usize = 0,
    view_proj: Mat4 = Mat4.identity,
};

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

    // Pipeline buckets in fixed order; the grouping key is defined once per
    // render so each cascade issues at most one applyPipeline per bucket.
    const Bucket = enum { regular_u16, regular_u32, inst_u16, inst_u32, skinned_u16, skinned_u32 };
    const bucket_order: [6]Bucket = .{ .regular_u16, .regular_u32, .inst_u16, .inst_u32, .skinned_u16, .skinned_u32 };

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

    fn renderBuckets(
        self: *ShadowPass,
        light_view_proj: Mat4,
        frustum: math.Frustum,
        counts: [6]usize,
        offsets: [6]usize,
        frame_id: u64,
        cascade_idx: ?usize,
        last_pipeline_id: *u32,
        draw_calls: *u32,
    ) void {
        for (bucket_order) |bucket| {
            const b_idx = @intFromEnum(bucket);
            const count = counts[b_idx];
            if (count == 0) continue;

            const pip_id = self.pipelineFor(bucket);
            if (pip_id == 0) continue;

            const bucket_meshes = self.binned_meshes.items[offsets[b_idx] .. offsets[b_idx] + count];
            for (bucket_meshes) |mesh| {
                // Deferred-creation meshes (off-context uploadGeometry) have
                // no buffers yet: drawing them would bind invalid handles.
                if (mesh.gpu_pending) continue;
                if (mesh.instances.items.len > 0) {
                    if (mesh.visible_instance_count == 0 or mesh.instance_buffer.id == 0) continue;

                    if (pip_id != last_pipeline_id.*) {
                        sg.applyPipeline(.{ .id = pip_id });
                        last_pipeline_id.* = pip_id;
                    }

                    var bind = sg.Bindings{};
                    bind.vertex_buffers[0] = mesh.vertex_buffer;
                    bind.vertex_buffers[1] = mesh.instance_buffer;
                    bind.index_buffer = mesh.index_buffer;
                    sg.applyBindings(bind);

                    const inst_vs = shadow_shd.VsInstParams{
                        .light_view_proj = light_view_proj,
                    };
                    sg.applyUniforms(shadow_shd.UB_vs_inst_params, sg.asRange(&inst_vs));
                    sg.draw(0, mesh.index_count, mesh.visible_instance_count);
                    draw_calls.* += 1;
                } else {
                    if (!mesh.is_visible) continue;
                    // Cached world transforms, filled here on first touch
                    // and reused by the main pass's queue build (same frame
                    // id). Bone attachment semantics match Mesh.getWorldMatrix.
                    const aabb_w = scene_render_queue.worldAABBCached(frame_id, mesh);
                    if (!frustum.intersectsAABB(aabb_w)) continue;

                    // Far cascade small object culling: tiny details produce sub-pixel shadows in distance
                    if (cascade_idx) |c_idx| {
                        const ext = aabb_w.extents();
                        const max_dim = @max(ext.x, @max(ext.y, ext.z));
                        if (c_idx == 2 and max_dim < 0.35) continue;
                        if (c_idx == 3 and max_dim < 0.75) continue;
                    }

                    if (pip_id != last_pipeline_id.*) {
                        sg.applyPipeline(.{ .id = pip_id });
                        last_pipeline_id.* = pip_id;
                    }

                    var bind = sg.Bindings{};
                    bind.vertex_buffers[0] = mesh.vertex_buffer;
                    bind.index_buffer = mesh.index_buffer;
                    sg.applyBindings(bind);

                    const model = scene_render_queue.worldMatrixCached(frame_id, mesh);
                    const shadow_vs = shadow_shd.VsParams{
                        .mvp = Mat4.mul(light_view_proj, model),
                    };
                    sg.applyUniforms(shadow_shd.UB_vs_params, sg.asRange(&shadow_vs));

                    if (mesh.skeleton) |skel| {
                        const vs_skin = shadow_shd.VsSkin{
                            .bones = skel.getRenderSkinMatrices().*,
                        };
                        sg.applyUniforms(shadow_shd.UB_vs_skin, sg.asRange(&vs_skin));
                    }

                    sg.draw(0, mesh.index_count, 1);
                    draw_calls.* += 1;
                }
            }
        }
    }

    pub const BinResult = struct {
        counts: [6]usize,
        offsets: [6]usize,
    };

    const ParallelShadowBinning = struct {
        meshes: []const *Mesh,
        span: usize,
        chunk_counts: [][6]usize,
        chunk_offsets: [][6]usize,
        out_binned: []*Mesh,

        fn countChunkRange(pass: *ParallelShadowBinning, start: usize, end: usize) void {
            for (start..end) |chunk_id| {
                const lo = chunk_id * pass.span;
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
                const hi = @min(lo + pass.span, pass.meshes.len);
                var cursors = pass.chunk_offsets[chunk_id];
                for (pass.meshes[lo..hi]) |mesh| {
                    if (!mesh.cast_shadows or mesh.is_lod_child or mesh.is_decal) continue;
                    const b = @intFromEnum(bucketFor(mesh));
                    pass.out_binned[cursors[b]] = mesh;
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
                self.binned_meshes.resize(self.allocator, total) catch return .{ .counts = counts, .offsets = offsets };
            } else {
                self.binned_meshes.shrinkRetainingCapacity(total);
            }

            for (0..6) |b| {
                var cur = offsets[b];
                for (0..chunk_count) |c| {
                    chunk_offsets[c][b] = cur;
                    cur += chunk_counts[c][b];
                }
            }

            pass.out_binned = self.binned_meshes.items;
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
            self.binned_meshes.resize(self.allocator, total) catch return .{ .counts = counts, .offsets = offsets };
        } else {
            self.binned_meshes.shrinkRetainingCapacity(total);
        }

        for (meshes) |mesh| {
            if (!mesh.cast_shadows or mesh.is_lod_child or mesh.is_decal) continue;
            const b = @intFromEnum(bucketFor(mesh));
            self.binned_meshes.items[cursors[b]] = mesh;
            cursors[b] += 1;
        }

        return .{ .counts = counts, .offsets = offsets };
    }

    pub fn render(
        self: *ShadowPass,
        meshes: []const *Mesh,
        frame_id: u64,
        cascades: [4]Mat4,
        spot_shadows: []const SpotShadowRenderInfo,
        pool: ?*jobs.Pool,
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

        // 1. Pre-bin shadow-casting meshes into the 6 pipeline buckets once.
        const binned = self.binMeshes(meshes, pool);
        const counts = binned.counts;
        const offsets = binned.offsets;

        for (0..4) |c_idx| {
            const light_view_proj = cascades[c_idx];
            const vx: i32 = if (c_idx % 2 == 1) CASCADE_RES else 0;
            const vy: i32 = if (c_idx >= 2) CASCADE_RES else 0;

            sg.applyViewport(vx, vy, CASCADE_RES, CASCADE_RES, false);
            sg.applyScissorRect(vx, vy, CASCADE_RES, CASCADE_RES, false);

            const c_frustum = math.Frustum.fromViewProjection(light_view_proj);
            self.renderBuckets(light_view_proj, c_frustum, counts, offsets, frame_id, c_idx, &last_pipeline_id, &draw_calls);
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
                self.renderBuckets(spot_info.view_proj, spot_frustum, counts, offsets, frame_id, null, &spot_last_pipeline_id, &draw_calls);
            }

            sg.endPass();
            self.spot_needs_clear = false;
        }

        return draw_calls;
    }

    pub fn deinit(self: *ShadowPass) void {
        self.binned_meshes.deinit(self.allocator);
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
        sg.destroySampler(self.sampler);
        sg.destroySampler(self.depth_sampler);
        sg.destroyImage(self.image);
        sg.destroyImage(self.spot_image);
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
