//! Shadow pass state and lifecycle. Split out of `shadow_pass.zig` (facade).
//!
//! This module owns the `ShadowPass` type: the atlas/pipeline fields, the
//! nested `ShadowDrawItem` + `PreparedShadowDraws` payloads, the trivial
//! lifecycle (`init`/`deinit`) and the small entry points (`prepare`,
//! `renderPrepared`, `render`) plus thin forwarders into the siblings below,
//! so every call site keeps working unchanged:
//!
//! - `types.zig` — atlas constants, render infos, `Bucket`/`BinResult`
//!   vocabulary, `bucketFor`, point-atlas math.
//! - `binning.zig` — `binMeshes` (serial + parallel mesh binning).
//! - `prepare.zig` — `prepareInto` (render-owned snapshot build).
//! - `buckets.zig` — `renderBuckets` (culling + bucketed depth draws).
//! - `csm.zig` / `spot.zig` / `point.zig` — the three atlas render paths
//!   composed by `renderPreparedFrom`.
//!
//! Anti-cycle rule (same as `particles/`, `profiler/`, `scene/`): siblings
//! take the pass as `anytype` and never import this module or the
//! `shadow_pass.zig` facade back; this module passes `self` straight
//! through. `shadow_pass.zig` re-exports `ShadowPass` under its historical
//! path.
const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const shadow_shd = @import("shadow_shader");
const math = @import("math");
const Mat4 = math.Mat4;
const mesh_mod = @import("../../mesh.zig");
const Mesh = mesh_mod.Mesh;
const Vertex = mesh_mod.Vertex;
const scene_render_queue = @import("../../scene/render_queue.zig");
const jobs = @import("../../jobs.zig");

const types = @import("types.zig");
const binning = @import("binning.zig");
const prepare_mod = @import("prepare.zig");
const csm = @import("csm.zig");
const spot = @import("spot.zig");
const point = @import("point.zig");

const SHADOW_ATLAS_SIZE = types.SHADOW_ATLAS_SIZE;
const SPOT_SHADOW_MAP_WIDTH = types.SPOT_SHADOW_MAP_WIDTH;
const SPOT_SHADOW_MAP_HEIGHT = types.SPOT_SHADOW_MAP_HEIGHT;
const POINT_SHADOW_MAP_WIDTH = types.POINT_SHADOW_MAP_WIDTH;
const POINT_SHADOW_MAP_HEIGHT = types.POINT_SHADOW_MAP_HEIGHT;
const SpotShadowRenderInfo = types.SpotShadowRenderInfo;
const PointShadowRenderInfo = types.PointShadowRenderInfo;

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

    // Aliases keeping the historical `ShadowPass.Bucket` /
    // `ShadowPass.bucket_order` / `ShadowPass.BinResult` paths working; the
    // vocabulary itself lives in `types.zig`.
    pub const Bucket = types.Bucket;
    pub const bucket_order: [6]Bucket = types.bucket_order;
    pub const BinResult = types.BinResult;

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
        /// Render-owned low-poly stand-in for distant CSM cascades
        /// (>= `types.SHADOW_LOD_FIRST_CASCADE`): borrowed GPU handles of
        /// the coarsest QEM-simplified LOD child (`types.shadowLodMesh`),
        /// snapshotted per prepare like the handles above. `false` (no
        /// valid stand-in: no LOD, skinned/morph source, pending upload,
        /// type mismatch, non-decimated child) fails safe to the
        /// high-poly handles above in every cascade. Skinned items never
        /// carry a stand-in. LOD children are excluded from binning, so the
        /// stand-in never double-draws; the main pass keeps its own
        /// camera-distance LOD pick untouched.
        lod_vertex_buffer: sg.Buffer = .{},
        lod_index_buffer: sg.Buffer = .{},
        lod_index_count: u32 = 0,
        has_shadow_lod: bool = false,
        /// Индекс копии скин-матриц в PreparedShadowDraws.skins (null = не скин).
        skin_index: ?u32 = null,
        bucket: Bucket = .regular_u16,
        is_instanced: bool = false,
        gpu_pending: bool = false,
        is_visible: bool = true,
        source_uid: u64 = 0,
        source_mesh: u32 = 0,
    };

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

    pub fn binMeshes(
        self: *ShadowPass,
        meshes: []const *Mesh,
        pool: ?*jobs.Pool,
    ) BinResult {
        return binning.binMeshes(self, meshes, pool);
    }

    pub fn prepareInto(
        self: *ShadowPass,
        out: *PreparedShadowDraws,
        meshes: []const *Mesh,
        cache_key: u64,
        instance_source: mesh_mod.InstanceSource,
        pool: ?*jobs.Pool,
    ) BinResult {
        return prepare_mod.prepareInto(self, out, meshes, cache_key, instance_source, pool);
    }

    pub fn prepare(
        self: *ShadowPass,
        meshes: []const *Mesh,
        cache_key: u64,
        instance_source: mesh_mod.InstanceSource,
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
        var draw_calls: u32 = 0;
        csm.renderCsm(self, prepared, cascades, &draw_calls);
        spot.renderSpot(self, prepared, spot_shadows, &draw_calls);
        point.renderPoint(self, prepared, point_shadows, &draw_calls);
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
        instance_source: mesh_mod.InstanceSource,
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

// --- Shared test helpers (headless; pub for sibling-leaf test blocks) ---

/// Test-only pass constructor lives here — next to its owner — because only
/// this module may name `ShadowPass` at top level without an import cycle
/// (leaves take the pass as `anytype`). Reached from moved tests through
/// block-scoped imports that exist only in test builds. Never re-exported
/// from the `shadow_pass.zig` facade.
pub fn testShadowPass(ally: std.mem.Allocator) ShadowPass {
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
