const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const outline_shd = @import("outline_shader");
const math = @import("math");
const Mat4 = math.Mat4;
const Vec3 = math.Vec3;
const Color4 = math.Color4;
const Mesh = @import("../mesh.zig").Mesh;
const Vertex = @import("../mesh.zig").Vertex;
const Skeleton = @import("../animation/skeleton.zig").Skeleton;
const MAX_BONES = @import("../animation/skeleton.zig").MAX_BONES;
const uniforms = @import("../scene/uniforms.zig");
const scene_render_queue = @import("../scene/render_queue.zig");

// Inverse-hull outline/highlight layer: highlighted meshes are redrawn with
// front-face culling (the inflated "inside out" hull), LESS_EQUAL depth
// test and no depth writes, inside the already-open main pass (same contract
// as DebugPass.render/SkyboxPass.render: the caller must have begun the main
// pass whose depth attachment is reused). The constant screen-space width
// comes from the shader (clip-depth-scaled NDC offset along the projected
// normal); no extra geometry or render targets are needed.
// All four mesh flavors are supported: rigid meshes, instanced meshes
// (per-instance model matrix, only the visible instance prefix is drawn),
// skinned meshes (matrix-palette skinning, matching skinned_pbr.glsl) and
// alpha-cutout cards (foliage: dilated from the projected bounds center and
// alpha-tested, so the halo follows the leaf silhouette, not the quad).

// Upper bound for the rim width; larger values would swallow small meshes.
pub const max_width_px: f32 = 16.0;

// NDC depth bias pushing the hull toward the camera (reduces rim shimmer at
// grazing angles). Zero by default: the LESS_EQUAL test already hides the
// covered hull interior.
pub const default_depth_bias: f32 = 0.0;

// Depth push-away for the cutout halo: the dilated card is offset behind the
// source card so the alpha-tested leaf wins the depth test and only the halo
// ring outside the leaf silhouette stays visible.
pub const cutout_depth_bias: f32 = 0.002;

// Viewport used for the px->NDC conversion, set via resize(). Starts at a
// guarded 1x1 so a forgotten resize() cannot divide by zero (the rim would
// just be oversized until the parent wires resize, see report).
var viewport_size: [2]f32 = .{ 1.0, 1.0 };

/// Sanitizes the rim width: non-finite collapses to 0 (no outline),
/// otherwise clamped to [0, max_width_px].
pub fn clampWidthPx(width_px: f32) f32 {
    if (!std.math.isFinite(width_px)) return 0.0;
    return std.math.clamp(width_px, 0.0, max_width_px);
}

/// CPU mirror of the shader px->NDC conversion: per-axis NDC offset of a
/// clamped width on the given viewport. Guards degenerate viewports
/// (never returns Inf/NaN).
pub fn ndcExpandForViewport(width_px: f32, viewport_w: f32, viewport_h: f32) [2]f32 {
    const w = clampWidthPx(width_px);
    const vw = if (std.math.isFinite(viewport_w)) @max(viewport_w, 1.0) else 1.0;
    const vh = if (std.math.isFinite(viewport_h)) @max(viewport_h, 1.0) else 1.0;
    return .{ w * 2.0 / vw, w * 2.0 / vh };
}

/// Offsets a vertex along its normal by scale (CPU-side hull math for
/// tooling/tests; the GPU path does the equivalent in clip space).
/// Non-finite scale leaves the position untouched.
pub fn expandVertex(pos: [3]f32, normal: [3]f32, scale: f32) [3]f32 {
    if (!std.math.isFinite(scale)) return pos;
    return .{
        pos[0] + normal[0] * scale,
        pos[1] + normal[1] * scale,
        pos[2] + normal[2] * scale,
    };
}

/// Packs the shader `params` uniform from the clamped width and the current
/// viewport set via resize().
pub fn outlineParamsFor(width_px: f32) [4]f32 {
    return .{ clampWidthPx(width_px), viewport_size[0], viewport_size[1], default_depth_bias };
}

/// Whether a mesh can take the outline path at all: visible and drawable
/// through indexed geometry with live GPU buffers. Rigid, instanced and
/// skinned meshes all qualify; render() routes each to its pipeline family.
/// Pure predicate, no GPU calls.
pub fn shouldOutlineMesh(mesh: *const Mesh) bool {
    if (!mesh.is_visible) return false;
    if (mesh.index_count == 0) return false;
    if (mesh.vertex_buffer.id == 0 or mesh.index_buffer.id == 0) return false;
    return true;
}

/// Alpha-cutout outline info for a rigid mesh: the albedo texture providing
/// the mask and the material's cutoff threshold.
pub const CutoutInfo = struct {
    texture: @import("../texture.zig").Texture,
    cutoff: f32,
};

/// Rigid meshes with an alpha-cutout material outline through the dedicated
/// dilate + alpha-test path (a flat card has no interior silhouette for the
/// inverse hull). Null for every other mesh flavor.
pub fn cutoutInfoFor(mesh: *const Mesh) ?CutoutInfo {
    const cutoff = uniforms.alphaCutoffFor(mesh.material);
    if (cutoff <= 0.0) return null;
    const tex = mesh.material.?.primaryTexture() orelse return null;
    return .{ .texture = tex, .cutoff = cutoff };
}

/// Full clip-space transform (no perspective division) of a world point.
fn toClip(m: Mat4, p: Vec3) [4]f32 {
    return .{
        m.m[0] * p.x + m.m[4] * p.y + m.m[8] * p.z + m.m[12],
        m.m[1] * p.x + m.m[5] * p.y + m.m[9] * p.z + m.m[13],
        m.m[2] * p.x + m.m[6] * p.y + m.m[10] * p.z + m.m[14],
        m.m[3] * p.x + m.m[7] * p.y + m.m[11] * p.z + m.m[15],
    };
}

/// Shared inverse-hull pipeline state (culling, depth, blend). Vertex layout
/// is configured per family by the configure*Desc functions below.
fn setInverseHullState(desc: *sg.PipelineDesc) void {
    desc.depth = .{
        .compare = .LESS_EQUAL,
        .write_enabled = false,
    };
    // Inverse hull: cull front faces, the inflated back faces form the rim.
    desc.cull_mode = .FRONT;
    desc.face_winding = .CCW;
    // Flat color with alpha support (same factors as the transparent twins).
    desc.colors[0].blend = .{
        .enabled = true,
        .src_factor_rgb = .SRC_ALPHA,
        .dst_factor_rgb = .ONE_MINUS_SRC_ALPHA,
        .src_factor_alpha = .ONE,
        .dst_factor_alpha = .ONE_MINUS_SRC_ALPHA,
    };
}

/// Rigid meshes: reuses the Mesh vertex buffer directly (same stride, only
/// the position/normal attributes are bound).
pub fn configureOutlineDesc(desc: *sg.PipelineDesc) void {
    setInverseHullState(desc);
    desc.layout.buffers[0] = .{ .stride = @sizeOf(Vertex) };
    desc.layout.attrs[outline_shd.ATTR_outline_position] = .{
        .format = .FLOAT3,
        .offset = @offsetOf(Vertex, "position"),
    };
    desc.layout.attrs[outline_shd.ATTR_outline_normal] = .{
        .format = .FLOAT3,
        .offset = @offsetOf(Vertex, "normal"),
    };
}

/// Instanced meshes: mesh vertex buffer (position/normal) plus the shared
/// per-instance model-matrix buffer (4 x FLOAT4, PER_INSTANCE), mirroring
/// the instanced/instanced_pbr pipeline layouts.
pub fn configureOutlineInstDesc(desc: *sg.PipelineDesc) void {
    setInverseHullState(desc);
    desc.layout.buffers[0] = .{ .stride = @sizeOf(Vertex) };
    desc.layout.attrs[outline_shd.ATTR_outline_inst_position] = .{
        .format = .FLOAT3,
        .offset = @offsetOf(Vertex, "position"),
    };
    desc.layout.attrs[outline_shd.ATTR_outline_inst_normal] = .{
        .format = .FLOAT3,
        .offset = @offsetOf(Vertex, "normal"),
    };
    desc.layout.buffers[1] = .{
        .step_func = .PER_INSTANCE,
        .step_rate = 1,
        .stride = @sizeOf(Mat4),
    };
    desc.layout.attrs[outline_shd.ATTR_outline_inst_inst_mat0] = .{ .buffer_index = 1, .offset = 0, .format = .FLOAT4 };
    desc.layout.attrs[outline_shd.ATTR_outline_inst_inst_mat1] = .{ .buffer_index = 1, .offset = 16, .format = .FLOAT4 };
    desc.layout.attrs[outline_shd.ATTR_outline_inst_inst_mat2] = .{ .buffer_index = 1, .offset = 32, .format = .FLOAT4 };
    desc.layout.attrs[outline_shd.ATTR_outline_inst_inst_mat3] = .{ .buffer_index = 1, .offset = 48, .format = .FLOAT4 };
}

/// Alpha-cutout cards: position + uv from the mesh vertex buffer; no face
/// culling (the dilated halo must show from both sides of the card).
pub fn configureOutlineCutoutDesc(desc: *sg.PipelineDesc) void {
    setInverseHullState(desc);
    desc.cull_mode = .NONE;
    desc.layout.buffers[0] = .{ .stride = @sizeOf(Vertex) };
    desc.layout.attrs[outline_shd.ATTR_outline_cutout_position] = .{
        .format = .FLOAT3,
        .offset = @offsetOf(Vertex, "position"),
    };
    desc.layout.attrs[outline_shd.ATTR_outline_cutout_texcoord0] = .{
        .format = .FLOAT2,
        .offset = @offsetOf(Vertex, "uv"),
    };
}

/// Skinned meshes: full Mesh.Vertex layout (position/normal/joints/weights)
/// plus the vs_skin bone palette uniform applied per draw.
pub fn configureOutlineSkinnedDesc(desc: *sg.PipelineDesc) void {
    setInverseHullState(desc);
    desc.layout.buffers[0] = .{ .stride = @sizeOf(Vertex) };
    desc.layout.attrs[outline_shd.ATTR_outline_skinned_position] = .{
        .format = .FLOAT3,
        .offset = @offsetOf(Vertex, "position"),
    };
    desc.layout.attrs[outline_shd.ATTR_outline_skinned_normal] = .{
        .format = .FLOAT3,
        .offset = @offsetOf(Vertex, "normal"),
    };
    desc.layout.attrs[outline_shd.ATTR_outline_skinned_joints] = .{
        .format = .FLOAT4,
        .offset = @offsetOf(Vertex, "joints"),
    };
    desc.layout.attrs[outline_shd.ATTR_outline_skinned_weights] = .{
        .format = .FLOAT4,
        .offset = @offsetOf(Vertex, "weights"),
    };
}

/// Self-contained per-item payload for outline rendering. Хранит только
/// render-owned снимки (модель, хендлы, индекс копии скина): живых указателей
/// на Mesh/Skeleton здесь нет.
///
/// Identity (stage-2 increment A, refactor-only): `source_uid` is the source
/// mesh's `Mesh.uid` (nonzero, stable for lifetime), `source_mesh` is the
/// mesh-list index at build time. `instance_buffer`/`visible_instance_count`
/// are provisional for game-built (`.build_view`) payloads until the latch
/// `patchInstanceRefs` finalizes them; identity is validated by uid at patch
/// time.
pub const OutlineDrawItem = struct {
    vertex_buffer: sg.Buffer = .{},
    index_buffer: sg.Buffer = .{},
    index_count: u32 = 0,
    instance_buffer: sg.Buffer = .{},
    visible_instance_count: u32 = 1,
    model: Mat4 = Mat4.identity,
    /// Индекс копии скин-матриц во внешнем SkinStorage (null = не скин).
    /// Хранилище переживает item и резолвится на draw-фазе.
    skin_index: ?u32 = null,
    cutout_view: ?sg.View = null,
    cutout_sampler: ?sg.Sampler = null,
    cutout_cutoff: f32 = 0.0,
    world_center: Vec3 = Vec3.zero,
    is_skinned: bool = false,
    is_instanced: bool = false,
    is_cutout: bool = false,
    is_u32: bool = false,
    gpu_pending: bool = false,
    is_visible: bool = true,
    source_uid: u64 = 0,
    source_mesh: u32 = 0,
};

/// Строит OutlineDrawItem из живого меша (только prepare-фаза). Скин
/// копируется в render-owned хранилище; OOM возвращает null — вызывающий
/// пропускает item, а не рисует с живыми матрицами.
/// Матрица мира — через mesh.getWorldMatrix() (тот же расчёт, что и
/// worldMatrixCached; поведение не менялось: Scene строит контур ДО очередей,
/// так что «прогретый очередями кеш» здесь неверен — просто свежий пересчёт,
/// контурный список короткий, значения идентичны кешированным).
/// P4: сигнатура расширена хранилищем/аллокатором — осознанное изменение
/// low-level API (см. passes/mod.zig); Scene и OutlinePass.render стабильны.
/// Stage-2B: `instance_source` selects the staged state (`.published` =
/// fallback's `instance_render`, `.build_view` = game-frozen provisional);
/// `cache_key` still tags nothing here (regular-center source stays
/// `mesh.cached_aabb`); `source_mesh` is the mesh-list index at build time
/// (uid validated by the latch patch). `mesh` is mutable for lazy `ensureUid`.
pub fn makeOutlineDrawItem(
    allocator: std.mem.Allocator,
    skins: *scene_render_queue.SkinStorage,
    mesh: *Mesh,
    cache_key: u64,
    source_mesh: u32,
    instance_source: @import("../mesh.zig").InstanceSource,
) ?OutlineDrawItem {
    _ = mesh.ensureUid();
    _ = cache_key;
    const skinned = mesh.skeleton != null;
    const instanced = !skinned and mesh.instances.items.len > 0;
    const cutout = if (!skinned and !instanced) cutoutInfoFor(mesh) else null;
    // P5: instanced outline reads the frame's staged render state (Scene
    // stages before capturing outline items); the regular cached center path
    // below is unchanged. Stage-2B: the state resolves via instance_source
    // (fallback `.published`, game build `.build_view` provisional).
    const staged = mesh.instanceRenderSource(instance_source).*;
    const aabb = if (instanced) staged.bounds else mesh.cached_aabb;
    const center = if (aabb.isValid()) aabb.center() else mesh.position;

    var skin_index: ?u32 = null;
    if (mesh.skeleton) |skel| {
        const src = skel.getRenderSkinMatrices();
        skins.ensureUnusedCapacity(allocator, 1) catch return null;
        skin_index = @intCast(skins.items.len);
        skins.appendAssumeCapacity(src.*);
    }

    return OutlineDrawItem{
        .vertex_buffer = mesh.vertex_buffer,
        .index_buffer = mesh.index_buffer,
        .index_count = mesh.index_count,
        .instance_buffer = staged.buffer,
        .visible_instance_count = if (instanced) staged.count else 1,
        .model = mesh.getWorldMatrix(),
        .skin_index = skin_index,
        .cutout_view = if (cutout) |c| c.texture.view else null,
        .cutout_sampler = if (cutout) |c| c.texture.sampler else null,
        .cutout_cutoff = if (cutout) |c| c.cutoff else 0.0,
        .world_center = center,
        .is_skinned = skinned,
        .is_instanced = instanced,
        .is_cutout = (cutout != null),
        .is_u32 = (mesh.index_type == .UINT32),
        .gpu_pending = mesh.gpu_pending,
        .is_visible = mesh.is_visible,
        .source_uid = mesh.uid,
        .source_mesh = source_mesh,
    };
}

pub const OutlinePass = struct {
    pipeline_u16: sg.Pipeline = .{},
    pipeline_u32: sg.Pipeline = .{},
    pipeline_inst_u16: sg.Pipeline = .{},
    pipeline_inst_u32: sg.Pipeline = .{},
    pipeline_cutout_u16: sg.Pipeline = .{},
    pipeline_cutout_u32: sg.Pipeline = .{},
    pipeline_skinned_u16: sg.Pipeline = .{},
    pipeline_skinned_u32: sg.Pipeline = .{},

    shader_rigid: sg.Shader = .{},
    shader_inst: sg.Shader = .{},
    shader_skin: sg.Shader = .{},
    shader_cutout: sg.Shader = .{},

    /// Sample count the pipelines were built for. Draw calls into the main
    /// pass must use the variant matching the target shape.
    sample_count: i32 = 1,
    /// Main-target color format the pipelines were built for.
    color_format: sg.PixelFormat = .RGBA16F,

    /// Same pass for an explicit target shape (sample count + color
    /// format): each main-target shape needs its own pipeline variant
    /// (sokol requires pipeline.sample_count to equal the attachment's on
    /// every draw).
    pub fn init(sample_count: i32, color_format: sg.PixelFormat) OutlinePass {
        // One shader object per program family: the @program entries in
        // outline.glsl generate separate desc functions, each with its own
        // uniform block table (only the skinned program declares vs_skin,
        // only the cutout program declares the albedo mask binding).
        const shd_rigid = sg.makeShader(outline_shd.outlineShaderDesc(sg.queryBackend()));
        const shd_inst = sg.makeShader(outline_shd.outlineInstShaderDesc(sg.queryBackend()));
        const shd_skin = sg.makeShader(outline_shd.outlineSkinnedShaderDesc(sg.queryBackend()));
        const shd_cutout = sg.makeShader(outline_shd.outlineCutoutShaderDesc(sg.queryBackend()));

        var desc_u16 = sg.PipelineDesc{ .shader = shd_rigid, .index_type = .UINT16, .sample_count = sample_count };
        desc_u16.colors[0].pixel_format = color_format;
        configureOutlineDesc(&desc_u16);
        const pip_u16 = sg.makePipeline(desc_u16);

        var desc_u32 = sg.PipelineDesc{ .shader = shd_rigid, .index_type = .UINT32, .sample_count = sample_count };
        desc_u32.colors[0].pixel_format = color_format;
        configureOutlineDesc(&desc_u32);
        const pip_u32 = sg.makePipeline(desc_u32);

        var desc_inst_u16 = sg.PipelineDesc{ .shader = shd_inst, .index_type = .UINT16, .sample_count = sample_count };
        desc_inst_u16.colors[0].pixel_format = color_format;
        configureOutlineInstDesc(&desc_inst_u16);
        const pip_inst_u16 = sg.makePipeline(desc_inst_u16);

        var desc_inst_u32 = sg.PipelineDesc{ .shader = shd_inst, .index_type = .UINT32, .sample_count = sample_count };
        desc_inst_u32.colors[0].pixel_format = color_format;
        configureOutlineInstDesc(&desc_inst_u32);
        const pip_inst_u32 = sg.makePipeline(desc_inst_u32);

        var desc_skin_u16 = sg.PipelineDesc{ .shader = shd_skin, .index_type = .UINT16, .sample_count = sample_count };
        desc_skin_u16.colors[0].pixel_format = color_format;
        configureOutlineSkinnedDesc(&desc_skin_u16);
        const pip_skin_u16 = sg.makePipeline(desc_skin_u16);

        var desc_skin_u32 = sg.PipelineDesc{ .shader = shd_skin, .index_type = .UINT32, .sample_count = sample_count };
        desc_skin_u32.colors[0].pixel_format = color_format;
        configureOutlineSkinnedDesc(&desc_skin_u32);
        const pip_skin_u32 = sg.makePipeline(desc_skin_u32);

        var desc_cut_u16 = sg.PipelineDesc{ .shader = shd_cutout, .index_type = .UINT16, .sample_count = sample_count };
        desc_cut_u16.colors[0].pixel_format = color_format;
        configureOutlineCutoutDesc(&desc_cut_u16);
        const pip_cut_u16 = sg.makePipeline(desc_cut_u16);

        var desc_cut_u32 = sg.PipelineDesc{ .shader = shd_cutout, .index_type = .UINT32, .sample_count = sample_count };
        desc_cut_u32.colors[0].pixel_format = color_format;
        configureOutlineCutoutDesc(&desc_cut_u32);
        const pip_cut_u32 = sg.makePipeline(desc_cut_u32);

        return .{
            .pipeline_u16 = pip_u16,
            .pipeline_u32 = pip_u32,
            .pipeline_inst_u16 = pip_inst_u16,
            .pipeline_inst_u32 = pip_inst_u32,
            .pipeline_cutout_u16 = pip_cut_u16,
            .pipeline_cutout_u32 = pip_cut_u32,
            .pipeline_skinned_u16 = pip_skin_u16,
            .pipeline_skinned_u32 = pip_skin_u32,
            .shader_rigid = shd_rigid,
            .shader_inst = shd_inst,
            .shader_skin = shd_skin,
            .shader_cutout = shd_cutout,
            .sample_count = sample_count,
            .color_format = color_format,
        };
    }

    /// Remembers the drawable size for the px->NDC conversion. The parent
    /// (Scene) must forward its framebuffer resizes here.
    pub fn resize(w: i32, h: i32) void {
        viewport_size[0] = @max(1.0, @as(f32, @floatFromInt(w)));
        viewport_size[1] = @max(1.0, @as(f32, @floatFromInt(h)));
    }

    /// Renders immutable outline items into the currently open main pass.
    /// skins — то же хранилище, в которое makeOutlineDrawItem складывал копии
    /// (резолв по skin_index уже после всех реаллокаций prepare-фазы).
    pub fn renderItems(
        self: *OutlinePass,
        view_proj: Mat4,
        camera_pos: Vec3,
        items: []const OutlineDrawItem,
        skins: []const [MAX_BONES]Mat4,
        color: Color4,
        width_px: f32,
    ) void {
        _ = camera_pos;
        if (items.len == 0) return;
        const width = clampWidthPx(width_px);
        if (width <= 0.0) return;
        if (!sg.isvalid()) return;

        for (items) |item| {
            if (!item.is_visible or item.gpu_pending or item.index_count == 0) continue;
            if (item.vertex_buffer.id == 0 or item.index_buffer.id == 0) continue;

            const skinned = item.is_skinned;
            const instanced = !skinned and item.is_instanced;
            const is_cutout = !skinned and !instanced and item.is_cutout;
            const is_u32 = item.is_u32;
            const pip = if (skinned)
                (if (is_u32) self.pipeline_skinned_u32 else self.pipeline_skinned_u16)
            else if (instanced)
                (if (is_u32) self.pipeline_inst_u32 else self.pipeline_inst_u16)
            else if (is_cutout)
                (if (is_u32) self.pipeline_cutout_u32 else self.pipeline_cutout_u16)
            else
                (if (is_u32) self.pipeline_u32 else self.pipeline_u16);
            if (sg.queryPipelineState(pip) != .VALID) continue;

            if (item.vertex_buffer.id == 0) continue;
            if (sg.isvalid()) {
                if (sg.queryBufferState(item.vertex_buffer) != .VALID) continue;
                if (item.index_buffer.id != 0 and sg.queryBufferState(item.index_buffer) != .VALID) continue;
            }

            var bind = sg.Bindings{};
            bind.vertex_buffers[0] = item.vertex_buffer;
            if (instanced) {
                if (item.instance_buffer.id == 0 or item.visible_instance_count == 0) continue;
                if (sg.isvalid() and sg.queryBufferState(item.instance_buffer) != .VALID) continue;
                bind.vertex_buffers[1] = item.instance_buffer;
            }
            if (is_cutout and item.cutout_view != null and item.cutout_sampler != null) {
                bind.views[outline_shd.VIEW_albedo_tex] = item.cutout_view.?;
                bind.samplers[outline_shd.SMP_smp] = item.cutout_sampler.?;
            }
            bind.index_buffer = item.index_buffer;
            sg.applyPipeline(pip);
            sg.applyBindings(bind);

            const model = item.model;
            const mvp = Mat4.mul(view_proj, model);
            const vs_params = outline_shd.VsParams{
                .mvp = mvp,
                .model = model,
                .color = color.toArray(),
                .params = if (is_cutout) blk: {
                    var p = outlineParamsFor(width);
                    p[3] = cutout_depth_bias;
                    break :blk p;
                } else outlineParamsFor(width),
            };
            sg.applyUniforms(outline_shd.UB_vs_params, sg.asRange(&vs_params));
            if (skinned) {
                // Битый индекс (билдером недостижимо): пропуск вместо stale-униформы.
                const bones = scene_render_queue.skinAt(skins, item.skin_index) orelse continue;
                const vs_skin = outline_shd.VsSkin{
                    .bones = bones.*,
                };
                sg.applyUniforms(outline_shd.UB_vs_skin, sg.asRange(&vs_skin));
            }
            if (is_cutout) {
                const clip = toClip(view_proj, item.world_center);
                const center_ndc: [4]f32 = if (clip[3] <= 0.001)
                    .{ 0, 0, 0, -1.0 }
                else
                    .{ clip[0] / clip[3], clip[1] / clip[3], 0, 1.0 };
                const vs_center = outline_shd.VsCenter{
                    .center_ndc = center_ndc,
                };
                sg.applyUniforms(outline_shd.UB_vs_center, sg.asRange(&vs_center));
                const fs_cutout = outline_shd.FsCutoutParams{
                    .cutout = .{ item.cutout_cutoff, 0, 0, 0 },
                };
                sg.applyUniforms(outline_shd.UB_fs_cutout_params, sg.asRange(&fs_cutout));
            }

            const instance_count: u32 = if (instanced) item.visible_instance_count else 1;
            sg.draw(0, item.index_count, instance_count);
        }
    }

    /// Each mesh routes to its pipeline family (rigid / instanced / skinned /
    /// alpha-cutout); meshes failing shouldOutlineMesh are skipped silently.
    /// Empty list and zero width are GPU-free no-ops.
    /// Immediate-режим: снимки (включая копии скинов) строятся здесь же во
    /// временное хранилище и рисуются синхронно до возврата — за пределы
    /// вызова живые указатели не утекают.
    pub fn render(self: *OutlinePass, view_proj: Mat4, camera_pos: Vec3, meshes: []const *Mesh, color: Color4, width_px: f32) void {
        if (meshes.len == 0) return;
        var stack_items: [32]OutlineDrawItem = undefined;
        var items: []OutlineDrawItem = undefined;
        var heap_buf: ?[]OutlineDrawItem = null;
        if (meshes.len <= 32) {
            items = stack_items[0..meshes.len];
        } else {
            heap_buf = std.heap.c_allocator.alloc(OutlineDrawItem, meshes.len) catch return;
            items = heap_buf.?;
        }
        defer if (heap_buf) |h| std.heap.c_allocator.free(h);

        var skins: scene_render_queue.SkinStorage = .empty;
        defer skins.deinit(std.heap.c_allocator);
        for (meshes, 0..) |m, i| {
            // Immediate mode has no frame cache: cache_key 0 is unused for
            // instance data (`.published` = instance_render, identical
            // behavior); source_mesh is the input slice index at build time.
            items[i] = makeOutlineDrawItem(std.heap.c_allocator, &skins, m, 0, @intCast(i), .published) orelse .{};
        }
        self.renderItems(view_proj, camera_pos, items, skins.items, color, width_px);
    }

    pub fn deinit(self: *OutlinePass) void {
        sg.destroyPipeline(self.pipeline_u16);
        sg.destroyPipeline(self.pipeline_u32);
        sg.destroyPipeline(self.pipeline_inst_u16);
        sg.destroyPipeline(self.pipeline_inst_u32);
        sg.destroyPipeline(self.pipeline_cutout_u16);
        sg.destroyPipeline(self.pipeline_cutout_u32);
        sg.destroyPipeline(self.pipeline_skinned_u16);
        sg.destroyPipeline(self.pipeline_skinned_u32);
        if (self.shader_rigid.id != 0) sg.destroyShader(self.shader_rigid);
        if (self.shader_inst.id != 0) sg.destroyShader(self.shader_inst);
        if (self.shader_skin.id != 0) sg.destroyShader(self.shader_skin);
        if (self.shader_cutout.id != 0) sg.destroyShader(self.shader_cutout);
        self.* = undefined;
    }
};

// Outline regression tests live in `outline_pass_tests.zig` (same directory,
// imported below so the test registry picks them up exactly once).

test {
    _ = @import("outline_pass_tests.zig");
}
