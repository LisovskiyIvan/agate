//! Velocity pipeline factory + destroy. Split out of `velocity_pass.zig`.
//!
//! Owns nothing: builds (or rebuilds on format change) the twelve opaque
//! pipelines (rigid/instanced/skinned x u16/u32 x single/double-sided),
//! the two particle reactive-marker pipelines, plus the twelve glass
//! reactive-marker pipelines (same shape/index/sidedness grid) for the
//! borrowed main-depth format, and destroys them. The pass is passed as
//! `anytype` to avoid a leaf-to-owner import (same discipline as
//! `passes/postprocess/targets.zig` and `passes/shadow/`); this module
//! never imports `velocity_pass.zig` back.
//!
//! All twenty-six builds preserve the locals-first, all-or-nothing
//! validation, partial-handle rollback, and destroy/publish ordering.
//! Opaque pipelines share the
//! shared-depth plan (writes off, EQUAL compare); all fourteen marker
//! pipelines share the marker plan (writes off, LESS_EQUAL compare).
//! Single/double-sided twins share every descriptor field except
//! `cull_mode` (BACK vs NONE): each twin uses the same layout and depth plan,
//! so the single-sided default is bit-identical to the pre-sidedness path.
const sokol = @import("sokol");
const sg = sokol.gfx;
const velocity_shd = @import("velocity_shader");
const math = @import("math");
const Mat4 = math.Mat4;
const mesh_mod = @import("../../mesh.zig");
const Vertex = mesh_mod.Vertex;
const particles_mod = @import("../../particles.zig");

/// Sidedness-to-cull mapping shared by every velocity/marker pipeline
/// (pure, no sg calls): single-sided culls back faces like the main
/// forward path; double-sided draws both (cull off) so backfaces emit the
/// same velocity/marker coverage the main pass shaded. Particle billboards
/// never use this (always cull NONE — view-facing quads, no sidedness).
pub fn cullModeFor(double_sided: bool) sg.CullMode {
    return if (double_sided) .NONE else .BACK;
}

/// Forces only the cull mode for sidedness on an already-built descriptor
/// (pure, no sg calls): every other field (shader, layout, depth EQUAL vs
/// LESS_EQUAL, writes off, formats, winding) is preserved, so the
/// double-sided twin differs from its single-sided default in exactly one
/// field.
pub fn applySidedness(desc: *sg.PipelineDesc, double_sided: bool) void {
    desc.cull_mode = cullModeFor(double_sided);
}

/// Common opaque-pipeline fields (shared-depth plan: writes off, EQUAL
/// compare against the borrowed main depth, RGBA16F target, CCW winding).
/// Cull comes from sidedness; layout is filled per shape afterwards.
fn opaqueCommon(shader: sg.Shader, index_type: sg.IndexType, depth_fmt: sg.PixelFormat, double_sided: bool) sg.PipelineDesc {
    var desc = sg.PipelineDesc{
        .shader = shader,
        .index_type = index_type,
        .sample_count = 1,
        .depth = .{
            .pixel_format = depth_fmt,
            .compare = .EQUAL,
            .write_enabled = false,
        },
        .face_winding = .CCW,
    };
    desc.colors[0].pixel_format = .RGBA16F;
    applySidedness(&desc, double_sided);
    return desc;
}

/// Common glass-marker fields (marker plan: writes off, LESS_EQUAL compare
/// — glass writes no main depth so its fragments sit in front of the stored
/// background depth, exactly like particle billboards). Cull from
/// sidedness; layout is filled per shape afterwards.
fn markerCommon(shader: sg.Shader, index_type: sg.IndexType, depth_fmt: sg.PixelFormat, double_sided: bool) sg.PipelineDesc {
    var desc = sg.PipelineDesc{
        .shader = shader,
        .index_type = index_type,
        .sample_count = 1,
        .depth = .{
            .pixel_format = depth_fmt,
            .compare = .LESS_EQUAL,
            .write_enabled = false,
        },
        .face_winding = .CCW,
    };
    desc.colors[0].pixel_format = .RGBA16F;
    applySidedness(&desc, double_sided);
    return desc;
}

fn layoutRigidVelocity(desc: *sg.PipelineDesc) void {
    desc.layout.buffers[0] = .{ .stride = @sizeOf(Vertex) };
    desc.layout.attrs[velocity_shd.ATTR_velocity_position] = .{
        .format = .FLOAT3,
        .offset = @offsetOf(Vertex, "position"),
    };
    desc.layout.attrs[velocity_shd.ATTR_velocity_color0] = .{
        .format = .FLOAT4,
        .offset = @offsetOf(Vertex, "color"),
    };
    desc.layout.attrs[velocity_shd.ATTR_velocity_texcoord0] = .{
        .format = .FLOAT2,
        .offset = @offsetOf(Vertex, "uv"),
    };
    desc.layout.attrs[velocity_shd.ATTR_velocity_texcoord1] = .{
        .format = .FLOAT2,
        .offset = @offsetOf(Vertex, "uv1"),
    };
}

fn layoutInstVelocity(desc: *sg.PipelineDesc) void {
    desc.layout.buffers[0] = .{ .stride = @sizeOf(Vertex) };
    desc.layout.attrs[velocity_shd.ATTR_velocity_instanced_position] = .{
        .buffer_index = 0,
        .format = .FLOAT3,
        .offset = @offsetOf(Vertex, "position"),
    };
    desc.layout.attrs[velocity_shd.ATTR_velocity_instanced_color0] = .{
        .buffer_index = 0,
        .format = .FLOAT4,
        .offset = @offsetOf(Vertex, "color"),
    };
    desc.layout.attrs[velocity_shd.ATTR_velocity_instanced_texcoord0] = .{
        .buffer_index = 0,
        .format = .FLOAT2,
        .offset = @offsetOf(Vertex, "uv"),
    };
    desc.layout.attrs[velocity_shd.ATTR_velocity_instanced_texcoord1] = .{
        .buffer_index = 0,
        .format = .FLOAT2,
        .offset = @offsetOf(Vertex, "uv1"),
    };
    desc.layout.buffers[1] = .{
        .step_func = .PER_INSTANCE,
        .step_rate = 1,
        .stride = @sizeOf(Mat4),
    };
    desc.layout.attrs[velocity_shd.ATTR_velocity_instanced_inst_mat0] = .{ .buffer_index = 1, .offset = 0, .format = .FLOAT4 };
    desc.layout.attrs[velocity_shd.ATTR_velocity_instanced_inst_mat1] = .{ .buffer_index = 1, .offset = 16, .format = .FLOAT4 };
    desc.layout.attrs[velocity_shd.ATTR_velocity_instanced_inst_mat2] = .{ .buffer_index = 1, .offset = 32, .format = .FLOAT4 };
    desc.layout.attrs[velocity_shd.ATTR_velocity_instanced_inst_mat3] = .{ .buffer_index = 1, .offset = 48, .format = .FLOAT4 };

    desc.layout.buffers[2] = .{
        .step_func = .PER_INSTANCE,
        .step_rate = 1,
        .stride = @sizeOf(Mat4),
    };
    desc.layout.attrs[velocity_shd.ATTR_velocity_instanced_prev_inst_mat0] = .{ .buffer_index = 2, .offset = 0, .format = .FLOAT4 };
    desc.layout.attrs[velocity_shd.ATTR_velocity_instanced_prev_inst_mat1] = .{ .buffer_index = 2, .offset = 16, .format = .FLOAT4 };
    desc.layout.attrs[velocity_shd.ATTR_velocity_instanced_prev_inst_mat2] = .{ .buffer_index = 2, .offset = 32, .format = .FLOAT4 };
    desc.layout.attrs[velocity_shd.ATTR_velocity_instanced_prev_inst_mat3] = .{ .buffer_index = 2, .offset = 48, .format = .FLOAT4 };
}

fn layoutSkinVelocity(desc: *sg.PipelineDesc) void {
    desc.layout.buffers[0] = .{ .stride = @sizeOf(Vertex) };
    desc.layout.attrs[velocity_shd.ATTR_velocity_skinned_position] = .{
        .format = .FLOAT3,
        .offset = @offsetOf(Vertex, "position"),
    };
    desc.layout.attrs[velocity_shd.ATTR_velocity_skinned_joints] = .{
        .format = .FLOAT4,
        .offset = @offsetOf(Vertex, "joints"),
    };
    desc.layout.attrs[velocity_shd.ATTR_velocity_skinned_weights] = .{
        .format = .FLOAT4,
        .offset = @offsetOf(Vertex, "weights"),
    };
    desc.layout.attrs[velocity_shd.ATTR_velocity_skinned_color0] = .{
        .format = .FLOAT4,
        .offset = @offsetOf(Vertex, "color"),
    };
    desc.layout.attrs[velocity_shd.ATTR_velocity_skinned_texcoord0] = .{
        .format = .FLOAT2,
        .offset = @offsetOf(Vertex, "uv"),
    };
    desc.layout.attrs[velocity_shd.ATTR_velocity_skinned_texcoord1] = .{
        .format = .FLOAT2,
        .offset = @offsetOf(Vertex, "uv1"),
    };
}

fn layoutGlassRigid(desc: *sg.PipelineDesc) void {
    desc.layout.buffers[0] = .{ .stride = @sizeOf(Vertex) };
    desc.layout.attrs[velocity_shd.ATTR_glass_marker_position] = .{
        .format = .FLOAT3,
        .offset = @offsetOf(Vertex, "position"),
    };
}

fn layoutGlassInst(desc: *sg.PipelineDesc) void {
    desc.layout.buffers[0] = .{ .stride = @sizeOf(Vertex) };
    desc.layout.attrs[velocity_shd.ATTR_glass_marker_inst_position] = .{
        .buffer_index = 0,
        .format = .FLOAT3,
        .offset = @offsetOf(Vertex, "position"),
    };
    desc.layout.buffers[1] = .{
        .step_func = .PER_INSTANCE,
        .step_rate = 1,
        .stride = @sizeOf(Mat4),
    };
    desc.layout.attrs[velocity_shd.ATTR_glass_marker_inst_inst_mat0] = .{ .buffer_index = 1, .offset = 0, .format = .FLOAT4 };
    desc.layout.attrs[velocity_shd.ATTR_glass_marker_inst_inst_mat1] = .{ .buffer_index = 1, .offset = 16, .format = .FLOAT4 };
    desc.layout.attrs[velocity_shd.ATTR_glass_marker_inst_inst_mat2] = .{ .buffer_index = 1, .offset = 32, .format = .FLOAT4 };
    desc.layout.attrs[velocity_shd.ATTR_glass_marker_inst_inst_mat3] = .{ .buffer_index = 1, .offset = 48, .format = .FLOAT4 };
}

fn layoutGlassSkin(desc: *sg.PipelineDesc) void {
    desc.layout.buffers[0] = .{ .stride = @sizeOf(Vertex) };
    desc.layout.attrs[velocity_shd.ATTR_glass_marker_skin_position] = .{
        .format = .FLOAT3,
        .offset = @offsetOf(Vertex, "position"),
    };
    desc.layout.attrs[velocity_shd.ATTR_glass_marker_skin_joints] = .{
        .format = .FLOAT4,
        .offset = @offsetOf(Vertex, "joints"),
    };
    desc.layout.attrs[velocity_shd.ATTR_glass_marker_skin_weights] = .{
        .format = .FLOAT4,
        .offset = @offsetOf(Vertex, "weights"),
    };
}

/// Bounded shape/sidedness/index factory: one descriptor per (shape,
/// index type, sidedness) triple. The double-sided twin is the
/// single-sided desc with only the cull forced off, so single-sided
/// defaults keep their exact pre-sidedness descriptors.
fn rigidDesc(shader: sg.Shader, index_type: sg.IndexType, depth_fmt: sg.PixelFormat, double_sided: bool) sg.PipelineDesc {
    var desc = opaqueCommon(shader, index_type, depth_fmt, double_sided);
    layoutRigidVelocity(&desc);
    return desc;
}

fn instDesc(shader: sg.Shader, index_type: sg.IndexType, depth_fmt: sg.PixelFormat, double_sided: bool) sg.PipelineDesc {
    var desc = opaqueCommon(shader, index_type, depth_fmt, double_sided);
    layoutInstVelocity(&desc);
    return desc;
}

fn skinDesc(shader: sg.Shader, index_type: sg.IndexType, depth_fmt: sg.PixelFormat, double_sided: bool) sg.PipelineDesc {
    var desc = opaqueCommon(shader, index_type, depth_fmt, double_sided);
    layoutSkinVelocity(&desc);
    return desc;
}

fn glassRigidDesc(shader: sg.Shader, index_type: sg.IndexType, depth_fmt: sg.PixelFormat, double_sided: bool) sg.PipelineDesc {
    var desc = markerCommon(shader, index_type, depth_fmt, double_sided);
    layoutGlassRigid(&desc);
    return desc;
}

fn glassInstDesc(shader: sg.Shader, index_type: sg.IndexType, depth_fmt: sg.PixelFormat, double_sided: bool) sg.PipelineDesc {
    var desc = markerCommon(shader, index_type, depth_fmt, double_sided);
    layoutGlassInst(&desc);
    return desc;
}

fn glassSkinDesc(shader: sg.Shader, index_type: sg.IndexType, depth_fmt: sg.PixelFormat, double_sided: bool) sg.PipelineDesc {
    var desc = markerCommon(shader, index_type, depth_fmt, double_sided);
    layoutGlassSkin(&desc);
    return desc;
}

/// Builds (or rebuilds on format change) the twelve opaque pipelines, the
/// two particle reactive-marker pipelines, plus the twelve glass
/// reactive-marker pipelines for the borrowed main-depth format.
/// Locals-first like everywhere else: the working set is
/// preserved when a rebuild fails. Opaque pipelines share the
/// shared-depth plan (writes off, EQUAL compare — see
/// velocityDepthState); all fourteen marker pipelines share the marker
/// plan (writes off, LESS_EQUAL compare — see reactiveMarkerDepthState).
/// All twenty-six are all-or-nothing: a failed build skips the whole pass
/// for the frame (fail-closed to the zero-mask fallback, never a half
/// pass).
pub fn ensurePipelinesFor(self: anytype, depth_fmt: sg.PixelFormat) bool {
    if (!sg.isvalid()) return false;
    if (depth_fmt == .DEFAULT or depth_fmt == .NONE) return false;
    if (self.depth_format == depth_fmt and
        sg.queryPipelineState(self.pip_rigid_u16) == .VALID and
        sg.queryPipelineState(self.pip_rigid_u32) == .VALID and
        sg.queryPipelineState(self.pip_rigid_u16_ds) == .VALID and
        sg.queryPipelineState(self.pip_rigid_u32_ds) == .VALID and
        sg.queryPipelineState(self.pip_inst_u16) == .VALID and
        sg.queryPipelineState(self.pip_inst_u32) == .VALID and
        sg.queryPipelineState(self.pip_inst_u16_ds) == .VALID and
        sg.queryPipelineState(self.pip_inst_u32_ds) == .VALID and
        sg.queryPipelineState(self.pip_skin_u16) == .VALID and
        sg.queryPipelineState(self.pip_skin_u32) == .VALID and
        sg.queryPipelineState(self.pip_skin_u16_ds) == .VALID and
        sg.queryPipelineState(self.pip_skin_u32_ds) == .VALID and
        sg.queryPipelineState(self.pip_marker_cpu) == .VALID and
        sg.queryPipelineState(self.pip_marker_gpu) == .VALID and
        sg.queryPipelineState(self.pip_marker_glass_rigid_u16) == .VALID and
        sg.queryPipelineState(self.pip_marker_glass_rigid_u32) == .VALID and
        sg.queryPipelineState(self.pip_marker_glass_rigid_u16_ds) == .VALID and
        sg.queryPipelineState(self.pip_marker_glass_rigid_u32_ds) == .VALID and
        sg.queryPipelineState(self.pip_marker_glass_inst_u16) == .VALID and
        sg.queryPipelineState(self.pip_marker_glass_inst_u32) == .VALID and
        sg.queryPipelineState(self.pip_marker_glass_inst_u16_ds) == .VALID and
        sg.queryPipelineState(self.pip_marker_glass_inst_u32_ds) == .VALID and
        sg.queryPipelineState(self.pip_marker_glass_skin_u16) == .VALID and
        sg.queryPipelineState(self.pip_marker_glass_skin_u32) == .VALID and
        sg.queryPipelineState(self.pip_marker_glass_skin_u16_ds) == .VALID and
        sg.queryPipelineState(self.pip_marker_glass_skin_u32_ds) == .VALID)
    {
        return true;
    }
    if (self.rigid_shader.id == 0 or self.inst_shader.id == 0 or self.skin_shader.id == 0) return false;
    if (self.marker_shader.id == 0 or self.marker_gpu_shader.id == 0) return false;
    if (self.glass_marker_shader.id == 0 or self.glass_marker_inst_shader.id == 0 or self.glass_marker_skin_shader.id == 0) return false;

    const pip_r_u16 = sg.makePipeline(rigidDesc(self.rigid_shader, .UINT16, depth_fmt, false));
    const pip_r_u32 = sg.makePipeline(rigidDesc(self.rigid_shader, .UINT32, depth_fmt, false));
    const pip_r_u16_ds = sg.makePipeline(rigidDesc(self.rigid_shader, .UINT16, depth_fmt, true));
    const pip_r_u32_ds = sg.makePipeline(rigidDesc(self.rigid_shader, .UINT32, depth_fmt, true));

    const pip_i_u16 = sg.makePipeline(instDesc(self.inst_shader, .UINT16, depth_fmt, false));
    const pip_i_u32 = sg.makePipeline(instDesc(self.inst_shader, .UINT32, depth_fmt, false));
    const pip_i_u16_ds = sg.makePipeline(instDesc(self.inst_shader, .UINT16, depth_fmt, true));
    const pip_i_u32_ds = sg.makePipeline(instDesc(self.inst_shader, .UINT32, depth_fmt, true));

    const pip_s_u16 = sg.makePipeline(skinDesc(self.skin_shader, .UINT16, depth_fmt, false));
    const pip_s_u32 = sg.makePipeline(skinDesc(self.skin_shader, .UINT32, depth_fmt, false));
    const pip_s_u16_ds = sg.makePipeline(skinDesc(self.skin_shader, .UINT16, depth_fmt, true));
    const pip_s_u32_ds = sg.makePipeline(skinDesc(self.skin_shader, .UINT32, depth_fmt, true));

    // Reactive-marker pipelines (C.3 v2): particle billboard coverage
    // with the constant band-alpha FS. Buffer 0 is the owned unit quad
    // (uv lanes ride unconsumed in the stride); buffer 1 is the
    // per-instance record (ParticleInstanceData for .cpu/.compute,
    // GpuParticleSlot x5 for .gpu). Single UINT16 index type (quad ib).
    var m_desc = sg.PipelineDesc{
        .shader = self.marker_shader,
        .index_type = .UINT16,
        .sample_count = 1,
        .depth = .{
            .pixel_format = depth_fmt,
            .compare = .LESS_EQUAL,
            .write_enabled = false,
        },
        .cull_mode = .NONE,
    };
    m_desc.colors[0].pixel_format = .RGBA16F;
    m_desc.layout.buffers[0] = .{ .stride = 4 * @sizeOf(f32) };
    m_desc.layout.attrs[velocity_shd.ATTR_reactive_marker_position] = .{
        .buffer_index = 0,
        .format = .FLOAT2,
        .offset = 0,
    };
    m_desc.layout.buffers[1] = .{
        .step_func = .PER_INSTANCE,
        .step_rate = 1,
        .stride = @sizeOf(particles_mod.ParticleInstanceData),
    };
    m_desc.layout.attrs[velocity_shd.ATTR_reactive_marker_inst_pos_size] = .{
        .buffer_index = 1,
        .format = .FLOAT4,
        .offset = @offsetOf(particles_mod.ParticleInstanceData, "pos_size"),
    };
    m_desc.layout.attrs[velocity_shd.ATTR_reactive_marker_inst_rotation] = .{
        .buffer_index = 1,
        .format = .FLOAT4,
        .offset = @offsetOf(particles_mod.ParticleInstanceData, "rotation_misc"),
    };
    const pip_m_cpu = sg.makePipeline(m_desc);

    var mg_desc = sg.PipelineDesc{
        .shader = self.marker_gpu_shader,
        .index_type = .UINT16,
        .sample_count = 1,
        .depth = .{
            .pixel_format = depth_fmt,
            .compare = .LESS_EQUAL,
            .write_enabled = false,
        },
        .cull_mode = .NONE,
    };
    mg_desc.colors[0].pixel_format = .RGBA16F;
    mg_desc.layout.buffers[0] = .{ .stride = 4 * @sizeOf(f32) };
    mg_desc.layout.attrs[velocity_shd.ATTR_reactive_marker_gpu_position] = .{
        .buffer_index = 0,
        .format = .FLOAT2,
        .offset = 0,
    };
    mg_desc.layout.buffers[1] = .{
        .step_func = .PER_INSTANCE,
        .step_rate = 1,
        .stride = @sizeOf(particles_mod.GpuParticleSlot),
    };
    mg_desc.layout.attrs[velocity_shd.ATTR_reactive_marker_gpu_gpu_slot0] = .{ .buffer_index = 1, .offset = 0, .format = .FLOAT4 };
    mg_desc.layout.attrs[velocity_shd.ATTR_reactive_marker_gpu_gpu_slot1] = .{ .buffer_index = 1, .offset = 16, .format = .FLOAT4 };
    mg_desc.layout.attrs[velocity_shd.ATTR_reactive_marker_gpu_gpu_slot2] = .{ .buffer_index = 1, .offset = 32, .format = .FLOAT4 };
    mg_desc.layout.attrs[velocity_shd.ATTR_reactive_marker_gpu_gpu_slot3] = .{ .buffer_index = 1, .offset = 48, .format = .FLOAT4 };
    mg_desc.layout.attrs[velocity_shd.ATTR_reactive_marker_gpu_gpu_slot4] = .{ .buffer_index = 1, .offset = 64, .format = .FLOAT4 };
    const pip_m_gpu = sg.makePipeline(mg_desc);

    // Glass reactive-marker pipelines (C.3 v2 glass half): transparent
    // redraw coverage with the constant band-alpha FS. Rigid/skinned
    // share the Vertex-stride buffer 0 (position only for rigid,
    // position+joints+weights for skinned); instanced adds the single
    // CURRENT instance buffer (no prev — coverage needs no history).
    // Sidedness mirrors the opaque path and the main blend twins: a
    // single-sided pane seen from behind draws nothing in the main pass,
    // so its marker must draw nothing either (cull NONE would stamp
    // reactive on background pixels); double-sided glass is covered via
    // its front faces on the single-sided twin and via both faces on the
    // cull-off twin. u16/u32 index-type pairs per shape like everywhere
    // else.
    const pip_g_rigid_u16 = sg.makePipeline(glassRigidDesc(self.glass_marker_shader, .UINT16, depth_fmt, false));
    const pip_g_rigid_u32 = sg.makePipeline(glassRigidDesc(self.glass_marker_shader, .UINT32, depth_fmt, false));
    const pip_g_rigid_u16_ds = sg.makePipeline(glassRigidDesc(self.glass_marker_shader, .UINT16, depth_fmt, true));
    const pip_g_rigid_u32_ds = sg.makePipeline(glassRigidDesc(self.glass_marker_shader, .UINT32, depth_fmt, true));

    const pip_g_inst_u16 = sg.makePipeline(glassInstDesc(self.glass_marker_inst_shader, .UINT16, depth_fmt, false));
    const pip_g_inst_u32 = sg.makePipeline(glassInstDesc(self.glass_marker_inst_shader, .UINT32, depth_fmt, false));
    const pip_g_inst_u16_ds = sg.makePipeline(glassInstDesc(self.glass_marker_inst_shader, .UINT16, depth_fmt, true));
    const pip_g_inst_u32_ds = sg.makePipeline(glassInstDesc(self.glass_marker_inst_shader, .UINT32, depth_fmt, true));

    const pip_g_skin_u16 = sg.makePipeline(glassSkinDesc(self.glass_marker_skin_shader, .UINT16, depth_fmt, false));
    const pip_g_skin_u32 = sg.makePipeline(glassSkinDesc(self.glass_marker_skin_shader, .UINT32, depth_fmt, false));
    const pip_g_skin_u16_ds = sg.makePipeline(glassSkinDesc(self.glass_marker_skin_shader, .UINT16, depth_fmt, true));
    const pip_g_skin_u32_ds = sg.makePipeline(glassSkinDesc(self.glass_marker_skin_shader, .UINT32, depth_fmt, true));

    if (sg.queryPipelineState(pip_r_u16) != .VALID or
        sg.queryPipelineState(pip_r_u32) != .VALID or
        sg.queryPipelineState(pip_r_u16_ds) != .VALID or
        sg.queryPipelineState(pip_r_u32_ds) != .VALID or
        sg.queryPipelineState(pip_i_u16) != .VALID or
        sg.queryPipelineState(pip_i_u32) != .VALID or
        sg.queryPipelineState(pip_i_u16_ds) != .VALID or
        sg.queryPipelineState(pip_i_u32_ds) != .VALID or
        sg.queryPipelineState(pip_s_u16) != .VALID or
        sg.queryPipelineState(pip_s_u32) != .VALID or
        sg.queryPipelineState(pip_s_u16_ds) != .VALID or
        sg.queryPipelineState(pip_s_u32_ds) != .VALID or
        sg.queryPipelineState(pip_m_cpu) != .VALID or
        sg.queryPipelineState(pip_m_gpu) != .VALID or
        sg.queryPipelineState(pip_g_rigid_u16) != .VALID or
        sg.queryPipelineState(pip_g_rigid_u32) != .VALID or
        sg.queryPipelineState(pip_g_rigid_u16_ds) != .VALID or
        sg.queryPipelineState(pip_g_rigid_u32_ds) != .VALID or
        sg.queryPipelineState(pip_g_inst_u16) != .VALID or
        sg.queryPipelineState(pip_g_inst_u32) != .VALID or
        sg.queryPipelineState(pip_g_inst_u16_ds) != .VALID or
        sg.queryPipelineState(pip_g_inst_u32_ds) != .VALID or
        sg.queryPipelineState(pip_g_skin_u16) != .VALID or
        sg.queryPipelineState(pip_g_skin_u32) != .VALID or
        sg.queryPipelineState(pip_g_skin_u16_ds) != .VALID or
        sg.queryPipelineState(pip_g_skin_u32_ds) != .VALID)
    {
        // Roll back every partial handle; the previous working set
        // (if any) stays installed for the next attempt.
        if (pip_r_u16.id != 0) sg.destroyPipeline(pip_r_u16);
        if (pip_r_u32.id != 0) sg.destroyPipeline(pip_r_u32);
        if (pip_r_u16_ds.id != 0) sg.destroyPipeline(pip_r_u16_ds);
        if (pip_r_u32_ds.id != 0) sg.destroyPipeline(pip_r_u32_ds);
        if (pip_i_u16.id != 0) sg.destroyPipeline(pip_i_u16);
        if (pip_i_u32.id != 0) sg.destroyPipeline(pip_i_u32);
        if (pip_i_u16_ds.id != 0) sg.destroyPipeline(pip_i_u16_ds);
        if (pip_i_u32_ds.id != 0) sg.destroyPipeline(pip_i_u32_ds);
        if (pip_s_u16.id != 0) sg.destroyPipeline(pip_s_u16);
        if (pip_s_u32.id != 0) sg.destroyPipeline(pip_s_u32);
        if (pip_s_u16_ds.id != 0) sg.destroyPipeline(pip_s_u16_ds);
        if (pip_s_u32_ds.id != 0) sg.destroyPipeline(pip_s_u32_ds);
        if (pip_m_cpu.id != 0) sg.destroyPipeline(pip_m_cpu);
        if (pip_m_gpu.id != 0) sg.destroyPipeline(pip_m_gpu);
        if (pip_g_rigid_u16.id != 0) sg.destroyPipeline(pip_g_rigid_u16);
        if (pip_g_rigid_u32.id != 0) sg.destroyPipeline(pip_g_rigid_u32);
        if (pip_g_rigid_u16_ds.id != 0) sg.destroyPipeline(pip_g_rigid_u16_ds);
        if (pip_g_rigid_u32_ds.id != 0) sg.destroyPipeline(pip_g_rigid_u32_ds);
        if (pip_g_inst_u16.id != 0) sg.destroyPipeline(pip_g_inst_u16);
        if (pip_g_inst_u32.id != 0) sg.destroyPipeline(pip_g_inst_u32);
        if (pip_g_inst_u16_ds.id != 0) sg.destroyPipeline(pip_g_inst_u16_ds);
        if (pip_g_inst_u32_ds.id != 0) sg.destroyPipeline(pip_g_inst_u32_ds);
        if (pip_g_skin_u16.id != 0) sg.destroyPipeline(pip_g_skin_u16);
        if (pip_g_skin_u32.id != 0) sg.destroyPipeline(pip_g_skin_u32);
        if (pip_g_skin_u16_ds.id != 0) sg.destroyPipeline(pip_g_skin_u16_ds);
        if (pip_g_skin_u32_ds.id != 0) sg.destroyPipeline(pip_g_skin_u32_ds);
        return false;
    }
    destroyPipelines(self);
    self.pip_rigid_u16 = pip_r_u16;
    self.pip_rigid_u32 = pip_r_u32;
    self.pip_rigid_u16_ds = pip_r_u16_ds;
    self.pip_rigid_u32_ds = pip_r_u32_ds;
    self.pip_inst_u16 = pip_i_u16;
    self.pip_inst_u32 = pip_i_u32;
    self.pip_inst_u16_ds = pip_i_u16_ds;
    self.pip_inst_u32_ds = pip_i_u32_ds;
    self.pip_skin_u16 = pip_s_u16;
    self.pip_skin_u32 = pip_s_u32;
    self.pip_skin_u16_ds = pip_s_u16_ds;
    self.pip_skin_u32_ds = pip_s_u32_ds;
    self.pip_marker_cpu = pip_m_cpu;
    self.pip_marker_gpu = pip_m_gpu;
    self.pip_marker_glass_rigid_u16 = pip_g_rigid_u16;
    self.pip_marker_glass_rigid_u32 = pip_g_rigid_u32;
    self.pip_marker_glass_rigid_u16_ds = pip_g_rigid_u16_ds;
    self.pip_marker_glass_rigid_u32_ds = pip_g_rigid_u32_ds;
    self.pip_marker_glass_inst_u16 = pip_g_inst_u16;
    self.pip_marker_glass_inst_u32 = pip_g_inst_u32;
    self.pip_marker_glass_inst_u16_ds = pip_g_inst_u16_ds;
    self.pip_marker_glass_inst_u32_ds = pip_g_inst_u32_ds;
    self.pip_marker_glass_skin_u16 = pip_g_skin_u16;
    self.pip_marker_glass_skin_u32 = pip_g_skin_u32;
    self.pip_marker_glass_skin_u16_ds = pip_g_skin_u16_ds;
    self.pip_marker_glass_skin_u32_ds = pip_g_skin_u32_ds;
    self.depth_format = depth_fmt;
    return true;
}

pub fn destroyPipelines(self: anytype) void {
    if (self.pip_rigid_u16.id != 0) sg.destroyPipeline(self.pip_rigid_u16);
    if (self.pip_rigid_u32.id != 0) sg.destroyPipeline(self.pip_rigid_u32);
    if (self.pip_rigid_u16_ds.id != 0) sg.destroyPipeline(self.pip_rigid_u16_ds);
    if (self.pip_rigid_u32_ds.id != 0) sg.destroyPipeline(self.pip_rigid_u32_ds);
    if (self.pip_inst_u16.id != 0) sg.destroyPipeline(self.pip_inst_u16);
    if (self.pip_inst_u32.id != 0) sg.destroyPipeline(self.pip_inst_u32);
    if (self.pip_inst_u16_ds.id != 0) sg.destroyPipeline(self.pip_inst_u16_ds);
    if (self.pip_inst_u32_ds.id != 0) sg.destroyPipeline(self.pip_inst_u32_ds);
    if (self.pip_skin_u16.id != 0) sg.destroyPipeline(self.pip_skin_u16);
    if (self.pip_skin_u32.id != 0) sg.destroyPipeline(self.pip_skin_u32);
    if (self.pip_skin_u16_ds.id != 0) sg.destroyPipeline(self.pip_skin_u16_ds);
    if (self.pip_skin_u32_ds.id != 0) sg.destroyPipeline(self.pip_skin_u32_ds);
    if (self.pip_marker_cpu.id != 0) sg.destroyPipeline(self.pip_marker_cpu);
    if (self.pip_marker_gpu.id != 0) sg.destroyPipeline(self.pip_marker_gpu);
    if (self.pip_marker_glass_rigid_u16.id != 0) sg.destroyPipeline(self.pip_marker_glass_rigid_u16);
    if (self.pip_marker_glass_rigid_u32.id != 0) sg.destroyPipeline(self.pip_marker_glass_rigid_u32);
    if (self.pip_marker_glass_rigid_u16_ds.id != 0) sg.destroyPipeline(self.pip_marker_glass_rigid_u16_ds);
    if (self.pip_marker_glass_rigid_u32_ds.id != 0) sg.destroyPipeline(self.pip_marker_glass_rigid_u32_ds);
    if (self.pip_marker_glass_inst_u16.id != 0) sg.destroyPipeline(self.pip_marker_glass_inst_u16);
    if (self.pip_marker_glass_inst_u32.id != 0) sg.destroyPipeline(self.pip_marker_glass_inst_u32);
    if (self.pip_marker_glass_inst_u16_ds.id != 0) sg.destroyPipeline(self.pip_marker_glass_inst_u16_ds);
    if (self.pip_marker_glass_inst_u32_ds.id != 0) sg.destroyPipeline(self.pip_marker_glass_inst_u32_ds);
    if (self.pip_marker_glass_skin_u16.id != 0) sg.destroyPipeline(self.pip_marker_glass_skin_u16);
    if (self.pip_marker_glass_skin_u32.id != 0) sg.destroyPipeline(self.pip_marker_glass_skin_u32);
    if (self.pip_marker_glass_skin_u16_ds.id != 0) sg.destroyPipeline(self.pip_marker_glass_skin_u16_ds);
    if (self.pip_marker_glass_skin_u32_ds.id != 0) sg.destroyPipeline(self.pip_marker_glass_skin_u32_ds);
    self.pip_rigid_u16 = .{};
    self.pip_rigid_u32 = .{};
    self.pip_rigid_u16_ds = .{};
    self.pip_rigid_u32_ds = .{};
    self.pip_inst_u16 = .{};
    self.pip_inst_u32 = .{};
    self.pip_inst_u16_ds = .{};
    self.pip_inst_u32_ds = .{};
    self.pip_skin_u16 = .{};
    self.pip_skin_u32 = .{};
    self.pip_skin_u16_ds = .{};
    self.pip_skin_u32_ds = .{};
    self.pip_marker_cpu = .{};
    self.pip_marker_gpu = .{};
    self.pip_marker_glass_rigid_u16 = .{};
    self.pip_marker_glass_rigid_u32 = .{};
    self.pip_marker_glass_rigid_u16_ds = .{};
    self.pip_marker_glass_rigid_u32_ds = .{};
    self.pip_marker_glass_inst_u16 = .{};
    self.pip_marker_glass_inst_u32 = .{};
    self.pip_marker_glass_inst_u16_ds = .{};
    self.pip_marker_glass_inst_u32_ds = .{};
    self.pip_marker_glass_skin_u16 = .{};
    self.pip_marker_glass_skin_u32 = .{};
    self.pip_marker_glass_skin_u16_ds = .{};
    self.pip_marker_glass_skin_u32_ds = .{};
    self.depth_format = .DEFAULT;
}
