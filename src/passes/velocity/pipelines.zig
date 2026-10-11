//! Velocity pipeline factory + destroy. Split out of `velocity_pass.zig`.
//!
//! Owns nothing: builds (or rebuilds on format change) the six opaque
//! pipelines, the two particle reactive-marker pipelines, plus the six glass
//! reactive-marker pipelines for the borrowed main-depth format, and destroys
//! them. The pass is passed as `anytype` to avoid a leaf-to-owner import
//! (same discipline as `passes/postprocess/targets.zig` and `passes/shadow/`);
//! this module never imports `velocity_pass.zig` back.
//!
//! Bodies are a mechanical move of `VelocityPass.ensurePipelinesFor` /
//! `VelocityPass.destroyPipelines`: all fourteen builds, the all-or-nothing
//! validation, the partial-handle rollback, and the destroy/publish ordering
//! are identical. Opaque pipelines share the shared-depth plan (writes off,
//! EQUAL compare); all eight marker pipelines share the marker plan (writes
//! off, LESS_EQUAL compare).
const sokol = @import("sokol");
const sg = sokol.gfx;
const velocity_shd = @import("velocity_shader");
const math = @import("math");
const Mat4 = math.Mat4;
const mesh_mod = @import("../../mesh.zig");
const Vertex = mesh_mod.Vertex;
const particles_mod = @import("../../particles.zig");

/// Builds (or rebuilds on format change) the six opaque pipelines, the
/// two particle reactive-marker pipelines, plus the six glass
/// reactive-marker pipelines for the borrowed main-depth format.
/// Locals-first like everywhere else: the working set is
/// preserved when a rebuild fails. Opaque pipelines share the
/// shared-depth plan (writes off, EQUAL compare — see
/// velocityDepthState); all eight marker pipelines share the marker
/// plan (writes off, LESS_EQUAL compare — see reactiveMarkerDepthState).
/// All fourteen are all-or-nothing: a failed build skips the whole pass
/// for the frame (fail-closed to the zero-mask fallback, never a half
/// pass).
pub fn ensurePipelinesFor(self: anytype, depth_fmt: sg.PixelFormat) bool {
    if (!sg.isvalid()) return false;
    if (depth_fmt == .DEFAULT or depth_fmt == .NONE) return false;
    if (self.depth_format == depth_fmt and
        sg.queryPipelineState(self.pip_rigid_u16) == .VALID and
        sg.queryPipelineState(self.pip_rigid_u32) == .VALID and
        sg.queryPipelineState(self.pip_inst_u16) == .VALID and
        sg.queryPipelineState(self.pip_inst_u32) == .VALID and
        sg.queryPipelineState(self.pip_skin_u16) == .VALID and
        sg.queryPipelineState(self.pip_skin_u32) == .VALID and
        sg.queryPipelineState(self.pip_marker_cpu) == .VALID and
        sg.queryPipelineState(self.pip_marker_gpu) == .VALID and
        sg.queryPipelineState(self.pip_marker_glass_rigid_u16) == .VALID and
        sg.queryPipelineState(self.pip_marker_glass_rigid_u32) == .VALID and
        sg.queryPipelineState(self.pip_marker_glass_inst_u16) == .VALID and
        sg.queryPipelineState(self.pip_marker_glass_inst_u32) == .VALID and
        sg.queryPipelineState(self.pip_marker_glass_skin_u16) == .VALID and
        sg.queryPipelineState(self.pip_marker_glass_skin_u32) == .VALID)
    {
        return true;
    }
    if (self.rigid_shader.id == 0 or self.inst_shader.id == 0 or self.skin_shader.id == 0) return false;
    if (self.marker_shader.id == 0 or self.marker_gpu_shader.id == 0) return false;
    if (self.glass_marker_shader.id == 0 or self.glass_marker_inst_shader.id == 0 or self.glass_marker_skin_shader.id == 0) return false;

    var r_desc = sg.PipelineDesc{
        .shader = self.rigid_shader,
        .index_type = .UINT16,
        .sample_count = 1,
        .depth = .{
            .pixel_format = depth_fmt,
            .compare = .EQUAL,
            .write_enabled = false,
        },
        .cull_mode = .BACK,
        .face_winding = .CCW,
    };
    r_desc.colors[0].pixel_format = .RGBA16F;
    r_desc.layout.buffers[0] = .{ .stride = @sizeOf(Vertex) };
    r_desc.layout.attrs[velocity_shd.ATTR_velocity_position] = .{
        .format = .FLOAT3,
        .offset = @offsetOf(Vertex, "position"),
    };
    r_desc.layout.attrs[velocity_shd.ATTR_velocity_color0] = .{
        .format = .FLOAT4,
        .offset = @offsetOf(Vertex, "color"),
    };
    r_desc.layout.attrs[velocity_shd.ATTR_velocity_texcoord0] = .{
        .format = .FLOAT2,
        .offset = @offsetOf(Vertex, "uv"),
    };
    r_desc.layout.attrs[velocity_shd.ATTR_velocity_texcoord1] = .{
        .format = .FLOAT2,
        .offset = @offsetOf(Vertex, "uv1"),
    };
    const pip_r_u16 = sg.makePipeline(r_desc);
    r_desc.index_type = .UINT32;
    const pip_r_u32 = sg.makePipeline(r_desc);

    var i_desc = sg.PipelineDesc{
        .shader = self.inst_shader,
        .index_type = .UINT16,
        .sample_count = 1,
        .depth = .{
            .pixel_format = depth_fmt,
            .compare = .EQUAL,
            .write_enabled = false,
        },
        .cull_mode = .BACK,
        .face_winding = .CCW,
    };
    i_desc.colors[0].pixel_format = .RGBA16F;
    i_desc.layout.buffers[0] = .{ .stride = @sizeOf(Vertex) };
    i_desc.layout.attrs[velocity_shd.ATTR_velocity_instanced_position] = .{
        .buffer_index = 0,
        .format = .FLOAT3,
        .offset = @offsetOf(Vertex, "position"),
    };
    i_desc.layout.attrs[velocity_shd.ATTR_velocity_instanced_color0] = .{
        .buffer_index = 0,
        .format = .FLOAT4,
        .offset = @offsetOf(Vertex, "color"),
    };
    i_desc.layout.attrs[velocity_shd.ATTR_velocity_instanced_texcoord0] = .{
        .buffer_index = 0,
        .format = .FLOAT2,
        .offset = @offsetOf(Vertex, "uv"),
    };
    i_desc.layout.attrs[velocity_shd.ATTR_velocity_instanced_texcoord1] = .{
        .buffer_index = 0,
        .format = .FLOAT2,
        .offset = @offsetOf(Vertex, "uv1"),
    };
    i_desc.layout.buffers[1] = .{
        .step_func = .PER_INSTANCE,
        .step_rate = 1,
        .stride = @sizeOf(Mat4),
    };
    i_desc.layout.attrs[velocity_shd.ATTR_velocity_instanced_inst_mat0] = .{ .buffer_index = 1, .offset = 0, .format = .FLOAT4 };
    i_desc.layout.attrs[velocity_shd.ATTR_velocity_instanced_inst_mat1] = .{ .buffer_index = 1, .offset = 16, .format = .FLOAT4 };
    i_desc.layout.attrs[velocity_shd.ATTR_velocity_instanced_inst_mat2] = .{ .buffer_index = 1, .offset = 32, .format = .FLOAT4 };
    i_desc.layout.attrs[velocity_shd.ATTR_velocity_instanced_inst_mat3] = .{ .buffer_index = 1, .offset = 48, .format = .FLOAT4 };

    i_desc.layout.buffers[2] = .{
        .step_func = .PER_INSTANCE,
        .step_rate = 1,
        .stride = @sizeOf(Mat4),
    };
    i_desc.layout.attrs[velocity_shd.ATTR_velocity_instanced_prev_inst_mat0] = .{ .buffer_index = 2, .offset = 0, .format = .FLOAT4 };
    i_desc.layout.attrs[velocity_shd.ATTR_velocity_instanced_prev_inst_mat1] = .{ .buffer_index = 2, .offset = 16, .format = .FLOAT4 };
    i_desc.layout.attrs[velocity_shd.ATTR_velocity_instanced_prev_inst_mat2] = .{ .buffer_index = 2, .offset = 32, .format = .FLOAT4 };
    i_desc.layout.attrs[velocity_shd.ATTR_velocity_instanced_prev_inst_mat3] = .{ .buffer_index = 2, .offset = 48, .format = .FLOAT4 };

    const pip_i_u16 = sg.makePipeline(i_desc);
    i_desc.index_type = .UINT32;
    const pip_i_u32 = sg.makePipeline(i_desc);

    var s_desc = sg.PipelineDesc{
        .shader = self.skin_shader,
        .index_type = .UINT16,
        .sample_count = 1,
        .depth = .{
            .pixel_format = depth_fmt,
            .compare = .EQUAL,
            .write_enabled = false,
        },
        .cull_mode = .BACK,
        .face_winding = .CCW,
    };
    s_desc.colors[0].pixel_format = .RGBA16F;
    s_desc.layout.buffers[0] = .{ .stride = @sizeOf(Vertex) };
    s_desc.layout.attrs[velocity_shd.ATTR_velocity_skinned_position] = .{
        .format = .FLOAT3,
        .offset = @offsetOf(Vertex, "position"),
    };
    s_desc.layout.attrs[velocity_shd.ATTR_velocity_skinned_joints] = .{
        .format = .FLOAT4,
        .offset = @offsetOf(Vertex, "joints"),
    };
    s_desc.layout.attrs[velocity_shd.ATTR_velocity_skinned_weights] = .{
        .format = .FLOAT4,
        .offset = @offsetOf(Vertex, "weights"),
    };
    s_desc.layout.attrs[velocity_shd.ATTR_velocity_skinned_color0] = .{
        .format = .FLOAT4,
        .offset = @offsetOf(Vertex, "color"),
    };
    s_desc.layout.attrs[velocity_shd.ATTR_velocity_skinned_texcoord0] = .{
        .format = .FLOAT2,
        .offset = @offsetOf(Vertex, "uv"),
    };
    s_desc.layout.attrs[velocity_shd.ATTR_velocity_skinned_texcoord1] = .{
        .format = .FLOAT2,
        .offset = @offsetOf(Vertex, "uv1"),
    };
    const pip_s_u16 = sg.makePipeline(s_desc);
    s_desc.index_type = .UINT32;
    const pip_s_u32 = sg.makePipeline(s_desc);

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
    // BACK cull like the opaque path (see the field docs); u16/u32
    // index-type pairs per shape like everywhere else.
    var gr_desc = sg.PipelineDesc{
        .shader = self.glass_marker_shader,
        .index_type = .UINT16,
        .sample_count = 1,
        .depth = .{
            .pixel_format = depth_fmt,
            .compare = .LESS_EQUAL,
            .write_enabled = false,
        },
        .cull_mode = .BACK,
        .face_winding = .CCW,
    };
    gr_desc.colors[0].pixel_format = .RGBA16F;
    gr_desc.layout.buffers[0] = .{ .stride = @sizeOf(Vertex) };
    gr_desc.layout.attrs[velocity_shd.ATTR_glass_marker_position] = .{
        .format = .FLOAT3,
        .offset = @offsetOf(Vertex, "position"),
    };
    const pip_g_rigid_u16 = sg.makePipeline(gr_desc);
    gr_desc.index_type = .UINT32;
    const pip_g_rigid_u32 = sg.makePipeline(gr_desc);

    var gi_desc = sg.PipelineDesc{
        .shader = self.glass_marker_inst_shader,
        .index_type = .UINT16,
        .sample_count = 1,
        .depth = .{
            .pixel_format = depth_fmt,
            .compare = .LESS_EQUAL,
            .write_enabled = false,
        },
        .cull_mode = .BACK,
        .face_winding = .CCW,
    };
    gi_desc.colors[0].pixel_format = .RGBA16F;
    gi_desc.layout.buffers[0] = .{ .stride = @sizeOf(Vertex) };
    gi_desc.layout.attrs[velocity_shd.ATTR_glass_marker_inst_position] = .{
        .buffer_index = 0,
        .format = .FLOAT3,
        .offset = @offsetOf(Vertex, "position"),
    };
    gi_desc.layout.buffers[1] = .{
        .step_func = .PER_INSTANCE,
        .step_rate = 1,
        .stride = @sizeOf(Mat4),
    };
    gi_desc.layout.attrs[velocity_shd.ATTR_glass_marker_inst_inst_mat0] = .{ .buffer_index = 1, .offset = 0, .format = .FLOAT4 };
    gi_desc.layout.attrs[velocity_shd.ATTR_glass_marker_inst_inst_mat1] = .{ .buffer_index = 1, .offset = 16, .format = .FLOAT4 };
    gi_desc.layout.attrs[velocity_shd.ATTR_glass_marker_inst_inst_mat2] = .{ .buffer_index = 1, .offset = 32, .format = .FLOAT4 };
    gi_desc.layout.attrs[velocity_shd.ATTR_glass_marker_inst_inst_mat3] = .{ .buffer_index = 1, .offset = 48, .format = .FLOAT4 };
    const pip_g_inst_u16 = sg.makePipeline(gi_desc);
    gi_desc.index_type = .UINT32;
    const pip_g_inst_u32 = sg.makePipeline(gi_desc);

    var gs_desc = sg.PipelineDesc{
        .shader = self.glass_marker_skin_shader,
        .index_type = .UINT16,
        .sample_count = 1,
        .depth = .{
            .pixel_format = depth_fmt,
            .compare = .LESS_EQUAL,
            .write_enabled = false,
        },
        .cull_mode = .BACK,
        .face_winding = .CCW,
    };
    gs_desc.colors[0].pixel_format = .RGBA16F;
    gs_desc.layout.buffers[0] = .{ .stride = @sizeOf(Vertex) };
    gs_desc.layout.attrs[velocity_shd.ATTR_glass_marker_skin_position] = .{
        .format = .FLOAT3,
        .offset = @offsetOf(Vertex, "position"),
    };
    gs_desc.layout.attrs[velocity_shd.ATTR_glass_marker_skin_joints] = .{
        .format = .FLOAT4,
        .offset = @offsetOf(Vertex, "joints"),
    };
    gs_desc.layout.attrs[velocity_shd.ATTR_glass_marker_skin_weights] = .{
        .format = .FLOAT4,
        .offset = @offsetOf(Vertex, "weights"),
    };
    const pip_g_skin_u16 = sg.makePipeline(gs_desc);
    gs_desc.index_type = .UINT32;
    const pip_g_skin_u32 = sg.makePipeline(gs_desc);

    if (sg.queryPipelineState(pip_r_u16) != .VALID or
        sg.queryPipelineState(pip_r_u32) != .VALID or
        sg.queryPipelineState(pip_i_u16) != .VALID or
        sg.queryPipelineState(pip_i_u32) != .VALID or
        sg.queryPipelineState(pip_s_u16) != .VALID or
        sg.queryPipelineState(pip_s_u32) != .VALID or
        sg.queryPipelineState(pip_m_cpu) != .VALID or
        sg.queryPipelineState(pip_m_gpu) != .VALID or
        sg.queryPipelineState(pip_g_rigid_u16) != .VALID or
        sg.queryPipelineState(pip_g_rigid_u32) != .VALID or
        sg.queryPipelineState(pip_g_inst_u16) != .VALID or
        sg.queryPipelineState(pip_g_inst_u32) != .VALID or
        sg.queryPipelineState(pip_g_skin_u16) != .VALID or
        sg.queryPipelineState(pip_g_skin_u32) != .VALID)
    {
        // Roll back every partial handle; the previous working set
        // (if any) stays installed for the next attempt.
        if (pip_r_u16.id != 0) sg.destroyPipeline(pip_r_u16);
        if (pip_r_u32.id != 0) sg.destroyPipeline(pip_r_u32);
        if (pip_i_u16.id != 0) sg.destroyPipeline(pip_i_u16);
        if (pip_i_u32.id != 0) sg.destroyPipeline(pip_i_u32);
        if (pip_s_u16.id != 0) sg.destroyPipeline(pip_s_u16);
        if (pip_s_u32.id != 0) sg.destroyPipeline(pip_s_u32);
        if (pip_m_cpu.id != 0) sg.destroyPipeline(pip_m_cpu);
        if (pip_m_gpu.id != 0) sg.destroyPipeline(pip_m_gpu);
        if (pip_g_rigid_u16.id != 0) sg.destroyPipeline(pip_g_rigid_u16);
        if (pip_g_rigid_u32.id != 0) sg.destroyPipeline(pip_g_rigid_u32);
        if (pip_g_inst_u16.id != 0) sg.destroyPipeline(pip_g_inst_u16);
        if (pip_g_inst_u32.id != 0) sg.destroyPipeline(pip_g_inst_u32);
        if (pip_g_skin_u16.id != 0) sg.destroyPipeline(pip_g_skin_u16);
        if (pip_g_skin_u32.id != 0) sg.destroyPipeline(pip_g_skin_u32);
        return false;
    }
    destroyPipelines(self);
    self.pip_rigid_u16 = pip_r_u16;
    self.pip_rigid_u32 = pip_r_u32;
    self.pip_inst_u16 = pip_i_u16;
    self.pip_inst_u32 = pip_i_u32;
    self.pip_skin_u16 = pip_s_u16;
    self.pip_skin_u32 = pip_s_u32;
    self.pip_marker_cpu = pip_m_cpu;
    self.pip_marker_gpu = pip_m_gpu;
    self.pip_marker_glass_rigid_u16 = pip_g_rigid_u16;
    self.pip_marker_glass_rigid_u32 = pip_g_rigid_u32;
    self.pip_marker_glass_inst_u16 = pip_g_inst_u16;
    self.pip_marker_glass_inst_u32 = pip_g_inst_u32;
    self.pip_marker_glass_skin_u16 = pip_g_skin_u16;
    self.pip_marker_glass_skin_u32 = pip_g_skin_u32;
    self.depth_format = depth_fmt;
    return true;
}

pub fn destroyPipelines(self: anytype) void {
    if (self.pip_rigid_u16.id != 0) sg.destroyPipeline(self.pip_rigid_u16);
    if (self.pip_rigid_u32.id != 0) sg.destroyPipeline(self.pip_rigid_u32);
    if (self.pip_inst_u16.id != 0) sg.destroyPipeline(self.pip_inst_u16);
    if (self.pip_inst_u32.id != 0) sg.destroyPipeline(self.pip_inst_u32);
    if (self.pip_skin_u16.id != 0) sg.destroyPipeline(self.pip_skin_u16);
    if (self.pip_skin_u32.id != 0) sg.destroyPipeline(self.pip_skin_u32);
    if (self.pip_marker_cpu.id != 0) sg.destroyPipeline(self.pip_marker_cpu);
    if (self.pip_marker_gpu.id != 0) sg.destroyPipeline(self.pip_marker_gpu);
    if (self.pip_marker_glass_rigid_u16.id != 0) sg.destroyPipeline(self.pip_marker_glass_rigid_u16);
    if (self.pip_marker_glass_rigid_u32.id != 0) sg.destroyPipeline(self.pip_marker_glass_rigid_u32);
    if (self.pip_marker_glass_inst_u16.id != 0) sg.destroyPipeline(self.pip_marker_glass_inst_u16);
    if (self.pip_marker_glass_inst_u32.id != 0) sg.destroyPipeline(self.pip_marker_glass_inst_u32);
    if (self.pip_marker_glass_skin_u16.id != 0) sg.destroyPipeline(self.pip_marker_glass_skin_u16);
    if (self.pip_marker_glass_skin_u32.id != 0) sg.destroyPipeline(self.pip_marker_glass_skin_u32);
    self.pip_rigid_u16 = .{};
    self.pip_rigid_u32 = .{};
    self.pip_inst_u16 = .{};
    self.pip_inst_u32 = .{};
    self.pip_skin_u16 = .{};
    self.pip_skin_u32 = .{};
    self.pip_marker_cpu = .{};
    self.pip_marker_gpu = .{};
    self.pip_marker_glass_rigid_u16 = .{};
    self.pip_marker_glass_rigid_u32 = .{};
    self.pip_marker_glass_inst_u16 = .{};
    self.pip_marker_glass_inst_u32 = .{};
    self.pip_marker_glass_skin_u16 = .{};
    self.pip_marker_glass_skin_u32 = .{};
    self.depth_format = .DEFAULT;
}
