const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const math = @import("math");
const Mat4 = math.Mat4;
const Color4 = math.Color4;

const debug_shd = @import("debug_shader");
const debug_pass = @import("../passes/debug_pass.zig");
const upload_meter = @import("../gpu_upload_meter.zig");

/// Renders a full-quad clear in the current viewport/scissor.
/// Used for per-camera viewport clears in multi-camera / PIP rendering.
pub const ViewportClearPass = struct {
    vb: sg.Buffer = .{},
    shader: sg.Shader = .{},
    pipeline: sg.Pipeline = .{},
    pipeline_msaa: sg.Pipeline = .{},

    pub fn deinit(self: *ViewportClearPass) void {
        if (!sg.isvalid()) return;
        if (self.pipeline.id != 0) sg.destroyPipeline(self.pipeline);
        if (self.pipeline_msaa.id != 0) sg.destroyPipeline(self.pipeline_msaa);
        if (self.shader.id != 0) sg.destroyShader(self.shader);
        if (self.vb.id != 0) sg.destroyBuffer(self.vb);
        self.* = .{};
    }

    pub fn ensureResources(self: *ViewportClearPass, samples: i32) void {
        if (self.vb.id == 0) {
            self.vb = sg.makeBuffer(.{
                .usage = .{ .vertex_buffer = true, .dynamic_update = true },
                .size = 16 * 6 * @sizeOf(debug_pass.Vertex),
            });
        }
        if (self.shader.id == 0) {
            self.shader = sg.makeShader(debug_shd.debugShaderDesc(sg.queryBackend()));
        }
        const target_pip = if (samples > 1) &self.pipeline_msaa else &self.pipeline;
        if (target_pip.id == 0) {
            var pip_desc = sg.PipelineDesc{
                .shader = self.shader,
                .index_type = .NONE,
                .primitive_type = .TRIANGLES,
                .depth = .{
                    .compare = .ALWAYS,
                    .write_enabled = true,
                },
                .cull_mode = .NONE,
                .sample_count = samples,
            };
            pip_desc.layout.buffers[0] = .{ .stride = @sizeOf(debug_pass.Vertex) };
            pip_desc.layout.attrs[debug_shd.ATTR_debug_position] = .{
                .format = .FLOAT3,
                .offset = @offsetOf(debug_pass.Vertex, "position"),
            };
            pip_desc.layout.attrs[debug_shd.ATTR_debug_color0] = .{
                .format = .FLOAT4,
                .offset = @offsetOf(debug_pass.Vertex, "color"),
            };
            target_pip.* = sg.makePipeline(pip_desc);
            if (sg.queryPipelineState(target_pip.*) != .VALID) {
                std.debug.print("[CLEAR PIPELINE FAILED]: shader_state={}, pip_state={}\n", .{
                    sg.queryShaderState(self.shader),
                    sg.queryPipelineState(target_pip.*),
                });
            }
        }
    }

    pub fn clear(self: *ViewportClearPass, color: Color4, samples: i32) void {
        self.ensureResources(samples);
        const pip = if (samples > 1) self.pipeline_msaa else self.pipeline;
        if (pip.id == 0 or self.vb.id == 0 or sg.queryPipelineState(pip) != .VALID) return;

        const clear_verts = [_]debug_pass.Vertex{
            .{ .position = .{ -1.0, -1.0, 1.0 }, .color = .{ color.r, color.g, color.b, color.a } },
            .{ .position = .{ 1.0, -1.0, 1.0 }, .color = .{ color.r, color.g, color.b, color.a } },
            .{ .position = .{ 1.0, 1.0, 1.0 }, .color = .{ color.r, color.g, color.b, color.a } },
            .{ .position = .{ -1.0, -1.0, 1.0 }, .color = .{ color.r, color.g, color.b, color.a } },
            .{ .position = .{ 1.0, 1.0, 1.0 }, .color = .{ color.r, color.g, color.b, color.a } },
            .{ .position = .{ -1.0, 1.0, 1.0 }, .color = .{ color.r, color.g, color.b, color.a } },
        };
        const offset = sg.appendBuffer(self.vb, sg.asRange(&clear_verts));
        if (offset < 0) return;
        // Учёт динамики: 6 вершин clear-квада через appendBuffer (байты те же — стрим в GPU-буфер).
        upload_meter.record(clear_verts.len * @sizeOf(debug_pass.Vertex));

        sg.applyPipeline(pip);
        var bind = sg.Bindings{};
        bind.vertex_buffers[0] = self.vb;
        bind.vertex_buffer_offsets[0] = offset;
        sg.applyBindings(bind);

        const vs_params = debug_shd.VsParams{
            .mvp = Mat4.identity,
        };
        sg.applyUniforms(debug_shd.UB_vs_params, sg.asRange(&vs_params));
        sg.draw(0, 6, 1);
    }
};
