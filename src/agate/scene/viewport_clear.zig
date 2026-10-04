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
    /// Single active pipeline slot, keyed by the exact target shape below.
    /// Rebuilt when the requested shape changes (context thread, between
    /// draws — never while bound).
    pipeline: sg.Pipeline = .{},
    shape_samples: i32 = 0,
    shape_format: sg.PixelFormat = .RGBA16F,

    pub fn deinit(self: *ViewportClearPass) void {
        if (!sg.isvalid()) return;
        if (self.pipeline.id != 0) sg.destroyPipeline(self.pipeline);
        if (self.shader.id != 0) sg.destroyShader(self.shader);
        if (self.vb.id != 0) sg.destroyBuffer(self.vb);
        self.* = .{};
    }

    /// Ensures the quad buffer, debug shader, and the pipeline for the
    /// exact target shape (sample count + color format).
    pub fn ensureResources(self: *ViewportClearPass, samples: i32, color_format: sg.PixelFormat) void {
        if (self.vb.id == 0) {
            self.vb = sg.makeBuffer(.{
                .usage = .{ .vertex_buffer = true, .dynamic_update = true },
                .size = 16 * 6 * @sizeOf(debug_pass.Vertex),
            });
        }
        if (self.shader.id == 0) {
            self.shader = sg.makeShader(debug_shd.debugShaderDesc(sg.queryBackend()));
        }
        if (self.pipeline.id == 0 or self.shape_samples != samples or self.shape_format != color_format) {
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
            pip_desc.colors[0].pixel_format = color_format;
            pip_desc.layout.buffers[0] = .{ .stride = @sizeOf(debug_pass.Vertex) };
            pip_desc.layout.attrs[debug_shd.ATTR_debug_position] = .{
                .format = .FLOAT3,
                .offset = @offsetOf(debug_pass.Vertex, "position"),
            };
            pip_desc.layout.attrs[debug_shd.ATTR_debug_color0] = .{
                .format = .FLOAT4,
                .offset = @offsetOf(debug_pass.Vertex, "color"),
            };
            const pip = sg.makePipeline(pip_desc);
            if (sg.queryPipelineState(pip) != .VALID) {
                std.debug.print("[CLEAR PIPELINE FAILED]: shader_state={}, pip_state={}\n", .{
                    sg.queryShaderState(self.shader),
                    sg.queryPipelineState(pip),
                });
                if (pip.id != 0) sg.destroyPipeline(pip);
                return;
            }
            if (self.pipeline.id != 0) sg.destroyPipeline(self.pipeline);
            self.pipeline = pip;
            self.shape_samples = samples;
            self.shape_format = color_format;
        }
    }

    /// Clears the current viewport with a full-quad draw for the exact
    /// target shape (sample count + color format).
    pub fn clear(self: *ViewportClearPass, color: Color4, samples: i32, color_format: sg.PixelFormat) void {
        self.ensureResources(samples, color_format);
        const pip = self.pipeline;
        if (pip.id == 0 or self.vb.id == 0 or sg.queryPipelineState(pip) != .VALID) return;
        if (self.shape_samples != samples or self.shape_format != color_format) return;

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
