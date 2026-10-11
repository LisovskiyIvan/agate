//! Main HDR target + TAA history lifecycle. GPU resources remain owned by
//! PostProcessPass; the pass is passed as anytype to avoid a leaf-to-owner import.
const sokol = @import("sokol");
const sg = sokol.gfx;
const post_shd = @import("postprocess_shader");

pub fn fullscreenPipelineDesc(
    shader: sg.Shader,
    color_fmt: sg.PixelFormat,
    sample_count: i32,
    depth_fmt: sg.PixelFormat,
) sg.PipelineDesc {
    var desc = sg.PipelineDesc{
        .shader = shader,
        .index_type = .UINT16,
        .depth = .{
            .pixel_format = depth_fmt,
            .compare = .ALWAYS,
            .write_enabled = false,
        },
        .cull_mode = .NONE,
        .sample_count = sample_count,
    };
    desc.colors[0].pixel_format = color_fmt;
    desc.layout.buffers[0] = .{ .stride = 4 * @sizeOf(f32) };
    desc.layout.attrs[post_shd.ATTR_postprocess_position] = .{
        .format = .FLOAT2,
        .offset = 0,
    };
    desc.layout.attrs[post_shd.ATTR_postprocess_texcoord0] = .{
        .format = .FLOAT2,
        .offset = 2 * @sizeOf(f32),
    };
    return desc;
}

pub fn fullscreenPipelineDescForShape(
    shader: sg.Shader,
    shape: anytype,
) sg.PipelineDesc {
    const resolved = shape.resolveEnvironment();
    return fullscreenPipelineDesc(shader, resolved.color_format, resolved.sample_count, resolved.depth_format);
}

pub fn destroyTargets(self: anytype) void {
    if (self.offscreen_color_att_view.id != 0) sg.destroyView(self.offscreen_color_att_view);
    if (self.offscreen_resolve_att_view.id != 0) sg.destroyView(self.offscreen_resolve_att_view);
    if (self.offscreen_resolve_tex_view.id != 0) sg.destroyView(self.offscreen_resolve_tex_view);
    if (self.offscreen_depth_att_view.id != 0) sg.destroyView(self.offscreen_depth_att_view);
    if (self.offscreen_depth_tex_view.id != 0) sg.destroyView(self.offscreen_depth_tex_view);
    if (self.offscreen_color_image.id != 0) sg.destroyImage(self.offscreen_color_image);
    if (self.offscreen_resolve_image.id != 0) sg.destroyImage(self.offscreen_resolve_image);
    if (self.offscreen_depth_image.id != 0) sg.destroyImage(self.offscreen_depth_image);
    self.offscreen_color_image = .{};
    self.offscreen_color_att_view = .{};
    self.offscreen_resolve_image = .{};
    self.offscreen_resolve_att_view = .{};
    self.offscreen_resolve_tex_view = .{};
    self.offscreen_depth_image = .{};
    self.offscreen_depth_att_view = .{};
    self.offscreen_depth_tex_view = .{};
    self.width = 0;
    self.height = 0;
    self.sample_count = 1;
    self.color_format = .DEFAULT;
    self.depth_format = .DEFAULT;
    destroyTaaHistory(self);
}

pub fn destroyTaaHistory(self: anytype) void {
    for (0..2) |i| {
        if (self.taa_att_views[i].id != 0) sg.destroyView(self.taa_att_views[i]);
        if (self.taa_tex_views[i].id != 0) sg.destroyView(self.taa_tex_views[i]);
        if (self.taa_images[i].id != 0) sg.destroyImage(self.taa_images[i]);
        self.taa_images[i] = .{};
        self.taa_att_views[i] = .{};
        self.taa_tex_views[i] = .{};
    }
    if (self.taa_pipeline.id != 0) sg.destroyPipeline(self.taa_pipeline);
    self.taa_pipeline = .{};
    self.taa_width = 0;
    self.taa_height = 0;
    self.taa_format = .DEFAULT;
    self.taa_read = 0;
}

pub fn taaReset(self: anytype) void {
    destroyTaaHistory(self);
}

pub fn ensureTaaHistory(self: anytype, width: i32, height: i32) bool {
    if (width <= 0 or height <= 0) return false;
    if (!sg.isvalid()) return false;
    const hist_fmt: sg.PixelFormat = .RGBA16F;
    if (self.taa_width == width and self.taa_height == height and self.taa_format == hist_fmt and
        self.taa_images[0].id != 0 and self.taa_images[1].id != 0 and self.taa_pipeline.id != 0) return false;

    destroyTaaHistory(self);

    // Store-first rollback: sokol make* returns nonzero FAILED handles,
    // so every handle lands in the struct before its state check and a
    // single destroyTaaHistory frees the failed handle plus all partial
    // handles. Views are destroyed before images.
    for (0..2) |i| {
        self.taa_images[i] = sg.makeImage(.{
            .usage = .{ .color_attachment = true },
            .width = width,
            .height = height,
            .pixel_format = hist_fmt,
            .sample_count = 1,
        });
        if (sg.queryImageState(self.taa_images[i]) != .VALID) {
            destroyTaaHistory(self);
            return false;
        }
        self.taa_att_views[i] = sg.makeView(.{
            .color_attachment = .{ .image = self.taa_images[i] },
        });
        if (sg.queryViewState(self.taa_att_views[i]) != .VALID) {
            destroyTaaHistory(self);
            return false;
        }
        self.taa_tex_views[i] = sg.makeView(.{
            .texture = .{ .image = self.taa_images[i] },
        });
        if (sg.queryViewState(self.taa_tex_views[i]) != .VALID) {
            destroyTaaHistory(self);
            return false;
        }
    }
    // Explicit 1x capture shape matched to the history format. The
    // display pipeline targets the swapchain shape and must not render
    // into the history targets.
    self.taa_pipeline = sg.makePipeline(fullscreenPipelineDesc(self.postprocess_shader, hist_fmt, 1, .NONE));
    if (sg.queryPipelineState(self.taa_pipeline) != .VALID) {
        destroyTaaHistory(self);
        return false;
    }
    self.taa_width = width;
    self.taa_height = height;
    self.taa_format = hist_fmt;
    self.taa_read = 0;
    return true;
}

pub fn taaAvailable(self: anytype) bool {
    if (self.taa_width <= 0 or self.taa_height <= 0) return false;
    if (!sg.isvalid()) return false;
    if (sg.queryPipelineState(self.taa_pipeline) != .VALID) return false;
    for (0..2) |i| {
        if (sg.queryImageState(self.taa_images[i]) != .VALID) return false;
        if (sg.queryViewState(self.taa_att_views[i]) != .VALID) return false;
        if (sg.queryViewState(self.taa_tex_views[i]) != .VALID) return false;
    }
    return true;
}

pub fn taaReadView(self: anytype) sg.View {
    return self.taa_tex_views[self.taa_read];
}

pub fn taaWriteAttView(self: anytype) sg.View {
    return self.taa_att_views[1 - self.taa_read];
}

pub fn targetsValid(self: anytype) bool {
    if (self.width <= 0 or self.height <= 0) return false;
    if (!sg.isvalid()) return false;
    if (sg.queryImageState(self.offscreen_color_image) != .VALID) return false;
    if (sg.queryViewState(self.offscreen_color_att_view) != .VALID) return false;
    if (sg.queryViewState(self.offscreen_resolve_tex_view) != .VALID) return false;
    if (sg.queryImageState(self.offscreen_depth_image) != .VALID) return false;
    if (sg.queryViewState(self.offscreen_depth_att_view) != .VALID) return false;
    if (self.sample_count > 1) {
        if (sg.queryImageState(self.offscreen_resolve_image) != .VALID) return false;
        if (sg.queryViewState(self.offscreen_resolve_att_view) != .VALID) return false;
    } else {
        if (sg.queryViewState(self.offscreen_depth_tex_view) != .VALID) return false;
    }
    return true;
}

pub fn resize(
    self: anytype,
    width: i32,
    height: i32,
    sample_count: i32,
) bool {
    if (width <= 0 or height <= 0) return false;
    if (!sg.isvalid()) return false;
    const samples: i32 = if (sample_count < 1) 1 else sample_count;
    const env_def = sg.queryDesc().environment.defaults;
    const actual_fmt: sg.PixelFormat = .RGBA16F;
    if (self.width == width and self.height == height and
        self.sample_count == samples and self.color_format == actual_fmt) return targetsValid(self);

    destroyTargets(self);

    const depth_fmt: sg.PixelFormat = if (env_def.depth_format != .DEFAULT and env_def.depth_format != .NONE) env_def.depth_format else .DEPTH;
    self.depth_format = depth_fmt;

    // Store-first rollback throughout: sokol make* returns nonzero
    // FAILED handles, so every handle lands in the struct before its
    // state check and a single destroyTargets frees the failed handle
    // plus all partial handles. Views are destroyed before images.
    //
    // Color: attachment at the full sample count; when resolving, a
    // separate 1x resolve image (usage.resolve_attachment) receives the
    // MSAA resolve at end of pass and carries the texture view.
    self.offscreen_color_image = sg.makeImage(.{
        .usage = .{ .color_attachment = true },
        .width = width,
        .height = height,
        .pixel_format = actual_fmt,
        .sample_count = samples,
    });
    if (sg.queryImageState(self.offscreen_color_image) != .VALID) {
        destroyTargets(self);
        return false;
    }
    self.offscreen_color_att_view = sg.makeView(.{
        .color_attachment = .{ .image = self.offscreen_color_image },
    });
    if (sg.queryViewState(self.offscreen_color_att_view) != .VALID) {
        destroyTargets(self);
        return false;
    }

    if (samples > 1) {
        self.offscreen_resolve_image = sg.makeImage(.{
            .usage = .{ .resolve_attachment = true },
            .width = width,
            .height = height,
            .pixel_format = actual_fmt,
            .sample_count = 1,
        });
        if (sg.queryImageState(self.offscreen_resolve_image) != .VALID) {
            destroyTargets(self);
            return false;
        }
        self.offscreen_resolve_att_view = sg.makeView(.{
            .resolve_attachment = .{ .image = self.offscreen_resolve_image },
        });
        if (sg.queryViewState(self.offscreen_resolve_att_view) != .VALID) {
            destroyTargets(self);
            return false;
        }
        self.offscreen_resolve_tex_view = sg.makeView(.{
            .texture = .{ .image = self.offscreen_resolve_image },
        });
        if (sg.queryViewState(self.offscreen_resolve_tex_view) != .VALID) {
            destroyTargets(self);
            return false;
        }
    } else {
        // 1x shape: postfx samples the color image directly.
        self.offscreen_resolve_tex_view = sg.makeView(.{
            .texture = .{ .image = self.offscreen_color_image },
        });
        if (sg.queryViewState(self.offscreen_resolve_tex_view) != .VALID) {
            destroyTargets(self);
            return false;
        }
    }

    // Depth: same sample count as color (sokol validation requires the
    // match). Only the 1x depth gets a texture view; MSAA depth is
    // write-only for the post chain (depth-consuming effects stay gated
    // off while MSAA is active).
    self.offscreen_depth_image = sg.makeImage(.{
        .usage = .{ .depth_stencil_attachment = true },
        .width = width,
        .height = height,
        .pixel_format = depth_fmt,
        .sample_count = samples,
    });
    if (sg.queryImageState(self.offscreen_depth_image) != .VALID) {
        destroyTargets(self);
        return false;
    }
    self.offscreen_depth_att_view = sg.makeView(.{
        .depth_stencil_attachment = .{ .image = self.offscreen_depth_image },
    });
    if (sg.queryViewState(self.offscreen_depth_att_view) != .VALID) {
        destroyTargets(self);
        return false;
    }
    if (samples == 1) {
        self.offscreen_depth_tex_view = sg.makeView(.{
            .texture = .{ .image = self.offscreen_depth_image },
        });
        if (sg.queryViewState(self.offscreen_depth_tex_view) != .VALID) {
            destroyTargets(self);
            return false;
        }
    }

    self.width = width;
    self.height = height;
    self.sample_count = samples;
    self.color_format = actual_fmt;
    if (!targetsValid(self)) {
        destroyTargets(self);
        return false;
    }
    return true;
}

pub fn depthSampleView(self: anytype) sg.View {
    if (self.offscreen_depth_tex_view.id != 0) return self.offscreen_depth_tex_view;
    return self.offscreen_resolve_tex_view;
}
