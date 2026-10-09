const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;

/// TargetShape defines the full shared geometry contract of a render target
/// or pass attachment: color format, depth format, stencil format, and sample count.
///
/// Unifying target shape eliminates implicit environment depth dependencies
/// across passes and ensures pipeline caches use an authoritative, collision-free key.
pub const TargetShape = struct {
    color_format: sg.PixelFormat = .RGBA16F,
    depth_format: sg.PixelFormat = .DEPTH,
    stencil_format: sg.PixelFormat = .NONE,
    sample_count: i32 = 1,

    pub fn init(color: sg.PixelFormat, depth: sg.PixelFormat, samples: i32) TargetShape {
        return .{
            .color_format = color,
            .depth_format = depth,
            .stencil_format = if (depth == .DEPTH_STENCIL) .DEPTH_STENCIL else .NONE,
            .sample_count = samples,
        };
    }

    /// Resolves `.DEFAULT` tokens to concrete formats.
    /// Scene HDR color defaults to RGBA16F; depth defaults to `defaultDepthFormat()`.
    pub fn resolveEnvironment(self: TargetShape) TargetShape {
        var resolved = self;
        if (resolved.color_format == .DEFAULT) {
            resolved.color_format = .RGBA16F;
        }
        if (resolved.depth_format == .DEFAULT) {
            resolved.depth_format = defaultDepthFormat();
        }
        if (resolved.depth_format == .DEPTH_STENCIL and resolved.stencil_format == .NONE) {
            resolved.stencil_format = .DEPTH_STENCIL;
        }
        return resolved;
    }

    /// Standard linear HDR scene target shape (RGBA16F + environment depth).
    pub fn defaultHdr(sample_count: i32) TargetShape {
        const depth_fmt = defaultDepthFormat();
        return .{
            .color_format = .RGBA16F,
            .depth_format = depth_fmt,
            .stencil_format = if (depth_fmt == .DEPTH_STENCIL) .DEPTH_STENCIL else .NONE,
            .sample_count = sample_count,
        };
    }

    pub fn eql(self: TargetShape, other: TargetShape) bool {
        return self.color_format == other.color_format and
            self.depth_format == other.depth_format and
            self.stencil_format == other.stencil_format and
            self.sample_count == other.sample_count;
    }

    /// 64-bit Wyhash key mixing color, depth, stencil, and sample count for pipeline caches.
    pub fn hash(self: TargetShape) u64 {
        var h = std.hash.Wyhash.init(0x7368_6170_6531_3030); // "shape100"
        h.update(std.mem.asBytes(&self.color_format));
        h.update(std.mem.asBytes(&self.depth_format));
        h.update(std.mem.asBytes(&self.stencil_format));
        h.update(std.mem.asBytes(&self.sample_count));
        return h.final();
    }
};

/// Swapchain depth format fallback chain: environment default, else DEPTH.
pub fn defaultDepthFormat() sg.PixelFormat {
    if (!sg.isvalid()) return .DEPTH;
    const env_def = sg.queryDesc().environment.defaults;
    return if (env_def.depth_format != .DEFAULT and env_def.depth_format != .NONE)
        env_def.depth_format
    else
        .DEPTH;
}
