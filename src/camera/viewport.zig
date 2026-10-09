//! Normalized viewport shared by all camera types. Split out of
//! `camera.zig` (facade): `camera.zig` re-exports `Viewport` so the public
//! API is unchanged. Leaf: no sibling imports.

const std = @import("std");

pub const Viewport = struct {
    x: f32 = 0.0,
    y: f32 = 0.0,
    width: f32 = 1.0,
    height: f32 = 1.0,

    /// Converts normalized viewport [0..1] to pixel rect given screen width and height.
    pub fn toPixelRect(self: Viewport, screen_w: i32, screen_h: i32) PixelRect {
        const sw: f32 = @floatFromInt(screen_w);
        const sh: f32 = @floatFromInt(screen_h);
        return .{
            .x = @intFromFloat(@round(self.x * sw)),
            .y = @intFromFloat(@round(self.y * sh)),
            .width = @max(1, @as(i32, @intFromFloat(@round(self.width * sw)))),
            .height = @max(1, @as(i32, @intFromFloat(@round(self.height * sh)))),
        };
    }

    pub const PixelRect = struct {
        x: i32,
        y: i32,
        width: i32,
        height: i32,

        pub fn aspect(self: PixelRect) f32 {
            const h: f32 = @floatFromInt(self.height);
            return if (h > 0.0) @as(f32, @floatFromInt(self.width)) / h else 1.0;
        }
    };
};
