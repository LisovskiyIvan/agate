const std = @import("std");
const sokol = @import("sokol");
const sapp = sokol.app;
const sg = sokol.gfx;
const slog = sokol.log;
const sglue = sokol.glue;
const z = @import("zenderer");

var gpa = std.heap.DebugAllocator(.{}){};
var scene: z.Scene = undefined;
var box: *z.Mesh = undefined;
var camera: z.ArcRotateCamera = undefined;

export fn init() callconv(.c) void {
    sg.setup(.{
        .environment = sglue.environment(),
        .logger = .{ .func = slog.func },
    });

    const allocator = gpa.allocator();
    scene = z.Scene.init(allocator);

    // Babylon.js style: настройка орбитальной камеры
    camera = z.ArcRotateCamera.init("MainCamera", .{
        .alpha = std.math.pi / 4.0,
        .beta = std.math.pi / 3.0,
        .radius = 5.5,
        .target = z.Vec3.zero,
    });
    scene.active_camera = camera;

    // Babylon.js style: свет HemisphericLight (небо + земля)
    _ = scene.createHemisphericLight("hemiLight", .{
        .direction = z.Vec3.new(0.6, 1.0, 0.4),
        .diffuse = z.Color3.white,
        .ground_color = z.Color3.new(0.2, 0.22, 0.28),
        .intensity = 1.0,
    });

    // Babylon.js style: создание меша через MeshBuilder
    box = z.MeshBuilder.createBox(&scene, "box", .{
        .size = 2.0,
    }) catch |err| {
        std.debug.panic("Failed to create box: {}", .{err});
    };
}

export fn frame() callconv(.c) void {
    const dt: f32 = @floatCast(sapp.frameDuration() * 60.0);

    // Вращаем куб
    box.rotation.x += 0.8 * dt;
    box.rotation.y += 1.6 * dt;

    scene.render();
}

export fn cleanup() callconv(.c) void {
    scene.deinit();
    _ = gpa.deinit();
    sg.shutdown();
}

var color_toggle: usize = 0;
const bg_colors = [_]z.Color4{
    z.Color4.new(0.12, 0.14, 0.18, 1.0),
    z.Color4.new(0.25, 0.12, 0.14, 1.0),
    z.Color4.new(0.12, 0.22, 0.16, 1.0),
    z.Color4.new(0.14, 0.18, 0.28, 1.0),
};

export fn event(ev: [*c]const sapp.Event) callconv(.c) void {
    switch (ev.*.type) {
        .MOUSE_DOWN => {
            color_toggle += 1;
            scene.clear_color = bg_colors[color_toggle % bg_colors.len];
        },
        .KEY_DOWN => switch (ev.*.key_code) {
            .SPACE => {
                color_toggle += 1;
                scene.clear_color = bg_colors[color_toggle % bg_colors.len];
            },
            .ESCAPE => sapp.quit(),
            else => {},
        },
        else => {},
    }
}

pub fn main() void {
    sapp.run(.{
        .init_cb = init,
        .frame_cb = frame,
        .cleanup_cb = cleanup,
        .event_cb = event,
        .window_title = "zenderer (Babylon.js-style 3D in Zig)",
        .width = 800,
        .height = 600,
        .sample_count = 4, // MSAA 4x
        .logger = .{ .func = slog.func },
    });
}
