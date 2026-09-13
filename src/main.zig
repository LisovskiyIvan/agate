const std = @import("std");
const sokol = @import("sokol");
const sapp = sokol.app;
const sg = sokol.gfx;
const slog = sokol.log;
const sglue = sokol.glue;
const z = @import("agate");

var gpa = std.heap.DebugAllocator(.{}){};
var scene: z.Scene = undefined;
var box: *z.Mesh = undefined;
var camera: z.ArcRotateCamera = undefined;

// CLI: --frames N quits after N rendered frames (0 = run until closed);
// --particles <cpu|gpu|compute> adds a demo particle system with that
// simulation mode (default: none). --frames is useful for headless smokes:
// `agate --frames 120` must complete without sokol validation errors.
var frame_limit: u32 = 0;
var frame_count: u32 = 0;
var particle_mode: ?z.SimulationMode = null;
var particles: *z.ParticleSystem = undefined;

fn parseArgs(args: std.process.Args) void {
    var it = std.process.Args.Iterator.init(args);
    _ = it.next(); // program name
    while (it.next()) |arg| {
        if (std.mem.eql(u8, arg, "--frames")) {
            if (it.next()) |n| {
                frame_limit = std.fmt.parseInt(u32, n, 10) catch 0;
            }
        } else if (std.mem.eql(u8, arg, "--particles")) {
            const mode = it.next() orelse break;
            if (std.mem.eql(u8, mode, "cpu")) {
                particle_mode = .cpu;
            } else if (std.mem.eql(u8, mode, "gpu")) {
                particle_mode = .gpu;
            } else if (std.mem.eql(u8, mode, "compute")) {
                particle_mode = .compute;
            }
        }
    }
}

export fn init() callconv(.c) void {
    sg.setup(.{
        .environment = sglue.environment(),
        .logger = .{ .func = slog.func },
        // Scene subsystems (forward + DS twins + shadow/skybox/particles +
        // postfx) create well over the 128-pipeline sokol default.
        .pipeline_pool_size = 256,
        .shader_pool_size = 64,
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
    scene.active_camera = .{ .arc_rotate = camera };

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

    if (particle_mode) |mode| {
        particles = scene.createParticleSystem("demo", 256) catch |err| {
            std.debug.panic("Failed to create particle system: {}", .{err});
        };
        particles.simulation_mode = mode;
        particles.emit_rate = 120.0;
        particles.is_emitting = true;
        particles.gravity = z.Vec3.new(0.0, -1.5, 0.0);
        particles.drag = 0.8;
        particles.emitter_box_max = z.Vec3.new(0.2, 0.2, 0.2);
        particles.direction_max = z.Vec3.new(0.5, 2.0, 0.5);
        particles.speed_max = 3.0;
        particles.size_start = 0.1;
        particles.size_end = 0.02;
    }
}

export fn frame() callconv(.c) void {
    const dt: f32 = @floatCast(sapp.frameDuration() * 60.0);

    // Вращаем куб
    box.rotation.x += 0.8 * dt;
    box.rotation.y += 1.6 * dt;

    if (particle_mode != null) {
        scene.updateParticles(@floatCast(sapp.frameDuration()));
    }

    scene.render();

    if (frame_limit != 0) {
        frame_count += 1;
        if (frame_count >= frame_limit) sapp.quit();
    }
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

pub fn main(minimal: std.process.Init.Minimal) void {
    parseArgs(minimal.args);
    sapp.run(.{
        .init_cb = init,
        .frame_cb = frame,
        .cleanup_cb = cleanup,
        .event_cb = event,
        .window_title = "agate (Babylon.js-style 3D in Zig)",
        .width = 800,
        .height = 600,
        .sample_count = 1,
        .logger = .{ .func = slog.func },
    });
}
