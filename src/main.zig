const std = @import("std");
const sokol = @import("sokol");

const sapp = sokol.app;
const sg = sokol.gfx;
const slog = sokol.log;
const sglue = sokol.glue;

const state = struct {
    var pass_action: sg.PassAction = undefined;
    var color_index: usize = 0;
};

// 4 цвета: клик мыши или пробел переключает фон
const colors = [_]sg.Color{
    .{ .r = 0.25, .g = 0.5, .b = 0.75, .a = 1.0 }, // сине-серый
    .{ .r = 0.85, .g = 0.3, .b = 0.3, .a = 1.0 }, // красный
    .{ .r = 0.3, .g = 0.75, .b = 0.4, .a = 1.0 }, // зелёный
    .{ .r = 0.9, .g = 0.75, .b = 0.25, .a = 1.0 }, // жёлтый
};

fn applyColor() void {
    state.pass_action.colors[0] = .{
        .load_action = .CLEAR,
        .clear_value = colors[state.color_index % colors.len],
    };
}

export fn init() callconv(.c) void {
    sg.setup(.{
        .environment = sglue.environment(),
        .logger = .{ .func = slog.func },
    });
    state.pass_action = .{};
    applyColor();
    std.debug.print("Backend: {}\n", .{sg.queryBackend()});
}

export fn frame() callconv(.c) void {
    sg.beginPass(.{ .action = state.pass_action, .swapchain = sglue.swapchain() });
    sg.endPass();
    sg.commit();
}
export fn event(ev: [*c]const sapp.Event) callconv(.c) void {
    switch (ev.*.type) {
        .MOUSE_DOWN => {
            state.color_index += 1;
            applyColor();
        },
        .KEY_DOWN => switch (ev.*.key_code) {
            .SPACE => {
                state.color_index += 1;
                applyColor();
            },
            .ESCAPE => sapp.quit(),
            else => {},
        },
        else => {},
    }
}
export fn cleanup() callconv(.c) void {
    sg.shutdown();
}

pub fn main() void {
    sapp.run(.{
        .init_cb = init,
        .frame_cb = frame,
        .cleanup_cb = cleanup,
        .event_cb = event,
        .window_title = "sokol zig: basic window",
        .width = 800,
        .height = 600,
        .logger = .{ .func = slog.func },
    });
}
