const std = @import("std");
const math = @import("math");
const Mat4 = math.Mat4;
const DebugLine = @import("../physics.zig").DebugLine;
const upload_meter = @import("../gpu_upload_meter.zig");
const debug_pass = @import("debug_pass.zig");
const DebugPass = debug_pass.DebugPass;
const Vertex = debug_pass.Vertex;
const packDebugLine = debug_pass.packDebugLine;
const verticesForLineCount = debug_pass.verticesForLineCount;
const grownCapacity = debug_pass.grownCapacity;
const max_capacity_lines = debug_pass.max_capacity_lines;

test "debug Vertex layout is tightly packed pos + rgba" {
    try std.testing.expectEqual(@as(usize, 28), @sizeOf(Vertex));
    try std.testing.expectEqual(@as(usize, 0), @offsetOf(Vertex, "position"));
    try std.testing.expectEqual(@as(usize, 12), @offsetOf(Vertex, "color"));
}

test "packDebugLine emits two verts with opaque color" {
    const line = DebugLine{
        .a = .{ .x = 1.0, .y = 2.0, .z = 3.0 },
        .b = .{ .x = 4.0, .y = 5.0, .z = 6.0 },
        .color = .{ 0.1, 0.9, 0.3 },
    };
    var pair: [2]Vertex = undefined;
    packDebugLine(line, &pair);
    try std.testing.expectEqual([3]f32{ 1.0, 2.0, 3.0 }, pair[0].position);
    try std.testing.expectEqual([3]f32{ 4.0, 5.0, 6.0 }, pair[1].position);
    try std.testing.expectEqual([4]f32{ 0.1, 0.9, 0.3, 1.0 }, pair[0].color);
    try std.testing.expectEqual([4]f32{ 0.1, 0.9, 0.3, 1.0 }, pair[1].color);
    try std.testing.expectEqual(@as(usize, 2), verticesForLineCount(1));
    try std.testing.expectEqual(@as(usize, 0), verticesForLineCount(0));
}

test "packDebugLine sanitizes non-finite and out-of-range inputs" {
    const nan = std.math.nan(f32);
    const inf = std.math.inf(f32);
    const line = DebugLine{
        .a = .{ .x = nan, .y = 1.0, .z = inf },
        .b = .{ .x = 0.0, .y = -inf, .z = 2.0 },
        .color = .{ 2.0, -1.0, nan },
    };
    var pair: [2]Vertex = undefined;
    packDebugLine(line, &pair);
    try std.testing.expectEqual([3]f32{ 0.0, 1.0, 0.0 }, pair[0].position);
    try std.testing.expectEqual([3]f32{ 0.0, 0.0, 2.0 }, pair[1].position);
    try std.testing.expectEqual([4]f32{ 1.0, 0.0, 0.0, 1.0 }, pair[0].color);
}

test "grownCapacity doubles and clamps at max" {
    try std.testing.expectEqual(@as(usize, 4096), grownCapacity(4096, 100));
    try std.testing.expectEqual(@as(usize, 4096), grownCapacity(4096, 4096));
    try std.testing.expectEqual(@as(usize, 8192), grownCapacity(4096, 4097));
    try std.testing.expectEqual(@as(usize, 16384), grownCapacity(4096, 16383));
    try std.testing.expectEqual(max_capacity_lines, grownCapacity(4096, max_capacity_lines * 4));
}

test "upload stages once headless, drawPrepared is a safe no-op" {
    const t = std.testing;
    _ = upload_meter.takeAndReset();
    var pass = DebugPass{
        .allocator = t.allocator,
        .pipeline = .{ .id = 1 },
        .vertex_buffer = .{ .id = 2 },
        .capacity_lines = 1,
    };
    defer pass.staging.deinit(t.allocator);
    try pass.staging.ensureTotalCapacity(t.allocator, verticesForLineCount(4));

    const lines = [_]DebugLine{
        .{ .a = .{ .x = 0, .y = 0, .z = 0 }, .b = .{ .x = 1, .y = 0, .z = 0 }, .color = .{ 1, 0, 0 } },
        .{ .a = .{ .x = 0, .y = 1, .z = 0 }, .b = .{ .x = 0, .y = 2, .z = 0 }, .color = .{ 0, 1, 0 } },
    };
    try t.expect(pass.upload(&lines));
    try t.expectEqual(@as(usize, 2), pass.capacity_lines);
    try t.expectEqual(@as(usize, 4), pass.prepared_verts);
    try t.expectEqual(@as(usize, 4), pass.staging.items.len);
    try t.expectEqual(@as(u64, 0), upload_meter.peek());

    const moved = [_]DebugLine{
        .{ .a = .{ .x = 9, .y = 0, .z = 0 }, .b = .{ .x = 10, .y = 0, .z = 0 }, .color = .{ 0, 0, 1 } },
    };
    try t.expect(pass.upload(&moved));
    try t.expectEqual(@as(usize, 2), pass.prepared_verts);
    try t.expectEqual(@as(f32, 9.0), pass.staging.items[0].position[0]);

    try t.expect(!pass.upload(&.{}));
    try t.expectEqual(@as(usize, 0), pass.prepared_verts);
    try t.expect(!pass.drawPrepared(Mat4.identity));
    try t.expectEqual(@as(u64, 0), upload_meter.peek());

    var dead = DebugPass{ .allocator = t.allocator };
    defer dead.staging.deinit(t.allocator);
    try t.expect(!dead.upload(&lines));
    try t.expectEqual(@as(usize, 0), dead.prepared_verts);
}

test "upload clamps at max capacity, oldest lines win" {
    const t = std.testing;
    var pass = DebugPass{
        .allocator = t.allocator,
        .pipeline = .{ .id = 1 },
        .vertex_buffer = .{ .id = 2 },
        .capacity_lines = max_capacity_lines,
    };
    defer pass.staging.deinit(t.allocator);
    try pass.staging.ensureTotalCapacity(t.allocator, verticesForLineCount(max_capacity_lines));

    const n = max_capacity_lines + 2;
    const lines = try t.allocator.alloc(DebugLine, n);
    defer t.allocator.free(lines);
    for (lines, 0..) |*ln, i| {
        const x: f32 = @floatFromInt(i);
        ln.* = .{
            .a = .{ .x = x, .y = 0, .z = 0 },
            .b = .{ .x = x + 0.5, .y = 0, .z = 0 },
            .color = .{ 1, 0, 0 },
        };
    }
    try t.expect(pass.upload(lines));
    try t.expectEqual(verticesForLineCount(max_capacity_lines), pass.prepared_verts);
    try t.expectEqual(@as(f32, 0.0), pass.staging.items[0].position[0]);
    const last: f32 = @floatFromInt(max_capacity_lines - 1);
    try t.expectEqual(last, pass.staging.items[pass.staging.items.len - 2].position[0]);
    try t.expectEqual(last + 0.5, pass.staging.items[pass.staging.items.len - 1].position[0]);
}
