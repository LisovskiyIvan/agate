//! Fuzz/robustness target for the PLY parser (loader/ply.zig), covering the
//! ASCII format and both binary endiannesses. Corpus and invariants follow
//! loader/obj_fuzz.zig; run notes in build.zig.

const std = @import("std");
const ply = @import("ply.zig");
const fzg = @import("../testing.zig");

const ascii_seed =
    \\ply
    \\format ascii 1.0
    \\element vertex 3
    \\property float x
    \\property float y
    \\property float z
    \\element face 1
    \\property list uchar int vertex_indices
    \\end_header
    \\0 0 0
    \\1 0 0
    \\0 1 0
    \\3 0 1 2
;

const bin_le_seed: []const u8 = blk: {
    const header =
        \\ply
        \\format binary_little_endian 1.0
        \\element vertex 3
        \\property float x
        \\property float y
        \\property float z
        \\element face 1
        \\property list uchar int vertex_indices
        \\end_header
    ;
    // 3 vertices * 3 f32 + (uchar count + 3 i32).
    var payload: [3 * 12 + 13]u8 = @splat(0);
    for (.{ 1.0, 0.0, 0.0, 0.0, 1.0, 0.0 }, 0..) |v, i| {
        std.mem.writeInt(u32, payload[i * 4 ..][0..4], @bitCast(@as(f32, v)), .little);
    }
    payload[36] = 3; // face index count
    for (.{ 0, 1, 2 }, 0..) |idx, i| {
        std.mem.writeInt(i32, payload[37 + i * 4 ..][0..4], idx, .little);
    }
    break :blk header ++ &payload;
};

const corpus = fzg.join(
    &[_][]const u8{ ascii_seed, bin_le_seed },
    fzg.join(
        // Cuts through magic, format line, element/property table and data.
        fzg.truncations(ascii_seed, &.{ 1, 3, 8, 20, 40, 60, 90, ascii_seed.len - 1 }),
        fzg.join(
            fzg.flips(ascii_seed, &.{ 0, 4, 10, 25, 45, 70 }),
            fzg.join(
                fzg.flips(bin_le_seed, &.{ 0, 6, 40, 90, 120 }),
                &[_][]const u8{
                    "",
                    "ply\n",
                    "ply\nformat binary_big_endian 1.0\nend_header\n",
                    "ply\nformat ascii 1.0\nelement vertex 99999999\nend_header\n",
                    "ply\nformat ascii 1.0\nelement vertex 3\nproperty int x\nend_header\na b c\n",
                    "ply\nformat ascii 1.0\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n5 0 1 2 3 4 5\n",
                },
            ),
        ),
    ),
);

fn testOne(_: void, smith: *std.testing.Smith) anyerror!void {
    @disableInstrumentation();
    var generated: [8192]u8 = undefined;
    const input = fzg.fuzzInput(smith, &generated);

    var data = ply.parse(std.testing.allocator, input) catch return;
    data.deinit(std.testing.allocator);
}

test "fuzz: ply.parse survives arbitrary bytes" {
    try std.testing.fuzz({}, testOne, .{ .corpus = corpus });
}

// ---------------------------------------------------------------------------
// Allocation-failure: the header table, vertex/face builders and the output
// lists must survive any single allocation failing.
// ---------------------------------------------------------------------------

fn parseSeedPly(alloc: std.mem.Allocator) !void {
    var data = try ply.parse(alloc, ascii_seed);
    data.deinit(alloc);
}

test "alloc-failure: ply.parse frees everything on OOM" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, parseSeedPly, .{});
}
