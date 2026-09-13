//! Fuzz/robustness target for the STL parser (loader/stl.zig), covering both
//! the ASCII tokenizer and the binary facet path. Corpus and invariants follow
//! loader/obj_fuzz.zig; run notes in build.zig.

const std = @import("std");
const stl = @import("stl.zig");
const fzg = @import("../testing.zig");

const ascii_seed =
    \\solid agate
    \\facet normal 0 0 1
    \\  outer loop
    \\    vertex 0 0 0
    \\    vertex 1 0 0
    \\    vertex 0 1 0
    \\  endloop
    \\endfacet
    \\endsolid agate
;

/// Minimal valid binary STL (80-byte header, count = 1, one facet whose
/// triangle is degenerate-but-wellformed). Built at comptime so the whole
/// corpus stays a comptime value.
const bin_seed: [84 + 50]u8 = blk: {
    var buf: [84 + 50]u8 = @splat(0);
    @memcpy(buf[0..5], "agate");
    std.mem.writeInt(u32, buf[80..84], 1, .little);
    // facet: normal +Z, vertices (0,0,0), (1,0,0), (0,1,0), attr 0.
    const one: [4]u8 = @bitCast(@as(f32, 1.0));
    @memcpy(buf[84 + 8 ..][0..4], &one); // normal.z
    @memcpy(buf[84 + 24 ..][0..4], &one); // v1.x
    @memcpy(buf[84 + 40 ..][0..4], &one); // v2.y
    break :blk buf;
};

const corpus = fzg.join(
    &[_][]const u8{ ascii_seed, &bin_seed },
    fzg.join(
        fzg.truncations(ascii_seed, &.{ 1, 5, 6, 12, 24, 48, 64, ascii_seed.len - 1 }),
        fzg.join(
            fzg.flips(ascii_seed, &.{ 0, 4, 10, 20, 40, 60 }),
            fzg.join(
                // Binary-specific: flip magic/header bytes (0..80), the count
                // field (80..84, makes size and count disagree) and facet
                // payload; plus an all-zeros "valid" binary facet.
                fzg.flips(&bin_seed, &.{ 0, 40, 80, 81, 84, 100 }),
                &[_][]const u8{
                    "",
                    "solid no facets\n",
                    "solid bad facet\nfacet normal 0 0\n",
                    "solid trunc\nfacet normal 0 0 1\n  outer loop\n    vertex 0 0 0\n",
                },
            ),
        ),
    ),
);

fn testOne(_: void, smith: *std.testing.Smith) anyerror!void {
    @disableInstrumentation();
    var generated: [8192]u8 = undefined;
    const input = fzg.fuzzInput(smith, &generated);

    var data = stl.parse(std.testing.allocator, input) catch return;
    data.deinit(std.testing.allocator);
}

test "fuzz: stl.parse survives arbitrary bytes" {
    try std.testing.fuzz({}, testOne, .{ .corpus = corpus });
}
