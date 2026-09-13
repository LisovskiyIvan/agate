//! Fuzz/robustness target for the OBJ parser (loader/obj.zig).
//!
//! Invariant: for ANY input the parser returns an error or a valid, fully
//! owned ObjData — no panic, no UB, no leaks (std.testing.allocator backs
//! every iteration and reports leaks as test failures).
//!
//! `zig build test` smoke-runs the corpus below once (valid seed +
//! truncations + single-byte flips at structural offsets + garbage).
//! `zig build test --fuzz[=limit]` runs the same target as a coverage-guided
//! campaign; see build.zig for the runner notes.

const std = @import("std");
const obj = @import("obj.zig");
const fzg = @import("../testing.zig");

/// Valid seed exercising v/vn/vt, the `v//n` form and negative `v/vt/n` faces.
const seed =
    \\v 0 0 0
    \\v 1 0 0
    \\v 0 1 0
    \\vn 0 0 1
    \\vt 0 0
    \\vt 1 1
    \\f 1//1 2//1 3//1
    \\f -1/2/-1 -2/1/-1 -3/2/-1
;

const corpus = fzg.join(
    &[_][]const u8{seed},
    fzg.join(
        // Truncations cut through every structural region: header-ish lines,
        // the vertex table and mid-face.
        fzg.truncations(seed, &.{ 1, 2, 4, 8, 16, 24, 32, 48, seed.len - 1 }),
        fzg.join(
            // Flips hit first bytes, numerals and face separators.
            fzg.flips(seed, &.{ 0, 1, 4, 12, 20, 30, 44 }),
            // Handpicked garbage: empty, comment-only, out-of-range indices,
            // non-finite floats (must parse to NaN/inf without crashing).
            &[_][]const u8{
                "",
                "#comment only\n",
                "f 9 9 9\n",
                "v nan nan inf\nf 1 2 3",
                "v 1e999 -1e999 0\nf 1 1 1\n",
            },
        ),
    ),
);

fn testOne(_: void, smith: *std.testing.Smith) anyerror!void {
    @disableInstrumentation();
    var generated: [8192]u8 = undefined;
    const input = fzg.fuzzInput(smith, &generated);

    var data = obj.parse(std.testing.allocator, input) catch return;
    data.deinit(std.testing.allocator);
}

test "fuzz: obj.parse survives arbitrary bytes" {
    try std.testing.fuzz({}, testOne, .{ .corpus = corpus });
}
