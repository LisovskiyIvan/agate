//! Fuzz/robustness target for the CSS theme parser (ui/css_parser.zig):
//! stylesheet parsing plus the leaf value parsers (colors, lengths,
//! durations, easing names).
//!
//! Invariant: any input yields an error or a theme with owned slices — no
//! panic, no UB, no leaks (each iteration parses through an arena backed by
//! std.testing.allocator, so a missed free surfaces at arena.deinit).
//! Run notes in build.zig.

const std = @import("std");
const css = @import("css_parser.zig");
const fzg = @import("../testing.zig");

/// One of every selector shape the subset supports (same fixture as the unit
/// tests: theme block, type selector, state class, hover/disabled deltas).
const seed =
    \\/* engine theme */
    \\theme { accent: #7cb3ff; transition_duration: 120ms; }
    \\button { background: #141925d9; corner_radius: 4px; }
    \\button:hover { background: rgb(36, 48, 66); border_color: #a6d4ff; }
    \\.danger { background: #b3261e; transition_easing: ease-in-cubic; }
    \\.danger:disabled { opacity: 0.4; }
;

const corpus = fzg.join(
    &[_][]const u8{seed},
    fzg.join(
        fzg.truncations(seed, &.{ 1, 2, 4, 8, 16, 32, 64, 120, seed.len - 1 }),
        fzg.join(
            fzg.flips(seed, &.{ 0, 1, 8, 20, 35, 50, 80, 110 }),
            &[_][]const u8{
                "",
                "{",
                "}",
                "theme { accent: ",
                ".x{background:",
                ".x{background: rgb(1,2,3",
                "theme{transition_duration:99999999999999999999s}",
                "a{corner_radius:-1e30px}b{opacity:nan}",
            },
        ),
    ),
);

fn testOne(_: void, smith: *std.testing.Smith) anyerror!void {
    @disableInstrumentation();
    var generated: [8192]u8 = undefined;
    const input = fzg.fuzzInput(smith, &generated);

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    _ = css.parseCss(arena.allocator(), input) catch {};
}

test "fuzz: parseCss survives arbitrary bytes" {
    try std.testing.fuzz({}, testOne, .{ .corpus = corpus });
}

// Value parsers: no allocation, so no arena — just "returns a value or null".

fn testValueOne(_: void, smith: *std.testing.Smith) anyerror!void {
    @disableInstrumentation();
    var buf: [256]u8 = undefined;
    const input = fzg.fuzzInput(smith, &buf);

    _ = css.parseColor(input);
    _ = css.parseLength(input);
    _ = css.parseDurationMs(input);
    _ = css.parseEasing(input);
}

test "fuzz: css value parsers survive arbitrary bytes" {
    try std.testing.fuzz({}, testValueOne, .{ .corpus = &[_][]const u8{
        "#7cb3ff",
        "rgb(1, 2, 3)",
        "4px",
        "120ms",
        "ease-in-cubic",
        "#",
        "rgb(",
        "9999999999999999999999px",
    } });
}
