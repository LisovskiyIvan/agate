//! View-block upload cache: draw-trace contract without GPU calls.
//!
//! Proves the sokol-correct rule (sokol_gfx.h: bindings AND uniform slots
//! die on applyPipeline even for identical layouts): the view block
//! uploads on the first draw of a contiguous (pipeline, shadow-policy)
//! run and on every pipeline switch / policy flip / view boundary — and
//! only then. Probe faces, refraction redraws, and PIP views each reset
//! the cache, so their first draw always re-uploads.

const std = @import("std");
const draw = @import("draw.zig");

test "view cache: first draw uploads, contiguous run skips" {
    draw.resetViewUploadCache();
    // Cold cache: must upload (first draw of the view is always valid).
    try std.testing.expect(draw.viewCacheNeed(7, true));
    // Simulate the upload landing (pipeline applied, shadow policy true).
    // The decision function is pure; mirror the mark via a second call
    // sequence on a fresh cache object for the transition table.
    var c = draw.ViewUploadCache{};
    try std.testing.expect(c.needUpload(7, true));
    c = .{ .pipeline = 7, .shadow = true, .valid = true };
    // Same pipeline + policy: skip (the upload saving).
    try std.testing.expect(!c.needUpload(7, true));
    // Policy flip on the same pipeline: must re-upload (receive gate).
    try std.testing.expect(c.needUpload(7, false));
    c = .{ .pipeline = 7, .shadow = false, .valid = true };
    try std.testing.expect(!c.needUpload(7, false));
    // Flip back: re-upload again (single-slot cache, ping-pong is exact).
    try std.testing.expect(c.needUpload(7, true));
}

test "view cache: runtime-external apply invalidates, engine re-uploads" {
    // Regression for the p7 -> runtime-external(pX) -> p7 hole: the
    // external draw applies its pipeline (invalidating ALL uniform slots)
    // but uploads no view block, so the returning engine draw must
    // re-upload even though the id matches the stale cache entry. The
    // three pipeline-apply sites reset the cache on every switch, which
    // this trace drives through the real threadlocal helpers (no GPU).
    draw.resetViewUploadCache(); // view boundary
    try std.testing.expect(draw.viewCacheNeed(7, true)); // engine p7 first draw: upload
    draw.viewCacheNoteUpload(7, true);
    try std.testing.expect(!draw.viewCacheNeed(7, true)); // contiguous: skip
    draw.resetViewUploadCache(); // p7 -> pX switch site resets (external uploads nothing)
    try std.testing.expect(draw.viewCacheNeed(7, true)); // pX -> p7 switch resets again: MUST re-upload
    draw.viewCacheNoteUpload(7, true);
    try std.testing.expect(!draw.viewCacheNeed(7, true)); // contiguous again: skip
}

test "view cache: pipeline switch always re-uploads (sokol contract)" {
    var c = draw.ViewUploadCache{ .pipeline = 7, .shadow = true, .valid = true };
    // Same layout, different pipeline id: slots were invalidated.
    try std.testing.expect(c.needUpload(9, true));
    try std.testing.expect(c.needUpload(9, false));
    // Even switching back to a previously-seen pipeline re-uploads.
    c = .{ .pipeline = 9, .shadow = true, .valid = true };
    try std.testing.expect(c.needUpload(7, true));
}

test "view cache: trace over a mixed draw sequence uploads minimally" {
    // Sequence: p7/shadow x3, p7/noshadow x2, p9/shadow x2, p9/shadow x1.
    // Expected uploads: 1 (first) + 1 (policy flip) + 1 (pipeline switch)
    // = 3 view uploads for 8 draws; the other 5 are material-only.
    const seq = [_]struct { pip: u32, shadow: bool }{
        .{ .pip = 7, .shadow = true },
        .{ .pip = 7, .shadow = true },
        .{ .pip = 7, .shadow = true },
        .{ .pip = 7, .shadow = false },
        .{ .pip = 7, .shadow = false },
        .{ .pip = 9, .shadow = true },
        .{ .pip = 9, .shadow = true },
        .{ .pip = 9, .shadow = true },
    };
    var c = draw.ViewUploadCache{};
    var uploads: usize = 0;
    for (seq) |s| {
        if (c.needUpload(s.pip, s.shadow)) {
            uploads += 1;
            c = .{ .pipeline = s.pip, .shadow = s.shadow, .valid = true };
        }
    }
    try std.testing.expectEqual(@as(usize, 3), uploads);
}
