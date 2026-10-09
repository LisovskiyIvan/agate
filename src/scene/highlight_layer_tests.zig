const std = @import("std");
const Mesh = @import("../mesh.zig").Mesh;
const highlight = @import("highlight_layer.zig");
const HighlightLayer = highlight.HighlightLayer;
const HighlightOptions = highlight.HighlightOptions;
const max_highlights = highlight.max_highlights;
const validateHighlightOptions = highlight.validateHighlightOptions;

test "highlight add/remove/clear/count with cap-8 error" {
    var layer = HighlightLayer{};
    var meshes: [9]Mesh = undefined;
    for (&meshes, 0..) |*m, i| {
        m.* = Mesh{ .name = "hl", .vertex_buffer = .{}, .index_buffer = .{}, .index_count = 3 };
        _ = i;
    }

    for (0..max_highlights) |i| {
        const id = try layer.add(&meshes[i], .{});
        try std.testing.expectEqual(i, id);
    }
    try std.testing.expectEqual(max_highlights, layer.highlightCount());
    try std.testing.expectError(error.TooManyHighlights, layer.add(&meshes[8], .{}));

    // Order-preserving remove: higher entries shift down.
    layer.remove(0);
    try std.testing.expectEqual(max_highlights - 1, layer.highlightCount());
    try std.testing.expectEqual(&meshes[1], layer.get(0).?.mesh);

    // Out-of-range remove is a no-op (probe/gui3d contract).
    layer.remove(99);
    try std.testing.expectEqual(max_highlights - 1, layer.highlightCount());

    // OOB get is null.
    try std.testing.expect(layer.get(max_highlights) == null);

    layer.clear();
    try std.testing.expectEqual(@as(usize, 0), layer.highlightCount());
    // The layer is reusable after clear.
    const id = try layer.add(&meshes[0], .{});
    try std.testing.expectEqual(@as(usize, 0), id);
}

test "highlight options validation rejects non-finite and out-of-range" {
    // Defaults validate.
    try validateHighlightOptions(.{});

    var bad = HighlightOptions{ .color = .{ std.math.nan(f32), 1.0, 1.0, 1.0 } };
    try std.testing.expectError(error.InvalidHighlightOptions, validateHighlightOptions(bad));
    bad = HighlightOptions{ .color = .{ 1.0, std.math.inf(f32), 1.0, 1.0 } };
    try std.testing.expectError(error.InvalidHighlightOptions, validateHighlightOptions(bad));
    bad = HighlightOptions{ .color = .{ 1.0, 1.0, 1.0, 1.0 }, .blur = std.math.nan(f32) };
    try std.testing.expectError(error.InvalidHighlightOptions, validateHighlightOptions(bad));
    bad = HighlightOptions{ .color = .{ 1.0, 1.0, 1.0, 1.0 }, .intensity = std.math.inf(f32) };
    try std.testing.expectError(error.InvalidHighlightOptions, validateHighlightOptions(bad));

    // Finite out-of-range values are hard errors, never silent clamps
    // (glow validateGlow precedent).
    bad = HighlightOptions{ .color = .{ 1.2, 1.0, 1.0, 1.0 } };
    try std.testing.expectError(error.InvalidHighlightOptions, validateHighlightOptions(bad));
    bad = HighlightOptions{ .color = .{ 1.0, 1.0, 1.0, -0.1 } };
    try std.testing.expectError(error.InvalidHighlightOptions, validateHighlightOptions(bad));
    bad = HighlightOptions{ .blur = -1.0 };
    try std.testing.expectError(error.InvalidHighlightOptions, validateHighlightOptions(bad));
    bad = HighlightOptions{ .intensity = -0.5 };
    try std.testing.expectError(error.InvalidHighlightOptions, validateHighlightOptions(bad));

    // Boundary values are legal (zero blur/intensity/alpha included).
    try validateHighlightOptions(.{ .color = .{ 0.0, 0.0, 0.0, 0.0 }, .blur = 0.0, .intensity = 0.0 });

    // add() surfaces the same error before touching the cap.
    var layer = HighlightLayer{};
    var mesh = Mesh{ .name = "hl_bad", .vertex_buffer = .{}, .index_buffer = .{}, .index_count = 3 };
    try std.testing.expectError(error.InvalidHighlightOptions, layer.add(&mesh, .{ .blur = -2.0 }));
    try std.testing.expectEqual(@as(usize, 0), layer.highlightCount());
}

test "highlight removeForMesh drops dead referents and keeps the rest" {
    var layer = HighlightLayer{};
    var m0 = Mesh{ .name = "hl0", .vertex_buffer = .{}, .index_buffer = .{}, .index_count = 3 };
    var m1 = Mesh{ .name = "hl1", .vertex_buffer = .{}, .index_buffer = .{}, .index_count = 3 };

    _ = try layer.add(&m0, .{});
    _ = try layer.add(&m1, .{ .color = .{ 1.0, 0.0, 0.0, 1.0 } });
    _ = try layer.add(&m0, .{ .intensity = 0.1 });

    // The destroyMesh path drops both m0 entries; m1 survives with its
    // options intact.
    layer.removeForMesh(&m0);
    try std.testing.expectEqual(@as(usize, 1), layer.highlightCount());
    try std.testing.expectEqual(&m1, layer.get(0).?.mesh);
    try std.testing.expectEqual([4]f32{ 1.0, 0.0, 0.0, 1.0 }, layer.get(0).?.options.color);

    // Unknown mesh is a no-op.
    layer.removeForMesh(&m0);
    try std.testing.expectEqual(@as(usize, 1), layer.highlightCount());
}

test "highlight add freezes uid and options" {
    var layer = HighlightLayer{};
    var mesh = Mesh{ .name = "hl_uid", .vertex_buffer = .{}, .index_buffer = .{}, .index_count = 3 };
    const opts = HighlightOptions{ .color = .{ 0.2, 0.4, 0.6, 1.0 }, .blur = 6.0, .intensity = 0.8 };
    const id = try layer.add(&mesh, opts);
    const entry = layer.get(id).?;
    try std.testing.expect(entry.uid != 0);
    try std.testing.expectEqual(mesh.uid, entry.uid);
    try std.testing.expectEqual(opts.color, entry.options.color);
    try std.testing.expectApproxEqAbs(opts.blur, entry.options.blur, 1e-6);
    try std.testing.expectApproxEqAbs(opts.intensity, entry.options.intensity, 1e-6);
}
