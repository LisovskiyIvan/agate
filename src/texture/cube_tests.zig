const std = @import("std");
const core = @import("core.zig");
const Texture = core.Texture;
const cube_mod = @import("cube.zig");
const CubeTexture = cube_mod.CubeTexture;

test "convertEquirectangularHDR maps gradient faces without NaN" {
    const allocator = std.testing.allocator;
    const pano_w: u32 = 4;
    const pano_h: u32 = 2;
    const face_size: u32 = 4;

    var pano: [4 * 2 * 4]f32 = undefined;
    for (0..pano_h) |y| {
        for (0..pano_w) |x| {
            const o = (y * pano_w + x) * 4;
            pano[o + 0] = @as(f32, @floatFromInt(x)) / @as(f32, @floatFromInt(pano_w - 1));
            pano[o + 1] = @as(f32, @floatFromInt(y)) / @as(f32, @floatFromInt(pano_h - 1));
            pano[o + 2] = 0.5;
            pano[o + 3] = 1.0;
        }
    }

    var cube = try CubeTexture.convertEquirectangularHDR(allocator, pano_w, pano_h, &pano, face_size);
    defer cube.deinit(allocator);

    try std.testing.expectEqual(face_size, cube.size);
    for (cube.faces) |maybe_face| {
        const face = maybe_face.?;
        try std.testing.expectEqual(@as(usize, face_size * face_size * 4), face.len);
        for (face) |bits| {
            const v = Texture.halfBitsToFloat(bits);
            try std.testing.expect(!std.math.isNan(v));
            try std.testing.expect(std.math.isFinite(v));
        }
    }
}

test "convertEquirectangularHDR validates dimensions and buffer size" {
    const allocator = std.testing.allocator;
    var pano: [2 * 1 * 4]f32 = [_]f32{0.0} ** (2 * 1 * 4);

    try std.testing.expectError(
        error.InvalidDimensions,
        CubeTexture.convertEquirectangularHDR(allocator, 0, 1, &pano, 2),
    );
    try std.testing.expectError(
        error.InvalidDimensions,
        CubeTexture.convertEquirectangularHDR(allocator, 2, 1, &pano, 0),
    );
    try std.testing.expectError(
        error.InvalidPanoramaSize,
        CubeTexture.convertEquirectangularHDR(allocator, 2, 1, pano[0..4], 2),
    );
}

test "fromEquirectangularFile validates input and missing files" {
    const allocator = std.testing.allocator;

    // face_size = 0 fails before any file IO.
    try std.testing.expectError(
        error.InvalidDimensions,
        CubeTexture.fromEquirectangularFile(allocator, "definitely/missing/file.png", 0),
    );
    // Missing file must fail cleanly (this also keeps the std.Io read path
    // analyzed, so future Zig API churn is caught by `zig build test`).
    try std.testing.expectError(
        error.FileNotFound,
        CubeTexture.fromEquirectangularFile(allocator, "definitely/missing/file.png", 4),
    );
}

// ---------------------------------------------------------------------------
// Golden test: HDR cube f16 mip chain (feeds textureLod in the IBL path)
// ---------------------------------------------------------------------------
test "buildRawFacesHdr builds the f16 mip chain with averaged values" {
    const allocator = std.testing.allocator;

    // One 2x2 face with R={1, 3, 5, 7}, G=B=0, A=1: level 1 must average
    // R to 4.0 exactly (f16-representable).
    var face: [2 * 2 * 4]u16 = undefined;
    const r_values = [_]f32{ 1, 3, 5, 7 };
    for (r_values, 0..) |rv, i| {
        face[i * 4 + 0] = Texture.floatToHalfBits(rv);
        face[i * 4 + 1] = 0;
        face[i * 4 + 2] = 0;
        face[i * 4 + 3] = Texture.floatToHalfBits(1.0);
    }
    var faces: [6][]const u16 = undefined;
    for (&faces) |*f| f.* = &face;

    var chain = try CubeTexture.buildRawFacesHdr(allocator, 2, faces);
    defer chain.deinit(allocator);

    try std.testing.expectEqual(@as(u32, 2), chain.num_levels);
    try std.testing.expectEqual(@as(u32, 2), chain.size);

    const l1 = chain.levels[1].?;
    try std.testing.expectEqual(@as(usize, 6 * 1 * 1 * 4), l1.len);
    // Face 0 averaged R = 4.0; every face inherits the same average.
    for (0..6) |f| {
        const o = f * 4;
        try std.testing.expectEqual(@as(f32, 4.0), Texture.halfBitsToFloat(l1[o + 0]));
        try std.testing.expectEqual(@as(f32, 0.0), Texture.halfBitsToFloat(l1[o + 1]));
        try std.testing.expectEqual(@as(f32, 1.0), Texture.halfBitsToFloat(l1[o + 3]));
    }
}

test "buildRawFacesHdr validates size and face buffers" {
    const allocator = std.testing.allocator;
    const face: [2 * 2 * 4]u16 = @splat(0);
    var faces: [6][]const u16 = @splat(&face);

    try std.testing.expectError(
        error.InvalidDimensions,
        CubeTexture.buildRawFacesHdr(allocator, 0, faces),
    );
    faces[5] = face[0..2];
    try std.testing.expectError(
        error.InvalidFaceBufferSize,
        CubeTexture.buildRawFacesHdr(allocator, 2, faces),
    );
}

test "initRawFaces validates dimensions without GPU upload" {
    const allocator = std.testing.allocator;
    const empty_faces: [6][]const u8 = @splat(&.{});
    try std.testing.expectError(
        error.InvalidDimensions,
        CubeTexture.initRawFaces(allocator, 0, empty_faces, false),
    );
    // Huge size fails on checked arithmetic before face-length checks.
    try std.testing.expectError(
        error.ImageTooLarge,
        CubeTexture.initRawFaces(allocator, 100000, empty_faces, false),
    );
    // Small size with short faces fails on length validation (no GPU).
    var one_pixel: [4]u8 = .{ 1, 2, 3, 255 };
    const bad_faces: [6][]const u8 = .{
        one_pixel[0..2], &one_pixel, &one_pixel, &one_pixel, &one_pixel, &one_pixel,
    };
    try std.testing.expectError(
        error.InvalidFaceBufferSize,
        CubeTexture.initRawFaces(allocator, 1, bad_faces, false),
    );
}

test "createProceduralSkybox rejects empty size" {
    const allocator = std.testing.allocator;
    try std.testing.expectError(
        error.InvalidDimensions,
        CubeTexture.createProceduralSkybox(allocator, .{ .size = 0 }),
    );
}

test "fromFiles reports missing faces without freeing uninitialized memory" {
    const allocator = std.testing.allocator;
    // First face missing: loaded == 0, the sized defer frees nothing.
    // C stbi allocations are not tracked by std.testing.allocator; this
    // only asserts the error contract, not leak accounting.
    const missing: [6][]const u8 = @splat("definitely/missing/face.png");
    try std.testing.expectError(
        error.ImageDecodeFailed,
        CubeTexture.fromFiles(allocator, missing),
    );
}

test "fromEquirectangular rejects huge faces before allocation" {
    const allocator = std.testing.allocator;
    // 1x1 RGBA PNG through the in-file fixture builder (no GPU involved).
    const scanlines = [_]u8{ 0, 10, 20, 30, 255 };
    const png = try core.TestPng.build(allocator, 1, 1, 8, 6, null, &scanlines);
    defer allocator.free(png);
    try std.testing.expectError(
        error.ImageTooLarge,
        CubeTexture.fromEquirectangular(allocator, png, 100000),
    );
}
