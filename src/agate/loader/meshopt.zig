//! EXT_meshopt_compression (glTF/GLB buffer views compressed with
//! meshoptimizer). Support is transparent: SceneLoader.appendGlb
//! accept such files without new parameters. cgltf parses the extension
//! metadata, and after cgltf_load_buffers the C glue
//! agate_cgltf_decode_meshopt (c_impl.c, vendored decoder in c/meshopt)
//! decompresses every compressed buffer view in place; everything downstream
//! (materials, meshes, skins, animations) keeps reading accessors unchanged.
//!
//! Supported (full EXT_meshopt_compression surface):
//!   - MODE_ATTRIBUTES (meshopt_decodeVertexBuffer)
//!   - MODE_TRIANGLES  (meshopt_decodeIndexBuffer)
//!   - MODE_INDICES    (meshopt_decodeIndexSequence)
//!   - attribute filters octahedral / quaternion / exponential / color
//!   - the legacy KHR_meshopt_compression spelling (cgltf maps both to the
//!     same struct, so both decode through the same path)
//!
//! Limitations:
//!   - The uncompressed fallback buffer gltfpack declares (`fallback: true`)
//!     is never read: compressed bytes are authoritative and the fallback
//!     data may legitimately be absent from the file.
//!   - Decompression is eager and single-threaded at load time, for the whole
//!     cgltf_data (no lazy per-primitive decode).
//!   - The meshopt stream carries no payload checksum: structurally broken
//!     blobs (bad header/framing) surface as a clean decode error, while a
//!     flipped payload byte may decode into silent garbage — the decoder is
//!     bounds-safe but does not verify data integrity.
//!   - Draco (KHR_draco_mesh_compression) and KTX2/Basisu textures are out of
//!     scope and still unsupported.
//!
//! Fixtures: meshopt_cube.glb is a 24-vertex cube packed with gltfpack 1.2
//! (`-c`, deterministic), with an uncompressed sibling meshopt_cube_plain.glb
//! produced by the same gltfpack pipeline minus compression. Both are
//! self-contained GLBs (embedded buffers only), so the tests parse them from
//! memory.

const std = @import("std");

const c = @import("../c.zig").c;

const compressed_glb = @embedFile("fixtures/meshopt_cube.glb");
const plain_glb = @embedFile("fixtures/meshopt_cube_plain.glb");

/// Parses an in-memory GLB and loads its embedded buffers. GLB buffers live
/// in the BIN chunk, so no file access happens.
fn parseGlbFromMemory(bytes: []const u8) !*c.cgltf_data {
    var options = std.mem.zeroes(c.cgltf_options);
    var data: ?*c.cgltf_data = null;

    const parse_res = c.cgltf_parse(&options, bytes.ptr, bytes.len, &data);
    if (parse_res != c.cgltf_result_success or data == null) {
        return error.GltfParseFailed;
    }
    const loaded = data orelse return error.GltfParseFailed;
    const load_res = c.cgltf_load_buffers(&options, loaded, null);
    if (load_res != c.cgltf_result_success) {
        c.cgltf_free(loaded);
        return error.GltfLoadBuffersFailed;
    }
    return loaded;
}

fn hasMeshoptViews(data: *const c.cgltf_data) bool {
    for (data.buffer_views[0..data.buffer_views_count]) |*view| {
        if (view.has_meshopt_compression != 0) return true;
    }
    return false;
}

test "meshopt: fixture declares the extension and compressed views" {
    const data = try parseGlbFromMemory(compressed_glb);
    defer c.cgltf_free(data);

    var found_ext = false;
    for (data.*.extensions_used[0..data.*.extensions_used_count]) |ext| {
        if (std.mem.order(u8, std.mem.sliceTo(ext, 0), "EXT_meshopt_compression") == .eq) {
            found_ext = true;
        }
    }
    try std.testing.expect(found_ext);
    try std.testing.expect(hasMeshoptViews(data));
    // gltfpack emits MODE_ATTRIBUTES for vertex streams and MODE_TRIANGLES
    // for indices; both must be present so the fixture covers both paths.
    var modes: u32 = 0;
    for (data.buffer_views[0..data.buffer_views_count]) |*view| {
        if (view.has_meshopt_compression == 0) continue;
        switch (view.meshopt_compression.mode) {
            c.cgltf_meshopt_compression_mode_attributes => modes |= 1,
            c.cgltf_meshopt_compression_mode_triangles => modes |= 2,
            else => {},
        }
    }
    try std.testing.expectEqual(@as(u32, 3), modes);
}

test "meshopt: compressed fixture decodes to the uncompressed baseline" {
    const compressed = try parseGlbFromMemory(compressed_glb);
    defer c.cgltf_free(compressed);
    const plain = try parseGlbFromMemory(plain_glb);
    defer c.cgltf_free(plain);

    const decode_res = c.agate_cgltf_decode_meshopt(
        &std.mem.zeroes(c.cgltf_options),
        compressed,
    );
    try std.testing.expect(decode_res == c.cgltf_result_success);

    // gltfpack -c only changes buffer serialization: after decoding, every
    // compressed view must hold exactly the bytes of the corresponding plain
    // view (same pack pipeline, so vertex/index data is bit-identical).
    try std.testing.expectEqual(plain.buffer_views_count, compressed.buffer_views_count);
    for (plain.buffer_views[0..plain.buffer_views_count], compressed.buffer_views[0..compressed.buffer_views_count]) |*pv, *cv| {
        try std.testing.expectEqual(pv.size, cv.meshopt_compression.count * cv.meshopt_compression.stride);
        const expected: [*]const u8 = @ptrCast(c.cgltf_buffer_view_data(pv));
        const actual: [*]const u8 = @ptrCast(c.cgltf_buffer_view_data(cv));
        try std.testing.expect(std.mem.order(u8, expected[0..pv.size], actual[0..pv.size]) == .eq);
    }
}

test "meshopt: decode is a no-op for uncompressed glTF" {
    const data = try parseGlbFromMemory(plain_glb);
    defer c.cgltf_free(data);

    try std.testing.expect(!hasMeshoptViews(data));
    const decode_res = c.agate_cgltf_decode_meshopt(
        &std.mem.zeroes(c.cgltf_options),
        data,
    );
    try std.testing.expect(decode_res == c.cgltf_result_success);
    // No view data materialized: views still resolve through buffer->data,
    // exactly as before meshopt support existed.
    for (data.buffer_views[0..data.buffer_views_count]) |*view| {
        try std.testing.expect(view.data == null);
    }
}

test "meshopt: plain fixture geometry matches golden values" {
    // The accessor reads cast into the buffer at 2/4-byte boundaries, so the
    // bytes must be aligned: @embedFile data has no alignment guarantee (this
    // test used to abort on an UBSan misaligned-load depending on where the
    // binary's rodata landed). Heap copies are max-aligned.
    const plain = try std.testing.allocator.dupe(u8, plain_glb);
    defer std.testing.allocator.free(plain);
    const data = try parseGlbFromMemory(plain);
    defer c.cgltf_free(data);

    const prim = &data.meshes[0].primitives[0];
    const pos_acc: *const c.cgltf_accessor = @ptrCast(prim.attributes[0].data);
    const idx_acc: *const c.cgltf_accessor = @ptrCast(prim.indices);

    try std.testing.expectEqual(@as(cgltf_size, 24), pos_acc.*.count);
    try std.testing.expectEqual(@as(cgltf_size, 36), idx_acc.*.count);

    // gltfpack quantizes positions to 14-bit integers (KHR_mesh_quantization),
    // so the accessor domain is raw integers 0..16383, not source units.
    try std.testing.expectEqualSlices(f32, &.{ 0, 0, 0 }, pos_acc.*.min[0..3]);
    try std.testing.expectEqualSlices(f32, &.{ 16383, 16383, 16383 }, pos_acc.*.max[0..3]);

    // Golden corners: first vertices of the packed cube, exactly as stored.
    var v: [3]f32 = undefined;
    try std.testing.expect(c.cgltf_accessor_read_float(pos_acc, 0, &v, 3) != 0);
    try std.testing.expectEqualSlices(f32, &.{ 0, 0, 16383 }, &v);
    try std.testing.expect(c.cgltf_accessor_read_float(pos_acc, 1, &v, 3) != 0);
    try std.testing.expectEqualSlices(f32, &.{ 16383, 0, 16383 }, &v);

    // Leading triangles of the packed index buffer.
    try std.testing.expectEqual(@as(c.cgltf_size, 0), c.cgltf_accessor_read_index(idx_acc, 0));
    try std.testing.expectEqual(@as(c.cgltf_size, 1), c.cgltf_accessor_read_index(idx_acc, 1));
    try std.testing.expectEqual(@as(c.cgltf_size, 2), c.cgltf_accessor_read_index(idx_acc, 2));
    try std.testing.expectEqual(@as(c.cgltf_size, 0), c.cgltf_accessor_read_index(idx_acc, 3));
    try std.testing.expectEqual(@as(c.cgltf_size, 2), c.cgltf_accessor_read_index(idx_acc, 4));
    try std.testing.expectEqual(@as(c.cgltf_size, 3), c.cgltf_accessor_read_index(idx_acc, 5));

    // Normals are i8-normalized: first face decodes to +Z.
    const nrm_acc: *const c.cgltf_accessor = @ptrCast(prim.attributes[1].data);
    try std.testing.expect(nrm_acc.*.normalized != 0);
    var n: [3]f32 = undefined;
    try std.testing.expect(c.cgltf_accessor_read_float(nrm_acc, 0, &n, 3) != 0);
    try std.testing.expectEqual(@as(f32, 0), n[0]);
    try std.testing.expectEqual(@as(f32, 0), n[1]);
    try std.testing.expectApproxEqAbs(@as(f32, 1), n[2], 1e-2);
}

test "meshopt: corrupted compressed blob is a clean decode error" {
    const copy = try std.testing.allocator.dupe(u8, compressed_glb);
    defer std.testing.allocator.free(copy);

    var options = std.mem.zeroes(c.cgltf_options);
    var data: ?*c.cgltf_data = null;
    const parse_res = c.cgltf_parse(&options, copy.ptr, copy.len, &data);
    try std.testing.expect(parse_res == c.cgltf_result_success);
    defer c.cgltf_free(data);

    // Smash the first compressed blob's magic byte. The meshopt stream does
    // not carry a payload checksum (payload corruption may decode into
    // garbage), but header corruption is always detected structurally.
    const mc = data.?.buffer_views[0].meshopt_compression;
    const bin_offset = @intFromPtr(data.?.bin) - @intFromPtr(copy.ptr);
    copy[bin_offset + mc.offset] ^= 0xff;

    const load_res = c.cgltf_load_buffers(&options, data, null);
    try std.testing.expect(load_res == c.cgltf_result_success);
    const decode_res = c.agate_cgltf_decode_meshopt(&options, data);
    try std.testing.expect(decode_res == c.cgltf_result_invalid_gltf);
}

const cgltf_size = c.cgltf_size;
