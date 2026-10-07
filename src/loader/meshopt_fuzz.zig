//! Fuzz/robustness target for the EXT_meshopt_compression decode path: the
//! full cgltf pipeline SceneLoader uses on untrusted files —
//! cgltf_parse -> cgltf_load_buffers -> agate_cgltf_decode_meshopt (C code on
//! attacker-controlled bytes).
//!
//! Invariant here is crash-safety only: cgltf allocates through C malloc (not
//! visible to std.testing.allocator) and malformed input just fails. Header
//! corruption must surface as a clean decode error; a flipped payload byte may
//! silently decode to garbage by design (no payload checksum — see
//! loader/meshopt.zig "Limitations").
//!
//! Seeds are the same gltfpack fixtures the unit tests use, plus truncations,
//! header flips and GLB-header-abuse variants. Run notes in build.zig.

const std = @import("std");
const c = @import("../c.zig").c;
const fzg = @import("../testing.zig");

const compressed_glb = @embedFile("fixtures/meshopt_cube.glb");
const plain_glb = @embedFile("fixtures/meshopt_cube_plain.glb");

/// GLB with a huge buffer length in the JSON chunk header region: pushes the
/// parser toward length-validation paths without crafting valid JSON.
const glb_garbage = "glTF" ++ "\x00\x00\x00\x00" ++ "\xff\xff\xff\x0f" ++ "{\"asset\":{},\"buffers\":[{\"byteLength\":18446744073709551615}]}";

const corpus = fzg.join(
    &[_][]const u8{ compressed_glb, plain_glb, glb_garbage },
    fzg.join(
        // Cuts through the 12-byte GLB header, the JSON chunk header and the
        // start of the BIN chunk.
        fzg.truncations(compressed_glb, &.{ 1, 4, 8, 12, 16, 20, 44, 100 }),
        fzg.join(
            // Flips the GLB magic/version/length fields and JSON chunk header.
            fzg.flips(compressed_glb, &.{ 0, 3, 4, 8, 12, 16, 20 }),
            fzg.flips(plain_glb, &.{ 0, 3, 12, 16 }),
        ),
    ),
);

fn testOne(_: void, smith: *std.testing.Smith) anyerror!void {
    @disableInstrumentation();
    var generated: [32768]u8 = undefined;
    const input = fzg.fuzzInput(smith, &generated);

    // The pipeline reads AND writes the buffer (meshopt decodes in place), so
    // the input needs a mutable copy.
    const copy = std.testing.allocator.dupe(u8, input) catch return;
    defer std.testing.allocator.free(copy);

    var options = std.mem.zeroes(c.cgltf_options);
    var data: ?*c.cgltf_data = null;
    if (c.cgltf_parse(&options, copy.ptr, copy.len, &data) != c.cgltf_result_success) return;
    defer c.cgltf_free(data);
    _ = c.cgltf_load_buffers(&options, data, null);
    _ = c.agate_cgltf_decode_meshopt(&options, data);
}

test "fuzz: meshopt glTF pipeline survives arbitrary bytes" {
    try std.testing.fuzz({}, testOne, .{ .corpus = corpus });
}
