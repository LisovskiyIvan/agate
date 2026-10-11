//! LDR CPU decode routing (stb/KTX2/DDS dispatch). Split out of `core.zig` (facade).
//!
//! `DecodeOptions` plus the GPU-free decode entries (`decodeMemory`/
//! `decodeFile`) and the unified routing union (`DecodedImage` via
//! `decodeImageMemory`/`decodeImageFile`: KTX2 block/Basis and DDS payloads
//! decode to `.block`, everything else to `.rgba`). Thread-safe; pair with
//! `fromRaw`/`fromRawBlock` on the main thread. Imports the `raw` leaf for
//! the RGBA8 containers/builders (one direction only, no cycle).

const std = @import("std");
const c = @import("../c.zig").c;
const ktx2 = @import("../ktx2.zig");
const dds = @import("../dds.zig");
const raw_mod = @import("core_raw.zig");

const RawTexture = raw_mod.RawTexture;
const buildRaw = raw_mod.buildRaw;
const buildRawSrgbFused = raw_mod.buildRawSrgbFused;

/// Decode-time switches for the CPU paths (GPU-free, thread-safe).
pub const DecodeOptions = struct {
    gen_mipmaps: bool = true,
    /// Explicit color space contract. Takes precedence over `slot` and `srgb_to_linear`.
    color_space: ?dds.TextureColorSpace = null,
    /// Semantic asset slot contract:
    /// - `.color`: albedo / emissive / base color (sRGB space)
    /// - `.data`: normal / metallic_roughness / occlusion / height / lookup (linear space)
    slot: ?dds.TextureSlot = null,
    /// Legacy fallback: converts RGB lanes before the mip chain is built, or selects
    /// the legacy DDS sRGB variant. Kept for backward compatibility.
    srgb_to_linear: bool = false,
    /// Transcode target for KTX2 Basis payloads (ETC1S/UASTC); ignored
    /// by every other format. Null = desktop-first .bc7 default (the
    /// fromRawBlock backend gate still rejects unsupported backends
    /// explicitly). Loaders snapshot per load via
    /// basisTargetForCurrentThread() so worker decodes already carry the
    /// backend-accurate target.
    basis_target: ?ktx2.BasisTarget = null,

    pub fn isColorSlot(self: DecodeOptions) bool {
        if (self.color_space) |cs| return cs == .srgb;
        if (self.slot) |s| return s.colorSpace() == .srgb;
        return self.srgb_to_linear;
    }
};
/// Decodes an in-memory image (PNG/JPEG/KTX2/...) to RGBA without
/// touching the GPU. Thread-safe; pair with `fromRaw`. KTX2 payloads
/// (magic sniff) route to the ktx2 reader: only its uncompressed LDR
/// subset decodes here — cube KTX2 files are rejected with
/// error.UnsupportedFaceCount (use ktx2.decodeCube instead), and Basis
/// payloads (ETC1S/UASTC) fail with error.BasisRequiresBlockDecode after
/// reporting the file's own validation reason for malformed files (use
/// decodeImageMemory or fromBasisMemory instead). DDS
/// payloads are block-compressed and have no RGBA8 form: they fail
/// here (malformed files with their own validation error, valid ones
/// with DdsRequiresBlockDecode) — use decodeImageMemory or
/// fromDdsMemory instead.
pub fn decodeMemory(allocator: std.mem.Allocator, bytes: []const u8, opts: DecodeOptions) !RawTexture {
    const is_color = opts.isColorSlot();
    if (ktx2.isBasisKtx2(bytes)) {
        // Report the file's own validation error when malformed (init or
        // subset gate), so a corrupt Basis file never degrades into a
        // generic stb failure; a valid one names the correct API instead.
        _ = try ktx2.basisInfo(bytes);
        return error.BasisRequiresBlockDecode;
    }
    if (dds.sniff(bytes)) {
        // Report the file's own validation error when malformed, so a
        // corrupt DDS never degrades into a generic stb failure; a
        // valid DDS names the correct API instead.
        var tmp = try dds.decodeBlock2D(allocator, bytes, .{
            .srgb = is_color,
            .color_space = opts.color_space,
            .slot = opts.slot,
        });
        tmp.deinit(allocator);
        return error.DdsRequiresBlockDecode;
    }
    if (ktx2.sniff(bytes)) {
        return ktx2.decode2D(allocator, bytes, .{
            .gen_mipmaps = opts.gen_mipmaps,
            // The caller's per-slot color/data decision is authoritative;
            // ktx2's format-tag auto detection applies only to its own
            // direct API (DecodeOptions.srgb_to_linear = null).
            .srgb_to_linear = is_color,
        });
    }

    var w: c_int = 0;
    var h: c_int = 0;
    var channels_in_file: c_int = 0;

    const data = c.stbi_load_from_memory(
        bytes.ptr,
        @intCast(bytes.len),
        &w,
        &h,
        &channels_in_file,
        4,
    );
    if (data == null) return error.ImageDecodeFailed;
    defer c.stbi_image_free(data);

    const width: u32 = @intCast(w);
    const height: u32 = @intCast(h);
    const size_bytes: usize = @as(usize, width) * @as(usize, height) * 4;
    // Fused L0 convert-copy (single pass, stride loop) so the box
    // filter still averages in linear space with bit-identical bytes.
    if (is_color) {
        var raw = try buildRawSrgbFused(allocator, width, height, data[0..size_bytes], opts.gen_mipmaps);
        raw.is_srgb = true;
        return raw;
    }
    var raw = try buildRaw(allocator, width, height, data[0..size_bytes], opts.gen_mipmaps);
    raw.is_srgb = false;
    return raw;
}
/// File variant of `decodeMemory`. Thread-safe; pair with `fromRaw`.
/// The file is read into memory first so KTX2 files (magic sniff) take
/// the same reader as in-memory payloads; everything else decodes via
/// stb from the buffered bytes.
pub fn decodeFile(allocator: std.mem.Allocator, file_path: []const u8, opts: DecodeOptions) !RawTexture {
    // Zig 0.16 removed std.fs.cwd(); read through the global
    // single-threaded Io (same pattern as CubeTexture.fromEquirectangularFile).
    const io = std.Io.Threaded.global_single_threaded.io();
    const file = try std.Io.Dir.cwd().openFile(io, file_path, .{});
    defer file.close(io);

    const file_size = try file.length(io);
    const bytes = try allocator.alloc(u8, std.math.cast(usize, file_size) orelse return error.ImageTooLarge);
    defer allocator.free(bytes);

    const read = try file.readPositionalAll(io, bytes, 0);
    if (read < bytes.len) return error.ImageDecodeFailed;
    return decodeMemory(allocator, bytes, opts);
}
/// CPU-decoded image in either engine form: RGBA8 levels or owned
/// block-compressed levels. GPU-free; upload the payload with fromRaw
/// (.rgba) or fromRawBlock (.block) on the main thread. Produced by
/// decodeImageMemory/decodeImageFile; free with deinit.
pub const DecodedImage = union(enum) {
    rgba: RawTexture,
    block: ktx2.RawBlockTexture,

    pub fn deinit(self: *DecodedImage, allocator: std.mem.Allocator) void {
        switch (self.*) {
            .rgba => |*r| r.deinit(allocator),
            .block => |*b| b.deinit(allocator),
        }
    }

    /// Exact bytes the upload hands to sg (asset byte budget).
    pub fn totalBytes(self: *const DecodedImage) usize {
        switch (self.*) {
            .rgba => |*r| {
                var total: usize = 0;
                for (r.levels[0..r.num_levels]) |level| {
                    if (level) |buf| total += buf.len;
                }
                return total;
            },
            .block => |*b| return b.totalBytes(),
        }
    }
};
/// decodeMemory plus block routing: KTX2 BC/ASTC and DDS BC payloads
/// decode to .block (owned slices, mip chain as authored, no
/// synthesis); KTX2 Basis payloads (ETC1S/UASTC) transcode to
/// opts.basis_target (null = desktop-first .bc7) — .block for bc7/astc,
/// .rgba for the rgba32 fallback; everything else behaves exactly like
/// decodeMemory (.rgba). Thread-safe; pair with fromRaw/fromRawBlock. A
/// Basis file that fails validation surfaces the reader error — never a
/// silent RGBA8 fallback.
/// decode_opts are RGBA8-only, except srgb_to_linear, which doubles as
/// the DDS legacy sRGB decision (DX10/KTX2 files carry their own tag):
/// gen_mipmaps has no effect on .block (chain and sRGB-ness ride in
/// the authored levels and the GPU format). For Basis, srgb_to_linear is
/// the explicit RGBA32-path decision (block targets follow the file DFD).
pub fn decodeImageMemory(allocator: std.mem.Allocator, bytes: []const u8, opts: DecodeOptions) !DecodedImage {
    const is_color = opts.isColorSlot();
    if (ktx2.isBasisKtx2(bytes)) {
        const t = opts.basis_target orelse .bc7;
        return try ktx2.decodeBasis2D(allocator, bytes, t, .{
            .gen_mipmaps = opts.gen_mipmaps,
            .srgb_to_linear = is_color,
        });
    }
    if (ktx2.isBlockKtx2(bytes)) {
        return .{ .block = try ktx2.decodeBlock2D(allocator, bytes) };
    }
    if (dds.sniff(bytes)) {
        return .{ .block = try dds.decodeBlock2D(allocator, bytes, .{
            .srgb = is_color,
            .color_space = opts.color_space,
            .slot = opts.slot,
        }) };
    }
    var raw_opts = opts;
    raw_opts.srgb_to_linear = is_color;
    return .{ .rgba = try decodeMemory(allocator, bytes, raw_opts) };
}
/// File variant of decodeImageMemory. Same buffered-read pattern as
/// decodeFile (Zig 0.16 removed std.fs.cwd()); thread-safe, pair with
/// fromRaw/fromRawBlock.
pub fn decodeImageFile(allocator: std.mem.Allocator, file_path: []const u8, opts: DecodeOptions) !DecodedImage {
    const io = std.Io.Threaded.global_single_threaded.io();
    const file = try std.Io.Dir.cwd().openFile(io, file_path, .{});
    defer file.close(io);

    const file_size = try file.length(io);
    const bytes = try allocator.alloc(u8, std.math.cast(usize, file_size) orelse return error.ImageTooLarge);
    defer allocator.free(bytes);

    const read = try file.readPositionalAll(io, bytes, 0);
    if (read < bytes.len) return error.ImageDecodeFailed;
    return decodeImageMemory(allocator, bytes, opts);
}
