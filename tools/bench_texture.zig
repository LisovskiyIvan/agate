//! Micro-benchmark: classic sRGB convert + `buildRaw` vs `buildRawSrgbFused`.
//!
//! Compares the pre-optimization L0 path (convert-in-place, then copy +
//! mip chain inside `buildRaw`) against the fused single-pass L0
//! convert-copy (`buildRawSrgbFused`). No stb decode, no GPU upload: both
//! sides run the same in-memory RGBA source through the same
//! box-filter chain; only the L0 convert/copy seam differs.
//!
//! RUN ONLY ON AN OTHERWISE IDLE TREE (explicit timing permission
//! required). Builders contend on the allocator and the run is meaningless
//! under load:
//!
//!     zig build bench-texture [-- <reps> [glb ...]]
//!
//! Without GLB paths the timed fixture is a deterministic 2048x2048 RGBA
//! source. With GLB paths, the color-slot texture images actually used by
//! the materials (baseColor, emissive, sheen color when present — the same
//! slots the loader decodes with `srgb_to_linear`, see
//! `loader/materials.zig colorSlotImageFlags`), unique by image, are parsed
//! with `cgltf_parse_file`/`cgltf_load_buffers` and decoded ONCE via
//! `Texture.decodeMemory` (`.{ .gen_mipmaps = false, .srgb_to_linear = false }`)
//! outside every timer; the real RGBA pixels then run the same
//! classic-vs-fused comparison with the same equality gate. stb decode is
//! never counted as a conversion stage. Images that are not embedded buffer
//! views, or that do not decode to RGBA (Basis/BC/DDS block payloads),
//! fail explicitly instead of timing something else.
//!
//! Heap is bounded: default mode is one 2048x2048 source (16 MiB) + one
//! scratch (16 MiB) + one live mip chain (~21 MiB); GLB mode holds one
//! decoded L0 per used image plus one scratch sized to the largest plus one
//! live chain; timed builds are freed before the other path runs. The
//! pre-timing equality gate holds both chains (~75 MiB in default mode).
const std = @import("std");
const agate = @import("agate");
const c = agate.c;
const sokol = agate.sokol;
const Texture = agate.Texture;

const default_reps: usize = 10;
const max_reps: usize = 50;
const warmup_reps: usize = 2;

/// Verbatim algorithm copy of the pre-optimization call-site helper
/// `convertSrgbToLinearInPlace` (`src/agate/texture/color.zig`): same LUT
/// (`agate.texture.srgbToLinearU8`), same per-byte `i % 4` shape, alpha
/// lane skipped. The `texture` facade does not re-export that helper, so
/// the old path is replicated here exactly instead of guessed.
fn classicConvertSrgbToLinearInPlace(pixels: []u8) void {
    const lut = agate.texture.srgbToLinearU8;
    for (pixels, 0..) |*byte, i| {
        if (i % 4 != 3) byte.* = lut(byte.*);
    }
}

const TimedRaw = struct {
    raw: Texture.RawTexture,
    ms: f64,
};

/// Classic path: refresh scratch (UNTRACKED, outside the timer), convert in
/// place, then `buildRaw` (copies L0 + builds the chain). The untimed
/// refresh favors classic: the old tree converted the stb-owned buffer
/// directly, while the bench needs a fresh mutable copy per rep.
fn buildClassic(
    alloc: std.mem.Allocator,
    width: u32,
    height: u32,
    src: []const u8,
    scratch: []u8,
) !TimedRaw {
    std.debug.assert(scratch.len == src.len);
    @memcpy(scratch, src);
    const t0 = sokol.time.now();
    classicConvertSrgbToLinearInPlace(scratch);
    const raw = try Texture.buildRaw(alloc, width, height, scratch, true);
    const dt_ms: f64 = @floatCast(sokol.time.ms(sokol.time.now() -% t0));
    return .{ .raw = raw, .ms = dt_ms };
}

fn buildFused(alloc: std.mem.Allocator, width: u32, height: u32, src: []const u8) !TimedRaw {
    const t0 = sokol.time.now();
    const raw = try Texture.buildRawSrgbFused(alloc, width, height, src, true);
    const dt_ms: f64 = @floatCast(sokol.time.ms(sokol.time.now() -% t0));
    return .{ .raw = raw, .ms = dt_ms };
}

/// Equality over the full output: dimensions, level count, and every byte
/// of every mip level.
fn checkEqual(classic: *const Texture.RawTexture, fused: *const Texture.RawTexture) !void {
    if (classic.width != fused.width or classic.height != fused.height) return error.BenchMismatch;
    if (classic.num_levels != fused.num_levels) return error.BenchMismatch;
    for (0..classic.num_levels) |m| {
        const a = classic.levels[m] orelse return error.BenchMismatch;
        const b = fused.levels[m] orelse return error.BenchMismatch;
        if (!std.mem.eql(u8, a, b)) return error.BenchMismatch;
    }
}

/// Deterministic RGBA fill (arithmetic hash of the index, no RNG API).
/// Alpha varies so the alpha-verbatim path is exercised.
fn fillDeterministic(px: []u8) void {
    var i: usize = 0;
    while (i < px.len) : (i += 1) {
        var x: u64 = @as(u64, i) * 0x9E3779B97F4A7C15 + 0x12345;
        x ^= x >> 29;
        x *%= 0xBF58476D1CE4E5B9;
        x ^= x >> 32;
        px[i] = @intCast((x >> 11) & 0xFF);
    }
}

fn lessThan(_: void, a: f64, b: f64) bool {
    return a < b;
}

fn median(xs: []f64) f64 {
    std.mem.sort(f64, xs, {}, lessThan);
    const n = xs.len;
    if (n % 2 == 1) return xs[n / 2];
    return (xs[n / 2 - 1] + xs[n / 2]) * 0.5;
}

fn checkFixture(alloc: std.mem.Allocator, w: u32, h: u32, scratch: []u8, src: []u8) !void {
    fillDeterministic(src);
    @memcpy(scratch, src);
    classicConvertSrgbToLinearInPlace(scratch);
    var classic = try Texture.buildRaw(alloc, w, h, scratch, true);
    defer classic.deinit(alloc);
    var fused = try Texture.buildRawSrgbFused(alloc, w, h, src, true);
    defer fused.deinit(alloc);
    try checkEqual(&classic, &fused);
    std.debug.print("equal: {d}x{d} {d} levels ({d} src bytes)\n", .{
        w, h, classic.num_levels, src.len,
    });
}

/// Bit-identity gate for one real (already decoded) RGBA source: one
/// classic + one fused build, full mip chain compared, before any timing.
/// `scratch` must be at least `pixels.len` bytes; its refresh stays outside
/// every timer, same as the timed path.
fn checkRealFixture(alloc: std.mem.Allocator, w: u32, h: u32, pixels: []const u8, scratch: []u8) !void {
    std.debug.assert(scratch.len >= pixels.len);
    const sub = scratch[0..pixels.len];
    @memcpy(sub, pixels);
    classicConvertSrgbToLinearInPlace(sub);
    var classic = try Texture.buildRaw(alloc, w, h, sub, true);
    defer classic.deinit(alloc);
    var fused = try Texture.buildRawSrgbFused(alloc, w, h, pixels, true);
    defer fused.deinit(alloc);
    try checkEqual(&classic, &fused);
}

/// Warmup (discarded) + measured alternating-order batches for one RGBA
/// source. Extracted so the default 2048 fixture and every GLB image share
/// the same loop; `tag` only prefixes the header line (null keeps the
/// historical default output byte-for-byte).
fn timeFixture(
    alloc: std.mem.Allocator,
    w: u32,
    h: u32,
    src: []const u8,
    scratch: []u8,
    reps: usize,
    tag: ?[]const u8,
) !void {
    const n = src.len;

    // Warmup (discarded): alternating order, frees each build.
    for (0..warmup_reps) |i| {
        if (i % 2 == 0) {
            const cl = try buildClassic(alloc, w, h, src, scratch);
            var cr = cl.raw;
            cr.deinit(alloc);
            const f = try buildFused(alloc, w, h, src);
            var fr = f.raw;
            fr.deinit(alloc);
        } else {
            const f = try buildFused(alloc, w, h, src);
            var fr = f.raw;
            fr.deinit(alloc);
            const cl = try buildClassic(alloc, w, h, src, scratch);
            var cr = cl.raw;
            cr.deinit(alloc);
        }
    }

    // Measured batches, alternating order to cancel drift.
    const classic_ms = try alloc.alloc(f64, reps);
    defer alloc.free(classic_ms);
    const fused_ms = try alloc.alloc(f64, reps);
    defer alloc.free(fused_ms);
    for (0..reps) |i| {
        if (i % 2 == 0) {
            const cl = try buildClassic(alloc, w, h, src, scratch);
            var cr = cl.raw;
            classic_ms[i] = cl.ms;
            cr.deinit(alloc);
            const f = try buildFused(alloc, w, h, src);
            var fr = f.raw;
            fused_ms[i] = f.ms;
            fr.deinit(alloc);
        } else {
            const f = try buildFused(alloc, w, h, src);
            var fr = f.raw;
            fused_ms[i] = f.ms;
            fr.deinit(alloc);
            const cl = try buildClassic(alloc, w, h, src, scratch);
            var cr = cl.raw;
            classic_ms[i] = cl.ms;
            cr.deinit(alloc);
        }
    }

    const c_med = median(classic_ms); // sorts in place
    const f_med = median(fused_ms); // sorts in place
    if (tag) |t| {
        std.debug.print("bench_texture [{s}]: classic convert+buildRaw vs buildRawSrgbFused (gen_mipmaps=true)\n", .{t});
    } else {
        std.debug.print("bench_texture: classic convert+buildRaw vs buildRawSrgbFused (gen_mipmaps=true)\n", .{});
    }
    std.debug.print("fixture: {d}x{d} rgba8, {d} src bytes, {d} levels, clock: sokol.time, units: ms\n", .{
        w, h, n, Texture.mipLevelCount(w, h),
    });
    std.debug.print("classic: median {d:.3} ms (min {d:.3}, max {d:.3}, n={d})\n", .{
        c_med, classic_ms[0], classic_ms[reps - 1], reps,
    });
    std.debug.print("fused:   median {d:.3} ms (min {d:.3}, max {d:.3}, n={d})\n", .{
        f_med, fused_ms[0], fused_ms[reps - 1], reps,
    });
    if (f_med > 0) {
        std.debug.print("ratio classic/fused: {d:.3} (raw numbers only, no claim)\n", .{c_med / f_med});
    }
}

/// One timed source: decoded-once real RGBA pixels (`label` borrows the GLB
/// argv slice, no copy).
const GlbFixture = struct {
    label: []const u8,
    w: u32,
    h: u32,
    pixels: []u8,
};

fn imageIndex(gltf: *c.cgltf_data, img: [*c]const c.cgltf_image) ?usize {
    for (0..gltf.images_count) |i| {
        if (&gltf.images[i] == img) return i;
    }
    return null;
}

/// Marks the image behind one color-slot texture view (baseColor, emissive,
/// sheen color — the slots the loader decodes with `srgb_to_linear`), once
/// per image. Mirrors `loader/materials.zig colorSlotImageFlags`, including
/// the KHR_texture_basisu fallback; data slots are never marked here.
fn markColorView(
    gltf: *c.cgltf_data,
    view: [*c]const c.cgltf_texture_view,
    seen: []bool,
    hits: *std.ArrayListUnmanaged(usize),
    alloc: std.mem.Allocator,
) !void {
    if (view == null) return;
    const tex = view.*.texture orelse return;
    var img: [*c]const c.cgltf_image = null;
    if (tex.*.image != null) img = tex.*.image;
    if (img == null and tex.*.has_basisu != 0) img = tex.*.basisu_image;
    if (img == null) return;
    const idx = imageIndex(gltf, img) orelse return;
    if (!seen[idx]) {
        seen[idx] = true;
        try hits.append(alloc, idx);
    }
}

/// Decodes every color-slot image of one GLB to raw RGBA L0, once, outside
/// every timer. Only embedded buffer-view images that decode to RGBA are
/// supported: anything else (external URI, Basis/block payloads) fails
/// explicitly so the bench never times a different path by accident.
fn loadGlbFixtures(alloc: std.mem.Allocator, path: []const u8, out: *std.ArrayListUnmanaged(GlbFixture)) !void {
    const path_z = try alloc.dupeZ(u8, path);
    defer alloc.free(path_z);

    var options = std.mem.zeroes(c.cgltf_options);
    var data: ?*c.cgltf_data = null;
    if (c.cgltf_parse_file(&options, path_z.ptr, &data) != c.cgltf_result_success or data == null) {
        return error.GltfParseFailed;
    }
    defer c.cgltf_free(data);
    if (c.cgltf_load_buffers(&options, data, path_z.ptr) != c.cgltf_result_success) {
        return error.GltfLoadBuffersFailed;
    }
    const gltf = data.?;
    if (gltf.images_count == 0) return error.NoColorSlotImages;

    const seen = try alloc.alloc(bool, gltf.images_count);
    defer alloc.free(seen);
    @memset(seen, false);
    var hits: std.ArrayListUnmanaged(usize) = .empty;
    defer hits.deinit(alloc);
    for (0..gltf.materials_count) |i| {
        const mat = &gltf.materials[i];
        if (mat.has_pbr_metallic_roughness != 0) {
            try markColorView(gltf, &mat.pbr_metallic_roughness.base_color_texture, seen, &hits, alloc);
        }
        try markColorView(gltf, &mat.emissive_texture, seen, &hits, alloc);
        if (mat.has_sheen != 0) {
            try markColorView(gltf, &mat.sheen.sheen_color_texture, seen, &hits, alloc);
        }
    }
    if (hits.items.len == 0) return error.NoColorSlotImages;

    for (hits.items) |idx| {
        const img = &gltf.images[idx];
        const bv = img.buffer_view orelse return error.UnsupportedEmbeddedImage;
        if (bv.*.buffer == null or bv.*.buffer.*.data == null) return error.UnsupportedEmbeddedImage;
        const raw_buf: [*]const u8 = @ptrCast(bv.*.buffer.*.data);
        const bytes = (raw_buf + bv.*.offset)[0..bv.*.size];
        var raw = Texture.decodeMemory(alloc, bytes, .{ .gen_mipmaps = false, .srgb_to_linear = false }) catch |e| {
            return if (e == error.OutOfMemory) e else error.UnsupportedEmbeddedImage;
        };
        defer raw.deinit(alloc);
        const l0 = raw.levels[0] orelse return error.UnsupportedEmbeddedImage;
        if (l0.len > (256 << 20)) return error.ImageTooLarge;
        const pixels = try alloc.dupe(u8, l0);
        errdefer alloc.free(pixels);
        try out.append(alloc, .{ .label = path, .w = raw.width, .h = raw.height, .pixels = pixels });
    }
}

pub fn main(init: std.process.Init) !void {
    sokol.time.setup();
    const alloc = std.heap.c_allocator;

    var reps: usize = default_reps;
    var glb_paths: std.ArrayListUnmanaged([]const u8) = .empty;
    defer glb_paths.deinit(alloc);
    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, alloc);
    defer args.deinit();
    _ = args.next();
    if (args.next()) |arg| {
        if (std.fmt.parseInt(usize, arg, 10)) |r| {
            reps = r;
            while (args.next()) |p| try glb_paths.append(alloc, p);
        } else |_| {
            try glb_paths.append(alloc, arg);
            while (args.next()) |p| try glb_paths.append(alloc, p);
        }
    }
    if (reps < 1) reps = 1;
    if (reps > max_reps) reps = max_reps;

    // GLB mode: real sRGB texture pixels from the given files.
    if (glb_paths.items.len > 0) {
        var fixtures: std.ArrayListUnmanaged(GlbFixture) = .empty;
        defer {
            for (fixtures.items) |f| alloc.free(f.pixels);
            fixtures.deinit(alloc);
        }
        for (glb_paths.items) |p| try loadGlbFixtures(alloc, p, &fixtures);

        var max_n: usize = 1;
        for (fixtures.items) |f| max_n = @max(max_n, f.pixels.len);
        const scratch = try alloc.alloc(u8, max_n);
        defer alloc.free(scratch);

        // Bit-identity gate over every fixture BEFORE any timing.
        for (fixtures.items) |f| {
            try checkRealFixture(alloc, f.w, f.h, f.pixels, scratch);
            std.debug.print("equal: {s} {d}x{d} ({d} src bytes)\n", .{ f.label, f.w, f.h, f.pixels.len });
        }
        std.debug.print("identity: all GLB fixtures bit-identical, proceeding to timing\n", .{});

        for (fixtures.items) |f| {
            try timeFixture(alloc, f.w, f.h, f.pixels, scratch[0..f.pixels.len], reps, f.label);
        }
        return;
    }

    // 1. Bit-identity gate for 1px / NPOT / RGBA shapes BEFORE any timing.
    //    Small buffers are reused for every fixture (largest first is not
    //    needed: each fixture allocs exactly w*h*4).
    {
        const shapes = [_][2]u32{ .{ 1, 1 }, .{ 3, 1 }, .{ 2, 2 }, .{ 2048, 2048 } };
        for (shapes) |sh| {
            const w: u32 = sh[0];
            const h: u32 = sh[1];
            const n = @as(usize, w) * h * 4;
            const src = try alloc.alloc(u8, n);
            defer alloc.free(src);
            const scratch = try alloc.alloc(u8, n);
            defer alloc.free(scratch);
            try checkFixture(alloc, w, h, scratch, src);
        }
    }
    std.debug.print("identity: all fixtures bit-identical, proceeding to timing\n", .{});

    // 2. Timed fixture: 2048x2048 representative RGBA source.
    const w: u32 = 2048;
    const h: u32 = 2048;
    const n = @as(usize, w) * h * 4;
    const src = try alloc.alloc(u8, n);
    defer alloc.free(src);
    const scratch = try alloc.alloc(u8, n);
    defer alloc.free(scratch);
    fillDeterministic(src);

    // 3. Warmup + measured batches, alternating order to cancel drift.
    try timeFixture(alloc, w, h, src, scratch, reps, null);
}
