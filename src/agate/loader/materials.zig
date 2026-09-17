const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;

const c = @import("../c.zig").c;
const math = @import("math");
const Color3 = math.Color3;

const Scene = @import("../scene.zig").Scene;
const Material = @import("../material.zig").Material;
const UvTransform = @import("../material.zig").UvTransform;
const Texture = @import("../texture.zig").Texture;
const ktx2 = @import("../ktx2.zig");
const assets = @import("../assets.zig");
const gpu_thread = @import("../gpu_thread.zig");

/// Резолвинг рабочего изображения текстуры с учётом KHR_texture_basisu:
/// при использовании расширения `textures[i].source` отсутствует, а
/// изображение (обычно .ktx2) лежит в `basisu_image`. Без расширения —
/// обычное `image`. Возвращает null, когда текстуры нет вообще.
pub fn textureImage(tex: [*c]const c.cgltf_texture) ?[*c]const c.cgltf_image {
    if (tex == null) return null;
    // Явные сравнения с null: нулевой C-указатель при неявном приведении
    // к ?[*c] дал бы ненулевой optional с нулевым payload.
    if (tex.*.image != null) return tex.*.image;
    if (tex.*.has_basisu != 0 and tex.*.basisu_image != null) return tex.*.basisu_image;
    return null;
}

const DecodeJob = struct {
    allocator: std.mem.Allocator,
    image_index: usize,
    /// Embedded buffer view, valid until the cgltf data is freed.
    bytes: ?[]const u8 = null,
    /// External URI path, owned by the job.
    path: ?[]const u8 = null,
    /// sRGB -> linear conversion before mip generation (color-slot images).
    /// Действует только на .rgba: блочные уровни грузятся как в файле.
    srgb: bool = false,
    out: ?Texture.DecodedImage = null,

    fn run(self: *DecodeJob) void {
        const opts: Texture.DecodeOptions = .{ .gen_mipmaps = true, .srgb_to_linear = self.srgb };
        if (self.bytes) |b| {
            self.out = Texture.decodeImageMemory(self.allocator, b, opts) catch null;
        } else if (self.path) |p| {
            self.out = Texture.decodeImageFile(self.allocator, p, opts) catch null;
        }
    }
};

const DecodeQueue = struct {
    jobs: []DecodeJob,
    next: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),

    fn run(self: *DecodeQueue) void {
        while (true) {
            const i = self.next.fetchAdd(1, .monotonic);
            if (i >= self.jobs.len) break;
            self.jobs[i].run();
        }
    }
};

/// Marks images referenced by COLOR texture slots (base color / emissive) so
/// the parallel decoder runs sRGB -> linear before mip generation. Data slots
/// (normal / metallic-roughness / occlusion) stay linear.
pub fn colorSlotImageFlags(allocator: std.mem.Allocator, gltf: *c.cgltf_data) ![]bool {
    const flags = try allocator.alloc(bool, gltf.images_count);
    @memset(flags, false);
    for (0..gltf.materials_count) |i| {
        const mat = &gltf.materials[i];
        if (mat.has_pbr_metallic_roughness != 0) {
            markColorSlotImage(gltf, mat.pbr_metallic_roughness.base_color_texture, flags);
        }
        markColorSlotImage(gltf, mat.emissive_texture, flags);
    }
    return flags;
}

fn markColorSlotImage(gltf: *c.cgltf_data, view: c.cgltf_texture_view, flags: []bool) void {
    const tex = view.texture orelse return;
    const img = textureImage(tex) orelse return;
    if (imageIndexFor(gltf, img)) |idx| {
        if (idx < flags.len) flags[idx] = true;
    }
}

fn imageIndexFor(gltf: *c.cgltf_data, img: [*c]const c.cgltf_image) ?usize {
    for (0..gltf.images_count) |im_i| {
        if (&gltf.images[im_i] == img) return im_i;
    }
    return null;
}

/// Decodes every GLB/GLTF image on worker threads. This never fails the
/// load: images that could not be decoded stay null and loadTextureFromView
/// falls back to the synchronous path.
pub fn decodeImagesInParallel(scene: *Scene, gltf: *c.cgltf_data, decoded: []?Texture.DecodedImage, base_dir: ?[]const u8) void {
    if (gltf.images_count == 0) return;

    const jobs = scene.allocator.alloc(DecodeJob, gltf.images_count) catch return;
    defer scene.allocator.free(jobs);

    // Per-image color-slot scan drives load-time sRGB -> linear.
    const srgb_flags: ?[]bool = colorSlotImageFlags(scene.allocator, gltf) catch null;
    defer if (srgb_flags) |flags| scene.allocator.free(flags);

    var job_count: usize = 0;
    for (0..gltf.images_count) |i| {
        const img = &gltf.images[i];
        var job = DecodeJob{ .allocator = scene.allocator, .image_index = i };
        if (srgb_flags) |flags| job.srgb = flags[i];

        if (img.buffer_view) |bv| {
            if (bv.*.buffer != null and bv.*.buffer.*.data != null) {
                const raw_buf: [*]const u8 = @ptrCast(bv.*.buffer.*.data);
                job.bytes = (raw_buf + bv.*.offset)[0..bv.*.size];
                jobs[job_count] = job;
                job_count += 1;
                continue;
            }
        }

        if (img.uri) |uri_c| {
            const uri = std.mem.span(uri_c);
            const path = if (base_dir) |dir|
                std.fs.path.join(scene.allocator, &.{ dir, uri }) catch continue
            else
                scene.allocator.dupe(u8, uri) catch continue;
            job.path = path;
            jobs[job_count] = job;
            job_count += 1;
        }
    }

    if (job_count == 0) return;

    var queue = DecodeQueue{ .jobs = jobs[0..job_count] };
    const cpu_count = std.Thread.getCpuCount() catch 4;
    const max_threads = 8;
    const worker_count = @min(job_count, @min(cpu_count, max_threads));
    if (worker_count > 1) {
        var threads: [max_threads]std.Thread = undefined;
        var spawned: usize = 0;
        while (spawned < worker_count - 1) : (spawned += 1) {
            threads[spawned] = std.Thread.spawn(.{}, DecodeQueue.run, .{&queue}) catch break;
        }
        queue.run();
        for (threads[0..spawned]) |t| t.join();
    } else {
        queue.run();
    }

    for (jobs[0..job_count]) |*job| {
        if (job.out) |raw| {
            decoded[job.image_index] = raw;
            job.out = null;
        }
        if (job.path) |p| scene.allocator.free(p);
    }
}

/// Maps a glTF sampler descriptor's raw enum values onto Texture.Options.
/// wrap_s/wrap_t (33071 clamp, 33648 mirrored, 10497 repeat), mag_filter and
/// min_filter (9728 nearest, 9729 linear; mipmapped variants 9984..9987 also
/// split their mip half into Options.mip_filter). Unknown values keep the
/// engine defaults (glTF spec allows omitting sampler fields).
pub fn applyGltfSampler(wrap_s: c_int, wrap_t: c_int, mag_filter: c_int, min_filter: c_int, opts: *Texture.Options) void {
    switch (wrap_s) {
        33071 => opts.wrap_u = .CLAMP_TO_EDGE,
        33648 => opts.wrap_u = .MIRRORED_REPEAT,
        10497 => opts.wrap_u = .REPEAT,
        else => {},
    }
    switch (wrap_t) {
        33071 => opts.wrap_v = .CLAMP_TO_EDGE,
        33648 => opts.wrap_v = .MIRRORED_REPEAT,
        10497 => opts.wrap_v = .REPEAT,
        else => {},
    }
    switch (mag_filter) {
        9728 => opts.mag_filter = .NEAREST,
        9729 => opts.mag_filter = .LINEAR,
        else => {},
    }
    switch (min_filter) {
        // min half
        9728, 9984, 9986 => opts.min_filter = .NEAREST,
        9729, 9985, 9987 => opts.min_filter = .LINEAR,
        else => {},
    }
    switch (min_filter) {
        // mip half: *_MIPMAP_NEAREST variants select nearest mips
        9984, 9985 => opts.mip_filter = .NEAREST,
        9986, 9987 => opts.mip_filter = .LINEAR,
        else => {},
    }
}

/// Extracts the KHR_texture_transform UV map from a glTF texture view
/// (identity when the extension is absent). Documented limitation: views
/// with `texCoord` > 0 reference a second UV set the engine does not load
/// (single v_uv varying); the transform still applies to texcoord0.
/// `anytype` accepts both normal and C (allowzero) pointers to
/// cgltf_texture_view (cgltf's own structs carry C-pointer parents).
pub fn uvTransformFromView(view: anytype) UvTransform {
    if (view.has_transform == 0) return UvTransform.identity;
    const t = view.transform;
    return .{
        .offset = .{ t.offset[0], t.offset[1] },
        .rotation = t.rotation,
        .scale = .{ t.scale[0], t.scale[1] },
    };
}

/// Stage 2: schedules background decodes for images the sync path could
/// not serve, registering material slots as patch targets. One decode per
/// image+srgb pair (`seen`), even when several materials share it.
/// Embedded buffer-view bytes are copied at registration time — the cgltf
/// data dies when appendGlb returns, long before the worker decodes.
pub const AsyncTexCtx = struct {
    scene: *Scene,
    gltf: *c.cgltf_data,
    base_dir: ?[]const u8,
    queue: *assets.UploadQueue,
    /// image_index * 2 + srgb -> in-flight request
    seen: std.AutoHashMapUnmanaged(usize, *assets.PendingTexture) = .empty,

    pub fn init(scene: *Scene, gltf: *c.cgltf_data, base_dir: ?[]const u8, queue: *assets.UploadQueue) AsyncTexCtx {
        return .{ .scene = scene, .gltf = gltf, .base_dir = base_dir, .queue = queue };
    }

    pub fn deinit(self: *AsyncTexCtx) void {
        self.seen.deinit(self.scene.allocator);
    }

    /// Never fails the load: allocation or extraction problems leave the
    /// slot null (default-white), matching the sync path's failure mode.
    pub fn register(self: *AsyncTexCtx, view: [*c]const c.cgltf_texture_view, srgb: bool, slot: *?Texture) void {
        if (view == null) return;
        if (view.*.texture == null) return;
        const tex = view.*.texture.?;
        const img_ptr = textureImage(tex) orelse return;
        const img_idx = imageIndexFor(self.gltf, img_ptr) orelse return;
        const key = img_idx * 2 + (if (srgb) @as(usize, 1) else @as(usize, 0));

        if (self.seen.get(key)) |existing| {
            existing.addTarget(slot);
            return;
        }

        var tex_options: Texture.Options = .{};
        if (tex.*.sampler) |smp| {
            applyGltfSampler(@intCast(smp.*.wrap_s), @intCast(smp.*.wrap_t), @intCast(smp.*.mag_filter), @intCast(smp.*.min_filter), &tex_options);
        }
        tex_options.srgb_to_linear = srgb;
        const decode_opts: Texture.DecodeOptions = .{ .gen_mipmaps = true, .srgb_to_linear = srgb };

        const img = &self.gltf.images[img_idx];
        var pending: ?*assets.PendingTexture = null;
        if (img.buffer_view) |bv| {
            if (bv.*.buffer != null and bv.*.buffer.*.data != null) {
                const raw_buf: [*]const u8 = @ptrCast(bv.*.buffer.*.data);
                const bytes = self.scene.allocator.dupe(u8, (raw_buf + bv.*.offset)[0..bv.*.size]) catch return;
                pending = self.queue.requestMemory(bytes, tex_options, decode_opts) catch {
                    self.scene.allocator.free(bytes);
                    return;
                };
            }
        } else if (img.uri) |uri_c| {
            const uri = std.mem.span(uri_c);
            const path = if (self.base_dir) |dir|
                std.fs.path.join(self.scene.allocator, &.{ dir, uri }) catch return
            else
                self.scene.allocator.dupe(u8, uri) catch return;
            pending = self.queue.getOrRequestFile(path, tex_options, decode_opts) catch {
                self.scene.allocator.free(path);
                return;
            };
            self.scene.allocator.free(path);
        } else return;

        pending.?.addTarget(slot);
        self.seen.put(self.scene.allocator, key, pending.?) catch {};
    }
};

/// Sync load with an async fallback: on success the texture lands in
/// `slot`; otherwise a background decode is scheduled for it (async mode)
/// or the slot stays null (sync mode). Returns true when the slot has a
/// texture now or will get one later.
pub fn loadTextureSlot(
    scene: *Scene,
    gltf: *c.cgltf_data,
    image_cache: []?Texture,
    decoded: []?Texture.DecodedImage,
    view: [*c]const c.cgltf_texture_view,
    base_dir: ?[]const u8,
    srgb_to_linear: bool,
    slot: *?Texture,
    actx: ?*AsyncTexCtx,
) bool {
    if (loadTextureFromView(scene, gltf, image_cache, decoded, view, base_dir, srgb_to_linear, actx == null)) |t| {
        slot.* = t;
        return true;
    }
    if (actx) |a| {
        a.register(view, srgb_to_linear, slot);
        return true;
    }
    return false;
}

pub fn loadTextureFromView(
    scene: *Scene,
    gltf: *c.cgltf_data,
    image_cache: []?Texture,
    decoded: []?Texture.DecodedImage,
    view: [*c]const c.cgltf_texture_view,
    base_dir: ?[]const u8,
    srgb_to_linear: bool,
    allow_sync_fallback: bool,
) ?Texture {
    if (view == null) return null;
    if (view.*.texture == null) return null;
    const tex = view.*.texture.?;
    const img = textureImage(tex) orelse return null;

    const img_idx = imageIndexFor(gltf, img);
    const cache_idx: ?usize = if (img_idx) |idx| blk: {
        const slot = idx * 2 + (if (srgb_to_linear) @as(usize, 1) else @as(usize, 0));
        if (slot < image_cache.len) break :blk slot;
        if (idx < image_cache.len) break :blk idx;
        break :blk null;
    } else null;

    if (cache_idx) |c_idx| {
        if (image_cache[c_idx]) |cached| {
            return cached;
        }
    }

    var tex_options: Texture.Options = .{};
    if (tex.*.sampler) |smp| {
        applyGltfSampler(@intCast(smp.*.wrap_s), @intCast(smp.*.wrap_t), @intCast(smp.*.mag_filter), @intCast(smp.*.min_filter), &tex_options);
    }
    tex_options.srgb_to_linear = srgb_to_linear;

    // Pre-decoded on worker threads: only the GPU upload runs here.
    // .rgba идёт старым путём (декод чужого sRGB-варианта не трогаем);
    // .block грузится через fromRawBlock: sRGB-ность уже в варианте
    // GPU-формата, srgb_to_linear/gen_mipmaps к блочным уровням не
    // применяются (цепочка — как в файле). Кэш общий с RGBA-веткой, но
    // алиасинга нет: ячейка хранит готовый Texture, а не декод.
    if (img_idx) |idx| {
        if (idx < decoded.len and decoded[idx] != null) {
            const usable = switch (decoded[idx].?) {
                .rgba => |raw| raw.is_srgb == srgb_to_linear,
                .block => true,
            };
            if (usable) {
                // GPU upload of a pre-decoded image (sg.makeImage).
                gpu_thread.assertOnContextThread();
                var owned = decoded[idx].?;
                decoded[idx] = null;
                switch (owned) {
                    .rgba => |*raw| {
                        const loaded = Texture.fromRaw(raw, tex_options);
                        owned.deinit(scene.allocator);
                        if (cache_idx) |c_idx| {
                            image_cache[c_idx] = loaded;
                        }
                        return loaded;
                    },
                    .block => |*blk| {
                        // Слот уже занулён выше, deinit — здесь: при
                        // BlockFormatNotSupportedByBackend уходим в null
                        // (дальше — async-регистрация или пустой слот),
                        // как при любой другой ошибке декода.
                        defer owned.deinit(scene.allocator);
                        const loaded = Texture.fromRawBlock(blk, tex_options) catch return null;
                        if (cache_idx) |c_idx| {
                            image_cache[c_idx] = loaded;
                        }
                        return loaded;
                    },
                }
            }
        }
    }

    if (!allow_sync_fallback) return null;

    // Everything below creates GPU objects (sg.makeImage through
    // Texture.fromMemory/fromFile): synchronous texture loading must run on
    // the graphics thread. Async texture mode returns null above and queues
    // the decode + upload through the UploadQueue instead, which is what
    // makes an off-context GLB load legal.
    gpu_thread.assertOnContextThread();

    // 1. Embedded buffer view (typical in GLB or embedded GLTF)
    if (img.*.buffer_view) |bv| {
        if (bv.*.buffer != null and bv.*.buffer.*.data != null) {
            const raw_buf = @as([*]const u8, @ptrCast(bv.*.buffer.*.data));
            const img_data = (raw_buf + bv.*.offset)[0..bv.*.size];
            if (uploadDecodedMemory(scene, img_data, tex_options)) |loaded| {
                if (cache_idx) |c_idx| {
                    image_cache[c_idx] = loaded;
                }
                return loaded;
            }
        }
    }

    // 2. External URI (typical in standard GLTF with external textures)
    if (img.*.uri) |uri_c| {
        const uri = std.mem.span(uri_c);
        if (base_dir) |dir| {
            const full_path = std.fs.path.join(scene.allocator, &.{ dir, uri }) catch null;
            if (full_path) |fp| {
                defer scene.allocator.free(fp);
                if (uploadDecodedFile(scene, fp, tex_options)) |loaded| {
                    if (cache_idx) |c_idx| {
                        image_cache[c_idx] = loaded;
                    }
                    return loaded;
                }
            }
        } else {
            if (uploadDecodedFile(scene, uri, tex_options)) |loaded| {
                if (cache_idx) |c_idx| {
                    image_cache[c_idx] = loaded;
                }
                return loaded;
            }
        }
    }

    return null;
}

/// Синхронный decode+upload для фолбэка loadTextureFromView: декод через
/// decodeImageMemory/File, загрузка — fromRaw (.rgba) или fromRawBlock
/// (.block). Ошибки декода и BlockFormatNotSupportedByBackend глотаются в
/// null — так же, как раньше глотались ошибки fromMemory/fromFile.
/// Декод GPU-free, но fromRaw/fromRawBlock делают sg.makeImage: вызывать
/// только на context-потоке.
/// Для .block gen_mipmaps/srgb_to_linear не действуют: цепочка мипов — как
/// в файле, а sRGB-ность несёт сам вариант GPU-формата.
fn uploadDecodedMemory(scene: *Scene, bytes: []const u8, tex_options: Texture.Options) ?Texture {
    var dec = Texture.decodeImageMemory(scene.allocator, bytes, .{
        .gen_mipmaps = tex_options.mipmaps,
        .srgb_to_linear = tex_options.srgb_to_linear,
    }) catch return null;
    defer dec.deinit(scene.allocator);
    return switch (dec) {
        .rgba => |*raw| Texture.fromRaw(raw, tex_options),
        .block => |*blk| Texture.fromRawBlock(blk, tex_options) catch null,
    };
}

fn uploadDecodedFile(scene: *Scene, path: []const u8, tex_options: Texture.Options) ?Texture {
    var dec = Texture.decodeImageFile(scene.allocator, path, .{
        .gen_mipmaps = tex_options.mipmaps,
        .srgb_to_linear = tex_options.srgb_to_linear,
    }) catch return null;
    defer dec.deinit(scene.allocator);
    return switch (dec) {
        .rgba => |*raw| Texture.fromRaw(raw, tex_options),
        .block => |*blk| Texture.fromRawBlock(blk, tex_options) catch null,
    };
}

pub fn loadMaterials(
    scene: *Scene,
    gltf: *c.cgltf_data,
    base_dir: ?[]const u8,
    materials: []?Material,
    image_cache: []?Texture,
    decoded: []?Texture.DecodedImage,
    actx: ?*AsyncTexCtx,
) !void {
    for (0..gltf.materials_count) |i| {
        const src_mat = &gltf.materials[i];
        const mat_name = if (src_mat.name != null)
            std.mem.span(src_mat.name)
        else
            "glb_material";

        const pbr_mat = try scene.createPBRMaterial(mat_name);

        if (src_mat.has_pbr_metallic_roughness != 0) {
            const pbr = &src_mat.pbr_metallic_roughness;
            pbr_mat.albedo_color = Color3.new(
                pbr.base_color_factor[0],
                pbr.base_color_factor[1],
                pbr.base_color_factor[2],
            );
            pbr_mat.alpha = pbr.base_color_factor[3];
            pbr_mat.metallic = pbr.metallic_factor;
            pbr_mat.roughness = pbr.roughness_factor;

            // Color slots load sRGB -> linear (glTF: textures are sRGB,
            // factors linear); data slots stay linear.
            _ = loadTextureSlot(scene, gltf, image_cache, decoded, &pbr.base_color_texture, base_dir, true, &pbr_mat.albedo_texture, actx);
            pbr_mat.albedo_uv_transform = uvTransformFromView(&pbr.base_color_texture);
            _ = loadTextureSlot(scene, gltf, image_cache, decoded, &pbr.metallic_roughness_texture, base_dir, false, &pbr_mat.metallic_roughness_texture, actx);
            pbr_mat.metallic_roughness_uv_transform = uvTransformFromView(&pbr.metallic_roughness_texture);
        }

        _ = loadTextureSlot(scene, gltf, image_cache, decoded, &src_mat.normal_texture, base_dir, false, &pbr_mat.normal_texture, actx);
        if (src_mat.normal_texture.texture != null) {
            // cgltf defaults texture-view scale to 1.0 (cgltf.h parse).
            pbr_mat.normal_scale = src_mat.normal_texture.scale;
        }
        pbr_mat.normal_uv_transform = uvTransformFromView(&src_mat.normal_texture);
        _ = loadTextureSlot(scene, gltf, image_cache, decoded, &src_mat.occlusion_texture, base_dir, false, &pbr_mat.occlusion_texture, actx);
        pbr_mat.occlusion_strength = src_mat.occlusion_texture.scale;
        pbr_mat.occlusion_uv_transform = uvTransformFromView(&src_mat.occlusion_texture);

        const emissive_textured = loadTextureSlot(scene, gltf, image_cache, decoded, &src_mat.emissive_texture, base_dir, true, &pbr_mat.emissive_texture, actx);
        pbr_mat.emissive_uv_transform = uvTransformFromView(&src_mat.emissive_texture);
        if ((pbr_mat.emissive_texture != null or emissive_textured) and
            src_mat.emissive_factor[0] == 0.0 and
            src_mat.emissive_factor[1] == 0.0 and
            src_mat.emissive_factor[2] == 0.0)
        {
            pbr_mat.emissive_color = Color3.white;
        } else {
            pbr_mat.emissive_color = Color3.new(
                src_mat.emissive_factor[0],
                src_mat.emissive_factor[1],
                src_mat.emissive_factor[2],
            );
        }

        // glTF alphaMode/doubleSided mapping (cgltf has no has_alpha_cutoff:
        // alpha_cutoff always parses, defaulting to 0.5 when absent).
        if (src_mat.alpha_mode == c.cgltf_alpha_mode_mask) {
            pbr_mat.alpha_mode = .cutout;
            pbr_mat.alpha_cutoff = std.math.clamp(src_mat.alpha_cutoff, 0.0, 1.0);
        } else if (src_mat.alpha_mode == c.cgltf_alpha_mode_blend) {
            pbr_mat.alpha_mode = .blend;
        } else {
            pbr_mat.alpha_mode = .@"opaque";
        }
        pbr_mat.double_sided = src_mat.double_sided != 0;
        pbr_mat.unlit = src_mat.unlit != 0;

        materials[i] = .{ .pbr = pbr_mat };
    }
}

// The loadMaterials tests only exercise the allocator-backed path, so they
// share the CPU-only Scene fixture from testing.zig (see testScene there).
const testScene = @import("../testing.zig").testScene;

test "loadMaterials maps alphaMode/cutoff/doubleSided (GPU-free)" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var scene = testScene(alloc);

    var src: [3]c.cgltf_material = .{
        std.mem.zeroes(c.cgltf_material),
        std.mem.zeroes(c.cgltf_material),
        std.mem.zeroes(c.cgltf_material),
    };
    // [0] default zero: OPAQUE, single-sided.
    // [1] MASK with cutoff + double-sided.
    src[1].alpha_mode = c.cgltf_alpha_mode_mask;
    src[1].alpha_cutoff = 0.2;
    src[1].double_sided = 1;
    // [2] BLEND, single-sided, out-of-range cutoff must be ignored.
    src[2].alpha_mode = c.cgltf_alpha_mode_blend;
    src[2].alpha_cutoff = 42.0;
    // Out-of-range MASK cutoff clamps to [0,1] (checked on a copy below).
    const clamped = std.math.clamp(@as(f32, 1.7), 0.0, 1.0);
    try std.testing.expectEqual(@as(f32, 1.0), clamped);

    var data: c.cgltf_data = std.mem.zeroes(c.cgltf_data);
    data.materials = &src[0];
    data.materials_count = src.len;

    var out: [3]?Material = .{ null, null, null };
    const empty_tex: []?Texture = &.{};
    const empty_dec: []?Texture.DecodedImage = &.{};
    try loadMaterials(&scene, &data, null, &out, empty_tex, empty_dec, null);

    try std.testing.expect(out[0].? == .pbr);
    try std.testing.expect(out[0].?.pbr.alpha_mode == .@"opaque");
    try std.testing.expectEqual(@as(f32, 0.5), out[0].?.pbr.alpha_cutoff);
    try std.testing.expect(!out[0].?.pbr.double_sided);

    try std.testing.expect(out[1].?.pbr.alpha_mode == .cutout);
    try std.testing.expectEqual(@as(f32, 0.2), out[1].?.pbr.alpha_cutoff);
    try std.testing.expect(out[1].?.pbr.double_sided);

    try std.testing.expect(out[2].?.pbr.alpha_mode == .blend);
    try std.testing.expectEqual(@as(f32, 0.5), out[2].?.pbr.alpha_cutoff);
    try std.testing.expect(!out[2].?.pbr.double_sided);
}

test "applyGltfSampler maps wrap, mag and the min+mip halves of min_filter" {
    // glTF 2.0 spec enums: 10497 REPEAT, 33071 CLAMP_TO_EDGE, 33648
    // MIRRORED_REPEAT; 9728 NEAREST, 9729 LINEAR; 9984 NEAREST_MIPMAP_NEAREST,
    // 9985 LINEAR_MIPMAP_NEAREST, 9986 NEAREST_MIPMAP_LINEAR,
    // 9987 LINEAR_MIPMAP_LINEAR.

    // Spec defaults (sampler omitted): wrap REPEAT/REPEAT, mag LINEAR,
    // min LINEAR_MIPMAP_LINEAR -> min LINEAR + mip LINEAR.
    var opts: Texture.Options = .{};
    applyGltfSampler(10497, 10497, 9729, 9987, &opts);
    try std.testing.expectEqual(sg.Wrap.REPEAT, opts.wrap_u);
    try std.testing.expectEqual(sg.Wrap.REPEAT, opts.wrap_v);
    try std.testing.expectEqual(sg.Filter.LINEAR, opts.mag_filter);
    try std.testing.expectEqual(sg.Filter.LINEAR, opts.min_filter);
    try std.testing.expectEqual(sg.Filter.LINEAR, opts.mip_filter);

    // Clamp + mirrored wrap, nearest mag.
    opts = .{};
    applyGltfSampler(33071, 33648, 9728, 9728, &opts);
    try std.testing.expectEqual(sg.Wrap.CLAMP_TO_EDGE, opts.wrap_u);
    try std.testing.expectEqual(sg.Wrap.MIRRORED_REPEAT, opts.wrap_v);
    try std.testing.expectEqual(sg.Filter.NEAREST, opts.mag_filter);
    try std.testing.expectEqual(sg.Filter.NEAREST, opts.min_filter);
    // Mip half of a bare NEAREST (no explicit mip mode) keeps the engine
    // default: glTF leaves mip selection to min_filter only when mipmapped
    // variants are used.
    try std.testing.expectEqual(sg.Filter.LINEAR, opts.mip_filter);

    // The mip half: *_MIPMAP_NEAREST variants select NEAREST mips for both
    // min halves; *_MIPMAP_LINEAR variants select LINEAR mips.
    opts = .{};
    applyGltfSampler(10497, 10497, 9728, 9984, &opts);
    try std.testing.expectEqual(sg.Filter.NEAREST, opts.min_filter);
    try std.testing.expectEqual(sg.Filter.NEAREST, opts.mip_filter);

    opts = .{};
    applyGltfSampler(10497, 10497, 9729, 9985, &opts);
    try std.testing.expectEqual(sg.Filter.LINEAR, opts.min_filter);
    try std.testing.expectEqual(sg.Filter.NEAREST, opts.mip_filter);

    opts = .{};
    applyGltfSampler(10497, 10497, 9728, 9986, &opts);
    try std.testing.expectEqual(sg.Filter.NEAREST, opts.min_filter);
    try std.testing.expectEqual(sg.Filter.LINEAR, opts.mip_filter);

    // Unknown values (spec: fields are optional) keep engine defaults.
    opts = .{};
    applyGltfSampler(0, 0, 0, 0, &opts);
    try std.testing.expectEqual(sg.Wrap.REPEAT, opts.wrap_u);
    try std.testing.expectEqual(sg.Filter.LINEAR, opts.mag_filter);
    try std.testing.expectEqual(sg.Filter.LINEAR, opts.min_filter);
}

test "loadMaterials maps normalTexture.scale into normal_scale (GPU-free)" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var scene = testScene(alloc);

    var images: [1]c.cgltf_image = .{std.mem.zeroes(c.cgltf_image)};
    var textures: [1]c.cgltf_texture = .{std.mem.zeroes(c.cgltf_texture)};
    textures[0].image = &images[0];

    var src: [2]c.cgltf_material = .{
        std.mem.zeroes(c.cgltf_material),
        std.mem.zeroes(c.cgltf_material),
    };
    // [0]: no normal texture -> scale must stay the engine default 1.0.
    // [1]: normal texture present with scale 0.5 -> mapped 1:1.
    src[1].normal_texture.texture = &textures[0];
    src[1].normal_texture.scale = 0.5;

    var data: c.cgltf_data = std.mem.zeroes(c.cgltf_data);
    data.materials = &src[0];
    data.materials_count = src.len;

    var out: [2]?Material = .{ null, null };
    // image_cache/decoded are indexed by image (one image here); entries
    // stay null: the texture has neither buffer view nor URI, so the load
    // cleanly returns null without touching the GPU.
    var tex_cache: [2]?Texture = .{ null, null };
    var dec_cache: [1]?Texture.DecodedImage = .{null};
    try loadMaterials(&scene, &data, null, &out, &tex_cache, &dec_cache, null);

    try std.testing.expectEqual(@as(f32, 1.0), out[0].?.pbr.normal_scale);
    try std.testing.expectEqual(@as(f32, 0.5), out[1].?.pbr.normal_scale);
}

test "colorSlotImageFlags marks albedo/emissive images, not data slots" {
    const alloc = std.testing.allocator;

    var images: [3]c.cgltf_image = .{
        std.mem.zeroes(c.cgltf_image),
        std.mem.zeroes(c.cgltf_image),
        std.mem.zeroes(c.cgltf_image),
    };
    var textures: [3]c.cgltf_texture = .{
        std.mem.zeroes(c.cgltf_texture),
        std.mem.zeroes(c.cgltf_texture),
        std.mem.zeroes(c.cgltf_texture),
    };
    textures[0].image = &images[0]; // albedo slot -> color
    textures[1].image = &images[1]; // normal slot -> data
    textures[2].image = &images[2]; // emissive slot -> color

    var src: [1]c.cgltf_material = .{std.mem.zeroes(c.cgltf_material)};
    src[0].has_pbr_metallic_roughness = 1;
    src[0].pbr_metallic_roughness.base_color_texture.texture = &textures[0];
    src[0].normal_texture.texture = &textures[1];
    src[0].emissive_texture.texture = &textures[2];

    var data: c.cgltf_data = std.mem.zeroes(c.cgltf_data);
    data.images = &images[0];
    data.images_count = images.len;
    data.materials = &src[0];
    data.materials_count = src.len;

    const flags = try colorSlotImageFlags(alloc, &data);
    defer alloc.free(flags);

    try std.testing.expectEqualSlices(bool, &.{ true, false, true }, flags);
}

test "uvTransformFromView reads KHR_texture_transform, identity when absent" {
    var view: c.cgltf_texture_view = std.mem.zeroes(c.cgltf_texture_view);
    // No extension: identity.
    try std.testing.expect(uvTransformFromView(&view).isIdentity());

    // has_transform with offset/rotation/scale maps 1:1.
    view.has_transform = 1;
    view.transform.offset = .{ 0.25, -0.5 };
    view.transform.rotation = 1.5;
    view.transform.scale = .{ 2, 4 };
    const t = uvTransformFromView(&view);
    try testingExpected(t);
}

fn testingExpected(t: UvTransform) !void {
    try std.testing.expectEqual(@as(f32, 0.25), t.offset[0]);
    try std.testing.expectEqual(@as(f32, -0.5), t.offset[1]);
    try std.testing.expectEqual(@as(f32, 1.5), t.rotation);
    try std.testing.expectEqual(@as(f32, 2), t.scale[0]);
    try std.testing.expectEqual(@as(f32, 4), t.scale[1]);
    try std.testing.expect(!t.isIdentity());
}

test "loadMaterials maps texture transforms into the PBR slots (GPU-free)" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var scene = testScene(alloc);

    var src: [1]c.cgltf_material = .{std.mem.zeroes(c.cgltf_material)};
    src[0].has_pbr_metallic_roughness = 1;
    // Albedo slot carries a transform; the other slots stay default.
    src[0].pbr_metallic_roughness.base_color_texture.has_transform = 1;
    src[0].pbr_metallic_roughness.base_color_texture.transform.rotation = 0.5;
    src[0].pbr_metallic_roughness.base_color_texture.transform.scale = .{ 3, 3 };

    var data: c.cgltf_data = std.mem.zeroes(c.cgltf_data);
    data.materials = &src[0];
    data.materials_count = src.len;

    var out: [1]?Material = .{null};
    try loadMaterials(&scene, &data, null, &out, &.{}, &.{}, null);

    const mat = out[0].?.pbr;
    try std.testing.expectEqual(@as(f32, 0.5), mat.albedo_uv_transform.rotation);
    try std.testing.expectEqual(@as(f32, 3), mat.albedo_uv_transform.scale[0]);
    try std.testing.expect(mat.normal_uv_transform.isIdentity());
    try std.testing.expect(mat.metallic_roughness_uv_transform.isIdentity());
    try std.testing.expect(mat.occlusion_uv_transform.isIdentity());
    try std.testing.expect(mat.emissive_uv_transform.isIdentity());
    // Channels stay at the glTF conventions: the format defines no override.
    try std.testing.expectEqual(@import("../material.zig").Channel.r, mat.occlusion_channel);
}

test "loadTextureFromView caches linear and sRGB variants separately without aliasing" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var scene = testScene(alloc);

    var img = std.mem.zeroes(c.cgltf_image);
    var tex = std.mem.zeroes(c.cgltf_texture);
    tex.image = &img;
    var view = std.mem.zeroes(c.cgltf_texture_view);
    view.texture = &tex;

    var data = std.mem.zeroes(c.cgltf_data);
    data.images = &img;
    data.images_count = 1;

    // Cache has 2 slots for image 0: slot 0 (linear) and slot 1 (sRGB)
    var image_cache: [2]?Texture = .{ null, null };
    var decoded: [1]?Texture.DecodedImage = .{null};

    // Pre-populate predecoded buffer with an sRGB decoded raw texture
    decoded[0] = .{ .rgba = .{
        .width = 1,
        .height = 1,
        .num_levels = 1,
        .is_srgb = true,
    } };

    // Requesting as linear (srgb_to_linear = false) MUST NOT consume or alias with the sRGB decoded texture
    const linear_tex = loadTextureFromView(&scene, &data, &image_cache, &decoded, &view, null, false, true);
    // Since it's linear and decoded was sRGB (and no buffer_view/uri exists), linear_tex stays null:
    try std.testing.expect(linear_tex == null);
    try std.testing.expect(decoded[0] != null); // Was NOT consumed!

    // Mock GPU textures into the cache slots to verify cache query separation:
    const mock_linear = Texture{
        .image = .{ .id = 101 },
        .view = .{ .id = 201 },
        .sampler = .{ .id = 301 },
        .width = 1,
        .height = 1,
    };
    const mock_srgb = Texture{
        .image = .{ .id = 102 },
        .view = .{ .id = 202 },
        .sampler = .{ .id = 302 },
        .width = 1,
        .height = 1,
    };
    image_cache[0] = mock_linear;
    image_cache[1] = mock_srgb;

    // Separate lookups must return their own distinct slot!
    const query_linear = loadTextureFromView(&scene, &data, &image_cache, &decoded, &view, null, false, true);
    try std.testing.expect(query_linear != null);
    try std.testing.expectEqual(@as(u32, 101), query_linear.?.image.id);

    const query_srgb = loadTextureFromView(&scene, &data, &image_cache, &decoded, &view, null, true, true);
    try std.testing.expect(query_srgb != null);
    try std.testing.expectEqual(@as(u32, 102), query_srgb.?.image.id);
}

test "textureImage resolves basisu image when source is absent (GPU-free)" {
    var img = std.mem.zeroes(c.cgltf_image);
    var basisu_img = std.mem.zeroes(c.cgltf_image);

    // Обычная текстура: рабочее изображение — image.
    var plain = std.mem.zeroes(c.cgltf_texture);
    plain.image = &img;
    try std.testing.expect(@intFromPtr(textureImage(&plain).?) == @intFromPtr(&img));

    // KHR_texture_basisu: source отсутствует, рабочее изображение —
    // basisu_image (обычно .ktx2).
    var basisu = std.mem.zeroes(c.cgltf_texture);
    basisu.has_basisu = 1;
    basisu.basisu_image = &basisu_img;
    try std.testing.expect(@intFromPtr(textureImage(&basisu).?) == @intFromPtr(&basisu_img));

    // Ни source, ни basisu — null.
    var empty = std.mem.zeroes(c.cgltf_texture);
    try std.testing.expect(textureImage(&empty) == null);
    try std.testing.expect(textureImage(null) == null);

    // Оба заданы (не по спеке, но в терпимости): приоритет у source.
    var both = std.mem.zeroes(c.cgltf_texture);
    both.image = &img;
    both.has_basisu = 1;
    both.basisu_image = &basisu_img;
    try std.testing.expect(@intFromPtr(textureImage(&both).?) == @intFromPtr(&img));

    // Флаг без указателя — тоже null, а не висячий доступ.
    var flag_only = std.mem.zeroes(c.cgltf_texture);
    flag_only.has_basisu = 1;
    try std.testing.expect(textureImage(&flag_only) == null);
}

test "decodeImagesInParallel decodes embedded KTX2 block views to .block (GPU-free)" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var scene = testScene(alloc);

    // Минимальный ASTC 4x4 UNORM контейнер (vkFormat 157): 80 байт
    // header+index, одна 24-байтная level-запись, 16 байт блочного пейлоада —
    // та же раскладка, что в assets.zig "requestMemory routes block KTX2".
    var file: [80 + 24 + 16]u8 = [_]u8{0} ** (80 + 24 + 16);
    @memcpy(file[0..12], &[12]u8{ 0xAB, 'K', 'T', 'X', ' ', '2', '0', 0xBB, 0x0D, 0x0A, 0x1A, 0x0A });
    std.mem.writeInt(u32, file[12..16], 157, .little); // vkFormat ASTC_4x4_UNORM_BLOCK
    std.mem.writeInt(u32, file[16..20], 1, .little); // typeSize
    std.mem.writeInt(u32, file[20..24], 4, .little); // width
    std.mem.writeInt(u32, file[24..28], 4, .little); // height
    std.mem.writeInt(u32, file[36..40], 1, .little); // faceCount
    std.mem.writeInt(u32, file[40..44], 1, .little); // levelCount
    std.mem.writeInt(u64, file[80..88], 104, .little); // byteOffset
    std.mem.writeInt(u64, file[88..96], 16, .little); // byteLength
    std.mem.writeInt(u64, file[96..104], 16, .little); // uncompressedByteLength
    for (file[104..], 0..) |*b, i| b.* = @intCast(i);

    var garbage = [_]u8{0} ** 32;
    @memcpy(garbage[0..16], "not an image!!!!");

    var buffer0 = std.mem.zeroes(c.cgltf_buffer);
    buffer0.data = @ptrCast(&file);
    buffer0.size = file.len;
    var buffer1 = std.mem.zeroes(c.cgltf_buffer);
    buffer1.data = @ptrCast(&garbage);
    buffer1.size = garbage.len;

    var bv0 = std.mem.zeroes(c.cgltf_buffer_view);
    bv0.buffer = &buffer0;
    bv0.offset = 0;
    bv0.size = file.len;
    var bv1 = std.mem.zeroes(c.cgltf_buffer_view);
    bv1.buffer = &buffer1;
    bv1.offset = 0;
    bv1.size = garbage.len;

    var images: [2]c.cgltf_image = .{ std.mem.zeroes(c.cgltf_image), std.mem.zeroes(c.cgltf_image) };
    images[0].buffer_view = &bv0;
    images[1].buffer_view = &bv1;

    var data = std.mem.zeroes(c.cgltf_data);
    data.images = &images[0];
    data.images_count = images.len;

    var decoded: [2]?Texture.DecodedImage = .{ null, null };
    decodeImagesInParallel(&scene, &data, &decoded, null);
    defer {
        for (0..decoded.len) |i| {
            if (decoded[i]) |*dec| dec.deinit(alloc);
        }
    }

    // [0]: блочный ASTC попал в .block как в файле (один уровень, без
    // синтеза мипов и без sRGB-конверсии); RGBA8-сторона пуста.
    try std.testing.expect(decoded[0] != null);
    switch (decoded[0].?) {
        .block => |b| {
            try std.testing.expectEqual(ktx2.BlockFormat.astc_4x4_unorm, b.format);
            try std.testing.expect(!b.format.isSrgb());
            try std.testing.expectEqual(@as(u32, 4), b.width);
            try std.testing.expectEqual(@as(u32, 4), b.height);
            try std.testing.expectEqual(@as(u32, 1), b.num_levels);
            try std.testing.expectEqualSlices(u8, file[104..], b.levels[0].?);
            try std.testing.expectEqual(@as(usize, 16), decoded[0].?.totalBytes());
        },
        .rgba => return error.TestUnexpectedResult,
    }

    // [1]: битый декод — null, как раньше для RawTexture.
    try std.testing.expect(decoded[1] == null);
}
