const std = @import("std");
const builtin = @import("builtin");
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
    /// Backend-accurate Basis transcode target, snapshotted per load on the
    /// calling thread (null = desktop-first .bc7 default). Workers never
    /// touch sg themselves.
    basis_target: ?ktx2.BasisTarget = null,
    out: ?Texture.DecodedImage = null,

    fn run(self: *DecodeJob) void {
        const opts: Texture.DecodeOptions = .{
            .gen_mipmaps = true,
            .srgb_to_linear = self.srgb,
            .slot = if (self.srgb) .color else .data,
            .color_space = if (self.srgb) .srgb else .linear,
            .basis_target = self.basis_target,
        };
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
        if (mat.has_sheen != 0) {
            markColorSlotImage(gltf, mat.sheen.sheen_color_texture, flags);
        }
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

    // One backend snapshot for every worker: Basis decodes transcode to BC7
    // on desktop, ASTC when supported, ETC2 when ASTC is unavailable, and
    // RGBA32 where no compressed target samples. Off-context (no sg) leaves null — the
    // desktop-first .bc7 default applies and fromRawBlock gates explicitly.
    const basis_target: ?ktx2.BasisTarget = if (sg.isvalid()) ktx2.preferredBasisTarget(Texture.queryBlockSupport()) else null;

    var job_count: usize = 0;
    for (0..gltf.images_count) |i| {
        const img = &gltf.images[i];
        var job = DecodeJob{ .allocator = scene.allocator, .image_index = i, .basis_target = basis_target };
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
    const cpu_count = if (builtin.single_threaded or builtin.cpu.arch.isWasm()) 1 else std.Thread.getCpuCount() catch 4;
    const max_threads = 8;
    const worker_count = @min(job_count, @min(cpu_count, max_threads));
    if (comptime !builtin.single_threaded and !builtin.cpu.arch.isWasm()) {
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

/// Base `Texture.Options` for one glTF texture: the engine defaults (which
/// since the anisotropy fix mirror Babylon's — wrap REPEAT, LINEAR
/// min/mag/mip, `max_anisotropy` = Babylon's
/// DEFAULT_ANISOTROPIC_FILTERING_LEVEL = 4), then the view's own glTF
/// sampler on top (a sampler-less texture keeps the defaults, exactly like
/// Babylon's loader), then the load-wide `max_anisotropy` override, then the
/// caller's per-slot sRGB decision. Shared by the sync and async texture
/// paths so both produce identical samplers.
pub fn textureOptionsFor(tex: [*c]const c.cgltf_texture, srgb: bool, max_anisotropy: ?u32) Texture.Options {
    var opts: Texture.Options = .{};
    if (tex.*.sampler) |smp| {
        applyGltfSampler(@intCast(smp.*.wrap_s), @intCast(smp.*.wrap_t), @intCast(smp.*.mag_filter), @intCast(smp.*.min_filter), &opts);
    }
    if (max_anisotropy) |a| opts.max_anisotropy = a;
    opts.srgb_to_linear = srgb;
    opts.slot = if (srgb) .color else .data;
    opts.color_space = if (srgb) .srgb else .linear;
    return opts;
}

/// KHR_texture_transform.texCoord overrides the base view's texCoord.
/// Unsupported coordinate sets fail the import rather than sampling UV0.
pub fn textureCoordFromView(view: anytype) error{UnsupportedTextureCoordinate}!u1 {
    const coord = if (view.*.has_transform != 0 and view.*.transform.has_texcoord != 0)
        view.*.transform.texcoord
    else
        view.*.texcoord;
    if (coord < 0 or coord > 1) return error.UnsupportedTextureCoordinate;
    return @intCast(coord);
}

/// Extracts the UV selector and KHR_texture_transform map. loadMaterials
/// validates coordinate sets before creating any materials or GPU objects.
/// `anytype` accepts both normal and C (allowzero) pointers to
/// cgltf_texture_view (cgltf's own structs carry C-pointer parents).
pub fn uvTransformFromView(view: anytype) error{UnsupportedTextureCoordinate}!UvTransform {
    const coord = try textureCoordFromView(view);
    if (view.*.has_transform == 0) return .{ .tex_coord = coord };
    const t = view.*.transform;
    return .{
        .offset = .{ t.offset[0], t.offset[1] },
        .rotation = t.rotation,
        .scale = .{ t.scale[0], t.scale[1] },
        .tex_coord = coord,
    };
}

pub fn textureViewsForMaterial(mat: anytype) [7][*c]const c.cgltf_texture_view {
    return .{
        &mat.*.pbr_metallic_roughness.base_color_texture,
        &mat.*.pbr_metallic_roughness.metallic_roughness_texture,
        &mat.*.normal_texture,
        &mat.*.occlusion_texture,
        &mat.*.emissive_texture,
        &mat.*.clearcoat.clearcoat_texture,
        &mat.*.sheen.sheen_color_texture,
    };
}

pub fn validateTextureCoordinates(gltf: *const c.cgltf_data) !void {
    for (0..gltf.materials_count) |i| {
        for (textureViewsForMaterial(&gltf.materials[i])) |view| {
            if (view.*.texture != null) _ = try textureCoordFromView(view);
        }
    }
}

/// Stage 2: schedules background decodes for images the sync path could
/// not serve, registering material slots as patch targets. One decode per
/// image+srgb pair (`seen`), even when several materials share it.
/// Embedded buffer-view bytes are copied at registration time — the cgltf
/// data dies when appendGlb returns, long before the worker decodes.
/// Registration runs on the calling (game) thread and is safe against a
/// concurrent context-side drain: `addTarget` serializes the list, and the
/// drain never writes the material slots itself — the game-side commit
/// (`UploadQueue.commitUploadedTargets`, every producer build) patches
/// them, so the slots stay game-owned end to end.
pub const AsyncTexCtx = struct {
    scene: *Scene,
    gltf: *c.cgltf_data,
    base_dir: ?[]const u8,
    queue: *assets.UploadQueue,
    /// Backend-accurate Basis target, snapshotted at init (same rule as the
    /// sync parallel decoder: null off-context). Rides decode_opts so the
    /// queue workers transcode to an uploadable target.
    basis_target: ?ktx2.BasisTarget = null,
    /// Load-wide sampler anisotropy override (SceneLoader.LoadOptions);
    /// null keeps Texture.Options' default (Babylon's 4). Rides every
    /// texture this load registers.
    max_anisotropy: ?u32 = null,
    /// image_index * 2 + srgb -> in-flight request
    seen: std.AutoHashMapUnmanaged(usize, *assets.PendingTexture) = .empty,

    pub fn init(scene: *Scene, gltf: *c.cgltf_data, base_dir: ?[]const u8, queue: *assets.UploadQueue) AsyncTexCtx {
        return .{
            .scene = scene,
            .gltf = gltf,
            .base_dir = base_dir,
            .queue = queue,
            .basis_target = if (sg.isvalid()) ktx2.preferredBasisTarget(Texture.queryBlockSupport()) else null,
        };
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

        const tex_options = textureOptionsFor(tex, srgb, self.max_anisotropy);
        const decode_opts: Texture.DecodeOptions = .{
            .gen_mipmaps = true,
            .srgb_to_linear = srgb,
            .slot = if (srgb) .color else .data,
            .color_space = if (srgb) .srgb else .linear,
            .basis_target = self.basis_target,
        };

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
    max_anisotropy: ?u32,
) bool {
    if (loadTextureFromView(scene, gltf, image_cache, decoded, view, base_dir, srgb_to_linear, actx == null, max_anisotropy)) |t| {
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
    max_anisotropy: ?u32,
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

    const tex_options = textureOptionsFor(tex, srgb_to_linear, max_anisotropy);

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
        .srgb_to_linear = tex_options.isColorSlot(),
        .color_space = tex_options.color_space,
        .slot = tex_options.slot,
        // Context thread here (asserted by the caller): snapshot the backend
        // so Basis payloads transcode to an uploadable target.
        .basis_target = Texture.basisTargetForCurrentThread(),
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
        .srgb_to_linear = tex_options.isColorSlot(),
        .color_space = tex_options.color_space,
        .slot = tex_options.slot,
        .basis_target = Texture.basisTargetForCurrentThread(),
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
    max_anisotropy: ?u32,
) !void {
    // Validate the entire load before publishing partial material state.
    try validateTextureCoordinates(gltf);
    for (0..gltf.materials_count) |i| {
        const src_mat = &gltf.materials[i];
        const mat_name = if (src_mat.name != null)
            std.mem.span(src_mat.name)
        else
            "glb_material";

        const pbr_mat = try scene.createPBRMaterial(mat_name);

        // glTF material default (Babylon's loading adapters). Babylon turns
        // specular anti-aliasing ON for every material the loader creates —
        // `babylonjs.loaders.js`, `PBRMaterialLoadingAdapter` constructor:
        //     class PBRMaterialLoadingAdapter {
        //         constructor(material) {
        //             this._specWorkflow = false;
        //             this._material = material;
        //             this._material.enableSpecularAntiAliasing = true;
        //         }
        // (byte 372234, reached from `_getOrCreateMaterialAdapter` at byte
        // 304156 → `_createDefaultMaterial` at byte 338350). A Babylon
        // PBRMaterial built by hand keeps its own default of `false`, so this
        // is a LOADER default, not a material one: without it a glTF helmet
        // and a hand-made ground plane in the same scene would shade
        // differently in Babylon while shading identically here.
        pbr_mat.specular_anti_aliasing = true;

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
            _ = loadTextureSlot(scene, gltf, image_cache, decoded, &pbr.base_color_texture, base_dir, true, &pbr_mat.albedo_texture, actx, max_anisotropy);
            pbr_mat.albedo_uv_transform = try uvTransformFromView(&pbr.base_color_texture);
            _ = loadTextureSlot(scene, gltf, image_cache, decoded, &pbr.metallic_roughness_texture, base_dir, false, &pbr_mat.metallic_roughness_texture, actx, max_anisotropy);
            pbr_mat.metallic_roughness_uv_transform = try uvTransformFromView(&pbr.metallic_roughness_texture);
        }

        _ = loadTextureSlot(scene, gltf, image_cache, decoded, &src_mat.normal_texture, base_dir, false, &pbr_mat.normal_texture, actx, max_anisotropy);
        if (src_mat.normal_texture.texture != null) {
            // cgltf defaults texture-view scale to 1.0 (cgltf.h parse).
            pbr_mat.normal_scale = src_mat.normal_texture.scale;
        }
        pbr_mat.normal_uv_transform = try uvTransformFromView(&src_mat.normal_texture);
        _ = loadTextureSlot(scene, gltf, image_cache, decoded, &src_mat.occlusion_texture, base_dir, false, &pbr_mat.occlusion_texture, actx, max_anisotropy);
        pbr_mat.occlusion_strength = src_mat.occlusion_texture.scale;
        pbr_mat.occlusion_uv_transform = try uvTransformFromView(&src_mat.occlusion_texture);

        const emissive_textured = loadTextureSlot(scene, gltf, image_cache, decoded, &src_mat.emissive_texture, base_dir, true, &pbr_mat.emissive_texture, actx, max_anisotropy);
        pbr_mat.emissive_uv_transform = try uvTransformFromView(&src_mat.emissive_texture);
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

        // KHR_materials_clearcoat
        if (src_mat.has_clearcoat != 0) {
            const cc = &src_mat.clearcoat;
            const cc_textured = loadTextureSlot(scene, gltf, image_cache, decoded, &cc.clearcoat_texture, base_dir, false, &pbr_mat.clearcoat.mask_texture, actx, max_anisotropy);
            pbr_mat.clearcoat.uv_transform = try uvTransformFromView(&cc.clearcoat_texture);
            pbr_mat.clearcoat.roughness = cc.clearcoat_roughness_factor;
            if ((pbr_mat.clearcoat.mask_texture != null or cc_textured) and cc.clearcoat_factor == 0.0) {
                pbr_mat.clearcoat.intensity = 1.0;
            } else {
                pbr_mat.clearcoat.intensity = cc.clearcoat_factor;
            }
        }

        // KHR_materials_sheen
        if (src_mat.has_sheen != 0) {
            const sh = &src_mat.sheen;
            pbr_mat.sheen.roughness = sh.sheen_roughness_factor;
            const sh_textured = loadTextureSlot(scene, gltf, image_cache, decoded, &sh.sheen_color_texture, base_dir, true, &pbr_mat.sheen.color_texture, actx, max_anisotropy);
            pbr_mat.sheen.uv_transform = try uvTransformFromView(&sh.sheen_color_texture);
            if ((pbr_mat.sheen.color_texture != null or sh_textured) and
                sh.sheen_color_factor[0] == 0.0 and
                sh.sheen_color_factor[1] == 0.0 and
                sh.sheen_color_factor[2] == 0.0)
            {
                pbr_mat.sheen.color = Color3.white;
                pbr_mat.sheen.intensity = 1.0;
            } else {
                pbr_mat.sheen.color = Color3.new(
                    sh.sheen_color_factor[0],
                    sh.sheen_color_factor[1],
                    sh.sheen_color_factor[2],
                );
                const has_color = (sh.sheen_color_factor[0] > 0.0 or sh.sheen_color_factor[1] > 0.0 or sh.sheen_color_factor[2] > 0.0);
                pbr_mat.sheen.intensity = if (has_color or pbr_mat.sheen.color_texture != null or sh_textured) 1.0 else 0.0;
            }
        }

        // KHR_materials_transmission
        if (src_mat.has_transmission != 0) {
            pbr_mat.transmission.factor = src_mat.transmission.transmission_factor;
        }

        // KHR_materials_ior
        if (src_mat.has_ior != 0) {
            pbr_mat.ior = src_mat.ior.ior;
            pbr_mat.transmission.ior = src_mat.ior.ior;
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

test {
    _ = @import("materials_tests.zig");
}
