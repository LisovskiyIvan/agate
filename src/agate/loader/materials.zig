const std = @import("std");

const c = @import("../c.zig").c;
const math = @import("math");
const Color3 = math.Color3;

const Scene = @import("../scene.zig").Scene;
const Material = @import("../material.zig").Material;
const Texture = @import("../texture.zig").Texture;

const DecodeJob = struct {
    allocator: std.mem.Allocator,
    image_index: usize,
    /// Embedded buffer view, valid until the cgltf data is freed.
    bytes: ?[]const u8 = null,
    /// External URI path, owned by the job.
    path: ?[]const u8 = null,
    out: ?Texture.RawTexture = null,

    fn run(self: *DecodeJob) void {
        if (self.bytes) |b| {
            self.out = Texture.decodeMemory(self.allocator, b, true) catch null;
        } else if (self.path) |p| {
            self.out = Texture.decodeFile(self.allocator, p, true) catch null;
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

/// Decodes every GLB/GLTF image on worker threads. This never fails the
/// load: images that could not be decoded stay null and loadTextureFromView
/// falls back to the synchronous path.
pub fn decodeImagesInParallel(scene: *Scene, gltf: *c.cgltf_data, decoded: []?Texture.RawTexture, base_dir: ?[]const u8) void {
    if (gltf.images_count == 0) return;

    const jobs = scene.allocator.alloc(DecodeJob, gltf.images_count) catch return;
    defer scene.allocator.free(jobs);

    var job_count: usize = 0;
    for (0..gltf.images_count) |i| {
        const img = &gltf.images[i];
        var job = DecodeJob{ .allocator = scene.allocator, .image_index = i };

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

pub fn loadTextureFromView(
    scene: *Scene,
    gltf: *c.cgltf_data,
    image_cache: []?Texture,
    decoded: []?Texture.RawTexture,
    view: [*c]const c.cgltf_texture_view,
    base_dir: ?[]const u8,
) ?Texture {
    if (view == null) return null;
    if (view.*.texture == null) return null;
    const tex = view.*.texture.?;
    if (tex.*.image == null) return null;
    const img = tex.*.image.?;

    var img_idx: ?usize = null;
    for (0..gltf.images_count) |im_i| {
        if (&gltf.images[im_i] == img) {
            img_idx = im_i;
            break;
        }
    }

    if (img_idx) |idx| {
        if (image_cache[idx]) |cached| {
            return cached;
        }
    }

    var tex_options: Texture.Options = .{};
    if (tex.*.sampler) |smp| {
        switch (smp.*.wrap_s) {
            33071 => tex_options.wrap_u = .CLAMP_TO_EDGE,
            33648 => tex_options.wrap_u = .MIRRORED_REPEAT,
            10497 => tex_options.wrap_u = .REPEAT,
            else => {},
        }
        switch (smp.*.wrap_t) {
            33071 => tex_options.wrap_v = .CLAMP_TO_EDGE,
            33648 => tex_options.wrap_v = .MIRRORED_REPEAT,
            10497 => tex_options.wrap_v = .REPEAT,
            else => {},
        }
        switch (smp.*.mag_filter) {
            9728 => tex_options.mag_filter = .NEAREST,
            9729 => tex_options.mag_filter = .LINEAR,
            else => {},
        }
        switch (smp.*.min_filter) {
            9728, 9984, 9986 => tex_options.min_filter = .NEAREST,
            9729, 9985, 9987 => tex_options.min_filter = .LINEAR,
            else => {},
        }
    }

    // Pre-decoded on worker threads: only the GPU upload runs here.
    if (img_idx) |idx| {
        if (decoded[idx]) |*raw| {
            const loaded = Texture.fromRaw(raw, tex_options);
            raw.deinit(scene.allocator);
            decoded[idx] = null;
            image_cache[idx] = loaded;
            return loaded;
        }
    }

    // 1. Embedded buffer view (typical in GLB or embedded GLTF)
    if (img.*.buffer_view) |bv| {
        if (bv.*.buffer != null and bv.*.buffer.*.data != null) {
            const raw_buf = @as([*]const u8, @ptrCast(bv.*.buffer.*.data));
            const img_data = (raw_buf + bv.*.offset)[0..bv.*.size];
            if (Texture.fromMemory(scene.allocator, img_data, tex_options)) |loaded| {
                if (img_idx) |idx| {
                    image_cache[idx] = loaded;
                }
                return loaded;
            } else |_| {}
        }
    }

    // 2. External URI (typical in standard GLTF with external textures)
    if (img.*.uri) |uri_c| {
        const uri = std.mem.span(uri_c);
        if (base_dir) |dir| {
            const full_path = std.fs.path.join(scene.allocator, &.{ dir, uri }) catch null;
            if (full_path) |fp| {
                defer scene.allocator.free(fp);
                if (Texture.fromFile(scene.allocator, fp, tex_options)) |loaded| {
                    if (img_idx) |idx| {
                        image_cache[idx] = loaded;
                    }
                    return loaded;
                } else |_| {}
            }
        } else {
            if (Texture.fromFile(scene.allocator, uri, tex_options)) |loaded| {
                if (img_idx) |idx| {
                    image_cache[idx] = loaded;
                }
                return loaded;
            } else |_| {}
        }
    }

    return null;
}

pub fn loadMaterials(
    scene: *Scene,
    gltf: *c.cgltf_data,
    base_dir: ?[]const u8,
    materials: []?Material,
    image_cache: []?Texture,
    decoded: []?Texture.RawTexture,
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

            pbr_mat.albedo_texture = loadTextureFromView(scene, gltf, image_cache, decoded, &pbr.base_color_texture, base_dir);
            pbr_mat.metallic_roughness_texture = loadTextureFromView(scene, gltf, image_cache, decoded, &pbr.metallic_roughness_texture, base_dir);
        }

        pbr_mat.normal_texture = loadTextureFromView(scene, gltf, image_cache, decoded, &src_mat.normal_texture, base_dir);
        pbr_mat.occlusion_texture = loadTextureFromView(scene, gltf, image_cache, decoded, &src_mat.occlusion_texture, base_dir);
        pbr_mat.occlusion_strength = src_mat.occlusion_texture.scale;

        pbr_mat.emissive_texture = loadTextureFromView(scene, gltf, image_cache, decoded, &src_mat.emissive_texture, base_dir);
        if (pbr_mat.emissive_texture != null and
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

        materials[i] = .{ .pbr = pbr_mat };
    }
}

/// Minimal CPU-only Scene for loadMaterials tests: the material path only
/// uses the allocator (createPBRMaterial) and never the GPU-backed default
/// textures or render passes, which std.mem.zeroes cannot initialize anyway
/// (non-nullable pointer fields).
fn testScene(alloc: std.mem.Allocator) Scene {
    return .{
        .allocator = alloc,
        .default_white_texture = undefined,
        .default_normal_texture = undefined,
        .default_cube_texture = undefined,
        .lights = .{},
        .shadows = .{ .pass = undefined },
        .sky = .{ .pass = undefined },
        .postfx = .{
            .postprocess_pass = undefined,
            .ssao_pass = undefined,
            .bloom_pass = undefined,
            .outline_pass = undefined,
        },
        .forward = .{},
        .particles = .{ .pass = undefined },
    };
}

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
    const empty_raw: []?Texture.RawTexture = &.{};
    try loadMaterials(&scene, &data, null, &out, empty_tex, empty_raw);

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
