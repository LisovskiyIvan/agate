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

        materials[i] = .{ .pbr = pbr_mat };
    }
}
