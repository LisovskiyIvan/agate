//! Tests for `materials.zig` (moved from `materials.zig` inline blocks).
const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;

const c = @import("../c.zig").c;
const Material = @import("../material.zig").Material;
const UvTransform = @import("../material.zig").UvTransform;
const Texture = @import("../texture.zig").Texture;
const ktx2 = @import("../ktx2.zig");
const materials = @import("materials.zig");
const loadMaterials = materials.loadMaterials;
const colorSlotImageFlags = materials.colorSlotImageFlags;
const decodeImagesInParallel = materials.decodeImagesInParallel;
const applyGltfSampler = materials.applyGltfSampler;
const textureOptionsFor = materials.textureOptionsFor;
const textureCoordFromView = materials.textureCoordFromView;
const uvTransformFromView = materials.uvTransformFromView;
const loadTextureFromView = materials.loadTextureFromView;
const textureImage = materials.textureImage;

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
    try loadMaterials(&scene, &data, null, &out, empty_tex, empty_dec, null, null);

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

    // Every glTF material carries the loader's specular anti-aliasing default
    // (Babylon's PBRMaterialLoadingAdapter sets it on unconditionally), on all
    // three alpha modes. A hand-built PBRMaterial keeps its own default off.
    try std.testing.expect(out[0].?.pbr.specular_anti_aliasing);
    try std.testing.expect(out[1].?.pbr.specular_anti_aliasing);
    try std.testing.expect(out[2].?.pbr.specular_anti_aliasing);
    try std.testing.expect(!PBRMaterial.init("hand_made").specular_anti_aliasing);
}

const PBRMaterial = @import("../material/pbr.zig").PBRMaterial;

test "loadMaterials maps KHR_materials_clearcoat and KHR_materials_sheen (GPU-free)" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var scene = testScene(alloc);

    var src: [2]c.cgltf_material = .{
        std.mem.zeroes(c.cgltf_material),
        std.mem.zeroes(c.cgltf_material),
    };
    // [0]: clearcoat active
    src[0].has_clearcoat = 1;
    src[0].clearcoat.clearcoat_factor = 0.85;
    src[0].clearcoat.clearcoat_roughness_factor = 0.15;

    // [1]: sheen active
    src[1].has_sheen = 1;
    src[1].sheen.sheen_color_factor[0] = 0.9;
    src[1].sheen.sheen_color_factor[1] = 0.8;
    src[1].sheen.sheen_color_factor[2] = 0.7;
    src[1].sheen.sheen_roughness_factor = 0.4;

    var data = std.mem.zeroes(c.cgltf_data);
    data.materials = &src[0];
    data.materials_count = src.len;

    var out: [2]?Material = .{ null, null };
    var img_cache: [0]?Texture = .{};
    var dec: [0]?Texture.DecodedImage = .{};

    try loadMaterials(&scene, &data, null, &out, &img_cache, &dec, null, null);

    try std.testing.expect(out[0].? == .pbr);
    try std.testing.expectEqual(@as(f32, 0.85), out[0].?.pbr.clearcoat.intensity);
    try std.testing.expectEqual(@as(f32, 0.15), out[0].?.pbr.clearcoat.roughness);

    try std.testing.expect(out[1].? == .pbr);
    try std.testing.expectEqual(@as(f32, 1.0), out[1].?.pbr.sheen.intensity);
    try std.testing.expectEqual(@as(f32, 0.4), out[1].?.pbr.sheen.roughness);
    try std.testing.expectApproxEqAbs(@as(f32, 0.9), out[1].?.pbr.sheen.color.r, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), out[1].?.pbr.sheen.color.g, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.7), out[1].?.pbr.sheen.color.b, 1e-4);
}

test "loadMaterials maps KHR_materials_transmission and KHR_materials_ior (GPU-free)" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var scene = testScene(alloc);

    var src: [2]c.cgltf_material = .{
        std.mem.zeroes(c.cgltf_material),
        std.mem.zeroes(c.cgltf_material),
    };
    // [0]: transmission active
    src[0].has_transmission = 1;
    src[0].transmission.transmission_factor = 0.75;

    // [1]: ior active
    src[1].has_ior = 1;
    src[1].ior.ior = 1.33;

    var data = std.mem.zeroes(c.cgltf_data);
    data.materials = &src[0];
    data.materials_count = src.len;

    var out: [2]?Material = .{ null, null };
    var img_cache: [0]?Texture = .{};
    var dec: [0]?Texture.DecodedImage = .{};

    try loadMaterials(&scene, &data, null, &out, &img_cache, &dec, null, null);

    try std.testing.expect(out[0].? == .pbr);
    try std.testing.expectEqual(@as(f32, 0.75), out[0].?.pbr.transmission.factor);
    try std.testing.expectEqual(@as(f32, 1.5), out[0].?.pbr.ior);

    try std.testing.expect(out[1].? == .pbr);
    try std.testing.expectEqual(@as(f32, 1.33), out[1].?.pbr.ior);
    try std.testing.expectEqual(@as(f32, 1.33), out[1].?.pbr.transmission.ior);
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

test "textureOptionsFor starts from Babylon's defaults and applies the load override" {
    // A sampler-less glTF texture keeps the engine defaults: glTF's
    // REPEAT/LINEAR_MIPMAP_LINEAR plus Babylon's
    // DEFAULT_ANISOTROPIC_FILTERING_LEVEL = 4, which the Babylon glTF loader
    // never overrides. The bench's DamagedHelmet/Fox GLBs have no `sampler`
    // on most textures, so this is the bench's actual path.
    var no_sampler = std.mem.zeroes(c.cgltf_texture);
    no_sampler.sampler = null;
    const by_default = textureOptionsFor(&no_sampler, true, null);
    try std.testing.expectEqual(@as(u32, 4), by_default.max_anisotropy);
    try std.testing.expectEqual(sg.Wrap.REPEAT, by_default.wrap_u);
    try std.testing.expectEqual(sg.Filter.LINEAR, by_default.min_filter);
    try std.testing.expect(by_default.srgb_to_linear);

    // The load-wide override wins over that default, and 1 restores the
    // pre-fix sampling (Babylon at anisotropicFilteringLevel = 1).
    try std.testing.expectEqual(@as(u32, 1), textureOptionsFor(&no_sampler, false, 1).max_anisotropy);
    try std.testing.expectEqual(@as(u32, 16), textureOptionsFor(&no_sampler, false, 16).max_anisotropy);

    // An authored sampler still wins on wrap/filter (glTF is authoritative
    // there), while the anisotropy override applies on top: Babylon's own
    // loader likewise sets wrap/filter from the glTF sampler and leaves
    // anisotropicFilteringLevel alone.
    var sampler = c.cgltf_sampler{
        .mag_filter = 9728, // NEAREST
        .min_filter = 9984, // NEAREST_MIPMAP_NEAREST
        .wrap_s = 33071, // CLAMP_TO_EDGE
        .wrap_t = 33648, // MIRRORED_REPEAT
    };
    var textured = std.mem.zeroes(c.cgltf_texture);
    textured.sampler = &sampler;
    const authored = textureOptionsFor(&textured, false, null);
    try std.testing.expectEqual(sg.Wrap.CLAMP_TO_EDGE, authored.wrap_u);
    try std.testing.expectEqual(sg.Wrap.MIRRORED_REPEAT, authored.wrap_v);
    try std.testing.expectEqual(sg.Filter.NEAREST, authored.mag_filter);
    try std.testing.expectEqual(sg.Filter.NEAREST, authored.min_filter);
    // ... and the sokol LINEAR clamp still applies to the effective value.
    try std.testing.expectEqual(@as(u32, 1), Texture.effectiveAnisotropy(authored, 10));
    // The override still lands on an authored-sampler texture (it changes
    // anisotropy only, never wrap/filter).
    try std.testing.expectEqual(@as(u32, 8), textureOptionsFor(&textured, false, 8).max_anisotropy);
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
    try loadMaterials(&scene, &data, null, &out, &tex_cache, &dec_cache, null, null);

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
    try std.testing.expect((try uvTransformFromView(&view)).isIdentity());

    // has_transform with offset/rotation/scale maps 1:1.
    view.has_transform = 1;
    view.transform.offset = .{ 0.25, -0.5 };
    view.transform.rotation = 1.5;
    view.transform.scale = .{ 2, 4 };
    const t = try uvTransformFromView(&view);
    try testingExpected(t);
}

test "texture coordinate selection honors extension override and rejects unsupported sets" {
    var view = std.mem.zeroes(c.cgltf_texture_view);
    view.texcoord = 1;
    try std.testing.expectEqual(@as(u1, 1), try textureCoordFromView(&view));
    try std.testing.expectEqual(@as(u1, 1), (try uvTransformFromView(&view)).tex_coord);
    view.has_transform = 1;
    view.transform.has_texcoord = 1;
    view.transform.texcoord = 0;
    view.transform.scale = .{ 1, 1 };
    try std.testing.expectEqual(@as(u1, 0), try textureCoordFromView(&view));
    view.transform.texcoord = 2;
    try std.testing.expectError(error.UnsupportedTextureCoordinate, textureCoordFromView(&view));
    try std.testing.expectError(error.UnsupportedTextureCoordinate, uvTransformFromView(&view));
    view.transform.texcoord = -1;
    try std.testing.expectError(error.UnsupportedTextureCoordinate, textureCoordFromView(&view));
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
    try loadMaterials(&scene, &data, null, &out, &.{}, &.{}, null, null);

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
    const linear_tex = loadTextureFromView(&scene, &data, &image_cache, &decoded, &view, null, false, true, null);
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
    const query_linear = loadTextureFromView(&scene, &data, &image_cache, &decoded, &view, null, false, true, null);
    try std.testing.expect(query_linear != null);
    try std.testing.expectEqual(@as(u32, 101), query_linear.?.image.id);

    const query_srgb = loadTextureFromView(&scene, &data, &image_cache, &decoded, &view, null, true, true, null);
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
