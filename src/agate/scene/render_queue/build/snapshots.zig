//! Queue snapshot durability (P4) integration tests: queued records must
//! survive source mutation, storage growth, and OOM without dangling or live
//! refs, and shader/skin snapshots must stay exact across the serial and
//! parallel paths. Test-only leaf: owns `expectP4QueueRefsValid` plus the P4
//! test blocks, exercising `frame.buildFrameQueues` (one-directional edge
//! `snapshots` → `frame`; never the `build.zig` facade). No production
//! logic lives here, so the facade does not re-export anything from it.
const std = @import("std");

const math = @import("math");
const Mat4 = math.Mat4;
const Frustum = math.Frustum;
const Vec3 = math.Vec3;
const BoundingBox = math.BoundingBox;
const Mesh = @import("../../../mesh.zig").Mesh;
const InstancedMesh = @import("../../../mesh.zig").InstancedMesh;
const material_mod = @import("../../../material.zig");
const Material = material_mod.Material;
const StandardMaterial = material_mod.StandardMaterial;
const Texture = @import("../../../texture.zig").Texture;
const skeleton_mod = @import("../../../animation/skeleton.zig");
const visibility = @import("../../../visibility/mod.zig");
const jobs = @import("../../../jobs.zig");
const stats_mod = @import("../../stats.zig");
const SceneStats = stats_mod.SceneStats;
const instance_staging = @import("../../instance_staging.zig");
const items = @import("../items.zig");
const RenderQueues = items.RenderQueues;
const RenderMeshItem = items.RenderMeshItem;
const RenderInstancedBatch = items.RenderInstancedBatch;
const CulledMesh = items.CulledMesh;
const TransparentKind = items.TransparentKind;
const TransparentDrawEntry = items.TransparentDrawEntry;
const sortTransparentDrawOrder = items.sortTransparentDrawOrder;
const MAX_BONES = items.MAX_BONES;
const cull = @import("../cull.zig");
const FrameCullContext = cull.FrameCullContext;
const worldMatrixCached = cull.worldMatrixCached;
const appendRenderItem = cull.appendRenderItem;
const cullNonInstancedMesh = cull.cullNonInstancedMesh;
const instances = @import("../instances.zig");
const submitInstancedMesh = instances.submitInstancedMesh;
const frame = @import("frame.zig");
const buildFrameQueues = frame.buildFrameQueues;

test "material alpha modes route regular and instanced draws with frozen cutoffs" {
    const ally = std.testing.allocator;
    var standard = StandardMaterial.init("standard");
    var pbr = material_mod.PBRMaterial.init("pbr");
    var shader = material_mod.ShaderMaterial.init("shader");
    const materials = [_]Material{
        .{ .standard = &standard },
        .{ .pbr = &pbr },
        .{ .shader_material = &shader },
    };

    // Exercise the queue builder, not just Material.isTransparent(). This
    // is CPU routing/snapshot proof, not a claim about shader output pixels.
    for (materials) |mat| {
        for ([_]material_mod.AlphaMode{ .@"opaque", .cutout, .blend }) |mode| {
            switch (mat) {
                inline else => |m| {
                    m.alpha_mode = mode;
                    m.alpha_cutoff = 0.37;
                },
            }
            var regular = Mesh{
                .name = "regular",
                .vertex_buffer = .{},
                .index_buffer = .{},
                .index_count = 3,
                .material = mat,
                .culling_strategy = .always_render,
                .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
            };
            var instance = InstancedMesh{ .name = "copy", .source_mesh = &regular };
            var instance_ptrs = [_]*InstancedMesh{&instance};
            var batched = Mesh{
                .name = "batched",
                .vertex_buffer = .{},
                .index_buffer = .{},
                .index_count = 3,
                .material = mat,
                .instances = .{ .items = &instance_ptrs, .capacity = 1 },
            };
            const meshes = [_]*Mesh{ &regular, &batched };
            var queues = RenderQueues{};
            defer queues.deinit(ally);
            var stats = SceneStats{};
            var culler = visibility.OcclusionCuller.init();
            buildFrameQueues(.{
                .allocator = ally,
                .meshes = &meshes,
                .cache_key = 1,
                .view_proj = Mat4.identity,
                .eye = Vec3.zero,
                .cull_frustum = false,
                .cull_occlusion = false,
                .occlusion_culler = &culler,
                .stats = &stats,
                .queues = &queues,
                .default_white_id = 1,
            });

            const blend = mode == .blend;
            try std.testing.expectEqual(@as(usize, if (blend) 0 else 1), queues.items.items.len);
            try std.testing.expectEqual(@as(usize, if (blend) 1 else 0), queues.transparent.items.len);
            try std.testing.expectEqual(@as(usize, if (blend) 0 else 1), queues.opaque_instanced.items.len);
            try std.testing.expectEqual(@as(usize, if (blend) 1 else 0), queues.transparent_instanced.items.len);
            try std.testing.expectEqual(@as(usize, if (blend) 2 else 0), queues.transparent_order.items.len);

            // Source mutation must not change the records already queued.
            switch (mat) {
                inline else => |m| {
                    m.alpha_mode = if (blend) .@"opaque" else .blend;
                    m.alpha_cutoff = 0.91;
                },
            }
            const item = if (blend) queues.transparent.items[0] else queues.items.items[0];
            const batch = if (blend) queues.transparent_instanced.items[0] else queues.opaque_instanced.items[0];
            const cutoff: f32 = if (mode == .cutout) 0.37 else 0;
            try std.testing.expectEqual(@as(u32, 0), item.mesh_index);
            try std.testing.expectEqual(@as(u32, 1), batch.source_mesh);
            try std.testing.expectEqual(cutoff, item.draw_record.alpha_cutoff);
            try std.testing.expectEqual(cutoff, batch.draw_record.alpha_cutoff);
            try std.testing.expectEqual(blend, batch.transparent);
        }
    }
}

// Очереди не хранят живых указателей: скриншот пережил мутацию TRS/
// материала и две публикации скелета, перезаписавшие исходный слот.
test "P4: queued snapshot survives source mutation and skeleton republication" {
    const ally = std.testing.allocator;
    const Skeleton = skeleton_mod.Skeleton;

    var std_mat = material_mod.StandardMaterial.init("snap");
    std_mat.diffuse_color = math.Color3.new(0.2, 0.4, 0.6);
    const mat: Material = .{ .standard = &std_mat };

    var blend_mat = material_mod.StandardMaterial.init("snap_blend");
    blend_mat.alpha_mode = .blend;
    blend_mat.diffuse_color = math.Color3.new(0.1, 0.2, 0.3);
    const blend: Material = .{ .standard = &blend_mat };

    const skel = try Skeleton.init(ally, 1);
    defer skel.deinit();
    skel.bones[0].local_position = Vec3.new(1, 0, 0);
    skel.update();

    const unit_box = BoundingBox.init(Vec3.new(-0.5, -0.5, -0.5), Vec3.new(0.5, 0.5, 0.5));
    var opaque_mesh = Mesh{
        .name = "skinned_opaque",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .position = Vec3.new(10, 0, 0),
        .local_bounding_box = unit_box,
        .culling_strategy = .always_render,
        .material = mat,
        .skeleton = skel,
    };
    var trans_mesh = Mesh{
        .name = "skinned_trans",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .position = Vec3.new(0, 0, 10),
        .local_bounding_box = unit_box,
        .culling_strategy = .always_render,
        .material = blend,
        .skeleton = skel,
    };

    var queues = RenderQueues{};
    defer queues.deinit(ally);
    var stats = SceneStats{};
    var culler = visibility.OcclusionCuller.init();
    const meshes = [_]*Mesh{ &opaque_mesh, &trans_mesh };
    buildFrameQueues(.{
        .allocator = ally,
        .meshes = &meshes,
        .cache_key = 7,
        .view_proj = Mat4.identity,
        .eye = Vec3.zero,
        .cull_frustum = false,
        .cull_occlusion = false,
        .occlusion_culler = &culler,
        .stats = &stats,
        .queues = &queues,
        .default_white_id = 1,
    });

    try std.testing.expectEqual(@as(usize, 1), queues.items.items.len);
    try std.testing.expectEqual(@as(usize, 1), queues.transparent.items.len);
    try std.testing.expectEqual(@as(usize, 2), queues.skin_storage.items.len);

    const oq = queues.items.items[0];
    const tq = queues.transparent.items[0];
    try std.testing.expect(oq.skin_index != null and tq.skin_index != null);

    // Мутация источников: TRS, материал, две публикации скелета (вторая
    // перезаписывает исходный слот новым значением x=5).
    opaque_mesh.position = Vec3.new(99, 99, 99);
    trans_mesh.position = Vec3.new(99, 99, 99);
    std_mat.diffuse_color = math.Color3.new(9, 9, 9);
    blend_mat.diffuse_color = math.Color3.new(9, 9, 9);
    skel.bones[0].local_position = Vec3.new(5, 0, 0);
    skel.update();
    skel.update();
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), skel.getRenderSkinMatrices()[0].m[12], 1e-4);

    // Снимки неизменны: модель, draw_record и копии скинов.
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), queues.items.items[0].model.m[12], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.2), queues.items.items[0].draw_record.base_color[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.3), queues.transparent.items[0].draw_record.base_color[2], 1e-6);
    for ([_]RenderMeshItem{ queues.items.items[0], queues.transparent.items[0] }) |it| {
        const bones = queues.skin_storage.items[it.skin_index.?];
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), bones[0].m[12], 1e-4);
    }
}

// Instanced-запись тоже snapshot: мутация материала/TRS после prepare
// не меняет draw_record батча.
test "P4: instanced batch record survives source mutation" {
    const ally = std.testing.allocator;

    var inst_mat = material_mod.StandardMaterial.init("inst_snap");
    inst_mat.diffuse_color = math.Color3.new(0.5, 0.25, 0.125);
    const unit_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1));
    var src = Mesh{
        .name = "inst_src",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = unit_box,
    };
    var mesh = Mesh{
        .name = "inst_parent",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 36,
        .index_type = .UINT16,
        .material = .{ .standard = &inst_mat },
    };
    var inst0 = InstancedMesh{ .name = "i0", .source_mesh = &src, .position = Vec3.new(0, 0, 10) };
    var ptrs = [_]*InstancedMesh{&inst0};
    mesh.instances = std.ArrayListUnmanaged(*InstancedMesh){ .items = &ptrs, .capacity = 1 };

    var queues = RenderQueues{};
    defer queues.deinit(ally);
    var stats = SceneStats{};
    var culler = visibility.OcclusionCuller.init();
    const meshes = [_]*Mesh{&mesh};
    buildFrameQueues(.{
        .allocator = ally,
        .meshes = &meshes,
        .cache_key = 3,
        .view_proj = Mat4.identity,
        .eye = Vec3.zero,
        .cull_frustum = false,
        .cull_occlusion = false,
        .occlusion_culler = &culler,
        .stats = &stats,
        .queues = &queues,
        .default_white_id = 1,
    });
    try std.testing.expectEqual(@as(usize, 1), queues.opaque_instanced.items.len);

    inst_mat.diffuse_color = math.Color3.new(9, 9, 9);
    mesh.position = Vec3.new(99, 0, 0);
    const batch = queues.opaque_instanced.items[0];
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), batch.draw_record.base_color[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.125), batch.draw_record.base_color[2], 1e-6);
}

// Копии скинов масштабируются числом skinned-draws (а не MAX_BONES на item),
// индексы стабильны при росте хранилища и независимы между видами.
test "P4: skin storage scales with skinned draws and stays stable across growth" {
    const ally = std.testing.allocator;
    const Skeleton = skeleton_mod.Skeleton;
    const count = 160;

    const skels = try ally.alloc(*Skeleton, count);
    defer ally.free(skels);
    const meshes_arr = try ally.alloc(Mesh, count);
    defer ally.free(meshes_arr);
    const ptrs = try ally.alloc(*Mesh, count);
    defer ally.free(ptrs);
    for (0..count) |i| {
        const sk = try Skeleton.init(ally, 1);
        skels[i] = sk;
        sk.bones[0].local_position = Vec3.new(@floatFromInt(i), 0, 0);
        sk.update();
        meshes_arr[i] = .{
            .name = "sk",
            .vertex_buffer = .{},
            .index_buffer = .{},
            .index_count = 3,
            .culling_strategy = .always_render,
            .skeleton = sk,
        };
        ptrs[i] = &meshes_arr[i];
    }
    defer for (skels) |sk| sk.deinit();

    var culler = visibility.OcclusionCuller.init();
    var qa = RenderQueues{};
    defer qa.deinit(ally);
    var qb = RenderQueues{};
    defer qb.deinit(ally);

    // Два вида (multi-camera): у каждой очереди своё хранилище.
    var sa = SceneStats{};
    buildFrameQueues(.{
        .allocator = ally,
        .meshes = ptrs,
        .cache_key = 11,
        .view_proj = Mat4.identity,
        .eye = Vec3.zero,
        .cull_frustum = false,
        .cull_occlusion = false,
        .occlusion_culler = &culler,
        .stats = &sa,
        .queues = &qa,
        .default_white_id = 1,
    });
    var sb = SceneStats{};
    buildFrameQueues(.{
        .allocator = ally,
        .meshes = ptrs,
        .cache_key = 11,
        .view_proj = Mat4.mul(Mat4.identity, Mat4.translation(Vec3.new(0, 0, 5))),
        .eye = Vec3.new(0, 0, 5),
        .cull_frustum = false,
        .cull_occlusion = false,
        .occlusion_culler = &culler,
        .stats = &sb,
        .queues = &qb,
        .default_white_id = 1,
    });

    try std.testing.expectEqual(@as(usize, count), qa.items.items.len);
    try std.testing.expectEqual(@as(usize, count), qa.skin_storage.items.len);
    try std.testing.expectEqual(@as(usize, count), qb.skin_storage.items.len);
    // Хранилище пережило несколько реаллокаций: каждый индекс резолвится
    // в копию своего скелета (x == номер меша).
    for (qa.items.items, qb.items.items) |a, b| {
        const ea: f32 = @floatFromInt(a.mesh_index);
        const eb: f32 = @floatFromInt(b.mesh_index);
        try std.testing.expectApproxEqAbs(ea, qa.skin_storage.items[a.skin_index.?][0].m[12], 1e-4);
        try std.testing.expectApproxEqAbs(eb, qb.skin_storage.items[b.skin_index.?][0].m[12], 1e-4);
    }

    // Мутация всех скелетов (x=1000+i, две публикации) — снимки целы.
    for (skels, 0..) |sk, i| {
        sk.bones[0].local_position = Vec3.new(1000.0 + @as(f32, @floatFromInt(i)), 0, 0);
        sk.update();
        sk.update();
    }
    for (qa.items.items) |a| {
        const ea: f32 = @floatFromInt(a.mesh_index);
        try std.testing.expectApproxEqAbs(ea, qa.skin_storage.items[a.skin_index.?][0].m[12], 1e-4);
    }

    // Reset/reuse: ёмкости retained, содержимое пересобрано корректно.
    const cap = qa.skin_storage.capacity;
    try std.testing.expect(cap >= count);
    qa.reset();
    try std.testing.expectEqual(@as(usize, 0), qa.items.items.len);
    try std.testing.expectEqual(@as(usize, 0), qa.skin_storage.items.len);
    try std.testing.expectEqual(cap, qa.skin_storage.capacity);
    var sa2 = SceneStats{};
    buildFrameQueues(.{
        .allocator = ally,
        .meshes = ptrs,
        .cache_key = 12,
        .view_proj = Mat4.identity,
        .eye = Vec3.zero,
        .cull_frustum = false,
        .cull_occlusion = false,
        .occlusion_culler = &culler,
        .stats = &sa2,
        .queues = &qa,
        .default_white_id = 1,
    });
    try std.testing.expectEqual(@as(usize, count), qa.items.items.len);
    for (qa.items.items) |a| {
        const ea: f32 = 1000.0 + @as(f32, @floatFromInt(a.mesh_index));
        try std.testing.expectApproxEqAbs(ea, qa.skin_storage.items[a.skin_index.?][0].m[12], 1e-4);
    }
}

// Фиксированная цена item не содержит MAX_BONES-матриц; пустые хранилища
// ничего не стоят, когда skinned/shader-draws отсутствуют.
test "P4: no fixed huge per-item skin cost" {
    try std.testing.expect(@sizeOf(RenderMeshItem) < 1024);
    try std.testing.expect(@sizeOf(RenderInstancedBatch) < 1024);
    // Один MAX_BONES-слот — 4 КиБ: item обязан быть кратно меньше.
    try std.testing.expect(@sizeOf(RenderMeshItem) * 7 < @sizeOf([MAX_BONES]Mat4));

    const ally = std.testing.allocator;
    var mesh = Mesh{
        .name = "plain",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .culling_strategy = .always_render,
    };
    var queues = RenderQueues{};
    defer queues.deinit(ally);
    var stats = SceneStats{};
    var culler = visibility.OcclusionCuller.init();
    const meshes = [_]*Mesh{&mesh};
    buildFrameQueues(.{
        .allocator = ally,
        .meshes = &meshes,
        .cache_key = 1,
        .view_proj = Mat4.identity,
        .eye = Vec3.zero,
        .cull_frustum = false,
        .cull_occlusion = false,
        .occlusion_culler = &culler,
        .stats = &stats,
        .queues = &queues,
        .default_white_id = 1,
    });
    try std.testing.expectEqual(@as(usize, 1), queues.items.items.len);
    try std.testing.expectEqual(@as(usize, 0), queues.skin_storage.items.len);
    try std.testing.expectEqual(@as(usize, 0), queues.shader_storage.items.len);
    try std.testing.expect(queues.items.items[0].skin_index == null);
    try std.testing.expect(queues.items.items[0].shader_index == null);
}

fn expectP4QueueRefsValid(queues: *const RenderQueues) !void {
    for (queues.items.items) |it| {
        if (it.is_skinned) {
            try std.testing.expect(it.skin_index != null);
            try std.testing.expect(it.skin_index.? < queues.skin_storage.items.len);
        } else {
            try std.testing.expect(it.skin_index == null);
        }
        if (it.shader_index) |s| try std.testing.expect(s < queues.shader_storage.items.len);
    }
    for (queues.transparent.items) |it| {
        if (it.is_skinned) {
            try std.testing.expect(it.skin_index != null);
            try std.testing.expect(it.skin_index.? < queues.skin_storage.items.len);
        } else {
            try std.testing.expect(it.skin_index == null);
        }
        if (it.shader_index) |s| try std.testing.expect(s < queues.shader_storage.items.len);
    }
    // Нет orphan-ссылок порядка: каждая запись указывает в существующий слот.
    for (queues.transparent_order.items) |e| {
        if (e.kind == .regular) {
            try std.testing.expect(e.index < queues.transparent.items.len);
        } else {
            try std.testing.expect(e.index < queues.transparent_instanced.items.len);
        }
    }
}

// OOM в любой точке prepare-фазы: item либо целиком в очереди с валидными
// индексами, либо отсутствует. Тихого отката к живым матрицам нет.
// std.testing.FailingAllocator роняет ровно n-ю аллокацию при прочих успешных —
// так достигаются и поздние отказы (transparent/skin/shader) после ранних успехов.
test "P4: OOM never leaves items with dangling or live skin refs" {
    const ally = std.testing.allocator;
    const Skeleton = skeleton_mod.Skeleton;

    var std_mat = material_mod.StandardMaterial.init("oom");
    const mat: Material = .{ .standard = &std_mat };
    var blend_mat = material_mod.StandardMaterial.init("oom_blend");
    blend_mat.alpha_mode = .blend;
    const blend: Material = .{ .standard = &blend_mat };
    var hook_mat = material_mod.ShaderMaterial.init("oom_hook");
    const hook: Material = .{ .shader_material = &hook_mat };
    var hook_blend_mat = material_mod.ShaderMaterial.init("oom_hook_blend");
    hook_blend_mat.alpha_mode = .blend;
    const hook_blend: Material = .{ .shader_material = &hook_blend_mat };

    const skel = try Skeleton.init(ally, 1);
    defer skel.deinit();
    skel.bones[0].local_position = Vec3.new(2, 0, 0);
    skel.update();

    var skinned = Mesh{
        .name = "oom_skinned",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .culling_strategy = .always_render,
        .material = mat,
        .skeleton = skel,
    };
    var skinned_trans = Mesh{
        .name = "oom_skinned_trans",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .culling_strategy = .always_render,
        .material = blend,
        .skeleton = skel,
    };
    var plain = Mesh{
        .name = "oom_plain",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .culling_strategy = .always_render,
        .material = mat,
    };
    var hooked = Mesh{
        .name = "oom_hook",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .culling_strategy = .always_render,
        .material = hook,
    };
    var hooked_trans = Mesh{
        .name = "oom_hook_trans",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .culling_strategy = .always_render,
        .material = hook_blend,
    };
    const meshes = [_]*Mesh{ &skinned, &skinned_trans, &plain, &hooked, &hooked_trans };
    var culler = visibility.OcclusionCuller.init();

    var full = RenderQueues{};
    defer full.deinit(ally);
    var stats_full = SceneStats{};
    buildFrameQueues(.{
        .allocator = ally,
        .meshes = &meshes,
        .cache_key = 1,
        .view_proj = Mat4.identity,
        .eye = Vec3.zero,
        .cull_frustum = false,
        .cull_occlusion = false,
        .occlusion_culler = &culler,
        .stats = &stats_full,
        .queues = &full,
        .default_white_id = 1,
    });
    try std.testing.expectEqual(@as(usize, 3), full.items.items.len);
    try std.testing.expectEqual(@as(usize, 2), full.transparent.items.len);
    try std.testing.expectEqual(@as(usize, 2), full.skin_storage.items.len);
    try std.testing.expectEqual(@as(usize, 2), full.shader_storage.items.len);
    try expectP4QueueRefsValid(&full);

    // Прогон по каждому n-му отказу отдельно: ранние успехи + поздний отказ.
    var saw_induced = false;
    var saw_partial = false;
    var n: usize = 0;
    while (n <= 40) : (n += 1) {
        var failing = std.testing.FailingAllocator.init(ally, .{ .fail_index = n });
        var q = RenderQueues{};
        defer q.deinit(failing.allocator());
        var st = SceneStats{};
        buildFrameQueues(.{
            .allocator = failing.allocator(),
            .meshes = &meshes,
            .cache_key = 2,
            .view_proj = Mat4.identity,
            .eye = Vec3.zero,
            .cull_frustum = false,
            .cull_occlusion = false,
            .occlusion_culler = &culler,
            .stats = &st,
            .queues = &q,
            .default_white_id = 1,
        });
        try std.testing.expect(q.items.items.len <= full.items.items.len);
        try std.testing.expect(q.transparent.items.len <= full.transparent.items.len);
        try expectP4QueueRefsValid(&q);
        if (failing.has_induced_failure) saw_induced = true;
        if (q.items.items.len < full.items.items.len or q.transparent.items.len < full.transparent.items.len) saw_partial = true;
    }
    // Хотя бы один отказ реально сработал и хотя бы один уронил item.
    try std.testing.expect(saw_induced);
    try std.testing.expect(saw_partial);
}

// Hook-снимки на уровне очередей: opaque/transparent записи несут точные копии
// tint/uniforms/texture/entry, мутация источника их не меняет; parallel-merge
// копирует shader-значения эквивалентно серийному пути.
test "P4: queue shader snapshots are exact and merge-equivalent" {
    const ally = std.testing.allocator;

    var hook_opaque = material_mod.ShaderMaterial.init("hook_opaque");
    hook_opaque.entry_index = 11;
    hook_opaque.tint_color = math.Color3.new(0.1, 0.2, 0.3);
    hook_opaque.alpha = 0.8;
    hook_opaque.texture = Texture{ .image = .{}, .view = .{ .id = 51 }, .sampler = .{ .id = 52 }, .width = 4, .height = 4 };
    hook_opaque.uniforms[0] = .{ 1, 2, 3, 4 };
    hook_opaque.uniforms[7] = .{ 5, 6, 7, 8 };

    var hook_blend = material_mod.ShaderMaterial.init("hook_blend");
    hook_blend.entry_index = 13;
    hook_blend.alpha_mode = .blend;
    hook_blend.tint_color = math.Color3.new(0.4, 0.5, 0.6);
    hook_blend.alpha = 0.7;
    hook_blend.texture = Texture{ .image = .{}, .view = .{ .id = 53 }, .sampler = .{ .id = 54 }, .width = 4, .height = 4 };
    hook_blend.uniforms[0] = .{ 9, 9, 9, 9 };

    var mo = Mesh{
        .name = "hook_opaque_mesh",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .culling_strategy = .always_render,
        .material = .{ .shader_material = &hook_opaque },
    };
    var mt = Mesh{
        .name = "hook_blend_mesh",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .culling_strategy = .always_render,
        .material = .{ .shader_material = &hook_blend },
    };
    const meshes = [_]*Mesh{ &mo, &mt };
    var culler = visibility.OcclusionCuller.init();

    var qs = RenderQueues{};
    defer qs.deinit(ally);
    var ss = SceneStats{};
    buildFrameQueues(.{
        .allocator = ally,
        .meshes = &meshes,
        .cache_key = 41,
        .view_proj = Mat4.identity,
        .eye = Vec3.zero,
        .cull_frustum = false,
        .cull_occlusion = false,
        .occlusion_culler = &culler,
        .stats = &ss,
        .queues = &qs,
        .default_white_id = 1,
    });
    try std.testing.expectEqual(@as(usize, 1), qs.items.items.len);
    try std.testing.expectEqual(@as(usize, 1), qs.transparent.items.len);
    try std.testing.expectEqual(@as(usize, 2), qs.shader_storage.items.len);

    // Parallel-merge копирует shader-значения эквивалентно серийному пути
    // (те же снимки в том же порядке).
    const pool = try jobs.Pool.init(ally, 2);
    defer pool.deinit();
    var qp = RenderQueues{};
    defer qp.deinit(ally);
    var sp = SceneStats{};
    buildFrameQueues(.{
        .allocator = ally,
        .meshes = &meshes,
        .cache_key = 42,
        .view_proj = Mat4.identity,
        .eye = Vec3.zero,
        .cull_frustum = false,
        .cull_occlusion = false,
        .occlusion_culler = &culler,
        .stats = &sp,
        .queues = &qp,
        .default_white_id = 1,
        .thread_pool = pool,
        .parallel_min_meshes = 1,
    });
    try std.testing.expectEqual(qs.shader_storage.items.len, qp.shader_storage.items.len);
    for (qs.shader_storage.items, qp.shader_storage.items) |a, b| {
        try std.testing.expectEqual(a.entry_index, b.entry_index);
        try std.testing.expectEqual(a.tint, b.tint);
        try std.testing.expectEqual(a.tex_view.id, b.tex_view.id);
        try std.testing.expectEqual(a.tex_sampler.id, b.tex_sampler.id);
        try std.testing.expectEqual(a.uniforms, b.uniforms);
    }
    try std.testing.expectEqual(
        qs.items.items[0].shader_index,
        qp.items.items[0].shader_index,
    );
    try std.testing.expectEqual(
        qs.transparent.items[0].shader_index,
        qp.transparent.items[0].shader_index,
    );

    // Мутация источников после prepare: обе очереди хранят точные копии.
    hook_opaque.tint_color = math.Color3.new(9, 9, 9);
    hook_opaque.alpha = 0.0;
    hook_opaque.texture = null;
    hook_opaque.uniforms[0] = .{ 9, 9, 9, 9 };
    hook_opaque.entry_index = 99;
    hook_blend.tint_color = math.Color3.new(8, 8, 8);
    hook_blend.uniforms[0] = .{ 8, 8, 8, 8 };
    hook_blend.entry_index = 98;

    for ([2]*const RenderQueues{ &qs, &qp }) |qq| {
        const so = qq.shader_storage.items[qq.items.items[0].shader_index.?];
        try std.testing.expectEqual(@as(u32, 11), so.entry_index);
        try std.testing.expectEqual([4]f32{ 0.1, 0.2, 0.3, 0.8 }, so.tint);
        try std.testing.expectEqual(@as(u32, 51), so.tex_view.id);
        try std.testing.expectEqual(@as(u32, 52), so.tex_sampler.id);
        try std.testing.expectEqual([4]f32{ 1, 2, 3, 4 }, so.uniforms[0]);
        try std.testing.expectEqual([4]f32{ 5, 6, 7, 8 }, so.uniforms[7]);
        const s_t = qq.shader_storage.items[qq.transparent.items[0].shader_index.?];
        try std.testing.expectEqual(@as(u32, 13), s_t.entry_index);
        try std.testing.expectEqual([4]f32{ 0.4, 0.5, 0.6, 0.7 }, s_t.tint);
        try std.testing.expectEqual(@as(u32, 53), s_t.tex_view.id);
        try std.testing.expectEqual([4]f32{ 9, 9, 9, 9 }, s_t.uniforms[0]);
    }
}

// Hook-sidedness как до P4: draw-путь hook-материалов использует собственный
// double_sided снимка, а не item.double_sided (куда decal- meshes форсят true
// для regular-пути). Decal с single-sided hook-материалом: item — double-sided
// (regular-контракт), снимок — single-sided (hook-контракт).
test "P4: hook shader snapshot preserves material sidedness without decal forcing" {
    const ally = std.testing.allocator;

    var hook_single = material_mod.ShaderMaterial.init("hook_single");
    hook_single.double_sided = false;

    var decal_hook = Mesh{
        .name = "decal_hook",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .culling_strategy = .always_render,
        .material = .{ .shader_material = &hook_single },
        .is_decal = true,
    };
    const meshes = [_]*Mesh{&decal_hook};
    var culler = visibility.OcclusionCuller.init();
    var queues = RenderQueues{};
    defer queues.deinit(ally);
    var stats = SceneStats{};
    buildFrameQueues(.{
        .allocator = ally,
        .meshes = &meshes,
        .cache_key = 43,
        .view_proj = Mat4.identity,
        .eye = Vec3.zero,
        .cull_frustum = false,
        .cull_occlusion = false,
        .occlusion_culler = &culler,
        .stats = &stats,
        .queues = &queues,
        .default_white_id = 1,
    });

    // Decal едет в transparent-очередь с форсированным double_sided (regular).
    try std.testing.expectEqual(@as(usize, 1), queues.transparent.items.len);
    const it = queues.transparent.items[0];
    try std.testing.expect(it.transparent and it.is_decal and it.double_sided);
    // А hook-снимок — single-sided, как sm.double_sided материала.
    const snap = queues.shader_storage.items[it.shader_index.?];
    try std.testing.expect(!snap.double_sided);

    // Мутация материала после prepare снимок не меняет.
    hook_single.double_sided = true;
    try std.testing.expect(!queues.shader_storage.items[it.shader_index.?].double_sided);
}

// Серийный и параллельный пути дают идентичные очереди включая копии скинов:
// воркеры только заимствуют слоты в скретч, копии делаются серийно в merge.
test "P4: parallel cull matches serial on skinned snapshots" {
    const ally = std.testing.allocator;
    const Skeleton = skeleton_mod.Skeleton;
    const count = 40;

    const skels = try ally.alloc(*Skeleton, count);
    defer ally.free(skels);
    const meshes_arr = try ally.alloc(Mesh, count);
    defer ally.free(meshes_arr);
    const ptrs = try ally.alloc(*Mesh, count);
    defer ally.free(ptrs);
    for (0..count) |i| {
        const sk = try Skeleton.init(ally, 2);
        skels[i] = sk;
        sk.bones[0].local_position = Vec3.new(@floatFromInt(i), 0, 0);
        sk.bones[1].local_position = Vec3.new(0, @floatFromInt(i), 0);
        sk.update();
        meshes_arr[i] = .{
            .name = "psk",
            .vertex_buffer = .{},
            .index_buffer = .{},
            .index_count = 3,
            .position = Vec3.new(@as(f32, @floatFromInt(i)) * 0.05, 0, 0),
            .culling_strategy = .always_render,
            .skeleton = sk,
        };
        ptrs[i] = &meshes_arr[i];
    }
    defer for (skels) |sk| sk.deinit();

    const pool = try jobs.Pool.init(ally, 2);
    defer pool.deinit();

    var culler = visibility.OcclusionCuller.init();
    var qs = RenderQueues{};
    defer qs.deinit(ally);
    var qp = RenderQueues{};
    defer qp.deinit(ally);
    var ss = SceneStats{};
    buildFrameQueues(.{
        .allocator = ally,
        .meshes = ptrs,
        .cache_key = 21,
        .view_proj = Mat4.identity,
        .eye = Vec3.zero,
        .cull_frustum = false,
        .cull_occlusion = false,
        .occlusion_culler = &culler,
        .stats = &ss,
        .queues = &qs,
        .default_white_id = 1,
    });
    var sp = SceneStats{};
    buildFrameQueues(.{
        .allocator = ally,
        .meshes = ptrs,
        .cache_key = 21,
        .view_proj = Mat4.identity,
        .eye = Vec3.zero,
        .cull_frustum = false,
        .cull_occlusion = false,
        .occlusion_culler = &culler,
        .stats = &sp,
        .queues = &qp,
        .default_white_id = 1,
        .thread_pool = pool,
        .parallel_min_meshes = 1,
    });

    try std.testing.expectEqual(qs.items.items.len, qp.items.items.len);
    try std.testing.expectEqual(@as(usize, count), qs.items.items.len);
    for (qs.items.items, qp.items.items) |a, b| {
        try std.testing.expectEqual(a.mesh_index, b.mesh_index);
        try std.testing.expectEqual(a.model, b.model);
        try std.testing.expect(a.skin_index != null and b.skin_index != null);
        const sa = qs.skin_storage.items[a.skin_index.?];
        const sb = qp.skin_storage.items[b.skin_index.?];
        try std.testing.expectEqual(sa, sb);
        const ea: f32 = @floatFromInt(a.mesh_index);
        try std.testing.expectApproxEqAbs(ea, sa[0].m[12], 1e-4);
        try std.testing.expectApproxEqAbs(ea, sa[1].m[13], 1e-4);
    }
}
