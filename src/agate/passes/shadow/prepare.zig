//! Render-owned snapshot build. Split out of `shadow_pass.zig` (facade).
//!
//! `prepareInto` is the core prepare algorithm, parameterized by the
//! destination payload: it bins meshes into the pass-owned `binned_meshes`
//! scratch (via `binning.zig`), then snapshots items + skin copies into
//! `out`. Takes the pass and the destination payload as `anytype` so this
//! module never imports `core.zig` or the facade back (same discipline as
//! `particles/*`, `profiler/*`).
//!
//! Moved tests reach `core.ShadowPass` helpers through block-scoped imports
//! that exist only in test builds.
const std = @import("std");
const math = @import("math");
const Mat4 = math.Mat4;
const mesh_mod = @import("../../mesh.zig");
const Mesh = mesh_mod.Mesh;
const InstancedMeshForP5 = mesh_mod.InstancedMesh;
const scene_render_queue = @import("../../scene/render_queue.zig");
const instance_staging = @import("../../scene/instance_staging.zig");
const outline_pass = @import("../outline_pass.zig");
const jobs = @import("../../jobs.zig");

const types = @import("types.zig");

/// Core prepare algorithm, parameterized by the destination payload: bins
/// meshes into the pass-owned `binned_meshes` scratch, then snapshots
/// items + skin copies into `out`. `out` may be the standalone
/// `prepared` (via `prepare`) or a Scene double-buffer slot (P7) — the
/// algorithm runs once here, never duplicated. OOM semantics preserved:
/// growth failure publishes a coherent-empty snapshot (items + skins +
/// zero counts, never stale), skin-copy failure flags the single item
/// gpu_pending without shifting bucket layout.
///
/// `cache_key` tags the world-matrix/AABB cache (fallback: `Scene.frame_id`,
/// game build: build-unique `(build_seq | (1<<63))`); `instance_source`
/// selects the instanced state (`.published` = fallback `instance_render`,
/// `.build_view` = game-frozen provisional). No new caches, no
/// invalidation change.
pub fn prepareInto(
    self: anytype,
    out: anytype,
    meshes: []const *Mesh,
    cache_key: u64,
    instance_source: mesh_mod.InstanceSource,
    pool: ?*jobs.Pool,
) types.BinResult {
    for (meshes) |m| _ = m.ensureUid();
    const binned = self.binMeshes(meshes, pool);
    const total = self.binned_meshes.items.len;
    if (total > out.items.items.len) {
        out.items.resize(self.allocator, total) catch {
            // Атомарность публикации (P4): рост не удался — пустой
            // coherent-снимок (предметы + скины + counts), а не stale-items
            // при очищенных скинах. Следующий кадр строится заново.
            out.items.clearRetainingCapacity();
            out.skins.clearRetainingCapacity();
            out.bin = .{
                .counts = [_]usize{0} ** 6,
                .offsets = [_]usize{0} ** 6,
            };
            return out.bin;
        };
    } else {
        out.items.shrinkRetainingCapacity(total);
    }
    out.skins.clearRetainingCapacity();

    for (self.binned_meshes.items, 0..) |mesh, idx| {
        const is_inst = mesh.instances.items.len > 0;
        // P5: instanced meshes read the frame's staged render state
        // (staged before this prepare); regular meshes use the fresh
        // world cache. No live instance/game-cache reads here.
        // Stage-2B: staged state resolves via instance_source
        // (fallback `.published`, game build `.build_view` provisional).
        const staged = mesh.instanceRenderSource(instance_source).*;
        const aabb_w = if (!is_inst) scene_render_queue.worldAABBCached(cache_key, mesh) else staged.bounds;
        const model = if (!is_inst) scene_render_queue.worldMatrixCached(cache_key, mesh) else Mat4.identity;
        // Копия скина в render-owned хранилище (prepare-фаза). Раскладка
        // items обязана оставаться 1:1 с binned_meshes (бакеты
        // режутся по counts/offsets), поэтому OOM помечает item флагом
        // gpu_pending — renderBuckets его пропускает, но слайсы бакетов
        // не съезжают. Живые матрицы и неверная поза исключены.
        var skin_index: ?u32 = null;
        var skin_oom = false;
        if (mesh.skeleton) |skel| {
            const src = skel.getRenderSkinMatrices();
            out.skins.ensureUnusedCapacity(self.allocator, 1) catch {
                skin_oom = true;
            };
            if (!skin_oom) {
                skin_index = @intCast(out.skins.items.len);
                out.skins.appendAssumeCapacity(src.*);
            }
        }
        const ext = aabb_w.extents();
        const max_dim = @max(ext.x, @max(ext.y, ext.z));

        // Distant-cascade shadow LOD stand-in (render-owned snapshot of the
        // coarsest QEM-simplified child; null = fail safe to high-poly).
        const shadow_lod = types.shadowLodMesh(mesh);

        out.items.items[idx] = .{
            .vertex_buffer = mesh.vertex_buffer,
            .index_buffer = mesh.index_buffer,
            .index_count = mesh.index_count,
            .instance_buffer = staged.buffer,
            .visible_instance_count = staged.count,
            .model = model,
            .world_aabb = aabb_w,
            .max_dim = max_dim,
            .skin_index = skin_index,
            .bucket = types.bucketFor(mesh),
            .is_instanced = is_inst,
            .gpu_pending = mesh.gpu_pending or skin_oom,
            // Instanced batch: drawable iff the staged buffer holds at least
            // one matrix (visible source mesh and/or visible instances).
            // Babylon renders the source mesh and its instances independently
            // of each other's `isVisible`, so the batch must not inherit the
            // source flag wholesale.
            .is_visible = if (is_inst) staged.count > 0 else mesh.is_visible,
            .source_uid = mesh.uid,
            .source_mesh = self.binned_source.items[idx],
            .lod_vertex_buffer = if (shadow_lod) |lm| lm.vertex_buffer else .{},
            .lod_index_buffer = if (shadow_lod) |lm| lm.index_buffer else .{},
            .lod_index_count = if (shadow_lod) |lm| lm.index_count else 0,
            .has_shadow_lod = shadow_lod != null,
        };
    }

    out.bin = binned;
    return binned;
}

// ---- P4 render-owned draw snapshot: регрессия владения. ----

const SkeletonForP4 = @import("../../animation/skeleton.zig").Skeleton;

// Подготовленный shadow-item не ссылается на живые данные: модель и копия
// скина пережили мутацию TRS и две публикации скелета.
test "P4: shadow item owns model and skin snapshots" {
    const core = @import("core.zig");
    const ally = std.testing.allocator;
    const skel = try SkeletonForP4.init(ally, 1);
    defer skel.deinit();
    skel.bones[0].local_position = math.Vec3.new(1, 0, 0);
    skel.update();

    const unit_box = math.BoundingBox.init(math.Vec3.new(-0.5, -0.5, -0.5), math.Vec3.new(0.5, 0.5, 0.5));
    var mesh = Mesh{
        .name = "shadow_skinned",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .position = math.Vec3.new(3, 0, 0),
        .local_bounding_box = unit_box,
        .skeleton = skel,
    };
    var pass = core.testShadowPass(ally);
    defer pass.binned_meshes.deinit(ally);
    defer pass.binned_source.deinit(ally);
    defer pass.prepared.deinit(ally);

    const meshes = [_]*Mesh{&mesh};
    _ = pass.prepare(&meshes, 5, .published, null);
    try std.testing.expectEqual(@as(usize, 1), pass.prepared.items.items.len);
    try std.testing.expectEqual(@as(usize, 1), pass.prepared.skins.items.len);
    const it = pass.prepared.items.items[0];
    try std.testing.expect(it.skin_index != null);

    mesh.position = math.Vec3.new(99, 99, 99);
    skel.bones[0].local_position = math.Vec3.new(5, 0, 0);
    skel.update();
    skel.update();
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), skel.getRenderSkinMatrices()[0].m[12], 1e-4);

    try std.testing.expectApproxEqAbs(@as(f32, 3.0), pass.prepared.items.items[0].model.m[12], 1e-4);
    const bones = pass.prepared.skins.items[pass.prepared.items.items[0].skin_index.?];
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), bones[0].m[12], 1e-4);
}

// Точечный OOM-аллокатор: падает ровно на n-й операции роста. Рост через
// remap и его fallback alloc+copy считаются ОДНОЙ операцией (флаги armed и
// coalesce): иначе число vtable-вызовов на рост зависит от того, смог ли
// backing-аллокатор расширить in-place, и стадия OOM неупорядочена.
// From-empty рост std сводит к alloc и ловится счётчиком напрямую.
const FailNthP4 = struct {
    backing: std.mem.Allocator,
    fail_on: usize,
    count: usize = 0,
    armed: bool = false,
    coalesce: bool = false,

    fn allocator(self: *FailNthP4) std.mem.Allocator {
        return .{
            .ptr = self,
            .vtable = &.{
                .alloc = allocFn,
                .resize = resizeFn,
                .remap = remapFn,
                .free = freeFn,
            },
        };
    }

    fn allocFn(ctx: *anyopaque, len: usize, alignment: std.mem.Alignment, ra: usize) ?[*]u8 {
        const self: *FailNthP4 = @ptrCast(@alignCast(ctx));
        if (self.armed) {
            self.armed = false;
            return null;
        }
        if (self.coalesce) {
            // Fallback после естественной неудачи remap: часть той же
            // операции роста, счётчик не тратится.
            self.coalesce = false;
            return self.backing.rawAlloc(len, alignment, ra);
        }
        self.count += 1;
        if (self.count == self.fail_on) return null;
        return self.backing.rawAlloc(len, alignment, ra);
    }

    fn resizeFn(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ra: usize) bool {
        const self: *FailNthP4 = @ptrCast(@alignCast(ctx));
        return self.backing.rawResize(memory, alignment, new_len, ra);
    }

    fn remapFn(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ra: usize) ?[*]u8 {
        const self: *FailNthP4 = @ptrCast(@alignCast(ctx));
        // Рост существующих буферов (ArrayList идёт через remap) — операция
        // роста; ужатие (shrink) пропускается без счёта.
        // From-empty рост std сводит к alloc и ловится выше.
        if (new_len > memory.len) {
            self.count += 1;
            if (self.count == self.fail_on) {
                self.armed = true;
                return null;
            }
            const res = self.backing.rawRemap(memory, alignment, new_len, ra);
            if (res == null) self.coalesce = true;
            return res;
        }
        return self.backing.rawRemap(memory, alignment, new_len, ra);
    }

    fn freeFn(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, ra: usize) void {
        const self: *FailNthP4 = @ptrCast(@alignCast(ctx));
        return self.backing.rawFree(memory, alignment, ra);
    }
};

// OOM копии скина: раскладка бакетов остаётся 1:1 (item на месте, помечен
// gpu_pending и пропускается рендером), нескinned-сосед рисуется как обычно.
test "P4: shadow skin OOM keeps bucket layout and skips the item" {
    const core = @import("core.zig");
    const ally = std.testing.allocator;
    const skel = try SkeletonForP4.init(ally, 1);
    defer skel.deinit();
    skel.bones[0].local_position = math.Vec3.new(1, 0, 0);
    skel.update();

    const unit_box = math.BoundingBox.init(math.Vec3.new(-0.5, -0.5, -0.5), math.Vec3.new(0.5, 0.5, 0.5));
    var skinned = Mesh{
        .name = "oom_skinned",
        .vertex_buffer = .{},
        .index_buffer = .{},
        // Маркер skinned-бакета: очереди не несут живых указателей.
        .index_count = 9,
        .local_bounding_box = unit_box,
        .skeleton = skel,
    };
    var plain = Mesh{
        .name = "oom_plain",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = unit_box,
    };
    const meshes = [_]*Mesh{ &skinned, &plain };

    // Аллокации свежего prepare: 1) binned_meshes.resize, 2) binned_source.resize,
    // 3) prepared.items.resize, 4) prepared.skins.ensure для skinned-меша.
    // Роняем четвёртую (stage-2A добавил binned_source как вторую).
    var limited = FailNthP4{ .backing = ally, .fail_on = 4 };
    var pass = core.testShadowPass(limited.allocator());
    defer pass.binned_meshes.deinit(limited.allocator());
    defer pass.binned_source.deinit(limited.allocator());
    defer pass.prepared.deinit(limited.allocator());

    const res = pass.prepare(&meshes, 9, .published, null);
    try std.testing.expectEqual(pass.binned_meshes.items.len, pass.prepared.items.items.len);
    try std.testing.expectEqual(@as(usize, 2), pass.prepared.items.items.len);
    try std.testing.expectEqual(@as(usize, 0), pass.prepared.skins.items.len);
    // Counts бакетов обязаны оставаться внутри длин (иначе renderBuckets
    // срежет за границу).
    var total: usize = 0;
    for (res.counts) |c| total += c;
    try std.testing.expectEqual(@as(usize, 2), total);
    for (res.counts, res.offsets) |c, o| try std.testing.expect(o + c <= pass.prepared.items.items.len);

    // Явная побакетная проверка: skinned-item помечен gpu_pending (рендер его
    // пропустит, в нескinned-позе он НЕ рисуется), plain-item чист.
    var skinned_seen = false;
    var plain_seen = false;
    for (pass.prepared.items.items) |it| {
        if (it.index_count == 9) {
            skinned_seen = true;
            try std.testing.expect(it.gpu_pending);
            try std.testing.expect(it.skin_index == null);
        } else {
            plain_seen = true;
            try std.testing.expect(!it.gpu_pending);
            try std.testing.expect(it.skin_index == null);
            try std.testing.expect(it.is_visible);
        }
    }
    try std.testing.expect(skinned_seen and plain_seen);
}

// Атомарность публикации при OOM роста: успешный skinned-prepare, затем
// вынужденный рост на каждой fallible-стадии. Возвращённые/last counts —
// пустой coherent-снимок БЕЗ чтения старых источников (один из них —
// freed-heap-меш, модель P3-destroyMesh между кадрами); binned-указатели,
// предметы, скины — всё пусто. Следующий кадр восстанавливается полностью.
test "P4: shadow prepare growth OOM publishes empty snapshot and recovers" {
    const core = @import("core.zig");
    const ally = std.testing.allocator;
    const skel = try SkeletonForP4.init(ally, 1);
    defer skel.deinit();
    skel.bones[0].local_position = math.Vec3.new(1, 0, 0);
    skel.update();

    const unit_box = math.BoundingBox.init(math.Vec3.new(-0.5, -0.5, -0.5), math.Vec3.new(0.5, 0.5, 0.5));
    var skinned = [8]Mesh{ undefined, undefined, undefined, undefined, undefined, undefined, undefined, undefined };
    for (&skinned, 0..) |*m, i| {
        m.* = Mesh{
            .name = "grow_skinned",
            .vertex_buffer = .{},
            .index_buffer = .{},
            .index_count = 9 + @as(u32, @intCast(i)),
            .local_bounding_box = unit_box,
            .skeleton = skel,
        };
    }
    var plains: [88]Mesh = undefined;
    for (&plains) |*m| {
        m.* = Mesh{
            .name = "grow_plain",
            .vertex_buffer = .{},
            .index_buffer = .{},
            .index_count = 3,
            .local_bounding_box = unit_box,
        };
    }
    // 96 мешей (8 skinned + 88 plains): заведомо больше любой стартовой
    // ёмкости кадра N (формула роста зависит от cache_line платформы —
    // stage-2A: u32-binned_source стартует с 19..35 слотов, поэтому 32 мешей
    // уже недостаточно), рост всех трёх таблиц (binned/binned_source/items)
    // гарантирован.
    var many: [96]*Mesh = undefined;
    for (&skinned, 0..) |*m, i| many[i] = m;
    for (&plains, 0..) |*m, i| many[8 + i] = m;

    var pass = core.testShadowPass(ally);
    defer pass.binned_meshes.deinit(ally);
    defer pass.binned_source.deinit(ally);
    defer pass.prepared.deinit(ally);

    // Успешный SKINNED-кадр N: heap-меш + skinned-меш. Очередь и skin storage
    // непусты, skinned-item валиден. Heap-меш затем уничтожается (модель
    // P3-destroyMesh между кадрами) — следующий prepare не должен его читать.
    // Кадр N+1 из 32 мешей гарантированно требует роста обеих таблиц.
    const heap_mesh = try ally.create(Mesh);
    heap_mesh.* = Mesh{
        .name = "grow_heap_plain",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = unit_box,
    };
    const frame_n = [_]*Mesh{ heap_mesh, &skinned[0] };
    const ok = pass.prepare(&frame_n, 1, .published, null);
    var ok_total: usize = 0;
    for (ok.counts) |c| ok_total += c;
    try std.testing.expectEqual(@as(usize, 2), ok_total);
    try std.testing.expectEqual(@as(usize, 2), pass.prepared.items.items.len);
    try std.testing.expectEqual(@as(usize, 1), pass.prepared.skins.items.len);

    // destroyMesh между кадрами: указатель в binned_meshes висячий до
    // следующего binMeshes; prepare при OOM не должен его разыменовывать.
    ally.destroy(heap_mesh);

    // Стадия A: падает рост binned_meshes — пустые counts И пустой binned-список:
    // prepare не читает ни freed-heap-меш, ни остальные старые источники.
    var lim_a = FailNthP4{ .backing = ally, .fail_on = 1 };
    pass.allocator = lim_a.allocator();
    const ra = pass.prepare(&many, 2, .published, null);
    for (ra.counts) |c| try std.testing.expectEqual(@as(usize, 0), c);
    for (pass.prepared.bin.counts) |c| try std.testing.expectEqual(@as(usize, 0), c);
    try std.testing.expectEqual(@as(usize, 0), pass.binned_meshes.items.len);
    try std.testing.expectEqual(@as(usize, 0), pass.binned_source.items.len);
    try std.testing.expectEqual(@as(usize, 0), pass.prepared.items.items.len);
    try std.testing.expectEqual(@as(usize, 0), pass.prepared.skins.items.len);

    // Стадия B: binned вырос (оба списка), падает рост prepared.items — пустой
    // снимок: ни предметов, ни скинов, counts нулевые. (Stage-2A: binned_source
    // как вторая аллокация, поэтому fail_on=3.)
    var lim_b = FailNthP4{ .backing = ally, .fail_on = 3 };
    pass.allocator = lim_b.allocator();
    const rb = pass.prepare(&many, 3, .published, null);
    for (rb.counts) |c| try std.testing.expectEqual(@as(usize, 0), c);
    for (pass.prepared.bin.counts) |c| try std.testing.expectEqual(@as(usize, 0), c);
    try std.testing.expectEqual(@as(usize, 0), pass.prepared.items.items.len);
    try std.testing.expectEqual(@as(usize, 0), pass.prepared.skins.items.len);

    // Восстановление следующим кадром с рабочим аллокатором.
    pass.allocator = ally;
    const rc = pass.prepare(&many, 4, .published, null);
    var rc_total: usize = 0;
    for (rc.counts) |c| rc_total += c;
    try std.testing.expectEqual(@as(usize, 96), rc_total);
    try std.testing.expectEqual(@as(usize, 96), pass.prepared.items.items.len);
    try std.testing.expectEqual(@as(usize, 8), pass.prepared.skins.items.len);
    for (rc.counts, rc.offsets) |c, o| try std.testing.expect(o + c <= pass.prepared.items.items.len);
    for (pass.prepared.items.items) |it| {
        if (it.skin_index) |s| {
            try std.testing.expect(s < pass.prepared.skins.items.len);
            try std.testing.expect(!it.gpu_pending);
        }
    }

    // Стадия C: та же атомарность через parallel-ветку binMeshes (порог 128;
    // stage-2A: 200 мешей, 20 skinned — заведомо больше retained-ёмкостей
    // 96-кадра выше, рост binned_meshes гарантирован). OOM роста
    // binned_meshes — пустые counts И пустой binned-список: старые указатели
    // (включая freed-heap кадра N) не читаются.
    const pool = try jobs.Pool.init(ally, 2);
    defer pool.deinit();
    const big_meshes = try ally.alloc(Mesh, 200);
    defer ally.free(big_meshes);
    const big_ptrs = try ally.alloc(*Mesh, 200);
    defer ally.free(big_ptrs);
    for (big_meshes, 0..) |*m, i| {
        m.* = if (i < 20) skinned[i % 8] else plains[0];
        big_ptrs[i] = m;
    }
    var lim_c = FailNthP4{ .backing = ally, .fail_on = 1 };
    pass.allocator = lim_c.allocator();
    const rd = pass.prepare(big_ptrs, 5, .published, pool);
    for (rd.counts) |c| try std.testing.expectEqual(@as(usize, 0), c);
    for (pass.prepared.bin.counts) |c| try std.testing.expectEqual(@as(usize, 0), c);
    try std.testing.expectEqual(@as(usize, 0), pass.binned_meshes.items.len);
    try std.testing.expectEqual(@as(usize, 0), pass.binned_source.items.len);
    try std.testing.expectEqual(@as(usize, 0), pass.prepared.items.items.len);
    try std.testing.expectEqual(@as(usize, 0), pass.prepared.skins.items.len);

    // Финальное восстановление parallel-кадром: все 200 на месте.
    pass.allocator = ally;
    const re = pass.prepare(big_ptrs, 6, .published, pool);
    var re_total: usize = 0;
    for (re.counts) |c| re_total += c;
    try std.testing.expectEqual(@as(usize, 200), re_total);
    try std.testing.expectEqual(@as(usize, 200), pass.prepared.items.items.len);
    try std.testing.expectEqual(@as(usize, 20), pass.prepared.skins.items.len);
    for (re.counts, re.offsets) |c, o| try std.testing.expect(o + c <= pass.prepared.items.items.len);
}

// ---- P5 instance staging ownership: читатели одного published state. ----

// Shadow/main/outline обязаны читать одно и то же опубликованное состояние
// кадра (bounds/count/handle) — ни прошлокадровых значений, ни живых
// instance/game-кешей. Без GPU-контекста хендлы пустые, но равенство
// источников и точные count/bounds ловят рассинхрон читателей.
test "P5: shadow, main batch and outline read identical published state" {
    const core = @import("core.zig");
    const ally = std.testing.allocator;
    const Vec3 = math.Vec3;
    const BoundingBox = math.BoundingBox;

    var src = Mesh{
        .name = "shared_src",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    var inst0 = InstancedMeshForP5{ .name = "s0", .source_mesh = &src, .position = Vec3.new(0, 0, 0) };
    var inst1 = InstancedMeshForP5{ .name = "s1", .source_mesh = &src, .position = Vec3.new(6, 0, 0) };
    var inst2 = InstancedMeshForP5{ .name = "s2", .source_mesh = &src, .position = Vec3.new(12, 0, 0), .is_visible = false };
    var ptrs = [_]*InstancedMeshForP5{ &inst0, &inst1, &inst2 };
    var parent = Mesh{
        .name = "shared_parent",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .instances = std.ArrayListUnmanaged(*InstancedMeshForP5){ .items = &ptrs, .capacity = 3 },
    };
    const meshes = [_]*Mesh{&parent};

    // Pre-stage кадра (как Scene.prepareFrame до shadow prepare).
    var stage_queues = scene_render_queue.RenderQueues{};
    defer stage_queues.deinit(ally);
    instance_staging.stageInstances(.{
        .allocator = ally,
        .instance_matrices = &stage_queues.instance_matrices,
        .thread_pool = null,
        .frame_id = 61,
        .eye = Vec3.zero,
    }, &meshes);
    try std.testing.expectEqual(@as(u32, 3), parent.instance_render.count);
    try std.testing.expectEqual(@as(u64, 61), parent.instance_render.staged_frame);

    // Shadow-читатель.
    var pass = core.testShadowPass(ally);
    defer pass.binned_meshes.deinit(ally);
    defer pass.binned_source.deinit(ally);
    defer pass.prepared.deinit(ally);
    _ = pass.prepare(&meshes, 61, .published, null);
    try std.testing.expectEqual(@as(usize, 1), pass.prepared.items.items.len);
    const shadow_item = pass.prepared.items.items[0];
    try std.testing.expect(shadow_item.is_instanced);

    // Main-читатель (view queue batch).
    var queues = scene_render_queue.RenderQueues{};
    defer queues.deinit(ally);
    var stats = @import("../../scene/stats.zig").SceneStats{};
    var culler = @import("../../visibility/mod.zig").OcclusionCuller.init();
    scene_render_queue.buildFrameQueues(.{
        .allocator = ally,
        .meshes = &meshes,
        .cache_key = 61,
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
    const batch = queues.opaque_instanced.items[0];

    // Outline-читатель.
    var skins = scene_render_queue.SkinStorage.empty;
    defer skins.deinit(ally);
    const outline_item = outline_pass.makeOutlineDrawItem(ally, &skins, &parent, 61, 0, .published) orelse
        return error.TestUnexpectedResult;

    // Все трое — один count (2 видимых, не 3 всего: count-only по
    // instances.len провалился бы), один хендл, одни границы.
    try std.testing.expectEqual(parent.instance_render.count, shadow_item.visible_instance_count);
    try std.testing.expectEqual(parent.instance_render.count, batch.visible_instance_count);
    try std.testing.expectEqual(parent.instance_render.count, outline_item.visible_instance_count);
    try std.testing.expectEqual(parent.instance_render.buffer.id, shadow_item.instance_buffer.id);
    try std.testing.expectEqual(parent.instance_render.buffer.id, batch.instance_buffer.id);
    try std.testing.expectEqual(parent.instance_render.buffer.id, outline_item.instance_buffer.id);
    try std.testing.expectEqual(parent.instance_render.bounds, shadow_item.world_aabb);
    try std.testing.expectEqual(parent.instance_render.bounds.center(), outline_item.world_center);
    // Точное значение: inst0 [-1,1] + inst1 [5,7] → центр staged границ x=3.
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), outline_item.world_center.x, 1e-4);
}

// ---- P5 revision: сбойный пре-стейдж дефинитивен для кадра. ----

// Transient pre-stage failure (scratch OOM) keeps the previous complete
// publish; the shadow snapshot taken from it must stay coherent with every
// main view even though a retry with a working allocator would succeed —
// Scene view builds (instances_prepared) consume the old state and never
// retry mid-frame. Next frame's successful pre-stage updates all readers.
// The transparent parent additionally pins primary-eye sort order: extra
// views with other eyes must not re-sort.
test "P5: failed pre-stage is definitive — views consume the old snapshot" {
    const core = @import("core.zig");
    const ally = std.testing.allocator;
    const Vec3 = math.Vec3;
    const BoundingBox = math.BoundingBox;
    const material = @import("../../material.zig");
    var blend_mat = material.StandardMaterial.init("blend");
    blend_mat.alpha_mode = .blend;

    var src = Mesh{
        .name = "defin_src",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    var inst0 = InstancedMeshForP5{ .name = "d0", .source_mesh = &src, .position = Vec3.new(0, 0, 0) };
    var inst1 = InstancedMeshForP5{ .name = "d1", .source_mesh = &src, .position = Vec3.new(6, 0, 0) };
    var ptrs = [_]*InstancedMeshForP5{ &inst0, &inst1 };
    var parent = Mesh{
        .name = "defin_parent",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .material = .{ .standard = &blend_mat },
        .instances = std.ArrayListUnmanaged(*InstancedMeshForP5){ .items = &ptrs, .capacity = 2 },
    };
    const meshes = [_]*Mesh{&parent};

    // Frame 71: successful pre-stage with the primary eye — count 2,
    // transparent sort back-to-front puts the farther inst1 (x=6) first.
    var stage_queues = scene_render_queue.RenderQueues{};
    defer stage_queues.deinit(ally);
    const primary_eye = Vec3.new(-50, 0, 0);
    instance_staging.stageInstances(.{
        .allocator = ally,
        .instance_matrices = &stage_queues.instance_matrices,
        .thread_pool = null,
        .frame_id = 71,
        .eye = primary_eye,
    }, &meshes);
    try std.testing.expectEqual(@as(u32, 3), parent.instance_render.count);
    try std.testing.expectEqual(@as(u64, 71), parent.instance_render.staged_frame);
    try std.testing.expectEqual(@as(usize, 3), stage_queues.instance_matrices.items.len);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), stage_queues.instance_matrices.items[0].m[12], 1e-4);
    const old_bounds = parent.instance_render.bounds;

    // Frame 72: hide inst1, then FAIL the pre-stage (fresh scratch +
    // failing allocator) — the frame-71 publish stays intact.
    inst1.is_visible = false;
    var bare = scene_render_queue.RenderQueues{};
    defer bare.deinit(ally);
    var failing = std.testing.FailingAllocator.init(ally, .{ .fail_index = 0 });
    instance_staging.stageInstances(.{
        .allocator = failing.allocator(),
        .instance_matrices = &bare.instance_matrices,
        .thread_pool = null,
        .frame_id = 72,
        .eye = primary_eye,
    }, &meshes);
    try std.testing.expectEqual(@as(u32, 3), parent.instance_render.count);
    try std.testing.expectEqual(@as(u64, 71), parent.instance_render.staged_frame);

    // Shadow snapshot captures the OLD state (count 2, old bounds).
    var pass = core.testShadowPass(ally);
    defer pass.binned_meshes.deinit(ally);
    defer pass.binned_source.deinit(ally);
    defer pass.prepared.deinit(ally);
    _ = pass.prepare(&meshes, 72, .published, null);
    try std.testing.expectEqual(@as(usize, 1), pass.prepared.items.items.len);
    const shadow_item = pass.prepared.items.items[0];
    try std.testing.expectEqual(@as(u32, 3), shadow_item.visible_instance_count);
    try std.testing.expectEqual(old_bounds, shadow_item.world_aabb);

    // Main build with a WORKING allocator: instances_prepared (as Scene
    // sets) forbids the mid-frame retry — the batch consumes the same old
    // state the shadow saw, and the view build stages nothing itself.
    var queues = scene_render_queue.RenderQueues{};
    defer queues.deinit(ally);
    var stats = @import("../../scene/stats.zig").SceneStats{};
    var culler = @import("../../visibility/mod.zig").OcclusionCuller.init();
    scene_render_queue.buildFrameQueues(.{
        .allocator = ally,
        .meshes = &meshes,
        .cache_key = 72,
        .view_proj = Mat4.identity,
        .eye = Vec3.zero,
        .cull_frustum = false,
        .cull_occlusion = false,
        .occlusion_culler = &culler,
        .stats = &stats,
        .queues = &queues,
        .default_white_id = 1,
        .instances_prepared = true,
    });
    try std.testing.expectEqual(@as(usize, 1), queues.transparent_instanced.items.len);
    try std.testing.expectEqual(shadow_item.visible_instance_count, queues.transparent_instanced.items[0].visible_instance_count);
    try std.testing.expectEqual(@as(u64, 71), parent.instance_render.staged_frame);
    try std.testing.expectEqual(@as(usize, 0), queues.instance_matrices.items.len);

    // Extra view with the opposite eye: still no retry, and the pre-stage
    // scratch keeps primary-eye order (inst1 first).
    var queues2 = scene_render_queue.RenderQueues{};
    defer queues2.deinit(ally);
    var stats2 = @import("../../scene/stats.zig").SceneStats{};
    scene_render_queue.buildFrameQueues(.{
        .allocator = ally,
        .meshes = &meshes,
        .cache_key = 72,
        .view_proj = Mat4.identity,
        .eye = Vec3.new(50, 0, 0),
        .cull_frustum = false,
        .cull_occlusion = false,
        .occlusion_culler = &culler,
        .stats = &stats2,
        .queues = &queues2,
        .default_white_id = 1,
        .instances_prepared = true,
    });
    try std.testing.expectEqual(@as(u64, 71), parent.instance_render.staged_frame);
    try std.testing.expectEqual(@as(u32, 3), parent.instance_render.count);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), stage_queues.instance_matrices.items[0].m[12], 1e-4);

    // Frame 73: successful pre-stage updates every reader to count 1.
    instance_staging.stageInstances(.{
        .allocator = ally,
        .instance_matrices = &stage_queues.instance_matrices,
        .thread_pool = null,
        .frame_id = 73,
        .eye = primary_eye,
    }, &meshes);
    try std.testing.expectEqual(@as(u32, 2), parent.instance_render.count);
    try std.testing.expectEqual(@as(u64, 73), parent.instance_render.staged_frame);
    _ = pass.prepare(&meshes, 73, .published, null);
    try std.testing.expectEqual(@as(u32, 2), pass.prepared.items.items[0].visible_instance_count);
    try std.testing.expectEqual(parent.instance_render.bounds, pass.prepared.items.items[0].world_aabb);
}

test "stage-2A: shadow items carry source uid and list index" {
    const core = @import("core.zig");
    const ally = std.testing.allocator;
    var skipped = Mesh{
        .name = "skip_no_shadow",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .cast_shadows = false,
    };
    var a = Mesh{
        .name = "shadow_a",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
    };
    var b = Mesh{
        .name = "shadow_b",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
    };
    const meshes = [_]*Mesh{ &skipped, &a, &b };
    var pass = core.testShadowPass(ally);
    defer pass.binned_meshes.deinit(ally);
    defer pass.binned_source.deinit(ally);
    defer pass.prepared.deinit(ally);
    _ = pass.prepare(&meshes, 17, .published, null);
    try std.testing.expectEqual(@as(usize, 2), pass.prepared.items.items.len);
    try std.testing.expectEqual(@as(usize, 2), pass.binned_source.items.len);
    // Binned order is bucket-grouped but source indices map back to the
    // input list (1 and 2; skipped mesh 0 never appears).
    for (pass.prepared.items.items) |it| {
        try std.testing.expect(it.source_uid != 0);
        try std.testing.expect(it.source_mesh == 1 or it.source_mesh == 2);
        if (it.source_mesh == 1) try std.testing.expectEqual(a.uid, it.source_uid);
        if (it.source_mesh == 2) try std.testing.expectEqual(b.uid, it.source_uid);
    }
    try std.testing.expect(a.uid != 0 and b.uid != 0 and a.uid != b.uid);
}

// ---- Shadow LOD: QEM stand-in snapshot + high-poly fallback. ----

// The stand-in must be genuinely simplified geometry: QEM-decimate a
// subdivided plane, use the decimated counts for the LOD child, and prove
// the snapshot carries the smaller geometry (never an alias of the source).
test "shadow LOD: prepare snapshots the QEM-simplified stand-in" {
    const core = @import("core.zig");
    const ally = std.testing.allocator;

    var plane_data = try mesh_mod.builders.buildPlaneData(ally, .{
        .width = 4.0,
        .height = 4.0,
        .subdivisions_x = 4,
        .subdivisions_y = 4,
    });
    defer plane_data.deinit(ally);
    const full_tris = plane_data.indices.len / 3;
    try std.testing.expect(full_tris > 4);

    var lod_geom = try mesh_mod.simplifyGeometry(ally, &plane_data, .{
        .target_ratio = 0.25,
        .preserve_border = true,
    });
    defer lod_geom.deinit(ally);
    const lod_tris = lod_geom.indices.len / 3;
    // Genuine decimation proof: strictly fewer triangles, valid indices.
    try std.testing.expect(lod_tris < full_tris);
    for (lod_geom.indices) |idx| try std.testing.expect(idx < lod_geom.vertices.len);

    const full_count: u32 = @intCast(plane_data.indices.len);
    const lod_count: u32 = @intCast(lod_geom.indices.len);
    var src = Mesh{
        .name = "shadow_lod_src",
        .vertex_buffer = .{ .id = 21 },
        .index_buffer = .{ .id = 22 },
        .index_count = full_count,
        .local_bounding_box = plane_data.bounds,
    };
    defer src.lod_levels.deinit(ally);
    var lod_child = Mesh{
        .name = "shadow_lod_child",
        .vertex_buffer = .{ .id = 23 },
        .index_buffer = .{ .id = 24 },
        .index_count = lod_count,
        .local_bounding_box = lod_geom.bounds,
    };
    try src.addLODLevel(ally, 40.0, &lod_child);

    var pass = core.testShadowPass(ally);
    defer pass.binned_meshes.deinit(ally);
    defer pass.binned_source.deinit(ally);
    defer pass.prepared.deinit(ally);
    const meshes = [_]*Mesh{&src};
    _ = pass.prepare(&meshes, 77, .published, null);
    try std.testing.expectEqual(@as(usize, 1), pass.prepared.items.items.len);
    const it = pass.prepared.items.items[0];
    // High-poly snapshot intact (near cascades + spot/point draw this).
    try std.testing.expectEqual(full_count, it.index_count);
    try std.testing.expectEqual(@as(u32, 21), it.vertex_buffer.id);
    // Stand-in snapshot carries the QEM-decimated geometry.
    try std.testing.expect(it.has_shadow_lod);
    try std.testing.expectEqual(lod_count, it.lod_index_count);
    try std.testing.expect(it.lod_index_count < it.index_count);
    try std.testing.expectEqual(@as(u32, 23), it.lod_vertex_buffer.id);
    try std.testing.expectEqual(@as(u32, 24), it.lod_index_buffer.id);
}

// Fallback matrix: no LOD, skinned source, and instanced coverage. A mesh
// without levels snapshots no stand-in; a skinned source never does (LOD
// children carry no skeleton); an instanced parent snapshots the stand-in
// like a regular mesh (the instance batch reuses it in far cascades).
test "shadow LOD: prepare falls back to high-poly without a valid stand-in" {
    const core = @import("core.zig");
    const ally = std.testing.allocator;
    const unit_box = math.BoundingBox.init(math.Vec3.new(-0.5, -0.5, -0.5), math.Vec3.new(0.5, 0.5, 0.5));

    var plain = Mesh{
        .name = "shadow_lod_plain",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 300,
        .local_bounding_box = unit_box,
    };
    var skinned_child = Mesh{
        .name = "shadow_lod_skinned_child",
        .vertex_buffer = .{ .id = 31 },
        .index_buffer = .{ .id = 32 },
        .index_count = 60,
        .local_bounding_box = unit_box,
    };
    const skel = try SkeletonForP4.init(ally, 1);
    defer skel.deinit();
    skel.update();
    var skinned = Mesh{
        .name = "shadow_lod_skinned",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 300,
        .local_bounding_box = unit_box,
        .skeleton = skel,
    };
    defer skinned.lod_levels.deinit(ally);
    try skinned.addLODLevel(ally, 40.0, &skinned_child);

    var inst_child = Mesh{
        .name = "shadow_lod_inst_child",
        .vertex_buffer = .{ .id = 41 },
        .index_buffer = .{ .id = 42 },
        .index_count = 90,
        .local_bounding_box = unit_box,
    };
    var inst_src = Mesh{
        .name = "shadow_lod_inst_src",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 6,
        .local_bounding_box = unit_box,
    };
    var inst = InstancedMeshForP5{ .name = "s0", .source_mesh = &inst_src };
    var ptrs = [_]*InstancedMeshForP5{&inst};
    var inst_parent = Mesh{
        .name = "shadow_lod_inst_parent",
        .vertex_buffer = .{ .id = 43 },
        .index_buffer = .{ .id = 44 },
        .index_count = 300,
        .local_bounding_box = unit_box,
        .instances = .{ .items = &ptrs, .capacity = 1 },
    };
    defer inst_parent.lod_levels.deinit(ally);
    try inst_parent.addLODLevel(ally, 40.0, &inst_child);

    const meshes = [_]*Mesh{ &plain, &skinned, &inst_parent };
    var pass = core.testShadowPass(ally);
    defer pass.binned_meshes.deinit(ally);
    defer pass.binned_source.deinit(ally);
    defer pass.prepared.deinit(ally);
    _ = pass.prepare(&meshes, 78, .published, null);
    try std.testing.expectEqual(@as(usize, 3), pass.prepared.items.items.len);

    for (pass.prepared.items.items) |it| {
        if (it.source_mesh == 0) {
            try std.testing.expect(!it.has_shadow_lod);
            try std.testing.expectEqual(@as(u32, 0), it.lod_index_count);
        } else if (it.source_mesh == 1) {
            try std.testing.expect(!it.has_shadow_lod);
            try std.testing.expectEqual(@as(u32, 0), it.lod_index_count);
            try std.testing.expectEqual(@as(u32, 300), it.index_count);
        } else {
            try std.testing.expectEqual(@as(u32, 2), it.source_mesh);
            try std.testing.expect(it.is_instanced);
            try std.testing.expect(it.has_shadow_lod);
            try std.testing.expectEqual(@as(u32, 90), it.lod_index_count);
            try std.testing.expectEqual(@as(u32, 41), it.lod_vertex_buffer.id);
        }
    }
}
