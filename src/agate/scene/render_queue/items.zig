//! Render-queue payload types: queue items, batches, render-owned snapshot
//! stores, sort/classify helpers. Leaf module of the `render_queue/` split:
//! imports only engine-external modules plus `material`/`skeleton`/
//! `morph_gpu`/`stats` — never the `render_queue.zig` facade and never the
//! sibling leaves (`cull`, `instances`, `build` import this module, not the
//! reverse). Documented anti-cycle rule: a leaf importing the facade would
//! make the facade's re-exports depend on their own consumers.
const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;

const math = @import("math");
const Mat4 = math.Mat4;
const Vec3 = math.Vec3;
const material_mod = @import("../../material.zig");
const Material = material_mod.Material;
const MaterialDrawRecord = material_mod.MaterialDrawRecord;
const skeleton_mod = @import("../../animation/skeleton.zig");
/// Переэкспорт для потребителей снапшот-хранилищ (postfx/scene/outline).
pub const MAX_BONES = skeleton_mod.MAX_BONES;
const morph_gpu = @import("../../mesh/morph_gpu.zig");
const stats_mod = @import("../stats.zig");
const SceneStats = stats_mod.SceneStats;

pub const RenderMeshItem = struct {
    model: Mat4,
    distance_sq: f32,
    is_pbr: bool,
    texture_id: u32,
    /// Index of the source mesh in FrameCullContext.meshes. Feeds the
    /// transparent order tie-break (see TransparentDrawEntry.seq) so serial
    /// and parallel paths break exact-distance ties identically.
    mesh_index: u32 = 0,
    /// Compact, GPU-ready material snapshot (~120-240 B): factors, uv transforms,
    /// cutoff, and texture views/samplers. Render passes consume this directly.
    draw_record: MaterialDrawRecord = .{},
    // True when the mesh material uses .blend alpha mode. Set at queue
    // build time; defaults to false so opaque behavior is unchanged.
    transparent: bool = false,
    // True when the mesh material is double-sided (face culling disabled).
    // Set at queue build time from materialIsDoubleSided; pipeline selection
    // reads only this snapshot, never the live material.
    // Defaults to false so single-sided behavior is unchanged.
    double_sided: bool = false,
    // True when the mesh is a projected decal. Decals route to the transparent
    // queue so they render after all opaque geometry with depth writes disabled,
    // eliminating depth-buffer z-fighting against the underlying surface.
    is_decal: bool = false,
    receive_shadows: bool = true,
    /// Индекс копии скин-матриц в RenderQueues.skin_storage (null = не скин).
    /// Указывает на render-owned снимок, а не на mutable-слоты скелета:
    /// draw-путь резолвит его через хранилище той же очереди уже после всех
    /// реаллокаций prepare-фазы, поэтому индекс стабилен при росте буфера.
    skin_index: ?u32 = null,
    /// Индекс ShaderDrawSnapshot в RenderQueues.shader_storage (null = не hook).
    /// Draw-путь hook-материалов читает только этот снимок.
    shader_index: ?u32 = null,
    /// Индекс CoatParams в RenderQueues.coat_storage (null = слои выключены).
    /// Только PBR-draws с включённым слоем (clearcoat/sheen/anisotropy/
    /// transmission/subsurface) занимают слот; все остальные рисуют из
    /// CoatParams.neutral. 8-байтовый индекс вместо факторов держит размер
    /// item в P4-лимите.
    coat_index: ?u32 = null,
    /// Per-mesh morph uniforms (weights and delta dimensions)
    morph_uniforms: morph_gpu.VsUniforms = .{
        .weights0 = .{ 0, 0, 0, 0 },
        .weights1 = .{ 0, 0, 0, 0 },
        .params = .{ 0, 1, 1, 0 },
    },
    /// Delta strip texture view for GPU-mode morphs
    morph_view: sg.View = .{},

    // Self-contained GPU geometry handles: decouples render execution from *Mesh lifetime
    vertex_buffer: sg.Buffer = .{},
    index_buffer: sg.Buffer = .{},
    index_count: u32 = 0,
    index_type: sg.IndexType = .UINT16,
    base_vertex: u32 = 0,
    is_u32: bool = false,
    is_skinned: bool = false,
};

/// Self-contained per-batch payload for instanced meshes. Decouples render execution
/// from *Mesh lifetime and state.
///
/// Identity (stage-2 increment A, refactor-only): `source_uid` is the source
/// mesh's `Mesh.uid` (nonzero, stable for lifetime), `source_mesh` is the
/// mesh-list index at build time. `instance_buffer`/`visible_instance_count`
/// are provisional for game-built (`.build_view`) payloads until the latch
/// `patchInstanceRefs` finalizes them from the post-latch `instance_render`;
/// identity is validated by uid at patch time (fail-closed zero on mismatch).
pub const RenderInstancedBatch = struct {
    vertex_buffer: sg.Buffer = .{},
    instance_buffer: sg.Buffer = .{},
    index_buffer: sg.Buffer = .{},
    index_count: u32 = 0,
    index_type: sg.IndexType = .UINT16,
    visible_instance_count: u32 = 0,
    is_pbr: bool = false,
    transparent: bool = false,
    double_sided: bool = false,
    is_decal: bool = false,
    receive_shadows: bool = true,
    draw_record: MaterialDrawRecord = .{},
    source_uid: u64 = 0,
    source_mesh: u32 = 0,
    /// Как RenderMeshItem.coat_index, но для инстансированных групп.
    coat_index: ?u32 = null,
};

/// Render-owned хранилище копий скин-матриц: один слот на skinned-draw кадра
/// (а не MAX_BONES на каждый item). Живёт в RenderQueues, переживает кадры
/// (clearRetainingCapacity в reset — без per-frame churn). Элементы резолвятся
/// по индексу уже на draw-фазе, поэтому реаллокации prepare-фазы безопасны.
pub const SkinStorage = std.ArrayListUnmanaged([MAX_BONES]Mat4);
/// Render-owned снимки hook-материалов: один на shader-draw кадра.
pub const ShaderStorage = std.ArrayListUnmanaged(material_mod.ShaderDrawSnapshot);
/// Render-owned снимки слоёв PBR (clearcoat/sheen/anisotropy/transmission/
/// subsurface): один на layer-draw кадра (только PBR с включённым слоем; остальные draws хранят coat_index = null
/// и рисуют из CoatParams.neutral — пустое хранилище ничего не стоит).
pub const CoatStorage = std.ArrayListUnmanaged(material_mod.CoatParams);

/// Чистый резолв копии скина по индексу (draw-фаза): null/out-of-range дают
/// null вместо чтения чужих данных. Skinned-вызывающий такой draw пропускает
/// (stale GPU-униформа предыдущего draw исключена); через билдеры недостижимо.
pub fn skinAt(skins: []const [MAX_BONES]Mat4, index: ?u32) ?*const [MAX_BONES]Mat4 {
    const i = index orelse return null;
    if (i >= skins.len) return null;
    return &skins[i];
}

/// Чистый резолв CoatParams по индексу (draw-фаза): null/out-of-range дают
/// null вместо чтения чужих данных; вызывающий подставляет
/// CoatParams.neutral (lobe выключен — бит-идентично легаси).
pub fn coatAt(coats: []const material_mod.CoatParams, index: ?u32) ?*const material_mod.CoatParams {
    const i = index orelse return null;
    if (i >= coats.len) return null;
    return &coats[i];
}

/// Prepare-локальный результат куллинга одного меша: готовый item плюс
/// заимствования живых данных, которые appendRenderItem копирует в
/// render-owned хранилища (и только потом кладёт item в очередь).
/// skin_src/shader_snap никогда не попадают в очереди и не читаются на draw.
pub const CulledMesh = struct {
    item: RenderMeshItem,
    /// Заимствованный опубликованный слот скелета (prepare-время; фазовый
    /// мьютекс держит update-поток вне prepare/render, слот стабилен до копии).
    skin_src: ?*const [MAX_BONES]Mat4 = null,
    /// Готовый снимок hook-материала (чистая копия, без живых указателей).
    shader_snap: ?material_mod.ShaderDrawSnapshot = null,
    /// Готовый снимок clearcoat/sheen-факторов (чистая копия 64 Б; в скретче
    /// параллельного пути — только значение, в очередь попадает через копию
    /// в appendRenderItem).
    coat: ?material_mod.CoatParams = null,
};

/// One transparent draw in the unified back-to-front pass. Regular items
/// index `queues.transparent`, instanced groups index
/// `queues.transparent_instanced`; both draw as a single batch at their
/// sorted position (no per-instance sorting/OIT). `seq` is the source mesh
/// index (RenderMeshItem.mesh_index) and only breaks exact distance ties
/// deterministically. Mesh order is identical across serial and parallel
/// paths, so the tie-break — and therefore the sorted order — matches
/// exactly between them.
pub const TransparentKind = enum { regular, instanced };
pub const TransparentDrawEntry = struct {
    distance_sq: f32,
    seq: u32,
    kind: TransparentKind,
    index: u32,
    is_decal: bool,
};

/// Reusable scratch for the parallel cull pass (buildFrameQueuesParallel):
/// per-chunk record buffers plus per-chunk stats. Owned per `RenderQueues`
/// (each view queue builds independently, so each keeps its own scratch)
/// and retained across frames: `ensure` grows the outer list and the
/// per-record capacities only when the chunk/span demand exceeds what is
/// already held, `reset` clears lengths for reuse, and `deinit` (via
/// `RenderQueues.deinit`) frees everything at shutdown. Steady-state frames
/// therefore perform zero allocations in the parallel setup path.
pub const ParallelCullScratch = struct {
    records: std.ArrayListUnmanaged(std.ArrayListUnmanaged(CulledMesh)) = .empty,
    chunk_stats: std.ArrayListUnmanaged(SceneStats) = .empty,

    /// Fit to hold `chunk_count` chunks of `span` records each. All fallible
    /// growth happens here, before any queue/stats write and before
    /// `forkJoin`, so callers keep the OOM→serial-fallback contract:
    /// failure leaves queues/stats untouched. Growth is atomic:
    /// on error the scratch is freed back to empty (same as the old
    /// per-call cleanup), so a failed setup owns no buffers even if the
    /// caller never runs deinit; the next call simply regrows.
    pub fn ensure(self: *ParallelCullScratch, allocator: std.mem.Allocator, chunk_count: usize, span: usize) !void {
        errdefer {
            self.deinit(allocator);
            self.* = .{};
        }
        try self.records.ensureTotalCapacity(allocator, chunk_count);
        while (self.records.items.len < chunk_count) self.records.appendAssumeCapacity(.empty);
        // Shrinking must release the trailing records' owned buffers before
        // truncating the list, otherwise they become unreachable and leak.
        while (self.records.items.len > chunk_count) {
            var r = self.records.pop().?;
            r.deinit(allocator);
        }
        for (self.records.items) |*r| try r.ensureTotalCapacity(allocator, span);
        try self.chunk_stats.ensureTotalCapacity(allocator, chunk_count);
        while (self.chunk_stats.items.len < chunk_count) self.chunk_stats.appendAssumeCapacity(.{});
        if (self.chunk_stats.items.len > chunk_count) self.chunk_stats.items.len = chunk_count;
    }

    /// Clear per-chunk record lengths and stats for reuse. The outer record
    /// list length is kept — its elements own the retained buffers, and
    /// truncating it would leak them; `ensure` re-fits lengths to the next
    /// call's chunk count.
    pub fn reset(self: *ParallelCullScratch) void {
        for (self.records.items) |*r| r.clearRetainingCapacity();
        self.chunk_stats.clearRetainingCapacity();
    }

    pub fn deinit(self: *ParallelCullScratch, allocator: std.mem.Allocator) void {
        for (self.records.items) |*r| r.deinit(allocator);
        self.records.deinit(allocator);
        self.chunk_stats.deinit(allocator);
    }
};

/// The four draw queues plus the per-frame instance-matrix staging buffer.
/// One view's worth: P7 retains 1 + MAX_CAMERAS of these inside the two
/// FrameDrawSlot slabs (Scene.draws) — buildFrameQueues fills the BACK
/// slot's queues during prepareFrame, render reads the published FRONT
/// slot's through Scene.preparedDraws. reset() clears lengths (retaining
/// capacity) for the next build; deinit frees once per slot at shutdown.
pub const RenderQueues = struct {
    items: std.ArrayListUnmanaged(RenderMeshItem) = .empty,
    // Transparent meshes (material alpha_mode == .blend), sorted strictly
    // back-to-front and drawn after every opaque mesh and instanced mesh.
    transparent: std.ArrayListUnmanaged(RenderMeshItem) = .empty,
    opaque_instanced: std.ArrayListUnmanaged(RenderInstancedBatch) = .empty,
    transparent_instanced: std.ArrayListUnmanaged(RenderInstancedBatch) = .empty,
    instance_matrices: std.ArrayListUnmanaged(Mat4) = .empty,
    // Unified transparent draw order across regular + instanced groups,
    // sorted by sortTransparentDrawOrder in renderSceneView. Retained
    // across frames (clearRetainingCapacity in reset: no per-frame churn).
    transparent_order: std.ArrayListUnmanaged(TransparentDrawEntry) = .empty,
    // Render-owned копии скин-матриц skinned-draws (резолв по skin_index на
    // draw-фазе) и снимки hook-материалов (резолв по shader_index). Рост
    // буферов в prepare-фазе безопасен: очереди хранят индексы, а не указатели.
    skin_storage: SkinStorage = .empty,
    shader_storage: ShaderStorage = .empty,
    // Render-owned снимки clearcoat/sheen-факторов (резолв по coat_index на
    // draw-фазе). Рост буферов в prepare-фазе безопасен: очереди хранят
    // индексы, а не указатели.
    coat_storage: CoatStorage = .empty,
    // Parallel-cull scratch (per-chunk record buffers + stats). Lives with
    // the view queues and is retained across frames (reset + ensure per
    // buildFrameQueuesParallel call, freed once in deinit), so repeated
    // per-view parallel culls reuse the same memory instead of
    // allocating/freeing chunk buffers on every call.
    parallel_scratch: ParallelCullScratch = .{},

    pub fn reset(self: *RenderQueues) void {
        self.items.clearRetainingCapacity();
        self.transparent.clearRetainingCapacity();
        self.opaque_instanced.clearRetainingCapacity();
        self.transparent_instanced.clearRetainingCapacity();
        self.instance_matrices.clearRetainingCapacity();
        self.transparent_order.clearRetainingCapacity();
        self.skin_storage.clearRetainingCapacity();
        self.shader_storage.clearRetainingCapacity();
        self.coat_storage.clearRetainingCapacity();
    }

    pub fn deinit(self: *RenderQueues, allocator: std.mem.Allocator) void {
        self.items.deinit(allocator);
        self.transparent.deinit(allocator);
        self.opaque_instanced.deinit(allocator);
        self.transparent_instanced.deinit(allocator);
        self.instance_matrices.deinit(allocator);
        self.transparent_order.deinit(allocator);
        self.skin_storage.deinit(allocator);
        self.shader_storage.deinit(allocator);
        self.coat_storage.deinit(allocator);
        self.parallel_scratch.deinit(allocator);
    }
};

pub fn sortRenderItems(_: void, a: RenderMeshItem, b: RenderMeshItem) bool {
    if (a.is_pbr != b.is_pbr) {
        return !a.is_pbr;
    }
    if (a.texture_id != b.texture_id) {
        return a.texture_id < b.texture_id;
    }
    return a.distance_sq < b.distance_sq;
}

// Unified transparent order: globally back-to-front by exact group distance
// so regular and instanced transparents interleave correctly. Strict weak
// ordering (exact distance, then decal, then source mesh index — never an
// epsilon band, which would be non-transitive: a≈b and b≈c need not imply
// a≈c). Exact ties (e.g. coplanar surfaces) keep decals after non-decals,
// then mesh order, for a deterministic result identical across paths.
pub fn sortTransparentDrawOrder(_: void, a: TransparentDrawEntry, b: TransparentDrawEntry) bool {
    if (a.distance_sq != b.distance_sq) {
        return a.distance_sq > b.distance_sq;
    }
    if (a.is_decal != b.is_decal) {
        return !a.is_decal;
    }
    return a.seq < b.seq;
}

// A mesh is transparent when its material opts into .blend alpha mode.
// Meshes without a material render opaque (legacy behavior).
// Cutout (.cutout) is NOT transparent: it renders in the opaque queue
// with depth writes on, discarding only sub-cutoff fragments in-shader.
pub fn materialIsTransparent(mat: ?Material) bool {
    if (mat) |m| return m.isTransparent();
    return false;
}

// A mesh is cutout when its material uses .cutout alpha mode.
// Cutout meshes stay in the opaque queue (see materialIsTransparent).
// Shadow depth passes ignore the alpha test and render full quads
// (documented limitation: cutout holes still cast solid shadows).
pub fn materialIsCutout(mat: ?Material) bool {
    if (mat) |m| return m.isCutout();
    return false;
}

// A mesh is double-sided when its material sets double_sided.
// Such meshes select the cull-off pipeline twins (opaque or blend).
// Meshes without a material render single-sided (legacy behavior).
pub fn materialIsDoubleSided(mat: ?Material) bool {
    if (mat) |m| return m.isDoubleSided();
    return false;
}

// Derives a transparent twin pipeline desc from an opaque base desc:
// same shader/layout/depth test, but SRC_ALPHA/ONE_MINUS_SRC_ALPHA
// blending with depth write disabled. Pure function (no GPU calls).
pub fn blendDescFor(base: sg.PipelineDesc) sg.PipelineDesc {
    var desc = base;
    desc.depth.write_enabled = false;
    desc.colors[0].blend = .{
        .enabled = true,
        .src_factor_rgb = .SRC_ALPHA,
        .dst_factor_rgb = .ONE_MINUS_SRC_ALPHA,
        .src_factor_alpha = .ONE,
        .dst_factor_alpha = .ONE_MINUS_SRC_ALPHA,
    };
    return desc;
}

test "cutout stays opaque, blend stays transparent" {
    const material = @import("../../material.zig");

    try std.testing.expect(!materialIsTransparent(null));
    try std.testing.expect(!materialIsCutout(null));
    try std.testing.expect(!materialIsDoubleSided(null));

    var std_mat = material.StandardMaterial.init("m");
    var pbr_mat = material.PBRMaterial.init("p");

    // Opaque default: opaque queue, not cutout, single-sided.
    try std.testing.expect(!materialIsTransparent(.{ .standard = &std_mat }));
    try std.testing.expect(!materialIsTransparent(.{ .pbr = &pbr_mat }));
    try std.testing.expect(!materialIsCutout(.{ .standard = &std_mat }));
    try std.testing.expect(!materialIsCutout(.{ .pbr = &pbr_mat }));

    // Cutout: still NOT in the transparent queue, but flagged cutout.
    std_mat.alpha_mode = .cutout;
    pbr_mat.alpha_mode = .cutout;
    try std.testing.expect(!materialIsTransparent(.{ .standard = &std_mat }));
    try std.testing.expect(!materialIsTransparent(.{ .pbr = &pbr_mat }));
    try std.testing.expect(materialIsCutout(.{ .standard = &std_mat }));
    try std.testing.expect(materialIsCutout(.{ .pbr = &pbr_mat }));

    // Blend: transparent queue, never cutout.
    std_mat.alpha_mode = .blend;
    pbr_mat.alpha_mode = .blend;
    try std.testing.expect(materialIsTransparent(.{ .standard = &std_mat }));
    try std.testing.expect(materialIsTransparent(.{ .pbr = &pbr_mat }));
    try std.testing.expect(!materialIsCutout(.{ .standard = &std_mat }));
    try std.testing.expect(!materialIsCutout(.{ .pbr = &pbr_mat }));

    // Double-sided is orthogonal to the alpha mode.
    try std.testing.expect(!materialIsDoubleSided(.{ .standard = &std_mat }));
    std_mat.double_sided = true;
    pbr_mat.double_sided = true;
    try std.testing.expect(materialIsDoubleSided(.{ .standard = &std_mat }));
    try std.testing.expect(materialIsDoubleSided(.{ .pbr = &pbr_mat }));
}

// ---- Ported from the legacy Scene inline suite. ----

test "transparent classification follows material alpha mode" {
    const material = @import("../../material.zig");

    try std.testing.expect(!materialIsTransparent(null));
    var std_mat = material.StandardMaterial.init("s");
    var pbr_mat = material.PBRMaterial.init("p");
    try std.testing.expect(!materialIsTransparent(Material{ .standard = &std_mat }));
    try std.testing.expect(!materialIsTransparent(Material{ .pbr = &pbr_mat }));
    std_mat.alpha_mode = .blend;
    pbr_mat.alpha_mode = .blend;
    try std.testing.expect(materialIsTransparent(Material{ .standard = &std_mat }));
    try std.testing.expect(materialIsTransparent(Material{ .pbr = &pbr_mat }));
}

test "opaque sort unchanged: state groups, front-to-back" {
    var items = [_]RenderMeshItem{
        .{ .model = Mat4.identity, .distance_sq = 9.0, .is_pbr = false, .texture_id = 2 },
        .{ .model = Mat4.identity, .distance_sq = 1.0, .is_pbr = false, .texture_id = 2 },
        .{ .model = Mat4.identity, .distance_sq = 5.0, .is_pbr = true, .texture_id = 1 },
        .{ .model = Mat4.identity, .distance_sq = 3.0, .is_pbr = false, .texture_id = 1 },
    };
    // Default transparent flag is false, so legacy items sort as before.
    try std.testing.expect(!items[0].transparent);
    std.mem.sort(RenderMeshItem, &items, {}, sortRenderItems);
    // Standard before PBR, texture id ascending, front-to-back within a group.
    try std.testing.expect(!items[0].is_pbr and items[0].texture_id == 1 and items[0].distance_sq == 3.0);
    try std.testing.expect(!items[1].is_pbr and items[1].texture_id == 2 and items[1].distance_sq == 1.0);
    try std.testing.expect(!items[2].is_pbr and items[2].texture_id == 2 and items[2].distance_sq == 9.0);
    try std.testing.expect(items[3].is_pbr and items[3].distance_sq == 5.0);
}

test "blendDescFor enables alpha blending without depth write" {
    const base = sg.PipelineDesc{
        .shader = .{},
        .index_type = .UINT32,
        .depth = .{ .compare = .LESS_EQUAL, .write_enabled = true },
        .cull_mode = .BACK,
        .face_winding = .CCW,
    };
    const blended = blendDescFor(base);
    try std.testing.expect(blended.colors[0].blend.enabled);
    try std.testing.expect(blended.colors[0].blend.src_factor_rgb == .SRC_ALPHA);
    try std.testing.expect(blended.colors[0].blend.dst_factor_rgb == .ONE_MINUS_SRC_ALPHA);
    try std.testing.expect(!blended.depth.write_enabled);
    try std.testing.expect(blended.depth.compare == .LESS_EQUAL);
    try std.testing.expect(blended.index_type == .UINT32);
    try std.testing.expect(blended.cull_mode == .BACK);
    // Pure function: the base desc is left untouched.
    try std.testing.expect(!base.colors[0].blend.enabled);
    try std.testing.expect(base.depth.write_enabled);
}

// ---- P4 render-owned draw snapshot: регрессия владения. ----

// Чистый селектор skinAt: null/out-of-range не читают чужое, валидный индекс
// отдаёт именно свою копию (draw-пути main/shadow/outline на нём же).
test "P4: skinAt resolves owned copies without stale reads" {
    var empty: [0][MAX_BONES]Mat4 = .{};
    try std.testing.expect(skinAt(&empty, null) == null);
    try std.testing.expect(skinAt(&empty, 0) == null);
    try std.testing.expect(skinAt(&empty, std.math.maxInt(u32)) == null);

    var one: [1][MAX_BONES]Mat4 = [_][MAX_BONES]Mat4{[_]Mat4{Mat4.identity} ** MAX_BONES};
    one[0][3] = Mat4.translation(Vec3.new(2, 0, 0));
    const got = skinAt(&one, 0) orelse return error.TestUnexpectedResult;
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), got[3].m[12], 1e-6);
    try std.testing.expect(skinAt(&one, 1) == null);
    try std.testing.expect(skinAt(&one, std.math.maxInt(u32)) == null);
}

test "coatAt resolves owned copies without stale reads" {
    const CoatParams = material_mod.CoatParams;
    var empty: [0]CoatParams = .{};
    try std.testing.expect(coatAt(&empty, null) == null);
    try std.testing.expect(coatAt(&empty, 0) == null);

    var one = [_]CoatParams{.{ .clearcoat_factors = .{ 0.8, 0.12, 0, 0 } }};
    const got = coatAt(&one, 0) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqualSlices(f32, &.{ 0.8, 0.12, 0, 0 }, &got.clearcoat_factors);
    // Neutral fallback for every miss (disabled lobe path).
    try std.testing.expect(coatAt(&one, 1) == null);
    try std.testing.expect(coatAt(&one, null) == null);
    const fallback = coatAt(&one, null) orelse &CoatParams.neutral;
    try std.testing.expectEqual(@as(f32, 0.0), fallback.clearcoat_factors[0]);
    try std.testing.expectEqual(@as(f32, 0.0), fallback.sheen_factors[0]);
}
