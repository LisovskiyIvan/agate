//! Headless ReleaseFast stress benchmark for scene/render-queue scaling.
//!
//! Drives the REAL engine path (`Scene.buildPreparedFrame`, which runs
//! instance pre-stage + `queue_builder.buildQueuesInto` =
//! `buildFrameQueues` cull + opaque/transparent sorts) with no GPU context
//! (`sg.isvalid()==false`, faked GPU-init flag so GPU halves publish
//! bounds/count without touching `sg.*`; render never runs).
//!
//! Two representative configs, same fixed camera, deterministic grid:
//!   - many-identical: N plain meshes sharing ONE opaque PBRMaterial
//!     (+1 shared transparent material for every 16th mesh).
//!   - many-distinct: N plain meshes each with its OWN PBRMaterial
//!     (every 16th `.blend`, rest opaque; colors vary deterministically).
//!   - many-instanced (`--mode instanced|all` only): `--meshes` total
//!     instances spread over parent meshes (256 each, one shared opaque
//!     material) plus one shared transparent parent (1024 instances,
//!     exercises the staging pair-sort). Drives the REAL instance path:
//!     CPU pre-stage of ALL parent batches (before culling) + queue build
//!     via `buildPreparedFrame`.
//!
//! Timed scope is the WHOLE `buildPreparedFrame` (pre-stage + cull +
//! sorts), never an isolated TRS/sort micro-section. The instanced
//! `per_staged` ns divides the whole-build median by the REAL staged
//! matrix count (`slot.primary.instance_matrices`), i.e. whole-build
//! amortized cost per staged matrix — not a per-stage attribution.
//!
//! Production allocator (`std.heap.smp_allocator`), checksum over the
//! published front queue (Wyhash of model translation + texture_id +
//! distance bits, `doNotOptimizeAway`) defeats DCE. Reports median/min/max
//! per-build ms plus ns/mesh; each case targets >=100ms total.
//!
//! Run: `zig build run-scene-scaling -- --meshes 12000 --frames 30`
//!   (from examples/; see examples/build.zig `run-scene-scaling`).

const std = @import("std");
const z = @import("agate");
const sokol = @import("sokol");

const Mode = enum { identical, distinct, both, instanced, all };

const Cfg = struct {
    meshes: usize = 24000,
    frames: usize = 50,
    warmup: usize = 3,
    mode: Mode = .both,
};

fn parseMode(s: []const u8) ?Mode {
    if (std.mem.eql(u8, s, "identical")) return .identical;
    if (std.mem.eql(u8, s, "distinct")) return .distinct;
    if (std.mem.eql(u8, s, "both")) return .both;
    if (std.mem.eql(u8, s, "instanced")) return .instanced;
    if (std.mem.eql(u8, s, "all")) return .all;
    return null;
}

fn parseArgs(args: []const []const u8) Cfg {
    var c = Cfg{};
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const a = args[i];
        if (std.mem.eql(u8, a, "--meshes") and i + 1 < args.len) {
            i += 1;
            c.meshes = std.fmt.parseInt(usize, args[i], 10) catch c.meshes;
        } else if (std.mem.startsWith(u8, a, "--meshes=")) {
            c.meshes = std.fmt.parseInt(usize, a["--meshes=".len..], 10) catch c.meshes;
        } else if (std.mem.eql(u8, a, "--frames") and i + 1 < args.len) {
            i += 1;
            c.frames = std.fmt.parseInt(usize, args[i], 10) catch c.frames;
        } else if (std.mem.startsWith(u8, a, "--frames=")) {
            c.frames = std.fmt.parseInt(usize, a["--frames=".len..], 10) catch c.frames;
        } else if (std.mem.eql(u8, a, "--warmup") and i + 1 < args.len) {
            i += 1;
            c.warmup = std.fmt.parseInt(usize, args[i], 10) catch c.warmup;
        } else if (std.mem.eql(u8, a, "--mode") and i + 1 < args.len) {
            i += 1;
            c.mode = parseMode(args[i]) orelse c.mode;
        } else if (std.mem.startsWith(u8, a, "--mode=")) {
            c.mode = parseMode(a["--mode=".len..]) orelse c.mode;
        }
    }
    if (c.meshes < 100) c.meshes = 100;
    if (c.meshes > 100000) c.meshes = 100000;
    if (c.frames < 5) c.frames = 5;
    if (c.frames > 200) c.frames = 200;
    return c;
}

fn lessF64(_: void, a: f64, b: f64) bool {
    return a < b;
}

fn median(xs: []f64) f64 {
    std.mem.sort(f64, xs, {}, lessF64);
    return xs[xs.len / 2];
}

/// Checksum over the just-built BACK slot primary queues (buildPreparedFrame
/// publishes to the back handoff; the front flips only at prepare latch
/// time, which never runs in this bench). Covers model translation,
/// texture_id, distance bits and counts so any queue-shape change
/// (cull/sort/batch regression) flips the value.
fn checksumBack(scene: *z.Scene) u64 {
    const back_n = scene.draws.slots.len;
    const idx = scene.build_slot.load(.monotonic) % back_n;
    const slot = &scene.draws.slots[idx];
    var h = std.hash.Wyhash.init(0x5ce45ca1e);
    const items = slot.primary.items.items;
    h.update(std.mem.asBytes(&items.len));
    for (items) |*it| {
        h.update(std.mem.asBytes(&it.model.m[12]));
        h.update(std.mem.asBytes(&it.model.m[13]));
        h.update(std.mem.asBytes(&it.model.m[14]));
        h.update(std.mem.asBytes(&it.texture_id));
        h.update(std.mem.asBytes(&it.distance_sq));
        // Material work proof: identical geometry with distinct materials
        // must checksum differently (draw_record covers factors/views).
        h.update(std.mem.asBytes(&it.draw_record));
    }
    const ti = slot.primary.transparent.items;
    h.update(std.mem.asBytes(&ti.len));
    for (ti) |*it| {
        h.update(std.mem.asBytes(&it.distance_sq));
        h.update(std.mem.asBytes(&it.texture_id));
    }
    h.update(std.mem.asBytes(&slot.primary.opaque_instanced.items.len));
    h.update(std.mem.asBytes(&slot.build_stats.rendered_meshes));
    h.update(std.mem.asBytes(&slot.build_stats.culled_meshes));
    const digest = h.final();
    std.mem.doNotOptimizeAway(digest);
    std.mem.doNotOptimizeAway(items.len);
    return digest;
}

const CaseResult = struct {
    median_ms: f64,
    min_ms: f64,
    max_ms: f64,
    total_ms: f64,
    checksum: u64,
    opaque_count: usize,
    transparent: usize,
    rendered: u32,
    culled: u32,
    inst_batches: usize = 0,
    instanced_count: u32 = 0,
    tinst_batches: usize = 0,
    tinst_staged: usize = 0,
    /// Instanced case only: total parent meshes created (opaque + the one
    /// transparent parent).
    n_parents: usize = 0,
    /// Instanced case only: total InstancedMesh objects generated
    /// (rounded workload plus the transparent batch, includes hidden instances, excludes the
    /// implicit per-parent self entries).
    generated_instances: usize = 0,
    /// Instanced case only: REAL staged matrix count from the back slot
    /// (`slot.primary.instance_matrices.items.len`) — every parent batch
    /// is pre-staged BEFORE culling, so this exceeds the visible/queued
    /// counts below. Denominator of `per_staged`.
    staged_matrices: usize = 0,
};

fn runCase(distinct: bool, cfg: Cfg, alloc: std.mem.Allocator) !CaseResult {
    var scene = z.testing.testScene(alloc);
    scene.enable_frustum_culling = true;
    scene.enable_occlusion_culling = false;
    scene.default_white_texture.view.id = 1;
    scene.shadows.enabled = false;

    const cam = z.Camera{ .arc_rotate = z.ArcRotateCamera.init("bench_cam", .{
        .alpha = std.math.pi / 4.0,
        .beta = std.math.pi / 3.0,
        .radius = 90.0,
        .target = z.Vec3.zero,
    }) };
    _ = scene.addCamera(.{ .name = "bench_cam", .camera = cam }) catch {};

    // Materials: 1 shared (identical) or N owned (distinct), plus one
    // shared transparent material for the every-16th transparent meshes.
    var shared_opaque = z.PBRMaterial.init("bench_opaque");
    var shared_trans = z.PBRMaterial.init("bench_trans");
    shared_trans.alpha_mode = .blend;
    var distinct_mats: []z.PBRMaterial = &.{};
    if (distinct) {
        distinct_mats = try alloc.alloc(z.PBRMaterial, cfg.meshes);
        for (distinct_mats, 0..) |*m, i| {
            m.* = z.PBRMaterial.init("bench_distinct");
            const fi: f32 = @floatFromInt(i);
            m.metallic = @mod(fi * 0.6180339, 1.0);
            m.roughness = 0.2 + 0.6 * @mod(fi * 0.381966, 1.0);
            if (i % 16 == 0) m.alpha_mode = .blend;
        }
    }

    const meshes = try alloc.alloc(z.Mesh, cfg.meshes);
    const unit_box = z.BoundingBox.init(z.Vec3.new(-0.5, -0.5, -0.5), z.Vec3.new(0.5, 0.5, 0.5));
    const grid_w: usize = 120;
    for (meshes, 0..) |*m, i| {
        const gx: f32 = @floatFromInt(i % grid_w);
        const gz: f32 = @floatFromInt(i / grid_w);
        const cols: f32 = @floatFromInt(grid_w);
        const rows: f32 = @floatFromInt((cfg.meshes + grid_w - 1) / grid_w);
        m.* = .{
            .name = "bench",
            .vertex_buffer = .{},
            .index_buffer = .{},
            .index_count = 36,
            .position = z.Vec3.new((gx - cols * 0.5) * 1.5, 0, (gz - rows * 0.5) * 1.5),
            .local_bounding_box = unit_box,
        };
        const is_trans = (i % 16 == 0);
        if (distinct) {
            m.material = .{ .pbr = &distinct_mats[i] };
        } else if (is_trans) {
            m.material = .{ .pbr = &shared_trans };
        } else {
            m.material = .{ .pbr = &shared_opaque };
        }
        try scene.meshes.append(alloc, m);
    }

    // Warmup (discarded): page faults + retained queue/scratch capacity.
    // publishFrameSnapshot per build: headless has no sokol app size, so
    // the live-pack fallback (sapp.width()==0) would build camera-less.
    for (0..cfg.warmup) |_| {
        scene.publishFrameSnapshot(16.0 / 9.0, 1280, 720);
        if (!scene.buildPreparedFrame()) std.debug.panic("bench warmup build failed", .{});
    }
    const warm_sum = checksumBack(&scene);

    const times = try alloc.alloc(f64, cfg.frames);
    defer alloc.free(times);
    var sum: f64 = 0;
    for (0..cfg.frames) |f| {
        scene.publishFrameSnapshot(16.0 / 9.0, 1280, 720);
        const t0 = sokol.time.now();
        if (!scene.buildPreparedFrame()) std.debug.panic("bench build failed at frame {d}", .{f});
        const dt_ms: f64 = @floatCast(sokol.time.ms(sokol.time.now() -% t0));
        times[f] = dt_ms;
        sum += dt_ms;
    }
    const sum_copy = sum;
    const med = median(times);
    var mn = times[0];
    var mx = times[0];
    for (times) |t| {
        mn = @min(mn, t);
        mx = @max(mx, t);
    }
    const sum2 = checksumBack(&scene);
    if (sum2 != warm_sum and cfg.warmup > 0) {
        // Static scene + fresh cache key per build still yields identical
        // queues; a flip here means nondeterminism, fail loudly.
        // (world-cache keys differ per build but matrices are pure.)
        std.debug.panic("bench: checksum drift warm={x} final={x} (queues must be deterministic)", .{ warm_sum, sum2 });
    }

    const bidx = scene.build_slot.load(.monotonic) % scene.draws.slots.len;
    const fslot = &scene.draws.slots[bidx];
    const res = CaseResult{
        .median_ms = med,
        .min_ms = mn,
        .max_ms = mx,
        .total_ms = sum_copy,
        .checksum = sum2,
        .opaque_count = fslot.primary.items.items.len,
        .transparent = fslot.primary.transparent.items.len,
        .rendered = fslot.build_stats.rendered_meshes,
        .culled = fslot.build_stats.culled_meshes,
    };

    // Sort probe (diagnostic, untimed from the case): re-sort a reversed
    // duplicate of the final opaque queue through the PRODUCTION indexed
    // path to attribute sort vs cull cost.
    {
        const src_items = fslot.primary.items.items;
        if (src_items.len > 0) {
            var pq: z.RenderQueues = .{};
            defer pq.deinit(alloc);
            try pq.items.appendSlice(alloc, src_items);
            var probe: [5]f64 = undefined;
            for (0..probe.len) |k| {
                pq.items.clearRetainingCapacity();
                try pq.items.appendSlice(alloc, src_items);
                std.mem.reverse(z.RenderMeshItem, pq.items.items);
                const t0 = sokol.time.now();
                z.sortOpaqueItemsIndexed(&pq, alloc);
                probe[k] = @floatCast(sokol.time.ms(sokol.time.now() -% t0));
            }
            std.mem.sort(f64, &probe, {}, lessF64);
            std.debug.print("sort_probe: n={d} median={d:.3}ms\n", .{ src_items.len, probe[probe.len / 2] });
        }
    }

    // Teardown: slot-owned queues first, then our lists.
    scene.draws.deinit(alloc);
    scene.meshes.deinit(alloc);
    scene.cameras.deinit(alloc);
    scene.profiler.deinit();
    alloc.free(meshes);
    if (distinct) alloc.free(distinct_mats);
    return res;
}

/// Checksum over the just-built BACK slot for the instanced case: the same
/// base queue coverage as checksumBack (kept inline so checksumBack itself
/// stays byte-identical for the two legacy cases) plus the instanced
/// payloads — batch counts/identities/counts and the staged matrix bytes —
/// so any staging-shape change (cull/sort/compaction regression) flips it.
///
/// Optimize-mode scope: this digest hashes RAW staged float bytes, and the
/// staging float math (rotation TRS, AABB transform) is FMA-contraction
/// sensitive, so ReleaseSafe and ReleaseFast digests LEGITIMATELY differ
/// (verified: each self-consistent across repeats, counts identical).
/// Compare strictly like-with-like WITHIN one optimize mode; the two legacy
/// checksums are mode-stable and stay the cross-mode anchor.
fn checksumInstancedBack(scene: *z.Scene) u64 {
    const back_n = scene.draws.slots.len;
    const idx = scene.build_slot.load(.monotonic) % back_n;
    const slot = &scene.draws.slots[idx];
    var h = std.hash.Wyhash.init(0x5ce45ca1e);
    const items = slot.primary.items.items;
    h.update(std.mem.asBytes(&items.len));
    for (items) |*it| {
        h.update(std.mem.asBytes(&it.model.m[12]));
        h.update(std.mem.asBytes(&it.model.m[13]));
        h.update(std.mem.asBytes(&it.model.m[14]));
        h.update(std.mem.asBytes(&it.texture_id));
        h.update(std.mem.asBytes(&it.distance_sq));
        h.update(std.mem.asBytes(&it.draw_record));
    }
    const ti = slot.primary.transparent.items;
    h.update(std.mem.asBytes(&ti.len));
    for (ti) |*it| {
        h.update(std.mem.asBytes(&it.distance_sq));
        h.update(std.mem.asBytes(&it.texture_id));
    }
    const ib = slot.primary.opaque_instanced.items;
    h.update(std.mem.asBytes(&ib.len));
    for (ib) |*b| {
        // No source_uid: uid-counter state depends on which cases ran
        // before in this process — the payload (counts/centers/matrices)
        // is the mode-independent identity.
        h.update(std.mem.asBytes(&b.visible_instance_count));
        h.update(std.mem.asBytes(&b.world_center));
        h.update(std.mem.asBytes(&b.draw_record));
    }
    const tib = slot.primary.transparent_instanced.items;
    h.update(std.mem.asBytes(&tib.len));
    for (tib) |*b| {
        h.update(std.mem.asBytes(&b.visible_instance_count));
        h.update(std.mem.asBytes(&b.world_center));
    }
    const staged = slot.primary.instance_matrices.items;
    h.update(std.mem.asBytes(&staged.len));
    h.update(std.mem.sliceAsBytes(staged));
    h.update(std.mem.asBytes(&slot.build_stats.rendered_meshes));
    h.update(std.mem.asBytes(&slot.build_stats.culled_meshes));
    const digest = h.final();
    std.mem.doNotOptimizeAway(digest);
    std.mem.doNotOptimizeAway(items.len);
    std.mem.doNotOptimizeAway(staged.len);
    return digest;
}

const inst_per_parent: usize = 256;
const transparent_instances: usize = 1024;

fn runCaseInstanced(cfg: Cfg, alloc: std.mem.Allocator) !CaseResult {
    var scene = z.testing.testScene(alloc);
    scene.enable_frustum_culling = true;
    scene.enable_occlusion_culling = false;
    scene.default_white_texture.view.id = 1;
    scene.shadows.enabled = false;

    const cam = z.Camera{ .arc_rotate = z.ArcRotateCamera.init("bench_cam", .{
        .alpha = std.math.pi / 4.0,
        .beta = std.math.pi / 3.0,
        .radius = 90.0,
        .target = z.Vec3.zero,
    }) };
    _ = scene.addCamera(.{ .name = "bench_cam", .camera = cam }) catch {};

    var shared_opaque = z.PBRMaterial.init("bench_inst_opaque");
    var shared_trans = z.PBRMaterial.init("bench_inst_trans");
    shared_trans.alpha_mode = .blend;

    // Parent count derives from the same --meshes budget (total opaque
    // instances ~= --meshes); the transparent parent is a fixed bounded
    // extra covering the staging pair-sort path.
    const n_parents: usize = @max(8, cfg.meshes / inst_per_parent);
    const total_opaque_inst = n_parents * inst_per_parent;
    const total_inst = total_opaque_inst + transparent_instances;

    const parents = try alloc.alloc(z.Mesh, n_parents + 1);
    const imem = try alloc.alloc(z.InstancedMesh, total_inst);
    const iptrs = try alloc.alloc(*z.InstancedMesh, total_inst);
    for (0..total_inst) |i| iptrs[i] = &imem[i];

    const unit_box = z.BoundingBox.init(z.Vec3.new(-0.5, -0.5, -0.5), z.Vec3.new(0.5, 0.5, 0.5));
    const grid_w: usize = 64;
    var cursor: usize = 0;
    for (parents, 0..) |*m, p| {
        const is_trans_parent = p == n_parents;
        const count = if (is_trans_parent) transparent_instances else inst_per_parent;
        const gx: f32 = @floatFromInt(p % 16);
        const gz: f32 = @floatFromInt(p / 16);
        // The transparent parent sits at the origin so its pair-sort path
        // stays covered at any --meshes (a grid-tail slot would cull out).
        const px: f32 = if (is_trans_parent) 0 else (gx - 8.0) * 12.0;
        const pz: f32 = if (is_trans_parent) 0 else (gz - 4.0) * 12.0;
        m.* = .{
            .name = "bench_inst",
            .vertex_buffer = .{},
            .index_buffer = .{},
            .index_count = 36,
            .position = z.Vec3.new(px, 0, pz),
            .local_bounding_box = unit_box,
        };
        if (is_trans_parent) {
            m.material = .{ .pbr = &shared_trans };
        } else {
            m.material = .{ .pbr = &shared_opaque };
        }
        const base_x = m.position.x;
        const base_z = m.position.z;
        for (0..count) |k| {
            const gi = cursor + k;
            const fi: f32 = @floatFromInt(gi);
            const lx: f32 = @floatFromInt(k % grid_w);
            const lz: f32 = @floatFromInt(k / grid_w);
            imem[gi] = .{
                .name = "bi",
                .source_mesh = m,
                .position = z.Vec3.new(base_x + (lx - 32.0) * 1.5, 0, base_z + (lz - 2.0) * 1.5),
                .rotation = z.Vec3.new(0, @mod(fi * 37.0, 360.0), 0),
                .scaling = z.Vec3.new(1, 1, 1),
                .is_visible = gi % 32 != 0,
            };
        }
        m.instances = .{ .items = iptrs[cursor .. cursor + count], .capacity = count };
        cursor += count;
        try scene.meshes.append(alloc, m);
    }

    for (0..cfg.warmup) |_| {
        scene.publishFrameSnapshot(16.0 / 9.0, 1280, 720);
        if (!scene.buildPreparedFrame()) std.debug.panic("bench warmup build failed", .{});
    }
    const warm_sum = checksumInstancedBack(&scene);

    const times = try alloc.alloc(f64, cfg.frames);
    defer alloc.free(times);
    var sum: f64 = 0;
    for (0..cfg.frames) |f| {
        scene.publishFrameSnapshot(16.0 / 9.0, 1280, 720);
        const t0 = sokol.time.now();
        if (!scene.buildPreparedFrame()) std.debug.panic("bench build failed at frame {d}", .{f});
        const dt_ms: f64 = @floatCast(sokol.time.ms(sokol.time.now() -% t0));
        times[f] = dt_ms;
        sum += dt_ms;
    }
    const sum_copy = sum;
    const med = median(times);
    var mn = times[0];
    var mx = times[0];
    for (times) |t| {
        mn = @min(mn, t);
        mx = @max(mx, t);
    }
    const sum2 = checksumInstancedBack(&scene);
    if (sum2 != warm_sum and cfg.warmup > 0) {
        std.debug.panic("bench: checksum drift warm={x} final={x} (queues must be deterministic)", .{ warm_sum, sum2 });
    }

    const bidx = scene.build_slot.load(.monotonic) % scene.draws.slots.len;
    const fslot = &scene.draws.slots[bidx];
    var inst_count: u32 = 0;
    for (fslot.primary.opaque_instanced.items) |*b| inst_count += b.visible_instance_count;
    var tinst_staged: usize = 0;
    for (fslot.primary.transparent_instanced.items) |*b| tinst_staged += b.visible_instance_count;
    const res = CaseResult{
        .median_ms = med,
        .min_ms = mn,
        .max_ms = mx,
        .total_ms = sum_copy,
        .checksum = sum2,
        .opaque_count = fslot.primary.items.items.len,
        .transparent = fslot.primary.transparent.items.len,
        .rendered = fslot.build_stats.rendered_meshes,
        .culled = fslot.build_stats.culled_meshes,
        .inst_batches = fslot.primary.opaque_instanced.items.len,
        .instanced_count = inst_count,
        .tinst_batches = fslot.primary.transparent_instanced.items.len,
        .tinst_staged = tinst_staged,
        .n_parents = parents.len,
        .generated_instances = total_inst,
        .staged_matrices = fslot.primary.instance_matrices.items.len,
    };

    scene.draws.deinit(alloc);
    scene.meshes.deinit(alloc);
    scene.cameras.deinit(alloc);
    scene.profiler.deinit();
    alloc.free(parents);
    alloc.free(imem);
    alloc.free(iptrs);
    return res;
}

pub fn main(init: std.process.Init) !void {
    sokol.time.setup();
    const alloc = std.heap.smp_allocator;

    var arg_buf: [32][]const u8 = undefined;
    var n_args: usize = 0;
    var it = try std.process.Args.Iterator.initAllocator(init.minimal.args, alloc);
    defer it.deinit();
    _ = it.next();
    while (it.next()) |a| {
        if (n_args < arg_buf.len) {
            arg_buf[n_args] = a;
            n_args += 1;
        }
    }
    const cfg = parseArgs(arg_buf[0..n_args]);

    std.debug.print("scene_scaling_bench: meshes={d} frames={d} warmup={d} allocator=smp clock=sokol.time units=ms\n", .{
        cfg.meshes, cfg.frames, cfg.warmup,
    });

    const run_ident = cfg.mode == .identical or cfg.mode == .both or cfg.mode == .all;
    const run_dist = cfg.mode == .distinct or cfg.mode == .both or cfg.mode == .all;
    const run_inst = cfg.mode == .instanced or cfg.mode == .all;

    if (run_ident) {
        const r = try runCase(false, cfg, alloc);
        const per_mesh_ns = r.median_ms * 1e6 / @as(f64, @floatFromInt(cfg.meshes));
        std.debug.print("case=many-identical n={d} opaque={d} transparent={d} rendered={d} culled={d} checksum={x}\n", .{
            cfg.meshes, r.opaque_count, r.transparent, r.rendered, r.culled, r.checksum,
        });
        std.debug.print("case=many-identical median={d:.3}ms min={d:.3} max={d:.3} total={d:.1}ms per_mesh={d:.0}ns\n", .{
            r.median_ms, r.min_ms, r.max_ms, r.total_ms, per_mesh_ns,
        });
        if (r.total_ms < 100.0) std.debug.print("warn: case total {d:.1}ms < 100ms; raise --meshes/--frames\n", .{r.total_ms});
    }
    if (run_dist) {
        const r = try runCase(true, cfg, alloc);
        const per_mesh_ns = r.median_ms * 1e6 / @as(f64, @floatFromInt(cfg.meshes));
        std.debug.print("case=many-distinct n={d} opaque={d} transparent={d} rendered={d} culled={d} checksum={x}\n", .{
            cfg.meshes, r.opaque_count, r.transparent, r.rendered, r.culled, r.checksum,
        });
        std.debug.print("case=many-distinct median={d:.3}ms min={d:.3} max={d:.3} total={d:.1}ms per_mesh={d:.0}ns\n", .{
            r.median_ms, r.min_ms, r.max_ms, r.total_ms, per_mesh_ns,
        });
        if (r.total_ms < 100.0) std.debug.print("warn: case total {d:.1}ms < 100ms; raise --meshes/--frames\n", .{r.total_ms});
    }
    if (run_inst) {
        const r = try runCaseInstanced(cfg, alloc);
        // Whole-build amortized cost per REAL staged matrix (pre-cull
        // staging output), NOT per visible instance and NOT an isolated
        // TRS/sort attribution.
        const per_staged_ns = r.median_ms * 1e6 / @as(f64, @floatFromInt(@max(1, r.staged_matrices)));
        const queued_total: usize = @as(usize, r.instanced_count) + r.tinst_staged;
        std.debug.print("case=many-instanced parents={d} generated_instances={d} staged_matrices={d} visible_batches={d} opaque_inst={d} tinst={d} queued_instances={d} opaque={d} transparent={d} rendered={d} culled={d} checksum={x}\n", .{
            r.n_parents, r.generated_instances, r.staged_matrices, r.inst_batches + r.tinst_batches, r.inst_batches, r.tinst_batches, queued_total, r.opaque_count, r.transparent, r.rendered, r.culled, r.checksum,
        });
        std.debug.print("case=many-instanced median={d:.3}ms min={d:.3} max={d:.3} total={d:.1}ms per_staged={d:.0}ns (whole buildPreparedFrame / staged matrix)\n", .{
            r.median_ms, r.min_ms, r.max_ms, r.total_ms, per_staged_ns,
        });
        if (r.total_ms < 100.0) std.debug.print("warn: case total {d:.1}ms < 100ms; raise --meshes/--frames\n", .{r.total_ms});
    }
}
