//! Shader-material prewarm tests: context-side warming BEFORE the pass opens,
//! lookup-only draws inside it. Headless throughout: real collection +
//! slot-management code runs against the real cache, while GPU creation is
//! driven through DummyCreator trace hooks (asserting creation never runs
//! with a pass open) with fake handle ids — no sg context required.

const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const math = @import("math");
const Mat4 = math.Mat4;

const forward_pipelines = @import("forward_pipelines.zig");
const ShaderMaterialCache = forward_pipelines.ShaderMaterialCache;
const ShaderMaterialSet = forward_pipelines.ShaderMaterialSet;
const PrewarmFilter = forward_pipelines.PrewarmFilter;
const collectShaderMaterialKeys = forward_pipelines.collectShaderMaterialKeys;
const shaderMaterialKeyForShape = forward_pipelines.shaderMaterialKeyForShape;
const render_queue = @import("render_queue.zig");
const RenderQueues = render_queue.RenderQueues;
const RenderMeshItem = render_queue.RenderMeshItem;
const material_mod = @import("../material.zig");
const shader_material = @import("../shader_material.zig");
const TargetShape = @import("../target_shape.zig").TargetShape;

// ---------------------------------------------------------------------------
// Fixtures.
// ---------------------------------------------------------------------------

fn dummyMakeShader(backend: sg.Backend) sg.Shader {
    _ = backend;
    return .{};
}

/// Second distinct registration key (ramp_wave is the static one). Runtime
/// registration is context-thread-only by contract; headless tests satisfy
/// that (no sg context). Idempotent across tests in this file.
fn dummyBIndex() !u32 {
    if (shader_material.indexForName("prewarm_dummy_b")) |i| return i;
    return try shader_material.registerRuntime(.{
        .name = "prewarm_dummy_b",
        .make_shader = &dummyMakeShader,
    });
}

fn rampIndex() !u32 {
    return shader_material.indexForName("ramp_wave") orelse error.TestUnexpectedResult;
}

fn rampKey() u64 {
    return shader_material.keyForName("ramp_wave");
}

fn snapFor(entry_index: u32) material_mod.ShaderDrawSnapshot {
    return .{ .entry_index = entry_index };
}

fn shaderItem(shader_storage_idx: u32) RenderMeshItem {
    return .{
        .model = Mat4.identity,
        .distance_sq = 1.0,
        .texture_id = 0,
        .index_count = 6,
        .shader_index = shader_storage_idx,
    };
}

/// Queues with two shader snapshots (0 = ramp_wave, 1 = dummy B) for tests
/// to reference. Caller owns; deinit when done.
fn twoSnapQueues(gpa: std.mem.Allocator) !RenderQueues {
    var q = RenderQueues{};
    errdefer q.deinit(gpa);
    try q.shader_storage.append(gpa, snapFor(try rampIndex()));
    try q.shader_storage.append(gpa, snapFor(try dummyBIndex()));
    return q;
}

// ---------------------------------------------------------------------------
// Dummy SG + creator trace hooks.
// ---------------------------------------------------------------------------

/// Dummy GPU: records resource creation and whether a pass is open. Fills
/// assert the outside-pass contract (creation with an open pass is a test
/// failure, mirroring the production rule that getOrCreate must never run
/// mid-pass).
const DummySg = struct {
    pass_open: bool = false,
    shader_makes: u32 = 0,
    pipeline_makes: u32 = 0,
    next_id: u32 = 1000,

    fn takeId(self: *DummySg) u32 {
        defer self.next_id += 1;
        return self.next_id;
    }
};

const DummyCreator = struct {
    sg: *DummySg,
    fail_keys: []const u64 = &.{},

    pub fn headlessOk() bool {
        return true;
    }

    pub fn create(self: DummyCreator, cache: *ShaderMaterialCache, key: u64) ?*const ShaderMaterialSet {
        return self.createInner(cache, key);
    }

    fn createInner(self: DummyCreator, cache: *ShaderMaterialCache, key: u64) ?*const ShaderMaterialSet {
        // Outside-pass contract: the production wiring warms before
        // beginPass; any creation under an open pass fails the test run.
        std.debug.assert(!self.sg.pass_open);
        // Invalid registry: no slot, no calls (mirrors getOrCreate).
        if (shader_material.entryForKey(key) == null) return null;
        for (self.fail_keys) |fk| if (fk == key) return null; // scripted GPU failure
        const slot = cache.claimEmptySlot() orelse return null; // full: overflow bumped
        self.sg.shader_makes += 1;
        self.sg.pipeline_makes += 8;
        slot.shader = .{ .id = self.sg.takeId() };
        slot.opaque_u16 = .{ .id = self.sg.takeId() };
        slot.opaque_u32 = .{ .id = self.sg.takeId() };
        slot.blend_u16 = .{ .id = self.sg.takeId() };
        slot.blend_u32 = .{ .id = self.sg.takeId() };
        slot.ds_opaque_u16 = .{ .id = self.sg.takeId() };
        slot.ds_opaque_u32 = .{ .id = self.sg.takeId() };
        slot.ds_blend_u16 = .{ .id = self.sg.takeId() };
        slot.ds_blend_u32 = .{ .id = self.sg.takeId() };
        slot.key = shaderMaterialKeyForShape(key, cache.shape);
        return slot;
    }
};

// ---------------------------------------------------------------------------
// Collection: ordering, eligibility, filters.
// ---------------------------------------------------------------------------

test "prewarm collect follows first-use draw order and dedupes" {
    const gpa = std.testing.allocator;
    var q = try twoSnapQueues(gpa);
    defer q.deinit(gpa);

    // Opaque: A, B, A-dup, plain (no shader).
    try q.items.append(gpa, shaderItem(0));
    try q.items.append(gpa, shaderItem(1));
    try q.items.append(gpa, shaderItem(0));
    try q.items.append(gpa, .{
        .model = Mat4.identity,
        .distance_sq = 2.0,
        .texture_id = 0,
        .index_count = 6,
    });
    // Transparent back-to-front: B first, then A, then an instanced entry
    // (batches carry no shader index and must not disturb the order).
    try q.transparent.append(gpa, shaderItem(1));
    try q.transparent.append(gpa, shaderItem(0));
    try q.transparent_order.append(gpa, .{ .distance_sq = 9.0, .seq = 1, .kind = .regular, .index = 0, .is_decal = false });
    try q.transparent_order.append(gpa, .{ .distance_sq = 4.0, .seq = 0, .kind = .regular, .index = 1, .is_decal = false });
    try q.transparent_order.append(gpa, .{ .distance_sq = 1.0, .seq = 2, .kind = .instanced, .index = 0, .is_decal = false });

    var out: [32]u64 = undefined;
    const total = collectShaderMaterialKeys(&q, .{}, &out);
    try std.testing.expectEqual(@as(usize, 2), total);
    try std.testing.expectEqual(shader_material.keyForName("ramp_wave"), out[0]);
    try std.testing.expectEqual(shader_material.keyForName("prewarm_dummy_b"), out[1]);
}

test "prewarm collect skips skinned, empty, invalid and out-of-range items" {
    const gpa = std.testing.allocator;
    var q = try twoSnapQueues(gpa);
    defer q.deinit(gpa);
    // Snapshot 2 points at no registration (invalid_index is out of range).
    try q.shader_storage.append(gpa, .{ .entry_index = shader_material.invalid_index });

    var skinned = shaderItem(0);
    skinned.is_skinned = true;
    try q.items.append(gpa, skinned);
    var empty = shaderItem(0);
    empty.index_count = 0;
    try q.items.append(gpa, empty);
    try q.items.append(gpa, shaderItem(99)); // out-of-range snapshot
    try q.items.append(gpa, shaderItem(2)); // unregistered entry
    try q.items.append(gpa, shaderItem(0)); // the only eligible one

    var out: [32]u64 = undefined;
    const total = collectShaderMaterialKeys(&q, .{}, &out);
    try std.testing.expectEqual(@as(usize, 1), total);
    try std.testing.expectEqual(rampKey(), out[0]);
}

test "prewarm collect filters cover the refraction variant" {
    const gpa = std.testing.allocator;
    var q = try twoSnapQueues(gpa);
    defer q.deinit(gpa);

    var refr = shaderItem(0);
    refr.draw_record = .{ .refractive = true };
    try q.items.append(gpa, refr); // opaque refractive A
    try q.items.append(gpa, shaderItem(1)); // opaque plain B
    try q.transparent.append(gpa, shaderItem(0)); // transparent A
    try q.transparent_order.append(gpa, .{ .distance_sq = 5.0, .seq = 0, .kind = .regular, .index = 0, .is_decal = false });

    const key_a = rampKey();
    const key_b = shader_material.keyForName("prewarm_dummy_b");

    var out: [32]u64 = undefined;
    var total = collectShaderMaterialKeys(&q, .{}, &out);
    try std.testing.expectEqual(@as(usize, 2), total);
    try std.testing.expectEqual(key_a, out[0]);
    try std.testing.expectEqual(key_b, out[1]);

    // Refraction capture draws opaque non-refractive only.
    total = collectShaderMaterialKeys(&q, .{ .include_transparent = false, .include_refractive = false }, &out);
    try std.testing.expectEqual(@as(usize, 1), total);
    try std.testing.expectEqual(key_b, out[0]);

    // Dropping only refractive keeps the transparent A (draw order: B, A).
    total = collectShaderMaterialKeys(&q, .{ .include_refractive = false }, &out);
    try std.testing.expectEqual(@as(usize, 2), total);
    try std.testing.expectEqual(key_b, out[0]);
    try std.testing.expectEqual(key_a, out[1]);
}

test "prewarm collect reports totals past a short window in first-use order" {
    const gpa = std.testing.allocator;
    var q = try twoSnapQueues(gpa);
    defer q.deinit(gpa);
    try q.items.append(gpa, shaderItem(0));
    try q.items.append(gpa, shaderItem(1));

    var tiny: [1]u64 = undefined;
    const total = collectShaderMaterialKeys(&q, .{}, &tiny);
    try std.testing.expectEqual(@as(usize, 2), total);
    try std.testing.expectEqual(rampKey(), tiny[0]);
}

// ---------------------------------------------------------------------------
// Real cache: headless skip, slot occupancy, overflow, shape isolation.
// ---------------------------------------------------------------------------

test "prewarm headless skips with zero GPU work" {
    try std.testing.expect(!sg.isvalid());
    var cache = ShaderMaterialCache{};
    const keys = [_]u64{ rampKey(), 0xdead_beef };
    const stats = cache.prewarmKeys(&keys);
    try std.testing.expect(stats.skipped);
    try std.testing.expectEqual(@as(usize, 2), stats.requested);
    try std.testing.expectEqual(@as(u32, 0), stats.warmed + stats.reused + stats.failed);
    try std.testing.expectEqual(@as(u32, 0), cache.overflow_count);
    for (&cache.slots) |*slot| try std.testing.expectEqual(@as(u64, 0), slot.key);

    const gpa = std.testing.allocator;
    var q = try twoSnapQueues(gpa);
    defer q.deinit(gpa);
    try q.items.append(gpa, shaderItem(0));
    const qstats = cache.prewarmQueues(&q, .{});
    try std.testing.expect(qstats.skipped);
    try std.testing.expectEqual(@as(u32, 0), cache.overflow_count);
}

test "dummy prewarm warms outside the pass; first draw resolves; repeat creates nothing" {
    const gpa = std.testing.allocator;
    var q = try twoSnapQueues(gpa);
    defer q.deinit(gpa);
    try q.items.append(gpa, shaderItem(0));
    try q.items.append(gpa, shaderItem(1));
    try q.transparent.append(gpa, shaderItem(0));
    try q.transparent_order.append(gpa, .{ .distance_sq = 5.0, .seq = 0, .kind = .regular, .index = 0, .is_decal = false });

    var buf: [32]u64 = undefined;
    const total = collectShaderMaterialKeys(&q, .{}, &buf);
    try std.testing.expectEqual(@as(usize, 2), total);

    var cache = ShaderMaterialCache{};
    var dsg = DummySg{};
    const creator = DummyCreator{ .sg = &dsg };

    // Prewarm with the pass closed: 1 shader + 8 pipelines per key.
    const stats = cache.prewarmKeysWith(buf[0..total], creator);
    try std.testing.expectEqual(@as(usize, 2), stats.requested);
    try std.testing.expectEqual(@as(u32, 2), stats.warmed);
    try std.testing.expectEqual(@as(u32, 0), stats.reused + stats.failed);
    try std.testing.expectEqual(@as(u32, 2), dsg.shader_makes);
    try std.testing.expectEqual(@as(u32, 16), dsg.pipeline_makes);

    // Warmed first draw: the draw's lookup-only step resolves every key.
    try std.testing.expect(cache.lookup(rampKey()) != null);
    try std.testing.expect(cache.lookup(shader_material.keyForName("prewarm_dummy_b")) != null);

    // In-pass draws perform lookups only: zero new resource creation.
    dsg.pass_open = true;
    for (q.items.items) |*item| {
        if (item.shader_index) |si| {
            const snap = q.shader_storage.items[si];
            const entry = shader_material.entry(snap.entry_index).?;
            try std.testing.expect(cache.lookup(entry.key) != null);
        }
    }
    try std.testing.expectEqual(@as(u32, 2), dsg.shader_makes);
    try std.testing.expectEqual(@as(u32, 16), dsg.pipeline_makes);
    dsg.pass_open = false;

    // Repeated prewarm reuses everything: no new creation.
    const again = cache.prewarmKeysWith(buf[0..total], creator);
    try std.testing.expectEqual(@as(u32, 0), again.warmed);
    try std.testing.expectEqual(@as(u32, 2), again.reused);
    try std.testing.expectEqual(@as(u32, 2), dsg.shader_makes);
    try std.testing.expectEqual(@as(u32, 16), dsg.pipeline_makes);
}

test "full cache preserves first-use order and counts overflow" {
    var cache = ShaderMaterialCache{};
    var dsg = DummySg{};
    // Claim all 32 slots directly through the real occupancy path.
    var i: u64 = 0;
    while (i < ShaderMaterialCache.max_entries) : (i += 1) {
        const slot = cache.claimEmptySlot() orelse return error.TestUnexpectedResult;
        slot.shader = .{ .id = dsg.takeId() };
        slot.opaque_u16 = .{ .id = dsg.takeId() };
        slot.key = 0x5000 + i;
    }
    try std.testing.expect(cache.lookup(0) == null);
    try std.testing.expect(cache.claimEmptySlot() == null);
    try std.testing.expectEqual(@as(u32, 1), cache.overflow_count);

    // A queued key against the full cache fails without consuming slots.
    const gpa = std.testing.allocator;
    var q = try twoSnapQueues(gpa);
    defer q.deinit(gpa);
    try q.items.append(gpa, shaderItem(0));
    const creator = DummyCreator{ .sg = &dsg };
    const stats = cache.prewarmQueuesWith(&q, .{}, creator);
    try std.testing.expectEqual(@as(usize, 1), stats.requested);
    try std.testing.expectEqual(@as(u32, 0), stats.warmed + stats.reused);
    try std.testing.expectEqual(@as(u32, 1), stats.failed);
    try std.testing.expectEqual(@as(u32, 0), stats.dropped);
    try std.testing.expect(cache.lookup(rampKey()) == null);
}

test "invalid registry and failed creation consume no slots" {
    var cache = ShaderMaterialCache{};
    var dsg = DummySg{};
    const key_b = shader_material.keyForName("prewarm_dummy_b");
    const creator = DummyCreator{ .sg = &dsg, .fail_keys = &.{key_b} };
    const keys = [_]u64{ rampKey(), 0xdead_beef, key_b };
    const stats = cache.prewarmKeysWith(&keys, creator);
    try std.testing.expectEqual(@as(u32, 1), stats.warmed);
    try std.testing.expectEqual(@as(u32, 2), stats.failed);
    try std.testing.expectEqual(@as(u32, 1), dsg.shader_makes);
    try std.testing.expectEqual(@as(u32, 0), cache.overflow_count);
    try std.testing.expect(cache.lookup(rampKey()) != null);
    try std.testing.expect(cache.lookup(0xdead_beef) == null);
    try std.testing.expect(cache.lookup(key_b) == null);
    var used: usize = 0;
    for (&cache.slots) |*s| if (s.key != 0) {
        used += 1;
    };
    try std.testing.expectEqual(@as(usize, 1), used);
}

test "shape toggle isolates variants across format, depth, stencil and samples" {
    const key = rampKey();
    const base = TargetShape{
        .sample_count = 1,
        .color_format = .RGBA16F,
        .depth_format = .DEPTH,
        .stencil_format = .NONE,
    };
    const k_base = shaderMaterialKeyForShape(key, base);
    // Every complete-shape lane participates in the slot key.
    try std.testing.expect(shaderMaterialKeyForShape(key, .{ .sample_count = 4, .color_format = .RGBA16F, .depth_format = .DEPTH, .stencil_format = .NONE }) != k_base);
    try std.testing.expect(shaderMaterialKeyForShape(key, .{ .sample_count = 1, .color_format = .BGRA8, .depth_format = .DEPTH, .stencil_format = .NONE }) != k_base);
    try std.testing.expect(shaderMaterialKeyForShape(key, .{ .sample_count = 1, .color_format = .RGBA16F, .depth_format = .DEPTH_STENCIL, .stencil_format = .DEPTH_STENCIL }) != k_base);
    try std.testing.expectEqual(k_base, shaderMaterialKeyForShape(key, base));

    // Warmed under one shape, invisible under another: no cross-talk.
    var cache_a = ShaderMaterialCache{ .shape = base };
    var cache_b = ShaderMaterialCache{ .shape = .{
        .sample_count = 4,
        .color_format = .RGBA16F,
        .depth_format = .DEPTH,
        .stencil_format = .NONE,
    } };
    var dsg = DummySg{};
    const creator = DummyCreator{ .sg = &dsg };
    const keys = [_]u64{key};
    const sa = cache_a.prewarmKeysWith(&keys, creator);
    try std.testing.expectEqual(@as(u32, 1), sa.warmed);
    try std.testing.expect(cache_a.lookup(key) != null);
    try std.testing.expect(cache_b.lookup(key) == null);
}
