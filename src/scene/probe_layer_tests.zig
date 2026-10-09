const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const probe_layer = @import("probe_layer.zig");
const ProbeLayer = probe_layer.ProbeLayer;
const Face = probe_layer.Face;
const faceDir = probe_layer.faceDir;
const faceUp = probe_layer.faceUp;
const faceView = probe_layer.faceView;
const mipSize = probe_layer.mipSize;
const max_mips = probe_layer.max_mips;
const max_lod = probe_layer.max_lod;
const bakeViewDesc = probe_layer.bakeViewDesc;
const selectProbe = probe_layer.selectProbe;
const captureFar = probe_layer.captureFar;
const targetBytes = probe_layer.targetBytes;
const ProbeGpu = probe_layer.ProbeGpu;
const ProbeFrameEntry = probe_layer.ProbeFrameEntry;

test "add fills slots and caps at four with a hard error" {
    var layer = ProbeLayer{};
    try std.testing.expectEqual(@as(usize, 0), layer.count);

    const a = try layer.add(Vec3.new(1, 2, 3), .{});
    const b = try layer.add(Vec3.zero, .{ .radius = 5.0, .enabled = false });
    try std.testing.expectEqual(@as(usize, 0), a);
    try std.testing.expectEqual(@as(usize, 1), b);
    try std.testing.expectEqual(@as(usize, 2), layer.count);
    try std.testing.expectEqual(Vec3.new(1, 2, 3), layer.probes[0].position);
    try std.testing.expectEqual(@as(f32, 10.0), layer.probes[0].radius);
    try std.testing.expectEqual(@as(f32, 5.0), layer.probes[1].radius);
    try std.testing.expect(!layer.probes[1].enabled);
    // New probes start dirty (scheduled) but uncaptured (selection falls
    // back until the first capture lands).
    try std.testing.expect(layer.probes[0].dirty);
    try std.testing.expect(!layer.probes[0].captured);

    _ = try layer.add(Vec3.zero, .{});
    _ = try layer.add(Vec3.zero, .{});
    try std.testing.expectEqual(@as(usize, 4), layer.count);
    try std.testing.expectError(error.TooManyReflectionProbes, layer.add(Vec3.zero, .{}));
    try std.testing.expectEqual(@as(usize, 4), layer.count);
}

test "remove retires the gpu target and keeps index order" {
    // Fake retire queue (duck-typed like GpuRetireQueue.retireProbeTarget):
    // records the retired payload without touching sg or imports.
    const FakeRetire = struct {
        calls: u32 = 0,
        pub fn retireProbeTarget(self: *@This(), allocator: std.mem.Allocator, gpu: ProbeGpu) void {
            _ = allocator;
            _ = gpu;
            self.calls += 1;
        }
    };
    const alloc = std.testing.allocator;
    var layer = ProbeLayer{};
    _ = try layer.add(Vec3.new(1, 0, 0), .{});
    _ = try layer.add(Vec3.new(2, 0, 0), .{});
    _ = try layer.add(Vec3.new(3, 0, 0), .{});

    var fake = FakeRetire{};
    // Out-of-range removal is a no-op (never retires).
    layer.remove(alloc, &fake, 9);
    try std.testing.expectEqual(@as(u32, 0), fake.calls);
    try std.testing.expectEqual(@as(usize, 3), layer.count);

    layer.remove(alloc, &fake, 1);
    try std.testing.expectEqual(@as(u32, 1), fake.calls);
    try std.testing.expectEqual(@as(usize, 2), layer.count);
    // Order-preserving: the tail shifted down (index 1 now holds x=3).
    try std.testing.expectEqual(@as(f32, 1.0), layer.probes[0].position.x);
    try std.testing.expectEqual(@as(f32, 3.0), layer.probes[1].position.x);
    // Removal always retires (even an empty pre-capture target: the entry
    // is a no-op destroy, but the discipline stays uniform).
    layer.remove(alloc, &fake, 0);
    try std.testing.expectEqual(@as(u32, 2), fake.calls);
    try std.testing.expectEqual(@as(usize, 1), layer.count);
}

test "dirty scheduling serves the lowest dirty enabled index" {
    var layer = ProbeLayer{};
    _ = try layer.add(Vec3.zero, .{});
    _ = try layer.add(Vec3.zero, .{ .enabled = false });
    _ = try layer.add(Vec3.zero, .{});
    // Fresh probes all start dirty.
    try std.testing.expectEqual(@as(usize, 3), layer.dirtyCount());
    // Index 1 is disabled: scheduling skips it even though it is dirty.
    try std.testing.expectEqual(@as(usize, 0), layer.nextDirtyIndex().?);

    layer.notifyCaptured(0);
    try std.testing.expect(!layer.probes[0].dirty);
    try std.testing.expect(layer.probes[0].captured);
    try std.testing.expectEqual(@as(usize, 2), layer.dirtyCount());
    // One capture per frame: the next frame takes index 2, not both.
    try std.testing.expectEqual(@as(usize, 2), layer.nextDirtyIndex().?);
    layer.notifyCaptured(2);
    // Only the disabled probe is still dirty — nothing schedulable.
    try std.testing.expectEqual(@as(?usize, null), layer.nextDirtyIndex());
    try std.testing.expectEqual(@as(usize, 1), layer.dirtyCount());

    // On-demand recapture of a single probe.
    layer.markDirty(0);
    try std.testing.expectEqual(@as(usize, 0), layer.nextDirtyIndex().?);
    layer.markAllDirty();
    try std.testing.expectEqual(@as(usize, 3), layer.dirtyCount());
    try std.testing.expectEqual(@as(usize, 0), layer.nextDirtyIndex().?);
    // Out-of-range marks are no-ops.
    layer.markDirty(42);
    layer.notifyCaptured(42);
    try std.testing.expectEqual(@as(usize, 3), layer.dirtyCount());
}

test "selectProbe picks the nearest enabled captured probe in radius" {
    const entry = ProbeFrameEntry{
        .position = Vec3.zero,
        .radius = 10.0,
        .enabled = true,
        .captured = true,
        .intensity = 0.5,
        .view = .{ .id = 7 },
        .sampler = .{ .id = 8 },
    };
    // No probes: no selection (caller takes the legacy path).
    try std.testing.expect(selectProbe(&.{}, Vec3.zero) == null);

    // Inside the radius: wins, carrying intensity/view/sampler.
    const one = [_]ProbeFrameEntry{entry};
    const sel = selectProbe(&one, Vec3.new(3, 4, 0)).?;
    try std.testing.expectEqual(@as(usize, 0), sel.index);
    try std.testing.expectEqual(@as(f32, 0.5), sel.intensity);
    try std.testing.expectEqual(@as(u32, 7), sel.view.id);
    try std.testing.expectEqual(@as(u32, 8), sel.sampler.id);

    // Outside the radius: falls back (5-4-0 is at distance 5; use 11).
    try std.testing.expect(selectProbe(&one, Vec3.new(11, 0, 0)) == null);

    // Disabled, uncaptured, or view-less entries never win.
    var off = entry;
    off.enabled = false;
    try std.testing.expect(selectProbe(&[_]ProbeFrameEntry{off}, Vec3.zero) == null);
    off = entry;
    off.captured = false;
    try std.testing.expect(selectProbe(&[_]ProbeFrameEntry{off}, Vec3.zero) == null);
    off = entry;
    off.view = .{};
    try std.testing.expect(selectProbe(&[_]ProbeFrameEntry{off}, Vec3.zero) == null);

    // Nearest wins; exact-distance ties resolve to the lowest index.
    var near = entry;
    near.position = Vec3.new(2, 0, 0);
    var far = entry;
    far.position = Vec3.new(-9, 0, 0);
    const two = [_]ProbeFrameEntry{ far, near };
    try std.testing.expectEqual(@as(usize, 1), selectProbe(&two, Vec3.zero).?.index);
    var tie_a = entry;
    tie_a.position = Vec3.new(5, 0, 0);
    var tie_b = entry;
    tie_b.position = Vec3.new(-5, 0, 0);
    const ties = [_]ProbeFrameEntry{ tie_a, tie_b };
    try std.testing.expectEqual(@as(usize, 0), selectProbe(&ties, Vec3.zero).?.index);
}

test "packFrame mirrors probe state for the snapshot" {
    var layer = ProbeLayer{};
    _ = try layer.add(Vec3.new(1, 2, 3), .{ .radius = 4.0, .intensity = 0.75 });
    _ = try layer.add(Vec3.zero, .{ .enabled = false });

    // Pre-capture: nothing usable (no GPU target), so selection stays off.
    var pack = layer.packFrame();
    try std.testing.expectEqual(@as(usize, 2), pack.count);
    try std.testing.expectEqual(Vec3.new(1, 2, 3), pack.entries[0].position);
    try std.testing.expectEqual(@as(f32, 4.0), pack.entries[0].radius);
    try std.testing.expectEqual(@as(f32, 0.75), pack.entries[0].intensity);
    try std.testing.expect(pack.entries[0].enabled);
    try std.testing.expect(!pack.entries[0].captured);
    try std.testing.expect(!pack.entries[1].enabled);
    // Disabled-probe path leaves state untouched: selection finds nothing.
    try std.testing.expect(selectProbe(pack.entries[0..pack.count], Vec3.new(1, 2, 3)) == null);

    // Simulate a landed capture (GPU target present): the entry flips to
    // usable with the full LOD range, and selection engages.
    layer.probes[0].captured = true;
    layer.probes[0].gpu.valid = true;
    layer.probes[0].gpu.tex_view = .{ .id = 11 };
    layer.probes[0].gpu.sampler = .{ .id = 12 };
    pack = layer.packFrame();
    try std.testing.expect(pack.entries[0].captured);
    try std.testing.expectEqual(max_lod, pack.entries[0].max_probe_lod);
    const sel = selectProbe(pack.entries[0..pack.count], Vec3.new(1, 2, 3)).?;
    try std.testing.expectEqual(@as(usize, 0), sel.index);
    try std.testing.expectEqual(@as(u32, 11), sel.view.id);
}

test "face tables cover the six cube axes with valid up vectors" {
    const faces = [_]Face{ .pos_x, .neg_x, .pos_y, .neg_y, .pos_z, .neg_z };
    const want_dirs = [_]Vec3{
        Vec3.new(1, 0, 0), Vec3.new(-1, 0, 0),
        Vec3.new(0, 1, 0), Vec3.new(0, -1, 0),
        Vec3.new(0, 0, 1), Vec3.new(0, 0, -1),
    };
    for (faces, 0..) |f, i| {
        // Unit axis directions in sokol cube-slice order.
        try std.testing.expectEqual(want_dirs[i], faceDir(f));
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), faceDir(f).length(), 1e-6);
        // No face looks parallel to its up (lookAt stays well-defined).
        try std.testing.expectApproxEqAbs(@as(f32, 0.0), faceDir(f).dot(faceUp(f)), 1e-6);
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), faceUp(f).length(), 1e-6);
    }
    // Every face view looks down camera -Z along its own axis: transforming
    // the face direction by its view matrix yields (0,0,-1), and the eye
    // maps to the origin.
    for (faces) |f| {
        const view = faceView(f, Vec3.new(5, -3, 2));
        const fwd = view.transformDirection(faceDir(f));
        try std.testing.expectApproxEqAbs(@as(f32, 0.0), fwd.x, 1e-5);
        try std.testing.expectApproxEqAbs(@as(f32, 0.0), fwd.y, 1e-5);
        try std.testing.expectApproxEqAbs(@as(f32, -1.0), fwd.z, 1e-5);
        try std.testing.expectEqual(Vec3.zero, view.transformPoint(Vec3.new(5, -3, 2)));
    }
}

test "captureFar scales with radius above a usable floor" {
    try std.testing.expectEqual(@as(f32, 50.0), captureFar(1.0));
    try std.testing.expectEqual(@as(f32, 50.0), captureFar(6.25));
    try std.testing.expectEqual(@as(f32, 80.0), captureFar(10.0));
    try std.testing.expectEqual(@as(i32, 128), mipSize(0));
    try std.testing.expectEqual(@as(i32, 64), mipSize(1));
    try std.testing.expectEqual(@as(i32, 1), mipSize(7));
    try std.testing.expectEqual(@as(i32, 1), mipSize(max_mips));
}

test "targetBytes accounts the HDR cube chain plus depth" {
    // RGBA16F cube 128..1 over 8 mips, all 6 faces, plus one 128x128 depth:
    // (16384+4096+1024+256+64+16+4+1)*8*6 + 128*128*4 = 1114096.
    try std.testing.expectEqual(@as(usize, 1114096), targetBytes());
}

test "probe convolution source view excludes output mips" {
    try @import("ibl_tests.zig").expectBakeSourcePlan(bakeViewDesc(.{ .id = 77 }));
}

test "ensureGpu fails closed without a gpu context" {
    // Headless (no sg context): no creation, dirty retained, no crash.
    var layer = ProbeLayer{};
    _ = try layer.add(Vec3.zero, .{});
    try std.testing.expect(!layer.ensureGpu(0));
    try std.testing.expect(layer.probes[0].dirty);
    try std.testing.expect(!layer.probes[0].gpu.valid);
    try std.testing.expect(!layer.ensureGpu(42));
}
