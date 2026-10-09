const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const Mat4 = math.Mat4;
const Ray = math.Ray;

const ui_mod = @import("../ui.zig");
const UICanvas = ui_mod.UICanvas;
const SceneStats = @import("stats.zig").SceneStats;

const sokol = @import("sokol");
const sg = sokol.gfx;

const gui3d = @import("gui3d_layer.zig");
const Gui3dLayer = gui3d.Gui3dLayer;
const Ui3dPanel = gui3d.Ui3dPanel;
const Ui3dTarget = gui3d.Ui3dTarget;
const max_panels = gui3d.max_panels;
const max_captures_per_frame = gui3d.max_captures_per_frame;
const max_canvas_dimension = gui3d.max_canvas_dimension;
const panelNormal = gui3d.panelNormal;
const panelRight = gui3d.panelRight;
const panelUp = gui3d.panelUp;
const panelCorners = gui3d.panelCorners;
const panelModel = gui3d.panelModel;
const intersectRayPanel = gui3d.intersectRayPanel;
const canvasPixelCoords = gui3d.canvasPixelCoords;
const targetBytes = gui3d.targetBytes;
const censusBytes = gui3d.censusBytes;

test "add fills slots and caps at four with a hard error" {
    const t = std.testing;
    var layer = Gui3dLayer{};
    defer layer.deinit(t.allocator);
    try t.expectEqual(@as(usize, 0), layer.panelCount());

    const a = try layer.add(t.allocator, "panel_a", Vec3.new(1, 2, 3), .{});
    const b = try layer.add(t.allocator, "panel_b", Vec3.zero, .{ .width = 4.0, .height = 2.0, .enabled = false });
    try t.expectEqual(@as(usize, 0), a);
    try t.expectEqual(@as(usize, 1), b);
    try t.expectEqual(@as(usize, 2), layer.panelCount());
    const pa = layer.get(0).?;
    try t.expectEqualStrings("panel_a", pa.name);
    try t.expect(pa.owns_name);
    try t.expectEqual(Vec3.new(1, 2, 3), pa.position);
    try t.expectEqual(@as(f32, 2.0), pa.width);
    try t.expectEqual(@as(f32, 1.0), pa.height);
    try t.expectEqual(@as(u32, 512), pa.canvas_width);
    // New panels start dirty (scheduled) but uncaptured (invisible until
    // the first capture lands).
    try t.expect(pa.dirty);
    try t.expect(!pa.captured);
    try t.expect(pa.canvas != null);
    const pb = layer.get(1).?;
    try t.expectEqual(@as(f32, 4.0), pb.width);
    try t.expect(!pb.enabled);

    _ = try layer.add(t.allocator, "c", Vec3.zero, .{});
    _ = try layer.add(t.allocator, "d", Vec3.zero, .{});
    try t.expectEqual(@as(usize, 4), layer.panelCount());
    try t.expectError(error.TooManyUi3dPanels, layer.add(t.allocator, "e", Vec3.zero, .{}));
    try t.expectEqual(@as(usize, 4), layer.panelCount());

    // Name lookup.
    try t.expectEqualStrings("panel_b", layer.getByName("panel_b").?.name);
    try t.expect(layer.getByName("missing") == null);
    try t.expect(layer.get(4) == null);
}

test "add rejects degenerate sizes and resolutions" {
    const t = std.testing;
    var layer = Gui3dLayer{};
    defer layer.deinit(t.allocator);
    try t.expectError(error.InvalidUi3dPanelSize, layer.add(t.allocator, "x", Vec3.zero, .{ .width = 0.0 }));
    try t.expectError(error.InvalidUi3dPanelSize, layer.add(t.allocator, "x", Vec3.zero, .{ .height = -1.0 }));
    try t.expectError(error.InvalidUi3dPanelSize, layer.add(t.allocator, "x", Vec3.zero, .{ .width = 4096.0 }));
    try t.expectError(error.InvalidUi3dPanelSize, layer.add(t.allocator, "x", Vec3.zero, .{ .canvas_width = 0 }));
    try t.expectError(
        error.InvalidUi3dPanelSize,
        layer.add(t.allocator, "x", Vec3.zero, .{ .canvas_width = max_canvas_dimension + 1 }),
    );
    try t.expectEqual(@as(usize, 0), layer.panelCount());
}

test "remove retires the gpu target and keeps index order" {
    // Fake retire queue (duck-typed like GpuRetireQueue.retireUi3dTarget):
    // records retired payloads without touching sg or imports.
    const FakeRetire = struct {
        calls: u32 = 0,
        buffers: u32 = 0,
        pub fn retireUi3dTarget(self: *@This(), allocator: std.mem.Allocator, target: Ui3dTarget) void {
            _ = allocator;
            _ = target;
            self.calls += 1;
        }
        pub fn retireBuffer(self: *@This(), allocator: std.mem.Allocator, buf: sg.Buffer) void {
            _ = allocator;
            _ = buf;
            self.buffers += 1;
        }
    };
    const t = std.testing;
    var layer = Gui3dLayer{};
    defer layer.deinit(t.allocator);
    _ = try layer.add(t.allocator, "one", Vec3.new(1, 0, 0), .{});
    _ = try layer.add(t.allocator, "two", Vec3.new(2, 0, 0), .{});
    _ = try layer.add(t.allocator, "three", Vec3.new(3, 0, 0), .{});

    var fake = FakeRetire{};
    // Out-of-range removal is a no-op (never retires).
    layer.remove(t.allocator, &fake, 9);
    try t.expectEqual(@as(u32, 0), fake.calls);
    try t.expectEqual(@as(usize, 3), layer.panelCount());

    layer.remove(t.allocator, &fake, 1);
    try t.expectEqual(@as(u32, 1), fake.calls);
    try t.expectEqual(@as(usize, 2), layer.panelCount());
    // Order-preserving: the tail shifted down (index 1 now holds x=3).
    try t.expectEqual(@as(f32, 1.0), layer.get(0).?.position.x);
    try t.expectEqualStrings("one", layer.get(0).?.name);
    try t.expectEqual(@as(f32, 3.0), layer.get(1).?.position.x);
    try t.expectEqualStrings("three", layer.get(1).?.name);
    // Removal always retires (even an empty pre-capture target).
    layer.remove(t.allocator, &fake, 0);
    try t.expectEqual(@as(u32, 2), fake.calls);
    try t.expectEqual(@as(usize, 1), layer.panelCount());
    // The surviving panel's canvas is intact (usable after a removal).
    layer.get(0).?.canvas.?.drawRect(0, 0, 10, 10, .{ .r = 1, .g = 1, .b = 1, .a = 1 });
    try t.expect(layer.get(0).?.canvas.?.vertices.items.len > 0);
}

test "dirty scheduling serves the lowest dirty enabled index" {
    const t = std.testing;
    var layer = Gui3dLayer{};
    defer layer.deinit(t.allocator);
    _ = try layer.add(t.allocator, "a", Vec3.zero, .{});
    _ = try layer.add(t.allocator, "b", Vec3.zero, .{ .enabled = false });
    _ = try layer.add(t.allocator, "c", Vec3.zero, .{});
    // Fresh panels all start dirty.
    try t.expectEqual(@as(usize, 3), layer.dirtyCount());
    // Index 1 is disabled: scheduling skips it even though it is dirty.
    try t.expectEqual(@as(usize, 0), layer.nextDirtyIndex().?);

    layer.notifyCaptured(0);
    try t.expect(!layer.get(0).?.dirty);
    try t.expect(layer.get(0).?.captured);
    try t.expectEqual(@as(usize, 2), layer.dirtyCount());
    // One capture per frame: the next frame takes index 2, not both.
    try t.expectEqual(@as(usize, 2), layer.nextDirtyIndex().?);
    try t.expectEqual(@as(usize, 1), max_captures_per_frame);
    layer.notifyCaptured(2);
    // Only the disabled panel is still dirty — nothing schedulable.
    try t.expectEqual(@as(?usize, null), layer.nextDirtyIndex());
    try t.expectEqual(@as(usize, 1), layer.dirtyCount());

    // On-demand recapture of a single panel.
    layer.markDirty(0);
    try t.expectEqual(@as(usize, 0), layer.nextDirtyIndex().?);
    layer.markAllDirty();
    try t.expectEqual(@as(usize, 3), layer.dirtyCount());
    try t.expectEqual(@as(usize, 0), layer.nextDirtyIndex().?);
    // Out-of-range marks are no-ops.
    layer.markDirty(42);
    layer.notifyCaptured(42);
    try t.expectEqual(@as(usize, 3), layer.dirtyCount());
}

test "quad corners follow position size and fixed yaw" {
    const t = std.testing;
    var panel = Ui3dPanel{
        .position = Vec3.new(10, 0, 0),
        .width = 4.0,
        .height = 2.0,
        .yaw_deg = 0.0,
    };
    // Yaw 0: right = +X, normal = +Z.
    try t.expectEqual(Vec3.new(1, 0, 0), panelRight(0.0));
    try t.expectEqual(Vec3.new(0, 0, 1), panelNormal(0.0));
    try t.expectEqual(Vec3.up, panelUp());
    const c = panelCorners(&panel);
    try t.expectEqual(Vec3.new(8, -1, 0), c[0]);
    try t.expectEqual(Vec3.new(12, -1, 0), c[1]);
    try t.expectEqual(Vec3.new(12, 1, 0), c[2]);
    try t.expectEqual(Vec3.new(8, 1, 0), c[3]);

    // Yaw 90: right = -Z, normal = +X (matches Mat4.rotationY columns).
    // Trig is approximate in f32 (cos(90°) ≈ -4.4e-8), so compare
    // component-wise with tolerance, including against rotationY itself.
    const r90 = panelRight(90.0);
    try t.expectApproxEqAbs(@as(f32, 0.0), r90.x, 1e-6);
    try t.expectApproxEqAbs(@as(f32, 0.0), r90.y, 1e-6);
    try t.expectApproxEqAbs(@as(f32, -1.0), r90.z, 1e-6);
    const n90 = panelNormal(90.0);
    try t.expectApproxEqAbs(@as(f32, 1.0), n90.x, 1e-6);
    try t.expectApproxEqAbs(@as(f32, 0.0), n90.y, 1e-6);
    try t.expectApproxEqAbs(@as(f32, 0.0), n90.z, 1e-6);
    const rot = Mat4.rotationY(90.0);
    const rr = rot.transformDirection(Vec3.new(1, 0, 0));
    try t.expectApproxEqAbs(r90.x, rr.x, 1e-6);
    try t.expectApproxEqAbs(r90.y, rr.y, 1e-6);
    try t.expectApproxEqAbs(r90.z, rr.z, 1e-6);
    const nn = rot.transformDirection(Vec3.new(0, 0, 1));
    try t.expectApproxEqAbs(n90.x, nn.x, 1e-6);
    try t.expectApproxEqAbs(n90.y, nn.y, 1e-6);
    try t.expectApproxEqAbs(n90.z, nn.z, 1e-6);
    panel.yaw_deg = 90.0;
    const c90 = panelCorners(&panel);
    try t.expectApproxEqAbs(@as(f32, 10.0), c90[0].x, 1e-5);
    try t.expectApproxEqAbs(@as(f32, -1.0), c90[0].y, 1e-5);
    try t.expectApproxEqAbs(@as(f32, 2.0), c90[0].z, 1e-5);
    try t.expectApproxEqAbs(@as(f32, 10.0), c90[1].x, 1e-5);
    try t.expectApproxEqAbs(@as(f32, -1.0), c90[1].y, 1e-5);
    try t.expectApproxEqAbs(@as(f32, -2.0), c90[1].z, 1e-5);

    // The model matrix maps the unit quad onto the same corners.
    panel.yaw_deg = 0.0;
    const m = panelModel(&panel);
    const local = [4]Vec3{
        Vec3.new(-0.5, -0.5, 0),
        Vec3.new(0.5, -0.5, 0),
        Vec3.new(0.5, 0.5, 0),
        Vec3.new(-0.5, 0.5, 0),
    };
    for (local, 0..) |lp, i| {
        const w = m.transformPoint(lp);
        try t.expectApproxEqAbs(c[i].x, w.x, 1e-5);
        try t.expectApproxEqAbs(c[i].y, w.y, 1e-5);
        try t.expectApproxEqAbs(c[i].z, w.z, 1e-5);
    }
}

test "ray-plane-UV picking hits misses and back faces" {
    const t = std.testing;
    var panel = Ui3dPanel{
        .position = Vec3.new(0, 0, 5),
        .width = 2.0,
        .height = 2.0,
        .yaw_deg = 180.0, // normal -Z, facing a camera at the origin
        .canvas_width = 512,
        .canvas_height = 256,
    };
    // Dead-on front hit: UV center.
    const front = intersectRayPanel(Ray.new(Vec3.zero, Vec3.new(0, 0, 1)), &panel).?;
    try t.expectApproxEqAbs(@as(f32, 0.5), front.u, 1e-6);
    try t.expectApproxEqAbs(@as(f32, 0.5), front.v, 1e-6);
    try t.expectApproxEqAbs(@as(f32, 5.0), front.t, 1e-5);

    // Off-center: at yaw 180 local +X points at world -X, so world x = +1
    // maps to u = 0 (canvas left) and world y = +1 to v = 0 (canvas top).
    const edge = intersectRayPanel(Ray.new(Vec3.new(1, 1, 0), Vec3.new(0, 0, 1)), &panel).?;
    try t.expectApproxEqAbs(@as(f32, 0.0), edge.u, 1e-5);
    try t.expectApproxEqAbs(@as(f32, 0.0), edge.v, 1e-5);

    // Misses: outside the rect, parallel to the plane, pointing away.
    try t.expect(intersectRayPanel(Ray.new(Vec3.new(3, 0, 0), Vec3.new(0, 0, 1)), &panel) == null);
    try t.expect(intersectRayPanel(Ray.new(Vec3.zero, Vec3.new(1, 0, 0)), &panel) == null);
    try t.expect(intersectRayPanel(Ray.new(Vec3.zero, Vec3.new(0, 0, -1)), &panel) == null);
    // Degenerate panel: no hit.
    panel.width = 0.0;
    try t.expect(intersectRayPanel(Ray.new(Vec3.zero, Vec3.new(0, 0, 1)), &panel) == null);
    panel.width = 2.0;

    // Back face (double-sided draw): a ray from behind hits with the same
    // UVs instead of being rejected.
    const back = intersectRayPanel(Ray.new(Vec3.new(0, 0, 10), Vec3.new(0, 0, -1)), &panel).?;
    try t.expectApproxEqAbs(@as(f32, 0.5), back.u, 1e-6);
    try t.expectApproxEqAbs(@as(f32, 0.5), back.v, 1e-6);
    try t.expectApproxEqAbs(@as(f32, 5.0), back.t, 1e-5);

    // Canvas mapping: center hit lands mid-canvas.
    const px = canvasPixelCoords(front.u, front.v, panel.canvas_width, panel.canvas_height);
    try t.expectApproxEqAbs(@as(f32, 256.0), px.x, 1e-4);
    try t.expectApproxEqAbs(@as(f32, 128.0), px.y, 1e-4);
}

test "layer pick selects the nearest captured panel and skips the rest" {
    const t = std.testing;
    var layer = Gui3dLayer{};
    defer layer.deinit(t.allocator);
    _ = try layer.add(t.allocator, "near", Vec3.new(0, 0, 3), .{ .width = 4, .height = 4, .yaw_deg = 180 });
    _ = try layer.add(t.allocator, "far", Vec3.new(0, 0, 6), .{ .width = 8, .height = 8, .yaw_deg = 180 });
    _ = try layer.add(t.allocator, "off", Vec3.zero, .{ .enabled = false });
    // Nothing captured yet: no pick (invisible until first capture).
    const ray = Ray.new(Vec3.zero, Vec3.new(0, 0, 1));
    try t.expect(layer.pick(ray) == null);

    layer.notifyCaptured(0);
    layer.notifyCaptured(1);
    layer.notifyCaptured(2); // disabled: still skipped
    const hit = layer.pick(ray).?;
    try t.expectEqual(@as(usize, 0), hit.panel_index);
    try t.expectApproxEqAbs(@as(f32, 3.0), hit.t, 1e-5);

    // Disabling the winner falls through to the next panel.
    layer.get(0).?.enabled = false;
    const hit2 = layer.pick(ray).?;
    try t.expectEqual(@as(usize, 1), hit2.panel_index);
}

test "injectPointer routes into the canvas input state" {
    const t = std.testing;
    var layer = Gui3dLayer{};
    defer layer.deinit(t.allocator);
    _ = try layer.add(t.allocator, "p", Vec3.zero, .{});
    const panel = layer.get(0).?;
    // Untouched canvas: parked offscreen, nothing pressed.
    try t.expect(!panel.canvas.?.mouse_down);
    try t.expect(!panel.canvas.?.mouse_clicked);

    panel.injectPointer(100.0, 50.0, true);
    try t.expectEqual([2]f32{ 100.0, 50.0 }, panel.canvas.?.mouse_pos);
    try t.expect(panel.canvas.?.mouse_down);
    try t.expect(panel.canvas.?.mouse_clicked);

    // A widget drawn after the inject sees hover + press (button reacts).
    const hovered = UICanvas.isPointInRect(
        panel.canvas.?.mouse_pos[0],
        panel.canvas.?.mouse_pos[1],
        90.0,
        40.0,
        60.0,
        30.0,
    );
    try t.expect(hovered);
    panel.canvas.?.drawButton("ok", 90.0, 40.0, 60.0, 30.0, 14.0, hovered, panel.canvas.?.mouse_down);
    try t.expect(panel.canvas.?.vertices.items.len > 0);

    // Release clears the buttons, keeps the hover position.
    panel.injectRelease();
    try t.expectEqual([2]f32{ 100.0, 50.0 }, panel.canvas.?.mouse_pos);
    try t.expect(!panel.canvas.?.mouse_down);
    try t.expect(!panel.canvas.?.mouse_clicked);
}

test "gpu entry points fail closed headless with dirty retained" {
    const t = std.testing;
    var layer = Gui3dLayer{};
    defer layer.deinit(t.allocator);
    _ = try layer.add(t.allocator, "p", Vec3.zero, .{});
    // No sokol context: no creation, dirty retained, no crash.
    try t.expect(!layer.ensureGpu(t.allocator, 0));
    try t.expect(layer.get(0).?.dirty);
    try t.expect(!layer.get(0).?.gpu.valid);
    try t.expect(!layer.ensureGpu(t.allocator, 42));

    const FakeRetire = struct {
        pub fn retireUi3dTarget(self: *@This(), allocator: std.mem.Allocator, target: Ui3dTarget) void {
            _ = self;
            _ = allocator;
            _ = target;
        }
        pub fn retireBuffer(self: *@This(), allocator: std.mem.Allocator, buf: sg.Buffer) void {
            _ = self;
            _ = allocator;
            _ = buf;
        }
    };
    var fake = FakeRetire{};
    try t.expect(!layer.capturePanel(t.allocator, &fake, 0));
    try t.expect(layer.get(0).?.dirty);
    try t.expect(!layer.get(0).?.captured);

    // Drawing with nothing drawable is a pure-CPU early-out (no context
    // needed, no crash): uncaptured panel draws nothing.
    try t.expectEqual(@as(usize, 0), layer.drawCount());
    var stats = std.mem.zeroes(SceneStats);
    layer.drawPanels(t.allocator, Mat4.identity, 1, .RGBA16F, &stats);
    try t.expectEqual(@as(u32, 0), stats.draw_calls);

    // A panel faked as drawable still draws nothing headless (isvalid
    // gate after the count check), and the census skips imageless targets
    // while the byte math stays pure.
    layer.get(0).?.captured = true;
    layer.get(0).?.gpu.valid = true;
    layer.drawPanels(t.allocator, Mat4.identity, 1, .RGBA16F, &stats);
    try t.expectEqual(@as(u32, 0), stats.draw_calls);
    try t.expectEqual(@as(usize, 0), layer.drawCount());
    try t.expectEqual(@as(usize, 0), layer.censusBytes());
    try t.expectEqual(@as(usize, 512 * 256 * 4), targetBytes(512, 256));
    layer.get(0).?.gpu.target.image = .{ .id = 77 };
    try t.expectEqual(@as(usize, 1), layer.drawCount());
    try t.expectEqual(@as(usize, 512 * 256 * 4), layer.censusBytes());
    layer.drawPanels(t.allocator, Mat4.identity, 1, .RGBA16F, &stats);
    try t.expectEqual(@as(u32, 0), stats.draw_calls);
}

test "disabled panels draw and schedule nothing" {
    const t = std.testing;
    var layer = Gui3dLayer{};
    defer layer.deinit(t.allocator);
    _ = try layer.add(t.allocator, "p", Vec3.zero, .{ .enabled = false });
    // Disabled + dirty: never scheduled (bit-identical frame needs no
    // capture pass and no quad draw).
    try t.expectEqual(@as(?usize, null), layer.nextDirtyIndex());
    try t.expectEqual(@as(usize, 0), layer.drawCount());
    var stats = std.mem.zeroes(SceneStats);
    // Even force-captured, a disabled panel stays undrawable.
    layer.get(0).?.captured = true;
    layer.get(0).?.gpu.valid = true;
    layer.get(0).?.gpu.target.image = .{ .id = 77 };
    try t.expectEqual(@as(usize, 0), layer.drawCount());
    layer.drawPanels(t.allocator, Mat4.identity, 1, .RGBA16F, &stats);
    try t.expectEqual(@as(u32, 0), stats.draw_calls);
}
