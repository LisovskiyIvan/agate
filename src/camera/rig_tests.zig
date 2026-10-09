const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const Mat4 = math.Mat4;
const Color4 = math.Color4;

const viewport_mod = @import("viewport.zig");
const Viewport = viewport_mod.Viewport;
const union_mod = @import("union.zig");
const Camera = union_mod.Camera;
const free_mod = @import("free.zig");
const FreeCamera = free_mod.FreeCamera;

const rig_mod = @import("rig.zig");
const CameraRig = rig_mod.CameraRig;
const CameraRigMode = rig_mod.CameraRigMode;
const CameraRigSlot = rig_mod.CameraRigSlot;
const StereoConvergenceMode = rig_mod.StereoConvergenceMode;

test "CameraRig single and dual presets" {
    const free: Camera = .{ .free = FreeCamera.init("master", .{}) };
    var rig = CameraRig.init(free);
    try std.testing.expectEqual(CameraRigMode.single, rig.mode);
    try std.testing.expectEqual(@as(usize, 1), rig.slot_count);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), rig.slots[0].viewport.width, 1e-5);

    rig.setupDualHorizontal(null);
    try std.testing.expectEqual(CameraRigMode.dual_horizontal, rig.mode);
    try std.testing.expectEqual(@as(usize, 2), rig.slot_count);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), rig.slots[0].viewport.width, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), rig.slots[1].viewport.x, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), rig.slots[1].viewport.width, 1e-5);

    rig.setupDualVertical(null);
    try std.testing.expectEqual(CameraRigMode.dual_vertical, rig.mode);
    try std.testing.expectEqual(@as(usize, 2), rig.slot_count);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), rig.slots[0].viewport.y, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), rig.slots[1].viewport.y, 1e-5);
}

test "CameraRig quad view and PIP presets" {
    const free: Camera = .{ .free = FreeCamera.init("master", .{}) };
    var rig = CameraRig.init(free);

    rig.setupQuadView(null, null, null);
    try std.testing.expectEqual(CameraRigMode.quad_view, rig.mode);
    try std.testing.expectEqual(@as(usize, 4), rig.slot_count);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), rig.slots[0].viewport.height, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), rig.slots[3].viewport.x, 1e-5);

    rig.setupPip(null, null);
    try std.testing.expectEqual(CameraRigMode.pip, rig.mode);
    try std.testing.expectEqual(@as(usize, 2), rig.slot_count);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), rig.slots[0].viewport.width, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.72), rig.slots[1].viewport.x, 1e-5);
}

test "CameraRig stereoscopic IPD and convergence" {
    const free: Camera = .{ .free = FreeCamera.init("master", .{
        .position = Vec3.new(0, 0, 10),
    }) };
    var rig = CameraRig.init(free);

    rig.setupStereo(.stereoscopic_side_by_side, 0.064, .parallel, null);
    try std.testing.expectEqual(CameraRigMode.stereoscopic_side_by_side, rig.mode);
    try std.testing.expectEqual(@as(usize, 2), rig.slot_count);

    // Call syncSlots multiple times to prove idempotency
    rig.syncSlots();
    rig.syncSlots();
    const left_pos = rig.slots[0].camera.getPosition();
    const right_pos = rig.slots[1].camera.getPosition();

    // With master at (0, 0, 10) facing -Z (free default yaw 0, pitch 0),
    // right vector is (1, 0, 0).
    // Left eye is at -0.032 on X, Right eye is at +0.032 on X.
    try std.testing.expectApproxEqAbs(@as(f32, -0.032), left_pos.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.032), right_pos.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), left_pos.z, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), right_pos.z, 1e-4);

    // Test toe-in convergence
    rig.setupStereo(.stereoscopic_side_by_side, 0.064, .toe_in, 2.0);
    rig.syncSlots();
    rig.syncSlots();
    const left_fwd = rig.slots[0].camera.getForward();
    const right_fwd = rig.slots[1].camera.getForward();
    // In toe-in, both eyes look at (0, 0, 10 - 2.0) = (0, 0, 8).
    // Left eye (at -0.032) looks toward +X to hit 0; right eye (at +0.032) looks toward -X.
    try std.testing.expect(left_fwd.x > 0.0);
    try std.testing.expect(right_fwd.x < 0.0);
}

test "CameraRig CAD quad view layout" {
    const free: Camera = .{ .free = FreeCamera.init("master", .{}) };
    var rig = CameraRig.init(free);
    const origin = Vec3.zero;
    rig.setupCadQuadView(origin, 15.0);

    try std.testing.expectEqual(@as(usize, 4), rig.slot_count);
    try std.testing.expectEqualStrings("Perspective", rig.slots[0].name);
    try std.testing.expectEqualStrings("TopView", rig.slots[1].name);
    try std.testing.expectEqualStrings("FrontView", rig.slots[2].name);
    try std.testing.expectEqualStrings("RightView", rig.slots[3].name);

    // Top view is at (0, 15, 0.0001)
    const top_pos = rig.slots[1].camera.getPosition();
    try std.testing.expectApproxEqAbs(@as(f32, 15.0), top_pos.y, 1e-3);

    // Front view is at (0, 0, 15)
    const front_pos = rig.slots[2].camera.getPosition();
    try std.testing.expectApproxEqAbs(@as(f32, 15.0), front_pos.z, 1e-3);

    // Right view is at (15, 0, 0)
    const right_pos = rig.slots[3].camera.getPosition();
    try std.testing.expectApproxEqAbs(@as(f32, 15.0), right_pos.x, 1e-3);
}

test "CameraRig custom slot and local offset" {
    const free: Camera = .{ .free = FreeCamera.init("master", .{
        .position = Vec3.new(0, 0, 0),
    }) };
    var rig = CameraRig.init(free);

    // Add a secondary camera offset 5 units right (+X) and 2 units up (+Y)
    const sub_cam = Camera{ .free = FreeCamera.init("sub", .{}) };
    const slot_idx = try rig.addSlot(.{
        .name = "Chaser",
        .camera = sub_cam,
        .viewport = .{ .x = 0.5, .y = 0.0, .width = 0.5, .height = 1.0 },
        .local_offset = Vec3.new(5.0, 2.0, 0.0),
        .sync_transform = true,
    });
    try std.testing.expectEqual(@as(usize, 1), slot_idx);
    try std.testing.expectEqual(@as(usize, 2), rig.slot_count);

    rig.syncSlots();
    const chaser_pos = rig.slots[1].camera.getPosition();
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), chaser_pos.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), chaser_pos.y, 1e-4);
}

test "CameraRig applyToScene and syncToScene" {
    const alloc = std.testing.allocator;
    const MockScene = struct {
        const MockEntry = struct {
            name: []const u8,
            camera: Camera,
            viewport: Viewport,
            clear_viewport: bool,
            clear_color: ?Color4,
            culling_mask: u32,
            enabled: bool,
        };
        cameras: std.ArrayListUnmanaged(MockEntry) = .empty,
        enable_multi_camera: bool = false,
        active_camera_index: ?usize = null,
        active_camera: ?Camera = null,
        allocator: std.mem.Allocator,

        pub fn removeCamera(self: *@This(), index: usize) void {
            if (index < self.cameras.items.len) {
                _ = self.cameras.orderedRemove(index);
            }
        }
        pub fn addCamera(self: *@This(), entry: MockEntry) !usize {
            const idx = self.cameras.items.len;
            try self.cameras.append(self.allocator, entry);
            return idx;
        }
        pub fn switchCamera(self: *@This(), index: usize) void {
            self.active_camera_index = index;
            if (index < self.cameras.items.len) {
                self.active_camera = self.cameras.items[index].camera;
            }
        }
    };

    var scene = MockScene{ .allocator = alloc };
    defer scene.cameras.deinit(alloc);

    const free: Camera = .{ .free = FreeCamera.init("master", .{}) };
    var rig = CameraRig.init(free);
    rig.setupDualHorizontal(null);

    try rig.applyToScene(&scene);
    try std.testing.expect(scene.enable_multi_camera);
    try std.testing.expectEqual(@as(usize, 2), scene.cameras.items.len);
    try std.testing.expectEqualStrings("LeftView", scene.cameras.items[0].name);
    try std.testing.expectEqualStrings("RightView", scene.cameras.items[1].name);
    try std.testing.expectEqual(@as(usize, 0), scene.active_camera_index.?);

    // Mutate rig viewport and test syncToScene
    rig.slots[1].viewport = .{ .x = 0.6, .y = 0.0, .width = 0.4, .height = 1.0 };
    rig.syncToScene(&scene);
    try std.testing.expectApproxEqAbs(@as(f32, 0.6), scene.cameras.items[1].viewport.x, 1e-5);
}
