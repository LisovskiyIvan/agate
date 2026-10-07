//! Multi-camera rig system supporting presets (dual horizontal/vertical,
//! quad view, picture-in-picture, stereoscopic VR 3D) and custom multi-camera
//! setups with automated viewport management and transform synchronization.
//!
//! Conforms to repository architectural rules:
//! - Leaf camera module under `camera/`
//! - No cycle back to `camera.zig` facade (imports sibling leaves directly)
//! - Decoupled scene integration via `anytype`

const std = @import("std");
const sokol = @import("sokol");
const math = @import("math");
const Vec3 = math.Vec3;
const Mat4 = math.Mat4;
const Color4 = math.Color4;

const viewport_mod = @import("viewport.zig");
const Viewport = viewport_mod.Viewport;
const union_mod = @import("union.zig");
const Camera = union_mod.Camera;
const target_mod = @import("target.zig");
const TargetCamera = target_mod.TargetCamera;

pub const MAX_RIG_SLOTS: usize = 8;

pub const CameraRigMode = enum {
    single,
    dual_horizontal,
    dual_vertical,
    quad_view,
    pip,
    stereoscopic_side_by_side,
    stereoscopic_over_under,
    custom,
};

pub const StereoConvergenceMode = enum {
    /// Parallel view axes for VR Head-Mounted Displays (HMDs) with optical collimation.
    parallel,
    /// Angled inward view axes to converge at a focal plane (zero-parallax display distance).
    toe_in,
};

pub const CameraRigSlot = struct {
    name: []const u8 = "RigSlot",
    camera: Camera,
    viewport: Viewport = .{},
    clear_viewport: bool = true,
    clear_color: ?Color4 = null,
    culling_mask: u32 = 0xFFFFFFFF,
    enabled: bool = true,
    /// Offset in master camera's local frame:
    /// x = right (+right, -left), y = up (+up, -down), z = forward (+forward, -back)
    local_offset: Vec3 = Vec3.zero,
    /// If set, the camera looks at this world-space position instead of matching orientation
    look_at_target: ?Vec3 = null,
    /// Whether this slot's position/orientation automatically syncs from the master camera
    sync_transform: bool = true,
};

pub const CameraRig = struct {
    mode: CameraRigMode = .single,
    master_camera: Camera,
    slots: [MAX_RIG_SLOTS]CameraRigSlot = undefined,
    slot_count: usize = 0,

    /// Interpupillary distance (IPD) baseline in meters (default 0.064m = 64mm human average)
    ipd: f32 = 0.064,

    /// Convergence distance in meters for stereoscopic toe-in mode
    convergence_distance: f32 = 2.0,

    /// Stereoscopic convergence mode
    stereo_convergence: StereoConvergenceMode = .parallel,

    /// Default Picture-In-Picture inset viewport (normalized [0..1])
    pip_viewport: Viewport = .{ .x = 0.72, .y = 0.05, .width = 0.25, .height = 0.25 },

    /// Creates a CameraRig with a single master camera filling the viewport.
    pub fn init(master: Camera) CameraRig {
        var rig = CameraRig{
            .master_camera = master,
        };
        rig.setupSingle();
        return rig;
    }

    /// Creates a CameraRig configured with a specific preset mode.
    pub fn initPreset(master: Camera, mode: CameraRigMode) CameraRig {
        var rig = CameraRig{
            .master_camera = master,
        };
        rig.setMode(mode);
        return rig;
    }

    fn createDefaultSubCamera(master: Camera, name: []const u8) Camera {
        const pos = master.getPosition();
        const fwd = master.getForward();
        return .{
            .target = TargetCamera.init(name, .{
                .position = pos,
                .target = pos.add(fwd.scale(10.0)),
                .fov_deg = master.getFovDeg(),
                .near = master.getNear(),
                .far = master.getFar(),
                .smoothing = 0.0,
            }),
        };
    }

    pub fn setMode(self: *CameraRig, mode: CameraRigMode) void {
        switch (mode) {
            .single => self.setupSingle(),
            .dual_horizontal => self.setupDualHorizontal(null),
            .dual_vertical => self.setupDualVertical(null),
            .quad_view => self.setupQuadView(null, null, null),
            .pip => self.setupPip(null, null),
            .stereoscopic_side_by_side => self.setupStereo(.stereoscopic_side_by_side, null, null, null),
            .stereoscopic_over_under => self.setupStereo(.stereoscopic_over_under, null, null, null),
            .custom => self.mode = .custom,
        }
    }

    pub fn setupSingle(self: *CameraRig) void {
        self.mode = .single;
        self.slots[0] = .{
            .name = "Master",
            .camera = self.master_camera,
            .viewport = .{ .x = 0.0, .y = 0.0, .width = 1.0, .height = 1.0 },
            .clear_viewport = true,
            .enabled = true,
            .sync_transform = true,
        };
        self.slot_count = 1;
    }

    pub fn setupDualHorizontal(self: *CameraRig, secondary: ?Camera) void {
        const sub = secondary orelse createDefaultSubCamera(self.master_camera, "RightView");
        self.mode = .dual_horizontal;
        self.slots[0] = .{
            .name = "LeftView",
            .camera = self.master_camera,
            .viewport = .{ .x = 0.0, .y = 0.0, .width = 0.5, .height = 1.0 },
            .clear_viewport = true,
            .enabled = true,
            .sync_transform = true,
        };
        self.slots[1] = .{
            .name = "RightView",
            .camera = sub,
            .viewport = .{ .x = 0.5, .y = 0.0, .width = 0.5, .height = 1.0 },
            .clear_viewport = true,
            .enabled = true,
            .sync_transform = true,
        };
        self.slot_count = 2;
        self.syncSlots();
    }

    pub fn setupDualVertical(self: *CameraRig, secondary: ?Camera) void {
        const sub = secondary orelse createDefaultSubCamera(self.master_camera, "BottomView");
        self.mode = .dual_vertical;
        self.slots[0] = .{
            .name = "TopView",
            .camera = self.master_camera,
            .viewport = .{ .x = 0.0, .y = 0.5, .width = 1.0, .height = 0.5 },
            .clear_viewport = true,
            .enabled = true,
            .sync_transform = true,
        };
        self.slots[1] = .{
            .name = "BottomView",
            .camera = sub,
            .viewport = .{ .x = 0.0, .y = 0.0, .width = 1.0, .height = 0.5 },
            .clear_viewport = true,
            .enabled = true,
            .sync_transform = true,
        };
        self.slot_count = 2;
        self.syncSlots();
    }

    pub fn setupQuadView(self: *CameraRig, tr: ?Camera, bl: ?Camera, br: ?Camera) void {
        const cam_tr = tr orelse createDefaultSubCamera(self.master_camera, "TopRight");
        const cam_bl = bl orelse createDefaultSubCamera(self.master_camera, "BottomLeft");
        const cam_br = br orelse createDefaultSubCamera(self.master_camera, "BottomRight");

        self.mode = .quad_view;
        self.slots[0] = .{
            .name = "TopLeft",
            .camera = self.master_camera,
            .viewport = .{ .x = 0.0, .y = 0.5, .width = 0.5, .height = 0.5 },
            .clear_viewport = true,
            .enabled = true,
            .sync_transform = true,
        };
        self.slots[1] = .{
            .name = "TopRight",
            .camera = cam_tr,
            .viewport = .{ .x = 0.5, .y = 0.5, .width = 0.5, .height = 0.5 },
            .clear_viewport = true,
            .enabled = true,
            .sync_transform = true,
        };
        self.slots[2] = .{
            .name = "BottomLeft",
            .camera = cam_bl,
            .viewport = .{ .x = 0.0, .y = 0.0, .width = 0.5, .height = 0.5 },
            .clear_viewport = true,
            .enabled = true,
            .sync_transform = true,
        };
        self.slots[3] = .{
            .name = "BottomRight",
            .camera = cam_br,
            .viewport = .{ .x = 0.5, .y = 0.0, .width = 0.5, .height = 0.5 },
            .clear_viewport = true,
            .enabled = true,
            .sync_transform = true,
        };
        self.slot_count = 4;
        self.syncSlots();
    }

    /// Sets up a 4-viewport CAD engineering layout:
    /// Slot 0: Perspective (Master) in Top-Right
    /// Slot 1: Top view in Top-Left (looking down along -Y)
    /// Slot 2: Front view in Bottom-Left (looking along -Z)
    /// Slot 3: Right view in Bottom-Right (looking along -X)
    pub fn setupCadQuadView(self: *CameraRig, target: Vec3, distance: f32) void {
        const d = if (distance > 0.001) distance else 10.0;

        self.mode = .quad_view;
        // Perspective (master)
        self.slots[0] = .{
            .name = "Perspective",
            .camera = self.master_camera,
            .viewport = .{ .x = 0.5, .y = 0.5, .width = 0.5, .height = 0.5 },
            .clear_viewport = true,
            .enabled = true,
            .sync_transform = true,
        };

        // Top view: eye at target + (0, d, 0.0001), looking at target, up = (0, 0, -1)
        const top_cam: Camera = .{
            .target = TargetCamera.init("TopView", .{
                .position = target.add(Vec3.new(0.0, d, 0.0001)),
                .target = target,
                .up = Vec3.new(0.0, 0.0, -1.0),
                .fov_deg = self.master_camera.getFovDeg(),
                .near = self.master_camera.getNear(),
                .far = self.master_camera.getFar() * 2.0,
                .smoothing = 0.0,
            }),
        };
        self.slots[1] = .{
            .name = "TopView",
            .camera = top_cam,
            .viewport = .{ .x = 0.0, .y = 0.5, .width = 0.5, .height = 0.5 },
            .clear_viewport = true,
            .enabled = true,
            .sync_transform = false,
            .look_at_target = target,
        };

        // Front view: eye at target + (0, 0, d), looking at target, up = (0, 1, 0)
        const front_cam: Camera = .{
            .target = TargetCamera.init("FrontView", .{
                .position = target.add(Vec3.new(0.0, 0.0, d)),
                .target = target,
                .up = Vec3.up,
                .fov_deg = self.master_camera.getFovDeg(),
                .near = self.master_camera.getNear(),
                .far = self.master_camera.getFar() * 2.0,
                .smoothing = 0.0,
            }),
        };
        self.slots[2] = .{
            .name = "FrontView",
            .camera = front_cam,
            .viewport = .{ .x = 0.0, .y = 0.0, .width = 0.5, .height = 0.5 },
            .clear_viewport = true,
            .enabled = true,
            .sync_transform = false,
            .look_at_target = target,
        };

        // Right view: eye at target + (d, 0, 0), looking at target, up = (0, 1, 0)
        const right_cam: Camera = .{
            .target = TargetCamera.init("RightView", .{
                .position = target.add(Vec3.new(d, 0.0, 0.0)),
                .target = target,
                .up = Vec3.up,
                .fov_deg = self.master_camera.getFovDeg(),
                .near = self.master_camera.getNear(),
                .far = self.master_camera.getFar() * 2.0,
                .smoothing = 0.0,
            }),
        };
        self.slots[3] = .{
            .name = "RightView",
            .camera = right_cam,
            .viewport = .{ .x = 0.5, .y = 0.0, .width = 0.5, .height = 0.5 },
            .clear_viewport = true,
            .enabled = true,
            .sync_transform = false,
            .look_at_target = target,
        };

        self.slot_count = 4;
    }

    pub fn setupPip(self: *CameraRig, pip_cam: ?Camera, pip_vp: ?Viewport) void {
        const sub = pip_cam orelse createDefaultSubCamera(self.master_camera, "PipView");
        const vp = pip_vp orelse self.pip_viewport;

        self.mode = .pip;
        self.slots[0] = .{
            .name = "MainView",
            .camera = self.master_camera,
            .viewport = .{ .x = 0.0, .y = 0.0, .width = 1.0, .height = 1.0 },
            .clear_viewport = true,
            .enabled = true,
            .sync_transform = true,
        };
        self.slots[1] = .{
            .name = "PipView",
            .camera = sub,
            .viewport = vp,
            .clear_viewport = true,
            .enabled = true,
            .sync_transform = true,
        };
        self.slot_count = 2;
        self.syncSlots();
    }

    pub fn setupStereo(
        self: *CameraRig,
        mode: CameraRigMode,
        ipd_opt: ?f32,
        convergence_opt: ?StereoConvergenceMode,
        convergence_dist_opt: ?f32,
    ) void {
        if (ipd_opt) |v| self.ipd = v;
        if (convergence_opt) |c| self.stereo_convergence = c;
        if (convergence_dist_opt) |d| self.convergence_distance = d;

        self.mode = if (mode == .stereoscopic_over_under) .stereoscopic_over_under else .stereoscopic_side_by_side;

        const left_vp: Viewport = if (self.mode == .stereoscopic_side_by_side)
            .{ .x = 0.0, .y = 0.0, .width = 0.5, .height = 1.0 }
        else
            .{ .x = 0.0, .y = 0.5, .width = 1.0, .height = 0.5 };

        const right_vp: Viewport = if (self.mode == .stereoscopic_side_by_side)
            .{ .x = 0.5, .y = 0.0, .width = 0.5, .height = 1.0 }
        else
            .{ .x = 0.0, .y = 0.0, .width = 1.0, .height = 0.5 };

        self.slots[0] = .{
            .name = "EyeLeft",
            .camera = createDefaultSubCamera(self.master_camera, "EyeLeft"),
            .viewport = left_vp,
            .clear_viewport = true,
            .enabled = true,
            .sync_transform = false,
        };
        self.slots[1] = .{
            .name = "EyeRight",
            .camera = createDefaultSubCamera(self.master_camera, "EyeRight"),
            .viewport = right_vp,
            .clear_viewport = true,
            .enabled = true,
            .sync_transform = false,
        };
        self.slot_count = 2;
        self.syncSlots();
    }

    pub fn addSlot(self: *CameraRig, slot: CameraRigSlot) !usize {
        if (self.slot_count >= MAX_RIG_SLOTS) return error.RigSlotsExceeded;
        const idx = self.slot_count;
        self.slots[idx] = slot;
        self.slot_count += 1;
        self.mode = .custom;
        return idx;
    }

    pub fn removeSlot(self: *CameraRig, index: usize) void {
        if (index >= self.slot_count) return;
        var i = index;
        while (i + 1 < self.slot_count) : (i += 1) {
            self.slots[i] = self.slots[i + 1];
        }
        self.slot_count -= 1;
    }

    pub fn getSlot(self: *CameraRig, index: usize) ?*CameraRigSlot {
        if (index < self.slot_count) return &self.slots[index];
        return null;
    }

    pub fn getSlotConst(self: *const CameraRig, index: usize) ?*const CameraRigSlot {
        if (index < self.slot_count) return &self.slots[index];
        return null;
    }

    pub fn getMaster(self: *CameraRig) *Camera {
        return &self.master_camera;
    }

    pub fn getMasterConst(self: *const CameraRig) *const Camera {
        return &self.master_camera;
    }

    pub fn handleEvent(self: *CameraRig, ev: [*c]const sokol.app.Event) void {
        self.master_camera.handleEvent(ev);
    }

    pub fn update(self: *CameraRig, dt: f32) void {
        // Update master camera simulation (inertia, keyboard, mouse)
        self.master_camera.update(dt);
        // Sync rig eye offsets and positions
        self.syncSlots();
    }

    /// Synchronizes sub-camera transforms from the master camera according
    /// to mode (stereoscopic IPD separation/toe-in, local offsets, or look-at targets).
    pub fn syncSlots(self: *CameraRig) void {
        if (self.slot_count == 0) return;
        const pos = self.master_camera.getPosition();
        const fwd = self.master_camera.getForward();
        const right = self.master_camera.getRight();
        const up = self.master_camera.getUp();

        switch (self.mode) {
            .single => {
                self.slots[0].camera = self.master_camera;
            },
            .stereoscopic_side_by_side, .stereoscopic_over_under => {
                if (self.slot_count < 2) return;
                const half_ipd = self.ipd * 0.5;
                const left_pos = pos.sub(right.scale(half_ipd));
                const right_pos = pos.add(right.scale(half_ipd));

                switch (self.stereo_convergence) {
                    .toe_in => {
                        const focal_pt = pos.add(fwd.scale(self.convergence_distance));
                        self.slots[0].camera.setLookAt(left_pos, focal_pt, up);
                        self.slots[1].camera.setLookAt(right_pos, focal_pt, up);
                    },
                    .parallel => {
                        const left_target = left_pos.add(fwd.scale(10.0));
                        const right_target = right_pos.add(fwd.scale(10.0));
                        self.slots[0].camera.setLookAt(left_pos, left_target, up);
                        self.slots[1].camera.setLookAt(right_pos, right_target, up);
                    },
                }
            },
            else => {
                // In preset modes (dual, quad, pip), slot 0 mirrors master camera
                if (self.mode != .custom and self.slot_count > 0 and self.slots[0].sync_transform) {
                    self.slots[0].camera = self.master_camera;
                }

                const start_idx: usize = if (self.mode == .custom) 0 else 1;
                for (self.slots[start_idx..self.slot_count]) |*slot| {
                    if (!slot.sync_transform) continue;

                    const world_offset = right.scale(slot.local_offset.x)
                        .add(up.scale(slot.local_offset.y))
                        .add(fwd.scale(slot.local_offset.z));
                    const slot_pos = pos.add(world_offset);
                    const target = if (slot.look_at_target) |t| t else slot_pos.add(fwd.scale(10.0));
                    slot.camera.setLookAt(slot_pos, target, up);
                }
            },
        }
    }

    /// Clears any existing cameras in `scene` and registers each slot as a `CameraEntry`,
    /// enabling `scene.enable_multi_camera` when multiple slots exist.
    pub fn applyToScene(self: *CameraRig, scene: anytype) !void {
        self.syncSlots();
        while (scene.cameras.items.len > 0) {
            scene.removeCamera(0);
        }
        for (self.slots[0..self.slot_count]) |slot| {
            _ = try scene.addCamera(.{
                .name = slot.name,
                .camera = slot.camera,
                .viewport = slot.viewport,
                .clear_viewport = slot.clear_viewport,
                .clear_color = slot.clear_color,
                .culling_mask = slot.culling_mask,
                .enabled = slot.enabled,
            });
        }
        scene.enable_multi_camera = (self.mode != .single and self.slot_count > 1);
        if (self.slot_count > 0) {
            scene.switchCamera(0);
        }
    }

    /// Fast per-frame update synchronizing live camera objects and viewports into an active `Scene`.
    pub fn syncToScene(self: *const CameraRig, scene: anytype) void {
        const n = @min(self.slot_count, scene.cameras.items.len);
        for (0..n) |i| {
            scene.cameras.items[i].camera = self.slots[i].camera;
            scene.cameras.items[i].viewport = self.slots[i].viewport;
            scene.cameras.items[i].enabled = self.slots[i].enabled;
            scene.cameras.items[i].culling_mask = self.slots[i].culling_mask;
        }
        if (scene.active_camera_index) |idx| {
            if (idx < scene.cameras.items.len) {
                scene.active_camera = scene.cameras.items[idx].camera;
            }
        }
    }
};

test "CameraRig single and dual presets" {
    const free: Camera = .{ .free = @import("free.zig").FreeCamera.init("master", .{}) };
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
    const free: Camera = .{ .free = @import("free.zig").FreeCamera.init("master", .{}) };
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
    const free: Camera = .{ .free = @import("free.zig").FreeCamera.init("master", .{
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
    const free: Camera = .{ .free = @import("free.zig").FreeCamera.init("master", .{}) };
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
    const free: Camera = .{ .free = @import("free.zig").FreeCamera.init("master", .{
        .position = Vec3.new(0, 0, 0),
    }) };
    var rig = CameraRig.init(free);

    // Add a secondary camera offset 5 units right (+X) and 2 units up (+Y)
    const sub_cam = Camera{ .free = @import("free.zig").FreeCamera.init("sub", .{}) };
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

    const free: Camera = .{ .free = @import("free.zig").FreeCamera.init("master", .{}) };
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
