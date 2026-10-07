//! Scene camera management: `CameraEntry` plus add/remove/get/switch/
//! cycle/set/update over the camera list. Split out of `scene.zig`
//! (facade); `CameraEntry` is re-exported by the facade (and root).
//!
/// Anti-cycle rule (same as `audio/*`, `profiler/*`): every function takes
/// the scene as `anytype` (a `*Scene` from `core.zig` in practice) and this
/// module never imports `core.zig` or the `scene.zig` facade back.
/// Cross-leaf helpers consumed here are `pub` in their home module but are
/// deliberately NOT re-exported by the facade.
const std = @import("std");
const Camera = @import("../camera.zig").Camera;
const Viewport = @import("../camera.zig").Viewport;
const Color4 = @import("math").Color4;

pub const CameraEntry = struct {
    name: []const u8,
    camera: Camera,
    owns_name: bool = false,
    enabled: bool = true,
    culling_mask: u32 = 0xFFFFFFFF,
    viewport: Viewport = .{},
    clear_viewport: bool = true,
    clear_color: ?Color4 = null,
};

// ---- Camera management ----

pub fn addCamera(self: anytype, entry: CameraEntry) !usize {
    var e = entry;
    if (e.culling_mask == 0xFFFFFFFF and e.camera.getCullingMask() != 0xFFFFFFFF) {
        e.culling_mask = e.camera.getCullingMask();
    }
    const cam_vp = e.camera.getViewport();
    if (e.viewport.x == 0.0 and e.viewport.y == 0.0 and e.viewport.width == 1.0 and e.viewport.height == 1.0 and
        (cam_vp.x != 0.0 or cam_vp.y != 0.0 or cam_vp.width != 1.0 or cam_vp.height != 1.0))
    {
        e.viewport = cam_vp;
    }
    const idx = self.cameras.items.len;
    try self.cameras.append(self.allocator, e);
    if (self.active_camera_index == null) {
        self.active_camera_index = idx;
        self.active_camera = e.camera;
    }
    return idx;
}

pub fn removeCamera(self: anytype, index: usize) void {
    if (index >= self.cameras.items.len) return;
    const entry = self.cameras.orderedRemove(index);
    if (entry.owns_name) {
        self.allocator.free(entry.name);
    }
    if (self.cameras.items.len == 0) {
        self.active_camera_index = null;
        self.active_camera = null;
    } else {
        const cur_idx = self.active_camera_index orelse 0;
        if (cur_idx >= self.cameras.items.len) {
            self.switchCamera(self.cameras.items.len - 1);
        } else {
            self.switchCamera(cur_idx);
        }
    }
}

pub fn getCamera(self: anytype, index: usize) ?*CameraEntry {
    if (index < self.cameras.items.len) return &self.cameras.items[index];
    return null;
}

pub fn getCameraByName(self: anytype, name: []const u8) ?*CameraEntry {
    for (self.cameras.items) |*entry| {
        if (std.mem.eql(u8, entry.name, name)) return entry;
    }
    return null;
}

pub fn switchCamera(self: anytype, index: usize) void {
    if (index >= self.cameras.items.len) return;
    if (self.active_camera_index) |curr_idx| {
        if (curr_idx < self.cameras.items.len and self.active_camera != null) {
            self.cameras.items[curr_idx].camera = self.active_camera.?;
        }
    }
    self.active_camera_index = index;
    self.active_camera = self.cameras.items[index].camera;
}

pub fn switchCameraByName(self: anytype, name: []const u8) bool {
    for (self.cameras.items, 0..) |entry, i| {
        if (std.mem.eql(u8, entry.name, name)) {
            self.switchCamera(i);
            return true;
        }
    }
    return false;
}

pub fn nextCamera(self: anytype) void {
    if (self.cameras.items.len == 0) return;
    const current = self.active_camera_index orelse 0;
    const next_idx = (current + 1) % self.cameras.items.len;
    self.switchCamera(next_idx);
}

pub fn prevCamera(self: anytype) void {
    if (self.cameras.items.len == 0) return;
    const current = self.active_camera_index orelse 0;
    const prev_idx = if (current == 0) self.cameras.items.len - 1 else current - 1;
    self.switchCamera(prev_idx);
}

pub fn getActiveCameraIndex(self: anytype) ?usize {
    return self.active_camera_index;
}

pub fn getActiveCameraName(self: anytype) ?[]const u8 {
    if (self.active_camera_index) |idx| {
        if (idx < self.cameras.items.len) return self.cameras.items[idx].name;
    }
    if (self.active_camera) |cam| return cam.getName();
    return null;
}

pub fn getCameraCount(self: anytype) usize {
    return self.cameras.items.len;
}

pub fn setActiveCamera(self: anytype, cam: ?Camera, owned_name: ?[]const u8) void {
    if (self.active_camera_owned_name) |old| {
        self.allocator.free(old);
    }
    self.active_camera = cam;
    self.active_camera_owned_name = owned_name;
    if (self.active_camera_index) |idx| {
        if (idx < self.cameras.items.len and cam != null) {
            self.cameras.items[idx].camera = cam.?;
        }
    }
}

pub fn updateCamera(self: anytype, dt: f32) void {
    if (self.enable_multi_camera) {
        for (self.cameras.items, 0..) |*entry, i| {
            if (entry.enabled) {
                entry.camera.update(dt);
                if (self.active_camera_index == i) {
                    self.active_camera = entry.camera;
                }
            }
        }
    } else if (self.active_camera) |*cam| {
        cam.update(dt);
        if (self.active_camera_index) |idx| {
            if (idx < self.cameras.items.len) {
                self.cameras.items[idx].camera = cam.*;
            }
        }
    }
}
