const std = @import("std");

const math = @import("math");
const Vec3 = math.Vec3;
const trail_mod = @import("../mesh/trail.zig");
const TrailMesh = trail_mod.TrailMesh;
const TrailOptions = trail_mod.TrailOptions;

/// Trail meshes: ribbon geometry that records a moving anchor each frame.
/// TrailMesh needs the owning *Scene for mesh creation, so Scene passes
/// itself into create (kept as anytype to avoid a scene.zig import cycle).
pub const TrailLayer = struct {
    meshes: std.ArrayListUnmanaged(*TrailMesh) = .empty,

    pub fn deinit(self: *TrailLayer, allocator: std.mem.Allocator) void {
        for (self.meshes.items) |tm| {
            tm.deinit();
            allocator.destroy(tm);
        }
        self.meshes.deinit(allocator);
    }

    pub fn create(self: *TrailLayer, scene: anytype, allocator: std.mem.Allocator, name: []const u8, options: TrailOptions) !*TrailMesh {
        const tm = try trail_mod.TrailMesh.init(scene, name, options);
        try self.meshes.append(allocator, tm);
        return tm;
    }

    /// Advances every trail; trails cull segments by distance to the camera.
    pub fn update(self: *TrailLayer, dt: f32, cam_pos: Vec3) void {
        for (self.meshes.items) |tm| {
            tm.update(dt, cam_pos);
        }
    }
};
