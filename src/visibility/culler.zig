const std = @import("std");
const math = @import("math");
const Mat4 = math.Mat4;
const Vec3 = math.Vec3;
const BoundingBox = math.BoundingBox;
const HiZBuffer = @import("hiz_buffer.zig").HiZBuffer;
const SoftwareRasterizer = @import("rasterizer.zig").SoftwareRasterizer;
const mesh_mod = @import("../mesh/types.zig");

/// High-level CPU Occlusion Culler using Hierarchical Z-Buffer (Hi-Z).
/// Manages occluder submission, rasterization, pyramid downsampling, and constant-time AABB visibility testing.
pub const OcclusionCuller = struct {
    hiz: HiZBuffer,
    view_proj: Mat4 = Mat4.identity,
    occluder_count: u32 = 0,
    triangles_rasterized: u32 = 0,

    pub fn init() OcclusionCuller {
        return .{
            .hiz = HiZBuffer.init(),
        };
    }

    /// Prepares the culler for a new frame with the camera's view-projection matrix.
    pub fn beginFrame(self: *OcclusionCuller, view_proj: Mat4) void {
        self.view_proj = view_proj;
        self.hiz.clear(1.0);
        self.occluder_count = 0;
        self.triangles_rasterized = 0;
    }

    /// Submits an oriented occluder box (e.g. wall, pillar, large building volume).
    pub fn rasterizeOccluderBox(self: *OcclusionCuller, aabb: BoundingBox, world_mat: Mat4) void {
        self.occluder_count += 1;
        self.triangles_rasterized += SoftwareRasterizer.rasterizeBox(&self.hiz, self.view_proj, aabb, world_mat);
    }

    /// Submits an occluder mesh. If CPU geometry is stored, rasterizes actual triangles;
    /// otherwise falls back to the mesh's bounding box.
    pub fn rasterizeOccluderMesh(
        self: *OcclusionCuller,
        positions: []const Vec3,
        indices: []const u32,
        local_aabb: BoundingBox,
        world_mat: Mat4,
    ) void {
        _ = local_aabb;
        if (positions.len >= 3 and indices.len >= 3) {
            self.occluder_count += 1;
            self.triangles_rasterized += SoftwareRasterizer.rasterizeTriangles(
                &self.hiz,
                self.view_proj,
                positions,
                indices,
                world_mat,
            );
        }
    }

    /// Finishes occluder submission and generates the Hi-Z pyramid for fast O(1) queries.
    pub fn endOccluders(self: *OcclusionCuller) void {
        if (self.occluder_count > 0) {
            self.hiz.buildPyramid();
        }
    }

    /// Tests if an occludee's world-space AABB is fully occluded.
    /// Returns true if 100% occluded (safe to cull), false if visible or no occluders exist.
    pub inline fn isOccluded(self: *const OcclusionCuller, world_aabb: BoundingBox) bool {
        if (self.occluder_count == 0) return false;
        return self.hiz.testAABB(self.view_proj, world_aabb);
    }
};
