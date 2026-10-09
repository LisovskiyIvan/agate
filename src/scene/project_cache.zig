const std = @import("std");
const math = @import("math");
const Mat4 = math.Mat4;
const Camera = @import("../camera.zig").Camera;
const scene_projection = @import("projection.zig");

/// Cached view-projection for projectPoint. The physics overlay calls
/// projectPoint ~4k times per frame; recomputing (and inverting) the VP
/// per call dominated its profile, so the matrix is reused until the
/// camera, viewport width or height changes.
pub const ProjectCache = struct {
    vp_valid: bool = false,
    vp: Mat4 = Mat4.identity,
    cam: ?Camera = null,
    w: f32 = 0.0,
    h: f32 = 0.0,

    /// Returns the view-projection for (cam, w, h), reusing the cached one
    /// when nothing that affects the matrices changed. Bit-identical to
    /// cam.getViewProjection(w / h).
    pub fn viewProjection(self: *ProjectCache, cam: Camera, w: f32, h: f32) Mat4 {
        if (self.vp_valid) {
            if (self.cam) |pc| {
                if (w == self.w and h == self.h and scene_projection.camerasEqualForProjection(pc, cam)) {
                    return self.vp;
                }
            }
        }
        const vp = cam.getViewProjection(w / h);
        self.vp = vp;
        self.cam = cam;
        self.w = w;
        self.h = h;
        self.vp_valid = true;
        return vp;
    }
};
