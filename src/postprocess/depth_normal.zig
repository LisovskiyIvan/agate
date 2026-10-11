//! Depth-derived surface normal shared by SSR, contact shadows and SSGI.
//! Pure CPU mirror of reconstructWorldNormal in shaders/postprocess.glsl.
const math = @import("math");
const Vec3 = math.Vec3;

pub fn depthNormalFromNeighbors(view_proj: math.Mat4, center: Vec3, right: Vec3, left: Vec3, up: Vec3, down: Vec3) Vec3 {
    // Perspective clip.w is linear view depth. Its world-space axis is
    // the fourth row of VP, independent of camera yaw, pitch and roll.
    const view_forward = Vec3.new(view_proj.m[3], view_proj.m[7], view_proj.m[11]);
    const dr = right.sub(center);
    const dl = center.sub(left);
    const du = up.sub(center);
    const dd = center.sub(down);
    const dx = if (@abs(dr.dot(view_forward)) < @abs(dl.dot(view_forward))) dr else dl;
    const dy = if (@abs(du.dot(view_forward)) < @abs(dd.dot(view_forward))) du else dd;
    return depthNormalFromGradients(dx, dy);
}

pub fn depthNormalFromGradients(dx: Vec3, dy: Vec3) Vec3 {
    const n = dx.cross(dy);
    const len_sq = n.lengthSq();
    // Relative degeneracy test (squared sine of the gradient angle), not
    // a cutoff in world units that turns small pixels into upward normals.
    const scale = dx.lengthSq() * dy.lengthSq();
    return if (len_sq > @max(scale * 1e-8, 1e-30)) n.scale(1.0 / @sqrt(len_sq)) else Vec3.zero;
}
