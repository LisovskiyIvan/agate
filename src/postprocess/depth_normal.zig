//! Depth-derived surface normal shared by SSR, contact shadows and SSGI.
//! Pure CPU mirror of reconstructWorldNormal in shaders/postprocess.glsl.
const math = @import("math");
const Vec3 = math.Vec3;

pub fn depthNormalFromGradients(dx: Vec3, dy: Vec3) Vec3 {
    const n = dx.cross(dy);
    const len_sq = n.lengthSq();
    // Relative degeneracy test (squared sine of the gradient angle), not
    // a cutoff in world units that turns small pixels into upward normals.
    const scale = dx.lengthSq() * dy.lengthSq();
    return if (len_sq > @max(scale * 1e-8, 1e-30)) n.scale(1.0 / @sqrt(len_sq)) else Vec3.zero;
}
