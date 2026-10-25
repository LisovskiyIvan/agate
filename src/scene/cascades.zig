const std = @import("std");

const math = @import("math");
const Vec3 = math.Vec3;
const Mat4 = math.Mat4;
const Camera = @import("../camera.zig").Camera;

// Pure cascade solver, with split distances passed explicitly.
// `norm_light_dir` must be the
// resolveSunDirection output (normalized sun direction).
pub fn computeCascades(camera: Camera, aspect: f32, norm_light_dir: Vec3, splits: [4]f32) [4]Mat4 {
    var result: [4]Mat4 = undefined;

    const cam_pos = camera.getPosition();
    var forward = camera.getForward();
    const fwd_len = forward.length();
    if (fwd_len > 0.0001) {
        forward = forward.scale(1.0 / fwd_len);
    } else {
        forward = Vec3.new(0, 0, -1);
    }

    const fov_rad = camera.getFovDeg() * (std.math.pi / 180.0);
    const tan_half_fov = @tan(fov_rad * 0.5);

    // Shaders select cascades by radial camera distance, not view-axis z.
    // At the screen corner z = distance / ray_stretch. Include the 12%
    // pre-split blend region in the next cascade as well; a depth slice
    // beginning at the split leaves a rectangular unshadowed gap there.
    const ray_stretch = @sqrt(1.0 + tan_half_fov * tan_half_fov * (1.0 + aspect * aspect));
    var split_near: f32 = 0.0;
    const cascade_res: f32 = 1024.0;
    const cascade_scale: f32 = cascade_res / (cascade_res - 16.0);

    const light_up = if (@abs(norm_light_dir.y) > 0.99) Vec3.new(0, 0, 1) else Vec3.up;
    const ref_light_view = Mat4.lookAt(Vec3.zero, norm_light_dir.scale(-1.0), light_up);
    const inv_ref = ref_light_view.invert() orelse Mat4.identity;

    const s = Vec3.new(ref_light_view.m[0], ref_light_view.m[4], ref_light_view.m[8]);
    const u = Vec3.new(ref_light_view.m[1], ref_light_view.m[5], ref_light_view.m[9]);
    const neg_f = Vec3.new(ref_light_view.m[2], ref_light_view.m[6], ref_light_view.m[10]);

    for (0..4) |i| {
        const z_near = split_near * 0.88 / ray_stretch;
        const z_far = splits[i];

        const half_h_far = tan_half_fov * z_far;
        const half_w_far = half_h_far * aspect;

        const center = cam_pos.add(forward.scale((z_near + z_far) * 0.5));
        const dz = (z_far - z_near) * 0.5;

        var radius = @sqrt(dz * dz + half_h_far * half_h_far + half_w_far * half_w_far);
        radius = @ceil(radius * 16.0) / 16.0;
        // Reserve 8 texels per edge: 6 for the atlas sampling inset and
        // 2 for the worst-case xy texel-snap displacement.
        radius *= cascade_scale;

        const texel_size = (2.0 * radius) / cascade_res;

        const c_ls = ref_light_view.transformPoint(center);
        const snapped_x = @floor(c_ls.x / texel_size) * texel_size;
        const snapped_y = @floor(c_ls.y / texel_size) * texel_size;

        const snapped_center_world = inv_ref.transformPoint(Vec3.new(snapped_x, snapped_y, c_ls.z));

        // Direct compose of light_proj * light_view without general 4x4 matrix multiplication
        const inv_r = 1.0 / radius;
        const inv_d = -1.0 / (2.0 * radius + 119.0);

        const v12 = -s.dot(snapped_center_world);
        const v13 = -u.dot(snapped_center_world);
        const v14 = -neg_f.dot(snapped_center_world) - (radius + 60.0);

        result[i].m[0] = s.x * inv_r;
        result[i].m[4] = s.y * inv_r;
        result[i].m[8] = s.z * inv_r;
        result[i].m[12] = v12 * inv_r;

        result[i].m[1] = u.x * inv_r;
        result[i].m[5] = u.y * inv_r;
        result[i].m[9] = u.z * inv_r;
        result[i].m[13] = v13 * inv_r;

        result[i].m[2] = neg_f.x * inv_d;
        result[i].m[6] = neg_f.y * inv_d;
        result[i].m[10] = neg_f.z * inv_d;
        result[i].m[14] = (v14 + 1.0) * inv_d;

        result[i].m[3] = 0.0;
        result[i].m[7] = 0.0;
        result[i].m[11] = 0.0;
        result[i].m[15] = 1.0;

        split_near = z_far;
    }

    return result;
}
