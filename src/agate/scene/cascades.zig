const std = @import("std");

const math = @import("math");
const Vec3 = math.Vec3;
const Mat4 = math.Mat4;
const Camera = @import("../camera.zig").Camera;

// Pure cascade solver: identical math to the legacy Scene method, with the
// split distances passed explicitly. `norm_light_dir` must be the
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

    var right = forward.cross(Vec3.up);
    const r_len = right.length();
    if (r_len > 0.0001) {
        right = right.scale(1.0 / r_len);
    } else {
        right = Vec3.new(1, 0, 0);
    }
    const up = right.cross(forward).normalize();

    const fov_rad = camera.getFovDeg() * (std.math.pi / 180.0);
    const tan_half_fov = @tan(fov_rad * 0.5);

    var z_near = camera.getNear();
    const cascade_res: f32 = 1024.0;

    for (0..4) |i| {
        const z_far = splits[i];

        const h_near = 2.0 * tan_half_fov * z_near;
        const w_near = h_near * aspect;
        const h_far = 2.0 * tan_half_fov * z_far;
        const w_far = h_far * aspect;

        const c_near = cam_pos.add(forward.scale(z_near));
        const c_far = cam_pos.add(forward.scale(z_far));

        const corners = [8]Vec3{
            c_near.add(up.scale(h_near * 0.5)).sub(right.scale(w_near * 0.5)),
            c_near.add(up.scale(h_near * 0.5)).add(right.scale(w_near * 0.5)),
            c_near.sub(up.scale(h_near * 0.5)).sub(right.scale(w_near * 0.5)),
            c_near.sub(up.scale(h_near * 0.5)).add(right.scale(w_near * 0.5)),

            c_far.add(up.scale(h_far * 0.5)).sub(right.scale(w_far * 0.5)),
            c_far.add(up.scale(h_far * 0.5)).add(right.scale(w_far * 0.5)),
            c_far.sub(up.scale(h_far * 0.5)).sub(right.scale(w_far * 0.5)),
            c_far.sub(up.scale(h_far * 0.5)).add(right.scale(w_far * 0.5)),
        };

        var center = Vec3.zero;
        for (corners) |c| {
            center = center.add(c);
        }
        center = center.scale(1.0 / 8.0);

        var radius: f32 = 0.0;
        for (corners) |c| {
            const d = c.sub(center).length();
            if (d > radius) radius = d;
        }
        radius = @ceil(radius * 16.0) / 16.0;

        const texel_size = (2.0 * radius) / cascade_res;

        const light_up = if (@abs(norm_light_dir.y) > 0.99) Vec3.new(0, 0, 1) else Vec3.up;
        const ref_light_view = Mat4.lookAt(Vec3.zero, norm_light_dir.scale(-1.0), light_up);

        const c_ls = ref_light_view.transformPoint(center);
        const snapped_x = @floor(c_ls.x / texel_size) * texel_size;
        const snapped_y = @floor(c_ls.y / texel_size) * texel_size;

        const inv_ref = ref_light_view.invert() orelse Mat4.identity;
        const snapped_center_world = inv_ref.transformPoint(Vec3.new(snapped_x, snapped_y, c_ls.z));

        const light_eye = snapped_center_world.add(norm_light_dir.scale(radius + 60.0));
        const light_view = Mat4.lookAt(light_eye, snapped_center_world, light_up);
        const light_proj = Mat4.orthographic(-radius, radius, -radius, radius, 1.0, 2.0 * radius + 120.0);

        result[i] = Mat4.mul(light_proj, light_view);
        z_near = z_far;
    }

    return result;
}
