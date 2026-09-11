const Camera = @import("../camera.zig").Camera;
const Vec3 = @import("math").Vec3;

// Pure view/projection snapshot compare. Only fields that affect the
// matrices are compared; name pointers, input state, and tuning fields
// (limits, sensitivity, speeds, lerp) are excluded.
pub fn camerasEqualForProjection(a: Camera, b: Camera) bool {
    switch (a) {
        .arc_rotate => |ac| switch (b) {
            .arc_rotate => |bc| {
                return ac.alpha == bc.alpha and ac.beta == bc.beta and ac.radius == bc.radius and
                    ac.target.x == bc.target.x and ac.target.y == bc.target.y and ac.target.z == bc.target.z and
                    ac.fov_deg == bc.fov_deg and ac.near == bc.near and ac.far == bc.far;
            },
            else => return false,
        },
        .free => |ac| switch (b) {
            .free => |bc| {
                return ac.position.x == bc.position.x and ac.position.y == bc.position.y and ac.position.z == bc.position.z and
                    ac.rotation.x == bc.rotation.x and ac.rotation.y == bc.rotation.y and ac.rotation.z == bc.rotation.z and
                    ac.fov_deg == bc.fov_deg and ac.near == bc.near and ac.far == bc.far;
            },
            else => return false,
        },
        .follow => |ac| switch (b) {
            .follow => |bc| {
                return ac.target_mesh == bc.target_mesh and
                    ac.position.x == bc.position.x and ac.position.y == bc.position.y and ac.position.z == bc.position.z and
                    ac.target_position.x == bc.target_position.x and ac.target_position.y == bc.target_position.y and ac.target_position.z == bc.target_position.z and
                    ac.radius == bc.radius and ac.height_offset == bc.height_offset and
                    ac.rotation_offset_deg == bc.rotation_offset_deg and
                    ac.fov_deg == bc.fov_deg and ac.near == bc.near and ac.far == bc.far;
            },
            else => return false,
        },
        .target => |ac| switch (b) {
            .target => |bc| {
                // Pending goals are included: a just-set goal means the next
                // update() will move the matrices, so the cache must miss.
                return vecEq(ac.position, bc.position) and vecEq(ac.target, bc.target) and
                    optVecEq(ac.desired_position, bc.desired_position) and
                    optVecEq(ac.desired_target, bc.desired_target) and
                    vecEq(ac.up, bc.up) and
                    ac.fov_deg == bc.fov_deg and ac.near == bc.near and ac.far == bc.far;
            },
            else => return false,
        },
        .fly => |ac| switch (b) {
            .fly => |bc| {
                // rotation.z (roll) tilts the view up vector, so all three
                // euler components participate in the view matrix.
                return vecEq(ac.position, bc.position) and vecEq(ac.rotation, bc.rotation) and
                    ac.fov_deg == bc.fov_deg and ac.near == bc.near and ac.far == bc.far;
            },
            else => return false,
        },
    }
}

fn vecEq(a: Vec3, b: Vec3) bool {
    return a.x == b.x and a.y == b.y and a.z == b.z;
}

fn optVecEq(a: ?Vec3, b: ?Vec3) bool {
    if (a == null and b == null) return true;
    if (a) |av| {
        if (b) |bv| return vecEq(av, bv);
    }
    return false;
}
