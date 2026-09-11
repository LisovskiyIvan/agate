const Camera = @import("../camera.zig").Camera;

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
    }
}
