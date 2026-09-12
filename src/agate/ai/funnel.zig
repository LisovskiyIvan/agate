const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;

pub const Portal = struct {
    left: Vec3,
    right: Vec3,
};

/// 2D signed area (cross product in horizontal XZ plane).
/// Returns positive if (a -> b -> c) is a counter-clockwise turn (c is to the left of a->b).
/// Returns negative if clockwise (c is to the right of a->b).
pub inline fn triArea2D(a: Vec3, b: Vec3, c: Vec3) f32 {
    return (b.x - a.x) * (c.z - a.z) - (c.x - a.x) * (b.z - a.z);
}

/// String-pulling (Funnel) algorithm.
/// Given a start point, end point, and sequence of shared portal edges between corridor triangles,
/// calculates the shortest smooth 3D polyline path without penetrating triangle obstacles.
pub fn stringPull(
    allocator: std.mem.Allocator,
    start: Vec3,
    end: Vec3,
    portals: []const Portal,
) ![]Vec3 {
    var path = std.ArrayList(Vec3).empty;
    errdefer path.deinit(allocator);

    // Start point is always the first waypoint
    try path.append(allocator, start);

    if (portals.len == 0) {
        try path.append(allocator, end);
        return try path.toOwnedSlice(allocator);
    }

    var apex = start;
    var portal_left = portals[0].left;
    var portal_right = portals[0].right;

    var left_idx: usize = 0;
    var right_idx: usize = 0;

    var i: usize = 1;
    while (i < portals.len) : (i += 1) {
        const p_left = portals[i].left;
        const p_right = portals[i].right;

        // --- Update right edge of funnel ---
        if (triArea2D(apex, portal_right, p_right) <= 0.0) {
            // p_right is to the right of or along apex->portal_right (widens or same) or narrows
            if (apex.sub(portal_right).lengthSq() < 1e-8 or triArea2D(apex, portal_left, p_right) > 0.0) {
                // Tighten right edge
                portal_right = p_right;
                right_idx = i;
            } else {
                // Funnel collapsed: p_right crossed over left edge!
                try path.append(allocator, portal_left);
                apex = portal_left;
                portal_left = apex;
                portal_right = apex;
                left_idx += 1;
                i = left_idx;
                if (i < portals.len) {
                    portal_left = portals[i].left;
                    portal_right = portals[i].right;
                    right_idx = i;
                }
                continue;
            }
        }

        // --- Update left edge of funnel ---
        if (triArea2D(apex, portal_left, p_left) >= 0.0) {
            if (apex.sub(portal_left).lengthSq() < 1e-8 or triArea2D(apex, portal_right, p_left) < 0.0) {
                // Tighten left edge
                portal_left = p_left;
                left_idx = i;
            } else {
                // Funnel collapsed: p_left crossed over right edge!
                try path.append(allocator, portal_right);
                apex = portal_right;
                portal_left = apex;
                portal_right = apex;
                right_idx += 1;
                i = right_idx;
                if (i < portals.len) {
                    portal_left = portals[i].left;
                    portal_right = portals[i].right;
                    left_idx = i;
                }
                continue;
            }
        }
    }

    // End point check with final funnel boundaries
    if (path.items.len == 0 or path.items[path.items.len - 1].sub(end).lengthSq() > 1e-6) {
        try path.append(allocator, end);
    }

    return try path.toOwnedSlice(allocator);
}
