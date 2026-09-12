const std = @import("std");

const math = @import("math");
const Vec3 = math.Vec3;

const lights = @import("../lights.zig");
const PointLight = lights.PointLight;
const SpotLight = lights.SpotLight;

// Camera-relative significance model (fixed contract):
//     score = intensity * range / (1 + dist_sq)
// where dist_sq is the squared distance from the light position to the
// camera. Rationale: quadratic falloff mirrors inverse-square decay
// without a singularity (the "+1" keeps the score finite at zero
// distance, converging to intensity * range), and range scales the score
// linearly as a proxy for reach. Spot cone angles are intentionally
// ignored: only the distance to the spot position matters, so a spot
// pointing away from the camera still scores by proximity. Cone culling
// is the caller's job. score*() are pure geometry helpers and do NOT
// check is_enabled; select*() skips disabled lights.
pub fn scorePoint(light: *const PointLight, camera_pos: Vec3) f32 {
    const dist_sq = Vec3.distanceSq(light.position, camera_pos);
    return light.intensity * light.range / (1.0 + dist_sq);
}

pub fn scoreSpot(light: *const SpotLight, camera_pos: Vec3) f32 {
    const dist_sq = Vec3.distanceSq(light.position, camera_pos);
    return light.intensity * light.range / (1.0 + dist_sq);
}

// Single-pass top-k insertion: no allocations, input stays untouched, and
// ties resolve to the lowest source index (stable, deterministic across frames).
// Duplicate pointers are distinct entries distinguished by source index.
// Cost is O(all.len * min(out.len, STACK_LIMIT)), evaluating scoreFn exactly
// once per enabled candidate.
fn selectTopK(
    comptime T: type,
    all: []const *T,
    camera_pos: Vec3,
    out: []*T,
    scoreFn: *const fn (*const T, Vec3) f32,
) usize {
    if (out.len == 0 or all.len == 0) return 0;
    const STACK_LIMIT = 128;
    const max_k = @min(out.len, STACK_LIMIT);

    var scores: [STACK_LIMIT]f32 = undefined;
    var indices: [STACK_LIMIT]usize = undefined;
    var count: usize = 0;

    for (all, 0..) |candidate, i| {
        if (!candidate.is_enabled) continue;
        const s = scoreFn(candidate, camera_pos);

        if (count == max_k) {
            const worst_s = scores[count - 1];
            const worst_i = indices[count - 1];
            if (s < worst_s or (s == worst_s and i >= worst_i)) {
                continue;
            }
        }

        var insert_pos: usize = count;
        for (0..count) |j| {
            if (s > scores[j] or (s == scores[j] and i < indices[j])) {
                insert_pos = j;
                break;
            }
        }

        if (insert_pos < max_k) {
            const shift_end = if (count < max_k) count else max_k - 1;
            var k_idx = shift_end;
            while (k_idx > insert_pos) : (k_idx -= 1) {
                out[k_idx] = out[k_idx - 1];
                scores[k_idx] = scores[k_idx - 1];
                indices[k_idx] = indices[k_idx - 1];
            }
            out[insert_pos] = candidate;
            scores[insert_pos] = s;
            indices[insert_pos] = i;
            if (count < max_k) count += 1;
        }
    }
    return count;
}

// Copies the most significant enabled lights into out (best first) and
// returns how many were written: min(out.len, enabled count).
pub fn selectPoint(lights_in: []const *PointLight, camera_pos: Vec3, out: []*PointLight) usize {
    return selectTopK(PointLight, lights_in, camera_pos, out, &scorePoint);
}

pub fn selectSpot(lights_in: []const *SpotLight, camera_pos: Vec3, out: []*SpotLight) usize {
    return selectTopK(SpotLight, lights_in, camera_pos, out, &scoreSpot);
}

test "selectPoint sorts by significance" {
    var l0 = PointLight{ .position = Vec3.new(1, 0, 0), .intensity = 1.0, .range = 10.0 };
    var l1 = PointLight{ .position = Vec3.new(2, 0, 0), .intensity = 1.0, .range = 10.0 };
    var l2 = PointLight{ .position = Vec3.zero, .intensity = 3.0, .range = 10.0 };
    var l3 = PointLight{ .position = Vec3.new(3, 0, 0), .intensity = 4.0, .range = 10.0 };
    var l4 = PointLight{ .position = Vec3.new(10, 0, 0), .intensity = 100.0, .range = 10.0 };
    var l5 = PointLight{ .position = Vec3.zero, .intensity = 1.0, .range = 10.0 };
    // Scores at origin: l2=30, l5=10, l4~=9.9, l0=5, l3=4, l1=2.
    const all = [_]*PointLight{ &l0, &l1, &l2, &l3, &l4, &l5 };
    var buf: [6]*PointLight = undefined;
    const n = selectPoint(&all, Vec3.zero, &buf);
    try std.testing.expectEqual(@as(usize, 6), n);
    const want = [_]*PointLight{ &l2, &l5, &l4, &l0, &l3, &l1 };
    for (want, buf) |w, g| try std.testing.expect(w == g);
}

test "selectPoint skips disabled lights" {
    var strong = PointLight{ .position = Vec3.zero, .intensity = 1000.0, .is_enabled = false };
    var weak = PointLight{ .position = Vec3.new(5, 0, 0), .intensity = 1.0 };
    const all = [_]*PointLight{ &strong, &weak };
    var buf: [2]*PointLight = undefined;
    const n = selectPoint(&all, Vec3.zero, &buf);
    try std.testing.expectEqual(@as(usize, 1), n);
    try std.testing.expect(buf[0] == &weak);
}

test "selectPoint with empty input returns zero" {
    var buf: [4]*PointLight = undefined;
    try std.testing.expectEqual(@as(usize, 0), selectPoint(&[_]*PointLight{}, Vec3.zero, &buf));
    var one = PointLight{};
    const all = [_]*PointLight{&one};
    try std.testing.expectEqual(@as(usize, 0), selectPoint(&all, Vec3.zero, &[_]*PointLight{}));
}

test "selectPoint caps output at out.len best" {
    var l0 = PointLight{ .position = Vec3.new(1, 0, 0), .intensity = 1.0, .range = 10.0 };
    var l1 = PointLight{ .position = Vec3.new(2, 0, 0), .intensity = 1.0, .range = 10.0 };
    var l2 = PointLight{ .position = Vec3.zero, .intensity = 3.0, .range = 10.0 };
    var l3 = PointLight{ .position = Vec3.new(3, 0, 0), .intensity = 4.0, .range = 10.0 };
    var l4 = PointLight{ .position = Vec3.new(10, 0, 0), .intensity = 100.0, .range = 10.0 };
    var l5 = PointLight{ .position = Vec3.zero, .intensity = 1.0, .range = 10.0 };
    const all = [_]*PointLight{ &l0, &l1, &l2, &l3, &l4, &l5 };
    var buf: [4]*PointLight = undefined;
    const n = selectPoint(&all, Vec3.zero, &buf);
    try std.testing.expectEqual(@as(usize, 4), n);
    const want = [_]*PointLight{ &l2, &l5, &l4, &l0 };
    for (want, buf) |w, g| try std.testing.expect(w == g);
}

test "selectPoint is stable for equal scores" {
    var a = PointLight{ .position = Vec3.new(1, 0, 0), .intensity = 2.0, .range = 5.0 };
    var b = PointLight{ .position = Vec3.new(-1, 0, 0), .intensity = 2.0, .range = 5.0 };
    var c = PointLight{ .position = Vec3.new(0, 1, 0), .intensity = 2.0, .range = 5.0 };
    const all = [_]*PointLight{ &a, &b, &c };
    var buf: [2]*PointLight = undefined;
    const n = selectPoint(&all, Vec3.zero, &buf);
    try std.testing.expectEqual(@as(usize, 2), n);
    try std.testing.expect(buf[0] == &a);
    try std.testing.expect(buf[1] == &b);
}

test "selectPoint prefers nearer light, all else equal" {
    var near = PointLight{ .position = Vec3.new(1, 0, 0), .intensity = 1.0, .range = 10.0 };
    var far = PointLight{ .position = Vec3.new(10, 0, 0), .intensity = 1.0, .range = 10.0 };
    try std.testing.expect(scorePoint(&near, Vec3.zero) > scorePoint(&far, Vec3.zero));
    const all = [_]*PointLight{ &far, &near };
    var buf: [2]*PointLight = undefined;
    const n = selectPoint(&all, Vec3.zero, &buf);
    try std.testing.expectEqual(@as(usize, 2), n);
    try std.testing.expect(buf[0] == &near);
    try std.testing.expect(buf[1] == &far);
}

test "selectSpot sorts by significance and caps at out.len" {
    var s0 = SpotLight{ .position = Vec3.new(1, 0, 0), .intensity = 1.0, .range = 10.0 };
    var s1 = SpotLight{ .position = Vec3.zero, .intensity = 5.0, .range = 10.0 };
    var s2 = SpotLight{ .position = Vec3.new(2, 0, 0), .intensity = 9.0, .range = 10.0 };
    // Scores at origin: s1=50, s2=18, s0=5.
    const all = [_]*SpotLight{ &s0, &s1, &s2 };
    var buf: [2]*SpotLight = undefined;
    const n = selectSpot(&all, Vec3.zero, &buf);
    try std.testing.expectEqual(@as(usize, 2), n);
    try std.testing.expect(buf[0] == &s1);
    try std.testing.expect(buf[1] == &s2);
}

test "selectSpot skips disabled lights" {
    var strong = SpotLight{ .position = Vec3.zero, .intensity = 1000.0, .is_enabled = false };
    var weak = SpotLight{ .position = Vec3.new(5, 0, 0), .intensity = 1.0 };
    const all = [_]*SpotLight{ &strong, &weak };
    var buf: [2]*SpotLight = undefined;
    const n = selectSpot(&all, Vec3.zero, &buf);
    try std.testing.expectEqual(@as(usize, 1), n);
    try std.testing.expect(buf[0] == &weak);
}

test "selectSpot with empty input returns zero" {
    var buf: [2]*SpotLight = undefined;
    try std.testing.expectEqual(@as(usize, 0), selectSpot(&[_]*SpotLight{}, Vec3.zero, &buf));
}

test "selectSpot is stable for equal scores" {
    var a = SpotLight{ .position = Vec3.new(1, 0, 0), .intensity = 2.0, .range = 5.0 };
    var b = SpotLight{ .position = Vec3.new(-1, 0, 0), .intensity = 2.0, .range = 5.0 };
    const all = [_]*SpotLight{ &a, &b };
    var buf: [2]*SpotLight = undefined;
    const n = selectSpot(&all, Vec3.zero, &buf);
    try std.testing.expectEqual(@as(usize, 2), n);
    try std.testing.expect(buf[0] == &a);
    try std.testing.expect(buf[1] == &b);
}

test "selectPoint handles duplicate pointers deterministically" {
    var a = PointLight{ .position = Vec3.new(1, 0, 0), .intensity = 2.0, .range = 5.0 };
    const all = [_]*PointLight{ &a, &a };
    var buf: [2]*PointLight = undefined;
    const n = selectPoint(&all, Vec3.zero, &buf);
    try std.testing.expectEqual(@as(usize, 2), n);
    try std.testing.expect(buf[0] == &a);
    try std.testing.expect(buf[1] == &a);
}
