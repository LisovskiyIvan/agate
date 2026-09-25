//! Tests for `light_selection.zig` (moved from `light_selection.zig` inline blocks).
const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const lights = @import("../lights.zig");
const PointLight = lights.PointLight;
const SpotLight = lights.SpotLight;
const sel = @import("light_selection.zig");
const scorePoint = sel.scorePoint;
const selectPoint = sel.selectPoint;
const selectSpot = sel.selectSpot;
const selectTopKHysteresis = sel.selectTopKHysteresis;
const Hysteresis = sel.Hysteresis;

const HPoint = Hysteresis(PointLight, 2, scorePoint);

fn hystLights() [3]PointLight {
    // Raw scores at the origin (range 10): ls[0] = 3*10/2 = 15,
    // ls[1] = 1*10/5 = 2, ls[2] = intensity*10/10 = intensity.
    return .{
        PointLight{ .position = Vec3.new(1, 0, 0), .intensity = 3.0, .range = 10.0 },
        PointLight{ .position = Vec3.new(2, 0, 0), .intensity = 1.0, .range = 10.0 },
        PointLight{ .position = Vec3.new(3, 0, 0), .intensity = 0.4, .range = 10.0 },
    };
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

test "selectTopKHysteresis: incumbency bonus keeps a borderline incumbent" {
    var ls = hystLights();
    const all = [_]*PointLight{ &ls[0], &ls[1], &ls[2] };
    // Challenger ls[2] at raw 2.2 beats incumbent ls[1]'s raw 2, but not the
    // bonus-boosted 2 * 1.5 = 3 — the incumbent keeps its slot.
    ls[2].intensity = 2.2;
    var out: [2]*PointLight = undefined;
    const prev = [_]?*PointLight{ &ls[0], &ls[1] };
    const n = selectTopKHysteresis(PointLight, &all, Vec3.zero, &scorePoint, &prev, 0.5, &out);
    try std.testing.expectEqual(@as(usize, 2), n);
    try std.testing.expect(out[0] == &ls[0]);
    try std.testing.expect(out[1] == &ls[1]); // kept only thanks to the bonus
}

test "selectTopKHysteresis: a decisive challenger displaces the incumbent" {
    var ls = hystLights();
    const all = [_]*PointLight{ &ls[0], &ls[1], &ls[2] };
    // Raw 40 beats the boosted incumbent ls[1] (3) by a wide margin.
    ls[2].intensity = 40.0;
    var out: [2]*PointLight = undefined;
    const prev = [_]?*PointLight{ &ls[0], &ls[1] };
    _ = selectTopKHysteresis(PointLight, &all, Vec3.zero, &scorePoint, &prev, 0.5, &out);
    try std.testing.expect(out[0] == &ls[2]);
    try std.testing.expect(out[1] == &ls[0]);
}

test "Hysteresis: fresh light fades in from zero" {
    var ls = hystLights();
    const all = [_]*PointLight{ &ls[0], &ls[1] };
    var h = HPoint{};
    var out: [2]HPoint.Packed = undefined;

    const dt: f32 = 0.25 / 2.0; // two frames per fade half
    var n = h.update(&all, Vec3.zero, dt, 0.5, 0.25, &out);
    try std.testing.expectEqual(@as(usize, 2), n);
    // First frame: both lights just entered, factor is still 0.
    try std.testing.expectEqual(@as(f32, 0.0), out[0].factor);
    try std.testing.expectEqual(@as(f32, 0.0), out[1].factor);

    n = h.update(&all, Vec3.zero, dt, 0.5, 0.25, &out);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), out[0].factor, 1e-6);

    _ = h.update(&all, Vec3.zero, dt, 0.5, 0.25, &out);
    n = h.update(&all, Vec3.zero, dt, 0.5, 0.25, &out);
    try std.testing.expectEqual(@as(f32, 1.0), out[0].factor); // fade done
}

test "Hysteresis: displaced light fades out in its slot, challenger waits then fades in" {
    var ls = hystLights();
    var all3 = [_]*PointLight{ &ls[0], &ls[1], &ls[2] };
    var h = HPoint{};
    var out: [2]HPoint.Packed = undefined;
    const dt: f32 = 0.25 / 2.0;

    // Warm up: ls[0] and ls[1] hold the slots at full strength.
    var n = h.update(all3[0..2], Vec3.zero, dt, 0.5, 0.25, &out);
    n = h.update(all3[0..2], Vec3.zero, dt, 0.5, 0.25, &out);
    n = h.update(all3[0..2], Vec3.zero, dt, 0.5, 0.25, &out);
    n = h.update(all3[0..2], Vec3.zero, dt, 0.5, 0.25, &out);
    try std.testing.expectEqual(@as(f32, 1.0), out[0].factor);

    // ls[2] becomes dominant (raw 4 beats the boosted incumbent ls[1] at 3):
    // it must displace ls[1], whose raw score is only 2.
    ls[2].intensity = 4.0;
    n = h.update(&all3, Vec3.zero, dt, 0.5, 0.25, &out);
    // Swap frame: ls[1] starts fading out at factor 1; no free slot, so
    // ls[2] waits and is NOT packed.
    try std.testing.expectEqual(@as(usize, 2), n);
    var saw_fading: bool = false;
    var saw_waiting: bool = false;
    for (out[0..n]) |p| {
        if (p.light == &ls[1]) {
            saw_fading = true;
            try std.testing.expect(p.factor > 0.0);
        }
        if (p.light == &ls[2]) saw_waiting = true;
    }
    try std.testing.expect(saw_fading);
    try std.testing.expect(!saw_waiting);

    // The exit takes T total: two dt=T/2 updates later ls[1] reached zero
    // and its slot is released; ls[2] claims it from factor 0.
    n = h.update(&all3, Vec3.zero, dt, 0.5, 0.25, &out);
    n = h.update(&all3, Vec3.zero, dt, 0.5, 0.25, &out);
    var found_challenger: ?HPoint.Packed = null;
    for (out[0..n]) |p| {
        if (p.light == &ls[2]) found_challenger = p;
        if (p.light == &ls[1]) try std.testing.expectEqual(@as(f32, 0.0), p.factor);
    }
    try std.testing.expect(found_challenger != null);
    try std.testing.expectEqual(@as(f32, 0.0), found_challenger.?.factor);

    // The fade-in then runs over the next T: halfway there by the next
    // update, complete one update after that.
    n = h.update(&all3, Vec3.zero, dt, 0.5, 0.25, &out);
    for (out[0..n]) |p| {
        if (p.light == &ls[2]) try std.testing.expectApproxEqAbs(@as(f32, 0.5), p.factor, 1e-6);
    }
    n = h.update(&all3, Vec3.zero, dt, 0.5, 0.25, &out);
    for (out[0..n]) |p| {
        if (p.light == &ls[2]) try std.testing.expectEqual(@as(f32, 1.0), p.factor);
    }
}

test "Hysteresis: re-selecting a fading light reverses the fade smoothly" {
    var ls = hystLights();
    var all3 = [_]*PointLight{ &ls[0], &ls[1], &ls[2] };
    var h = HPoint{};
    var out: [2]HPoint.Packed = undefined;
    const dt: f32 = 0.25 / 4.0;

    for (0..8) |_| _ = h.update(all3[0..2], Vec3.zero, dt, 0.5, 0.25, &out);

    // Displace ls[1] with a strong ls[2]; ls[1] starts fading out.
    ls[2].intensity = 4.0;
    _ = h.update(&all3, Vec3.zero, dt, 0.5, 0.25, &out);
    // Halfway through the fade-out ls[2] weakens again: ls[1] wins its slot
    // back. Its factor must CONTINUE from the exit level, not jump to 0.
    ls[2].intensity = 0.0; // disabled-level score (still enabled, score 0)
    const n = h.update(&all3, Vec3.zero, dt, 0.5, 0.25, &out);
    var ls1_factor: f32 = 1.0;
    var ls2_factor: f32 = -1.0;
    for (out[0..n]) |p| {
        if (p.light == &ls[1]) ls1_factor = p.factor;
        if (p.light == &ls[2]) ls2_factor = p.factor;
    }
    // ls[1] reversed mid-fade: still partially visible, not dropped.
    try std.testing.expect(ls1_factor > 0.0 and ls1_factor < 1.0);
    // ls[2] lost the selection; if it was packed fading in, it must now be
    // on its way down (never instant vanish while visible).
    if (ls2_factor >= 0.0) try std.testing.expect(ls2_factor < 1.0);
}

test "Hysteresis: destroyed light pruned without packing a stale pointer" {
    var ls = hystLights();
    const all = [_]*PointLight{ &ls[0], &ls[1] };
    var h = HPoint{};
    var out: [2]HPoint.Packed = undefined;
    for (0..4) |_| _ = h.update(&all, Vec3.zero, 0.25, 0.5, 0.25, &out);

    // Simulate ls[1] destruction: rebuild the list without it. The tracker
    // must release its slot the same frame and never emit the pointer.
    const after = [_]*PointLight{&ls[0]};
    const n = h.update(&after, Vec3.zero, 0.25, 0.5, 0.25, &out);
    try std.testing.expectEqual(@as(usize, 1), n);
    try std.testing.expect(out[0].light == &ls[0]);
    // The freed slot is reusable immediately.
    var ls_new = PointLight{ .position = Vec3.new(1, 0, 0), .intensity = 1.0, .range = 10.0 };
    const after2 = [_]*PointLight{ &ls[0], &ls_new };
    const n2 = h.update(&after2, Vec3.zero, 0.25, 0.5, 0.25, &out);
    try std.testing.expectEqual(@as(usize, 2), n2);
}
