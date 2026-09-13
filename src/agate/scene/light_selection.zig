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
// ties resolve to the lowest source index (stable, deterministic across
// frames). Duplicate pointers are distinct entries distinguished by source
// index. Cost is O(all.len * min(out.len, STACK_LIMIT)), evaluating the
// score exactly once per enabled candidate; `score.raw()` lets the caller
// decorate the base score (the hysteresis flavor boosts incumbents there).
fn insertTopK(
    comptime T: type,
    all: []const *T,
    camera_pos: Vec3,
    out: []*T,
    score: anytype,
) usize {
    if (out.len == 0 or all.len == 0) return 0;
    const STACK_LIMIT = 128;
    const max_k = @min(out.len, STACK_LIMIT);

    var buf_scores: [STACK_LIMIT]f32 = undefined;
    var buf_indices: [STACK_LIMIT]usize = undefined;
    var count: usize = 0;

    for (all, 0..) |candidate, i| {
        if (!candidate.is_enabled) continue;
        const s = score.raw(candidate, camera_pos);

        if (count == max_k) {
            const worst_s = buf_scores[count - 1];
            const worst_i = buf_indices[count - 1];
            if (s < worst_s or (s == worst_s and i >= worst_i)) {
                continue;
            }
        }

        var insert_pos: usize = count;
        for (0..count) |j| {
            if (s > buf_scores[j] or (s == buf_scores[j] and i < buf_indices[j])) {
                insert_pos = j;
                break;
            }
        }

        if (insert_pos < max_k) {
            const shift_end = if (count < max_k) count else max_k - 1;
            var k_idx = shift_end;
            while (k_idx > insert_pos) : (k_idx -= 1) {
                out[k_idx] = out[k_idx - 1];
                buf_scores[k_idx] = buf_scores[k_idx - 1];
                buf_indices[k_idx] = buf_indices[k_idx - 1];
            }
            out[insert_pos] = candidate;
            buf_scores[insert_pos] = s;
            buf_indices[insert_pos] = i;
            if (count < max_k) count += 1;
        }
    }
    return count;
}

fn plainScoreFn(comptime T: type, scoreFn: *const fn (*const T, Vec3) f32) plainScoreCtx(T) {
    return .{ .scoreFn = scoreFn };
}

fn plainScoreCtx(comptime T: type) type {
    return struct {
        scoreFn: *const fn (*const T, Vec3) f32,
        fn raw(self: @This(), l: *const T, p: Vec3) f32 {
            return self.scoreFn(l, p);
        }
    };
}

// Copies the most significant enabled lights into out (best first) and
// returns how many were written: min(out.len, enabled count).
pub fn selectPoint(lights_in: []const *PointLight, camera_pos: Vec3, out: []*PointLight) usize {
    return insertTopK(PointLight, lights_in, camera_pos, out, plainScoreFn(PointLight, &scorePoint));
}

pub fn selectSpot(lights_in: []const *SpotLight, camera_pos: Vec3, out: []*SpotLight) usize {
    return insertTopK(SpotLight, lights_in, camera_pos, out, plainScoreFn(SpotLight, &scoreSpot));
}

/// Hysteresis-aware top-k: lights chosen last frame get their raw score
/// multiplied by (1 + bonus), so a challenger must be decisively better to
/// displace an incumbent. This is what turns per-frame boundary flapping
/// (two lights trading 4th place every frame) into rare, decisive swaps.
pub fn selectTopKHysteresis(
    comptime T: type,
    all: []const *T,
    camera_pos: Vec3,
    scoreFn: *const fn (*const T, Vec3) f32,
    prev_chosen: []const ?*T,
    bonus: f32,
    out: []*T,
) usize {
    const Ctx = struct {
        scoreFn: *const fn (*const T, Vec3) f32,
        prev: []const ?*T,
        bonus: f32,
        fn raw(self: @This(), l: *const T, p: Vec3) f32 {
            var s = self.scoreFn(l, p);
            for (self.prev) |pc| {
                if (pc != null and pc.? == l) {
                    s *= 1.0 + self.bonus;
                    break;
                }
            }
            return s;
        }
    };
    return insertTopK(T, all, camera_pos, out, Ctx{
        .scoreFn = scoreFn,
        .prev = prev_chosen,
        .bonus = bonus,
    });
}

/// Sequential cross-fade state machine for the fixed light slots (4 point +
/// 2 spot in the shader uniform arrays). The shader cannot address more
/// lights than the array holds, so a true simultaneous cross-fade of a swap
/// (old light out while the new one in) is impossible with fixed slots; this
/// tracker instead makes transitions SEQUENTIAL: the displaced light keeps
/// its slot and fades out to zero, and only then does the waiting challenger
/// fade in from zero — the rendered total intensity never jumps.
///
/// Slot assignments are sticky: a light keeps its slot while it stays
/// selected, so per-slot data (e.g. spot shadow atlas tiles keyed by slot
/// index) stays stable. Fade state lives only on packed lights; a challenger
/// that found no free slot simply waits (its fade timer starts when it first
/// appears), and a fade reverses smoothly if the light is re-selected
/// mid-fade (the same `t` runs the other direction).
pub fn Hysteresis(comptime T: type, comptime slots: usize, comptime scoreFn: fn (*const T, Vec3) f32) type {
    return struct {
        const Self = @This();

        const FadeKind = enum { enter, exit };
        const Fade = struct { t: f32, kind: FadeKind };

        /// Lights chosen by the last update(); feeds the incumbency bonus.
        prev_chosen: [slots]?*T = [_]?*T{null} ** slots,
        /// Current slot -> light mapping. Null = free slot.
        packed_slots: [slots]?*T = [_]?*T{null} ** slots,
        /// Fade progress per packed light (indexed parallel to packed_slots).
        fades: [slots]?Fade = [_]?Fade{null} ** slots,

        pub const Packed = struct { light: *T, slot: usize, factor: f32 };

        pub fn reset(self: *Self) void {
            self.prev_chosen = [_]?*T{null} ** slots;
            self.packed_slots = [_]?*T{null} ** slots;
            self.fades = [_]?Fade{null} ** slots;
        }

        fn factorOf(fade: ?Fade, fade_time: f32) f32 {
            const f = fade orelse return 1.0;
            const t01 = @min(f.t / fade_time, 1.0);
            return switch (f.kind) {
                .enter => t01,
                .exit => 1.0 - t01,
            };
        }

        /// Advances fades by `dt`, re-selects the top-k lights (with the
        /// incumbency bonus), resolves sticky slot assignment with
        /// sequential enter/exit fades, and writes the packed subset into
        /// `out` (unordered; every entry carries its intensity multiplier).
        /// Lights that vanish from `all` or get disabled are pruned from the
        /// fade state the same frame, so stale pointers are never packed.
        pub fn update(
            self: *Self,
            all: []const *T,
            camera_pos: Vec3,
            dt: f32,
            bonus: f32,
            fade_time: f32,
            out: []Packed,
        ) usize {
            // 1) Advance fades of currently packed lights. A finished exit
            // frees its slot for this frame's assignment; a finished enter
            // just drops its entry (factor reaches steady 1.0).
            for (&self.packed_slots, 0..) |maybe_light, s| {
                const light = maybe_light orelse continue;
                var alive = false;
                for (all) |l| {
                    if (l == light and l.is_enabled) {
                        alive = true;
                        break;
                    }
                }
                if (!alive) {
                    // Destroyed or disabled since last frame: release the
                    // slot immediately (no fade — user intent, not a pop).
                    self.packed_slots[s] = null;
                    self.fades[s] = null;
                    continue;
                }
                if (self.fades[s]) |*f| {
                    f.t += dt;
                    if (f.t >= fade_time) {
                        // A finished exit releases the slot THIS frame so a
                        // waiting challenger can claim it (the exiting light
                        // reached factor 0, so the hand-off is seamless). A
                        // finished enter just reaches the steady factor 1.0.
                        if (f.kind == .exit) self.packed_slots[s] = null;
                        self.fades[s] = null;
                    }
                }
            }

            // 2) Re-select with the incumbency bonus.
            var chosen_buf: [slots]*T = undefined;
            const num_chosen = selectTopKHysteresis(
                T,
                all,
                camera_pos,
                scoreFn,
                &self.prev_chosen,
                bonus,
                &chosen_buf,
            );
            const chosen = chosen_buf[0..num_chosen];

            // 3) Sticky slot assignment. Pass A: keep every packed light that
            // is still chosen (steady factor 1 or an in-progress enter fade);
            // a packed light that is no longer chosen starts (or continues)
            // its exit fade IN its slot.
            for (&self.packed_slots, 0..) |maybe_light, s| {
                const light = maybe_light orelse continue;
                var still_chosen = false;
                for (chosen) |c| {
                    if (c == light) {
                        still_chosen = true;
                        break;
                    }
                }
                if (still_chosen) {
                    // Reverse direction if it was fading out.
                    if (self.fades[s]) |f| {
                        if (f.kind == .exit) {
                            self.fades[s] = .{ .t = fade_time - f.t, .kind = .enter };
                        }
                    }
                } else {
                    // Displaced: start (fresh exit from the steady level) or
                    // continue the exit fade in its slot. An in-progress
                    // enter converts to an exit from the same visual level;
                    // an ongoing exit just keeps advancing (step 1).
                    if (self.fades[s]) |f| {
                        if (f.kind == .enter) {
                            self.fades[s] = .{ .t = fade_time - f.t, .kind = .exit };
                        }
                    } else {
                        self.fades[s] = .{ .t = 0, .kind = .exit };
                    }
                }
            }

            // Pass B: chosen lights that hold no slot yet (fresh entrants or
            // challengers that waited for a slot) take freed slots, best
            // first. They start fading in from zero; if no slot is free the
            // light waits (stays unpacked, factor irrelevant until packed).
            for (chosen) |c| {
                var has_slot = false;
                for (self.packed_slots) |pl| {
                    if (pl != null and pl.? == c) {
                        has_slot = true;
                        break;
                    }
                }
                if (has_slot) continue;
                for (&self.packed_slots, 0..) |*pl, s| {
                    if (pl.* == null) {
                        pl.* = c;
                        self.fades[s] = .{ .t = 0, .kind = .enter };
                        break;
                    }
                }
            }

            // 4) Remember the chosen set for next frame's bonus.
            for (&self.prev_chosen, 0..) |*pc, i| {
                pc.* = if (i < chosen.len) chosen[i] else null;
            }

            // 5) Emit the packed subset with intensity multipliers.
            var count: usize = 0;
            for (&self.packed_slots, 0..) |maybe_light, s| {
                const light = maybe_light orelse continue;
                if (count == out.len) break;
                out[count] = .{
                    .light = light,
                    .slot = s,
                    .factor = factorOf(self.fades[s], fade_time),
                };
                count += 1;
            }
            return count;
        }
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

// -- Hysteresis -------------------------------------------------------------

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
