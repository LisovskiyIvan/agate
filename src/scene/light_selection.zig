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

// -- Hysteresis -------------------------------------------------------------
