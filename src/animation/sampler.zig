const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const Quat = math.Quat;
const easing_mod = @import("easing.zig");
pub const EasingType = easing_mod.EasingType;

pub const AnimationPath = enum {
    translation,
    rotation,
    scale,
    weights,
};

pub const AnimationInterpolation = enum {
    linear,
    step,
    cubic_spline,
};

/// Scalar cubic Hermite basis: interpolates p0 -> p1 over s in [0, 1] with
/// endpoint slopes m0 (at p0) and m1 (at p1). Callers pre-scale glTF
/// per-second tangents by the keyframe interval: m = tangent * dt.
/// Exact on the endpoints: hermite(p0, m0, p1, m1, 0) == p0, (... , 1) == p1.
fn hermiteScalar(p0: f32, m0: f32, p1: f32, m1: f32, s: f32) f32 {
    const s2 = s * s;
    const s3 = s2 * s;
    return (2.0 * s3 - 3.0 * s2 + 1.0) * p0 +
        (s3 - 2.0 * s2 + s) * m0 +
        (-2.0 * s3 + 3.0 * s2) * p1 +
        (s3 - s2) * m1;
}

/// A named timestamp marker on an animation timeline. Owned by the group via
/// setEvents (names are duplicated); the slice passed to setEvents is only
/// borrowed. Events outside the group's [from, to] range never fire.
pub const AnimationEvent = struct {
    time: f32,
    name: []const u8,
};

pub const AnimationSampler = struct {
    timestamps: []const f32,
    outputs: []const f32,
    interpolation: AnimationInterpolation = .linear,

    /// Cubic (glTF CUBICSPLINE) layout: outputs hold stride*3 floats per key
    /// in (in-tangent, value, out-tangent) order, so a valid track needs at
    /// least keys * stride * 3 floats. Returns the number of keyframes that
    /// can actually be sampled (timestamps capped by the outputs capacity).
    /// LINEAR/STEP tracks do not use this helper.
    fn cubicFrames(self: AnimationSampler, stride: usize) usize {
        if (self.timestamps.len == 0 or stride == 0) return 0;
        return @min(self.timestamps.len, self.outputs.len / (stride * 3));
    }

    fn cubicValue3(self: AnimationSampler, key: usize) Vec3 {
        const base = (key * 3 + 1) * 3;
        return Vec3.new(self.outputs[base], self.outputs[base + 1], self.outputs[base + 2]);
    }

    fn cubicQuatValue(self: AnimationSampler, key: usize) Quat {
        const base = (key * 3 + 1) * 4;
        return (Quat{
            .x = self.outputs[base],
            .y = self.outputs[base + 1],
            .z = self.outputs[base + 2],
            .w = self.outputs[base + 3],
        }).normalize();
    }

    /// True cubic-spline (Hermite) evaluation for a vec3 track. Tangents are
    /// per-second slopes scaled by the keyframe interval dt, per the glTF
    /// spec. Easing is ignored here on purpose: the Hermite curve already
    /// defines the intra-keyframe shape, easing only warps the LINEAR/STEP
    /// factor. Short/truncated buffers never crash: sampling degrades to the
    /// closest usable keyframe value (or zero when nothing is usable).
    fn sampleVec3Cubic(self: AnimationSampler, time: f32) Vec3 {
        const usable = self.cubicFrames(3);
        if (usable == 0) return Vec3.zero;
        if (usable == 1 or time <= self.timestamps[0]) return self.cubicValue3(0);
        if (time >= self.timestamps[usable - 1]) return self.cubicValue3(usable - 1);
        var idx = self.findKeyframeIndex(time);
        if (idx >= usable - 1) idx = usable - 2;
        const t0 = self.timestamps[idx];
        const t1 = self.timestamps[idx + 1];
        const dt = t1 - t0;
        if (dt <= 0.0) return self.cubicValue3(idx);
        const s = std.math.clamp((time - t0) / dt, 0.0, 1.0);
        const k0 = idx * 9;
        const k1 = (idx + 1) * 9;
        return Vec3.new(
            hermiteScalar(self.outputs[k0 + 3], self.outputs[k0 + 6] * dt, self.outputs[k1 + 3], self.outputs[k1] * dt, s),
            hermiteScalar(self.outputs[k0 + 4], self.outputs[k0 + 7] * dt, self.outputs[k1 + 4], self.outputs[k1 + 1] * dt, s),
            hermiteScalar(self.outputs[k0 + 5], self.outputs[k0 + 8] * dt, self.outputs[k1 + 5], self.outputs[k1 + 2] * dt, s),
        );
    }

    /// Quaternion cubic-spline: component-wise Hermite on the vec4 track,
    /// then normalized (glTF-correct). Antipodal keyframes (dot < 0) take the
    /// short path by negating the next value and its in-tangent together,
    /// which preserves the derivative direction.
    fn sampleQuatCubic(self: AnimationSampler, time: f32) Quat {
        const usable = self.cubicFrames(4);
        if (usable == 0) return Quat.identity;
        if (usable == 1 or time <= self.timestamps[0]) return self.cubicQuatValue(0);
        if (time >= self.timestamps[usable - 1]) return self.cubicQuatValue(usable - 1);
        var idx = self.findKeyframeIndex(time);
        if (idx >= usable - 1) idx = usable - 2;
        const t0 = self.timestamps[idx];
        const t1 = self.timestamps[idx + 1];
        const dt = t1 - t0;
        if (dt <= 0.0) return self.cubicQuatValue(idx);
        const s = std.math.clamp((time - t0) / dt, 0.0, 1.0);
        const k0 = idx * 12;
        const k1 = (idx + 1) * 12;
        const dot = self.outputs[k0 + 4] * self.outputs[k1 + 4] +
            self.outputs[k0 + 5] * self.outputs[k1 + 5] +
            self.outputs[k0 + 6] * self.outputs[k1 + 6] +
            self.outputs[k0 + 7] * self.outputs[k1 + 7];
        const flip: f32 = if (dot < 0.0) -1.0 else 1.0;
        return (Quat{
            .x = hermiteScalar(self.outputs[k0 + 4], self.outputs[k0 + 8] * dt, flip * self.outputs[k1 + 4], flip * self.outputs[k1] * dt, s),
            .y = hermiteScalar(self.outputs[k0 + 5], self.outputs[k0 + 9] * dt, flip * self.outputs[k1 + 5], flip * self.outputs[k1 + 1] * dt, s),
            .z = hermiteScalar(self.outputs[k0 + 6], self.outputs[k0 + 10] * dt, flip * self.outputs[k1 + 6], flip * self.outputs[k1 + 2] * dt, s),
            .w = hermiteScalar(self.outputs[k0 + 7], self.outputs[k0 + 11] * dt, flip * self.outputs[k1 + 7], flip * self.outputs[k1 + 3] * dt, s),
        }).normalize();
    }

    pub fn sampleVec3(self: AnimationSampler, time: f32) Vec3 {
        return self.sampleVec3Eased(time, .linear);
    }

    /// Samples a vec3 track, warping the intra-keyframe factor with an easing
    /// curve. STEP ignores easing. CUBICSPLINE evaluates the true Hermite
    /// spline (easing intentionally ignored: it only warps the linear
    /// factor, never the Hermite shape).
    pub fn sampleVec3Eased(self: AnimationSampler, time: f32, easing: EasingType) Vec3 {
        if (self.timestamps.len == 0) return Vec3.zero;
        if (self.interpolation == .cubic_spline) return self.sampleVec3Cubic(time);
        if (time <= self.timestamps[0]) {
            return Vec3.new(self.outputs[0], self.outputs[1], self.outputs[2]);
        }
        const last_idx = self.timestamps.len - 1;
        if (time >= self.timestamps[last_idx]) {
            const base = last_idx * 3;
            return Vec3.new(self.outputs[base], self.outputs[base + 1], self.outputs[base + 2]);
        }

        const idx = self.findKeyframeIndex(time);
        const t0 = self.timestamps[idx];
        const t1 = self.timestamps[idx + 1];
        const raw = if (t1 > t0) (time - t0) / (t1 - t0) else 0.0;

        const base0 = idx * 3;
        const v0 = Vec3.new(self.outputs[base0], self.outputs[base0 + 1], self.outputs[base0 + 2]);

        if (self.interpolation == .step) return v0;

        const factor = easing_mod.evaluate(easing, raw);
        const base1 = (idx + 1) * 3;
        const v1 = Vec3.new(self.outputs[base1], self.outputs[base1 + 1], self.outputs[base1 + 2]);
        return Vec3.lerp(v0, v1, factor);
    }

    pub fn sampleQuat(self: AnimationSampler, time: f32) Quat {
        return self.sampleQuatEased(time, .linear);
    }

    /// Quaternion variant of sampleVec3Eased: slerp factor is eased, STEP
    /// ignores easing, CUBICSPLINE evaluates component-wise Hermite plus
    /// normalize (easing ignored, same rule as for vec3).
    pub fn sampleQuatEased(self: AnimationSampler, time: f32, easing: EasingType) Quat {
        if (self.timestamps.len == 0) return Quat.identity;
        if (self.interpolation == .cubic_spline) return self.sampleQuatCubic(time);
        if (time <= self.timestamps[0]) {
            const q = Quat{
                .x = self.outputs[0],
                .y = self.outputs[1],
                .z = self.outputs[2],
                .w = self.outputs[3],
            };
            return q.normalize();
        }
        const last_idx = self.timestamps.len - 1;
        if (time >= self.timestamps[last_idx]) {
            const base = last_idx * 4;
            const q = Quat{
                .x = self.outputs[base],
                .y = self.outputs[base + 1],
                .z = self.outputs[base + 2],
                .w = self.outputs[base + 3],
            };
            return q.normalize();
        }

        const idx = self.findKeyframeIndex(time);
        const t0 = self.timestamps[idx];
        const t1 = self.timestamps[idx + 1];
        const raw = if (t1 > t0) (time - t0) / (t1 - t0) else 0.0;

        const base0 = idx * 4;
        const q0 = Quat{
            .x = self.outputs[base0],
            .y = self.outputs[base0 + 1],
            .z = self.outputs[base0 + 2],
            .w = self.outputs[base0 + 3],
        };

        if (self.interpolation == .step) return q0.normalize();

        const factor = easing_mod.evaluate(easing, raw);
        const base1 = (idx + 1) * 4;
        const q1 = Quat{
            .x = self.outputs[base1],
            .y = self.outputs[base1 + 1],
            .z = self.outputs[base1 + 2],
            .w = self.outputs[base1 + 3],
        };
        return Quat.slerp(q0, q1, factor);
    }

    /// Samples a weights (morph) track with out.len values per keyframe.
    /// Per-component mirror of sampleVec3Eased: STEP holds the keyframe,
    /// LINEAR lerps with the eased factor, CUBICSPLINE runs Hermite per
    /// target (easing ignored). Undersized outputs or missing keys leave out
    /// untouched; truncated cubic buffers clamp to the closest usable value.
    pub fn sampleWeightsInto(self: AnimationSampler, time: f32, out: []f32, easing: EasingType) void {
        const n = out.len;
        if (n == 0 or self.timestamps.len == 0) return;
        if (self.interpolation == .cubic_spline) {
            self.sampleWeightsCubic(time, out);
            return;
        }
        const frames = self.outputs.len / n;
        if (frames == 0) return;
        const keys = @min(self.timestamps.len, frames);
        if (keys == 0) return;
        if (time <= self.timestamps[0]) {
            @memcpy(out, self.outputs[0..n]);
            return;
        }
        const last = keys - 1;
        if (time >= self.timestamps[last]) {
            @memcpy(out, self.outputs[last * n .. last * n + n]);
            return;
        }
        var idx = self.findKeyframeIndex(time);
        if (idx >= last) idx = last - 1;
        const t0 = self.timestamps[idx];
        const t1 = self.timestamps[idx + 1];
        const raw = if (t1 > t0) (time - t0) / (t1 - t0) else 0.0;

        const base0 = idx * n;
        if (self.interpolation == .step) {
            @memcpy(out, self.outputs[base0 .. base0 + n]);
            return;
        }
        const factor = easing_mod.evaluate(easing, raw);
        const base1 = (idx + 1) * n;
        for (0..n) |j| {
            out[j] = self.outputs[base0 + j] + (self.outputs[base1 + j] - self.outputs[base0 + j]) * factor;
        }
    }

    /// Per-target Hermite mirror of sampleVec3Cubic: each keyframe holds
    /// (in-tangent, value, out-tangent) per morph target, i.e. 3*n floats.
    fn sampleWeightsCubic(self: AnimationSampler, time: f32, out: []f32) void {
        const n = out.len;
        const usable = self.cubicFrames(n);
        if (usable == 0) return;
        if (usable == 1 or time <= self.timestamps[0]) {
            @memcpy(out, self.outputs[n .. n + n]);
            return;
        }
        if (time >= self.timestamps[usable - 1]) {
            const base = ((usable - 1) * 3 + 1) * n;
            @memcpy(out, self.outputs[base .. base + n]);
            return;
        }
        var idx = self.findKeyframeIndex(time);
        if (idx >= usable - 1) idx = usable - 2;
        const t0 = self.timestamps[idx];
        const t1 = self.timestamps[idx + 1];
        const dt = t1 - t0;
        if (dt <= 0.0) {
            const base = (idx * 3 + 1) * n;
            @memcpy(out, self.outputs[base .. base + n]);
            return;
        }
        const s = std.math.clamp((time - t0) / dt, 0.0, 1.0);
        const k0 = idx * n * 3;
        const k1 = (idx + 1) * n * 3;
        for (0..n) |j| {
            out[j] = hermiteScalar(
                self.outputs[k0 + n + j],
                self.outputs[k0 + 2 * n + j] * dt,
                self.outputs[k1 + n + j],
                self.outputs[k1 + j] * dt,
                s,
            );
        }
    }

    fn findKeyframeIndex(self: AnimationSampler, time: f32) usize {
        if (self.timestamps.len < 2) return 0;
        var low: usize = 0;
        var high: usize = self.timestamps.len - 1;
        while (low < high - 1) {
            const mid = low + (high - low) / 2;
            if (self.timestamps[mid] <= time) {
                low = mid;
            } else {
                high = mid;
            }
        }
        return low;
    }
};
