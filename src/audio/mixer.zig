//! Real-time mix core: block renderer + per-voice synth kernels.
//! Split out of `audio.zig` (facade).
//!
//! This module owns `renderFrames` (drain, filter sync, topological bus
//! order, voice collection, the silence fast-path, chunked block rendering
//! with leaf-to-root bus summation and soft clipping) plus the synth
//! kernels it dispatches to (`renderNoise`, `renderTone`, `renderSample`)
//! and the per-buffer `ActiveVoice` render state. The callback path stays
//! exactly as before the split: 0-allocation, lock-free (atomics only),
//! no synchronization or memory behavior touched.
//!
//! Anti-cycle rule (same as `profiler/*`, `particles/*`): `renderFrames`
//! takes the engine as `anytype` (a `*AudioEngine` from `engine.zig` in
//! practice) and this module never imports `engine.zig` or the `audio.zig`
//! facade back. Command draining reaches the `commands.zig` sibling
//! directly; master/capacity state reads go through the engine-resident
//! methods via `anytype`. Kernels and `ActiveVoice` are private here —
//! they were private before the split and only `renderFrames` uses them.

const std = @import("std");
const dsp = @import("dsp.zig");
const types = @import("types.zig");
const commands = @import("commands.zig");

const Voice = types.Voice;
const BiquadFilterType = dsp.BiquadFilterType;
const max_voices = types.max_voices;
const max_bus_capacity = types.max_bus_capacity;
const chunk_samples = types.chunk_samples;

// Local alias so the moved body stays byte-identical.
const drainCommands = commands.drainCommands;

/// Per-buffer render state for one voice, built once in `renderFrames`
/// so the inner sample loops do no sqrt, no voice scan and no pan math.
const ActiveVoice = struct {
    v: *Voice,
    kind: VoiceKind,
    gl: f32, // volume * master * left pan gain
    gr: f32, // volume * master * right pan gain
    duration: f32,
    inv_duration: f32, // 1 / max(duration, 1e-6)
    freq0: f32, // synth only: ramp slopes per second of voice time
    freq_slope: f32,
    cutoff0: f32,
    cutoff_slope: f32,
    clip: ?[]const f32 = null, // sample only: interleaved stereo frames
    frames: usize = 0,
    step: f64 = 1.0, // clip_rate / engine_rate * rate
    loop: bool = false,
    step_one: bool = false, // step == 1 from an integer pos: copy, no lerp
};

const VoiceKind = types.VoiceKind;

/// Mixes stereo-interleaved frames and advances voice state. Called by
/// the stream callback and directly by tests.
pub fn renderFrames(self: anytype, buffer: []f32) void {
    drainCommands(self);
    std.debug.assert(buffer.len % 2 == 0);
    const total: usize = buffer.len / 2;
    if (total == 0) return;
    const dt: f32 = 1.0 / self.sample_rate;
    const master: f32 = if (self.isMuted()) 0.0 else self.getMasterVolume();
    const cap = self.getBusCapacity();

    // 1. Update Biquad filter parameters if changed atomically
    for (0..cap) |i| {
        if (self.bus_active[i].load(.acquire)) {
            const f_type: BiquadFilterType = @enumFromInt(self.bus_filter_type[i].load(.acquire));
            const f_cut: f32 = @bitCast(self.bus_filter_cutoff[i].load(.acquire));
            const f_q: f32 = @bitCast(self.bus_filter_q[i].load(.acquire));
            if (self.bus_filters[i].filter_type != f_type or
                self.bus_filters[i].cutoff != f_cut or
                self.bus_filters[i].q != f_q or
                self.bus_filters[i].last_sample_rate != self.sample_rate)
            {
                self.bus_filters[i].setParams(f_type, f_cut, f_q, self.sample_rate);
            }
        }
    }

    // 2. Build topological evaluation order (leaves first, parents after children)
    var active_buses: [max_bus_capacity]u8 = undefined;
    var active_count: usize = 0;
    var heights: [max_bus_capacity]u8 = [_]u8{0} ** max_bus_capacity;
    var parents: [max_bus_capacity]u8 = undefined;

    for (0..cap) |i| {
        if (self.bus_active[i].load(.acquire)) {
            active_buses[active_count] = @intCast(i);
            active_count += 1;
            parents[i] = self.bus_parent[i].load(.acquire);
        }
    }

    if (active_count > 1) {
        var pass: usize = 0;
        while (pass < 8) : (pass += 1) {
            var changed = false;
            for (active_buses[0..active_count]) |b| {
                const p = parents[b];
                if (p < cap and p != b and self.bus_active[p].load(.acquire)) {
                    if (heights[p] <= heights[b]) {
                        heights[p] = heights[b] + 1;
                        changed = true;
                    }
                }
            }
            if (!changed) break;
        }

        var i: usize = 1;
        while (i < active_count) : (i += 1) {
            const key = active_buses[i];
            const key_h = heights[key];
            var j: usize = i;
            while (j > 0 and heights[active_buses[j - 1]] > key_h) : (j -= 1) {
                active_buses[j] = active_buses[j - 1];
            }
            active_buses[j] = key;
        }
    }

    // 3. Collect active voices
    var list: [max_voices]ActiveVoice = undefined;
    var voice_count: usize = 0;
    for (&self.voices) |*v| {
        if (!v.active) continue;
        const pan = std.math.clamp(v.pan, -1.0, 1.0);
        const g = v.volume;
        const dur = @max(v.duration, 1e-6);
        var a = ActiveVoice{
            .v = v,
            .kind = v.kind,
            .gl = g * @sqrt(0.5 * (1.0 - pan)),
            .gr = g * @sqrt(0.5 * (1.0 + pan)),
            .duration = v.duration,
            .inv_duration = 1.0 / dur,
            .freq0 = v.freq,
            .freq_slope = (v.freq_end - v.freq) / dur,
            .cutoff0 = v.cutoff,
            .cutoff_slope = (v.cutoff_end - v.cutoff) / dur,
            .step = v.sample_step,
            .loop = v.loop,
        };
        if (v.kind == .sample) {
            const clip = v.clip orelse {
                v.active = false;
                continue;
            };
            if (clip.frames == 0 or clip.sample_rate <= 0.0 or clip.samples.len < clip.frames * 2) {
                v.active = false;
                continue;
            }
            if (!std.math.isFinite(v.sample_step) or v.sample_step <= 0.0) {
                v.active = false;
                continue;
            }
            var pos = v.sample_pos;
            if (!std.math.isFinite(pos)) {
                v.active = false;
                continue;
            }
            const n: f64 = @floatFromInt(clip.frames);
            if (pos < 0.0 or pos >= n) {
                if (v.loop) {
                    pos = @mod(pos, n);
                    if (!(pos >= 0.0)) pos += n;
                    v.sample_pos = pos;
                } else {
                    v.active = false;
                    continue;
                }
            }
            a.clip = clip.samples[0 .. clip.frames * 2];
            a.frames = clip.frames;
            a.step_one = v.sample_step == 1.0 and pos == @floor(pos);
            list[voice_count] = a;
        } else {
            list[voice_count] = a;
        }
        voice_count += 1;
    }

    // Fast-path: if no voices, no active reverbs, and no active streams, silence buffer
    var any_reverb = false;
    for (&self.reverbs) |*r| {
        if (r.active) {
            any_reverb = true;
            break;
        }
    }
    var any_stream = false;
    for (&self.streams) |*maybe_s| {
        if (maybe_s.*) |s| {
            if (s.state.load(.acquire) == .playing) {
                any_stream = true;
                break;
            }
        }
    }
    if (voice_count == 0 and !any_reverb and !any_stream) {
        @memset(buffer, 0.0);
        return;
    }

    // 4. Chunked block rendering (64 frames = 128 samples per block)
    var offset: usize = 0;
    var master_chunk: [chunk_samples]f32 = undefined;
    var bus_has_audio: [max_bus_capacity]bool = undefined;

    while (offset < buffer.len) {
        const remaining = buffer.len - offset;
        const cur_samples = @min(remaining, chunk_samples);

        @memset(master_chunk[0..cur_samples], 0.0);
        for (active_buses[0..active_count]) |b| {
            @memset(self.bus_chunks[b][0..cur_samples], 0.0);
            bus_has_audio[b] = false;
        }

        for (list[0..voice_count]) |*a| {
            if (!a.v.active) continue;
            const target = if (a.v.bus) |b_id| blk: {
                const idx = @intFromEnum(b_id);
                if (idx < cap and self.bus_active[idx].load(.acquire)) {
                    bus_has_audio[idx] = true;
                    break :blk self.bus_chunks[idx][0..cur_samples];
                }
                break :blk master_chunk[0..cur_samples];
            } else master_chunk[0..cur_samples];

            switch (a.kind) {
                .noise_burst => renderNoise(a.*, dt, target),
                .thump => renderTone(true, a.*, dt, target),
                .blip => renderTone(false, a.*, dt, target),
                .sample => renderSample(a.*, dt, target),
            }
        }

        // Render active audio streams
        for (&self.streams) |*maybe_s| {
            const s = maybe_s.* orelse continue;
            if (s.state.load(.acquire) != .playing) continue;
            const bus_opt = s.getBus();
            const target = if (bus_opt) |b_id| blk: {
                const idx = @intFromEnum(b_id);
                if (idx < cap and self.bus_active[idx].load(.acquire)) {
                    bus_has_audio[idx] = true;
                    break :blk self.bus_chunks[idx][0..cur_samples];
                }
                break :blk master_chunk[0..cur_samples];
            } else master_chunk[0..cur_samples];

            _ = s.renderToBuffer(target);
        }

        // Process buses in leaf-to-root topological order
        for (active_buses[0..active_count]) |b| {
            const r_slot = self.bus_reverb_slot[b].load(.acquire);
            const has_rev = (r_slot < types.max_reverbs and self.reverbs[r_slot].active);
            const occ: f32 = @bitCast(self.bus_occlusion[b].load(.acquire));

            if (!bus_has_audio[b] and !has_rev) continue;

            if (self.bus_filters[b].filter_type != .none) {
                self.bus_filters[b].processBuffer(self.bus_chunks[b][0..cur_samples]);
            }

            if (occ > 0.001) {
                const o_cfg = self.bus_occlusion_config[b];
                const occ_cutoff = std.math.lerp(o_cfg.max_cutoff, o_cfg.min_cutoff, occ);
                self.bus_occlusion_filters[b].setParams(.lowpass, occ_cutoff, 0.7071, self.sample_rate);
                self.bus_occlusion_filters[b].processBuffer(self.bus_chunks[b][0..cur_samples]);
            }

            if (has_rev) {
                self.reverbs[r_slot].processBuffer(self.bus_chunks[b][0..cur_samples]);
            }

            if (self.bus_muted[b].load(.acquire)) {
                @memset(self.bus_chunks[b][0..cur_samples], 0.0);
            } else {
                var vol: f32 = @bitCast(self.bus_volumes[b].load(.acquire));
                if (occ > 0.001) {
                    vol *= std.math.lerp(1.0, self.bus_occlusion_config[b].min_volume, occ);
                }
                if (vol != 1.0) {
                    for (self.bus_chunks[b][0..cur_samples]) |*s| {
                        s.* *= vol;
                    }
                }
            }

            const p = parents[b];
            if (p < cap and p != b and self.bus_active[p].load(.acquire)) {
                for (0..cur_samples) |k| {
                    self.bus_chunks[p][k] += self.bus_chunks[b][k];
                }
                bus_has_audio[p] = true;
            } else {
                for (0..cur_samples) |k| {
                    master_chunk[k] += self.bus_chunks[b][k];
                }
            }
        }

        if (master == 0.0) {
            @memset(buffer[offset .. offset + cur_samples], 0.0);
        } else {
            var k: usize = 0;
            while (k < cur_samples) : (k += 2) {
                const l = master_chunk[k] * master;
                const r = master_chunk[k + 1] * master;
                buffer[offset + k] = l / (1.0 + @abs(l));
                buffer[offset + k + 1] = r / (1.0 + @abs(r));
            }
        }

        offset += cur_samples;
    }
}

/// Noise burst core: per-sample LCG noise + one-pole lowpass. The
/// cutoff ramp slope and filter constant are hoisted; `alpha` stays
/// per-sample since the filter state integrates every sample.
inline fn renderNoise(a: ActiveVoice, dt: f32, buffer: []f32) void {
    const v = a.v;
    const omega: f32 = 2.0 * std.math.pi * dt; // hoisted filter constant
    var t = v.t;
    var seed = v.seed;
    var lp = v.lp;
    const total: usize = buffer.len / 2;
    var j: usize = 0;
    while (j < total) : (j += 1) {
        t += dt;
        if (t >= a.duration) break;
        const k = t * a.inv_duration;
        const env = (1.0 - k) * (1.0 - k);
        const cutoff = @max(a.cutoff0 + a.cutoff_slope * t, 30.0);
        const alpha = 1.0 - @exp(-omega * cutoff);
        seed = seed *% 1664525 +% 1013904223;
        const u: f32 = @floatFromInt(seed >> 9); // 0..2^23-1
        const n = u * (2.0 / 8388608.0) - 1.0;
        lp += alpha * (n - lp);
        const s = lp * 1.6 * env;
        buffer[j * 2] += s * a.gl;
        buffer[j * 2 + 1] += s * a.gr;
    }
    v.t = t;
    v.seed = seed;
    v.lp = lp;
    if (t >= a.duration) v.active = false;
}

/// Oscillator core: phase and envelope stay per-sample (they *are* the
/// signal); only the freq ramp slope is hoisted to one multiply-add.
/// `thump_click` is comptime so `.blip` pays no branch for the click.
inline fn renderTone(comptime thump_click: bool, a: ActiveVoice, dt: f32, buffer: []f32) void {
    const v = a.v;
    const two_pi: f32 = 2.0 * std.math.pi;
    var t = v.t;
    var phase = v.phase;
    const total: usize = buffer.len / 2;
    var j: usize = 0;
    while (j < total) : (j += 1) {
        t += dt;
        if (t >= a.duration) break;
        const k = t * a.inv_duration;
        const env = (1.0 - k) * (1.0 - k);
        const freq = @max(a.freq0 + a.freq_slope * t, 20.0);
        phase += freq * dt;
        const s = @sin(phase * two_pi);
        const click: f32 = if (thump_click and t < 0.008) 0.5 * (1.0 - t / 0.008) else 0.0;
        const o = (s * 0.8 + click) * env;
        buffer[j * 2] += o * a.gl;
        buffer[j * 2 + 1] += o * a.gr;
    }
    v.t = t;
    v.phase = phase;
    if (t >= a.duration) v.active = false;
}

/// Sample playback core. Clip validity, null/empty checks and the
/// initial position wrap happen once in `renderFrames`; here only the
/// position advance can end/wrap the voice, with unchanged semantics:
/// the last frame stays audible, then a non-looping voice deactivates.
inline fn renderSample(a: ActiveVoice, dt: f32, buffer: []f32) void {
    const v = a.v;
    const s = a.clip.?; // validated during setup
    const n: f64 = @floatFromInt(a.frames);
    var pos = v.sample_pos;
    var t = v.t;
    var lp_l = v.lp;
    var lp_r = v.lp_r;
    const filter_active = a.cutoff0 < 19000.0;
    const omega: f32 = 2.0 * std.math.pi * dt;
    const alpha: f32 = if (filter_active) 1.0 - @exp(-omega * @max(a.cutoff0, 40.0)) else 1.0;
    const total: usize = buffer.len / 2;
    var j: usize = 0;
    if (a.step_one) {
        // Fast path: 1:1 step from an integer position, direct copy.
        while (j < total) : (j += 1) {
            t += dt;
            if (a.loop and a.duration > 0.0 and t >= a.duration) t = @mod(t, a.duration);
            if (pos >= n) {
                if (a.loop) {
                    pos = @mod(pos, n);
                } else {
                    v.sample_pos = pos;
                    v.t = t;
                    v.lp = lp_l;
                    v.lp_r = lp_r;
                    v.active = false;
                    return;
                }
            }
            const idx: usize = @intFromFloat(pos);
            var smp_l = s[idx * 2];
            var smp_r = s[idx * 2 + 1];
            if (filter_active) {
                lp_l += alpha * (smp_l - lp_l);
                lp_r += alpha * (smp_r - lp_r);
                if (@abs(lp_l) < 1e-15) lp_l = 0.0;
                if (@abs(lp_r) < 1e-15) lp_r = 0.0;
                smp_l = lp_l;
                smp_r = lp_r;
            }
            buffer[j * 2] += smp_l * a.gl;
            buffer[j * 2 + 1] += smp_r * a.gr;
            pos += 1.0;
        }
    } else {
        // Resampling path: linear interpolation between frames.
        while (j < total) : (j += 1) {
            t += dt;
            if (a.loop and a.duration > 0.0 and t >= a.duration) t = @mod(t, a.duration);
            if (pos >= n) {
                if (a.loop) {
                    pos = @mod(pos, n);
                } else {
                    v.sample_pos = pos;
                    v.t = t;
                    v.lp = lp_l;
                    v.lp_r = lp_r;
                    v.active = false;
                    return;
                }
            }
            const fl = @floor(pos);
            const idx0: usize = @intFromFloat(fl);
            const frac: f32 = @floatCast(pos - fl);
            const x = @min(idx0, a.frames - 1);
            const y = @min(x + 1, a.frames - 1);
            var smp_l = s[x * 2] + (s[y * 2] - s[x * 2]) * frac;
            var smp_r = s[x * 2 + 1] + (s[y * 2 + 1] - s[x * 2 + 1]) * frac;
            if (filter_active) {
                lp_l += alpha * (smp_l - lp_l);
                lp_r += alpha * (smp_r - lp_r);
                if (@abs(lp_l) < 1e-15) lp_l = 0.0;
                if (@abs(lp_r) < 1e-15) lp_r = 0.0;
                smp_l = lp_l;
                smp_r = lp_r;
            }
            buffer[j * 2] += smp_l * a.gl;
            buffer[j * 2 + 1] += smp_r * a.gr;
            pos += a.step;
        }
    }
    v.sample_pos = pos;
    v.t = t;
    v.lp = lp_l;
    v.lp_r = lp_r;
}
