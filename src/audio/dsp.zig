const std = @import("std");

/// Biquad IIR filter types according to Robert Bristow-Johnson Audio EQ Cookbook.
pub const BiquadFilterType = enum(u8) {
    none = 0,
    lowpass = 1,
    highpass = 2,
    bandpass = 3,
    notch = 4,
};

/// High-level filter configuration for an audio bus.
pub const BusFilterConfig = struct {
    filter_type: BiquadFilterType = .none,
    cutoff: f32 = 1000.0, // Cutoff frequency in Hz (10.0 .. 20000.0)
    q: f32 = 0.7071, // Resonance / Q factor (0.1 .. 20.0, 0.7071 = Butterworth)
};

/// Freeverb-style algorithmic reverb configuration for an audio bus.
pub const BusReverbConfig = struct {
    room_size: f32 = 0.5, // 0.0 .. 1.0 (controls feedback / decay time)
    damping: f32 = 0.5, // 0.0 .. 1.0 (high-frequency absorption)
    width: f32 = 1.0, // 0.0 .. 1.0 (stereo spread)
    wet: f32 = 0.33, // 0.0 .. 2.0 (reverberated signal level)
    dry: f32 = 0.67, // 0.0 .. 2.0 (direct signal level)
    freeze: bool = false, // Infinite decay tail mode
};

/// 2nd-order Direct Form II Transposed biquad filter.
/// Provides superior numerical stability for 32-bit floats and minimal state storage.
pub const BiquadFilter = struct {
    // Normalized coefficients (a0 = 1.0)
    b0: f32 = 1.0,
    b1: f32 = 0.0,
    b2: f32 = 0.0,
    a1: f32 = 0.0,
    a2: f32 = 0.0,

    // TDF-II delay registers for Left and Right channels
    s1_l: f32 = 0.0,
    s2_l: f32 = 0.0,
    s1_r: f32 = 0.0,
    s2_r: f32 = 0.0,

    filter_type: BiquadFilterType = .none,
    cutoff: f32 = 1000.0,
    q: f32 = 0.7071,
    last_sample_rate: f32 = 44100.0,

    pub fn setParams(self: *BiquadFilter, filter_type: BiquadFilterType, cutoff_hz: f32, q_val: f32, sample_rate: f32) void {
        self.filter_type = filter_type;
        self.cutoff = cutoff_hz;
        self.q = q_val;
        self.last_sample_rate = sample_rate;

        if (filter_type == .none or sample_rate <= 0.0) {
            self.b0 = 1.0;
            self.b1 = 0.0;
            self.b2 = 0.0;
            self.a1 = 0.0;
            self.a2 = 0.0;
            return;
        }

        const nyquist = sample_rate * 0.499;
        const fc = std.math.clamp(cutoff_hz, 10.0, nyquist);
        const q = std.math.clamp(q_val, 0.1, 20.0);
        const w0 = 2.0 * std.math.pi * (fc / sample_rate);
        const cos_w0 = @cos(w0);
        const sin_w0 = @sin(w0);
        const alpha = sin_w0 / (2.0 * q);

        var b0_raw: f32 = 1.0;
        var b1_raw: f32 = 0.0;
        var b2_raw: f32 = 0.0;
        var a0_raw: f32 = 1.0;
        var a1_raw: f32 = 0.0;
        var a2_raw: f32 = 0.0;

        switch (filter_type) {
            .none => return,
            .lowpass => {
                b0_raw = (1.0 - cos_w0) * 0.5;
                b1_raw = 1.0 - cos_w0;
                b2_raw = (1.0 - cos_w0) * 0.5;
                a0_raw = 1.0 + alpha;
                a1_raw = -2.0 * cos_w0;
                a2_raw = 1.0 - alpha;
            },
            .highpass => {
                b0_raw = (1.0 + cos_w0) * 0.5;
                b1_raw = -(1.0 + cos_w0);
                b2_raw = (1.0 + cos_w0) * 0.5;
                a0_raw = 1.0 + alpha;
                a1_raw = -2.0 * cos_w0;
                a2_raw = 1.0 - alpha;
            },
            .bandpass => {
                b0_raw = alpha;
                b1_raw = 0.0;
                b2_raw = -alpha;
                a0_raw = 1.0 + alpha;
                a1_raw = -2.0 * cos_w0;
                a2_raw = 1.0 - alpha;
            },
            .notch => {
                b0_raw = 1.0;
                b1_raw = -2.0 * cos_w0;
                b2_raw = 1.0;
                a0_raw = 1.0 + alpha;
                a1_raw = -2.0 * cos_w0;
                a2_raw = 1.0 - alpha;
            },
        }

        const inv_a0 = 1.0 / a0_raw;
        self.b0 = b0_raw * inv_a0;
        self.b1 = b1_raw * inv_a0;
        self.b2 = b2_raw * inv_a0;
        self.a1 = a1_raw * inv_a0;
        self.a2 = a2_raw * inv_a0;
    }

    pub inline fn processSample(self: *BiquadFilter, in_l: f32, in_r: f32) struct { l: f32, r: f32 } {
        if (self.filter_type == .none) return .{ .l = in_l, .r = in_r };

        // Left channel TDF-II
        const out_l = self.b0 * in_l + self.s1_l;
        self.s1_l = self.b1 * in_l - self.a1 * out_l + self.s2_l;
        self.s2_l = self.b2 * in_l - self.a2 * out_l;

        // Right channel TDF-II
        const out_r = self.b0 * in_r + self.s1_r;
        self.s1_r = self.b1 * in_r - self.a1 * out_r + self.s2_r;
        self.s2_r = self.b2 * in_r - self.a2 * out_r;

        // Anti-denormal protection (flush tiny residual values to 0.0)
        if (@abs(self.s1_l) < 1e-15) self.s1_l = 0.0;
        if (@abs(self.s2_l) < 1e-15) self.s2_l = 0.0;
        if (@abs(self.s1_r) < 1e-15) self.s1_r = 0.0;
        if (@abs(self.s2_r) < 1e-15) self.s2_r = 0.0;

        return .{ .l = out_l, .r = out_r };
    }

    pub fn processBuffer(self: *BiquadFilter, buffer: []f32) void {
        if (self.filter_type == .none) return;
        var i: usize = 0;
        while (i < buffer.len) : (i += 2) {
            const res = self.processSample(buffer[i], buffer[i + 1]);
            buffer[i] = res.l;
            buffer[i + 1] = res.r;
        }
    }

    pub fn resetState(self: *BiquadFilter) void {
        self.s1_l = 0.0;
        self.s2_l = 0.0;
        self.s1_r = 0.0;
        self.s2_r = 0.0;
    }
};

/// Lowpass feedback comb filter with power-of-two circular buffer.
pub const CombFilter = struct {
    pub const max_size: usize = 2048;
    pub const mask: usize = max_size - 1;

    buffer: [max_size]f32 = [_]f32{0.0} ** max_size,
    buf_idx: usize = 0,
    filter_store: f32 = 0.0,
    delay: usize = 1116,

    pub inline fn process(self: *CombFilter, input: f32, feedback: f32, damp1: f32, damp2: f32) f32 {
        const read_idx = (self.buf_idx + max_size - self.delay) & mask;
        const output = self.buffer[read_idx];
        self.filter_store = (output * damp2) + (self.filter_store * damp1);
        self.buffer[self.buf_idx] = input + (self.filter_store * feedback);
        self.buf_idx = (self.buf_idx + 1) & mask;
        return output;
    }

    pub fn clear(self: *CombFilter) void {
        @memset(&self.buffer, 0.0);
        self.filter_store = 0.0;
        self.buf_idx = 0;
    }
};

/// Allpass filter with power-of-two circular buffer.
pub const AllpassFilter = struct {
    pub const max_size: usize = 1024;
    pub const mask: usize = max_size - 1;

    buffer: [max_size]f32 = [_]f32{0.0} ** max_size,
    buf_idx: usize = 0,
    delay: usize = 556,

    pub inline fn process(self: *AllpassFilter, input: f32) f32 {
        const read_idx = (self.buf_idx + max_size - self.delay) & mask;
        const buf_out = self.buffer[read_idx];
        const output = -input + buf_out;
        self.buffer[self.buf_idx] = input + (buf_out * 0.5);
        self.buf_idx = (self.buf_idx + 1) & mask;
        return output;
    }

    pub fn clear(self: *AllpassFilter) void {
        @memset(&self.buffer, 0.0);
        self.buf_idx = 0;
    }
};

/// Freeverb stereo algorithmic reverberator.
/// Employs 8 parallel lowpass comb filters and 4 serial allpass filters per channel.
pub const ReverbProcessor = struct {
    // Base delay tunings at 44.1 kHz
    const comb_tuning_l = [8]usize{ 1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617 };
    const comb_tuning_r = [8]usize{ 1139, 1211, 1300, 1379, 1445, 1514, 1580, 1640 };
    const allpass_tuning_l = [4]usize{ 556, 441, 341, 225 };
    const allpass_tuning_r = [4]usize{ 579, 464, 364, 248 };

    combs_l: [8]CombFilter = [_]CombFilter{CombFilter{}} ** 8,
    combs_r: [8]CombFilter = [_]CombFilter{CombFilter{}} ** 8,
    allpasses_l: [4]AllpassFilter = [_]AllpassFilter{AllpassFilter{}} ** 4,
    allpasses_r: [4]AllpassFilter = [_]AllpassFilter{AllpassFilter{}} ** 4,

    config: BusReverbConfig = .{},
    active: bool = false,
    bus_id: u8 = 0xFF,
    sample_rate: f32 = 44100.0,

    pub fn init(sample_rate: f32) ReverbProcessor {
        var rev = ReverbProcessor{};
        rev.setSampleRate(sample_rate);
        return rev;
    }

    pub fn setSampleRate(self: *ReverbProcessor, rate: f32) void {
        const r = @max(rate, 8000.0);
        self.sample_rate = r;
        const scale = r / 44100.0;

        for (&self.combs_l, 0..) |*c, i| {
            const raw = @as(f32, @floatFromInt(comb_tuning_l[i])) * scale;
            c.delay = std.math.clamp(@as(usize, @intFromFloat(raw)), 1, CombFilter.max_size - 2);
        }
        for (&self.combs_r, 0..) |*c, i| {
            const raw = @as(f32, @floatFromInt(comb_tuning_r[i])) * scale;
            c.delay = std.math.clamp(@as(usize, @intFromFloat(raw)), 1, CombFilter.max_size - 2);
        }
        for (&self.allpasses_l, 0..) |*a, i| {
            const raw = @as(f32, @floatFromInt(allpass_tuning_l[i])) * scale;
            a.delay = std.math.clamp(@as(usize, @intFromFloat(raw)), 1, AllpassFilter.max_size - 2);
        }
        for (&self.allpasses_r, 0..) |*a, i| {
            const raw = @as(f32, @floatFromInt(allpass_tuning_r[i])) * scale;
            a.delay = std.math.clamp(@as(usize, @intFromFloat(raw)), 1, AllpassFilter.max_size - 2);
        }
    }

    pub fn setConfig(self: *ReverbProcessor, config: BusReverbConfig) void {
        self.config = config;
    }

    pub fn processBuffer(self: *ReverbProcessor, buffer: []f32) void {
        if (!self.active) return;
        const cfg = self.config;
        const feedback: f32 = if (cfg.freeze) 1.0 else std.math.clamp(cfg.room_size * 0.28 + 0.7, 0.0, 0.98);
        const damp: f32 = std.math.clamp(cfg.damping, 0.0, 1.0);
        const damp1: f32 = damp * 0.4;
        const damp2: f32 = 1.0 - damp1;
        const gain: f32 = 0.015;
        const dry: f32 = std.math.clamp(cfg.dry, 0.0, 2.0);
        const wet: f32 = std.math.clamp(cfg.wet, 0.0, 2.0);
        const width: f32 = std.math.clamp(cfg.width, 0.0, 1.0);
        const wet1: f32 = wet * (width * 0.5 + 0.5);
        const wet2: f32 = wet * ((1.0 - width) * 0.5);

        var i: usize = 0;
        while (i < buffer.len) : (i += 2) {
            const in_l = buffer[i];
            const in_r = buffer[i + 1];

            // 8 parallel combs for Left
            var out_l: f32 = 0.0;
            for (&self.combs_l) |*c| {
                out_l += c.process(in_l * gain, feedback, damp1, damp2);
            }

            // 8 parallel combs for Right
            var out_r: f32 = 0.0;
            for (&self.combs_r) |*c| {
                out_r += c.process(in_r * gain, feedback, damp1, damp2);
            }

            // 4 series allpasses for Left
            for (&self.allpasses_l) |*a| {
                out_l = a.process(out_l);
            }

            // 4 series allpasses for Right
            for (&self.allpasses_r) |*a| {
                out_r = a.process(out_r);
            }

            // Anti-denormal protection
            if (@abs(out_l) < 1e-15) out_l = 0.0;
            if (@abs(out_r) < 1e-15) out_r = 0.0;

            // Stereo wet/dry mix with spatial width crossfeed
            buffer[i] = dry * in_l + wet1 * out_l + wet2 * out_r;
            buffer[i + 1] = dry * in_r + wet1 * out_r + wet2 * out_l;
        }
    }

    pub fn clear(self: *ReverbProcessor) void {
        for (&self.combs_l) |*c| c.clear();
        for (&self.combs_r) |*c| c.clear();
        for (&self.allpasses_l) |*a| a.clear();
        for (&self.allpasses_r) |*a| a.clear();
    }
};
