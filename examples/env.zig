//! Shared getenv helpers for the native examples (q0-interior,
//! hdr-showcase, gpu-timing). Same import mechanism as
//! gpu_phase_bench.zig: `@import("env.zig")`, no build.zig change needed.

const std = @import("std");
const z = @import("agate");

/// True when the env var is set to "1" or "true".
pub fn envFlag(name: [*:0]const u8) bool {
    const value = std.c.getenv(name) orelse return false;
    const s = std.mem.span(value);
    return std.mem.eql(u8, s, "1") or std.mem.eql(u8, s, "true");
}

/// True when the env var is set to "1", "true", "on" or "yes".
/// Delegates to the engine rule so gpu-timing keeps its wider spelling set.
pub fn envFlagAny(name: [*:0]const u8) bool {
    const raw = std.c.getenv(name);
    return z.gpu_timing.parseEnvFlag(if (raw) |v| std.mem.span(v) else null);
}

/// f32 env var with fallback on unset/unparseable.
pub fn envF32(name: [*:0]const u8, fallback: f32) f32 {
    const value = std.c.getenv(name) orelse return fallback;
    return std.fmt.parseFloat(f32, std.mem.span(value)) catch fallback;
}
