//! No-op Tracy fallback wired as the "ztracy" import when the engine is
//! built without `-Dtracy=true` (the default). Mirrors exactly the ztracy
//! surface the engine calls, so instrumented code compiles and runs
//! unchanged with zero overhead (every call inlines to nothing). The real
//! client (ztracy repo + vendored Tracy C++ library) is only fetched and
//! linked for `-Dtracy=true` native builds; see docs/profiling.md.

const std = @import("std");
const Src = std.builtin.SourceLocation;

pub const ZoneCtx = struct {
    pub inline fn End(self: ZoneCtx) void {
        _ = self;
    }
};

pub inline fn FrameMark() void {}

pub inline fn FrameMarkNamed(name: [*:0]const u8) void {
    _ = name;
}

pub inline fn ZoneN(comptime src: Src, name: [*:0]const u8) ZoneCtx {
    _ = src;
    _ = name;
    return .{};
}

pub inline fn ZoneNC(comptime src: Src, name: [*:0]const u8, color: u32) ZoneCtx {
    _ = src;
    _ = name;
    _ = color;
    return .{};
}
