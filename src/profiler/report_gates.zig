//! Report gates and byte formatting. Split out of `report.zig` (facade).
//!
//! Availability predicates (`hasGpuData`/`hasGpuPassData`) plus `formatBytes`.
//! Leaf: std + `types` only, so the report graph stays acyclic.

const std = @import("std");
const types = @import("types.zig");

const FrameRecord = types.FrameRecord;

/// True when at least one frame carries an AVAILABLE GPU frame sample
/// (submission id != 0 — valid quantized zeros count). All GPU-specific
/// report output (HTML/MD/trace) is gated on this so the disabled path
/// (no ids) stays byte-identical to the pre-GPU output.
pub fn hasGpuData(frames: []const FrameRecord) bool {
    for (frames) |f| {
        if (f.gpu_frame_submit != 0) return true;
    }
    return false;
}

/// True when at least one frame carries an AVAILABLE per-pass GPU sample
/// (any phase submission id != 0, zeros included). Gated separately from
/// `hasGpuData`, so frame-scope-only sessions (no per-pass samples) and
/// the disabled path both render exactly the pre-per-pass output.
pub fn hasGpuPassData(frames: []const FrameRecord) bool {
    for (frames) |f| {
        if (f.gpu_shadow_submit != 0 or f.gpu_main_submit != 0 or f.gpu_post_submit != 0) return true;
    }
    return false;
}

/// Helper to format byte values (e.g. "12.4 MB").
pub fn formatBytes(allocator: std.mem.Allocator, bytes: usize) ![]u8 {
    if (bytes < 1024) {
        return std.fmt.allocPrint(allocator, "{d} B", .{bytes});
    } else if (bytes < 1024 * 1024) {
        return std.fmt.allocPrint(allocator, "{d:.1} KB", .{@as(f32, @floatFromInt(bytes)) / 1024.0});
    } else if (bytes < 1024 * 1024 * 1024) {
        return std.fmt.allocPrint(allocator, "{d:.2} MB", .{@as(f32, @floatFromInt(bytes)) / (1024.0 * 1024.0)});
    } else {
        return std.fmt.allocPrint(allocator, "{d:.2} GB", .{@as(f32, @floatFromInt(bytes)) / (1024.0 * 1024.0 * 1024.0)});
    }
}
