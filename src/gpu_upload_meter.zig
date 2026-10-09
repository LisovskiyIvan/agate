//! Frame-level byte meter for dynamic GPU buffer updates.
//!
//! Tracks bytes sent via sg.updateBuffer / sg.appendBuffer during a frame.
//! Uses an atomic counter so worker threads can safely record transfers
//! during parallel instance staging. Reset once per frame on the context thread.

const std = @import("std");
const builtin = @import("builtin");

const MeterInt = if (builtin.cpu.arch.isWasm()) usize else u64;
var pending_bytes: std.atomic.Value(MeterInt) = std.atomic.Value(MeterInt).init(0);

/// Records `bytes` written to a GPU buffer.
pub fn record(bytes: usize) void {
    _ = pending_bytes.fetchAdd(@as(MeterInt, @intCast(bytes)), .monotonic);
}

/// Takes the accumulated byte count and resets the counter to zero.
pub fn takeAndReset() u64 {
    return @as(u64, pending_bytes.swap(0, .acq_rel));
}

/// Reads the current byte count without resetting.
pub fn peek() u64 {
    return @as(u64, pending_bytes.load(.monotonic));
}
