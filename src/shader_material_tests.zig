const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;

const sm = @import("shader_material.zig");
const keyForName = sm.keyForName;
const indexForName = sm.indexForName;
const entry = sm.entry;
const entryForKey = sm.entryForKey;
const entryCount = sm.entryCount;
const registerRuntime = sm.registerRuntime;
const defaultUniformStorage = sm.defaultUniformStorage;
const setUniform = sm.setUniform;
const uniformBytes = sm.uniformBytes;
const invalid_index = sm.invalid_index;
const Param = sm.Param;

test "keyForName is deterministic and collision-free for distinct names" {
    // Golden literal: Wyhash(0, "x") — guards against the hash algorithm or
    // seed ever drifting (keys persist only in-process, but the cache
    // contract relies on run-to-run determinism).
    try std.testing.expectEqual(@as(u64, 4738888789374899184), keyForName("x"));
    try std.testing.expectEqual(keyForName("toon"), keyForName("toon"));
    try std.testing.expect(keyForName("toon") != keyForName("toon2"));
}

test "static registry exposes build.zig materials by name and key" {
    // The example material registered in build.zig's user_shader_materials.
    const idx = indexForName("ramp_wave") orelse return error.TestUnexpectedResult;
    const e = entry(idx).?;
    try std.testing.expectEqualStrings("ramp_wave", e.name);
    try std.testing.expectEqual(keyForName("ramp_wave"), e.key);
    try std.testing.expect(e.engine_template);
    try std.testing.expect(e.user_ub != null);
    try std.testing.expect(e.params.len > 0);
    // Key path agrees with the name path (same registration).
    try std.testing.expectEqual(e, entryForKey(e.key).?);
}

test "runtime registration appends entries with deterministic keys" {
    const dummy = struct {
        fn makeShader(_: sg.Backend) sg.Shader {
            return .{};
        }
    };
    const count_before = entryCount();
    const idx = try registerRuntime(.{ .name = "test_runtime_mat", .make_shader = dummy.makeShader, .user_ub = 3 });
    defer sm.runtime_len -= 1; // rollback: keep the registry deterministic per test run

    try std.testing.expectEqual(count_before, idx);
    const e = entry(idx).?;
    try std.testing.expectEqualStrings("test_runtime_mat", e.name);
    try std.testing.expectEqual(keyForName("test_runtime_mat"), e.key);
    try std.testing.expect(!e.engine_template);
    try std.testing.expectEqual(@as(u32, 3), e.user_ub.?);
    try std.testing.expectEqual(e, entry(idx).?);
    // Name path finds runtime entries too.
    try std.testing.expectEqual(idx, indexForName("test_runtime_mat").?);

    // Duplicate names are rejected.
    try std.testing.expectError(error.DuplicateName, registerRuntime(.{ .name = "test_runtime_mat", .make_shader = dummy.makeShader }));
}

test "external uniform window: boundary offsets register, overflow is a hard error" {
    const dummy = struct {
        fn makeShader(_: sg.Backend) sg.Shader {
            return .{};
        }
    };
    // Boundary: two floats in slot 0 plus a vec4 filling slot 1 — exactly
    // the 2-vec4 v1 window (words 0..7).
    const fitting = [_]Param{
        .{ .name = "u_time", .offset = 0, .comps = 1 },
        .{ .name = "u_intensity", .offset = 1, .comps = 1 },
        .{ .name = "u_tint", .offset = 4, .comps = 4 },
    };
    const idx = try registerRuntime(.{ .name = "test_ext_fit", .make_shader = dummy.makeShader, .user_ub = 1, .params = &fitting });
    defer sm.runtime_len -= 1;
    try std.testing.expectEqual(@as(usize, 3), entry(idx).?.params.len);

    // Overflow: a float in word 8 (first word past the window).
    const past_end = [_]Param{.{ .name = "u_nope", .offset = 8, .comps = 1 }};
    try std.testing.expectError(error.UniformLimitExceeded, registerRuntime(.{ .name = "test_ext_past", .make_shader = dummy.makeShader, .params = &past_end }));
    // Straddling vec4 (words 5..8) and unknown component counts are
    // rejected too — never silently clamped or truncated.
    const straddle = [_]Param{.{ .name = "u_straddle", .offset = 5, .comps = 4 }};
    try std.testing.expectError(error.UniformLimitExceeded, registerRuntime(.{ .name = "test_ext_straddle", .make_shader = dummy.makeShader, .params = &straddle }));
    const bad_comps = [_]Param{.{ .name = "u_vec2", .offset = 0, .comps = 2 }};
    try std.testing.expectError(error.UniformLimitExceeded, registerRuntime(.{ .name = "test_ext_comps", .make_shader = dummy.makeShader, .params = &bad_comps }));
    // Rejected registrations leave no entry behind.
    try std.testing.expect(indexForName("test_ext_past") == null);
    try std.testing.expect(indexForName("test_ext_straddle") == null);
    try std.testing.expect(indexForName("test_ext_comps") == null);

    // Wire size: zero, non-vec4-multiple, and over-budget blocks are
    // rejected (the draw uploads exactly user_bytes — a mismatch would
    // trip sokol validation at draw time instead).
    try std.testing.expectError(error.UniformLimitExceeded, registerRuntime(.{ .name = "test_ext_zero", .make_shader = dummy.makeShader, .user_bytes = 0 }));
    try std.testing.expectError(error.UniformLimitExceeded, registerRuntime(.{ .name = "test_ext_20", .make_shader = dummy.makeShader, .user_bytes = 20 }));
    try std.testing.expectError(error.UniformLimitExceeded, registerRuntime(.{ .name = "test_ext_48", .make_shader = dummy.makeShader, .user_bytes = 48 }));
    // A 1-vec4 window registers, but the slot-1 vec4 no longer fits it.
    const narrow = [_]Param{.{ .name = "u_a", .offset = 0, .comps = 1 }};
    const narrow_idx = try registerRuntime(.{ .name = "test_ext_narrow", .make_shader = dummy.makeShader, .user_bytes = 16, .params = &narrow });
    defer sm.runtime_len -= 1;
    try std.testing.expectEqual(@as(u16, 16), entry(narrow_idx).?.user_bytes);
    try std.testing.expectError(error.UniformLimitExceeded, registerRuntime(.{ .name = "test_ext_narrow2", .make_shader = dummy.makeShader, .user_bytes = 16, .params = &fitting }));
    try std.testing.expectEqual(@as(u16, 32), entry(idx).?.user_bytes);
}

test "defaultUniformStorage applies declared defaults" {
    const params = [_]Param{
        .{ .name = "u_a", .offset = 0, .comps = 1, .default = .{ 2.5, 0, 0, 0 } },
        .{ .name = "u_b", .offset = 4, .comps = 4, .default = .{ 1, 2, 3, 4 } },
    };
    const storage = defaultUniformStorage(&params);
    try std.testing.expectEqual(@as(f32, 2.5), storage[0][0]);
    try std.testing.expectEqual(@as(f32, 0), storage[0][1]);
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4 }, &storage[1]);
    // Everything else stays zero.
    try std.testing.expectEqual(@as(f32, 0), storage[7][3]);
}

test "setUniform packs by declarative offsets (float lanes and vec4 slots)" {
    const params = [_]Param{
        .{ .name = "u_speed", .offset = 0, .comps = 1 },
        .{ .name = "u_tint", .offset = 4, .comps = 4 },
        .{ .name = "u_amp", .offset = 8, .comps = 1 },
    };
    var storage = defaultUniformStorage(&params);

    try setUniform(&storage, &params, "u_speed", .{ .scalar = 3.0 });
    try setUniform(&storage, &params, "u_tint", .{ .vector = .{ 0.1, 0.2, 0.3, 0.4 } });
    try setUniform(&storage, &params, "u_amp", .{ .scalar = 0.75 });

    // u_speed sits in slot 0 lane x; u_amp in slot 2 lane x.
    try std.testing.expectEqual(@as(f32, 3.0), storage[0][0]);
    try std.testing.expectEqualSlices(f32, &.{ 0.1, 0.2, 0.3, 0.4 }, &storage[1]);
    try std.testing.expectEqual(@as(f32, 0.75), storage[2][0]);

    // Unknown names and component mismatches are hard errors (typo guard).
    try std.testing.expectError(error.UnknownParam, setUniform(&storage, &params, "u_nope", .{ .scalar = 1 }));
    try std.testing.expectError(error.CompMismatch, setUniform(&storage, &params, "u_speed", .{ .vector = .{ 1, 2, 3, 4 } }));
    try std.testing.expectError(error.CompMismatch, setUniform(&storage, &params, "u_tint", .{ .scalar = 1 }));

    // The packed storage serializes into the exact 128-byte UB payload.
    try std.testing.expectEqual(@as(usize, 128), uniformBytes(&storage).len);
}

test "invalid indices resolve to no entry (defensive draw path)" {
    try std.testing.expect(entry(invalid_index) == null);
    try std.testing.expect(entry(9999) == null);
    try std.testing.expect(indexForName("does_not_exist") == null);
    try std.testing.expect(entryForKey(0xDEADBEEF) == null);
}
