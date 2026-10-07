const std = @import("std");

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const is_web = target.result.cpu.arch.isWasm();

    const dep_agate = b.dependency("agate", .{
        .target = target,
        .optimize = optimize,
    });
    const mod_agate = dep_agate.module("agate");

    const dep_sokol = dep_agate.builder.dependency("sokol", .{
        .target = target,
        .optimize = optimize,
        .wgpu = is_web,
    });
    const mod_sokol = dep_sokol.module("sokol");

    // 1. Render Target Basic (RTT)
    const rtt_mod = b.createModule(.{
        .root_source_file = b.path("render_target_basic.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "sokol", .module = mod_sokol },
            .{ .name = "agate", .module = mod_agate },
        },
    });
    const rtt_exe = b.addExecutable(.{
        .name = "rtt-smoke",
        .root_module = rtt_mod,
    });
    b.installArtifact(rtt_exe);
    const run_rtt = b.addRunArtifact(rtt_exe);
    if (b.args) |args| run_rtt.addArgs(args);
    b.step("run-rtt", "Run render-target and refraction smoke (needs GPU/display)").dependOn(&run_rtt.step);

    // 2. GPU Timing
    const timing_mod = b.createModule(.{
        .root_source_file = b.path(if (is_web) "gpu_timing_web.zig" else "gpu_timing.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "sokol", .module = mod_sokol },
            .{ .name = "agate", .module = mod_agate },
        },
    });
    const timing_exe = b.addExecutable(.{
        .name = "gpu-timing",
        .root_module = timing_mod,
    });
    b.installArtifact(timing_exe);
    const run_timing = b.addRunArtifact(timing_exe);
    if (b.args) |args| run_timing.addArgs(args);
    b.step("run-gpu-timing", "Run GPU timing/lifecycle smoke (needs GPU/display)").dependOn(&run_timing.step);

    // 3. HDR Showcase
    const hdr_mod = b.createModule(.{
        .root_source_file = b.path(if (is_web) "hdr_showcase_web.zig" else "hdr_showcase.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "sokol", .module = mod_sokol },
            .{ .name = "agate", .module = mod_agate },
        },
    });
    const hdr_exe = b.addExecutable(.{
        .name = "hdr-showcase",
        .root_module = hdr_mod,
    });
    b.installArtifact(hdr_exe);
    const run_hdr = b.addRunArtifact(hdr_exe);
    if (b.args) |args| run_hdr.addArgs(args);
    b.step("run-hdr", "Run the HDR studio showcase").dependOn(&run_hdr.step);
}
