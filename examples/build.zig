const std = @import("std");
const agate_build = @import("agate");

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

    if (is_web) {
        const emsdk = agate_build.getEmsdk(dep_agate);
        const sdk_install = agate_build.sokol.emSdkInstallStep(b, emsdk, .{});
        b.step("install-emsdk", "Install the workspace Emscripten SDK").dependOn(sdk_install);
        for ([_]struct { name: []const u8, source: []const u8 }{
            .{ .name = "gpu-timing", .source = "gpu_timing_web.zig" },
            .{ .name = "hdr-showcase", .source = "hdr_showcase_web.zig" },
            .{ .name = "q0-interior", .source = "q0_interior_web.zig" },
            .{ .name = "taa-quality", .source = "taa_quality_web.zig" },
            .{ .name = "render-scale-gate", .source = "render_scale_gate_web.zig" },
        }) |example| {
            const lib = b.addLibrary(.{
                .name = example.name,
                .linkage = .static,
                .root_module = b.createModule(.{
                    .root_source_file = b.path(example.source),
                    .target = target,
                    .optimize = optimize,
                    .imports = &.{
                        .{ .name = "sokol", .module = mod_sokol },
                        .{ .name = "agate", .module = mod_agate },
                    },
                }),
            });
            const link = try agate_build.sokol.emLinkStep(b, .{
                .target = target,
                .optimize = optimize,
                .lib_main = lib,
                .emsdk = emsdk,
                .use_webgpu = true,
                .use_emmalloc = true,
                .shell_file_path = agate_build.sokolShellPath(dep_agate),
                .pre_js = if (std.mem.eql(u8, example.name, "q0-interior"))
                    &.{b.path("q0_web_env.js")}
                else if (std.mem.eql(u8, example.name, "taa-quality"))
                    &.{b.path("taa_quality_web_env.js")}
                else if (std.mem.eql(u8, example.name, "render-scale-gate"))
                    &.{b.path("render_scale_gate_web_env.js")}
                else
                    &.{},
                .extra_args = &.{ "-sSTACK_SIZE=1MB", "-sALLOW_MEMORY_GROWTH=1" },
            });
            b.getInstallStep().dependOn(&link.step);
            b.step(b.fmt("web-{s}", .{example.name}), "Build a browser WebGPU example").dependOn(&link.step);
        }
        return;
    }

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

    // 4. Runtime Worker (threaded staged frame lifecycle; native only —
    //    the worker degrades to an inline serial fallback on wasm)
    const rw_mod = b.createModule(.{
        .root_source_file = b.path("runtime_worker.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "sokol", .module = mod_sokol },
            .{ .name = "agate", .module = mod_agate },
        },
    });
    const rw_exe = b.addExecutable(.{
        .name = "runtime-worker",
        .root_module = rw_mod,
    });
    b.installArtifact(rw_exe);
    const run_rw = b.addRunArtifact(rw_exe);
    if (b.args) |args| run_rw.addArgs(args);
    b.step("run-runtime-worker", "Run the Runtime worker-thread smoke (needs GPU/display)").dependOn(&run_rw.step);

    // 5. Q0 calibration interior (fill-bound render-scale A/B workload)
    const q0_mod = b.createModule(.{
        .root_source_file = b.path("q0_interior.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "sokol", .module = mod_sokol },
            .{ .name = "agate", .module = mod_agate },
        },
    });
    const q0_exe = b.addExecutable(.{
        .name = "q0-interior",
        .root_module = q0_mod,
    });
    b.installArtifact(q0_exe);
    const run_q0 = b.addRunArtifact(q0_exe);
    if (b.args) |args| run_q0.addArgs(args);
    b.step("run-q0", "Run the Q0 calibration interior (fill-bound A/B; AGATE_Q0_FRAMES/AGATE_Q0_TIMINGS)").dependOn(&run_q0.step);

    // Harness unit tests: gpu_phase_bench verdict-rule tests (no window;
    // live sampling/occlusion paths are covered by the windowed runs).
    const bench_test_mod = b.createModule(.{
        .root_source_file = b.path("gpu_phase_bench.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "sokol", .module = mod_sokol },
        },
    });
    const bench_tests = b.addTest(.{ .root_module = bench_test_mod });
    const run_bench_tests = b.addRunArtifact(bench_tests);
    b.step("test", "Run example harness unit tests").dependOn(&run_bench_tests.step);

    // 6. Chaos Runner (deterministic seeded stress host; headless CPU-only
    //    staged path vs windowed full GPU path — see examples/chaos_runner.zig)
    const chaos_mod = b.createModule(.{
        .root_source_file = b.path("chaos_runner.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "sokol", .module = mod_sokol },
            .{ .name = "agate", .module = mod_agate },
        },
    });
    const chaos_exe = b.addExecutable(.{
        .name = "chaos-runner",
        .root_module = chaos_mod,
    });
    b.installArtifact(chaos_exe);
    const run_chaos = b.addRunArtifact(chaos_exe);
    if (b.args) |args| run_chaos.addArgs(args);
    b.step("run-chaos", "Run the seeded chaos stress host (see docs/chaos.md)").dependOn(&run_chaos.step);

    // 7. Scene scaling bench (headless CPU queue-build stress; no GPU/display —
    //    see examples/scene_scaling_bench.zig)
    const scaling_mod = b.createModule(.{
        .root_source_file = b.path("scene_scaling_bench.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "sokol", .module = mod_sokol },
            .{ .name = "agate", .module = mod_agate },
        },
    });
    const scaling_exe = b.addExecutable(.{
        .name = "scene-scaling-bench",
        .root_module = scaling_mod,
    });
    b.installArtifact(scaling_exe);
    const run_scaling = b.addRunArtifact(scaling_exe);
    if (b.args) |args| run_scaling.addArgs(args);
    b.step("run-scene-scaling", "Run the headless scene/render-queue scaling bench (ReleaseFast, smp)").dependOn(&run_scaling.step);

    // 8. Deterministic temporal-quality fixture (held same-frame captures).
    const taa_mod = b.createModule(.{
        .root_source_file = b.path("taa_quality.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "sokol", .module = mod_sokol },
            .{ .name = "agate", .module = mod_agate },
        },
    });
    const taa_exe = b.addExecutable(.{
        .name = "taa-quality",
        .root_module = taa_mod,
    });
    b.installArtifact(taa_exe);
    const run_taa = b.addRunArtifact(taa_exe);
    if (b.args) |args| run_taa.addArgs(args);
    b.step("run-taa-quality", "Run the deterministic temporal-quality GPU fixture").dependOn(&run_taa.step);

    // 8b. Render-scale + clustered point-light pixel fixture (same held-frame
    // capture protocol; headless Chromium gate lives in tools/).
    const rsg_mod = b.createModule(.{
        .root_source_file = b.path("render_scale_gate.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "sokol", .module = mod_sokol },
            .{ .name = "agate", .module = mod_agate },
        },
    });
    const rsg_exe = b.addExecutable(.{
        .name = "render-scale-gate",
        .root_module = rsg_mod,
    });
    b.installArtifact(rsg_exe);
    const run_rsg = b.addRunArtifact(rsg_exe);
    if (b.args) |args| run_rsg.addArgs(args);
    b.step("run-render-scale-gate", "Run the render-scale + clustered point-light GPU fixture").dependOn(&run_rsg.step);

    // 9. Pure CPU uniform preparation cost (no sg.applyUniforms/GPU upload).
    const uniform_mod = b.createModule(.{
        .root_source_file = b.path("draw_uniform_bench.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "sokol", .module = mod_sokol },
            .{ .name = "agate", .module = mod_agate },
        },
    });
    const uniform_exe = b.addExecutable(.{
        .name = "draw-uniform-bench",
        .root_module = uniform_mod,
    });
    b.installArtifact(uniform_exe);
    const run_uniform = b.addRunArtifact(uniform_exe);
    if (b.args) |args| run_uniform.addArgs(args);
    b.step("run-draw-uniform", "Measure CPU uniform builders (ReleaseFast, checksum parity)").dependOn(&run_uniform.step);

    // 10. Headless light-pack + clustered-tile CPU bench (audit A/B seam).
    const light_pack_mod = b.createModule(.{
        .root_source_file = b.path("light_pack_bench.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "sokol", .module = mod_sokol },
            .{ .name = "agate", .module = mod_agate },
        },
    });
    const light_pack_exe = b.addExecutable(.{
        .name = "light-pack-bench",
        .root_module = light_pack_mod,
    });
    b.installArtifact(light_pack_exe);
    const run_light_pack = b.addRunArtifact(light_pack_exe);
    if (b.args) |args| run_light_pack.addArgs(args);
    b.step("run-light-pack", "Measure LightRig.packFrame + clustered tile build (ReleaseFast, smp)").dependOn(&run_light_pack.step);

    // 11. Headless fork-join pool bench (worker count / job shape sweep).
    const jobs_mod = b.createModule(.{
        .root_source_file = b.path("jobs_bench.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "sokol", .module = mod_sokol },
            .{ .name = "agate", .module = mod_agate },
        },
    });
    const jobs_exe = b.addExecutable(.{
        .name = "jobs-bench",
        .root_module = jobs_mod,
    });
    b.installArtifact(jobs_exe);
    const run_jobs = b.addRunArtifact(jobs_exe);
    if (b.args) |args| run_jobs.addArgs(args);
    b.step("run-jobs-bench", "Measure Pool.forkJoin vs inline across worker counts (ReleaseFast)").dependOn(&run_jobs.step);
}
