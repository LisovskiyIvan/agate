const std = @import("std");
const Build = std.Build;
const sokol = @import("sokol");

pub fn build(b: *Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const dep_sokol = b.dependency("sokol", .{
        .target = target,
        .optimize = optimize,
    });
    const mod_sokol = dep_sokol.module("sokol");
    const mod_math = b.createModule(.{ .root_source_file = b.path("src/agate/math.zig") });

    // Шейдеры: единая таблица "имя модуля -> вход/выход", slang общий для всех.
    // NOTE: SHADOW_ATLAS_SIZE не передаётся через .defines: sokol-shdc этой
    // версии разворачивает любой дефайн в `#define NAME (1)` (проверено запуском
    // бинарника в /tmp: форма NAME=VALUE игнорируется с варнингом, форма
    // NAME VALUE даёт пустой макрос). Значение 2048.0 живёт как fallback в
    // самих .glsl и как ShadowPass.SHADOW_ATLAS_SIZE в passes/shadow_pass.zig.
    const ShaderSpec = struct {
        name: []const u8,
        input: []const u8,
        output: []const u8,
    };
    const shader_specs = [_]ShaderSpec{
        .{ .name = "shader", .input = "src/agate/shaders/standard.glsl", .output = "standard_shader.zig" },
        .{ .name = "pbr_shader", .input = "src/agate/shaders/pbr.glsl", .output = "pbr_shader.zig" },
        .{ .name = "skinned_pbr_shader", .input = "src/agate/shaders/skinned_pbr.glsl", .output = "skinned_pbr_shader.zig" },
        .{ .name = "instanced_shader", .input = "src/agate/shaders/instanced.glsl", .output = "instanced_shader.zig" },
        .{ .name = "instanced_pbr_shader", .input = "src/agate/shaders/instanced_pbr.glsl", .output = "instanced_pbr_shader.zig" },
        .{ .name = "shadow_shader", .input = "src/agate/shaders/shadow.glsl", .output = "shadow_shader.zig" },
        .{ .name = "skybox_shader", .input = "src/agate/shaders/skybox.glsl", .output = "skybox_shader.zig" },
        .{ .name = "postprocess_shader", .input = "src/agate/shaders/postprocess.glsl", .output = "postprocess_shader.zig" },
        .{ .name = "particle_shader", .input = "src/agate/shaders/particle.glsl", .output = "particle_shader.zig" },
        .{ .name = "ui_shader", .input = "src/agate/shaders/ui.glsl", .output = "ui_shader.zig" },
        .{ .name = "ssao_shader", .input = "src/agate/shaders/ssao.glsl", .output = "ssao_shader.zig" },
        .{ .name = "ssao_blur_shader", .input = "src/agate/shaders/ssao_blur.glsl", .output = "ssao_blur_shader.zig" },
        .{ .name = "debug_shader", .input = "src/agate/shaders/debug.glsl", .output = "debug_shader.zig" },
        .{ .name = "bloom_down_shader", .input = "src/agate/shaders/bloom_down.glsl", .output = "bloom_down_shader.zig" },
        .{ .name = "bloom_up_shader", .input = "src/agate/shaders/bloom_up.glsl", .output = "bloom_up_shader.zig" },
        .{ .name = "outline_shader", .input = "src/agate/shaders/outline.glsl", .output = "outline_shader.zig" },
    };

    const dep_shdc = dep_sokol.builder.dependency("shdc", .{});
    var shader_modules: [shader_specs.len]*Build.Module = undefined;
    for (shader_specs, 0..) |spec, i| {
        const shader_mod = try sokol.shdc.createModule(b, spec.name, mod_sokol, .{
            .shdc_dep = dep_shdc,
            .input = spec.input,
            .output = spec.output,
            .slang = .{
                .glsl410 = true, // Linux (GL)
                .metal_macos = true, // macOS (Metal)
                .hlsl5 = true, // Windows (D3D11)
            },
        });
        shader_mod.addImport("math", mod_math);
        shader_modules[i] = shader_mod;
    }

    // Главный модуль библиотеки agate
    var agate_imports: [2 + shader_specs.len]Build.Module.Import = undefined;
    agate_imports[0] = .{ .name = "sokol", .module = mod_sokol };
    for (shader_specs, 0..) |spec, i| {
        agate_imports[1 + i] = .{ .name = spec.name, .module = shader_modules[i] };
    }
    agate_imports[1 + shader_specs.len] = .{ .name = "math", .module = mod_math };
    const mod_agate = b.addModule("agate", .{
        .root_source_file = b.path("src/agate/root.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &agate_imports,
    });
    mod_agate.addIncludePath(b.path("src/agate/c"));
    mod_agate.addIncludePath(b.path("src/agate/c/box3d/include"));
    mod_agate.addCSourceFile(.{
        .file = b.path("src/agate/c/c_impl.c"),
        // Asset decoding dominates scene startup at -O0. Keep Zig debug checks
        // and the selected release optimization mode unchanged.
        // stb_image only compiles its NEON paths when asked; without the define
        // JPEG decode runs scalar on Apple Silicon (measured ~2x slower).
        .flags = blk: {
            const neon = target.result.cpu.arch == .aarch64;
            if (optimize == .Debug) {
                break :blk if (neon) &.{ "-std=c99", "-O2", "-DSTBI_NEON" } else &.{ "-std=c99", "-O2" };
            }
            break :blk if (neon) &.{ "-std=c99", "-DSTBI_NEON" } else &.{"-std=c99"};
        },
    });
    // Box3D v0.1.0, vendored C17 sources (MIT). Public headers under
    // src/agate/c/box3d/include, internal headers resolve inside src/.
    mod_agate.addCSourceFiles(.{
        .files = &.{
            "src/agate/c/box3d/src/aabb.c",
            "src/agate/c/box3d/src/arena_allocator.c",
            "src/agate/c/box3d/src/bitset.c",
            "src/agate/c/box3d/src/block_allocator.c",
            "src/agate/c/box3d/src/body.c",
            "src/agate/c/box3d/src/broad_phase.c",
            "src/agate/c/box3d/src/capsule.c",
            "src/agate/c/box3d/src/compound.c",
            "src/agate/c/box3d/src/constraint_graph.c",
            "src/agate/c/box3d/src/contact.c",
            "src/agate/c/box3d/src/contact_solver.c",
            "src/agate/c/box3d/src/convex_manifold.c",
            "src/agate/c/box3d/src/core.c",
            "src/agate/c/box3d/src/distance.c",
            "src/agate/c/box3d/src/distance_joint.c",
            "src/agate/c/box3d/src/dynamic_tree.c",
            "src/agate/c/box3d/src/height_field.c",
            "src/agate/c/box3d/src/hull.c",
            "src/agate/c/box3d/src/id_pool.c",
            "src/agate/c/box3d/src/island.c",
            "src/agate/c/box3d/src/joint.c",
            "src/agate/c/box3d/src/manifold.c",
            "src/agate/c/box3d/src/math_functions.c",
            "src/agate/c/box3d/src/mesh.c",
            "src/agate/c/box3d/src/mesh_contact.c",
            "src/agate/c/box3d/src/motor_joint.c",
            "src/agate/c/box3d/src/mover.c",
            "src/agate/c/box3d/src/name_cache.c",
            "src/agate/c/box3d/src/parallel_for.c",
            "src/agate/c/box3d/src/parallel_joint.c",
            "src/agate/c/box3d/src/physics_world.c",
            "src/agate/c/box3d/src/prismatic_joint.c",
            "src/agate/c/box3d/src/recording.c",
            "src/agate/c/box3d/src/recording_replay.c",
            "src/agate/c/box3d/src/revolute_joint.c",
            "src/agate/c/box3d/src/scheduler.c",
            "src/agate/c/box3d/src/sensor.c",
            "src/agate/c/box3d/src/shape.c",
            "src/agate/c/box3d/src/simd.c",
            "src/agate/c/box3d/src/solver.c",
            "src/agate/c/box3d/src/solver_set.c",
            "src/agate/c/box3d/src/sphere.c",
            "src/agate/c/box3d/src/spherical_joint.c",
            "src/agate/c/box3d/src/table.c",
            "src/agate/c/box3d/src/timer.c",
            "src/agate/c/box3d/src/triangle_manifold.c",
            "src/agate/c/box3d/src/types.c",
            "src/agate/c/box3d/src/weld_joint.c",
            "src/agate/c/box3d/src/wheel_joint.c",
            "src/agate/c/box3d/src/world_snapshot.c",
        },
        .flags = &.{"-std=c17"},
    });
    // meshoptimizer v1.2 (MIT), decoder-only subset vendored under
    // src/agate/c/meshopt. Compiled as C++: the decoder sources are
    // runtime-free C++ (no exceptions/RTTI/stdlib), so no libc++ link is
    // needed and the module's existing libc link stays intact.
    mod_agate.addCSourceFiles(.{
        .files = &.{
            "src/agate/c/meshopt/indexcodec.cpp",
            "src/agate/c/meshopt/vertexcodec.cpp",
            "src/agate/c/meshopt/vertexfilter.cpp",
        },
        .flags = &.{ "-std=c++17", "-fno-exceptions", "-fno-rtti" },
    });
    mod_agate.link_libc = true;
    mod_agate.linkSystemLibrary("m", .{});

    const exe = b.addExecutable(.{
        .name = "agate",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "sokol", .module = mod_sokol },
                .{ .name = "agate", .module = mod_agate },
            },
        }),
    });
    b.installArtifact(exe);

    const run = b.addRunArtifact(exe);
    b.step("run", "Run the window").dependOn(&run.step);
    const lib_tests = b.addTest(.{
        .root_module = mod_agate,
    });
    const run_lib_tests = b.addRunArtifact(lib_tests);
    const test_step = b.step("test", "Run library tests");
    test_step.dependOn(&run_lib_tests.step);

    const fmt = b.addFmt(.{ .paths = &.{"src"}, .check = true });
    b.step("fmt", "Check formatting with zig fmt").dependOn(&fmt.step);
}
