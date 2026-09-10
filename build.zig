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
    const mod_math = b.createModule(.{ .root_source_file = b.path("src/zenderer/math.zig") });

    // Шейдер: src/zenderer/shaders/standard.glsl -> Zig-модуль "shader"
    const dep_shdc = dep_sokol.builder.dependency("shdc", .{});
    const mod_shader = try sokol.shdc.createModule(b, "shader", mod_sokol, .{
        .shdc_dep = dep_shdc,
        .input = "src/zenderer/shaders/standard.glsl",
        .output = "standard_shader.zig",
        .slang = .{
            .glsl410 = true, // Linux (GL)
            .metal_macos = true, // macOS (Metal)
            .hlsl5 = true, // Windows (D3D11)
        },
    });
    mod_shader.addImport("math", mod_math);

    // Шейдер PBR: src/zenderer/shaders/pbr.glsl -> Zig-модуль "pbr_shader"
    const mod_pbr_shader = try sokol.shdc.createModule(b, "pbr_shader", mod_sokol, .{
        .shdc_dep = dep_shdc,
        .input = "src/zenderer/shaders/pbr.glsl",
        .output = "pbr_shader.zig",
        .slang = .{
            .glsl410 = true,
            .metal_macos = true,
            .hlsl5 = true,
        },
    });
    mod_pbr_shader.addImport("math", mod_math);

    // Шейдер Skinned PBR: src/zenderer/shaders/skinned_pbr.glsl -> Zig-модуль "skinned_pbr_shader"
    const mod_skinned_pbr_shader = try sokol.shdc.createModule(b, "skinned_pbr_shader", mod_sokol, .{
        .shdc_dep = dep_shdc,
        .input = "src/zenderer/shaders/skinned_pbr.glsl",
        .output = "skinned_pbr_shader.zig",
        .slang = .{
            .glsl410 = true,
            .metal_macos = true,
            .hlsl5 = true,
        },
    });
    mod_skinned_pbr_shader.addImport("math", mod_math);

    // Шейдер Instanced: src/zenderer/shaders/instanced.glsl -> Zig-модуль "instanced_shader"
    const mod_instanced_shader = try sokol.shdc.createModule(b, "instanced_shader", mod_sokol, .{
        .shdc_dep = dep_shdc,
        .input = "src/zenderer/shaders/instanced.glsl",
        .output = "instanced_shader.zig",
        .slang = .{
            .glsl410 = true,
            .metal_macos = true,
            .hlsl5 = true,
        },
    });
    mod_instanced_shader.addImport("math", mod_math);

    // Шейдер Shadow: src/zenderer/shaders/shadow.glsl -> Zig-модуль "shadow_shader"
    const mod_shadow_shader = try sokol.shdc.createModule(b, "shadow_shader", mod_sokol, .{
        .shdc_dep = dep_shdc,
        .input = "src/zenderer/shaders/shadow.glsl",
        .output = "shadow_shader.zig",
        .slang = .{
            .glsl410 = true,
            .metal_macos = true,
            .hlsl5 = true,
        },
    });
    mod_shadow_shader.addImport("math", mod_math);

    // Шейдер Skybox: src/zenderer/shaders/skybox.glsl -> Zig-модуль "skybox_shader"
    const mod_skybox_shader = try sokol.shdc.createModule(b, "skybox_shader", mod_sokol, .{
        .shdc_dep = dep_shdc,
        .input = "src/zenderer/shaders/skybox.glsl",
        .output = "skybox_shader.zig",
        .slang = .{
            .glsl410 = true,
            .metal_macos = true,
            .hlsl5 = true,
        },
    });
    mod_skybox_shader.addImport("math", mod_math);

    // Шейдер PostProcess: src/zenderer/shaders/postprocess.glsl -> Zig-модуль "postprocess_shader"
    const mod_postprocess_shader = try sokol.shdc.createModule(b, "postprocess_shader", mod_sokol, .{
        .shdc_dep = dep_shdc,
        .input = "src/zenderer/shaders/postprocess.glsl",
        .output = "postprocess_shader.zig",
        .slang = .{
            .glsl410 = true,
            .metal_macos = true,
            .hlsl5 = true,
        },
    });
    mod_postprocess_shader.addImport("math", mod_math);

    // Шейдер Particle: src/zenderer/shaders/particle.glsl -> Zig-модуль "particle_shader"
    const mod_particle_shader = try sokol.shdc.createModule(b, "particle_shader", mod_sokol, .{
        .shdc_dep = dep_shdc,
        .input = "src/zenderer/shaders/particle.glsl",
        .output = "particle_shader.zig",
        .slang = .{
            .glsl410 = true,
            .metal_macos = true,
            .hlsl5 = true,
        },
    });
    mod_particle_shader.addImport("math", mod_math);

    // Шейдер UI & Text: src/zenderer/shaders/ui.glsl -> Zig-модуль "ui_shader"
    const mod_ui_shader = try sokol.shdc.createModule(b, "ui_shader", mod_sokol, .{
        .shdc_dep = dep_shdc,
        .input = "src/zenderer/shaders/ui.glsl",
        .output = "ui_shader.zig",
        .slang = .{
            .glsl410 = true,
            .metal_macos = true,
            .hlsl5 = true,
        },
    });
    mod_ui_shader.addImport("math", mod_math);

    // Шейдер SSAO: src/zenderer/shaders/ssao.glsl -> Zig-модуль "ssao_shader"
    const mod_ssao_shader = try sokol.shdc.createModule(b, "ssao_shader", mod_sokol, .{
        .shdc_dep = dep_shdc,
        .input = "src/zenderer/shaders/ssao.glsl",
        .output = "ssao_shader.zig",
        .slang = .{
            .glsl410 = true,
            .metal_macos = true,
            .hlsl5 = true,
        },
    });
    mod_ssao_shader.addImport("math", mod_math);

    // Шейдер SSAO Blur: src/zenderer/shaders/ssao_blur.glsl -> Zig-модуль "ssao_blur_shader"
    const mod_ssao_blur_shader = try sokol.shdc.createModule(b, "ssao_blur_shader", mod_sokol, .{
        .shdc_dep = dep_shdc,
        .input = "src/zenderer/shaders/ssao_blur.glsl",
        .output = "ssao_blur_shader.zig",
        .slang = .{
            .glsl410 = true,
            .metal_macos = true,
            .hlsl5 = true,
        },
    });
    mod_ssao_blur_shader.addImport("math", mod_math);

    // Главный модуль библиотеки zenderer
    const mod_zenderer = b.addModule("zenderer", .{
        .root_source_file = b.path("src/zenderer/root.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "sokol", .module = mod_sokol },
            .{ .name = "shader", .module = mod_shader },
            .{ .name = "pbr_shader", .module = mod_pbr_shader },
            .{ .name = "skinned_pbr_shader", .module = mod_skinned_pbr_shader },
            .{ .name = "instanced_shader", .module = mod_instanced_shader },
            .{ .name = "shadow_shader", .module = mod_shadow_shader },
            .{ .name = "skybox_shader", .module = mod_skybox_shader },
            .{ .name = "postprocess_shader", .module = mod_postprocess_shader },
            .{ .name = "particle_shader", .module = mod_particle_shader },
            .{ .name = "ui_shader", .module = mod_ui_shader },
            .{ .name = "ssao_shader", .module = mod_ssao_shader },
            .{ .name = "ssao_blur_shader", .module = mod_ssao_blur_shader },
            .{ .name = "math", .module = mod_math },
        },
    });
    mod_zenderer.addIncludePath(b.path("src/zenderer/c"));
    mod_zenderer.addIncludePath(b.path("src/zenderer/c/box3d/include"));
    mod_zenderer.addCSourceFile(.{
        .file = b.path("src/zenderer/c/c_impl.c"),
        // Asset decoding dominates scene startup at -O0. Keep Zig debug checks
        // and the selected release optimization mode unchanged.
        .flags = if (optimize == .Debug) &.{ "-std=c99", "-O2" } else &.{"-std=c99"},
    });
    // Box3D v0.1.0, vendored C17 sources (MIT). Public headers under
    // src/zenderer/c/box3d/include, internal headers resolve inside src/.
    mod_zenderer.addCSourceFiles(.{
        .files = &.{
            "src/zenderer/c/box3d/src/aabb.c",
            "src/zenderer/c/box3d/src/arena_allocator.c",
            "src/zenderer/c/box3d/src/bitset.c",
            "src/zenderer/c/box3d/src/block_allocator.c",
            "src/zenderer/c/box3d/src/body.c",
            "src/zenderer/c/box3d/src/broad_phase.c",
            "src/zenderer/c/box3d/src/capsule.c",
            "src/zenderer/c/box3d/src/compound.c",
            "src/zenderer/c/box3d/src/constraint_graph.c",
            "src/zenderer/c/box3d/src/contact.c",
            "src/zenderer/c/box3d/src/contact_solver.c",
            "src/zenderer/c/box3d/src/convex_manifold.c",
            "src/zenderer/c/box3d/src/core.c",
            "src/zenderer/c/box3d/src/distance.c",
            "src/zenderer/c/box3d/src/distance_joint.c",
            "src/zenderer/c/box3d/src/dynamic_tree.c",
            "src/zenderer/c/box3d/src/height_field.c",
            "src/zenderer/c/box3d/src/hull.c",
            "src/zenderer/c/box3d/src/id_pool.c",
            "src/zenderer/c/box3d/src/island.c",
            "src/zenderer/c/box3d/src/joint.c",
            "src/zenderer/c/box3d/src/manifold.c",
            "src/zenderer/c/box3d/src/math_functions.c",
            "src/zenderer/c/box3d/src/mesh.c",
            "src/zenderer/c/box3d/src/mesh_contact.c",
            "src/zenderer/c/box3d/src/motor_joint.c",
            "src/zenderer/c/box3d/src/mover.c",
            "src/zenderer/c/box3d/src/name_cache.c",
            "src/zenderer/c/box3d/src/parallel_for.c",
            "src/zenderer/c/box3d/src/parallel_joint.c",
            "src/zenderer/c/box3d/src/physics_world.c",
            "src/zenderer/c/box3d/src/prismatic_joint.c",
            "src/zenderer/c/box3d/src/recording.c",
            "src/zenderer/c/box3d/src/recording_replay.c",
            "src/zenderer/c/box3d/src/revolute_joint.c",
            "src/zenderer/c/box3d/src/scheduler.c",
            "src/zenderer/c/box3d/src/sensor.c",
            "src/zenderer/c/box3d/src/shape.c",
            "src/zenderer/c/box3d/src/simd.c",
            "src/zenderer/c/box3d/src/solver.c",
            "src/zenderer/c/box3d/src/solver_set.c",
            "src/zenderer/c/box3d/src/sphere.c",
            "src/zenderer/c/box3d/src/spherical_joint.c",
            "src/zenderer/c/box3d/src/table.c",
            "src/zenderer/c/box3d/src/timer.c",
            "src/zenderer/c/box3d/src/triangle_manifold.c",
            "src/zenderer/c/box3d/src/types.c",
            "src/zenderer/c/box3d/src/weld_joint.c",
            "src/zenderer/c/box3d/src/wheel_joint.c",
            "src/zenderer/c/box3d/src/world_snapshot.c",
        },
        .flags = &.{"-std=c17"},
    });
    mod_zenderer.link_libc = true;
    mod_zenderer.linkSystemLibrary("m", .{});

    const exe = b.addExecutable(.{
        .name = "zenderer",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "sokol", .module = mod_sokol },
                .{ .name = "zenderer", .module = mod_zenderer },
            },
        }),
    });
    b.installArtifact(exe);

    const run = b.addRunArtifact(exe);
    b.step("run", "Run the window").dependOn(&run.step);
    const lib_tests = b.addTest(.{
        .root_module = mod_zenderer,
    });
    const run_lib_tests = b.addRunArtifact(lib_tests);
    const test_step = b.step("test", "Run library tests");
    test_step.dependOn(&run_lib_tests.step);
}
