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

    // Главный модуль библиотеки zenderer
    const mod_zenderer = b.addModule("zenderer", .{
        .root_source_file = b.path("src/zenderer/root.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "sokol", .module = mod_sokol },
            .{ .name = "shader", .module = mod_shader },
            .{ .name = "pbr_shader", .module = mod_pbr_shader },
            .{ .name = "instanced_shader", .module = mod_instanced_shader },
            .{ .name = "shadow_shader", .module = mod_shadow_shader },
            .{ .name = "skybox_shader", .module = mod_skybox_shader },
            .{ .name = "postprocess_shader", .module = mod_postprocess_shader },
            .{ .name = "particle_shader", .module = mod_particle_shader },
            .{ .name = "math", .module = mod_math },
        },
    });
    mod_zenderer.addIncludePath(b.path("src/zenderer/c"));
    mod_zenderer.addCSourceFile(.{
        .file = b.path("src/zenderer/c/c_impl.c"),
        .flags = &.{"-std=c99"},
    });
    mod_zenderer.link_libc = true;

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
