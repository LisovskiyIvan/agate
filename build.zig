const std = @import("std");
const Build = std.Build;
const sokol = @import("sokol");

// ---------------------------------------------------------------------------
// User shader materials (hook level).
//
// Add one entry + one GLSL snippet file to get a custom shader material
// without touching any engine source: the snippet is merged into the base
// template's hook markers at build time (see
// src/agate/shader_material/merge.zig), compiled by sokol-shdc under
// glsl430 + metal_macos + hlsl5 (glsl430: the base templates carry the
// wave-30 clustered storage blocks, and SSBO syntax needs GLSL 4.30+),
// and registered in the runtime registry
// under `name`. Engine side picks it up by name:
//
//   const idx = agate.shader_material.indexForName("my_effect");
//   const mat = try scene.createShaderMaterial("fx", "my_effect");
//   mesh.material = .{ .shader_material = mat };
//
// Snippet format:
//   // agate shader material: my_effect
//   // base: standard            (or "pbr")
//   // @param u_speed float = 2.0
//   // @param u_tint vec4 = 1 0.5 0 1
//   // @hook(albedo)
//   base.rgb = ...;
//
// Hooks: vertex (vs, morphed_pos/morphed_nrm), albedo, emissive (pbr only),
// post_lighting, decls (owned by the generated user uniform block).
// Snippets must be plain statement-level GLSL valid for GLSL 430, Metal and
// HLSL 5 (the merge output is compiled under all three slangs).
// ---------------------------------------------------------------------------
pub const UserShaderMaterial = struct {
    /// Registry name ([a-z0-9_]); also the runtime lookup key (Wyhash).
    name: []const u8,
    /// Path to the snippet file (build-root relative).
    snippet: []const u8,
    /// Engine template the snippet hooks into.
    base: Base = .standard,
    pub const Base = enum { standard, pbr };
};

pub const user_shader_materials = [_]UserShaderMaterial{
    .{ .name = "ramp_wave", .snippet = "examples/shader_materials/ramp_wave.glsl" },
    // Material library v1 presets (see src/agate/material_library.zig):
    // procedural Sky/Gradient/Grid/TriPlanar constructors over the same
    // hook-merge pipeline. Each entry compiles its snippet into the
    // standard template under glsl430/metal_macos/hlsl5 at build time.
    .{ .name = "matlib_sky", .snippet = "examples/shader_materials/matlib_sky.glsl" },
    .{ .name = "matlib_gradient", .snippet = "examples/shader_materials/matlib_gradient.glsl" },
    .{ .name = "matlib_grid", .snippet = "examples/shader_materials/matlib_grid.glsl" },
    .{ .name = "matlib_triplanar", .snippet = "examples/shader_materials/matlib_triplanar.glsl" },
};

// ---------------------------------------------------------------------------
// Public: user-owned shader compilation (ShaderMaterial v1 external path).
//
// A downstream project compiles its OWN .glsl (sokol-shdc format, like
// src/agate/shaders/*.glsl) without editing any agate source, from its own
// build.zig:
//
//   const agate_build = @import("agate"); // agate's build.zig, like sokol's
//   const dep_agate = b.dependency("agate", .{ .target = target, .optimize = optimize });
//   const my_shader = try agate_build.compileUserShader(b, dep_agate, .{
//       .name = "my_shader",
//       .input = "shaders/my.glsl", // downstream build-root relative
//       .target = target,
//       .optimize = optimize,
//   });
//   exe.root_module.addImport("my_shader", my_shader);
//
// At startup the downstream registers the compiled shader for the CURRENT
// backend and points a material at it (single-context-thread contract, see
// shader_material.registerRuntime):
//
//   const my_mod = @import("my_shader");
//   const Entry = struct {
//       fn makeShader(backend: sg.Backend) sg.Shader {
//           return sg.makeShader(my_mod.myShaderDesc(backend));
//       }
//   };
//   _ = try z.shader_material.registerRuntime(.{
//       .name = "my_effect",
//       .make_shader = Entry.makeShader,
//       .engine_template = false,
//       .user_ub = my_mod.UB_my_user_block,
//       .params = &my_params, // f32 offsets into the user uniform storage
//   });
//   const mat = scene.createShaderMaterial("fx", "my_effect") orelse unreachable;
//   mesh.material = .{ .shader_material = mat };
//
// sokol/shdc resolve THROUGH dep_agate.builder, so the downstream shares
// agate's single sokol module instance (no second vendor/sokol dependency,
// no duplicate sg state). Pass the same target/optimize you used for the
// agate dependency itself. The generated module unconditionally
// `@import("math")`, wired here to agate's math facade.
// ---------------------------------------------------------------------------

/// Backend codegen shared by the engine forward shaders and user shaders
/// (glsl430 carries the template SSBO syntax; metal_macos/hlsl5 cover the
/// remaining desktop backends).
pub const engine_shader_slang = sokol.shdc.Slang{
    .glsl430 = true,
    .metal_macos = true,
    .hlsl5 = true,
};

pub const UserShaderSpec = struct {
    /// Zig module name for the generated shader (also the @import name).
    name: []const u8,
    /// Downstream build-root-relative path to the .glsl (sokol-shdc format:
    /// @vs/@fs/@program blocks, see src/agate/shaders/standard.glsl).
    input: []const u8,
    /// Generated file name; default "<name>.zig".
    output: ?[]const u8 = null,
    /// Slang set; default engine_shader_slang. shdc compiles every leg in
    /// one invocation, so a broken Metal/HLSL leg fails the build here —
    /// never silently at runtime on another OS.
    slang: ?sokol.shdc.Slang = null,
    /// Must match the target/optimize of the downstream's agate dependency
    /// (used to resolve agate's sokol/shdc instances for this build).
    target: Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
};

pub fn compileUserShader(b: *Build, dep_agate: *Build.Dependency, spec: UserShaderSpec) !*Build.Module {
    if (spec.name.len == 0 or spec.input.len == 0) return error.UserShaderBadSpec;
    const dep_sokol = dep_agate.builder.dependency("sokol", .{
        .target = spec.target,
        .optimize = spec.optimize,
    });
    const mod_sokol = dep_sokol.module("sokol");
    const dep_shdc = dep_sokol.builder.dependency("shdc", .{});
    const shader_mod = try sokol.shdc.createModule(b, spec.name, mod_sokol, .{
        .shdc_dep = dep_shdc,
        .input = spec.input,
        .output = spec.output orelse b.fmt("{s}.zig", .{spec.name}),
        .slang = spec.slang orelse engine_shader_slang,
    });
    const mod_math = b.createModule(.{
        .root_source_file = dep_agate.path("src/agate/math.zig"),
    });
    shader_mod.addImport("math", mod_math);
    return shader_mod;
}

// ---------------------------------------------------------------------------
// Test aggregation, dir-based.
//
// src/agate/tests.zig is GENERATED from the src/agate tree, so a newly added
// module's tests are picked up without touching any hand-maintained import
// list (the historical failure mode: tests silently stopped reaching the
// runner while the suite stayed green).
//
// Why not `refAllDeclsRecursive`? Removed from std in 0.16; a hand-rolled
// recursion over the facade would recurse into sokol/C bindings (compile-time
// explosion) and loops on the engine's cyclic file imports. Why not plain
// `refAllDecls`? One level only: files whose decls nothing references stay
// unanalyzed, and their tests silently vanish — exactly the rot we are
// guarding against. A directory walk is total: every module file is imported,
// dead or uncompiled files surface as build errors.
//
// Zig 0.16 requires `@import` operands to be string literals, so the
// generated literals must live in a real file. That file is tracked as
// src/agate/tests.zig, but a normal `zig build` (including builds that use
// agate as a dependency) MUST NOT rewrite it: silent source mutation breaks
// packaging reproducibility and races with concurrent edits.
//
// Explicit workflow (run from an agate checkout, then commit the result):
//   zig build update-tests   # regenerate src/agate/tests.zig from the tree
//   zig build test           # fails via CheckFile while the registry is stale
//
// Ordinary library/exe builds never run the CheckFile step, and the vendored
// test runner (tools/test_runner.zig) is only referenced by the `test` step,
// so dependency users building the library do not need it.
// ---------------------------------------------------------------------------
fn computeTestRegistryBytes(b: *Build) []const u8 {
    const gpa = b.allocator;
    const io = b.graph.io;

    var paths: std.ArrayListUnmanaged([]const u8) = .empty;
    defer {
        for (paths.items) |p| gpa.free(p);
        paths.deinit(gpa);
    }

    const src_path = b.path("src/agate").getPath3(b, null);
    var dir = src_path.root_dir.handle.openDir(io, src_path.subPathOrDot(), .{ .iterate = true }) catch
        @panic("test registry: cannot open src/agate");
    defer dir.close(io);

    var walker = dir.walk(gpa) catch @panic("test registry: OOM");
    defer walker.deinit();
    while (walker.next(io) catch |err| @panic(@errorName(err))) |entry| {
        if (entry.kind != .file) continue;
        if (!std.mem.endsWith(u8, entry.path, ".zig")) continue;
        if (excludedFromTestRegistry(entry.path)) continue;
        const dup = gpa.dupe(u8, entry.path) catch @panic("test registry: OOM");
        paths.append(gpa, dup) catch @panic("test registry: OOM");
    }
    std.mem.sort([]const u8, paths.items, {}, struct {
        fn less(_: void, a: []const u8, bb: []const u8) bool {
            return std.mem.order(u8, a, bb) == .lt;
        }
    }.less);

    var out: std.ArrayListUnmanaged(u8) = .empty;
    // NOTE: `out` intentionally leaks: `b.allocator` lives for the whole
    // configure phase, and both CheckFile (`expected_exact`) and
    // UpdateSourceFiles (`addBytesToSource`) retain the slice without copying.
    out.appendSlice(gpa,
        \\//! GENERATED by build.zig (computeTestRegistryBytes) from the src/agate tree.
        \\//! Do not edit by hand: regenerate with `zig build update-tests`.
        \\//! Every module file under src/agate is imported so its `test` blocks
        \\//! reach the runner; math is a separate build module and is imported
        \\//! by name (its own test block aggregates math/*.zig).
        \\
        \\test {
        \\    _ = @import("math");
        \\
    ) catch @panic("test registry: OOM");
    for (paths.items) |p| {
        out.print(gpa, "    _ = @import(\"{s}\");\n", .{p}) catch @panic("test registry: OOM");
    }
    out.appendSlice(gpa, "}\n") catch @panic("test registry: OOM");

    return out.items;
}

/// Registry exclusions, each with the reason it is imported elsewhere (or not
/// at all) instead of by the generated list.
fn excludedFromTestRegistry(path: []const u8) bool {
    // Library facade: the test binary's import chain lives here already;
    // importing it from the registry would force-analyze the entire pub
    // facade (refAllDecls-style) and cycle back into the registry.
    if (std.mem.eql(u8, path, "root.zig")) return true;
    // The registry itself.
    if (std.mem.eql(u8, path, "tests.zig")) return true;
    // Separate build module, emitted as the literal `@import("math")`.
    if (std.mem.eql(u8, path, "math.zig")) return true;
    // Aggregated by math.zig's own test block under the "math" module.
    if (std.mem.startsWith(u8, path, "math/")) return true;
    // Host CLI for the build-time shader merge; compiled by build.zig as
    // merge_shader_material on every build, not part of the library.
    if (std.mem.eql(u8, path, "shader_material/tool_main.zig")) return true;
    return false;
}

/// Clang flags disabling every sanitizer-coverage feature zig enables for C
/// sources in fuzz mode. See the c_impl.c flag block for the rationale.
const no_sancov = "-fno-sanitize-coverage=trace-pc-guard,inline-8bit-counters,pc-table,indirect-calls,trace-cmp,trace-div,trace-gep,inline-bool-flag";

pub fn build(b: *Build) !void {
    // Test registry: compute-only at configure time. Never writes to source
    // here, so ordinary builds (including as a dependency) are reproducible.
    // `zig build test` fails while the registry is stale; regenerate with
    // `zig build update-tests` (UpdateSourceFiles, make-phase mutation).
    const expected_test_registry = computeTestRegistryBytes(b);
    const update_sources = b.addUpdateSourceFiles();
    update_sources.addBytesToSource(expected_test_registry, "src/agate/tests.zig");
    b.step("update-tests", "Regenerate src/agate/tests.zig from the src/agate tree").dependOn(&update_sources.step);

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
        /// Необязательный пер-модульный набор слэнгов. Дефолт (null) — общий
        /// набор glsl410/metal_macos/hlsl5. Compute-шейдеры не выражаются в
        /// GLSL 410, поэтому им нужен свой вызов shdc с glsl430 (доступность
        /// на рантайме всё равно гейтится sg.queryFeatures().compute, см.
        /// src/agate/compute.zig).
        slang: ?sokol.shdc.Slang = null,
    };
    const default_slang = sokol.shdc.Slang{
        .glsl410 = true, // Linux (GL)
        .metal_macos = true, // macOS (Metal)
        .hlsl5 = true, // Windows (D3D11)
    };
    // Forward shaders with fragment-stage storage buffers (clustered
    // lights, wave 30): SSBO syntax is not valid GLSL 4.10, so these five
    // compile the GLSL leg as 4.30 (a strict superset for everything else
    // they use). The Metal/HLSL legs are unchanged. Runtime GLSL needs a
    // 4.3+ context (true on Linux desktop GL; macOS desktop GL caps at 4.1
    // and never supported storage buffers — see src/agate/compute.zig).
    const forward_slang = sokol.shdc.Slang{
        .glsl430 = true,
        .metal_macos = true,
        .hlsl5 = true,
    };
    const shader_specs = [_]ShaderSpec{
        .{ .name = "shader", .input = "src/agate/shaders/standard.glsl", .output = "standard_shader.zig", .slang = forward_slang },
        .{ .name = "pbr_shader", .input = "src/agate/shaders/pbr.glsl", .output = "pbr_shader.zig", .slang = forward_slang },
        .{ .name = "skinned_pbr_shader", .input = "src/agate/shaders/skinned_pbr.glsl", .output = "skinned_pbr_shader.zig", .slang = forward_slang },
        .{ .name = "instanced_shader", .input = "src/agate/shaders/instanced.glsl", .output = "instanced_shader.zig", .slang = forward_slang },
        .{ .name = "instanced_pbr_shader", .input = "src/agate/shaders/instanced_pbr.glsl", .output = "instanced_pbr_shader.zig", .slang = forward_slang },
        .{ .name = "shadow_shader", .input = "src/agate/shaders/shadow.glsl", .output = "shadow_shader.zig" },
        .{ .name = "skybox_shader", .input = "src/agate/shaders/skybox.glsl", .output = "skybox_shader.zig" },
        .{ .name = "postprocess_shader", .input = "src/agate/shaders/postprocess.glsl", .output = "postprocess_shader.zig" },
        .{ .name = "particle_shader", .input = "src/agate/shaders/particle.glsl", .output = "particle_shader.zig" },
        // Stateful compute particles (wave 25): compute slang set (410 has
        // no compute); runtime availability still gates on
        // compute.supported(), see src/agate/compute.zig.
        .{
            .name = "particle_compute_shader",
            .input = "src/agate/shaders/particle_compute.glsl",
            .output = "particle_compute_shader.zig",
            .slang = .{ .glsl430 = true, .metal_macos = true, .hlsl5 = true },
        },
        .{ .name = "ui_shader", .input = "src/agate/shaders/ui.glsl", .output = "ui_shader.zig" },
        .{ .name = "ssao_shader", .input = "src/agate/shaders/ssao.glsl", .output = "ssao_shader.zig" },
        .{ .name = "ssao_blur_shader", .input = "src/agate/shaders/ssao_blur.glsl", .output = "ssao_blur_shader.zig" },
        .{ .name = "debug_shader", .input = "src/agate/shaders/debug.glsl", .output = "debug_shader.zig" },
        .{ .name = "bloom_down_shader", .input = "src/agate/shaders/bloom_down.glsl", .output = "bloom_down_shader.zig" },
        .{ .name = "bloom_up_shader", .input = "src/agate/shaders/bloom_up.glsl", .output = "bloom_up_shader.zig" },
        .{ .name = "glow_extract_shader", .input = "src/agate/shaders/glow_extract.glsl", .output = "glow_extract_shader.zig" },
        .{ .name = "glow_blur_shader", .input = "src/agate/shaders/glow_blur.glsl", .output = "glow_blur_shader.zig" },
        .{ .name = "outline_shader", .input = "src/agate/shaders/outline.glsl", .output = "outline_shader.zig" },
        .{ .name = "probe_mip_shader", .input = "src/agate/shaders/probe_mip.glsl", .output = "probe_mip_shader.zig" },
        .{ .name = "ui3d_panel_shader", .input = "src/agate/shaders/ui3d_panel.glsl", .output = "ui3d_panel_shader.zig" },
    };

    const dep_shdc = dep_sokol.builder.dependency("shdc", .{});
    var shader_modules: [shader_specs.len]*Build.Module = undefined;
    for (shader_specs, 0..) |spec, i| {
        const shader_mod = try sokol.shdc.createModule(b, spec.name, mod_sokol, .{
            .shdc_dep = dep_shdc,
            .input = spec.input,
            .output = spec.output,
            .slang = spec.slang orelse default_slang,
        });
        shader_mod.addImport("math", mod_math);
        shader_modules[i] = shader_mod;
    }

    // Shader materials: hook-merge each user snippet into its base template,
    // compile the merged GLSL with sokol-shdc (same slangs as the engine
    // shaders) and generate the `shader_material_registry` module the
    // runtime resolves registrations from. sokol.shdc.createModule only
    // accepts build-root paths, so the shdc invocation is replicated here
    // with a LazyPath input (the merge step's output).
    const mod_registry = try createShaderMaterialRegistry(b, mod_sokol, mod_math, dep_shdc);

    // Главный модуль библиотеки agate
    var agate_imports: [3 + shader_specs.len]Build.Module.Import = undefined;
    agate_imports[0] = .{ .name = "sokol", .module = mod_sokol };
    for (shader_specs, 0..) |spec, i| {
        agate_imports[1 + i] = .{ .name = spec.name, .module = shader_modules[i] };
    }
    agate_imports[1 + shader_specs.len] = .{ .name = "math", .module = mod_math };
    agate_imports[2 + shader_specs.len] = .{ .name = "shader_material_registry", .module = mod_registry };
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
            // -fno-sanitize-coverage: in fuzz mode (`zig build test --fuzz`)
            // zig adds clang sancov instrumentation to this graph's C sources;
            // its fuzzer runtime cannot account for the extra counters and
            // panics at init ("pc counters length and pcs length do not
            // match"). C decoders still execute under the fuzzer, they just
            // contribute no coverage feedback. User cflags land after zig's.
            if (optimize == .Debug) {
                break :blk if (neon)
                    &.{ "-std=c99", "-O2", "-DSTBI_NEON", no_sancov }
                else
                    &.{ "-std=c99", "-O2", no_sancov };
            }
            break :blk if (neon)
                &.{ "-std=c99", "-DSTBI_NEON", no_sancov }
            else
                &.{ "-std=c99", no_sancov };
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
        .flags = &.{ "-std=c17", no_sancov },
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
        .flags = &.{ "-std=c++17", "-fno-exceptions", "-fno-rtti", no_sancov },
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
    if (b.args) |args| run.addArgs(args);
    b.step("run", "Run the window").dependOn(&run.step);
    const lib_tests = b.addTest(.{
        .root_module = mod_agate,
        // Vendored runner (tools/test_runner.zig): stock 0.16.0 fails to
        // compile ANY fuzz-mode build (`*builtin.StackTrace` vs
        // `*std.debug.StackTrace` in the -ffuzz-only branch). Same runner
        // serves both modes: plain `zig build test` runs every fuzz target's
        // corpus once (smoke pass), while
        //
        //     zig build test --fuzz[=limit]
        //
        // switches the build into coverage-guided deep fuzzing (-ffuzz) —
        // e.g. `--fuzz=10K` caps iterations; without a limit it runs until
        // interrupted. Findings are summarized in a "FUZZING REPORT".
        .test_runner = .{ .path = b.path("tools/test_runner.zig"), .mode = .server },
    });
    const run_lib_tests = b.addRunArtifact(lib_tests);
    // Stale-registry gate: runs only for `zig build test` (make phase via
    // CheckFile), leaving ordinary library/exe builds untouched. The vendored
    // runner stays lazy: `b.path("tools/test_runner.zig")` above only
    // materializes when this test compile actually builds.
    const check_test_registry = b.addCheckFile(b.path("src/agate/tests.zig"), .{ .expected_exact = expected_test_registry });
    run_lib_tests.step.dependOn(&check_test_registry.step);
    const test_step = b.step("test", "Run library tests (fails while src/agate/tests.zig is stale; run `zig build update-tests`)");
    test_step.dependOn(&run_lib_tests.step);

    const fmt = b.addFmt(.{ .paths = &.{"src"}, .check = true });
    b.step("fmt", "Check formatting with zig fmt").dependOn(&fmt.step);
}

// Hook-level shader material pipeline: merge tool -> sokol-shdc -> generated
// registry module. Returns the `shader_material_registry` module (always
// exists; `entries` is empty when user_shader_materials is empty).
fn createShaderMaterialRegistry(
    b: *Build,
    mod_sokol: *Build.Module,
    mod_math: *Build.Module,
    dep_shdc: *Build.Dependency,
) !*Build.Module {
    // Host tool that performs the hook merge (see shader_material/merge.zig).
    const merge_tool = b.addExecutable(.{
        .name = "merge_shader_material",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/agate/shader_material/tool_main.zig"),
            // Host tool: runs inside the build graph.
            .target = b.graph.host,
            .optimize = .ReleaseSafe,
        }),
    });

    const shdc_exe = dep_shdc.path(try sokol.shdc.getShdcSubPath());

    var registry_src: std.ArrayListUnmanaged(u8) = .empty;
    var registry_import_names: std.ArrayListUnmanaged([]const u8) = .empty;
    var registry_import_modules: std.ArrayListUnmanaged(*Build.Module) = .empty;

    try registry_src.appendSlice(b.allocator,
        \\// GENERATED by build.zig from the `user_shader_materials` table.
        \\// Do not edit: changes belong in build.zig / the snippet files.
        \\const sg = @import("sokol").gfx;
        \\
        \\pub const Base = enum { standard, pbr };
        \\pub const Param = struct {
        \\    name: []const u8,
        \\    offset: u8,
        \\    comps: u8,
        \\    default: [4]f32 = .{ 0, 0, 0, 0 },
        \\};
        \\pub const Entry = struct {
        \\    name: []const u8,
        \\    key: u64,
        \\    base: Base,
        \\    vs_ub: u32,
        \\    fs_ub: u32,
        \\    user_ub: ?u32,
        \\    vs_user_ub: ?u32,
        \\    params: []const Param,
        \\    make_shader: *const fn (sg.Backend) sg.Shader,
        \\};
        \\
        \\
    );

    var seen_names: std.StringHashMapUnmanaged(void) = .empty;
    for (user_shader_materials) |mat| {
        // Names become Zig identifiers (module/fn names) and registry keys.
        if (mat.name.len == 0) return error.ShaderMaterialBadName;
        for (mat.name) |ch| {
            const ok = (ch >= 'a' and ch <= 'z') or (ch >= '0' and ch <= '9') or ch == '_';
            if (!ok) return error.ShaderMaterialBadName;
        }
        if (seen_names.contains(mat.name)) return error.ShaderMaterialDuplicateName;
        try seen_names.put(b.allocator, mat.name, {});

        const base_glsl = switch (mat.base) {
            .standard => "src/agate/shaders/standard.glsl",
            .pbr => "src/agate/shaders/pbr.glsl",
        };
        const prog_name: []const u8 = switch (mat.base) {
            .standard => "standardShaderDesc",
            .pbr => "pbrShaderDesc",
        };

        // 1. Merge template + snippet (host tool step).
        const run_merge = b.addRunArtifact(merge_tool);
        run_merge.addArg("--template");
        run_merge.addFileArg(b.path(base_glsl));
        run_merge.addArg("--snippet");
        run_merge.addFileArg(b.path(mat.snippet));
        run_merge.addArg("--base");
        run_merge.addArg(@tagName(mat.base));
        run_merge.addArg("--name");
        run_merge.addArg(mat.name);
        run_merge.addArg("--out-glsl");
        const merged_glsl = run_merge.addOutputFileArg(b.fmt("shader_mat_{s}.glsl", .{mat.name}));
        run_merge.addArg("--out-params");
        const params_zig = run_merge.addOutputFileArg(b.fmt("shader_mat_{s}_params.zig", .{mat.name}));

        // 2. sokol-shdc on the merged GLSL — same slangs as the engine
        // forward shader table (glsl430 / metal_macos / hlsl5: hook
        // materials inherit the clustered storage blocks from the
        // templates, and SSBO syntax needs GLSL 4.30+). sokol.shdc's
        // createModule only accepts build-root paths, so the invocation is
        // replicated here to feed it the merge step's LazyPath output.
        // argv[0] = the sokol-shdc binary (resolved eagerly like
        // sokol.shdc does for zig 0.16).
        const run_shdc = b.addSystemCommand(&.{shdc_exe.getPath(b)});
        run_shdc.addArgs(&.{ "-l", "glsl430:metal_macos:hlsl5", "-f", "sokol_zig" });
        run_shdc.addArg("--input");
        run_shdc.addFileArg(merged_glsl);
        run_shdc.addArg("--output");
        const shader_zig = run_shdc.addOutputFileArg(b.fmt("shader_mat_{s}_shader.zig", .{mat.name}));

        const shader_mod = b.addModule(b.fmt("shader_mat_{s}", .{mat.name}), .{
            .root_source_file = shader_zig,
        });
        shader_mod.addImport("sokol", mod_sokol);
        shader_mod.addImport("math", mod_math);

        const params_mod = b.addModule(b.fmt("shader_mat_{s}_params", .{mat.name}), .{
            .root_source_file = params_zig,
        });

        // 3. Registry entry (comptime conversion of the generated params
        // table; key matches shader_material.keyForName — Wyhash of the name).
        const key = std.hash.Wyhash.hash(0, mat.name);
        try registry_src.print(b.allocator,
            \\const {s}_mod = @import("shader_mat_{s}");
            \\const {s}_params_mod = @import("shader_mat_{s}_params");
            \\fn {s}_make_shader(backend: sg.Backend) sg.Shader {{
            \\    return sg.makeShader({s}_mod.{s}(backend));
            \\}}
            \\fn {s}_params() []const Param {{
            \\    const frozen = comptime blk: {{
            \\        const raw = {s}_params_mod.params;
            \\        var arr: [raw.len]Param = undefined;
            \\        for (raw, 0..) |r, i| arr[i] = .{{ .name = r.name, .offset = r.offset, .comps = r.comps, .default = r.default }};
            \\        break :blk arr;
            \\    }};
            \\    return &frozen;
            \\}}
            \\const {s}_entry = Entry{{
            \\    .name = "{s}",
            \\    .key = 0x{X},
            \\    .base = .{s},
            \\    .vs_ub = {s}_mod.UB_vs_params,
            \\    .fs_ub = {s}_mod.UB_fs_params,
            \\    .user_ub = if (@hasDecl({s}_mod, "UB_sm_user_params")) {s}_mod.UB_sm_user_params else null,
            \\    .vs_user_ub = if (@hasDecl({s}_mod, "UB_sm_user_params_vs")) {s}_mod.UB_sm_user_params_vs else null,
            \\    .params = {s}_params(),
            \\    .make_shader = {s}_make_shader,
            \\}};
            \\
        , .{ mat.name, mat.name, mat.name, mat.name, mat.name, mat.name, prog_name, mat.name, mat.name, mat.name, mat.name, key, @tagName(mat.base), mat.name, mat.name, mat.name, mat.name, mat.name, mat.name, mat.name, mat.name });

        try registry_import_names.appendSlice(b.allocator, &.{ b.fmt("shader_mat_{s}", .{mat.name}), b.fmt("shader_mat_{s}_params", .{mat.name}) });
        try registry_import_modules.appendSlice(b.allocator, &.{ shader_mod, params_mod });
    }

    try registry_src.appendSlice(b.allocator, "pub const entries = [_]Entry{");
    for (user_shader_materials) |mat| {
        try registry_src.print(b.allocator, "{s}_entry,", .{mat.name});
    }
    try registry_src.appendSlice(b.allocator, "};\n");

    const registry_path = b.addWriteFiles().add("shader_material_registry.zig", registry_src.items);
    const registry_mod = b.addModule("shader_material_registry", .{
        .root_source_file = registry_path,
    });
    registry_mod.addImport("sokol", mod_sokol);
    for (registry_import_names.items, registry_import_modules.items) |name, mod| {
        registry_mod.addImport(name, mod);
    }
    return registry_mod;
}
