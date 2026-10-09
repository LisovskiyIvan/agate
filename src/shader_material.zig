//! Shader materials: custom shaders without touching engine sources.
//!
//! Two registration levels share one runtime model:
//!
//! 1. Build-time hook materials (priority, `engine_template = true`):
//!    build.zig's `user_shader_materials` table compiles a user GLSL snippet
//!    (see shader_material/merge.zig) into a sokol-shdc module and generates
//!    the static registry (`shader_material_registry`). Pipelines use the
//!    engine's standard/pbr vertex layout and uniform contract, so the draw
//!    path just selects a different pipeline.
//!
//! 2. Runtime-registered sources (`registerRuntime`): the user supplies a
//!    `*const fn (sg.Backend) sg.Shader` that builds a FULL sg.ShaderDesc
//!    for the CURRENT backend (no cross-compilation: the source strings must
//!    be written for the backend you ship on — sokol takes backend-specific
//!    sources via sg_shader_desc.vs/fs.source). With `engine_template =
//!    false` the draw path only binds vertex/index buffers plus the
//!    material's texture at view slot 0 and skips the engine frame uniforms.
//!
//! Uniform model: every shader material owns a fixed 8x vec4 (128 byte)
//! user uniform block (see merge.zig for the GLSL side). Values are packed
//! by a declarative Param table (name -> f32 offset). Sokol UB slots follow
//! the engine contract: 0 = vs_params {mvp, model}, 1 = engine fs_params
//! (frame uniforms, engine templates only), 2 = vs_morph (engine templates
//! only), user_ub = sm_user_params (declared per material).
//!
//! GPU-free by design: pipeline creation lives in
//! scene/forward_pipelines.zig (ShaderMaterialCache); this module only
//! resolves registrations and packs uniform bytes.

const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const gpu_thread = @import("gpu_thread.zig");

// Build-time generated registry (may be empty). One-way dependency: this
// module imports it; it never imports back into agate.
const registry = @import("shader_material_registry");

/// Re-export of the build-time merge engine (hook injection + golden tests).
pub const merge = @import("shader_material/merge.zig");

/// Which engine template family a material renders with. Drives the vertex
/// layout (pbr adds the tangent attribute) and the engine uniform contract.
pub const Base = enum {
    standard,
    pbr,
};

/// One declared user uniform: an f32-aligned window inside the 32-float
/// (8x vec4) user uniform storage. Mirrors merge.Param / generated params.
pub const Param = struct {
    name: []const u8,
    /// f32 offset inside the 32-float user uniform storage.
    offset: u8,
    /// 1 (float) or 4 (vec4).
    comps: u8,
    default: [4]f32 = .{ 0, 0, 0, 0 },
};

/// Uniform slot contract constants (sokol UB slots are a shared slot pool
/// across stages; the engine templates bind 0/1/2 explicitly).
pub const vs_params_ub: u32 = 0; // {mat4 mvp, mat4 model}
pub const fs_params_ub: u32 = 1; // engine frame uniforms (engine templates)
pub const vs_morph_ub: u32 = 2; // GPU morphs (engine templates only)

/// Runtime view of one registered shader material (static or runtime).
pub const Entry = struct {
    name: []const u8,
    /// Stable cache key: Wyhash of the registration name (see keyForName).
    key: u64,
    base: Base,
    /// True when the shader was compiled from an engine template via the
    /// build-time hook merger: it declares the full engine uniform/texture
    /// contract (frame uniforms, shadow maps, GPU morphs).
    engine_template: bool = true,
    vs_ub: u32 = vs_params_ub,
    fs_ub: u32 = fs_params_ub,
    /// UB slot of the material's fs-stage sm_user_params block, when the
    /// snippet declares params.
    user_ub: ?u32 = null,
    /// Declared byte size of the fs user uniform block. The draw uploads
    /// exactly this many leading bytes of the packed storage (sokol
    /// validates the upload size against the shader's declared block).
    /// Hook blocks always declare the full 8x vec4 (128).
    user_bytes: u16 = 128,
    /// UB slot of the vs-stage sm_user_params_vs block (only when a snippet
    /// uses params inside @hook(vertex)); receives the same 128 bytes.
    vs_user_ub: ?u32 = null,
    params: []const Param = &.{},
    /// Creates (and caches) the backend sg.Shader. Must return a valid
    /// shader for sg.queryBackend().
    make_shader: *const fn (sg.Backend) sg.Shader,
};

/// Index value for "no registration resolved".
pub const invalid_index: u32 = std.math.maxInt(u32);

// ---------------------------------------------------------------------------
// Static registry (build-time hook materials).
// ---------------------------------------------------------------------------

const static_len = registry.entries.len;

/// Converts the generated registry table (its own standalone types) into the
/// runtime Entry view, at comptime.
const static_table: [static_len]Entry = blk: {
    var out: [static_len]Entry = undefined;
    for (registry.entries, 0..) |e, i| {
        out[i] = .{
            .name = e.name,
            .key = e.key,
            .base = switch (e.base) {
                .standard => .standard,
                .pbr => .pbr,
            },
            .engine_template = true,
            .vs_ub = e.vs_ub,
            .fs_ub = e.fs_ub,
            .user_ub = e.user_ub,
            .user_bytes = 128,
            .vs_user_ub = e.vs_user_ub,
            .params = convParams(e.params),
            .make_shader = e.make_shader,
        };
    }
    break :blk out;
};

fn convParams(comptime raw: []const registry.Param) []const Param {
    comptime {
        var out: [raw.len]Param = undefined;
        for (raw, 0..) |r, i| {
            out[i] = .{ .name = r.name, .offset = r.offset, .comps = r.comps, .default = r.default };
        }
        const frozen = out;
        return &frozen;
    }
}

// ---------------------------------------------------------------------------
// Runtime registry (level 2: full custom sources).
// ---------------------------------------------------------------------------

/// Hard cap for runtime registrations (no allocator, fixed slots).
pub const max_runtime_entries = 16;

/// V1 budget for EXTERNAL (runtime-registered) user uniforms: at most 2x
/// vec4 (8 f32 words) of the shared 8x vec4 user uniform storage. The
/// external path has no merge tool to pack a declarative table, so the
/// window is deliberately small: registerRuntime rejects params that do
/// not fit with error.UniformLimitExceeded (explicit, never a silent
/// clamp). Static hook materials keep the full 8-vec4 merge contract
/// (merge.user_slot_count) — unchanged.
pub const max_external_uniform_vec4: u32 = 2;
/// F32 word budget matching max_external_uniform_vec4.
pub const max_external_uniform_words: u8 = max_external_uniform_vec4 * 4;

var runtime_table: [max_runtime_entries]Entry = undefined;
pub var runtime_len: usize = 0;

/// Descriptor for a runtime-registered shader material.
///
/// EXTERNAL-SHADER CONTRACT (ShaderMaterial v1, engine_template = false):
/// the custom .glsl is written in sokol-shdc format and compiled OFFLINE
/// with the engine slang set (glsl430:metal_macos:hlsl5) through agate's
/// public build API (build.zig compileUserShader) — no runtime
/// cross-compilation, no engine source edits. The draw path renders with
/// the STANDARD rigid vertex layout (position FLOAT3, normal FLOAT3,
/// color0 FLOAT4, texcoord0 UV FLOAT2, in that attribute order — declare
/// the same prefix in the custom @vs), the standard-opaque base state
/// (depth LESS_EQUAL + write, BACK cull, CCW; blend/cull-off twins follow
/// the material's alpha_mode/double_sided like Standard), and this bind
/// contract:
///   - UB `vs_ub` (default 0) carries {mat4 mvp, mat4 model} — the custom
///     @vs must declare `layout(binding = 0) uniform vs_params` with two
///     mat4 members in that order.
///   - the material texture lands on view slot 0 (+ sampler slot 0), the
///     optional second texture (ShaderMaterial.texture1) on view/sampler
///     slot 1. sokol tolerates slots the shader does not declare.
///   - `user_ub` (when set) receives the packed user uniform bytes
///     (uniformBytes: full 128-byte storage; the shader reads its own
///     window — v1 budget is max_external_uniform_vec4).
/// Only rigid geometry: skinned meshes skip drawing (documented non-goal,
/// like hook v1); instanced meshes draw through the regular queue without
/// per-instance data (non-goal). No sg calls happen without a context:
/// make_shader runs lazily on first use from the render thread; an
/// invalid shader (id 0) makes the draw path skip the mesh.
pub const RuntimeDesc = struct {
    /// Unique registration name; the key is Wyhash(name).
    name: []const u8,
    base: Base = .standard,
    /// Builds the full sg.ShaderDesc for the current backend. The source
    /// strings must be written FOR THAT BACKEND (sokol accepts
    /// backend-specific sources; agate does not cross-compile).
    make_shader: *const fn (sg.Backend) sg.Shader,
    /// True when the custom shader follows the engine template contract
    /// (UB 0 {mvp, model}, UB 1 engine fs_params, shadow/morph textures).
    /// Runtime sources usually set false: the draw path then binds only
    /// vertex/index buffers and the material texture at view slot 0.
    engine_template: bool = false,
    /// UB slot that receives the user uniform bytes (when the custom
    /// shader declares a user block; see user_bytes).
    user_ub: ?u32 = null,
    /// Declared byte size of the custom shader's fs user uniform block
    /// (default = the full v1 window, 2x vec4). The draw uploads exactly
    /// this many leading bytes of the packed storage, so it must equal
    /// the block size in the .glsl (sokol rejects size mismatches).
    /// Must be a nonzero multiple of 16 within the v1 budget
    /// (max_external_uniform_vec4); params must fit inside
    /// user_bytes/4 words. Violations are error.UniformLimitExceeded.
    user_bytes: u16 = max_external_uniform_vec4 * 16,
    params: []const Param = &.{},
};

/// Registers a runtime shader material. Single-context-thread contract
/// (asserted): the draw resolves entries (`entry`/`entryForKey`) from render
/// while holding no lock, so a worker registering concurrently would race
/// the render's pipeline-cache lookup. Register during setup / on the
/// context thread between submissions — never from the update side or a
/// jobs worker while render is in flight.
///
/// Every declared param must fit the v1 external window
/// (offset + comps <= max_external_uniform_words); overflow is a hard
/// error.UniformLimitExceeded, never a silent clamp or truncation.
pub fn registerRuntime(desc: RuntimeDesc) error{ RegistryFull, DuplicateName, UniformLimitExceeded }!u32 {
    gpu_thread.assertOnContextThread();
    if (runtime_len >= max_runtime_entries) return error.RegistryFull;
    if (indexForName(desc.name) != null) return error.DuplicateName;
    // Wire size: nonzero vec4 multiple inside the v1 budget. The draw
    // uploads exactly user_bytes, so anything else fails sokol validation
    // at draw time — reject it here instead, with a name.
    if (desc.user_bytes == 0 or desc.user_bytes % 16 != 0 or desc.user_bytes > max_external_uniform_vec4 * 16)
        return error.UniformLimitExceeded;
    const words: u16 = desc.user_bytes / 4;
    for (desc.params) |p| {
        if (p.comps != 1 and p.comps != 4) return error.UniformLimitExceeded;
        if (@as(u16, p.offset) + p.comps > words) return error.UniformLimitExceeded;
    }
    const index: u32 = @intCast(static_len + runtime_len);
    runtime_table[runtime_len] = .{
        .name = desc.name,
        .key = keyForName(desc.name),
        .base = desc.base,
        .engine_template = desc.engine_template,
        .vs_ub = vs_params_ub,
        .fs_ub = fs_params_ub,
        .user_ub = desc.user_ub,
        .user_bytes = desc.user_bytes,
        .vs_user_ub = null,
        .params = desc.params,
        .make_shader = desc.make_shader,
    };
    runtime_len += 1;
    return index;
}

fn findInRuntime(name: []const u8) ?u32 {
    for (runtime_table[0..runtime_len], 0..) |*e, i| {
        if (std.mem.eql(u8, e.name, name)) return @intCast(static_len + i);
    }
    return null;
}

// ---------------------------------------------------------------------------
// Lookup.
// ---------------------------------------------------------------------------

pub fn entryCount() usize {
    return static_len + runtime_len;
}

/// Returns the registration for an index obtained via indexForName /
/// registerRuntime. Null for invalid_index / out of range (defensive: the
/// draw path skips such materials instead of rendering garbage).
pub fn entry(index: u32) ?*const Entry {
    if (index < static_len) return &static_table[index];
    const runtime_index = index - static_len;
    if (runtime_index < runtime_len) return &runtime_table[runtime_index];
    return null;
}

/// Stable pipeline-cache key for a registration name. Deterministic across
/// runs (Wyhash, no random seed).
pub fn keyForName(name: []const u8) u64 {
    return std.hash.Wyhash.hash(0, name);
}

/// Resolves a registration by name (static table first, then runtime).
pub fn indexForName(name: []const u8) ?u32 {
    for (&static_table, 0..) |*e, i| {
        if (std.mem.eql(u8, e.name, name)) return @intCast(i);
    }
    return findInRuntime(name);
}

/// Resolves a registration by its stable key (pipeline-cache path).
pub fn entryForKey(key: u64) ?*const Entry {
    for (&static_table) |*e| {
        if (e.key == key) return e;
    }
    for (runtime_table[0..runtime_len]) |*e| {
        if (e.key == key) return e;
    }
    return null;
}

// ---------------------------------------------------------------------------
// Uniform packing.
// ---------------------------------------------------------------------------

/// The fixed 8x vec4 (128 byte) user uniform storage owned by each material.
pub const UniformStorage = [merge.user_slot_count][4]f32;

/// Value accepted by setUniform.
pub const UniformValue = union(enum) {
    /// float param (comps == 1).
    scalar: f32,
    /// vec4 param (comps == 4).
    vector: [4]f32,
};

pub const SetUniformError = error{
    UnknownParam,
    /// Scalar write into a vec4 param or vice versa.
    CompMismatch,
};

/// Finds a declared param by name.
pub fn findParam(params: []const Param, name: []const u8) ?Param {
    for (params) |p| {
        if (std.mem.eql(u8, p.name, name)) return p;
    }
    return null;
}

/// Storage initialized from the declared param defaults (0 elsewhere).
pub fn defaultUniformStorage(params: []const Param) UniformStorage {
    var storage: UniformStorage = .{.{ 0, 0, 0, 0 }} ** merge.user_slot_count;
    for (params) |p| {
        const slot = p.offset / 4;
        switch (p.comps) {
            1 => storage[slot][p.offset % 4] = p.default[0],
            4 => storage[slot] = p.default,
            else => {},
        }
    }
    return storage;
}

/// Packs one named value into the storage (declarative table lookup).
pub fn setUniform(storage: *UniformStorage, params: []const Param, name: []const u8, value: UniformValue) SetUniformError!void {
    const p = findParam(params, name) orelse return error.UnknownParam;
    const slot = p.offset / 4;
    switch (value) {
        .scalar => |v| {
            if (p.comps != 1) return error.CompMismatch;
            storage[slot][p.offset % 4] = v;
        },
        .vector => |v| {
            if (p.comps != 4) return error.CompMismatch;
            storage[slot] = v;
        },
    }
}

/// Raw bytes for sg.applyUniforms on the entry's user UB.
pub fn uniformBytes(storage: *const UniformStorage) []const u8 {
    return std.mem.sliceAsBytes(storage);
}
