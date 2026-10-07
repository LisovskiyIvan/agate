//! Scene GPU/teardown lifecycle: full `deinit`, offscreen-resize hook,
//! post/SSAO config setters, async save/load, skybox setup, and the lazy
//! MSAA forward-pipeline twin. Split out of `scene.zig` (facade).
//!
/// Anti-cycle rule (same as `audio/*`, `profiler/*`): every function takes
/// the scene as `anytype` (a `*Scene` from `core.zig` in practice) and this
/// module never imports `core.zig` or the `scene.zig` facade back.
/// Cross-leaf helpers consumed here are `pub` in their home module but are
/// deliberately NOT re-exported by the facade.
const gpu_thread = @import("../gpu_thread.zig");
const sokol = @import("sokol");
const sg = sokol.gfx;
const serialization = @import("../serialization.zig");
const CubeTexture = @import("../texture.zig").CubeTexture;
const SkyboxOptions = @import("../texture.zig").SkyboxOptions;
const PostProcessOptions = @import("../postprocess.zig").PostProcessOptions;
const SSAOOptions = @import("../ssao.zig").SSAOOptions;
const scene_forward = @import("forward_pipelines.zig");
const scene_content = @import("content.zig");

// ---- Offscreen targets & post-processing config. ----

/// Window-resize hook: resizes every viewport-sized offscreen target
/// (postprocess, bloom, glow, SSAO, outline). The per-frame post chain resizes
/// postprocess/bloom/glow lazily, but SSAO has no other resize path — call
/// this when the window size changes and post-processing/SSAO is in use.
///
/// Render-owned targets: CONTEXT THREAD ONLY (asserted), never a worker
/// or the update side — a concurrent update must never observe
/// half-resized attachments, and sokol resizes are context-bound.
pub fn resizeOffscreen(self: anytype, width: i32, height: i32) void {
    gpu_thread.assertOnContextThread();
    self.postfx.resizeAll(width, height);
}

pub fn setPostProcess(self: anytype, config: PostProcessOptions) void {
    self.post_process = config;
}

pub fn setSSAO(self: anytype, config: SSAOOptions) void {
    self.ssao = config;
}

// ---- Serialization (off-thread save/load). ----

/// Stage 3, slice 3: captures scene state under snapshot semantics (<0.1 ms)
/// and dispatches serialization + file I/O to a background TaskRunner.
/// Neither the game thread nor the sapp render thread blocks on disk I/O.
pub fn saveStateFileAsync(self: anytype, path: []const u8) !*serialization.AsyncSaveTask {
    const snap = try serialization.capture(self.allocator, self);
    errdefer {
        var s = snap;
        s.deinit(self.allocator);
    }
    const runner = if (self.io_runner) |r| r else return error.NoTaskRunner;
    return serialization.saveFileAsync(self.allocator, runner, snap, path);
}

/// Loads a scene file off-thread; caller polls task.isDone() and calls
/// restoreSceneState(scene, &task.result.?) on the game thread.
pub fn loadStateFileAsync(self: anytype, path: []const u8) !*serialization.AsyncLoadTask {
    const runner = if (self.io_runner) |r| r else return error.NoTaskRunner;
    return serialization.loadFileAsync(self.allocator, runner, path);
}

// ---- Skybox. ----

pub fn setSkybox(self: anytype, cube: CubeTexture) void {
    self.sky.setSkybox(cube);
}

pub fn createDefaultSkybox(self: anytype, config: SkyboxOptions) !void {
    try self.sky.createDefault(self.allocator, config);
}

/// Forward pipeline set matching the exact target shape (sample count +
/// color format). Returns the base `forward` set when both match the base
/// shape; otherwise returns the lazy `forward_msaa` slot, recreated when
/// the requested shape differs (it can also hold a 1x different-format
/// variant). Borrows the base family shaders. The returned pointer aliases
/// Scene state; never call mid-pass (creation destroys the previous twin
/// of that slot).
pub fn forwardFor(self: anytype, samples: i32, color_format: sg.PixelFormat) *scene_forward.ForwardPipelines {
    if (samples == self.forward.sample_count and color_format == self.forward.color_format) return &self.forward;
    if (self.forward_msaa == null or self.forward_msaa.?.sample_count != samples or self.forward_msaa.?.color_format != color_format) {
        if (self.forward_msaa) |*fw| fw.deinit();
        if (self.forward.family_shaders) |fs| {
            self.forward_msaa = scene_forward.ForwardPipelines.initWithShaders(samples, color_format, fs);
        } else {
            self.forward_msaa = scene_forward.ForwardPipelines.init(samples, color_format);
        }
    }
    return &self.forward_msaa.?;
}

pub fn deinit(self: anytype) void {
    // Live GPU teardown requires the registered context owner in every build
    // mode. Unregistered headless tools may perform CPU-only cleanup.
    gpu_thread.assertOnContextThread();
    self.profiler.deinit();
    // In-flight decodes target material fields; join them before any
    // mesh/material teardown can free those fields.
    if (self.uploads) |*q| {
        q.deinit();
        self.uploads = null;
    }
    // Join file-I/O tasks next: save tasks own their SceneState snapshot
    // and load tasks own their result, so both must finish before the
    // allocator-backed state they touch is torn down below.
    if (self.io_runner) |r| {
        r.deinit();
        self.io_runner = null;
    }
    for (self.cameras.items) |entry| {
        if (entry.owns_name) {
            self.allocator.free(entry.name);
        }
    }
    self.cameras.deinit(self.allocator);
    self.viewport_clear.deinit();

    if (self.active_camera_owned_name) |n| {
        self.allocator.free(n);
        self.active_camera_owned_name = null;
    }
    self.active_camera = null;

    self.decals.deinit();

    // Deferred off-context destroys that never reached a render-start
    // flush (queued meshes are already unlinked from `meshes`, so this
    // cannot double-free with deinitMeshes below). deinit забирает и
    // незавершённые эпохи: приложения без render не оставляют хвостов.
    self.gpu_retire.deinit(self.allocator);

    // Physics before meshes: bodies keep raw `mesh` pointers and bulk
    // teardown does not remove them individually (per-mesh destroyMesh
    // does). Destroying the world first closes that dangling window.
    self.physics.deinit(self.allocator);

    // Soft bodies before meshes: the layer frees solver + staging CPU
    // memory only; meshes/materials below (or already retired) own the
    // GPU side. Bodies must not outlive this call with mesh pointers.
    self.softbodies.deinit(self.allocator);

    scene_content.deinitMeshes(self.allocator, &self.meshes);
    scene_content.deinitPbrMaterials(self.allocator, &self.pbr_materials);
    scene_content.deinitShaderMaterials(self.allocator, &self.shader_materials);

    self.lights.deinit(self.allocator);

    self.draws.deinit(self.allocator);

    self.trails.deinit(self.allocator);
    for (self.greased_lines.items) |gl| gl.deinit();
    self.greased_lines.deinit(self.allocator);
    // sim domain (matches the sim-routed creates in sim_api): nav meshes +
    // agents, particle systems + their prepared/build frames were all
    // allocated with sim_allocator, so they must be freed with it here.
    self.nav.deinit(self.sim_allocator);

    self.default_white_texture.deinit();
    self.default_normal_texture.deinit();
    self.default_cube_texture.deinit();

    self.shadows.deinit();
    self.sky.deinit();
    self.probes.deinit();
    // render domain, CPU scratch only: rebuildCpuForSlot (view_render) grows
    // cpu_lights/headers/indices with render_allocator and this frees them.
    // upload/retireBuffers stay core: their allocator funds appends into the
    // core-owned gpu_retire queue (growth retire + add/remove retire).
    self.clustered.deinit(self.render_allocator);
    self.gui3d.deinit(self.allocator);

    self.forward.deinit();
    if (self.forward_msaa) |*fw| fw.deinit();
    self.forward_msaa = null;

    scene_content.deinitAnimations(self.allocator, &self.animation_groups, &self.skeletons);

    self.outline_meshes.deinit(self.allocator);
    self.postfx.deinit();
    self.refraction.deinit();

    // sim domain (matches createParticleSystem in sim_api).
    self.particles.deinit(self.sim_allocator);

    if (self.ui_canvas) |*u| {
        u.deinit();
    }
    // Borrowed GPU IDs only — frees the frame-owned CPU copies.
    self.ui_frame.deinit(self.allocator);
}
