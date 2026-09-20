const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const sapp = sokol.app;
const sglue = sokol.glue;
const postprocess = @import("postprocess.zig");
const PostProcessOptions = postprocess.PostProcessOptions;
const ssao = @import("ssao.zig");
const SSAOOptions = ssao.SSAOOptions;
const particles = @import("particles.zig");
const ParticleSystem = particles.ParticleSystem;

const math = @import("math");
const Vec3 = math.Vec3;
const Mat4 = math.Mat4;
const Color3 = math.Color3;
const Color4 = math.Color4;
const BoundingBox = math.BoundingBox;
const Ray = math.Ray;

const physics = @import("physics.zig");
const PhysicsWorld = physics.PhysicsWorld;
const RigidBody = physics.RigidBody;
const PickingInfo = physics.PickingInfo;
const ColliderType = physics.ColliderType;

const ui = @import("ui.zig");
const UICanvas = ui.UICanvas;
const audio = @import("audio.zig");

const AnimationGroup = @import("animation/animation.zig").AnimationGroup;
const Skeleton = @import("animation/skeleton.zig").Skeleton;

const camera_mod = @import("camera.zig");
const Camera = camera_mod.Camera;
const Viewport = camera_mod.Viewport;
const debug_shd = @import("debug_shader");
const debug_pass = @import("passes/debug_pass.zig");
const outline_pass = @import("passes/outline_pass.zig");
const lights = @import("lights.zig");
const HemisphericLight = lights.HemisphericLight;
const DirectionalLight = lights.DirectionalLight;
const DirectionalLightOptions = lights.DirectionalLightOptions;
const PointLight = lights.PointLight;
const PointLightOptions = lights.PointLightOptions;
const SpotLight = lights.SpotLight;
const SpotLightOptions = lights.SpotLightOptions;
const Mesh = @import("mesh.zig").Mesh;
const StagedInstanceRecord = @import("mesh.zig").StagedInstanceRecord;
const decal_mod = @import("mesh/decal.zig");
const DecalManager = decal_mod.DecalManager;
const trail_mod = @import("mesh/trail.zig");
const TrailMesh = trail_mod.TrailMesh;
const TrailOptions = trail_mod.TrailOptions;
const csg_mod = @import("mesh/csg.zig");
const greased_mod = @import("mesh/greased_line.zig");
const GreasedLineOptions = greased_mod.GreasedLineOptions;
const GreasedLineMesh = greased_mod.GreasedLineMesh;
const simplify_mod = @import("mesh/simplify.zig");
const SimplifyOptions = simplify_mod.SimplifyOptions;
const LODLevelSpec = simplify_mod.LODLevelSpec;
const ai_mod = @import("ai.zig");
const NavMesh = ai_mod.NavMesh;
const NavAgent = ai_mod.NavAgent;
const StandardMaterial = @import("material.zig").StandardMaterial;
const PBRMaterial = @import("material.zig").PBRMaterial;
const ShaderMaterial = @import("material.zig").ShaderMaterial;
const Texture = @import("texture.zig").Texture;
const CubeTexture = @import("texture.zig").CubeTexture;
const SkyboxOptions = @import("texture.zig").SkyboxOptions;
const visibility = @import("visibility/mod.zig");
const serialization = @import("serialization.zig");

// Scene subsystems. Each owns its state (and GPU resources) plus the logic
// that belongs to it; Scene is the owner/orchestrator facade. Subsystems
// never import scene.zig — Scene passes everything they need as parameters.
const scene_stats = @import("scene/stats.zig");
pub const SceneStats = scene_stats.SceneStats;
const scene_render_queue = @import("scene/render_queue.zig");
const scene_instance_staging = @import("scene/instance_staging.zig");
const jobs = @import("jobs.zig");
const assets_mod = @import("assets.zig");
const handoff_mod = @import("handoff.zig");
const gpu_thread = @import("gpu_thread.zig");
const upload_meter = @import("gpu_upload_meter.zig");
pub const RenderMeshItem = scene_render_queue.RenderMeshItem;
const scene_lights = @import("scene/light_rig.zig");
const scene_shadow = @import("scene/shadow_system.zig");
const scene_sky = @import("scene/sky_layer.zig");
const scene_postfx = @import("scene/postfx_stack.zig");
const scene_forward = @import("scene/forward_pipelines.zig");
const scene_particles = @import("scene/particle_layer.zig");
const scene_decals = @import("scene/decal_layer.zig");
const scene_trails = @import("scene/trail_layer.zig");
const scene_nav = @import("scene/nav_layer.zig");
const scene_physics = @import("scene/physics_layer.zig");
const scene_project = @import("scene/project_cache.zig");
const scene_retire = @import("scene/gpu_retire.zig");
const scene_ui_frame = @import("scene/ui_frame.zig");
pub const UiFrame = scene_ui_frame.UiFrame;
const scene_frame_draws = @import("scene/frame_draws.zig");
/// P7 published consumable draw payload (one coherent prepared frame).
pub const FrameDrawSlot = scene_frame_draws.FrameDrawSlot;
/// P7 two retained owning queue slots (the variable-list double buffer).
pub const FrameDraws = scene_frame_draws.FrameDraws;
const scene_content = @import("scene/content.zig");
const scene_animation = @import("scene/animation_runtime.zig");
const scene_picking = @import("scene/picking.zig");
const scene_uniforms = @import("scene/uniforms.zig");
const FrameContext = scene_uniforms.FrameContext;
const scene_draw = @import("scene/draw.zig");
const scene_msaa = @import("scene/msaa.zig");
pub const scene_snapshot = @import("scene/snapshot.zig");
pub const SceneFrameSnapshot = scene_snapshot.SceneFrameSnapshot;
pub const CameraSnapshot = scene_snapshot.CameraSnapshot;
pub const profiler_mod = @import("profiler.zig");
pub const Profiler = profiler_mod.Profiler;
pub const FrameRecord = profiler_mod.FrameRecord;
pub const MemorySnapshot = profiler_mod.MemorySnapshot;

pub const CameraEntry = struct {
    name: []const u8,
    camera: Camera,
    owns_name: bool = false,
    enabled: bool = true,
    culling_mask: u32 = 0xFFFFFFFF,
    viewport: Viewport = .{},
    clear_viewport: bool = true,
    clear_color: ?Color4 = null,
};

/// Milliseconds elapsed since a `sokol.time.now()` tick. Cheap, no
/// allocation; used for the SceneStats phase timings.
fn msSince(t0: u64) f32 {
    return @floatCast(sokol.time.ms(sokol.time.now() -% t0));
}

/// The scene: content registries (flat, iterated directly by loaders,
/// tooling and serialization), a handful of cross-cutting config flags, and
/// one field per render subsystem. New features should add state to the
/// matching subsystem in scene/, not to this struct.
///
/// Threading body contract (actual update||render): the Scene allocator
/// MUST be thread-safe (GPA `.thread_safe = true` — prepare/render lazy
/// caches and jobs workers allocate from it). The frame snapshot and every
/// render pass / render-owned cache live EXCLUSIVELY on the context thread
/// (prepare + render, sequential); workers and the update side never touch
/// them. No raw `sg.destroy*` from workers or the update side — all GPU
/// teardown funnels through `gpu_retire` (context-thread flush) or deinit.
/// The `light_pack` consumed-copy fallback in `updateLights` stays
/// game-owned under update-vs-prepare exclusion (render reads only the
/// snapshot copy) — no atomic mailbox needed there.
pub const Scene = struct {
    allocator: std.mem.Allocator,

    // ---- Content registries (kept flat: external code iterates them). ----
    meshes: std.ArrayListUnmanaged(*Mesh) = .empty,
    /// P3: очередь ретенции GPU-мешей с epoch-семантикой
    /// (scene/gpu_retire.zig). destroyMesh вне context-потока только отвязывает
    /// меш и кладёт его сюда (retireMesh — с любого потока); уничтожает
    /// (sg.* + free) только context-поток во flush (начало кадра) и в deinit.
    /// Уже отвязанные меши невидимы для deinitMeshes — двойного free нет.
    /// Tripwire P6: новые kind'ы записей (не только меши/буферы) добавлять в
    /// GpuRetireQueue, новых очередей в Scene не заводить. (P5 — instance
    /// buffer payload — уже там: см. retireBuffer.)
    gpu_retire: scene_retire.GpuRetireQueue = .{},
    /// Epoch, начатый последним prepareFrame. render завершает его на ВСЕХ
    /// выходах (включая ранний возврат без камеры), поэтому epoch — на кадр,
    /// а не на камеру/view.
    retire_epoch: scene_retire.Epoch = 0,
    materials: std.ArrayListUnmanaged(*StandardMaterial) = .empty,
    pbr_materials: std.ArrayListUnmanaged(*PBRMaterial) = .empty,
    shader_materials: std.ArrayListUnmanaged(*ShaderMaterial) = .empty,
    animation_groups: std.ArrayListUnmanaged(*AnimationGroup) = .empty,
    skeletons: std.ArrayListUnmanaged(*Skeleton) = .empty,

    // ---- Camera management ----
    cameras: std.ArrayListUnmanaged(CameraEntry) = .empty,
    active_camera_index: ?usize = null,
    enable_multi_camera: bool = false,
    clear_pipeline: sg.Pipeline = .{},
    clear_pipeline_msaa: sg.Pipeline = .{},
    clear_shader: sg.Shader = .{},
    clear_vb: sg.Buffer = .{},

    active_camera: ?Camera = null,
    active_camera_owned_name: ?[]const u8 = null,
    clear_color: Color4 = Color4.new(0.12, 0.14, 0.18, 1.0),
    default_material: StandardMaterial = StandardMaterial.init("default"),
    default_white_texture: Texture,
    default_normal_texture: Texture,
    default_cube_texture: CubeTexture,

    // ---- Cross-cutting culling config and per-frame stats. ----
    enable_frustum_culling: bool = true,
    enable_occlusion_culling: bool = true,
    occlusion_culler: visibility.OcclusionCuller = visibility.OcclusionCuller.init(),
    /// Context-owned per-frame counters/timings (see scene/stats.zig).
    /// Update-side code NEVER writes here directly (render reads it
    /// concurrently with update); the update tick arrives via
    /// `pending_update_ms` + `recordUpdateTime` and is transferred by
    /// prepareFrame.
    stats: SceneStats = .{},
    /// Staged update-phase timing: the game side writes ONLY this field via
    /// `recordUpdateTime` (under update-vs-prepare exclusion); prepareFrame
    /// transfers the last tick into `stats.update_ms`. Separate word from
    /// every stats field, so update||render shares no memory here.
    pending_update_ms: f32 = 0,
    // Bumped once per render(); Mesh.cached_* entries tagged with this are fresh.
    frame_id: u64 = 0,

    // ---- Render subsystems. ----
    // All lights: hemispheric sun, optional directional sun, point/spot lists.
    lights: scene_lights.LightRig,
    // CSM shadow state + GPU shadow depth pass.
    shadows: scene_shadow.ShadowSystem,
    // Skybox texture/exposure + IBL intensity + skybox pass.
    sky: scene_sky.SkyboxLayer,
    // Offscreen target, SSAO, bloom, composite pass + outline highlights.
    postfx: scene_postfx.PostFXStack,
    // Forward GPU pipeline sets (opaque/blend/double-sided per family).
    forward: scene_forward.ForwardPipelines,
    // Particle systems + billboard pass.
    particles: scene_particles.ParticleLayer,
    /// Stage 3, slice 2: light selection + packing (including the
    /// incumbency-hysteresis fade simulation) runs in `updateLights`
    /// during the update phase and publishes through this mailbox; render
    /// takes the newest pack at frame start. Plain data end to end, so
    /// the group is already thread-ready for the game/render split.
    light_handoff: handoff_mod.Handoff(scene_lights.LightRig.FramePack, 2) = .{},
    /// Last consumed pack — the fallback render uses when no new light
    /// pack was published since the previous frame (e.g. PIP: several
    /// render calls per one update).
    light_pack: scene_lights.LightRig.FramePack = std.mem.zeroes(scene_lights.LightRig.FramePack),
    /// Async texture decode/upload pipeline. Drained at the top of
    /// render(); deinit'd FIRST in deinit so in-flight decodes finish
    /// before any material they target is freed. Null = synchronous loads.
    uploads: ?assets_mod.UploadQueue = null,
    /// Dedicated file-I/O runner for async save/load (saveStateFileAsync /
    /// loadStateFileAsync). Decode work stays on `uploads.runner`, so long
    /// disk I/O can never starve texture decodes. 1 thread: file ops are
    /// serial by nature. Null = NoTaskRunner, same fallback as uploads.
    io_runner: ?*jobs.TaskRunner = null,
    /// Last prepareFrame texture-upload tally, published into
    /// `stats.uploaded_*_frame` at render start (prepareFrame runs before
    /// the per-frame stats reset, so the tally is staged here first).
    frame_uploads: assets_mod.UploadQueue.DrainResult = .{},
    // Dynamic decals.
    decals: scene_decals.DecalLayer = .{},
    // Trail meshes.
    trails: scene_trails.TrailLayer = .{},
    // Greased line meshes.
    greased_lines: std.ArrayListUnmanaged(*GreasedLineMesh) = .empty,
    // Nav meshes and agents.
    nav: scene_nav.NavLayer = .{},
    // Physics world + debug wireframe overlay.
    physics: scene_physics.PhysicsIntegration = .{},
    // Per-frame draw queues + instance staging.
    // P7 double buffer: two retained owning slots encompassing PRIMARY +
    // ALL PIP view queues, outline items+skins, and prepared shadow
    // items+skins+bin ranges. prepare builds the back slot, render reads the
    // published front slot via preparedDraws() — the ONLY low-level draw
    // accessor (no legacy field aliases). See scene/frame_draws.zig for the
    // ownership/lifecycle contract.
    draws: scene_frame_draws.FrameDraws = .{},
    // projectPoint view-projection cache.
    project: scene_project.ProjectCache = .{},

    // Public post-processing configs. Flat on purpose: sandbox tooling reads
    // and writes these two directly ~100 times; the stack that consumes them
    // lives in `postfx`.
    post_process: PostProcessOptions = .{},
    ssao: SSAOOptions = .{},

    // Requested MSAA sample count for the offscreen main render target
    // (PASS 2). 1 = off (default). Effective count = scene/msaa.zig policy:
    // clamped per backend, applied only when the post chain owns the main
    // pass, 1x on the swapchain path. While active it suppresses the
    // depth-consuming post effects (SSAO/SSR/DoF) — sokol has no depth
    // resolve, so an MSAA depth attachment cannot feed them (warned once).
    // Set before the first render(); changing it later recreates the MSAA
    // pipeline/target twins on the next MSAA frame.
    msaa_sample_count: i32 = 1,

    // Forward pipeline twin built for the effective MSAA sample count
    // (sokol requires pipeline.sample_count to match every main-target
    // attachment). Null until the first MSAA frame.
    forward_msaa: ?scene_forward.ForwardPipelines = null,
    warn_msaa_format: scene_msaa.WarnOnce = .{},

    // Highlighted meshes for the inverse-hull outline (postfx holds the
    // settings + pass). Kept flat: mock scenes in mesh tests construct it.
    // The prepared outline items+skins live in the P7 slots (preparedDraws).
    outline_meshes: std.ArrayListUnmanaged(*Mesh) = .empty,

    /// Mailbox for publishing frame-level camera/light/pass state from the
    /// simulation thread. Meshes and materials are still live scene objects,
    /// so threaded applications must keep update-vs-prepare exclusion
    /// (producer update, consumer prepare); update may overlap render.
    frame_handoff: handoff_mod.Handoff(scene_snapshot.SceneFrameSnapshot, 2) = .{},
    /// Last consumed frame snapshot (context/prepare + render ownership:
    /// prepare writes it, render reads it; update/build NEVER touch it, so a
    /// concurrent update cannot race the in-flight draw).
    frame_snapshot: scene_snapshot.SceneFrameSnapshot = .{},
    /// Producer-owned build snapshot (game/update phase ownership): the exact
    /// generation the last `buildPreparedFrame` culled/built queues against.
    /// Plain value camera/pass state (Camera.name slices alias, but draw
    /// never dereferences names — no deep copy). The latch copies it into
    /// `frame_snapshot` verbatim; render never reads this.
    build_snapshot: scene_snapshot.SceneFrameSnapshot = .{},
    /// Flag indicating whether prepareFrame() has already run for this frame.
    frame_prepared: bool = false,
    /// Reuse guard owned by `renderReuse` on the context thread (set around
    /// the inner `render()` call, never observed concurrently): tells render
    /// to skip the `prepareFrame` fallback and the profiler `recordFrame`
    /// tail, so the already-consumed front is re-drawn as-is.
    rendering_reuse: bool = false,
    /// Monotonic presented-frame counter, written only on the context thread:
    /// every presented frame (normal renders and `renderReuse` re-presents
    /// alike) bumps it once and uses it as the profiler `recordFrame`
    /// frame_id label, so wall-pacing stats see consecutive presents with no
    /// gaps. Re-presented rows repeat the consumed frame's counters while
    /// pacing/timestamps are real.
    profiler_frame_seq: u64 = 0,

    /// Stage 1 producer-build handoff (game/update phase → prepare latch):
    /// `buildPreparedFrame` (game side, CPU-only, sg-free) stages instance
    /// matrices into the back-slot scratch + per-mesh previews, freezes the
    /// slot-owned staged records, and captures
    /// the particle/physics CPU build frames, then bumps `build_seq`.
    /// `prepareFrame` (context side) consumes the build when `build_seq !=
    /// last_latched_seq` (GPU halves over the slot records + latch copies,
    /// no CPU restaging, no live preview reads) and
    /// otherwise runs the historical inline path — so behavior stays correct
    /// when `buildPreparedFrame` is never called. Two builds before a latch:
    /// newest wins (single preview store recomputed, scratch overwritten,
    /// record array reset and refilled).
    /// A missed build never resurfaces a stale frame: every prepare (latch
    /// or fallback) publishes a fresh slot; render reuses the last published
    /// front only when prepare itself is not called (unchanged).
    /// Plain counters, no atomics (update-vs-prepare exclusion); the build
    /// touches no stats/profiler/frame_id/retire_epoch.
    build_seq: u64 = 0,
    last_latched_seq: u64 = 0,
    /// Back-slot index the last `buildPreparedFrame` wrote; the latch
    /// asserts it still is the back index (no intervening publish).
    build_slot: usize = 0,
    /// Game-owned queue-build stats (stage-2 increment B): `buildPreparedFrame`
    /// clears this and the game-side `buildQueuesInto` accumulates the queue
    /// counters here (never `self.stats`, which stays context-owned). The
    /// latch merges it into `self.stats` via `SceneStats.mergeFrom` (after
    /// the latch's own reset) and resets it to `.{}`. Plain struct, phase
    /// ownership (game write, context merge), never observed concurrently.
    build_stats: SceneStats = .{},

    // 2D & 3D UI canvas (lazy; created via createUI()).
    ui_canvas: ?UICanvas = null,
    /// P6 render-owned UI frame: prepareFrame captures CPU geometry + draw
    /// params out of `ui_canvas` and uploads at the prepare/context
    /// boundary; render draws this frame (upload-free), never the live
    /// canvas. Single-frame Scene ownership, no registry, no P7 overlap.
    ui_frame: UiFrame = .{},

    // Built-in flight recorder & memory profiler.
    profiler: profiler_mod.Profiler,

    // Frame uniform types shared with the draw path (see scene/uniforms.zig).

    pub fn initInto(self: *Scene, allocator: std.mem.Allocator) void {
        self.* = Scene{
            .allocator = allocator,
            .profiler = profiler_mod.Profiler.init(allocator),
            .default_white_texture = Texture.createWhite1x1(),
            .default_normal_texture = Texture.createFlatNormal1x1(),
            .default_cube_texture = CubeTexture.createDefault1x1(.{ 25, 30, 40, 255 }),
            .lights = scene_lights.LightRig.init("hemi", .{
                .direction = math.Vec3.new(0.5, 1.0, 0.3),
                .diffuse = Color3.white,
                .ground_color = Color3.new(0.2, 0.25, 0.3),
                .intensity = 1.0,
            }),
            .shadows = scene_shadow.ShadowSystem.init(allocator),
            .sky = scene_sky.SkyboxLayer.init(),
            .postfx = scene_postfx.PostFXStack.init(),
            .forward = scene_forward.ForwardPipelines.init(),
            .particles = scene_particles.ParticleLayer.init(),
        };
        // Async texture decode/uploads (stage 2): failures degrade to a
        // null queue and all loads take the synchronous path.
        self.uploads = assets_mod.UploadQueue.init(allocator, 2) catch null;
        // Dedicated file-I/O runner (1 thread): async save/load must not
        // share the decode runner, or long file I/O starves decodes.
        self.io_runner = jobs.TaskRunner.init(allocator, 1) catch null;
    }

    pub fn init(allocator: std.mem.Allocator) Scene {
        var self: Scene = undefined;
        self.initInto(allocator);
        return self;
    }

    // ---- Offscreen targets & post-processing config. ----

    /// Window-resize hook: resizes every viewport-sized offscreen target
    /// (postprocess, bloom, SSAO, outline). The per-frame post chain resizes
    /// postprocess/bloom lazily, but SSAO has no other resize path — call
    /// this when the window size changes and post-processing/SSAO is in use.
    ///
    /// Render-owned targets: CONTEXT THREAD ONLY (asserted), never a worker
    /// or the update side — a concurrent update must never observe
    /// half-resized attachments, and sokol resizes are context-bound.
    pub fn resizeOffscreen(self: *Scene, width: i32, height: i32) void {
        gpu_thread.assertOnContextThread();
        self.postfx.resizeAll(width, height);
    }

    pub fn setPostProcess(self: *Scene, config: PostProcessOptions) void {
        self.post_process = config;
    }

    pub fn setSSAO(self: *Scene, config: SSAOOptions) void {
        self.ssao = config;
    }

    // ---- Serialization (off-thread save/load). ----

    /// Stage 3, slice 3: captures scene state under snapshot semantics (<0.1 ms)
    /// and dispatches serialization + file I/O to a background TaskRunner.
    /// Neither the game thread nor the sapp render thread blocks on disk I/O.
    pub fn saveStateFileAsync(self: *Scene, path: []const u8) !*serialization.AsyncSaveTask {
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
    pub fn loadStateFileAsync(self: *Scene, path: []const u8) !*serialization.AsyncLoadTask {
        const runner = if (self.io_runner) |r| r else return error.NoTaskRunner;
        return serialization.loadFileAsync(self.allocator, runner, path);
    }

    // ---- Skybox. ----

    pub fn setSkybox(self: *Scene, cube: CubeTexture) void {
        self.sky.setSkybox(cube);
    }

    pub fn createDefaultSkybox(self: *Scene, config: SkyboxOptions) !void {
        try self.sky.createDefault(self.allocator, config);
    }

    // ---- Lights. ----

    pub fn createHemisphericLight(self: *Scene, name: []const u8, options: lights.HemisphericLightOptions) HemisphericLight {
        return self.lights.setHemispheric(name, options);
    }

    pub fn createPointLight(self: *Scene, name: []const u8, options: PointLightOptions) !*PointLight {
        return self.lights.createPointLight(self.allocator, name, options);
    }

    pub fn createSpotLight(self: *Scene, name: []const u8, options: SpotLightOptions) !*SpotLight {
        return self.lights.createSpotLight(self.allocator, name, options);
    }

    pub fn createDirectionalLight(self: *Scene, name: []const u8, options: DirectionalLightOptions) !*DirectionalLight {
        return self.lights.createDirectionalLight(self.allocator, name, options);
    }

    /// Appends a shadowless directional fill (uniform slots 1..3, creation
    /// order). Hard-errors past 4 suns total (see
    /// LightRig.addDirectionalLight); fills are session-local.
    pub fn addDirectionalLight(self: *Scene, name: []const u8, options: DirectionalLightOptions) !*DirectionalLight {
        return self.lights.addDirectionalLight(self.allocator, name, options);
    }

    // ---- Content registries: materials & meshes. ----

    pub fn createStandardMaterial(self: *Scene, name: []const u8) !*StandardMaterial {
        const mat = try self.allocator.create(StandardMaterial);
        mat.* = StandardMaterial.init(name);
        try self.materials.append(self.allocator, mat);
        return mat;
    }

    pub fn createPBRMaterial(self: *Scene, name: []const u8) !*PBRMaterial {
        const mat = try self.allocator.create(PBRMaterial);
        mat.* = PBRMaterial.init(name);
        try self.pbr_materials.append(self.allocator, mat);
        return mat;
    }

    // Creates a custom-shader material bound to a registered shader (by
    // name — build.zig `user_shader_materials` or registerRuntime). Returns
    // null when the shader name is not registered.
    pub fn createShaderMaterial(self: *Scene, name: []const u8, shader_name: []const u8) ?*ShaderMaterial {
        const mat = self.allocator.create(ShaderMaterial) catch return null;
        mat.* = ShaderMaterial.initForShader(shader_name, name) orelse {
            self.allocator.destroy(mat);
            return null;
        };
        self.shader_materials.append(self.allocator, mat) catch {
            self.allocator.destroy(mat);
            return null;
        };
        return mat;
    }

    /// Stages the update-phase wall time measured by the app around
    /// Scene.update. Update-side write (game thread, under update-vs-prepare
    /// phase ownership); prepareFrame transfers the last tick into
    /// stats.update_ms. This is the ONLY update-side timing write — direct
    /// `scene.stats.*` writes from the update thread are forbidden (stats is
    /// context-owned; render reads it concurrently with update).
    pub fn recordUpdateTime(self: *Scene, ms: f32) void {
        self.pending_update_ms = ms;
    }

    /// Unlinks `mat` from the registry and frees the CPU material. Safe under
    /// update||render WITHOUT any GPU retire: prepared draw records carry
    /// GPU handle VALUES (views/samplers), never CPU material refs, and
    /// materials own no GPU objects needing deinit — the in-flight frame's
    /// baked copies stay valid. (No audit-driven retire-all-materials/textures
    /// queue: that would be a false-positive fix for a non-issue.)
    pub fn destroyPBRMaterial(self: *Scene, mat: *PBRMaterial) void {
        for (self.pbr_materials.items, 0..) |m, i| {
            if (m == mat) {
                _ = self.pbr_materials.swapRemove(i);
                break;
            }
        }
        self.allocator.destroy(mat);
    }

    pub fn removeMesh(self: *Scene, mesh: *Mesh) bool {
        for (self.meshes.items, 0..) |m, i| {
            if (m == mesh) {
                _ = self.meshes.swapRemove(i);
                return true;
            }
        }
        return false;
    }

    pub fn destroyMesh(self: *Scene, mesh: *Mesh) void {
        _ = self.removeMesh(mesh);
        for (self.outline_meshes.items, 0..) |m, i| {
            if (m == mesh) {
                _ = self.outline_meshes.swapRemove(i);
                break;
            }
        }
        // Referent cleanup: neutralize every cross-mesh reference to `mesh`
        // before its storage is freed. Runs before the sync/deferred branch
        // below so both paths are covered. Never cascade-destroys: orphaned
        // meshes stay alive under their own transform.
        // Physics: drop the rigid body bound to this mesh, if any.
        if (self.physics.getWorld()) |pw| {
            if (pw.findBody(mesh)) |body| pw.removeBody(body);
        }
        // Hierarchy: orphan children and detach bone attachments hosted by
        // the destroyed mesh (this also covers decal meshes parented to a
        // destroyed target via createDecal).
        for (self.meshes.items) |child| {
            if (child.parent == mesh) child.parent = null;
            if (child.attach_bone) |ab| {
                if (ab.host_mesh == mesh) child.detachFromBone();
            }
        }
        // LOD: order-preserving removal keeps the distance-sorted band
        // order; the parent renders its own geometry for the freed band
        // instead of holding a dangling pointer.
        for (self.meshes.items) |other| {
            var i: usize = 0;
            while (i < other.lod_levels.items.len) {
                if (other.lod_levels.items[i].mesh == mesh) {
                    _ = other.lod_levels.orderedRemove(i);
                } else {
                    i += 1;
                }
            }
        }
        // Animation groups: morph targets bind slices INTO the mesh (weights
        // slice + the dirty flag), unlike node transforms whose pointers are
        // scene-owned. Tombstone matching targets instead of removing them so
        // NodeChannel.target indices stay valid (applyNodesAtTime skips empty
        // weight slices) and free the group-owned rest snapshot.
        for (self.animation_groups.items) |ag| {
            for (ag.morph_targets) |*mt| {
                const binds_mesh = mt.dirty == &mesh.morph_dirty or
                    (mt.weights.len > 0 and mt.weights.ptr == mesh.morph_weights.ptr);
                if (!binds_mesh) continue;
                // rest_weights belongs to the group's allocator, not the
                // scene's; today they are the same instance.
                if (mt.rest_weights.len > 0) ag.allocator.free(mt.rest_weights);
                mt.weights = &.{};
                mt.rest_weights = &.{};
                mt.dirty = null;
            }
        }
        // Decals: drop manager instances of this mesh and free their
        // material so neither dangles nor leaks. Safe against DecalManager
        // iteration: update/destroyOldest/clear remove the instance BEFORE
        // calling destroyMesh, so this scan only mutates the list on
        // user-initiated destroys where no manager iteration is in flight.
        if (self.decals.manager) |*dm| {
            var i: usize = 0;
            while (i < dm.instances.items.len) {
                if (dm.instances.items[i].mesh == mesh) {
                    const removed = dm.instances.orderedRemove(i);
                    self.destroyPBRMaterial(removed.material);
                } else {
                    i += 1;
                }
            }
        }
        // Decal expiration calls this from Scene.update on the game thread,
        // where sg.destroyBuffer is illegal: unlink now, destroy the GPU
        // resources at the next render-start flush on the context thread
        // (epoch-ретенция: запись ждёт завершения текущего кадра).
        if (!gpu_thread.isOnContextThread()) {
            self.gpu_retire.retireMesh(self.allocator, mesh);
            return;
        }
        mesh.deinit(self.allocator);
        self.allocator.destroy(mesh);
    }

    // ---- Decals / particles / trails / CSG / nav. ----

    pub fn getOrCreateDecalManager(self: *Scene, max_decals: usize) *DecalManager {
        return self.decals.getOrCreate(self, max_decals);
    }

    pub fn updateDecals(self: *Scene, dt: f32) void {
        self.decals.update(dt);
    }

    pub fn createParticleSystem(self: *Scene, name: []const u8, capacity: usize) !*ParticleSystem {
        return self.particles.create(self.allocator, name, capacity);
    }

    /// Explicit particle stepping: GPU simulation modes either run or return
    /// an error (particles.UpdateError) — never a silent CPU downgrade.
    pub fn updateParticles(self: *Scene, dt: f32) particles.UpdateError!void {
        try self.particles.update(dt);
    }

    pub fn createTrailMesh(self: *Scene, name: []const u8, options: TrailOptions) !*TrailMesh {
        return self.trails.create(self, self.allocator, name, options);
    }

    pub fn updateTrails(self: *Scene, dt: f32) void {
        const cam_pos = if (self.active_camera) |cam| cam.getPosition() else Vec3.zero;
        self.trails.update(dt, cam_pos);
    }

    pub fn createCSGMesh(self: *Scene, name: []const u8, csg_solid: *const csg_mod.CSG) !*Mesh {
        return csg_solid.toMesh(self, name);
    }

    pub fn createGreasedLine(self: *Scene, name: []const u8, options: GreasedLineOptions) !*Mesh {
        var data = try greased_mod.buildGreasedLineData(self.allocator, options);
        defer data.deinit(self.allocator);
        return @import("mesh/mesh.zig").uploadGeometry(self, name, data);
    }

    pub fn createGreasedLineMesh(self: *Scene, name: []const u8, options: GreasedLineOptions) !*GreasedLineMesh {
        const gl = try GreasedLineMesh.init(self, name, options);
        try self.greased_lines.append(self.allocator, gl);
        return gl;
    }

    pub fn simplifyMesh(self: *Scene, name: []const u8, source_mesh: *Mesh, options: SimplifyOptions) !*Mesh {
        return simplify_mod.simplifyMesh(self.allocator, self, name, source_mesh, options);
    }

    pub fn generateLODLevels(self: *Scene, source_mesh: *Mesh, specs: []const LODLevelSpec) !void {
        return simplify_mod.generateLODLevels(self.allocator, self, source_mesh, specs);
    }

    pub fn createNavMeshFromTriangles(
        self: *Scene,
        positions: []const [3]f32,
        indices: []const u32,
        max_slope_rad: f32,
    ) !*ai_mod.NavMesh {
        return self.nav.createMeshFromTriangles(self.allocator, positions, indices, max_slope_rad);
    }

    pub fn createNavMeshGrid(
        self: *Scene,
        min_x: f32,
        max_x: f32,
        min_z: f32,
        max_z: f32,
        elevation_y: f32,
        subdiv_x: usize,
        subdiv_z: usize,
        obstacles: []const BoundingBox,
    ) !*ai_mod.NavMesh {
        return self.nav.createMeshGrid(self.allocator, min_x, max_x, min_z, max_z, elevation_y, subdiv_x, subdiv_z, obstacles);
    }

    pub fn createNavAgent(self: *Scene, nav_mesh: *const ai_mod.NavMesh, start_pos: Vec3) !*ai_mod.NavAgent {
        return self.nav.createAgent(self.allocator, nav_mesh, start_pos);
    }

    pub fn updateNavAgents(self: *Scene, dt: f32) void {
        self.nav.updateAgents(dt);
    }

    // ---- Animation / physics / camera / picking / UI / projection. ----

    pub fn updateAnimations(self: *Scene, dt: f32) void {
        scene_animation.updateAnimations(self.animation_groups.items, self.skeletons.items, self.meshes.items, dt);
    }

    pub fn enablePhysics(self: *Scene, gravity: ?Vec3) *PhysicsWorld {
        return self.physics.enable(self.allocator, gravity);
    }

    /// Finds the rigid body previously created for `mesh`, if any.
    pub fn getRigidBody(self: *Scene, mesh: *const Mesh) ?*RigidBody {
        if (self.physics.getWorld()) |pw| {
            return pw.findBody(mesh);
        }
        return null;
    }

    pub fn createRigidBody(self: *Scene, mesh: *Mesh, collider: ColliderType, mass: f32) !*RigidBody {
        const pw = self.physics.getWorld() orelse self.physics.enable(self.allocator, null);
        return pw.createBody(mesh, collider, mass);
    }

    /// Creates a rigid body with collision filter / sensor / event options.
    pub fn createRigidBodyWith(self: *Scene, mesh: *Mesh, collider: ColliderType, mass: f32, options: physics.BodyOptions) !*RigidBody {
        const pw = self.physics.getWorld() orelse self.physics.enable(self.allocator, null);
        return pw.createBodyWith(mesh, collider, mass, options);
    }

    pub fn updatePhysics(self: *Scene, dt: f32) void {
        self.physics.step(dt);
    }

    // ---- Camera management ----

    pub fn addCamera(self: *Scene, entry: CameraEntry) !usize {
        var e = entry;
        if (e.culling_mask == 0xFFFFFFFF and e.camera.getCullingMask() != 0xFFFFFFFF) {
            e.culling_mask = e.camera.getCullingMask();
        }
        const cam_vp = e.camera.getViewport();
        if (e.viewport.x == 0.0 and e.viewport.y == 0.0 and e.viewport.width == 1.0 and e.viewport.height == 1.0 and
            (cam_vp.x != 0.0 or cam_vp.y != 0.0 or cam_vp.width != 1.0 or cam_vp.height != 1.0))
        {
            e.viewport = cam_vp;
        }
        const idx = self.cameras.items.len;
        try self.cameras.append(self.allocator, e);
        if (self.active_camera_index == null) {
            self.active_camera_index = idx;
            self.active_camera = e.camera;
        }
        return idx;
    }

    pub fn removeCamera(self: *Scene, index: usize) void {
        if (index >= self.cameras.items.len) return;
        const entry = self.cameras.orderedRemove(index);
        if (entry.owns_name) {
            self.allocator.free(entry.name);
        }
        if (self.cameras.items.len == 0) {
            self.active_camera_index = null;
            self.active_camera = null;
        } else {
            const cur_idx = self.active_camera_index orelse 0;
            if (cur_idx >= self.cameras.items.len) {
                self.switchCamera(self.cameras.items.len - 1);
            } else {
                self.switchCamera(cur_idx);
            }
        }
    }

    pub fn getCamera(self: *Scene, index: usize) ?*CameraEntry {
        if (index < self.cameras.items.len) return &self.cameras.items[index];
        return null;
    }

    pub fn getCameraByName(self: *Scene, name: []const u8) ?*CameraEntry {
        for (self.cameras.items) |*entry| {
            if (std.mem.eql(u8, entry.name, name)) return entry;
        }
        return null;
    }

    pub fn switchCamera(self: *Scene, index: usize) void {
        if (index >= self.cameras.items.len) return;
        if (self.active_camera_index) |curr_idx| {
            if (curr_idx < self.cameras.items.len and self.active_camera != null) {
                self.cameras.items[curr_idx].camera = self.active_camera.?;
            }
        }
        self.active_camera_index = index;
        self.active_camera = self.cameras.items[index].camera;
    }

    pub fn switchCameraByName(self: *Scene, name: []const u8) bool {
        for (self.cameras.items, 0..) |entry, i| {
            if (std.mem.eql(u8, entry.name, name)) {
                self.switchCamera(i);
                return true;
            }
        }
        return false;
    }

    pub fn nextCamera(self: *Scene) void {
        if (self.cameras.items.len == 0) return;
        const current = self.active_camera_index orelse 0;
        const next_idx = (current + 1) % self.cameras.items.len;
        self.switchCamera(next_idx);
    }

    pub fn prevCamera(self: *Scene) void {
        if (self.cameras.items.len == 0) return;
        const current = self.active_camera_index orelse 0;
        const prev_idx = if (current == 0) self.cameras.items.len - 1 else current - 1;
        self.switchCamera(prev_idx);
    }

    pub fn getActiveCameraIndex(self: Scene) ?usize {
        return self.active_camera_index;
    }

    pub fn getActiveCameraName(self: Scene) ?[]const u8 {
        if (self.active_camera_index) |idx| {
            if (idx < self.cameras.items.len) return self.cameras.items[idx].name;
        }
        if (self.active_camera) |cam| return cam.getName();
        return null;
    }

    pub fn getCameraCount(self: Scene) usize {
        return self.cameras.items.len;
    }

    pub fn setActiveCamera(self: *Scene, cam: ?Camera, owned_name: ?[]const u8) void {
        if (self.active_camera_owned_name) |old| {
            self.allocator.free(old);
        }
        self.active_camera = cam;
        self.active_camera_owned_name = owned_name;
        if (self.active_camera_index) |idx| {
            if (idx < self.cameras.items.len and cam != null) {
                self.cameras.items[idx].camera = cam.?;
            }
        }
    }

    pub fn updateCamera(self: *Scene, dt: f32) void {
        if (self.enable_multi_camera) {
            for (self.cameras.items, 0..) |*entry, i| {
                if (entry.enabled) {
                    entry.camera.update(dt);
                    if (self.active_camera_index == i) {
                        self.active_camera = entry.camera;
                    }
                }
            }
        } else if (self.active_camera) |*cam| {
            cam.update(dt);
            if (self.active_camera_index) |idx| {
                if (idx < self.cameras.items.len) {
                    self.cameras.items[idx].camera = cam.*;
                }
            }
        }
    }

    /// Viewport-aware picking ray. In multi-camera mode the visually relevant
    /// enabled camera under the cursor wins (draw overlay order: active first,
    /// then the rest in index order, topmost covering entry). Falls back to a
    /// forward ray from the origin when no camera covers the cursor (or no
    /// camera exists); pick() maps the same case to PickingInfo{} (no hit).
    /// Fullscreen viewports reduce to the legacy whole-window formula.
    pub fn createPickingRay(self: *Scene, screen_x: f32, screen_y: f32) Ray {
        const w = sapp.widthf();
        const h = sapp.heightf();
        if (w <= 0.0 or h <= 0.0) return Ray.new(Vec3.zero, Vec3.forward);
        if (self.enable_multi_camera and self.cameras.items.len > 0) {
            const idx = scene_picking.selectPickIndex(
                self.cameras.items,
                self.active_camera_index,
                screen_x,
                screen_y,
                w,
                h,
            ) orelse return Ray.new(Vec3.zero, Vec3.forward);
            const entry = self.cameras.items[idx];
            return scene_picking.createPickingRayViewport(entry.camera, screen_x, screen_y, w, h, entry.viewport);
        }
        if (self.active_camera) |cam| {
            const vp = if (self.active_camera_index) |ai| (if (ai < self.cameras.items.len)
                self.cameras.items[ai].viewport
            else
                cam.getViewport()) else cam.getViewport();
            if (!scene_picking.viewportContainsPoint(vp, screen_x, screen_y, w, h)) {
                return Ray.new(cam.getPosition(), Vec3.forward);
            }
            return scene_picking.createPickingRayViewport(cam, screen_x, screen_y, w, h, vp);
        }
        return Ray.new(Vec3.zero, Vec3.forward);
    }

    pub fn pickWithRay(self: *Scene, r: Ray) PickingInfo {
        return scene_picking.pickWithRay(self.meshes.items, self.physics.getWorld(), r);
    }

    /// Raycast adapter matching audio.RaycastFn for audio occlusion queries.
    pub fn audioRaycastAdapter(origin: Vec3, direction: Vec3, max_distance: f32, user_data: ?*anyopaque) bool {
        const self: *Scene = @ptrCast(@alignCast(user_data orelse return false));
        const r = Ray.new(origin, direction);
        const info = self.pickWithRay(r);
        return info.hit and info.distance <= max_distance;
    }

    /// Evaluates audio occlusion between listener and emitter through scene meshes and colliders.
    pub fn evaluateAudioOcclusion(
        self: *Scene,
        listener_pos: Vec3,
        emitter_pos: Vec3,
        config: audio.AudioOcclusionConfig,
    ) f32 {
        return audio.evaluateRaycastOcclusion(listener_pos, emitter_pos, config, audioRaycastAdapter, self);
    }

    /// Viewport-aware pick: resolves the camera like createPickingRay but
    /// returns PickingInfo{} (no hit) when no enabled camera covers the
    /// cursor, instead of casting a fallback ray into the scene.
    pub fn pick(self: *Scene, screen_x: f32, screen_y: f32) PickingInfo {
        const w = sapp.widthf();
        const h = sapp.heightf();
        if (w <= 0.0 or h <= 0.0) return PickingInfo{};
        if (self.enable_multi_camera and self.cameras.items.len > 0) {
            const idx = scene_picking.selectPickIndex(
                self.cameras.items,
                self.active_camera_index,
                screen_x,
                screen_y,
                w,
                h,
            ) orelse return PickingInfo{};
            const entry = self.cameras.items[idx];
            const r = scene_picking.createPickingRayViewport(entry.camera, screen_x, screen_y, w, h, entry.viewport);
            return self.pickWithRay(r);
        }
        const r = self.createPickingRay(screen_x, screen_y);
        // Single-camera PIP: cursor outside the viewport is a clean miss.
        if (self.active_camera) |cam| {
            const vp = if (self.active_camera_index) |ai| (if (ai < self.cameras.items.len)
                self.cameras.items[ai].viewport
            else
                cam.getViewport()) else cam.getViewport();
            if (!scene_picking.viewportContainsPoint(vp, screen_x, screen_y, w, h)) {
                return PickingInfo{};
            }
        } else {
            return PickingInfo{};
        }
        return self.pickWithRay(r);
    }

    pub fn createUI(self: *Scene) !*UICanvas {
        if (self.ui_canvas == null) {
            self.ui_canvas = try UICanvas.init(self.allocator);
        }
        return &self.ui_canvas.?;
    }

    pub fn getUI(self: *Scene) ?*UICanvas {
        if (self.ui_canvas) |*u| return u;
        return null;
    }

    pub fn projectPoint(self: *Scene, world_pos: Vec3) ?math.Vec2 {
        const cam = self.active_camera orelse return null;
        const w = sapp.widthf();
        const h = sapp.heightf();
        if (w <= 0.0 or h <= 0.0) return null;
        const vp = self.project.viewProjection(cam, w, h);
        return vp.projectPoint(world_pos, w, h);
    }

    pub fn handleEvent(self: *Scene, ev: [*c]const sapp.Event) void {
        if (self.active_camera) |*cam| {
            cam.handleEvent(ev);
            if (self.active_camera_index) |idx| {
                if (idx < self.cameras.items.len) {
                    self.cameras.items[idx].camera = cam.*;
                }
            }
        }
    }

    // ---- Rendering. ----

    /// True when the swapchain color AND depth formats both support MSAA at
    /// runtime (the main target mirrors them; see postprocess_pass.resize).
    /// sg.queryPixelformat is the sokol-provided backend gate.
    fn mainTargetFormatsMsaaCapable() bool {
        const env_def = sg.queryDesc().environment.defaults;
        const color_fmt: sg.PixelFormat = if (env_def.color_format != .DEFAULT and env_def.color_format != .NONE) env_def.color_format else .BGRA8;
        const depth_fmt: sg.PixelFormat = if (env_def.depth_format != .DEFAULT and env_def.depth_format != .NONE) env_def.depth_format else .DEPTH;
        return sg.queryPixelformat(color_fmt).msaa and sg.queryPixelformat(depth_fmt).msaa;
    }

    /// Forward pipeline set matching the MSAA main-target shape; created
    /// lazily (and recreated on count changes) on the first MSAA frame. The
    /// returned pointer aliases Scene state.
    fn ensureForwardMsaa(self: *Scene, samples: i32) *scene_forward.ForwardPipelines {
        if (self.forward_msaa == null or self.forward_msaa.?.sample_count != samples) {
            if (self.forward_msaa) |*fw| fw.deinit();
            if (self.forward.family_shaders) |fs| {
                self.forward_msaa = scene_forward.ForwardPipelines.initSampledWithShaders(samples, fs);
            } else {
                self.forward_msaa = scene_forward.ForwardPipelines.initSampled(samples);
            }
        }
        return &self.forward_msaa.?;
    }

    fn ensureClearResources(self: *Scene, samples: i32) void {
        if (self.clear_vb.id == 0) {
            self.clear_vb = sg.makeBuffer(.{
                .usage = .{ .vertex_buffer = true, .dynamic_update = true },
                .size = 16 * 6 * @sizeOf(debug_pass.Vertex),
            });
        }
        if (self.clear_shader.id == 0) {
            self.clear_shader = sg.makeShader(debug_shd.debugShaderDesc(sg.queryBackend()));
        }
        const target_pip = if (samples > 1) &self.clear_pipeline_msaa else &self.clear_pipeline;
        if (target_pip.id == 0) {
            var pip_desc = sg.PipelineDesc{
                .shader = self.clear_shader,
                .index_type = .NONE,
                .primitive_type = .TRIANGLES,
                .depth = .{
                    .compare = .ALWAYS,
                    .write_enabled = true,
                },
                .cull_mode = .NONE,
                .sample_count = samples,
            };
            pip_desc.layout.buffers[0] = .{ .stride = @sizeOf(debug_pass.Vertex) };
            pip_desc.layout.attrs[debug_shd.ATTR_debug_position] = .{
                .format = .FLOAT3,
                .offset = @offsetOf(debug_pass.Vertex, "position"),
            };
            pip_desc.layout.attrs[debug_shd.ATTR_debug_color0] = .{
                .format = .FLOAT4,
                .offset = @offsetOf(debug_pass.Vertex, "color"),
            };
            target_pip.* = sg.makePipeline(pip_desc);
            if (sg.queryPipelineState(target_pip.*) != .VALID) {
                std.debug.print("[CLEAR PIPELINE FAILED]: shader_state={}, pip_state={}\n", .{
                    sg.queryShaderState(self.clear_shader),
                    sg.queryPipelineState(target_pip.*),
                });
            }
        }
    }

    fn clearCurrentViewport(self: *Scene, color: Color4, samples: i32) void {
        self.ensureClearResources(samples);
        const pip = if (samples > 1) self.clear_pipeline_msaa else self.clear_pipeline;
        if (pip.id == 0 or self.clear_vb.id == 0 or sg.queryPipelineState(pip) != .VALID) return;

        const clear_verts = [_]debug_pass.Vertex{
            .{ .position = .{ -1.0, -1.0, 1.0 }, .color = .{ color.r, color.g, color.b, color.a } },
            .{ .position = .{ 1.0, -1.0, 1.0 }, .color = .{ color.r, color.g, color.b, color.a } },
            .{ .position = .{ 1.0, 1.0, 1.0 }, .color = .{ color.r, color.g, color.b, color.a } },
            .{ .position = .{ -1.0, -1.0, 1.0 }, .color = .{ color.r, color.g, color.b, color.a } },
            .{ .position = .{ 1.0, 1.0, 1.0 }, .color = .{ color.r, color.g, color.b, color.a } },
            .{ .position = .{ -1.0, 1.0, 1.0 }, .color = .{ color.r, color.g, color.b, color.a } },
        };
        const offset = sg.appendBuffer(self.clear_vb, sg.asRange(&clear_verts));
        if (offset < 0) return;
        // Учёт динамики: 6 вершин clear-квада через appendBuffer (байты те же — стрим в GPU-буфер).
        upload_meter.record(clear_verts.len * @sizeOf(debug_pass.Vertex));

        sg.applyPipeline(pip);
        var bind = sg.Bindings{};
        bind.vertex_buffers[0] = self.clear_vb;
        bind.vertex_buffer_offsets[0] = offset;
        sg.applyBindings(bind);

        const vs_params = debug_shd.VsParams{
            .mvp = Mat4.identity,
        };
        sg.applyUniforms(debug_shd.UB_vs_params, sg.asRange(&vs_params));
        sg.draw(0, 6, 1);
    }

    /// Shared queue/shadow/outline build parameters (stage-2 increment B):
    /// one internal function `buildQueuesInto` fills a `FrameDrawSlot` from
    /// these plus the explicit `snap` cameras (never `Scene.frame_snapshot`:
    /// the build must not read the consumed render snapshot). Fallback passes
    /// `&frame_snapshot` with `frame_id` as `cache_key`, `&self.stats`, the
    /// snapshot primary eye, the resolved snapshot sky/ibl, `true`, and
    /// `.published` (today's exact behavior). The game-side build passes
    /// `&build_snapshot` with the build-unique key `(build_seq | (1<<63))`
    /// (high bit set: cannot collide with any context `frame_id`, which
    /// counts up from 0), `&self.build_stats`, the frozen snapshot eye,
    /// the exact snapshot sky/ibl, `true`, and `.build_view`
    /// (provisional buffer/count until the latch patch).
    /// Payload identity invariant: every instanced batch / shadow item /
    /// outline item carries `source_uid` (stable `Mesh.uid`) + `source_mesh`
    /// (mesh-list index at build time); the latch validates uid before
    /// finalizing provisional handles (fail-closed zero).
    /// `eye` is currently informational (views sort by their own snapshot
    /// eye; the build passes the live eye for future transparent-sort use
    /// and for the instance CPU staging eye, which is threaded separately).
    /// `stats` stays a direct pointer (deferred via `build_stats` + merge).
    pub const QueueBuildParams = struct {
        snap: *const SceneFrameSnapshot,
        cache_key: u64,
        stats: *SceneStats,
        eye: Vec3,
        sky_texture: ?CubeTexture,
        ibl_intensity: f32,
        instances_prepared: bool,
        instance_source: @import("mesh.zig").InstanceSource,
    };

    // internal, used by scene tests
    pub fn prepareViewQueues(
        self: *Scene,
        queues: *scene_render_queue.RenderQueues,
        cam_snap: scene_snapshot.CameraSnapshot,
        sky_texture: ?CubeTexture,
        ibl_intensity: f32,
        cache_key: u64,
        stats: *SceneStats,
        instances_prepared: bool,
        instance_source: @import("mesh.zig").InstanceSource,
    ) void {
        queues.reset();

        scene_render_queue.buildFrameQueues(.{
            .allocator = self.allocator,
            .meshes = self.meshes.items,
            .cache_key = cache_key,
            .instance_source = instance_source,
            .view_proj = cam_snap.view_proj,
            .eye = cam_snap.eye,
            .cull_frustum = self.enable_frustum_culling,
            .cull_occlusion = self.enable_occlusion_culling,
            .culling_mask = cam_snap.culling_mask,
            .occlusion_culler = &self.occlusion_culler,
            .stats = stats,
            .queues = queues,
            .default_white_id = self.default_white_texture.view.id,
            .default_material = &self.default_material,
            .default_white = &self.default_white_texture,
            .default_normal = &self.default_normal_texture,
            .default_cube = &self.default_cube_texture,
            .sky_texture = sky_texture,
            .ibl_intensity = ibl_intensity,
            .default_morph_view = self.forward.default_morph_view,
            .thread_pool = jobs.global,
            // P5: grown instance buffers retire into the epoch queue; the
            // pre-stage above is definitive for this frame, so view builds
            // never retry staging mid-frame (failure coherence).
            .gpu_retire = &self.gpu_retire,
            .instances_prepared = instances_prepared,
        });

        std.mem.sort(RenderMeshItem, queues.items.items, {}, scene_render_queue.sortRenderItems);
        // Unified transparent order: regular + instanced groups globally
        // back-to-front by group distance (one entry per batch).
        std.mem.sort(
            scene_render_queue.TransparentDrawEntry,
            queues.transparent_order.items,
            {},
            scene_render_queue.sortTransparentDrawOrder,
        );
    }

    /// Shared queue/shadow/outline builder (stage-2 increment B): the code
    /// previously inline in `prepareFrame` (outline capture loop + shadow
    /// `prepareInto` + view-queue `prepareViewQueues` calls) in one internal
    /// function, callable from the game-side `buildPreparedFrame` (with
    /// `&build_snapshot` + `.build_view` + build-unique cache key +
    /// `&build_stats`) and from the fallback latch (with `&frame_snapshot` +
    /// `.published` + frame_id + `&self.stats`). Reads cameras ONLY from
    /// `params.snap` (never `self.frame_snapshot`/`self.build_snapshot`
    /// directly), so the build generation stays frozen. Producer `.build_view`
    /// freezes on the snapshot shadow switch alone; fallback keeps the
    /// historical live `shadows.enabled` gate.
    /// Fallback passes today's exact values (see `prepareFrame`) and stays
    /// bit-identical to the old inline path. Does NOT reset `back` (the
    /// caller reset before consume, as before). sg-free when
    /// `instances_prepared=true` (view builds never retry staging mid-frame).
    // internal, used by scene tests
    pub fn buildQueuesInto(self: *Scene, back: *FrameDrawSlot, params: QueueBuildParams) void {
        const snap = params.snap;
        // `eye` is informational today: views sort by their own snapshot eye
        // (see prepareViewQueues); the build passes the live eye for future
        // transparent-sort use. The instance CPU staging eye is threaded
        // separately (stageInstancesCpu / InstanceStageContext).
        _ = params.eye;

        // P5: outline capture is unconditional, as before — immediately
        // after the conditional pre-stage above and before conditional
        // shadow/view warming below. With a GPU context instanced items
        // snapshot the resolved staged state (`.published` = this frame's
        // `instance_render`, `.build_view` = provisional `instance_build_view`
        // until the latch patch); regular items are captured before
        // worldMatrixCached warming, exactly like the historical pre-queue
        // capture, so their cached-center behavior is unchanged.
        //
        // Outline identity domain (stage-2B fix): `source_mesh` MUST be the
        // mesh-list index into `self.meshes.items` (patch resolves against
        // that list with uid validation). The outline-list position is a
        // different domain (subset, any order) and MUST NOT be stored. Each
        // outline mesh is resolved to its mesh-list index here; an outline
        // mesh absent from the mesh list gets the OOB sentinel `meshes.len`
        // (deterministic, still emitted so the fallback — which never patches
        // — stays bit-identical; the latch patch fail-closes the sentinel via
        // its OOB branch).
        for (self.outline_meshes.items) |m| {
            if (m.gpu_pending or !m.is_visible or m.index_count == 0) continue;
            var src_idx: u32 = @intCast(self.meshes.items.len);
            for (self.meshes.items, 0..) |sm, si| {
                if (sm == m) {
                    src_idx = @intCast(si);
                    break;
                }
            }
            if (outline_pass.makeOutlineDrawItem(self.allocator, &back.outline_skins, m, params.cache_key, src_idx, params.instance_source)) |it| {
                back.outline_items.append(self.allocator, it) catch {};
            }
        }

        const is_gpu_init = (self.default_white_texture.view.id != 0);
        if (!is_gpu_init) return;

        // Shadow pass preparation into the back slot (disabled shadows —
        // or no camera — leave the reset-empty payload: coherent, never
        // the front slot's prior bins). Producer `.build_view` freezes on the
        // snapshot switch alone; fallback keeps the historical live gate.
        if (snap.has_camera and snap.shadows_enabled and (params.instance_source == .build_view or self.shadows.enabled)) {
            _ = self.shadows.pass.prepareInto(&back.shadow, self.meshes.items, params.cache_key, params.instance_source, jobs.global);
        }

        // View queues preparation (params carry the resolved sky/ibl).
        if (snap.has_camera) {
            if (snap.enable_multi_camera and snap.camera_count > 0) {
                const active_idx = snap.active_camera_idx;
                const primary_snap = if (active_idx < snap.camera_count) snap.cameras[active_idx] else snap.primary_cam;
                self.prepareViewQueues(&back.primary, primary_snap, params.sky_texture, params.ibl_intensity, params.cache_key, params.stats, params.instances_prepared, params.instance_source);
                for (snap.cameras[0..snap.camera_count], 0..) |entry, i| {
                    if (i == active_idx or !entry.enabled) continue;
                    self.prepareViewQueues(&back.views[i], entry, params.sky_texture, params.ibl_intensity, params.cache_key, params.stats, params.instances_prepared, params.instance_source);
                }
            } else {
                self.prepareViewQueues(&back.primary, snap.primary_cam, params.sky_texture, params.ibl_intensity, params.cache_key, params.stats, params.instances_prepared, params.instance_source);
            }
        }
    }

    fn renderSceneView(
        self: *Scene,
        cam_snap: scene_snapshot.CameraSnapshot,
        queues: *const scene_render_queue.RenderQueues,
        outline_items: []const outline_pass.OutlineDrawItem,
        outline_skins: []const [scene_render_queue.MAX_BONES]Mat4,
        samples: i32,
        snap: *const scene_snapshot.SceneFrameSnapshot,
        env: scene_draw.Environment,
    ) void {
        const view_proj = cam_snap.view_proj;
        const eye = cam_snap.eye;

        var frame_ctx = FrameContext{
            .view_proj = view_proj,
            .eye = eye,
            .sun_dir = snap.sun_dir,
            .sun_color = snap.sun_color,
            .sun_intensity = snap.sun_intensity,
            .directional_dir = snap.light_pack.directional_dir,
            .directional_color_int = snap.light_pack.directional_color_int,
            .cascades = snap.cascades,
            .light_counts = snap.light_pack.counts,
            .point_pos_range = snap.light_pack.point_pos_range,
            .point_color_int = snap.light_pack.point_color_int,
            .spot_pos_range = snap.light_pack.spot_pos_range,
            .spot_dir_inner = snap.light_pack.spot_dir_inner,
            .spot_color_outer = snap.light_pack.spot_color_outer,
            .spot_intensity = snap.light_pack.spot_intensity,
            .spot_view_proj = snap.light_pack.spot_view_proj,
            .spot_shadow_params = snap.light_pack.spot_shadow_params,
            .point_view_proj = snap.light_pack.point_view_proj,
            .point_shadow_params = snap.light_pack.point_shadow_params,
        };

        var shadow_state_with = env.shadow_uniforms;
        shadow_state_with.mesh_receive_shadows = true;
        const u_with = scene_uniforms.buildFrameUniforms(shadow_state_with, &frame_ctx);

        var shadow_state_no = env.shadow_uniforms;
        shadow_state_no.mesh_receive_shadows = false;
        const u_no = scene_uniforms.buildFrameUniforms(shadow_state_no, &frame_ctx);

        frame_ctx.uniforms_with_shadows = &u_with;
        frame_ctx.uniforms_without_shadows = &u_no;

        var current_pipeline_id: u32 = 0;

        // Opaque regular meshes first (front-to-back, early-Z).
        for (queues.items.items) |item| {
            scene_draw.drawRegularItem(&env, item, &frame_ctx, &current_pipeline_id, queues.skin_storage.items, queues.shader_storage.items, queues.coat_storage.items);
        }

        // Opaque instanced meshes.
        for (queues.opaque_instanced.items) |batch| {
            scene_draw.drawInstancedBatch(&env, batch, &frame_ctx, &current_pipeline_id, queues.coat_storage.items);
        }

        // Transparent pass: regular items and instanced groups interleaved in
        // one global back-to-front order (each instanced group draws as a
        // single batch at its sorted position; no per-instance sorting).
        for (queues.transparent_order.items) |entry| {
            switch (entry.kind) {
                .regular => {
                    if (entry.index < queues.transparent.items.len) {
                        scene_draw.drawRegularItem(&env, queues.transparent.items[entry.index], &frame_ctx, &current_pipeline_id, queues.skin_storage.items, queues.shader_storage.items, queues.coat_storage.items);
                    }
                },
                .instanced => {
                    if (entry.index < queues.transparent_instanced.items.len) {
                        scene_draw.drawInstancedBatch(&env, queues.transparent_instanced.items[entry.index], &frame_ctx, &current_pipeline_id, queues.coat_storage.items);
                    }
                },
            }
        }

        // Inverse-hull outline for highlighted meshes (P7: published slot
        // payload, never live Scene fields).
        self.postfx.renderOutlineItems(
            view_proj,
            eye,
            outline_items,
            outline_skins,
            samples,
            &self.stats,
            snap.outline_enabled,
            snap.outline_color,
            snap.outline_width_px,
        );

        // Physics debug lines: prepared capture only — no live world,
        // no show_debug read at draw time (update may step the world
        // concurrently). One committed upload (prepare), one draw per view.
        self.physics.renderDebugPrepared(view_proj, samples, &self.stats);

        // Skybox Pass: captured enabled/texture/exposure only. The cube is
        // the snapshot sky texture orelse the snapshot's render-owned
        // default copy — never self.sky.* / self.default_cube_texture live.
        self.sky.renderPrepared(
            snap.sky_enabled,
            cam_snap.camera,
            cam_snap.aspect,
            snap.sky_texture orelse snap.default_cube,
            snap.sky_exposure,
            samples,
            &self.stats,
        );

        // Particle Pass: prepared frame only — no live ParticleSystem reads
        // at draw time (the update side may step systems concurrently).
        self.particles.renderPrepared(cam_snap.camera, cam_snap.aspect, samples, &self.stats);
    }

    /// Packs the current camera, light, shadow, and environment state into an immutable
    /// frame snapshot that can be published to the render thread.
    pub fn packFrameSnapshot(self: *Scene, aspect: f32, cur_w: i32, cur_h: i32) scene_snapshot.SceneFrameSnapshot {
        const w = if (cur_w > 0) cur_w else sapp.width();
        const h = if (cur_h > 0) cur_h else sapp.height();
        const eff_aspect = if (aspect > 0.0) aspect else (if (h > 0) @as(f32, @floatFromInt(w)) / @as(f32, @floatFromInt(h)) else 1.0);

        var snap = scene_snapshot.SceneFrameSnapshot{
            .frame_id = self.frame_id,
            .aspect = eff_aspect,
            .screen_w = w,
            .screen_h = h,
        };

        const primary_cam_opt = self.active_camera orelse (if (self.cameras.items.len > 0) self.cameras.items[0].camera else null);
        if (primary_cam_opt == null) {
            snap.has_camera = false;
            return snap;
        }
        snap.has_camera = true;

        snap.enable_multi_camera = self.enable_multi_camera;
        snap.active_camera_idx = self.active_camera_index orelse 0;
        snap.camera_count = @min(self.cameras.items.len, scene_snapshot.MAX_CAMERAS);

        for (0..snap.camera_count) |i| {
            const entry = self.cameras.items[i];
            const cam_rect = entry.viewport.toPixelRect(w, h);
            const cam_aspect = cam_rect.aspect();
            snap.cameras[i] = scene_snapshot.CameraSnapshot{
                .camera = entry.camera,
                .view_proj = entry.camera.getViewProjection(cam_aspect),
                .eye = entry.camera.getPosition(),
                .viewport = entry.viewport,
                .culling_mask = entry.culling_mask,
                .clear_viewport = entry.clear_viewport,
                .clear_color = entry.clear_color,
                .aspect = cam_aspect,
                .enabled = entry.enabled,
            };
        }

        if (snap.enable_multi_camera and snap.camera_count > 0 and snap.active_camera_idx < snap.camera_count) {
            snap.primary_cam = snap.cameras[snap.active_camera_idx];
        } else {
            const vp = primary_cam_opt.?.getViewport();
            const rect = vp.toPixelRect(w, h);
            const cam_aspect = rect.aspect();
            snap.primary_cam = scene_snapshot.CameraSnapshot{
                .camera = primary_cam_opt.?,
                .view_proj = primary_cam_opt.?.getViewProjection(cam_aspect),
                .eye = primary_cam_opt.?.getPosition(),
                .viewport = vp,
                .culling_mask = primary_cam_opt.?.getCullingMask(),
                .aspect = cam_aspect,
            };
        }

        snap.sun_dir = self.lights.sunDirection();
        snap.sun_color = self.lights.sunColor();
        snap.sun_intensity = self.lights.sunIntensity();
        snap.cascades = self.shadows.computeCascades(snap.primary_cam.camera, snap.primary_cam.aspect, snap.sun_dir);

        var lp = self.light_pack;
        _ = self.light_handoff.takeLatest(&lp);
        self.light_pack = lp;
        snap.light_pack = lp;

        snap.shadows_enabled = self.shadows.enabled;
        snap.shadow_uniforms = self.shadows.uniformState(self.lights.hemi.ground_color);
        snap.sky_texture = self.sky.texture;
        snap.sky_enabled = self.sky.enabled;
        snap.sky_exposure = self.sky.exposure;
        snap.ibl_intensity = self.sky.ibl_intensity;
        // Render-owned default copies (plain GPU-handle values): the draw
        // binds these, never the live Scene.default_*_texture fields.
        snap.default_white = self.default_white_texture;
        snap.default_normal = self.default_normal_texture;
        snap.default_cube = self.default_cube_texture;
        snap.clear_color = self.clear_color;
        snap.msaa_sample_count = self.msaa_sample_count;
        snap.post_process = self.post_process;
        snap.ssao = self.ssao;
        snap.outline_enabled = self.postfx.outline_enabled;
        snap.outline_color = self.postfx.outline_color;
        snap.outline_width_px = self.postfx.outline_width_px;

        return snap;
    }

    /// Publishes a complete frame snapshot through the lock-free mailbox.
    /// When the mailbox is saturated (consumer lagging, both slots
    /// published), stale published slots are drained first so the NEWEST
    /// snapshot wins — otherwise prepareFrame's takeLatest would resurface
    /// an older published frame over the newer fallback.
    ///
    /// Render-ownership: the saturated fallback NEVER writes the consumed
    /// `frame_snapshot` directly. Render reads `frame_snapshot`
    /// concurrently with update (update||render overlap), so a producer-side
    /// overwrite would race the draw; the last-unclaimable tick is DROPPED
    /// instead (newest published frame stays, this one is skipped). Producer
    /// (update) vs consumer (prepare) stay excluded under phase_mutex, which
    /// is what makes releasePublished safe here.
    pub fn publishFrameSnapshot(self: *Scene, aspect: f32, cur_w: i32, cur_h: i32) void {
        const snap = self.packFrameSnapshot(aspect, cur_w, cur_h);
        if (self.frame_handoff.claim()) |i| {
            self.frame_handoff.slot(i).* = snap;
            self.frame_handoff.publish(i);
        } else {
            // Saturated: drop stale published frames (consumer is excluded
            // by phase ownership here) and publish the newest.
            self.frame_handoff.releasePublished();
            if (self.frame_handoff.claim()) |i| {
                self.frame_handoff.slot(i).* = snap;
                self.frame_handoff.publish(i);
            }
            // Still unclaimable (a slot is held in WRITING state): DROP.
            // Never fall back to `self.frame_snapshot = snap` — the
            // consumed snapshot belongs to the in-flight render.
        }
    }

    /// Stage 3: prepares GPU uploads and acquires the frame-level snapshot.
    /// Phase contract (actual update||render): prepare + render run
    /// SEQUENTIALLY on the context thread (next prepare NEVER concurrent
    /// with render); update-vs-prepare stay excluded under phase_mutex, but
    /// update CAN overlap render — so phase ownership NO LONGER spans
    /// prepareFrame() AND render(), only update-vs-prepare. The draw phase
    /// therefore reads ONLY render-owned captures: P4 mesh payload
    /// (regular/instanced очереди, shadow-bins, outline-items — trails
    /// included: Trail.update is CPU-only staging, the prepare flush
    /// uploads it, and P7 bakes handle/model/count values), P6 UI frame,
    /// physics-debug capture + committed upload, particle prepared frame,
    /// sky params + default texture copies (snapshot), light pack
    /// (snapshot). Живые Mesh/Material/Skeleton/мир физики/sky/системы
    /// частиц во время отрисовки недоступны. Заимствованными остаются
    /// только GPU-хендлы под фазовым мьютексом/P3
    /// (буферы/вью/сэмплеры/пайплайны).
    /// At most `upload_budget_per_frame` textures upload per call; leftover
    /// `.ready` slots ride to subsequent frames instead of stalling one frame.
    pub const upload_budget_per_frame: usize = 4;
    /// Per-frame texture-upload byte cap. A 2K RGBA8 map with mips is ~21 MiB,
    /// so a pure count budget lets a few big textures stall a frame (frames
    /// #1-#3 of the live profile uploaded 85/65/46 MB at 4 textures/frame).
    /// 8 MiB keeps the worst single-frame burst under one small mip chain
    /// while still finishing a 2K map in ~3 frames; the overshoot rule always
    /// uploads at least one texture per call so a lone oversized texture
    /// still makes progress. Tune only with a profiled reason.
    pub const upload_byte_budget_per_frame: usize = 8 * 1024 * 1024;
    /// P6 UI handoff: captures CPU geometry + draw parameters out of the
    /// live canvas into the render-owned frame and uploads at this
    /// prepare/context boundary (phase mutex held, context thread). Runs
    /// after the UI build (sandbox builds under the same mutex before
    /// prepareFrame) and after the frame snapshot above is final, so the
    /// captured screen size matches the selected scene snapshot (same
    /// fallback as render: snapshot dims when positive, live sapp dims
    /// otherwise — a zero placeholder snapshot never hides a valid GPU
    /// frame). Snapshots canvas presence separately from nonempty/gpu_ready
    /// so the historical phantom+1 counters survive empty frames. Missing
    /// canvas or camera-less frames fail close to coherent-empty — never a
    /// stale prior overlay, never an unsafe upload/draw.
    fn captureUiFrame(self: *Scene) void {
        const canvas = if (self.ui_canvas) |*c| c else {
            self.ui_frame.clearEmpty();
            self.ui_frame.canvas_present = false;
            return;
        };
        self.ui_frame.canvas_present = true;
        if (!self.frame_snapshot.has_camera) {
            self.ui_frame.clearEmpty();
            self.ui_frame.canvas_present = true;
            return;
        }
        const w = if (self.frame_snapshot.screen_w > 0) self.frame_snapshot.screen_w else sapp.width();
        const h = if (self.frame_snapshot.screen_h > 0) self.frame_snapshot.screen_h else sapp.height();
        self.ui_frame.capture(
            self.allocator,
            canvas,
            @floatFromInt(w),
            @floatFromInt(h),
        );
        _ = self.ui_frame.upload(canvas, .{
            .allocator = self.allocator,
            .retire_queue = &self.gpu_retire,
        });
    }

    /// P7 published consumable draw payload: the prepared mesh draw lists
    /// (PRIMARY + ALL PIP view queues with their skin/shader side stores,
    /// outline items+skins, prepared shadow items+skins+bin ranges), with the
    /// frame_id/retire_epoch that built them. The ONLY low-level draw
    /// accessor — render and P5/P7 tests read through here, never raw fields.
    /// Scope is mesh draws only: UI (P6 ui_frame), physics-debug lines
    /// (prepared_lines + committed DebugPass upload), sky params + default
    /// copies (frame_snapshot), and particles (prepared frame) are separate
    /// payloads — none of them reads live subsystems at draw time. Trail
    /// meshes ride these same queues: Trail.update stages CPU-side, the
    /// prepare flush uploads, and the queue build bakes the values.
    ///
    /// Borrow rules: every GPU handle inside is BORROWED (phase mutex / P3
    /// epochs, no second GPU copies); CPU slot retention is NOT a GPU
    /// lifetime pin. GPU consumability ends at the consuming render's return
    /// or — when no render consumes the frame — at the START of the next
    /// prepareFrame (a repeated prepare discards the pending frame before
    /// GpuRetire.begin/flush, and that flush may tear down its borrowed
    /// handles); it also ends at deinit. Retained CPU storage may be reused
    /// as back scratch by any prepare, so the returned pointer (and any slice
    /// taken from it) is consumable only while frame_prepared is set or
    /// during the render call consuming this frame (including the inner
    /// render of `renderReuse`, which re-draws the already-consumed front
    /// without a prepare) — never across a prepare boundary. Render
    /// completes the frame epoch on all returns (no-camera too); the reuse
    /// re-run re-completes the same epoch, which is idempotent (no-op).
    /// One pending frame, no concurrent prepare/render — but update
    /// CAN overlap render (update-vs-prepare stay excluded instead).
    pub fn preparedDraws(self: *const Scene) *const FrameDrawSlot {
        return &self.draws.slots[self.draws.front];
    }

    /// Latch patch finalizing game-built provisional handles (stage-2
    /// increment B, context side, allocation-free): after
    /// `stageInstancesLatch` publishes, every instanced payload entry built
    /// with `.build_view` is re-resolved by identity (`source_mesh` index +
    /// `source_uid` validation) against the SLOT-OWNED staged records — never
    /// against the live mesh list.
    ///
    /// Payload identity invariant: `source_uid` is the stable `Mesh.uid`,
    /// `source_mesh` the mesh-list index at build time. Provisional vs
    /// finalized: at build time `instance_buffer`/`visible_instance_count`
    /// (plus shadow `world_aabb`/`max_dim`, outline `world_center`) came from
    /// the provisional `instance_build_view` (frozen count/bounds + old
    /// handle); here they are finalized from the post-latch RECORD mirror
    /// (which the latch copied from its guarded `instance_render`
    /// write-back, so the values are identical to reading live state —
    /// without the race).
    ///
    /// Records are appended in mesh-list order (strictly increasing
    /// `mesh_index`), so the lookup below is a linear scan with early exit.
    /// A mesh-list mutation between build and latch can therefore NOT slip
    /// a stale entry through: the latch guard already fail-closed the
    /// affected records (`staged_frame != frame_id`), and the patch zeroes
    /// them here. The live mesh list is validated at latch time, not patch
    /// time — the patch performs zero live reads (outline stale fallbacks
    /// use the record-frozen `mesh_position`, not live `mesh.position`).
    ///
    /// Per entry (only `is_instanced` shadow/outline items and all instanced
    /// batches; regular items have no provisional handle and are skipped):
    /// - find the record with `mesh_index == source_mesh`; missing (mesh
    ///   list shrank, or the mesh never froze a record) → fail-closed zero.
    ///   Outline center falls back to `Vec3.zero` (no record to read).
    /// - `record.uid != source_uid` (reorder changed the list) →
    ///   fail-closed zero; outline center falls back to the record-frozen
    ///   `mesh_position` (the mesh is known via the record).
    /// - `record.staged_frame != frame_id` (latch skipped/failed this
    ///   record) → fail-closed zero with the same outline fallback.
    /// - else copy `buffer`/`count` (+ shadow `world_aabb` from
    ///   `record.bounds` with `max_dim` recomputed from extents exactly as
    ///   `prepareInto` did, outline `world_center` from `record.bounds`
    ///   center or the frozen `mesh_position` when invalid).
    /// - Transparent order entries referencing zeroed batches need no distance
    ///   change: the draw skips `count==0` batches, so order is harmless.
    /// - Culling/inclusion stay frozen at build time (bounds/model/distance
    ///   are NOT repatched): a live TRS mutation between build and latch
    ///   never alters this frame's sets, only the next build sees it.
    fn findStagedRecord(records: []const StagedInstanceRecord, idx: usize) ?*const StagedInstanceRecord {
        for (records) |*rec| {
            const ri: usize = rec.mesh_index;
            if (ri == idx) return rec;
            if (ri > idx) break;
        }
        return null;
    }

    fn patchInstanceRefs(self: *Scene, back: *FrameDrawSlot) void {
        _ = self;
        const records = back.staged_instances.items;
        const fid = back.frame_id;
        // Queue batches: primary + all views, opaque + transparent.
        const queue_lists = [_]*std.ArrayListUnmanaged(scene_render_queue.RenderInstancedBatch){
            &back.primary.opaque_instanced, &back.primary.transparent_instanced,
        };
        for (queue_lists) |list| patchBatchList(list, records, fid);
        for (&back.views) |*q| {
            patchBatchList(&q.opaque_instanced, records, fid);
            patchBatchList(&q.transparent_instanced, records, fid);
        }
        // Shadow items (instanced only).
        for (back.shadow.items.items) |*it| {
            if (!it.is_instanced) continue;
            const rec = findStagedRecord(records, it.source_mesh) orelse {
                it.instance_buffer = .{};
                it.visible_instance_count = 0;
                it.world_aabb = BoundingBox.zero;
                it.max_dim = 0;
                continue;
            };
            if (rec.uid != it.source_uid or rec.staged_frame != fid) {
                it.instance_buffer = .{};
                it.visible_instance_count = 0;
                it.world_aabb = BoundingBox.zero;
                it.max_dim = 0;
                continue;
            }
            it.instance_buffer = rec.buffer;
            it.visible_instance_count = rec.count;
            it.world_aabb = rec.bounds;
            const ext = rec.bounds.extents();
            it.max_dim = @max(ext.x, @max(ext.y, ext.z));
        }
        // Outline items (instanced only).
        for (back.outline_items.items) |*it| {
            if (!it.is_instanced) continue;
            const rec = findStagedRecord(records, it.source_mesh) orelse {
                it.instance_buffer = .{};
                it.visible_instance_count = 0;
                it.world_center = Vec3.zero;
                continue;
            };
            if (rec.uid != it.source_uid or rec.staged_frame != fid) {
                it.instance_buffer = .{};
                it.visible_instance_count = 0;
                it.world_center = rec.mesh_position;
                continue;
            }
            it.instance_buffer = rec.buffer;
            it.visible_instance_count = rec.count;
            it.world_center = if (rec.bounds.isValid()) rec.bounds.center() else rec.mesh_position;
        }
    }

    fn patchBatchList(
        list: *std.ArrayListUnmanaged(scene_render_queue.RenderInstancedBatch),
        records: []const StagedInstanceRecord,
        fid: u64,
    ) void {
        for (list.items) |*b| {
            const rec = findStagedRecord(records, b.source_mesh) orelse {
                b.instance_buffer = .{};
                b.visible_instance_count = 0;
                continue;
            };
            if (rec.uid != b.source_uid or rec.staged_frame != fid) {
                b.instance_buffer = .{};
                b.visible_instance_count = 0;
                continue;
            }
            b.instance_buffer = rec.buffer;
            b.visible_instance_count = rec.count;
        }
    }

    /// Stage-2 increment B producer build (game/update phase, CPU-only,
    /// sg-free): stages the CPU halves the prepare latch will consume —
    /// instance matrices into the back-slot scratch + per-mesh previews +
    /// the slot-owned staged records, the
    /// particle build frame, the physics debug build capture — then freezes
    /// the provisional `instance_build_view` per mesh and builds the full
    /// queue/shadow/outline payload into the back slot via the shared
    /// `buildQueuesInto` (with `.build_view` + build-unique cache key +
    /// `&build_stats`), and bumps `build_seq`.
    ///
    /// Call AFTER the sim mutations of the tick (update boundary), BEFORE
    /// the context `prepareFrame`; sequential with update, excluded vs
    /// prepare (phase ownership, plain fields, no atomics). Callable from
    /// any non-pool thread (game thread or a spawned worker — never
    /// concurrent with update or prepare); also callable on the
    /// context/single thread. Two builds before a latch: newest wins (the
    /// single preview store is recomputed, the scratch overwritten) — no
    /// build queue, bounded, no allocs beyond retained capacity.
    ///
    /// Build-view freeze: for each mesh with a fresh preview
    /// (`preview.build_seq == build_seq`) set `instance_build_view` from the
    /// preview count/bounds/hash + the current `instance_render`
    /// buffer/capacity/uploaded_count/staged_frame (provisional handle); for
    /// a stale/absent preview set all-zero (invisible). The queue build then
    /// resolves instanced state via `.build_view`; the latch `patchInstanceRefs`
    /// finalizes handles after `stageInstancesLatch`.
    ///
    /// Snapshot: consumes the newest published tick into the producer-owned
    /// `build_snapshot` (update-vs-prepare excluded) BEFORE any CPU staging;
    /// when nothing new was published ALWAYS packs fresh live state — never
    /// reuses the consumed render `frame_snapshot`, so the build works
    /// without a publish and after camera removal. Staging eye, queue
    /// culling, shadow switch, and sky/ibl all freeze on this generation
    /// (fixed-size snapshot values only; live meshes/materials/culling flags
    /// stay live by design). Never reads or writes `frame_snapshot`.
    /// Cache key is the build-unique `(build_seq | (1<<63))` (high bit set:
    /// cannot collide with any context `frame_id`). Stats accumulate into
    /// game-owned `build_stats` (cleared at build start), merged by the latch.
    ///
    /// Touches NOTHING else: no sg.*, no GpuRetire begin/complete/flush (view
    /// builds run with `instances_prepared=true`, never retrying staging),
    /// no frame_id/retire_epoch (stamped by the latch), no UI canvas/frame,
    /// no `self.stats`, no profiler. Mutates under game-phase ownership only:
    /// back-slot queues/shadow/outline + scratch, previews/build_views,
    /// staged records,
    /// particle/physics build frames, `build_snapshot` (refreshed), shadow
    /// bin scratch, occlusion-culler frame state, world-matrix cache (tagged
    /// with the build key), and `build_stats`.
    ///
    /// App contract: no latch is possible while a build runs (update-vs-
    /// prepare exclusion), and the mesh list MUST NOT be mutated between a
    /// build and its latch (validated by uid at patch time: a violation
    /// fail-closes stale entries to invisible instead of corrupting). A mesh
    /// whose upload finishes between build and latch, or whose segment OOMs,
    /// keeps its previous complete `instance_render` for one frame
    /// (documented, coherent); the next funded build+latch picks it up.
    pub fn buildPreparedFrame(self: *Scene) void {
        // Deliberately NO gpu_thread assert: this runs on the game side or
        // a spawned worker. Everything below is sg-free (the CPU staging
        // half, the plain captures, the CPU queue/shadow/outline build with
        // instances_prepared=true); any sg.* here would be a bug.
        self.build_stats = .{};
        self.build_seq +%= 1;
        const seq = self.build_seq;
        const back_idx = self.draws.backIndex();
        const back = &self.draws.slots[back_idx];
        back.reset();
        self.build_slot = back_idx;
        // Producer snapshot FIRST (update-vs-prepare excluded): consume the
        // newest published tick into the producer-owned build_snapshot. When
        // nothing new was published, ALWAYS pack fresh live state — never
        // reuse the consumed render snapshot, so the build works without a
        // publish and after camera removal. Never touches frame_snapshot
        // (update may overlap render). Everything below freezes on this
        // generation (fixed-size snapshot values only; live meshes/materials/
        // culling flags stay live by design).
        if (!self.frame_handoff.takeLatest(&self.build_snapshot)) {
            const cur_w = sapp.width();
            const cur_h = sapp.height();
            const aspect = if (cur_h > 0) @as(f32, @floatFromInt(cur_w)) / @as(f32, @floatFromInt(cur_h)) else 1.0;
            self.build_snapshot = self.packFrameSnapshot(aspect, cur_w, cur_h);
        }
        // Frozen snapshot eye (zero when camera-less): the CPU staging sort
        // and the queue build both use this generation, never the live eye.
        const eye = if (self.build_snapshot.has_camera) self.build_snapshot.primary_cam.eye else Vec3.zero;
        scene_instance_staging.stageInstancesCpu(.{
            .allocator = self.allocator,
            .scratch = &back.primary.instance_matrices,
            .thread_pool = jobs.global,
            .eye = eye,
        }, self.meshes.items, seq);
        // Freeze the provisional build view for the queue build below.
        for (self.meshes.items) |m| {
            _ = m.ensureUid();
            if (m.instance_preview.build_seq == seq) {
                m.instance_build_view = .{
                    .buffer = m.instance_render.buffer,
                    .capacity = m.instance_render.capacity,
                    .count = m.instance_preview.count,
                    .bounds = m.instance_preview.bounds,
                    .hash = m.instance_preview.hash,
                    .uploaded_count = m.instance_render.uploaded_count,
                    .staged_frame = m.instance_render.staged_frame,
                };
            } else {
                m.instance_build_view = .{};
            }
        }
        // Freeze the slot-owned staged records for the prepare latch below
        // (same fresh-preview set as the build-view freeze above). The latch
        // and `patchInstanceRefs` consume these — never live previews.
        scene_instance_staging.freezeStagedRecords(self.allocator, &back.staged_instances, self.meshes.items, seq);
        self.particles.buildCapture(self.allocator, seq);
        self.physics.buildDebug(self.allocator, seq);
        // Game-side queue/shadow/outline build (sg-free: instances_prepared).
        // The shared builder resets each view queue (including primary's
        // instance_matrices scratch) — but that scratch holds the CPU-staged
        // matrices the latch GPU half still needs. Swap it out across the
        // build and restore after: the queue build with instances_prepared
        // never appends to it, so the staged segments survive intact.
        {
            const build_key = seq | (@as(u64, 1) << 63);
            const sky_tex = self.build_snapshot.sky_texture;
            const ibl_int = self.build_snapshot.ibl_intensity;
            const saved_scratch = back.primary.instance_matrices;
            back.primary.instance_matrices = .empty;
            self.buildQueuesInto(back, .{
                .snap = &self.build_snapshot,
                .cache_key = build_key,
                .stats = &self.build_stats,
                .eye = eye,
                .sky_texture = sky_tex,
                .ibl_intensity = ibl_int,
                .instances_prepared = true,
                .instance_source = .build_view,
            });
            // The builder left the swapped-in list empty (no staging appends
            // under instances_prepared); discard it (zero capacity, no leak)
            // and restore the staged scratch.
            back.primary.instance_matrices = saved_scratch;
        }
    }

    pub fn prepareFrame(self: *Scene) void {
        // Владение фазой: prepare выполняется на context-потоке
        // ПОСЛЕДОВАТЕЛЬНО с render (один поток, next prepare NEVER
        // concurrent with render); update-поток в это время ИСКЛЮЧЁН
        // (phase_mutex update-vs-prepare), а во время render — НЕТ: update
        // CAN overlap render. Внутри prepare — только контекстные операции:
        // flushPendingGpuUploads, стейджинг инстансов, shadow prepare,
        // построение очередей + CPU-capture (debug/particles/UI) и их GPU
        // upload. Draw-фаза ниже читает только render-owned снимки.
        gpu_thread.assertOnContextThread();
        // P7: repeated prepare discards the previous pending frame BEFORE
        // GpuRetire.begin/flush below: its borrowed handles may be torn down
        // by the flush, so preparedDraws() payloads from that frame lose GPU
        // consumability from this point (retained CPU storage may be reused
        // as back scratch). The new publish at the end re-associates
        // frame_id/retire_epoch.
        self.frame_prepared = false;
        // Начало кадра (P3): новый epoch ретенции. flush ниже (внутри
        // flushPendingGpuUploads) уничтожит только завершённые эпохи —
        // запись текущего кадра ждёт его конца. begin заодно закрывает
        // предыдущий незакрытый epoch (discarded-pending контракт выше).
        self.retire_epoch = self.gpu_retire.begin();
        // update_ms приходит staged (recordUpdateTime -> pending_update_ms,
        // игровой поток), prepare_ms пишет app на этом же context-потоке
        // вокруг prepareFrame (см. main): оба сохраняются через сброс, всё
        // остальное обнуляется под новый кадр. Прямых stats-записей с
        // update-потока нет — stats читает render конкурентно с update.
        const keep_prepare_ms = self.stats.prepare_ms;
        self.stats = .{};
        self.stats.update_ms = self.pending_update_ms;
        self.stats.prepare_ms = keep_prepare_ms;
        // Сброс счётчика динамических обновлений на начало кадра: всё, что
        // запишут flushPendingGpuUploads, стейджинг инстансов и UI/debug
        // upload'ы ниже — всё внутри prepare — плюс clear-append'ы внутри
        // render, сложится в stats.updated_bytes_frame перед
        // Profiler.recordFrame. На троттлинг текстур не влияет.
        _ = upload_meter.takeAndReset();
        self.frame_id +%= 1;

        if (self.uploads) |*q| {
            self.frame_uploads = q.drainCountedBudget(upload_budget_per_frame, upload_byte_budget_per_frame);
        } else {
            self.frame_uploads = .{};
        }
        self.stats.uploaded_textures_frame = std.math.cast(u32, self.frame_uploads.count) orelse std.math.maxInt(u32);
        self.stats.uploaded_bytes_frame = self.frame_uploads.bytes;
        self.flushPendingGpuUploads();

        // Particle prepared frame: capture the retained plain frame here,
        // after the flush above and BEFORE the update/render unlock below.
        // Reads live systems for the LAST time this frame; the draw below
        // sees only the capture. Stage 1: when the game side built a fresh
        // frame (`build_seq` newer than `last_latched_seq`), latch the build
        // instead of restaging from live systems; otherwise the historical
        // inline capture (apps without `buildPreparedFrame` are unchanged).
        const have_build = self.build_seq != self.last_latched_seq;
        if (have_build) {
            self.particles.latchFrame(self.allocator);
        } else {
            self.particles.captureFrame(self.allocator);
        }

        // Physics debug wireframe capture (CPU): world.appendDebugLines runs
        // HERE in prepare — never inside render. The draw below reads only
        // the capture (prepared_visible/prepared_lines). Same stage 1 shape
        // as particles: latch a fresh build, else capture inline.
        if (have_build) {
            self.physics.latchDebug(self.allocator);
        } else {
            self.physics.captureDebug(self.allocator);
        }

        // Snapshot generations: with a fresh game build the latch publishes
        // the EXACT build generation (`build_snapshot`, by value) — a newer
        // publication after the build stays queued for the next build or
        // fallback and never mixes culling/camera generations into these
        // queues. Without a build the historical takeLatest-else-pack runs.
        if (have_build) {
            self.frame_snapshot = self.build_snapshot;
        } else {
            var snap = self.frame_snapshot;
            if (self.frame_handoff.takeLatest(&snap)) {
                self.frame_snapshot = snap;
            } else {
                const cur_w = sapp.width();
                const cur_h = sapp.height();
                const aspect = if (cur_h > 0) @as(f32, @floatFromInt(cur_w)) / @as(f32, @floatFromInt(cur_h)) else 1.0;
                self.frame_snapshot = self.packFrameSnapshot(aspect, cur_w, cur_h);
            }
        }

        // Debug line upload (GPU): the frame's single updateBuffer, once per
        // prepare no matter how many PIP views render below. Samples follow
        // the same MSAA policy as render (same snapshot inputs, same
        // result). Headless the whole upload is skipped (the CPU capture
        // above already ran for tests) — no sg.* without a context.
        if (sg.isvalid()) {
            const upload_samples = scene_msaa.effectiveSampleCount(self.frame_snapshot.msaa_sample_count, .{
                .post_enabled = self.frame_snapshot.post_process.enabled,
                .formats_msaa_capable = mainTargetFormatsMsaaCapable(),
                .backend = sg.queryBackend(),
            });
            self.physics.uploadDebug(self.allocator, upload_samples);
        }

        const is_gpu_init = (self.default_white_texture.view.id != 0);
        // P7: build the BACK slot in place, then publish with one index flip
        // at the end. The front slot is untouched during the build:
        // allocator failure in the back corrupts nothing consumable.
        const back_idx = self.draws.backIndex();
        const back = &self.draws.slots[back_idx];
        if (have_build) {
            // Stage-2 latch: the game-side build already reset this back
            // slot, staged the instance scratch + previews + staged records +
            // build_views, and
            // built the queue/shadow/outline payload with `.build_view`
            // (provisional handles). Reset-before-consume would erase the
            // build, and rebuilding would unfreeze the sets — so do NOT call
            // buildQueuesInto here. Only stamp the prepare-owned frame/epoch,
            // run the GPU halves over the slot records + scratch, finalize the
            // provisional handles with patchInstanceRefs, then merge the
            // deferred build_stats. Meshes with no record (OOM-skipped,
            // post-build meshes) keep their previous complete
            // `instance_render` and
            // patch to invisible — no partial publish. Mesh-list mutation
            // between build and latch fail-closes by guard at latch time and
            // by record lookup at patch time (see patch docs).
            //
            // Remaining live coupling (next slice): the update-vs-prepare
            // mutex still guards the whole handoff, and the latch still
            // writes back `mesh.instance_render` (mirrored into the record so
            // the patch itself is self-contained).
            std.debug.assert(self.build_slot == back_idx);
            back.frame_id = self.frame_id;
            back.retire_epoch = self.retire_epoch;
            if (is_gpu_init) {
                // Same position/gate as the historical pre-stage: the GPU half
                // must run before the patch finalizes handles (and would have
                // run before the shadow snapshot historically). Same retire
                // queue and eye source shape (the eye itself was consumed at
                // build time for the transparent sort; only the guard reads
                // the snapshot here). The latch consumes the slot-owned
                // staged records frozen by buildPreparedFrame — no live
                // preview reads; a mesh-list mutation between build and latch
                // trips the per-record guard and fail-closes.
                if (self.frame_snapshot.has_camera) {
                    scene_instance_staging.stageInstancesLatch(.{
                        .allocator = self.allocator,
                        .frame_id = self.frame_id,
                        .retire_queue = &self.gpu_retire,
                    }, back.staged_instances.items, self.meshes.items, &back.primary.instance_matrices);
                }
            }
            self.patchInstanceRefs(back);
            // Deferred stats merge (stage-2B): the game-side queue build
            // accumulated into build_stats; fold the queue counters into the
            // context-owned self.stats (already reset above, so upload
            // tallies/prepare_ms/update_ms are preserved) and clear the build
            // stats for the next tick. build_stats deferred merge.
            self.stats.mergeFrom(&self.build_stats);
            self.build_stats = .{};
            self.last_latched_seq = self.build_seq;
        } else {
            // Inline fallback (no fresh build): reset first — every list,
            // including disabled views/shadow bins, so a skipped path can never
            // resurface the other slot's prior frame — then stage fully.
            back.reset();
            back.frame_id = self.frame_id;
            back.retire_epoch = self.retire_epoch;
            if (is_gpu_init) {
                // Pre-stage instance data before the shadow pass: ShadowPass.prepare
                // snapshots the published render state (bounds/buffer/count),
                // so staging must run first or shadows lag one frame. Same scratch
                // and eye the first view queue would use; the frame guard keeps it
                // once per frame, shared by all view queues. Grown-away old
                // buffers retire into the epoch queue (P5), never destroyed inline.
                // Staging scratch is the back primary's list (view builds with
                // instances_prepared never retry mid-frame — failure coherence).
                if (self.frame_snapshot.has_camera) {
                    scene_instance_staging.stageInstances(.{
                        .allocator = self.allocator,
                        .instance_matrices = &back.primary.instance_matrices,
                        .thread_pool = jobs.global,
                        .frame_id = self.frame_id,
                        .eye = self.frame_snapshot.primary_cam.eye,
                        .retire_queue = &self.gpu_retire,
                    }, self.meshes.items);
                }
            }
        }

        // Shared builder — fallback only (stage-2B): when no fresh game build
        // exists, outline + shadow + view queues build here with exactly
        // today's values (&frame_snapshot, frame_id as cache_key, &self.stats,
        // snapshot eye, resolved sky/ibl, true, `.published`) — bit-identical
        // to the old inline path. When have_build the payload was already
        // built game-side (`.build_view` + patch above); rebuilding would
        // unfreeze the sets.
        if (!have_build) {
            const sky_tex = self.frame_snapshot.sky_texture orelse self.sky.texture;
            const ibl_int = self.frame_snapshot.ibl_intensity;
            self.buildQueuesInto(back, .{
                .snap = &self.frame_snapshot,
                .cache_key = self.frame_id,
                .stats = &self.stats,
                .eye = self.frame_snapshot.primary_cam.eye,
                .sky_texture = sky_tex,
                .ibl_intensity = ibl_int,
                .instances_prepared = true,
                .instance_source = .published,
            });
        }

        self.captureUiFrame();

        // P7 publish: one index flip, no list copies. Newest wins — a repeated
        // prepare's back overwrote nothing consumable until this point.
        self.draws.publish(back_idx);
        self.frame_prepared = true;
    }

    /// Stage 3, slice 2: update-phase light packing. Selects the top-k
    /// point/spot lights for the camera (advancing the incumbency-
    /// hysteresis fades with `dt`) and stores the uniform-ready pack.
    /// Called once per frame BEFORE render(); render consumes
    /// `snapshot.light_pack` (via frame_snapshot), never `self.light_pack`
    /// and never live light state — so the direct `self.light_pack` fallback
    /// below stays game/prepare-side ownership (update vs prepare excluded
    /// under phase_mutex) and needs no atomic mailbox. No new mailbox: the
    /// render reads the snapshot copy taken by packFrameSnapshot.
    pub fn updateLights(self: *Scene, dt: f32) void {
        // Zero-eye fallback keeps the pack defined for camera-less scenes
        // (render early-returns without a camera anyway).
        const eye = if (self.active_camera) |cam|
            cam.getPosition()
        else if (self.cameras.items.len > 0)
            self.cameras.items[0].camera.getPosition()
        else
            Vec3.zero;
        const pack = self.lights.packFrame(eye, self.shadows.enabled, dt);
        // Publish through the mailbox. Same-thread today (publish is
        // visible to this frame's render takeLatest); after the split the
        // same call sequence crosses the thread boundary unchanged.
        if (self.light_handoff.claim()) |i| {
            self.light_handoff.slot(i).* = pack;
            self.light_handoff.publish(i);
        } else {
            // Same saturation shape as publishFrameSnapshot: drain stale
            // published packs (consumer excluded by phase ownership) so the
            // newest pack wins; direct fallback only on a still-failed claim.
            self.light_handoff.releasePublished();
            if (self.light_handoff.claim()) |i| {
                self.light_handoff.slot(i).* = pack;
                self.light_handoff.publish(i);
            } else {
                // Both slots still published (consumer lagging): fall back to
                // the consumed copy as the carrier.
                self.light_pack = pack;
            }
        }
    }

    /// Stage 3, slice 2: the game-side update entry point. Everything the
    /// simulation advances per frame, in one call, in the canonical order
    /// (camera -> lights -> physics -> animations -> particles -> decals);
    /// render() then consumes the published frame values (light_pack,
    /// frame_snapshot, prepared draws/UI/debug/sky/particles) without
    /// simulating anything itself.
    ///
    /// Runs on the game side and MAY overlap render (update||render): it must
    /// touch ONLY update-owned state (live cameras/lights/world/meshes/
    /// materials/canvas/particles/trails/nav + the mailboxes +
    /// pending_update_ms). It must NEVER touch stats/Profiler/render-owned
    /// caches or prepared payloads, and never call sg.* (uploads flush on
    /// the context thread in prepare). Update-vs-prepare stay excluded
    /// under phase_mutex.
    ///
    /// Deliberately NOT included: `updateTrails` and `updateNavAgents` —
    /// both require real-seconds dt (the 60fps-normalized dt breaks their
    /// SI tuning), so apps drive them explicitly with their own time base.
    /// Those explicit CPU mutators run under the SAME update-vs-prepare
    /// exclusion as this entry point (game side, never concurrent with
    /// prepare; render sees only their staged/uploaded results) — the fact
    /// that `Scene.update` skips them is a dt-base distinction, not a
    /// thread-ownership one: no render path traces into their live state.
    ///
    /// Stage 1: after this (and any explicit mutators), the app may call
    /// `buildPreparedFrame` — still on the game side, still under the same
    /// exclusion — to stage the CPU halves the prepare latch will consume.
    /// Skipping the build is always legal: `prepareFrame` then runs the
    /// historical inline path.
    pub fn update(self: *Scene, dt: f32) particles.UpdateError!void {
        self.updateCamera(dt);
        self.updateLights(dt);
        self.updatePhysics(dt);
        self.updateAnimations(dt);
        try self.updateParticles(dt);
        self.updateDecals(dt);

        const cur_w = sapp.width();
        const cur_h = sapp.height();
        const aspect = if (cur_h > 0) @as(f32, @floatFromInt(cur_w)) / @as(f32, @floatFromInt(cur_h)) else 1.0;
        self.publishFrameSnapshot(aspect, cur_w, cur_h);
    }

    /// Stage 3: uploads the GPU buffers that the update phase staged
    /// (particle instances, trail geometry). Called at render start so
    /// every sg.* touch stays on the context thread; the update phase is
    /// free of sg.* calls.
    pub fn flushPendingGpuUploads(self: *Scene) void {
        gpu_thread.assertOnContextThread();
        // Deferred off-context destroys first: unlinking already happened in
        // destroyMesh, this completes the GPU teardown (deinit + free) for
        // entries whose epoch already completed (see GpuRetireQueue.flush).
        self.gpu_retire.flush(self.allocator);
        // Deferred off-context creations (uploadGeometry): finish the vertex/
        // index buffers before queue building can reference them. Plain scan
        // over meshes — the loop below already visits every mesh, so the
        // pending check adds no traversal, just one branch per mesh.
        for (self.meshes.items) |m| m.finishGpuUpload(self.allocator);
        for (self.particles.systems.items) |ps| ps.flushGpuUploads();
        for (self.trails.meshes.items) |tm| tm.flushGpuUploads();
        for (self.greased_lines.items) |gl| gl.flushGpuUploads();
        for (self.meshes.items) |m| m.flushGpuUploads();
    }

    /// Render entry: draws the frame prepared by prepareFrame (context
    /// thread, SEQUENTIAL with prepare — never concurrent; update MAY run
    /// concurrently on the game side). Reads ONLY render-owned captures
    /// (prepared draws, frame_snapshot incl. sky/default copies, ui_frame,
    /// debug capture + committed upload, particle prepared frame) plus
    /// BORROWED GPU handles under P3 epochs. No global phase lock is taken
    /// here: the app unlocks update-vs-prepare ownership BEFORE calling
    /// render (see main), and prepare already ran. Takes no sg.* outside the
    /// context thread (asserted). Every subsystem above is snapshot-driven;
    /// no live subsystem reads remain on this path.
    ///
    /// Non-blocking consumer: when the app skipped the phase-lock acquire it
    /// calls `renderReuse` instead, which re-draws the already-consumed front
    /// through this same function with `rendering_reuse` set — the
    /// `prepareFrame` fallback below is skipped and the profiler tail is
    /// suppressed while the stats are still accumulating inside; the
    /// `renderReuse` wrapper records the re-presented frame after restoring
    /// them. Every presented frame is recorded once, including reuses.
    pub fn render(self: *Scene) void {
        gpu_thread.assertOnContextThread();
        if (!self.frame_prepared and !self.rendering_reuse) {
            self.prepareFrame();
        }
        self.frame_prepared = false;
        // Конец кадра (P3): epoch, начатый в prepareFrame, закрывается на ВСЕХ
        // выходах render — включая ранний возврат без камеры ниже. Поэтому
        // epoch — на кадр, а не на камеру/view.
        defer self.gpu_retire.complete(self.retire_epoch);

        // P7: consume the published front slot (const payloads only). Valid
        // for this render; the next prepareFrame invalidates it.
        const draws = self.preparedDraws();
        const snap = &self.frame_snapshot;
        if (!snap.has_camera) {
            // Камеры нет — UI/debug-проходов не будет: переносим только
            // prepare-фазу динамики, чтобы счётчик не утёк в следующий кадр.
            self.stats.updated_bytes_frame = upload_meter.takeAndReset();
            return;
        }

        const cur_w = if (snap.screen_w > 0) snap.screen_w else sapp.width();
        const cur_h = if (snap.screen_h > 0) snap.screen_h else sapp.height();

        // Effective main-target MSAA sample count for this frame
        // (scene/msaa.zig holds the policy and the backend matrix).
        const samples = scene_msaa.effectiveSampleCount(snap.msaa_sample_count, .{
            .post_enabled = snap.post_process.enabled,
            .formats_msaa_capable = mainTargetFormatsMsaaCapable(),
            .backend = sg.queryBackend(),
        });
        if (snap.post_process.enabled and snap.msaa_sample_count > 1 and samples == 1) {
            // Only the runtime format gate can nullify a > 1 request here
            // (clamping lands on a valid count, post-off forces 1 upstream).
            _ = self.warn_msaa_format.warn(
                "msaa: x{} requested but the main target formats cannot MSAA on this backend; running 1x",
                .{snap.msaa_sample_count},
            );
        }

        // TAA sub-pixel jitter (context thread, render-owned): the snapshot
        // view_proj stays UNJITTERED (prepare built queues/culling from it,
        // i.e. the conservative unjittered frustum); the jittered matrix
        // below drives the main-pass draws and the postfx reprojection for
        // this frame so depth/color/history line up. The index is the
        // snapshot frame_id, so a reused frame repeats its jitter instead of
        // advancing history against identical content. Forced off under MSAA
        // (no depth resolve for the velocity term; PostFXStack forces the
        // composite side off the same way).
        const taa_on = snap.post_process.enabled and snap.post_process.taa_enabled and samples == 1;
        var taa_view_proj = snap.primary_cam.view_proj;
        if (taa_on) {
            const jpx = postprocess.taaJitter(snap.frame_id, snap.post_process.taa_jitter_scale);
            taa_view_proj = postprocess.applyTaaJitterToViewProj(snap.primary_cam.view_proj, jpx, cur_w, cur_h);
        }

        // 1. Directional Light Cascaded Shadow View-Projections
        const cascades = snap.cascades;
        const light_pack = snap.light_pack;

        // ==============================================
        // PASS 1: OFFSCREEN SHADOW DEPTH PASS
        // ==============================================
        if (snap.shadows_enabled) {
            const t_shadow = sokol.time.now();
            const shadow_draws = self.shadows.pass.renderPreparedFrom(
                &draws.shadow,
                cascades,
                light_pack.spot_shadows[0..light_pack.num_spot_shadows],
                light_pack.point_shadows[0..light_pack.num_point_shadows],
            );
            self.stats.shadow_draw_calls += shadow_draws;
            self.stats.draw_calls += shadow_draws;
            self.stats.shadow_ms = msSince(t_shadow);
        }

        // ==============================================
        // PASS 2: MAIN SCENE RENDER PASS
        // ==============================================
        var main_pass_action = sg.PassAction{};
        main_pass_action.colors[0] = .{
            .load_action = .CLEAR,
            .clear_value = .{
                .r = snap.clear_color.r,
                .g = snap.clear_color.g,
                .b = snap.clear_color.b,
                .a = snap.clear_color.a,
            },
        };
        main_pass_action.depth = .{
            .load_action = .CLEAR,
            .clear_value = 1.0,
            .store_action = .STORE,
        };

        // Offscreen target when post-processing is on, swapchain otherwise.
        const t_main = sokol.time.now();
        self.postfx.beginMainPass(main_pass_action, snap.post_process.enabled, samples, cur_w, cur_h);

        // Render-owned draw environment: every fallback below is a snapshot
        // COPY (default textures + sky captured at prepare) — the draw never
        // dereferences game-mutatable Scene.default_*_texture / sky fields,
        // so update may run concurrently with this whole pass. (The
        // default_material pointer is gone: null-material draws were already
        // baked into draw_record at prepare; the draw never needed it.)
        const env = scene_draw.Environment{
            // Pipeline set must match the main target's sample count: the
            // 1x set for the legacy/swapchain path, the MSAA twin otherwise.
            // Render-owned lazy caches (forward_msaa, clear_*, sky/debug/
            // particle/outline MSAA twins, postfx targets, shader-material
            // cache): touched ONLY on this context thread in prepare/render,
            // never by update — safe under overlap given the thread-safe
            // Scene allocator (GPA .thread_safe = true); no prewarm needed
            // just because the fields live in Scene.
            .pipelines = if (samples > 1) self.ensureForwardMsaa(samples) else &self.forward,
            .stats = &self.stats,
            .default_white = snap.default_white,
            .default_normal = snap.default_normal,
            .default_cube = snap.default_cube,
            .sky_texture = snap.sky_texture,
            .ibl_intensity = snap.ibl_intensity,
            .shadow_pass = &self.shadows.pass,
            .shadow_uniforms = snap.shadow_uniforms,
        };

        if (snap.enable_multi_camera and snap.camera_count > 0) {
            const active_idx = snap.active_camera_idx;
            const primary_snap = if (active_idx < snap.camera_count) snap.cameras[active_idx] else snap.primary_cam;
            const primary_rect = primary_snap.viewport.toPixelRect(cur_w, cur_h);
            sg.applyViewport(primary_rect.x, primary_rect.y, primary_rect.width, primary_rect.height, true);
            sg.applyScissorRect(primary_rect.x, primary_rect.y, primary_rect.width, primary_rect.height, true);
            // Only the TAA view (primary) is jittered; secondary views keep
            // their snapshot matrices.
            var primary_jittered = primary_snap;
            if (taa_on) primary_jittered.view_proj = taa_view_proj;
            self.renderSceneView(primary_jittered, &draws.primary, draws.outline_items.items, draws.outline_skins.items, samples, snap, env);

            for (snap.cameras[0..snap.camera_count], 0..) |entry, i| {
                if (i == active_idx or !entry.enabled) continue;
                const rect = entry.viewport.toPixelRect(cur_w, cur_h);
                sg.applyViewport(rect.x, rect.y, rect.width, rect.height, true);
                sg.applyScissorRect(rect.x, rect.y, rect.width, rect.height, true);

                if (entry.clear_viewport) {
                    const clr = entry.clear_color orelse snap.clear_color;
                    self.clearCurrentViewport(clr, samples);
                }

                self.renderSceneView(entry, &draws.views[i], draws.outline_items.items, draws.outline_skins.items, samples, snap, env);
            }
            // Restore full viewport
            sg.applyViewport(0, 0, cur_w, cur_h, true);
            sg.applyScissorRect(0, 0, cur_w, cur_h, true);
        } else {
            const vp = snap.primary_cam.viewport;
            const rect = vp.toPixelRect(cur_w, cur_h);
            sg.applyViewport(rect.x, rect.y, rect.width, rect.height, true);
            sg.applyScissorRect(rect.x, rect.y, rect.width, rect.height, true);

            var primary_jittered = snap.primary_cam;
            if (taa_on) primary_jittered.view_proj = taa_view_proj;
            self.renderSceneView(primary_jittered, &draws.primary, draws.outline_items.items, draws.outline_skins.items, samples, snap, env);

            if (rect.width != cur_w or rect.height != cur_h or rect.x != 0 or rect.y != 0) {
                sg.applyViewport(0, 0, cur_w, cur_h, true);
                sg.applyScissorRect(0, 0, cur_w, cur_h, true);
            }
        }

        if (!snap.post_process.enabled) {
            // P6: single fullscreen UI draw AFTER the per-view
            // viewport/scissor restoration above (both single- and
            // multi-camera paths restore before this point). Draw site
            // selection reads ONLY the prepared presence flag — never the
            // live canvas — so the snapshot boundary is complete at
            // prepare; the frame itself carries no canvas reference.
            // Counter semantics unchanged (including the historical +1
            // whenever a canvas existed at prepare, even for an empty
            // frame).
            if (self.ui_frame.canvas_present) {
                self.ui_frame.drawPrepared();
                self.stats.post_draw_calls += 1;
                self.stats.draw_calls += 1;
            }
        }

        sg.endPass();
        self.stats.main_ms = msSince(t_main);

        // ==============================================
        // PASS 2.5 (SSAO) + 2.75 (bloom) + 3 (composite & UI overlay)
        // ==============================================
        const t_post = sokol.time.now();
        self.postfx.renderChain(.{
            .post = snap.post_process,
            .ssao = snap.ssao,
            .camera = snap.primary_cam.camera,
            .aspect = snap.primary_cam.aspect,
            // Jittered when TAA is on (same matrix the color pass drew
            // with); otherwise exactly the snapshot matrix as before.
            .view_proj = taa_view_proj,
            .eye = snap.primary_cam.eye,
            .sun_dir = snap.sun_dir,
            .sun_color = snap.sun_color,
            .default_white_view = snap.default_white.view,
            .main_samples = samples,
            .ui = if (self.ui_frame.canvas_present) &self.ui_frame else null,
            .stats = &self.stats,
        }, cur_w, cur_h);

        sg.commit();
        self.stats.post_ms = msSince(t_post);

        // Перенос динамики в кадровую метрику: prepare-фаза (flush, стейджинг,
        // UI/debug upload'ы) уже накоплена в счётчике с prepareFrame, сюда
        // добавились только clear-append'ы main-прохода выше. После take
        // счётчик чист для следующего кадра.
        self.stats.updated_bytes_frame += upload_meter.takeAndReset();

        if (self.profiler.isRecording() and !self.rendering_reuse) {
            self.profiler_frame_seq +%= 1;
            self.profiler.recordFrame(self.profiler_frame_seq, &self.stats);
        }
    }

    /// True once a prepare published a frame (front slot `frame_id != 0`).
    /// The app blocks once for the first prepare while this is false instead
    /// of reusing: before the first successful prepare nothing is consumable,
    /// and reuse must not prepare without phase ownership (the game side may
    /// be mid-mutation).
    pub fn hasConsumableFrame(self: *const Scene) bool {
        return self.draws.slots[self.draws.front].frame_id != 0;
    }

    /// Non-blocking render-consumer reuse: re-draws the current front slot
    /// without a prepare, for frames where the app skipped the phase-lock
    /// acquire (lock contended) instead of stalling the present.
    ///
    /// Reuse contract: the caller reuses only when `hasConsumableFrame()`
    /// is true (debug assert below enforces it: front slot `frame_id != 0`,
    /// i.e. at least one prepare published a frame). The app skipped prepare
    /// because the phase lock was busy, so `frame_prepared` is false and the
    /// front slot holds the last consumed frame. Calling with a pending
    /// prepared frame (`frame_prepared` true) is a contract violation
    /// (debug assert) — consume pending frames with `render`, never
    /// `renderReuse`.
    ///
    /// No GpuRetire begin/flush runs here (no prepare): pending retire
    /// entries (queue + overflow[8]) stay queued until the next successful
    /// prepare's leading flush, which destroys only entries whose epoch
    /// already completed — the reused frame's borrowed handles stay valid
    /// precisely because no prepare ran. The inner render's
    /// `complete(retire_epoch)` re-run is idempotent (same epoch: no-op), as
    /// is the `frame_prepared = false` store; no other render step is
    /// prepare-frame-epoch dependent.
    ///
    /// Skip-streak retention (honest bound): every skipped prepare defers
    /// flush AND pending-upload completion, so `GpuRetireQueue.pending`
    /// grows with skip-streak × destroy-rate until the next successful
    /// prepare drains it (overflow[8] covers only OOM appends, not streak
    /// growth). Previously this was bounded by the phase-mutex coupling
    /// (every frame ran prepare+flush); with reuse the bound is the app's
    /// contract — do not streak reuse indefinitely — documented, not capped.
    /// Borrowed handles stay valid throughout the streak (no flush ran), and
    /// new meshes stay `gpu_pending` (invisible: queue builds skip them until
    /// a successful prepare completes their uploads) — staleness visible as
    /// missing objects, never corruption.
    ///
    /// Stats are saved and restored around the inner render because the frame
    /// was already recorded; the upload meter is still drained by the inner
    /// render and its tally discarded with the restored stats.
    ///
    /// Recording: every presented frame is recorded once, including reuses.
    /// After the restore above, while still recording, the seq is bumped and
    /// the re-shown frame recorded with the consumed frame's metrics
    /// (identical draws; wall pacing/timestamps are real). The inner render's
    /// own tail stays suppressed via `rendering_reuse`, so no double record.
    pub fn renderReuse(self: *Scene) void {
        gpu_thread.assertOnContextThread();
        std.debug.assert(!self.frame_prepared);
        std.debug.assert(self.draws.slots[self.draws.front].frame_id != 0);
        const saved_stats = self.stats;
        self.rendering_reuse = true;
        defer self.rendering_reuse = false;
        self.render();
        self.stats = saved_stats;
        if (self.profiler.isRecording()) {
            self.profiler_frame_seq +%= 1;
            self.profiler.recordFrame(self.profiler_frame_seq, &self.stats);
        }
    }

    pub fn deinit(self: *Scene) void {
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
        if (self.clear_vb.id != 0) sg.destroyBuffer(self.clear_vb);
        if (self.clear_pipeline.id != 0) sg.destroyPipeline(self.clear_pipeline);
        if (self.clear_pipeline_msaa.id != 0) sg.destroyPipeline(self.clear_pipeline_msaa);
        if (self.clear_shader.id != 0) sg.destroyShader(self.clear_shader);

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

        scene_content.deinitMeshes(self.allocator, &self.meshes);
        scene_content.deinitMaterials(self.allocator, &self.materials);
        scene_content.deinitPbrMaterials(self.allocator, &self.pbr_materials);
        scene_content.deinitShaderMaterials(self.allocator, &self.shader_materials);

        self.lights.deinit(self.allocator);

        self.draws.deinit(self.allocator);

        self.trails.deinit(self.allocator);
        for (self.greased_lines.items) |gl| gl.deinit();
        self.greased_lines.deinit(self.allocator);
        self.nav.deinit(self.allocator);

        self.default_white_texture.deinit();
        self.default_normal_texture.deinit();
        self.default_cube_texture.deinit();

        self.shadows.deinit();
        self.sky.deinit();

        self.forward.deinit();
        if (self.forward_msaa) |*fw| fw.deinit();
        self.forward_msaa = null;

        scene_content.deinitAnimations(self.allocator, &self.animation_groups, &self.skeletons);

        self.outline_meshes.deinit(self.allocator);
        self.postfx.deinit();

        self.particles.deinit(self.allocator);

        if (self.ui_canvas) |*u| {
            u.deinit();
        }
        // Borrowed GPU IDs only — frees the frame-owned CPU copies.
        self.ui_frame.deinit(self.allocator);
    }

    // ---- Profiling & Diagnostics API ----
    //
    // Ownership (actual update||render): the Profiler is render-owned.
    // start/stop/reset/recordFrame/save* run on the CONTEXT thread between
    // submissions (the F8 window callback shares that thread — no concurrent
    // render there; no broad IO-backend refactor). recordFrame is called only
    // at the end of render. captureMemorySnapshot reads LIVE registries
    // (meshes/materials/textures), so it additionally requires update
    // exclusion (phase lock held) — never from a worker, never concurrently
    // with update. The game/update side never touches the Profiler.

    /// Starts recording per-frame performance metrics.
    pub fn startProfiling(self: *Scene) void {
        self.profiler.start();
    }

    /// Stops recording per-frame performance metrics.
    pub fn stopProfiling(self: *Scene) void {
        self.profiler.stop();
    }

    /// Clears any recorded frame history.
    pub fn resetProfiling(self: *Scene) void {
        self.profiler.reset();
    }

    /// Returns true if profiling is currently active.
    pub fn isProfiling(self: *const Scene) bool {
        return self.profiler.isRecording();
    }

    /// Captures a snapshot of current memory allocations (CPU objects & GPU VRAM).
    /// Context thread + update excluded (phase lock): reads live registries.
    pub fn captureMemorySnapshot(self: *Scene) !*const profiler_mod.MemorySnapshot {
        return self.profiler.captureMemorySnapshot(self);
    }

    /// Saves an interactive HTML report to `path`.
    pub fn saveProfileReportHtml(self: *Scene, path: []const u8) !void {
        try self.profiler.saveReportHtml(self, path);
    }

    /// Saves a Markdown report to `path`.
    pub fn saveProfileReportMd(self: *Scene, path: []const u8) !void {
        try self.profiler.saveReportMd(self, path);
    }

    /// Saves Chrome Trace Event JSON to `path`.
    pub fn saveProfileTraceJson(self: *Scene, path: []const u8) !void {
        try self.profiler.saveTraceJson(path);
    }

    /// Saves all reports (HTML, Markdown, Chrome Trace JSON) to `<base_path>.html`,
    /// `<base_path>.md`, and `<base_path>.json`.
    pub fn saveProfileReports(self: *Scene, base_path: []const u8) !void {
        try self.profiler.saveReports(self, base_path);
    }
};
