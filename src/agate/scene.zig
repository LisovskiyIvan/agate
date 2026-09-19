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
    /// so threaded applications must keep coarse phase ownership through the
    /// complete render until a full per-item render payload exists.
    frame_handoff: handoff_mod.Handoff(scene_snapshot.SceneFrameSnapshot, 2) = .{},
    /// Last consumed frame snapshot.
    frame_snapshot: scene_snapshot.SceneFrameSnapshot = .{},
    /// Flag indicating whether prepareFrame() has already run for this frame.
    frame_prepared: bool = false,

    /// Stage 1 producer-build handoff (game/update phase → prepare latch):
    /// `buildPreparedFrame` (game side, CPU-only, sg-free) stages instance
    /// matrices into the back-slot scratch + per-mesh previews and captures
    /// the particle/physics CPU build frames, then bumps `build_seq`.
    /// `prepareFrame` (context side) consumes the build when `build_seq !=
    /// last_latched_seq` (GPU halves + latch copies, no CPU restaging) and
    /// otherwise runs the historical inline path — so behavior stays correct
    /// when `buildPreparedFrame` is never called. Two builds before a latch:
    /// newest wins (single preview store recomputed, scratch overwritten).
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

    fn prepareViewQueues(
        self: *Scene,
        queues: *scene_render_queue.RenderQueues,
        cam_snap: scene_snapshot.CameraSnapshot,
        sky_texture: ?CubeTexture,
        ibl_intensity: f32,
    ) void {
        queues.reset();

        scene_render_queue.buildFrameQueues(.{
            .allocator = self.allocator,
            .meshes = self.meshes.items,
            .frame_id = self.frame_id,
            .view_proj = cam_snap.view_proj,
            .eye = cam_snap.eye,
            .cull_frustum = self.enable_frustum_culling,
            .cull_occlusion = self.enable_occlusion_culling,
            .culling_mask = cam_snap.culling_mask,
            .occlusion_culler = &self.occlusion_culler,
            .stats = &self.stats,
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
            .instances_prepared = true,
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
            scene_draw.drawRegularItem(&env, item, &frame_ctx, &current_pipeline_id, queues.skin_storage.items, queues.shader_storage.items);
        }

        // Opaque instanced meshes.
        for (queues.opaque_instanced.items) |batch| {
            scene_draw.drawInstancedBatch(&env, batch, &frame_ctx, &current_pipeline_id);
        }

        // Transparent pass: regular items and instanced groups interleaved in
        // one global back-to-front order (each instanced group draws as a
        // single batch at its sorted position; no per-instance sorting).
        for (queues.transparent_order.items) |entry| {
            switch (entry.kind) {
                .regular => {
                    if (entry.index < queues.transparent.items.len) {
                        scene_draw.drawRegularItem(&env, queues.transparent.items[entry.index], &frame_ctx, &current_pipeline_id, queues.skin_storage.items, queues.shader_storage.items);
                    }
                },
                .instanced => {
                    if (entry.index < queues.transparent_instanced.items.len) {
                        scene_draw.drawInstancedBatch(&env, queues.transparent_instanced.items[entry.index], &frame_ctx, &current_pipeline_id);
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
    /// during the render call consuming this frame — never across a prepare
    /// boundary. Render completes the frame epoch on all returns (no-camera
    /// too). One pending frame, no concurrent prepare/render — but update
    /// CAN overlap render (update-vs-prepare stay excluded instead).
    pub fn preparedDraws(self: *const Scene) *const FrameDrawSlot {
        return &self.draws.slots[self.draws.front];
    }

    /// Stage 1 producer build (game/update phase, CPU-only, sg-free): stages
    /// the CPU halves the prepare latch will consume — instance matrices
    /// into the back-slot scratch + per-mesh previews, the particle build
    /// frame, the physics debug build capture — and bumps `build_seq`.
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
    /// Touches NOTHING else: no sg.*, no UI canvas, no P7 queues/shadow/
    /// outline (those stay prepare-time), no GpuRetire begin/complete/flush,
    /// no frame_id/retire_epoch, no stats, no profiler. Starts by resetting
    /// the back slot (the latch consumes it without reset; the consumed
    /// slot becomes front and is reset again by the next build or the
    /// inline fallback). The transparent-sort eye comes from the live
    /// cameras (freshest post-mutation value; the inline fallback uses the
    /// snapshot eye — sort order may differ between paths, counts/bounds
    /// never do).
    ///
    /// App contract: no latch is possible while a build runs (update-vs-
    /// prepare exclusion), and no mesh is created/destroyed between a build
    /// and its latch. A mesh whose upload finishes between build and latch,
    /// or whose segment OOMs, keeps its previous complete `instance_render`
    /// for one frame (documented, coherent); the next funded build+latch
    /// picks it up.
    pub fn buildPreparedFrame(self: *Scene) void {
        // Deliberately NO gpu_thread assert: this runs on the game side or
        // a spawned worker. Everything below is sg-free (the CPU staging
        // half, the plain captures); any sg.* here would be a bug.
        self.build_seq +%= 1;
        const seq = self.build_seq;
        const back_idx = self.draws.backIndex();
        const back = &self.draws.slots[back_idx];
        back.reset();
        self.build_slot = back_idx;
        // Freshest post-mutation eye (same source shape as updateLights):
        // the live camera, else the first registered camera, else zero.
        const eye = if (self.active_camera) |cam|
            cam.getPosition()
        else if (self.cameras.items.len > 0)
            self.cameras.items[0].camera.getPosition()
        else
            Vec3.zero;
        scene_instance_staging.stageInstancesCpu(.{
            .allocator = self.allocator,
            .scratch = &back.primary.instance_matrices,
            .thread_pool = jobs.global,
            .eye = eye,
        }, self.meshes.items, seq);
        self.particles.buildCapture(self.allocator, seq);
        self.physics.buildDebug(self.allocator, seq);
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

        var snap = self.frame_snapshot;
        if (self.frame_handoff.takeLatest(&snap)) {
            self.frame_snapshot = snap;
        } else {
            const cur_w = sapp.width();
            const cur_h = sapp.height();
            const aspect = if (cur_h > 0) @as(f32, @floatFromInt(cur_w)) / @as(f32, @floatFromInt(cur_h)) else 1.0;
            self.frame_snapshot = self.packFrameSnapshot(aspect, cur_w, cur_h);
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
            // Stage 1 latch: the game-side build already reset this back
            // slot and staged the instance scratch into it — reset-before-
            // consume would erase the build, so only stamp the prepare-owned
            // frame/epoch (reset by `buildPreparedFrame`, rebuilt here by
            // the queue/shadow/outline builds below) and run the GPU halves
            // over the previews + scratch. Stale previews (OOM-skipped,
            // post-build meshes) keep their previous complete
            // `instance_render` — no partial publish. Queues/outline/shadow
            // below rebuild exactly as in the inline path.
            std.debug.assert(self.build_slot == back_idx);
            back.frame_id = self.frame_id;
            back.retire_epoch = self.retire_epoch;
            if (is_gpu_init) {
                // Same position/gate as the historical pre-stage: the shadow
                // pass snapshots the published render state, so the GPU half
                // must run first or shadows lag one frame. Same retire queue
                // and eye source shape (the eye itself was consumed at build
                // time for the transparent sort; only the guard reads the
                // snapshot here).
                if (self.frame_snapshot.has_camera) {
                    scene_instance_staging.stageInstancesLatch(.{
                        .allocator = self.allocator,
                        .frame_id = self.frame_id,
                        .retire_queue = &self.gpu_retire,
                        .build_seq = self.build_seq,
                    }, self.meshes.items, &back.primary.instance_matrices);
                }
            }
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

        // P5: outline capture is unconditional, as before — immediately
        // after the conditional pre-stage above and before conditional
        // shadow/view warming below. With a GPU context instanced items
        // snapshot this frame's staged bounds/count/handle; regular
        // items are captured before worldMatrixCached warming, exactly
        // like the historical pre-queue capture, so their cached-center
        // behavior is unchanged.
        for (self.outline_meshes.items) |m| {
            if (m.gpu_pending or !m.is_visible or m.index_count == 0) continue;
            if (outline_pass.makeOutlineDrawItem(self.allocator, &back.outline_skins, m)) |it| {
                back.outline_items.append(self.allocator, it) catch {};
            }
        }

        if (is_gpu_init) {
            // Shadow pass preparation into the back slot (disabled shadows —
            // or no camera — leave the reset-empty payload: coherent, never
            // the front slot's prior bins).
            if (self.frame_snapshot.has_camera and self.frame_snapshot.shadows_enabled and self.shadows.enabled) {
                _ = self.shadows.pass.prepareInto(&back.shadow, self.meshes.items, self.frame_id, jobs.global);
            }

            // View queues preparation
            const sky_tex = self.frame_snapshot.sky_texture orelse self.sky.texture;
            const ibl_int = self.frame_snapshot.ibl_intensity;
            if (self.frame_snapshot.has_camera) {
                if (self.frame_snapshot.enable_multi_camera and self.frame_snapshot.camera_count > 0) {
                    const active_idx = self.frame_snapshot.active_camera_idx;
                    const primary_snap = if (active_idx < self.frame_snapshot.camera_count) self.frame_snapshot.cameras[active_idx] else self.frame_snapshot.primary_cam;
                    self.prepareViewQueues(&back.primary, primary_snap, sky_tex, ibl_int);
                    for (self.frame_snapshot.cameras[0..self.frame_snapshot.camera_count], 0..) |entry, i| {
                        if (i == active_idx or !entry.enabled) continue;
                        self.prepareViewQueues(&back.views[i], entry, sky_tex, ibl_int);
                    }
                } else {
                    self.prepareViewQueues(&back.primary, self.frame_snapshot.primary_cam, sky_tex, ibl_int);
                }
            }
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
    pub fn render(self: *Scene) void {
        gpu_thread.assertOnContextThread();
        if (!self.frame_prepared) {
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
            self.renderSceneView(primary_snap, &draws.primary, draws.outline_items.items, draws.outline_skins.items, samples, snap, env);

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

            self.renderSceneView(snap.primary_cam, &draws.primary, draws.outline_items.items, draws.outline_skins.items, samples, snap, env);

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
            .view_proj = snap.primary_cam.view_proj,
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

        if (self.profiler.isRecording()) {
            self.profiler.recordFrame(self.frame_id, &self.stats);
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

test "Scene camera switching and cycling" {
    const ally = std.testing.allocator;
    var scene: Scene = undefined;
    scene.allocator = ally;
    scene.cameras = .empty;
    scene.active_camera_index = null;
    scene.active_camera = null;
    scene.active_camera_owned_name = null;
    scene.enable_multi_camera = true;

    defer {
        for (scene.cameras.items) |entry| {
            if (entry.owns_name) ally.free(entry.name);
        }
        scene.cameras.deinit(ally);
    }

    const cam1 = Camera{ .free = camera_mod.FreeCamera.init("Cam1", .{}) };
    const cam2 = Camera{ .arc_rotate = camera_mod.ArcRotateCamera.init("Cam2", .{}) };
    const cam3 = Camera{ .fly = camera_mod.FlyCamera.init("Cam3", .{}) };

    const idx0 = try scene.addCamera(.{ .name = "Cam1", .camera = cam1 });
    try std.testing.expectEqual(@as(usize, 0), idx0);
    try std.testing.expectEqual(@as(usize, 0), scene.getActiveCameraIndex().?);
    try std.testing.expectEqualStrings("Cam1", scene.getActiveCameraName().?);

    _ = try scene.addCamera(.{ .name = "Cam2", .camera = cam2 });
    _ = try scene.addCamera(.{ .name = "Cam3", .camera = cam3 });
    try std.testing.expectEqual(@as(usize, 3), scene.getCameraCount());

    // Cycle next: 0 -> 1 -> 2 -> 0
    scene.nextCamera();
    try std.testing.expectEqual(@as(usize, 1), scene.getActiveCameraIndex().?);
    try std.testing.expectEqualStrings("Cam2", scene.getActiveCameraName().?);

    scene.nextCamera();
    try std.testing.expectEqual(@as(usize, 2), scene.getActiveCameraIndex().?);
    try std.testing.expectEqualStrings("Cam3", scene.getActiveCameraName().?);

    scene.nextCamera();
    try std.testing.expectEqual(@as(usize, 0), scene.getActiveCameraIndex().?);

    // Cycle prev: 0 -> 2 -> 1
    scene.prevCamera();
    try std.testing.expectEqual(@as(usize, 2), scene.getActiveCameraIndex().?);

    // Switch by name
    try std.testing.expect(scene.switchCameraByName("Cam2"));
    try std.testing.expectEqual(@as(usize, 1), scene.getActiveCameraIndex().?);
    try std.testing.expect(!scene.switchCameraByName("NonExistent"));
}

test "updateLights packs point lights into the frame payload" {
    const alloc = std.testing.allocator;
    var scene = @import("testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);

    _ = try scene.createPointLight("probe", .{
        .position = Vec3.zero,
        .intensity = 10.0,
        .range = 10.0,
    });

    // One point light packed into slot 0 (the light sits at the pack eye,
    // so its score is maximal). Hysteresis fades a newly seen light in
    // over fade_time (0.25 s), so the first pack carries a partial factor.
    // The pack travels through the mailbox: render-side consumption goes
    // via takeLatest, exactly what the render pass does.
    scene.updateLights(0.016);
    var pack_out: scene_lights.LightRig.FramePack = undefined;
    try std.testing.expect(scene.light_handoff.takeLatest(&pack_out));
    // First sight: packed with a partial enter-fade factor (intensity
    // lane = intensity * factor < full intensity).
    try std.testing.expectEqual(@as(f32, 1.0), pack_out.counts[0]);
    try std.testing.expect(pack_out.point_color_int[0][3] < 10.0);
    var frame: usize = 0;
    while (frame < 32) : (frame += 1) {
        scene.updateLights(0.016);
        // Consume per frame like render does: a fresh pack lands each
        // update, so the mailbox never lags.
        _ = scene.light_handoff.takeLatest(&pack_out);
    }
    // Fully faded in (0.512 s >= fade_time): the intensity lane carries
    // the light's exact intensity, and the incumbent keeps its slot.
    try std.testing.expectEqual(@as(f32, 1.0), pack_out.counts[0]);
    try std.testing.expectEqual(@as(f32, 10.0), pack_out.point_color_int[0][3]);
    // A second take without a new publish consumes nothing (last pack
    // stands on the render side) — the PIP multi-render pattern.
    try std.testing.expect(!scene.light_handoff.takeLatest(&pack_out));
}

test "publishFrameSnapshot and prepareFrame snapshot handoff" {
    const alloc = std.testing.allocator;
    var scene = @import("testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);

    const cam = Camera{ .free = camera_mod.FreeCamera.init("Cam1", .{}) };
    _ = try scene.addCamera(.{ .name = "Cam1", .camera = cam });

    scene.publishFrameSnapshot(16.0 / 9.0, 1920, 1080);
    try std.testing.expect(!scene.frame_prepared);

    scene.prepareFrame();
    try std.testing.expect(scene.frame_prepared);
    try std.testing.expect(scene.frame_snapshot.has_camera);
    try std.testing.expectEqual(@as(i32, 1920), scene.frame_snapshot.screen_w);
    try std.testing.expectEqual(@as(i32, 1080), scene.frame_snapshot.screen_h);
}

test "saturated frame mailbox keeps the newest snapshot" {
    const alloc = std.testing.allocator;
    var scene = @import("testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);

    // Reproducer for the fallback-overwrite bug: three publishes without a
    // consuming prepareFrame saturate the 2-slot mailbox. The third publish
    // must drain the stale slots and land, so prepareFrame takes 300 — not
    // an older published frame over the newer fallback. packFrameSnapshot
    // sets screen_w before the no-camera early-out, so no camera is needed.
    scene.publishFrameSnapshot(1.0, 100, 100);
    scene.publishFrameSnapshot(1.0, 200, 200);
    scene.publishFrameSnapshot(1.0, 300, 300);

    scene.prepareFrame();
    try std.testing.expect(scene.frame_prepared);
    try std.testing.expectEqual(@as(i32, 300), scene.frame_snapshot.screen_w);
    try std.testing.expectEqual(@as(i32, 300), scene.frame_snapshot.screen_h);
}

test "saturated light mailbox keeps the newest pack" {
    const alloc = std.testing.allocator;
    var scene = @import("testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);

    // Same saturation shape for lights: fill both slots with a sentinel,
    // then update through the saturated mailbox. updateLights must drain the
    // stale sentinels and publish the fresh pack, so takeLatest never
    // resurfaces one.
    var sentinel = std.mem.zeroes(scene_lights.LightRig.FramePack);
    sentinel.counts[0] = 7.0;
    for (0..2) |_| {
        const i = scene.light_handoff.claim().?;
        scene.light_handoff.slot(i).* = sentinel;
        scene.light_handoff.publish(i);
    }
    try std.testing.expect(scene.light_handoff.claim() == null);

    scene.updateLights(0.016);
    var pack_out: scene_lights.LightRig.FramePack = undefined;
    try std.testing.expect(scene.light_handoff.takeLatest(&pack_out));
    // Fresh pack from the light-less rig: zero lights, not the sentinel.
    try std.testing.expectEqual(@as(f32, 0.0), pack_out.counts[0]);
    try std.testing.expect(!scene.light_handoff.takeLatest(&pack_out));
}

test "destroyMesh removes the physics body" {
    const alloc = std.testing.allocator;
    var scene = @import("testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.physics.deinit(alloc);
    defer scene.meshes.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);
    defer scene.outline_meshes.deinit(alloc);

    const m = try alloc.create(Mesh);
    m.* = @import("testing.zig").testMesh("body_mesh");
    try scene.meshes.append(alloc, m);

    _ = try scene.createRigidBody(m, .box, 1.0);
    try std.testing.expect(scene.getRigidBody(m) != null);

    scene.destroyMesh(m);
    try std.testing.expectEqual(@as(usize, 0), scene.meshes.items.len);
    try std.testing.expectEqual(@as(usize, 0), scene.physics.getWorld().?.bodies.items.len);
}

test "destroyMesh orphans children and detaches bone links" {
    const alloc = std.testing.allocator;
    var scene = @import("testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.meshes.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);
    defer scene.outline_meshes.deinit(alloc);

    const parent = try alloc.create(Mesh);
    parent.* = @import("testing.zig").testMesh("parent");
    try scene.meshes.append(alloc, parent);
    const child = try alloc.create(Mesh);
    child.* = @import("testing.zig").testMesh("child");
    child.parent = parent;
    try scene.meshes.append(alloc, child);
    const attached = try alloc.create(Mesh);
    attached.* = @import("testing.zig").testMesh("attached");
    attached.attachToBone(parent, 2);
    try scene.meshes.append(alloc, attached);

    scene.destroyMesh(parent);
    try std.testing.expect(child.parent == null);
    try std.testing.expect(attached.attach_bone == null);
    // No cascade: the orphans stay alive under their own transform.
    try std.testing.expectEqual(@as(usize, 2), scene.meshes.items.len);

    scene.destroyMesh(child);
    scene.destroyMesh(attached);
    try std.testing.expectEqual(@as(usize, 0), scene.meshes.items.len);
}

test "destroyMesh removes LOD entries preserving order" {
    const alloc = std.testing.allocator;
    var scene = @import("testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.meshes.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);
    defer scene.outline_meshes.deinit(alloc);

    const parent = try alloc.create(Mesh);
    parent.* = @import("testing.zig").testMesh("lod_parent");
    try scene.meshes.append(alloc, parent);
    const c1 = try alloc.create(Mesh);
    c1.* = @import("testing.zig").testMesh("lod_c1");
    try scene.meshes.append(alloc, c1);
    const c2 = try alloc.create(Mesh);
    c2.* = @import("testing.zig").testMesh("lod_c2");
    try scene.meshes.append(alloc, c2);
    const c3 = try alloc.create(Mesh);
    c3.* = @import("testing.zig").testMesh("lod_c3");
    try scene.meshes.append(alloc, c3);

    try parent.addLODLevel(alloc, 10.0, c1);
    try parent.addLODLevel(alloc, 20.0, c2);
    try parent.addLODLevel(alloc, 30.0, c3);

    scene.destroyMesh(c2);
    try std.testing.expectEqual(@as(usize, 2), parent.lod_levels.items.len);
    try std.testing.expectEqual(@as(f32, 10.0), parent.lod_levels.items[0].distance);
    try std.testing.expect(parent.lod_levels.items[0].mesh.? == c1);
    try std.testing.expectEqual(@as(f32, 30.0), parent.lod_levels.items[1].distance);
    try std.testing.expect(parent.lod_levels.items[1].mesh.? == c3);

    scene.destroyMesh(c1);
    scene.destroyMesh(c3);
    scene.destroyMesh(parent);
    try std.testing.expectEqual(@as(usize, 0), scene.meshes.items.len);
}

test "destroyMesh drops decal instances and frees material" {
    const alloc = std.testing.allocator;
    var scene = @import("testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.decals.deinit();
    defer scene.meshes.deinit(alloc);
    defer scene.pbr_materials.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);
    defer scene.outline_meshes.deinit(alloc);

    // Manual instance (no projection/GPU): the manager only stores the
    // mesh/material pair, so removal + material teardown is fully
    // exercisable without projecting any decal geometry.
    const dm = scene.getOrCreateDecalManager(8);
    const dm_mesh = try alloc.create(Mesh);
    dm_mesh.* = @import("testing.zig").testMesh("decal_mesh");
    try scene.meshes.append(alloc, dm_mesh);
    const mat = try scene.createPBRMaterial("decal_mat");
    try dm.instances.append(alloc, .{
        .mesh = dm_mesh,
        .material = mat,
        .base_color = Color3.white,
        .lifetime = 0.0,
        .fade_duration = 1.0,
    });

    scene.destroyMesh(dm_mesh);
    try std.testing.expectEqual(@as(usize, 0), dm.instances.items.len);
    try std.testing.expectEqual(@as(usize, 0), scene.pbr_materials.items.len);
    try std.testing.expectEqual(@as(usize, 0), scene.meshes.items.len);
}

test "pending destroy overflow drains on flush" {
    const alloc = std.testing.allocator;
    var scene = @import("testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.meshes.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);

    // Drive the allocation-free spillover directly: the off-context OOM
    // enqueue itself needs fault injection, but the drain path (flush and
    // deinit share it) is covered here with a buffer-free mesh, so no sg.*
    // call is involved. Запись помечена завершённым epoch, чтобы flush её
    // забрал — правило epoch относится и к overflow.
    const m = try alloc.create(Mesh);
    m.* = @import("testing.zig").testMesh("overflow_mesh");
    const e = scene.gpu_retire.begin();
    scene.gpu_retire.overflow[0] = .{ .kind = .mesh, .mesh = m, .epoch = e };
    scene.gpu_retire.overflow_len = 1;
    scene.gpu_retire.complete(e);

    scene.flushPendingGpuUploads();
    try std.testing.expectEqual(@as(usize, 0), scene.gpu_retire.retainedCount());
}

test "destroyMesh вне контекста: ретенция + flush на контекстном потоке" {
    const alloc = std.testing.allocator;
    // Маркер идемпотентен: главный поток тестов уже помечен gpu_thread-тестами
    // (идут раньше по реестру), воркер ниже всё равно чужой. Без маркера тест
    // был бы синхронным и бессмысленным.
    gpu_thread.markContextThread();
    var scene = @import("testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.meshes.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);
    defer scene.outline_meshes.deinit(alloc);

    const m = try alloc.create(Mesh);
    m.* = @import("testing.zig").testMesh("offctx_mesh");
    try scene.meshes.append(alloc, m);

    // Воркер — не-контекстный поток: destroyMesh обязан только отвязать меш
    // (физика/иерархия/LOD зачищены) и положить его в ретенцию, без sg.*.
    const Job = struct {
        scene: *Scene,
        mesh: *Mesh,
        fn run(j: @This()) void {
            j.scene.destroyMesh(j.mesh);
        }
    };
    const t = try std.Thread.spawn(.{}, Job.run, .{Job{ .scene = &scene, .mesh = m }});
    t.join();
    try std.testing.expectEqual(@as(usize, 0), scene.meshes.items.len);
    try std.testing.expectEqual(@as(usize, 1), scene.gpu_retire.retainedCount());

    // Конец кадра + flush на контекстном потоке: ретенция освобождена.
    scene.gpu_retire.complete(scene.gpu_retire.current());
    scene.flushPendingGpuUploads();
    try std.testing.expectEqual(@as(usize, 0), scene.gpu_retire.retainedCount());
}

test "prepareFrame stages an empty upload tally without an upload queue" {
    const alloc = std.testing.allocator;
    var scene = @import("testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);

    // No uploads queue and no io runner on the fixture: prepareFrame takes
    // the synchronous path and stages a zero tally for the stats publish.
    scene.prepareFrame();
    try std.testing.expect(scene.frame_prepared);
    try std.testing.expectEqual(@as(usize, 0), scene.frame_uploads.count);
    try std.testing.expectEqual(@as(u64, 0), scene.frame_uploads.bytes);
    // Без GPU-контекста динамических апдейтов нет: метрика нулевая,
    // счётчик meter сброшен в начале prepareFrame.
    try std.testing.expectEqual(@as(u64, 0), scene.stats.updated_bytes_frame);
    try std.testing.expectEqual(@as(u64, 0), upload_meter.peek());
}

test "prepareFrame rebuilds outline snapshots without GPU" {
    const alloc = std.testing.allocator;
    var scene = @import("testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.outline_meshes.deinit(alloc);
    defer scene.draws.deinit(alloc);

    // No GPU context on the fixture (default textures zeroed): outline
    // capture still runs unconditionally after the (skipped) pre-stage, as
    // before P5 — snapshots clear and rebuild every prepareFrame (P7: into
    // the published slot, read via preparedDraws()).
    var mesh = Mesh{
        .name = "headless_outline",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .position = Vec3.new(4, 0, 0),
    };
    try scene.outline_meshes.append(alloc, &mesh);

    scene.prepareFrame();
    try std.testing.expect(scene.frame_prepared);
    try std.testing.expectEqual(@as(usize, 1), scene.preparedDraws().outline_items.items.len);
    // Rebuild, not accumulate.
    scene.prepareFrame();
    try std.testing.expectEqual(@as(usize, 1), scene.preparedDraws().outline_items.items.len);
    // Clearing works: a hidden mesh rebuilds to empty.
    mesh.is_visible = false;
    scene.prepareFrame();
    try std.testing.expectEqual(@as(usize, 0), scene.preparedDraws().outline_items.items.len);
}

test "prepareFrame transfers staged update tick, preserves prepare_ms" {
    const alloc = std.testing.allocator;
    var scene = @import("testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);

    // Игровая фаза сложила тик через recordUpdateTime (НЕ в stats),
    // прошлый кадр оставил счётчики и post_ms: сброс prepareFrame обязан
    // перенести staged update_ms и сохранить prepare_ms, остальное обнулить.
    // Прямая запись stats.update_ms с update-стороны запрещена (контракт
    // recordUpdateTime): этот тест пишет только staged поле + prepare_ms.
    scene.recordUpdateTime(2.5);
    scene.stats.prepare_ms = 1.25;
    scene.stats.draw_calls = 41;
    scene.stats.triangles = 1000;
    scene.stats.post_ms = 3.0;

    scene.prepareFrame();

    try std.testing.expectEqual(@as(f32, 2.5), scene.stats.update_ms);
    try std.testing.expectEqual(@as(f32, 1.25), scene.stats.prepare_ms);
    try std.testing.expectEqual(@as(u32, 0), scene.stats.draw_calls);
    try std.testing.expectEqual(@as(u32, 0), scene.stats.triangles);
    try std.testing.expectEqual(@as(f32, 0.0), scene.stats.post_ms);
    // prepare_ms следующего кадра app перезапишет поверх после prepareFrame
    // (frame() в main) — handoff не мешает новому замеру.
}

test "recordUpdateTime stages without touching stats (update||render disjoint)" {
    const alloc = std.testing.allocator;
    var scene = @import("testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);

    scene.prepareFrame();
    try std.testing.expectEqual(@as(f32, 0.0), scene.stats.update_ms);

    // Update-сторона (game thread): только staged поле, stats не тронуты —
    // render может читать stats конкурентно.
    scene.recordUpdateTime(7.5);
    try std.testing.expectEqual(@as(f32, 7.5), scene.pending_update_ms);
    try std.testing.expectEqual(@as(f32, 0.0), scene.stats.update_ms);

    // Последний тик wins до prepare; prepare переносит его в stats.
    scene.recordUpdateTime(8.25);
    scene.prepareFrame();
    try std.testing.expectEqual(@as(f32, 8.25), scene.stats.update_ms);
}

test "async save/load report NoTaskRunner without an io runner" {
    const alloc = std.testing.allocator;
    var scene = @import("testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);

    // The fixture has no io_runner: capture succeeds, dispatch fails with
    // NoTaskRunner (and frees the snapshot — testing.allocator verifies).
    try std.testing.expectError(error.NoTaskRunner, scene.saveStateFileAsync("no_runner.agsc"));
    try std.testing.expectError(error.NoTaskRunner, scene.loadStateFileAsync("no_runner.agsc"));
}

test "P6: prepareFrame captures UI into the render-owned frame" {
    const alloc = std.testing.allocator;
    var scene = @import("testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.ui_frame.deinit(alloc);

    // Headless canvas (no GPU init): draws only fill CPU-side lists.
    scene.ui_canvas = UICanvas{
        .allocator = alloc,
        .font_texture = std.mem.zeroes(Texture),
    };
    defer {
        if (scene.ui_canvas) |*c| {
            c.vertices.deinit(alloc);
            c.indices.deinit(alloc);
        }
        scene.ui_canvas = null;
    }
    const canvas_ui = &scene.ui_canvas.?;
    canvas_ui.drawRect(10, 20, 30, 40, Color4.white);
    canvas_ui.drawText("ok", 0, 0, 16.0, Color4.white);

    const cam = Camera{ .free = camera_mod.FreeCamera.init("Cam1", .{}) };
    _ = try scene.addCamera(.{ .name = "Cam1", .camera = cam });
    scene.publishFrameSnapshot(16.0 / 9.0, 1920, 1080);

    scene.prepareFrame();
    try std.testing.expect(scene.frame_prepared);
    // Frame owns copies of the full geometry; dims come from the scene
    // snapshot (not live sapp values); borrowed handle IDs are captured.
    try std.testing.expect(scene.ui_frame.has_capture);
    try std.testing.expectEqual(canvas_ui.vertices.items.len, scene.ui_frame.vertices.items.len);
    try std.testing.expectEqual(canvas_ui.indices.items.len, scene.ui_frame.indices.items.len);
    try std.testing.expectEqual(@as(f32, 1920.0), scene.ui_frame.screen_w);
    try std.testing.expectEqual(@as(f32, 1080.0), scene.ui_frame.screen_h);
    try std.testing.expectEqual(canvas_ui.pipeline.id, scene.ui_frame.pipeline.id);
    // Headless: staged but not drawable, zero meter bytes, stats clean.
    try std.testing.expect(!scene.ui_frame.gpu_ready);
    try std.testing.expectEqual(@as(u64, 0), upload_meter.peek());
    try std.testing.expectEqual(@as(u64, 0), scene.stats.updated_bytes_frame);
    // Upload-free draw of a never-uploaded frame: safe no-op.
    scene.ui_frame.drawPrepared();
    try std.testing.expectEqual(@as(u64, 0), upload_meter.peek());

    // Newest capture wins across prepares.
    canvas_ui.drawRect(1, 2, 3, 4, Color4.white);
    scene.publishFrameSnapshot(16.0 / 9.0, 1920, 1080);
    scene.prepareFrame();
    try std.testing.expectEqual(canvas_ui.vertices.items.len, scene.ui_frame.vertices.items.len);
}

test "P6: camera-less prepare clears stale UI (no prior overlay)" {
    const alloc = std.testing.allocator;
    var scene = @import("testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.ui_frame.deinit(alloc);

    scene.ui_canvas = UICanvas{
        .allocator = alloc,
        .font_texture = std.mem.zeroes(Texture),
    };
    defer {
        if (scene.ui_canvas) |*c| {
            c.vertices.deinit(alloc);
            c.indices.deinit(alloc);
        }
        scene.ui_canvas = null;
    }
    scene.ui_canvas.?.drawRect(0, 0, 10, 10, Color4.white);

    const cam = Camera{ .free = camera_mod.FreeCamera.init("Cam1", .{}) };
    _ = try scene.addCamera(.{ .name = "Cam1", .camera = cam });
    scene.publishFrameSnapshot(1.0, 640, 480);
    scene.prepareFrame();
    try std.testing.expect(scene.ui_frame.has_capture);

    // Camera lost: the next prepare must not redisplay the prior overlay.
    for (scene.cameras.items) |entry| {
        if (entry.owns_name) alloc.free(entry.name);
    }
    scene.cameras.clearRetainingCapacity();
    scene.active_camera = null;
    scene.active_camera_index = null;
    scene.prepareFrame();
    try std.testing.expect(!scene.ui_frame.has_capture);
    try std.testing.expectEqual(@as(usize, 0), scene.ui_frame.vertices.items.len);
    scene.ui_frame.drawPrepared();
    try std.testing.expectEqual(@as(u64, 0), upload_meter.peek());
}

test "P6: prepare snapshots canvas presence apart from content" {
    const alloc = std.testing.allocator;
    var scene = @import("testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.ui_frame.deinit(alloc);

    const cam = Camera{ .free = camera_mod.FreeCamera.init("Cam1", .{}) };
    _ = try scene.addCamera(.{ .name = "Cam1", .camera = cam });

    // No canvas at all: presence false, content empty.
    scene.publishFrameSnapshot(1.0, 640, 480);
    scene.prepareFrame();
    try std.testing.expect(!scene.ui_frame.canvas_present);
    try std.testing.expect(!scene.ui_frame.has_capture);

    // Canvas exists but drew nothing: presence true (draw sites still
    // select the frame, preserving the phantom+1 counters), no capture.
    scene.ui_canvas = UICanvas{
        .allocator = alloc,
        .font_texture = std.mem.zeroes(Texture),
    };
    defer {
        if (scene.ui_canvas) |*c| {
            c.vertices.deinit(alloc);
            c.indices.deinit(alloc);
        }
        scene.ui_canvas = null;
    }
    scene.publishFrameSnapshot(1.0, 640, 480);
    scene.prepareFrame();
    try std.testing.expect(scene.ui_frame.canvas_present);
    try std.testing.expect(!scene.ui_frame.has_capture);
    scene.ui_frame.drawPrepared();
    try std.testing.expectEqual(@as(u64, 0), upload_meter.peek());

    // Canvas removed again: presence follows prepare, never sticks.
    if (scene.ui_canvas) |*c| {
        c.vertices.deinit(alloc);
        c.indices.deinit(alloc);
    }
    scene.ui_canvas = null;
    scene.publishFrameSnapshot(1.0, 640, 480);
    scene.prepareFrame();
    try std.testing.expect(!scene.ui_frame.canvas_present);
}

// ---- P7 double-buffered prepared draws. ----

// Headless full-path integration runs prepareFrame with a faked GPU-init
// flag (default_white_texture.view.id != 0) plus a CPU-only shadow pass
// (allocator + empty scratch/payload, zero GPU handles). No sg.* fires on
// the prepare path for plain/skinned/hook meshes: instance staging skips
// non-instanced meshes (and guards the rest with sg.isvalid), shadow
// prepareInto and the view-queue builds are pure CPU snapshots, and UI
// capture is sg-guarded (P6). Render never runs headless.
fn p7CpuShadowPass(scene: *Scene, alloc: std.mem.Allocator) void {
    // CPU-only stand-in: prepareInto/binMeshes never touch GPU handles
    // headless (render never runs), so every handle field is safely zero and
    // only allocator + scratch/payload carry state. Full literal — no
    // undefined fields left unread.
    scene.shadows.pass = .{
        .allocator = alloc,
        .image = .{},
        .attachment_view = .{},
        .texture_view = .{},
        .sampler = .{},
        .depth_sampler = .{},
        .spot_image = .{},
        .spot_attachment_view = .{},
        .spot_texture_view = .{},
        .spot_needs_clear = false,
        .pipeline_u16 = .{},
        .pipeline_u32 = .{},
        .inst_pipeline_u16 = .{},
        .inst_pipeline_u32 = .{},
        .skinned_pipeline_u16 = .{},
        .skinned_pipeline_u32 = .{},
        .shadow_shader = .{},
        .inst_shader = .{},
        .skinned_shader = .{},
        .binned_meshes = .empty,
        .prepared = .{},
    };
}

fn p7ShadowTotal(draws: *const FrameDrawSlot) usize {
    var total: usize = 0;
    for (draws.shadow.bin.counts) |c| total += c;
    return total;
}

fn p7FindByMeshIndex(items: []const RenderMeshItem, idx: u32) ?RenderMeshItem {
    for (items) |it| {
        if (it.mesh_index == idx) return it;
    }
    return null;
}

test "P7: slots alternate, newest wins, front intact while building back" {
    const alloc = std.testing.allocator;
    var scene = @import("testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.meshes.deinit(alloc);
    defer scene.outline_meshes.deinit(alloc);
    defer scene.draws.deinit(alloc);
    defer scene.shadows.pass.binned_meshes.deinit(alloc);
    defer scene.shadows.pass.prepared.deinit(alloc);
    scene.enable_frustum_culling = false;
    scene.enable_occlusion_culling = false;
    scene.default_white_texture.view.id = 1;
    p7CpuShadowPass(&scene, alloc);

    const skel = try Skeleton.init(alloc, 1);
    defer skel.deinit();
    skel.bones[0].local_position = Vec3.new(1, 0, 0);
    skel.update();

    var hook_mat = ShaderMaterial{ .name = "p7_hook" };
    var mesh_a = Mesh{
        .name = "p7_a",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .position = Vec3.new(4, 0, 0),
    };
    var mesh_s = Mesh{
        .name = "p7_s",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .position = Vec3.new(1, 0, 0),
        .skeleton = skel,
    };
    var mesh_h = Mesh{
        .name = "p7_h",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .position = Vec3.new(2, 0, 0),
        .material = .{ .shader_material = &hook_mat },
    };
    try scene.meshes.append(alloc, &mesh_a);
    try scene.meshes.append(alloc, &mesh_s);
    try scene.meshes.append(alloc, &mesh_h);
    try scene.outline_meshes.append(alloc, &mesh_a);

    // PIP enabled so a views[i] output is part of every frame (and of the
    // zero-alloc proof below): the second camera sees all three meshes.
    scene.enable_multi_camera = true;
    const cam = Camera{ .free = camera_mod.FreeCamera.init("Cam1", .{}) };
    const cam2 = Camera{ .free = camera_mod.FreeCamera.init("Cam2", .{}) };
    _ = try scene.addCamera(.{ .name = "Cam1", .camera = cam });
    _ = try scene.addCamera(.{ .name = "Cam2", .camera = cam2 });
    scene.active_camera_index = 0;
    scene.publishFrameSnapshot(16.0 / 9.0, 1920, 1080);

    scene.prepareFrame();
    try std.testing.expect(scene.frame_prepared);
    const front0 = scene.draws.front;
    const d0 = scene.preparedDraws();
    try std.testing.expectEqual(@as(usize, 3), d0.primary.items.items.len);
    try std.testing.expectEqual(@as(usize, 1), d0.primary.skin_storage.items.len);
    try std.testing.expectEqual(@as(usize, 1), d0.primary.shader_storage.items.len);
    try std.testing.expectEqual(@as(usize, 3), d0.views[1].items.items.len);
    try std.testing.expectEqual(@as(usize, 1), d0.outline_items.items.len);
    try std.testing.expectEqual(@as(usize, 3), p7ShadowTotal(d0));
    try std.testing.expectEqual(@as(usize, 3), d0.shadow.items.items.len);
    try std.testing.expectEqual(@as(usize, 1), d0.shadow.skins.items.len);
    try std.testing.expectEqual(scene.frame_id, d0.frame_id);
    try std.testing.expectEqual(scene.retire_epoch, d0.retire_epoch);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), d0.outline_items.items[0].model.m[12], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), p7FindByMeshIndex(d0.primary.items.items, 0).?.model.m[12], 1e-4);

    // Front intact while building back: move mesh_a, build the OTHER slot
    // directly, and prove the published front is untouched (lists, skins,
    // shader, outline, shadow ranges) while the back sees the new state.
    mesh_a.position = Vec3.new(9, 0, 0);
    const back_idx = 1 - front0;
    // New frame id for the manual back build (as prepareFrame would bump):
    // the world-matrix cache keys on it, and the published front snapshot
    // must stay at the old pose regardless.
    scene.frame_id +%= 1;
    scene.prepareViewQueues(&scene.draws.slots[back_idx].primary, scene.frame_snapshot.primary_cam, null, 1.0);
    const front_still = &scene.draws.slots[front0];
    try std.testing.expectEqual(@as(usize, 3), front_still.primary.items.items.len);
    try std.testing.expectEqual(@as(usize, 1), front_still.primary.skin_storage.items.len);
    try std.testing.expectEqual(@as(usize, 1), front_still.primary.shader_storage.items.len);
    try std.testing.expectEqual(@as(usize, 1), front_still.outline_items.items.len);
    try std.testing.expectEqual(@as(usize, 3), front_still.shadow.items.items.len);
    try std.testing.expectEqual(@as(usize, 1), front_still.shadow.skins.items.len);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), front_still.outline_items.items[0].model.m[12], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), p7FindByMeshIndex(front_still.primary.items.items, 0).?.model.m[12], 1e-4);
    const back_built = &scene.draws.slots[back_idx];
    try std.testing.expectApproxEqAbs(@as(f32, 9.0), p7FindByMeshIndex(back_built.primary.items.items, 0).?.model.m[12], 1e-4);
    // View builds never touch outline: the scratch back holds none.
    try std.testing.expectEqual(@as(usize, 0), back_built.outline_items.items.len);

    // Warmup: the second slot is still cold, so build it with a full prepare
    // first — the refusal proof below needs both slots warm.
    scene.publishFrameSnapshot(16.0 / 9.0, 1920, 1080);
    scene.prepareFrame();
    try std.testing.expectEqual(1 - front0, scene.draws.front);
    const d1 = scene.preparedDraws();
    try std.testing.expectEqual(@as(usize, 3), d1.primary.items.items.len);
    try std.testing.expectEqual(@as(usize, 1), d1.primary.skin_storage.items.len);
    try std.testing.expectEqual(@as(usize, 1), d1.primary.shader_storage.items.len);
    try std.testing.expectEqual(@as(usize, 3), d1.views[1].items.items.len);
    try std.testing.expectApproxEqAbs(@as(f32, 9.0), d1.outline_items.items[0].model.m[12], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 9.0), p7FindByMeshIndex(d1.primary.items.items, 0).?.model.m[12], 1e-4);
    try std.testing.expectEqual(@as(usize, 3), p7ShadowTotal(d1));
    try std.testing.expectEqual(@as(usize, 1), d1.shadow.skins.items.len);
    // The discarded front is retained as scratch, not cleared or copied.
    const old_slot = &scene.draws.slots[front0];
    try std.testing.expectEqual(@as(usize, 3), old_slot.primary.items.items.len);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), old_slot.outline_items.items[0].model.m[12], 1e-4);

    // Zero-alloc proof: both slots are warm now (slot B built by prepare#1,
    // slot A by prepare#2), so two further prepares — one per slot as back —
    // must not touch the allocator at all. The wrapper refuses ANY fresh
    // alloc (fail_index=0, flagged in has_induced_failure) and ANY
    // resize/remap (resize_fail_index=0); a refused growth always degrades
    // the frame (dropped items/counts), so the flag plus the expected
    // counts + selected poses under allocation refusal together prove zero
    // allocator traffic. Same wrapper feeds
    // scene.allocator and the shadow pass allocator.
    // Scope (narrow): this fixture exercises 3 regular meshes
    // (plain/skinned/hook), the serial cull path (no worker pool), one
    // outline item, shadow bin+snapshot, and primary+PIP view builds — with
    // no canvas, no instancing, no LOD/morphs. NOT covered by this proof:
    // parallel_scratch growth, instanced staging/buffer growth, UI capture
    // growth, or any parallel/GPU path.
    var expect_front = scene.draws.front;
    var round: usize = 0;
    while (round < 2) : (round += 1) {
        var refusing = std.testing.FailingAllocator.init(alloc, .{
            .fail_index = 0,
            .resize_fail_index = 0,
        });
        scene.publishFrameSnapshot(16.0 / 9.0, 1920, 1080);
        const saved_alloc = scene.allocator;
        scene.allocator = refusing.allocator();
        scene.shadows.pass.allocator = refusing.allocator();
        scene.prepareFrame();
        scene.allocator = saved_alloc;
        scene.shadows.pass.allocator = saved_alloc;
        try std.testing.expect(!refusing.has_induced_failure);
        expect_front = 1 - expect_front;
        try std.testing.expectEqual(expect_front, scene.draws.front);
        const dz = scene.preparedDraws();
        try std.testing.expectEqual(@as(usize, 3), dz.primary.items.items.len);
        try std.testing.expectEqual(@as(usize, 1), dz.primary.skin_storage.items.len);
        try std.testing.expectEqual(@as(usize, 1), dz.primary.shader_storage.items.len);
        try std.testing.expectEqual(@as(usize, 3), dz.views[1].items.items.len);
        try std.testing.expectEqual(@as(usize, 1), dz.outline_items.items.len);
        try std.testing.expectApproxEqAbs(@as(f32, 9.0), dz.outline_items.items[0].model.m[12], 1e-4);
        try std.testing.expectApproxEqAbs(@as(f32, 9.0), p7FindByMeshIndex(dz.primary.items.items, 0).?.model.m[12], 1e-4);
        try std.testing.expectEqual(@as(usize, 3), dz.shadow.items.items.len);
        try std.testing.expectEqual(@as(usize, 1), dz.shadow.skins.items.len);
        try std.testing.expectEqual(@as(usize, 3), p7ShadowTotal(dz));
        try std.testing.expectEqual(scene.frame_id, dz.frame_id);
        try std.testing.expectEqual(scene.retire_epoch, dz.retire_epoch);
    }
}

test "P7: main and PIP view slots stay isolated" {
    const alloc = std.testing.allocator;
    var scene = @import("testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.meshes.deinit(alloc);
    defer scene.draws.deinit(alloc);
    defer scene.shadows.pass.binned_meshes.deinit(alloc);
    defer scene.shadows.pass.prepared.deinit(alloc);
    scene.enable_frustum_culling = false;
    scene.enable_occlusion_culling = false;
    scene.enable_multi_camera = true;
    scene.default_white_texture.view.id = 1;
    p7CpuShadowPass(&scene, alloc);

    var mesh_a = Mesh{
        .name = "pip_a",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .layer_mask = 0b01,
    };
    var mesh_b = Mesh{
        .name = "pip_b",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .position = Vec3.new(5, 0, 0),
        .layer_mask = 0b10,
    };
    try scene.meshes.append(alloc, &mesh_a);
    try scene.meshes.append(alloc, &mesh_b);

    const cam0 = Camera{ .free = camera_mod.FreeCamera.init("Main", .{}) };
    const cam1 = Camera{ .free = camera_mod.FreeCamera.init("Pip", .{}) };
    const cam2 = Camera{ .free = camera_mod.FreeCamera.init("Off", .{}) };
    _ = try scene.addCamera(.{ .name = "Main", .camera = cam0 });
    _ = try scene.addCamera(.{ .name = "Pip", .camera = cam1 });
    _ = try scene.addCamera(.{ .name = "Off", .camera = cam2 });
    scene.cameras.items[1].culling_mask = 0b10;
    scene.cameras.items[2].enabled = false;
    scene.active_camera_index = 0;
    scene.publishFrameSnapshot(16.0 / 9.0, 1920, 1080);

    scene.prepareFrame();
    const draws = scene.preparedDraws();
    // Primary (all-mask active camera) sees both meshes.
    try std.testing.expectEqual(@as(usize, 2), draws.primary.items.items.len);
    // PIP slot 1 (mask 0b10) sees only mesh_b; slot 2 (disabled) is
    // reset-empty so no prior frame can resurface through it.
    try std.testing.expectEqual(@as(usize, 1), draws.views[1].items.items.len);
    try std.testing.expectEqual(@as(u32, 1), draws.views[1].items.items[0].mesh_index);
    try std.testing.expectEqual(@as(usize, 0), draws.views[2].items.items.len);
    // Shadow ignores view masks: both meshes binned in the same slot.
    try std.testing.expectEqual(@as(usize, 2), p7ShadowTotal(draws));

    // Disable the PIP view: the next publish clears its slot instead of
    // resurfacing the mesh_b frame above.
    scene.cameras.items[1].enabled = false;
    scene.publishFrameSnapshot(16.0 / 9.0, 1920, 1080);
    scene.prepareFrame();
    const draws2 = scene.preparedDraws();
    try std.testing.expectEqual(@as(usize, 0), draws2.views[1].items.items.len);
    try std.testing.expectEqual(@as(usize, 2), draws2.primary.items.items.len);
    try std.testing.expectEqual(@as(usize, 2), p7ShadowTotal(draws2));
}

test "P7: repeated prepare wins newest, no duplicate outline/UI capture" {
    const alloc = std.testing.allocator;
    var scene = @import("testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.meshes.deinit(alloc);
    defer scene.outline_meshes.deinit(alloc);
    defer scene.draws.deinit(alloc);
    defer scene.ui_frame.deinit(alloc);
    scene.enable_frustum_culling = false;
    scene.enable_occlusion_culling = false;

    scene.ui_canvas = UICanvas{
        .allocator = alloc,
        .font_texture = std.mem.zeroes(Texture),
    };
    defer {
        if (scene.ui_canvas) |*c| {
            c.vertices.deinit(alloc);
            c.indices.deinit(alloc);
        }
        scene.ui_canvas = null;
    }
    const canvas_ui = &scene.ui_canvas.?;
    canvas_ui.drawRect(10, 20, 30, 40, Color4.white);

    var mesh = Mesh{
        .name = "rep_outline",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .position = Vec3.new(4, 0, 0),
    };
    try scene.outline_meshes.append(alloc, &mesh);

    const cam = Camera{ .free = camera_mod.FreeCamera.init("Cam1", .{}) };
    _ = try scene.addCamera(.{ .name = "Cam1", .camera = cam });
    scene.publishFrameSnapshot(16.0 / 9.0, 1920, 1080);

    scene.prepareFrame();
    const front0 = scene.draws.front;
    try std.testing.expectEqual(@as(usize, 1), scene.preparedDraws().outline_items.items.len);
    const ui_n = canvas_ui.vertices.items.len;
    try std.testing.expectEqual(ui_n, scene.ui_frame.vertices.items.len);

    // More UI + moved outline, then a repeated prepare with no new camera
    // publish (fallback snapshot path): newest wins, nothing accumulates.
    canvas_ui.drawRect(1, 2, 3, 4, Color4.white);
    mesh.position = Vec3.new(7, 0, 0);
    scene.prepareFrame();
    try std.testing.expectEqual(1 - front0, scene.draws.front);
    try std.testing.expectEqual(@as(usize, 1), scene.preparedDraws().outline_items.items.len);
    try std.testing.expectApproxEqAbs(@as(f32, 7.0), scene.preparedDraws().outline_items.items[0].model.m[12], 1e-4);
    try std.testing.expectEqual(canvas_ui.vertices.items.len, scene.ui_frame.vertices.items.len);
    try std.testing.expect(canvas_ui.vertices.items.len > ui_n);
}

test "P7: no-camera/headless and disabled shadows clear coherently, epochs consumed" {
    const alloc = std.testing.allocator;
    var scene = @import("testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.meshes.deinit(alloc);
    defer scene.outline_meshes.deinit(alloc);
    defer scene.draws.deinit(alloc);
    defer scene.shadows.pass.binned_meshes.deinit(alloc);
    defer scene.shadows.pass.prepared.deinit(alloc);
    scene.enable_frustum_culling = false;
    scene.enable_occlusion_culling = false;
    scene.default_white_texture.view.id = 1;
    p7CpuShadowPass(&scene, alloc);

    var mesh = Mesh{
        .name = "clr_mesh",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
    };
    try scene.meshes.append(alloc, &mesh);
    try scene.outline_meshes.append(alloc, &mesh);

    const cam = Camera{ .free = camera_mod.FreeCamera.init("Cam1", .{}) };
    _ = try scene.addCamera(.{ .name = "Cam1", .camera = cam });
    scene.publishFrameSnapshot(1.0, 640, 480);
    scene.prepareFrame();
    try std.testing.expectEqual(@as(usize, 1), scene.preparedDraws().primary.items.items.len);
    try std.testing.expectEqual(@as(usize, 1), p7ShadowTotal(scene.preparedDraws()));
    const epoch1 = scene.retire_epoch;

    // Camera lost: views + shadow publish coherent-empty (no resurface of
    // the frame above); outline stays unconditional (existing semantics).
    for (scene.cameras.items) |entry| {
        if (entry.owns_name) alloc.free(entry.name);
    }
    scene.cameras.clearRetainingCapacity();
    scene.active_camera = null;
    scene.active_camera_index = null;
    scene.prepareFrame();
    const cleared = scene.preparedDraws();
    try std.testing.expect(!scene.frame_snapshot.has_camera);
    try std.testing.expectEqual(@as(usize, 0), cleared.primary.items.items.len);
    try std.testing.expectEqual(@as(usize, 0), cleared.views[0].items.items.len);
    try std.testing.expectEqual(@as(usize, 0), cleared.shadow.items.items.len);
    try std.testing.expectEqual(@as(usize, 0), cleared.shadow.skins.items.len);
    for (cleared.shadow.bin.counts) |c| try std.testing.expectEqual(@as(usize, 0), c);
    try std.testing.expectEqual(@as(usize, 1), cleared.outline_items.items.len);
    // Repeated prepare discarded the pending frame: begin auto-closed its
    // epoch before the flush tore down anything it borrowed.
    try std.testing.expectEqual(epoch1 + 1, scene.retire_epoch);
    try std.testing.expectEqual(epoch1, scene.gpu_retire.lastCompleted());

    // No-camera render still completes the current epoch (headless-safe:
    // the early return runs before any sg.*).
    scene.render();
    try std.testing.expect(!scene.frame_prepared);
    try std.testing.expectEqual(scene.retire_epoch, scene.gpu_retire.lastCompleted());

    // Camera back but shadows disabled: primary rebuilds, shadow stays empty.
    const cam2 = Camera{ .free = camera_mod.FreeCamera.init("Cam2", .{}) };
    _ = try scene.addCamera(.{ .name = "Cam2", .camera = cam2 });
    scene.shadows.enabled = false;
    scene.publishFrameSnapshot(1.0, 640, 480);
    scene.prepareFrame();
    const noshadow = scene.preparedDraws();
    try std.testing.expectEqual(@as(usize, 1), noshadow.primary.items.items.len);
    try std.testing.expectEqual(@as(usize, 0), noshadow.shadow.items.items.len);
    for (noshadow.shadow.bin.counts) |c| try std.testing.expectEqual(@as(usize, 0), c);
}

test "P7: allocator-failure back stays coherent and recovers without stale items" {
    const alloc = std.testing.allocator;
    var scene = @import("testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.meshes.deinit(alloc);
    defer scene.outline_meshes.deinit(alloc);
    defer scene.draws.deinit(alloc);
    defer scene.shadows.pass.binned_meshes.deinit(alloc);
    defer scene.shadows.pass.prepared.deinit(alloc);
    scene.enable_frustum_culling = false;
    scene.enable_occlusion_culling = false;
    scene.default_white_texture.view.id = 1;
    p7CpuShadowPass(&scene, alloc);

    const skel = try Skeleton.init(alloc, 1);
    defer skel.deinit();
    skel.bones[0].local_position = Vec3.new(1, 0, 0);
    skel.update();

    var mesh_a = Mesh{
        .name = "oom_a",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .position = Vec3.new(4, 0, 0),
    };
    var mesh_s = Mesh{
        .name = "oom_s",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .skeleton = skel,
    };
    try scene.meshes.append(alloc, &mesh_a);
    try scene.meshes.append(alloc, &mesh_s);
    try scene.outline_meshes.append(alloc, &mesh_a);

    const cam = Camera{ .free = camera_mod.FreeCamera.init("Cam1", .{}) };
    _ = try scene.addCamera(.{ .name = "Cam1", .camera = cam });
    scene.publishFrameSnapshot(1.0, 640, 480);
    scene.prepareFrame();
    try std.testing.expectEqual(@as(usize, 2), scene.preparedDraws().primary.items.items.len);
    try std.testing.expectEqual(@as(usize, 2), p7ShadowTotal(scene.preparedDraws()));

    // Failing back build: everything fallible drops (existing per-item /
    // coherent-empty OOM semantics — never a full-frame transaction abort),
    // and the publish stays index-coherent: no skin/shader/order index
    // escapes its side store, no bin range escapes the items.
    const real_alloc = scene.allocator;
    var failing = std.testing.FailingAllocator.init(alloc, .{ .fail_index = 0 });
    scene.allocator = failing.allocator();
    scene.shadows.pass.allocator = failing.allocator();
    scene.publishFrameSnapshot(1.0, 640, 480);
    scene.prepareFrame();
    scene.allocator = real_alloc;
    scene.shadows.pass.allocator = real_alloc;
    const oom = scene.preparedDraws();
    for (oom.primary.items.items) |it| {
        if (it.skin_index) |s| try std.testing.expect(s < oom.primary.skin_storage.items.len);
        if (it.shader_index) |s| try std.testing.expect(s < oom.primary.shader_storage.items.len);
    }
    for (oom.primary.transparent.items) |it| {
        if (it.skin_index) |s| try std.testing.expect(s < oom.primary.skin_storage.items.len);
        if (it.shader_index) |s| try std.testing.expect(s < oom.primary.shader_storage.items.len);
    }
    for (oom.outline_items.items) |it| {
        if (it.skin_index) |s| try std.testing.expect(s < oom.outline_skins.items.len);
    }
    for (oom.primary.transparent_order.items) |e| {
        switch (e.kind) {
            .regular => try std.testing.expect(e.index < oom.primary.transparent.items.len),
            .instanced => try std.testing.expect(e.index < oom.primary.transparent_instanced.items.len),
        }
    }
    for (oom.shadow.bin.counts, oom.shadow.bin.offsets) |c, o| {
        try std.testing.expect(o + c <= oom.shadow.items.items.len);
    }
    for (oom.shadow.items.items) |it| {
        if (it.skin_index) |s| try std.testing.expect(s < oom.shadow.skins.items.len);
    }

    // Recovery with the working allocator: no stale items, full frame back.
    mesh_a.position = Vec3.new(6, 0, 0);
    scene.publishFrameSnapshot(1.0, 640, 480);
    scene.prepareFrame();
    const rec = scene.preparedDraws();
    try std.testing.expectEqual(@as(usize, 2), rec.primary.items.items.len);
    try std.testing.expectEqual(@as(usize, 1), rec.primary.skin_storage.items.len);
    try std.testing.expectEqual(@as(usize, 1), rec.outline_items.items.len);
    try std.testing.expectEqual(@as(usize, 2), p7ShadowTotal(rec));
    try std.testing.expectEqual(@as(usize, 1), rec.shadow.skins.items.len);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), rec.outline_items.items[0].model.m[12], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), p7FindByMeshIndex(rec.primary.items.items, 0).?.model.m[12], 1e-4);
}

// ---- Actual update||render boundary: render-owned captures. ----

test "saturated frame mailbox drops instead of overwriting the consumed snapshot" {
    const alloc = std.testing.allocator;
    var scene = @import("testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);

    // Consumed snapshot (screen dims are set before the no-camera
    // early-out, so no camera is needed for this ownership proof).
    scene.publishFrameSnapshot(1.0, 111, 111);
    scene.prepareFrame();
    try std.testing.expectEqual(@as(i32, 111), scene.frame_snapshot.screen_w);

    // Force the last-unclaimable case: hold BOTH slots in WRITING (a lagging
    // holder). releasePublished only drains PUBLISHED slots, so the claim
    // still fails — the tick must DROP, never overwrite the consumed
    // snapshot under a concurrent render.
    const a = scene.frame_handoff.claim().?;
    const b = scene.frame_handoff.claim().?;
    try std.testing.expect(scene.frame_handoff.claim() == null);
    scene.publishFrameSnapshot(1.0, 999, 999);
    try std.testing.expectEqual(@as(i32, 111), scene.frame_snapshot.screen_w);

    // Finish the held claims as stale publishes; the next real tick drains
    // them and the newest wins (no resurfacing of 222/333).
    scene.frame_handoff.slot(a).* = scene.packFrameSnapshot(1.0, 222, 222);
    scene.frame_handoff.publish(a);
    scene.frame_handoff.slot(b).* = scene.packFrameSnapshot(1.0, 333, 333);
    scene.frame_handoff.publish(b);
    scene.publishFrameSnapshot(1.0, 444, 444);
    scene.prepareFrame();
    try std.testing.expectEqual(@as(i32, 444), scene.frame_snapshot.screen_w);
}

test "debug capture: off/world-empty stay empty, capture immutable, OOM coherent" {
    const alloc = std.testing.allocator;
    var scene = @import("testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.physics.deinit(alloc);

    // No world + off: empty, invisible.
    scene.physics.captureDebug(alloc);
    try std.testing.expect(!scene.physics.prepared_visible);
    try std.testing.expectEqual(@as(usize, 0), scene.physics.prepared_lines.items.len);

    _ = scene.physics.enable(alloc, null);
    const m = try alloc.create(Mesh);
    defer alloc.destroy(m);
    m.* = @import("testing.zig").testMesh("dbg_cap");
    _ = try scene.physics.getWorld().?.createBody(m, .box, 0.0);

    // World present but off: still empty (draw site selects on the capture).
    scene.physics.captureDebug(alloc);
    try std.testing.expect(!scene.physics.prepared_visible);
    try std.testing.expectEqual(@as(usize, 0), scene.physics.prepared_lines.items.len);

    scene.physics.show_debug = true;
    scene.physics.captureDebug(alloc);
    try std.testing.expect(scene.physics.prepared_visible);
    try std.testing.expectEqual(@as(usize, 12), scene.physics.prepared_lines.items.len);

    // Immutable under later mutation without recapture (the update side may
    // step/move while render draws the capture).
    const x0 = scene.physics.prepared_lines.items[0].a.x;
    m.position = Vec3.new(5, 0, 0);
    scene.physics.captureDebug(alloc);
    try std.testing.expectEqual(@as(usize, 12), scene.physics.prepared_lines.items.len);
    try std.testing.expectApproxEqAbs(x0 + 5.0, scene.physics.prepared_lines.items[0].a.x, 1e-4);
    scene.physics.step(0.016);
    try std.testing.expectApproxEqAbs(x0 + 5.0, scene.physics.prepared_lines.items[0].a.x, 1e-4);

    // Headless upload/draw: safe no-ops, stats clean, no pass created.
    var stats = SceneStats{};
    scene.physics.uploadDebug(alloc, 1);
    try std.testing.expect(scene.physics.debug_pass == null);
    scene.physics.renderDebugPrepared(Mat4.identity, 1, &stats);
    try std.testing.expectEqual(@as(u32, 0), stats.draw_calls);
    try std.testing.expectEqual(@as(u32, 0), stats.main_draw_calls);

    // Non-empty capture + present-but-empty upload: still no count (the
    // counters gate on the issued-draw report, never on visibility alone).
    scene.physics.debug_pass = @import("passes/debug_pass.zig").DebugPass{ .allocator = alloc };
    try std.testing.expect(scene.physics.prepared_visible);
    try std.testing.expectEqual(@as(usize, 12), scene.physics.prepared_lines.items.len);
    scene.physics.renderDebugPrepared(Mat4.identity, 1, &stats);
    try std.testing.expectEqual(@as(u32, 0), stats.draw_calls);
    try std.testing.expectEqual(@as(u32, 0), stats.main_draw_calls);
    // Pull the fake pass back out: DebugPass.deinit issues sg.destroy*
    // (context-only, traps headless) — headless teardown is staging-only,
    // so the fixture deinit below must see null here.
    var fake_pass = scene.physics.debug_pass.?;
    scene.physics.debug_pass = null;
    fake_pass.staging.deinit(alloc);

    // Off clears (no stale overlay).
    scene.physics.show_debug = false;
    scene.physics.captureDebug(alloc);
    try std.testing.expect(!scene.physics.prepared_visible);
    try std.testing.expectEqual(@as(usize, 0), scene.physics.prepared_lines.items.len);
    // Retained capacity survives the clear (no per-frame churn).
    try std.testing.expect(scene.physics.prepared_lines.capacity >= 12);
}

test "debug capture OOM fail-closes coherent-empty then recovers" {
    const alloc = std.testing.allocator;
    var scene = @import("testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.physics.deinit(alloc);

    _ = scene.physics.enable(alloc, null);
    const m = try alloc.create(Mesh);
    defer alloc.destroy(m);
    m.* = @import("testing.zig").testMesh("dbg_oom");
    _ = try scene.physics.getWorld().?.createBody(m, .box, 0.0);
    scene.physics.show_debug = true;

    // Unfunded capture on empty capacity: the upfront reserve fails ->
    // coherent-empty (never a partial half).
    var failing_fresh = std.testing.FailingAllocator.init(alloc, .{ .fail_index = 0 });
    scene.physics.captureDebug(failing_fresh.allocator());
    try std.testing.expect(!scene.physics.prepared_visible);
    try std.testing.expectEqual(@as(usize, 0), scene.physics.prepared_lines.items.len);

    // Funded prime (12 lines, retained capacity high-water).
    scene.physics.captureDebug(alloc);
    try std.testing.expect(scene.physics.prepared_visible);
    try std.testing.expectEqual(@as(usize, 12), scene.physics.prepared_lines.items.len);

    // Force growth past retained capacity: a second body needs 24 lines,
    // so the reserve MUST allocate -> the failing allocator fails ->
    // coherent-empty with capacity retained (no shrink, no partial).
    const m2 = try alloc.create(Mesh);
    defer alloc.destroy(m2);
    m2.* = @import("testing.zig").testMesh("dbg_oom2");
    _ = try scene.physics.getWorld().?.createBody(m2, .box, 0.0);
    var failing_grow = std.testing.FailingAllocator.init(alloc, .{ .fail_index = 0 });
    scene.physics.captureDebug(failing_grow.allocator());
    try std.testing.expect(!scene.physics.prepared_visible);
    try std.testing.expectEqual(@as(usize, 0), scene.physics.prepared_lines.items.len);
    try std.testing.expect(scene.physics.prepared_lines.capacity >= 12);

    // Recovery funds the full 24.
    scene.physics.captureDebug(alloc);
    try std.testing.expect(scene.physics.prepared_visible);
    try std.testing.expectEqual(@as(usize, 24), scene.physics.prepared_lines.items.len);

    // Warm capture refuses fresh allocs entirely: steady-state captures
    // are allocation-free (no per-frame churn).
    var refusing = std.testing.FailingAllocator.init(alloc, .{ .fail_index = 0, .resize_fail_index = 0 });
    scene.physics.captureDebug(refusing.allocator());
    try std.testing.expect(!refusing.has_induced_failure);
    try std.testing.expect(scene.physics.prepared_visible);
    try std.testing.expectEqual(@as(usize, 24), scene.physics.prepared_lines.items.len);
}

test "sky snapshot freezes enabled/texture/exposure/defaults at prepare" {
    const alloc = std.testing.allocator;
    var scene = @import("testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);

    const cam = Camera{ .free = camera_mod.FreeCamera.init("Cam1", .{}) };
    _ = try scene.addCamera(.{ .name = "Cam1", .camera = cam });

    // Fake cube: plain-data handles, no GPU involved.
    var cube = std.mem.zeroes(CubeTexture);
    cube.view.id = 77;
    cube.sampler.id = 78;
    scene.sky.setSkybox(cube);
    scene.sky.exposure = 2.0;
    scene.sky.ibl_intensity = 0.5;

    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    scene.prepareFrame();
    try std.testing.expect(scene.frame_snapshot.sky_enabled);
    try std.testing.expectEqual(@as(f32, 2.0), scene.frame_snapshot.sky_exposure);
    try std.testing.expectEqual(@as(f32, 0.5), scene.frame_snapshot.ibl_intensity);
    try std.testing.expectEqual(@as(u32, 77), scene.frame_snapshot.sky_texture.?.view.id);

    // Mutate live AFTER prepare: the consumed snapshot stays frozen
    // (a concurrent update cannot change the in-flight packet). Covers
    // every render-owned sky/default field: enabled, exposure, the sky
    // cube, and all three default copies. (Sampler ids, not view ids: the
    // fixture has no GPU init, and a nonzero default-white view id would
    // open the GPU queue-build path with an uninitialized shadow pass.)
    scene.sky.enabled = false;
    scene.sky.exposure = 9.0;
    scene.sky.texture = null;
    scene.default_white_texture.sampler.id = 5;
    scene.default_normal_texture.sampler.id = 6;
    scene.default_cube_texture.sampler.id = 7;
    try std.testing.expect(scene.frame_snapshot.sky_enabled);
    try std.testing.expectEqual(@as(f32, 2.0), scene.frame_snapshot.sky_exposure);
    try std.testing.expectEqual(@as(u32, 77), scene.frame_snapshot.sky_texture.?.view.id);
    try std.testing.expectEqual(@as(u32, 0), scene.frame_snapshot.default_white.sampler.id);
    try std.testing.expectEqual(@as(u32, 0), scene.frame_snapshot.default_normal.sampler.id);
    try std.testing.expectEqual(@as(u32, 0), scene.frame_snapshot.default_cube.sampler.id);

    // Headless prepared draw: safe no-op, stats clean (the fixture pass is
    // undefined, but the sg.isvalid() short-circuit returns first).
    var stats = SceneStats{};
    scene.sky.renderPrepared(
        scene.frame_snapshot.sky_enabled,
        scene.frame_snapshot.primary_cam.camera,
        scene.frame_snapshot.primary_cam.aspect,
        scene.frame_snapshot.sky_texture orelse scene.frame_snapshot.default_cube,
        scene.frame_snapshot.sky_exposure,
        1,
        &stats,
    );
    try std.testing.expectEqual(@as(u32, 0), stats.draw_calls);

    // A disabled snapshot draws nothing even with the live layer enabled.
    scene.sky.enabled = true;
    scene.sky.renderPrepared(
        false,
        scene.frame_snapshot.primary_cam.camera,
        scene.frame_snapshot.primary_cam.aspect,
        scene.frame_snapshot.sky_texture orelse scene.frame_snapshot.default_cube,
        scene.frame_snapshot.sky_exposure,
        1,
        &stats,
    );
    try std.testing.expectEqual(@as(u32, 0), stats.draw_calls);

    // pack/publish/prepare round-trip: the NEXT publish captures the newest
    // live values (nothing stale), proving each stage reads live state at
    // prepare time and freezes it after.
    scene.sky.enabled = true;
    scene.sky.exposure = 4.0;
    var cube2 = std.mem.zeroes(CubeTexture);
    cube2.view.id = 88;
    scene.sky.setSkybox(cube2);
    scene.default_normal_texture.sampler.id = 66;
    scene.default_cube_texture.sampler.id = 67;
    // Direct pack reads live.
    const repacked = scene.packFrameSnapshot(16.0 / 9.0, 800, 600);
    try std.testing.expect(repacked.sky_enabled);
    try std.testing.expectEqual(@as(f32, 4.0), repacked.sky_exposure);
    try std.testing.expectEqual(@as(u32, 88), repacked.sky_texture.?.view.id);
    try std.testing.expectEqual(@as(u32, 66), repacked.default_normal.sampler.id);
    try std.testing.expectEqual(@as(u32, 67), repacked.default_cube.sampler.id);
    // Publish + prepare consume the newest mailbox frame.
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    scene.prepareFrame();
    try std.testing.expect(scene.frame_snapshot.sky_enabled);
    try std.testing.expectEqual(@as(f32, 4.0), scene.frame_snapshot.sky_exposure);
    try std.testing.expectEqual(@as(u32, 88), scene.frame_snapshot.sky_texture.?.view.id);
    try std.testing.expectEqual(@as(u32, 5), scene.frame_snapshot.default_white.sampler.id);
    try std.testing.expectEqual(@as(u32, 66), scene.frame_snapshot.default_normal.sampler.id);
    try std.testing.expectEqual(@as(u32, 67), scene.frame_snapshot.default_cube.sampler.id);
}

test "worker churn after prepare cannot mutate consumed snapshot/timings" {
    const alloc = std.testing.allocator;
    var scene = @import("testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.physics.deinit(alloc);

    const cam = Camera{ .free = camera_mod.FreeCamera.init("Cam1", .{}) };
    _ = try scene.addCamera(.{ .name = "Cam1", .camera = cam });
    scene.sky.enabled = true;
    scene.sky.exposure = 1.5;
    scene.recordUpdateTime(3.0);
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    scene.prepareFrame();
    try std.testing.expect(scene.frame_snapshot.has_camera);
    try std.testing.expectEqual(@as(f32, 3.0), scene.stats.update_ms);

    const snap_w = scene.frame_snapshot.screen_w;
    const snap_exp = scene.frame_snapshot.sky_exposure;
    const snap_clear_r = scene.frame_snapshot.clear_color.r;

    // Worker = the next update tick hammering live state. Spawned AFTER
    // prepare (a happens-before edge — deterministic, no sleeps/timing).
    // It mutates ONLY update-owned words: live sky fields, the staged tick,
    // clear color. Snapshot/stats are never touched by it (disjoint fields).
    const Worker = struct {
        scene: *Scene,
        iters: usize,
        fn run(self: @This()) void {
            var i: usize = 0;
            while (i < self.iters) : (i += 1) {
                self.scene.sky.exposure = 9.0;
                self.scene.sky.enabled = false;
                self.scene.recordUpdateTime(99.0);
                self.scene.clear_color = Color4.new(1, 0, 0, 1);
            }
        }
    };
    const t = try std.Thread.spawn(.{}, Worker.run, .{Worker{ .scene = &scene, .iters = 500 }});
    t.join();

    // Consumed render state frozen despite the churn.
    try std.testing.expectEqual(snap_w, scene.frame_snapshot.screen_w);
    try std.testing.expectEqual(snap_exp, scene.frame_snapshot.sky_exposure);
    try std.testing.expect(scene.frame_snapshot.sky_enabled);
    try std.testing.expectEqual(snap_clear_r, scene.frame_snapshot.clear_color.r);
    try std.testing.expectEqual(@as(f32, 3.0), scene.stats.update_ms);
    try std.testing.expectEqual(@as(f32, 99.0), scene.pending_update_ms);

    // Next prepare picks up the newest live state (newest wins).
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    scene.prepareFrame();
    try std.testing.expect(!scene.frame_snapshot.sky_enabled);
    try std.testing.expectEqual(@as(f32, 9.0), scene.frame_snapshot.sky_exposure);
    try std.testing.expectEqual(@as(f32, 99.0), scene.stats.update_ms);
}

// ---- Stage 1 producer-build handoff: buildPreparedFrame + prepare latch. ---
//
// Headless full-path pattern (mirrors the P7 tests): faked GPU-init flag
// (default_white_texture.view.id != 0) + CPU-only shadow pass + disabled
// culling. sg.isvalid() stays false, so the GPU halves publish bounds/count
// without touching sg.*. Render never runs headless past no-camera.

fn stage1FillInstances(src: *Mesh, mem: []@import("mesh.zig").InstancedMesh, ptrs: []*@import("mesh.zig").InstancedMesh, x0: f32) void {
    for (mem, 0..) |*inst, i| {
        const fi: f32 = @floatFromInt(i);
        inst.* = .{ .name = "s1", .source_mesh = src, .position = Vec3.new(x0 + fi * 2.0, 0, 0) };
        ptrs[i] = inst;
    }
}

fn stage1Scene(alloc: std.mem.Allocator) Scene {
    var scene = @import("testing.zig").testScene(alloc);
    scene.enable_frustum_culling = false;
    scene.enable_occlusion_culling = false;
    scene.default_white_texture.view.id = 1;
    return scene;
}

test "stage1: worker build + main latch publishes previews; post-build mutation invisible" {
    const InstancedMesh = @import("mesh.zig").InstancedMesh;
    const alloc = std.testing.allocator;
    var scene = stage1Scene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.meshes.deinit(alloc);
    defer scene.outline_meshes.deinit(alloc);
    defer scene.draws.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);
    defer scene.shadows.pass.binned_meshes.deinit(alloc);
    defer scene.shadows.pass.prepared.deinit(alloc);
    p7CpuShadowPass(&scene, alloc);

    const cam = Camera{ .free = camera_mod.FreeCamera.init("Cam1", .{}) };
    _ = try scene.addCamera(.{ .name = "Cam1", .camera = cam });

    var src = Mesh{
        .name = "s1_src",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    const n: usize = 4;
    const mem = try alloc.alloc(InstancedMesh, n);
    defer alloc.free(mem);
    const ptrs = try alloc.alloc(*InstancedMesh, n);
    defer alloc.free(ptrs);
    stage1FillInstances(&src, mem, ptrs, 0);
    var parent = Mesh{
        .name = "s1_parent",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .instances = .{ .items = ptrs, .capacity = n },
    };
    try scene.meshes.append(alloc, &parent);

    // Victim: heap mesh (destroyMesh frees it via the retire queue) with
    // heap instances (Mesh.deinit destroys them at flush).
    const victim = try alloc.create(Mesh);
    victim.* = @import("testing.zig").testMesh("s1_victim");
    victim.local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1));
    for (0..2) |i| {
        const inst = try alloc.create(InstancedMesh);
        inst.* = .{ .name = "v", .source_mesh = victim, .position = Vec3.new(@floatFromInt(i), 0, 0) };
        try victim.instances.append(alloc, inst);
    }
    try scene.meshes.append(alloc, victim);

    // Mark the context on main: the worker below is off-context, so its
    // destroyMesh must take the retire-queue path (never sg.* off-thread).
    gpu_thread.markContextThread();

    // Worker = the game side: build, then destroy the victim inside the
    // build→latch window (sequential in the worker — no data race; the
    // spawn/join edges carry the happens-before). Main only joins.
    const Builder = struct {
        scene: *Scene,
        victim: *Mesh,
        fn run(self: @This()) void {
            self.scene.buildPreparedFrame();
            self.scene.destroyMesh(self.victim);
        }
    };
    const t = try std.Thread.spawn(.{}, Builder.run, .{Builder{ .scene = &scene, .victim = victim }});
    t.join();

    // Off-context destroy unlinked the victim into the retire queue; the
    // build recorded previews for both meshes (victim included).
    try std.testing.expectEqual(@as(u64, 1), scene.build_seq);
    try std.testing.expectEqual(@as(usize, 1), scene.meshes.items.len);
    try std.testing.expectEqual(@as(usize, 1), scene.gpu_retire.retainedCount());
    try std.testing.expectEqual(@as(u64, 1), parent.instance_preview.build_seq);
    try std.testing.expectEqual(@as(u32, 4), parent.instance_preview.count);
    try std.testing.expectEqual(@as(u64, 1), victim.instance_preview.build_seq);

    // Live mutation AFTER the build: the published instance_render must
    // reflect the build, not the mutation.
    mem[0].position = Vec3.new(100, 0, 0);
    const preview_bounds = parent.instance_preview.bounds;

    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    scene.prepareFrame();
    try std.testing.expect(scene.frame_prepared);
    try std.testing.expectEqual(@as(u64, 1), scene.last_latched_seq);
    try std.testing.expectEqual(@as(u32, 4), parent.instance_render.count);
    try std.testing.expectEqual(preview_bounds, parent.instance_render.bounds);
    try std.testing.expectEqual(scene.frame_id, parent.instance_render.staged_frame);
    // The CPU preview carries the matrix-bytes hash for the GPU dedup gate;
    // headless (no sg context) the GPU half is skipped, so instance_render
    // keeps the stale hash (0) by design — bounds/count still publish.
    try std.testing.expect(parent.instance_preview.hash != 0);
    try std.testing.expectEqual(@as(u64, 0), parent.instance_render.hash);
    // A 100-unit move would have shifted the bounds; the latch kept build
    // time (max.x well under the mutated span).
    try std.testing.expect(parent.instance_render.bounds.max.x < 50.0);
    // The destroyed victim is gone from the latch (no use-after-free) and
    // the published queues carry only the surviving mesh.
    const draws = scene.preparedDraws();
    try std.testing.expectEqual(@as(usize, 1), draws.primary.opaque_instanced.items.len);
    try std.testing.expectEqual(@as(u32, 4), draws.primary.opaque_instanced.items[0].visible_instance_count);
}

test "stage1: two builds before latch, newest wins" {
    const InstancedMesh = @import("mesh.zig").InstancedMesh;
    const alloc = std.testing.allocator;
    var scene = stage1Scene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.meshes.deinit(alloc);
    defer scene.outline_meshes.deinit(alloc);
    defer scene.draws.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);
    defer scene.shadows.pass.binned_meshes.deinit(alloc);
    defer scene.shadows.pass.prepared.deinit(alloc);
    p7CpuShadowPass(&scene, alloc);

    const cam = Camera{ .free = camera_mod.FreeCamera.init("Cam1", .{}) };
    _ = try scene.addCamera(.{ .name = "Cam1", .camera = cam });

    var src = Mesh{
        .name = "s1_src",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    const n: usize = 3;
    const mem = try alloc.alloc(InstancedMesh, n);
    defer alloc.free(mem);
    const ptrs = try alloc.alloc(*InstancedMesh, n);
    defer alloc.free(ptrs);
    stage1FillInstances(&src, mem, ptrs, 0);
    var parent = Mesh{
        .name = "s1_parent",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .instances = .{ .items = ptrs, .capacity = n },
    };
    try scene.meshes.append(alloc, &parent);

    scene.buildPreparedFrame();
    const first_bounds = parent.instance_preview.bounds;
    // Mutate, rebuild: the single preview store is recomputed in place.
    mem[2].position = Vec3.new(40, 0, 0);
    scene.buildPreparedFrame();
    try std.testing.expectEqual(@as(u64, 2), scene.build_seq);
    try std.testing.expectEqual(@as(u64, 2), parent.instance_preview.build_seq);
    try std.testing.expect(parent.instance_preview.bounds.max.x > first_bounds.max.x + 10.0);

    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    scene.prepareFrame();
    try std.testing.expectEqual(@as(u64, 2), scene.last_latched_seq);
    try std.testing.expectEqual(parent.instance_preview.bounds, parent.instance_render.bounds);
    try std.testing.expect(parent.instance_render.bounds.max.x > first_bounds.max.x + 10.0);
}

test "stage1: no build runs the inline fallback with identical counts/bounds" {
    const InstancedMesh = @import("mesh.zig").InstancedMesh;
    const alloc = std.testing.allocator;
    var scene = stage1Scene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.meshes.deinit(alloc);
    defer scene.outline_meshes.deinit(alloc);
    defer scene.draws.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);
    defer scene.shadows.pass.binned_meshes.deinit(alloc);
    defer scene.shadows.pass.prepared.deinit(alloc);
    p7CpuShadowPass(&scene, alloc);

    const cam = Camera{ .free = camera_mod.FreeCamera.init("Cam1", .{}) };
    _ = try scene.addCamera(.{ .name = "Cam1", .camera = cam });

    var src = Mesh{
        .name = "s1_src",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    const n: usize = 5;
    const mem = try alloc.alloc(InstancedMesh, n);
    defer alloc.free(mem);
    const ptrs = try alloc.alloc(*InstancedMesh, n);
    defer alloc.free(ptrs);
    stage1FillInstances(&src, mem, ptrs, 0);
    mem[4].is_visible = false;
    var parent = Mesh{
        .name = "s1_parent",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .instances = .{ .items = ptrs, .capacity = n },
    };
    try scene.meshes.append(alloc, &parent);

    // Latch path first.
    scene.buildPreparedFrame();
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    scene.prepareFrame();
    try std.testing.expectEqual(@as(u64, 1), scene.last_latched_seq);
    const latched_bounds = parent.instance_render.bounds;
    const latched_count = parent.instance_render.count;
    const latched_hash = parent.instance_render.hash;
    try std.testing.expectEqual(@as(u32, 4), latched_count);

    // Same live state, no fresh build: the inline fallback must publish the
    // identical counts/bounds/hash (only staged_frame advances).
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    scene.prepareFrame();
    try std.testing.expectEqual(@as(u64, 1), scene.last_latched_seq);
    try std.testing.expectEqual(latched_count, parent.instance_render.count);
    try std.testing.expectEqual(latched_bounds, parent.instance_render.bounds);
    try std.testing.expectEqual(latched_hash, parent.instance_render.hash);
}

test "stage1: serial same-thread build+latch parity" {
    const InstancedMesh = @import("mesh.zig").InstancedMesh;
    const alloc = std.testing.allocator;
    var scene = stage1Scene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.meshes.deinit(alloc);
    defer scene.outline_meshes.deinit(alloc);
    defer scene.draws.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);
    defer scene.shadows.pass.binned_meshes.deinit(alloc);
    defer scene.shadows.pass.prepared.deinit(alloc);
    p7CpuShadowPass(&scene, alloc);

    const cam = Camera{ .free = camera_mod.FreeCamera.init("Cam1", .{}) };
    _ = try scene.addCamera(.{ .name = "Cam1", .camera = cam });

    var src = Mesh{
        .name = "s1_src",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    const n: usize = 6;
    const mem = try alloc.alloc(InstancedMesh, n);
    defer alloc.free(mem);
    const ptrs = try alloc.alloc(*InstancedMesh, n);
    defer alloc.free(ptrs);
    stage1FillInstances(&src, mem, ptrs, 10);
    mem[1].is_visible = false;
    var parent = Mesh{
        .name = "s1_parent",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .instances = .{ .items = ptrs, .capacity = n },
    };
    try scene.meshes.append(alloc, &parent);

    scene.buildPreparedFrame();
    try std.testing.expectEqual(scene.draws.backIndex(), scene.build_slot);
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    scene.prepareFrame();

    try std.testing.expect(scene.frame_prepared);
    try std.testing.expectEqual(scene.build_seq, scene.last_latched_seq);
    try std.testing.expectEqual(parent.instance_preview.count, parent.instance_render.count);
    try std.testing.expectEqual(@as(u32, 5), parent.instance_render.count);
    try std.testing.expectEqual(parent.instance_preview.bounds, parent.instance_render.bounds);
    try std.testing.expectEqual(scene.frame_id, parent.instance_render.staged_frame);
    try std.testing.expectEqual(scene.frame_id, scene.preparedDraws().frame_id);
    try std.testing.expectEqual(scene.retire_epoch, scene.preparedDraws().retire_epoch);
}

test "stage1: OOM build advances nothing, latch keeps previous, then recovers" {
    const InstancedMesh = @import("mesh.zig").InstancedMesh;
    const alloc = std.testing.allocator;
    var scene = stage1Scene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.meshes.deinit(alloc);
    defer scene.outline_meshes.deinit(alloc);
    defer scene.draws.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);
    defer scene.shadows.pass.binned_meshes.deinit(alloc);
    defer scene.shadows.pass.prepared.deinit(alloc);
    p7CpuShadowPass(&scene, alloc);

    const cam = Camera{ .free = camera_mod.FreeCamera.init("Cam1", .{}) };
    _ = try scene.addCamera(.{ .name = "Cam1", .camera = cam });

    var src = Mesh{
        .name = "s1_src",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    const n: usize = 4;
    const mem = try alloc.alloc(InstancedMesh, n);
    defer alloc.free(mem);
    const ptrs = try alloc.alloc(*InstancedMesh, n);
    defer alloc.free(ptrs);
    stage1FillInstances(&src, mem, ptrs, 0);
    var parent = Mesh{
        .name = "s1_parent",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .instances = .{ .items = ptrs, .capacity = n },
    };
    try scene.meshes.append(alloc, &parent);

    // Prime: funded build + latch publishes the complete state.
    scene.buildPreparedFrame();
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    scene.prepareFrame();
    const primed_bounds = parent.instance_render.bounds;
    const primed_hash = parent.instance_render.hash;
    try std.testing.expectEqual(@as(u32, 4), parent.instance_render.count);

    // Mutate live, then build unfunded: drop ALL scratch capacity so the
    // segment really allocates, and refuse the first alloc. The preview
    // must not advance (scene seq still does — the latch will skip it).
    mem[0].position = Vec3.new(100, 0, 0);
    for (&scene.draws.slots) |*slot| slot.primary.instance_matrices.clearAndFree(alloc);
    const real_alloc = scene.allocator;
    var failing = std.testing.FailingAllocator.init(alloc, .{ .fail_index = 0 });
    scene.allocator = failing.allocator();
    scene.buildPreparedFrame();
    scene.allocator = real_alloc;
    try std.testing.expectEqual(@as(u64, 2), scene.build_seq);
    try std.testing.expectEqual(@as(u64, 1), parent.instance_preview.build_seq);
    try std.testing.expect(failing.has_induced_failure);

    // Latch: the stale mesh is skipped — previous complete state stands,
    // no partial publish.
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    scene.prepareFrame();
    try std.testing.expectEqual(@as(u64, 2), scene.last_latched_seq);
    try std.testing.expectEqual(@as(u32, 4), parent.instance_render.count);
    try std.testing.expectEqual(primed_bounds, parent.instance_render.bounds);
    try std.testing.expectEqual(primed_hash, parent.instance_render.hash);

    // Recovery: a funded build + latch publishes the mutated state.
    scene.buildPreparedFrame();
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    scene.prepareFrame();
    try std.testing.expectEqual(@as(u64, 3), parent.instance_preview.build_seq);
    try std.testing.expect(parent.instance_render.bounds.max.x > primed_bounds.max.x + 10.0);
}

test "stage1: particle build on worker + latch on main freezes the frame" {
    const alloc = std.testing.allocator;
    var scene = stage1Scene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.meshes.deinit(alloc);
    defer scene.outline_meshes.deinit(alloc);
    defer scene.draws.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);
    defer scene.particles.systems.deinit(alloc);
    defer scene.particles.frame.deinit(alloc);
    defer scene.particles.build_frame.deinit(alloc);

    // CPU-only system with fake borrowed handle ids (never deinited via
    // ps.deinit — that would issue sg.destroy* on the fake ids; members are
    // freed manually below).
    const parts = try alloc.alloc(particles.Particle, 4);
    defer alloc.free(parts);
    const insts = try alloc.alloc(particles.ParticleInstanceData, 4);
    defer alloc.free(insts);
    const scratch = try alloc.alloc(u8, 4);
    defer alloc.free(scratch);
    var ps = ParticleSystem{
        .name = "s1",
        .allocator = alloc,
        .particles = parts,
        .instances = insts,
        .alive_scratch = scratch,
        .capacity = 4,
        .instance_buffer = .{ .id = 11 },
        .prng = std.Random.DefaultPrng.init(42),
    };
    ps.active_count = 3;
    try scene.particles.systems.append(alloc, &ps);

    // Worker = the game side: build only. Main only joins, then prepares.
    const Builder = struct {
        scene: *Scene,
        fn run(self: @This()) void {
            self.scene.buildPreparedFrame();
        }
    };
    const t = try std.Thread.spawn(.{}, Builder.run, .{Builder{ .scene = &scene }});
    t.join();
    try std.testing.expectEqual(@as(u64, 1), scene.particles.build_seq);
    try std.testing.expectEqual(@as(usize, 1), scene.particles.build_frame.items.len);
    try std.testing.expectEqual(@as(usize, 0), scene.particles.frame.items.len);

    // Live mutation after the build must not reach the render-owned frame.
    ps.active_count = 1;
    ps.instance_buffer = .{ .id = 99 };
    scene.prepareFrame();
    try std.testing.expect(scene.frame_prepared);
    try std.testing.expectEqual(@as(u64, 1), scene.particles.latched_seq);
    try std.testing.expectEqual(@as(usize, 1), scene.particles.frame.items.len);
    try std.testing.expectEqual(@as(usize, 3), scene.particles.frame.items[0].active_count);
    try std.testing.expectEqual(@as(u32, 11), scene.particles.frame.items[0].instance_buffer.id);
    // renderPrepared keeps reading `frame` only (headless no-op here).
    try std.testing.expectEqual(@as(usize, 1), scene.particles.build_frame.items.len);
    ps.active_count = 4;
    try std.testing.expectEqual(@as(usize, 3), scene.particles.frame.items[0].active_count);
}

test "stage1: physics build on worker + latch on main freezes the capture" {
    const alloc = std.testing.allocator;
    var scene = stage1Scene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.meshes.deinit(alloc);
    defer scene.outline_meshes.deinit(alloc);
    defer scene.draws.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);
    defer scene.physics.deinit(alloc);

    _ = scene.physics.enable(alloc, null);
    var pmesh = @import("testing.zig").testMesh("s1_phys");
    _ = try scene.physics.getWorld().?.createBody(&pmesh, .box, 0.0);
    scene.physics.show_debug = true;

    const Builder = struct {
        scene: *Scene,
        fn run(self: @This()) void {
            self.scene.buildPreparedFrame();
        }
    };
    const t = try std.Thread.spawn(.{}, Builder.run, .{Builder{ .scene = &scene }});
    t.join();
    try std.testing.expectEqual(@as(u64, 1), scene.physics.build_seq);
    try std.testing.expect(scene.physics.build_visible);
    try std.testing.expectEqual(@as(usize, 12), scene.physics.build_lines.items.len);
    try std.testing.expect(!scene.physics.prepared_visible);

    // Live mutation after the build must not reach the prepared capture.
    const x0 = scene.physics.build_lines.items[0].a.x;
    pmesh.position = Vec3.new(5, 0, 0);
    scene.prepareFrame();
    try std.testing.expect(scene.frame_prepared);
    try std.testing.expectEqual(@as(u64, 1), scene.physics.latched_seq);
    try std.testing.expect(scene.physics.prepared_visible);
    try std.testing.expectEqual(@as(usize, 12), scene.physics.prepared_lines.items.len);
    try std.testing.expectApproxEqAbs(x0, scene.physics.prepared_lines.items[0].a.x, 1e-4);
    scene.physics.step(0.016);
    try std.testing.expectApproxEqAbs(x0, scene.physics.prepared_lines.items[0].a.x, 1e-4);
}

test "stage1: instances cleared between build and latch take the regular path" {
    const InstancedMesh = @import("mesh.zig").InstancedMesh;
    const alloc = std.testing.allocator;
    var scene = stage1Scene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.meshes.deinit(alloc);
    defer scene.outline_meshes.deinit(alloc);
    defer scene.draws.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);
    defer scene.shadows.pass.binned_meshes.deinit(alloc);
    defer scene.shadows.pass.prepared.deinit(alloc);
    p7CpuShadowPass(&scene, alloc);

    const cam = Camera{ .free = camera_mod.FreeCamera.init("Cam1", .{}) };
    _ = try scene.addCamera(.{ .name = "Cam1", .camera = cam });

    var src = Mesh{
        .name = "s1_src",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    const n: usize = 3;
    const mem = try alloc.alloc(InstancedMesh, n);
    defer alloc.free(mem);
    const ptrs = try alloc.alloc(*InstancedMesh, n);
    defer alloc.free(ptrs);
    stage1FillInstances(&src, mem, ptrs, 0);
    var parent = Mesh{
        .name = "s1_parent",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .instances = .{ .items = ptrs, .capacity = n },
    };
    try scene.meshes.append(alloc, &parent);

    // Build while instanced: the preview records 3 (the build never
    // publishes — instance_render stays empty until a latch or fallback).
    scene.buildPreparedFrame();
    try std.testing.expectEqual(@as(u32, 3), parent.instance_preview.count);
    try std.testing.expectEqual(@as(u32, 0), parent.instance_render.count);

    // Drop all instances before the latch: both the latch and the fallback
    // skip empty instance lists, and the queue routes the mesh as regular —
    // the stale preview is never read, no instanced item is emitted.
    parent.instances.clearRetainingCapacity();
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    scene.prepareFrame();
    try std.testing.expect(scene.frame_prepared);
    try std.testing.expectEqual(@as(u64, 1), scene.last_latched_seq);
    try std.testing.expectEqual(@as(u32, 0), parent.instance_render.count);
    const draws = scene.preparedDraws();
    try std.testing.expectEqual(@as(usize, 0), draws.primary.opaque_instanced.items.len);
    try std.testing.expect(p7FindByMeshIndex(draws.primary.items.items, 0) != null);
}

test "stage1: mesh reorder + post-build add keeps surviving slices exact" {
    const InstancedMesh = @import("mesh.zig").InstancedMesh;
    const alloc = std.testing.allocator;
    var scene = stage1Scene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.meshes.deinit(alloc);
    defer scene.outline_meshes.deinit(alloc);
    defer scene.draws.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);
    defer scene.shadows.pass.binned_meshes.deinit(alloc);
    defer scene.shadows.pass.prepared.deinit(alloc);
    p7CpuShadowPass(&scene, alloc);

    const cam = Camera{ .free = camera_mod.FreeCamera.init("Cam1", .{}) };
    _ = try scene.addCamera(.{ .name = "Cam1", .camera = cam });

    var src = Mesh{
        .name = "s1_src",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    const mem_a = try alloc.alloc(InstancedMesh, 3);
    defer alloc.free(mem_a);
    const ptrs_a = try alloc.alloc(*InstancedMesh, 3);
    defer alloc.free(ptrs_a);
    stage1FillInstances(&src, mem_a, ptrs_a, 0);
    var mesh_a = Mesh{
        .name = "s1_a",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .instances = .{ .items = ptrs_a, .capacity = 3 },
    };
    const mem_b = try alloc.alloc(InstancedMesh, 2);
    defer alloc.free(mem_b);
    const ptrs_b = try alloc.alloc(*InstancedMesh, 2);
    defer alloc.free(ptrs_b);
    stage1FillInstances(&src, mem_b, ptrs_b, 50);
    var mesh_b = Mesh{
        .name = "s1_b",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .instances = .{ .items = ptrs_b, .capacity = 2 },
    };
    try scene.meshes.append(alloc, &mesh_a);
    try scene.meshes.append(alloc, &mesh_b);

    // Scratch is concatenated [A0 A1 A2 B0 B1]: A.lo=0, B.lo=3.
    scene.buildPreparedFrame();
    try std.testing.expectEqual(@as(usize, 0), mesh_a.instance_preview.scratch_lo);
    try std.testing.expectEqual(@as(usize, 3), mesh_b.instance_preview.scratch_lo);

    // Reorder (swapRemove unlinks A) and add a fresh mesh C whose preview
    // predates every build (build_seq 0).
    _ = scene.meshes.swapRemove(0);
    const mem_c = try alloc.alloc(InstancedMesh, 2);
    defer alloc.free(mem_c);
    const ptrs_c = try alloc.alloc(*InstancedMesh, 2);
    defer alloc.free(ptrs_c);
    stage1FillInstances(&src, mem_c, ptrs_c, 200);
    var mesh_c = Mesh{
        .name = "s1_c",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .instances = .{ .items = ptrs_c, .capacity = 2 },
    };
    try scene.meshes.append(alloc, &mesh_c);

    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    scene.prepareFrame();
    // B's slice [3,5) is exact: bounds match its own preview (A's segment
    // would give min.x near 0, B's gives 49).
    try std.testing.expectEqual(@as(u32, 2), mesh_b.instance_render.count);
    try std.testing.expectEqual(mesh_b.instance_preview.bounds, mesh_b.instance_render.bounds);
    try std.testing.expectApproxEqAbs(@as(f32, 49.0), mesh_b.instance_render.bounds.min.x, 1e-4);
    // C (never built) is skipped: no publish, no queue batch.
    try std.testing.expectEqual(@as(u64, 0), mesh_c.instance_preview.build_seq);
    try std.testing.expectEqual(@as(u32, 0), mesh_c.instance_render.count);
    try std.testing.expectEqual(std.math.maxInt(u64), mesh_c.instance_render.staged_frame);
    const draws = scene.preparedDraws();
    try std.testing.expectEqual(@as(usize, 1), draws.primary.opaque_instanced.items.len);
    try std.testing.expectEqual(@as(u32, 2), draws.primary.opaque_instanced.items[0].visible_instance_count);
}

test "stage1: truncated scratch between build and latch is skipped safely" {
    const InstancedMesh = @import("mesh.zig").InstancedMesh;
    const alloc = std.testing.allocator;
    var scene = stage1Scene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.meshes.deinit(alloc);
    defer scene.outline_meshes.deinit(alloc);
    defer scene.draws.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);
    defer scene.shadows.pass.binned_meshes.deinit(alloc);
    defer scene.shadows.pass.prepared.deinit(alloc);
    p7CpuShadowPass(&scene, alloc);

    const cam = Camera{ .free = camera_mod.FreeCamera.init("Cam1", .{}) };
    _ = try scene.addCamera(.{ .name = "Cam1", .camera = cam });

    var src = Mesh{
        .name = "s1_src",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    const n: usize = 2;
    const mem = try alloc.alloc(InstancedMesh, n);
    defer alloc.free(mem);
    const ptrs = try alloc.alloc(*InstancedMesh, n);
    defer alloc.free(ptrs);
    stage1FillInstances(&src, mem, ptrs, 0);
    var parent = Mesh{
        .name = "s1_parent",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .instances = .{ .items = ptrs, .capacity = n },
    };
    try scene.meshes.append(alloc, &parent);

    // Prime: funded build + latch publishes the complete state.
    scene.buildPreparedFrame();
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    scene.prepareFrame();
    const primed_bounds = parent.instance_render.bounds;
    try std.testing.expectEqual(@as(u32, 2), parent.instance_render.count);

    // Mutate, rebuild, then truncate the scratch before the latch (a
    // contract violation the latch must survive): the out-of-range slice is
    // skipped via the bounds check and the previous state stands.
    mem[0].position = Vec3.new(100, 0, 0);
    scene.buildPreparedFrame();
    try std.testing.expectEqual(@as(u64, 2), parent.instance_preview.build_seq);
    scene.draws.backSlot().primary.instance_matrices.clearRetainingCapacity();
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    scene.prepareFrame();
    try std.testing.expectEqual(@as(u64, 2), scene.last_latched_seq);
    try std.testing.expectEqual(@as(u32, 2), parent.instance_render.count);
    try std.testing.expectEqual(primed_bounds, parent.instance_render.bounds);

    // Recovery: a funded build + latch publishes the mutated state.
    scene.buildPreparedFrame();
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    scene.prepareFrame();
    try std.testing.expect(parent.instance_render.bounds.max.x > primed_bounds.max.x + 10.0);
}
