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
const RayHit = math.RayHit;

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
const outline_pass = @import("passes/outline_pass.zig");
const lights = @import("lights.zig");
const HemisphericLight = lights.HemisphericLight;
const DirectionalLight = lights.DirectionalLight;
const DirectionalLightOptions = lights.DirectionalLightOptions;
const PointLight = lights.PointLight;
const PointLightOptions = lights.PointLightOptions;
const SpotLight = lights.SpotLight;
const SpotLightOptions = lights.SpotLightOptions;
const AreaLight = lights.AreaLight;
const AreaLightOptions = lights.AreaLightOptions;
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
const tags_mod = @import("tags.zig");
pub const TagSet = tags_mod.TagSet;
pub const TagQuery = tags_mod.TagQuery;

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
const scene_clustered = @import("scene/clustered_lights.zig");
pub const ClusteredPointLight = lights.ClusteredPointLight;
pub const ClusteredPointLightOptions = lights.ClusteredPointLightOptions;
const scene_shadow = @import("scene/shadow_system.zig");
const scene_sky = @import("scene/sky_layer.zig");
const scene_probes = @import("scene/probe_layer.zig");
const scene_probe_render = @import("scene/probe_render.zig");
pub const ReflectionProbe = scene_probes.ReflectionProbe;
pub const ReflectionProbeOptions = scene_probes.ReflectionProbeOptions;
const scene_gui3d = @import("scene/gui3d_layer.zig");
pub const Ui3dPanel = scene_gui3d.Ui3dPanel;
pub const Ui3dPanelOptions = scene_gui3d.Ui3dPanelOptions;
pub const Ui3dFaceMode = scene_gui3d.Ui3dFaceMode;
pub const Ui3dPickHit = scene_gui3d.Ui3dPickHit;
const scene_postfx = @import("scene/postfx_stack.zig");
const scene_forward = @import("scene/forward_pipelines.zig");
const scene_particles = @import("scene/particle_layer.zig");
const scene_decals = @import("scene/decal_layer.zig");
const scene_trails = @import("scene/trail_layer.zig");
const scene_nav = @import("scene/nav_layer.zig");
const scene_physics = @import("scene/physics_layer.zig");
const softbody_mod = @import("softbody.zig");
pub const SoftBody = softbody_mod.SoftBody;
pub const SoftBodyLayer = softbody_mod.SoftBodyLayer;
pub const Cloth = softbody_mod.Cloth;
pub const ClothOptions = softbody_mod.ClothOptions;
pub const SphereCollider = softbody_mod.SphereCollider;
const scene_project = @import("scene/project_cache.zig");
const scene_retire = @import("scene/gpu_retire.zig");
const scene_ui_frame = @import("scene/ui_frame.zig");
pub const UiFrame = scene_ui_frame.UiFrame;
const scene_ui_capture = @import("scene/ui_capture.zig");
const scene_frame_draws = @import("scene/frame_draws.zig");
/// P7 published consumable draw payload (one coherent prepared frame).
pub const FrameDrawSlot = scene_frame_draws.FrameDrawSlot;
/// P7 three retained owning queue slots (the variable-list triple buffer
/// with a consumer pin/lease protocol).
pub const FrameDraws = scene_frame_draws.FrameDraws;
const scene_content = @import("scene/content.zig");
const scene_animation = @import("scene/animation_runtime.zig");
const scene_picking = @import("scene/picking.zig");
const scene_uniforms = @import("scene/uniforms.zig");
const FrameContext = scene_uniforms.FrameContext;
const scene_draw = @import("scene/draw.zig");
const scene_msaa = @import("scene/msaa.zig");
const scene_viewport_clear = @import("scene/viewport_clear.zig");
const scene_view_render = @import("scene/view_render.zig");
const scene_queue_builder = @import("scene/queue_builder.zig");
const scene_frame_render = @import("scene/frame_render.zig");
const scene_patch_instances = @import("scene/patch_instance_refs.zig");
const scene_frame_build = @import("scene/frame_build.zig");
const scene_frame_prepare = @import("scene/frame_prepare.zig");
pub const QueueBuildParams = scene_queue_builder.QueueBuildParams;
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
    viewport_clear: scene_viewport_clear.ViewportClearPass = .{},

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
    // Reflection probes (wave 25, v1): on-demand cube captures feeding the
    // PBR/standard ambient terms. Empty by default: with no enabled probe
    // every draw takes today's ambient/skybox path bit-identically.
    probes: scene_probes.ProbeLayer = .{},
    // Clustered forward point lights (wave 30, v1): render-owned tile
    // scratch + storage buffers for the EXTRA pool beyond the legacy
    // lanes. Empty by default: with no clustered light staged every draw
    // takes the exact legacy path (zeroed lanes, count uniform 0).
    clustered: scene_clustered.ClusteredGpuCache = .{},
    // 3D GUI panels (wave 28, v1): on-demand world-space UI quads. Empty
    // by default: with no panels every capture/draw hook early-outs with
    // zero sg.* calls, so rendering stays bit-identical.
    gui3d: scene_gui3d.Gui3dLayer = .{},
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
    // PBD cloth bodies + their deformable meshes (wave 29, v1). Empty by
    // default: every hook early-outs over an empty list, so update and
    // rendering stay bit-identical with no soft bodies.
    softbodies: SoftBodyLayer = .{},
    // Per-frame draw queues + instance staging.
    // P7 triple buffer: three retained owning slots encompassing PRIMARY +
    // ALL PIP view queues, outline items+skins, and prepared shadow
    // items+skins+bin ranges. prepare builds a back slot, render reads the
    // published front slot via preparedDraws() — the ONLY low-level draw
    // accessor (no legacy field aliases) — while holding a consumer pin on
    // it (see scene/frame_draws.zig for the ownership/lifecycle contract
    // and the pin/lease protocol).
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
    /// Prepare-side working/compat copy of the frame snapshot (wave 27):
    /// the fallback prepare path latches the mailbox (or a fresh pack) here
    /// and immediately stages it into the consumed back slot
    /// (`FrameDrawSlot.snapshot`); the build path mirrors the staged slot
    /// copy here for compatibility. Prepare reads the CONSUMED SLOT's staged
    /// copy (never this field past the latch point), and render/reuse/the UI
    /// latch never read this field at all — they read the front slot's staged
    /// snapshot — so a concurrent game-side mutation cannot tear the
    /// in-flight draw. Update/build NEVER touch this field (update||render
    /// overlap would race the draw); the game-private working copy on the
    /// producer side is `build_snapshot`.
    frame_snapshot: scene_snapshot.SceneFrameSnapshot = .{},
    /// Producer-owned build snapshot (game/update phase ownership): the exact
    /// generation the last `buildPreparedFrame` culled/built queues against.
    /// Plain value camera/pass state (Camera.name slices alias, but draw
    /// never dereferences names — no deep copy). `buildPreparedFrame` stages
    /// it into the claim slot's `snapshot` by value; the latch consumes the
    /// STAGED copy (staged wins over a post-build `build_snapshot` mutation),
    /// mirroring it into `frame_snapshot` for compatibility. The build NEVER
    /// reads/writes `frame_snapshot`, otherwise it races the draw under
    /// update||render.
    build_snapshot: scene_snapshot.SceneFrameSnapshot = .{},
    /// Flag indicating whether prepareFrame() has already run for this frame.
    frame_prepared: bool = false,
    /// Reuse guard owned by `renderReuse` on the context thread (set around
    /// the inner `render()` call, never observed concurrently): tells render
    /// to skip the `prepareFrame` fallback and the profiler `recordFrame`
    /// tail, so the already-consumed front is re-drawn as-is.
    rendering_reuse: bool = false,
    /// Consecutive `renderReuse` presents without an intervening successful
    /// prepare (context thread only: reset by `prepareFrame`, bumped by
    /// `renderReuse`). Observable staleness: retire/upload intents pile up
    /// for exactly this many frames (`GpuRetireQueue.pending_cap` bounds the
    /// pileup, `retainedCount`/`cappedDropCount` expose it). Read via
    /// `reuseStreak()`; never written by update.
    reuse_streak: u64 = 0,
    /// True only while prepareFrame's own flushPendingGpuUploads runs
    /// (context thread): the render pipeline commits that frame's buffer,
    /// so the flush must not. Standalone flushes (quiesced-context
    /// completion outside the frame pipeline) commit themselves — see the
    /// flushPendingGpuUploads tail.
    flush_in_prepare: bool = false,
    /// Monotonic presented-frame counter, written only on the context thread:
    /// every presented frame (normal renders and `renderReuse` re-presents
    /// alike) bumps it once and uses it as the profiler `recordFrame`
    /// frame_id label, so wall-pacing stats see consecutive presents with no
    /// gaps. Re-presented rows repeat the consumed frame's counters while
    /// pacing/timestamps are real.
    profiler_frame_seq: u64 = 0,

    /// Stage 1 producer-build handoff (game/update phase → prepare latch):
    /// `buildPreparedFrame` (game side, CPU-only, sg-free) first commits the
    /// last published latch outcomes to the live meshes (game-side
    /// `commitPublishedRecords` over the front slot — the old guarded
    /// `instance_render` write-back, ordered after publish, never concurrent
    /// with the context), then stages instance
    /// matrices into the back-slot scratch + per-mesh previews, freezes the
    /// slot-owned staged records, and captures
    /// the particle/physics CPU build frames, then bumps `build_seq`.
    /// `prepareFrame` (context side) consumes the build when `build_seq !=
    /// last_latched_seq` (GPU halves over the slot records + latch copies,
    /// no CPU restaging, no live mesh reads or writes — outcomes land in
    /// the slot records for the game-side commit) and
    /// otherwise runs the historical inline path — so behavior stays correct
    /// when `buildPreparedFrame` is never called. Two builds before a latch:
    /// newest wins (single preview store recomputed, scratch overwritten,
    /// record array reset and refilled).
    /// A missed build never resurfaces a stale frame: every prepare (latch
    /// or fallback) publishes a fresh slot; render reuses the last published
    /// front only when prepare itself is not called (unchanged).
    /// Handoff edge (wave 30, atomic): `build_seq` is release-stored by the
    /// producer (`BuildClaim.publish`) only after the whole build payload is
    /// staged (slot queues/records/snapshot, particle/physics build frames,
    /// `build_slot`), and acquire-loaded by the context latch (`prepareFrame`
    /// freshness check + `last_latched_seq` stamp) — the release/acquire pair
    /// orders the payload before the generation the latch consumes. The claim
    /// reserve (`tryClaimBuildSlot`) monotonic-loads the committed generation
    /// (single producer: no commit races, the publish is the release edge).
    /// `last_latched_seq` is stamped context-side only (monotonic store; the
    /// producer never touches it) — atomic so the handoff-edge comparison
    /// itself is race-free once game||prepare overlap. The build touches no
    /// stats/profiler/frame_id/retire_epoch.
    build_seq: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    last_latched_seq: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    /// Game-side UI CPU packet sequence (lock-free-publication slice 2, b):
    /// `stageUiPacket` (game side, sg-free) records live canvas geometry
    /// into the back slot's `ui_vertices`/`ui_indices` + `ui_packet` header
    /// and bumps `ui_packet_seq`; `prepareFrame` latches the packet into
    /// `ui_frame` when `ui_packet_seq != last_latched_ui_seq` and stamps it
    /// consumed. Independent of `build_seq`: UI-only apps never call
    /// `buildPreparedFrame`. Handoff edge (wave 30, atomic):
    /// `stageUiPacketInto` release-bumps `ui_packet_seq` only after the slot
    /// packet bytes + header are staged, and the context latch
    /// (`captureUiFrame`) acquire-loads it before consuming the packet and
    /// monotonic-stamping `last_latched_ui_seq` consumed — the release/
    /// acquire pair orders the packet before the generation the latch reads.
    /// `last_latched_ui_seq` is stamped context-side only (the producer never
    /// touches it). The stage touches no stats/profiler/frame_id/epoch and
    /// no GPU state at all (CPU list copies + seq bump only).
    ui_packet_seq: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    last_latched_ui_seq: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    /// How many prepares actually consumed a staged packet as the geometry
    /// source (the `capturePacket` path, not the legacy canvas read and not
    /// the staged-absence clear). Observability only: lets an app/fixture
    /// prove the staged path is live rather than merely staged. Monotonic;
    /// read via `uiPacketLatchedCount()`. Plain u64, stays plain: incremented
    /// only by the context latch (`captureUiFrame`) and read context-side —
    /// never shared across the handoff edge.
    ui_packet_latched: u64 = 0,
    /// Back-slot index the last `buildPreparedFrame` wrote (stamped by
    /// `BuildClaim.publish` with a release store BEFORE the `build_seq`
    /// release, so it rides the same handoff edge); the latch acquire-reads
    /// it after observing a fresh seq and asserts it still is the back index
    /// (no intervening publish).
    build_slot: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),
    /// Game-owned queue-build stats accumulator (stage-2 increment B,
    /// slot-staged since wave 31 second slice): `buildIntoClaimedSlot`
    /// clears this at build start, the game-side `buildQueuesInto`
    /// accumulates the queue counters here (never `self.stats`, which stays
    /// context-owned), and the build freezes a plain copy into the claimed
    /// slot's `build_stats` (staged-wins over any post-build accumulation,
    /// same precedent as the snapshot). The prepare latch merges the SLOT
    /// copy into `self.stats` via `SceneStats.mergeFrom` (after the latch's
    /// own reset) and zeroes both copies. The live field itself never
    /// crosses the handoff edge — a concurrent game-side accumulation
    /// cannot race the context-side merge.
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

    // ---- Reflection probes (wave 25, v1). ----
    //
    // On-demand environment captures: each probe renders the prepared draw
    // list plus the sky into a 128px cube target from its position, then
    // nearby PBR/standard draws sample it for their ambient terms (nearest
    // enabled probe within its radius wins, no blending). Captures run on
    // the context thread inside `render`, at most one per frame (lowest
    // dirty + enabled index first; the rest wait for later frames), and
    // never inside `renderReuse` (reuse re-presents the captured state).
    // Before a probe's first capture lands, selection skips it, so new (or
    // disabled) probes leave rendering bit-identical.
    //
    // Capacity: at most `scene_probes.max_probes` (4); `addReflectionProbe`
    // past the cap is a hard `error.TooManyReflectionProbes`.
    //
    // Explicit non-goals (see scene/probe_layer.zig): box projection /
    // parallax, probe blending / weights, per-frame real-time updates,
    // specular occlusion, irradiance SH, editor tooling.

    /// Adds a reflection probe at `position`; returns its index. The probe
    /// starts dirty + uncaptured: the next `render` captures it (one capture
    /// per frame) and draws fall back until then.
    pub fn addReflectionProbe(self: *Scene, position: Vec3, options: ReflectionProbeOptions) error{TooManyReflectionProbes}!usize {
        return self.probes.add(position, options);
    }

    /// Removes probe `index`, retiring its cube target through the epoch
    /// retire queue (safe under update||render overlap: the context thread
    /// destroys at the next flush). Order-preserving: higher indices shift
    /// down. Out-of-range indices are a no-op.
    pub fn removeReflectionProbe(self: *Scene, index: usize) void {
        self.probes.remove(self.allocator, &self.gpu_retire, index);
    }

    /// Live probe state (position/radius/enabled/intensity are freely
    /// mutable game-side under update-vs-prepare exclusion). Null when
    /// out of range.
    pub fn getReflectionProbe(self: *Scene, index: usize) ?*ReflectionProbe {
        if (index >= self.probes.count) return null;
        return &self.probes.probes[index];
    }

    pub fn reflectionProbeCount(self: *const Scene) usize {
        return self.probes.count;
    }

    /// Requests an on-demand recapture of probe `index` on the next render
    /// (actual GPU work happens there, at most one probe per frame).
    /// Out-of-range is a no-op.
    pub fn captureReflectionProbe(self: *Scene, index: usize) void {
        self.probes.markDirty(index);
    }

    /// Requests a recapture of every probe (each still captures on its own
    /// frame: one per frame maximum).
    pub fn captureDirtyReflectionProbes(self: *Scene) void {
        self.probes.markAllDirty();
    }

    /// How many probes currently want a capture (observability for tests
    /// and tooling).
    pub fn probeDirtyCount(self: *const Scene) usize {
        return self.probes.dirtyCount();
    }

    // ---- 3D GUI panels (wave 28, v1). ----
    //
    // On-demand world-space UI: each panel owns a private UICanvas (the app
    // draws into it with the normal canvas API) rendered into a private
    // color RT sized to the canvas resolution, then drawn in the main pass
    // as an unlit/emissive double-sided quad. Captures run on the context
    // thread inside `render`, at most `max_captures_per_frame` (1) per frame
    // (lowest dirty + enabled index first; the rest wait), and never inside
    // `renderReuse`. Before a panel's first capture lands, drawing and
    // picking skip it, so new (or disabled) panels leave rendering
    // bit-identical.
    //
    // Capacity: at most `scene_gui3d.max_panels` (4); `addUi3dPanel` past
    // the cap is a hard `error.TooManyUi3dPanels`. Face mode v1 is fixed yaw
    // only (see scene/gui3d_layer.zig for the policy, picking contract with
    // `injectPointer`, and the explicit non-goals).

    /// Adds a 3D GUI panel at `position`; returns its index. The name is
    /// duped (Scene owns panel names; freed on remove/deinit). The panel
    /// starts dirty + uncaptured: the next `render` captures it (one capture
    /// per frame) while drawing/picking skip it until then. Headless-safe
    /// (only CPU lists + name allocation; GPU targets are lazy).
    pub fn addUi3dPanel(
        self: *Scene,
        name: []const u8,
        position: Vec3,
        options: Ui3dPanelOptions,
    ) error{ TooManyUi3dPanels, InvalidUi3dPanelSize, OutOfMemory }!usize {
        return self.gui3d.add(self.allocator, name, position, options);
    }

    /// Removes panel `index`, retiring its RT target through the epoch
    /// retire queue (safe under update||render overlap: the context thread
    /// destroys at the next flush). Order-preserving: higher indices shift
    /// down. Out-of-range indices are a no-op.
    pub fn removeUi3dPanel(self: *Scene, index: usize) void {
        self.gui3d.remove(self.allocator, &self.gpu_retire, index);
    }

    /// Live panel state (position/yaw/enabled/canvas are freely mutable
    /// game-side under update-vs-prepare exclusion). Null when out of range.
    pub fn getUi3dPanel(self: *Scene, index: usize) ?*Ui3dPanel {
        return self.gui3d.get(index);
    }

    /// Live panel state by name. Null when no panel matches.
    pub fn getUi3dPanelByName(self: *Scene, name: []const u8) ?*Ui3dPanel {
        return self.gui3d.getByName(name);
    }

    pub fn ui3dPanelCount(self: *const Scene) usize {
        return self.gui3d.panelCount();
    }

    /// Requests an on-demand recapture of panel `index` on the next render
    /// (actual GPU work happens there, at most one panel per frame).
    /// Out-of-range is a no-op.
    pub fn markUi3dPanelDirty(self: *Scene, index: usize) void {
        self.gui3d.markDirty(index);
    }

    /// Requests a recapture of every panel (each still captures on its own
    /// frame: one per frame maximum).
    pub fn markAllUi3dPanelsDirty(self: *Scene) void {
        self.gui3d.markAllDirty();
    }

    /// How many panels currently want a capture (observability for tests
    /// and tooling).
    pub fn ui3dDirtyCount(self: *const Scene) usize {
        return self.gui3d.dirtyCount();
    }

    /// Picks the nearest drawable 3D panel under the cursor. The ray is
    /// built from the STAGED snapshot camera (`primary_cam.view_proj`,
    /// fullscreen mapping — never the live camera), so game-side callers
    /// under update-vs-prepare exclusion observe the same camera the frame
    /// was prepared against. Returns the panel index plus canvas pixel
    /// coordinates; the app routes those into
    /// `panel.injectPointer(x, y, pressed)` itself (no focus system, no
    /// keyboard routing in v1). Null when no camera/stage is present or no
    /// drawable panel is hit. Headless-safe (pure CPU; no `sapp.*` reads —
    /// snapshot dims gate instead).
    pub fn pickUi3dPanel(self: *Scene, mouse_x: f32, mouse_y: f32) ?Ui3dPickHit {
        const front = self.draws.frontIndex();
        const snap = &self.draws.slots[front].snapshot;
        if (!snap.has_camera) return null;
        const w: f32 = @floatFromInt(snap.screen_w);
        const h: f32 = @floatFromInt(snap.screen_h);
        if (w <= 0.0 or h <= 0.0) return null;
        const inv_vp = snap.primary_cam.view_proj.invert() orelse return null;
        const ndc_x = (2.0 * mouse_x) / w - 1.0;
        const ndc_y = 1.0 - (2.0 * mouse_y) / h;
        const near_pt = inv_vp.transformPoint(Vec3.new(ndc_x, ndc_y, 0.0));
        const far_pt = inv_vp.transformPoint(Vec3.new(ndc_x, ndc_y, 1.0));
        const dir = far_pt.sub(near_pt).normalize();
        return self.gui3d.pick(Ray.new(near_pt, dir));
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

    // ---- Rect area lights (wave 26, v1). ----
    //
    // A bounded, additive, OFF-by-default feature: with zero area lights
    // every appended uniform lane is zeroed and all five forward shaders
    // skip the area loop, rendering bit-identically to today.
    //
    // Orientation is two half-extent vectors (`right` = local +X axis scaled
    // by half-width, `up` = local +Y axis scaled by half-height; the
    // emitting normal is cross(right, up)). Degenerate rects (zero area)
    // emit nothing.
    //
    // Explicit non-goals in v1: area-light shadows (unshadowed, document
    // accordingly), LTC integration (closest-point approximation instead —
    // see the shader header), glTF import (KHR_lights_punctual has no rect
    // type; area lights are API-only), persistence (session-local like
    // directional fills: save/load never writes them, load never clears
    // live ones).
    //
    // Capacity: at most `lights.max_area_lights` (2); past the cap is a
    // hard `error.TooManyAreaLights`.

    /// Appends a rect area light; returns the live pointer (creation order
    /// == uniform slot order). Hard-errors past the cap.
    pub fn addAreaLight(self: *Scene, name: []const u8, options: AreaLightOptions) !*AreaLight {
        return self.lights.addAreaLight(self.allocator, name, options);
    }

    /// Removes area light `index`, destroying it. Order-preserving: higher
    /// indices shift down. Out-of-range indices are a no-op.
    pub fn removeAreaLight(self: *Scene, index: usize) void {
        self.lights.removeAreaLight(self.allocator, index);
    }

    /// Live area-light state (center/right/up/color/intensity/enabled are
    /// freely mutable game-side under update-vs-prepare exclusion). Null
    /// when out of range.
    pub fn getAreaLight(self: *Scene, index: usize) ?*AreaLight {
        return self.lights.getAreaLight(index);
    }

    /// Number of owned area lights (at most lights.max_area_lights).
    pub fn areaLightCount(self: *const Scene) usize {
        return self.lights.areaLightCount();
    }

    // ---- Clustered forward point lights (wave 30, v1). ----
    //
    // A bounded, additive, OFF-by-default pool of EXTRA point lights beyond
    // the legacy 4-slot top-k lanes: with zero clustered lights every
    // appended pack lane is zeroed, the tile build writes empty headers,
    // and all five forward shaders skip the clustered loop, rendering
    // bit-identically to today.
    //
    // Staging (1-frame lag, probe-pack pattern): add/remove/moves mutate
    // game-side values under update-vs-prepare exclusion; `updateLights`
    // stages them into the light handoff, `packFrameSnapshot` freezes the
    // staged copy, and the context thread rebuilds the 2D screen tiles
    // from that frozen copy during render — so a move is visible on the
    // NEXT presented frame, never the current one. Removing a light also
    // retires the live tile storage buffers through the epoch retire queue
    // (safe under update||render overlap: the context thread destroys at
    // the next flush); the next context rebuild recreates exact-fit
    // buffers.
    //
    // Explicit non-goals in v1: shadows for clustered lights (unshadowed,
    // document accordingly), depth-aware tiles (2D full-depth columns with
    // documented over-inclusion), hysteresis/fades, glTF import, and
    // persistence (session-local like directional fills and area lights:
    // save/load never writes the pool, load never clears live lights).
    //
    // Capacity: at most `lights.max_clustered_lights` (64); past the cap
    // is a hard `error.TooManyClusteredLights`.

    /// Appends a clustered forward point light at `position`; returns its
    /// index (pack order follows creation order). Hard-errors past the cap.
    /// Also retires any live tile storage buffers (uniform discipline, so
    /// stale GPU can never serve the next frame).
    pub fn addClusteredPointLight(self: *Scene, position: Vec3, options: ClusteredPointLightOptions) error{TooManyClusteredLights}!usize {
        const idx = try self.lights.addClusteredPointLight(position, options);
        self.clustered.retireBuffers(self.allocator, &self.gpu_retire);
        return idx;
    }

    /// Removes clustered light `index`, retiring its tile storage buffers
    /// through the epoch retire queue (safe under update||render overlap).
    /// Order-preserving: higher indices shift down. Out-of-range indices
    /// are a no-op (and never retire).
    pub fn removeClusteredPointLight(self: *Scene, index: usize) void {
        if (index >= self.lights.clusteredPointLightCount()) return;
        self.lights.removeClusteredPointLight(index);
        self.clustered.retireBuffers(self.allocator, &self.gpu_retire);
    }

    /// Live clustered-light state (position/color/intensity/radius/enabled
    /// are freely mutable game-side under update-vs-prepare exclusion;
    /// edits stage through updateLights and appear next frame). Null when
    /// out of range.
    pub fn getClusteredPointLight(self: *Scene, index: usize) ?*ClusteredPointLight {
        return self.lights.getClusteredPointLight(index);
    }

    /// Number of owned clustered lights (at most
    /// lights.max_clustered_lights).
    pub fn clusteredPointLightCount(self: *const Scene) usize {
        return self.lights.clusteredPointLightCount();
    }

    // ---- Soft bodies / PBD cloth (wave 29, v1). ----
    //
    // A bounded, additive, OFF-by-default feature: with zero soft bodies
    // `updateSoftBodies` and the flush loop iterate an empty list, so update
    // and rendering stay bit-identical to today.
    //
    // `addSoftBodyCloth` creates a PBD cloth solver plus the textured
    // double-sided standard-material mesh it deforms every scene update
    // (game side, inside `Scene.update`'s sequential ordering). Per-frame
    // vertex uploads follow the sanctioned dynamic-upload discipline from
    // the morph audit (game stages CPU vertices + sets the pending flag,
    // the context-thread flush issues the single `sg.updateBuffer` +
    // meter record — the TrailMesh pattern, not the morph fields). Removal
    // always retires the mesh through the epoch retire queue, so it is safe
    // from the game thread under update||render overlap.
    //
    // Solver state is session-local (like particle systems and trail nodes);
    // serialization never writes it. See softbody.zig for the solver
    // parameters, the exact strain-limiting statement and the non-goals.
    //
    // Capacity: at most `softbody.max_bodies` (4); past the cap is a hard
    // `error.TooManySoftBodies`.

    /// Creates a cloth + its deformable mesh; returns the live body (index
    /// order == creation order). Hard-errors past the cap or on invalid
    /// options. Headless/off-context safe (GPU buffers defer to the first
    /// context-thread flush).
    pub fn addSoftBodyCloth(self: *Scene, name: []const u8, options: ClothOptions) softbody_mod.SoftBodyError!*SoftBody {
        return self.softbodies.create(self, name, options);
    }

    /// Removes body `index`: unlinks + frees the solver side and retires its
    /// mesh through the epoch queue (context thread completes the teardown
    /// at the next flush). Order-preserving: higher indices shift down.
    /// Out-of-range indices are a hard `error.UnknownSoftBody`.
    pub fn removeSoftBodyCloth(self: *Scene, index: usize) softbody_mod.SoftBodyError!void {
        const mesh = try self.softbodies.extractAt(self.allocator, index);
        _ = self.removeMesh(mesh);
        self.gpu_retire.retireMesh(self.allocator, mesh);
    }

    /// Live body state (solver fields, pins, wind, colliders are freely
    /// mutable game-side under update-vs-prepare exclusion). Null when out
    /// of range.
    pub fn getSoftBody(self: *Scene, index: usize) ?*SoftBody {
        return self.softbodies.get(index);
    }

    /// Live body state by mesh name. Null when no body matches.
    pub fn getSoftBodyByName(self: *Scene, name: []const u8) ?*SoftBody {
        for (self.softbodies.bodies.items) |b| {
            if (std.mem.eql(u8, b.mesh.name, name)) return b;
        }
        return null;
    }

    /// Number of owned cloth bodies (at most softbody.max_bodies).
    pub fn softBodyCount(self: *const Scene) usize {
        return self.softbodies.count();
    }

    /// Advances every enabled cloth and stages its vertex upload (game side,
    /// no sg.*). Called from `Scene.update`; apps that need their own time
    /// base may also drive it explicitly like `updateTrails`.
    pub fn updateSoftBodies(self: *Scene, dt: f32) void {
        self.softbodies.update(dt);
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
        // Soft bodies: drop the cloth body bound to this mesh, if any (frees
        // the solver side only; the mesh itself proceeds below as usual).
        self.softbodies.removeForMesh(self.allocator, mesh);
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

    // ---- Mesh search, tags & queries ----

    pub fn getMeshByName(self: *Scene, name: []const u8) ?*Mesh {
        for (self.meshes.items) |m| {
            if (std.mem.eql(u8, m.name, name)) return m;
        }
        return null;
    }

    /// Returns a list of all scene meshes that have the specified tag (case-insensitive).
    pub fn getMeshesByTag(self: *Scene, allocator: std.mem.Allocator, tag_str: []const u8) !std.ArrayListUnmanaged(*Mesh) {
        var list = std.ArrayListUnmanaged(*Mesh).empty;
        errdefer list.deinit(allocator);
        for (self.meshes.items) |m| {
            if (m.hasTag(tag_str)) {
                try list.append(allocator, m);
            }
        }
        return list;
    }

    /// Returns a list of all scene meshes matching the given boolean tag query expression (e.g. "enemy & (boss | elite)").
    pub fn getMeshesByQuery(self: *Scene, allocator: std.mem.Allocator, query_str: []const u8) !std.ArrayListUnmanaged(*Mesh) {
        var list = std.ArrayListUnmanaged(*Mesh).empty;
        errdefer list.deinit(allocator);
        var q = try TagQuery.parse(allocator, query_str);
        defer q.deinit();

        for (self.meshes.items) |m| {
            if (m.tags.matches(&q)) {
                try list.append(allocator, m);
            }
        }
        return list;
    }

    /// Counts how many scene meshes have the specified tag.
    pub fn countMeshesByTag(self: *Scene, tag_str: []const u8) usize {
        var n: usize = 0;
        for (self.meshes.items) |m| {
            if (m.hasTag(tag_str)) n += 1;
        }
        return n;
    }

    /// Counts how many scene meshes match the given boolean tag query expression.
    pub fn countMeshesByQuery(self: *Scene, query_str: []const u8) usize {
        var n: usize = 0;
        for (self.meshes.items) |m| {
            if (m.matchesTagQuery(query_str)) n += 1;
        }
        return n;
    }

    /// Finds the first scene mesh with the specified tag, or null if none found.
    pub fn findFirstMeshByTag(self: *Scene, tag_str: []const u8) ?*Mesh {
        for (self.meshes.items) |m| {
            if (m.hasTag(tag_str)) return m;
        }
        return null;
    }

    /// Finds the first scene mesh matching the boolean tag query expression, or null if none found.
    pub fn findFirstMeshByQuery(self: *Scene, query_str: []const u8) ?*Mesh {
        for (self.meshes.items) |m| {
            if (m.matchesTagQuery(query_str)) return m;
        }
        return null;
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

    /// Raycasts into the scene and returns the closest hit mesh that matches the tag query.
    pub fn pickWithRayTag(self: *Scene, r: Ray, query_str: []const u8) PickingInfo {
        var closest_dist: f32 = std.math.inf(f32);
        var best_hit: ?RayHit = null;
        var best_mesh: ?*Mesh = null;

        for (self.meshes.items) |mesh| {
            if (!mesh.matchesTagQuery(query_str)) continue;
            if (mesh.is_lod_child or mesh.is_decal or mesh.gpu_pending) continue;

            const box = if (mesh.cached_frame == self.frame_id) mesh.cached_aabb else mesh.getWorldBoundingBox();
            if (r.intersectsAABBNormal(box)) |hit| {
                if (hit.distance < closest_dist) {
                    closest_dist = hit.distance;
                    best_hit = hit;
                    best_mesh = mesh;
                }
            }
        }

        return .{
            .hit = best_mesh != null,
            .distance = if (best_mesh != null) closest_dist else 0.0,
            .picked_mesh = best_mesh,
            .picked_point = if (best_hit) |h| h.point else Vec3.zero,
            .picked_normal = if (best_hit) |h| h.normal else Vec3.up,
            .picked_instance = null,
        };
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

    /// Forward pipeline set matching the MSAA main-target shape; created
    /// lazily (and recreated on count changes) on the first MSAA frame. The
    /// returned pointer aliases Scene state.
    pub fn ensureForwardMsaa(self: *Scene, samples: i32) *scene_forward.ForwardPipelines {
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

    /// Shared queue/shadow/outline build parameters (stage-2 increment B):
    /// one internal function `buildQueuesInto` fills a `FrameDrawSlot` from
    /// these plus the explicit `snap` cameras (never `Scene.frame_snapshot`
    /// nor the slot's staged copy: the build must not read the consumed
    /// render snapshot). Fallback passes the staged slot snapshot with
    /// `frame_id` as `cache_key`, `&self.stats`, the snapshot primary eye,
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
    /// `stats` stays a direct pointer: the game-side build passes
    /// `&self.build_stats` (the live accumulator, frozen into the claim
    /// slot at build end and merged from there by the latch), the fallback
    /// passes `&self.stats` directly.
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
        scene_queue_builder.prepareViewQueues(self, queues, cam_snap, sky_texture, ibl_intensity, cache_key, stats, instances_prepared, instance_source);
    }

    // internal, used by scene tests
    pub fn buildQueuesInto(self: *Scene, back: *FrameDrawSlot, params: QueueBuildParams) void {
        scene_queue_builder.buildQueuesInto(self, back, params);
    }

    pub fn renderSceneView(
        self: *Scene,
        cam_snap: scene_snapshot.CameraSnapshot,
        queues: *const scene_render_queue.RenderQueues,
        outline_items: []const outline_pass.OutlineDrawItem,
        outline_skins: []const [scene_render_queue.MAX_BONES]Mat4,
        samples: i32,
        snap: *const scene_snapshot.SceneFrameSnapshot,
        env: scene_draw.Environment,
    ) void {
        scene_view_render.renderSceneView(self, cam_snap, queues, outline_items, outline_skins, samples, snap, env);
    }

    /// Runs at most one pending reflection-probe capture (wave 25, v1).
    /// Context thread only; called from `render` between the shadow depth
    /// pass and the main pass, never from `renderReuse` (which re-presents
    /// the consumed front, probe snapshot included). Takes the staged slot
    /// snapshot from the caller (the front slot's copy `render` already
    /// reads) — never the live `frame_snapshot`.
    ///
    /// What one capture does, in order: ensure the probe's GPU target (fail
    /// closed when headless or on creation failure — the probe stays dirty
    /// and is retried on a later frame), render the six cube faces from the
    /// probe position (prepared PRIMARY draw list: opaque + opaque-instanced
    /// + transparent, plus the sky; no PIP views, outline, particles,
    /// physics-debug, or UI in v1), then run the box-prefilter blit chain
    /// over the mip levels. No `UploadQueue`/texture-streaming interaction
    /// and no `sg.updateBuffer` traffic on this path — only draws into
    /// probe-owned targets — so the single-upload-per-frame discipline is
    /// untouched. When several probes are dirty, the lowest dirty + enabled
    /// index captures now and the rest wait for later frames (one capture
    /// per frame maximum). The fresh content reaches draws one prepare
    /// later (the snapshot is packed in `prepareFrame`, before `render`
    /// captures) — a documented one-frame lag.
    pub fn captureDirtyProbes(self: *Scene, snap: *const SceneFrameSnapshot) void {
        scene_probe_render.captureDirtyProbes(self, snap);
    }

    /// Runs at most `max_captures_per_frame` (1) pending 3D-GUI panel
    /// captures (wave 28, v1). Lowest dirty + enabled index first; the rest
    /// wait for later frames. Fail-closed like the probe path (headless or
    /// creation failure keeps the panel dirty for retry). The layer owns the
    /// upload + offscreen RT pass; this only budgets the count.
    pub fn captureDirtyUi3dPanels(self: *Scene) void {
        var n: usize = 0;
        while (n < scene_gui3d.max_captures_per_frame) : (n += 1) {
            const idx = self.gui3d.nextDirtyIndex() orelse return;
            if (!self.gui3d.capturePanel(self.allocator, &self.gpu_retire, idx)) return;
        }
    }

    /// Packs the current camera, light, shadow, and environment state into an immutable
    /// frame snapshot that can be published to the render thread.
    pub fn packFrameSnapshot(self: *Scene, aspect: f32, cur_w: i32, cur_h: i32) scene_snapshot.SceneFrameSnapshot {
        return scene_snapshot.packFrameSnapshot(self, aspect, cur_w, cur_h);
    }

    /// Publishes a complete frame snapshot through the lock-free mailbox.
    /// When the mailbox is saturated (consumer lagging, both slots
    /// published), stale published slots are drained first so the NEWEST
    /// snapshot wins — otherwise prepareFrame's takeLatest would resurface
    /// an older published frame over the newer fallback.
    pub fn publishFrameSnapshot(self: *Scene, aspect: f32, cur_w: i32, cur_h: i32) void {
        scene_snapshot.publishFrameSnapshot(self, aspect, cur_w, cur_h);
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
    /// Game-side UI CPU packet staging (lock-free-publication slice 2, b):
    /// records the live canvas CPU geometry into the back slot's
    /// slot-owned packet lists (`ui_vertices`/`ui_indices` + `ui_packet`
    /// header) for the prepare latch to consume into `ui_frame`.
    ///
    /// Runs on the game side (any non-pool thread under update-vs-prepare
    /// exclusion — same exclusion as `buildPreparedFrame`), sg-free by
    /// design: CPU list copies into retained slot capacity + plain handle
    /// stamps + one seq bump. No GPU upload and no GPU handle mutation on
    /// this path — the draw handles (pipeline, resolved font view/sampler,
    /// canvas buffer ids) and the writer watermark (`ui_upload_seq`) are
    /// frozen as plain values for the latch to consume instead of reading
    /// the live canvas; screen dims already come from the staged slot
    /// snapshot at latch time. Call AFTER the tick's UI build
    /// (canvas holds the frame's geometry) and, when `buildPreparedFrame`
    /// is also used, AFTER it (the build resets the back slot and would
    /// wipe an earlier packet; the latch then degrades to the legacy canvas
    /// read, which still holds the content — correct, just an extra copy).
    /// Staging twice before a latch: newest wins (lists overwritten, single
    /// header). OOM mid-stage: the packet is marked invalid and the latch
    /// takes the legacy path (the canvas still holds the content) — never a
    /// partial packet. Skipping the stage is always legal: with
    /// `ui_packet_seq == last_latched_ui_seq` the legacy path runs
    /// bit-identically and the slot lists stay empty.
    pub fn stageUiPacket(self: *Scene) void {
        // Sequential-only: targets the unlocked back index — must never run
        // concurrently with prepareFrame (on the concurrent path use
        // `BuildClaim.stageUi`, which targets the claimed slot, never the
        // one prepare is latching). Since wave 31 prepare additionally holds
        // its slot WRITING for the whole prepare, a concurrent legacy stage
        // would not even target the latched slot (the unlocked back index
        // skips WRITING slots) — it would stage into a slot nobody latches,
        // silently dropping the packet. The prohibition stands.
        self.stageUiPacketInto(self.draws.backIndex());
    }

    /// Shared UI stage core behind `stageUiPacket` and `BuildClaim.stageUi`:
    /// the exact historical stage body targeted at `slot`.
    fn stageUiPacketInto(self: *Scene, slot: usize) void {
        scene_ui_capture.stageUiPacketInto(self, slot);
    }

    /// P6 UI handoff: captures CPU geometry + draw parameters out of the
    /// live canvas into the render-owned frame and uploads at this
    /// prepare/context boundary (phase mutex held, context thread).
    pub fn captureUiFrame(self: *Scene, snap: *const SceneFrameSnapshot, back: *FrameDrawSlot) void {
        scene_ui_capture.captureUiFrame(self, snap, back);
    }

    /// P7 published consumable draw payload: the prepared mesh draw lists
    /// (PRIMARY + ALL PIP view queues with their skin/shader side stores,
    /// outline items+skins, prepared shadow items+skins+bin ranges), with the
    /// frame_id/retire_epoch that built them, PLUS the staged frame snapshot
    /// (wave 27: render/reuse read the slot's copy, never live state). The
    /// ONLY low-level draw accessor — render and P5/P7 tests read through
    /// here, never raw fields.
    /// Scope is mesh draws only: UI (P6 ui_frame), physics-debug lines
    /// (prepared_lines + committed DebugPass upload), sky params + default
    /// copies (staged slot snapshot), and particles (prepared frame) are
    /// separate payloads — none of them reads live subsystems at draw time. Trail
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
    /// without a prepare) — never across a prepare boundary — UNLESS the
    /// caller holds a consumer pin on the slot (`FrameDraws.pin`), which
    /// render itself does for the whole draw (see render). Render
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
    /// (which the game-side commit applies verbatim to `instance_render`,
    /// so the values are identical to reading live state — without the
    /// race).
    ///
    /// Records are appended in mesh-list order (strictly increasing
    /// `mesh_index`), so the lookup below is a linear scan with early exit.
    /// A mesh-list mutation between build and latch can therefore NOT slip
    /// a stale entry through the COMMIT (the game-side guard skips displaced
    /// meshes there); the patch itself resolves purely from the records —
    /// a record the latch published finalizes, a record the latch
    /// fail-closed (`staged_frame != frame_id`) zeroes. The live mesh list
    /// is validated at commit time, not patch time — the patch performs
    /// zero live reads (outline stale fallbacks use the record-frozen
    /// `mesh_position`, not live `mesh.position`).
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
    pub fn patchInstanceRefs(self: *Scene, back: *FrameDrawSlot) void {
        _ = self;
        scene_patch_instances.patchInstanceRefs(back);
    }

    /// Concurrent-build claim (wave 29, game side): reserve a free draw slot
    /// for the next build — a slot that is neither pinned nor front under
    /// the lease protocol. Returns null when every non-front slot is pinned
    /// or claimed (consumer lagging): skip the frame instead of blocking
    /// (the documented latest-wins drop, counted in
    /// `draws.saturation_skips`), never stall.
    ///
    /// The claim reserves (but does not commit) the next build generation
    /// (`seq = build_seq + 1`): `build()` fills the slot against it,
    /// `publish()` commits it (`build_seq`, `build_slot`) and releases the
    /// slot into the prepare handoff WITHOUT flipping `front` (the flip
    /// stays context-owned in `prepareFrame`, so the sequential
    /// build→prepare flow below is bit-identical), `cancel()` drops the
    /// claim without committing anything (the slot's provisional contents
    /// are ignored by prepare and reset by the next claim). Do not
    /// interleave a legacy `buildPreparedFrame` between claim and publish
    /// (debug-asserted: the reserved seq must still be exactly next).
    ///
    /// Threading: the claim holder owns the claimed slot's payload
    /// exclusively (producer writes, no lock needed); the lease mutex pairs
    /// those writes with the consumer's post-pin reads. Everything ELSE the
    /// build touches (live meshes/canvas, `build_snapshot`, particle/
    /// physics build frames and their layer seqs, per-mesh previews) is still
    /// phase-excluded today — the handoff seq words themselves are atomic
    /// since wave 30 (release/acquire, see the field docs), but the payload
    /// they order is not — see the adoption checklist in
    /// scene/frame_draws.zig. The claim API alone does not remove the phase
    /// mutex.
    pub fn tryClaimBuildSlot(self: *Scene) ?BuildClaim {
        const slot = self.draws.claimBack() orelse return null;
        // Single-producer reserve: monotonic load suffices, the publish
        // below is the release edge that commits the generation.
        return .{ .scene = self, .slot = slot, .seq = self.build_seq.load(.monotonic) +% 1 };
    }

    /// Game-side build claim: a reserved draw slot plus its reserved build
    /// generation. Fill via `build()` (the real build core, shared with the
    /// sequential path), optionally `stageUi()` (into the claimed slot —
    /// AFTER `build`, which resets the slot), then exactly one terminal
    /// call: `publish()` (hand to prepare) or `cancel()` (drop).
    pub const BuildClaim = struct {
        scene: *Scene,
        slot: usize,
        seq: u64,
        completed: bool = false,

        /// Run the shared build core into the claimed slot (commit of the
        /// last published latch outcomes, CPU staging, record freeze, queue/
        /// shadow/outline build). sg-free, game side. Repeatable (newest
        /// wins); the generation is only committed by `publish()`.
        pub fn build(self: *BuildClaim) void {
            self.scene.buildIntoClaimedSlot(self.slot, self.seq);
        }

        /// Stage the live canvas CPU packet into the CLAIMED slot (concurrent
        /// path). Same newest-wins/OOM rules as `stageUiPacket`, but never
        /// the latched slot: with overlapping build(N+1, game) and
        /// prepare(N, context) the claim owns a different slot than the one
        /// prepare latches. Call AFTER `build()` when both are used (the
        /// build resets the slot); UI-only claims (no `build()`) are legal.
        pub fn stageUi(self: *BuildClaim) void {
            self.scene.stageUiPacketInto(self.slot);
        }

        /// Hand the built slot to prepare: commit the reserved generation
        /// (`build_seq`, `build_slot`) and release WRITING (payload kept —
        /// prepare consumes it and flips `front` itself at latch time).
        /// No front flip here: after publish the slot is still the back
        /// index, exactly as after the legacy build.
        pub fn publish(self: *BuildClaim) void {
            std.debug.assert(!self.completed);
            self.completed = true;
            const s = self.scene;
            std.debug.assert(self.seq == s.build_seq.load(.monotonic) +% 1);
            // Release edge: the slot index first, then the generation — the
            // latch acquire-reads `build_seq` before `build_slot`, so both
            // land ordered after the whole staged payload.
            s.build_slot.store(self.slot, .release);
            s.build_seq.store(self.seq, .release);
            s.draws.releaseHandoff(self.slot) catch {};
        }

        /// Drop the claim without committing (seq/handoff untouched: prepare
        /// sees no fresh build and runs its fallback; provisional previews
        /// stamped with the uncommitted seq are overwritten by the next
        /// build and never latched). The slot is reset by the next claim.
        /// Safe to call on an unconsumed claim; double-terminal is a bug
        /// (debug-asserted).
        pub fn cancel(self: *BuildClaim) void {
            std.debug.assert(!self.completed);
            self.completed = true;
            self.scene.draws.cancelClaim(self.slot) catch {};
        }
    };

    /// Stage-2 increment B producer build (game/update phase, CPU-only,
    /// sg-free): sequential convenience over the claim flow above
    /// (`tryClaimBuildSlot` + `build` + `publish` under one call — the SAME
    /// code path, so behavior cannot diverge; sequential callers observe no
    /// change). Commits the last published latch outcomes to the live
    /// meshes, then stages the CPU halves the prepare latch will consume —
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
    /// prepare (phase ownership; the handoff seq words are atomic since wave
    /// 30, everything else plain). Callable from
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
    /// `build_snapshot` (update-vs-prepare excluded) BEFORE any CPU staging,
    /// then freezes that generation into the claim slot's staged `snapshot`
    /// (wave 27, by value); when nothing new was published ALWAYS packs
    /// fresh live state — never reuses the consumed render snapshot, so the
    /// build works without a publish and after camera removal. Staging eye,
    /// queue culling, shadow switch, and sky/ibl all freeze on this
    /// generation (fixed-size snapshot values only; live meshes/materials/
    /// culling flags stay live by design). Never reads or writes
    /// `frame_snapshot` (the latch consumes the staged slot copy, staged
    /// wins over a post-build `build_snapshot` mutation).
    /// Cache key is the build-unique `(build_seq | (1<<63))` (high bit set:
    /// cannot collide with any context `frame_id`). Stats accumulate into
    /// the game-owned `build_stats` accumulator (cleared at build start)
    /// and freeze into the claim slot's staged `build_stats` copy at build
    /// end; the latch merges the slot copy.
    ///
    /// Touches NOTHING else: no sg.*, no GpuRetire begin/complete/flush (view
    /// builds run with `instances_prepared=true`, never retrying staging),
    /// no frame_id/retire_epoch (stamped by the latch), no UI canvas/frame,
    /// no `self.stats`, no profiler. Mutates under game-phase ownership only:
    /// live mesh `instance_render` (commit of the last published latch
    /// outcomes, guarded — see above), back-slot queues/shadow/outline +
    /// scratch, previews/build_views,
    /// staged records, the staged slot `snapshot`, the staged slot
    /// `build_stats` copy,
    /// particle/physics build frames, `build_snapshot` (refreshed), shadow
    /// bin scratch, occlusion-culler frame state, world-matrix cache (tagged
    /// with the build key), and the live `build_stats` accumulator.
    ///
    /// App contract: no latch is possible while a build runs (update-vs-
    /// prepare exclusion), and the mesh list MUST NOT be mutated between a
    /// build and its latch (a violation no longer fail-closes at latch time —
    /// the latch stages whatever the slot owns — but the commit guard skips
    /// the displaced meshes and the patch already resolved the payload from
    /// the records; the next funded build recomputes). A mesh
    /// whose upload finishes between build and latch, or whose segment OOMs,
    /// keeps its previous complete `instance_render` for one frame
    /// (documented, coherent); the next funded build+latch picks it up.
    pub fn buildPreparedFrame(self: *Scene) void {
        // Sequential path holds no pins or claims across the build, so with
        // 3 slots one is always free (same guarantee as the old backIndex
        // assert — a null here is unreachable, never a skip).
        var claim = self.tryClaimBuildSlot() orelse unreachable;
        claim.build();
        claim.publish();
    }

    /// Shared build core behind `buildPreparedFrame` and `BuildClaim.build`:
    /// the exact historical build body targeted at the claimed `slot` under
    /// the reserved generation `seq` (preview stamps, record freeze, build
    /// cache key). Commits NOTHING global: `build_seq`/`build_slot` are
    /// stamped by `BuildClaim.publish`, so a cancelled claim leaves no
    /// handoff behind.
    ///
    /// EPOCH DISCIPLINE (wave 29): this core must never call
    /// `GpuRetire.begin`/`complete`/`flush` and never passes a `retire_queue`
    /// anywhere — epochs stay context-owned (prepare/render own the
    /// begin/complete pairing); enforced by test.
    fn buildIntoClaimedSlot(self: *Scene, slot: usize, seq: u64) void {
        scene_frame_build.buildIntoClaimedSlot(self, slot, seq);
    }

    pub fn prepareFrame(self: *Scene) void {
        scene_frame_prepare.prepareFrame(self);
    }

    /// Stage 3, slice 2: update-phase light packing. Selects the top-k
    /// point/spot lights for the camera (advancing the incumbency-
    /// hysteresis fades with `dt`) and stores the uniform-ready pack.
    /// Called once per frame BEFORE render(); render consumes
    /// `snapshot.light_pack` (via the staged slot snapshot), never
    /// `self.light_pack` and never live light state — so the direct
    /// `self.light_pack` fallback below stays game/prepare-side ownership
    /// (update vs prepare excluded under phase_mutex) and needs no atomic
    /// mailbox. No new mailbox: the render reads the snapshot copy taken by
    /// packFrameSnapshot.
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
    /// (camera -> lights -> physics -> animations -> soft bodies ->
    /// particles -> decals);
    /// render() then consumes the published frame values (light_pack,
    /// staged slot snapshot, prepared draws/UI/debug/sky/particles) without
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
        self.updateSoftBodies(dt);
        try self.updateParticles(dt);
        self.updateDecals(dt);

        const cur_w = sapp.width();
        const cur_h = sapp.height();
        const aspect = if (cur_h > 0) @as(f32, @floatFromInt(cur_w)) / @as(f32, @floatFromInt(cur_h)) else 1.0;
        self.publishFrameSnapshot(aspect, cur_w, cur_h);
    }

    /// Stage 3: uploads the GPU buffers that the update phase staged
    /// (particle instances, soft-body cloth vertices, trail geometry).
    /// Called at render start so every sg.* touch stays on the context
    /// thread; the update phase is free of sg.* calls.
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
        for (self.softbodies.bodies.items) |b| b.flushGpuUploads();
        for (self.trails.meshes.items) |tm| tm.flushGpuUploads();
        for (self.greased_lines.items) |gl| gl.flushGpuUploads();
        for (self.meshes.items) |m| m.flushGpuUploads();
        // Standalone completion (quiesced-context drains — NOT the
        // prepareFrame path, whose frame a render commits): the
        // compute-particle dispatch above can open the frame's command
        // buffer, and a buffer that is never committed keeps its in-flight
        // semaphore forever — sg_shutdown waits NUM_INFLIGHT_FRAMES signals
        // unconditionally and hangs at exit (observed: waits=commits+1 in
        // an instrumented soak). Nil-buffer commit is a no-op; headless
        // (no sg.setup) skips.
        if (!self.flush_in_prepare and sg.isvalid()) sg.commit();
    }

    /// Render entry: draws the frame prepared by prepareFrame (context
    /// thread, SEQUENTIAL with prepare — never concurrent; update MAY run
    /// concurrently on the game side). Reads ONLY render-owned captures
    /// (prepared draws + the staged slot snapshot incl. sky/default copies,
    /// ui_frame, debug capture + committed upload, particle prepared frame)
    /// plus BORROWED GPU handles under P3 epochs. No global phase lock is
    /// taken here: the app unlocks update-vs-prepare ownership BEFORE calling
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
        scene_frame_render.render(self);
    }

    /// True once a prepare published a frame (front slot `frame_id != 0`).
    /// The app blocks once for the first prepare while this is false instead
    /// of reusing: before the first successful prepare nothing is consumable,
    /// and reuse must not prepare without phase ownership (the game side may
    /// be mid-mutation).
    pub fn hasConsumableFrame(self: *const Scene) bool {
        return self.draws.slots[self.draws.front].frame_id != 0;
    }

    /// Current reuse streak: consecutive `renderReuse` presents since the
    /// last successful prepare (0 right after any prepare). Context thread
    /// only. Together with `pendingRetires()` this bounds what a skip streak
    /// can hide: new uploads stay pending, retires stay queued (capped).
    pub fn reuseStreak(self: *const Scene) u64 {
        return self.reuse_streak;
    }

    /// Prepares that consumed a staged UI packet as the geometry source
    /// (`capturePacket`), i.e. proof the staged path is live. The legacy
    /// canvas read and the staged-absence clear do not count.
    pub fn uiPacketLatchedCount(self: *const Scene) u64 {
        return self.ui_packet_latched;
    }

    /// Retire entries currently awaiting the next successful prepare's
    /// leading flush (queue + overflow spillover). Grows with
    /// streak × destroy-rate, bounded by `GpuRetireQueue.pending_cap`.
    pub fn pendingRetires(self: *Scene) usize {
        return self.gpu_retire.retainedCount();
    }

    /// Non-blocking render-consumer reuse: re-draws the current front slot
    /// without a prepare, for frames where the app skipped the phase-lock
    /// acquire (lock contended) instead of stalling the present.
    ///
    /// Takes no lease claim (nothing is built, nothing publishes): reuse
    /// only re-presents the pinned front through the inner `render` — the
    /// claim protocol is untouched, and a concurrent game claim can proceed
    /// against any other slot while the present holds its pin.
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
    /// Skip-streak retention (bounded, observable): every skipped prepare
    /// defers flush AND pending-upload completion, so `GpuRetireQueue.pending`
    /// grows with skip-streak × destroy-rate until the next successful
    /// prepare drains it. The growth is CAPPED (`pending_cap`, default 8192 —
    /// steady state holds a handful; anything past the cap is a counted +
    /// logged drop, never silent) and OBSERVABLE (`reuseStreak()` counts the
    /// streak, `pendingRetires()`/`cappedDropCount()` the pileup and drops).
    /// Previously this was bounded only by the phase-mutex coupling (every
    /// frame ran prepare+flush); with reuse the bound is the cap plus the
    /// app's contract — do not streak reuse indefinitely — documented here,
    /// not wished away. Borrowed handles stay valid throughout the streak
    /// (no flush ran), and new meshes stay `gpu_pending` (invisible: queue
    /// builds skip them until a successful prepare completes their uploads)
    /// — staleness visible as missing objects, never corruption.
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
        scene_frame_render.renderReuse(self);
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
        self.probes.deinit();
        self.clustered.deinit(self.allocator);
        self.gui3d.deinit(self.allocator);

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
