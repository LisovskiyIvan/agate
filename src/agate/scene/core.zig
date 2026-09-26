//! Owner of the `Scene` orchestrator type (fields, `CameraEntry`,
//! `BuildClaim`, trivial lifecycle, thin forwarders). Split out of
//! `scene.zig`, which is now a pure re-export facade.
//!
//! Method bodies live in focused leaves (`scene/<leaf>.zig`, each taking
//! the scene as `anytype`); this file keeps the public `Scene` API
//! identical so every call site keeps working unchanged.
//!
//! Anti-cycle rule: leaves never import `core.zig` or the facade.
const std = @import("std");
const sokol = @import("sokol");
const sapp = sokol.app;
const postprocess = @import("../postprocess.zig");
const PostProcessOptions = postprocess.PostProcessOptions;
const ssao = @import("../ssao.zig");
const SSAOOptions = ssao.SSAOOptions;
const particles = @import("../particles.zig");
const ParticleSystem = particles.ParticleSystem;

const math = @import("math");
const Vec3 = math.Vec3;
const Mat4 = math.Mat4;
const Color3 = math.Color3;
const Color4 = math.Color4;
const BoundingBox = math.BoundingBox;
const Ray = math.Ray;

const physics = @import("../physics.zig");
const PhysicsWorld = physics.PhysicsWorld;
const RigidBody = physics.RigidBody;
const PickingInfo = physics.PickingInfo;
const ColliderType = physics.ColliderType;

const ui = @import("../ui.zig");
const UICanvas = ui.UICanvas;
const audio = @import("../audio.zig");

const AnimationGroup = @import("../animation/animation.zig").AnimationGroup;
const Skeleton = @import("../animation/skeleton.zig").Skeleton;

const camera_mod = @import("../camera.zig");
const Camera = camera_mod.Camera;
const outline_pass = @import("../passes/outline_pass.zig");
const lights = @import("../lights.zig");
const HemisphericLight = lights.HemisphericLight;
const DirectionalLight = lights.DirectionalLight;
const DirectionalLightOptions = lights.DirectionalLightOptions;
const PointLight = lights.PointLight;
const PointLightOptions = lights.PointLightOptions;
const SpotLight = lights.SpotLight;
const SpotLightOptions = lights.SpotLightOptions;
const AreaLight = lights.AreaLight;
const AreaLightOptions = lights.AreaLightOptions;
const Mesh = @import("../mesh.zig").Mesh;
const decal_mod = @import("../mesh/decal.zig");
const DecalManager = decal_mod.DecalManager;
const trail_mod = @import("../mesh/trail.zig");
const TrailMesh = trail_mod.TrailMesh;
const TrailOptions = trail_mod.TrailOptions;
const csg_mod = @import("../mesh/csg.zig");
const greased_mod = @import("../mesh/greased_line.zig");
const GreasedLineOptions = greased_mod.GreasedLineOptions;
const GreasedLineMesh = greased_mod.GreasedLineMesh;
const simplify_mod = @import("../mesh/simplify.zig");
const SimplifyOptions = simplify_mod.SimplifyOptions;
const LODLevelSpec = simplify_mod.LODLevelSpec;
const ai_mod = @import("../ai.zig");
const NavMesh = ai_mod.NavMesh;
const NavAgent = ai_mod.NavAgent;
const StandardMaterial = @import("../material.zig").StandardMaterial;
const PBRMaterial = @import("../material.zig").PBRMaterial;
const ShaderMaterial = @import("../material.zig").ShaderMaterial;
const Texture = @import("../texture.zig").Texture;
const CubeTexture = @import("../texture.zig").CubeTexture;
const SkyboxOptions = @import("../texture.zig").SkyboxOptions;
const visibility = @import("../visibility/mod.zig");
const serialization = @import("../serialization.zig");

// Scene subsystems. Each owns its state (and GPU resources) plus the logic
// that belongs to it; Scene is the owner/orchestrator facade. Subsystems
// never import scene.zig — Scene passes everything they need as parameters.
const scene_stats = @import("stats.zig");
const SceneStats = scene_stats.SceneStats;
const scene_render_queue = @import("render_queue.zig");
const jobs = @import("../jobs.zig");
const assets_mod = @import("../assets.zig");
const handoff_mod = @import("../handoff.zig");
const scene_lights = @import("light_rig.zig");
const scene_clustered = @import("clustered_lights.zig");
const ClusteredPointLight = lights.ClusteredPointLight;
const ClusteredPointLightOptions = lights.ClusteredPointLightOptions;
const scene_shadow = @import("shadow_system.zig");
const scene_sky = @import("sky_layer.zig");
const scene_probes = @import("probe_layer.zig");
const ReflectionProbe = scene_probes.ReflectionProbe;
const ReflectionProbeOptions = scene_probes.ReflectionProbeOptions;
const scene_gui3d = @import("gui3d_layer.zig");
const Ui3dPanel = scene_gui3d.Ui3dPanel;
const Ui3dPanelOptions = scene_gui3d.Ui3dPanelOptions;
const Ui3dPickHit = scene_gui3d.Ui3dPickHit;
const scene_highlight = @import("highlight_layer.zig");
const HighlightLayer = scene_highlight.HighlightLayer;
const HighlightEntry = scene_highlight.HighlightEntry;
const HighlightOptions = scene_highlight.HighlightOptions;
const scene_postfx = @import("postfx_stack.zig");
const scene_forward = @import("forward_pipelines.zig");
const scene_particles = @import("particle_layer.zig");
const scene_decals = @import("decal_layer.zig");
const scene_trails = @import("trail_layer.zig");
const scene_nav = @import("nav_layer.zig");
const scene_physics = @import("physics_layer.zig");
const softbody_mod = @import("../softbody.zig");
const SoftBody = softbody_mod.SoftBody;
const SoftBodyLayer = softbody_mod.SoftBodyLayer;
const ClothOptions = softbody_mod.ClothOptions;
const SphereCollider = softbody_mod.SphereCollider;
const scene_project = @import("project_cache.zig");
const scene_retire = @import("gpu_retire.zig");
const scene_ui_frame = @import("ui_frame.zig");
const UiFrame = scene_ui_frame.UiFrame;
const scene_frame_draws = @import("frame_draws.zig");
/// P7 published consumable draw payload (one coherent prepared frame).
const FrameDrawSlot = scene_frame_draws.FrameDrawSlot;
/// P7 three retained owning queue slots (the variable-list triple buffer
/// with a consumer pin/lease protocol).
const FrameDraws = scene_frame_draws.FrameDraws;
const scene_draw = @import("draw.zig");
const scene_msaa = @import("msaa.zig");
const scene_viewport_clear = @import("viewport_clear.zig");
const scene_queue_builder = @import("queue_builder.zig");
const QueueBuildParams = scene_queue_builder.QueueBuildParams;
const scene_snapshot = @import("snapshot.zig");
const SceneFrameSnapshot = scene_snapshot.SceneFrameSnapshot;
const CameraSnapshot = scene_snapshot.CameraSnapshot;
const profiler_mod = @import("../profiler.zig");
const Profiler = profiler_mod.Profiler;
const FrameRecord = profiler_mod.FrameRecord;
const MemorySnapshot = profiler_mod.MemorySnapshot;

const scene_lifecycle = @import("lifecycle.zig");
const scene_registry = @import("registry.zig");
const scene_cameras = @import("cameras.zig");
const scene_lights_api = @import("lights_api.zig");
const scene_attachments = @import("attachments.zig");
const scene_sim = @import("sim_api.zig");
const scene_query = @import("query_api.zig");
const scene_frame = @import("frame_api.zig");
const scene_profile = @import("profile_api.zig");
const scene_upload_packets = @import("upload_packets.zig");
const mesh_mod = @import("../mesh.zig");

pub const CameraEntry = scene_cameras.CameraEntry;

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
///
/// Resource-lifetime contract (see `scene/registry.zig` + `agate/API.md`):
/// meshes/materials/trails are Scene-owned and die through the matching
/// `destroy*` (mesh GPU buffers retire by epoch off-context and flush at
/// the next render start; `deinit` drains everything). Mesh `name` slices
/// are borrowed (`owns_name == false`) or Scene-allocator-owned
/// (`owns_name == true`, freed in `Mesh.deinit`) — plain fields, no
/// language-level privacy; rename only via `renameMesh`. Particle systems
/// are Scene-owned until `deinit` (context-thread only): no per-system
/// destroy exists, because `ParticleSystem.deinit` issues `sg.destroy*`
/// inline and prepared frames borrow its handle ids.
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
    /// Latest update timing crosses the producer/prepare boundary atomically.
    /// Prepare may finish a slot-owned frame while the producer starts the
    /// next tick, so a plain float here would race even though stats itself
    /// remains context-owned.
    pending_update_ms: std.atomic.Value(f32) = std.atomic.Value(f32).init(0),
    /// Latest physics timing, published with the same single-writer rule.
    pending_physics_ms: std.atomic.Value(f32) = std.atomic.Value(f32).init(0),
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

    // MSAA depth-prepass gate (default OFF): when true AND the effective
    // main-target sample count is > 1, opaque primary-view geometry is
    // drawn a second time with depth-only pipelines into a 1x depth
    // texture (PASS 1.7, before the main pass), so the depth-consuming
    // post effects (SSAO/SSR/DoF/Fog/MotionBlur) keep running under MSAA
    // instead of being suppressed. Off path is bit-identical (no extra
    // pass, no extra VRAM, suppression as before). TAA stays suppressed
    // under MSAA regardless (v1 non-goal). Known approximations, see
    // passes/msaa_depth_pass.zig: 1x depth vs MSAA color can differ by a
    // pixel at geometric edges; cutout renders opaque and GPU morphs
    // contribute base positions (shadow-map precedent); hook-material,
    // transparent, decal, particle and secondary-view geometry is skipped
    // (primary view only, v1 non-goal). Set before the first render();
    // toggling later allocates/frees the 1x target on the next frame.
    msaa_depth_prepass: bool = false,

    // Forward pipeline twin built for the effective MSAA sample count
    // (sokol requires pipeline.sample_count to match every main-target
    // attachment). Null until the first MSAA frame.
    forward_msaa: ?scene_forward.ForwardPipelines = null,
    warn_msaa_format: scene_msaa.WarnOnce = .{},

    // Mesh-vanish probe state (pure diagnostic for the symptom "all meshes
    // gone from the main view while skybox + particles keep rendering").
    // Render-owned, context thread only (written by probeMeshVanish in
    // scene/frame_render.zig): rate-limit cursor + latched active flag for
    // the recovery line. Defaults keep every existing Scene construction
    // valid; the probe never panics, never fails, never allocates.
    mesh_vanish_last_log_frame: u64 = 0,
    mesh_vanish_active: bool = false,
    mesh_vanish_armed: bool = false,

    // Highlighted meshes for the inverse-hull outline (postfx holds the
    // settings + pass). Kept flat: mock scenes in mesh tests construct it.
    // The prepared outline items+skins live in the P7 slots (preparedDraws).
    outline_meshes: std.ArrayListUnmanaged(*Mesh) = .empty,
    // Per-mesh highlight entries (highlight layer v1, mask-RT inner glow).
    // Fixed-size layer (no allocation, no deinit — the entries borrow the
    // mesh pointers; the staged items+handles live in the P7 slots). Empty
    // by default: with zero highlights the whole mask/blur/composite chain
    // is gated off and rendering stays bit-identical.
    highlights: scene_highlight.HighlightLayer = .{},

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
    /// Context-only linear-use guard for the split prepare token. A begin
    /// claims a slot and must be paired with exactly one finish or cancel.
    prepare_claim_generation: u64 = 0,
    prepare_claim_active: bool = false,
    prepare_claim_slot: usize = 0,
    prepare_claim_seq: u64 = 0,
    prepare_claim_have_build: bool = false,
    prepare_claim_has_handoff: bool = false,
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
    /// slot-owned staged records, captures
    /// the particle/physics CPU build frames and freezes them into the
    /// claimed slot (`particle_draws`, `physics_lines`/`physics_visible` —
    /// wave 32 freeze-then-latch, so the latch never reads the shared
    /// staging stores), then bumps `build_seq`.
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
    /// staged (slot queues/records/snapshot/stats, frozen particle/physics
    /// slot captures, `build_slot`), and acquire-loaded by the context latch (`prepareFrame`
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
    /// Cooperative game-side yield for the concurrent-build path (wave 38,
    /// contention-yield): nanoseconds the game thread parks inside
    /// `tryClaimBuildSlot` — but ONLY when the previous build is still
    /// unconsumed (`build_seq != last_latched_seq`) — when nonzero.
    /// Default 0 = OFF: the default (phase-locked) path never sets it, so
    /// it stays behavior-identical (one predictable not-taken branch, no
    /// syscall, no atomics beyond the existing handoff words). Under
    /// EXPERIMENTAL `--concurrent-build` the app may set it (startup only,
    /// before the game thread spawns; game-side read): the game is then
    /// overproducing (hundreds of builds/s vs 60 latched frames/s), so
    /// parking before reserving the next slot donates its core to the
    /// context thread's prepare/render + pool workers instead of burning
    /// CPU on builds the latch will supersede — the phase mutex used to
    /// provide this yield implicitly by parking the game on contention
    /// (§6.38 diagnosis). Backpressure, not a flat tax: when the consumer
    /// is caught up the claim proceeds immediately (freshness preserved,
    /// zero added latency); the park also runs BEFORE the lease reserve,
    /// so no WRITING slot is ever held while parked. Overhead when
    /// enabled and behind: exactly one nanosleep per skipped-ahead tick;
    /// when 0 or caught up: a single branch. Plain u64, stays plain:
    /// written once by the app at startup, read only by the game-side
    /// claim — never across the handoff edge. Tune only with wall numbers
    /// (dips / 1% low at fixed prepared=100% + waitC=0), never blindly:
    /// too small changes nothing, too large slows the sim tick rate
    /// (ticks/s floor: the latch needs one fresh build per 16.7 ms frame).
    concurrent_yield_ns: u64 = 0,
    /// Lock-free staged prepare (phase 2, default ON): `Runtime.beginPrepare*`
    /// sets this from its `producer_exclusion` knob on every begin
    /// (`setProducerExclusion(true)` clears it again). It gates the last
    /// live-list fallbacks on the fresh-build path (`captureUiFrame` degrades
    /// an invalid staged UI packet to coherent-empty instead of re-reading
    /// the live canvas lists); the upload-packet flush/commit needs no flag
    /// (it never touches live state either way). Plain bool, written on the
    /// context thread, read only there.
    lock_free_prepare: bool = false,
    /// Last published frame whose staged-upload outcomes were committed
    /// (`commitSlotResults` at the next build). Game side only (written
    /// by the build core, never across the handoff edge): a repeat build
    /// without an intervening latch observes the same `front.frame_id`
    /// and skips the commit, so outcomes are never double-applied.
    last_upload_commit_frame: u64 = 0,
    /// Dropped `stageHostBytes` payloads (OOM fail-closed clears). Game
    /// side only (written by the producer claim, read by the host/tests —
    /// never across the handoff edge). Expected to stay zero: the encode
    /// is fixed-size into retained slot capacity.
    host_bytes_oob_drops: u64 = 0,
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

    /// See `scene/lifecycle.zig` (owns the body + docs).
    pub fn resizeOffscreen(self: *Scene, width: i32, height: i32) void {
        scene_lifecycle.resizeOffscreen(self, width, height);
    }

    /// See `scene/lifecycle.zig` (owns the body + docs).
    pub fn setPostProcess(self: *Scene, config: PostProcessOptions) void {
        scene_lifecycle.setPostProcess(self, config);
    }

    /// See `scene/lifecycle.zig` (owns the body + docs).
    pub fn setSSAO(self: *Scene, config: SSAOOptions) void {
        scene_lifecycle.setSSAO(self, config);
    }

    /// See `scene/lifecycle.zig` (owns the body + docs).
    pub fn saveStateFileAsync(self: *Scene, path: []const u8) !*serialization.AsyncSaveTask {
        return scene_lifecycle.saveStateFileAsync(self, path);
    }

    /// See `scene/lifecycle.zig` (owns the body + docs).
    pub fn loadStateFileAsync(self: *Scene, path: []const u8) !*serialization.AsyncLoadTask {
        return scene_lifecycle.loadStateFileAsync(self, path);
    }

    /// See `scene/lifecycle.zig` (owns the body + docs).
    pub fn setSkybox(self: *Scene, cube: CubeTexture) void {
        scene_lifecycle.setSkybox(self, cube);
    }

    /// See `scene/lifecycle.zig` (owns the body + docs).
    pub fn createDefaultSkybox(self: *Scene, config: SkyboxOptions) !void {
        return scene_lifecycle.createDefaultSkybox(self, config);
    }

    /// See `scene/attachments.zig` (owns the body + docs).
    pub fn addReflectionProbe(self: *Scene, position: Vec3, options: ReflectionProbeOptions) error{TooManyReflectionProbes}!usize {
        return scene_attachments.addReflectionProbe(self, position, options);
    }

    /// See `scene/attachments.zig` (owns the body + docs).
    pub fn removeReflectionProbe(self: *Scene, index: usize) void {
        scene_attachments.removeReflectionProbe(self, index);
    }

    /// See `scene/attachments.zig` (owns the body + docs).
    pub fn getReflectionProbe(self: *Scene, index: usize) ?*ReflectionProbe {
        return scene_attachments.getReflectionProbe(self, index);
    }

    /// See `scene/attachments.zig` (owns the body + docs).
    pub fn reflectionProbeCount(self: *const Scene) usize {
        return scene_attachments.reflectionProbeCount(self);
    }

    /// See `scene/attachments.zig` (owns the body + docs).
    pub fn captureReflectionProbe(self: *Scene, index: usize) void {
        scene_attachments.captureReflectionProbe(self, index);
    }

    /// See `scene/attachments.zig` (owns the body + docs).
    pub fn captureDirtyReflectionProbes(self: *Scene) void {
        scene_attachments.captureDirtyReflectionProbes(self);
    }

    /// See `scene/attachments.zig` (owns the body + docs).
    pub fn probeDirtyCount(self: *const Scene) usize {
        return scene_attachments.probeDirtyCount(self);
    }

    /// See `scene/attachments.zig` (owns the body + docs).
    pub fn addUi3dPanel(
        self: *Scene,
        name: []const u8,
        position: Vec3,
        options: Ui3dPanelOptions,
    ) error{ TooManyUi3dPanels, InvalidUi3dPanelSize, OutOfMemory }!usize {
        return scene_attachments.addUi3dPanel(self, name, position, options);
    }

    /// See `scene/attachments.zig` (owns the body + docs).
    pub fn removeUi3dPanel(self: *Scene, index: usize) void {
        scene_attachments.removeUi3dPanel(self, index);
    }

    /// See `scene/attachments.zig` (owns the body + docs).
    pub fn getUi3dPanel(self: *Scene, index: usize) ?*Ui3dPanel {
        return scene_attachments.getUi3dPanel(self, index);
    }

    /// See `scene/attachments.zig` (owns the body + docs).
    pub fn getUi3dPanelByName(self: *Scene, name: []const u8) ?*Ui3dPanel {
        return scene_attachments.getUi3dPanelByName(self, name);
    }

    /// See `scene/attachments.zig` (owns the body + docs).
    pub fn ui3dPanelCount(self: *const Scene) usize {
        return scene_attachments.ui3dPanelCount(self);
    }

    /// See `scene/attachments.zig` (owns the body + docs).
    pub fn markUi3dPanelDirty(self: *Scene, index: usize) void {
        scene_attachments.markUi3dPanelDirty(self, index);
    }

    /// See `scene/attachments.zig` (owns the body + docs).
    pub fn markAllUi3dPanelsDirty(self: *Scene) void {
        scene_attachments.markAllUi3dPanelsDirty(self);
    }

    /// See `scene/attachments.zig` (owns the body + docs).
    pub fn ui3dDirtyCount(self: *const Scene) usize {
        return scene_attachments.ui3dDirtyCount(self);
    }

    /// See `scene/attachments.zig` (owns the body + docs).
    pub fn pickUi3dPanel(self: *Scene, mouse_x: f32, mouse_y: f32) ?Ui3dPickHit {
        return scene_attachments.pickUi3dPanel(self, mouse_x, mouse_y);
    }

    /// See `scene/attachments.zig` (owns the body + docs).
    pub fn addHighlightMesh(self: *Scene, mesh: *Mesh, options: HighlightOptions) error{ TooManyHighlights, InvalidHighlightOptions }!usize {
        return scene_attachments.addHighlightMesh(self, mesh, options);
    }

    /// See `scene/attachments.zig` (owns the body + docs).
    pub fn removeHighlightMesh(self: *Scene, index: usize) void {
        scene_attachments.removeHighlightMesh(self, index);
    }

    /// See `scene/attachments.zig` (owns the body + docs).
    pub fn clearHighlights(self: *Scene) void {
        scene_attachments.clearHighlights(self);
    }

    /// See `scene/attachments.zig` (owns the body + docs).
    pub fn getHighlightMesh(self: *Scene, index: usize) ?*HighlightEntry {
        return scene_attachments.getHighlightMesh(self, index);
    }

    /// See `scene/attachments.zig` (owns the body + docs).
    pub fn highlightCount(self: *const Scene) usize {
        return scene_attachments.highlightCount(self);
    }

    /// See `scene/lights_api.zig` (owns the body + docs).
    pub fn createHemisphericLight(self: *Scene, name: []const u8, options: lights.HemisphericLightOptions) HemisphericLight {
        return scene_lights_api.createHemisphericLight(self, name, options);
    }

    /// See `scene/lights_api.zig` (owns the body + docs).
    pub fn createPointLight(self: *Scene, name: []const u8, options: PointLightOptions) !*PointLight {
        return scene_lights_api.createPointLight(self, name, options);
    }

    /// See `scene/lights_api.zig` (owns the body + docs).
    pub fn createSpotLight(self: *Scene, name: []const u8, options: SpotLightOptions) !*SpotLight {
        return scene_lights_api.createSpotLight(self, name, options);
    }

    /// See `scene/lights_api.zig` (owns the body + docs).
    pub fn createDirectionalLight(self: *Scene, name: []const u8, options: DirectionalLightOptions) !*DirectionalLight {
        return scene_lights_api.createDirectionalLight(self, name, options);
    }

    /// See `scene/lights_api.zig` (owns the body + docs).
    pub fn addDirectionalLight(self: *Scene, name: []const u8, options: DirectionalLightOptions) !*DirectionalLight {
        return scene_lights_api.addDirectionalLight(self, name, options);
    }

    /// See `scene/lights_api.zig` (owns the body + docs).
    pub fn setSunAngles(self: *Scene, azimuth_rad: f32, elevation_rad: f32) void {
        scene_lights_api.setSunAngles(self, azimuth_rad, elevation_rad);
    }

    /// See `scene/lights_api.zig` (owns the body + docs).
    pub fn setSunColorTemperature(self: *Scene, kelvin: f32) void {
        scene_lights_api.setSunColorTemperature(self, kelvin);
    }

    /// See `scene/lights_api.zig` (owns the body + docs).
    pub fn addAreaLight(self: *Scene, name: []const u8, options: AreaLightOptions) !*AreaLight {
        return scene_lights_api.addAreaLight(self, name, options);
    }

    /// See `scene/lights_api.zig` (owns the body + docs).
    pub fn removeAreaLight(self: *Scene, index: usize) void {
        scene_lights_api.removeAreaLight(self, index);
    }

    /// See `scene/lights_api.zig` (owns the body + docs).
    pub fn getAreaLight(self: *Scene, index: usize) ?*AreaLight {
        return scene_lights_api.getAreaLight(self, index);
    }

    /// See `scene/lights_api.zig` (owns the body + docs).
    pub fn areaLightCount(self: *const Scene) usize {
        return scene_lights_api.areaLightCount(self);
    }

    /// See `scene/lights_api.zig` (owns the body + docs).
    pub fn addClusteredPointLight(self: *Scene, position: Vec3, options: ClusteredPointLightOptions) error{TooManyClusteredLights}!usize {
        return scene_lights_api.addClusteredPointLight(self, position, options);
    }

    /// See `scene/lights_api.zig` (owns the body + docs).
    pub fn removeClusteredPointLight(self: *Scene, index: usize) void {
        scene_lights_api.removeClusteredPointLight(self, index);
    }

    /// See `scene/lights_api.zig` (owns the body + docs).
    pub fn getClusteredPointLight(self: *Scene, index: usize) ?*ClusteredPointLight {
        return scene_lights_api.getClusteredPointLight(self, index);
    }

    /// See `scene/lights_api.zig` (owns the body + docs).
    pub fn clusteredPointLightCount(self: *const Scene) usize {
        return scene_lights_api.clusteredPointLightCount(self);
    }

    /// See `scene/attachments.zig` (owns the body + docs).
    pub fn addSoftBodyCloth(self: *Scene, name: []const u8, options: ClothOptions) softbody_mod.SoftBodyError!*SoftBody {
        return scene_attachments.addSoftBodyCloth(self, name, options);
    }

    /// See `scene/attachments.zig` (owns the body + docs).
    pub fn removeSoftBodyCloth(self: *Scene, index: usize) softbody_mod.SoftBodyError!void {
        return scene_attachments.removeSoftBodyCloth(self, index);
    }

    /// See `scene/attachments.zig` (owns the body + docs).
    pub fn getSoftBody(self: *Scene, index: usize) ?*SoftBody {
        return scene_attachments.getSoftBody(self, index);
    }

    /// See `scene/attachments.zig` (owns the body + docs).
    pub fn getSoftBodyByName(self: *Scene, name: []const u8) ?*SoftBody {
        return scene_attachments.getSoftBodyByName(self, name);
    }

    /// See `scene/attachments.zig` (owns the body + docs).
    pub fn softBodyCount(self: *const Scene) usize {
        return scene_attachments.softBodyCount(self);
    }

    /// See `scene/attachments.zig` (owns the body + docs).
    pub fn updateSoftBodies(self: *Scene, dt: f32) void {
        scene_attachments.updateSoftBodies(self, dt);
    }

    /// See `scene/registry.zig` (owns the body + docs).
    pub fn createStandardMaterial(self: *Scene, name: []const u8) !*StandardMaterial {
        return scene_registry.createStandardMaterial(self, name);
    }

    /// See `scene/registry.zig` (owns the body + docs).
    pub fn createPBRMaterial(self: *Scene, name: []const u8) !*PBRMaterial {
        return scene_registry.createPBRMaterial(self, name);
    }

    /// See `scene/registry.zig` (owns the body + docs).
    pub fn createShaderMaterial(self: *Scene, name: []const u8, shader_name: []const u8) ?*ShaderMaterial {
        return scene_registry.createShaderMaterial(self, name, shader_name);
    }

    /// See `scene/frame_api.zig` (owns the body + docs).
    pub fn recordUpdateTime(self: *Scene, ms: f32) void {
        scene_frame.recordUpdateTime(self, ms);
    }

    /// See `scene/frame_api.zig` (owns the body + docs).
    pub fn recordPhysicsTime(self: *Scene, ms: f32) void {
        scene_frame.recordPhysicsTime(self, ms);
    }

    /// See `scene/registry.zig` (owns the body + docs).
    pub fn destroyPBRMaterial(self: *Scene, mat: *PBRMaterial) void {
        scene_registry.destroyPBRMaterial(self, mat);
    }

    /// See `scene/registry.zig` (owns the body + docs).
    pub fn removeMesh(self: *Scene, mesh: *Mesh) bool {
        return scene_registry.removeMesh(self, mesh);
    }

    /// See `scene/registry.zig` (owns the body + docs).
    pub fn destroyMesh(self: *Scene, mesh: *Mesh) void {
        scene_registry.destroyMesh(self, mesh);
    }

    /// See `scene/registry.zig` (owns the body + docs).
    pub fn renameMesh(self: *Scene, mesh: *Mesh, new_name: []const u8) !void {
        return scene_registry.renameMesh(self, mesh, new_name);
    }

    /// See `scene/registry.zig` (owns the body + docs).
    pub fn destroyTrailMesh(self: *Scene, trail: *TrailMesh) void {
        scene_registry.destroyTrailMesh(self, trail);
    }

    /// See `scene/registry.zig` (owns the body + docs).
    pub fn getMeshByName(self: *Scene, name: []const u8) ?*Mesh {
        return scene_registry.getMeshByName(self, name);
    }

    /// See `scene/registry.zig` (owns the body + docs).
    pub fn getMeshesByTag(self: *Scene, allocator: std.mem.Allocator, tag_str: []const u8) !std.ArrayListUnmanaged(*Mesh) {
        return scene_registry.getMeshesByTag(self, allocator, tag_str);
    }

    /// See `scene/registry.zig` (owns the body + docs).
    pub fn getMeshesByQuery(self: *Scene, allocator: std.mem.Allocator, query_str: []const u8) !std.ArrayListUnmanaged(*Mesh) {
        return scene_registry.getMeshesByQuery(self, allocator, query_str);
    }

    /// See `scene/registry.zig` (owns the body + docs).
    pub fn countMeshesByTag(self: *Scene, tag_str: []const u8) usize {
        return scene_registry.countMeshesByTag(self, tag_str);
    }

    /// See `scene/registry.zig` (owns the body + docs).
    pub fn countMeshesByQuery(self: *Scene, query_str: []const u8) usize {
        return scene_registry.countMeshesByQuery(self, query_str);
    }

    /// See `scene/registry.zig` (owns the body + docs).
    pub fn findFirstMeshByTag(self: *Scene, tag_str: []const u8) ?*Mesh {
        return scene_registry.findFirstMeshByTag(self, tag_str);
    }

    /// See `scene/registry.zig` (owns the body + docs).
    pub fn findFirstMeshByQuery(self: *Scene, query_str: []const u8) ?*Mesh {
        return scene_registry.findFirstMeshByQuery(self, query_str);
    }

    /// See `scene/sim_api.zig` (owns the body + docs).
    pub fn getOrCreateDecalManager(self: *Scene, max_decals: usize) *DecalManager {
        return scene_sim.getOrCreateDecalManager(self, max_decals);
    }

    /// See `scene/sim_api.zig` (owns the body + docs).
    pub fn updateDecals(self: *Scene, dt: f32) void {
        scene_sim.updateDecals(self, dt);
    }

    /// See `scene/sim_api.zig` (owns the body + docs).
    pub fn createParticleSystem(self: *Scene, name: []const u8, capacity: usize) !*ParticleSystem {
        return scene_sim.createParticleSystem(self, name, capacity);
    }

    /// See `scene/sim_api.zig` (owns the body + docs).
    pub fn updateParticles(self: *Scene, dt: f32) particles.UpdateError!void {
        return scene_sim.updateParticles(self, dt);
    }

    /// See `scene/sim_api.zig` (owns the body + docs).
    pub fn createTrailMesh(self: *Scene, name: []const u8, options: TrailOptions) !*TrailMesh {
        return scene_sim.createTrailMesh(self, name, options);
    }

    /// See `scene/sim_api.zig` (owns the body + docs).
    pub fn updateTrails(self: *Scene, dt: f32) void {
        scene_sim.updateTrails(self, dt);
    }

    /// See `scene/sim_api.zig` (owns the body + docs).
    pub fn createCSGMesh(self: *Scene, name: []const u8, csg_solid: *const csg_mod.CSG) !*Mesh {
        return scene_sim.createCSGMesh(self, name, csg_solid);
    }

    /// See `scene/sim_api.zig` (owns the body + docs).
    pub fn createGreasedLine(self: *Scene, name: []const u8, options: GreasedLineOptions) !*Mesh {
        return scene_sim.createGreasedLine(self, name, options);
    }

    /// See `scene/sim_api.zig` (owns the body + docs).
    pub fn createGreasedLineMesh(self: *Scene, name: []const u8, options: GreasedLineOptions) !*GreasedLineMesh {
        return scene_sim.createGreasedLineMesh(self, name, options);
    }

    /// See `scene/sim_api.zig` (owns the body + docs).
    pub fn simplifyMesh(self: *Scene, name: []const u8, source_mesh: *Mesh, options: SimplifyOptions) !*Mesh {
        return scene_sim.simplifyMesh(self, name, source_mesh, options);
    }

    /// See `scene/sim_api.zig` (owns the body + docs).
    pub fn generateLODLevels(self: *Scene, source_mesh: *Mesh, specs: []const LODLevelSpec) !void {
        return scene_sim.generateLODLevels(self, source_mesh, specs);
    }

    /// See `scene/sim_api.zig` (owns the body + docs).
    pub fn createNavMeshFromTriangles(
        self: *Scene,
        positions: []const [3]f32,
        indices: []const u32,
        max_slope_rad: f32,
    ) !*ai_mod.NavMesh {
        return scene_sim.createNavMeshFromTriangles(self, positions, indices, max_slope_rad);
    }

    /// See `scene/sim_api.zig` (owns the body + docs).
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
        return scene_sim.createNavMeshGrid(self, min_x, max_x, min_z, max_z, elevation_y, subdiv_x, subdiv_z, obstacles);
    }

    /// See `scene/sim_api.zig` (owns the body + docs).
    pub fn createNavAgent(self: *Scene, nav_mesh: *const ai_mod.NavMesh, start_pos: Vec3) !*ai_mod.NavAgent {
        return scene_sim.createNavAgent(self, nav_mesh, start_pos);
    }

    /// See `scene/sim_api.zig` (owns the body + docs).
    pub fn updateNavAgents(self: *Scene, dt: f32) void {
        scene_sim.updateNavAgents(self, dt);
    }

    /// See `scene/sim_api.zig` (owns the body + docs).
    pub fn updateAnimations(self: *Scene, dt: f32) void {
        scene_sim.updateAnimations(self, dt);
    }

    /// See `scene/sim_api.zig` (owns the body + docs).
    pub fn enablePhysics(self: *Scene, gravity: ?Vec3) *PhysicsWorld {
        return scene_sim.enablePhysics(self, gravity);
    }

    /// See `scene/sim_api.zig` (owns the body + docs).
    pub fn getRigidBody(self: *Scene, mesh: *const Mesh) ?*RigidBody {
        return scene_sim.getRigidBody(self, mesh);
    }

    /// See `scene/sim_api.zig` (owns the body + docs).
    pub fn createRigidBody(self: *Scene, mesh: *Mesh, collider: ColliderType, mass: f32) !*RigidBody {
        return scene_sim.createRigidBody(self, mesh, collider, mass);
    }

    /// See `scene/sim_api.zig` (owns the body + docs).
    pub fn createRigidBodyWith(self: *Scene, mesh: *Mesh, collider: ColliderType, mass: f32, options: physics.BodyOptions) !*RigidBody {
        return scene_sim.createRigidBodyWith(self, mesh, collider, mass, options);
    }

    /// See `scene/sim_api.zig` (owns the body + docs).
    pub fn updatePhysics(self: *Scene, dt: f32) void {
        scene_sim.updatePhysics(self, dt);
    }

    /// See `scene/cameras.zig` (owns the body + docs).
    pub fn addCamera(self: *Scene, entry: CameraEntry) !usize {
        return scene_cameras.addCamera(self, entry);
    }

    /// See `scene/cameras.zig` (owns the body + docs).
    pub fn removeCamera(self: *Scene, index: usize) void {
        scene_cameras.removeCamera(self, index);
    }

    /// See `scene/cameras.zig` (owns the body + docs).
    pub fn getCamera(self: *Scene, index: usize) ?*CameraEntry {
        return scene_cameras.getCamera(self, index);
    }

    /// See `scene/cameras.zig` (owns the body + docs).
    pub fn getCameraByName(self: *Scene, name: []const u8) ?*CameraEntry {
        return scene_cameras.getCameraByName(self, name);
    }

    /// See `scene/cameras.zig` (owns the body + docs).
    pub fn switchCamera(self: *Scene, index: usize) void {
        scene_cameras.switchCamera(self, index);
    }

    /// See `scene/cameras.zig` (owns the body + docs).
    pub fn switchCameraByName(self: *Scene, name: []const u8) bool {
        return scene_cameras.switchCameraByName(self, name);
    }

    /// See `scene/cameras.zig` (owns the body + docs).
    pub fn nextCamera(self: *Scene) void {
        scene_cameras.nextCamera(self);
    }

    /// See `scene/cameras.zig` (owns the body + docs).
    pub fn prevCamera(self: *Scene) void {
        scene_cameras.prevCamera(self);
    }

    /// See `scene/cameras.zig` (owns the body + docs).
    pub fn getActiveCameraIndex(self: Scene) ?usize {
        return scene_cameras.getActiveCameraIndex(self);
    }

    /// See `scene/cameras.zig` (owns the body + docs).
    pub fn getActiveCameraName(self: Scene) ?[]const u8 {
        return scene_cameras.getActiveCameraName(self);
    }

    /// See `scene/cameras.zig` (owns the body + docs).
    pub fn getCameraCount(self: Scene) usize {
        return scene_cameras.getCameraCount(self);
    }

    /// See `scene/cameras.zig` (owns the body + docs).
    pub fn setActiveCamera(self: *Scene, cam: ?Camera, owned_name: ?[]const u8) void {
        scene_cameras.setActiveCamera(self, cam, owned_name);
    }

    /// See `scene/cameras.zig` (owns the body + docs).
    pub fn updateCamera(self: *Scene, dt: f32) void {
        scene_cameras.updateCamera(self, dt);
    }

    /// See `scene/query_api.zig` (owns the body + docs).
    pub fn createPickingRay(self: *Scene, screen_x: f32, screen_y: f32) Ray {
        return scene_query.createPickingRay(self, screen_x, screen_y);
    }

    /// See `scene/query_api.zig` (owns the body + docs).
    pub fn pickWithRay(self: *Scene, r: Ray) PickingInfo {
        return scene_query.pickWithRay(self, r);
    }

    /// See `scene/query_api.zig` (owns the body + docs).
    pub fn pickWithRayTag(self: *Scene, r: Ray, query_str: []const u8) PickingInfo {
        return scene_query.pickWithRayTag(self, r, query_str);
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

    /// See `scene/query_api.zig` (owns the body + docs).
    pub fn pick(self: *Scene, screen_x: f32, screen_y: f32) PickingInfo {
        return scene_query.pick(self, screen_x, screen_y);
    }

    /// See `scene/query_api.zig` (owns the body + docs).
    pub fn createUI(self: *Scene) !*UICanvas {
        return scene_query.createUI(self);
    }

    /// See `scene/query_api.zig` (owns the body + docs).
    pub fn getUI(self: *Scene) ?*UICanvas {
        return scene_query.getUI(self);
    }

    /// See `scene/query_api.zig` (owns the body + docs).
    pub fn projectPoint(self: *Scene, world_pos: Vec3) ?math.Vec2 {
        return scene_query.projectPoint(self, world_pos);
    }

    /// See `scene/query_api.zig` (owns the body + docs).
    pub fn handleEvent(self: *Scene, ev: [*c]const sapp.Event) void {
        scene_query.handleEvent(self, ev);
    }

    /// See `scene/lifecycle.zig` (owns the body + docs).
    pub fn ensureForwardMsaa(self: *Scene, samples: i32) *scene_forward.ForwardPipelines {
        return scene_lifecycle.ensureForwardMsaa(self, samples);
    }

    /// See `scene/frame_api.zig` (owns the body + docs).
    pub fn prepareViewQueues(
        self: *Scene,
        queues: *scene_render_queue.RenderQueues,
        cam_snap: scene_snapshot.CameraSnapshot,
        sky_texture: ?CubeTexture,
        ibl_intensity: f32,
        cache_key: u64,
        stats: *SceneStats,
        instances_prepared: bool,
        instance_source: mesh_mod.InstanceSource,
    ) void {
        scene_frame.prepareViewQueues(self, queues, cam_snap, sky_texture, ibl_intensity, cache_key, stats, instances_prepared, instance_source);
    }

    /// See `scene/frame_api.zig` (owns the body + docs).
    pub fn buildQueuesInto(self: *Scene, back: *FrameDrawSlot, params: QueueBuildParams) void {
        scene_frame.buildQueuesInto(self, back, params);
    }

    /// See `scene/frame_api.zig` (owns the body + docs).
    pub fn renderSceneView(
        self: *Scene,
        cam_snap: scene_snapshot.CameraSnapshot,
        queues: *const scene_render_queue.RenderQueues,
        outline_items: []const outline_pass.OutlineDrawItem,
        outline_skins: []const [scene_render_queue.MAX_BONES]Mat4,
        samples: i32,
        snap: *const scene_snapshot.SceneFrameSnapshot,
        env: scene_draw.Environment,
        view_slot: usize,
    ) void {
        scene_frame.renderSceneView(self, cam_snap, queues, outline_items, outline_skins, samples, snap, env, view_slot);
    }

    /// See `scene/frame_api.zig` (owns the body + docs).
    pub fn captureDirtyProbes(self: *Scene, snap: *const SceneFrameSnapshot) void {
        scene_frame.captureDirtyProbes(self, snap);
    }

    /// See `scene/frame_api.zig` (owns the body + docs).
    pub fn captureDirtyUi3dPanels(self: *Scene) void {
        scene_frame.captureDirtyUi3dPanels(self);
    }

    /// See `scene/frame_api.zig` (owns the body + docs).
    pub fn packFrameSnapshot(self: *Scene, aspect: f32, cur_w: i32, cur_h: i32) scene_snapshot.SceneFrameSnapshot {
        return scene_frame.packFrameSnapshot(self, aspect, cur_w, cur_h);
    }

    /// See `scene/frame_api.zig` (owns the body + docs).
    pub fn publishFrameSnapshot(self: *Scene, aspect: f32, cur_w: i32, cur_h: i32) void {
        scene_frame.publishFrameSnapshot(self, aspect, cur_w, cur_h);
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
    /// See `scene/frame_api.zig` (owns the body + docs).
    pub fn stageUiPacket(self: *Scene) void {
        scene_frame.stageUiPacket(self);
    }

    /// See `scene/frame_api.zig` (owns the body + docs).
    fn stageUiPacketInto(self: *Scene, slot: usize) void {
        scene_frame.stageUiPacketInto(self, slot);
    }

    /// See `scene/frame_api.zig` (owns the body + docs).
    pub fn captureUiFrame(self: *Scene, snap: *const SceneFrameSnapshot, back: *FrameDrawSlot) void {
        scene_frame.captureUiFrame(self, snap, back);
    }

    /// See `scene/frame_api.zig` (owns the body + docs).
    pub fn preparedDraws(self: *const Scene) *const FrameDrawSlot {
        return scene_frame.preparedDraws(self);
    }

    /// See `scene/frame_api.zig` (owns the body + docs).
    pub fn patchInstanceRefs(self: *Scene, back: *FrameDrawSlot) void {
        scene_frame.patchInstanceRefs(self, back);
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
    /// build touches (live meshes/canvas, `build_snapshot`, per-mesh
    /// previews, the particle/physics shared build frames and their layer
    /// seqs) is still phase-excluded today — the handoff seq words
    /// themselves are atomic since wave 30 (release/acquire, see the field
    /// docs), and since wave 32 the particle/physics payload the latch
    /// consumes rides the slot too (frozen copies, never the shared
    /// stores) — but the remaining live reads are not — see the adoption
    /// checklist in scene/frame_draws.zig. The claim API alone does not
    /// remove the phase mutex.
    pub fn tryClaimBuildSlot(self: *Scene) ?BuildClaim {
        // Contention-yield (wave 38): when enabled AND the previous build
        // is still unconsumed, park BEFORE reserving — the consumer is
        // behind, so another build now only adds CPU contention for a
        // payload the latch will supersede with a newer one. Gated on
        // nonzero first: the default path (yield 0) executes one not-taken
        // branch and touches no atomics here. The park holds no lease
        // (reserve happens after), so prepare's latch can always proceed.
        if (self.concurrent_yield_ns != 0 and
            self.build_seq.load(.monotonic) != self.last_latched_seq.load(.monotonic))
        {
            jobs.sleepNs(self.concurrent_yield_ns);
        }
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
        did_build: bool = false,

        /// Run the shared build core into the claimed slot (commit of the
        /// last published latch outcomes, CPU staging, record freeze, queue/
        /// shadow/outline build). sg-free, game side. Repeatable (newest
        /// wins); the generation is only committed by `publish()`.
        pub fn build(self: *BuildClaim) void {
            self.scene.buildIntoClaimedSlot(self.slot, self.seq);
            self.did_build = true;
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

        /// Copy small host-owned bytes into the CLAIMED slot's frozen host
        /// area (`FrameDrawSlot.host_bytes`), for the context to read via
        /// `PrepareClaim.host_bytes` instead of live host state (phase 2
        /// lock-free prepare: picked-name bytes, memory-summary tallies).
        /// Slot-owned copy (the claim owns the slot); fail-closed on OOM
        /// (the area is cleared, so the context sees absent bytes rather
        /// than a half-written payload). Call AFTER `build()` like
        /// `stageUi` (the build resets the slot) and BEFORE `publish()`.
        /// sg-free, game side. Repeatable (newest wins).
        pub fn stageHostBytes(self: *BuildClaim, bytes: []const u8) void {
            const slot = self.scene.draws.slotAt(self.slot);
            slot.host_bytes.clearRetainingCapacity();
            slot.host_bytes.appendSlice(self.scene.allocator, bytes) catch {
                slot.host_bytes.clearRetainingCapacity();
                // Observable fail-closed: the encode is fixed-size (≤128B
                // in practice) into retained slot capacity, so OOM here is
                // practically impossible after the first success — but a
                // silent clear would hide it forever. Game-side plain
                // counter (same discipline as last_upload_commit_frame:
                // written only by the producer, never across the handoff).
                self.scene.host_bytes_oob_drops +%= 1;
            };
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
            // UI-only claims are supported, so stamp every published slot
            // here as well as the full build core. Prepare consumes this
            // exact generation after newer producer claims may publish.
            const slot = s.draws.slotAt(self.slot);
            slot.build_seq = self.seq;
            slot.has_scene_build = self.did_build;
            s.draws.releaseHandoffWithSeq(self.slot, self.seq, &s.build_slot, &s.build_seq) catch {
                // A failed publish must not strand the producer lease. The
                // generation remains uncommitted and prepare sees no handoff.
                s.draws.cancelClaim(self.slot) catch {};
            };
        }

        /// Drop the claim without committing (seq/handoff untouched: prepare
        /// sees no fresh build and runs its fallback; provisional previews
        /// stamped with the uncommitted seq are overwritten by the next
        /// build and never latched). Staged-upload flags frozen into the
        /// dropped slot are re-armed on their live owners
        /// (token/index/uid-validated), so a cancelled claim consumes no
        /// upload — the next funded build re-freezes from the intact live
        /// arrays. The slot payload itself is ignored by prepare and reset
        /// by the next claim. Safe to call on an unconsumed claim;
        /// double-terminal is a bug (debug-asserted).
        pub fn cancel(self: *BuildClaim) void {
            std.debug.assert(!self.completed);
            self.completed = true;
            scene_upload_packets.restageDroppedSlot(self.scene, &self.scene.draws.slots[self.slot]);
            self.scene.draws.cancelClaim(self.slot) catch {};
        }
    };

    /// See `scene/frame_api.zig` (owns the body + docs).
    pub fn buildPreparedFrame(self: *Scene) void {
        scene_frame.buildPreparedFrame(self);
    }

    /// See `scene/frame_api.zig` (owns the body + docs).
    fn buildIntoClaimedSlot(self: *Scene, slot: usize, seq: u64) void {
        scene_frame.buildIntoClaimedSlot(self, slot, seq);
    }

    /// See `scene/frame_api.zig` (owns the body + docs).
    pub fn prepareFrame(self: *Scene) void {
        scene_frame.prepareFrame(self);
    }

    /// Split context prepare at the caller's producer/live-state lock
    /// boundary. See `scene/frame_api.zig` for the ownership contract.
    pub const PrepareClaim = scene_frame.PrepareClaim;

    pub fn beginStagedPrepare(self: *Scene) ?PrepareClaim {
        return scene_frame.beginStagedPrepare(self);
    }

    pub fn finishStagedPrepare(self: *Scene, claim: PrepareClaim) void {
        scene_frame.finishStagedPrepare(self, claim);
    }

    pub fn cancelStagedPrepare(self: *Scene, claim: PrepareClaim) void {
        scene_frame.cancelStagedPrepare(self, claim);
    }

    /// See `scene/lights_api.zig` (owns the body + docs).
    pub fn updateLights(self: *Scene, dt: f32) void {
        scene_lights_api.updateLights(self, dt);
    }

    /// See `scene/frame_api.zig` (owns the body + docs).
    pub fn update(self: *Scene, dt: f32) particles.UpdateError!void {
        return scene_frame.update(self, dt);
    }

    /// See `scene/frame_api.zig` (owns the body + docs).
    pub fn flushPendingGpuUploads(self: *Scene) void {
        scene_frame.flushPendingGpuUploads(self);
    }

    /// See `scene/frame_api.zig` (owns the body + docs).
    pub fn render(self: *Scene) void {
        scene_frame.render(self);
    }

    /// See `scene/frame_api.zig` (owns the body + docs).
    pub fn hasConsumableFrame(self: *const Scene) bool {
        return scene_frame.hasConsumableFrame(self);
    }

    /// See `scene/frame_api.zig` (owns the body + docs).
    pub fn reuseStreak(self: *const Scene) u64 {
        return scene_frame.reuseStreak(self);
    }

    /// See `scene/frame_api.zig` (owns the body + docs).
    pub fn uiPacketLatchedCount(self: *const Scene) u64 {
        return scene_frame.uiPacketLatchedCount(self);
    }

    /// See `scene/frame_api.zig` (owns the body + docs).
    pub fn pendingRetires(self: *Scene) usize {
        return scene_frame.pendingRetires(self);
    }

    /// See `scene/frame_api.zig` (owns the body + docs).
    pub fn renderReuse(self: *Scene) void {
        scene_frame.renderReuse(self);
    }

    /// See `scene/lifecycle.zig` (owns the body + docs).
    pub fn deinit(self: *Scene) void {
        scene_lifecycle.deinit(self);
    }

    /// See `scene/profile_api.zig` (owns the body + docs).
    pub fn startProfiling(self: *Scene) void {
        scene_profile.startProfiling(self);
    }

    /// See `scene/profile_api.zig` (owns the body + docs).
    pub fn stopProfiling(self: *Scene) void {
        scene_profile.stopProfiling(self);
    }

    /// See `scene/profile_api.zig` (owns the body + docs).
    pub fn resetProfiling(self: *Scene) void {
        scene_profile.resetProfiling(self);
    }

    /// See `scene/profile_api.zig` (owns the body + docs).
    pub fn isProfiling(self: *const Scene) bool {
        return scene_profile.isProfiling(self);
    }

    /// See `scene/profile_api.zig` (owns the body + docs).
    pub fn captureMemorySnapshot(self: *Scene) !*const profiler_mod.MemorySnapshot {
        return scene_profile.captureMemorySnapshot(self);
    }

    /// See `scene/profile_api.zig` (owns the body + docs).
    pub fn saveProfileReportHtml(self: *Scene, path: []const u8) !void {
        return scene_profile.saveProfileReportHtml(self, path);
    }

    /// See `scene/profile_api.zig` (owns the body + docs).
    pub fn saveProfileReportMd(self: *Scene, path: []const u8) !void {
        return scene_profile.saveProfileReportMd(self, path);
    }

    /// See `scene/profile_api.zig` (owns the body + docs).
    pub fn saveProfileTraceJson(self: *Scene, path: []const u8) !void {
        return scene_profile.saveProfileTraceJson(self, path);
    }

    /// See `scene/profile_api.zig` (owns the body + docs).
    pub fn saveProfileReports(self: *Scene, base_path: []const u8) !void {
        return scene_profile.saveProfileReports(self, base_path);
    }

    /// See `scene/profile_api.zig` (owns the body + docs).
    pub fn saveProfileReportsAsync(
        self: *Scene,
        base_path: []const u8,
        files: profiler_mod.ReportFiles,
        out: *[3]?*profiler_mod.ReportWriteTask,
    ) !void {
        return scene_profile.saveProfileReportsAsync(self, base_path, files, out);
    }
};
