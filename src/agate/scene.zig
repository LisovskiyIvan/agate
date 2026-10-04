//! Facade for the scene modules. The `scene.zig` orchestrator was split
//! into an owner + focused leaves under `scene/` following the repo pattern
//! (free functions + thin forwarders; Zig 0.16 has no usingnamespace; see
//! `audio.zig`, `profiler.zig`, `camera.zig`, `ttf.zig`):
//!
//! - `scene/core.zig` — owns the `Scene` type: content-registry fields,
//!   camera state, cross-cutting config/stats, every render-subsystem field,
//!   the frame-handoff mailboxes, `CameraEntry`, `BuildClaim`, the trivial
//!   lifecycle (`initInto`/`init`) plus thin forwarders into the siblings
//!   below, so every call site keeps working unchanged. Imports every leaf.
//! - `scene/lifecycle.zig` — full `deinit`, offscreen resize, post/SSAO
//!   setters, async save/load, skybox setup, lazy MSAA pipeline twin.
//! - `scene/registry.zig` — material/mesh create/destroy + mesh search by
//!   name/tag/query (incl. the `destroyMesh` cross-layer referent cleanup).
//! - `scene/cameras.zig` — `CameraEntry` + camera add/remove/get/switch/
//!   cycle/set/update.
//! - `scene/lights_api.zig` — hemispheric/point/spot/directional/area/
//!   clustered light API + update-phase `updateLights` packing.
//! - `scene/attachments.zig` — bounded optional attachments: reflection
//!   probes, 3D-GUI panels (+ picking), per-mesh highlights, PBD cloth.
//! - `scene/sim_api.zig` — simulation/content-builder attach points:
//!   decals, particles, trails, CSG/greased-line/simplify builders, nav,
//!   animations, physics.
//! - `scene/query_api.zig` — picking rays/picks, UI canvas accessors,
//!   projection, event routing.
//! - `scene/frame_api.zig` — frame orchestration: queue/view builds,
//!   snapshot publish, UI packet staging, prepared-draw accessors, the
//!   producer-build claim core, prepare/update/flush/render + reuse guards.
//! - `scene/profile_api.zig` — thin `profiler` forwarders.
//! - Pre-existing leaves, untouched: `scene/content.zig`,
//!   `scene/frame_build.zig`, `scene/frame_draws.zig`,
//!   `scene/frame_prepare.zig`, `scene/frame_render.zig`,
//!   `scene/render_queue.zig` (+ `render_queue/`), `scene/picking.zig`,
//!   `scene/snapshot.zig`, `scene/stats.zig`, `scene/tests.zig`, and the
//!   layer/state leaves (`light_rig`, `shadow_system`, `sky_layer`,
//!   `probe_layer`, `probe_render`, `clustered_lights`, `gui3d_layer`,
//!   `highlight_layer`, `postfx_stack`, `forward_pipelines`, `pipelines`,
//!   `particle_layer`, `decal_layer`, `trail_layer`, `nav_layer`,
//!   `physics_layer`, `animation_runtime`, `queue_builder`, `draw`,
//!   `view_render`, `viewport_clear`, `msaa`, `project_cache`,
//!   `projection`, `uniforms`, `ui_frame`, `ui_capture`,
//!   `patch_instance_refs`, `instance_staging`, `gpu_retire`, `cascades`,
//!   `light_selection`, `shadow_pcss`).
//!
//! Everything that was public before the split is re-exported here
//! unchanged; consumers (`root.zig`, loaders, serialization, sandbox, all
//! `scene/*` tests) see the same API as when everything lived in this file.
//!
//! Documented anti-cycle rule: leaves must never import this facade —
//! importing it back would make the re-exports depend on their own
//! consumers. Method bodies take the scene as `anytype` (same discipline
//! as `profiler/*` taking a generic profiler), so library code has no
//! leaf-to-owner edge at all; leaf-to-leaf calls use direct sibling
//! imports. Cross-leaf helpers (`frame_api.stageUiPacketInto`,
//! `frame_api.buildIntoClaimedSlot`) are `pub` in their home module for
//! the owner (`core.zig`) but are deliberately NOT re-exported here, so the
//! public surface is identical to the pre-split file.
//!
//! Honest structural notes:
//! - `core.audioRaycastAdapter` + `core.evaluateAudioOcclusion` stay on the
//!   owner: the adapter casts `user_data` to the concrete `*Scene`, which a
//!   leaf cannot name without importing the owner back.
//! - `Scene.tryClaimBuildSlot` + `Scene.BuildClaim` stay on the owner: the
//!   claim holds a `*Scene`. `BuildClaim.build/stageUi` reach the frame
//!   leaf directly (`frame_api.buildIntoClaimedSlot`,
//!   `frame_api.stageUiPacketInto`) instead of `self.scene.*` — private
//!   owner methods cannot be called through `anytype` (same finding as the
//!   audio split); behavior is identical, only the call path changed.
//! - `createGreasedLine` now reaches `mesh/mesh.zig` through a top-level
//!   leaf import instead of an inline `@import`, and `prepareViewQueues`
//!   names `mesh_mod.InstanceSource` instead of an inline `@import` —
//!   module-prefix substitutions only.
//! - The dead private `msSince` helper (no callers before the split) is
//!   preserved verbatim in `core.zig`.
const core = @import("scene/core.zig");
const scene_cameras = @import("scene/cameras.zig");

/// The scene orchestrator (lives in `scene/core.zig`).
pub const Scene = core.Scene;
/// Optional per-domain allocator config (lives in `scene/core.zig`).
pub const AllocatorConfig = core.AllocatorConfig;
/// Camera list entry (lives in `scene/cameras.zig`).
pub const CameraEntry = scene_cameras.CameraEntry;

const tags_mod = @import("tags.zig");
pub const TagSet = tags_mod.TagSet;
pub const TagQuery = tags_mod.TagQuery;

const scene_stats = @import("scene/stats.zig");
pub const SceneStats = scene_stats.SceneStats;
const scene_render_queue = @import("scene/render_queue.zig");
pub const RenderMeshItem = scene_render_queue.RenderMeshItem;
const lights = @import("lights.zig");
pub const ClusteredPointLight = lights.ClusteredPointLight;
pub const ClusteredPointLightOptions = lights.ClusteredPointLightOptions;
pub const ClusteredSpotLight = lights.ClusteredSpotLight;
pub const ClusteredSpotLightOptions = lights.ClusteredSpotLightOptions;
const scene_probes = @import("scene/probe_layer.zig");
pub const ReflectionProbe = scene_probes.ReflectionProbe;
pub const ReflectionProbeOptions = scene_probes.ReflectionProbeOptions;
const scene_gui3d = @import("scene/gui3d_layer.zig");
pub const Ui3dPanel = scene_gui3d.Ui3dPanel;
pub const Ui3dPanelOptions = scene_gui3d.Ui3dPanelOptions;
pub const Ui3dFaceMode = scene_gui3d.Ui3dFaceMode;
pub const Ui3dPickHit = scene_gui3d.Ui3dPickHit;
const scene_highlight = @import("scene/highlight_layer.zig");
pub const HighlightLayer = scene_highlight.HighlightLayer;
pub const HighlightEntry = scene_highlight.HighlightEntry;
pub const HighlightOptions = scene_highlight.HighlightOptions;
const softbody_mod = @import("softbody.zig");
pub const SoftBody = softbody_mod.SoftBody;
pub const SoftBodyLayer = softbody_mod.SoftBodyLayer;
pub const Cloth = softbody_mod.Cloth;
pub const ClothOptions = softbody_mod.ClothOptions;
pub const SphereCollider = softbody_mod.SphereCollider;
const scene_ui_frame = @import("scene/ui_frame.zig");
pub const UiFrame = scene_ui_frame.UiFrame;
const scene_frame_draws = @import("scene/frame_draws.zig");
/// P7 published consumable draw payload (one coherent prepared frame).
pub const FrameDrawSlot = scene_frame_draws.FrameDrawSlot;
/// P7 three retained owning queue slots (the variable-list triple buffer
/// with a consumer pin/lease protocol).
pub const FrameDraws = scene_frame_draws.FrameDraws;
const scene_queue_builder = @import("scene/queue_builder.zig");
pub const QueueBuildParams = scene_queue_builder.QueueBuildParams;
pub const scene_snapshot = @import("scene/snapshot.zig");
pub const SceneFrameSnapshot = scene_snapshot.SceneFrameSnapshot;
pub const CameraSnapshot = scene_snapshot.CameraSnapshot;
pub const profiler_mod = @import("profiler.zig");
pub const Profiler = profiler_mod.Profiler;
pub const FrameRecord = profiler_mod.FrameRecord;
pub const MemorySnapshot = profiler_mod.MemorySnapshot;

// Test-only seam exposure: the P5 live-GPU gate arms
// `instance_staging.testArmGrowthFailOnce()` to exercise the real makeBuffer
// .FAILED staging branch, then reads `testLastInjectedFailId()` to prove the
// failed handle itself was destroyed. Same module instance the frame leaves
// use (file-path dedup); one not-taken branch when unarmed, no behavior
// change otherwise.
pub const instance_staging = @import("scene/instance_staging.zig");

// Test-only seam exposure: the P5 live-GPU gate arms
// `mesh_deferred.testArmMeshFailOnce()` to exercise the real makeBuffer
// .FAILED deferred-creation branch in Mesh.finishGpuUpload, then reads
// `testLastInjectedMeshFailId()` to prove the failed handle itself was
// destroyed. Same module instance the flush path uses (file-path dedup);
// one not-taken branch when unarmed, no behavior change otherwise.
pub const mesh_deferred = @import("mesh/mesh.zig");
