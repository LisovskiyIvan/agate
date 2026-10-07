//! Scene facade: public types and configuration live in `scene/core.zig`;
//! lifecycle, content, simulation and staged-frame work live in `scene/`.
//!
//! Leaves must not import this facade: that would cycle through their own
//! consumers. They take the scene as `anytype` and call sibling modules
//! directly. Concrete `*Scene` callbacks and the owning `BuildClaim` stay
//! on `core.zig`; internal cross-leaf helpers are not re-exported here.
//!
//! GPU rendering consumes immutable staged slots. Producer building, context
//! staging, consumer leases and epoch retirement are coordinated by the frame
//! modules; see `frame_api.zig`, `frame_draws.zig` and `gpu_retire.zig`.
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
