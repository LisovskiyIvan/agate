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

const tags_mod = @import("tags.zig");

const scene_stats = @import("scene/stats.zig");
pub const SceneStats = scene_stats.SceneStats;
const scene_render_queue = @import("scene/render_queue.zig");
pub const RenderMeshItem = scene_render_queue.RenderMeshItem;
pub const sortRenderItems = scene_render_queue.sortRenderItems;
pub const sortOpaqueItemsIndexed = scene_render_queue.sortOpaqueItemsIndexed;
pub const RenderQueues = scene_render_queue.RenderQueues;
const lights = @import("lights.zig");
const scene_probes = @import("scene/probe_layer.zig");
pub const ReflectionProbe = scene_probes.ReflectionProbe;
const scene_gui3d = @import("scene/gui3d_layer.zig");
pub const Ui3dPanel = scene_gui3d.Ui3dPanel;
const scene_highlight = @import("scene/highlight_layer.zig");
pub const HighlightLayer = scene_highlight.HighlightLayer;
pub const HighlightEntry = scene_highlight.HighlightEntry;
pub const HighlightOptions = scene_highlight.HighlightOptions;
const softbody_mod = @import("softbody.zig");
const scene_ui_frame = @import("scene/ui_frame.zig");
const scene_frame_draws = @import("scene/frame_draws.zig");
/// P7 published consumable draw payload (one coherent prepared frame).
pub const FrameDrawSlot = scene_frame_draws.FrameDrawSlot;
const scene_queue_builder = @import("scene/queue_builder.zig");
pub const scene_snapshot = @import("scene/snapshot.zig");
pub const SceneFrameSnapshot = scene_snapshot.SceneFrameSnapshot;
pub const cascades = @import("scene/cascades.zig");

// Test-only seam exposure: the P5 live-GPU gate arms
// `instance_staging.testArmGrowthFailOnce()` to exercise the real makeBuffer
// .FAILED staging branch, then reads `testLastInjectedFailId()` to prove the
// failed handle itself was destroyed. Same module instance the frame leaves
// use (file-path dedup); one not-taken branch when unarmed, no behavior
// change otherwise.
pub const instance_staging = @import("scene/instance_staging.zig");
