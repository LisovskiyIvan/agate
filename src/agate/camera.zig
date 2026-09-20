//! Facade for the camera modules. The `camera.zig` camera set was split
//! into focused leaves under `camera/` following the repo pattern (one
//! self-contained type per leaf plus a dispatcher; Zig 0.16 has no
//! usingnamespace; see `profiler.zig`, `scene/render_queue.zig`,
//! `serialization.zig`, `ui.zig`):
//!
//! - `camera/viewport.zig` — shared `Viewport` (+ `PixelRect`) used by every
//!   camera type. Leaf: no sibling imports.
//! - `camera/arc_rotate.zig` — orbit camera (`ArcRotateCameraOptions`,
//!   `ArcRotateCamera`). Imports `viewport` only.
//! - `camera/free.zig` — WASD + drag camera (`FreeCameraOptions`,
//!   `FreeCamera`). Imports `viewport` only.
//! - `camera/follow.zig` — chase camera (`FollowCameraOptions`,
//!   `FollowCamera`). Imports `viewport` + `../mesh.zig` (target link).
//! - `camera/target.zig` — observer camera (`TargetCameraOptions`,
//!   `TargetCamera`). Imports `viewport` only.
//! - `camera/fly.zig` — 6-DOF flight camera (`FlyCameraOptions`,
//!   `FlyCamera`). Imports `viewport` only.
//! - `camera/union.zig` — polymorphic `Camera` union dispatching over the
//!   five leaves. Imports the siblings, never this facade.
//!
//! Everything that was public before the split is re-exported here
//! unchanged; consumers (`scene.zig`, `scene/*`, `root.zig`,
//! `serialization/*`) see the same API as when everything lived in this
//! file. Unlike `ui/*` or `profiler/*`, no `anytype` forwarders are needed:
//! each camera struct is fully owned by its leaf and the union dispatches
//! to the concrete types directly.
//!
//! Documented anti-cycle rule: leaves must never import this facade —
//! importing it back would make the re-exports depend on their own
//! consumers.
const viewport_mod = @import("camera/viewport.zig");
const arc_rotate_mod = @import("camera/arc_rotate.zig");
const free_mod = @import("camera/free.zig");
const follow_mod = @import("camera/follow.zig");
const target_mod = @import("camera/target.zig");
const fly_mod = @import("camera/fly.zig");
const union_mod = @import("camera/union.zig");

// Shared viewport (lives in camera/viewport.zig).
pub const Viewport = viewport_mod.Viewport;

// Orbit camera (lives in camera/arc_rotate.zig).
pub const ArcRotateCameraOptions = arc_rotate_mod.ArcRotateCameraOptions;
pub const ArcRotateCamera = arc_rotate_mod.ArcRotateCamera;

// WASD + drag camera (lives in camera/free.zig).
pub const FreeCameraOptions = free_mod.FreeCameraOptions;
pub const FreeCamera = free_mod.FreeCamera;

// Chase camera (lives in camera/follow.zig).
pub const FollowCameraOptions = follow_mod.FollowCameraOptions;
pub const FollowCamera = follow_mod.FollowCamera;

// Observer camera (lives in camera/target.zig).
pub const TargetCameraOptions = target_mod.TargetCameraOptions;
pub const TargetCamera = target_mod.TargetCamera;

// 6-DOF flight camera (lives in camera/fly.zig).
pub const FlyCameraOptions = fly_mod.FlyCameraOptions;
pub const FlyCamera = fly_mod.FlyCamera;

// Polymorphic union (lives in camera/union.zig).
pub const Camera = union_mod.Camera;
