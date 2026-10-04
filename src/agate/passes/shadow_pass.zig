//! Facade for the shadow modules. The `shadow_pass.zig` shadow pass was split
//! into focused leaves under `shadow/` following the repo pattern (owner
//! type + thin forwarders; Zig 0.16 has no usingnamespace; see
//! `profiler.zig`, `particles.zig`, `texture.zig`, `camera.zig`):
//!
//! - `shadow/types.zig` — shared vocabulary: atlas constants
//!   (`SHADOW_ATLAS_SIZE`, `SPOT_*`, `POINT_*`), `SpotShadowRenderInfo`,
//!   `PointShadowRenderInfo`, the pipeline `Bucket` vocabulary with
//!   `bucket_order`, `BinResult` ranges, the pure `bucketFor` classifier and
//!   the point-atlas math (`pointFaceForDir`, `pointTileOrigin`). Leaf:
//!   imports `math` + `mesh` only.
//! - `shadow/core.zig` — owns the `ShadowPass` type: fields, the nested
//!   `ShadowDrawItem` + `PreparedShadowDraws` payloads, the trivial lifecycle
//!   (`init`/`deinit`), the small entry points (`prepare`/`renderPrepared`/
//!   `render`) and thin forwarders into the siblings below, so every call
//!   site keeps working unchanged. Also hosts the shared test helper.
//! - `shadow/binning.zig` — `binMeshes` (serial + parallel mesh binning).
//!   Imports `types` only.
//! - `shadow/prepare.zig` — `prepareInto` (render-owned snapshot build).
//!   Imports `types` (+ `mesh`, `scene/render_queue` as before).
//! - `shadow/buckets.zig` — `renderBuckets` (per-item culling, pipeline
//!   grouping, uniform packing) shared by all three atlas paths. Imports
//!   `types` (+ `scene/render_queue` for the skin resolve, as before).
//! - `shadow/csm.zig` — `renderCsm` (4 cascade tiles of the CSM atlas).
//! - `shadow/spot.zig` — `renderSpot` (spot atlas tiles).
//! - `shadow/point.zig` — `renderPoint` (point-light atlas face tiles).
//!
//! Everything that was public before the split is re-exported here
//! unchanged; consumers (`scene.zig`, `scene/*`, `passes/mod.zig`,
//! `profiler/snapshot.zig`) see the same API as when everything lived in
//! this file.
//!
//! Documented anti-cycle rule: leaves must never import this facade —
//! importing it back would make the re-exports depend on their own
//! consumers. The method bodies take the pass as `anytype` (same discipline
//! as `particles/*` taking a generic system), so library code has no
//! leaf-to-owner edge at all; moved tests reach `core.ShadowPass` helpers
//! through block-scoped imports that exist only in test builds.
//! Cross-leaf helpers (`types.bucketFor`, `buckets.renderBuckets`) are `pub`
//! in their home module for the sibling that needs them but are deliberately
//! NOT re-exported here, so the public surface is identical to the pre-split
//! file.
const core = @import("shadow/core.zig");
const types = @import("shadow/types.zig");

// Shadow pass owner (lives in shadow/core.zig).
pub const ShadowPass = core.ShadowPass;

// Atlas constants (live in shadow/types.zig).
pub const SHADOW_ATLAS_SIZE = types.SHADOW_ATLAS_SIZE;
pub const SPOT_SHADOW_SLOTS = types.SPOT_SHADOW_SLOTS;
pub const SPOT_SHADOW_MAP_WIDTH = types.SPOT_SHADOW_MAP_WIDTH;
pub const SPOT_SHADOW_MAP_HEIGHT = types.SPOT_SHADOW_MAP_HEIGHT;
pub const SPOT_SHADOW_RES = types.SPOT_SHADOW_RES;
pub const POINT_SHADOW_SLOTS = types.POINT_SHADOW_SLOTS;
pub const POINT_SHADOW_FACES = types.POINT_SHADOW_FACES;
pub const POINT_SHADOW_RES = types.POINT_SHADOW_RES;
pub const POINT_SHADOW_MAP_WIDTH = types.POINT_SHADOW_MAP_WIDTH;
pub const POINT_SHADOW_MAP_HEIGHT = types.POINT_SHADOW_MAP_HEIGHT;

// Per-light render infos (live in shadow/types.zig).
pub const SpotShadowRenderInfo = types.SpotShadowRenderInfo;
pub const PointShadowRenderInfo = types.PointShadowRenderInfo;

// Atlas math (lives in shadow/types.zig).
pub const pointFaceForDir = types.pointFaceForDir;
pub const pointTileOrigin = types.pointTileOrigin;
pub const spotTileOrigin = types.spotTileOrigin;
