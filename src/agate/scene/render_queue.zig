//! Facade for the render-queue modules. The ~3.9k-line `render_queue.zig`
//! was split into focused leaves under `render_queue/` following the repo
//! pattern (free functions + thin forwarders; Zig 0.16 has no
//! usingnamespace; see `scene/`, `physics/` precedent):
//!
//! - `render_queue/items.zig` — payload structs (`RenderMeshItem`,
//!   `RenderInstancedBatch`, `TransparentDrawEntry`, skin/shader/coat
//!   storages, `ParallelCullScratch`, `RenderQueues`) plus pure
//!   sort/classify helpers. Leaf: no sibling imports.
//! - `render_queue/cull.zig` — `FrameCullContext`, world-matrix cache,
//!   plain-mesh cull, material-record build, queue append. Imports `items`.
//! - `render_queue/instances.zig` — instanced submit/staging glue.
//!   Imports `items` + `cull`.
//! - `render_queue/build.zig` — `buildFrameQueues` plus the parallel cull
//!   pass and merge tail. Imports `items` + `cull` + `instances`.
//!
//! Everything that was public before the split is re-exported here
//! unchanged; consumers (`scene.zig`, `scene/*`, `passes/*`, `root.zig`)
//! see the same API as when everything lived in this file.
//!
//! Documented anti-cycle rule: leaves must never import this facade —
//! importing it back would make the re-exports depend on their own
//! consumers. Cross-leaf helpers (`cull.appendRenderItem`,
//! `cull.buildMaterialRecord`, `cull.cullNonInstancedMesh`,
//! `instances.submitInstancedMesh`) are `pub` in their home module for the
//! sibling that needs them but are deliberately NOT re-exported here, so
//! the public surface is identical to the pre-split file.
const items = @import("render_queue/items.zig");
const cull = @import("render_queue/cull.zig");
const build = @import("render_queue/build.zig");

// Re-exports of the extracted leaf modules (public API unchanged).
pub const MAX_BONES = items.MAX_BONES;
pub const RenderMeshItem = items.RenderMeshItem;
pub const RenderInstancedBatch = items.RenderInstancedBatch;
pub const SkinStorage = items.SkinStorage;
pub const ShaderStorage = items.ShaderStorage;
pub const CoatStorage = items.CoatStorage;
pub const skinAt = items.skinAt;
pub const coatAt = items.coatAt;
pub const CulledMesh = items.CulledMesh;
pub const TransparentKind = items.TransparentKind;
pub const TransparentDrawEntry = items.TransparentDrawEntry;
pub const ParallelCullScratch = items.ParallelCullScratch;
pub const RenderQueues = items.RenderQueues;
pub const sortRenderItems = items.sortRenderItems;
pub const sortTransparentDrawOrder = items.sortTransparentDrawOrder;
pub const materialIsTransparent = items.materialIsTransparent;
pub const materialIsCutout = items.materialIsCutout;
pub const materialIsDoubleSided = items.materialIsDoubleSided;
pub const blendDescFor = items.blendDescFor;
pub const worldMatrixCached = cull.worldMatrixCached;
pub const worldAABBCached = cull.worldAABBCached;
pub const FrameCullContext = cull.FrameCullContext;
pub const buildFrameQueues = build.buildFrameQueues;
