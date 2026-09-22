//! Facade for the mesh builders modules. The `mesh/builders.zig` CPU geometry
//! builders were split into focused leaves under `mesh/builders/` following
//! the repo pattern (per-responsibility leaves with same-named re-exports;
//! Zig 0.16 has no usingnamespace; see `scene/render_queue/build.zig`,
//! `camera.zig`, `texture.zig`, `profiler.zig`):
//!
//! - `builders/common.zig` — shared trig tables, grid-quad index writers,
//!   frame-seed helper, and 2D ear-clipping predicates. Imports
//!   `../types.zig` + `../tangents.zig` only.
//! - `builders/solids.zig` — closed primitive solids (box, ground, terrain,
//!   sphere, cylinder, capsule) with their option structs. Imports `common`
//!   only.
//! - `builders/revolve.zig` — flat and revolved parametric surfaces (plane,
//!   torus, torus-knot, disc) with their option structs. Imports `common`
//!   only.
//! - `builders/sweep.zig` — path-swept and lofted surfaces (ribbon, lathe,
//!   tube, lines) with their option structs plus the shared
//!   parallel-transport frame machinery. Imports `common` only.
//! - `builders/extrude.zig` — 2D profile to 3D prism with its option struct.
//!   Imports `common` only.
//! - `builders/polygon.zig` — 2D shape (holes, planes, depth) triangulation
//!   with its option structs. Imports `common` only.
//!
//! Everything that was public before the split is re-exported here
//! unchanged; consumers (`mesh.zig`, `mesh/builder.zig`, `mesh/tests.zig`)
//! see the same API as when everything lived in this file. These are pure
//! CPU geometry builders — no GPU, no threads.
//!
//! Documented anti-cycle rule: leaves must never import this facade —
//! importing it back would make the re-exports depend on their own
//! consumers. The ear-clipping predicates (`common.orient2d`,
//! `common.pointInTriangle2d`, `common.isEarTip`,
//! `common.segmentsCross2d`) are `pub` in their home module for the
//! extrude/polygon siblings but are deliberately NOT re-exported here, so
//! the public surface is identical to the pre-split file.
const common = @import("builders/common.zig");
const solids = @import("builders/solids.zig");
const revolve = @import("builders/revolve.zig");
const sweep = @import("builders/sweep.zig");
const extrude = @import("builders/extrude.zig");
const polygon = @import("builders/polygon.zig");

// Shared trig/index/frame helpers (live in builders/common.zig).
pub const TrigEntry = common.TrigEntry;
pub const trigEntry = common.trigEntry;
pub const buildTrigTable = common.buildTrigTable;
pub const storeQuad = common.storeQuad;
pub const storeQuadFlipped = common.storeQuadFlipped;
pub const appendGridQuad = common.appendGridQuad;
pub const appendGridQuadFlipped = common.appendGridQuadFlipped;
pub const resolveFrameSeed = common.resolveFrameSeed;

// Closed primitive solids (live in builders/solids.zig).
pub const BoxOptions = solids.BoxOptions;
pub const buildBoxData = solids.buildBoxData;
pub const GroundOptions = solids.GroundOptions;
pub const buildGroundData = solids.buildGroundData;
pub const TerrainOptions = solids.TerrainOptions;
pub const buildTerrainData = solids.buildTerrainData;
pub const SphereOptions = solids.SphereOptions;
pub const buildSphereData = solids.buildSphereData;
pub const CylinderOptions = solids.CylinderOptions;
pub const buildCylinderData = solids.buildCylinderData;
pub const CapsuleOptions = solids.CapsuleOptions;
pub const buildCapsuleData = solids.buildCapsuleData;

// Flat and revolved parametric surfaces (live in builders/revolve.zig).
pub const PlaneOptions = revolve.PlaneOptions;
pub const buildPlaneData = revolve.buildPlaneData;
pub const TorusOptions = revolve.TorusOptions;
pub const buildTorusData = revolve.buildTorusData;
pub const TorusKnotOptions = revolve.TorusKnotOptions;
pub const buildTorusKnotData = revolve.buildTorusKnotData;
pub const DiscOptions = revolve.DiscOptions;
pub const buildDiscData = revolve.buildDiscData;

// Path-swept and lofted surfaces (live in builders/sweep.zig).
pub const RibbonOptions = sweep.RibbonOptions;
pub const buildRibbonData = sweep.buildRibbonData;
pub const LatheOptions = sweep.LatheOptions;
pub const buildLatheData = sweep.buildLatheData;
pub const TubeOptions = sweep.TubeOptions;
pub const buildTubeData = sweep.buildTubeData;
pub const LinesOptions = sweep.LinesOptions;
pub const buildLinesData = sweep.buildLinesData;

// 2D profile to 3D prism (lives in builders/extrude.zig).
pub const ExtrudeOptions = extrude.ExtrudeOptions;
pub const buildExtrudeData = extrude.buildExtrudeData;

// 2D shape triangulation (lives in builders/polygon.zig).
pub const PolygonSideOrientation = polygon.PolygonSideOrientation;
pub const PolygonPlane = polygon.PolygonPlane;
pub const PolygonOptions = polygon.PolygonOptions;
pub const buildPolygonData = polygon.buildPolygonData;
