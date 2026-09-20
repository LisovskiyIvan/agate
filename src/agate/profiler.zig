//! Facade for the profiler modules. The `profiler.zig` flight recorder
//! was split into focused leaves under `profiler/` following the repo
//! pattern (free functions + thin forwarders; Zig 0.16 has no
//! usingnamespace; see `scene/render_queue.zig`, `serialization.zig`,
//! `ui.zig`):
//!
//! - `profiler/types.zig` — data types (`FrameRecord`, `MemorySnapshot`,
//!   `SessionSummary`, diagnostics, `dominantPhase`). Leaf: no sibling imports.
//! - `profiler/report.zig` — standalone HTML/Markdown/Chrome-trace
//!   generators over plain data values. Imports `types` only.
//! - `profiler/core.zig` — owns the `Profiler` type: recorder fields, the
//!   trivial lifecycle plus the report/save glue, and thin forwarders into
//!   the siblings below, so every call site keeps working unchanged.
//! - `profiler/recording.zig` — `recordFrame` (per-frame ring-buffer insert).
//! - `profiler/snapshot.zig` — `captureMemorySnapshot` (CPU/GPU memory census).
//! - `profiler/summary.zig` — `summarize` (session statistics + pacing).
//! - `profiler/diagnostics.zig` — `analyze` (bottleneck findings).
//!
//! Everything that was public before the split is re-exported here
//! unchanged; consumers (`scene.zig`, `scene/*`, `root.zig`) see the same
//! API as when everything lived in this file.
//!
//! Documented anti-cycle rule: leaves must never import this facade —
//! importing it back would make the re-exports depend on their own
//! consumers. The method bodies take the profiler as `anytype` (same
//! discipline as `ui/*` taking a generic canvas), so library code has no
//! leaf-to-owner edge at all; moved tests reach `core.Profiler` through a
//! block-scoped import that exists only in test builds.
const core = @import("profiler/core.zig");
const types = @import("profiler/types.zig");

// Profiler type (lives in profiler/core.zig).
pub const Profiler = core.Profiler;

// Re-exported so `@import("profiler.zig").FrameRecord` and every existing
// caller keep working; the data types live in profiler/types.zig.
pub const FrameRecord = types.FrameRecord;
pub const TextureMemoryRecord = types.TextureMemoryRecord;
pub const MeshMemoryRecord = types.MeshMemoryRecord;
pub const RenderTargetRecord = types.RenderTargetRecord;
pub const MemorySnapshot = types.MemorySnapshot;
pub const SessionSummary = types.SessionSummary;
pub const DiagnosticSeverity = types.DiagnosticSeverity;
pub const DiagnosticFinding = types.DiagnosticFinding;
pub const PhaseCulprit = types.PhaseCulprit;
pub const dominantPhase = types.dominantPhase;
