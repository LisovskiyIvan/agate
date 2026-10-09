//! Facade for the frame-queue build modules. The `render_queue/build.zig`
//! frame assembly was split into focused leaves under `build/` following the
//! repo pattern (free functions + thin forwarders; Zig 0.16 has no
//! usingnamespace; see `scene/render_queue.zig`, `camera.zig`,
//! `profiler.zig`):
//!
//! - `build/frame.zig` — `buildFrameQueues` (occluder rasterization,
//!   parallel dispatch with OOM fallback, serial cull loop). Imports
//!   `items` + `cull` + `instances` + the `parallel` sibling.
//! - `build/parallel.zig` — `ParallelCull` + `buildFrameQueuesParallel`
//!   (world-matrix warming, chunked parallel cull, deterministic merge,
//!   serial instanced tail). Imports `items` + `cull` + `instances`.
//! - `build/equivalence_tests.zig` — serial/parallel equivalence + OOM-fallback
//!   integration tests (`FailFirstN`). Test-only leaf importing `frame`.
//! - `build/snapshots_tests.zig` — snapshot-durability (P4) tests +
//!   `expectP4QueueRefsValid`. Test-only leaf importing `frame`.
//!
//! Everything that was public before the split is re-exported here
//! unchanged; consumers (`render_queue.zig`, `scene.zig`) see the same API
//! as when everything lived in this file.
//!
//! Documented anti-cycle rule: leaves must never import this facade —
//! importing it back would make the re-exports depend on their own
//! consumers. `parallel.buildFrameQueuesParallel` is `pub` in its home
//! module for the `frame` sibling but is deliberately NOT re-exported here,
//! so the public surface is identical to the pre-split file.
const frame = @import("build/frame.zig");

// Frame-queue assembly (lives in build/frame.zig).
pub const buildFrameQueues = frame.buildFrameQueues;
