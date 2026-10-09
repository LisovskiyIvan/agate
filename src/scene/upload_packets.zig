//! Slot-owned dynamic-upload packets (producer freeze-then-latch,
//! lock-free publication).
//!
//! Producer side (`stageUploads`, game/update phase, sg-free): copies every
//! per-frame GPU staging payload into the claimed `FrameDrawSlot` by value
//! (morph vertex bytes, particle CPU/GPU/compute staging bytes, trail verts
//! + indices, softbody verts + indices, greased verts + indices,
//! pending-mesh creation geometry) plus frozen buffer ids / counts / bounds
//! / allocation sizes. Empty-but-dirty owners freeze empty packets so the
//! staged flush observes them exactly like the quiesced drain. The producer
//! clears each live dirty flag AT STAGE TIME (when it freezes the packet);
//! a cancelled claim must not consume the upload, so `BuildClaim.cancel`
//! re-arms every flag staged into the dropped slot via
//! `restageDroppedSlot` (game-side, token-validated).
//!
//! Context side (`flushSlotUploads`, beginPrepare with a fresh build):
//! uploads from the slot packets and creates deferred buffers from FROZEN
//! sizes, then records per-packet outcomes IN THE SLOT DESCRIPTORS
//! (`delivered` + created handles + consumed window). THE CONTEXT NEVER
//! WRITES GAME-OWNED STATE on this path: no live flag clears, no live
//! scalar publishes, no live handle installs, no live array frees, no live
//! staging memcpy. No live staging BYTES drive uploads (upload is direct from
//! packet bytes with frozen counts only) and no live mutable lengths are read
//! (allocation sizes ride frozen in the packets; `capacity` fields are
//! immutable after system init — documented at each freeze site).
//!
//! Game side (`commitSlotResults`, next build, under the game lock, front
//! slot under a read lease): validates each packet outcome by
//! index/token(/uid) and applies it — installs created handles (a replaced
//! non-zero live handle retires through the thread-safe queue; in practice
//! the live id is always zero here), publishes scalars (trail
//! index_count/bounds, pending-mesh vertex_count/morph flag, compute ring
//! advance + dt consume), frees consumed pending arrays, and re-arms flags
//! for every undelivered packet so the next funded build retries. Applied
//! once per published frame (`last_upload_commit_frame` vs
//! `front.frame_id` in the build core — a repeat build without an
//! intervening latch skips, so outcomes are never double-applied).
//!
//! Memory ordering: the context writes outcomes into its CLAIMED slot
//! (private until publish); the publish release edge (`tryPublish` /
//! `releaseHandoffWithSeq` under the lease mutex) pairs with the
//! producer's `pinFrontReader` acquire (same mutex) before the commit
//! reads them. Payload reads/writes on distinct slots need no further
//! locking (single producer; see frame_draws.zig).
//!
//! Deferred morph-delta textures (`morph_upload_pending`, GPU-mode only)
//! ride the pending-mesh packet: the producer packs the RGBA32F delta
//! pixels into `pending_delta_data` at stage time (write-once), the context
//! creates the delta image + view from those frozen bytes alongside the
//! base buffers, and the commit installs everything atomically — the mesh
//! leaves `gpu_pending` fully drawable (no base-pose frame).
//!
//! `flushPendingGpuUploads` is a separate quiesced completion drain: its
//! `finishGpuUpload` scans require a stopped/excluded producer, never replace
//! staged frame preparation, and are not called by the normal renderer.
//!
//! Identity: descriptors carry `token` (@intFromPtr of the live owner) +
//! list index (+ uid for meshes). The COMMIT validates token/index (/uid)
//! with a pointer compare first (never dereferencing a stale pointer):
//! mismatch skips fail-closed (previous complete GPU state stands, the
//! created handles retire through the queue, retry next build). Mesh-list
//! or registry mutation between build and latch violates the app contract;
//! the guard keeps it coherent, never corrupt.
//!
//! OOM: any packet that fails to stage is skipped like an OOM-skipped
//! instance segment (no partial packet: data truncated back, descriptor
//! unwritten). The stage-time flag clear is then uncovered — the next
//! build re-freezes from the intact live arrays because the mutation that
//! set the flag is still staged live... EXCEPT a flag cleared with no
//! packet and no newer mutation would be lost. To close that hole the
//! stage clears a flag ONLY when its packet is successfully frozen (clear
//! after append, per owner); an OOM-skipped owner keeps its flag set.
//! Undeliverable packets (creation OOM, headless, unsupported backend)
//! record `delivered = false` and the commit re-arms the flags, so the
//! next funded build retries from the intact live arrays.
//! Layout: `upload_packets_stage.zig` (producer freeze),
//! `upload_packets_flush.zig` (context flush),
//! `upload_packets_commit.zig` (game-side commit + re-arm),
//! `upload_packets_transient.zig` (reuse-frame rewrite). This file is
//! a facade re-exporting the public API so callers are unchanged.

const stage_mod = @import("upload_packets_stage.zig");
const flush_mod = @import("upload_packets_flush.zig");
const commit_mod = @import("upload_packets_commit.zig");
const transient_mod = @import("upload_packets_transient.zig");

pub const stageUploads = stage_mod.stageUploads;
pub const flushSlotUploads = flush_mod.flushSlotUploads;
pub const commitSlotResults = commit_mod.commitSlotResults;
pub const restageDroppedSlot = commit_mod.restageDroppedSlot;
pub const rewriteTransientWrites = transient_mod.rewriteTransientWrites;

// Staged-upload regression tests live in `upload_packets_tests.zig` (same directory,
// imported below so the test registry picks them up exactly once).
