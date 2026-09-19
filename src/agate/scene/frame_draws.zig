//! P7 double-buffered prepared draw payload: two retained owning queue slots
//! covering the prepared mesh draw lists — PRIMARY + ALL PIP view queues
//! (with their skin/shader side stores), outline items+skins, and prepared
//! shadow items+skins+bin ranges. Scope is mesh draws only (trail meshes
//! ride these same queues — Trail.update stages CPU-side and the prepare
//! flush uploads before the queue build bakes the values): UI stays outside
//! (P6 single-owned ui_frame), and particles, physics-debug lines, and sky
//! carry their own prepared frames/payloads — never these slots.
//!
//! Ownership / lifecycle block:
//! - Owns only CPU-side queue storage (ArrayList buffers, skin/shader copies,
//!   sort-order entries). GPU handles inside items (buffers/views/samplers/
//!   pipelines) are BORROWED under the phase mutex / P3 epoch discipline —
//!   this buffer never destroys them, never duplicates GPU buffers, and never
//!   changes the P3 retire algorithm. CPU slot retention is NOT a GPU
//!   lifetime pin: a retained slot may reference handles a GpuRetire flush
//!   already destroyed.
//! - Sequential phase contract: ONE pending frame, no concurrent
//!   prepare/render (same context thread). prepare builds the BACK slot in
//!   place (reset first), then publishes with a single index flip — never a
//!   shallow ArrayList copy (that would double-free), never a per-frame deep
//!   clone. Render reads only the published FRONT slot through const
//!   payloads. Update CAN overlap render (actual update||render boundary):
//!   the mailbox producer (update) vs consumer (prepare) stay excluded under
//!   phase_mutex instead — phase ownership no longer spans prepare+render.
//! - GPU consumability of a published slot ends at the consuming render's
//!   return or — when no render consumes it — at the START of the next
//!   prepareFrame: a repeated prepare discards the pending frame BEFORE
//!   GpuRetire.begin/flush (Scene.prepareFrame clears its consumable flag
//!   first), and that flush may tear down the borrowed handles the old
//!   front references. It also ends at Scene.deinit. The new publish at
//!   the end of prepare re-associates frame_id/retire_epoch.
//! - CPU storage outlives consumability (retained capacity, reused as the
//!   next back scratch), but that carries NO snapshot-lifetime or pinning
//!   guarantee: contents may be reset and rebuilt by any prepare. Getter
//!   pointers (Scene.preparedDraws and any slice taken from it) are therefore
//!   consumable only while frame_prepared is set or during the render call
//!   consuming the frame — never across a prepare boundary. No generic SPSC
//!   mailbox/leases/state machines: this is the minimal 2-slot lifetime for
//!   the variable-list boundary; fixed-size state (frame_snapshot/light
//!   mailbox) is untouched.

const std = @import("std");
const render_queue = @import("render_queue.zig");
const snapshot_mod = @import("snapshot.zig");
const retire_mod = @import("gpu_retire.zig");
const outline_pass = @import("../passes/outline_pass.zig");
const shadow_pass = @import("../passes/shadow_pass.zig");

pub const RenderQueues = render_queue.RenderQueues;
pub const SkinStorage = render_queue.SkinStorage;
pub const OutlineDrawItem = outline_pass.OutlineDrawItem;
pub const PreparedShadowDraws = shadow_pass.ShadowPass.PreparedShadowDraws;
pub const Epoch = retire_mod.Epoch;

/// One coherent prepared frame: the prepared mesh draw lists (view queues,
// outline, shadow) the render phase consumes. Built whole into the BACK
// slot, then published by index flip; render touches it only through const
// references while it is the consumable front (see header).
//
// Stage-2 increment B payload identity: every instanced batch / shadow item /
// outline item carries `source_uid` (stable `Mesh.uid`) + `source_mesh`
// (mesh-list index at build time). Game-built (`.build_view`) payloads hold
// provisional `instance_buffer`/`visible_instance_count` (plus shadow
// `world_aabb`/`max_dim`, outline `world_center`) until the latch
// `patchInstanceRefs` finalizes them from the post-latch `instance_render`
// (fail-closed zero on uid mismatch or stale publish); fallback
// (`.published`) payloads are final at build time.
pub const FrameDrawSlot = struct {
    primary: RenderQueues = .{},
    views: [snapshot_mod.MAX_CAMERAS]RenderQueues = [_]RenderQueues{.{}} ** snapshot_mod.MAX_CAMERAS,
    outline_items: std.ArrayListUnmanaged(OutlineDrawItem) = .empty,
    outline_skins: SkinStorage = .empty,
    shadow: PreparedShadowDraws = .{},
    /// Scene.frame_id that built this slot.
    frame_id: u64 = 0,
    /// GpuRetire epoch opened by the prepareFrame that built this slot.
    retire_epoch: Epoch = 0,

    /// Clear lengths for reuse, retaining all capacity. Covers EVERY list —
    /// including disabled views and disabled shadow bins — so a skipped path
    /// can never resurface the other slot's prior frame.
    pub fn reset(self: *FrameDrawSlot) void {
        self.primary.reset();
        for (&self.views) |*q| q.reset();
        self.outline_items.clearRetainingCapacity();
        self.outline_skins.clearRetainingCapacity();
        self.shadow.reset();
        self.frame_id = 0;
        self.retire_epoch = 0;
    }

    pub fn deinit(self: *FrameDrawSlot, allocator: std.mem.Allocator) void {
        self.primary.deinit(allocator);
        for (&self.views) |*q| q.deinit(allocator);
        self.outline_items.deinit(allocator);
        self.outline_skins.deinit(allocator);
        self.shadow.deinit(allocator);
    }
};

/// The two retained slots. `front` is the published consumable slot;
/// `1 - front` is the build scratch. Flipping `front` IS the publish — the
/// lists themselves never move.
pub const FrameDraws = struct {
    slots: [2]FrameDrawSlot = .{ .{}, .{} },
    front: usize = 0,

    pub fn backIndex(self: *const FrameDraws) usize {
        return 1 - self.front;
    }

    pub fn backSlot(self: *FrameDraws) *FrameDrawSlot {
        return &self.slots[1 - self.front];
    }

    pub fn publish(self: *FrameDraws, back_idx: usize) void {
        std.debug.assert(back_idx == 1 - self.front);
        self.front = back_idx;
    }

    pub fn deinit(self: *FrameDraws, allocator: std.mem.Allocator) void {
        for (&self.slots) |*s| s.deinit(allocator);
    }
};
