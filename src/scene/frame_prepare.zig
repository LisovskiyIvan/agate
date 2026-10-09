const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const gpu_thread = @import("../gpu_thread.zig");
const upload_meter = @import("../gpu_upload_meter.zig");
const scene_instance_staging = @import("instance_staging.zig");
const SceneFrameSnapshot = @import("snapshot.zig").SceneFrameSnapshot;

/// A claimed staged producer frame. `beginPrepare` claims the latest fully
/// built handoff and holds its slot for the whole prepare; `finishPrepare`
/// consumes only that slot plus context-owned state. Single staged path:
/// null means nothing fresh — never a live read.
pub const PrepareClaim = struct {
    token_id: u64,
    back_idx: usize,
    build_seq: usize,
    /// Frozen host bytes staged by the producer (`BuildClaim.stageHostBytes`
    /// into the claimed slot's `host_bytes`): the context reads this copy
    /// instead of live host state. Valid from `beginPrepare` until the
    /// paired `finishPrepare`/`cancelPrepare` releases the slot claim.
    host_bytes: []const u8 = &.{},
};

/// Begins staged preparation on the context thread. Claims the latest fully
/// built producer handoff (`claimLatestHandoff`, always requiring a full
/// scene build) and holds its slot for the whole prepare. Returns null when
/// no fresh producer frame is ready — never a live fallback. While the claim
/// is held a concurrent game `tryClaimBuildSlot` skips this slot (or
/// saturates, counted) and `pin` refuses it (`SlotBusy`, counted).
pub fn beginPrepare(scene: anytype) ?PrepareClaim {
    gpu_thread.assertOnContextThread();
    if (scene.prepare_claim_active) return null;
    const handoff = scene.draws.claimLatestHandoff(
        &scene.build_slot,
        &scene.build_seq,
        scene.last_latched_seq.load(.monotonic),
        true,
    ) catch return null;
    const h = handoff orelse return null;
    const build_seq: usize = h.seq;
    const back_idx: usize = h.slot;
    const back = scene.draws.slotAt(back_idx);
    scene.frame_prepared = false;
    scene.reuse_streak = 0;
    scene.retire_epoch = scene.gpu_retire.begin();
    const keep_prepare_ms = scene.stats.prepare_ms;
    scene.stats = .{};
    scene.stats.update_ms = scene.pending_update_ms.load(.acquire);
    scene.stats.physics_ms = scene.pending_physics_ms.load(.acquire);
    scene.stats.prepare_ms = keep_prepare_ms;
    _ = upload_meter.takeAndReset();
    scene.frame_id +%= 1;

    const upload_budget_per_frame = @TypeOf(scene.*).upload_budget_per_frame;
    const upload_byte_budget_per_frame = @TypeOf(scene.*).upload_byte_budget_per_frame;

    if (scene.uploads) |*q| {
        scene.frame_uploads = q.drainCountedBudget(upload_budget_per_frame, upload_byte_budget_per_frame);
    } else {
        scene.frame_uploads = .{};
    }
    scene.stats.uploaded_textures_frame = std.math.cast(u32, scene.frame_uploads.count) orelse std.math.maxInt(u32);
    scene.stats.uploaded_bytes_frame = scene.frame_uploads.bytes;
    scene.flush_in_prepare = true;
    @import("upload_packets.zig").flushSlotUploads(scene, back);
    scene.flush_in_prepare = false;

    scene.particles.latchSlotFrame(scene.sim_allocator, back.particle_draws.items);
    scene.physics.latchSlotDebug(scene.allocator, back.physics_lines.items, back.physics_visible);

    scene.frame_snapshot = back.snapshot;
    const staged: *const SceneFrameSnapshot = &back.snapshot;

    scene.captureUiFrame(staged, back);

    std.debug.assert(!scene.prepare_claim_active);
    scene.prepare_claim_generation +%= 1;
    if (scene.prepare_claim_generation == 0) scene.prepare_claim_generation = 1;
    scene.prepare_claim_active = true;
    scene.prepare_claim_slot = back_idx;
    scene.prepare_claim_seq = build_seq;
    return .{
        .token_id = scene.prepare_claim_generation,
        .back_idx = back_idx,
        .build_seq = build_seq,
        .host_bytes = back.host_bytes.items,
    };
}

/// Contract-violation recovery for `finishPrepare`/`cancelPrepare`.
///
/// A mismatched claim means the caller violated the pairing contract
/// (stale token, double finish/cancel, or lost token after a fresh begin).
/// The historical `assert(false)` panicked in debug and — worse — silently
/// WEDGED ReleaseFast: an unpaired begin leaves `prepare_claim_active`
/// stuck, every future `beginPrepare` returns null at its active-claim
/// guard, and the pipeline degrades to reuse-forever with no diagnostic.
///
/// Instead: log the violation and release the ACTIVE claim defensively
/// (cancel semantics — slot lease released, begin-side retire epoch
/// completed, `frame_prepared` cleared). The frame is lost either way;
/// this only chooses a live pipeline with a visible error over a silent
/// permanent wedge. A stale claim against NO active claim is a no-op log
/// line (idempotent double release).
fn recoverMismatchedClaim(scene: anytype, comptime action: []const u8, claim: PrepareClaim) void {
    if (!scene.prepare_claim_active) {
        std.log.warn("agate frame_prepare: {s} with stale claim (token {d}, build_seq {d}) but no active claim — already finished or cancelled?", .{ action, claim.token_id, claim.build_seq });
        return;
    }
    std.log.warn("agate frame_prepare: {s} claim mismatch (got token {d}, active token {d}) — releasing the active claim to keep the frame pipeline running", .{ action, claim.token_id, scene.prepare_claim_generation });
    const slot = scene.prepare_claim_slot;
    const seq = scene.prepare_claim_seq;
    scene.prepare_claim_active = false;
    scene.draws.cancelHandoffClaim(slot, seq, &scene.build_slot, &scene.build_seq) catch {};
    scene.frame_prepared = false;
    scene.gpu_retire.complete(scene.retire_epoch);
}

/// Finishes staged preparation using the exact slot claimed by `beginPrepare`.
/// The producer lock may be released before this call: a staged claim
/// consumes only slot-owned and context-owned state, never live state.
pub fn finishPrepare(scene: anytype, claim: PrepareClaim) void {
    gpu_thread.assertOnContextThread();
    if (!matchesPrepareClaim(scene, claim)) {
        recoverMismatchedClaim(scene, "finishPrepare", claim);
        return;
    }
    scene.prepare_claim_active = false;
    const back_idx = claim.back_idx;
    const build_seq = claim.build_seq;
    const back = scene.draws.slotAt(back_idx);

    const staged: *const SceneFrameSnapshot = &back.snapshot;
    if (sg.isvalid()) {
        const upload_samples = @import("postfx_stack.zig").PostFXStack.targetSampleCount(staged.msaa_sample_count);
        scene.physics.uploadDebug(scene.allocator, upload_samples, .RGBA16F);
    }

    const is_gpu_init = (scene.default_white_texture.view.id != 0);
    back.frame_id = scene.frame_id;
    back.retire_epoch = scene.retire_epoch;
    if (is_gpu_init) {
        if (back.snapshot.has_camera) {
            // Retained prior-generation payloads for velocity pairing: every
            // slot's staged frame, records, and matrix scratch. The latch
            // only reads these (never live meshes); a missing or mismatched
            // entry falls back to zero motion per record.
            var prev_sources: [scene.draws.slots.len]scene_instance_staging.PrevFrameSource = undefined;
            for (0..scene.draws.slots.len) |i| {
                const slot = &scene.draws.slots[i];
                prev_sources[i] = .{
                    .frame_id = slot.frame_id,
                    .records = slot.staged_instances.items,
                    .scratch = slot.primary.instance_matrices.items,
                };
            }
            scene_instance_staging.stageInstancesLatch(.{
                .allocator = scene.allocator,
                .frame_id = scene.frame_id,
                .retire_queue = &scene.gpu_retire,
                .prev_frames = &prev_sources,
            }, back.staged_instances.items, &back.primary.instance_matrices);
        }
    }
    scene.patchInstanceRefs(back);
    scene.stats.mergeFrom(&back.build_stats);
    back.build_stats = .{};

    scene.draws.tryPublish(back_idx) catch {
        scene.draws.cancelHandoffClaim(back_idx, build_seq, &scene.build_slot, &scene.build_seq) catch {};
        scene.gpu_retire.complete(scene.retire_epoch);
        return;
    };
    scene.last_latched_seq.store(build_seq, .monotonic);
    scene.frame_prepared = true;
}

/// Cancels an in-progress staged prepare and releases its slot lease. The
/// begin-side retire epoch is completed because no render will consume it.
pub fn cancelPrepare(scene: anytype, claim: PrepareClaim) void {
    gpu_thread.assertOnContextThread();
    if (!matchesPrepareClaim(scene, claim)) {
        recoverMismatchedClaim(scene, "cancelPrepare", claim);
        return;
    }
    scene.prepare_claim_active = false;
    scene.draws.cancelHandoffClaim(claim.back_idx, claim.build_seq, &scene.build_slot, &scene.build_seq) catch {};
    scene.frame_prepared = false;
    scene.gpu_retire.complete(scene.retire_epoch);
}

fn matchesPrepareClaim(scene: anytype, claim: PrepareClaim) bool {
    return scene.prepare_claim_active and
        scene.prepare_claim_generation == claim.token_id and
        scene.prepare_claim_slot == claim.back_idx and
        scene.prepare_claim_seq == claim.build_seq;
}
