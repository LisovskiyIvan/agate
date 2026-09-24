const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const sapp = sokol.app;
const gpu_thread = @import("../gpu_thread.zig");
const upload_meter = @import("../gpu_upload_meter.zig");
const jobs = @import("../jobs.zig");
const scene_msaa = @import("msaa.zig");
const scene_instance_staging = @import("instance_staging.zig");
const SceneFrameSnapshot = @import("snapshot.zig").SceneFrameSnapshot;

/// A claimed producer frame split at the update-vs-prepare ownership
/// boundary. `beginPrepare` performs all operations that can touch live
/// producer state; after it returns, `finishPrepare` consumes only this
/// claimed slot and context-owned state.
pub const PrepareClaim = struct {
    token_id: u64,
    back_idx: usize,
    build_seq: u64,
    have_build: bool,
    has_handoff: bool,
};

/// Begins preparation on the context thread. The caller must exclude the
/// producer from live-scene mutation for this call. `allow_fallback` is for
/// sequential/legacy callers only: the inline path captures live state and
/// must stay inside the same exclusion window. Concurrent callers pass false;
/// if there is no new producer frame this is a no-op, never a live read.
pub fn beginPrepare(scene: anytype, allow_fallback: bool) ?PrepareClaim {
    // Владение фазой: prepare выполняется на context-потоке
    // ПОСЛЕДОВАТЕЛЬНО с render (один поток, next prepare NEVER
    // concurrent with render); update-поток в это время ИСКЛЮЧЁН
    // (phase_mutex update-vs-prepare), а во время render — НЕТ: update
    // CAN overlap render. Внутри prepare — только контекстные операции:
    // flushPendingGpuUploads, стейджинг инстансов, shadow prepare,
    // построение очередей + CPU-capture (debug/particles/UI) и их GPU
    // upload. Draw-фаза ниже читает только render-owned снимки.
    gpu_thread.assertOnContextThread();
    if (scene.prepare_claim_active) return null;
    // Wave-31 lease claim (concurrent-build prerequisite): prepare
    // resolves AND holds its working slot through the lease protocol for
    // the duration of the whole prepare. Fallback (no fresh build)
    // claims any free slot (`claimBack` — sequentially exactly the old
    // `backIndex`); the latch path claims the handoff slot the game
    // published (`claimLatestHandoff` — the `(build_slot, build_seq)` pair
    // plus the slot stamp are read under one lease-mutex critical section,
    // so a publish landing between separate loads can never pair an old
    // generation with a newer slot; UI-only handoffs are left pending when
    // `allow_fallback == false`). While the claim is held a concurrent game
    // `tryClaimBuildSlot` skips this slot (or saturates, counted) and
    // `pin` refuses it (`SlotBusy`, counted): no concurrent
    // `BuildClaim.publish`/`stageUi` can target the slot prepare is
    // consuming, and prepare never resets a game-held slot. The publish
    // flip itself stays context-owned exactly as before — now via the
    // locked `tryPublish` at the end. Contention degrades to a counted
    // skip, never a wedge: the counters are bumped inside the lease
    // calls, nothing is consumed yet (no retire/stats/frame side effects
    // below have run). The skip discards nothing: a pending frame from
    // an earlier prepare stays consumable (`frame_prepared` untouched —
    // still the freshest prepared); with nothing pending it stays false,
    // so `render`'s fallback drops the present instead of mislabeling a
    // stale front. On the latch path the build stays fresh
    // (`last_latched_seq` unstamped until the finish publishes) for the
    // next prepare.
    // Visibility: publication and this claim meet under the lease mutex
    // (releaseHandoffWithSeq vs claimLatestHandoff), so the staged build
    // payload is visible on the latch path below without a separate
    // acquire edge on the seq words.
    const handoff = scene.draws.claimLatestHandoff(
        &scene.build_slot,
        &scene.build_seq,
        scene.last_latched_seq.load(.monotonic),
        !allow_fallback,
    ) catch return null;
    const has_handoff = handoff != null;
    const have_build = if (handoff) |h| h.has_scene_build else false;
    if (!has_handoff and !allow_fallback) return null;
    const build_seq: u64 = if (handoff) |h| h.seq else 0;
    const back_idx: usize = if (handoff) |h| h.slot else blk: {
        break :blk scene.draws.claimBack() orelse {
            return null;
        };
    };
    const back = scene.draws.slotAt(back_idx);
    // P7: repeated prepare discards the previous pending frame BEFORE
    // GpuRetire.begin/flush below: its borrowed handles may be torn down
    // by the flush, so preparedDraws() payloads from that frame lose GPU
    // consumability from this point (retained CPU storage may be reused
    // as back scratch). The new publish at the end re-associates
    // frame_id/retire_epoch.
    scene.frame_prepared = false;
    // A successful prepare ends any reuse streak: from here on the
    // leading flush below drains retire/upload intents again, so the
    // streak × destroy-rate pileup is bounded by the streak length
    // (observable via reuseStreak()/pendingRetires(), capped by
    // GpuRetireQueue.pending_cap).
    scene.reuse_streak = 0;
    // Начало кадра (P3): новый epoch ретенции. flush ниже (внутри
    // flushPendingGpuUploads) уничтожит только завершённые эпохи —
    // запись текущего кадра ждёт его конца. begin заодно закрывает
    // предыдущий незакрытый epoch (discarded-pending контракт выше).
    scene.retire_epoch = scene.gpu_retire.begin();
    // update_ms приходит staged (recordUpdateTime -> pending_update_ms,
    // игровой поток), prepare_ms пишет app на этом же context-потоке
    // вокруг prepareFrame (см. main): оба сохраняются через сброс, всё
    // остальное обнуляется под новый кадр. Прямых stats-записей с
    // update-потока нет — stats читает render конкурентно с update.
    const keep_prepare_ms = scene.stats.prepare_ms;
    scene.stats = .{};
    scene.stats.update_ms = scene.pending_update_ms.load(.acquire);
    scene.stats.physics_ms = scene.pending_physics_ms.load(.acquire);
    scene.stats.prepare_ms = keep_prepare_ms;
    // Сброс счётчика динамических обновлений на начало кадра: всё, что
    // запишут flushPendingGpuUploads, стейджинг инстансов и UI/debug
    // upload'ы ниже — всё внутри prepare — плюс clear-append'ы внутри
    // render, сложится в stats.updated_bytes_frame перед
    // Profiler.recordFrame. На троттлинг текстур не влияет.
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
    // The render pipeline commits this frame's buffer at the end of
    // render(); a standalone flushPendingGpuUploads (quiesced-context
    // completion) has no following render and must commit itself.
    scene.flush_in_prepare = true;
    scene.flushPendingGpuUploads();
    scene.flush_in_prepare = false;

    // Particle prepared frame: capture the retained plain frame here,
    // after the flush above and BEFORE the update/render unlock below.
    // Reads live systems for the LAST time this frame; the draw below
    // sees only the capture. Stage 1: when the game side built a fresh
    // frame (`have_build`, latched at the top under the lease claim),
    // consume the slot-frozen capture (`latchSlotFrame` reads the claimed
    // slot's `particle_draws`, never the shared `build_frame` — wave 32
    // freeze-then-latch, so a colliding game-thread build cannot tear
    // the record); otherwise the historical inline capture (apps without
    // `buildPreparedFrame` are unchanged).
    if (have_build) {
        scene.particles.latchSlotFrame(scene.allocator, back.particle_draws.items);
    } else {
        scene.particles.captureFrame(scene.allocator);
    }

    // Physics debug wireframe capture (CPU): world.appendDebugLines runs
    // HERE in prepare — never inside render. The draw below reads only
    // the capture (prepared_visible/prepared_lines). Same stage 1 shape
    // as particles: consume the slot-frozen capture on a fresh build
    // (`latchSlotDebug` reads the claimed slot's `physics_lines`/
    // `physics_visible`, never the shared staging store), else capture
    // inline.
    if (have_build) {
        scene.physics.latchSlotDebug(scene.allocator, back.physics_lines.items, back.physics_visible);
    } else {
        scene.physics.captureDebug(scene.allocator);
    }

    // Snapshot generations (wave 27, slot-owned): the consumed slot
    // carries the staged copy — prepare reads the slot it is consuming,
    // never the live working copies past this latch point. With a fresh
    // game build the slot already stages the EXACT build generation
    // (frozen by `buildPreparedFrame`; staged wins over a post-build
    // `build_snapshot` mutation) and the latch mirrors it into
    // `frame_snapshot` for compatibility — a newer mailbox publication
    // after the build stays queued for the next build or fallback and
    // never mixes culling/camera generations into these queues. Without
    // a build the historical takeLatest-else-pack runs into
    // `frame_snapshot`; the fallback branch below stages it into the
    // claimed slot before consuming it.
    // (`back` is the lease-held slot claimed at the top: `front` moves
    // only on the publish at the end, so the claim stays stable across
    // the latch, and the staged UI packet lives in this same slot.)
    if (have_build) {
        // The handoff slot is already held via `claimLatestHandoff`:
        // no intervening publish could have moved `front` (only prepare
        // flips it, and the latch has not run yet; a second prepare is
        // token-blocked), and no concurrent claim could have taken it
        // (claims skip WRITING slots).
        scene.frame_snapshot = back.snapshot;
    } else {
        var snap = scene.frame_snapshot;
        if (scene.frame_handoff.takeLatest(&snap)) {
            scene.frame_snapshot = snap;
        } else {
            const cur_w = sapp.width();
            const cur_h = sapp.height();
            const aspect = if (cur_h > 0) @as(f32, @floatFromInt(cur_w)) / @as(f32, @floatFromInt(cur_h)) else 1.0;
            scene.frame_snapshot = scene.packFrameSnapshot(aspect, cur_w, cur_h);
        }
    }
    // The staged snapshot prepare consumes from here on: the back slot's
    // frozen copy on the build path, the just-latched working copy on the
    // fallback path (staged into the slot by the queue branch below —
    // same bytes, so the UI latch and the MSAA sample gate already see
    // the prepared generation either way).
    const staged: *const SceneFrameSnapshot = if (have_build) &back.snapshot else &scene.frame_snapshot;

    // UI packet latch BEFORE the queue branches below: the fallback
    // branch resets the claimed slot (wiping a staged UI packet), while
    // the latch path consumes it — so the packet must land in `ui_frame`
    // first. Order vs debug/instance uploads is irrelevant (disjoint
    // buffers; the meter sums identically).
    scene.captureUiFrame(staged, back);

    std.debug.assert(!scene.prepare_claim_active);
    scene.prepare_claim_generation +%= 1;
    if (scene.prepare_claim_generation == 0) scene.prepare_claim_generation = 1;
    scene.prepare_claim_active = true;
    scene.prepare_claim_slot = back_idx;
    scene.prepare_claim_seq = build_seq;
    scene.prepare_claim_have_build = have_build;
    scene.prepare_claim_has_handoff = has_handoff;
    return .{
        .token_id = scene.prepare_claim_generation,
        .back_idx = back_idx,
        .build_seq = build_seq,
        .have_build = have_build,
        .has_handoff = has_handoff,
    };
}

/// Finishes preparation using the exact slot claimed by `beginPrepare`.
/// The caller may release producer/live-state exclusion before this function
/// only when `claim.have_build` is true. A fallback claim reads live meshes
/// and must remain serialized for the whole call.
pub fn finishPrepare(scene: anytype, claim: PrepareClaim) void {
    gpu_thread.assertOnContextThread();
    if (!matchesPrepareClaim(scene, claim)) {
        std.debug.assert(false);
        return;
    }
    // Consume the one-shot token before any side effects: a duplicate finish
    // cannot re-run uploads/merges or double-publish the claimed slot.
    scene.prepare_claim_active = false;
    const back_idx = claim.back_idx;
    const build_seq = claim.build_seq;
    const have_build = claim.have_build;
    const back = scene.draws.slotAt(back_idx);

    // Debug line upload (GPU): the frame's single updateBuffer, once per
    // prepare no matter how many PIP views render below. This consumes the
    // render-owned capture created in beginPrepare (or the serialized legacy
    // capture) and does not read producer-owned physics state.
    const staged: *const SceneFrameSnapshot = if (have_build) &back.snapshot else &scene.frame_snapshot;
    if (sg.isvalid()) {
        const upload_samples = scene_msaa.effectiveSampleCount(staged.msaa_sample_count, .{
            .post_enabled = staged.post_process.enabled,
            .formats_msaa_capable = scene_msaa.mainTargetFormatsMsaaCapable(),
            .backend = sg.queryBackend(),
        });
        scene.physics.uploadDebug(scene.allocator, upload_samples);
    }

    const is_gpu_init = (scene.default_white_texture.view.id != 0);
    // P7: build the CLAIMED slot in place, then publish with one index
    // flip at the end. The front slot is untouched during the build:
    // allocator failure in the claimed slot corrupts nothing consumable.
    // (The claim is held since the top, across the UI latch above;
    // `front` has not moved since, so `back` still addresses the build
    // scratch, and no concurrent claim could have taken it.)
    if (have_build) {
        // Stage-2 latch: the game-side build already committed the last
        // publish, reset this back slot, staged the instance scratch +
        // previews + staged records + build_views, and
        // built the queue/shadow/outline payload with `.build_view`
        // (provisional handles). Reset-before-consume would erase the
        // build, and rebuilding would unfreeze the sets — so do NOT call
        // buildQueuesInto here. Only stamp the prepare-owned frame/epoch,
        // run the GPU halves over the slot records + scratch, finalize the
        // provisional handles with patchInstanceRefs, then merge the
        // slot-staged build_stats copy. Meshes with no record (OOM-skipped,
        // post-build meshes) keep their previous complete
        // `instance_render` and
        // patch to invisible — no partial publish. A mesh-list mutation
        // between build and latch stages through here untouched (the
        // latch reads no live list) and is caught by the game-side
        // commit guard at the next build; the patch finalizes whatever
        // the records published (see patch docs).
        //
        // Remaining live coupling (next slice): the update-vs-prepare
        // mutex still guards the whole handoff. The latch itself is now
        // live-touch-free (GPU halves over the slot-owned staged records
        // + scratch, outcomes mirrored into the records, payloads
        // finalized from the mirrors): the game-side commit at the next
        // build applies the mirrors to the live meshes. The staged slot
        // snapshot above is already frozen (wave 27) — the latch reads
        // it, never the live `build_snapshot`/`frame_snapshot`. Queue
        // stats have no shared accumulator at all (wave 39): the producer
        // wrote this slot's `build_stats` directly and the merge below
        // folds that immutable copy into context-owned `stats`.
        back.frame_id = scene.frame_id;
        back.retire_epoch = scene.retire_epoch;
        if (is_gpu_init) {
            // Same position/gate as the historical pre-stage: the GPU half
            // must run before the patch finalizes handles (and would have
            // run before the shadow snapshot historically). Same retire
            // queue and eye source shape (the eye itself was consumed at
            // build time for the transparent sort). The latch consumes
            // the slot-owned staged records frozen by buildPreparedFrame
            // plus the slot scratch — no live mesh reads, no live mesh
            // writes; a mesh-list mutation between build and latch is
            // caught later by the game-side commit guard, never here.
            if (back.snapshot.has_camera) {
                scene_instance_staging.stageInstancesLatch(.{
                    .allocator = scene.allocator,
                    .frame_id = scene.frame_id,
                    .retire_queue = &scene.gpu_retire,
                }, back.staged_instances.items, &back.primary.instance_matrices);
            }
        }
        scene.patchInstanceRefs(back);
        // Deferred stats merge: fold this immutable slot's queue counters
        // into context-owned stats and clear the consumed slot for reuse.
        scene.stats.mergeFrom(&back.build_stats);
        back.build_stats = .{};
    } else {
        // Inline fallback (no fresh build): reset first — every list
        // plus the staged snapshot, so a skipped path can never resurface
        // the other slot's prior frame — then stage fully.
        back.reset();
        back.frame_id = scene.frame_id;
        back.retire_epoch = scene.retire_epoch;
        // Wave 27: freeze the just-latched working copy into the consumed
        // slot before anything reads it below (plain copy, same bytes the
        // UI latch and the MSAA gate above already consumed via `staged`).
        back.snapshot = scene.frame_snapshot;
        if (is_gpu_init) {
            // Pre-stage instance data before the shadow pass: ShadowPass.prepare
            // snapshots the published render state (bounds/buffer/count),
            // so staging must run first or shadows lag one frame. Same scratch
            // and eye the first view queue would use; the frame guard keeps it
            // once per frame, shared by all view queues. Grown-away old
            // buffers retire into the epoch queue (P5), never destroyed inline.
            // Staging scratch is the back primary's list (view builds with
            // instances_prepared never retry mid-frame — failure coherence).
            if (back.snapshot.has_camera) {
                scene_instance_staging.stageInstances(.{
                    .allocator = scene.allocator,
                    .instance_matrices = &back.primary.instance_matrices,
                    .thread_pool = jobs.global,
                    .frame_id = scene.frame_id,
                    .eye = back.snapshot.primary_cam.eye,
                    .retire_queue = &scene.gpu_retire,
                }, scene.meshes.items);
            }
        }
    }

    // Shared builder — fallback only (stage-2B): when no fresh game build
    // exists, outline + shadow + view queues build here with exactly
    // today's values (staged slot snapshot, frame_id as cache_key,
    // &self.stats, snapshot eye, resolved sky/ibl, true, `.published`) —
    // bit-identical to the old inline path. When have_build the payload
    // was already built game-side (`.build_view` + patch above);
    // rebuilding would unfreeze the sets.
    if (!have_build) {
        const sky_tex = back.snapshot.sky_texture orelse scene.sky.texture;
        const ibl_int = back.snapshot.ibl_intensity;
        scene.buildQueuesInto(back, .{
            .snap = &back.snapshot,
            .cache_key = scene.frame_id,
            .stats = &scene.stats,
            .eye = back.snapshot.primary_cam.eye,
            .sky_texture = sky_tex,
            .ibl_intensity = ibl_int,
            .instances_prepared = true,
            .instance_source = .published,
        });
    }

    // P7 publish: one index flip, no list copies. Newest wins — a repeated
    // prepare's back overwrote nothing consumable until this point.
    // (UI already latched above, before the queue branches.)
    // Context-owned flip exactly as before, now via the locked publish:
    // this clears the claim taken at the top and hands the slot to the
    // consumer in one step. `pin` refuses WRITING slots, so no
    // concurrent pin could have landed on our slot and the publish
    // cannot refuse under correct API use; the defensive branch releases
    // the claim and skips instead of wedging (a missed release degrades
    // to a counted skip — the refusal itself is counted inside
    // `tryPublish`).
    scene.draws.tryPublish(back_idx) catch {
        if (claim.has_handoff) {
            scene.draws.cancelHandoffClaim(back_idx, build_seq, &scene.build_slot, &scene.build_seq) catch {};
        } else {
            scene.draws.cancelClaim(back_idx) catch {};
        }
        // No render will consume this frame: close the epoch begun in
        // beginPrepare here (same pairing as cancelPrepare), not just via
        // the next begin's auto-close.
        scene.gpu_retire.complete(scene.retire_epoch);
        return;
    };
    if (claim.has_handoff) {
        // Consume the exact handoff only after successful publication. A
        // newer producer generation may already be pending and must remain
        // visible to the next begin.
        scene.last_latched_seq.store(build_seq, .monotonic);
    }
    scene.frame_prepared = true;
}

/// Cancels an in-progress split prepare and releases its slot lease. The
/// begin-side retire epoch is completed because no render will consume it.
pub fn cancelPrepare(scene: anytype, claim: PrepareClaim) void {
    gpu_thread.assertOnContextThread();
    if (!matchesPrepareClaim(scene, claim)) {
        std.debug.assert(false);
        return;
    }
    scene.prepare_claim_active = false;
    if (claim.has_handoff) {
        scene.draws.cancelHandoffClaim(claim.back_idx, claim.build_seq, &scene.build_slot, &scene.build_seq) catch {};
    } else {
        scene.draws.cancelClaim(claim.back_idx) catch {};
    }
    scene.frame_prepared = false;
    scene.gpu_retire.complete(scene.retire_epoch);
}

fn matchesPrepareClaim(scene: anytype, claim: PrepareClaim) bool {
    return scene.prepare_claim_active and
        scene.prepare_claim_generation == claim.token_id and
        scene.prepare_claim_slot == claim.back_idx and
        scene.prepare_claim_seq == claim.build_seq and
        scene.prepare_claim_have_build == claim.have_build and
        scene.prepare_claim_has_handoff == claim.has_handoff;
}

/// Serialized compatibility entry point. Applications that overlap producer
/// updates with context preparation should use `beginPrepare(..., false)`,
/// release their producer lock, then call `finishPrepare`.
pub fn prepareFrame(scene: anytype) void {
    const claim = beginPrepare(scene, true) orelse return;
    finishPrepare(scene, claim);
}
