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

/// Prepares the scene frame on the context thread.
/// Orchestrates upload draining, snapshot latching, instance staging,
/// debug wireframe uploads, queue building, and triple-buffer publication.
pub fn prepareFrame(scene: anytype) void {
    // Владение фазой: prepare выполняется на context-потоке
    // ПОСЛЕДОВАТЕЛЬНО с render (один поток, next prepare NEVER
    // concurrent with render); update-поток в это время ИСКЛЮЧЁН
    // (phase_mutex update-vs-prepare), а во время render — НЕТ: update
    // CAN overlap render. Внутри prepare — только контекстные операции:
    // flushPendingGpuUploads, стейджинг инстансов, shadow prepare,
    // построение очередей + CPU-capture (debug/particles/UI) и их GPU
    // upload. Draw-фаза ниже читает только render-owned снимки.
    gpu_thread.assertOnContextThread();
    // Wave-31 lease claim (concurrent-build prerequisite): prepare
    // resolves AND holds its working slot through the lease protocol for
    // the duration of the whole prepare. Fallback (no fresh build)
    // claims any free slot (`claimBack` — sequentially exactly the old
    // `backIndex`); the latch path claims the handoff slot the game
    // published (`claimSlot(build_slot)` — sequentially exactly the old
    // `build_slot == backIndex` assert, now fail-closed instead of
    // debug-only). While the claim is held a concurrent game
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
    // (`last_latched_seq` unstamped) for the next prepare.
    // Acquire: pairs with the publish release-store, so the staged build
    // payload is visible on the latch path below.
    const have_build = scene.build_seq.load(.acquire) != scene.last_latched_seq.load(.monotonic);
    const back_idx: usize = if (have_build) blk: {
        const wanted = scene.build_slot.load(.acquire);
        scene.draws.claimSlot(wanted) catch {
            return;
        };
        break :blk wanted;
    } else blk: {
        break :blk scene.draws.claimBack() orelse {
            return;
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
    scene.stats.update_ms = scene.pending_update_ms;
    scene.stats.physics_ms = scene.pending_physics_ms;
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
        // The handoff slot is already held via `claimSlot(build_slot)`:
        // no intervening publish could have moved `front` (only prepare
        // flips it, and the latch has not run yet), and no concurrent
        // claim could have taken it (claims skip WRITING slots).
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

    // Debug line upload (GPU): the frame's single updateBuffer, once per
    // prepare no matter how many PIP views render below. Samples follow
    // the same MSAA policy as render (same snapshot inputs, same
    // result). Headless the whole upload is skipped (the CPU capture
    // above already ran for tests) — no sg.* without a context.
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
        // it, never the live `build_snapshot`/`frame_snapshot` — and
        // the staged slot build_stats below likewise (wave 31 second
        // slice): the latch merges the slot copy, never the live
        // `build_stats` accumulator.
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
        // Deferred stats merge (stage-2B, slot-staged since wave 31
        // second slice): the game-side queue build accumulated into the
        // live `build_stats` accumulator and froze a copy into this
        // slot; fold the SLOT copy's queue counters into the
        // context-owned self.stats (already reset above, so upload
        // tallies/prepare_ms/update_ms are preserved) and clear both
        // copies for the next tick. Sourcing the merge from the slot —
        // never the shared field — is what lets a concurrent game-side
        // accumulation race nothing here.
        scene.stats.mergeFrom(&back.build_stats);
        back.build_stats = .{};
        scene.build_stats = .{};
        // Context-side stamp only (the producer never touches this word).
        scene.last_latched_seq.store(scene.build_seq.load(.monotonic), .monotonic);
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
        scene.draws.cancelClaim(back_idx) catch {};
        return;
    };
    scene.frame_prepared = true;
}
