const std = @import("std");
const sokol = @import("sokol");
const sapp = sokol.app;
const math = @import("math");
const Vec3 = math.Vec3;
const jobs = @import("../jobs.zig");
const scene_instance_staging = @import("instance_staging.zig");

/// Shared build core behind `buildPreparedFrame` and `BuildClaim.build`:
/// the exact historical build body targeted at the claimed `slot` under
/// the reserved generation `seq` (preview stamps, record freeze, build
/// cache key). Commits NOTHING global: `build_seq`/`build_slot` are
/// stamped by `BuildClaim.publish`, so a cancelled claim leaves no
/// handoff behind.
///
/// EPOCH DISCIPLINE (wave 29): this core must never call
/// `GpuRetire.begin`/`complete`/`flush` and never passes a `retire_queue`
/// anywhere — epochs stay context-owned (prepare/render own the
/// begin/complete pairing); enforced by test.
pub fn buildIntoClaimedSlot(scene: anytype, slot: usize, seq: u64) void {
    // Deliberately NO gpu_thread assert: this runs on the game side or
    // a spawned worker. Everything below is sg-free (the commit, the CPU
    // staging half, the plain captures, the CPU queue/shadow/outline
    // build with instances_prepared=true); any sg.* here would be a bug.
    // Commit the last published latch outcomes FIRST. Resolve and hold a
    // shared read lease on the exact front: render may already be pinning it,
    // and prepare may advance front while this producer build continues.
    // The read lease prevents either case from reclaiming/resetting the slot
    // while its records are copied into the live meshes.
    {
        const front_idx = scene.draws.pinFrontReader();
        defer scene.draws.unpinReader(front_idx) catch {};
        const front = &scene.draws.slots[front_idx];
        scene_instance_staging.commitPublishedRecords(front.staged_instances.items, scene.meshes.items, front.frame_id);
    }
    const back = &scene.draws.slots[slot];
    back.reset();
    back.build_seq = seq;
    back.has_scene_build = true;
    // Producer snapshot FIRST (update-vs-prepare excluded): consume the
    // newest published tick into the producer-owned build_snapshot. When
    // nothing new was published, ALWAYS pack fresh live state — never
    // reuse the consumed render snapshot, so the build works without a
    // publish and after camera removal. Never touches frame_snapshot
    // (update may overlap render). Everything below freezes on this
    // generation (fixed-size snapshot values only; live meshes/materials/
    // culling flags stay live by design).
    if (!scene.frame_handoff.takeLatest(&scene.build_snapshot)) {
        const cur_w = sapp.width();
        const cur_h = sapp.height();
        const aspect = if (cur_h > 0) @as(f32, @floatFromInt(cur_w)) / @as(f32, @floatFromInt(cur_h)) else 1.0;
        scene.build_snapshot = scene.packFrameSnapshot(aspect, cur_w, cur_h);
    }
    // Wave 27 slot-owned snapshot: freeze this generation into the claim
    // slot by value (plain copy — Camera.name slices alias but are never
    // dereferenced by draw; GPU handles are borrowed values under P3).
    // The prepare latch consumes THIS copy (staged wins over any
    // post-build `build_snapshot` mutation); `frame_snapshot` is untouched.
    back.snapshot = scene.build_snapshot;
    // Frozen snapshot eye (zero when camera-less): the CPU staging sort
    // and the queue build both use this generation, never the live eye.
    const eye = if (scene.build_snapshot.has_camera) scene.build_snapshot.primary_cam.eye else Vec3.zero;
    scene_instance_staging.stageInstancesCpu(.{
        .allocator = scene.allocator,
        .scratch = &back.primary.instance_matrices,
        .thread_pool = jobs.global,
        .eye = eye,
    }, scene.meshes.items, seq);
    // Freeze the provisional build view for the queue build below.
    for (scene.meshes.items) |m| {
        _ = m.ensureUid();
        if (m.instance_preview.build_seq == seq) {
            m.instance_build_view = .{
                .buffer = m.instance_render.buffer,
                .capacity = m.instance_render.capacity,
                .count = m.instance_preview.count,
                .bounds = m.instance_preview.bounds,
                .hash = m.instance_preview.hash,
                .uploaded_count = m.instance_render.uploaded_count,
                .staged_frame = m.instance_render.staged_frame,
            };
        } else {
            m.instance_build_view = .{};
        }
    }
    // Freeze the slot-owned staged records for the prepare latch below
    // (same fresh-preview set as the build-view freeze above). The latch
    // and `patchInstanceRefs` consume these — never live previews.
    scene_instance_staging.freezeStagedRecords(scene.allocator, &back.staged_instances, scene.meshes.items, seq);
    scene.particles.buildCapture(scene.allocator, seq);
    scene.physics.buildDebug(scene.allocator, seq);
    // Wave 32 freeze-then-latch (lock-free-publication slices 4/5): freeze
    // the just-captured particle/physics build frames into the claimed
    // slot by value. The prepare latch consumes THESE copies
    // (`latchSlotFrame`/`latchSlotDebug`) — never the shared
    // `build_frame`/`build_lines` staging stores — so a game-thread build
    // colliding with the context-side latch cannot deliver a torn record
    // for one frame. Rides the same publish release edge as the rest of
    // the staged payload (`build_slot` then `build_seq` in
    // `BuildClaim.publish`); staged-wins on OOM (fail-closes to
    // coherent-empty, same precedent as the snapshot/stats slices above).
    // The shared stores keep their existing semantics (sequential flow
    // bit-identical, direct layer tests/tooling unaffected).
    scene.particles.stageIntoSlot(scene.allocator, &back.particle_draws);
    scene.physics.stageIntoSlot(scene.allocator, &back.physics_lines, &back.physics_visible);
    // Game-side queue/shadow/outline build (sg-free: instances_prepared).
    // The shared builder resets each view queue (including primary's
    // instance_matrices scratch) — but that scratch holds the CPU-staged
    // matrices the latch GPU half still needs. Swap it out across the
    // build and restore after: the queue build with instances_prepared
    // never appends to it, so the staged segments survive intact.
    {
        const build_key = seq | (@as(u64, 1) << 63);
        const sky_tex = scene.build_snapshot.sky_texture;
        const ibl_int = scene.build_snapshot.ibl_intensity;
        const saved_scratch = back.primary.instance_matrices;
        back.primary.instance_matrices = .empty;
        scene.buildQueuesInto(back, .{
            .snap = &scene.build_snapshot,
            .cache_key = build_key,
            .stats = &back.build_stats,
            .eye = eye,
            .sky_texture = sky_tex,
            .ibl_intensity = ibl_int,
            .instances_prepared = true,
            .instance_source = .build_view,
        });
        // The builder left the swapped-in list empty (no staging appends
        // under instances_prepared); discard it (zero capacity, no leak)
        // and restore the staged scratch.
        back.primary.instance_matrices = saved_scratch;
    }
}
