//! Context-side slot flush from frozen upload packets.
//! Section of the `upload_packets` facade (pure move, see
//! `upload_packets.zig` for the full stage/flush/commit contract).

const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const gpu_thread = @import("../gpu_thread.zig");
const upload_meter = @import("../gpu_upload_meter.zig");
const mesh_deferred = @import("../mesh/mesh.zig");

/// Context-side slot flush (beginPrepare, fresh-build path only). Uploads
/// from `slot` packets and creates deferred buffers from FROZEN sizes,
/// recording per-packet outcomes in the slot descriptors for the
/// game-side commit. THE CONTEXT NEVER WRITES GAME-OWNED STATE HERE: no
/// live flag clears, no live scalar publishes, no live handle installs, no
/// live array frees, no live staging memcpy. No live staging bytes drive
/// uploads (compute uploads direct from packet bytes) and no live mutable
/// state is read at all — only the slot packets, the stable allocator,
/// and context-owned services (retire queue, upload meter, sg).
/// Headless-safe: with no sg context every packet records undelivered and
/// the commit re-arms the flags, only the sg.* calls are skipped. See the
/// module header for the full contract.
pub fn flushSlotUploads(scene: anytype, slot: anytype) void {
    gpu_thread.assertOnContextThread();
    scene.gpu_retire.flush(scene.allocator);
    // Re-flush guard: a cancelled prepare re-latches the same slot without
    // a rebuild, so outcomes from the previous attempt must not linger
    // (an undelivered packet would look delivered). Slot-owned writes into
    // the claimed slot — race-free by the lease protocol.
    resetOutcomes(slot);
    flushPendingCreations(scene, slot);
    flushMorphs(scene, slot);
    flushParticleCpu(scene, slot);
    flushParticleGpu(scene, slot);
    flushParticleCompute(scene, slot);
    flushTrails(scene, slot);
    flushSoftbodies(scene, slot);
    flushGreased(scene, slot);
    if (!scene.flush_in_prepare and sg.isvalid()) sg.commit();
}

/// Clears every packet outcome back to undelivered (see flushSlotUploads).
fn resetOutcomes(slot: anytype) void {
    for (slot.morph_uploads.items) |*up| up.delivered = false;
    for (slot.p_cpu_uploads.items) |*up| {
        up.delivered = false;
        up.created_buffer_id = 0;
    }
    for (slot.p_gpu_uploads.items) |*up| {
        up.delivered = false;
        up.created_buffer_id = 0;
    }
    for (slot.p_compute_uploads.items) |*up| {
        up.delivered = false;
        up.unsupported = false;
        up.consumed_staged = 0;
        up.consumed_dt = 0.0;
        // `dispatches` is an attempt counter, NOT a delivery outcome: a
        // cancelled prepare re-flushes the same slot, and the re-flush must
        // still transfer the earlier attempt's dispatch. Only the commit
        // (exactly-once transfer) or a fresh packet list (slot reset)
        // clears it.
        up.created_state_buffer_id = 0;
        up.created_spawn_buffer_id = 0;
        up.created_draw_buffer_id = 0;
        up.created_state_view_id = 0;
        up.created_spawn_view_id = 0;
        up.created_draw_view_id = 0;
        up.created_shader_id = 0;
        up.created_pipeline_id = 0;
    }
    for (slot.trail_uploads.items) |*up| {
        up.delivered = false;
        up.created_vertex_buffer_id = 0;
        up.created_index_buffer_id = 0;
    }
    for (slot.soft_uploads.items) |*up| {
        up.delivered = false;
        up.created_vertex_buffer_id = 0;
        up.created_index_buffer_id = 0;
    }
    for (slot.greased_uploads.items) |*up| {
        up.delivered = false;
        up.full_delivered = false;
        up.created_vertex_buffer_id = 0;
        up.created_index_buffer_id = 0;
    }
    for (slot.pending_uploads.items) |*up| {
        up.delivered = false;
        up.created_vertex_buffer_id = 0;
        up.created_index_buffer_id = 0;
        up.created_delta_image_id = 0;
        up.created_delta_view_id = 0;
    }
}

fn flushPendingCreations(scene: anytype, slot: anytype) void {
    // No live reads, no live writes: creation runs purely from frozen
    // geometry/sizes; the game-side commit installs the handles (guarded)
    // or re-arms `gpu_pending` for retry. Headless records undelivered.
    if (!sg.isvalid()) return;
    for (slot.pending_uploads.items) |*up| {
        const v_end = up.vert_lo + up.vert_count;
        const i_end = up.index_lo + up.index_count;
        if (v_end > slot.pending_verts.items.len or i_end > slot.pending_indices.items.len) continue;
        const verts = slot.pending_verts.items[up.vert_lo..v_end];
        const idx32 = slot.pending_indices.items[up.index_lo..i_end];
        var vbuf: sg.Buffer = .{};
        var ibuf: sg.Buffer = .{};
        if (up.dynamic_update) {
            // Vertex creation through the canonical mesh seam (same descs as
            // the drain's finish; frozen slot bytes, never live mesh state).
            // Index creation stays a regular sg.makeBuffer here.
            vbuf = mesh_deferred.makeDeferredMeshVertexBuffer(.{
                .usage = .{ .vertex_buffer = true, .write_transient = true },
                .size = verts.len * @sizeOf(@TypeOf(verts[0])),
            });
            // Nonzero id may still be FAILED state (pool exhaustion is id 0
            // only); draw rejects FAILED permanently, so release the failed
            // slot and stay undelivered for retry. Same contract as the
            // instance-staging growth path.
            if (vbuf.id == 0 or sg.queryBufferState(vbuf) != .VALID) {
                if (vbuf.id != 0) sg.destroyBuffer(vbuf);
                continue;
            }
            if (verts.len > 0) {
                sg.writeBufferTransient(.{
                    .dst = .{ .buffer = vbuf },
                    .src = .{ .data = sg.asRange(verts) },
                });
            }
            if (up.index_type_is_u16) {
                const tmp = scene.allocator.alloc(u16, idx32.len) catch {
                    sg.destroyBuffer(vbuf);
                    continue;
                };
                defer scene.allocator.free(tmp);
                for (idx32, 0..) |v, k| tmp[k] = @intCast(v);
                ibuf = sg.makeBuffer(.{ .usage = .{ .index_buffer = true }, .data = sg.asRange(tmp) });
            } else {
                ibuf = sg.makeBuffer(.{ .usage = .{ .index_buffer = true }, .data = sg.asRange(idx32) });
            }
            if (ibuf.id == 0 or sg.queryBufferState(ibuf) != .VALID) {
                if (ibuf.id != 0) sg.destroyBuffer(ibuf);
                sg.destroyBuffer(vbuf);
                continue;
            }
            up.created_vertex_buffer_id = vbuf.id;
            up.created_index_buffer_id = ibuf.id;
        } else {
            vbuf = mesh_deferred.makeDeferredMeshVertexBuffer(.{ .data = sg.asRange(verts) });
            if (vbuf.id == 0 or sg.queryBufferState(vbuf) != .VALID) {
                if (vbuf.id != 0) sg.destroyBuffer(vbuf);
                continue;
            }
            if (up.index_type_is_u16) {
                const tmp = scene.allocator.alloc(u16, idx32.len) catch {
                    sg.destroyBuffer(vbuf);
                    continue;
                };
                defer scene.allocator.free(tmp);
                for (idx32, 0..) |v, k| tmp[k] = @intCast(v);
                ibuf = sg.makeBuffer(.{ .usage = .{ .index_buffer = true }, .data = sg.asRange(tmp) });
            } else {
                ibuf = sg.makeBuffer(.{ .usage = .{ .index_buffer = true }, .data = sg.asRange(idx32) });
            }
            if (ibuf.id == 0 or sg.queryBufferState(ibuf) != .VALID) {
                if (ibuf.id != 0) sg.destroyBuffer(ibuf);
                sg.destroyBuffer(vbuf);
                continue;
            }
            up.created_vertex_buffer_id = vbuf.id;
            up.created_index_buffer_id = ibuf.id;
        }
        // Delivered: the commit publishes vertex_count (when live is zero),
        // re-arms morph_upload_needed for dynamic updates, clears
        // pending_dynamic_update, and frees the consumed live arrays.
        // GPU-morph meshes additionally land their delta texture here, from
        // the frozen packed bytes (never live morph_targets): any failure
        // tears the fresh buffers back down inline (context thread, legal)
        // and records undelivered, so the commit re-arms both flags and the
        // next build retries the whole finish atomically — the mesh stays
        // skipped by the queue meanwhile (no base-pose frame, no panic).
        if (up.morph_delta_pending) {
            const d_end = up.delta_lo + up.delta_count;
            const want: usize = @as(usize, up.delta_width) * up.delta_height * 4;
            if (d_end > slot.pending_delta_data.items.len or up.delta_count != want or want == 0) {
                sg.destroyBuffer(vbuf);
                sg.destroyBuffer(ibuf);
                // Invariant: created_* records live-or-zero handles only.
                up.created_vertex_buffer_id = 0;
                up.created_index_buffer_id = 0;
                continue;
            }
            var img_desc = sg.ImageDesc{
                .width = @intCast(up.delta_width),
                .height = @intCast(up.delta_height),
                .pixel_format = .RGBA32F,
            };
            img_desc.data.mip_levels[0] = sg.asRange(slot.pending_delta_data.items[up.delta_lo..d_end]);
            const img = sg.makeImage(img_desc);
            // Nonzero id may still be FAILED state (same contract as the
            // buffer creations above): tear everything down inline and stay
            // undelivered so the commit re-arms both flags atomically.
            if (img.id == 0 or sg.queryImageState(img) != .VALID) {
                if (img.id != 0) sg.destroyImage(img);
                sg.destroyBuffer(vbuf);
                sg.destroyBuffer(ibuf);
                // Invariant: created_* records live-or-zero handles only.
                up.created_vertex_buffer_id = 0;
                up.created_index_buffer_id = 0;
                continue;
            }
            const view = sg.makeView(.{ .texture = .{ .image = img } });
            if (view.id == 0 or sg.queryViewState(view) != .VALID) {
                if (view.id != 0) sg.destroyView(view);
                sg.destroyImage(img);
                sg.destroyBuffer(vbuf);
                sg.destroyBuffer(ibuf);
                // Invariant: created_* records live-or-zero handles only.
                up.created_vertex_buffer_id = 0;
                up.created_index_buffer_id = 0;
                continue;
            }
            up.created_delta_image_id = img.id;
            up.created_delta_view_id = view.id;
            upload_meter.record(up.delta_count * @sizeOf(f32));
        }
        up.delivered = true;
    }
}

fn flushMorphs(scene: anytype, slot: anytype) void {
    _ = scene;
    // No live reads, no live writes. The frozen target id is uploaded to
    // directly: sokol ids are generation-tagged, so a stale id (owner
    // destroyed between build and latch — an app-contract violation the
    // commit guard also skips) fail-closes in the driver, never aliases a
    // recycled buffer. Morph buffers are updated in place and never
    // replaced, so no foreign-buffer case exists here by construction.
    for (slot.morph_uploads.items) |*up| {
        const end = up.data_lo + up.count;
        if (end > slot.morph_data.items.len) continue;
        if (up.count == 0) {
            up.delivered = true;
            continue;
        }
        if (!sg.isvalid()) continue;
        if (up.buffer_id == 0) continue;
        if (!sg.queryBufferUsage(.{ .id = up.buffer_id }).write_transient) {
            up.delivered = true;
            continue;
        }
        sg.writeBufferTransient(.{
            .dst = .{ .buffer = .{ .id = up.buffer_id } },
            .src = .{ .data = sg.asRange(slot.morph_data.items[up.data_lo..end]) },
        });
        if (up.dirty) {
            upload_meter.record(up.count * @sizeOf(@TypeOf(slot.morph_data.items[0])));
        }
        up.delivered = true;
    }
}

fn flushParticleCpu(scene: anytype, slot: anytype) void {
    // No live reads, no live writes. Deferred creation sizes from the
    // FROZEN capacity (immutable after init); the outcome carries the
    // created buffer for the game-side commit, or stays undelivered for
    // retry. Instance buffers are never replaced (updated in place), so
    // the frozen target id is uploaded to directly (see flushMorphs).
    for (slot.p_cpu_uploads.items) |*up| {
        const end = up.data_lo + up.count;
        if (end > slot.p_cpu_data.items.len) continue;
        var target_id = up.buffer_id;
        if (target_id == 0) {
            // Deferred first-use creation (staged by the producer as
            // "no buffer yet"): frozen capacity, never the live field.
            if (!sg.isvalid()) continue;
            if (up.capacity == 0) continue;
            const created = sg.makeBuffer(.{
                .usage = .{ .vertex_buffer = true, .write_transient = true },
                .size = up.capacity * @sizeOf(@TypeOf(slot.p_cpu_data.items[0])),
            });
            // Nonzero id may still be FAILED state (same contract as
            // flushPendingCreations): release the failed slot and stay
            // undelivered; the commit re-arms for retry.
            if (created.id == 0 or sg.queryBufferState(created) != .VALID) {
                if (created.id != 0) sg.destroyBuffer(created);
                continue;
            }
            up.created_buffer_id = created.id;
            target_id = created.id;
        }
        if (up.count == 0) {
            up.delivered = true;
            continue;
        }
        if (!sg.isvalid()) continue;
        sg.writeBufferTransient(.{
            .dst = .{ .buffer = .{ .id = target_id } },
            .src = .{ .data = sg.asRange(slot.p_cpu_data.items[up.data_lo..end]) },
        });
        if (up.dirty) {
            upload_meter.record(up.count * @sizeOf(@TypeOf(slot.p_cpu_data.items[0])));
        }
        up.delivered = true;
    }
    _ = scene;
}

fn flushParticleGpu(scene: anytype, slot: anytype) void {
    // Same shape as flushParticleCpu (see above).
    for (slot.p_gpu_uploads.items) |*up| {
        const end = up.data_lo + up.count;
        if (end > slot.p_gpu_data.items.len) continue;
        var target_id = up.buffer_id;
        if (target_id == 0) {
            if (!sg.isvalid()) continue;
            if (up.capacity == 0) continue;
            const created = sg.makeBuffer(.{
                .usage = .{ .vertex_buffer = true, .write_transient = true },
                .size = up.capacity * @sizeOf(@TypeOf(slot.p_gpu_data.items[0])),
            });
            // Nonzero id may still be FAILED state (same contract as
            // flushPendingCreations): release the failed slot and stay
            // undelivered; the commit re-arms for retry.
            if (created.id == 0 or sg.queryBufferState(created) != .VALID) {
                if (created.id != 0) sg.destroyBuffer(created);
                continue;
            }
            up.created_buffer_id = created.id;
            target_id = created.id;
        }
        if (up.count == 0) {
            up.delivered = true;
            continue;
        }
        if (!sg.isvalid()) continue;
        sg.writeBufferTransient(.{
            .dst = .{ .buffer = .{ .id = target_id } },
            .src = .{ .data = sg.asRange(slot.p_gpu_data.items[up.data_lo..end]) },
        });
        if (up.dirty) {
            upload_meter.record(up.count * @sizeOf(@TypeOf(slot.p_gpu_data.items[0])));
        }
        up.delivered = true;
    }
    _ = scene;
}

fn flushParticleCompute(scene: anytype, slot: anytype) void {
    // Direct staged upload from packet bytes: no live staging bytes
    // drive uploads, only frozen counts. No live reads, no live writes: creation/dispatch run on
    // frozen ids + frozen capacity, outcomes ride the descriptor for the
    // game-side commit (ring advance, dt consume, handle installs,
    // unsupported latch) or stay undelivered for retry.
    const compute = @import("../compute.zig");
    const pc_shd = @import("particle_compute_shader");
    const workgroup_size = @import("../particles/types.zig").compute_workgroup_size;
    for (slot.p_compute_uploads.items) |*up| {
        const end = up.data_lo + up.data_count;
        if (end > slot.p_compute_data.items.len) continue;
        if (!sg.isvalid()) {
            // Headless: full no-op that records undelivered for retry,
            // exactly like the drain's `if (!sg.isvalid()) return`.
            // The frozen bytes stay in the slot; the next funded build
            // re-freezes from the intact live window.
            continue;
        }
        if (!compute.supported()) {
            // Latch unsupported context-side (mirrors the direct drain:
            // never a silent fallback, never a retry spin) — published
            // through the outcome, installed by the commit.
            up.unsupported = true;
            continue;
        }
        const cap = up.capacity;
        if (cap == 0) continue;
        // Creation from frozen capacity + frozen request flags. Partial
        // progress is recorded into the outcome even on failure so the
        // commit can install it and the retry completes the rest.
        var state_id = up.state_buffer_id;
        var spawn_id = up.spawn_buffer_id;
        var draw_id = up.draw_buffer_id;
        var state_view = up.state_view_id;
        var spawn_view = up.spawn_view_id;
        var draw_view = up.draw_view_id;
        var shader_id = up.shader_id;
        var pipeline_id = up.pipeline_id;
        if (up.buffers_pending) {
            if (state_id == 0) {
                const zeros = scene.allocator.alloc(u8, cap * @sizeOf(@import("../particles/types.zig").ComputeParticleState)) catch {
                    slot.build_stats.build_oom_drops += 1;
                    continue;
                };
                defer scene.allocator.free(zeros);
                @memset(zeros, 0);
                const created = sg.makeBuffer(.{
                    .usage = .{ .storage_buffer = true },
                    .data = sg.Range{ .ptr = zeros.ptr, .size = zeros.len },
                });
                // FAILED-state guard (same contract as the buffer sites
                // above): destroy the failed slot inline and keep the
                // partial-progress semantics below (already-recorded
                // handles ride the outcome for the commit to install).
                if (created.id == 0 or sg.queryBufferState(created) != .VALID) {
                    if (created.id != 0) sg.destroyBuffer(created);
                    continue;
                }
                up.created_state_buffer_id = created.id;
                state_id = created.id;
            }
            if (spawn_id == 0) {
                const created = sg.makeBuffer(.{
                    .usage = .{ .storage_buffer = true, .write_transient = true },
                    .size = cap * @sizeOf(@import("../particles/types.zig").GpuParticleSlot),
                });
                if (created.id == 0 or sg.queryBufferState(created) != .VALID) {
                    if (created.id != 0) sg.destroyBuffer(created);
                    continue;
                }
                up.created_spawn_buffer_id = created.id;
                spawn_id = created.id;
            }
            if (draw_id == 0) {
                const created = sg.makeBuffer(.{
                    .usage = .{ .vertex_buffer = true, .storage_buffer = true },
                    .size = cap * @sizeOf(@import("../particles/types.zig").ParticleInstanceData),
                });
                if (created.id == 0 or sg.queryBufferState(created) != .VALID) {
                    if (created.id != 0) sg.destroyBuffer(created);
                    continue;
                }
                up.created_draw_buffer_id = created.id;
                draw_id = created.id;
            }
            if (state_view == 0 and state_id != 0) {
                const created = compute.makeStorageView(.{ .id = state_id }, "compute-particles-state");
                // FAILED views are destroyed inline: this flush runs on the
                // context thread (same-thread teardown precedent as
                // deinitComputeGpuObjects), so no retire queue is needed.
                if (created.id == 0 or sg.queryViewState(created) != .VALID) {
                    if (created.id != 0) sg.destroyView(created);
                    continue;
                }
                up.created_state_view_id = created.id;
                state_view = created.id;
            }
            if (spawn_view == 0 and spawn_id != 0) {
                const created = compute.makeStorageView(.{ .id = spawn_id }, "compute-particles-spawn");
                if (created.id == 0 or sg.queryViewState(created) != .VALID) {
                    if (created.id != 0) sg.destroyView(created);
                    continue;
                }
                up.created_spawn_view_id = created.id;
                spawn_view = created.id;
            }
            if (draw_view == 0 and draw_id != 0) {
                const created = compute.makeStorageView(.{ .id = draw_id }, "compute-particles-draw");
                if (created.id == 0 or sg.queryViewState(created) != .VALID) {
                    if (created.id != 0) sg.destroyView(created);
                    continue;
                }
                up.created_draw_view_id = created.id;
                draw_view = created.id;
            }
            if (shader_id == 0) {
                const created = sg.makeShader(pc_shd.particleComputeShaderDesc(sg.queryBackend()));
                if (created.id == 0 or sg.queryShaderState(created) != .VALID) {
                    if (created.id != 0) sg.destroyShader(created);
                    continue;
                }
                up.created_shader_id = created.id;
                shader_id = created.id;
            }
            if (pipeline_id == 0 and shader_id != 0) {
                const created = compute.makePipeline(.{ .id = shader_id }, "compute-particles");
                if (created.id == 0 or sg.queryPipelineState(created) != .VALID) {
                    if (created.id != 0) sg.destroyPipeline(created);
                    continue;
                }
                up.created_pipeline_id = created.id;
                pipeline_id = created.id;
            }
        }
        if (state_id == 0) continue;
        if (up.state_clear_pending) {
            const zeros = scene.allocator.alloc(u8, cap * @sizeOf(@import("../particles/types.zig").ComputeParticleState)) catch {
                slot.build_stats.build_oom_drops += 1;
                continue;
            };
            defer scene.allocator.free(zeros);
            @memset(zeros, 0);
            const created_buf = sg.makeBuffer(.{
                .usage = .{ .storage_buffer = true },
                .data = sg.Range{ .ptr = zeros.ptr, .size = zeros.len },
            });
            if (created_buf.id != 0 and sg.queryBufferState(created_buf) == .VALID) {
                const created_view = compute.makeStorageView(created_buf, "compute-particles-state");
                if (created_view.id != 0 and sg.queryViewState(created_view) == .VALID) {
                    up.created_state_buffer_id = created_buf.id;
                    up.created_state_view_id = created_view.id;
                    state_id = created_buf.id;
                    state_view = created_view.id;
                } else {
                    if (created_view.id != 0) sg.destroyView(created_view);
                    sg.destroyBuffer(created_buf);
                    continue;
                }
            } else {
                if (created_buf.id != 0) sg.destroyBuffer(created_buf);
                continue;
            }
            upload_meter.record(zeros.len);
        }
        // Creation incomplete (a make* failed; partial handles ride the
        // outcome for the commit to install): keep everything staged,
        // consume nothing.
        if (pipeline_id == 0) continue;
        const pkt = slot.p_compute_data.items[up.data_lo..end];
        const will_dispatch = (up.flush_pending or up.data_count > 0) and up.high_water > 0;
        if (spawn_id != 0) {
            if (up.data_count > 0) {
                sg.writeBufferTransient(.{
                    .dst = .{ .buffer = .{ .id = spawn_id } },
                    .src = .{ .data = sg.asRange(pkt) },
                });
                upload_meter.record(up.data_count * @sizeOf(@TypeOf(pkt[0])));
            } else if (will_dispatch) {
                var dummy: @import("../particles/types.zig").GpuParticleSlot = undefined;
                @memset(std.mem.asBytes(&dummy), 0);
                sg.writeBufferTransient(.{
                    .dst = .{ .buffer = .{ .id = spawn_id } },
                    .src = .{ .data = sg.asRange(&dummy) },
                });
            }
        }
        // Dispatch on staged spawns even without a frame update (a
        // sub-emitter child may hold spawns while its own update did not
        // run; dt 0 then integrates nothing but still applies respawns).
        // Consumes the window/dt only here, via the outcome below.
        if (up.flush_pending or up.data_count > 0) {
            if (up.high_water > 0) {
                const params = pc_shd.CsParams{
                    .dyn = .{ up.dt_accum, up.drag, 0.0, 0.0 },
                    .grav = .{ up.gravity[0], up.gravity[1], up.gravity[2], 0.0 },
                    .sheet = .{
                        @floatFromInt(up.sheet_cols),
                        @floatFromInt(up.sheet_rows),
                        up.sheet_loops,
                        0.0,
                    },
                    .addr = .{
                        @floatFromInt(cap),
                        @floatFromInt(up.stage_base),
                        @floatFromInt(up.data_count),
                        0.0,
                    },
                };
                sg.beginPass(.{ .compute = true, .label = "compute-particles" });
                sg.applyPipeline(.{ .id = pipeline_id });
                var bind = sg.Bindings{};
                bind.views[pc_shd.VIEW_cs_state] = .{ .id = state_view };
                bind.views[pc_shd.VIEW_cs_spawn] = .{ .id = spawn_view };
                bind.views[pc_shd.VIEW_cs_draw] = .{ .id = draw_view };
                sg.applyBindings(bind);
                sg.applyUniforms(pc_shd.UB_cs_params, sg.asRange(&params));
                sg.dispatch(@intCast(compute.groupCount(up.high_water, workgroup_size)), 1, 1);
                sg.endPass();
                // Slot-owned attempt outcome: counted only after the real
                // dispatch above (headless/unsupported/empty paths never
                // reach here, so they stay 0 — never faked).
                up.dispatches +%= 1;
            }
            up.consumed_staged = up.data_count;
            up.consumed_dt = up.dt_accum;
        }
        up.delivered = true;
    }
}

fn flushTrails(scene: anytype, slot: anytype) void {
    // No live reads, no live writes. Creation sizes from the FROZEN caps
    // (fixed at trail init); uploads target the frozen ids directly (trail
    // buffers are only ever created from zero, never replaced, so no
    // foreign-buffer case exists by construction — same rationale as
    // flushMorphs). Scalars (index_count/bounds) and created handles ride
    // the outcome for the game-side commit.
    _ = scene;
    for (slot.trail_uploads.items) |*up| {
        const v_end = up.vert_lo + up.vert_count;
        const i_end = up.index_lo + up.index_count;
        if (v_end > slot.trail_verts.items.len or i_end > slot.trail_indices.items.len) continue;
        var vertex_id = up.vertex_buffer_id;
        var index_id = up.index_buffer_id;
        if (up.buffers_pending) {
            if (!sg.isvalid()) continue;
            const vb = sg.makeBuffer(.{
                .usage = .{ .vertex_buffer = true, .write_transient = true },
                .size = up.vert_cap * @sizeOf(@TypeOf(slot.trail_verts.items[0])),
            });
            const ib = sg.makeBuffer(.{
                .usage = .{ .index_buffer = true, .write_transient = true },
                .size = up.index_cap * @sizeOf(@TypeOf(slot.trail_indices.items[0])),
            });
            // Pair-atomic VALID-only creation: a nonzero FAILED id must not
            // reach the commit (it would strand a dead live handle with the
            // pending flags cleared). The else arm already destroys any
            // nonzero handle, so FAILED follows the pool-exhaustion path.
            if (vb.id != 0 and ib.id != 0 and sg.queryBufferState(vb) == .VALID and sg.queryBufferState(ib) == .VALID) {
                up.created_vertex_buffer_id = vb.id;
                up.created_index_buffer_id = ib.id;
                vertex_id = vb.id;
                index_id = ib.id;
            } else {
                if (vb.id != 0) sg.destroyBuffer(vb);
                if (ib.id != 0) sg.destroyBuffer(ib);
                continue;
            }
        }
        if (sg.isvalid()) {
            if (up.vert_count > 0 and vertex_id != 0) {
                sg.writeBufferTransient(.{
                    .dst = .{ .buffer = .{ .id = vertex_id } },
                    .src = .{ .data = sg.asRange(slot.trail_verts.items[up.vert_lo..v_end]) },
                });
                if (up.dirty) {
                    upload_meter.record(up.vert_count * @sizeOf(@TypeOf(slot.trail_verts.items[0])));
                }
            }
            if (up.index_count > 0 and index_id != 0) {
                sg.writeBufferTransient(.{
                    .dst = .{ .buffer = .{ .id = index_id } },
                    .src = .{ .data = sg.asRange(slot.trail_indices.items[up.index_lo..i_end]) },
                });
                if (up.dirty) {
                    upload_meter.record(up.index_count * @sizeOf(@TypeOf(slot.trail_indices.items[0])));
                }
            }
            up.delivered = true;
        } else {
            // Headless: an empty packet with no creation still delivers
            // (the commit publishes the zero scalars exactly like the
            // quiesced drain, consuming the generation); any payload or
            // creation need records undelivered for retry once a context
            // exists. Uploading to a zero id headless would be a no-op
            // mistaken for delivery, so the zero-id + payload case also
            // stays undelivered.
            const needs_creation = up.buffers_pending;
            const has_payload = up.vert_count > 0 or up.index_count > 0;
            if (!needs_creation and !has_payload) up.delivered = true;
        }
    }
}

fn flushSoftbodies(scene: anytype, slot: anytype) void {
    // No live reads, no live writes. Creation sizes from frozen data (the
    // vertex buffer from the frozen cap, the u32 index buffer from the
    // frozen index bytes); uploads target the frozen id directly (soft
    // buffers are only ever created from zero, never replaced — same
    // rationale as flushMorphs). Outcomes ride the descriptor for the
    // game-side commit.
    _ = scene;
    for (slot.soft_uploads.items) |*up| {
        const end = up.data_lo + up.vert_count;
        if (end > slot.soft_data.items.len) continue;
        const i_end = up.index_lo + up.index_count;
        if (i_end > slot.soft_indices.items.len) continue;
        var vertex_id = up.vertex_buffer_id;
        if (up.buffers_pending) {
            if (!sg.isvalid()) continue;
            const vb = sg.makeBuffer(.{
                .usage = .{ .vertex_buffer = true, .write_transient = true },
                .size = up.vert_cap * @sizeOf(@TypeOf(slot.soft_data.items[0])),
            });
            // Cloth index buffers are always u32 (vertices cap at 4096, so
            // this wastes nothing material and keeps one upload path).
            // Bytes come from the frozen packet, never the live array.
            const ib = sg.makeBuffer(.{
                .usage = .{ .index_buffer = true },
                .data = sg.asRange(slot.soft_indices.items[up.index_lo..i_end]),
            });
            // Pair-atomic VALID-only creation (see flushTrails): a nonzero
            // FAILED id follows the pool-exhaustion path via the else arm.
            if (vb.id != 0 and ib.id != 0 and sg.queryBufferState(vb) == .VALID and sg.queryBufferState(ib) == .VALID) {
                up.created_vertex_buffer_id = vb.id;
                up.created_index_buffer_id = ib.id;
                vertex_id = vb.id;
            } else {
                if (vb.id != 0) sg.destroyBuffer(vb);
                if (ib.id != 0) sg.destroyBuffer(ib);
                continue;
            }
        }
        if (sg.isvalid()) {
            if (up.vert_count == 0) {
                up.delivered = true;
                continue;
            }
            if (vertex_id == 0) continue;
            sg.writeBufferTransient(.{
                .dst = .{ .buffer = .{ .id = vertex_id } },
                .src = .{ .data = sg.asRange(slot.soft_data.items[up.data_lo..end]) },
            });
            if (up.dirty) {
                upload_meter.record(up.vert_count * @sizeOf(@TypeOf(slot.soft_data.items[0])));
            }
            up.delivered = true;
        } else {
            // Headless rule (see flushTrails): empty + no creation still
            // delivers; anything else retries.
            if (!up.buffers_pending and up.vert_count == 0) up.delivered = true;
        }
    }
}

fn flushGreased(scene: anytype, slot: anytype) void {
    // No live reads, no live writes. Creation sizes from the FROZEN caps;
    // uploads target the frozen ids directly (greased buffers are only
    // ever created from zero, never replaced — same rationale as
    // flushMorphs). Outcomes ride the descriptor for the game-side commit.
    _ = scene;
    for (slot.greased_uploads.items) |*up| {
        const v_end = up.vert_lo + up.vert_count;
        if (v_end > slot.greased_verts.items.len) continue;
        const i_end = up.index_lo + up.index_count;
        if (i_end > slot.greased_indices.items.len) continue;
        // Headless rule (see flushTrails): empty + existing targets still
        // delivers — the generation is consumed. Headless always records
        // undelivered (the flag is retained for retry); the commit re-arms
        // for retry once a context exists.
        if (!sg.isvalid()) continue;
        var vertex_id = up.vertex_buffer_id;
        var index_id = up.index_buffer_id;
        var just_created = false;
        if (vertex_id == 0) {
            const vb = sg.makeBuffer(.{
                .usage = .{ .vertex_buffer = true, .write_transient = true },
                .size = up.vert_cap * @sizeOf(@TypeOf(slot.greased_verts.items[0])),
            });
            const ib = sg.makeBuffer(.{
                .usage = .{ .index_buffer = true, .write_transient = true },
                .size = up.index_cap * @sizeOf(@TypeOf(slot.greased_indices.items[0])),
            });
            if (vb.id == 0 or ib.id == 0 or sg.queryBufferState(vb) != .VALID or sg.queryBufferState(ib) != .VALID) {
                if (vb.id != 0) sg.destroyBuffer(vb);
                if (ib.id != 0) sg.destroyBuffer(ib);
                // Creation failed (pool exhaustion or FAILED state):
                // undelivered, the commit re-arms for retry like the drain's
                // path.
                continue;
            }
            up.created_vertex_buffer_id = vb.id;
            up.created_index_buffer_id = ib.id;
            vertex_id = vb.id;
            index_id = ib.id;
            just_created = true;
        }
        if (up.vert_count > 0 and vertex_id != 0) {
            sg.writeBufferTransient(.{
                .dst = .{ .buffer = .{ .id = vertex_id } },
                .src = .{ .data = sg.asRange(slot.greased_verts.items[up.vert_lo..v_end]) },
            });
            if (up.dirty) {
                upload_meter.record(up.vert_count * @sizeOf(@TypeOf(slot.greased_verts.items[0])));
            }
        }
        // A just-created index buffer is empty: it needs the full index
        // upload even when the frozen full flag was false (the stage
        // froze indices whenever live buffers were missing, precisely for
        // this case).
        if (up.index_count > 0 and index_id != 0) {
            sg.writeBufferTransient(.{
                .dst = .{ .buffer = .{ .id = index_id } },
                .src = .{ .data = sg.asRange(slot.greased_indices.items[up.index_lo..i_end]) },
            });
            if (up.full_upload or just_created) {
                if (up.dirty) {
                    upload_meter.record(up.index_count * @sizeOf(@TypeOf(slot.greased_indices.items[0])));
                }
                up.full_delivered = true;
            }
        } else if (up.full_upload and up.index_count == 0) {
            // Full requested but nothing frozen (degenerate): the full
            // requirement is satisfied vacuously.
            up.full_delivered = true;
        }
        up.delivered = true;
    }
}
