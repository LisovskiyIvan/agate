//! UI GPU upload + draw path: the canvas-owned buffer pair (`UiBufferEnsure` /
//! `ensureUiBufferPair`), the single-allowed-per-frame upload (`uploadUiBuffers`),
//! the upload-free draw (`drawUiBuffers`), the P6 same-frame window
//! (`isUploadOpen` / `markUiUploaded`), and the standalone immediate `render`.
//!
//! Split out of `ui.zig` (facade). Functions take a generic `canvas: anytype`
//! (a `*UICanvas` in practice) so this module never imports `../ui.zig` —
//! following the `scene/` precedent, subsystems never import the facade.
//! The pure batch math (`UIVertex`, `clampedVertCount`, `grownCapacity`,
//! `batchUploadBytes`) is reused from `draw.zig` directly; the font binding
//! is reached through the canvas forwarders. `ui.zig` owns the `UICanvas`
//! type and provides thin forwarders so every call site (including the P6
//! frame path in `scene/ui_frame.zig` and the 3D-GUI layer) keeps working
//! exactly as before.

const sokol = @import("sokol");
const sg = sokol.gfx;

const ui_shd = @import("ui_shader");

const ui_draw = @import("draw.zig");
const UIVertex = ui_draw.UIVertex;
const upload_meter = @import("../gpu_upload_meter.zig");

/// True when this sokol frame's single `updateBuffer` per buffer is
/// already spent on the canvas-owned pair. Read-only SDK metadata
/// (`queryStats`, no GPU writes), `isvalid`-gated and headless-safe:
/// without a context no commit can exist, so the armed flag alone
/// decides (never-uploaded canvases stay closed; tests override it
/// directly to simulate an open window).
pub fn isUploadOpen(canvas: anytype) bool {
    if (!canvas.ui_upload_armed) return false;
    if (!sg.isvalid()) return true;
    return sg.queryStats().prev_frame.frame_index == canvas.ui_upload_commit;
}

/// Marks a successful upload: arms the window with the current commit
/// watermark and bumps the upload sequence. Called by both upload paths
/// (frame + legacy immediate) right after their `updateBuffer` pair
/// lands — so the sequence identifies the last writer unambiguously,
/// even for two uploads inside one sokol frame (the watermark alone
/// cannot: the marker rotates only at commit).
pub fn markUiUploaded(canvas: anytype) void {
    canvas.ui_upload_commit = if (sg.isvalid()) sg.queryStats().prev_frame.frame_index else 0;
    canvas.ui_upload_armed = true;
    canvas.ui_upload_seq +%= 1;
}

/// Outcome of `ensureUiBufferPair`: buffers to upload into + install,
/// with per-buffer replacement flags. On `ok == false` the current
/// pair is untouched (only uncommitted new handles were destroyed).
pub const UiBufferEnsure = struct {
    ok: bool = false,
    vertex_buffer: sg.Buffer = .{},
    index_buffer: sg.Buffer = .{},
    capacity_vertices: usize = 0,
    capacity_indices: usize = 0,
    replaced_vb: bool = false,
    replaced_ib: bool = false,
};

/// Ensures canvas GPU buffers cover the batch: creates + VALIDates ALL
/// needed replacements BEFORE returning, never installs a FAILED
/// handle. Current handles are VALID-checked too — an id==0 (or
/// otherwise invalid) current with a large capacity is recreated, never
/// updated blindly. On any failure only the uncommitted new handle(s)
/// are destroyed and `ok` is false with the current pair untouched.
/// No retire here: the caller installs the pair and retires/destroys
/// the replaced handles itself (queue for the Scene frame path with
/// live epochs; immediate destroy for the standalone immediate path,
/// whose caller asserts no outstanding snapshots). Caller must hold a
/// sokol context (all `sg.*` below assert it).
pub fn ensureUiBufferPair(
    cur_vb: sg.Buffer,
    cur_ib: sg.Buffer,
    cap_v: usize,
    cap_i: usize,
    need_v: usize,
    need_i: usize,
) UiBufferEnsure {
    // A failed makeBuffer may hand out a nonzero FAILED id (pool
    // exhaustion is id == 0 only): validity is the state query, here
    // and for the current pair.
    const want_vb = need_v > cap_v or cur_vb.id == 0 or sg.queryBufferState(cur_vb) != .VALID;
    const want_ib = need_i > cap_i or cur_ib.id == 0 or sg.queryBufferState(cur_ib) != .VALID;
    const target_cap_v = if (need_v > cap_v) ui_draw.grownCapacity(cap_v, need_v) else cap_v;
    const target_cap_i = if (need_i > cap_i) ui_draw.grownCapacity(cap_i, need_i) else cap_i;
    var new_vb: sg.Buffer = .{};
    var new_ib: sg.Buffer = .{};
    if (want_vb) {
        new_vb = sg.makeBuffer(.{
            .usage = .{ .vertex_buffer = true, .write_transient = true },
            .size = target_cap_v * @sizeOf(UIVertex),
        });
        if (new_vb.id == 0 or sg.queryBufferState(new_vb) != .VALID) {
            if (new_vb.id != 0) sg.destroyBuffer(new_vb);
            return .{ .ok = false };
        }
    }
    if (want_ib) {
        new_ib = sg.makeBuffer(.{
            .usage = .{ .index_buffer = true, .write_transient = true },
            .size = target_cap_i * @sizeOf(u16),
        });
        if (new_ib.id == 0 or sg.queryBufferState(new_ib) != .VALID) {
            if (new_ib.id != 0) sg.destroyBuffer(new_ib);
            if (new_vb.id != 0) sg.destroyBuffer(new_vb);
            return .{ .ok = false };
        }
    }
    return .{
        .ok = true,
        .vertex_buffer = if (want_vb) new_vb else cur_vb,
        .index_buffer = if (want_ib) new_ib else cur_ib,
        .capacity_vertices = target_cap_v,
        .capacity_indices = target_cap_i,
        .replaced_vb = want_vb,
        .replaced_ib = want_ib,
    };
}

/// Uploads one UI batch into the given buffers (the single allowed
/// update per buffer per sokol frame) and records the exact bytes.
/// Shared by the legacy immediate render and the P6 frame upload; both
/// callers resolve WHICH buffers first (growth ownership differs).
pub fn uploadUiBuffers(
    vertex_buffer: sg.Buffer,
    index_buffer: sg.Buffer,
    verts: []const UIVertex,
    indices: []const u16,
) void {
    sg.writeBufferTransient(.{
        .dst = .{ .buffer = vertex_buffer },
        .src = .{ .data = sg.asRange(verts) },
    });
    sg.writeBufferTransient(.{
        .dst = .{ .buffer = index_buffer },
        .src = .{ .data = sg.asRange(indices) },
    });
    // Учёт динамики: весь UI-батч кадра (вершины + u16-индексы).
    upload_meter.record(ui_draw.batchUploadBytes(verts.len, indices.len));
}

/// Draws an already-uploaded UI batch (no upload, no meter). Shared by
/// the legacy immediate render and the P6 upload-free frame draw.
pub fn drawUiBuffers(
    pipeline: sg.Pipeline,
    vertex_buffer: sg.Buffer,
    index_buffer: sg.Buffer,
    font_view: sg.View,
    font_sampler: sg.Sampler,
    screen_w: f32,
    screen_h: f32,
    index_count: usize,
) void {
    if (pipeline.id == 0) return;
    sg.applyPipeline(pipeline);

    var bind = sg.Bindings{};
    bind.vertex_buffers[0] = vertex_buffer;
    bind.index_buffer = index_buffer;
    bind.views[ui_shd.VIEW_font_tex] = font_view;
    bind.samplers[ui_shd.SMP_smp] = font_sampler;
    sg.applyBindings(bind);

    const vs_params = ui_shd.VsParams{
        .screen_size = .{ screen_w, screen_h, 0.0, 0.0 },
    };
    sg.applyUniforms(ui_shd.UB_vs_params, sg.asRange(&vs_params));

    sg.draw(0, @intCast(index_count), 1);
}

/// Uploads dynamic batch buffers and executes the UI render pass.
/// Standalone immediate path, signature unchanged: the caller owns the
/// frame discipline (a `sg.commit` between renders reopens the window
/// automatically — no manual reset, standalone `render();
/// sg.commit(); render()` draws twice).
///
/// Borrower lifetime: this call is a buffer WRITER — it invalidates any
/// previously committed `UiFrame` on this canvas (same buffers
/// overwritten, or replaced on growth). Drawing an old frame after this
/// render without a recapture is invalid; Scene fail-closes that mix via
/// the upload identity, standalone callers own the discipline. No
/// blanket ban: epochs between uploads are freely usable — only
/// cross-writer consumption without recapture is invalid, and retired /
/// destroyed snapshots must never be consumed.
pub fn render(canvas: anytype, screen_w: f32, screen_h: f32) void {
    if (canvas.vertices.items.len == 0 or canvas.indices.items.len == 0) return;
    if (screen_w <= 0.0 or screen_h <= 0.0) return;
    // P6 same-frame guard: this sokol frame's single update on these
    // buffers is already spent — fail close (no upload, no draw: the
    // live lists may differ from the GPU-resident data). Checked before
    // the context gate so the policy holds headless too.
    if (isUploadOpen(canvas)) return;
    // Headless/tools (no sokol context): safe no-op instead of an
    // assert trap inside sg.*.
    if (!sg.isvalid()) return;

    // usize on purpose: `@min(usize, u16)` resolves to u16 in Zig 0.16, and
    // 48 B/vertex would then overflow the u16 multiply at 1366 vertices.
    const vert_count: usize = ui_draw.clampedVertCount(canvas.vertices.items.len);
    // Shared validate-all routine (same as the Scene frame path):
    // every needed replacement is created + VALIDated before any
    // upload/install, so a FAILED handle is never installed and the
    // current pair survives a pair failure untouched.
    const ensured = ensureUiBufferPair(
        canvas.vertex_buffer,
        canvas.index_buffer,
        canvas.capacity_vertices,
        canvas.capacity_indices,
        vert_count,
        canvas.indices.items.len,
    );
    if (!ensured.ok) return;
    // Standalone ownership: replaced handles are destroyed immediately —
    // the caller asserts no outstanding snapshots reference them (the
    // Scene frame path retires through the epoch queue instead).
    if (ensured.replaced_vb) {
        if (canvas.vertex_buffer.id != 0) sg.destroyBuffer(canvas.vertex_buffer);
        canvas.vertex_buffer = ensured.vertex_buffer;
        canvas.capacity_vertices = ensured.capacity_vertices;
    }
    if (ensured.replaced_ib) {
        if (canvas.index_buffer.id != 0) sg.destroyBuffer(canvas.index_buffer);
        canvas.index_buffer = ensured.index_buffer;
        canvas.capacity_indices = ensured.capacity_indices;
    }

    uploadUiBuffers(canvas.vertex_buffer, canvas.index_buffer, canvas.vertices.items[0..vert_count], canvas.indices.items);
    markUiUploaded(canvas);

    drawUiBuffers(
        canvas.pipeline,
        canvas.vertex_buffer,
        canvas.index_buffer,
        canvas.activeFontView(),
        canvas.activeFontSampler(),
        screen_w,
        screen_h,
        canvas.indices.items.len,
    );
}
