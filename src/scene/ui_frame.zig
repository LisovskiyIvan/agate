//! P6 UI handoff: Scene UI draw consumes an immutable render-owned frame.
//!
//! Ownership block (mirrors the P4/P5 shape, one boundary at a time):
//! - Producer `BuildClaim.stageUi` freezes canvas geometry in its claimed
//!   slot. Context-side staged prepare calls `capturePacket` on those bytes;
//!   it never copies live canvas geometry. `upload` grows/refreshes the
//!   context-owned canvas GPU pair and spends one `updateBuffer` per buffer.
//!   Standalone callers may explicitly use `capture(canvas)` outside Scene.
//!   Neither capture nor upload runs during prepared drawing.
//! - READS the committed packet at draw time (`drawPrepared`, upload-free):
//!   Scene's direct path and `PostFXStack.renderChain` draw the frame, never
//!   the live canvas lists, style/input state, or live pipeline/font fields.
//!   Draw site selection uses the prepared `canvas_present` flag, never a
//!   live `ui_canvas` read — the snapshot boundary is complete at prepare.
//! - GPU buffers stay canvas-owned (no second full-capacity VB/IB set for
//!   snapshots): the frame owns CPU copies (retained capacity) and captures
//!   buffer/pipeline/font IDs. Handles are borrowed under the phase lock and
//!   the canvas owner's lifetime — no font/texture refcount redesign. The
//!   frame itself never references the canvas at draw (packet only).
//! - Growth retires replaced buffers through the existing
//!   `GpuRetireQueue.retireBuffer` (same epochs, no new queue/kind); the
//!   null-queue standalone case destroys immediately with the documented
//!   caller obligation (no outstanding snapshots), exactly like P5 staging.
//!   Both upload paths (prepared frame + standalone immediate) share the single
//!   `UICanvas.ensureUiBufferPair` routine: every needed replacement is
//!   created + VALIDated before any upload/install/retire.
//! - Same-sokol-frame policy: FIRST-COMMITTED-WINS, automatic. The window
//!   is the canvas-resident commit watermark (`UICanvas.isUploadOpen`,
//!   compared against `sg.queryStats().prev_frame.frame_index`, which
//!   rotates at every `sg.commit` even with stats disabled): a real commit
//!   reopens the window with no caller action, so standalone
//!   `render(); sg.commit(); render()` draws twice. A capture landing
//!   inside an open window drops the recapture ENTIRELY — CPU copies, draw
//!   packet, counts and fields stay exactly the committed frame A (no
//!   allocation, no overwrite) — until the next capture after a commit
//!   publishes B. Preservation additionally requires the committed-upload
//!   IDENTITY to still match the canvas (sequence + buffer IDs stamped by
//!   the successful `upload`): any other writer in between (immediate
//!   render overwriting the same buffers, or a growth replacement
//!   destroying them) invalidates the borrowed packet, and the capture
//!   fail-closes to coherent-empty instead of drawing destroyed bytes or
//!   wrong counts — no re-upload until the next commit + capture, which
//!   recovers. No P7 buffering, no pending-B auto-upload promise: an
//!   `upload()` after the commit with nothing staged returns
//!   `nothing_to_do`. There is deliberately NO content-hash dedup:
//!   steady-state byte totals stay identical to the legacy
//!   upload-every-frame path (only the prepare-vs-render attribution moves).
//! - Borrow lifetime: the packet borrows canvas-owned buffers/handles, so a
//!   committed frame stays drawable only until the NEXT upload to the same
//!   canvas by ANY path or a buffer replacement. Drawing an old frame after
//!   another writer without a recapture is a caller-obligation violation
//!   (invalidated by any buffer write); Scene prepare re-captures every
//!   frame and fail-closes the mix via the identity above. Usable epochs
//!   between uploads are unconstrained; retired/destroyed snapshots must
//!   never be consumed.
//! - Failure coherence: both CPU reserves happen BEFORE publish (OOM gives a
//!   coherent empty frame, never a partial half); growth creates + VALIDates
//!   ALL needed replacements before uploading/committing/retiring anything
//!   (a FAILED state is possible with a nonzero id, so the id is never the
//!   validity check); any failure keeps the owned GPU pair valid and drops
//!   the frame to coherent-empty for a next-frame retry.
//! - Prepare/render are sequential on the context thread. Variable-length
//!   producer geometry crosses only through leased slot-owned copies; Scene
//!   holds the committed GPU packet directly. Update may overlap prepare and
//!   render: both consume frozen geometry, never the live canvas lists. Canvas
//!   GPU metadata and lifetime changes still require context ownership/quiesce.
//!
//! Headless note: every `sg.*` WRITE sits behind `sg.isvalid()`, so unit
//! tests exercise capture/packet/policy purely on CPU. `capture` may query
//! read-only SDK frame metadata through `isUploadOpen` (no GPU writes,
//! headless-safe). The GPU half (growth VALID states, retire timing, real
//! upload bytes, drawable + 0-bytes draws, same-buffer failure) needs the
//! live harness (parent lane).

const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;

const ui_mod = @import("../ui.zig");
const UICanvas = ui_mod.UICanvas;
const UIVertex = ui_mod.UIVertex;
const math = @import("math");
const Color4 = math.Color4;
const gpu_thread = @import("../gpu_thread.zig");
const upload_meter = @import("../gpu_upload_meter.zig");
const gpu_retire = @import("gpu_retire.zig");

/// Result of `UiFrame.upload`: explicit so the same-frame policy and the
/// failure paths stay observable without a GPU.
pub const UploadResult = enum {
    /// Geometry uploaded: packet is drawable.
    uploaded,
    /// Frame empty, or packet already on the GPU: no `sg.*` touched.
    nothing_to_do,
    /// Staged packet predates an open window (another path uploaded this
    /// sokol frame): nothing uploaded, retry after a commit.
    deferred_open_window,
    /// No sokol context (headless/tools/tests): packet staged, not drawable.
    no_context,
    /// Growth failed: owned GPU pair untouched, frame dropped to
    /// coherent-empty, retry next frame.
    failed_coherent_empty,
};

/// Slot-owned UI CPU packet header (lock-free-publication slice 2, b):
/// the game side (`Scene.stageUiPacket`, sg-free) records live canvas CPU
/// geometry into the back `FrameDrawSlot` lists (`ui_vertices`/`ui_indices`)
/// and stamps this header; the prepare latch consumes the packet into
/// `UiFrame` instead of reading the live canvas lists. Plain CPU data only —
/// geometry + presence + staged draw handles (below). The game path never
/// uploads or mutates a GPU handle.
///
/// Handle-liveness contract: the staged handles are BORROWED values under
/// the same discipline as every other GPU handle in the slot (phase mutex /
/// P3 epochs — never destroyed/retired through the packet). The app must not
/// destroy the canvas pipeline/font/buffers between stage and latch; a
/// violation draws the same way a destroyed borrowed slot handle would
/// (caller obligation, like the mesh-queue borrowed handles).
pub const UiPacketHandles = struct {
    pipeline: sg.Pipeline = .{},
    font_view: sg.View = .{},
    font_sampler: sg.Sampler = .{},
    vertex_buffer: sg.Buffer = .{},
    index_buffer: sg.Buffer = .{},
};

pub const UiPacketState = struct {
    /// A stage published a packet (geometry or staged absence) for the latch.
    valid: bool = false,
    /// The canvas existed at stage time. Staged absence (false) latches to
    /// the same clear as the legacy missing-canvas path.
    canvas_present: bool = false,
    /// Draw handles frozen at stage time (pipeline, resolved font
    /// view/sampler, canvas buffer ids): the latch consumes these instead of
    /// reading the live canvas, so a canvas handle mutation between stage
    /// and latch cannot leak into this frame (staged wins — same precedent
    /// as the staged geometry). Unused on the staged-absence path.
    handles: UiPacketHandles = .{},
    /// Canvas `ui_upload_seq` at stage time: informational only (which canvas
    /// write the staged bytes came from; debuggability). The same-sokol-frame
    /// window identity (`decideWindow`) deliberately keeps reading the LIVE
    /// canvas: an intervening writer between stage and latch must invalidate
    /// the frame, and only the live watermark can show it.
    upload_seq: u64 = 0,
};

/// Context for the GPU half of capture. Mirrors P5's staging context: the
/// allocator funds the retire enqueue, the queue may be null for standalone
/// low-level callers (tests, one-off tooling) with the caller obligation
/// that no live snapshot still references the replaced buffers — legal
/// because the upload block already requires the context thread. A null
/// queue is NOT a guarantee about snapshots; the caller takes
/// responsibility for them.
pub const UiUploadContext = struct {
    allocator: std.mem.Allocator,
    retire_queue: ?*gpu_retire.GpuRetireQueue = null,
};

/// One prepared UI frame: frame-owned CPU copies plus the committed draw
/// packet describing exactly the GPU-resident data. Held directly by Scene
/// (single-frame ownership); never crosses a mailbox.
pub const UiFrame = struct {
    const Self = @This();

    // Frame-owned CPU copies (retained capacity across captures).
    vertices: std.ArrayListUnmanaged(UIVertex) = .empty,
    indices: std.ArrayListUnmanaged(u16) = .empty,

    // Committed draw packet: describes GPU-resident data, fixed for the
    // prepared frame. Published by capture, refreshed to post-growth IDs by
    // upload, consumed by drawPrepared.
    vert_count: usize = 0,
    index_count: usize = 0,
    screen_w: f32 = 0,
    screen_h: f32 = 0,
    pipeline: sg.Pipeline = .{},
    font_view: sg.View = .{},
    font_sampler: sg.Sampler = .{},
    vertex_buffer: sg.Buffer = .{},
    index_buffer: sg.Buffer = .{},

    /// A capture was published (CPU newest + packet staged).
    has_capture: bool = false,
    /// The staged packet is on the GPU (set by upload, live only).
    gpu_ready: bool = false,
    /// Staged packet not yet uploaded.
    needs_upload: bool = false,
    /// Committed-upload identity: canvas `ui_upload_seq` stamped by the
    /// successful `upload`, paired with the packet's buffer IDs. An
    /// open-window capture preserves first-wins only while this still
    /// identifies the canvas's last writer; any other writer (immediate
    /// render, growth replacement) invalidates it → fail-close empty.
    /// Reset by `clearEmpty` (Scene's missing-canvas/no-camera clears
    /// included) and by `deinit`.
    upload_seq: u64 = 0,
    /// A canvas existed at prepare (snapshotted by Scene, independent of
    /// nonempty/gpu_ready so the historical phantom+1 draw counters survive
    /// empty frames). Untouched by `clearEmpty` (content, not presence);
    /// reset by `deinit`.
    canvas_present: bool = false,

    /// Frees the frame-owned CPU copies. Borrowed GPU IDs are never
    /// destroyed here: buffers stay canvas-owned, retired olds belong to
    /// the retire queue — owners unchanged, no double-free.
    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        self.vertices.deinit(allocator);
        self.indices.deinit(allocator);
        self.* = .{};
    }

    /// Drops the frame CONTENT to coherent-empty, retaining capacity for
    /// the next capture. `canvas_present` is prepare-scoped metadata, not
    /// content, and stays untouched (Scene sets it explicitly every
    /// prepare, including the fail-close clears).
    pub fn clearEmpty(self: *Self) void {
        self.vertices.clearRetainingCapacity();
        self.indices.clearRetainingCapacity();
        self.vert_count = 0;
        self.index_count = 0;
        self.screen_w = 0;
        self.screen_h = 0;
        self.pipeline = .{};
        self.font_view = .{};
        self.font_sampler = .{};
        self.vertex_buffer = .{};
        self.index_buffer = .{};
        self.has_capture = false;
        self.gpu_ready = false;
        self.needs_upload = false;
        self.upload_seq = 0;
    }

    /// Shared window-policy entry for `capture` (live canvas) and
    /// `capturePacket` (slot-owned packet): FIRST-COMMITTED-WINS with
    /// the committed-upload identity check. `preserve_committed` leaves the
    /// whole frame untouched (early return, no allocation); `fail_closed`
    /// has already cleared to coherent-empty (early return); `proceed` runs
    /// the content guards + reserve + copy of the caller. Always reads the
    /// LIVE canvas: the window and the writer watermark are live context
    /// state (like the P3 epochs), never freezable.
    const WindowDecision = enum { preserve_committed, fail_closed, proceed };

    fn decideWindow(self: *Self, canvas: *const UICanvas) WindowDecision {
        // Whole-frame coherence first: nothing allocated, nothing touched
        // unless our own committed upload still owns the canvas buffers.
        if (canvas.isUploadOpen()) {
            const own_committed = self.gpu_ready and self.upload_seq != 0 and
                self.upload_seq == canvas.ui_upload_seq and
                self.vertex_buffer.id == canvas.vertex_buffer.id and
                self.index_buffer.id == canvas.index_buffer.id;
            if (!own_committed) self.clearEmpty();
            return if (own_committed) .preserve_committed else .fail_closed;
        }
        return .proceed;
    }

    /// Captures CPU geometry + draw parameters out of the live canvas.
    /// FIRST-COMMITTED-WINS: when the upload window is open (a committed
    /// upload already spent this sokol frame's single update), the recapture
    /// is dropped ENTIRELY — CPU copies, draw packet, counts and fields stay
    /// exactly the committed frame — but ONLY while the committed-upload
    /// identity still matches the canvas (sequence + buffer IDs): the frame
    /// must be `gpu_ready` from our own successful `upload` with no other
    /// writer since. A foreign write in between (immediate render
    /// overwriting the same buffers, or a growth replacement destroying
    /// them) fail-closes to coherent-empty — never a draw of destroyed
    /// bytes or wrong counts — with no re-upload until the next commit +
    /// capture, which recovers. All of this happens before any
    /// allocation/reserve or list overwrite. Missing content or invalid dims
    /// fail close to coherent-empty (explicit Scene policy for the
    /// empty/removal paths, taken only with a closed window). Reserves BOTH
    /// lists before publishing either (OOM → coherent empty, never a
    /// partial half). May query read-only SDK frame metadata via
    /// `isUploadOpen` (no GPU writes, headless-safe); otherwise no `sg.*`.
    pub fn capture(self: *Self, allocator: std.mem.Allocator, canvas: *const UICanvas, screen_w: f32, screen_h: f32) void {
        switch (self.decideWindow(canvas)) {
            .preserve_committed, .fail_closed => return,
            .proceed => {},
        }
        if (canvas.vertices.items.len == 0 or canvas.indices.items.len == 0) {
            self.clearEmpty();
            return;
        }
        if (screen_w <= 0.0 or screen_h <= 0.0) {
            self.clearEmpty();
            return;
        }
        self.captureSlices(
            allocator,
            canvas.vertices.items,
            canvas.indices.items,
            screen_w,
            screen_h,
            .{
                .pipeline = canvas.pipeline,
                .font_view = canvas.activeFontView(),
                .font_sampler = canvas.activeFontSampler(),
                .vertex_buffer = canvas.vertex_buffer,
                .index_buffer = canvas.index_buffer,
            },
        );
    }

    /// Captures out of a slot-owned game-side packet instead of the live
    /// canvas lists (lock-free-publication slice 2, b): the geometry AND the
    /// draw handles come from the staged packet (`Scene.stageUiPacket`),
    /// everything else — window policy, guards, upload identity — reads the
    /// live canvas exactly like `capture`. The canvas lists AND the canvas
    /// handles are never touched here, so a game-side mutation of the canvas
    /// between stage and latch cannot leak into this frame (staged bytes and
    /// staged handles win) and a canvas cleared after the stage still latches
    /// the staged content.
    pub fn capturePacket(
        self: *Self,
        allocator: std.mem.Allocator,
        canvas: *const UICanvas,
        verts: []const UIVertex,
        indices: []const u16,
        screen_w: f32,
        screen_h: f32,
        handles: UiPacketHandles,
    ) void {
        switch (self.decideWindow(canvas)) {
            .preserve_committed, .fail_closed => return,
            .proceed => {},
        }
        if (verts.len == 0 or indices.len == 0) {
            self.clearEmpty();
            return;
        }
        if (screen_w <= 0.0 or screen_h <= 0.0) {
            self.clearEmpty();
            return;
        }
        self.captureSlices(allocator, verts, indices, screen_w, screen_h, handles);
    }

    /// Shared content path of `capture`/`capturePacket`: reserves BOTH lists
    /// before publishing either (OOM → coherent empty, never a partial
    /// half), then copies + publishes the packet (clamped counts, dims,
    /// staged handles). Window + empty/dims guards ran in the caller.
    fn captureSlices(
        self: *Self,
        allocator: std.mem.Allocator,
        verts: []const UIVertex,
        indices: []const u16,
        screen_w: f32,
        screen_h: f32,
        handles: UiPacketHandles,
    ) void {
        const vert_count = UICanvas.clampedVertCount(verts.len);
        const index_count = indices.len;
        // Reserve both BEFORE publish: either both land or the frame is
        // coherent-empty (no old-index/new-vertex mix on OOM).
        self.vertices.ensureTotalCapacity(allocator, verts.len) catch {
            self.clearEmpty();
            return;
        };
        self.indices.ensureTotalCapacity(allocator, index_count) catch {
            self.clearEmpty();
            return;
        };
        self.vertices.clearRetainingCapacity();
        self.indices.clearRetainingCapacity();
        self.vertices.appendSlice(allocator, verts) catch {
            self.clearEmpty();
            return;
        };
        self.indices.appendSlice(allocator, indices) catch {
            self.clearEmpty();
            return;
        };
        self.vert_count = vert_count;
        self.index_count = index_count;
        self.screen_w = screen_w;
        self.screen_h = screen_h;
        self.pipeline = handles.pipeline;
        self.font_view = handles.font_view;
        self.font_sampler = handles.font_sampler;
        self.vertex_buffer = handles.vertex_buffer;
        self.index_buffer = handles.index_buffer;
        self.has_capture = true;
        self.gpu_ready = false;
        self.needs_upload = true;
    }

    /// GPU half of capture: ensures canvas buffers through the shared
    /// `ensureUiBufferPair` routine (all replacements created + VALIDated
    /// before any upload/commit/retire), uploads the staged packet, retires
    /// replaced buffers, arms the upload window. Idempotent: a second call
    /// without a new capture (or with an empty frame) touches no `sg.*`.
    /// No context → `no_context`.
    pub fn upload(self: *Self, canvas: *UICanvas, ctx: UiUploadContext) UploadResult {
        if (!self.has_capture or !self.needs_upload) return .nothing_to_do;
        if (canvas.isUploadOpen()) return .deferred_open_window;
        if (!sg.isvalid()) return .no_context;
        gpu_thread.assertOnContextThread();

        const ensured = UICanvas.ensureUiBufferPair(
            canvas.vertex_buffer,
            canvas.index_buffer,
            canvas.capacity_vertices,
            canvas.capacity_indices,
            self.vert_count,
            self.index_count,
        );
        if (!ensured.ok) {
            // Pair failure: current pair untouched (only uncommitted new
            // handles were destroyed above), frame dropped to
            // coherent-empty for a next-frame retry.
            self.clearEmpty();
            return .failed_coherent_empty;
        }

        // The frame's single update per buffer this sokol frame, into the
        // ensured (possibly replaced) targets.
        UICanvas.uploadUiBuffers(
            ensured.vertex_buffer,
            ensured.index_buffer,
            self.vertices.items[0..self.vert_count],
            self.indices.items[0..self.index_count],
        );
        canvas.markUiUploaded();

        // Commit: install ensured buffers + capacities, retire the replaced
        // pair (same epochs, no new queue/kind). Uploads above already
        // landed, so in-flight snapshots keep reading valid geometry.
        if (ensured.replaced_vb) {
            const old = canvas.vertex_buffer;
            canvas.vertex_buffer = ensured.vertex_buffer;
            canvas.capacity_vertices = ensured.capacity_vertices;
            if (old.id != 0) {
                if (ctx.retire_queue) |q| {
                    q.retireBuffer(ctx.allocator, old);
                } else {
                    sg.destroyBuffer(old);
                }
            }
        }
        if (ensured.replaced_ib) {
            const old = canvas.index_buffer;
            canvas.index_buffer = ensured.index_buffer;
            canvas.capacity_indices = ensured.capacity_indices;
            if (old.id != 0) {
                if (ctx.retire_queue) |q| {
                    q.retireBuffer(ctx.allocator, old);
                } else {
                    sg.destroyBuffer(old);
                }
            }
        }

        // Packet fixup: capture ran pre-growth, so refresh to the
        // installed (post-growth) IDs the draw must bind. The identity
        // stamps this exact upload (sequence + installed IDs).
        self.vertex_buffer = canvas.vertex_buffer;
        self.index_buffer = canvas.index_buffer;
        self.gpu_ready = true;
        self.needs_upload = false;
        self.upload_seq = canvas.ui_upload_seq;
        return .uploaded;
    }

    /// Upload-free draw of the committed packet. Draws nothing unless a
    /// capture was uploaded (never uploads, never records meter bytes).
    /// `isvalid`-gated: headless/tools get a safe no-op, never an assert
    /// trap and never a handmade readiness assert.
    pub fn drawPrepared(self: *const Self) void {
        if (!sg.isvalid()) return;
        if (!self.has_capture or !self.gpu_ready) return;
        if (self.vert_count == 0 or self.index_count == 0) return;
        if (self.screen_w <= 0.0 or self.screen_h <= 0.0) return;
        UICanvas.drawUiBuffers(
            self.pipeline,
            self.vertex_buffer,
            self.index_buffer,
            self.font_view,
            self.font_sampler,
            self.screen_w,
            self.screen_h,
            self.index_count,
        );
    }
};
