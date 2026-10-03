//! Clustered forward point lights, wave 30 (v1): a bounded, additive,
//! OFF-by-default pool of up to `lights.max_clustered_lights` (64) EXTRA
//! point lights beyond the legacy 4-slot top-k lanes.
//!
//! Data flow (1-frame lag, probe-pack pattern):
//! 1. Game side owns values (`LightRig.clustered`, fixed array + count).
//!    `Scene.addClusteredPointLight` returns an index; remove/get/count and
//!    per-light `enabled` follow the probe/area contracts (order-preserving
//!    removal, out-of-range no-op, hard `error.TooManyClusteredLights`).
//! 2. `LightRig.packFrame` stages every ENABLED clustered light verbatim
//!    into `FramePack` (appended last; zeroed lanes when empty, so the empty
//!    pool renders bit-identically). No hysteresis, no significance
//!    selection, no shadows — tiling does the culling, and v1 lights are
//!    unshadowed by design.
//! 3. The staged snapshot carries the pack (`SceneFrameSnapshot.light_pack`)
//!    into prepare/render. `Scene.renderSceneView` (context thread) rebuilds
//!    the 2D screen tiles from the STAGED copy (view-projection, light
//!    positions, ranges) — never live light state — uploads the storage
//!    buffers (metered), and binds them per draw.
//! 4. The five forward shaders map each pixel to its tile, iterate the
//!    tile's light indices, and apply the SAME point-light math as their
//!    legacy lanes (no shadows). Empty pool (count 0) skips the loop, so
//!    output is bit-identical to the legacy path.
//!
//! Tiling (2D screen columns, CPU build):
//! - Fixed 64x64 px tiles (`tile_size_px`) over the view's pixel rect.
//! - Per (tile, light): sphere-vs-4-side-plane test extracted from the
//!   staged view-projection (near/far ignored on purpose: 2D columns, so
//!   depth over-inclusion is documented, never under-inclusion — a
//!   behind-camera sphere overlapping the frustum is INCLUDED, a fully
//!   outside one is excluded, NaN positions exclude safely).
//! - CPU build is O(tiles x lights) per view (<= 64 lights by cap); exact
//!   buffer sizing (headers = tiles, indices <= tiles x count) keeps steady
//!   state allocation-free. Compute-shader build is an explicit non-goal.
//! - Multi-view (PIP, probe faces): tiles rebuild per rendered view from
//!   that view's staged camera. Probe captures render with the clustered
//!   lanes zeroed (legacy lanes only, documented v1 scope).
//!
//! GPU lifecycle (probe_layer precedent):
//! - All `sg.*` is context-thread only and gated on `sg.isvalid()`:
//!   headless rebuilds keep fresh CPU arrays while the GPU side fails
//!   closed (invalid cache binds the shared 16-byte dummy, count uniform 0
//!   takes the legacy shader path).
//! - Every `sg.updateBuffer` is paired with `upload_meter.record`.
//! - One buffer set per rendered view slot (`MAX_VIEW_SLOTS`): each view
//!   uploads only its own slot exactly once per frame, so no buffer is
//!   ever updated twice in one frame (sokol allows a single update per
//!   buffer per frame).
//! - Buffer replacement (viewport/light-count resize) retires the old
//!   buffer through `GpuRetireQueue.retireBuffer` (never destroys inline),
//!   and every light add/remove retires the live buffers first (uniform
//!   discipline like probe removal: stale GPU is gone before the next
//!   context rebuild recreates exact-fit buffers).
//! - Session-local like directional fills and area lights: serialization
//!   never persists the pool.
//!
//! Explicit non-goals (v1): shadows for clustered lights, depth-aware
//! (2.5D) tiles, hysteresis/fade, glTF import, persistence, compute build,
//! per-tile light caps (full lists; worst-case index bytes are
//! tiles x count x 4, documented), dirty-tracking (rebuild+upload run per
//! view while any clustered light is staged).

const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const math = @import("math");
const Vec3 = math.Vec3;
const Mat4 = math.Mat4;

const lights = @import("../lights.zig");
const compute = @import("../compute.zig");
const upload_meter = @import("../gpu_upload_meter.zig");
const gpu_thread = @import("../gpu_thread.zig");
const snapshot = @import("snapshot.zig");

/// One GPU buffer set per rendered view (slot 0 = primary, slots 1+ =
/// secondary PIP views, allocated lazily on first use). Sokol allows
/// exactly one `sg.updateBuffer` per buffer per frame, so multi-view
/// frames (primary + secondaries in ONE sokol frame) must never share a
/// buffer set across views: each view rebuilds its own tiles and uploads
/// only its own slot, giving the structural invariant that every slot's
/// buffers are updated at most once per frame.
// Captures must not reuse a primary/PIP buffer in the same sokol frame.
// Both extra slots allocate GPU buffers lazily, just like camera slots.
pub const REFRACTION_VIEW_SLOT: usize = snapshot.MAX_CAMERAS;
pub const RTT_VIEW_SLOT: usize = snapshot.MAX_CAMERAS + 1;
pub const MAX_VIEW_SLOTS: usize = snapshot.MAX_CAMERAS + 2;

/// Screen tile edge in pixels. Fixed: keeps the tile math, the shader
/// divisor, and the header sizing on one constant.
pub const tile_size_px: u32 = 64;

/// Staging copy of one clustered light for the tile build: position +
/// radius (w) and color + intensity (w). Mirrors the FramePack lanes.
pub const ClusteredLightStage = struct {
    pos_range: [4]f32,
    color_int: [4]f32,
};

/// GPU mirror of one clustered light (std430: two vec4, 32 bytes, stride
/// is a multiple of the 16-byte vec4 alignment). Uploaded verbatim from
/// the staged pack; the shader indexes it by the tile index list.
pub const ClusterLightGpu = struct {
    pos_range: [4]f32 = .{ 0, 0, 0, 0 },
    color_int: [4]f32 = .{ 0, 0, 0, 0 },
};

/// GPU mirror of one tile header (std430 uvec2: byte offset into the
/// index list + light count for the tile). Zeroed headers (count 0) make
/// the shader loop a no-op for empty tiles.
pub const ClusterTileGpu = struct {
    offset: u32 = 0,
    count: u32 = 0,
};

/// Grid dimensions for a viewport, in tiles (ceiling division). Zero in a
/// dimension when the viewport is empty — the build then writes nothing.
pub const TileGrid = struct {
    x: u32,
    y: u32,
};

pub fn tilesForViewport(screen_w: i32, screen_h: i32) TileGrid {
    if (screen_w <= 0 or screen_h <= 0) return .{ .x = 0, .y = 0 };
    const t: u32 = tile_size_px;
    const w: u32 = @intCast(screen_w);
    const h: u32 = @intCast(screen_h);
    return .{ .x = (w + t - 1) / t, .y = (h + t - 1) / t };
}

/// Pixel rect of one rendered view inside the window (sokol viewport
/// convention: origin at the bottom-left, y up). Tiles live on the WINDOW
/// grid (gl_FragCoord space, y up on GL; the shader flips y for
/// Metal/D3D), while each tile's NDC bounds map through its view rect —
/// so fullscreen and sub-viewport (PIP) views tile correctly from the same
/// staged snapshot.
pub const ViewRect = struct {
    x: i32 = 0,
    y: i32 = 0,
    w: i32 = 0,
    h: i32 = 0,
};

/// NDC rect (x0, x1, y0, y1, y-up) of window-grid tile (tx, ty). Tile
/// (0, 0) is the bottom-left window tile, matching the gl_FragCoord
/// convention the shader uses. Tiles outside the view rect map outside
/// [-1, 1] (valid planes; they simply collect only huge lights).
/// Degenerate view rects yield a zero rect (planes go no-constraint, i.e.
/// over-include — the view renders nothing anyway).
pub fn tileNdcRect(tx: u32, ty: u32, view: ViewRect, win_w: i32, win_h: i32) [4]f32 {
    const t: f32 = @floatFromInt(tile_size_px);
    const fx: f32 = @floatFromInt(tx);
    const fy: f32 = @floatFromInt(ty);
    const vx: f32 = @floatFromInt(view.x);
    const vy: f32 = @floatFromInt(view.y);
    const vw: f32 = @floatFromInt(@max(view.w, 0));
    const vh: f32 = @floatFromInt(@max(view.h, 0));
    const ww: f32 = @floatFromInt(@max(win_w, 0));
    const wh: f32 = @floatFromInt(@max(win_h, 0));
    if (!(vw > 0.0) or !(vh > 0.0)) return .{ 0, 0, 0, 0 };
    const px0 = fx * t;
    const px1 = @min(fx * t + t, ww);
    const py0 = fy * t;
    const py1 = @min(fy * t + t, wh);
    return .{
        (px0 - vx) / vw * 2.0 - 1.0,
        (px1 - vx) / vw * 2.0 - 1.0,
        (py0 - vy) / vh * 2.0 - 1.0,
        (py1 - vy) / vh * 2.0 - 1.0,
    };
}

/// True when the sphere (center, radius) overlaps the tile frustum column
/// defined by the NDC rect. Tests only the four side planes extracted from
/// the view-projection rows (near/far ignored: 2D columns, documented
/// over-inclusion). Provably over-inclusive-only: a fully outside sphere is
/// excluded, everything else (including behind-camera spheres reaching into
/// the frustum) is kept. NaN centers exclude safely (comparisons false).
fn sphereOverlapsTileColumn(vp: Mat4, rect: [4]f32, center: Vec3, radius: f32) bool {
    // VP rows (column-major m): row_i = (m[i], m[i+4], m[i+8], m[i+12]).
    const r0 = [_]f32{ vp.m[0], vp.m[4], vp.m[8], vp.m[12] };
    const r1 = [_]f32{ vp.m[1], vp.m[5], vp.m[9], vp.m[13] };
    const r3 = [_]f32{ vp.m[3], vp.m[7], vp.m[11], vp.m[15] };
    // Inside half-spaces: (r0 - x*r3).p >= 0, (x*r3 - r0).p >= 0, same for y.
    const planes = [_][4]f32{
        .{ r0[0] - rect[0] * r3[0], r0[1] - rect[0] * r3[1], r0[2] - rect[0] * r3[2], r0[3] - rect[0] * r3[3] },
        .{ rect[1] * r3[0] - r0[0], rect[1] * r3[1] - r0[1], rect[1] * r3[2] - r0[2], rect[1] * r3[3] - r0[3] },
        .{ r1[0] - rect[2] * r3[0], r1[1] - rect[2] * r3[1], r1[2] - rect[2] * r3[2], r1[3] - rect[2] * r3[3] },
        .{ rect[3] * r3[0] - r1[0], rect[3] * r3[1] - r1[1], rect[3] * r3[2] - r1[2], rect[3] * r3[3] - r1[3] },
    };
    const p = [_]f32{ center.x, center.y, center.z, 1.0 };
    for (planes) |pl| {
        const n_len = @sqrt(pl[0] * pl[0] + pl[1] * pl[1] + pl[2] * pl[2]);
        if (!(n_len > 0.0)) continue; // Degenerate plane: no constraint.
        const dist = (pl[0] * p[0] + pl[1] * p[1] + pl[2] * p[2] + pl[3] * p[3]) / n_len;
        if (!(dist >= -radius)) return false;
    }
    return true;
}

/// Builds per-tile clustered-light index lists from staged pack data.
/// Tiles row-major (tile id = ty * tiles_x + tx); indices within a tile in
/// light order — fully deterministic for a given pack + camera.
/// Skips lanes with intensity <= 0 or radius <= 0 (disabled/degenerate pack
/// as absent, mirroring the shader range gate). `headers` (len
/// tiles_x*tiles_y) is zeroed here; `indices` must hold tiles_x*tiles_y*count
/// worst case (exact-fit sizing by the caller) — overflow saturates
/// defensively (unreachable at exact-fit sizing). Returns indices written.
/// Pure (no `sg.*`): headless-testable and safe on any thread.
pub fn buildTileLists(
    pos_range: []const [4]f32,
    color_int: []const [4]f32,
    count: usize,
    view_proj: Mat4,
    tiles_x: u32,
    tiles_y: u32,
    view: ViewRect,
    win_w: i32,
    win_h: i32,
    headers: [][2]u32,
    indices: []u32,
) usize {
    const tiles: usize = @as(usize, tiles_x) * @as(usize, tiles_y);
    std.debug.assert(headers.len >= tiles);
    for (headers[0..tiles]) |*h| h.* = .{ 0, 0 };
    if (tiles == 0 or count == 0) return 0;
    const n = @min(count, pos_range.len, color_int.len, lights.max_clustered_lights);
    var written: usize = 0;
    var ty: u32 = 0;
    while (ty < tiles_y) : (ty += 1) {
        var tx: u32 = 0;
        while (tx < tiles_x) : (tx += 1) {
            const tid: usize = @as(usize, ty) * @as(usize, tiles_x) + @as(usize, tx);
            const rect = tileNdcRect(tx, ty, view, win_w, win_h);
            const start = written;
            var li: usize = 0;
            while (li < n) : (li += 1) {
                const pr = pos_range[li];
                const ci = color_int[li];
                if (!(ci[3] > 0.0) or !(pr[3] > 0.0)) continue;
                const center = Vec3.new(pr[0], pr[1], pr[2]);
                if (!sphereOverlapsTileColumn(view_proj, rect, center, pr[3])) continue;
                if (written >= indices.len) break; // Defensive saturate.
                indices[written] = @intCast(li);
                written += 1;
            }
            headers[tid] = .{ @intCast(start), @intCast(written - start) };
        }
    }
    return written;
}

/// View triple for one draw (see ClusteredGpuCache.bindingViews).
pub const ClusterBindingViews = struct {
    lights: sg.View = .{},
    tiles: sg.View = .{},
    indices: sg.View = .{},
};

/// One view's GPU buffer set: the three storage buffers (+ views) the
/// forward shaders read for that view's tiles. Slot 0 is the primary
/// view; slots 1+ are secondary views, allocated lazily on first use
/// (single-camera frames touch only slot 0 — no VRAM/behavior change
/// for the single-view path). Each slot's buffers are updated at most
/// once per frame because each view uploads its own slot exactly once.
pub const ClusteredViewSlot = struct {
    light_buffer: sg.Buffer = .{},
    header_buffer: sg.Buffer = .{},
    index_buffer: sg.Buffer = .{},
    light_view: sg.View = .{},
    header_view: sg.View = .{},
    index_view: sg.View = .{},
    light_cap: usize = 0,
    header_cap: usize = 0,
    index_cap: usize = 0,
    /// True when this slot's GPU buffers mirror that view's last upload
    /// (bind real views). False binds the shared dummy (fail-closed:
    /// legacy shader path) while other slots keep their own state.
    live: bool = false,
};

/// Render-owned clustered GPU state: CPU tile scratch plus one storage
/// buffer set PER VIEW SLOT (+ views) the forward shaders read, and a
/// shared 16-byte dummy bound whenever no real data is live (so declared
/// SSBO slots are always valid on a live context, even on the legacy
/// path).
/// All `sg.*` happens on the context thread (`upload`/`deinit`);
/// `rebuildCpu`/`retireBuffers` are sg-free (any thread, headless-safe).
pub const ClusteredGpuCache = struct {
    // CPU scratch (rebuilt per view while any clustered light is staged;
    // shared across views: each view rebuilds then immediately uploads
    // its own slot before the next view rebuilds).
    cpu_lights: std.ArrayListUnmanaged(ClusterLightGpu) = .empty,
    cpu_headers: std.ArrayListUnmanaged(ClusterTileGpu) = .empty,
    cpu_indices: std.ArrayListUnmanaged(u32) = .empty,
    tiles_x: u32 = 0,
    tiles_y: u32 = 0,
    staged_count: usize = 0,
    // GPU storage per view slot (exact-fit; replaced buffers retire,
    // never destroy inline).
    slots: [MAX_VIEW_SLOTS]ClusteredViewSlot = [_]ClusteredViewSlot{.{}} ** MAX_VIEW_SLOTS,
    dummy_buffer: sg.Buffer = .{},
    dummy_view: sg.View = .{},

    /// Clamps an arbitrary view slot to the slot range (defensive: all
    /// in-tree callers already pass a valid slot; out-of-range input
    /// addresses the last slot instead of trapping).
    pub fn clampSlot(view_slot: usize) usize {
        return @min(view_slot, MAX_VIEW_SLOTS - 1);
    }

    /// True when `view_slot`'s GPU buffers mirror that view's last upload
    /// (that view's draws bind the real storage views). Pure (no `sg.*`).
    pub fn isLive(self: *const ClusteredGpuCache, view_slot: usize) bool {
        return self.slots[clampSlot(view_slot)].live;
    }

    /// Rebuilds the CPU tile scratch from staged pack data for one view.
    /// Headless-safe (no `sg.*`); marks that view's GPU slot stale (its
    /// `live` goes false) so the next context `upload` for the same slot
    /// refreshes it — other slots keep their own liveness (their draws
    /// already ran or are still pending with their own buffers intact).
    /// Never fails on empty input (zeroed headers, empty indices).
    pub fn rebuildCpu(
        self: *ClusteredGpuCache,
        allocator: std.mem.Allocator,
        pos_range: []const [4]f32,
        color_int: []const [4]f32,
        count: usize,
        view_proj: Mat4,
        win_w: i32,
        win_h: i32,
        view: ViewRect,
    ) !void {
        return self.rebuildCpuForSlot(
            allocator,
            pos_range,
            color_int,
            count,
            view_proj,
            win_w,
            win_h,
            view,
            0,
        );
    }

    /// Slot-explicit rebuild: same as `rebuildCpu` but clears the liveness
    /// of `view_slot` (the view being rebuilt). View callers pass their
    /// own slot so a secondary rebuild never clears the primary's live
    /// flag (or vice versa).
    pub fn rebuildCpuForSlot(
        self: *ClusteredGpuCache,
        allocator: std.mem.Allocator,
        pos_range: []const [4]f32,
        color_int: []const [4]f32,
        count: usize,
        view_proj: Mat4,
        win_w: i32,
        win_h: i32,
        view: ViewRect,
        view_slot: usize,
    ) !void {
        const grid = tilesForViewport(win_w, win_h);
        self.tiles_x = grid.x;
        self.tiles_y = grid.y;
        self.slots[clampSlot(view_slot)].live = false;
        const tiles: usize = @as(usize, grid.x) * @as(usize, grid.y);
        const n = @min(count, pos_range.len, color_int.len, lights.max_clustered_lights);
        self.staged_count = n;

        self.cpu_lights.clearRetainingCapacity();
        var li: usize = 0;
        while (li < n) : (li += 1) {
            try self.cpu_lights.append(allocator, .{ .pos_range = pos_range[li], .color_int = color_int[li] });
        }

        self.cpu_headers.clearRetainingCapacity();
        try self.cpu_headers.ensureTotalCapacity(allocator, tiles);
        self.cpu_headers.items.len = tiles;
        for (self.cpu_headers.items) |*h| h.* = .{};

        self.cpu_indices.clearRetainingCapacity();
        try self.cpu_indices.ensureTotalCapacity(allocator, tiles * n);
        self.cpu_indices.items.len = tiles * n;

        var as_pairs: [][2]u32 = &.{};
        if (tiles > 0) {
            // ClusterTileGpu is two packed u32 (offset, count): reinterpret
            // as the pairs buildTileLists writes (layout pinned below).
            as_pairs = @ptrCast(self.cpu_headers.items);
        }
        const written = buildTileLists(pos_range, color_int, n, view_proj, grid.x, grid.y, view, win_w, win_h, as_pairs, self.cpu_indices.items);
        self.cpu_indices.items.len = written;
    }

    /// Ensures the shared 16-byte dummy buffer + view exist so draws can
    /// bind valid views on the legacy path. Context thread only;
    /// fail-closed headless (no `sg.*`, dummy stays zero).
    pub fn ensureDummyViews(self: *ClusteredGpuCache) void {
        gpu_thread.assertOnContextThread();
        if (!sg.isvalid()) return;
        self.ensureDummy();
    }

    /// Creates/refreshes `view_slot`'s GPU buffers from the CPU scratch
    /// (which the caller just rebuilt for that view). Context thread only;
    /// fails closed headless (`!sg.isvalid()` keeps everything staged for
    /// a later retry). Replaced buffers retire through `retire_queue`
    /// (same contract as instance-staging growth). Every
    /// `sg.updateBuffer` is metered. Only this slot's three buffers are
    /// touched — so each slot is updated at most once per frame (its view
    /// uploads exactly once) and the sokol one-update-per-buffer rule
    /// holds under multi-view. Returns true when the dummy is live so the
    /// draw can bind unconditionally (a sizing failure for THIS slot
    /// returns false and that view falls back to the dummy/legacy path
    /// while other slots keep their own state).
    pub fn upload(self: *ClusteredGpuCache, allocator: std.mem.Allocator, retire_queue: anytype, view_slot: usize) bool {
        gpu_thread.assertOnContextThread();
        if (!sg.isvalid()) return false;
        self.ensureDummy();
        if (self.dummy_view.id == 0) return false;
        const s = clampSlot(view_slot);
        if (!self.ensureSized(allocator, retire_queue, s)) return false;
        const slot = &self.slots[s];

        if (self.cpu_lights.items.len > 0) {
            sg.updateBuffer(slot.light_buffer, sg.asRange(self.cpu_lights.items));
            upload_meter.record(self.cpu_lights.items.len * @sizeOf(ClusterLightGpu));
        }
        if (self.cpu_headers.items.len > 0) {
            sg.updateBuffer(slot.header_buffer, sg.asRange(self.cpu_headers.items));
            upload_meter.record(self.cpu_headers.items.len * @sizeOf(ClusterTileGpu));
        }
        if (self.cpu_indices.items.len > 0) {
            sg.updateBuffer(slot.index_buffer, sg.asRange(self.cpu_indices.items));
            upload_meter.record(self.cpu_indices.items.len * @sizeOf(u32));
        }
        self.ensureViews(s);
        slot.live = slot.light_view.id != 0 and slot.header_view.id != 0 and slot.index_view.id != 0;
        return self.dummy_view.id != 0;
    }

    /// View triple for one draw from slot 0 (the primary view): the real
    /// storage views when that slot mirrors its last upload, else the
    /// shared dummy on all three slots (declared SSBO slots stay valid;
    /// the count uniform keeps the shader on the legacy path). Pure (no
    /// `sg.*`). Multi-view draws use `bindingViewsForSlot` with their own
    /// slot instead.
    pub fn bindingViews(self: *const ClusteredGpuCache) ClusterBindingViews {
        return self.bindingViewsForSlot(0);
    }

    /// View triple for one draw from `view_slot`: the real storage views
    /// when THAT slot mirrors its view's last upload, else the shared
    /// dummy (a failing secondary falls back to legacy while the primary
    /// keeps clustered). Pure (no `sg.*`).
    pub fn bindingViewsForSlot(self: *const ClusteredGpuCache, view_slot: usize) ClusterBindingViews {
        const slot = &self.slots[clampSlot(view_slot)];
        if (slot.live) {
            return .{ .lights = slot.light_view, .tiles = slot.header_view, .indices = slot.index_view };
        }
        return .{ .lights = self.dummy_view, .tiles = self.dummy_view, .indices = self.dummy_view };
    }

    /// Retires every allocated slot's three live storage buffers through
    /// the epoch retire queue (sg-free: only an epoch stamp + append per
    /// buffer, like probe removal) and zeroes the handles/views/caps plus
    /// all liveness flags. Called on every clustered add/remove so stale
    /// GPU can never serve a newer frame; the next context rebuild
    /// recreates exact-fit buffers per slot on demand. Safe on any
    /// thread, headless-safe (untouched secondary slots hold zero ids and
    /// are skipped — only allocated buffers reach the queue).
    pub fn retireBuffers(self: *ClusteredGpuCache, allocator: std.mem.Allocator, retire_queue: anytype) void {
        for (&self.slots) |*slot| {
            if (slot.light_buffer.id != 0) retire_queue.retireBuffer(allocator, slot.light_buffer);
            if (slot.header_buffer.id != 0) retire_queue.retireBuffer(allocator, slot.header_buffer);
            if (slot.index_buffer.id != 0) retire_queue.retireBuffer(allocator, slot.index_buffer);
            slot.light_buffer = .{};
            slot.header_buffer = .{};
            slot.index_buffer = .{};
            slot.light_view = .{};
            slot.header_view = .{};
            slot.index_view = .{};
            slot.light_cap = 0;
            slot.header_cap = 0;
            slot.index_cap = 0;
            slot.live = false;
        }
    }

    /// Context-thread teardown: destroys every allocated slot's views
    /// (guarded: view destroys assert a valid context, probe precedent)
    /// then buffers, plus the shared dummy; frees CPU scratch. Pending
    /// retire entries for already-removed buffers drain via
    /// `GpuRetireQueue.deinit` separately — no double destroy (handles
    /// were zeroed at retire time).
    pub fn deinit(self: *ClusteredGpuCache, allocator: std.mem.Allocator) void {
        gpu_thread.assertOnContextThread();
        if (sg.isvalid()) {
            for (&self.slots) |*slot| {
                for ([_]*sg.View{ &slot.light_view, &slot.header_view, &slot.index_view }) |v| {
                    if (v.*.id != 0) {
                        sg.destroyView(v.*);
                        v.* = .{};
                    }
                }
            }
            if (self.dummy_view.id != 0) {
                sg.destroyView(self.dummy_view);
                self.dummy_view = .{};
            }
        } else {
            for (&self.slots) |*slot| {
                slot.light_view = .{};
                slot.header_view = .{};
                slot.index_view = .{};
            }
            self.dummy_view = .{};
        }
        for (&self.slots) |*slot| {
            for ([_]*sg.Buffer{ &slot.light_buffer, &slot.header_buffer, &slot.index_buffer }) |b| {
                if (b.*.id != 0) {
                    sg.destroyBuffer(b.*);
                    b.* = .{};
                }
            }
        }
        if (self.dummy_buffer.id != 0) {
            sg.destroyBuffer(self.dummy_buffer);
            self.dummy_buffer = .{};
        }
        self.cpu_lights.deinit(allocator);
        self.cpu_headers.deinit(allocator);
        self.cpu_indices.deinit(allocator);
        self.* = .{};
    }

    fn ensureDummy(self: *ClusteredGpuCache) void {
        if (self.dummy_buffer.id == 0) {
            self.dummy_buffer = sg.makeBuffer(.{
                .usage = .{ .storage_buffer = true },
                .size = 64,
            });
        }
        if (self.dummy_buffer.id != 0 and self.dummy_view.id == 0) {
            self.dummy_view = compute.makeStorageView(self.dummy_buffer, "clustered-dummy");
        }
    }

    /// Sizes `view_slot`'s three buffers to the current CPU scratch
    /// (power-of-two growth with a 64 B floor, minimum one element so the
    /// views are always bindable). Secondary slots allocate lazily here on
    /// first use; untouched slots hold no buffers. Growth retires the old
    /// buffer through the same `gpu_retire` queue as before (never destroys
    /// inline) and drops its storage view so `ensureViews` rebuilds it
    /// against the new buffer. Failed creation (buffer id 0 / pool
    /// exhaustion) fails that slot closed.
    fn ensureSized(self: *ClusteredGpuCache, allocator: std.mem.Allocator, retire_queue: anytype, view_slot: usize) bool {
        const slot = &self.slots[clampSlot(view_slot)];
        const need_lights = @max(self.cpu_lights.items.len, 1);
        const need_headers = @max(self.cpu_headers.items.len, 1);
        const need_indices = @max(self.cpu_indices.items.len, 1);
        if (!ensureBuffer(allocator, retire_queue, &slot.light_buffer, &slot.light_view, &slot.light_cap, need_lights * @sizeOf(ClusterLightGpu))) return false;
        if (!ensureBuffer(allocator, retire_queue, &slot.header_buffer, &slot.header_view, &slot.header_cap, need_headers * @sizeOf(ClusterTileGpu))) return false;
        if (!ensureBuffer(allocator, retire_queue, &slot.index_buffer, &slot.index_view, &slot.index_cap, need_indices * @sizeOf(u32))) return false;
        return true;
    }

    /// Growth size: next power of two with a 64 B floor. A camera sweep
    /// grows the index scratch as lights sweep across tiles; exact-fit
    /// sizing made every crossing a retire+create churn event (buffer-pool
    /// pressure + view churn), pow2 bounds it to a handful per session.
    fn growBytes(need_bytes: usize) usize {
        const floor_bytes: usize = 64;
        if (need_bytes <= floor_bytes) return floor_bytes;
        return std.math.ceilPowerOfTwo(usize, need_bytes) catch need_bytes;
    }

    fn ensureBuffer(
        allocator: std.mem.Allocator,
        retire_queue: anytype,
        buffer: *sg.Buffer,
        view: *sg.View,
        cap: *usize,
        need_bytes: usize,
    ) bool {
        if (buffer.*.id != 0 and cap.* >= need_bytes) return true;
        if (buffer.*.id != 0) {
            retire_queue.retireBuffer(allocator, buffer.*);
            buffer.* = .{};
            // Drop the storage view together with its buffer: once the
            // retire queue flushes, the stale view is dead at bind time
            // (VALIDATE_ABND_VIEW_ALIVE) and a non-zero stale view id made
            // `ensureViews` skip recreation while `slot.live` stayed true —
            // every clustered draw then failed sg_apply_bindings and drew
            // nothing (the "clustered ON + rotate camera = empty screen"
            // bisect; samplers in the same failing bindings were
            // collateral validation output). ensureViews recreates the
            // view against the new buffer on the upload below.
            view.* = .{};
            cap.* = 0;
        }
        const grow = growBytes(need_bytes);
        buffer.* = sg.makeBuffer(.{
            .usage = .{ .storage_buffer = true, .dynamic_update = true },
            .size = grow,
        });
        if (buffer.*.id == 0) return false;
        cap.* = grow;
        return true;
    }

    fn ensureViews(self: *ClusteredGpuCache, view_slot: usize) void {
        const slot = &self.slots[clampSlot(view_slot)];
        if (slot.light_view.id == 0 and slot.light_buffer.id != 0) {
            slot.light_view = compute.makeStorageView(slot.light_buffer, "clustered-lights-view");
        }
        if (slot.header_view.id == 0 and slot.header_buffer.id != 0) {
            slot.header_view = compute.makeStorageView(slot.header_buffer, "clustered-tiles-view");
        }
        if (slot.index_view.id == 0 and slot.index_buffer.id != 0) {
            slot.index_view = compute.makeStorageView(slot.index_buffer, "clustered-indices-view");
        }
    }
};

test "growBytes rounds up to pow2 with a 64 B floor" {
    try std.testing.expectEqual(@as(usize, 64), ClusteredGpuCache.growBytes(0));
    try std.testing.expectEqual(@as(usize, 64), ClusteredGpuCache.growBytes(1));
    try std.testing.expectEqual(@as(usize, 64), ClusteredGpuCache.growBytes(64));
    try std.testing.expectEqual(@as(usize, 128), ClusteredGpuCache.growBytes(65));
    try std.testing.expectEqual(@as(usize, 256), ClusteredGpuCache.growBytes(200));
    try std.testing.expectEqual(@as(usize, 4096), ClusteredGpuCache.growBytes(2132));
    try std.testing.expectEqual(@as(usize, 8192), ClusteredGpuCache.growBytes(6240));
}

test "tilesForViewport is ceiling division with empty-viewport zero" {
    try std.testing.expectEqual(TileGrid{ .x = 0, .y = 0 }, tilesForViewport(0, 480));
    try std.testing.expectEqual(TileGrid{ .x = 0, .y = 0 }, tilesForViewport(640, 0));
    try std.testing.expectEqual(TileGrid{ .x = 0, .y = 0 }, tilesForViewport(-3, 480));
    try std.testing.expectEqual(TileGrid{ .x = 1, .y = 1 }, tilesForViewport(64, 64));
    try std.testing.expectEqual(TileGrid{ .x = 2, .y = 2 }, tilesForViewport(128, 65));
    try std.testing.expectEqual(TileGrid{ .x = 30, .y = 17 }, tilesForViewport(1920, 1080));
}

test "tileNdcRect covers the full NDC box bottom-left first" {
    const full = ViewRect{ .x = 0, .y = 0, .w = 128, .h = 128 };
    // 2x2 grid over a 128px window: tile (0,0) is bottom-left
    // [-1,0]x[-1,0], (1,1) top-right.
    try std.testing.expectEqual([4]f32{ -1, 0, -1, 0 }, tileNdcRect(0, 0, full, 128, 128));
    try std.testing.expectEqual([4]f32{ 0, 1, 0, 1 }, tileNdcRect(1, 1, full, 128, 128));
    try std.testing.expectEqual([4]f32{ -1, 0, 0, 1 }, tileNdcRect(0, 1, full, 128, 128));
    // Edge tiles meet exactly at the shared border (no gaps/overlaps).
    const wide = ViewRect{ .x = 0, .y = 0, .w = 256, .h = 64 };
    const left = tileNdcRect(0, 0, wide, 256, 64);
    const right = tileNdcRect(1, 0, wide, 256, 64);
    try std.testing.expectEqual(left[1], right[0]);
    // Sub-viewport: a left-half view maps its own pixels to full NDC.
    const half = ViewRect{ .x = 0, .y = 0, .w = 64, .h = 128 };
    try std.testing.expectEqual([4]f32{ -1, 1, -1, 0 }, tileNdcRect(0, 0, half, 128, 128));
    // Degenerate view rect: zero rect (planes go no-constraint upstream).
    try std.testing.expectEqual([4]f32{ 0, 0, 0, 0 }, tileNdcRect(0, 0, .{}, 128, 128));
}

test "buildTileLists assigns center lights, excludes off-screen and dead lanes" {
    // Identity view-projection: world == NDC (w = 1 everywhere).
    const vp = Mat4.identity;
    const full = ViewRect{ .x = 0, .y = 0, .w = 128, .h = 128 };
    const tx: u32 = 2;
    const ty: u32 = 2;
    var headers = [_][2]u32{.{ 0, 0 }} ** 4;
    var indices = [_]u32{0} ** 16;
    const pos = [_][4]f32{
        .{ 0.75, 0.75, 0.0, 0.5 }, // small light deep inside tile (1,1)
        .{ 5.0, 5.0, 0.0, 0.5 }, // off-screen entirely: excluded
        .{ -0.75, -0.75, 0.0, 0.5 }, // small light deep inside tile (0,0)
    };
    const col = [_][4]f32{
        .{ 1, 0, 0, 2.0 },
        .{ 0, 1, 0, 2.0 },
        .{ 0, 0, 1, 2.0 },
    };
    const n = buildTileLists(&pos, &col, 3, vp, tx, ty, full, 128, 128, &headers, &indices);
    // Light 0 only in tile (1,1) [id 3], light 2 only in tile (0,0) [id 0].
    try std.testing.expectEqual(@as(u32, 1), headers[3][1]);
    try std.testing.expectEqual(@as(u32, 1), headers[0][1]);
    try std.testing.expectEqual(@as(u32, 0), headers[1][1]);
    try std.testing.expectEqual(@as(u32, 0), headers[2][1]);
    try std.testing.expectEqual(@as(usize, 2), n);
    try std.testing.expectEqual(@as(u32, 2), indices[headers[0][0]]);
    try std.testing.expectEqual(@as(u32, 0), indices[headers[3][0]]);

    // Disabled lane (intensity 0) and degenerate radius pack as absent.
    const col_off = [_][4]f32{
        .{ 1, 0, 0, 0.0 },
        .{ 0, 1, 0, 2.0 },
        .{ 0, 0, 1, 2.0 },
    };
    const pos_flat = [_][4]f32{
        .{ 0.75, 0.75, 0.0, 0.0 },
        .{ 5.0, 5.0, 0.0, 0.5 },
        .{ -0.75, -0.75, 0.0, 0.5 },
    };
    const n2 = buildTileLists(&pos_flat, &col_off, 3, vp, tx, ty, full, 128, 128, &headers, &indices);
    try std.testing.expectEqual(@as(usize, 1), n2);
    try std.testing.expectEqual(@as(u32, 0), headers[3][1]);
    try std.testing.expectEqual(@as(u32, 2), indices[headers[0][0]]);
}

test "buildTileLists perspective: front lights assign, behind lights exclude unless huge" {
    const proj = Mat4.perspective(90.0, 1.0, 0.1, 100.0);
    const view = Mat4.lookAt(Vec3.zero, Vec3.new(0, 0, -1), Vec3.up);
    const vp = Mat4.mul(proj, view);
    const full = ViewRect{ .x = 0, .y = 0, .w = 128, .h = 128 };
    var headers = [_][2]u32{.{ 0, 0 }} ** 4;
    var indices = [_]u32{0} ** 8;
    // Light 5 units ahead: center tile column (x=0 spans tiles (0,*) and (1,*)).
    const pos = [_][4]f32{
        .{ 0, 0, -5, 1.0 },
        .{ 0, 0, 5, 1.0 }, // behind the eye, small: excluded everywhere
        .{ 0, 0, 5, 100.0 }, // behind but enormous: over-included (documented)
    };
    const col = [_][4]f32{
        .{ 1, 1, 1, 1.0 },
        .{ 1, 1, 1, 1.0 },
        .{ 1, 1, 1, 1.0 },
    };
    const n = buildTileLists(&pos, &col, 3, vp, 2, 2, full, 128, 128, &headers, &indices);
    var front_hits: u32 = 0;
    var behind_small_hits: u32 = 0;
    var behind_huge_hits: u32 = 0;
    for (headers) |h| {
        for (indices[h[0] .. h[0] + h[1]]) |li| {
            if (li == 0) front_hits += 1;
            if (li == 1) behind_small_hits += 1;
            if (li == 2) behind_huge_hits += 1;
        }
    }
    try std.testing.expect(front_hits > 0);
    try std.testing.expectEqual(@as(u32, 0), behind_small_hits);
    try std.testing.expect(behind_huge_hits > 0);
    try std.testing.expectEqual(@as(usize, front_hits + behind_huge_hits), n);
}

test "buildTileLists follows each view's own camera (PIP views independent)" {
    const proj = Mat4.perspective(90.0, 1.0, 0.1, 100.0);
    const full = ViewRect{ .x = 0, .y = 0, .w = 128, .h = 128 };
    const pos = [_][4]f32{.{ 0, 0, -5, 1.0 }};
    const col = [_][4]f32{.{ 1, 1, 1, 1.0 }};
    var headers = [_][2]u32{.{ 0, 0 }} ** 4;
    var indices = [_]u32{0} ** 4;
    // Camera A faces the light: some tile lists it.
    const vp_a = Mat4.mul(proj, Mat4.lookAt(Vec3.zero, Vec3.new(0, 0, -1), Vec3.up));
    const na = buildTileLists(&pos, &col, 1, vp_a, 2, 2, full, 128, 128, &headers, &indices);
    try std.testing.expect(na > 0);
    // Camera B faces away (light fully behind it): nothing lists it, so a
    // PIP view never inherits another view's tile lists.
    const vp_b = Mat4.mul(proj, Mat4.lookAt(Vec3.zero, Vec3.new(0, 0, 1), Vec3.up));
    const nb = buildTileLists(&pos, &col, 1, vp_b, 2, 2, full, 128, 128, &headers, &indices);
    try std.testing.expectEqual(@as(usize, 0), nb);
    for (headers) |h| try std.testing.expectEqual(@as(u32, 0), h[1]);
}

test "buildTileLists is empty-safe and deterministic" {
    var headers = [_][2]u32{.{ 9, 9 }} ** 1;
    var indices = [_]u32{0} ** 1;
    const full = ViewRect{ .x = 0, .y = 0, .w = 64, .h = 64 };
    // Empty viewport: nothing written, headers zeroed.
    const n0 = buildTileLists(&.{}, &.{}, 0, Mat4.identity, 0, 0, .{}, 0, 0, &.{}, &.{});
    try std.testing.expectEqual(@as(usize, 0), n0);
    // Zero lights on a live grid: headers zeroed, no indices.
    const n1 = buildTileLists(&.{}, &.{}, 0, Mat4.identity, 1, 1, full, 64, 64, &headers, &indices);
    try std.testing.expectEqual(@as(usize, 0), n1);
    try std.testing.expectEqual([2]u32{ 0, 0 }, headers[0]);
}

test "gpu mirror structs match the std430 shader layout" {
    // ClusterLightGpu == struct { vec4; vec4 }: 32 bytes, 16-aligned lanes.
    try std.testing.expectEqual(@as(usize, 32), @sizeOf(ClusterLightGpu));
    try std.testing.expectEqual(@as(usize, 0), @offsetOf(ClusterLightGpu, "pos_range"));
    try std.testing.expectEqual(@as(usize, 16), @offsetOf(ClusterLightGpu, "color_int"));
    // ClusterTileGpu == uvec2 (offset, count): 8 bytes, 4-aligned lanes.
    try std.testing.expectEqual(@as(usize, 8), @sizeOf(ClusterTileGpu));
    try std.testing.expectEqual(@as(usize, 0), @offsetOf(ClusterTileGpu, "offset"));
    try std.testing.expectEqual(@as(usize, 4), @offsetOf(ClusterTileGpu, "count"));
    // Index entries are plain u32 (one per tile-list slot).
    try std.testing.expectEqual(@as(usize, 4), @sizeOf(u32));
    // The header reinterpret in rebuildCpu (ClusterTileGpu <-> [2]u32).
    try std.testing.expectEqual(@sizeOf(ClusterTileGpu), @sizeOf([2]u32));
}

test "rebuildCpu is headless-safe and stages exact-fit scratch" {
    const alloc = std.testing.allocator;
    var cache = ClusteredGpuCache{};
    defer {
        cache.cpu_lights.deinit(alloc);
        cache.cpu_headers.deinit(alloc);
        cache.cpu_indices.deinit(alloc);
    }
    const pos = [_][4]f32{.{ 0.75, 0.75, 0.0, 5.0 }};
    const col = [_][4]f32{.{ 1, 0, 0, 2.0 }};
    const full = ViewRect{ .x = 0, .y = 0, .w = 128, .h = 128 };
    try cache.rebuildCpu(alloc, &pos, &col, 1, Mat4.identity, 128, 128, full);
    try std.testing.expectEqual(@as(u32, 2), cache.tiles_x);
    try std.testing.expectEqual(@as(u32, 2), cache.tiles_y);
    try std.testing.expectEqual(@as(usize, 1), cache.staged_count);
    try std.testing.expectEqual(@as(usize, 1), cache.cpu_lights.items.len);
    try std.testing.expectEqual(@as(usize, 4), cache.cpu_headers.items.len);
    try std.testing.expect(!cache.isLive(0));
    // No context headless: upload fails closed for every slot, handles
    // stay zero.
    const FakeRetire = struct {
        calls: u32 = 0,
        fn retireBuffer(self: *@This(), allocator: std.mem.Allocator, buf: sg.Buffer) void {
            _ = allocator;
            _ = buf;
            self.calls += 1;
        }
    };
    var fake = FakeRetire{};
    try std.testing.expect(!sg.isvalid());
    var s: usize = 0;
    while (s < MAX_VIEW_SLOTS) : (s += 1) {
        try std.testing.expect(!cache.upload(alloc, &fake, s));
        try std.testing.expectEqual(@as(u32, 0), cache.slots[s].light_buffer.id);
        try std.testing.expect(!cache.isLive(s));
    }
    // Dummy fallback binds (zero views headless — the draw never runs
    // without a context; the count uniform gates the shader path).
    const views = cache.bindingViews();
    try std.testing.expectEqual(@as(u32, 0), views.lights.id);

    // Empty pool: zeroed headers, empty indices, still headless-safe.
    try cache.rebuildCpu(alloc, &.{}, &.{}, 0, Mat4.identity, 128, 128, full);
    try std.testing.expectEqual(@as(usize, 0), cache.cpu_lights.items.len);
    try std.testing.expectEqual(@as(usize, 0), cache.cpu_indices.items.len);
    for (cache.cpu_headers.items) |h| {
        try std.testing.expectEqual(@as(u32, 0), h.count);
    }
}

test "retireBuffers moves live buffers into the retire queue and zeroes handles" {
    const alloc = std.testing.allocator;
    const FakeRetire = struct {
        calls: u32 = 0,
        fn retireBuffer(self: *@This(), allocator: std.mem.Allocator, buf: sg.Buffer) void {
            _ = allocator;
            _ = buf;
            self.calls += 1;
        }
    };
    var cache = ClusteredGpuCache{};
    cache.slots[0] = .{
        .light_buffer = .{ .id = 7 },
        .header_buffer = .{ .id = 8 },
        .index_buffer = .{ .id = 9 },
        .light_view = .{ .id = 70 },
        .live = true,
        .light_cap = 64,
        .header_cap = 32,
        .index_cap = 128,
    };
    // A second live slot retires too (multi-view frames allocate 1+N).
    cache.slots[2].index_buffer = .{ .id = 11 };
    cache.slots[2].live = true;
    var fake = FakeRetire{};
    cache.retireBuffers(alloc, &fake);
    // All live storage buffers retire (views are context-owned and die
    // with the buffers' generation; the next upload recreates them).
    // Untouched slots hold zero ids and are skipped by retireBuffers.
    try std.testing.expectEqual(@as(u32, 4), fake.calls);
    try std.testing.expectEqual(@as(u32, 0), cache.slots[0].light_buffer.id);
    try std.testing.expectEqual(@as(u32, 0), cache.slots[0].header_buffer.id);
    try std.testing.expectEqual(@as(u32, 0), cache.slots[0].index_buffer.id);
    try std.testing.expectEqual(@as(u32, 0), cache.slots[0].light_view.id);
    try std.testing.expect(!cache.isLive(0));
    try std.testing.expect(!cache.isLive(2));
    try std.testing.expectEqual(@as(usize, 0), cache.slots[0].light_cap);
}

test "view slots clamp defensively and cover every camera" {
    // All cameras fit; capture slots are disjoint from them and each other.
    try std.testing.expectEqual(snapshot.MAX_CAMERAS + 2, MAX_VIEW_SLOTS);
    try std.testing.expect(REFRACTION_VIEW_SLOT >= snapshot.MAX_CAMERAS);
    try std.testing.expect(RTT_VIEW_SLOT > REFRACTION_VIEW_SLOT);
    try std.testing.expect(RTT_VIEW_SLOT < MAX_VIEW_SLOTS);
    try std.testing.expectEqual(@as(usize, 0), ClusteredGpuCache.clampSlot(0));
    try std.testing.expectEqual(@as(usize, 3), ClusteredGpuCache.clampSlot(3));
    try std.testing.expectEqual(MAX_VIEW_SLOTS - 1, ClusteredGpuCache.clampSlot(MAX_VIEW_SLOTS - 1));
    try std.testing.expectEqual(MAX_VIEW_SLOTS - 1, ClusteredGpuCache.clampSlot(MAX_VIEW_SLOTS));
    try std.testing.expectEqual(MAX_VIEW_SLOTS - 1, ClusteredGpuCache.clampSlot(std.math.maxInt(usize)));

    // Out-of-range reads never trap: they observe the last slot.
    var cache = ClusteredGpuCache{};
    cache.slots[MAX_VIEW_SLOTS - 1].live = true;
    try std.testing.expect(cache.isLive(std.math.maxInt(usize)));
    const views = cache.bindingViewsForSlot(std.math.maxInt(usize));
    // Live but viewless headless: real (zero-id) views, never the dummy
    // path confusion — liveness and view handles stay consistent.
    try std.testing.expectEqual(cache.slots[MAX_VIEW_SLOTS - 1].light_view.id, views.lights.id);
}

test "rebuildCpuForSlot clears only its own slot's liveness" {
    const alloc = std.testing.allocator;
    var cache = ClusteredGpuCache{};
    defer {
        cache.cpu_lights.deinit(alloc);
        cache.cpu_headers.deinit(alloc);
        cache.cpu_indices.deinit(alloc);
    }
    const pos = [_][4]f32{.{ 0.75, 0.75, 0.0, 5.0 }};
    const col = [_][4]f32{.{ 1, 0, 0, 2.0 }};
    const full = ViewRect{ .x = 0, .y = 0, .w = 128, .h = 128 };
    // Simulate two uploaded views (headless: liveness flags only, no
    // sg.*): rebuilding the secondary must not clear the primary.
    cache.slots[0].live = true;
    cache.slots[1].live = true;
    try cache.rebuildCpuForSlot(alloc, &pos, &col, 1, Mat4.identity, 128, 128, full, 1);
    try std.testing.expect(cache.isLive(0));
    try std.testing.expect(!cache.isLive(1));
    // The legacy single-arg rebuild targets the primary slot only.
    cache.slots[1].live = true;
    try cache.rebuildCpu(alloc, &pos, &col, 1, Mat4.identity, 128, 128, full);
    try std.testing.expect(!cache.isLive(0));
    try std.testing.expect(cache.isLive(1));
}

test "bindingViewsForSlot falls back to the shared dummy per slot" {
    var cache = ClusteredGpuCache{};
    // Fresh cache: every slot binds the (zero, headless) dummy.
    var s: usize = 0;
    while (s < MAX_VIEW_SLOTS) : (s += 1) {
        const v = cache.bindingViewsForSlot(s);
        try std.testing.expectEqual(@as(u32, 0), v.lights.id);
        try std.testing.expectEqual(@as(u32, 0), v.tiles.id);
        try std.testing.expectEqual(@as(u32, 0), v.indices.id);
    }
    // A live slot with real views binds them; a dead slot still binds
    // the dummy — a failing secondary falls back to legacy while the
    // primary keeps clustered.
    cache.slots[0].live = true;
    cache.slots[0].light_view = .{ .id = 21 };
    cache.slots[0].header_view = .{ .id = 22 };
    cache.slots[0].index_view = .{ .id = 23 };
    const primary = cache.bindingViews();
    try std.testing.expectEqual(@as(u32, 21), primary.lights.id);
    try std.testing.expectEqual(@as(u32, 22), primary.tiles.id);
    try std.testing.expectEqual(@as(u32, 23), primary.indices.id);
    const secondary = cache.bindingViewsForSlot(1);
    try std.testing.expectEqual(cache.dummy_view.id, secondary.lights.id);
}
