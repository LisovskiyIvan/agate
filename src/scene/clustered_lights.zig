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

/// GPU mirror of one clustered light (std430: four vec4, 64 bytes).
/// Uploaded verbatim from the staged pack; the shader indexes it by the tile index list.
pub const ClusterLightGpu = extern struct {
    pos_range: [4]f32 = .{ 0, 0, 0, 0 },
    color_int: [4]f32 = .{ 0, 0, 0, 0 },
    dir_inner: [4]f32 = .{ 0, 0, 0, -2.0 },
    spot_params: [4]f32 = .{ -2.0, 0, 0, 0 },
};

/// GPU mirror of one tile header (std430 uvec2: byte offset into the
/// index list + light count for the tile). Zeroed headers (count 0) make
/// the shader loop a no-op for empty tiles.
pub const ClusterTileGpu = struct {
    offset: u32 = 0,
    count: u32 = 0,
};

/// Staging copy of one clustered spot light for the tile build:
/// position + range (w), direction + cos_inner (w),
/// color + cos_outer (w), intensity.
pub const ClusteredSpotStage = struct {
    pos_range: [4]f32,
    dir_inner: [4]f32,
    color_outer: [4]f32,
    intensity: [4]f32,
};

/// GPU mirror of one clustered spot light (std430: four vec4, 64 bytes).
pub const ClusterSpotGpu = struct {
    pos_range: [4]f32 = .{ 0, 0, 0, 0 },
    dir_inner: [4]f32 = .{ 0, 0, 0, 0 },
    color_outer: [4]f32 = .{ 0, 0, 0, 0 },
    intensity: [4]f32 = .{ 0, 0, 0, 0 },
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

/// True when the cone bounded by the sphere (center, range) overlaps the tile column.
pub fn spotOverlapsTileColumn(vp: Mat4, rect: [4]f32, center: Vec3, range: f32) bool {
    return sphereOverlapsTileColumn(vp, rect, center, range);
}

/// Builds per-tile clustered-light index lists from ClusterLightGpu slice.
pub fn buildTileListsFromLights(
    lights_slice: []const ClusterLightGpu,
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
    if (tiles == 0 or lights_slice.len == 0) return 0;
    const n = @min(lights_slice.len, lights.max_clustered_lights);
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
                const pr = lights_slice[li].pos_range;
                const ci = lights_slice[li].color_int;
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
    const n = @min(count, pos_range.len, color_int.len, lights.max_clustered_lights);
    var lights_buf: [lights.max_clustered_lights]ClusterLightGpu = undefined;
    for (0..n) |i| {
        lights_buf[i] = .{
            .pos_range = pos_range[i],
            .color_int = color_int[i],
            .dir_inner = .{ 0, 0, 0, -2.0 },
            .spot_params = .{ -2.0, 0, 0, 0 },
        };
    }
    return buildTileListsFromLights(lights_buf[0..n], view_proj, tiles_x, tiles_y, view, win_w, win_h, headers, indices);
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

    /// Rebuilds CPU tile scratch directly from a ClusterLightGpu slice for a specific slot.
    pub fn rebuildCpuFromLights(
        self: *ClusteredGpuCache,
        allocator: std.mem.Allocator,
        lights_slice: []const ClusterLightGpu,
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
        const n = @min(lights_slice.len, lights.max_clustered_lights);
        self.staged_count = n;

        self.cpu_lights.clearRetainingCapacity();
        try self.cpu_lights.appendSlice(allocator, lights_slice[0..n]);

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
        const written = buildTileListsFromLights(lights_slice[0..n], view_proj, grid.x, grid.y, view, win_w, win_h, as_pairs, self.cpu_indices.items);
        self.cpu_indices.items.len = written;
    }

    /// Slot-explicit rebuild: legacy entry point wrapping rebuildCpuFromLights.
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
        const n = @min(count, pos_range.len, color_int.len, lights.max_clustered_lights);
        var lights_buf: [lights.max_clustered_lights]ClusterLightGpu = undefined;
        for (0..n) |i| {
            lights_buf[i] = .{
                .pos_range = pos_range[i],
                .color_int = color_int[i],
                .dir_inner = .{ 0, 0, 0, -2.0 },
                .spot_params = .{ -2.0, 0, 0, 0 },
            };
        }
        return self.rebuildCpuFromLights(allocator, lights_buf[0..n], view_proj, win_w, win_h, view, view_slot);
    }

    /// Ensures the shared 16-byte dummy buffer + view exist so draws can
    /// bind valid views on the legacy path. Context thread only;
    /// fail-closed headless (no `sg.*`, dummy stays zero).
    pub fn ensureDummyViews(self: *ClusteredGpuCache) void {
        gpu_thread.assertOnContextThread();
        for (&self.slots) |*slot| {
            slot.live = false;
        }
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

        const dummy_light = [_]ClusterLightGpu{.{}};
        const light_data = if (self.cpu_lights.items.len > 0) self.cpu_lights.items else &dummy_light;
        sg.writeBufferTransient(.{
            .dst = .{ .buffer = slot.light_buffer },
            .src = .{ .data = sg.asRange(light_data) },
        });
        if (self.cpu_lights.items.len > 0) {
            upload_meter.record(self.cpu_lights.items.len * @sizeOf(ClusterLightGpu));
        }

        const dummy_header = [_]ClusterTileGpu{.{}};
        const header_data = if (self.cpu_headers.items.len > 0) self.cpu_headers.items else &dummy_header;
        sg.writeBufferTransient(.{
            .dst = .{ .buffer = slot.header_buffer },
            .src = .{ .data = sg.asRange(header_data) },
        });
        if (self.cpu_headers.items.len > 0) {
            upload_meter.record(self.cpu_headers.items.len * @sizeOf(ClusterTileGpu));
        }

        const dummy_idx = [_]u32{0};
        const idx_data = if (self.cpu_indices.items.len > 0) self.cpu_indices.items else &dummy_idx;
        sg.writeBufferTransient(.{
            .dst = .{ .buffer = slot.index_buffer },
            .src = .{ .data = sg.asRange(idx_data) },
        });
        if (self.cpu_indices.items.len > 0) {
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
    pub fn growBytes(need_bytes: usize) usize {
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
            .usage = .{ .storage_buffer = true, .write_transient = true },
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
