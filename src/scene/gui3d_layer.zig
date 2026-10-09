//! 3D GUI panels, wave 28 (v1): Babylon-style world-space UI.
//!
//! Design (bounded, additive, OFF by default):
//! - At most `max_panels` (4) panels per scene; `add` past the cap is a hard
//!   `error.TooManyUi3dPanels` (never a silent clamp), mirroring the
//!   reflection-probe cap in `probe_layer.zig`.
//! - Each panel owns a private `UICanvas` (CPU-only until first capture: the
//!   app draws into it with the normal canvas API) plus a private RGBA8
//!   color render target sized to the canvas pixel resolution (no depth
//!   target in v1 — the quad depth-tests against the scene depth buffer at
//!   draw time instead). With no panels every draw/capture hook early-outs
//!   on a pure CPU count check with zero `sg.*` calls, so rendering is
//!   bit-identical to before this wave.
//! - Render on demand: `markDirty` flags one panel; `Scene.render` captures
//!   at most `max_captures_per_frame` (1) dirty + enabled panels per frame
//!   (lowest index first; the rest wait for later frames), uploading the
//!   canvas batch and re-rendering it into the panel RT in one offscreen
//!   pass. Panels are NEVER re-rendered every frame by default: a clean
//!   panel keeps its RT content indefinitely. A new panel starts dirty +
//!   uncaptured, so it is invisible until its first capture lands.
//! - The quad is drawn inside the main pass (in `renderSceneView`, after the
//!   transparent queue, before outlines) as an unlit/emissive textured
//!   double-sided quad sampling the RT: depth test LESS_EQUAL (occluded by
//!   opaque geometry like any world object), depth write OFF, alpha blend
//!   ON, cull NONE. The panel pipeline is rebuilt when the main-target
//!   sample count changes (sokol rejects sample-count mismatches).
//! - Shared GPU state (one layer): the bitmap font texture (same atlas as
//!   screen UI, via `UICanvas.makeFontTexture`), the UI-batch pipeline used
//!   for RT captures, the panel quad pipeline/shader, and one static unit
//!   quad. Per-panel GPU state (RT image + sample/attach views + sampler,
//!   dynamic VB/IB) is created lazily on the context thread and retired
//!   through `GpuRetireQueue` on removal (new `ui3d` entry kind, same
//!   epochs — no new queue, per the P6 tripwire).
//! - Face mode v1: FIXED YAW only. The panel keeps its authored yaw about Y
//!   (`panelRight`/`panelNormal` derive from `Mat4.rotationY`, so the quad
//!   model matrix and the picking basis agree exactly). A `billboard`
//!   variant is future work (see non-goals).
//!
//! Picking contract (v1): `Scene.pickUi3dPanel` builds a ray from the STAGED
//! snapshot camera (`SceneFrameSnapshot.primary_cam.view_proj`, fullscreen
//! mapping — never the live camera) and intersects live panel planes,
//! nearest hit wins. The result carries the panel index plus canvas pixel
//! coordinates; `panel.injectPointer(x, y, pressed)` / `injectRelease()`
//! route into the panel canvas's existing `setInput` state so immediate-mode
//! widgets react (hover/press). The app decides what to do with the pick
//! result. No focus system, no keyboard routing, no multi-camera PIP views
//! (primary snapshot view only).
//!
//! Explicit non-goals (v1): billboard/spherical facing, depth-write ordering
//! guarantees vs transparent geometry beyond the draw position (after the
//! transparent queue), multi-monitor DPI, keyboard focus, per-panel camera
//! layers, per-panel TTF fonts (captures always bind the shared bitmap
//! atlas), interaction with the P6 screen-UI pass, probe captures of panels
//! (probe faces render the raw draw lists, never the panel quads), and
//! per-frame auto-refresh.
//!
//! Threading: owned by `Scene` (context side for GPU work, game side for
//! add/remove/mark under update-vs-prepare exclusion, like the probe/sky
//! layers). Panel transforms are live-read at draw; mutating a panel
//! concurrently with render needs the same exclusion as add/remove.
//!
//! Headless behavior: every `sg.*` call is gated on `sg.isvalid()`, so all
//! bookkeeping (add/remove/dirty/quad math/ray-UV picking/input routing) is
//! CPU-testable without a GPU context. Without a context `ensureGpu` and
//! `capturePanel` fail closed (false), dirty flags are retained, and
//! `drawPanels` is a pure-CPU early-out.

const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;

const math = @import("math");
const Vec3 = math.Vec3;
const Mat4 = math.Mat4;
const Ray = math.Ray;

const ui_mod = @import("../ui.zig");
const UICanvas = ui_mod.UICanvas;
const UIVertex = ui_mod.UIVertex;
const Texture = @import("../texture.zig").Texture;
const ui_shd = @import("ui_shader");
const panel_shd = @import("ui3d_panel_shader");
const gpu_thread = @import("../gpu_thread.zig");
const SceneStats = @import("stats.zig").SceneStats;

/// Fixed capacity: at most this many 3D panels per scene. Mirrors the
/// reflection-probe cap (probe_layer.max_probes).
pub const max_panels = 4;
/// Render-on-demand budget: at most this many dirty panels are (re-)captured
/// per `Scene.render`. Lowest dirty + enabled index first; the rest wait for
/// later frames, so a `markAllDirty` burst converges in at most `max_panels`
/// frames with bounded per-frame cost (one small RT upload + offscreen UI
/// pass each). Mirrors the one-probe-per-frame capture policy.
pub const max_captures_per_frame = 1;
/// Largest canvas dimension (pixels) per axis. Bounds RT VRAM per panel to
/// 2048*2048*4 = 16 MiB worst case; larger requests are a hard error.
pub const max_canvas_dimension: u32 = 2048;
/// Initial dynamic VB/IB capacities per panel (grown via the shared
/// `UICanvas.ensureUiBufferPair` formula on demand).
pub const initial_cap_v: usize = 4096;
pub const initial_cap_i: usize = 6144;

/// Face mode v1: fixed yaw only (see module docs). The enum reserves the API
/// shape; `billboard` arrives as a later wave.
pub const Ui3dFaceMode = enum {
    fixed_yaw,
};

/// Creation options for one panel. A new panel starts dirty + uncaptured, so
/// selection/picking/drawing skip it until the first capture lands and
/// pre-capture frames stay bit-identical.
pub const Ui3dPanelOptions = struct {
    /// Quad size in world units.
    width: f32 = 2.0,
    height: f32 = 1.0,
    /// Canvas pixel resolution backing the RT (e.g. 512x256).
    canvas_width: u32 = 512,
    canvas_height: u32 = 256,
    /// Fixed yaw about Y, degrees (see module docs).
    yaw_deg: f32 = 0.0,
    face_mode: Ui3dFaceMode = .fixed_yaw,
    /// Disabled panels are skipped by capture scheduling, drawing, and
    /// picking (but keep their dirty flag, so re-enabling recaptures).
    enabled: bool = true,
};

/// Retire payload for one panel target: the RT image, its sample + attach
/// views, its sampler, and the panel-owned dynamic UI buffers. Plain
/// handles, moved by value into the retire queue on removal (probe
/// precedent: uniform no-op-destroy discipline for empty targets).
pub const Ui3dTarget = struct {
    image: sg.Image = .{},
    tex_view: sg.View = .{},
    sampler: sg.Sampler = .{},
    attach_view: sg.View = .{},
    vb: sg.Buffer = .{},
    ib: sg.Buffer = .{},

    pub fn deinit(self: *Ui3dTarget) void {
        // Headless guard (same reason as ProbeGpu.deinit): sokol destroys
        // assert a valid context, and tests retire + flush empty targets
        // without one. Either way the value resets.
        defer self.* = .{};
        if (!sg.isvalid()) return;
        if (self.sampler.id != 0) sg.destroySampler(self.sampler);
        if (self.tex_view.id != 0) sg.destroyView(self.tex_view);
        if (self.attach_view.id != 0) sg.destroyView(self.attach_view);
        if (self.vb.id != 0) sg.destroyBuffer(self.vb);
        if (self.ib.id != 0) sg.destroyBuffer(self.ib);
        if (self.image.id != 0) sg.destroyImage(self.image);
        self.* = .{};
    }
};

/// Per-panel GPU state: the retire payload plus dynamic-buffer capacities
/// and the validity flag (set once the RT + buffers exist).
pub const PanelGpu = struct {
    target: Ui3dTarget = .{},
    cap_v: usize = initial_cap_v,
    cap_i: usize = initial_cap_i,
    valid: bool = false,
};

/// One scene panel: CPU state plus its lazily created GPU target. The canvas
/// is null for unused slots (fixed array + count, probe-layer shape); `add`
/// installs a CPU-only canvas (`UICanvas.initCpuOnly` — headless-safe),
/// whose lists the capture path uploads.
pub const Ui3dPanel = struct {
    name: []const u8 = "",
    /// Mesh/CameraEntry convention: caller-owned slices stay false; `add`
    /// dupes the name and sets true, `remove`/layer-deinit frees it.
    owns_name: bool = false,
    position: Vec3 = Vec3.zero,
    yaw_deg: f32 = 0.0,
    width: f32 = 2.0,
    height: f32 = 1.0,
    canvas_width: u32 = 512,
    canvas_height: u32 = 256,
    face_mode: Ui3dFaceMode = .fixed_yaw,
    enabled: bool = true,
    /// True when the RT content is stale and needs a recapture. Set at `add`
    /// and by `markDirty`, cleared by `notifyCaptured`.
    dirty: bool = true,
    /// True once at least one capture completed. Drawing and picking require
    /// this (plus `enabled` and a live target), so pre-capture frames skip
    /// the panel entirely.
    captured: bool = false,
    canvas: ?UICanvas = null,
    gpu: PanelGpu = .{},

    /// Flags the panel for recapture on the next render.
    pub fn markDirty(self: *Ui3dPanel) void {
        self.dirty = true;
    }

    /// Routes a pointer press/move into the panel canvas's existing input
    /// state (`UICanvas.setInput`): hover position plus down + clicked while
    /// `pressed`. Immediate-mode widgets read this state on their next draw
    /// into the canvas, so a pick-then-inject sequence drives hover/press.
    /// No focus tracking in v1: every inject overwrites the single canvas
    /// state (see module docs).
    pub fn injectPointer(self: *Ui3dPanel, x: f32, y: f32, pressed: bool) void {
        if (self.canvas) |*c| c.setInput(x, y, pressed, pressed);
    }

    /// Routes a pointer release: clears down + clicked, keeps the position
    /// (hover stays where the cursor is).
    pub fn injectRelease(self: *Ui3dPanel) void {
        if (self.canvas) |*c| {
            c.setInput(c.mouse_pos[0], c.mouse_pos[1], false, false);
        }
    }
};

/// Estimated VRAM bytes of one panel RT (RGBA8, no depth in v1). Pure (no
/// GPU calls); feeds the profiler's render-target census.
pub fn targetBytes(canvas_width: u32, canvas_height: u32) usize {
    return @as(usize, canvas_width) * @as(usize, canvas_height) * 4;
}

/// Panel right axis (local +X) for `yaw_deg`, matching `Mat4.rotationY`
/// column 0 exactly (quad model and picking basis agree by construction).
pub fn panelRight(yaw_deg: f32) Vec3 {
    const rad = yaw_deg * std.math.pi / 180.0;
    return Vec3.new(@cos(rad), 0.0, -@sin(rad));
}

/// Panel normal (local +Z) for `yaw_deg`, matching `Mat4.rotationY` column
/// 2 exactly.
pub fn panelNormal(yaw_deg: f32) Vec3 {
    const rad = yaw_deg * std.math.pi / 180.0;
    return Vec3.new(@sin(rad), 0.0, @cos(rad));
}

/// Panel up axis: world Y in v1 (fixed yaw never rolls).
pub fn panelUp() Vec3 {
    return Vec3.up;
}

/// World-space quad corners: 0 = bottom-left, 1 = bottom-right, 2 =
/// top-right, 3 = top-left (CCW seen from the normal side).
pub fn panelCorners(panel: *const Ui3dPanel) [4]Vec3 {
    const r = panelRight(panel.yaw_deg).scale(panel.width * 0.5);
    const u = panelUp().scale(panel.height * 0.5);
    return .{
        panel.position.sub(r).sub(u),
        panel.position.add(r).sub(u),
        panel.position.add(r).add(u),
        panel.position.sub(r).add(u),
    };
}

/// Model matrix for the quad: T(position) * RotY(yaw) * S(width, height, 1)
/// over the unit quad (local XY plane, corners at +/-0.5).
pub fn panelModel(panel: *const Ui3dPanel) Mat4 {
    const t = Mat4.translation(panel.position);
    const r = Mat4.rotationY(panel.yaw_deg);
    const s = Mat4.scaling(Vec3.new(panel.width, panel.height, 1.0));
    return Mat4.mul(t, Mat4.mul(r, s));
}

/// Ray/plane UV hit: `u` runs left-to-right, `v` runs top-to-bottom in
/// canvas space (canvas origin top-left, y down), `t` is the ray parameter.
/// Double-sided by design (the draw is cull-NONE): rays from behind the
/// plane hit with the same UVs — no facing rejection, only the parallel and
/// behind-origin guards from `intersectsPlane` plus the 0..1 rect test.
pub const PanelUvHit = struct {
    u: f32,
    v: f32,
    t: f32,
};

pub fn intersectRayPanel(ray: Ray, panel: *const Ui3dPanel) ?PanelUvHit {
    if (panel.width <= 0.0 or panel.height <= 0.0) return null;
    const t = ray.intersectsPlane(panel.position, panelNormal(panel.yaw_deg)) orelse return null;
    const hit = ray.getPoint(t);
    const rel = hit.sub(panel.position);
    const right = panelRight(panel.yaw_deg);
    const up = panelUp();
    const u = rel.dot(right) / panel.width + 0.5;
    const v = 0.5 - rel.dot(up) / panel.height;
    if (u < 0.0 or u > 1.0 or v < 0.0 or v > 1.0) return null;
    return .{ .u = u, .v = v, .t = t };
}

/// Maps a UV hit to canvas pixel coordinates (fractional, clamped to the
/// canvas rect). Feeds `injectPointer` directly.
pub fn canvasPixelCoords(u: f32, v: f32, canvas_width: u32, canvas_height: u32) struct { x: f32, y: f32 } {
    const w: f32 = @floatFromInt(canvas_width);
    const h: f32 = @floatFromInt(canvas_height);
    return .{
        .x = std.math.clamp(u * w, 0.0, @max(w - 1.0, 0.0)),
        .y = std.math.clamp(v * h, 0.0, @max(h - 1.0, 0.0)),
    };
}

/// Pick result: winning panel index, UVs, ray distance, and canvas pixels.
pub const Ui3dPickHit = struct {
    panel_index: usize,
    u: f32,
    v: f32,
    t: f32,
    canvas_x: f32,
    canvas_y: f32,
};

/// One vertex of the shared unit quad (local XY plane, +/-0.5). UV v = 0 at
/// the quad top to match canvas space and picking ray intersection (top = v: 0.0,
/// bottom = v: 1.0), displaying the offscreen canvas upright.
pub const PanelVertex = struct {
    position: [3]f32,
    uv: [2]f32,
};

const unit_quad_vertices = [_]PanelVertex{
    .{ .position = .{ -0.5, 0.5, 0.0 }, .uv = .{ 0.0, 0.0 } },
    .{ .position = .{ 0.5, 0.5, 0.0 }, .uv = .{ 1.0, 0.0 } },
    .{ .position = .{ 0.5, -0.5, 0.0 }, .uv = .{ 1.0, 1.0 } },
    .{ .position = .{ -0.5, -0.5, 0.0 }, .uv = .{ 0.0, 1.0 } },
};
const unit_quad_indices = [_]u16{ 0, 1, 2, 0, 2, 3 };

/// Owns the scene's 3D panels plus the shared capture/draw resources (font
/// texture, UI-batch pipeline for RT captures, panel quad pipeline/shader,
/// static unit quad). Fixed-size array + count (probe-layer shape):
/// slots `0..count` are live, the tail is stale garbage after removals.
pub const Gui3dLayer = struct {
    panels: [max_panels]Ui3dPanel = [_]Ui3dPanel{.{}} ** max_panels,
    count: usize = 0,
    // Shared capture/draw resources (context thread, lazy).
    font: Texture = std.mem.zeroes(Texture),
    font_ready: bool = false,
    ui_pipeline: sg.Pipeline = .{},
    panel_shader: sg.Shader = .{},
    panel_pipeline: sg.Pipeline = .{},
    /// Main-target sample count `panel_pipeline` was built for (0 = none).
    pipeline_samples: i32 = 0,
    /// Main-target color format `panel_pipeline` was built for.
    pipeline_format: sg.PixelFormat = .RGBA16F,
    quad_vb: sg.Buffer = .{},
    quad_ib: sg.Buffer = .{},

    /// Adds a panel; returns its index. Past `max_panels` this is a hard
    /// error (never a silent clamp or replacement), mirroring
    /// `ProbeLayer.add`. Degenerate world sizes or out-of-range canvas
    /// resolutions are hard `InvalidUi3dPanelSize` errors. The name is
    /// duped (Scene owns panel names, mesh/CameraEntry convention with
    /// `owns_name = true`). New panels start dirty + uncaptured: the next
    /// render captures them (one per frame) while picking/drawing skip them.
    /// Headless-safe: only CPU lists + name allocation happen here; GPU
    /// targets are created lazily by `ensureGpu`/`capturePanel`.
    pub fn add(
        self: *Gui3dLayer,
        allocator: std.mem.Allocator,
        name: []const u8,
        position: Vec3,
        options: Ui3dPanelOptions,
    ) error{ TooManyUi3dPanels, InvalidUi3dPanelSize, OutOfMemory }!usize {
        if (self.count >= max_panels) return error.TooManyUi3dPanels;
        if (!(options.width > 0.0) or !(options.height > 0.0)) return error.InvalidUi3dPanelSize;
        if (options.width > 1024.0 or options.height > 1024.0) return error.InvalidUi3dPanelSize;
        if (options.canvas_width == 0 or options.canvas_height == 0) return error.InvalidUi3dPanelSize;
        if (options.canvas_width > max_canvas_dimension or options.canvas_height > max_canvas_dimension) {
            return error.InvalidUi3dPanelSize;
        }
        const owned_name = try allocator.dupe(u8, name);
        errdefer allocator.free(owned_name);
        const idx = self.count;
        self.panels[idx] = .{
            .name = owned_name,
            .owns_name = true,
            .position = position,
            .yaw_deg = options.yaw_deg,
            .width = options.width,
            .height = options.height,
            .canvas_width = options.canvas_width,
            .canvas_height = options.canvas_height,
            .face_mode = options.face_mode,
            .enabled = options.enabled,
            .dirty = true,
            .captured = false,
            .canvas = UICanvas.initCpuOnly(allocator),
            .gpu = .{},
        };
        self.count += 1;
        return idx;
    }

    /// Removes panel `index`, retiring its GPU target through `retire_queue`
    /// (any thread: only an epoch stamp + append happen here; the context
    /// thread destroys at the next flush). Order-preserving: higher indices
    /// shift down, so callers must not cache indices across removals.
    /// Out-of-range indices are a no-op (same contract as
    /// `Scene.removeCamera`). Removal always retires (even an empty
    /// pre-capture target: the entry is a no-op destroy, but the discipline
    /// stays uniform — probe precedent).
    pub fn remove(self: *Gui3dLayer, allocator: std.mem.Allocator, retire_queue: anytype, index: usize) void {
        if (index >= self.count) return;
        var panel = &self.panels[index];
        retire_queue.retireUi3dTarget(allocator, panel.gpu.target);
        panel.gpu = .{};
        if (panel.canvas) |*c| {
            c.vertices.deinit(c.allocator);
            c.indices.deinit(c.allocator);
            panel.canvas = null;
        }
        if (panel.owns_name) {
            allocator.free(panel.name);
        }
        for (index..self.count - 1) |k| self.panels[k] = self.panels[k + 1];
        self.count -= 1;
        // Tail past `count` is stale garbage by design (aliased lists/names
        // must never be touched): neutralize the vacated slot so a stray
        // read cannot double-free.
        self.panels[self.count] = .{};
    }

    /// Live panel state (position/yaw/enabled/canvas are freely mutable by
    /// the app under update-vs-prepare exclusion). Null past the end.
    pub fn get(self: *Gui3dLayer, index: usize) ?*Ui3dPanel {
        if (index >= self.count) return null;
        return &self.panels[index];
    }

    pub fn getByName(self: *Gui3dLayer, name: []const u8) ?*Ui3dPanel {
        for (self.panels[0..self.count]) |*p| {
            if (std.mem.eql(u8, p.name, name)) return p;
        }
        return null;
    }

    pub fn panelCount(self: *const Gui3dLayer) usize {
        return self.count;
    }

    /// Flags one panel for recapture on the next render. Out-of-range is a
    /// no-op.
    pub fn markDirty(self: *Gui3dLayer, index: usize) void {
        if (index >= self.count) return;
        self.panels[index].dirty = true;
    }

    /// Flags every panel for recapture (each still captures on its own
    /// frame: at most `max_captures_per_frame` per frame, see
    /// `nextDirtyIndex`).
    pub fn markAllDirty(self: *Gui3dLayer) void {
        for (self.panels[0..self.count]) |*p| p.dirty = true;
    }

    /// Lowest-index dirty + enabled panel, or null when nothing needs a
    /// capture. Disabled panels never schedule (capturing them would waste
    /// the frame's single capture slot on content drawing skips).
    pub fn nextDirtyIndex(self: *const Gui3dLayer) ?usize {
        for (self.panels[0..self.count], 0..) |*p, i| {
            if (p.dirty and p.enabled) return i;
        }
        return null;
    }

    /// How many panels currently want a capture (enabled or not).
    pub fn dirtyCount(self: *const Gui3dLayer) usize {
        var n: usize = 0;
        for (self.panels[0..self.count]) |*p| {
            if (p.dirty) n += 1;
        }
        return n;
    }

    /// Records a completed capture: clears `dirty`, sets `captured` so
    /// drawing and picking start using the panel.
    pub fn notifyCaptured(self: *Gui3dLayer, index: usize) void {
        if (index >= self.count) return;
        self.panels[index].dirty = false;
        self.panels[index].captured = true;
    }

    /// Pure-CPU count of panels the main pass would draw (enabled +
    /// captured + live target). The draw hook checks this FIRST with zero
    /// `sg.*` calls, which is what keeps panel-less frames bit-identical.
    pub fn drawCount(self: *const Gui3dLayer) usize {
        var n: usize = 0;
        for (self.panels[0..self.count]) |*p| {
            if (p.enabled and p.captured and p.gpu.valid and p.gpu.target.image.id != 0) n += 1;
        }
        return n;
    }

    /// Nearest enabled + captured panel intersected by `ray`, or null.
    /// Pure (no GPU calls): `captured` gates visibility, never the target.
    pub fn pick(self: *const Gui3dLayer, ray: Ray) ?Ui3dPickHit {
        var best: ?Ui3dPickHit = null;
        for (self.panels[0..self.count], 0..) |*p, i| {
            if (!p.enabled or !p.captured) continue;
            const h = intersectRayPanel(ray, p) orelse continue;
            if (best == null or h.t < best.?.t) {
                const px = canvasPixelCoords(h.u, h.v, p.canvas_width, p.canvas_height);
                best = .{
                    .panel_index = i,
                    .u = h.u,
                    .v = h.v,
                    .t = h.t,
                    .canvas_x = px.x,
                    .canvas_y = px.y,
                };
            }
        }
        return best;
    }

    /// VRAM census: summed RT bytes over panels with a live target. Pure.
    pub fn censusBytes(self: *const Gui3dLayer) usize {
        var total: usize = 0;
        for (self.panels[0..self.count]) |*p| {
            if (p.gpu.target.image.id == 0) continue;
            total += targetBytes(p.canvas_width, p.canvas_height);
        }
        return total;
    }

    /// Ensures the RT target of panel `index` plus the shared
    /// capture/draw resources exist. Context thread only (called from
    /// `Scene.render`). Fails closed (false) without a GPU context or on
    /// creation failure; the panel stays dirty and is retried on a later
    /// frame. Disabled panels still ensure (re-enabling must not stall on
    /// creation), but are never scheduled for capture while disabled.
    pub fn ensureGpu(self: *Gui3dLayer, allocator: std.mem.Allocator, index: usize) bool {
        if (!sg.isvalid()) return false;
        if (index >= self.count) return false;
        if (!self.ensureShared(allocator)) return false;
        const panel = &self.panels[index];
        if (!panel.gpu.valid and !createTarget(allocator, panel)) return false;
        panel.gpu.valid = true;
        return true;
    }

    fn ensureShared(self: *Gui3dLayer, allocator: std.mem.Allocator) bool {
        if (!self.font_ready) {
            self.font = ui_mod.UICanvas.makeFontTexture(allocator) catch return false;
            self.font_ready = true;
        }
        if (self.ui_pipeline.id == 0 and !createUiPipeline(self)) return false;
        if (self.quad_vb.id == 0 or self.quad_ib.id == 0) {
            if (!createQuad(self)) return false;
        }
        if (self.panel_shader.id == 0 and !createPanelShader(self)) return false;
        return true;
    }

    fn defaultColorFormat() sg.PixelFormat {
        const env_def = sg.queryDesc().environment.defaults;
        return if (env_def.color_format != .DEFAULT and env_def.color_format != .NONE) env_def.color_format else .BGRA8;
    }

    fn createUiPipeline(self: *Gui3dLayer) bool {
        // Same descriptor as the screen-UI canvas pipeline (UICanvas.init):
        // the RT capture reuses the identical pixel→NDC mapping with the
        // canvas resolution as the screen size.
        var desc = sg.PipelineDesc{
            .shader = sg.makeShader(ui_shd.uiShaderDesc(sg.queryBackend())),
            .index_type = .UINT16,
            .depth = .{
                .pixel_format = .NONE,
                .compare = .ALWAYS,
                .write_enabled = false,
            },
            .cull_mode = .NONE,
        };
        if (desc.shader.id == 0) return false;
        errdefer sg.destroyShader(desc.shader);
        desc.colors[0].pixel_format = defaultColorFormat();
        desc.colors[0].blend = .{
            .enabled = true,
            .src_factor_rgb = .SRC_ALPHA,
            .dst_factor_rgb = .ONE_MINUS_SRC_ALPHA,
            .src_factor_alpha = .ONE,
            .dst_factor_alpha = .ONE_MINUS_SRC_ALPHA,
        };
        desc.layout.buffers[0] = .{ .stride = @sizeOf(UIVertex) };
        desc.layout.attrs[ui_shd.ATTR_ui_position] = .{
            .buffer_index = 0,
            .format = .FLOAT2,
            .offset = @offsetOf(UIVertex, "position"),
        };
        desc.layout.attrs[ui_shd.ATTR_ui_texcoord0] = .{
            .buffer_index = 0,
            .format = .FLOAT2,
            .offset = @offsetOf(UIVertex, "uv"),
        };
        desc.layout.attrs[ui_shd.ATTR_ui_color0] = .{
            .buffer_index = 0,
            .format = .FLOAT4,
            .offset = @offsetOf(UIVertex, "color"),
        };
        desc.layout.attrs[ui_shd.ATTR_ui_mode_params] = .{
            .buffer_index = 0,
            .format = .FLOAT4,
            .offset = @offsetOf(UIVertex, "mode_params"),
        };
        const pip = sg.makePipeline(desc);
        if (pip.id == 0) return false;
        self.ui_pipeline = pip;
        return true;
    }

    fn createQuad(self: *Gui3dLayer) bool {
        const vb = sg.makeBuffer(.{ .data = sg.asRange(&unit_quad_vertices) });
        if (vb.id == 0) return false;
        errdefer sg.destroyBuffer(vb);
        const ib = sg.makeBuffer(.{
            .usage = .{ .index_buffer = true },
            .data = sg.asRange(&unit_quad_indices),
        });
        if (ib.id == 0) return false;
        self.quad_vb = vb;
        self.quad_ib = ib;
        return true;
    }

    fn createPanelShader(self: *Gui3dLayer) bool {
        const shd = sg.makeShader(panel_shd.ui3dPanelShaderDesc(sg.queryBackend()));
        if (shd.id == 0) return false;
        self.panel_shader = shd;
        return true;
    }

    fn createTarget(allocator: std.mem.Allocator, panel: *Ui3dPanel) bool {
        _ = allocator;
        const w: i32 = @intCast(panel.canvas_width);
        const h: i32 = @intCast(panel.canvas_height);
        const img = sg.makeImage(.{
            .usage = .{ .color_attachment = true },
            .width = w,
            .height = h,
            .pixel_format = defaultColorFormat(),
            .sample_count = 1,
        });
        if (img.id == 0) return false;
        errdefer sg.destroyImage(img);

        const sample_view = sg.makeView(.{ .texture = .{ .image = img } });
        if (sample_view.id == 0) return false;
        errdefer sg.destroyView(sample_view);

        const attach_view = sg.makeView(.{ .color_attachment = .{ .image = img } });
        if (attach_view.id == 0) return false;
        errdefer sg.destroyView(attach_view);

        const smp = sg.makeSampler(.{
            .min_filter = .LINEAR,
            .mag_filter = .LINEAR,
            .wrap_u = .CLAMP_TO_EDGE,
            .wrap_v = .CLAMP_TO_EDGE,
        });
        if (smp.id == 0) return false;
        errdefer sg.destroySampler(smp);

        const vb = sg.makeBuffer(.{
            .usage = .{ .vertex_buffer = true, .write_transient = true },
            .size = panel.gpu.cap_v * @sizeOf(UIVertex),
        });
        if (vb.id == 0) return false;
        errdefer sg.destroyBuffer(vb);
        const ib = sg.makeBuffer(.{
            .usage = .{ .index_buffer = true, .write_transient = true },
            .size = panel.gpu.cap_i * @sizeOf(u16),
        });
        if (ib.id == 0) return false;

        panel.gpu.target = .{
            .image = img,
            .tex_view = sample_view,
            .sampler = smp,
            .attach_view = attach_view,
            .vb = vb,
            .ib = ib,
        };
        return true;
    }

    /// GPU half of the on-demand capture (context thread only): ensures the
    /// target, uploads the canvas batch into the panel-owned buffers (same
    /// single-update-per-frame discipline as the P6 UI path, retired on
    /// growth through `retire_queue`), and renders it into the RT in one
    /// offscreen pass. An empty canvas clears the RT to transparent instead.
    /// Returns false without capturing (headless, disabled, open upload
    /// window, or creation failure) with the dirty flag retained for retry.
    pub fn capturePanel(self: *Gui3dLayer, allocator: std.mem.Allocator, retire_queue: anytype, index: usize) bool {
        if (!sg.isvalid()) return false;
        gpu_thread.assertOnContextThread();
        if (index >= self.count) return false;
        const panel = &self.panels[index];
        if (!panel.enabled) return false;
        const canvas = if (panel.canvas) |*c| c else return false;
        // Same-sokol-frame policy (P6 discipline): this frame's single
        // update on these buffers is already spent — defer, keep dirty.
        if (canvas.isUploadOpen()) return false;
        if (!self.ensureGpu(allocator, index)) return false;

        const w_f: f32 = @floatFromInt(panel.canvas_width);
        const h_f: f32 = @floatFromInt(panel.canvas_height);

        var pass_action = sg.PassAction{};
        pass_action.colors[0] = .{
            .load_action = .CLEAR,
            .clear_value = .{ .r = 0.0, .g = 0.0, .b = 0.0, .a = 0.0 },
        };
        var pass = sg.Pass{ .action = pass_action };
        pass.attachments.colors[0] = panel.gpu.target.attach_view;
        sg.beginPass(pass);
        sg.applyViewport(0, 0, @intCast(panel.canvas_width), @intCast(panel.canvas_height), true);
        sg.applyScissorRect(0, 0, @intCast(panel.canvas_width), @intCast(panel.canvas_height), true);

        if (canvas.vertices.items.len > 0 and canvas.indices.items.len > 0) {
            const vert_count = UICanvas.clampedVertCount(canvas.vertices.items.len);
            const ensured = UICanvas.ensureUiBufferPair(
                panel.gpu.target.vb,
                panel.gpu.target.ib,
                panel.gpu.cap_v,
                panel.gpu.cap_i,
                vert_count,
                canvas.indices.items.len,
            );
            if (ensured.ok) {
                UICanvas.uploadUiBuffers(
                    ensured.vertex_buffer,
                    ensured.index_buffer,
                    canvas.vertices.items[0..vert_count],
                    canvas.indices.items,
                );
                canvas.markUiUploaded();
                if (ensured.replaced_vb) {
                    const old = panel.gpu.target.vb;
                    panel.gpu.target.vb = ensured.vertex_buffer;
                    panel.gpu.cap_v = ensured.capacity_vertices;
                    if (old.id != 0) retire_queue.retireBuffer(allocator, old);
                }
                if (ensured.replaced_ib) {
                    const old = panel.gpu.target.ib;
                    panel.gpu.target.ib = ensured.index_buffer;
                    panel.gpu.cap_i = ensured.capacity_indices;
                    if (old.id != 0) retire_queue.retireBuffer(allocator, old);
                }
                UICanvas.drawUiBuffers(
                    self.ui_pipeline,
                    panel.gpu.target.vb,
                    panel.gpu.target.ib,
                    self.font.view,
                    self.font.sampler,
                    w_f,
                    h_f,
                    canvas.indices.items.len,
                );
            }
        }
        sg.endPass();
        self.notifyCaptured(index);
        return true;
    }

    /// Draws every drawable panel quad into the CURRENT main pass for the
    /// exact target shape (called from `renderSceneView` after the
    /// transparent queue). Zero `sg.*`
    /// calls unless at least one panel is drawable — the pure `drawCount`
    /// check runs first, so panel-less frames are bit-identical.
    pub fn drawPanels(self: *Gui3dLayer, allocator: std.mem.Allocator, view_proj: Mat4, samples: i32, color_format: sg.PixelFormat, stats: *SceneStats) void {
        if (self.drawCount() == 0) return;
        if (!sg.isvalid()) return;
        gpu_thread.assertOnContextThread();
        if (!self.ensureShared(allocator)) return;
        if (!self.ensurePanelPipeline(samples, color_format)) return;
        for (self.panels[0..self.count]) |*panel| {
            if (!panel.enabled or !panel.captured or !panel.gpu.valid) continue;
            if (panel.gpu.target.image.id == 0) continue;
            const mvp = Mat4.mul(view_proj, panelModel(panel));
            sg.applyPipeline(self.panel_pipeline);
            var bind = sg.Bindings{};
            bind.vertex_buffers[0] = self.quad_vb;
            bind.index_buffer = self.quad_ib;
            bind.views[panel_shd.VIEW_panel_tex] = panel.gpu.target.tex_view;
            bind.samplers[panel_shd.SMP_smp] = panel.gpu.target.sampler;
            sg.applyBindings(bind);
            const vs_params = panel_shd.VsParams{ .mvp = mvp };
            sg.applyUniforms(panel_shd.UB_vs_params, sg.asRange(&vs_params));
            sg.draw(0, 6, 1);
            stats.draw_calls += 1;
        }
    }

    /// (Re)builds the quad pipeline for the exact main-target shape.
    /// The old pipeline is destroyed immediately on change (context thread,
    /// between draws — never while bound). Fails closed (false) on
    /// creation failure; the previous pipeline (if any) stays bound-able.
    fn ensurePanelPipeline(self: *Gui3dLayer, samples: i32, color_format: sg.PixelFormat) bool {
        if (self.panel_pipeline.id != 0 and self.pipeline_samples == samples and self.pipeline_format == color_format) return true;
        var desc = sg.PipelineDesc{
            .shader = self.panel_shader,
            .index_type = .UINT16,
            .depth = .{ .compare = .LESS_EQUAL, .write_enabled = false },
            .cull_mode = .NONE,
            .sample_count = samples,
        };
        desc.colors[0].pixel_format = color_format;
        desc.colors[0].blend = .{
            .enabled = true,
            .src_factor_rgb = .SRC_ALPHA,
            .dst_factor_rgb = .ONE_MINUS_SRC_ALPHA,
            .src_factor_alpha = .ONE,
            .dst_factor_alpha = .ONE_MINUS_SRC_ALPHA,
        };
        desc.layout.buffers[0] = .{ .stride = @sizeOf(PanelVertex) };
        desc.layout.attrs[panel_shd.ATTR_ui3d_panel_position] = .{
            .buffer_index = 0,
            .format = .FLOAT3,
            .offset = @offsetOf(PanelVertex, "position"),
        };
        desc.layout.attrs[panel_shd.ATTR_ui3d_panel_texcoord0] = .{
            .buffer_index = 0,
            .format = .FLOAT2,
            .offset = @offsetOf(PanelVertex, "uv"),
        };
        const pip = sg.makePipeline(desc);
        if (pip.id == 0) return false;
        if (self.panel_pipeline.id != 0) sg.destroyPipeline(self.panel_pipeline);
        self.panel_pipeline = pip;
        self.pipeline_samples = samples;
        self.pipeline_format = color_format;
        return true;
    }

    /// Destroys every live panel target plus the shared resources. Context
    /// thread only in production (Scene.deinit, sg still alive); every
    /// destroy is additionally `sg.isvalid()`-gated so headless tests can
    /// tear down CPU state (names, lists) without a context. Pending retire
    /// entries for already-removed panels are drained by
    /// `GpuRetireQueue.deinit` separately — no double destroy.
    pub fn deinit(self: *Gui3dLayer, allocator: std.mem.Allocator) void {
        for (self.panels[0..self.count]) |*panel| {
            panel.gpu.target.deinit();
            if (panel.canvas) |*c| {
                c.vertices.deinit(c.allocator);
                c.indices.deinit(c.allocator);
                panel.canvas = null;
            }
            if (panel.owns_name) {
                allocator.free(panel.name);
                panel.name = "";
                panel.owns_name = false;
            }
        }
        self.count = 0;
        if (sg.isvalid()) {
            if (self.panel_pipeline.id != 0) sg.destroyPipeline(self.panel_pipeline);
            if (self.panel_shader.id != 0) sg.destroyShader(self.panel_shader);
            if (self.ui_pipeline.id != 0) sg.destroyPipeline(self.ui_pipeline);
            if (self.quad_vb.id != 0) sg.destroyBuffer(self.quad_vb);
            if (self.quad_ib.id != 0) sg.destroyBuffer(self.quad_ib);
            if (self.font_ready) self.font.deinit();
        }
        self.* = .{};
    }
};
