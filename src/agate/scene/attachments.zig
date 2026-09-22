//! Scene bounded optional attachments: reflection probes, 3D-GUI panels
//! (+ `pickUi3dPanel`), per-mesh highlights, PBD cloth bodies.
//! Split out of `scene.zig` (facade). All bodies are thin layer calls
//! or small staged-snapshot reads.
//!
/// Anti-cycle rule (same as `audio/*`, `profiler/*`): every function takes
/// the scene as `anytype` (a `*Scene` from `core.zig` in practice) and this
/// module never imports `core.zig` or the `scene.zig` facade back.
/// Cross-leaf helpers consumed here are `pub` in their home module but are
/// deliberately NOT re-exported by the facade.
const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const Ray = math.Ray;
const mesh_mod = @import("../mesh.zig");
const Mesh = mesh_mod.Mesh;
const softbody_mod = @import("../softbody.zig");
const SoftBody = softbody_mod.SoftBody;
const ClothOptions = softbody_mod.ClothOptions;
const scene_probes = @import("probe_layer.zig");
const ReflectionProbe = scene_probes.ReflectionProbe;
const ReflectionProbeOptions = scene_probes.ReflectionProbeOptions;
const scene_gui3d = @import("gui3d_layer.zig");
const Ui3dPanel = scene_gui3d.Ui3dPanel;
const Ui3dPanelOptions = scene_gui3d.Ui3dPanelOptions;
const Ui3dPickHit = scene_gui3d.Ui3dPickHit;
const scene_highlight = @import("highlight_layer.zig");
const HighlightEntry = scene_highlight.HighlightEntry;
const HighlightOptions = scene_highlight.HighlightOptions;

// ---- Reflection probes (wave 25, v1). ----
//
// On-demand environment captures: each probe renders the prepared draw
// list plus the sky into a 128px cube target from its position, then
// nearby PBR/standard draws sample it for their ambient terms (nearest
// enabled probe within its radius wins, no blending). Captures run on
// the context thread inside `render`, at most one per frame (lowest
// dirty + enabled index first; the rest wait for later frames), and
// never inside `renderReuse` (reuse re-presents the captured state).
// Before a probe's first capture lands, selection skips it, so new (or
// disabled) probes leave rendering bit-identical.
//
// Capacity: at most `scene_probes.max_probes` (4); `addReflectionProbe`
// past the cap is a hard `error.TooManyReflectionProbes`.
//
// Explicit non-goals (see scene/probe_layer.zig): box projection /
// parallax, probe blending / weights, per-frame real-time updates,
// specular occlusion, irradiance SH, editor tooling.

/// Adds a reflection probe at `position`; returns its index. The probe
/// starts dirty + uncaptured: the next `render` captures it (one capture
/// per frame) and draws fall back until then.
pub fn addReflectionProbe(self: anytype, position: Vec3, options: ReflectionProbeOptions) error{TooManyReflectionProbes}!usize {
    return self.probes.add(position, options);
}

/// Removes probe `index`, retiring its cube target through the epoch
/// retire queue (safe under update||render overlap: the context thread
/// destroys at the next flush). Order-preserving: higher indices shift
/// down. Out-of-range indices are a no-op.
pub fn removeReflectionProbe(self: anytype, index: usize) void {
    self.probes.remove(self.allocator, &self.gpu_retire, index);
}

/// Live probe state (position/radius/enabled/intensity are freely
/// mutable game-side under update-vs-prepare exclusion). Null when
/// out of range.
pub fn getReflectionProbe(self: anytype, index: usize) ?*ReflectionProbe {
    if (index >= self.probes.count) return null;
    return &self.probes.probes[index];
}

pub fn reflectionProbeCount(self: anytype) usize {
    return self.probes.count;
}

/// Requests an on-demand recapture of probe `index` on the next render
/// (actual GPU work happens there, at most one probe per frame).
/// Out-of-range is a no-op.
pub fn captureReflectionProbe(self: anytype, index: usize) void {
    self.probes.markDirty(index);
}

/// Requests a recapture of every probe (each still captures on its own
/// frame: one per frame maximum).
pub fn captureDirtyReflectionProbes(self: anytype) void {
    self.probes.markAllDirty();
}

/// How many probes currently want a capture (observability for tests
/// and tooling).
pub fn probeDirtyCount(self: anytype) usize {
    return self.probes.dirtyCount();
}

// ---- 3D GUI panels (wave 28, v1). ----
//
// On-demand world-space UI: each panel owns a private UICanvas (the app
// draws into it with the normal canvas API) rendered into a private
// color RT sized to the canvas resolution, then drawn in the main pass
// as an unlit/emissive double-sided quad. Captures run on the context
// thread inside `render`, at most `max_captures_per_frame` (1) per frame
// (lowest dirty + enabled index first; the rest wait), and never inside
// `renderReuse`. Before a panel's first capture lands, drawing and
// picking skip it, so new (or disabled) panels leave rendering
// bit-identical.
//
// Capacity: at most `scene_gui3d.max_panels` (4); `addUi3dPanel` past
// the cap is a hard `error.TooManyUi3dPanels`. Face mode v1 is fixed yaw
// only (see scene/gui3d_layer.zig for the policy, picking contract with
// `injectPointer`, and the explicit non-goals).

/// Adds a 3D GUI panel at `position`; returns its index. The name is
/// duped (Scene owns panel names; freed on remove/deinit). The panel
/// starts dirty + uncaptured: the next `render` captures it (one capture
/// per frame) while drawing/picking skip it until then. Headless-safe
/// (only CPU lists + name allocation; GPU targets are lazy).
pub fn addUi3dPanel(
    self: anytype,
    name: []const u8,
    position: Vec3,
    options: Ui3dPanelOptions,
) error{ TooManyUi3dPanels, InvalidUi3dPanelSize, OutOfMemory }!usize {
    return self.gui3d.add(self.allocator, name, position, options);
}

/// Removes panel `index`, retiring its RT target through the epoch
/// retire queue (safe under update||render overlap: the context thread
/// destroys at the next flush). Order-preserving: higher indices shift
/// down. Out-of-range indices are a no-op.
pub fn removeUi3dPanel(self: anytype, index: usize) void {
    self.gui3d.remove(self.allocator, &self.gpu_retire, index);
}

/// Live panel state (position/yaw/enabled/canvas are freely mutable
/// game-side under update-vs-prepare exclusion). Null when out of range.
pub fn getUi3dPanel(self: anytype, index: usize) ?*Ui3dPanel {
    return self.gui3d.get(index);
}

/// Live panel state by name. Null when no panel matches.
pub fn getUi3dPanelByName(self: anytype, name: []const u8) ?*Ui3dPanel {
    return self.gui3d.getByName(name);
}

pub fn ui3dPanelCount(self: anytype) usize {
    return self.gui3d.panelCount();
}

/// Requests an on-demand recapture of panel `index` on the next render
/// (actual GPU work happens there, at most one panel per frame).
/// Out-of-range is a no-op.
pub fn markUi3dPanelDirty(self: anytype, index: usize) void {
    self.gui3d.markDirty(index);
}

/// Requests a recapture of every panel (each still captures on its own
/// frame: one per frame maximum).
pub fn markAllUi3dPanelsDirty(self: anytype) void {
    self.gui3d.markAllDirty();
}

/// How many panels currently want a capture (observability for tests
/// and tooling).
pub fn ui3dDirtyCount(self: anytype) usize {
    return self.gui3d.dirtyCount();
}

/// Picks the nearest drawable 3D panel under the cursor. The ray is
/// built from the STAGED snapshot camera (`primary_cam.view_proj`,
/// fullscreen mapping — never the live camera), so game-side callers
/// under update-vs-prepare exclusion observe the same camera the frame
/// was prepared against. Returns the panel index plus canvas pixel
/// coordinates; the app routes those into
/// `panel.injectPointer(x, y, pressed)` itself (no focus system, no
/// keyboard routing in v1). Null when no camera/stage is present or no
/// drawable panel is hit. Headless-safe (pure CPU; no `sapp.*` reads —
/// snapshot dims gate instead).
pub fn pickUi3dPanel(self: anytype, mouse_x: f32, mouse_y: f32) ?Ui3dPickHit {
    const front = self.draws.frontIndex();
    const snap = &self.draws.slots[front].snapshot;
    if (!snap.has_camera) return null;
    const w: f32 = @floatFromInt(snap.screen_w);
    const h: f32 = @floatFromInt(snap.screen_h);
    if (w <= 0.0 or h <= 0.0) return null;
    const inv_vp = snap.primary_cam.view_proj.invert() orelse return null;
    const ndc_x = (2.0 * mouse_x) / w - 1.0;
    const ndc_y = 1.0 - (2.0 * mouse_y) / h;
    const near_pt = inv_vp.transformPoint(Vec3.new(ndc_x, ndc_y, 0.0));
    const far_pt = inv_vp.transformPoint(Vec3.new(ndc_x, ndc_y, 1.0));
    const dir = far_pt.sub(near_pt).normalize();
    return self.gui3d.pick(Ray.new(near_pt, dir));
}

// ---- Highlight layer (per-mesh colored inner glow, v1). ----
//
// Babylon.js HighlightLayer parity: each highlighted mesh carries its
// own color/blur/intensity (unlike the single-color inverse-hull
// outline). Mask-RT inner glow (see scene/highlight_layer.zig): the
// prepare stage freezes world matrices + proxy geometry handles into
// the frame slot, a render pass draws them flat-colored into a
// half-res mask RT, blurs it with the glow-style separable Gaussian,
// and the composite adds the inner glow (raw mask minus blurred halo,
// floored at zero, x2) after the glow block, before the
// grading chain. With zero highlights the chain is gated off and
// rendering stays bit-identical.
//
// Capacity: at most `scene_highlight.max_highlights` (8);
// `addHighlightMesh` past the cap is a hard
// `error.TooManyHighlights`, invalid options a hard
// `error.InvalidHighlightOptions` (never silent clamps — probe/gui3d
// precedent).
//
// Explicit non-goals (v1, see scene/highlight_layer.zig): skinned
// meshes (skipped at capture), instanced meshes (template proxy only),
// no occlusion-aware highlight (additive, not depth-tested), no
// per-instance highlight, primary view only. Transient: entries
// reference live meshes and are never serialized (defaults on load).

/// Highlights `mesh` with `options`; returns its index (id). Headless-safe
/// (CPU bookkeeping only; GPU targets are pass-owned and lazy).
pub fn addHighlightMesh(self: anytype, mesh: *Mesh, options: HighlightOptions) error{ TooManyHighlights, InvalidHighlightOptions }!usize {
    return self.highlights.add(mesh, options);
}

/// Removes highlight `index`. Order-preserving: higher indices shift
/// down. Out-of-range indices are a no-op. Entries own no GPU
/// resources, so no retire queue is involved (probe/gui3d remove take
/// one for their pass-owned targets; highlights have none).
pub fn removeHighlightMesh(self: anytype, index: usize) void {
    self.highlights.remove(index);
}

/// Drops every highlight (back to the bit-identical no-highlight path).
pub fn clearHighlights(self: anytype) void {
    self.highlights.clear();
}

/// Live highlight entry state. Null when out of range.
pub fn getHighlightMesh(self: anytype, index: usize) ?*HighlightEntry {
    return self.highlights.get(index);
}

pub fn highlightCount(self: anytype) usize {
    return self.highlights.highlightCount();
}

// ---- Soft bodies / PBD cloth (wave 29, v1). ----
//
// A bounded, additive, OFF-by-default feature: with zero soft bodies
// `updateSoftBodies` and the flush loop iterate an empty list, so update
// and rendering stay bit-identical to today.
//
// `addSoftBodyCloth` creates a PBD cloth solver plus the textured
// double-sided standard-material mesh it deforms every scene update
// (game side, inside `Scene.update`'s sequential ordering). Per-frame
// vertex uploads follow the sanctioned dynamic-upload discipline from
// the morph audit (game stages CPU vertices + sets the pending flag,
// the context-thread flush issues the single `sg.updateBuffer` +
// meter record — the TrailMesh pattern, not the morph fields). Removal
// always retires the mesh through the epoch retire queue, so it is safe
// from the game thread under update||render overlap.
//
// Solver state is session-local (like particle systems and trail nodes);
// serialization never writes it. See softbody.zig for the solver
// parameters, the exact strain-limiting statement and the non-goals.
//
// Capacity: at most `softbody.max_bodies` (4); past the cap is a hard
// `error.TooManySoftBodies`.

/// Creates a cloth + its deformable mesh; returns the live body (index
/// order == creation order). Hard-errors past the cap or on invalid
/// options. Headless/off-context safe (GPU buffers defer to the first
/// context-thread flush).
pub fn addSoftBodyCloth(self: anytype, name: []const u8, options: ClothOptions) softbody_mod.SoftBodyError!*SoftBody {
    return self.softbodies.create(self, name, options);
}

/// Removes body `index`: unlinks + frees the solver side and retires its
/// mesh through the epoch queue (context thread completes the teardown
/// at the next flush). Order-preserving: higher indices shift down.
/// Out-of-range indices are a hard `error.UnknownSoftBody`.
pub fn removeSoftBodyCloth(self: anytype, index: usize) softbody_mod.SoftBodyError!void {
    const mesh = try self.softbodies.extractAt(self.allocator, index);
    _ = self.removeMesh(mesh);
    self.gpu_retire.retireMesh(self.allocator, mesh);
}

/// Live body state (solver fields, pins, wind, colliders are freely
/// mutable game-side under update-vs-prepare exclusion). Null when out
/// of range.
pub fn getSoftBody(self: anytype, index: usize) ?*SoftBody {
    return self.softbodies.get(index);
}

/// Live body state by mesh name. Null when no body matches.
pub fn getSoftBodyByName(self: anytype, name: []const u8) ?*SoftBody {
    for (self.softbodies.bodies.items) |b| {
        if (std.mem.eql(u8, b.mesh.name, name)) return b;
    }
    return null;
}

/// Number of owned cloth bodies (at most softbody.max_bodies).
pub fn softBodyCount(self: anytype) usize {
    return self.softbodies.count();
}

/// Advances every enabled cloth and stages its vertex upload (game side,
/// no sg.*). Called from `Scene.update`; apps that need their own time
/// base may also drive it explicitly like `updateTrails`.
pub fn updateSoftBodies(self: anytype, dt: f32) void {
    self.softbodies.update(dt);
}
