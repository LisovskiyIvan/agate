//! Screen-space slab refraction v1 (linear HDR). No work/resources without
//! opt-in draws. The background target is half-res linear HDR (RGBA16F) and
//! renders the same clipped opaque geometry the main view draws.
const std = @import("std");
const sg = @import("sokol").gfx;
const target = @import("../render_target.zig");
const draw = @import("draw.zig");
const snapshot = @import("snapshot.zig");
const queues_mod = @import("render_queue.zig");
const math = @import("math");
const Vec3 = math.Vec3;

/// Computes the refracted direction of an incident view vector using Snell's law in 3D.
/// V is direction toward viewer; N is surface normal (both normalized).
/// Returns null if total internal reflection occurs.
pub fn refractRay(V: Vec3, N: Vec3, ior: f32) ?Vec3 {
    const safe_ior = @max(1.001, ior);
    const eta = 1.0 / safe_ior;
    const cos_theta_i = N.dot(V);
    const k = 1.0 - eta * eta * (1.0 - cos_theta_i * cos_theta_i);
    if (k < 0.0) return null; // Total internal reflection
    const factor = eta * cos_theta_i - @sqrt(k);
    return V.scale(-eta).add(N.scale(factor));
}

/// Internal reflection fallback when TIR occurs.
pub fn reflectRay(V: Vec3, N: Vec3) Vec3 {
    return V.scale(-1.0).add(N.scale(2.0 * N.dot(V)));
}

/// Computes dispersed IORs [R, G, B] given base IOR and dispersion strength [0, 0.5].
pub fn refractDispersionIors(ior: f32, dispersion: f32) [3]f32 {
    const safe_ior = @max(1.001, ior);
    const disp = std.math.clamp(dispersion, 0.0, 0.5);
    return .{
        @max(1.001, safe_ior * (1.0 - disp * 0.04)),
        safe_ior,
        safe_ior * (1.0 + disp * 0.04),
    };
}

/// Computes the projected exit point of the refracted ray exiting an optical slab.
pub fn slabExitPoint(world_pos: Vec3, ray: Vec3, N: Vec3, thickness: f32) Vec3 {
    const denom = @max(-ray.dot(N), 0.1);
    const dist = thickness / denom;
    return world_pos.add(ray.scale(dist));
}

/// Computes screen-edge vignette falloff weight [0, 1] with margin (typically 0.08).
pub fn screenEdgeWeight(screen_coord: [2]f32, margin: f32) f32 {
    const u = screen_coord[0];
    const v = screen_coord[1];
    if (u <= 0.0 or u >= 1.0 or v <= 0.0 or v >= 1.0) return 0.0;
    const m = @max(0.001, margin);
    const ex0 = std.math.clamp(u / m, 0.0, 1.0);
    const ex1 = std.math.clamp((1.0 - u) / m, 0.0, 1.0);
    const ey0 = std.math.clamp(v / m, 0.0, 1.0);
    const ey1 = std.math.clamp((1.0 - v) / m, 0.0, 1.0);
    const sx0 = ex0 * ex0 * (3.0 - 2.0 * ex0);
    const sx1 = ex1 * ex1 * (3.0 - 2.0 * ex1);
    const sy0 = ey0 * ey0 * (3.0 - 2.0 * ey0);
    const sy1 = ey1 * ey1 * (3.0 - 2.0 * ey1);
    return sx0 * sx1 * sy0 * sy1;
}

/// Dielectric Fresnel reflectance (Schlick approximation).
pub fn dielectricFresnel(cos_theta: f32, ior: f32) f32 {
    const safe_ior = @max(1.001, ior);
    var f0 = (safe_ior - 1.0) / (safe_ior + 1.0);
    f0 = f0 * f0;
    const ct = std.math.clamp(cos_theta, 0.0, 1.0);
    const x = 1.0 - ct;
    return f0 + (1.0 - f0) * (x * x * x * x * x);
}

pub const RefractionCapture = struct {
    target: target.RenderTarget = .{},
    captured_frame: ?u64 = null,

    pub fn deinit(self: *RefractionCapture) void {
        self.target.deinit();
        self.captured_frame = null;
    }
};

pub fn needsCapture(queues: *const queues_mod.RenderQueues) bool {
    for (queues.transparent.items) |item| if (item.draw_record.refractive) return true;
    for (queues.transparent_instanced.items) |batch| if (batch.draw_record.refractive) return true;
    return false;
}

pub fn capture(scene: anytype, draws: anytype, snap: *const snapshot.SceneFrameSnapshot, env: *draw.Environment) void {
    // One camera in v1: never feed a PIP camera another view's background.
    // Reuse re-presents the previous image without uploads or recapture.
    if (!sg.isvalid() or !snap.has_camera or snap.enable_multi_camera or
        snap.screen_w <= 0 or snap.screen_h <= 0 or !needsCapture(&draws.primary)) return;
    const rt = &scene.refraction.target;
    const cam = snap.primary_cam;
    const rect = cam.viewport.toPixelRect(snap.screen_w, snap.screen_h);
    if (rect.width <= 0 or rect.height <= 0) return;
    const width: u32 = @intCast(@max(1, @divTrunc(rect.width, 2)));
    const height: u32 = @intCast(@max(1, @divTrunc(rect.height, 2)));
    if (!scene.rendering_reuse) {
        scene.refraction.captured_frame = null;
        if (!rt.isValid()) {
            rt.* = target.RenderTarget.create(.{ .width = width, .height = height, .color_format = .RGBA16F }) catch return;
        } else if (!rt.resize(width, height)) return;
        if (rt.queuesSampleSelf(&draws.primary, false)) return;
        // Resolve the forward set matching the exact target shape BEFORE opening the pass:
        // forwardForShape may recreate the twin, which must never happen mid-pass.
        const fwd = scene.forwardForShape(rt.shape());
        // Shader-material prewarm for THIS target shape (opaque
        // non-refractive only — exactly what the capture below draws),
        // before the pass opens so capture draws stay lookup-only.
        _ = fwd.prewarmShaderMaterials(&draws.primary, .{ .include_transparent = false, .include_refractive = false });
        if (!rt.begin(snap.clear_color, 1)) return;
        var capture_env = env.*;
        capture_env.pipelines = fwd;
        capture_env.capture_opaque_only = true;
        var capture_snap = snap.*;
        capture_snap.screen_w = @intCast(width);
        capture_snap.screen_h = @intCast(height);
        var capture_cam = cam;
        capture_cam.viewport = .{};
        scene.renderSceneView(capture_cam, &draws.primary, &.{}, &.{}, 1, &capture_snap, capture_env, @import("clustered_lights.zig").REFRACTION_VIEW_SLOT);
        rt.end();
        scene.refraction.captured_frame = snap.frame_id;
    }
    if (scene.refraction.captured_frame != snap.frame_id or !rt.isValid() or rt.width != width or rt.height != height) return;
    env.refraction_view = rt.sampleView();
    env.refraction_sampler = rt.sampleSampler();
    env.refraction_view_proj = cam.view_proj;
    env.refraction_capture = .{ 1, @floatFromInt(width), @floatFromInt(height), 0 };
}
