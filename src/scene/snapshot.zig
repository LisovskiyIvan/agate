const std = @import("std");
const sokol = @import("sokol");
const sapp = sokol.app;
const math = @import("math");
const Vec3 = math.Vec3;
const Mat4 = math.Mat4;
const Color3 = math.Color3;
const Color4 = math.Color4;

const camera_mod = @import("../camera.zig");
const Camera = camera_mod.Camera;
const Viewport = camera_mod.Viewport;

const texture_mod = @import("../texture.zig");
const CubeTexture = texture_mod.CubeTexture;

const light_rig = @import("light_rig.zig");
const probe_layer = @import("probe_layer.zig");
const uniforms = @import("uniforms.zig");
const postprocess = @import("../postprocess.zig");
const gi = @import("../gi/ddgi.zig");
const PostProcessOptions = postprocess.PostProcessOptions;
const ssao = @import("../ssao.zig");
const SSAOOptions = ssao.SSAOOptions;

pub const MAX_CAMERAS: usize = 8;

/// Derived camera snapshot published by the simulation thread.
/// Render passes use this snapshot to avoid touching live camera state.
pub const CameraSnapshot = struct {
    camera: Camera = .{ .free = camera_mod.FreeCamera.init("", .{}) },
    view_proj: Mat4 = Mat4.identity,
    eye: Vec3 = Vec3.zero,
    viewport: Viewport = .{},
    culling_mask: u32 = 0xFFFFFFFF,
    clear_viewport: bool = true,
    clear_color: ?Color4 = null,
    aspect: f32 = 1.0,
    enabled: bool = true,
};

/// Frame-level camera, light, and pass state published by the simulation side.
/// Holds configuration snapshots (cameras, sun, post-process parameters) for consumption
/// by the render phase, decoupled from per-mesh draw queues and GPU resources.
pub const SceneFrameSnapshot = struct {
    frame_id: u64 = 0,
    aspect: f32 = 1.0,
    screen_w: i32 = 0,
    screen_h: i32 = 0,

    // Cameras
    has_camera: bool = false,
    primary_cam: CameraSnapshot = .{},
    cameras: [MAX_CAMERAS]CameraSnapshot = [_]CameraSnapshot{.{}} ** MAX_CAMERAS,
    camera_count: usize = 0,
    active_camera_idx: usize = 0,
    enable_multi_camera: bool = false,

    // Sun & lighting
    sun_dir: Vec3 = Vec3.new(0, 1, 0),
    sun_color: Color3 = Color3.white,
    sun_intensity: f32 = 1.0,
    cascades: [4]Mat4 = [_]Mat4{Mat4.identity} ** 4,
    light_pack: light_rig.LightRig.FramePack = .{},
    /// Reflection-probe state for the frame (wave 25): packed per
    /// `ProbeLayer.packFrame` at snapshot time (plain data + borrowed cube
    /// view/sampler VALUES, never live layer refs). The draw selects the
    /// winning probe per object from these entries; empty (count 0) means
    /// every draw takes today's ambient/skybox path bit-identically.
    /// Reused verbatim by `renderReuse` (no capture there by design).
    probe_pack: probe_layer.FramePack = .{},

    // Environment & passes config
    shadows_enabled: bool = true,
    shadow_uniforms: uniforms.ShadowState = .{
        .ground_color = Color3.black,
        .enable_shadows = true,
        .mesh_receive_shadows = true,
        .bias = 0.002,
        .intensity = 1.0,
        .normal_bias = 0.005,
        .softness = 1.0,
        .debug_cascades = false,
        .splits = .{ 0, 0, 0, 0 },
    },
    sky_texture: ?CubeTexture = null,
    /// Captured skybox switch + exposure: render reads ONLY these, never the
    /// live SkyboxLayer fields (update may mutate them concurrently with
    /// render once update||render overlap is real).
    sky_enabled: bool = false,
    sky_exposure: f32 = 1.0,
    /// Analytic-dome switch + turbidity captured from Scene.phys_sky (both
    /// false/default with the model off): the skybox pass renders the
    /// analytic dome instead of the cubemap when sky_analytic is set.
    sky_analytic: bool = false,
    sky_turbidity: f32 = 2.6,
    /// Pre-resolved skybox uniforms (sun + cloud/night extras), captured
    /// from Scene.phys_sky alongside the triple below. Off-valued unless
    /// the analytic dome is live; the draw binds it verbatim.
    sky_params: postprocess.sky.AnalyticParams = .{
        .sky_sun = .{ 0.0, 1.0, 0.0, 2.6 },
        .sky_ctrl = .{ 0.0, 0.0, 0.0, 0.0 },
        .sky_sun_color = .{ 0.0, 0.0, 0.0, 0.0 },
    },
    /// Render-owned copies of the shared default textures, captured at
    /// prepare. The draw path (scene/draw.zig Environment) binds these
    /// values — never the game-mutatable Scene.default_*_texture fields —
    /// so a concurrent update cannot race the draw's fallback sampling.
    /// Plain GPU-handle structs (no CPU refs), copied by value.
    default_white: texture_mod.Texture = .{
        .image = .{},
        .view = .{},
        .sampler = .{},
        .width = 1,
        .height = 1,
    },
    default_normal: texture_mod.Texture = .{
        .image = .{},
        .view = .{},
        .sampler = .{},
        .width = 1,
        .height = 1,
    },
    default_cube: CubeTexture = .{
        .image = .{},
        .view = .{},
        .sampler = .{},
        .size = 1,
    },
    ibl_intensity: f32 = 1.0,
    clear_color: Color4 = Color4.new(0, 0, 0, 1),
    msaa_sample_count: i32 = 1,
    post_process: PostProcessOptions = .{},
    ssao: SSAOOptions = .{},
    // DDGI v1 spike: grid options + proxy AABB scene frozen at snapshot
    // time (render reads ONLY these, never live meshes — update may run
    // concurrently with render). Empty (count 0, options disabled) with
    // the feature off: no per-frame mesh scan, zeroed sample lanes.
    ddgi: gi.DdgiOptions = .{},
    ddgi_proxy_count: usize = 0,
    ddgi_proxies: [gi.MAX_PROXIES]gi.ProxyBox = [_]gi.ProxyBox{.{
        .min = .{ 0, 0, 0 },
        .max = .{ 0, 0, 0 },
        .albedo = .{ 0, 0, 0 },
    }} ** gi.MAX_PROXIES,
    // Inverse-hull outline settings
    outline_enabled: bool = false,
    outline_color: Color4 = Color4.new(1.0, 0.5, 0.0, 1.0),
    outline_width_px: f32 = 2.0,
};

/// Packs the current camera, light, shadow, and environment state of `scene` into an immutable
/// frame snapshot that can be published to the render thread.
pub fn packFrameSnapshot(scene: anytype, aspect: f32, cur_w: i32, cur_h: i32) SceneFrameSnapshot {
    const w = if (cur_w > 0) cur_w else sapp.width();
    const h = if (cur_h > 0) cur_h else sapp.height();
    const eff_aspect = if (aspect > 0.0) aspect else (if (h > 0) @as(f32, @floatFromInt(w)) / @as(f32, @floatFromInt(h)) else 1.0);

    var snap = SceneFrameSnapshot{
        .frame_id = scene.frame_id,
        .aspect = eff_aspect,
        .screen_w = w,
        .screen_h = h,
    };

    const primary_cam_opt = scene.active_camera orelse (if (scene.cameras.items.len > 0) scene.cameras.items[0].camera else null);
    if (primary_cam_opt == null) {
        snap.has_camera = false;
        return snap;
    }
    snap.has_camera = true;

    snap.enable_multi_camera = scene.enable_multi_camera;
    snap.active_camera_idx = scene.active_camera_index orelse 0;
    snap.camera_count = @min(scene.cameras.items.len, MAX_CAMERAS);

    for (0..snap.camera_count) |i| {
        const entry = scene.cameras.items[i];
        const cam_rect = entry.viewport.toPixelRect(w, h);
        const cam_aspect = cam_rect.aspect();
        snap.cameras[i] = CameraSnapshot{
            .camera = entry.camera,
            .view_proj = entry.camera.getViewProjection(cam_aspect),
            .eye = entry.camera.getPosition(),
            .viewport = entry.viewport,
            .culling_mask = entry.culling_mask,
            .clear_viewport = entry.clear_viewport,
            .clear_color = entry.clear_color,
            .aspect = cam_aspect,
            .enabled = entry.enabled,
        };
    }

    if (snap.enable_multi_camera and snap.camera_count > 0 and snap.active_camera_idx < snap.camera_count) {
        snap.primary_cam = snap.cameras[snap.active_camera_idx];
    } else {
        const vp = primary_cam_opt.?.getViewport();
        const rect = vp.toPixelRect(w, h);
        const cam_aspect = rect.aspect();
        snap.primary_cam = CameraSnapshot{
            .camera = primary_cam_opt.?,
            .view_proj = primary_cam_opt.?.getViewProjection(cam_aspect),
            .eye = primary_cam_opt.?.getPosition(),
            .viewport = vp,
            .culling_mask = primary_cam_opt.?.getCullingMask(),
            .aspect = cam_aspect,
        };
    }

    snap.sun_dir = scene.lights.sunDirection();
    snap.sun_color = scene.lights.sunColor();
    snap.sun_intensity = scene.lights.sunIntensity();
    // Analytic physical sky (opt-in): an elevation/azimuth override moves
    // the sun BEFORE cascades, so CSM follows the model sun. Disabled:
    // this block runs nothing and every lane below keeps its legacy value.
    if (scene.phys_sky.enabled) {
        snap.sun_dir = postprocess.sky.modelSunDir(scene.phys_sky, snap.sun_dir);
    }
    snap.cascades = scene.shadows.computeCascades(snap.primary_cam.camera, snap.primary_cam.aspect, snap.sun_dir);

    var lp = scene.light_pack;
    _ = scene.light_handoff.takeLatest(&lp);
    scene.light_pack = lp;
    snap.light_pack = lp;

    snap.shadows_enabled = scene.shadows.enabled;
    snap.shadow_uniforms = scene.shadows.uniformState(scene.lights.hemi.ground_color);
    // Hemispheric light model (Babylon): direction/diffuse/intensity are real
    // shading inputs, not decoration — see shaders/common/hemi.glsl.
    snap.shadow_uniforms.hemi_dir = scene.lights.hemi.direction;
    snap.shadow_uniforms.hemi_diffuse = scene.lights.hemi.diffuse;
    snap.shadow_uniforms.hemi_intensity = scene.lights.hemi.intensity;
    snap.sky_texture = scene.sky.texture;
    snap.sky_enabled = scene.sky.enabled;
    snap.sky_exposure = scene.sky.exposure;
    snap.ibl_intensity = scene.sky.ibl_intensity;
    // Reflection-probe state for the draw's per-object selection (plain
    // data + borrowed cube view/sampler values, never live layer refs).
    snap.probe_pack = scene.probes.packFrame();
    // Render-owned default copies (plain GPU-handle values): the draw
    // binds these, never the live Scene.default_*_texture fields.
    snap.default_white = scene.default_white_texture;
    snap.default_normal = scene.default_normal_texture;
    snap.default_cube = scene.default_cube_texture;
    snap.clear_color = scene.clear_color;
    snap.msaa_sample_count = scene.msaa_sample_count;
    snap.post_process = scene.post_process;
    snap.ssao = scene.ssao;
    // DDGI proxy scene (v1 spike): collected under the snapshot/build
    // exclusion from live meshes, frozen here for the render phase.
    snap.ddgi = scene.ddgi;
    snap.ddgi_proxy_count = 0;
    if (scene.ddgi.enabled) {
        snap.ddgi_proxy_count = gi.collectProxies(scene.meshes.items, &snap.ddgi_proxies);
    }
    // Analytic triple (opt-in, post-copy so the fog tint lands on the
    // snapshot copy, never on live scene state): model-driven sun
    // color/intensity, hemispheric sky/ground ambient, and fog tint ride
    // the EXISTING lanes (no layout change); per-output drive_* flags
    // keep manual values where the artist wants them. Disabled:
    // resolveFrame returns the manual values verbatim, so every
    // assignment below is a no-op and the frame stays bit-identical.
    {
        const m = postprocess.sky.ManualFrame{
            .sun_color = .{ snap.sun_color.r, snap.sun_color.g, snap.sun_color.b },
            .sun_intensity = snap.sun_intensity,
            .hemi_diffuse = .{ snap.shadow_uniforms.hemi_diffuse.r, snap.shadow_uniforms.hemi_diffuse.g, snap.shadow_uniforms.hemi_diffuse.b },
            .hemi_ground = .{ snap.shadow_uniforms.ground_color.r, snap.shadow_uniforms.ground_color.g, snap.shadow_uniforms.ground_color.b },
            .hemi_intensity = snap.shadow_uniforms.hemi_intensity,
            .fog_color = .{ snap.post_process.fog_color[0], snap.post_process.fog_color[1], snap.post_process.fog_color[2] },
        };
        const r = postprocess.sky.resolveFrame(scene.phys_sky, snap.sun_dir, m);
        snap.sun_color = Color3.new(r.sun_color[0], r.sun_color[1], r.sun_color[2]);
        snap.sun_intensity = r.sun_intensity;
        // Slot 0 mirrors the primary shadow-caster: keep it consistent
        // with the model sun, but never conjure a sun that is absent
        // (zero intensity = no primary directional, stays zeroed).
        if (scene.phys_sky.enabled and scene.phys_sky.drive_sun and snap.light_pack.directional_color_int[0][3] != 0.0) {
            snap.light_pack.directional_dir[0] = .{ snap.sun_dir.x, snap.sun_dir.y, snap.sun_dir.z, 0.0 };
            snap.light_pack.directional_color_int[0] = .{ r.sun_color[0], r.sun_color[1], r.sun_color[2], r.sun_intensity };
        }
        snap.shadow_uniforms.hemi_diffuse = Color3.new(r.hemi_sky[0], r.hemi_sky[1], r.hemi_sky[2]);
        snap.shadow_uniforms.ground_color = Color3.new(r.hemi_ground[0], r.hemi_ground[1], r.hemi_ground[2]);
        snap.post_process.fog_color = r.fog_tint;
        snap.sky_analytic = r.analytic;
        snap.sky_turbidity = r.turbidity;
        // Skybox uniforms (dome path only; the default above already parks
        // the off state). Resolved from the MODEL sun: clouds/night ride
        // the appended lanes, sun color/intensity mirror the triple.
        if (r.analytic) {
            const se = std.math.clamp(snap.sun_dir.y, -1.0, 1.0);
            snap.sky_params = postprocess.sky.analyticParamsFull(
                snap.sun_dir,
                r.turbidity,
                Color3.new(r.sun_color[0], r.sun_color[1], r.sun_color[2]),
                r.sun_intensity,
                true,
                scene.phys_sky,
                se,
            );
        }
    }
    snap.outline_enabled = scene.postfx.outline_enabled;
    snap.outline_color = scene.postfx.outline_color;
    snap.outline_width_px = scene.postfx.outline_width_px;

    return snap;
}

/// Publishes a complete frame snapshot through the lock-free mailbox.
/// When the mailbox is saturated (consumer lagging, both slots
/// published), stale published slots are drained first so the NEWEST
/// snapshot wins — otherwise the producer build's takeLatest would resurface
/// an older published frame over the newer fallback.
///
/// Render-ownership: the saturated fallback NEVER writes the consumed
/// snapshot directly. Render reads the front slot's STAGED snapshot
/// (`FrameDrawSlot.snapshot`) concurrently with update (update||render
/// overlap), so a producer-side overwrite would race the draw; the
/// last-unclaimable tick is DROPPED instead (newest published frame
/// stays, this one is skipped). Producer (update) vs consumer (prepare)
/// stay excluded under phase_mutex, which is what makes
/// releasePublished safe here.
pub fn publishFrameSnapshot(scene: anytype, aspect: f32, cur_w: i32, cur_h: i32) void {
    const snap = packFrameSnapshot(scene, aspect, cur_w, cur_h);
    if (scene.frame_handoff.claim()) |i| {
        scene.frame_handoff.slot(i).* = snap;
        scene.frame_handoff.publish(i);
    } else {
        // Saturated: drop stale published frames (consumer is excluded
        // by phase ownership here) and publish the newest.
        scene.frame_handoff.releasePublished();
        if (scene.frame_handoff.claim()) |i| {
            scene.frame_handoff.slot(i).* = snap;
            scene.frame_handoff.publish(i);
        }
        // Still unclaimable (a slot is held in WRITING state): DROP.
        // Never fall back to `self.frame_snapshot = snap` — the
        // consumed snapshot belongs to the in-flight render.
    }
}
