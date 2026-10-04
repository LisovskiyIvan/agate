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

/// Frame-level camera/light/pass state published by the simulation side.
/// Это frame mailbox (камеры/свет/конфиг проходов), а НЕ подготовленные
/// per-view draw-записи: per-mesh очереди (RenderQueues), shadow-bins и
/// outline-items строятся позже в staged begin/finish и живут отдельно — P7: в трёх
/// retained-слотах Scene.draws (см. scene/frame_draws.zig: FrameDrawSlot),
/// а не в этом snapshot.
/// P4 покрывает только mesh-payload очередей; update-vs-prepare остаются
/// исключены фазовым мьютексом (producer update, consumer prepare), а update
/// CAN overlap render — поэтому saturated-фолбэк publishFrameSnapshot
/// НИКОГДА не пишет consumed snapshot (только drop), иначе гонка с
/// draw. GPU-ресурсы заимствуются под фазовым мьютексом/P3; UI покрыт P6
/// (render-owned кадр в Scene), debug — prepared capture + committed upload,
/// sky/defaults — копии в этом snapshot, light pack — snapshot-копия.
/// Producer-билд (`buildPreparedFrame`) владеет отдельной копией этого же
/// типа (`Scene.build_snapshot`): очереди строятся против неё, staged-копия
/// замораживается в claim-слоте (`FrameDrawSlot.snapshot`), а latch
/// потребляет STAGED-копию (staged wins над пост-билд мутацией
/// `build_snapshot`; `frame_snapshot` зеркалит staged для совместимости) —
/// build НИКОГДА не читает/пишет `frame_snapshot`, иначе гонка с draw при
/// update||render.
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
    // Carried for the PASS 1.7 depth-prepass decision (render thread reads
    // only the snapshot, never live Scene fields). Default off.
    msaa_depth_prepass: bool = false,
    post_process: PostProcessOptions = .{},
    ssao: SSAOOptions = .{},
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
    snap.msaa_depth_prepass = scene.msaa_depth_prepass;
    snap.post_process = scene.post_process;
    snap.ssao = scene.ssao;
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
