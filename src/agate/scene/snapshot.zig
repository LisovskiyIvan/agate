const std = @import("std");
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
/// outline-items строятся позже в prepareFrame и живут отдельно — P7: в двух
/// retained-слотах Scene.draws (см. scene/frame_draws.zig: FrameDrawSlot),
/// а не в этом snapshot.
/// P4 покрывает только mesh-payload очередей; update-vs-prepare остаются
/// исключены фазовым мьютексом (producer update, consumer prepare), а update
/// CAN overlap render — поэтому saturated-фолбэк publishFrameSnapshot
/// НИКОГДА не пишет consumed frame_snapshot (только drop), иначе гонка с
/// draw. GPU-ресурсы заимствуются под фазовым мьютексом/P3; UI покрыт P6
/// (render-owned кадр в Scene), debug — prepared capture + committed upload,
/// sky/defaults — копии в этом snapshot, light pack — snapshot-копия.
/// Producer-билд (`buildPreparedFrame`) владеет отдельной копией этого же
/// типа (`Scene.build_snapshot`): очереди строятся против неё, а latch
/// копирует её в consumed `frame_snapshot` один в один — build НИКОГДА не
/// читает/пишет `frame_snapshot`, иначе гонка с draw при update||render.
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
    post_process: PostProcessOptions = .{},
    ssao: SSAOOptions = .{},
    // Inverse-hull outline settings
    outline_enabled: bool = false,
    outline_color: Color4 = Color4.new(1.0, 0.5, 0.0, 1.0),
    outline_width_px: f32 = 2.0,
};

test "SceneFrameSnapshot default initialization" {
    const snap = SceneFrameSnapshot{};
    try std.testing.expect(!snap.has_camera);
    try std.testing.expectEqual(@as(usize, 0), snap.camera_count);
    try std.testing.expectEqual(@as(i32, 1), snap.msaa_sample_count);
}
