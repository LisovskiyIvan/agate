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
/// outline-items строятся позже в prepareFrame и живут отдельно.
/// P4 покрывает только mesh-payload очередей; фазовый мьютекс по-прежнему
/// обязателен (GPU-ресурсы заимствуются, а UI/debug/particles/trails —
/// вне P4, см. P5-P7).
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
