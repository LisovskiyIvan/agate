const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const sapp = sokol.app;
const sglue = sokol.glue;
const postprocess = @import("postprocess.zig");
const PostProcessOptions = postprocess.PostProcessOptions;
const ssao = @import("ssao.zig");
const SSAOOptions = ssao.SSAOOptions;
const particles = @import("particles.zig");
const ParticleSystem = particles.ParticleSystem;

const math = @import("math");
const Vec3 = math.Vec3;
const Mat4 = math.Mat4;
const Color3 = math.Color3;
const Color4 = math.Color4;
const BoundingBox = math.BoundingBox;
const Ray = math.Ray;

const physics = @import("physics.zig");
const PhysicsWorld = physics.PhysicsWorld;
const RigidBody = physics.RigidBody;
const PickingInfo = physics.PickingInfo;
const ColliderType = physics.ColliderType;

const ui = @import("ui.zig");
const UICanvas = ui.UICanvas;

const AnimationGroup = @import("animation/animation.zig").AnimationGroup;
const Skeleton = @import("animation/skeleton.zig").Skeleton;

const camera_mod = @import("camera.zig");
const Camera = camera_mod.Camera;
const Viewport = camera_mod.Viewport;
const debug_shd = @import("debug_shader");
const debug_pass = @import("passes/debug_pass.zig");
const lights = @import("lights.zig");
const HemisphericLight = lights.HemisphericLight;
const DirectionalLight = lights.DirectionalLight;
const DirectionalLightOptions = lights.DirectionalLightOptions;
const PointLight = lights.PointLight;
const PointLightOptions = lights.PointLightOptions;
const SpotLight = lights.SpotLight;
const SpotLightOptions = lights.SpotLightOptions;
const Mesh = @import("mesh.zig").Mesh;
const decal_mod = @import("mesh/decal.zig");
const DecalManager = decal_mod.DecalManager;
const trail_mod = @import("mesh/trail.zig");
const TrailMesh = trail_mod.TrailMesh;
const TrailOptions = trail_mod.TrailOptions;
const csg_mod = @import("mesh/csg.zig");
const ai_mod = @import("ai.zig");
const NavMesh = ai_mod.NavMesh;
const NavAgent = ai_mod.NavAgent;
const StandardMaterial = @import("material.zig").StandardMaterial;
const PBRMaterial = @import("material.zig").PBRMaterial;
const ShaderMaterial = @import("material.zig").ShaderMaterial;
const Texture = @import("texture.zig").Texture;
const CubeTexture = @import("texture.zig").CubeTexture;
const SkyboxOptions = @import("texture.zig").SkyboxOptions;
const visibility = @import("visibility/mod.zig");
const serialization = @import("serialization.zig");

// Scene subsystems. Each owns its state (and GPU resources) plus the logic
// that belongs to it; Scene is the owner/orchestrator facade. Subsystems
// never import scene.zig — Scene passes everything they need as parameters.
const scene_stats = @import("scene/stats.zig");
pub const SceneStats = scene_stats.SceneStats;
const scene_render_queue = @import("scene/render_queue.zig");
const jobs = @import("jobs.zig");
const assets_mod = @import("assets.zig");
const handoff_mod = @import("handoff.zig");
pub const RenderMeshItem = scene_render_queue.RenderMeshItem;
const scene_lights = @import("scene/light_rig.zig");
const scene_shadow = @import("scene/shadow_system.zig");
const scene_sky = @import("scene/sky_layer.zig");
const scene_postfx = @import("scene/postfx_stack.zig");
const scene_forward = @import("scene/forward_pipelines.zig");
const scene_particles = @import("scene/particle_layer.zig");
const scene_decals = @import("scene/decal_layer.zig");
const scene_trails = @import("scene/trail_layer.zig");
const scene_nav = @import("scene/nav_layer.zig");
const scene_physics = @import("scene/physics_layer.zig");
const scene_project = @import("scene/project_cache.zig");
const scene_content = @import("scene/content.zig");
const scene_animation = @import("scene/animation_runtime.zig");
const scene_picking = @import("scene/picking.zig");
const scene_uniforms = @import("scene/uniforms.zig");
const FrameContext = scene_uniforms.FrameContext;
const scene_draw = @import("scene/draw.zig");
const scene_msaa = @import("scene/msaa.zig");
pub const scene_snapshot = @import("scene/snapshot.zig");
pub const SceneFrameSnapshot = scene_snapshot.SceneFrameSnapshot;
pub const CameraSnapshot = scene_snapshot.CameraSnapshot;

pub const CameraEntry = struct {
    name: []const u8,
    camera: Camera,
    owns_name: bool = false,
    enabled: bool = true,
    culling_mask: u32 = 0xFFFFFFFF,
    viewport: Viewport = .{},
    clear_viewport: bool = true,
    clear_color: ?Color4 = null,
};

/// The scene: content registries (flat, iterated directly by loaders,
/// tooling and serialization), a handful of cross-cutting config flags, and
/// one field per render subsystem. New features should add state to the
/// matching subsystem in scene/, not to this struct.
pub const Scene = struct {
    allocator: std.mem.Allocator,

    // ---- Content registries (kept flat: external code iterates them). ----
    meshes: std.ArrayListUnmanaged(*Mesh) = .empty,
    materials: std.ArrayListUnmanaged(*StandardMaterial) = .empty,
    pbr_materials: std.ArrayListUnmanaged(*PBRMaterial) = .empty,
    shader_materials: std.ArrayListUnmanaged(*ShaderMaterial) = .empty,
    animation_groups: std.ArrayListUnmanaged(*AnimationGroup) = .empty,
    skeletons: std.ArrayListUnmanaged(*Skeleton) = .empty,

    // ---- Camera management ----
    cameras: std.ArrayListUnmanaged(CameraEntry) = .empty,
    active_camera_index: ?usize = null,
    enable_multi_camera: bool = false,
    clear_pipeline: sg.Pipeline = .{},
    clear_pipeline_msaa: sg.Pipeline = .{},
    clear_shader: sg.Shader = .{},
    clear_vb: sg.Buffer = .{},

    active_camera: ?Camera = null,
    active_camera_owned_name: ?[]const u8 = null,
    clear_color: Color4 = Color4.new(0.12, 0.14, 0.18, 1.0),
    default_material: StandardMaterial = StandardMaterial.init("default"),
    default_white_texture: Texture,
    default_normal_texture: Texture,
    default_cube_texture: CubeTexture,

    // ---- Cross-cutting culling config and per-frame stats. ----
    enable_frustum_culling: bool = true,
    enable_occlusion_culling: bool = true,
    occlusion_culler: visibility.OcclusionCuller = visibility.OcclusionCuller.init(),
    stats: SceneStats = .{},
    // Bumped once per render(); Mesh.cached_* entries tagged with this are fresh.
    frame_id: u64 = 0,

    // ---- Render subsystems. ----
    // All lights: hemispheric sun, optional directional sun, point/spot lists.
    lights: scene_lights.LightRig,
    // CSM shadow state + GPU shadow depth pass.
    shadows: scene_shadow.ShadowSystem,
    // Skybox texture/exposure + IBL intensity + skybox pass.
    sky: scene_sky.SkyboxLayer,
    // Offscreen target, SSAO, bloom, composite pass + outline highlights.
    postfx: scene_postfx.PostFXStack,
    // Forward GPU pipeline sets (opaque/blend/double-sided per family).
    forward: scene_forward.ForwardPipelines,
    // Particle systems + billboard pass.
    particles: scene_particles.ParticleLayer,
    /// Stage 3, slice 2: light selection + packing (including the
    /// incumbency-hysteresis fade simulation) runs in `updateLights`
    /// during the update phase and publishes through this mailbox; render
    /// takes the newest pack at frame start. Plain data end to end, so
    /// the group is already thread-ready for the game/render split.
    light_handoff: handoff_mod.Handoff(scene_lights.LightRig.FramePack, 2) = .{},
    /// Last consumed pack — the fallback render uses when no new light
    /// pack was published since the previous frame (e.g. PIP: several
    /// render calls per one update).
    light_pack: scene_lights.LightRig.FramePack = std.mem.zeroes(scene_lights.LightRig.FramePack),
    /// Async texture decode/upload pipeline. Drained at the top of
    /// render(); deinit'd FIRST in deinit so in-flight decodes finish
    /// before any material they target is freed. Null = synchronous loads.
    uploads: ?assets_mod.UploadQueue = null,
    // Dynamic decals.
    decals: scene_decals.DecalLayer = .{},
    // Trail meshes.
    trails: scene_trails.TrailLayer = .{},
    // Nav meshes and agents.
    nav: scene_nav.NavLayer = .{},
    // Physics world + debug wireframe overlay.
    physics: scene_physics.PhysicsIntegration = .{},
    // Per-frame draw queues + instance staging.
    queues: scene_render_queue.RenderQueues = .{},
    // projectPoint view-projection cache.
    project: scene_project.ProjectCache = .{},

    // Public post-processing configs. Flat on purpose: sandbox tooling reads
    // and writes these two directly ~100 times; the stack that consumes them
    // lives in `postfx`.
    post_process: PostProcessOptions = .{},
    ssao: SSAOOptions = .{},

    // Requested MSAA sample count for the offscreen main render target
    // (PASS 2). 1 = off (default). Effective count = scene/msaa.zig policy:
    // clamped per backend, applied only when the post chain owns the main
    // pass, 1x on the swapchain path. While active it suppresses the
    // depth-consuming post effects (SSAO/SSR/DoF) — sokol has no depth
    // resolve, so an MSAA depth attachment cannot feed them (warned once).
    // Set before the first render(); changing it later recreates the MSAA
    // pipeline/target twins on the next MSAA frame.
    msaa_sample_count: i32 = 1,

    // Forward pipeline twin built for the effective MSAA sample count
    // (sokol requires pipeline.sample_count to match every main-target
    // attachment). Null until the first MSAA frame.
    forward_msaa: ?scene_forward.ForwardPipelines = null,
    warn_msaa_format: scene_msaa.WarnOnce = .{},

    // Highlighted meshes for the inverse-hull outline (postfx holds the
    // settings + pass). Kept flat: mock scenes in mesh tests construct it.
    outline_meshes: std.ArrayListUnmanaged(*Mesh) = .empty,
    render_outline_meshes: std.ArrayListUnmanaged(*Mesh) = .empty,

    /// Mailbox for publishing frame snapshots from the simulation thread.
    /// The render thread consumes the newest snapshot without holding coarse
    /// locks during GPU draw calls, enabling strictly non-blocking rendering.
    frame_handoff: handoff_mod.Handoff(scene_snapshot.SceneFrameSnapshot, 2) = .{},
    /// Last consumed frame snapshot.
    frame_snapshot: scene_snapshot.SceneFrameSnapshot = .{},
    /// Flag indicating whether prepareFrame() has already run for this frame.
    frame_prepared: bool = false,

    // 2D & 3D UI canvas (lazy; created via createUI()).
    ui_canvas: ?UICanvas = null,

    // Frame uniform types shared with the draw path (see scene/uniforms.zig).

    pub fn initInto(self: *Scene, allocator: std.mem.Allocator) void {
        self.* = Scene{
            .allocator = allocator,
            .default_white_texture = Texture.createWhite1x1(),
            .default_normal_texture = Texture.createFlatNormal1x1(),
            .default_cube_texture = CubeTexture.createDefault1x1(.{ 25, 30, 40, 255 }),
            .lights = scene_lights.LightRig.init("hemi", .{
                .direction = math.Vec3.new(0.5, 1.0, 0.3),
                .diffuse = Color3.white,
                .ground_color = Color3.new(0.2, 0.25, 0.3),
                .intensity = 1.0,
            }),
            .shadows = scene_shadow.ShadowSystem.init(allocator),
            .sky = scene_sky.SkyboxLayer.init(),
            .postfx = scene_postfx.PostFXStack.init(),
            .forward = scene_forward.ForwardPipelines.init(),
            .particles = scene_particles.ParticleLayer.init(),
        };
        // Async texture decode/uploads (stage 2): failures degrade to a
        // null queue and all loads take the synchronous path.
        self.uploads = assets_mod.UploadQueue.init(allocator, 2) catch null;
    }

    pub fn init(allocator: std.mem.Allocator) Scene {
        var self: Scene = undefined;
        self.initInto(allocator);
        return self;
    }

    // ---- Offscreen targets & post-processing config. ----

    /// Window-resize hook: resizes every viewport-sized offscreen target
    /// (postprocess, bloom, SSAO, outline). The per-frame post chain resizes
    /// postprocess/bloom lazily, but SSAO has no other resize path — call
    /// this when the window size changes and post-processing/SSAO is in use.
    pub fn resizeOffscreen(self: *Scene, width: i32, height: i32) void {
        self.postfx.resizeAll(width, height);
    }

    pub fn setPostProcess(self: *Scene, config: PostProcessOptions) void {
        self.post_process = config;
    }

    pub fn setSSAO(self: *Scene, config: SSAOOptions) void {
        self.ssao = config;
    }

    // ---- Serialization (off-thread save/load). ----

    /// Stage 3, slice 3: captures scene state under snapshot semantics (<0.1 ms)
    /// and dispatches serialization + file I/O to a background TaskRunner.
    /// Neither the game thread nor the sapp render thread blocks on disk I/O.
    pub fn saveStateFileAsync(self: *Scene, path: []const u8) !*serialization.AsyncSaveTask {
        const snap = try serialization.capture(self.allocator, self);
        errdefer {
            var s = snap;
            s.deinit(self.allocator);
        }
        const runner = if (self.uploads) |*q| q.runner else return error.NoTaskRunner;
        return serialization.saveFileAsync(self.allocator, runner, snap, path);
    }

    /// Loads a scene file off-thread; caller polls task.isDone() and calls
    /// restoreSceneState(scene, &task.result.?) on the game thread.
    pub fn loadStateFileAsync(self: *Scene, path: []const u8) !*serialization.AsyncLoadTask {
        const runner = if (self.uploads) |*q| q.runner else return error.NoTaskRunner;
        return serialization.loadFileAsync(self.allocator, runner, path);
    }

    // ---- Skybox. ----

    pub fn setSkybox(self: *Scene, cube: CubeTexture) void {
        self.sky.setSkybox(cube);
    }

    pub fn createDefaultSkybox(self: *Scene, config: SkyboxOptions) !void {
        try self.sky.createDefault(self.allocator, config);
    }

    // ---- Lights. ----

    pub fn createHemisphericLight(self: *Scene, name: []const u8, options: lights.HemisphericLightOptions) HemisphericLight {
        return self.lights.setHemispheric(name, options);
    }

    pub fn createPointLight(self: *Scene, name: []const u8, options: PointLightOptions) !*PointLight {
        return self.lights.createPointLight(self.allocator, name, options);
    }

    pub fn createSpotLight(self: *Scene, name: []const u8, options: SpotLightOptions) !*SpotLight {
        return self.lights.createSpotLight(self.allocator, name, options);
    }

    pub fn createDirectionalLight(self: *Scene, name: []const u8, options: DirectionalLightOptions) !*DirectionalLight {
        return self.lights.createDirectionalLight(self.allocator, name, options);
    }

    // ---- Content registries: materials & meshes. ----

    pub fn createStandardMaterial(self: *Scene, name: []const u8) !*StandardMaterial {
        const mat = try self.allocator.create(StandardMaterial);
        mat.* = StandardMaterial.init(name);
        try self.materials.append(self.allocator, mat);
        return mat;
    }

    pub fn createPBRMaterial(self: *Scene, name: []const u8) !*PBRMaterial {
        const mat = try self.allocator.create(PBRMaterial);
        mat.* = PBRMaterial.init(name);
        try self.pbr_materials.append(self.allocator, mat);
        return mat;
    }

    // Creates a custom-shader material bound to a registered shader (by
    // name — build.zig `user_shader_materials` or registerRuntime). Returns
    // null when the shader name is not registered.
    pub fn createShaderMaterial(self: *Scene, name: []const u8, shader_name: []const u8) ?*ShaderMaterial {
        const mat = self.allocator.create(ShaderMaterial) catch return null;
        mat.* = ShaderMaterial.initForShader(shader_name, name) orelse {
            self.allocator.destroy(mat);
            return null;
        };
        self.shader_materials.append(self.allocator, mat) catch {
            self.allocator.destroy(mat);
            return null;
        };
        return mat;
    }

    pub fn destroyPBRMaterial(self: *Scene, mat: *PBRMaterial) void {
        for (self.pbr_materials.items, 0..) |m, i| {
            if (m == mat) {
                _ = self.pbr_materials.swapRemove(i);
                break;
            }
        }
        self.allocator.destroy(mat);
    }

    pub fn removeMesh(self: *Scene, mesh: *Mesh) bool {
        for (self.meshes.items, 0..) |m, i| {
            if (m == mesh) {
                _ = self.meshes.swapRemove(i);
                return true;
            }
        }
        return false;
    }

    pub fn destroyMesh(self: *Scene, mesh: *Mesh) void {
        _ = self.removeMesh(mesh);
        for (self.outline_meshes.items, 0..) |m, i| {
            if (m == mesh) {
                _ = self.outline_meshes.swapRemove(i);
                break;
            }
        }
        mesh.deinit(self.allocator);
        self.allocator.destroy(mesh);
    }

    // ---- Decals / particles / trails / CSG / nav. ----

    pub fn getOrCreateDecalManager(self: *Scene, max_decals: usize) *DecalManager {
        return self.decals.getOrCreate(self, max_decals);
    }

    pub fn updateDecals(self: *Scene, dt: f32) void {
        self.decals.update(dt);
    }

    pub fn createParticleSystem(self: *Scene, name: []const u8, capacity: usize) !*ParticleSystem {
        return self.particles.create(self.allocator, name, capacity);
    }

    /// Explicit particle stepping: GPU simulation modes either run or return
    /// an error (particles.UpdateError) — never a silent CPU downgrade.
    pub fn updateParticles(self: *Scene, dt: f32) particles.UpdateError!void {
        try self.particles.update(dt);
    }

    pub fn createTrailMesh(self: *Scene, name: []const u8, options: TrailOptions) !*TrailMesh {
        return self.trails.create(self, self.allocator, name, options);
    }

    pub fn updateTrails(self: *Scene, dt: f32) void {
        const cam_pos = if (self.active_camera) |cam| cam.getPosition() else Vec3.zero;
        self.trails.update(dt, cam_pos);
    }

    pub fn createCSGMesh(self: *Scene, name: []const u8, csg_solid: *const csg_mod.CSG) !*Mesh {
        return csg_solid.toMesh(self, name);
    }

    pub fn createNavMeshFromTriangles(
        self: *Scene,
        positions: []const [3]f32,
        indices: []const u32,
        max_slope_rad: f32,
    ) !*ai_mod.NavMesh {
        return self.nav.createMeshFromTriangles(self.allocator, positions, indices, max_slope_rad);
    }

    pub fn createNavMeshGrid(
        self: *Scene,
        min_x: f32,
        max_x: f32,
        min_z: f32,
        max_z: f32,
        elevation_y: f32,
        subdiv_x: usize,
        subdiv_z: usize,
        obstacles: []const BoundingBox,
    ) !*ai_mod.NavMesh {
        return self.nav.createMeshGrid(self.allocator, min_x, max_x, min_z, max_z, elevation_y, subdiv_x, subdiv_z, obstacles);
    }

    pub fn createNavAgent(self: *Scene, nav_mesh: *const ai_mod.NavMesh, start_pos: Vec3) !*ai_mod.NavAgent {
        return self.nav.createAgent(self.allocator, nav_mesh, start_pos);
    }

    pub fn updateNavAgents(self: *Scene, dt: f32) void {
        self.nav.updateAgents(dt);
    }

    // ---- Animation / physics / camera / picking / UI / projection. ----

    pub fn updateAnimations(self: *Scene, dt: f32) void {
        scene_animation.updateAnimations(self.animation_groups.items, self.skeletons.items, self.meshes.items, dt);
    }

    pub fn enablePhysics(self: *Scene, gravity: ?Vec3) *PhysicsWorld {
        return self.physics.enable(self.allocator, gravity);
    }

    /// Finds the rigid body previously created for `mesh`, if any.
    pub fn getRigidBody(self: *Scene, mesh: *const Mesh) ?*RigidBody {
        if (self.physics.getWorld()) |pw| {
            return pw.findBody(mesh);
        }
        return null;
    }

    pub fn createRigidBody(self: *Scene, mesh: *Mesh, collider: ColliderType, mass: f32) !*RigidBody {
        const pw = self.physics.getWorld() orelse self.physics.enable(self.allocator, null);
        return pw.createBody(mesh, collider, mass);
    }

    /// Creates a rigid body with collision filter / sensor / event options.
    pub fn createRigidBodyWith(self: *Scene, mesh: *Mesh, collider: ColliderType, mass: f32, options: physics.BodyOptions) !*RigidBody {
        const pw = self.physics.getWorld() orelse self.physics.enable(self.allocator, null);
        return pw.createBodyWith(mesh, collider, mass, options);
    }

    pub fn updatePhysics(self: *Scene, dt: f32) void {
        self.physics.step(dt);
    }

    // ---- Camera management ----

    pub fn addCamera(self: *Scene, entry: CameraEntry) !usize {
        var e = entry;
        if (e.culling_mask == 0xFFFFFFFF and e.camera.getCullingMask() != 0xFFFFFFFF) {
            e.culling_mask = e.camera.getCullingMask();
        }
        const cam_vp = e.camera.getViewport();
        if (e.viewport.x == 0.0 and e.viewport.y == 0.0 and e.viewport.width == 1.0 and e.viewport.height == 1.0 and
            (cam_vp.x != 0.0 or cam_vp.y != 0.0 or cam_vp.width != 1.0 or cam_vp.height != 1.0))
        {
            e.viewport = cam_vp;
        }
        const idx = self.cameras.items.len;
        try self.cameras.append(self.allocator, e);
        if (self.active_camera_index == null) {
            self.active_camera_index = idx;
            self.active_camera = e.camera;
        }
        return idx;
    }

    pub fn removeCamera(self: *Scene, index: usize) void {
        if (index >= self.cameras.items.len) return;
        const entry = self.cameras.orderedRemove(index);
        if (entry.owns_name) {
            self.allocator.free(entry.name);
        }
        if (self.cameras.items.len == 0) {
            self.active_camera_index = null;
            self.active_camera = null;
        } else {
            const cur_idx = self.active_camera_index orelse 0;
            if (cur_idx >= self.cameras.items.len) {
                self.switchCamera(self.cameras.items.len - 1);
            } else {
                self.switchCamera(cur_idx);
            }
        }
    }

    pub fn getCamera(self: *Scene, index: usize) ?*CameraEntry {
        if (index < self.cameras.items.len) return &self.cameras.items[index];
        return null;
    }

    pub fn getCameraByName(self: *Scene, name: []const u8) ?*CameraEntry {
        for (self.cameras.items) |*entry| {
            if (std.mem.eql(u8, entry.name, name)) return entry;
        }
        return null;
    }

    pub fn switchCamera(self: *Scene, index: usize) void {
        if (index >= self.cameras.items.len) return;
        if (self.active_camera_index) |curr_idx| {
            if (curr_idx < self.cameras.items.len and self.active_camera != null) {
                self.cameras.items[curr_idx].camera = self.active_camera.?;
            }
        }
        self.active_camera_index = index;
        self.active_camera = self.cameras.items[index].camera;
    }

    pub fn switchCameraByName(self: *Scene, name: []const u8) bool {
        for (self.cameras.items, 0..) |entry, i| {
            if (std.mem.eql(u8, entry.name, name)) {
                self.switchCamera(i);
                return true;
            }
        }
        return false;
    }

    pub fn nextCamera(self: *Scene) void {
        if (self.cameras.items.len == 0) return;
        const current = self.active_camera_index orelse 0;
        const next_idx = (current + 1) % self.cameras.items.len;
        self.switchCamera(next_idx);
    }

    pub fn prevCamera(self: *Scene) void {
        if (self.cameras.items.len == 0) return;
        const current = self.active_camera_index orelse 0;
        const prev_idx = if (current == 0) self.cameras.items.len - 1 else current - 1;
        self.switchCamera(prev_idx);
    }

    pub fn getActiveCameraIndex(self: Scene) ?usize {
        return self.active_camera_index;
    }

    pub fn getActiveCameraName(self: Scene) ?[]const u8 {
        if (self.active_camera_index) |idx| {
            if (idx < self.cameras.items.len) return self.cameras.items[idx].name;
        }
        if (self.active_camera) |cam| return cam.getName();
        return null;
    }

    pub fn getCameraCount(self: Scene) usize {
        return self.cameras.items.len;
    }

    pub fn setActiveCamera(self: *Scene, cam: ?Camera, owned_name: ?[]const u8) void {
        if (self.active_camera_owned_name) |old| {
            self.allocator.free(old);
        }
        self.active_camera = cam;
        self.active_camera_owned_name = owned_name;
        if (self.active_camera_index) |idx| {
            if (idx < self.cameras.items.len and cam != null) {
                self.cameras.items[idx].camera = cam.?;
            }
        }
    }

    pub fn updateCamera(self: *Scene, dt: f32) void {
        if (self.enable_multi_camera) {
            for (self.cameras.items, 0..) |*entry, i| {
                if (entry.enabled) {
                    entry.camera.update(dt);
                    if (self.active_camera_index == i) {
                        self.active_camera = entry.camera;
                    }
                }
            }
        } else if (self.active_camera) |*cam| {
            cam.update(dt);
            if (self.active_camera_index) |idx| {
                if (idx < self.cameras.items.len) {
                    self.cameras.items[idx].camera = cam.*;
                }
            }
        }
    }

    pub fn createPickingRay(self: *Scene, screen_x: f32, screen_y: f32) Ray {
        const cam = self.active_camera orelse return Ray.new(Vec3.zero, Vec3.forward);
        return scene_picking.createPickingRay(cam, screen_x, screen_y);
    }

    pub fn pickWithRay(self: *Scene, r: Ray) PickingInfo {
        return scene_picking.pickWithRay(self.meshes.items, self.physics.getWorld(), r);
    }

    pub fn pick(self: *Scene, screen_x: f32, screen_y: f32) PickingInfo {
        const r = self.createPickingRay(screen_x, screen_y);
        return self.pickWithRay(r);
    }

    pub fn createUI(self: *Scene) !*UICanvas {
        if (self.ui_canvas == null) {
            self.ui_canvas = try UICanvas.init(self.allocator);
        }
        return &self.ui_canvas.?;
    }

    pub fn getUI(self: *Scene) ?*UICanvas {
        if (self.ui_canvas) |*u| return u;
        return null;
    }

    pub fn projectPoint(self: *Scene, world_pos: Vec3) ?math.Vec2 {
        const cam = self.active_camera orelse return null;
        const w = sapp.widthf();
        const h = sapp.heightf();
        if (w <= 0.0 or h <= 0.0) return null;
        const vp = self.project.viewProjection(cam, w, h);
        return vp.projectPoint(world_pos, w, h);
    }

    pub fn handleEvent(self: *Scene, ev: [*c]const sapp.Event) void {
        if (self.active_camera) |*cam| {
            cam.handleEvent(ev);
            if (self.active_camera_index) |idx| {
                if (idx < self.cameras.items.len) {
                    self.cameras.items[idx].camera = cam.*;
                }
            }
        }
    }

    // ---- Rendering. ----

    /// True when the swapchain color AND depth formats both support MSAA at
    /// runtime (the main target mirrors them; see postprocess_pass.resize).
    /// sg.queryPixelformat is the sokol-provided backend gate.
    fn mainTargetFormatsMsaaCapable() bool {
        const env_def = sg.queryDesc().environment.defaults;
        const color_fmt: sg.PixelFormat = if (env_def.color_format != .DEFAULT and env_def.color_format != .NONE) env_def.color_format else .BGRA8;
        const depth_fmt: sg.PixelFormat = if (env_def.depth_format != .DEFAULT and env_def.depth_format != .NONE) env_def.depth_format else .DEPTH;
        return sg.queryPixelformat(color_fmt).msaa and sg.queryPixelformat(depth_fmt).msaa;
    }

    /// Forward pipeline set matching the MSAA main-target shape; created
    /// lazily (and recreated on count changes) on the first MSAA frame. The
    /// returned pointer aliases Scene state.
    fn ensureForwardMsaa(self: *Scene, samples: i32) *scene_forward.ForwardPipelines {
        if (self.forward_msaa == null or self.forward_msaa.?.sample_count != samples) {
            if (self.forward_msaa) |*fw| fw.deinit();
            if (self.forward.family_shaders) |fs| {
                self.forward_msaa = scene_forward.ForwardPipelines.initSampledWithShaders(samples, fs);
            } else {
                self.forward_msaa = scene_forward.ForwardPipelines.initSampled(samples);
            }
        }
        return &self.forward_msaa.?;
    }

    fn ensureClearResources(self: *Scene, samples: i32) void {
        if (self.clear_vb.id == 0) {
            self.clear_vb = sg.makeBuffer(.{
                .usage = .{ .vertex_buffer = true, .dynamic_update = true },
                .size = 16 * 6 * @sizeOf(debug_pass.Vertex),
            });
        }
        if (self.clear_shader.id == 0) {
            self.clear_shader = sg.makeShader(debug_shd.debugShaderDesc(sg.queryBackend()));
        }
        const target_pip = if (samples > 1) &self.clear_pipeline_msaa else &self.clear_pipeline;
        if (target_pip.id == 0) {
            var pip_desc = sg.PipelineDesc{
                .shader = self.clear_shader,
                .index_type = .NONE,
                .primitive_type = .TRIANGLES,
                .depth = .{
                    .compare = .ALWAYS,
                    .write_enabled = true,
                },
                .cull_mode = .NONE,
                .sample_count = samples,
            };
            pip_desc.layout.buffers[0] = .{ .stride = @sizeOf(debug_pass.Vertex) };
            pip_desc.layout.attrs[debug_shd.ATTR_debug_position] = .{
                .format = .FLOAT3,
                .offset = @offsetOf(debug_pass.Vertex, "position"),
            };
            pip_desc.layout.attrs[debug_shd.ATTR_debug_color0] = .{
                .format = .FLOAT4,
                .offset = @offsetOf(debug_pass.Vertex, "color"),
            };
            target_pip.* = sg.makePipeline(pip_desc);
            if (sg.queryPipelineState(target_pip.*) != .VALID) {
                std.debug.print("[CLEAR PIPELINE FAILED]: shader_state={}, pip_state={}\n", .{
                    sg.queryShaderState(self.clear_shader),
                    sg.queryPipelineState(target_pip.*),
                });
            }
        }
    }

    fn clearCurrentViewport(self: *Scene, color: Color4, samples: i32) void {
        self.ensureClearResources(samples);
        const pip = if (samples > 1) self.clear_pipeline_msaa else self.clear_pipeline;
        if (pip.id == 0 or self.clear_vb.id == 0 or sg.queryPipelineState(pip) != .VALID) return;

        const clear_verts = [_]debug_pass.Vertex{
            .{ .position = .{ -1.0, -1.0, 1.0 }, .color = .{ color.r, color.g, color.b, color.a } },
            .{ .position = .{ 1.0, -1.0, 1.0 }, .color = .{ color.r, color.g, color.b, color.a } },
            .{ .position = .{ 1.0, 1.0, 1.0 }, .color = .{ color.r, color.g, color.b, color.a } },
            .{ .position = .{ -1.0, -1.0, 1.0 }, .color = .{ color.r, color.g, color.b, color.a } },
            .{ .position = .{ 1.0, 1.0, 1.0 }, .color = .{ color.r, color.g, color.b, color.a } },
            .{ .position = .{ -1.0, 1.0, 1.0 }, .color = .{ color.r, color.g, color.b, color.a } },
        };
        const offset = sg.appendBuffer(self.clear_vb, sg.asRange(&clear_verts));
        if (offset < 0) return;

        sg.applyPipeline(pip);
        var bind = sg.Bindings{};
        bind.vertex_buffers[0] = self.clear_vb;
        bind.vertex_buffer_offsets[0] = offset;
        sg.applyBindings(bind);

        const vs_params = debug_shd.VsParams{
            .mvp = Mat4.identity,
        };
        sg.applyUniforms(debug_shd.UB_vs_params, sg.asRange(&vs_params));
        sg.draw(0, 6, 1);
    }

    fn renderSceneView(
        self: *Scene,
        cam_snap: scene_snapshot.CameraSnapshot,
        samples: i32,
        snap: *const scene_snapshot.SceneFrameSnapshot,
        env: scene_draw.Environment,
    ) void {
        const view_proj = cam_snap.view_proj;
        const eye = cam_snap.eye;

        self.queues.reset();

        scene_render_queue.buildFrameQueues(.{
            .allocator = self.allocator,
            .meshes = self.meshes.items,
            .frame_id = self.frame_id,
            .view_proj = view_proj,
            .eye = eye,
            .cull_frustum = self.enable_frustum_culling,
            .cull_occlusion = self.enable_occlusion_culling,
            .culling_mask = cam_snap.culling_mask,
            .occlusion_culler = &self.occlusion_culler,
            .stats = &self.stats,
            .queues = &self.queues,
            .default_white_id = self.default_white_texture.view.id,
            .default_material = &self.default_material,
            .default_white = &self.default_white_texture,
            .default_normal = &self.default_normal_texture,
            .default_cube = &self.default_cube_texture,
            .sky_texture = env.sky_texture,
            .ibl_intensity = env.ibl_intensity,
            .default_morph_view = self.forward.default_morph_view,
            .thread_pool = jobs.global,
        });

        std.mem.sort(RenderMeshItem, self.queues.items.items, {}, scene_render_queue.sortRenderItems);
        std.mem.sort(RenderMeshItem, self.queues.transparent.items, {}, scene_render_queue.sortTransparentBackToFront);

        const frame_ctx = FrameContext{
            .view_proj = view_proj,
            .eye = eye,
            .sun_dir = snap.sun_dir,
            .sun_color = snap.sun_color,
            .sun_intensity = snap.sun_intensity,
            .cascades = snap.cascades,
            .light_counts = snap.light_pack.counts,
            .point_pos_range = snap.light_pack.point_pos_range,
            .point_color_int = snap.light_pack.point_color_int,
            .spot_pos_range = snap.light_pack.spot_pos_range,
            .spot_dir_inner = snap.light_pack.spot_dir_inner,
            .spot_color_outer = snap.light_pack.spot_color_outer,
            .spot_intensity = snap.light_pack.spot_intensity,
            .spot_view_proj = snap.light_pack.spot_view_proj,
            .spot_shadow_params = snap.light_pack.spot_shadow_params,
        };

        var current_pipeline_id: u32 = 0;

        // Opaque regular meshes first (front-to-back, early-Z).
        for (self.queues.items.items) |item| {
            scene_draw.drawRegularItem(&env, item, &frame_ctx, &current_pipeline_id);
        }

        // Opaque instanced meshes.
        for (self.queues.opaque_instanced.items) |mesh| {
            scene_draw.drawInstancedMesh(&env, mesh, &frame_ctx, &current_pipeline_id);
        }

        // Transparent regular meshes (strict back-to-front, blended).
        for (self.queues.transparent.items) |item| {
            scene_draw.drawRegularItem(&env, item, &frame_ctx, &current_pipeline_id);
        }

        // Transparent instanced meshes.
        for (self.queues.transparent_instanced.items) |mesh| {
            scene_draw.drawInstancedMesh(&env, mesh, &frame_ctx, &current_pipeline_id);
        }

        // Inverse-hull outline for highlighted meshes
        self.postfx.renderOutlineExplicit(
            view_proj,
            eye,
            self.render_outline_meshes.items,
            samples,
            &self.stats,
            snap.outline_enabled,
            snap.outline_color,
            snap.outline_width_px,
        );

        // Physics debug lines
        self.physics.renderDebug(self.allocator, view_proj, samples, &self.stats);

        // Skybox Pass
        self.sky.render(cam_snap.camera, cam_snap.aspect, self.default_cube_texture, samples, &self.stats);

        // Particle Pass
        self.particles.render(cam_snap.camera, cam_snap.aspect, samples, &self.stats);
    }

    /// Packs the current camera, light, shadow, and environment state into an immutable
    /// frame snapshot that can be published to the render thread.
    pub fn packFrameSnapshot(self: *Scene, aspect: f32, cur_w: i32, cur_h: i32) scene_snapshot.SceneFrameSnapshot {
        const w = if (cur_w > 0) cur_w else sapp.width();
        const h = if (cur_h > 0) cur_h else sapp.height();
        const eff_aspect = if (aspect > 0.0) aspect else (if (h > 0) @as(f32, @floatFromInt(w)) / @as(f32, @floatFromInt(h)) else 1.0);

        var snap = scene_snapshot.SceneFrameSnapshot{
            .frame_id = self.frame_id,
            .aspect = eff_aspect,
            .screen_w = w,
            .screen_h = h,
        };

        const primary_cam_opt = self.active_camera orelse (if (self.cameras.items.len > 0) self.cameras.items[0].camera else null);
        if (primary_cam_opt == null) {
            snap.has_camera = false;
            return snap;
        }
        snap.has_camera = true;

        snap.enable_multi_camera = self.enable_multi_camera;
        snap.active_camera_idx = self.active_camera_index orelse 0;
        snap.camera_count = @min(self.cameras.items.len, scene_snapshot.MAX_CAMERAS);

        for (0..snap.camera_count) |i| {
            const entry = self.cameras.items[i];
            const cam_rect = entry.viewport.toPixelRect(w, h);
            const cam_aspect = cam_rect.aspect();
            snap.cameras[i] = scene_snapshot.CameraSnapshot{
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
            snap.primary_cam = scene_snapshot.CameraSnapshot{
                .camera = primary_cam_opt.?,
                .view_proj = primary_cam_opt.?.getViewProjection(cam_aspect),
                .eye = primary_cam_opt.?.getPosition(),
                .viewport = vp,
                .culling_mask = primary_cam_opt.?.getCullingMask(),
                .aspect = cam_aspect,
            };
        }

        snap.sun_dir = self.lights.sunDirection();
        snap.sun_color = self.lights.sunColor();
        snap.sun_intensity = self.lights.sunIntensity();
        snap.cascades = self.shadows.computeCascades(snap.primary_cam.camera, snap.primary_cam.aspect, snap.sun_dir);

        var lp = self.light_pack;
        _ = self.light_handoff.takeLatest(&lp);
        self.light_pack = lp;
        snap.light_pack = lp;

        snap.shadows_enabled = self.shadows.enabled;
        snap.shadow_uniforms = self.shadows.uniformState(self.lights.hemi.ground_color);
        snap.sky_texture = self.sky.texture;
        snap.ibl_intensity = self.sky.ibl_intensity;
        snap.clear_color = self.clear_color;
        snap.msaa_sample_count = self.msaa_sample_count;
        snap.post_process = self.post_process;
        snap.ssao = self.ssao;
        snap.outline_enabled = self.postfx.outline_enabled;
        snap.outline_color = self.postfx.outline_color;
        snap.outline_width_px = self.postfx.outline_width_px;

        return snap;
    }

    /// Publishes a complete frame snapshot through the lock-free mailbox.
    pub fn publishFrameSnapshot(self: *Scene, aspect: f32, cur_w: i32, cur_h: i32) void {
        const snap = self.packFrameSnapshot(aspect, cur_w, cur_h);
        if (self.frame_handoff.claim()) |i| {
            self.frame_handoff.slot(i).* = snap;
            self.frame_handoff.publish(i);
        } else {
            self.frame_snapshot = snap;
        }
    }

    /// Stage 3: prepares GPU uploads and acquires the frame snapshot.
    /// In threaded mode, calling this under the brief handoff lock (<0.05 ms)
    /// allows render() to execute all GPU passes completely unlocked.
    pub fn prepareFrame(self: *Scene) void {
        if (self.uploads) |*q| _ = q.drain();
        self.flushPendingGpuUploads();

        var snap = self.frame_snapshot;
        if (self.frame_handoff.takeLatest(&snap)) {
            self.frame_snapshot = snap;
        } else {
            const cur_w = sapp.width();
            const cur_h = sapp.height();
            const aspect = if (cur_h > 0) @as(f32, @floatFromInt(cur_w)) / @as(f32, @floatFromInt(cur_h)) else 1.0;
            self.frame_snapshot = self.packFrameSnapshot(aspect, cur_w, cur_h);
        }

        self.render_outline_meshes.clearRetainingCapacity();
        self.render_outline_meshes.appendSlice(self.allocator, self.outline_meshes.items) catch {};

        self.frame_prepared = true;
    }

    /// Stage 3, slice 2: update-phase light packing. Selects the top-k
    /// point/spot lights for the camera (advancing the incumbency-
    /// hysteresis fades with `dt`) and stores the uniform-ready pack.
    /// Called once per frame BEFORE render(); render consumes
    /// `self.light_pack` without touching light state.
    pub fn updateLights(self: *Scene, dt: f32) void {
        // Zero-eye fallback keeps the pack defined for camera-less scenes
        // (render early-returns without a camera anyway).
        const eye = if (self.active_camera) |cam|
            cam.getPosition()
        else if (self.cameras.items.len > 0)
            self.cameras.items[0].camera.getPosition()
        else
            Vec3.zero;
        const pack = self.lights.packFrame(eye, self.shadows.enabled, dt);
        // Publish through the mailbox. Same-thread today (publish is
        // visible to this frame's render takeLatest); after the split the
        // same call sequence crosses the thread boundary unchanged.
        if (self.light_handoff.claim()) |i| {
            self.light_handoff.slot(i).* = pack;
            self.light_handoff.publish(i);
        } else {
            // Both slots still published (consumer lagging): fall back to
            // the consumed copy as the carrier.
            self.light_pack = pack;
        }
    }

    /// Stage 3, slice 2: the game-side update entry point. Everything the
    /// simulation advances per frame, in one call, in the canonical order
    /// (camera -> lights -> physics -> animations -> particles -> decals);
    /// render() then consumes the published frame values (light_pack, frame_snapshot)
    /// without simulating anything itself.
    ///
    /// Deliberately NOT included: `updateTrails` and `updateNavAgents` —
    /// both require real-seconds dt (the 60fps-normalized dt breaks their
    /// SI tuning), so apps drive them explicitly with their own time base.
    pub fn update(self: *Scene, dt: f32) particles.UpdateError!void {
        self.updateCamera(dt);
        self.updateLights(dt);
        self.updatePhysics(dt);
        self.updateAnimations(dt);
        try self.updateParticles(dt);
        self.updateDecals(dt);

        const cur_w = sapp.width();
        const cur_h = sapp.height();
        const aspect = if (cur_h > 0) @as(f32, @floatFromInt(cur_w)) / @as(f32, @floatFromInt(cur_h)) else 1.0;
        self.publishFrameSnapshot(aspect, cur_w, cur_h);
    }

    /// Stage 3: uploads the GPU buffers that the update phase staged
    /// (particle instances, trail geometry). Called at render start so
    /// every sg.* touch stays on the context thread; the update phase is
    /// free of sg.* calls.
    pub fn flushPendingGpuUploads(self: *Scene) void {
        for (self.particles.systems.items) |ps| ps.flushGpuUploads();
        for (self.trails.meshes.items) |tm| tm.flushGpuUploads();
        for (self.meshes.items) |m| m.flushGpuUploads();
    }

    pub fn render(self: *Scene) void {
        if (!self.frame_prepared) {
            self.prepareFrame();
        }
        self.frame_prepared = false;

        const snap = &self.frame_snapshot;
        if (!snap.has_camera) return;

        const cur_w = if (snap.screen_w > 0) snap.screen_w else sapp.width();
        const cur_h = if (snap.screen_h > 0) snap.screen_h else sapp.height();

        // Effective main-target MSAA sample count for this frame
        // (scene/msaa.zig holds the policy and the backend matrix).
        const samples = scene_msaa.effectiveSampleCount(snap.msaa_sample_count, .{
            .post_enabled = snap.post_process.enabled,
            .formats_msaa_capable = mainTargetFormatsMsaaCapable(),
            .backend = sg.queryBackend(),
        });
        if (snap.post_process.enabled and snap.msaa_sample_count > 1 and samples == 1) {
            // Only the runtime format gate can nullify a > 1 request here
            // (clamping lands on a valid count, post-off forces 1 upstream).
            _ = self.warn_msaa_format.warn(
                "msaa: x{} requested but the main target formats cannot MSAA on this backend; running 1x",
                .{snap.msaa_sample_count},
            );
        }

        self.stats = .{};
        self.frame_id +%= 1;

        // 1. Directional Light Cascaded Shadow View-Projections
        const cascades = snap.cascades;
        const light_pack = snap.light_pack;

        // ==============================================
        // PASS 1: OFFSCREEN SHADOW DEPTH PASS
        // ==============================================
        if (snap.shadows_enabled) {
            const shadow_draws = self.shadows.pass.render(self.meshes.items, self.frame_id, cascades, light_pack.spot_shadows[0..light_pack.num_spot_shadows]);
            self.stats.shadow_draw_calls += shadow_draws;
            self.stats.draw_calls += shadow_draws;
        }

        // ==============================================
        // PASS 2: MAIN SCENE RENDER PASS
        // ==============================================
        var main_pass_action = sg.PassAction{};
        main_pass_action.colors[0] = .{
            .load_action = .CLEAR,
            .clear_value = .{
                .r = snap.clear_color.r,
                .g = snap.clear_color.g,
                .b = snap.clear_color.b,
                .a = snap.clear_color.a,
            },
        };
        main_pass_action.depth = .{
            .load_action = .CLEAR,
            .clear_value = 1.0,
            .store_action = .STORE,
        };

        // Offscreen target when post-processing is on, swapchain otherwise.
        self.postfx.beginMainPass(main_pass_action, snap.post_process.enabled, samples, cur_w, cur_h);

        const env = scene_draw.Environment{
            // Pipeline set must match the main target's sample count: the
            // 1x set for the legacy/swapchain path, the MSAA twin otherwise.
            .pipelines = if (samples > 1) self.ensureForwardMsaa(samples) else &self.forward,
            .stats = &self.stats,
            .default_material = &self.default_material,
            .default_white = &self.default_white_texture,
            .default_normal = &self.default_normal_texture,
            .default_cube = &self.default_cube_texture,
            .sky_texture = snap.sky_texture orelse self.sky.texture,
            .ibl_intensity = snap.ibl_intensity,
            .shadow_pass = &self.shadows.pass,
            .shadow_uniforms = snap.shadow_uniforms,
        };

        if (snap.enable_multi_camera and snap.camera_count > 0) {
            const active_idx = snap.active_camera_idx;
            const primary_snap = if (active_idx < snap.camera_count) snap.cameras[active_idx] else snap.primary_cam;
            const primary_rect = primary_snap.viewport.toPixelRect(cur_w, cur_h);
            sg.applyViewport(primary_rect.x, primary_rect.y, primary_rect.width, primary_rect.height, true);
            sg.applyScissorRect(primary_rect.x, primary_rect.y, primary_rect.width, primary_rect.height, true);
            self.renderSceneView(primary_snap, samples, snap, env);

            for (snap.cameras[0..snap.camera_count], 0..) |entry, i| {
                if (i == active_idx or !entry.enabled) continue;
                const rect = entry.viewport.toPixelRect(cur_w, cur_h);
                sg.applyViewport(rect.x, rect.y, rect.width, rect.height, true);
                sg.applyScissorRect(rect.x, rect.y, rect.width, rect.height, true);

                if (entry.clear_viewport) {
                    const clr = entry.clear_color orelse snap.clear_color;
                    self.clearCurrentViewport(clr, samples);
                }

                self.renderSceneView(entry, samples, snap, env);
            }
            // Restore full viewport
            sg.applyViewport(0, 0, cur_w, cur_h, true);
            sg.applyScissorRect(0, 0, cur_w, cur_h, true);
        } else {
            const vp = snap.primary_cam.viewport;
            const rect = vp.toPixelRect(cur_w, cur_h);
            sg.applyViewport(rect.x, rect.y, rect.width, rect.height, true);
            sg.applyScissorRect(rect.x, rect.y, rect.width, rect.height, true);

            self.renderSceneView(snap.primary_cam, samples, snap, env);

            if (rect.width != cur_w or rect.height != cur_h or rect.x != 0 or rect.y != 0) {
                sg.applyViewport(0, 0, cur_w, cur_h, true);
                sg.applyScissorRect(0, 0, cur_w, cur_h, true);
            }
        }

        if (!snap.post_process.enabled) {
            if (self.ui_canvas) |*ui_c| {
                ui_c.render(sapp.widthf(), sapp.heightf());
                self.stats.post_draw_calls += 1;
                self.stats.draw_calls += 1;
            }
        }

        sg.endPass();

        // ==============================================
        // PASS 2.5 (SSAO) + 2.75 (bloom) + 3 (composite & UI overlay)
        // ==============================================
        self.postfx.renderChain(.{
            .post = snap.post_process,
            .ssao = snap.ssao,
            .camera = snap.primary_cam.camera,
            .aspect = snap.primary_cam.aspect,
            .view_proj = snap.primary_cam.view_proj,
            .eye = snap.primary_cam.eye,
            .sun_dir = snap.sun_dir,
            .sun_color = snap.sun_color,
            .default_white_view = self.default_white_texture.view,
            .main_samples = samples,
            .ui = if (self.ui_canvas) |*u| u else null,
            .stats = &self.stats,
        }, cur_w, cur_h);

        sg.commit();
    }

    pub fn deinit(self: *Scene) void {
        // In-flight decodes target material fields; join them before any
        // mesh/material teardown can free those fields.
        if (self.uploads) |*q| {
            q.deinit();
            self.uploads = null;
        }
        for (self.cameras.items) |entry| {
            if (entry.owns_name) {
                self.allocator.free(entry.name);
            }
        }
        self.cameras.deinit(self.allocator);
        if (self.clear_vb.id != 0) sg.destroyBuffer(self.clear_vb);
        if (self.clear_pipeline.id != 0) sg.destroyPipeline(self.clear_pipeline);
        if (self.clear_pipeline_msaa.id != 0) sg.destroyPipeline(self.clear_pipeline_msaa);
        if (self.clear_shader.id != 0) sg.destroyShader(self.clear_shader);

        if (self.active_camera_owned_name) |n| {
            self.allocator.free(n);
            self.active_camera_owned_name = null;
        }
        self.active_camera = null;

        self.decals.deinit();

        scene_content.deinitMeshes(self.allocator, &self.meshes);
        scene_content.deinitMaterials(self.allocator, &self.materials);
        scene_content.deinitPbrMaterials(self.allocator, &self.pbr_materials);
        scene_content.deinitShaderMaterials(self.allocator, &self.shader_materials);

        self.lights.deinit(self.allocator);

        self.queues.deinit(self.allocator);

        self.trails.deinit(self.allocator);
        self.nav.deinit(self.allocator);

        self.default_white_texture.deinit();
        self.default_normal_texture.deinit();
        self.default_cube_texture.deinit();

        self.shadows.deinit();
        self.sky.deinit();

        self.forward.deinit();
        if (self.forward_msaa) |*fw| fw.deinit();
        self.forward_msaa = null;

        scene_content.deinitAnimations(self.allocator, &self.animation_groups, &self.skeletons);

        self.outline_meshes.deinit(self.allocator);
        self.render_outline_meshes.deinit(self.allocator);
        self.postfx.deinit();

        self.particles.deinit(self.allocator);

        self.physics.deinit(self.allocator);

        if (self.ui_canvas) |*u| {
            u.deinit();
        }
    }
};

test "Scene camera switching and cycling" {
    const ally = std.testing.allocator;
    var scene: Scene = undefined;
    scene.allocator = ally;
    scene.cameras = .empty;
    scene.active_camera_index = null;
    scene.active_camera = null;
    scene.active_camera_owned_name = null;
    scene.enable_multi_camera = true;

    defer {
        for (scene.cameras.items) |entry| {
            if (entry.owns_name) ally.free(entry.name);
        }
        scene.cameras.deinit(ally);
    }

    const cam1 = Camera{ .free = camera_mod.FreeCamera.init("Cam1", .{}) };
    const cam2 = Camera{ .arc_rotate = camera_mod.ArcRotateCamera.init("Cam2", .{}) };
    const cam3 = Camera{ .fly = camera_mod.FlyCamera.init("Cam3", .{}) };

    const idx0 = try scene.addCamera(.{ .name = "Cam1", .camera = cam1 });
    try std.testing.expectEqual(@as(usize, 0), idx0);
    try std.testing.expectEqual(@as(usize, 0), scene.getActiveCameraIndex().?);
    try std.testing.expectEqualStrings("Cam1", scene.getActiveCameraName().?);

    _ = try scene.addCamera(.{ .name = "Cam2", .camera = cam2 });
    _ = try scene.addCamera(.{ .name = "Cam3", .camera = cam3 });
    try std.testing.expectEqual(@as(usize, 3), scene.getCameraCount());

    // Cycle next: 0 -> 1 -> 2 -> 0
    scene.nextCamera();
    try std.testing.expectEqual(@as(usize, 1), scene.getActiveCameraIndex().?);
    try std.testing.expectEqualStrings("Cam2", scene.getActiveCameraName().?);

    scene.nextCamera();
    try std.testing.expectEqual(@as(usize, 2), scene.getActiveCameraIndex().?);
    try std.testing.expectEqualStrings("Cam3", scene.getActiveCameraName().?);

    scene.nextCamera();
    try std.testing.expectEqual(@as(usize, 0), scene.getActiveCameraIndex().?);

    // Cycle prev: 0 -> 2 -> 1
    scene.prevCamera();
    try std.testing.expectEqual(@as(usize, 2), scene.getActiveCameraIndex().?);

    // Switch by name
    try std.testing.expect(scene.switchCameraByName("Cam2"));
    try std.testing.expectEqual(@as(usize, 1), scene.getActiveCameraIndex().?);
    try std.testing.expect(!scene.switchCameraByName("NonExistent"));
}

test "updateLights packs point lights into the frame payload" {
    const alloc = std.testing.allocator;
    var scene = @import("testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);

    _ = try scene.createPointLight("probe", .{
        .position = Vec3.zero,
        .intensity = 10.0,
        .range = 10.0,
    });

    // One point light packed into slot 0 (the light sits at the pack eye,
    // so its score is maximal). Hysteresis fades a newly seen light in
    // over fade_time (0.25 s), so the first pack carries a partial factor.
    // The pack travels through the mailbox: render-side consumption goes
    // via takeLatest, exactly what the render pass does.
    scene.updateLights(0.016);
    var pack_out: scene_lights.LightRig.FramePack = undefined;
    try std.testing.expect(scene.light_handoff.takeLatest(&pack_out));
    // First sight: packed with a partial enter-fade factor (intensity
    // lane = intensity * factor < full intensity).
    try std.testing.expectEqual(@as(f32, 1.0), pack_out.counts[0]);
    try std.testing.expect(pack_out.point_color_int[0][3] < 10.0);
    var frame: usize = 0;
    while (frame < 32) : (frame += 1) {
        scene.updateLights(0.016);
        // Consume per frame like render does: a fresh pack lands each
        // update, so the mailbox never lags.
        _ = scene.light_handoff.takeLatest(&pack_out);
    }
    // Fully faded in (0.512 s >= fade_time): the intensity lane carries
    // the light's exact intensity, and the incumbent keeps its slot.
    try std.testing.expectEqual(@as(f32, 1.0), pack_out.counts[0]);
    try std.testing.expectEqual(@as(f32, 10.0), pack_out.point_color_int[0][3]);
    // A second take without a new publish consumes nothing (last pack
    // stands on the render side) — the PIP multi-render pattern.
    try std.testing.expect(!scene.light_handoff.takeLatest(&pack_out));
}

test "publishFrameSnapshot and prepareFrame snapshot handoff" {
    const alloc = std.testing.allocator;
    var scene = @import("testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);

    const cam = Camera{ .free = camera_mod.FreeCamera.init("Cam1", .{}) };
    _ = try scene.addCamera(.{ .name = "Cam1", .camera = cam });

    scene.publishFrameSnapshot(16.0 / 9.0, 1920, 1080);
    try std.testing.expect(!scene.frame_prepared);

    scene.prepareFrame();
    try std.testing.expect(scene.frame_prepared);
    try std.testing.expect(scene.frame_snapshot.has_camera);
    try std.testing.expectEqual(@as(i32, 1920), scene.frame_snapshot.screen_w);
    try std.testing.expectEqual(@as(i32, 1080), scene.frame_snapshot.screen_h);
}

