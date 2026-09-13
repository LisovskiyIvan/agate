const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const sapp = sokol.app;
const postprocess = @import("postprocess.zig");
pub const PostProcessConfig = postprocess.PostProcessConfig;
pub const TonemappingType = postprocess.TonemappingType;
const ssao = @import("ssao.zig");
pub const SSAOConfig = ssao.SSAOConfig;
const particles = @import("particles.zig");
pub const ParticleSystem = particles.ParticleSystem;
pub const ParticleBlendMode = particles.ParticleBlendMode;
pub const Particle = particles.Particle;

const math = @import("math");
const Vec3 = math.Vec3;
const Mat4 = math.Mat4;
const Color3 = math.Color3;
const Color4 = math.Color4;
const BoundingBox = math.BoundingBox;
const Ray = math.Ray;

const physics = @import("physics.zig");
pub const PhysicsWorld = physics.PhysicsWorld;
pub const RigidBody = physics.RigidBody;
pub const PickingInfo = physics.PickingInfo;
pub const ColliderType = physics.ColliderType;

const ui = @import("ui.zig");
pub const UICanvas = ui.UICanvas;
pub const UIVertex = ui.UIVertex;

const AnimationGroup = @import("animation/animation.zig").AnimationGroup;
const Skeleton = @import("animation/skeleton.zig").Skeleton;

const Camera = @import("camera.zig").Camera;
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
pub const DecalManager = decal_mod.DecalManager;
const trail_mod = @import("mesh/trail.zig");
pub const TrailMesh = trail_mod.TrailMesh;
pub const TrailOptions = trail_mod.TrailOptions;
pub const DecalProjector = decal_mod.DecalProjector;
const csg_mod = @import("mesh/csg.zig");
pub const CSG = csg_mod.CSG;
const ai_mod = @import("ai.zig");
pub const NavMesh = ai_mod.NavMesh;
pub const NavNode = ai_mod.NavNode;
pub const NavAgent = ai_mod.NavAgent;
const StandardMaterial = @import("material.zig").StandardMaterial;
const PBRMaterial = @import("material.zig").PBRMaterial;
const ShaderMaterial = @import("material.zig").ShaderMaterial;
const Texture = @import("texture.zig").Texture;
const CubeTexture = @import("texture.zig").CubeTexture;
const SkyboxConfig = @import("texture.zig").SkyboxConfig;
const visibility = @import("visibility/mod.zig");

// Scene subsystems. Each owns its state (and GPU resources) plus the logic
// that belongs to it; Scene is the owner/orchestrator facade. Subsystems
// never import scene.zig — Scene passes everything they need as parameters.
const scene_stats = @import("scene/stats.zig");
pub const SceneStats = scene_stats.SceneStats;
const scene_render_queue = @import("scene/render_queue.zig");
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
const scene_draw = @import("scene/draw.zig");

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
    active_camera: ?Camera = null,
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
    post_process: PostProcessConfig = .{},
    ssao: SSAOConfig = .{},

    // Highlighted meshes for the inverse-hull outline (postfx holds the
    // settings + pass). Kept flat: mock scenes in mesh tests construct it.
    outline_meshes: std.ArrayListUnmanaged(*Mesh) = .empty,

    // 2D & 3D UI canvas (lazy; created via createUI()).
    ui_canvas: ?UICanvas = null,

    // Frame uniform types shared with the draw path (aliases kept so
    // `Scene.FrameContext` keeps resolving for tests and tooling).
    pub const FrameContext = scene_uniforms.FrameContext;
    pub const FrameUniforms = scene_uniforms.FrameUniforms;

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
    }

    pub fn init(allocator: std.mem.Allocator) Scene {
        var self: Scene = undefined;
        self.initInto(allocator);
        return self;
    }

    // ---- Offscreen targets & post-processing config. ----

    pub fn resizeOffscreen(self: *Scene, width: i32, height: i32) void {
        self.postfx.resizeAll(width, height);
    }

    pub fn setPostProcess(self: *Scene, config: PostProcessConfig) void {
        self.post_process = config;
    }

    pub fn setSSAO(self: *Scene, config: SSAOConfig) void {
        self.ssao = config;
    }

    // ---- Skybox. ----

    pub fn setSkybox(self: *Scene, cube: CubeTexture) void {
        self.sky.setSkybox(cube);
    }

    pub fn createDefaultSkybox(self: *Scene, config: SkyboxConfig) !void {
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

    pub fn updateParticles(self: *Scene, dt: f32) void {
        self.particles.update(dt);
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

    pub fn getPhysicsWorld(self: *Scene) ?*PhysicsWorld {
        return self.physics.getWorld();
    }

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

    pub fn updateCamera(self: *Scene, dt: f32) void {
        if (self.active_camera) |*cam| {
            cam.update(dt);
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
        }
    }

    // ---- Shadow / uniform plumbing kept for API compatibility. ----

    pub fn computeCascades(self: *Scene, camera: Camera, aspect: f32) [4]Mat4 {
        return self.shadows.computeCascades(camera, aspect, self.lights.sunDirection());
    }

    // A mesh is transparent when its material opts into .blend alpha mode.
    // Meshes without a material render opaque (legacy behavior).
    pub fn materialIsTransparent(mat: ?@import("material.zig").Material) bool {
        return scene_render_queue.materialIsTransparent(mat);
    }

    pub fn frameUniforms(self: *Scene, mesh: *const Mesh, ctx: FrameContext) FrameUniforms {
        var state = self.shadows.uniformState(self.lights.hemi.ground_color);
        state.mesh_receive_shadows = mesh.receive_shadows;
        return scene_uniforms.buildFrameUniforms(state, ctx);
    }

    // ---- Rendering. ----

    pub fn render(self: *Scene) void {
        const camera = self.active_camera orelse return;
        const aspect = sapp.widthf() / sapp.heightf();
        // Sun resolved once per frame; reused by cascades, mesh uniforms, postprocess.
        const sun_dir = self.lights.sunDirection();
        const sun_color = self.lights.sunColor();
        const sun_intensity = self.lights.sunIntensity();
        const view_proj = camera.getViewProjection(aspect);
        const eye = camera.getPosition();

        self.stats = .{};
        self.frame_id +%= 1;
        self.queues.reset();

        // 1. Directional Light Cascaded Shadow View-Projections
        const cascades = self.shadows.computeCascades(camera, aspect, sun_dir);

        // Phase -1/0: occluder rasterization, frustum/occlusion culling,
        // LOD picking, instance buffers and queue fill.
        scene_render_queue.buildFrameQueues(.{
            .allocator = self.allocator,
            .meshes = self.meshes.items,
            .frame_id = self.frame_id,
            .view_proj = view_proj,
            .eye = eye,
            .cull_frustum = self.enable_frustum_culling,
            .cull_occlusion = self.enable_occlusion_culling,
            .occlusion_culler = &self.occlusion_culler,
            .stats = &self.stats,
            .queues = &self.queues,
            .default_white_id = self.default_white_texture.view.id,
        });

        // Pack Point & Spot Lights (top-k selection + uniform arrays +
        // spot shadow infos for the depth pass). dt drives the slot
        // hand-off fades (incumbency-hysteresis light selection).
        const dt: f32 = @floatCast(sapp.frameDuration());
        const light_pack = self.lights.packFrame(eye, self.shadows.enabled, dt);

        // ==============================================
        // PASS 1: OFFSCREEN SHADOW DEPTH PASS
        // ==============================================
        if (self.shadows.enabled) {
            const shadow_draws = self.shadows.pass.render(self.meshes.items, self.frame_id, cascades, light_pack.spot_shadows[0..light_pack.num_spot_shadows]);
            self.stats.shadow_draw_calls += shadow_draws;
            self.stats.draw_calls += shadow_draws;
        }

        // ==============================================
        // PASS 2: MAIN SCENE RENDER PASS
        // ==============================================
        const cur_w = sapp.width();
        const cur_h = sapp.height();

        var main_pass_action = sg.PassAction{};
        main_pass_action.colors[0] = .{
            .load_action = .CLEAR,
            .clear_value = .{
                .r = self.clear_color.r,
                .g = self.clear_color.g,
                .b = self.clear_color.b,
                .a = self.clear_color.a,
            },
        };
        main_pass_action.depth = .{
            .load_action = .CLEAR,
            .clear_value = 1.0,
            .store_action = .STORE,
        };

        // Offscreen target when post-processing is on, swapchain otherwise.
        self.postfx.beginMainPass(main_pass_action, self.post_process.enabled, cur_w, cur_h);

        // State sorting: opaque items group by shader type and textures,
        // Front-to-Back Early-Z. Transparent items sort strictly
        // back-to-front in their own queue and draw after all opaque work.
        std.mem.sort(RenderMeshItem, self.queues.items.items, {}, scene_render_queue.sortRenderItems);
        std.mem.sort(RenderMeshItem, self.queues.transparent.items, {}, scene_render_queue.sortTransparentBackToFront);

        const frame_ctx = FrameContext{
            .view_proj = view_proj,
            .eye = eye,
            .sun_dir = sun_dir,
            .sun_color = sun_color,
            .sun_intensity = sun_intensity,
            .cascades = cascades,
            .light_counts = light_pack.counts,
            .point_pos_range = light_pack.point_pos_range,
            .point_color_int = light_pack.point_color_int,
            .spot_pos_range = light_pack.spot_pos_range,
            .spot_dir_inner = light_pack.spot_dir_inner,
            .spot_color_outer = light_pack.spot_color_outer,
            .spot_intensity = light_pack.spot_intensity,
            .spot_view_proj = light_pack.spot_view_proj,
            .spot_shadow_params = light_pack.spot_shadow_params,
        };

        const env = scene_draw.Environment{
            .pipelines = &self.forward,
            .stats = &self.stats,
            .default_material = &self.default_material,
            .default_white = &self.default_white_texture,
            .default_normal = &self.default_normal_texture,
            .default_cube = &self.default_cube_texture,
            .sky_texture = self.sky.texture,
            .ibl_intensity = self.sky.ibl_intensity,
            .shadow_pass = &self.shadows.pass,
            .shadow_uniforms = self.shadows.uniformState(self.lights.hemi.ground_color),
        };

        var current_pipeline_id: u32 = 0;

        // Opaque regular meshes first (front-to-back, early-Z). Transparent
        // items live in queues.transparent and are drawn below, after every
        // opaque mesh including instanced ones.
        for (self.queues.items.items) |item| {
            scene_draw.drawRegularItem(env, item, frame_ctx, &current_pipeline_id);
        }

        // Opaque instanced meshes.
        for (self.queues.opaque_instanced.items) |mesh| {
            scene_draw.drawInstancedMesh(env, mesh, frame_ctx, &current_pipeline_id);
        }

        // Transparent regular meshes (strict back-to-front, blended).
        for (self.queues.transparent.items) |item| {
            scene_draw.drawRegularItem(env, item, frame_ctx, &current_pipeline_id);
        }

        // Transparent instanced meshes last, drawn as-is (no per-instance
        // sorting; documented limitation).
        for (self.queues.transparent_instanced.items) |mesh| {
            scene_draw.drawInstancedMesh(env, mesh, frame_ctx, &current_pipeline_id);
        }

        // Inverse-hull outline for highlighted meshes: inside the main pass,
        // depth-tested, no depth write, drawn after all surface geometry.
        self.postfx.renderOutline(view_proj, eye, self.outline_meshes.items, &self.stats);

        // Physics debug lines (3D pass, depth-tested, no depth write).
        self.physics.renderDebug(self.allocator, view_proj, &self.stats);

        // Skybox Pass
        self.sky.render(camera, aspect, self.default_cube_texture, &self.stats);

        // Particle Pass
        self.particles.render(camera, aspect, &self.stats);

        if (!self.post_process.enabled) {
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
            .post = self.post_process,
            .ssao = self.ssao,
            .camera = camera,
            .aspect = aspect,
            .view_proj = view_proj,
            .eye = eye,
            .sun_dir = sun_dir,
            .sun_color = sun_color,
            .default_white_view = self.default_white_texture.view,
            .ui = if (self.ui_canvas) |*u| u else null,
            .stats = &self.stats,
        }, cur_w, cur_h);

        sg.commit();
    }

    pub fn deinit(self: *Scene) void {
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

        scene_content.deinitAnimations(self.allocator, &self.animation_groups, &self.skeletons);

        self.outline_meshes.deinit(self.allocator);
        self.postfx.deinit();

        self.particles.deinit(self.allocator);

        self.physics.deinit(self.allocator);

        if (self.ui_canvas) |*u| {
            u.deinit();
        }
    }
};
