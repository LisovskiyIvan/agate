const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const sapp = sokol.app;
const sglue = sokol.glue;
const shd = @import("shader");
const pbr_shd = @import("pbr_shader");
const skinned_pbr_shd = @import("skinned_pbr_shader");
const inst_shd = @import("instanced_shader");
const inst_pbr_shd = @import("instanced_pbr_shader");
const passes = @import("passes/mod.zig");
const postprocess = @import("postprocess.zig");
pub const PostProcessConfig = postprocess.PostProcessConfig;
pub const TonemappingType = postprocess.TonemappingType;
const ssao = @import("ssao.zig");
pub const SSAOConfig = ssao.SSAOConfig;
const particles = @import("particles.zig");
pub const ParticleSystem = particles.ParticleSystem;
pub const ParticleBlendMode = particles.ParticleBlendMode;
pub const Particle = particles.Particle;
const AnimationGroup = @import("animation/animation.zig").AnimationGroup;
const evaluateSkeleton = @import("animation/animation.zig").evaluateSkeleton;
const Skeleton = @import("animation/skeleton.zig").Skeleton;

const math = @import("math");
const Vec3 = math.Vec3;
const Mat4 = math.Mat4;
const Color3 = math.Color3;
const Color4 = math.Color4;
const BoundingBox = math.BoundingBox;
const Frustum = math.Frustum;
const Ray = math.Ray;
const RayHit = math.RayHit;

const physics = @import("physics.zig");
pub const PhysicsWorld = physics.PhysicsWorld;
pub const RigidBody = physics.RigidBody;
pub const PickingInfo = physics.PickingInfo;
pub const ColliderType = physics.ColliderType;

const ui = @import("ui.zig");
pub const UICanvas = ui.UICanvas;
pub const UIVertex = ui.UIVertex;

const Camera = @import("camera.zig").Camera;
const scene_render_queue = @import("scene/render_queue.zig");
const scene_pipelines = @import("scene/pipelines.zig");
const scene_cascades = @import("scene/cascades.zig");
const scene_uniforms = @import("scene/uniforms.zig");
const scene_projection = @import("scene/projection.zig");
const scene_draw = @import("scene/draw.zig");
const scene_light_selection = @import("scene/light_selection.zig");
const lights = @import("lights.zig");
const HemisphericLight = lights.HemisphericLight;
const DirectionalLight = lights.DirectionalLight;
const DirectionalLightOptions = lights.DirectionalLightOptions;
const PointLight = lights.PointLight;
const PointLightOptions = lights.PointLightOptions;
const SpotLight = lights.SpotLight;
const SpotLightOptions = lights.SpotLightOptions;
const Mesh = @import("mesh.zig").Mesh;
const Vertex = @import("mesh.zig").Vertex;
const InstancedMesh = @import("mesh.zig").InstancedMesh;
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
const Material = @import("material.zig").Material;
const Texture = @import("texture.zig").Texture;
const CubeTexture = @import("texture.zig").CubeTexture;
const SkyboxConfig = @import("texture.zig").SkyboxConfig;
const visibility = @import("visibility/mod.zig");

pub const SceneStats = struct {
    total_meshes: u32 = 0,
    rendered_meshes: u32 = 0,
    culled_meshes: u32 = 0,
    occluded_meshes: u32 = 0,
    occluders_count: u32 = 0,
    occluder_triangles: u32 = 0,
    draw_calls: u32 = 0,
    triangles: u32 = 0,
    pipeline_switches: u32 = 0,
};

pub const RenderMeshItem = scene_render_queue.RenderMeshItem;

pub const Scene = struct {
    allocator: std.mem.Allocator,
    meshes: std.ArrayListUnmanaged(*Mesh) = .empty,
    materials: std.ArrayListUnmanaged(*StandardMaterial) = .empty,
    pbr_materials: std.ArrayListUnmanaged(*PBRMaterial) = .empty,
    active_camera: ?Camera = null,
    clear_color: Color4 = Color4.new(0.12, 0.14, 0.18, 1.0),
    light: HemisphericLight = .{},
    // Optional scene sun. Null keeps the legacy hemispheric sun exactly.
    directional_light: ?*DirectionalLight = null,
    point_lights: std.ArrayListUnmanaged(*PointLight) = .empty,
    spot_lights: std.ArrayListUnmanaged(*SpotLight) = .empty,
    default_material: StandardMaterial = StandardMaterial.init("default"),
    default_white_texture: Texture,
    default_normal_texture: Texture,
    default_cube_texture: CubeTexture,

    enable_frustum_culling: bool = true,
    enable_occlusion_culling: bool = true,
    occlusion_culler: visibility.OcclusionCuller = visibility.OcclusionCuller.init(),
    stats: SceneStats = .{},
    // Bumped once per render(); Mesh.cached_* entries tagged with this are fresh.
    frame_id: u64 = 0,

    // Cached view-projection for projectPoint (physics overlay calls it ~4k/frame).
    project_vp_valid: bool = false,
    project_vp: Mat4 = Mat4.identity,
    project_cam: ?Camera = null,
    project_w: f32 = 0.0,
    project_h: f32 = 0.0,

    // Render Passes
    shadow_pass: passes.ShadowPass,
    skybox_pass: passes.SkyboxPass,
    particle_pass: passes.ParticlePass,
    postprocess_pass: passes.PostProcessPass,
    ssao_pass: passes.SSAOPass,
    bloom_pass: passes.BloomPass,
    outline_pass: passes.OutlinePass,
    // Lazily created on first use; renders physics debug wireframes in 3D.
    debug_pass: ?passes.DebugPass = null,
    show_physics_debug: bool = false,
    debug_lines: std.ArrayListUnmanaged(physics.DebugLine) = .empty,

    // Inverse-hull outline (highlight layer) settings.
    outline_meshes: std.ArrayListUnmanaged(*Mesh) = .empty,
    outline_enabled: bool = false,
    outline_color: Color4 = Color4.new(1.0, 0.5, 0.0, 1.0),
    outline_width_px: f32 = 2.0,

    // Shadow Mapping settings (Cascaded Shadow Maps with 16-sample Poisson PCF)
    enable_shadows: bool = true,
    shadow_bias: f32 = 0.0012,
    shadow_normal_bias: f32 = 0.02,
    shadow_intensity: f32 = 0.75,
    shadow_softness: f32 = 1.5,
    shadow_debug_cascades: bool = false,
    // PCSS (percentage-closer soft shadows) for the CSM sun. Disabled keeps
    // the legacy 16x Poisson PCF path bit-identical.
    pcss_enabled: bool = false,
    pcss_light_size: f32 = 0.02,
    pcss_blocker_radius: f32 = 0.01,
    cascade_splits: [4]f32 = .{ 10.0, 26.0, 65.0, 150.0 },
    cascade_matrices: [4]Mat4 = [_]Mat4{Mat4.identity} ** 4,

    // Skybox & IBL settings
    skybox_texture: ?CubeTexture = null,
    skybox_enabled: bool = false,
    skybox_exposure: f32 = 1.0,
    ibl_intensity: f32 = 1.0,

    // Mesh Forward Pipelines
    pipeline_u16: sg.Pipeline = .{},
    pipeline_u32: sg.Pipeline = .{},
    pipeline_pbr_u16: sg.Pipeline = .{},
    pipeline_pbr_u32: sg.Pipeline = .{},
    pipeline_skinned_pbr_u16: sg.Pipeline = .{},
    pipeline_skinned_pbr_u32: sg.Pipeline = .{},
    pipeline_instanced_u16: sg.Pipeline = .{},
    pipeline_instanced_u32: sg.Pipeline = .{},
    pipeline_instanced_pbr_u16: sg.Pipeline = .{},
    pipeline_instanced_pbr_u32: sg.Pipeline = .{},

    // Transparent twin pipelines: same shaders/layouts as above, but with
    // alpha blending (SRC_ALPHA, ONE_MINUS_SRC_ALPHA), depth test on and
    // depth write off. Selected per item when item.transparent is true.
    pipeline_blend_u16: sg.Pipeline = .{},
    pipeline_blend_u32: sg.Pipeline = .{},
    pipeline_pbr_blend_u16: sg.Pipeline = .{},
    pipeline_pbr_blend_u32: sg.Pipeline = .{},
    pipeline_skinned_pbr_blend_u16: sg.Pipeline = .{},
    pipeline_skinned_pbr_blend_u32: sg.Pipeline = .{},
    pipeline_instanced_blend_u16: sg.Pipeline = .{},
    pipeline_instanced_blend_u32: sg.Pipeline = .{},
    pipeline_instanced_pbr_blend_u16: sg.Pipeline = .{},
    pipeline_instanced_pbr_blend_u32: sg.Pipeline = .{},

    // Double-sided (cull-off) twins for every family; selected per item when
    // the material sets double_sided. Created in initPipelines, freed in deinit.
    ds_pipelines: scene_pipelines.DoubleSidedPipelines = .{},

    // Skeletal Animation Groups & Skeletons
    animation_groups: std.ArrayListUnmanaged(*AnimationGroup) = .empty,
    skeletons: std.ArrayListUnmanaged(*Skeleton) = .empty,

    // Post-Processing & SSAO settings
    post_process: PostProcessConfig = .{},
    ssao: SSAOConfig = .{},

    // Particle Systems
    particle_systems: std.ArrayListUnmanaged(*particles.ParticleSystem) = .empty,

    // Physics World
    physics_world: ?PhysicsWorld = null,

    // 2D & 3D UI Canvas
    ui_canvas: ?UICanvas = null,

    // Dynamic Decal Manager
    decal_manager: ?decal_mod.DecalManager = null,

    // Dynamic Trail Meshes
    trail_meshes: std.ArrayListUnmanaged(*trail_mod.TrailMesh) = .empty,

    // Navigation & AI
    nav_meshes: std.ArrayListUnmanaged(*ai_mod.NavMesh) = .empty,
    nav_agents: std.ArrayListUnmanaged(*ai_mod.NavAgent) = .empty,

    render_queue: std.ArrayListUnmanaged(RenderMeshItem) = .empty,
    // Transparent meshes (material alpha_mode == .blend), sorted strictly
    // back-to-front and drawn after every opaque mesh and instanced mesh.
    transparent_queue: std.ArrayListUnmanaged(RenderMeshItem) = .empty,
    opaque_instanced_queue: std.ArrayListUnmanaged(*Mesh) = .empty,
    transparent_instanced_queue: std.ArrayListUnmanaged(*Mesh) = .empty,
    instance_matrices: std.ArrayListUnmanaged(Mat4) = .empty,

    pub fn initInto(self: *Scene, allocator: std.mem.Allocator) void {
        self.* = Scene{
            .allocator = allocator,
            .default_white_texture = Texture.createWhite1x1(),
            .default_normal_texture = Texture.createFlatNormal1x1(),
            .default_cube_texture = CubeTexture.createDefault1x1(.{ 25, 30, 40, 255 }),
            .shadow_pass = passes.ShadowPass.init(allocator),
            .skybox_pass = passes.SkyboxPass.init(),
            .particle_pass = passes.ParticlePass.init(),
            .postprocess_pass = passes.PostProcessPass.init(),
            .ssao_pass = passes.SSAOPass.init(),
            .bloom_pass = passes.BloomPass.init(),
            .outline_pass = passes.OutlinePass.init(),
            .light = HemisphericLight.init("hemi", .{
                .direction = math.Vec3.new(0.5, 1.0, 0.3),
                .diffuse = Color3.white,
                .ground_color = Color3.new(0.2, 0.25, 0.3),
                .intensity = 1.0,
            }),
        };
        self.initPipelines();
    }

    pub fn init(allocator: std.mem.Allocator) Scene {
        var self: Scene = undefined;
        self.initInto(allocator);
        return self;
    }

    // One shader + one vertex layout feeds an opaque u16/u32 pair plus its
    // transparent blend twins. Only initPipelines uses this table.
    // (Moved to scene/pipelines.zig; aliased here so Scene internals and
    // tests keep the same names.)
    const PipelineFamily = scene_pipelines.PipelineFamily;

    // Fills the vertex layout for a family exactly as the legacy hand-written
    // descs did. Thin forwarder; real table lives in scene/pipelines.zig.
    fn pipelineLayoutFor(family: PipelineFamily, desc: *sg.PipelineDesc) void {
        return scene_pipelines.pipelineLayoutFor(family, desc);
    }

    // Creates an opaque u16/u32 pair plus transparent blend twins from one
    // base desc. Thin forwarder to scene/pipelines.zig.
    fn makePipelinePair(base: sg.PipelineDesc, opaque_u16: *sg.Pipeline, opaque_u32: *sg.Pipeline, blend_u16: *sg.Pipeline, blend_u32: *sg.Pipeline) void {
        return scene_pipelines.makePipelinePair(base, opaque_u16, opaque_u32, blend_u16, blend_u32);
    }

    fn initPipelines(self: *Scene) void {
        // One shader handle per family, shared by the opaque/blend pairs and
        // the double-sided twins.
        const family_shaders = scene_pipelines.DoubleSidedSourceShaders{
            .standard = sg.makeShader(shd.standardShaderDesc(sg.queryBackend())),
            .pbr = sg.makeShader(pbr_shd.pbrShaderDesc(sg.queryBackend())),
            .instanced = sg.makeShader(inst_shd.instancedShaderDesc(sg.queryBackend())),
            .instanced_pbr = sg.makeShader(inst_pbr_shd.instancedPbrShaderDesc(sg.queryBackend())),
            .skinned_pbr = sg.makeShader(skinned_pbr_shd.skinnedPbrShaderDesc(sg.queryBackend())),
        };
        const specs = [_]struct {
            shader: sg.Shader,
            family: PipelineFamily,
            opaque_u16: *sg.Pipeline,
            opaque_u32: *sg.Pipeline,
            blend_u16: *sg.Pipeline,
            blend_u32: *sg.Pipeline,
        }{
            .{
                .shader = family_shaders.standard,
                .family = .standard,
                .opaque_u16 = &self.pipeline_u16,
                .opaque_u32 = &self.pipeline_u32,
                .blend_u16 = &self.pipeline_blend_u16,
                .blend_u32 = &self.pipeline_blend_u32,
            },
            .{
                .shader = family_shaders.pbr,
                .family = .pbr,
                .opaque_u16 = &self.pipeline_pbr_u16,
                .opaque_u32 = &self.pipeline_pbr_u32,
                .blend_u16 = &self.pipeline_pbr_blend_u16,
                .blend_u32 = &self.pipeline_pbr_blend_u32,
            },
            .{
                .shader = family_shaders.instanced,
                .family = .instanced,
                .opaque_u16 = &self.pipeline_instanced_u16,
                .opaque_u32 = &self.pipeline_instanced_u32,
                .blend_u16 = &self.pipeline_instanced_blend_u16,
                .blend_u32 = &self.pipeline_instanced_blend_u32,
            },
            .{
                .shader = family_shaders.instanced_pbr,
                .family = .instanced_pbr,
                .opaque_u16 = &self.pipeline_instanced_pbr_u16,
                .opaque_u32 = &self.pipeline_instanced_pbr_u32,
                .blend_u16 = &self.pipeline_instanced_pbr_blend_u16,
                .blend_u32 = &self.pipeline_instanced_pbr_blend_u32,
            },
            .{
                .shader = family_shaders.skinned_pbr,
                .family = .skinned_pbr,
                .opaque_u16 = &self.pipeline_skinned_pbr_u16,
                .opaque_u32 = &self.pipeline_skinned_pbr_u32,
                .blend_u16 = &self.pipeline_skinned_pbr_blend_u16,
                .blend_u32 = &self.pipeline_skinned_pbr_blend_u32,
            },
        };

        for (specs) |spec| {
            var desc = sg.PipelineDesc{
                .shader = spec.shader,
                .index_type = .UINT16,
                .depth = .{
                    .compare = .LESS_EQUAL,
                    .write_enabled = true,
                },
                .cull_mode = .BACK,
                .face_winding = .CCW,
            };
            pipelineLayoutFor(spec.family, &desc);
            makePipelinePair(desc, spec.opaque_u16, spec.opaque_u32, spec.blend_u16, spec.blend_u32);
        }

        self.ds_pipelines.initFromShaders(family_shaders);

        inline for (.{
            .{ .pipe = self.pipeline_u16, .msg = "pipeline_u16 failed to create!" },
            .{ .pipe = self.pipeline_u32, .msg = "pipeline_u32 failed to create!" },
            .{ .pipe = self.pipeline_pbr_u16, .msg = "pipeline_pbr_u16 failed to create!" },
            .{ .pipe = self.pipeline_pbr_u32, .msg = "pipeline_pbr_u32 failed to create!" },
            .{ .pipe = self.pipeline_skinned_pbr_u16, .msg = "pipeline_skinned_pbr_u16 failed to create!" },
            .{ .pipe = self.pipeline_skinned_pbr_u32, .msg = "pipeline_skinned_pbr_u32 failed to create!" },
            .{ .pipe = self.pipeline_instanced_u16, .msg = "pipeline_instanced_u16 failed to create!" },
            .{ .pipe = self.pipeline_instanced_u32, .msg = "pipeline_instanced_u32 failed to create!" },
            .{ .pipe = self.pipeline_instanced_pbr_u16, .msg = "pipeline_instanced_pbr_u16 failed to create!" },
            .{ .pipe = self.pipeline_instanced_pbr_u32, .msg = "pipeline_instanced_pbr_u32 failed to create!" },
            .{ .pipe = self.pipeline_blend_u16, .msg = "pipeline_blend_u16 failed to create!" },
            .{ .pipe = self.pipeline_blend_u32, .msg = "pipeline_blend_u32 failed to create!" },
            .{ .pipe = self.pipeline_pbr_blend_u16, .msg = "pipeline_pbr_blend_u16 failed to create!" },
            .{ .pipe = self.pipeline_pbr_blend_u32, .msg = "pipeline_pbr_blend_u32 failed to create!" },
            .{ .pipe = self.pipeline_skinned_pbr_blend_u16, .msg = "pipeline_skinned_pbr_blend_u16 failed to create!" },
            .{ .pipe = self.pipeline_skinned_pbr_blend_u32, .msg = "pipeline_skinned_pbr_blend_u32 failed to create!" },
            .{ .pipe = self.pipeline_instanced_blend_u16, .msg = "pipeline_instanced_blend_u16 failed to create!" },
            .{ .pipe = self.pipeline_instanced_blend_u32, .msg = "pipeline_instanced_blend_u32 failed to create!" },
            .{ .pipe = self.pipeline_instanced_pbr_blend_u16, .msg = "pipeline_instanced_pbr_blend_u16 failed to create!" },
            .{ .pipe = self.pipeline_instanced_pbr_blend_u32, .msg = "pipeline_instanced_pbr_blend_u32 failed to create!" },
        }) |entry| {
            if (entry.pipe.id == 0) @panic(entry.msg);
        }
    }

    pub fn resizeOffscreen(self: *Scene, width: i32, height: i32) void {
        self.postprocess_pass.resize(width, height);
        self.ssao_pass.resize(width, height);
        self.bloom_pass.resize(width, height);
        passes.OutlinePass.resize(width, height);
    }

    pub fn setPostProcess(self: *Scene, config: PostProcessConfig) void {
        self.post_process = config;
    }

    pub fn setSSAO(self: *Scene, config: SSAOConfig) void {
        self.ssao = config;
    }

    pub fn setSkybox(self: *Scene, cube: CubeTexture) void {
        self.skybox_texture = cube;
        self.skybox_enabled = true;
    }

    pub fn createDefaultSkybox(self: *Scene, config: SkyboxConfig) !void {
        const cube = try CubeTexture.createProceduralSkybox(self.allocator, config);
        self.setSkybox(cube);
    }

    pub fn createHemisphericLight(self: *Scene, name: []const u8, options: lights.HemisphericLightOptions) HemisphericLight {
        self.light = HemisphericLight.init(name, options);
        return self.light;
    }

    pub fn createPointLight(self: *Scene, name: []const u8, options: PointLightOptions) !*PointLight {
        const pl = try self.allocator.create(PointLight);
        pl.* = PointLight.init(name, options);
        try self.point_lights.append(self.allocator, pl);
        return pl;
    }

    pub fn createSpotLight(self: *Scene, name: []const u8, options: SpotLightOptions) !*SpotLight {
        const sl = try self.allocator.create(SpotLight);
        sl.* = SpotLight.init(name, options);
        try self.spot_lights.append(self.allocator, sl);
        return sl;
    }

    // Creates (or replaces) the single scene sun. Replacing destroys the
    // previous light so at most one directional light is owned at a time.
    pub fn createDirectionalLight(self: *Scene, name: []const u8, options: DirectionalLightOptions) !*DirectionalLight {
        if (self.directional_light) |old| {
            if (old.owns_name) self.allocator.free(old.name);
            self.allocator.destroy(old);
            self.directional_light = null;
        }
        const dl = try self.allocator.create(DirectionalLight);
        dl.* = DirectionalLight.init(name, options);
        self.directional_light = dl;
        return dl;
    }

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

    pub fn getOrCreateDecalManager(self: *Scene, max_decals: usize) *decal_mod.DecalManager {
        if (self.decal_manager == null) {
            self.decal_manager = decal_mod.DecalManager.init(self, max_decals);
        }
        return &self.decal_manager.?;
    }

    pub fn updateDecals(self: *Scene, dt: f32) void {
        if (self.decal_manager) |*dm| {
            dm.update(dt);
        }
    }

    pub fn createParticleSystem(self: *Scene, name: []const u8, capacity: usize) !*particles.ParticleSystem {
        const ps = try particles.ParticleSystem.init(self.allocator, name, capacity);
        try self.particle_systems.append(self.allocator, ps);
        return ps;
    }

    pub fn updateParticles(self: *Scene, dt: f32) void {
        for (self.particle_systems.items) |ps| {
            ps.update(dt);
        }
    }

    pub fn createTrailMesh(self: *Scene, name: []const u8, options: trail_mod.TrailOptions) !*trail_mod.TrailMesh {
        const tm = try trail_mod.TrailMesh.init(self, name, options);
        try self.trail_meshes.append(self.allocator, tm);
        return tm;
    }

    pub fn updateTrails(self: *Scene, dt: f32) void {
        const cam_pos = if (self.active_camera) |cam| cam.getPosition() else Vec3.zero;
        for (self.trail_meshes.items) |tm| {
            tm.update(dt, cam_pos);
        }
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
        const ptr = try self.allocator.create(ai_mod.NavMesh);
        errdefer self.allocator.destroy(ptr);
        ptr.* = try ai_mod.NavMesh.buildFromTriangles(self.allocator, positions, indices, max_slope_rad);
        try self.nav_meshes.append(self.allocator, ptr);
        return ptr;
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
        const ptr = try self.allocator.create(ai_mod.NavMesh);
        errdefer self.allocator.destroy(ptr);
        ptr.* = try ai_mod.NavMesh.buildGrid(self.allocator, min_x, max_x, min_z, max_z, elevation_y, subdiv_x, subdiv_z, obstacles);
        try self.nav_meshes.append(self.allocator, ptr);
        return ptr;
    }

    pub fn createNavAgent(self: *Scene, nav_mesh: *const ai_mod.NavMesh, start_pos: Vec3) !*ai_mod.NavAgent {
        const ptr = try self.allocator.create(ai_mod.NavAgent);
        errdefer self.allocator.destroy(ptr);
        ptr.* = ai_mod.NavAgent.init(self.allocator, nav_mesh, start_pos);
        try self.nav_agents.append(self.allocator, ptr);
        return ptr;
    }

    pub fn updateNavAgents(self: *Scene, dt: f32) void {
        for (self.nav_agents.items) |ag| {
            ag.update(dt);
        }
    }

    pub fn updateAnimations(self: *Scene, dt: f32) void {
        for (self.animation_groups.items) |ag| {
            ag.update(dt);
        }

        for (self.skeletons.items) |skel| {
            var active_base: [16]*AnimationGroup = undefined;
            var base_count: usize = 0;
            var active_add: [16]*AnimationGroup = undefined;
            var add_count: usize = 0;

            for (self.animation_groups.items) |ag| {
                if (ag.skeleton == skel and ag.is_playing and ag.weight > 0.0001) {
                    if (ag.is_additive) {
                        if (add_count < active_add.len) {
                            active_add[add_count] = ag;
                            add_count += 1;
                        }
                    } else {
                        if (base_count < active_base.len) {
                            active_base[base_count] = ag;
                            base_count += 1;
                        }
                    }
                }
            }

            evaluateSkeleton(skel, active_base[0..base_count], active_add[0..add_count]);
        }

        // CPU morph targets: weights tracks only flag meshes dirty above;
        // blend base + deltas here once per frame (applyMorphs early-outs
        // when clean), before the render queue is built in render().
        for (self.meshes.items) |mesh| {
            if (mesh.hasMorphTargets()) mesh.applyMorphs();
        }
    }

    pub fn enablePhysics(self: *Scene, gravity: ?Vec3) *PhysicsWorld {
        if (self.physics_world == null) {
            self.physics_world = PhysicsWorld.init(self.allocator);
        }
        if (gravity) |g| {
            self.physics_world.?.gravity = g;
        }
        return &self.physics_world.?;
    }

    pub fn getPhysicsWorld(self: *Scene) ?*PhysicsWorld {
        if (self.physics_world) |*pw| return pw;
        return null;
    }

    pub fn getRigidBody(self: *Scene, mesh: *const Mesh) ?*RigidBody {
        if (self.physics_world) |*pw| {
            return pw.findBody(mesh);
        }
        return null;
    }

    pub fn createRigidBody(self: *Scene, mesh: *Mesh, collider: ColliderType, mass: f32) !*RigidBody {
        const pw = if (self.physics_world) |*p| p else self.enablePhysics(null);
        return pw.createBody(mesh, collider, mass);
    }

    /// Creates a rigid body with collision filter / sensor / event options.
    pub fn createRigidBodyWith(self: *Scene, mesh: *Mesh, collider: ColliderType, mass: f32, options: physics.BodyOptions) !*RigidBody {
        const pw = if (self.physics_world) |*p| p else self.enablePhysics(null);
        return pw.createBodyWith(mesh, collider, mass, options);
    }

    pub fn updatePhysics(self: *Scene, dt: f32) void {
        if (self.physics_world) |*pw| {
            pw.step(dt);
        }
    }

    pub fn updateCamera(self: *Scene, dt: f32) void {
        if (self.active_camera) |*cam| {
            cam.update(dt);
        }
    }

    pub fn createPickingRay(self: *Scene, screen_x: f32, screen_y: f32) Ray {
        const cam = self.active_camera orelse return Ray.new(Vec3.zero, Vec3.forward);
        const w = sapp.widthf();
        const h = sapp.heightf();
        if (w <= 0.0 or h <= 0.0) return Ray.new(cam.getPosition(), Vec3.forward);

        const aspect = w / h;
        const vp = cam.getViewProjection(aspect);
        const inv_vp = vp.invert() orelse return Ray.new(cam.getPosition(), Vec3.forward);

        const ndc_x = (2.0 * screen_x) / w - 1.0;
        const ndc_y = 1.0 - (2.0 * screen_y) / h;

        const near_pt = inv_vp.transformPoint(Vec3.new(ndc_x, ndc_y, 0.0));
        const far_pt = inv_vp.transformPoint(Vec3.new(ndc_x, ndc_y, 1.0));
        const dir = far_pt.sub(near_pt).normalize();

        return Ray.new(near_pt, dir);
    }

    pub fn pickWithRay(self: *Scene, r: Ray) PickingInfo {
        var closest_dist: f32 = std.math.inf(f32);
        var best_hit: ?RayHit = null;
        var best_mesh: ?*Mesh = null;

        for (self.meshes.items) |mesh| {
            if (!mesh.is_visible or mesh.is_lod_child or mesh.is_decal) continue;

            if (mesh.instances.items.len > 0) {
                continue;
            }

            const model = mesh.getWorldMatrix();
            const world_aabb = mesh.local_bounding_box.transform(model);

            if (self.getRigidBody(mesh)) |body| {
                if (body.collider == .sphere) {
                    const radius = body.sphere_radius * mesh.scaling.x;
                    if (r.intersectsSphereNormal(mesh.position, radius)) |hit| {
                        if (hit.distance < closest_dist) {
                            closest_dist = hit.distance;
                            best_hit = hit;
                            best_mesh = mesh;
                        }
                    }
                    continue;
                }
            }

            if (r.intersectsAABBNormal(world_aabb)) |hit| {
                if (hit.distance < closest_dist) {
                    closest_dist = hit.distance;
                    best_hit = hit;
                    best_mesh = mesh;
                }
            }
        }

        if (best_hit) |hit| {
            return PickingInfo{
                .hit = true,
                .distance = hit.distance,
                .picked_point = hit.point,
                .picked_normal = hit.normal,
                .picked_mesh = best_mesh,
            };
        }

        return PickingInfo{};
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

    // Pure view/projection snapshot compare. Thin forwarder; the real
    // compare lives in scene/projection.zig.
    fn camerasEqualForProjection(a: Camera, b: Camera) bool {
        return scene_projection.camerasEqualForProjection(a, b);
    }

    // Cached VP reuse for projectPoint; bit-identical to getViewProjection.
    fn cachedProjectViewProjection(self: *Scene, cam: Camera, w: f32, h: f32) Mat4 {
        if (self.project_vp_valid) {
            if (self.project_cam) |pc| {
                if (w == self.project_w and h == self.project_h and camerasEqualForProjection(pc, cam)) {
                    return self.project_vp;
                }
            }
        }
        const vp = cam.getViewProjection(w / h);
        self.project_vp = vp;
        self.project_cam = cam;
        self.project_w = w;
        self.project_h = h;
        self.project_vp_valid = true;
        return vp;
    }

    pub fn projectPoint(self: *Scene, world_pos: Vec3) ?math.Vec2 {
        const cam = self.active_camera orelse return null;
        const w = sapp.widthf();
        const h = sapp.heightf();
        if (w <= 0.0 or h <= 0.0) return null;
        const vp = self.cachedProjectViewProjection(cam, w, h);
        return vp.projectPoint(world_pos, w, h);
    }

    pub fn handleEvent(self: *Scene, ev: [*c]const sapp.Event) void {
        if (self.active_camera) |*cam| {
            cam.handleEvent(ev);
        }
    }

    fn sortRenderItems(_: void, a: RenderMeshItem, b: RenderMeshItem) bool {
        return scene_render_queue.sortRenderItems({}, a, b);
    }

    // Transparent items sort strictly back-to-front (by squared camera
    // distance) so alpha blending composites in the correct order.
    fn sortTransparentBackToFront(_: void, a: RenderMeshItem, b: RenderMeshItem) bool {
        return scene_render_queue.sortTransparentBackToFront({}, a, b);
    }

    // A mesh is transparent when its material opts into .blend alpha mode.
    // Meshes without a material render opaque (legacy behavior).
    pub fn materialIsTransparent(mat: ?Material) bool {
        return scene_render_queue.materialIsTransparent(mat);
    }

    // Derives a transparent twin pipeline desc from an opaque base desc:
    // same shader/layout/depth test, but SRC_ALPHA/ONE_MINUS_SRC_ALPHA
    // blending with depth write disabled. Pure function (no GPU calls).
    fn blendDescFor(base: sg.PipelineDesc) sg.PipelineDesc {
        return scene_render_queue.blendDescFor(base);
    }

    // Selects the forward pipeline for a regular (non-instanced) queue item.
    // Transparent items resolve to the blend twins; opaque selection is
    // identical to the legacy logic, so existing pipeline ids are untouched.
    fn pipelineForRegularItem(self: *Scene, item: RenderMeshItem) u32 {
        return scene_pipelines.pipelineForRegularItem(self, item);
    }
    /// World matrix computed at most once per render() call.
    /// Parent chains resolve through the same cache, so hierarchies stay O(depth) total.
    fn worldMatrixCached(self: *Scene, mesh: *Mesh) Mat4 {
        if (mesh.cached_frame == self.frame_id) return mesh.cached_matrix;
        const trs = Mat4.fromRotationTranslationScale(mesh.position, mesh.rotation, mesh.scaling);
        const local = Mat4.mul(trs, mesh.base_matrix);
        const world = if (mesh.parent) |p| Mat4.mul(self.worldMatrixCached(p), local) else local;
        mesh.cached_matrix = world;
        mesh.cached_aabb = mesh.local_bounding_box.transform(world);
        mesh.cached_frame = self.frame_id;
        return world;
    }

    fn worldAABBCached(self: *Scene, mesh: *Mesh) BoundingBox {
        if (mesh.cached_frame == self.frame_id) return mesh.cached_aabb;
        _ = self.worldMatrixCached(mesh);
        return mesh.cached_aabb;
    }

    pub fn computeCascades(self: *Scene, camera: Camera, aspect: f32) [4]Mat4 {
        const sun_dir = lights.resolveSunDirection(self.directional_light, self.light);
        return self.computeCascadesWithSun(camera, aspect, sun_dir);
    }

    // Hoisted-sun variant: norm_light_dir must be resolveSunDirection output.
    // Thin wrapper; the solver lives in scene/cascades.zig.
    fn computeCascadesWithSun(self: *Scene, camera: Camera, aspect: f32, norm_light_dir: Vec3) [4]Mat4 {
        return scene_cascades.computeCascades(camera, aspect, norm_light_dir, self.cascade_splits);
    }

    // Per-frame constants shared by the regular and instanced draw helpers.
    // (Moved to scene/uniforms.zig; aliased here so draw paths and tests
    // keep the `Scene.FrameContext` name.)
    const FrameContext = scene_uniforms.FrameContext;

    // Fragment uniforms shared by the standard, PBR and instanced shaders:
    // identical names, types and values in all three FsParams structs.
    // Material-specific factors (diffuse_color / base_color_factor /
    // pbr_factors / emissive_factor) stay with the caller.
    const FrameUniforms = scene_uniforms.FrameUniforms;

    // Packs the shared fragment uniforms once per draw. Values are
    // bit-identical to the legacy per-shader literals.
    // Thin wrapper over scene/uniforms.zig: Scene state is packed into an
    // explicit ShadowState so the core builder stays pure (no Scene import).
    // pub: called from scene/draw.zig through a generic `scene` parameter.
    pub fn frameUniforms(self: *Scene, mesh: *const Mesh, ctx: FrameContext) FrameUniforms {
        return scene_uniforms.buildFrameUniforms(.{
            .ground_color = self.light.ground_color,
            .enable_shadows = self.enable_shadows,
            .mesh_receive_shadows = mesh.receive_shadows,
            .bias = self.shadow_bias,
            .intensity = self.shadow_intensity,
            .normal_bias = self.shadow_normal_bias,
            .softness = self.shadow_softness,
            .debug_cascades = self.shadow_debug_cascades,
            .pcss_enabled = self.pcss_enabled,
            .pcss_light_size = self.pcss_light_size,
            .pcss_blocker_radius = self.pcss_blocker_radius,
            .splits = self.cascade_splits,
        }, ctx);
    }

    // Draws one regular (non-instanced) queue item: pipeline select, bind,
    // uniforms, draw. Shared by the opaque pass and the transparent pass
    // (pipeline choice follows item.transparent). Updates stats.
    // Thin forwarder; the real path lives in scene/draw.zig.
    fn drawRegularItem(self: *Scene, item: RenderMeshItem, ctx: FrameContext, current_pipeline_id: *u32) void {
        return scene_draw.drawRegularItem(self, item, ctx, current_pipeline_id);
    }

    // Draws one instanced mesh with the currently visible instance buffer.
    // Transparent instanced meshes use the blend twin pipeline and are drawn
    // as-is (no per-instance back-to-front sort); the caller draws them
    // after all opaque geometry. Updates stats.
    // Thin forwarder; the real path lives in scene/draw.zig.
    fn drawInstancedMesh(self: *Scene, mesh: *Mesh, ctx: FrameContext, current_pipeline_id: *u32) void {
        return scene_draw.drawInstancedMesh(self, mesh, ctx, current_pipeline_id);
    }

    pub fn render(self: *Scene) void {
        const camera = self.active_camera orelse return;
        const aspect = sapp.widthf() / sapp.heightf();
        // Sun resolved once per frame; reused by cascades, mesh uniforms, postprocess.
        const sun_dir = lights.resolveSunDirection(self.directional_light, self.light);
        const sun_color = lights.resolveSunColor(self.directional_light, self.light);
        const sun_intensity = lights.resolveSunIntensity(self.directional_light, self.light);
        const view_proj = camera.getViewProjection(aspect);
        const frustum = Frustum.fromViewProjection(view_proj);
        const eye = camera.getPosition();

        self.stats = .{};
        self.frame_id +%= 1;
        self.render_queue.clearRetainingCapacity();
        self.transparent_queue.clearRetainingCapacity();
        self.opaque_instanced_queue.clearRetainingCapacity();
        self.transparent_instanced_queue.clearRetainingCapacity();

        // 1. Directional Light Cascaded Shadow View-Projections
        const cascades = self.computeCascadesWithSun(camera, aspect, sun_dir);
        self.cascade_matrices = cascades;

        // Phase -1: Occlusion Culling setup & occluder rasterization
        if (self.enable_occlusion_culling) {
            self.occlusion_culler.beginFrame(view_proj);
            for (self.meshes.items) |m| {
                if (!m.is_lod_child and m.is_visible and m.is_occluder) {
                    const m_world = self.worldMatrixCached(m);
                    self.occlusion_culler.rasterizeOccluderMesh(
                        m.cpu_positions,
                        m.cpu_indices,
                        m.local_bounding_box,
                        m_world,
                    );
                }
            }
            self.occlusion_culler.endOccluders();
            self.stats.occluders_count = self.occlusion_culler.occluder_count;
            self.stats.occluder_triangles = self.occlusion_culler.triangles_rasterized;
        }

        // Phase 0: Pre-filter meshes and populate instance buffers using SIMD 4-wide batching
        for (self.meshes.items) |mesh| {
            if (mesh.is_lod_child) continue;
            if (mesh.instances.items.len > 0) {
                self.instance_matrices.clearRetainingCapacity();
                const total_insts = mesh.instances.items.len;
                var inst_idx: usize = 0;

                // SIMD 4-wide batching
                while (inst_idx + 4 <= total_insts) : (inst_idx += 4) {
                    const inst0 = mesh.instances.items[inst_idx + 0];
                    const inst1 = mesh.instances.items[inst_idx + 1];
                    const inst2 = mesh.instances.items[inst_idx + 2];
                    const inst3 = mesh.instances.items[inst_idx + 3];

                    self.stats.total_meshes += 4;

                    inst0.updateCachedTransforms();
                    inst1.updateCachedTransforms();
                    inst2.updateCachedTransforms();
                    inst3.updateCachedTransforms();

                    const m0 = inst0.cached_world_matrix;
                    const m1 = inst1.cached_world_matrix;
                    const m2 = inst2.cached_world_matrix;
                    const m3 = inst3.cached_world_matrix;

                    const b0 = inst0.cached_bounding_box;
                    const b1 = inst1.cached_bounding_box;
                    const b2 = inst2.cached_bounding_box;
                    const b3 = inst3.cached_bounding_box;

                    if (self.enable_frustum_culling) {
                        const c0 = b0.center();
                        const c1 = b1.center();
                        const c2 = b2.center();
                        const c3 = b3.center();

                        const e0 = b0.extents();
                        const e1 = b1.extents();
                        const e2 = b2.extents();
                        const e3 = b3.extents();

                        const c_x: @Vector(4, f32) = .{ c0.x, c1.x, c2.x, c3.x };
                        const c_y: @Vector(4, f32) = .{ c0.y, c1.y, c2.y, c3.y };
                        const c_z: @Vector(4, f32) = .{ c0.z, c1.z, c2.z, c3.z };

                        const ex_x: @Vector(4, f32) = .{ e0.x, e1.x, e2.x, e3.x };
                        const ex_y: @Vector(4, f32) = .{ e0.y, e1.y, e2.y, e3.y };
                        const ex_z: @Vector(4, f32) = .{ e0.z, e1.z, e2.z, e3.z };

                        const vis = frustum.intersectsAABB4(c_x, c_y, c_z, ex_x, ex_y, ex_z);

                        if (inst0.is_visible and (inst0.culling_strategy == .always_render or vis[0])) {
                            if (self.enable_occlusion_culling and !mesh.is_occluder and inst0.culling_strategy != .always_render and self.occlusion_culler.isOccluded(b0)) {
                                self.stats.occluded_meshes += 1;
                                self.stats.culled_meshes += 1;
                            } else {
                                self.stats.rendered_meshes += 1;
                                self.instance_matrices.append(self.allocator, m0) catch {};
                            }
                        } else {
                            self.stats.culled_meshes += 1;
                        }

                        if (inst1.is_visible and (inst1.culling_strategy == .always_render or vis[1])) {
                            if (self.enable_occlusion_culling and !mesh.is_occluder and inst1.culling_strategy != .always_render and self.occlusion_culler.isOccluded(b1)) {
                                self.stats.occluded_meshes += 1;
                                self.stats.culled_meshes += 1;
                            } else {
                                self.stats.rendered_meshes += 1;
                                self.instance_matrices.append(self.allocator, m1) catch {};
                            }
                        } else {
                            self.stats.culled_meshes += 1;
                        }

                        if (inst2.is_visible and (inst2.culling_strategy == .always_render or vis[2])) {
                            if (self.enable_occlusion_culling and !mesh.is_occluder and inst2.culling_strategy != .always_render and self.occlusion_culler.isOccluded(b2)) {
                                self.stats.occluded_meshes += 1;
                                self.stats.culled_meshes += 1;
                            } else {
                                self.stats.rendered_meshes += 1;
                                self.instance_matrices.append(self.allocator, m2) catch {};
                            }
                        } else {
                            self.stats.culled_meshes += 1;
                        }

                        if (inst3.is_visible and (inst3.culling_strategy == .always_render or vis[3])) {
                            if (self.enable_occlusion_culling and !mesh.is_occluder and inst3.culling_strategy != .always_render and self.occlusion_culler.isOccluded(b3)) {
                                self.stats.occluded_meshes += 1;
                                self.stats.culled_meshes += 1;
                            } else {
                                self.stats.rendered_meshes += 1;
                                self.instance_matrices.append(self.allocator, m3) catch {};
                            }
                        } else {
                            self.stats.culled_meshes += 1;
                        }
                    } else {
                        if (inst0.is_visible) {
                            self.stats.rendered_meshes += 1;
                            self.instance_matrices.append(self.allocator, m0) catch {};
                        }
                        if (inst1.is_visible) {
                            self.stats.rendered_meshes += 1;
                            self.instance_matrices.append(self.allocator, m1) catch {};
                        }
                        if (inst2.is_visible) {
                            self.stats.rendered_meshes += 1;
                            self.instance_matrices.append(self.allocator, m2) catch {};
                        }
                        if (inst3.is_visible) {
                            self.stats.rendered_meshes += 1;
                            self.instance_matrices.append(self.allocator, m3) catch {};
                        }
                    }
                }

                // Remainder instances
                while (inst_idx < total_insts) : (inst_idx += 1) {
                    const inst = mesh.instances.items[inst_idx];
                    self.stats.total_meshes += 1;
                    if (!inst.is_visible) continue;
                    inst.updateCachedTransforms();
                    const m = inst.cached_world_matrix;
                    const world_aabb = inst.cached_bounding_box;
                    if (self.enable_frustum_culling and inst.culling_strategy != .always_render) {
                        if (!frustum.intersectsAABB(world_aabb)) {
                            self.stats.culled_meshes += 1;
                            continue;
                        }
                    }
                    if (self.enable_occlusion_culling and !mesh.is_occluder and inst.culling_strategy != .always_render) {
                        if (self.occlusion_culler.isOccluded(world_aabb)) {
                            self.stats.occluded_meshes += 1;
                            self.stats.culled_meshes += 1;
                            continue;
                        }
                    }
                    self.stats.rendered_meshes += 1;
                    self.instance_matrices.append(self.allocator, m) catch continue;
                }

                const visible_count = self.instance_matrices.items.len;
                mesh.visible_instance_count = @intCast(visible_count);
                if (visible_count > 0) {
                    if (materialIsTransparent(mesh.material)) {
                        self.transparent_instanced_queue.append(self.allocator, mesh) catch {};
                    } else {
                        self.opaque_instanced_queue.append(self.allocator, mesh) catch {};
                    }

                    if (mesh.instance_buffer.id == 0 or mesh.instance_buffer_capacity < visible_count) {
                        if (mesh.instance_buffer.id != 0) {
                            sg.destroyBuffer(mesh.instance_buffer);
                        }
                        const new_cap = @max(visible_count, mesh.instance_buffer_capacity * 2);
                        mesh.instance_buffer = sg.makeBuffer(.{
                            .usage = .{ .vertex_buffer = true, .dynamic_update = true },
                            .size = new_cap * @sizeOf(Mat4),
                        });
                        mesh.instance_buffer_capacity = new_cap;
                        // Fresh buffer: must upload, then record hash.
                        sg.updateBuffer(mesh.instance_buffer, sg.asRange(self.instance_matrices.items[0..visible_count]));
                        mesh.instance_hash = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(self.instance_matrices.items[0..visible_count]));
                        mesh.instance_uploaded_count = visible_count;
                    } else {
                        // Static instance sets skip the driver upload entirely.
                        const h = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(self.instance_matrices.items[0..visible_count]));
                        if (visible_count != mesh.instance_uploaded_count or h != mesh.instance_hash) {
                            sg.updateBuffer(mesh.instance_buffer, sg.asRange(self.instance_matrices.items[0..visible_count]));
                            mesh.instance_hash = h;
                            mesh.instance_uploaded_count = visible_count;
                        }
                    }
                }
            } else {
                self.stats.total_meshes += 1;
                if (!mesh.is_visible) continue;

                var render_mesh = mesh;
                if (mesh.lod_levels.items.len > 0) {
                    const dist = if (mesh.cached_aabb.isValid()) mesh.cached_aabb.center().distance(eye) else mesh.position.distance(eye);
                    const active_lod = mesh.getLOD(dist);
                    if (active_lod) |lod| {
                        render_mesh = lod;
                        if (lod != mesh) {
                            lod.position = mesh.position;
                            lod.rotation = mesh.rotation;
                            lod.scaling = mesh.scaling;
                            lod.parent = mesh.parent;
                            lod.base_matrix = mesh.base_matrix;
                            if (lod.material == null) {
                                lod.material = mesh.material;
                            }
                        }
                    } else {
                        // Beyond max distance, culled
                        self.stats.culled_meshes += 1;
                        continue;
                    }
                }

                const model = self.worldMatrixCached(render_mesh);
                const world_aabb = render_mesh.cached_aabb;

                if (self.enable_frustum_culling and render_mesh.culling_strategy != .always_render) {
                    if (!frustum.intersectsAABB(world_aabb)) {
                        self.stats.culled_meshes += 1;
                        continue;
                    }
                }

                if (self.enable_occlusion_culling and !render_mesh.is_occluder and render_mesh.culling_strategy != .always_render) {
                    if (self.occlusion_culler.isOccluded(world_aabb)) {
                        self.stats.occluded_meshes += 1;
                        self.stats.culled_meshes += 1;
                        continue;
                    }
                }

                self.stats.rendered_meshes += 1;

                const is_pbr = if (render_mesh.material) |m| (m == .pbr) else false;
                const tex_id: u32 = if (render_mesh.material) |m| switch (m) {
                    .pbr => |p| if (p.albedo_texture) |t| t.view.id else self.default_white_texture.view.id,
                    .standard => |s| if (s.diffuse_texture) |t| t.view.id else self.default_white_texture.view.id,
                } else self.default_white_texture.view.id;

                const d_sq = world_aabb.center().sub(eye).lengthSq();
                const transparent = materialIsTransparent(render_mesh.material) or render_mesh.is_decal;
                const target_queue = if (transparent) &self.transparent_queue else &self.render_queue;
                target_queue.append(self.allocator, .{
                    .mesh = render_mesh,
                    .model = model,
                    .distance_sq = d_sq,
                    .is_pbr = is_pbr,
                    .texture_id = tex_id,
                    .transparent = transparent,
                    .double_sided = scene_render_queue.materialIsDoubleSided(render_mesh.material) or render_mesh.is_decal,
                    .is_decal = render_mesh.is_decal,
                }) catch continue;
            }
        }

        // Pack Point & Spot Lights. Directions are normalized once here so the
        // fragment shaders can use them raw (no per-pixel normalize()).
        // The directional light overrides the legacy hemispheric sun here;
        // ambient stays hemispheric ground_color below.
        var light_counts = [4]f32{ 0.0, 0.0, 0.0, 0.0 };
        var point_pos_range = [_][4]f32{.{ 0.0, 0.0, 0.0, 0.0 }} ** 4;
        var point_color_int = [_][4]f32{.{ 0.0, 0.0, 0.0, 0.0 }} ** 4;
        var spot_pos_range = [_][4]f32{.{ 0.0, 0.0, 0.0, 0.0 }} ** 2;
        var spot_dir_inner = [_][4]f32{.{ 0.0, 0.0, 0.0, 0.0 }} ** 2;
        var spot_color_outer = [_][4]f32{.{ 0.0, 0.0, 0.0, 0.0 }} ** 2;
        var spot_intensity = [_][4]f32{.{ 0.0, 0.0, 0.0, 0.0 }} ** 2;
        var spot_view_proj = [_]Mat4{Mat4.identity} ** 2;
        var spot_shadow_params = [_][4]f32{.{ 0.0, 0.0, 0.0, 0.0 }} ** 2;

        // Pick the most relevant lights for the camera before packing; the
        // shader uniform arrays only hold 4 point + 2 spot slots.
        var point_buf: [4]*PointLight = undefined;
        const num_point = scene_light_selection.selectPoint(self.point_lights.items, eye, &point_buf);
        for (point_buf[0..num_point], 0..) |pl, i| {
            point_pos_range[i] = .{ pl.position.x, pl.position.y, pl.position.z, pl.range };
            point_color_int[i] = .{ pl.color.r, pl.color.g, pl.color.b, pl.intensity };
        }
        light_counts[0] = @floatFromInt(num_point);

        var spot_buf: [2]*SpotLight = undefined;
        const num_spot = scene_light_selection.selectSpot(self.spot_lights.items, eye, &spot_buf);
        var spot_shadows_buf: [2]passes.SpotShadowRenderInfo = undefined;
        var num_spot_shadows: usize = 0;

        for (spot_buf[0..num_spot], 0..) |sl, i| {
            const dir = sl.direction.normalize();
            const cos_inner = @cos(sl.inner_angle_deg * (std.math.pi / 180.0));
            const cos_outer = @cos(sl.outer_angle_deg * (std.math.pi / 180.0));
            spot_pos_range[i] = .{ sl.position.x, sl.position.y, sl.position.z, sl.range };
            spot_dir_inner[i] = .{ dir.x, dir.y, dir.z, cos_inner };
            spot_color_outer[i] = .{ sl.color.r, sl.color.g, sl.color.b, cos_outer };
            spot_intensity[i] = .{ sl.intensity, 0.0, 0.0, 0.0 };

            if (sl.cast_shadows and self.enable_shadows and sl.is_enabled) {
                const svp = sl.getShadowViewProj();
                spot_view_proj[i] = svp;
                spot_shadow_params[i] = .{ 1.0, sl.shadow_bias, sl.shadow_normal_bias, 0.0 };
                spot_shadows_buf[num_spot_shadows] = .{
                    .spot_index = i,
                    .view_proj = svp,
                };
                num_spot_shadows += 1;
            } else {
                spot_shadow_params[i] = .{ 0.0, 0.0, 0.0, 0.0 };
            }
        }
        light_counts[1] = @floatFromInt(num_spot);

        // ==============================================
        // PASS 1: OFFSCREEN SHADOW DEPTH PASS
        // ==============================================
        if (self.enable_shadows) {
            const shadow_draws = self.shadow_pass.render(self.meshes.items, self.frame_id, cascades, spot_shadows_buf[0..num_spot_shadows]);
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

        if (self.post_process.enabled) {
            self.postprocess_pass.resize(cur_w, cur_h);
            self.bloom_pass.resize(cur_w, cur_h);
            var offscreen_pass = sg.Pass{
                .action = main_pass_action,
            };
            offscreen_pass.attachments.colors[0] = self.postprocess_pass.offscreen_color_att_view;
            offscreen_pass.attachments.depth_stencil = self.postprocess_pass.offscreen_depth_att_view;
            if (self.postprocess_pass.sample_count > 1) {
                offscreen_pass.attachments.resolves[0] = self.postprocess_pass.offscreen_resolve_att_view;
            }
            sg.beginPass(offscreen_pass);
        } else {
            sg.beginPass(.{
                .action = main_pass_action,
                .swapchain = sglue.swapchain(),
            });
        }

        // Viewport for pixel-width outline expansion (static, shared).
        passes.OutlinePass.resize(cur_w, cur_h);

        // State sorting: opaque items group by shader type and textures,
        // Front-to-Back Early-Z. Transparent items sort strictly
        // back-to-front in their own queue and draw after all opaque work.
        std.mem.sort(RenderMeshItem, self.render_queue.items, {}, sortRenderItems);
        std.mem.sort(RenderMeshItem, self.transparent_queue.items, {}, sortTransparentBackToFront);

        const frame_ctx = FrameContext{
            .view_proj = view_proj,
            .eye = eye,
            .sun_dir = sun_dir,
            .sun_color = sun_color,
            .sun_intensity = sun_intensity,
            .cascades = cascades,
            .light_counts = light_counts,
            .point_pos_range = point_pos_range,
            .point_color_int = point_color_int,
            .spot_pos_range = spot_pos_range,
            .spot_dir_inner = spot_dir_inner,
            .spot_color_outer = spot_color_outer,
            .spot_intensity = spot_intensity,
            .spot_view_proj = spot_view_proj,
            .spot_shadow_params = spot_shadow_params,
        };

        var current_pipeline_id: u32 = 0;

        // Opaque regular meshes first (front-to-back, early-Z). Transparent
        // items live in transparent_queue and are drawn below, after every
        // opaque mesh including instanced ones.
        for (self.render_queue.items) |item| {
            self.drawRegularItem(item, frame_ctx, &current_pipeline_id);
        }

        // Opaque instanced meshes.
        for (self.opaque_instanced_queue.items) |mesh| {
            self.drawInstancedMesh(mesh, frame_ctx, &current_pipeline_id);
        }

        // Transparent regular meshes (strict back-to-front, blended).
        for (self.transparent_queue.items) |item| {
            self.drawRegularItem(item, frame_ctx, &current_pipeline_id);
        }

        // Transparent instanced meshes last, drawn as-is (no per-instance
        // sorting; documented limitation).
        for (self.transparent_instanced_queue.items) |mesh| {
            self.drawInstancedMesh(mesh, frame_ctx, &current_pipeline_id);
        }

        // Inverse-hull outline for highlighted meshes: inside the main pass,
        // depth-tested, no depth write, drawn after all surface geometry.
        if (self.outline_enabled and self.outline_meshes.items.len > 0) {
            self.outline_pass.render(view_proj, eye, self.outline_meshes.items, self.outline_color, self.outline_width_px);
            self.stats.draw_calls += @intCast(self.outline_meshes.items.len);
        }

        // Physics debug lines (3D pass, depth-tested, no depth write).
        if (self.show_physics_debug) {
            if (self.physics_world) |*pw| {
                self.debug_lines.clearRetainingCapacity();
                pw.appendDebugLines(self.allocator, &self.debug_lines) catch {};
                if (self.debug_lines.items.len > 0) {
                    if (self.debug_pass == null) {
                        self.debug_pass = passes.DebugPass.init(self.allocator) catch null;
                    }
                    if (self.debug_pass) |*dp| {
                        dp.render(view_proj, self.debug_lines.items);
                        self.stats.draw_calls += 1;
                    }
                }
            }
        }

        // Skybox Pass
        if (self.skybox_enabled) {
            self.skybox_pass.render(camera, aspect, self.skybox_texture orelse self.default_cube_texture, self.skybox_exposure);
            self.stats.draw_calls += 1;
            self.stats.triangles += 12;
        }

        // Particle Pass
        if (self.particle_systems.items.len > 0) {
            self.particle_pass.render(self.particle_systems.items, camera, aspect);
            for (self.particle_systems.items) |ps| {
                if (ps.active_count > 0) {
                    self.stats.draw_calls += 1;
                    self.stats.triangles += 2 * @as(u32, @intCast(ps.active_count));
                }
            }
        }

        if (!self.post_process.enabled) {
            if (self.ui_canvas) |*ui_c| {
                ui_c.render(sapp.widthf(), sapp.heightf());
                self.stats.draw_calls += 1;
            }
        }

        sg.endPass();

        // ==============================================
        // PASS 2.5: SCREEN-SPACE AMBIENT OCCLUSION (SSAO)
        // ==============================================
        var ssao_view = self.default_white_texture.view;
        const ssao_active = self.ssao.enabled or self.ssao.debug_mode;
        if (self.post_process.enabled and ssao_active) {
            self.ssao_pass.render(
                camera,
                aspect,
                self.postprocess_pass.offscreen_depth_tex_view,
                self.ssao,
                cur_w,
                cur_h,
            );
            ssao_view = self.ssao_pass.ssao_blur_tex_view;
            self.stats.draw_calls += 2;
            self.stats.triangles += 4;
        }

        // ==============================================
        // PASS 2.75: BLOOM MIP PYRAMID (optional)
        // ==============================================
        // Multi-pass glow; when disabled the composite shader keeps its legacy
        // in-shader bloom and this binds the resolved scene view placeholder.
        var bloom_view = self.postprocess_pass.offscreen_resolve_tex_view;
        if (self.post_process.enabled and self.post_process.bloom_enabled and self.post_process.bloom_pyramid) {
            bloom_view = self.bloom_pass.render(
                self.postprocess_pass.offscreen_resolve_tex_view,
                self.post_process.bloom_threshold,
                self.post_process.bloom_pyramid_mips,
                cur_w,
                cur_h,
            );
            const mips = postprocess.clampBloomMips(self.post_process.bloom_pyramid_mips);
            self.stats.draw_calls += 2 * @as(u32, mips) - 1;
        }
        self.postprocess_pass.setBloomTexture(bloom_view);

        // ==============================================
        // PASS 3: FULLSCREEN POST-PROCESSING PASS
        // ==============================================
        if (self.post_process.enabled) {
            var swap_action = sg.PassAction{};
            swap_action.colors[0] = .{
                .load_action = .DONTCARE,
            };
            sg.beginPass(.{
                .action = swap_action,
                .swapchain = sglue.swapchain(),
            });

            const inv_view_proj = view_proj.invert() orelse Mat4.identity;

            self.postprocess_pass.render(
                self.post_process,
                ssao_view,
                self.ssao.enabled,
                self.ssao.debug_mode,
                self.ssao.intensity,
                cur_w,
                cur_h,
                view_proj,
                inv_view_proj,
                eye,
                sun_dir,
                sun_color,
                camera.getNear(),
                camera.getFar(),
            );
            self.stats.draw_calls += 1;
            self.stats.triangles += 2;

            // Render 2D UI overlay on top of post-processed swapchain
            if (self.ui_canvas) |*ui_c| {
                ui_c.render(@floatFromInt(cur_w), @floatFromInt(cur_h));
                self.stats.draw_calls += 1;
            }

            sg.endPass();
        }

        sg.commit();
    }

    pub fn deinit(self: *Scene) void {
        if (self.decal_manager) |*dm| {
            dm.deinit();
            self.decal_manager = null;
        }

        for (self.meshes.items) |m| {
            m.deinit(self.allocator);
            self.allocator.destroy(m);
        }
        self.meshes.deinit(self.allocator);

        for (self.materials.items) |mat| {
            if (mat.diffuse_texture) |*t| {
                t.deinit();
            }
            self.allocator.destroy(mat);
        }
        self.materials.deinit(self.allocator);

        var destroyed_views = std.AutoHashMap(u32, void).init(self.allocator);
        defer destroyed_views.deinit();

        for (self.pbr_materials.items) |mat| {
            inline for (.{ "albedo_texture", "normal_texture", "metallic_roughness_texture", "emissive_texture", "occlusion_texture" }) |field| {
                if (@field(mat, field)) |*t| {
                    if (t.view.id != 0 and !destroyed_views.contains(t.view.id)) {
                        destroyed_views.put(t.view.id, {}) catch {};
                        t.deinit();
                    }
                }
            }
            self.allocator.destroy(mat);
        }
        self.pbr_materials.deinit(self.allocator);

        for (self.point_lights.items) |pl| {
            if (pl.owns_name) self.allocator.free(pl.name);
            self.allocator.destroy(pl);
        }
        self.point_lights.deinit(self.allocator);

        for (self.spot_lights.items) |sl| {
            if (sl.owns_name) self.allocator.free(sl.name);
            self.allocator.destroy(sl);
        }
        self.spot_lights.deinit(self.allocator);

        if (self.directional_light) |dl| {
            if (dl.owns_name) self.allocator.free(dl.name);
            self.allocator.destroy(dl);
            self.directional_light = null;
        }

        self.render_queue.deinit(self.allocator);
        self.transparent_queue.deinit(self.allocator);
        self.opaque_instanced_queue.deinit(self.allocator);
        self.transparent_instanced_queue.deinit(self.allocator);
        self.instance_matrices.deinit(self.allocator);
        self.debug_lines.deinit(self.allocator);

        for (self.trail_meshes.items) |tm| {
            tm.deinit();
            self.allocator.destroy(tm);
        }
        self.trail_meshes.deinit(self.allocator);

        for (self.nav_agents.items) |agent| {
            agent.deinit();
            self.allocator.destroy(agent);
        }
        self.nav_agents.deinit(self.allocator);

        for (self.nav_meshes.items) |nm| {
            nm.deinit();
            self.allocator.destroy(nm);
        }
        self.nav_meshes.deinit(self.allocator);

        self.default_white_texture.deinit();
        self.default_normal_texture.deinit();
        self.default_cube_texture.deinit();
        if (self.skybox_texture) |*c| {
            c.deinit();
        }

        sg.destroyPipeline(self.pipeline_u16);
        sg.destroyPipeline(self.pipeline_u32);
        sg.destroyPipeline(self.pipeline_pbr_u16);
        sg.destroyPipeline(self.pipeline_pbr_u32);
        sg.destroyPipeline(self.pipeline_skinned_pbr_u16);
        sg.destroyPipeline(self.pipeline_skinned_pbr_u32);
        sg.destroyPipeline(self.pipeline_instanced_u16);
        sg.destroyPipeline(self.pipeline_instanced_u32);
        sg.destroyPipeline(self.pipeline_blend_u16);
        sg.destroyPipeline(self.pipeline_blend_u32);
        sg.destroyPipeline(self.pipeline_pbr_blend_u16);
        sg.destroyPipeline(self.pipeline_pbr_blend_u32);
        sg.destroyPipeline(self.pipeline_skinned_pbr_blend_u16);
        sg.destroyPipeline(self.pipeline_skinned_pbr_blend_u32);
        sg.destroyPipeline(self.pipeline_instanced_blend_u16);
        sg.destroyPipeline(self.pipeline_instanced_blend_u32);
        sg.destroyPipeline(self.pipeline_instanced_pbr_u16);
        sg.destroyPipeline(self.pipeline_instanced_pbr_u32);
        sg.destroyPipeline(self.pipeline_instanced_pbr_blend_u16);
        sg.destroyPipeline(self.pipeline_instanced_pbr_blend_u32);

        for (self.animation_groups.items) |ag| {
            ag.deinit();
        }
        self.animation_groups.deinit(self.allocator);

        for (self.skeletons.items) |skel| {
            skel.deinit();
        }
        self.skeletons.deinit(self.allocator);

        // Deinitialize passes
        self.shadow_pass.deinit();
        self.skybox_pass.deinit();
        self.particle_pass.deinit();
        self.postprocess_pass.deinit();
        self.ssao_pass.deinit();
        self.bloom_pass.deinit();
        self.outline_pass.deinit();
        if (self.debug_pass) |*dp| dp.deinit();
        self.outline_meshes.deinit(self.allocator);
        self.ds_pipelines.deinit();

        for (self.particle_systems.items) |ps| {
            ps.deinit();
            self.allocator.destroy(ps);
        }
        self.particle_systems.deinit(self.allocator);

        if (self.physics_world) |*pw| {
            pw.deinit();
        }

        if (self.ui_canvas) |*u| {
            u.deinit();
        }
    }
};

test "project VP cache: equal cameras hit" {
    const a: Camera = .{ .arc_rotate = .{ .alpha = 0.5, .beta = 1.0, .radius = 8.0 } };
    const b: Camera = .{ .arc_rotate = .{ .alpha = 0.5, .beta = 1.0, .radius = 8.0 } };
    try std.testing.expect(Scene.camerasEqualForProjection(a, b));
    // Name pointers and input state do not affect the matrices.
    const c: Camera = .{ .arc_rotate = .{ .alpha = 0.5, .beta = 1.0, .radius = 8.0, .name = "other", .is_dragging = true } };
    try std.testing.expect(Scene.camerasEqualForProjection(a, c));

    const fa: Camera = .{ .free = .{ .position = Vec3.new(1.0, 2.0, 3.0) } };
    const fb: Camera = .{ .free = .{ .position = Vec3.new(1.0, 2.0, 3.0) } };
    try std.testing.expect(Scene.camerasEqualForProjection(fa, fb));

    const ga: Camera = .{ .follow = .{ .target_position = Vec3.new(1.0, 0.0, 0.0), .radius = 5.0 } };
    const gb: Camera = .{ .follow = .{ .target_position = Vec3.new(1.0, 0.0, 0.0), .radius = 5.0 } };
    try std.testing.expect(Scene.camerasEqualForProjection(ga, gb));
}

test "project VP cache: changed field misses" {
    const a: Camera = .{ .arc_rotate = .{ .radius = 8.0 } };
    const b: Camera = .{ .arc_rotate = .{ .radius = 9.0 } };
    try std.testing.expect(!Scene.camerasEqualForProjection(a, b));

    const fa: Camera = .{ .free = .{ .position = Vec3.new(1.0, 2.0, 3.0) } };
    const fb: Camera = .{ .free = .{ .position = Vec3.new(1.0, 2.0, 4.0) } };
    try std.testing.expect(!Scene.camerasEqualForProjection(fa, fb));

    const ga: Camera = .{ .follow = .{ .radius = 5.0 } };
    const gb: Camera = .{ .follow = .{ .radius = 6.0 } };
    try std.testing.expect(!Scene.camerasEqualForProjection(ga, gb));

    // Distinct target meshes miss even when every scalar matches.
    var m1: Mesh = undefined;
    var m2: Mesh = undefined;
    const ha: Camera = .{ .follow = .{ .target_mesh = &m1 } };
    const hb: Camera = .{ .follow = .{ .target_mesh = &m2 } };
    try std.testing.expect(!Scene.camerasEqualForProjection(ha, hb));
}

test "project VP cache: different union variants miss" {
    const arc: Camera = .{ .arc_rotate = .{} };
    const free: Camera = .{ .free = .{} };
    const follow: Camera = .{ .follow = .{} };
    try std.testing.expect(!Scene.camerasEqualForProjection(arc, free));
    try std.testing.expect(!Scene.camerasEqualForProjection(free, follow));
    try std.testing.expect(!Scene.camerasEqualForProjection(follow, arc));
}

test "project VP cache: target and fly cameras" {
    // Equal target pairs hit; tuning-only (smoothing) differences still hit.
    const ta: Camera = .{ .target = .{ .position = Vec3.new(0.0, 0.0, 5.0), .target = Vec3.zero } };
    const tb: Camera = .{ .target = .{ .position = Vec3.new(0.0, 0.0, 5.0), .target = Vec3.zero } };
    try std.testing.expect(Scene.camerasEqualForProjection(ta, tb));
    const tc: Camera = .{ .target = .{ .position = Vec3.new(0.0, 0.0, 5.0), .target = Vec3.zero, .smoothing = 0.0 } };
    try std.testing.expect(Scene.camerasEqualForProjection(ta, tc));

    // Moved target point, moved position, pending goal, and changed up miss.
    const td: Camera = .{ .target = .{ .position = Vec3.new(0.0, 0.0, 5.0), .target = Vec3.new(1.0, 0.0, 0.0) } };
    try std.testing.expect(!Scene.camerasEqualForProjection(ta, td));
    const te: Camera = .{ .target = .{ .position = Vec3.new(0.0, 1.0, 5.0), .target = Vec3.zero } };
    try std.testing.expect(!Scene.camerasEqualForProjection(ta, te));
    var tf: Camera = .{ .target = .{ .position = Vec3.new(0.0, 0.0, 5.0), .target = Vec3.zero } };
    tf.target.desired_target = Vec3.new(0.0, 1.0, 0.0);
    try std.testing.expect(!Scene.camerasEqualForProjection(ta, tf));
    const tg: Camera = .{ .target = .{ .position = Vec3.new(0.0, 5.0, 0.0), .target = Vec3.zero, .up = Vec3.new(0.0, 0.0, 1.0) } };
    const th: Camera = .{ .target = .{ .position = Vec3.new(0.0, 5.0, 0.0), .target = Vec3.zero } };
    try std.testing.expect(!Scene.camerasEqualForProjection(tg, th));

    // Equal fly pairs hit; roll-only differences miss (roll tilts view up).
    const fa: Camera = .{ .fly = .{ .position = Vec3.new(1.0, 2.0, 3.0) } };
    const fb: Camera = .{ .fly = .{ .position = Vec3.new(1.0, 2.0, 3.0) } };
    try std.testing.expect(Scene.camerasEqualForProjection(fa, fb));
    const fc: Camera = .{ .fly = .{ .position = Vec3.new(1.0, 2.0, 3.0), .rotation = Vec3.new(0.0, 0.0, 90.0) } };
    try std.testing.expect(!Scene.camerasEqualForProjection(fa, fc));
    const fd: Camera = .{ .fly = .{ .position = Vec3.new(1.0, 2.0, 4.0) } };
    try std.testing.expect(!Scene.camerasEqualForProjection(fa, fd));

    // Cross-variant pairs (old and new) always miss.
    const arc: Camera = .{ .arc_rotate = .{} };
    const free: Camera = .{ .free = .{} };
    const follow: Camera = .{ .follow = .{} };
    try std.testing.expect(!Scene.camerasEqualForProjection(ta, fa));
    try std.testing.expect(!Scene.camerasEqualForProjection(fa, ta));
    try std.testing.expect(!Scene.camerasEqualForProjection(arc, ta));
    try std.testing.expect(!Scene.camerasEqualForProjection(ta, arc));
    try std.testing.expect(!Scene.camerasEqualForProjection(free, fa));
    try std.testing.expect(!Scene.camerasEqualForProjection(fa, free));
    try std.testing.expect(!Scene.camerasEqualForProjection(follow, ta));
    try std.testing.expect(!Scene.camerasEqualForProjection(fa, follow));
}

test "transparent classification follows material alpha mode" {
    try std.testing.expect(!Scene.materialIsTransparent(null));
    var std_mat = StandardMaterial.init("s");
    var pbr_mat = PBRMaterial.init("p");
    try std.testing.expect(!Scene.materialIsTransparent(Material{ .standard = &std_mat }));
    try std.testing.expect(!Scene.materialIsTransparent(Material{ .pbr = &pbr_mat }));
    std_mat.alpha_mode = .blend;
    pbr_mat.alpha_mode = .blend;
    try std.testing.expect(Scene.materialIsTransparent(Material{ .standard = &std_mat }));
    try std.testing.expect(Scene.materialIsTransparent(Material{ .pbr = &pbr_mat }));
}

test "transparent queue sorts strictly back-to-front" {
    var m: Mesh = undefined;
    var items = [_]RenderMeshItem{
        .{ .mesh = &m, .model = Mat4.identity, .distance_sq = 1.0, .is_pbr = false, .texture_id = 0, .transparent = true },
        .{ .mesh = &m, .model = Mat4.identity, .distance_sq = 9.0, .is_pbr = false, .texture_id = 0, .transparent = true },
        .{ .mesh = &m, .model = Mat4.identity, .distance_sq = 4.0, .is_pbr = true, .texture_id = 1, .transparent = true },
    };
    std.mem.sort(RenderMeshItem, &items, {}, Scene.sortTransparentBackToFront);
    try std.testing.expect(items[0].distance_sq == 9.0);
    try std.testing.expect(items[1].distance_sq == 4.0);
    try std.testing.expect(items[2].distance_sq == 1.0);
}

test "opaque sort unchanged: state groups, front-to-back" {
    var m: Mesh = undefined;
    var items = [_]RenderMeshItem{
        .{ .mesh = &m, .model = Mat4.identity, .distance_sq = 9.0, .is_pbr = false, .texture_id = 2 },
        .{ .mesh = &m, .model = Mat4.identity, .distance_sq = 1.0, .is_pbr = false, .texture_id = 2 },
        .{ .mesh = &m, .model = Mat4.identity, .distance_sq = 5.0, .is_pbr = true, .texture_id = 1 },
        .{ .mesh = &m, .model = Mat4.identity, .distance_sq = 3.0, .is_pbr = false, .texture_id = 1 },
    };
    // Default transparent flag is false, so legacy items sort as before.
    try std.testing.expect(!items[0].transparent);
    std.mem.sort(RenderMeshItem, &items, {}, Scene.sortRenderItems);
    // Standard before PBR, texture id ascending, front-to-back within a group.
    try std.testing.expect(!items[0].is_pbr and items[0].texture_id == 1 and items[0].distance_sq == 3.0);
    try std.testing.expect(!items[1].is_pbr and items[1].texture_id == 2 and items[1].distance_sq == 1.0);
    try std.testing.expect(!items[2].is_pbr and items[2].texture_id == 2 and items[2].distance_sq == 9.0);
    try std.testing.expect(items[3].is_pbr and items[3].distance_sq == 5.0);
}

test "blendDescFor enables alpha blending without depth write" {
    const base = sg.PipelineDesc{
        .shader = .{},
        .index_type = .UINT32,
        .depth = .{ .compare = .LESS_EQUAL, .write_enabled = true },
        .cull_mode = .BACK,
        .face_winding = .CCW,
    };
    const blended = Scene.blendDescFor(base);
    try std.testing.expect(blended.colors[0].blend.enabled);
    try std.testing.expect(blended.colors[0].blend.src_factor_rgb == .SRC_ALPHA);
    try std.testing.expect(blended.colors[0].blend.dst_factor_rgb == .ONE_MINUS_SRC_ALPHA);
    try std.testing.expect(!blended.depth.write_enabled);
    try std.testing.expect(blended.depth.compare == .LESS_EQUAL);
    try std.testing.expect(blended.index_type == .UINT32);
    try std.testing.expect(blended.cull_mode == .BACK);
    // Pure function: the base desc is left untouched.
    try std.testing.expect(!base.colors[0].blend.enabled);
    try std.testing.expect(base.depth.write_enabled);
}

test "pipeline selection follows transparency flag" {
    var scene: Scene = undefined;
    scene.pipeline_u16.id = 11;
    scene.pipeline_u32.id = 12;
    scene.pipeline_blend_u16.id = 13;
    scene.pipeline_blend_u32.id = 14;
    scene.pipeline_pbr_u16.id = 21;
    scene.pipeline_pbr_u32.id = 22;
    scene.pipeline_pbr_blend_u16.id = 23;
    scene.pipeline_pbr_blend_u32.id = 24;
    scene.pipeline_skinned_pbr_u16.id = 31;
    scene.pipeline_skinned_pbr_u32.id = 32;
    scene.pipeline_skinned_pbr_blend_u16.id = 33;
    scene.pipeline_skinned_pbr_blend_u32.id = 34;

    var mesh_obj: Mesh = undefined;
    mesh_obj.index_type = .UINT16;
    mesh_obj.skeleton = null;

    const opaque_std = RenderMeshItem{ .mesh = &mesh_obj, .model = Mat4.identity, .distance_sq = 1.0, .is_pbr = false, .texture_id = 0, .transparent = false };
    var blend_std = opaque_std;
    blend_std.transparent = true;
    try std.testing.expect(scene.pipelineForRegularItem(opaque_std) == 11);
    try std.testing.expect(scene.pipelineForRegularItem(blend_std) == 13);

    mesh_obj.index_type = .UINT32;
    try std.testing.expect(scene.pipelineForRegularItem(opaque_std) == 12);
    try std.testing.expect(scene.pipelineForRegularItem(blend_std) == 14);

    const opaque_pbr = RenderMeshItem{ .mesh = &mesh_obj, .model = Mat4.identity, .distance_sq = 1.0, .is_pbr = true, .texture_id = 0, .transparent = false };
    var blend_pbr = opaque_pbr;
    blend_pbr.transparent = true;
    try std.testing.expect(scene.pipelineForRegularItem(opaque_pbr) == 22);
    try std.testing.expect(scene.pipelineForRegularItem(blend_pbr) == 24);

    var skel: Skeleton = undefined;
    mesh_obj.skeleton = &skel;
    try std.testing.expect(scene.pipelineForRegularItem(opaque_pbr) == 32);
    try std.testing.expect(scene.pipelineForRegularItem(blend_pbr) == 34);
    mesh_obj.index_type = .UINT16;
    try std.testing.expect(scene.pipelineForRegularItem(opaque_pbr) == 31);
    try std.testing.expect(scene.pipelineForRegularItem(blend_pbr) == 33);
}

test "frameUniforms packs shared lighting state verbatim" {
    var scene: Scene = undefined;
    scene.light.ground_color = Color3.new(0.2, 0.25, 0.3);
    scene.enable_shadows = true;
    scene.shadow_bias = 0.0012;
    scene.shadow_intensity = 0.75;
    scene.shadow_normal_bias = 0.02;
    scene.shadow_softness = 1.5;
    scene.shadow_debug_cascades = true;
    scene.cascade_splits = .{ 10.0, 26.0, 65.0, 150.0 };

    var mesh_obj: Mesh = undefined;
    mesh_obj.receive_shadows = true;

    const cascades = [_]Mat4{ Mat4.identity, Mat4.identity, Mat4.identity, Mat4.identity };
    const ctx = Scene.FrameContext{
        .view_proj = Mat4.identity,
        .eye = Vec3.new(1.0, 2.0, 3.0),
        .sun_dir = Vec3.new(0.5, 1.0, 0.3),
        .sun_color = Color3.new(1.0, 0.9, 0.8),
        .sun_intensity = 2.0,
        .cascades = cascades,
        .light_counts = .{ 1.0, 0.0, 0.0, 0.0 },
        .point_pos_range = [_][4]f32{.{ 1, 2, 3, 10 }} ** 4,
        .point_color_int = [_][4]f32{.{ 1, 1, 1, 1 }} ** 4,
        .spot_pos_range = [_][4]f32{.{ 4, 5, 6, 20 }} ** 2,
        .spot_dir_inner = [_][4]f32{.{ 0, -1, 0, 0.9 }} ** 2,
        .spot_color_outer = [_][4]f32{.{ 1, 1, 1, 0.7 }} ** 2,
        .spot_intensity = [_][4]f32{.{ 3, 0, 0, 0 }} ** 2,
        .spot_view_proj = [_]Mat4{ Mat4.identity, Mat4.identity },
        .spot_shadow_params = [_][4]f32{.{ 0, 0, 0, 0 }} ** 2,
    };

    const f = scene.frameUniforms(&mesh_obj, ctx);
    try std.testing.expectEqual([4]f32{ 1.0, 2.0, 3.0, 4.0 }, f.eye_pos);
    try std.testing.expectEqual([4]f32{ 0.5, 1.0, 0.3, 2048.0 }, f.light_dir);
    try std.testing.expectEqual([4]f32{ 1.0, 0.9, 0.8, 2.0 }, f.light_color);
    try std.testing.expectEqual([4]f32{ 0.2, 0.25, 0.3, 1.0 }, f.ambient_color);
    try std.testing.expectEqual([4]f32{ 0.0012, 0.75, 0.02, 1.5 }, f.shadow_params);
    try std.testing.expectEqual([4]f32{ 10.0, 26.0, 65.0, 150.0 }, f.shadow_splits);
    try std.testing.expectEqual(cascades, f.cascade_view_proj);
    try std.testing.expectEqual([4]f32{ 1.0, 0.0, 0.02, 0.01 }, f.cascade_debug);
    try std.testing.expectEqual([4]f32{ 1.0, 0.0, 0.0005, 0.01 }, f.light_counts);
    try std.testing.expectEqual(ctx.point_pos_range, f.point_pos_range);
    try std.testing.expectEqual(ctx.spot_intensity, f.spot_intensity);

    // Disabled shadows (globally or per-mesh) zero the bias/intensity lanes
    // but keep normal bias and softness, exactly like the legacy literals.
    scene.enable_shadows = false;
    const off = scene.frameUniforms(&mesh_obj, ctx);
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.02, 1.5 }, off.shadow_params);
    scene.enable_shadows = true;
    mesh_obj.receive_shadows = false;
    const skipped = scene.frameUniforms(&mesh_obj, ctx);
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.02, 1.5 }, skipped.shadow_params);
}

test "pipeline layout table preserves per-family vertex layouts" {
    var desc = sg.PipelineDesc{
        .shader = .{},
        .index_type = .UINT16,
        .depth = .{ .compare = .LESS_EQUAL, .write_enabled = true },
        .cull_mode = .BACK,
        .face_winding = .CCW,
    };

    Scene.pipelineLayoutFor(.standard, &desc);
    try std.testing.expect(desc.layout.buffers[0].stride == @as(i32, @intCast(@sizeOf(Vertex))));
    try std.testing.expect(desc.layout.attrs[shd.ATTR_standard_position].format == .FLOAT3);
    try std.testing.expect(desc.layout.attrs[shd.ATTR_standard_position].offset == @as(i32, @intCast(@offsetOf(Vertex, "position"))));
    try std.testing.expect(desc.layout.attrs[shd.ATTR_standard_texcoord0].format == .FLOAT2);

    desc = sg.PipelineDesc{
        .shader = .{},
        .index_type = .UINT16,
        .depth = .{ .compare = .LESS_EQUAL, .write_enabled = true },
        .cull_mode = .BACK,
        .face_winding = .CCW,
    };
    Scene.pipelineLayoutFor(.pbr, &desc);
    try std.testing.expect(desc.layout.attrs[pbr_shd.ATTR_pbr_tangent].format == .FLOAT4);
    try std.testing.expect(desc.layout.attrs[pbr_shd.ATTR_pbr_tangent].offset == @as(i32, @intCast(@offsetOf(Vertex, "tangent"))));

    desc = sg.PipelineDesc{
        .shader = .{},
        .index_type = .UINT16,
        .depth = .{ .compare = .LESS_EQUAL, .write_enabled = true },
        .cull_mode = .BACK,
        .face_winding = .CCW,
    };
    Scene.pipelineLayoutFor(.instanced, &desc);
    try std.testing.expect(desc.layout.buffers[1].stride == @as(i32, @intCast(@sizeOf(Mat4))));
    try std.testing.expect(desc.layout.buffers[1].step_func == .PER_INSTANCE);
    try std.testing.expect(desc.layout.attrs[inst_shd.ATTR_instanced_inst_mat0].offset == 0);
    try std.testing.expect(desc.layout.attrs[inst_shd.ATTR_instanced_inst_mat3].offset == 48);

    desc = sg.PipelineDesc{
        .shader = .{},
        .index_type = .UINT16,
        .depth = .{ .compare = .LESS_EQUAL, .write_enabled = true },
        .cull_mode = .BACK,
        .face_winding = .CCW,
    };
    Scene.pipelineLayoutFor(.skinned_pbr, &desc);
    try std.testing.expect(desc.layout.attrs[skinned_pbr_shd.ATTR_skinned_pbr_joints].offset == @as(i32, @intCast(@offsetOf(Vertex, "joints"))));
    try std.testing.expect(desc.layout.attrs[skinned_pbr_shd.ATTR_skinned_pbr_weights].offset == @as(i32, @intCast(@offsetOf(Vertex, "weights"))));
}
