const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const sapp = sokol.app;
const sglue = sokol.glue;
const shd = @import("shader");
const pbr_shd = @import("pbr_shader");
const inst_shd = @import("instanced_shader");
const passes = @import("passes/mod.zig");
const postprocess = @import("postprocess.zig");
pub const PostProcessConfig = postprocess.PostProcessConfig;
pub const TonemappingType = postprocess.TonemappingType;
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

const ArcRotateCamera = @import("camera.zig").ArcRotateCamera;
const lights = @import("lights.zig");
const HemisphericLight = lights.HemisphericLight;
const PointLight = lights.PointLight;
const PointLightOptions = lights.PointLightOptions;
const SpotLight = lights.SpotLight;
const SpotLightOptions = lights.SpotLightOptions;
const Mesh = @import("mesh.zig").Mesh;
const Vertex = @import("mesh.zig").Vertex;
const InstancedMesh = @import("mesh.zig").InstancedMesh;
const StandardMaterial = @import("material.zig").StandardMaterial;
const PBRMaterial = @import("material.zig").PBRMaterial;
const Material = @import("material.zig").Material;
const Texture = @import("texture.zig").Texture;
const CubeTexture = @import("texture.zig").CubeTexture;
const SkyboxConfig = @import("texture.zig").SkyboxConfig;

pub const SceneStats = struct {
    total_meshes: u32 = 0,
    rendered_meshes: u32 = 0,
    culled_meshes: u32 = 0,
    draw_calls: u32 = 0,
    triangles: u32 = 0,
    pipeline_switches: u32 = 0,
};

pub const RenderMeshItem = struct {
    mesh: *Mesh,
    model: Mat4,
    distance_sq: f32,
    is_pbr: bool,
    texture_id: u32,
};

pub const Scene = struct {
    allocator: std.mem.Allocator,
    meshes: std.ArrayListUnmanaged(*Mesh) = .empty,
    materials: std.ArrayListUnmanaged(*StandardMaterial) = .empty,
    pbr_materials: std.ArrayListUnmanaged(*PBRMaterial) = .empty,
    active_camera: ?ArcRotateCamera = null,
    clear_color: Color4 = Color4.new(0.12, 0.14, 0.18, 1.0),
    light: HemisphericLight = .{},
    point_lights: std.ArrayListUnmanaged(*PointLight) = .empty,
    spot_lights: std.ArrayListUnmanaged(*SpotLight) = .empty,
    default_material: StandardMaterial = StandardMaterial.init("default"),
    default_white_texture: Texture,
    default_normal_texture: Texture,
    default_black_texture: Texture,
    default_cube_texture: CubeTexture,

    enable_frustum_culling: bool = true,
    stats: SceneStats = .{},

    // Render Passes
    shadow_pass: passes.ShadowPass,
    skybox_pass: passes.SkyboxPass,
    particle_pass: passes.ParticlePass,
    postprocess_pass: passes.PostProcessPass,

    // Shadow Mapping settings
    enable_shadows: bool = true,
    shadow_bias: f32 = 0.003,
    shadow_intensity: f32 = 0.75,
    shadow_extent: f32 = 25.0,
    shadow_near: f32 = 0.5,
    shadow_far: f32 = 80.0,

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
    pipeline_instanced_u16: sg.Pipeline = .{},
    pipeline_instanced_u32: sg.Pipeline = .{},

    // Post-Processing settings
    post_process: PostProcessConfig = .{},

    // Particle Systems
    particle_systems: std.ArrayListUnmanaged(*particles.ParticleSystem) = .empty,

    // Physics World
    physics_world: ?PhysicsWorld = null,

    // 2D & 3D UI Canvas
    ui_canvas: ?UICanvas = null,

    render_queue: std.ArrayListUnmanaged(RenderMeshItem) = .empty,
    instance_matrices: std.ArrayListUnmanaged(Mat4) = .empty,

    pub fn init(allocator: std.mem.Allocator) Scene {
        var self = Scene{
            .allocator = allocator,
            .default_white_texture = Texture.createWhite1x1(),
            .default_normal_texture = Texture.createFlatNormal1x1(),
            .default_black_texture = Texture.createBlack1x1(),
            .default_cube_texture = CubeTexture.createDefault1x1(.{ 25, 30, 40, 255 }),
            .shadow_pass = passes.ShadowPass.init(),
            .skybox_pass = passes.SkyboxPass.init(),
            .particle_pass = passes.ParticlePass.init(),
            .postprocess_pass = passes.PostProcessPass.init(),
            .light = HemisphericLight.init("hemi", .{
                .direction = math.Vec3.new(0.5, 1.0, 0.3),
                .diffuse = Color3.white,
                .ground_color = Color3.new(0.2, 0.25, 0.3),
                .intensity = 1.0,
            }),
        };
        self.initPipelines();
        return self;
    }

    fn initPipelines(self: *Scene) void {
        // 1. Standard Material pipelines (Phong / Blinn-Phong)
        var pip_desc = sg.PipelineDesc{
            .shader = sg.makeShader(shd.standardShaderDesc(sg.queryBackend())),
            .index_type = .UINT16,
            .depth = .{
                .compare = .LESS_EQUAL,
                .write_enabled = true,
            },
            .cull_mode = .BACK,
            .face_winding = .CCW,
        };
        pip_desc.layout.buffers[0] = .{ .stride = @sizeOf(Vertex) };
        pip_desc.layout.attrs[shd.ATTR_standard_position] = .{ .format = .FLOAT3, .offset = @offsetOf(Vertex, "position") };
        pip_desc.layout.attrs[shd.ATTR_standard_normal] = .{ .format = .FLOAT3, .offset = @offsetOf(Vertex, "normal") };
        pip_desc.layout.attrs[shd.ATTR_standard_color0] = .{ .format = .FLOAT4, .offset = @offsetOf(Vertex, "color") };
        pip_desc.layout.attrs[shd.ATTR_standard_texcoord0] = .{ .format = .FLOAT2, .offset = @offsetOf(Vertex, "uv") };

        self.pipeline_u16 = sg.makePipeline(pip_desc);
        pip_desc.index_type = .UINT32;
        self.pipeline_u32 = sg.makePipeline(pip_desc);

        // 2. PBR Material pipelines (Cook-Torrance BRDF + Normal / MR / Emissive / AO)
        var pbr_desc = sg.PipelineDesc{
            .shader = sg.makeShader(pbr_shd.pbrShaderDesc(sg.queryBackend())),
            .index_type = .UINT16,
            .depth = .{
                .compare = .LESS_EQUAL,
                .write_enabled = true,
            },
            .cull_mode = .BACK,
            .face_winding = .CCW,
        };
        pbr_desc.layout.buffers[0] = .{ .stride = @sizeOf(Vertex) };
        pbr_desc.layout.attrs[pbr_shd.ATTR_pbr_position] = .{ .format = .FLOAT3, .offset = @offsetOf(Vertex, "position") };
        pbr_desc.layout.attrs[pbr_shd.ATTR_pbr_normal] = .{ .format = .FLOAT3, .offset = @offsetOf(Vertex, "normal") };
        pbr_desc.layout.attrs[pbr_shd.ATTR_pbr_color0] = .{ .format = .FLOAT4, .offset = @offsetOf(Vertex, "color") };
        pbr_desc.layout.attrs[pbr_shd.ATTR_pbr_texcoord0] = .{ .format = .FLOAT2, .offset = @offsetOf(Vertex, "uv") };
        pbr_desc.layout.attrs[pbr_shd.ATTR_pbr_tangent] = .{ .format = .FLOAT4, .offset = @offsetOf(Vertex, "tangent") };

        self.pipeline_pbr_u16 = sg.makePipeline(pbr_desc);
        pbr_desc.index_type = .UINT32;
        self.pipeline_pbr_u32 = sg.makePipeline(pbr_desc);

        // 3. Instanced Standard pipelines
        var inst_desc = sg.PipelineDesc{
            .shader = sg.makeShader(inst_shd.instancedShaderDesc(sg.queryBackend())),
            .index_type = .UINT16,
            .depth = .{
                .compare = .LESS_EQUAL,
                .write_enabled = true,
            },
            .cull_mode = .BACK,
            .face_winding = .CCW,
        };
        inst_desc.layout.buffers[0] = .{ .stride = @sizeOf(Vertex) };
        inst_desc.layout.attrs[inst_shd.ATTR_instanced_position] = .{ .buffer_index = 0, .format = .FLOAT3, .offset = @offsetOf(Vertex, "position") };
        inst_desc.layout.attrs[inst_shd.ATTR_instanced_normal] = .{ .buffer_index = 0, .format = .FLOAT3, .offset = @offsetOf(Vertex, "normal") };
        inst_desc.layout.attrs[inst_shd.ATTR_instanced_color0] = .{ .buffer_index = 0, .format = .FLOAT4, .offset = @offsetOf(Vertex, "color") };
        inst_desc.layout.attrs[inst_shd.ATTR_instanced_texcoord0] = .{ .buffer_index = 0, .format = .FLOAT2, .offset = @offsetOf(Vertex, "uv") };

        inst_desc.layout.buffers[1] = .{
            .step_func = .PER_INSTANCE,
            .step_rate = 1,
            .stride = @sizeOf(Mat4),
        };
        inst_desc.layout.attrs[inst_shd.ATTR_instanced_inst_mat0] = .{ .buffer_index = 1, .offset = 0, .format = .FLOAT4 };
        inst_desc.layout.attrs[inst_shd.ATTR_instanced_inst_mat1] = .{ .buffer_index = 1, .offset = 16, .format = .FLOAT4 };
        inst_desc.layout.attrs[inst_shd.ATTR_instanced_inst_mat2] = .{ .buffer_index = 1, .offset = 32, .format = .FLOAT4 };
        inst_desc.layout.attrs[inst_shd.ATTR_instanced_inst_mat3] = .{ .buffer_index = 1, .offset = 48, .format = .FLOAT4 };

        self.pipeline_instanced_u16 = sg.makePipeline(inst_desc);
        inst_desc.index_type = .UINT32;
        self.pipeline_instanced_u32 = sg.makePipeline(inst_desc);
    }

    pub fn resizeOffscreen(self: *Scene, width: i32, height: i32) void {
        self.postprocess_pass.resize(width, height);
    }

    pub fn setPostProcess(self: *Scene, config: PostProcessConfig) void {
        self.post_process = config;
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

    pub fn updatePhysics(self: *Scene, dt: f32) void {
        if (self.physics_world) |*pw| {
            pw.step(dt);
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
            if (!mesh.is_visible) continue;

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

    pub fn projectPoint(self: *Scene, world_pos: Vec3) ?math.Vec2 {
        const cam = self.active_camera orelse return null;
        const w = sapp.widthf();
        const h = sapp.heightf();
        if (w <= 0.0 or h <= 0.0) return null;
        const vp = cam.getViewProjection(w / h);
        return vp.projectPoint(world_pos, w, h);
    }

    pub fn handleEvent(self: *Scene, ev: [*c]const sapp.Event) void {
        if (self.active_camera) |*cam| {
            cam.handleEvent(ev);
        }
    }

    fn sortRenderItems(_: void, a: RenderMeshItem, b: RenderMeshItem) bool {
        if (a.is_pbr != b.is_pbr) {
            return !a.is_pbr;
        }
        if (a.texture_id != b.texture_id) {
            return a.texture_id < b.texture_id;
        }
        return a.distance_sq < b.distance_sq;
    }

    pub fn render(self: *Scene) void {
        const camera = self.active_camera orelse return;
        const aspect = sapp.widthf() / sapp.heightf();
        const view_proj = camera.getViewProjection(aspect);
        const frustum = Frustum.fromViewProjection(view_proj);
        const eye = camera.getPosition();

        self.stats = .{};
        self.render_queue.clearRetainingCapacity();

        // 1. Directional Light Shadow View-Projection
        const light_pos = self.light.direction.scale(35.0);
        const light_view = Mat4.lookAt(light_pos, Vec3.zero, Vec3.up);
        const light_proj = Mat4.orthographic(
            -self.shadow_extent,
            self.shadow_extent,
            -self.shadow_extent,
            self.shadow_extent,
            self.shadow_near,
            self.shadow_far,
        );
        const light_view_proj = Mat4.mul(light_proj, light_view);

        // Phase 0: Pre-filter meshes and populate instance buffers using SIMD 4-wide batching
        for (self.meshes.items) |mesh| {
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

                    const m0 = inst0.getWorldMatrix();
                    const m1 = inst1.getWorldMatrix();
                    const m2 = inst2.getWorldMatrix();
                    const m3 = inst3.getWorldMatrix();

                    const b0 = mesh.local_bounding_box.transform(m0);
                    const b1 = mesh.local_bounding_box.transform(m1);
                    const b2 = mesh.local_bounding_box.transform(m2);
                    const b3 = mesh.local_bounding_box.transform(m3);

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

                        if (inst0.is_visible and (inst0.culling_strategy != .frustum or vis[0])) {
                            self.stats.rendered_meshes += 1;
                            self.instance_matrices.append(self.allocator, m0) catch {};
                        } else {
                            self.stats.culled_meshes += 1;
                        }

                        if (inst1.is_visible and (inst1.culling_strategy != .frustum or vis[1])) {
                            self.stats.rendered_meshes += 1;
                            self.instance_matrices.append(self.allocator, m1) catch {};
                        } else {
                            self.stats.culled_meshes += 1;
                        }

                        if (inst2.is_visible and (inst2.culling_strategy != .frustum or vis[2])) {
                            self.stats.rendered_meshes += 1;
                            self.instance_matrices.append(self.allocator, m2) catch {};
                        } else {
                            self.stats.culled_meshes += 1;
                        }

                        if (inst3.is_visible and (inst3.culling_strategy != .frustum or vis[3])) {
                            self.stats.rendered_meshes += 1;
                            self.instance_matrices.append(self.allocator, m3) catch {};
                        } else {
                            self.stats.culled_meshes += 1;
                        }
                    } else {
                        if (inst0.is_visible) { self.stats.rendered_meshes += 1; self.instance_matrices.append(self.allocator, m0) catch {}; }
                        if (inst1.is_visible) { self.stats.rendered_meshes += 1; self.instance_matrices.append(self.allocator, m1) catch {}; }
                        if (inst2.is_visible) { self.stats.rendered_meshes += 1; self.instance_matrices.append(self.allocator, m2) catch {}; }
                        if (inst3.is_visible) { self.stats.rendered_meshes += 1; self.instance_matrices.append(self.allocator, m3) catch {}; }
                    }
                }

                // Remainder instances
                while (inst_idx < total_insts) : (inst_idx += 1) {
                    const inst = mesh.instances.items[inst_idx];
                    self.stats.total_meshes += 1;
                    if (!inst.is_visible) continue;
                    const m = inst.getWorldMatrix();
                    if (self.enable_frustum_culling and inst.culling_strategy == .frustum) {
                        const world_aabb = mesh.local_bounding_box.transform(m);
                        if (!frustum.intersectsAABB(world_aabb)) {
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
                    }
                    sg.updateBuffer(mesh.instance_buffer, sg.asRange(self.instance_matrices.items));
                }
            } else {
                self.stats.total_meshes += 1;
                if (!mesh.is_visible) continue;

                const model = mesh.getWorldMatrix();
                const world_aabb = mesh.local_bounding_box.transform(model);

                if (self.enable_frustum_culling and mesh.culling_strategy == .frustum) {
                    if (!frustum.intersectsAABB(world_aabb)) {
                        self.stats.culled_meshes += 1;
                        continue;
                    }
                }

                self.stats.rendered_meshes += 1;

                const is_pbr = if (mesh.material) |m| (m == .pbr) else false;
                const tex_id: u32 = if (mesh.material) |m| switch (m) {
                    .pbr => |p| if (p.albedo_texture) |t| t.view.id else self.default_white_texture.view.id,
                    .standard => |s| if (s.diffuse_texture) |t| t.view.id else self.default_white_texture.view.id,
                } else self.default_white_texture.view.id;

                const d_sq = world_aabb.center().sub(eye).lengthSq();
                self.render_queue.append(self.allocator, .{
                    .mesh = mesh,
                    .model = model,
                    .distance_sq = d_sq,
                    .is_pbr = is_pbr,
                    .texture_id = tex_id,
                }) catch continue;
            }
        }

        // ==============================================
        // PASS 1: OFFSCREEN SHADOW DEPTH PASS
        // ==============================================
        if (self.enable_shadows) {
            self.shadow_pass.render(self.meshes.items, light_view_proj);
            self.stats.draw_calls += 1;
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
        };

        if (self.post_process.enabled) {
            self.postprocess_pass.resize(cur_w, cur_h);
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

        // State sorting: Group by shader type and textures, Front-to-Back Early-Z
        std.mem.sort(RenderMeshItem, self.render_queue.items, {}, sortRenderItems);

        // Pack Point & Spot Lights
        var light_counts = [4]f32{ 0.0, 0.0, 0.0, 0.0 };
        var point_pos_range = [_][4]f32{ .{ 0.0, 0.0, 0.0, 0.0 } } ** 4;
        var point_color_int = [_][4]f32{ .{ 0.0, 0.0, 0.0, 0.0 } } ** 4;
        var spot_pos_range = [_][4]f32{ .{ 0.0, 0.0, 0.0, 0.0 } } ** 2;
        var spot_dir_inner = [_][4]f32{ .{ 0.0, 0.0, 0.0, 0.0 } } ** 2;
        var spot_color_outer = [_][4]f32{ .{ 0.0, 0.0, 0.0, 0.0 } } ** 2;
        var spot_intensity = [_][4]f32{ .{ 0.0, 0.0, 0.0, 0.0 } } ** 2;

        var num_point: usize = 0;
        for (self.point_lights.items) |pl| {
            if (!pl.is_enabled) continue;
            if (num_point >= 4) break;
            point_pos_range[num_point] = .{ pl.position.x, pl.position.y, pl.position.z, pl.range };
            point_color_int[num_point] = .{ pl.color.r, pl.color.g, pl.color.b, pl.intensity };
            num_point += 1;
        }
        light_counts[0] = @floatFromInt(num_point);

        var num_spot: usize = 0;
        for (self.spot_lights.items) |sl| {
            if (!sl.is_enabled) continue;
            if (num_spot >= 2) break;
            const dir = sl.direction.normalize();
            const cos_inner = @cos(sl.inner_angle_deg * (std.math.pi / 180.0));
            const cos_outer = @cos(sl.outer_angle_deg * (std.math.pi / 180.0));
            spot_pos_range[num_spot] = .{ sl.position.x, sl.position.y, sl.position.z, sl.range };
            spot_dir_inner[num_spot] = .{ dir.x, dir.y, dir.z, cos_inner };
            spot_color_outer[num_spot] = .{ sl.color.r, sl.color.g, sl.color.b, cos_outer };
            spot_intensity[num_spot] = .{ sl.intensity, 0.0, 0.0, 0.0 };
            num_spot += 1;
        }
        light_counts[1] = @floatFromInt(num_spot);

        var current_pipeline_id: u32 = 0;

        // Render Regular Meshes
        for (self.render_queue.items) |item| {
            const mesh = item.mesh;
            const model = item.model;
            const mvp = Mat4.mul(view_proj, model);

            const pip = if (item.is_pbr)
                (if (mesh.index_type == .UINT32) self.pipeline_pbr_u32 else self.pipeline_pbr_u16)
            else
                (if (mesh.index_type == .UINT32) self.pipeline_u32 else self.pipeline_u16);

            if (pip.id != current_pipeline_id) {
                sg.applyPipeline(pip);
                current_pipeline_id = pip.id;
                self.stats.pipeline_switches += 1;
            }

            var bind = sg.Bindings{};
            bind.vertex_buffers[0] = mesh.vertex_buffer;
            bind.index_buffer = mesh.index_buffer;

            if (item.is_pbr) {
                const pbr_mat = if (mesh.material) |m| m.pbr else null;
                const albedo_tex = if (pbr_mat) |p| (p.albedo_texture orelse self.default_white_texture) else self.default_white_texture;
                const normal_tex = if (pbr_mat) |p| (p.normal_texture orelse self.default_normal_texture) else self.default_normal_texture;
                const mr_tex = if (pbr_mat) |p| (p.metallic_roughness_texture orelse self.default_white_texture) else self.default_white_texture;
                const emissive_tex = if (pbr_mat) |p| (p.emissive_texture orelse self.default_white_texture) else self.default_white_texture;
                const occlusion_tex = if (pbr_mat) |p| (p.occlusion_texture orelse self.default_white_texture) else self.default_white_texture;

                bind.views[pbr_shd.VIEW_albedo_tex] = albedo_tex.view;
                bind.views[pbr_shd.VIEW_normal_tex] = normal_tex.view;
                bind.views[pbr_shd.VIEW_metallic_roughness_tex] = mr_tex.view;
                bind.views[pbr_shd.VIEW_emissive_tex] = emissive_tex.view;
                bind.views[pbr_shd.VIEW_occlusion_tex] = occlusion_tex.view;
                bind.samplers[pbr_shd.SMP_smp] = albedo_tex.sampler;

                // Environment IBL Cubemap & Shadow Depth Map
                const cube = self.skybox_texture orelse self.default_cube_texture;
                bind.views[pbr_shd.VIEW_env_tex] = cube.view;
                bind.samplers[pbr_shd.SMP_env_smp] = cube.sampler;

                bind.views[pbr_shd.VIEW_shadow_tex] = self.shadow_pass.texture_view;
                bind.samplers[pbr_shd.SMP_shadow_smp] = self.shadow_pass.sampler;

                sg.applyBindings(bind);

                const vs_params = pbr_shd.VsParams{
                    .mvp = mvp,
                    .model = model,
                    .light_view_proj = light_view_proj,
                };
                sg.applyUniforms(pbr_shd.UB_vs_params, sg.asRange(&vs_params));

                const mat_albedo = if (pbr_mat) |p| p.getAlbedoColor4() else [4]f32{ 1, 1, 1, 1 };
                const metallic = if (pbr_mat) |p| p.metallic else 0.0;
                const roughness = if (pbr_mat) |p| p.roughness else 0.5;
                const emissive_col = if (pbr_mat) |p| [4]f32{ p.emissive_color.r, p.emissive_color.g, p.emissive_color.b, 1.0 } else [4]f32{ 0, 0, 0, 1 };

                const fs_params = pbr_shd.FsParams{
                    .eye_pos = .{ eye.x, eye.y, eye.z, 1.0 },
                    .light_dir = .{ self.light.direction.x, self.light.direction.y, self.light.direction.z, 0.0 },
                    .light_color = .{ self.light.diffuse.r, self.light.diffuse.g, self.light.diffuse.b, self.light.intensity },
                    .ambient_color = .{ self.light.ground_color.r, self.light.ground_color.g, self.light.ground_color.b, 1.0 },
                    .base_color_factor = mat_albedo,
                    .pbr_factors = .{ metallic, roughness, self.ibl_intensity, 0.0 },
                    .emissive_factor = emissive_col,
                    .shadow_params = .{
                        if (self.enable_shadows and mesh.receive_shadows) self.shadow_bias else 0.0,
                        if (self.enable_shadows and mesh.receive_shadows) self.shadow_intensity else 0.0,
                        0.0,
                        0.0,
                    },
                    .light_counts = light_counts,
                    .point_pos_range = point_pos_range,
                    .point_color_int = point_color_int,
                    .spot_pos_range = spot_pos_range,
                    .spot_dir_inner = spot_dir_inner,
                    .spot_color_outer = spot_color_outer,
                    .spot_intensity = spot_intensity,
                };
                sg.applyUniforms(pbr_shd.UB_fs_params, sg.asRange(&fs_params));
            } else {
                const std_mat = if (mesh.material) |m| m.standard else &self.default_material;
                const tex = if (std_mat.diffuse_texture) |t| t else self.default_white_texture;

                bind.views[shd.VIEW_diffuse_tex] = tex.view;
                bind.samplers[shd.SMP_smp] = tex.sampler;

                bind.views[shd.VIEW_shadow_tex] = self.shadow_pass.texture_view;
                bind.samplers[shd.SMP_shadow_smp] = self.shadow_pass.sampler;

                sg.applyBindings(bind);

                const vs_params = shd.VsParams{
                    .mvp = mvp,
                    .model = model,
                    .light_view_proj = light_view_proj,
                };
                sg.applyUniforms(shd.UB_vs_params, sg.asRange(&vs_params));

                const fs_params = shd.FsParams{
                    .light_dir = .{ self.light.direction.x, self.light.direction.y, self.light.direction.z, 0.0 },
                    .light_color = .{ self.light.diffuse.r, self.light.diffuse.g, self.light.diffuse.b, self.light.intensity },
                    .ambient_color = .{ self.light.ground_color.r, self.light.ground_color.g, self.light.ground_color.b, 1.0 },
                    .diffuse_color = std_mat.getDiffuseColor4(),
                    .shadow_params = .{
                        if (self.enable_shadows and mesh.receive_shadows) self.shadow_bias else 0.0,
                        if (self.enable_shadows and mesh.receive_shadows) self.shadow_intensity else 0.0,
                        0.0,
                        0.0,
                    },
                    .light_counts = light_counts,
                    .point_pos_range = point_pos_range,
                    .point_color_int = point_color_int,
                    .spot_pos_range = spot_pos_range,
                    .spot_dir_inner = spot_dir_inner,
                    .spot_color_outer = spot_color_outer,
                    .spot_intensity = spot_intensity,
                };
                sg.applyUniforms(shd.UB_fs_params, sg.asRange(&fs_params));
            }

            sg.draw(0, mesh.index_count, 1);
            self.stats.draw_calls += 1;
            self.stats.triangles += mesh.index_count / 3;
        }

        // Render Instanced Meshes
        for (self.meshes.items) |mesh| {
            if (mesh.instances.items.len == 0) continue;
            if (mesh.visible_instance_count == 0 or mesh.instance_buffer.id == 0) continue;

            const pip = if (mesh.index_type == .UINT32) self.pipeline_instanced_u32 else self.pipeline_instanced_u16;
            if (pip.id != current_pipeline_id) {
                sg.applyPipeline(pip);
                current_pipeline_id = pip.id;
                self.stats.pipeline_switches += 1;
            }

            var bind = sg.Bindings{};
            bind.vertex_buffers[0] = mesh.vertex_buffer;
            bind.vertex_buffers[1] = mesh.instance_buffer;
            bind.index_buffer = mesh.index_buffer;

            const std_mat = if (mesh.material) |m| switch (m) {
                .standard => |s| s,
                .pbr => &self.default_material,
            } else &self.default_material;
            const tex = if (std_mat.diffuse_texture) |t| t else self.default_white_texture;

            bind.views[inst_shd.VIEW_diffuse_tex] = tex.view;
            bind.samplers[inst_shd.SMP_smp] = tex.sampler;

            bind.views[inst_shd.VIEW_shadow_tex] = self.shadow_pass.texture_view;
            bind.samplers[inst_shd.SMP_shadow_smp] = self.shadow_pass.sampler;

            sg.applyBindings(bind);

            const inst_vs = inst_shd.VsParams{
                .view_proj = view_proj,
                .light_view_proj = light_view_proj,
            };
            sg.applyUniforms(inst_shd.UB_vs_params, sg.asRange(&inst_vs));

            const inst_fs = inst_shd.FsParams{
                .light_dir = .{ self.light.direction.x, self.light.direction.y, self.light.direction.z, 0.0 },
                .light_color = .{ self.light.diffuse.r, self.light.diffuse.g, self.light.diffuse.b, self.light.intensity },
                .ambient_color = .{ self.light.ground_color.r, self.light.ground_color.g, self.light.ground_color.b, 1.0 },
                .diffuse_color = std_mat.getDiffuseColor4(),
                .shadow_params = .{
                    if (self.enable_shadows and mesh.receive_shadows) self.shadow_bias else 0.0,
                    if (self.enable_shadows and mesh.receive_shadows) self.shadow_intensity else 0.0,
                    0.0,
                    0.0,
                },
                .light_counts = light_counts,
                .point_pos_range = point_pos_range,
                .point_color_int = point_color_int,
                .spot_pos_range = spot_pos_range,
                .spot_dir_inner = spot_dir_inner,
                .spot_color_outer = spot_color_outer,
                .spot_intensity = spot_intensity,
            };
            sg.applyUniforms(inst_shd.UB_fs_params, sg.asRange(&inst_fs));

            sg.draw(0, mesh.index_count, mesh.visible_instance_count);
            self.stats.draw_calls += 1;
            self.stats.triangles += (mesh.index_count / 3) * mesh.visible_instance_count;
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

            self.postprocess_pass.render(self.post_process, cur_w, cur_h);
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
            inline for (.{"albedo_texture", "normal_texture", "metallic_roughness_texture", "emissive_texture", "occlusion_texture"}) |field| {
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
            self.allocator.destroy(pl);
        }
        self.point_lights.deinit(self.allocator);

        for (self.spot_lights.items) |sl| {
            self.allocator.destroy(sl);
        }
        self.spot_lights.deinit(self.allocator);

        self.render_queue.deinit(self.allocator);
        self.instance_matrices.deinit(self.allocator);

        self.default_white_texture.deinit();
        self.default_normal_texture.deinit();
        self.default_black_texture.deinit();
        self.default_cube_texture.deinit();
        if (self.skybox_texture) |*c| {
            c.deinit();
        }

        sg.destroyPipeline(self.pipeline_u16);
        sg.destroyPipeline(self.pipeline_u32);
        sg.destroyPipeline(self.pipeline_pbr_u16);
        sg.destroyPipeline(self.pipeline_pbr_u32);
        sg.destroyPipeline(self.pipeline_instanced_u16);
        sg.destroyPipeline(self.pipeline_instanced_u32);

        // Deinitialize passes
        self.shadow_pass.deinit();
        self.skybox_pass.deinit();
        self.particle_pass.deinit();
        self.postprocess_pass.deinit();

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
