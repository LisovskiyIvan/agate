const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const sapp = sokol.app;
const sglue = sokol.glue;
const shd = @import("shader");
const pbr_shd = @import("pbr_shader");
const inst_shd = @import("instanced_shader");

const math = @import("math");
const Mat4 = math.Mat4;
const Color3 = math.Color3;
const Color4 = math.Color4;
const BoundingBox = math.BoundingBox;
const Frustum = math.Frustum;

const ArcRotateCamera = @import("camera.zig").ArcRotateCamera;
const HemisphericLight = @import("lights.zig").HemisphericLight;
const Mesh = @import("mesh.zig").Mesh;
const InstancedMesh = @import("mesh.zig").InstancedMesh;
const StandardMaterial = @import("material.zig").StandardMaterial;
const PBRMaterial = @import("material.zig").PBRMaterial;
const Material = @import("material.zig").Material;
const Texture = @import("texture.zig").Texture;

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
    default_material: StandardMaterial = StandardMaterial.init("default"),
    default_white_texture: Texture,

    enable_frustum_culling: bool = true,
    stats: SceneStats = .{},

    pipeline_u16: sg.Pipeline = .{},
    pipeline_u32: sg.Pipeline = .{},
    pipeline_pbr_u16: sg.Pipeline = .{},
    pipeline_pbr_u32: sg.Pipeline = .{},
    pipeline_instanced_u16: sg.Pipeline = .{},
    pipeline_instanced_u32: sg.Pipeline = .{},
    pass_action: sg.PassAction = .{},

    render_queue: std.ArrayListUnmanaged(RenderMeshItem) = .empty,
    instance_matrices: std.ArrayListUnmanaged(Mat4) = .empty,

    pub fn init(allocator: std.mem.Allocator) Scene {
        var self = Scene{
            .allocator = allocator,
            .default_white_texture = Texture.createWhite1x1(),
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
        // 1. Standard pipelines
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

        pip_desc.layout.attrs[shd.ATTR_standard_position] = .{ .format = .FLOAT3 };
        pip_desc.layout.attrs[shd.ATTR_standard_normal] = .{ .format = .FLOAT3 };
        pip_desc.layout.attrs[shd.ATTR_standard_color0] = .{ .format = .FLOAT4 };
        pip_desc.layout.attrs[shd.ATTR_standard_texcoord0] = .{ .format = .FLOAT2 };

        self.pipeline_u16 = sg.makePipeline(pip_desc);

        pip_desc.index_type = .UINT32;
        self.pipeline_u32 = sg.makePipeline(pip_desc);

        // 2. PBR pipelines
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

        pbr_desc.layout.attrs[pbr_shd.ATTR_pbr_position] = .{ .format = .FLOAT3 };
        pbr_desc.layout.attrs[pbr_shd.ATTR_pbr_normal] = .{ .format = .FLOAT3 };
        pbr_desc.layout.attrs[pbr_shd.ATTR_pbr_color0] = .{ .format = .FLOAT4 };
        pbr_desc.layout.attrs[pbr_shd.ATTR_pbr_texcoord0] = .{ .format = .FLOAT2 };

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

        // Buffer 0: Per-vertex attributes
        inst_desc.layout.buffers[0] = .{};
        inst_desc.layout.attrs[inst_shd.ATTR_instanced_position] = .{ .buffer_index = 0, .format = .FLOAT3 };
        inst_desc.layout.attrs[inst_shd.ATTR_instanced_normal] = .{ .buffer_index = 0, .format = .FLOAT3 };
        inst_desc.layout.attrs[inst_shd.ATTR_instanced_color0] = .{ .buffer_index = 0, .format = .FLOAT4 };
        inst_desc.layout.attrs[inst_shd.ATTR_instanced_texcoord0] = .{ .buffer_index = 0, .format = .FLOAT2 };

        // Buffer 1: Per-instance 4x4 matrix
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

    pub fn createHemisphericLight(self: *Scene, name: []const u8, options: @import("lights.zig").HemisphericLightOptions) HemisphericLight {
        const light = HemisphericLight.init(name, options);
        self.light = light;
        return light;
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

    pub fn handleEvent(self: *Scene, ev: [*c]const sapp.Event) void {
        if (self.active_camera) |*cam| {
            cam.handleEvent(ev);
        }
    }

    fn sortRenderItems(_: void, a: RenderMeshItem, b: RenderMeshItem) bool {
        // 1. Group by shader type (Standard first, then PBR)
        if (a.is_pbr != b.is_pbr) {
            return !a.is_pbr;
        }
        // 2. Group by texture ID to minimize texture binding switches
        if (a.texture_id != b.texture_id) {
            return a.texture_id < b.texture_id;
        }
        // 3. Front-to-back distance for Early-Z hardware rejection
        return a.distance_sq < b.distance_sq;
    }

    pub fn render(self: *Scene) void {
        self.stats = .{};
        self.render_queue.clearRetainingCapacity();

        self.pass_action.colors[0] = .{
            .load_action = .CLEAR,
            .clear_value = .{
                .r = self.clear_color.r,
                .g = self.clear_color.g,
                .b = self.clear_color.b,
                .a = self.clear_color.a,
            },
        };

        const aspect = sapp.widthf() / sapp.heightf();
        const view_proj = if (self.active_camera) |cam|
            cam.getViewProjection(aspect)
        else
            Mat4.perspective(60.0, aspect, 0.1, 100.0);

        const eye = if (self.active_camera) |cam| cam.getPosition() else math.Vec3.new(0, 0, 5);
        const frustum = Frustum.fromViewProjection(view_proj);

        sg.beginPass(.{
            .action = self.pass_action,
            .swapchain = sglue.swapchain(),
        });

        var current_pipeline: sg.Pipeline = .{};

        // Phase 1: Process all meshes (Instanced meshes drawn directly; standard meshes queued & sorted)
        for (self.meshes.items) |mesh| {
            if (mesh.instances.items.len > 0) {
                // Instanced mesh rendering
                self.instance_matrices.clearRetainingCapacity();
                for (mesh.instances.items) |inst| {
                    self.stats.total_meshes += 1;
                    if (!inst.is_visible) continue;
                    if (self.enable_frustum_culling and inst.culling_strategy == .frustum) {
                        if (!frustum.intersectsAABB(inst.getWorldBoundingBox())) {
                            self.stats.culled_meshes += 1;
                            continue;
                        }
                    }
                    self.stats.rendered_meshes += 1;
                    self.instance_matrices.append(self.allocator, inst.getWorldMatrix()) catch continue;
                }

                const visible_count = self.instance_matrices.items.len;
                if (visible_count == 0) continue;

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

                const pip = if (mesh.index_type == .UINT32) self.pipeline_instanced_u32 else self.pipeline_instanced_u16;
                if (pip.id != current_pipeline.id) {
                    sg.applyPipeline(pip);
                    current_pipeline = pip;
                    self.stats.pipeline_switches += 1;
                }

                const vs_params = inst_shd.VsParams{ .view_proj = view_proj };
                const mat = if (mesh.material) |m| switch (m) {
                    .standard => |s| s,
                    else => &self.default_material,
                } else &self.default_material;

                const fs_params = inst_shd.FsParams{
                    .light_dir = .{ self.light.direction.x, self.light.direction.y, self.light.direction.z, 0.0 },
                    .light_color = .{ self.light.diffuse.r, self.light.diffuse.g, self.light.diffuse.b, self.light.intensity },
                    .ambient_color = .{ self.light.ground_color.r, self.light.ground_color.g, self.light.ground_color.b, 1.0 },
                    .diffuse_color = mat.getDiffuseColor4(),
                };

                const tex = if (mat.diffuse_texture) |t| t else self.default_white_texture;
                var bind = sg.Bindings{};
                bind.vertex_buffers[0] = mesh.vertex_buffer;
                bind.vertex_buffers[1] = mesh.instance_buffer;
                bind.index_buffer = mesh.index_buffer;
                bind.views[inst_shd.VIEW_diffuse_tex] = tex.view;
                bind.samplers[inst_shd.SMP_smp] = tex.sampler;

                sg.applyBindings(bind);
                sg.applyUniforms(inst_shd.UB_vs_params, sg.asRange(&vs_params));
                sg.applyUniforms(inst_shd.UB_fs_params, sg.asRange(&fs_params));
                sg.draw(0, mesh.index_count, @intCast(visible_count));

                self.stats.draw_calls += 1;
                self.stats.triangles += (mesh.index_count / 3) * @as(u32, @intCast(visible_count));
            } else {
                // Non-instanced mesh
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

        // Phase 2: State Sorting (Material & Early-Z front-to-back)
        std.mem.sort(RenderMeshItem, self.render_queue.items, {}, sortRenderItems);

        // Phase 3: Render sorted non-instanced meshes
        for (self.render_queue.items) |item| {
            const mesh = item.mesh;
            const model = item.model;
            const mvp = Mat4.mul(view_proj, model);

            var bind = sg.Bindings{};
            bind.vertex_buffers[0] = mesh.vertex_buffer;
            bind.index_buffer = mesh.index_buffer;

            if (item.is_pbr) {
                const pbr_mat = mesh.material.?.pbr;
                const pip = if (mesh.index_type == .UINT32) self.pipeline_pbr_u32 else self.pipeline_pbr_u16;
                if (pip.id != current_pipeline.id) {
                    sg.applyPipeline(pip);
                    current_pipeline = pip;
                    self.stats.pipeline_switches += 1;
                }

                const vs_params = pbr_shd.VsParams{
                    .mvp = mvp,
                    .model = model,
                };

                const fs_params = pbr_shd.FsParams{
                    .eye_pos = .{ eye.x, eye.y, eye.z, 1.0 },
                    .light_dir = .{ self.light.direction.x, self.light.direction.y, self.light.direction.z, 0.0 },
                    .light_color = .{ self.light.diffuse.r, self.light.diffuse.g, self.light.diffuse.b, self.light.intensity },
                    .ambient_color = .{ self.light.ground_color.r, self.light.ground_color.g, self.light.ground_color.b, 1.0 },
                    .base_color_factor = pbr_mat.getAlbedoColor4(),
                    .pbr_factors = .{ pbr_mat.metallic, pbr_mat.roughness, 1.0, 0.0 },
                };

                const tex = if (pbr_mat.albedo_texture) |t| t else self.default_white_texture;
                bind.views[pbr_shd.VIEW_albedo_tex] = tex.view;
                bind.samplers[pbr_shd.SMP_smp] = tex.sampler;

                sg.applyBindings(bind);
                sg.applyUniforms(pbr_shd.UB_vs_params, sg.asRange(&vs_params));
                sg.applyUniforms(pbr_shd.UB_fs_params, sg.asRange(&fs_params));
                sg.draw(0, mesh.index_count, 1);

                self.stats.draw_calls += 1;
                self.stats.triangles += mesh.index_count / 3;
            } else {
                // Standard pipeline
                const pip = if (mesh.index_type == .UINT32) self.pipeline_u32 else self.pipeline_u16;
                if (pip.id != current_pipeline.id) {
                    sg.applyPipeline(pip);
                    current_pipeline = pip;
                    self.stats.pipeline_switches += 1;
                }

                const vs_params = shd.VsParams{
                    .mvp = mvp,
                    .model = model,
                };

                const mat = if (mesh.material) |m| switch (m) {
                    .standard => |s| s,
                    else => &self.default_material,
                } else &self.default_material;

                const fs_params = shd.FsParams{
                    .light_dir = .{ self.light.direction.x, self.light.direction.y, self.light.direction.z, 0.0 },
                    .light_color = .{ self.light.diffuse.r, self.light.diffuse.g, self.light.diffuse.b, self.light.intensity },
                    .ambient_color = .{ self.light.ground_color.r, self.light.ground_color.g, self.light.ground_color.b, 1.0 },
                    .diffuse_color = mat.getDiffuseColor4(),
                };

                const tex = if (mat.diffuse_texture) |t| t else self.default_white_texture;
                bind.views[shd.VIEW_diffuse_tex] = tex.view;
                bind.samplers[shd.SMP_smp] = tex.sampler;

                sg.applyBindings(bind);
                sg.applyUniforms(shd.UB_vs_params, sg.asRange(&vs_params));
                sg.applyUniforms(shd.UB_fs_params, sg.asRange(&fs_params));
                sg.draw(0, mesh.index_count, 1);

                self.stats.draw_calls += 1;
                self.stats.triangles += mesh.index_count / 3;
            }
        }

        sg.endPass();
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

        for (self.pbr_materials.items) |mat| {
            if (mat.albedo_texture) |*t| {
                t.deinit();
            }
            self.allocator.destroy(mat);
        }
        self.pbr_materials.deinit(self.allocator);

        self.render_queue.deinit(self.allocator);
        self.instance_matrices.deinit(self.allocator);

        self.default_white_texture.deinit();
        sg.destroyPipeline(self.pipeline_u16);
        sg.destroyPipeline(self.pipeline_u32);
        sg.destroyPipeline(self.pipeline_pbr_u16);
        sg.destroyPipeline(self.pipeline_pbr_u32);
        sg.destroyPipeline(self.pipeline_instanced_u16);
        sg.destroyPipeline(self.pipeline_instanced_u32);
    }
};
