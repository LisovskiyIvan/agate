const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const sapp = sokol.app;
const sglue = sokol.glue;
const shd = @import("shader");
const pbr_shd = @import("pbr_shader");
const inst_shd = @import("instanced_shader");
const shadow_shd = @import("shadow_shader");
const sky_shd = @import("skybox_shader");
const post_shd = @import("postprocess_shader");
const part_shd = @import("particle_shader");
const postprocess = @import("postprocess.zig");
pub const PostProcessConfig = postprocess.PostProcessConfig;
pub const TonemappingType = postprocess.TonemappingType;
const particles = @import("particles.zig");
pub const ParticleSystem = particles.ParticleSystem;
pub const ParticleBlendMode = particles.ParticleBlendMode;
pub const Particle = particles.Particle;

const math = @import("math");
const Mat4 = math.Mat4;
const Color3 = math.Color3;
const Color4 = math.Color4;
const BoundingBox = math.BoundingBox;
const Frustum = math.Frustum;

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

    enable_frustum_culling: bool = true,
    stats: SceneStats = .{},

    // Shadow Mapping
    enable_shadows: bool = true,
    shadow_bias: f32 = 0.003,
    shadow_intensity: f32 = 0.75,
    shadow_extent: f32 = 25.0,
    shadow_near: f32 = 0.5,
    shadow_far: f32 = 80.0,
    shadow_image: sg.Image = .{},
    shadow_attachment_view: sg.View = .{},
    shadow_texture_view: sg.View = .{},
    shadow_sampler: sg.Sampler = .{},
    shadow_pipeline_u16: sg.Pipeline = .{},
    shadow_pipeline_u32: sg.Pipeline = .{},
    shadow_inst_pipeline_u16: sg.Pipeline = .{},
    shadow_inst_pipeline_u32: sg.Pipeline = .{},

    // Skybox & IBL
    default_cube_texture: CubeTexture,
    skybox_texture: ?CubeTexture = null,
    skybox_pipeline: sg.Pipeline = .{},
    skybox_mesh_vb: sg.Buffer = .{},
    skybox_mesh_ib: sg.Buffer = .{},
    skybox_enabled: bool = false,
    skybox_exposure: f32 = 1.0,
    ibl_intensity: f32 = 1.0,

    pipeline_u16: sg.Pipeline = .{},
    pipeline_u32: sg.Pipeline = .{},
    pipeline_pbr_u16: sg.Pipeline = .{},
    pipeline_pbr_u32: sg.Pipeline = .{},
    pipeline_instanced_u16: sg.Pipeline = .{},
    pipeline_instanced_u32: sg.Pipeline = .{},
    pass_action: sg.PassAction = .{},

    // Post-Processing Pipeline
    post_process: PostProcessConfig = .{},
    postprocess_pipeline: sg.Pipeline = .{},
    postprocess_sampler: sg.Sampler = .{},
    postprocess_quad_vb: sg.Buffer = .{},
    postprocess_quad_ib: sg.Buffer = .{},

    // Offscreen Framebuffer for Post-Processing (supports both MSAA and non-MSAA)
    offscreen_width: i32 = 0,
    offscreen_height: i32 = 0,
    offscreen_sample_count: i32 = 1,
    offscreen_color_image: sg.Image = .{},
    offscreen_color_att_view: sg.View = .{},
    offscreen_resolve_image: sg.Image = .{},
    offscreen_resolve_att_view: sg.View = .{},
    offscreen_resolve_tex_view: sg.View = .{},
    offscreen_depth_image: sg.Image = .{},
    offscreen_depth_att_view: sg.View = .{},

    // Particle Systems
    particle_systems: std.ArrayListUnmanaged(*particles.ParticleSystem) = .empty,
    default_particle_texture: Texture,
    particle_quad_vb: sg.Buffer = .{},
    particle_quad_ib: sg.Buffer = .{},
    particle_pipeline_additive: sg.Pipeline = .{},
    particle_pipeline_alphablend: sg.Pipeline = .{},

    render_queue: std.ArrayListUnmanaged(RenderMeshItem) = .empty,
    instance_matrices: std.ArrayListUnmanaged(Mat4) = .empty,

    pub fn init(allocator: std.mem.Allocator) Scene {
        const depth_img = sg.makeImage(.{
            .usage = .{ .depth_stencil_attachment = true },
            .width = 2048,
            .height = 2048,
            .pixel_format = .DEPTH,
            .sample_count = 1,
        });
        const att_view = sg.makeView(.{
            .depth_stencil_attachment = .{ .image = depth_img },
        });
        const tex_view = sg.makeView(.{
            .texture = .{ .image = depth_img },
        });
        const smp = sg.makeSampler(.{
            .min_filter = .LINEAR,
            .mag_filter = .LINEAR,
            .wrap_u = .CLAMP_TO_EDGE,
            .wrap_v = .CLAMP_TO_EDGE,
            .compare = .LESS_EQUAL,
        });

        var self = Scene{
            .allocator = allocator,
            .default_white_texture = Texture.createWhite1x1(),
            .default_normal_texture = Texture.createFlatNormal1x1(),
            .default_black_texture = Texture.createBlack1x1(),
            .default_cube_texture = CubeTexture.createDefault1x1(.{ 25, 30, 40, 255 }),
            .default_particle_texture = Texture.createDefaultParticleDot32(),
            .shadow_image = depth_img,
            .shadow_attachment_view = att_view,
            .shadow_texture_view = tex_view,
            .shadow_sampler = smp,
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

        pip_desc.layout.buffers[0] = .{ .stride = @sizeOf(Vertex) };
        pip_desc.layout.attrs[shd.ATTR_standard_position] = .{ .format = .FLOAT3, .offset = @offsetOf(Vertex, "position") };
        pip_desc.layout.attrs[shd.ATTR_standard_normal] = .{ .format = .FLOAT3, .offset = @offsetOf(Vertex, "normal") };
        pip_desc.layout.attrs[shd.ATTR_standard_color0] = .{ .format = .FLOAT4, .offset = @offsetOf(Vertex, "color") };
        pip_desc.layout.attrs[shd.ATTR_standard_texcoord0] = .{ .format = .FLOAT2, .offset = @offsetOf(Vertex, "uv") };

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

        // Buffer 0: Per-vertex attributes
        inst_desc.layout.buffers[0] = .{ .stride = @sizeOf(Vertex) };
        inst_desc.layout.attrs[inst_shd.ATTR_instanced_position] = .{ .buffer_index = 0, .format = .FLOAT3, .offset = @offsetOf(Vertex, "position") };
        inst_desc.layout.attrs[inst_shd.ATTR_instanced_normal] = .{ .buffer_index = 0, .format = .FLOAT3, .offset = @offsetOf(Vertex, "normal") };
        inst_desc.layout.attrs[inst_shd.ATTR_instanced_color0] = .{ .buffer_index = 0, .format = .FLOAT4, .offset = @offsetOf(Vertex, "color") };
        inst_desc.layout.attrs[inst_shd.ATTR_instanced_texcoord0] = .{ .buffer_index = 0, .format = .FLOAT2, .offset = @offsetOf(Vertex, "uv") };

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

        // 4. Shadow Depth pipelines (regular meshes)
        var shadow_pip_desc = sg.PipelineDesc{
            .shader = sg.makeShader(shadow_shd.shadowShaderDesc(sg.queryBackend())),
            .index_type = .UINT16,
            .sample_count = 1,
            .depth = .{
                .pixel_format = .DEPTH,
                .compare = .LESS_EQUAL,
                .write_enabled = true,
                .bias = 2.0,
                .bias_slope_scale = 2.0,
            },
            .cull_mode = .BACK,
            .face_winding = .CCW,
        };
        shadow_pip_desc.colors[0].pixel_format = .NONE;
        shadow_pip_desc.layout.buffers[0] = .{ .stride = @sizeOf(Vertex) };
        shadow_pip_desc.layout.attrs[shadow_shd.ATTR_shadow_position] = .{
            .format = .FLOAT3,
            .offset = @offsetOf(Vertex, "position"),
        };

        self.shadow_pipeline_u16 = sg.makePipeline(shadow_pip_desc);
        shadow_pip_desc.index_type = .UINT32;
        self.shadow_pipeline_u32 = sg.makePipeline(shadow_pip_desc);

        // 5. Shadow Depth pipelines (instanced meshes)
        var shadow_inst_desc = sg.PipelineDesc{
            .shader = sg.makeShader(shadow_shd.shadowInstancedShaderDesc(sg.queryBackend())),
            .index_type = .UINT16,
            .sample_count = 1,
            .depth = .{
                .pixel_format = .DEPTH,
                .compare = .LESS_EQUAL,
                .write_enabled = true,
                .bias = 2.0,
                .bias_slope_scale = 2.0,
            },
            .cull_mode = .BACK,
            .face_winding = .CCW,
        };
        shadow_inst_desc.colors[0].pixel_format = .NONE;
        shadow_inst_desc.layout.buffers[0] = .{ .stride = @sizeOf(Vertex) };
        shadow_inst_desc.layout.attrs[shadow_shd.ATTR_shadow_instanced_position] = .{
            .buffer_index = 0,
            .format = .FLOAT3,
            .offset = @offsetOf(Vertex, "position"),
        };
        shadow_inst_desc.layout.buffers[1] = .{
            .step_func = .PER_INSTANCE,
            .step_rate = 1,
            .stride = @sizeOf(Mat4),
        };
        shadow_inst_desc.layout.attrs[shadow_shd.ATTR_shadow_instanced_inst_mat0] = .{ .buffer_index = 1, .offset = 0, .format = .FLOAT4 };
        shadow_inst_desc.layout.attrs[shadow_shd.ATTR_shadow_instanced_inst_mat1] = .{ .buffer_index = 1, .offset = 16, .format = .FLOAT4 };
        shadow_inst_desc.layout.attrs[shadow_shd.ATTR_shadow_instanced_inst_mat2] = .{ .buffer_index = 1, .offset = 32, .format = .FLOAT4 };
        shadow_inst_desc.layout.attrs[shadow_shd.ATTR_shadow_instanced_inst_mat3] = .{ .buffer_index = 1, .offset = 48, .format = .FLOAT4 };

        self.shadow_inst_pipeline_u16 = sg.makePipeline(shadow_inst_desc);
        shadow_inst_desc.index_type = .UINT32;
        self.shadow_inst_pipeline_u32 = sg.makePipeline(shadow_inst_desc);

        // 5. Skybox pipeline & cube geometry
        const skybox_positions = [_][3]f32{
            .{ -1.0, -1.0, -1.0 }, // 0
            .{  1.0, -1.0, -1.0 }, // 1
            .{  1.0,  1.0, -1.0 }, // 2
            .{ -1.0,  1.0, -1.0 }, // 3
            .{ -1.0, -1.0,  1.0 }, // 4
            .{  1.0, -1.0,  1.0 }, // 5
            .{  1.0,  1.0,  1.0 }, // 6
            .{ -1.0,  1.0,  1.0 }, // 7
        };

        const skybox_indices = [_]u16{
            // Front (-Z)
            0, 2, 1,  0, 3, 2,
            // Back (+Z)
            4, 5, 6,  4, 6, 7,
            // Left (-X)
            0, 4, 7,  0, 7, 3,
            // Right (+X)
            1, 2, 6,  1, 6, 5,
            // Top (+Y)
            3, 7, 6,  3, 6, 2,
            // Bottom (-Y)
            0, 1, 5,  0, 5, 4,
        };

        self.skybox_mesh_vb = sg.makeBuffer(.{
            .data = sg.asRange(&skybox_positions),
        });
        self.skybox_mesh_ib = sg.makeBuffer(.{
            .usage = .{ .index_buffer = true },
            .data = sg.asRange(&skybox_indices),
        });

        var sky_desc = sg.PipelineDesc{
            .shader = sg.makeShader(sky_shd.skyboxShaderDesc(sg.queryBackend())),
            .index_type = .UINT16,
            .depth = .{
                .compare = .LESS_EQUAL,
                .write_enabled = false,
            },
            .cull_mode = .NONE,
        };
        sky_desc.layout.buffers[0] = .{ .stride = @sizeOf([3]f32) };
        sky_desc.layout.attrs[sky_shd.ATTR_skybox_position] = .{
            .format = .FLOAT3,
            .offset = 0,
        };
        self.skybox_pipeline = sg.makePipeline(sky_desc);

        // 6. Fullscreen Post-Processing Pipeline
        const quad_vertices = [_]f32{
            // pos.x, pos.y, uv.x, uv.y
            -1.0, -1.0, 0.0, 0.0,
             1.0, -1.0, 1.0, 0.0,
             1.0,  1.0, 1.0, 1.0,
            -1.0,  1.0, 0.0, 1.0,
        };
        const quad_indices = [_]u16{
            0, 1, 2,
            0, 2, 3,
        };
        self.postprocess_quad_vb = sg.makeBuffer(.{
            .data = sg.asRange(&quad_vertices),
        });
        self.postprocess_quad_ib = sg.makeBuffer(.{
            .usage = .{ .index_buffer = true },
            .data = sg.asRange(&quad_indices),
        });
        self.postprocess_sampler = sg.makeSampler(.{
            .min_filter = .LINEAR,
            .mag_filter = .LINEAR,
            .wrap_u = .CLAMP_TO_EDGE,
            .wrap_v = .CLAMP_TO_EDGE,
        });

        var pp_desc = sg.PipelineDesc{
            .shader = sg.makeShader(post_shd.postprocessShaderDesc(sg.queryBackend())),
            .index_type = .UINT16,
            .depth = .{
                .compare = .ALWAYS,
                .write_enabled = false,
            },
            .cull_mode = .NONE,
        };
        pp_desc.layout.buffers[0] = .{ .stride = 4 * @sizeOf(f32) };
        pp_desc.layout.attrs[post_shd.ATTR_postprocess_position] = .{
            .format = .FLOAT2,
            .offset = 0,
        };
        pp_desc.layout.attrs[post_shd.ATTR_postprocess_texcoord0] = .{
            .format = .FLOAT2,
            .offset = 2 * @sizeOf(f32),
        };
        self.postprocess_pipeline = sg.makePipeline(pp_desc);

        // 7. Particle Pipelines & Billboard Quad
        const particle_quad_vertices = [_]f32{
            // x,     y,     u,   v
            -0.5, -0.5,  0.0, 0.0,
             0.5, -0.5,  1.0, 0.0,
             0.5,  0.5,  1.0, 1.0,
            -0.5,  0.5,  0.0, 1.0,
        };
        const particle_quad_indices = [_]u16{
            0, 1, 2,
            0, 2, 3,
        };
        self.particle_quad_vb = sg.makeBuffer(.{
            .data = sg.asRange(&particle_quad_vertices),
        });
        self.particle_quad_ib = sg.makeBuffer(.{
            .usage = .{ .index_buffer = true },
            .data = sg.asRange(&particle_quad_indices),
        });

        var part_desc = sg.PipelineDesc{
            .shader = sg.makeShader(part_shd.particleShaderDesc(sg.queryBackend())),
            .index_type = .UINT16,
            .depth = .{
                .compare = .LESS_EQUAL,
                .write_enabled = false,
            },
            .cull_mode = .NONE,
        };
        // Buffer 0: Unit quad
        part_desc.layout.buffers[0] = .{ .stride = 4 * @sizeOf(f32) };
        part_desc.layout.attrs[part_shd.ATTR_particle_position] = .{
            .buffer_index = 0,
            .format = .FLOAT2,
            .offset = 0,
        };
        part_desc.layout.attrs[part_shd.ATTR_particle_texcoord0] = .{
            .buffer_index = 0,
            .format = .FLOAT2,
            .offset = 2 * @sizeOf(f32),
        };

        // Buffer 1: Dynamic per-instance data
        part_desc.layout.buffers[1] = .{
            .step_func = .PER_INSTANCE,
            .step_rate = 1,
            .stride = @sizeOf(particles.ParticleInstanceData),
        };
        part_desc.layout.attrs[part_shd.ATTR_particle_inst_pos_size] = .{
            .buffer_index = 1,
            .format = .FLOAT4,
            .offset = 0,
        };
        part_desc.layout.attrs[part_shd.ATTR_particle_inst_color] = .{
            .buffer_index = 1,
            .format = .FLOAT4,
            .offset = 4 * @sizeOf(f32),
        };

        // Additive pipeline
        part_desc.colors[0].blend = .{
            .enabled = true,
            .src_factor_rgb = .SRC_ALPHA,
            .dst_factor_rgb = .ONE,
            .src_factor_alpha = .ONE,
            .dst_factor_alpha = .ONE,
        };
        self.particle_pipeline_additive = sg.makePipeline(part_desc);

        // AlphaBlend pipeline
        part_desc.colors[0].blend = .{
            .enabled = true,
            .src_factor_rgb = .SRC_ALPHA,
            .dst_factor_rgb = .ONE_MINUS_SRC_ALPHA,
            .src_factor_alpha = .ONE,
            .dst_factor_alpha = .ONE_MINUS_SRC_ALPHA,
        };
        self.particle_pipeline_alphablend = sg.makePipeline(part_desc);
    }

    pub fn resizeOffscreen(self: *Scene, width: i32, height: i32) void {
        if (width <= 0 or height <= 0) return;
        if (self.offscreen_color_image.id != 0) {
            sg.destroyImage(self.offscreen_color_image);
            sg.destroyView(self.offscreen_color_att_view);
            if (self.offscreen_resolve_image.id != 0) {
                sg.destroyImage(self.offscreen_resolve_image);
                sg.destroyView(self.offscreen_resolve_att_view);
                sg.destroyView(self.offscreen_resolve_tex_view);
            } else {
                sg.destroyView(self.offscreen_resolve_tex_view);
            }
            sg.destroyImage(self.offscreen_depth_image);
            sg.destroyView(self.offscreen_depth_att_view);
            self.offscreen_resolve_image = .{};
            self.offscreen_resolve_att_view = .{};
            self.offscreen_resolve_tex_view = .{};
        }

        const sw = sglue.swapchain();
        const color_fmt: sg.PixelFormat = if (sw.color_format != .DEFAULT and sw.color_format != .NONE) sw.color_format else .BGRA8;
        const depth_fmt: sg.PixelFormat = if (sw.depth_format != .DEFAULT and sw.depth_format != .NONE) sw.depth_format else .DEPTH_STENCIL;
        const sample_count: i32 = if (sw.sample_count > 1) sw.sample_count else 1;

        const col_img = sg.makeImage(.{
            .usage = .{ .color_attachment = true },
            .width = width,
            .height = height,
            .pixel_format = color_fmt,
            .sample_count = sample_count,
        });
        const col_att = sg.makeView(.{
            .color_attachment = .{ .image = col_img },
        });

        if (sample_count > 1) {
            const res_img = sg.makeImage(.{
                .usage = .{ .resolve_attachment = true },
                .width = width,
                .height = height,
                .pixel_format = color_fmt,
                .sample_count = 1,
            });
            const res_att = sg.makeView(.{
                .resolve_attachment = .{ .image = res_img },
            });
            const res_tex = sg.makeView(.{
                .texture = .{ .image = res_img },
            });
            self.offscreen_resolve_image = res_img;
            self.offscreen_resolve_att_view = res_att;
            self.offscreen_resolve_tex_view = res_tex;
        } else {
            const col_tex = sg.makeView(.{
                .texture = .{ .image = col_img },
            });
            self.offscreen_resolve_tex_view = col_tex;
        }

        const depth_img = sg.makeImage(.{
            .usage = .{ .depth_stencil_attachment = true },
            .width = width,
            .height = height,
            .pixel_format = depth_fmt,
            .sample_count = sample_count,
        });
        const depth_att = sg.makeView(.{
            .depth_stencil_attachment = .{ .image = depth_img },
        });

        self.offscreen_width = width;
        self.offscreen_height = height;
        self.offscreen_sample_count = sample_count;
        self.offscreen_color_image = col_img;
        self.offscreen_color_att_view = col_att;
        self.offscreen_depth_image = depth_img;
        self.offscreen_depth_att_view = depth_att;
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
        const light = HemisphericLight.init(name, options);
        self.light = light;
        return light;
    }

    pub fn createPointLight(self: *Scene, name: []const u8, options: PointLightOptions) !*PointLight {
        const light = try self.allocator.create(PointLight);
        light.* = PointLight.init(name, options);
        try self.point_lights.append(self.allocator, light);
        return light;
    }

    pub fn createSpotLight(self: *Scene, name: []const u8, options: SpotLightOptions) !*SpotLight {
        const light = try self.allocator.create(SpotLight);
        light.* = SpotLight.init(name, options);
        try self.spot_lights.append(self.allocator, light);
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

        const aspect = sapp.widthf() / sapp.heightf();
        const view_proj = if (self.active_camera) |cam|
            cam.getViewProjection(aspect)
        else
            Mat4.perspective(60.0, aspect, 0.1, 100.0);

        const eye = if (self.active_camera) |cam| cam.getPosition() else math.Vec3.new(0, 0, 5);
        const frustum = Frustum.fromViewProjection(view_proj);

        // Directional Light View-Projection Matrix
        const light_dir = self.light.direction.normalize();
        const light_target = if (self.active_camera) |cam| cam.target else math.Vec3.zero;
        const light_pos = light_target.add(light_dir.scale(35.0));
        var light_up = math.Vec3.new(0.0, 1.0, 0.0);
        if (@abs(light_dir.x) < 0.001 and @abs(light_dir.z) < 0.001) {
            light_up = math.Vec3.new(0.0, 0.0, 1.0);
        }
        const light_view = Mat4.lookAt(light_pos, light_target, light_up);
        const light_proj = Mat4.orthographic(
            -self.shadow_extent,
            self.shadow_extent,
            -self.shadow_extent,
            self.shadow_extent,
            self.shadow_near,
            self.shadow_far,
        );
        const light_view_proj = Mat4.mul(light_proj, light_view);

        // Phase 0: Pre-filter meshes and populate instance buffers
        for (self.meshes.items) |mesh| {
            if (mesh.instances.items.len > 0) {
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
            const shadow_pass = sg.Pass{
                .action = .{
                    .depth = .{
                        .load_action = .CLEAR,
                        .store_action = .STORE,
                        .clear_value = 1.0,
                    },
                },
                .attachments = .{
                    .depth_stencil = self.shadow_attachment_view,
                },
            };
            sg.beginPass(shadow_pass);

            var cur_shadow_pip: sg.Pipeline = .{};

            for (self.meshes.items) |mesh| {
                if (!mesh.cast_shadows) continue;

                if (mesh.instances.items.len > 0) {
                    if (mesh.instance_buffer.id == 0 or mesh.visible_instance_count == 0) continue;

                    const pip = if (mesh.index_type == .UINT32) self.shadow_inst_pipeline_u32 else self.shadow_inst_pipeline_u16;
                    if (pip.id != cur_shadow_pip.id) {
                        sg.applyPipeline(pip);
                        cur_shadow_pip = pip;
                    }

                    var bind = sg.Bindings{};
                    bind.vertex_buffers[0] = mesh.vertex_buffer;
                    bind.vertex_buffers[1] = mesh.instance_buffer;
                    bind.index_buffer = mesh.index_buffer;
                    sg.applyBindings(bind);

                    const vs_inst_params = shadow_shd.VsInstParams{
                        .light_view_proj = light_view_proj,
                    };
                    sg.applyUniforms(shadow_shd.UB_vs_inst_params, sg.asRange(&vs_inst_params));
                    sg.draw(0, mesh.index_count, mesh.visible_instance_count);
                } else {
                    const pip = if (mesh.index_type == .UINT32) self.shadow_pipeline_u32 else self.shadow_pipeline_u16;
                    if (pip.id != cur_shadow_pip.id) {
                        sg.applyPipeline(pip);
                        cur_shadow_pip = pip;
                    }

                    const model = mesh.getWorldMatrix();
                    const mvp = Mat4.mul(light_view_proj, model);

                    var bind = sg.Bindings{};
                    bind.vertex_buffers[0] = mesh.vertex_buffer;
                    bind.index_buffer = mesh.index_buffer;
                    sg.applyBindings(bind);

                    const vs_params = shadow_shd.VsParams{
                        .mvp = mvp,
                    };
                    sg.applyUniforms(shadow_shd.UB_vs_params, sg.asRange(&vs_params));
                    sg.draw(0, mesh.index_count, 1);
                }
            }

            sg.endPass();
        }

        // ==============================================
        // PASS 2: MAIN SWAPCHAIN RENDER PASS
        // ==============================================
        self.pass_action.colors[0] = .{
            .load_action = .CLEAR,
            .clear_value = .{
                .r = self.clear_color.r,
                .g = self.clear_color.g,
                .b = self.clear_color.b,
                .a = self.clear_color.a,
            },
        };

        // Pack up to 4 Point Lights and 2 Spot Lights for fragment shaders
        var light_counts = [4]f32{ 0.0, 0.0, 0.0, 0.0 };
        var point_pos_range = [_][4]f32{[_]f32{ 0.0, 0.0, 0.0, 0.0 }} ** 4;
        var point_color_int = [_][4]f32{[_]f32{ 0.0, 0.0, 0.0, 0.0 }} ** 4;
        var spot_pos_range = [_][4]f32{[_]f32{ 0.0, 0.0, 0.0, 0.0 }} ** 2;
        var spot_dir_inner = [_][4]f32{[_]f32{ 0.0, 0.0, 0.0, 0.0 }} ** 2;
        var spot_color_outer = [_][4]f32{[_]f32{ 0.0, 0.0, 0.0, 0.0 }} ** 2;
        var spot_intensity = [_][4]f32{[_]f32{ 0.0, 0.0, 0.0, 0.0 }} ** 2;

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

        // ==============================================
        // PASS 2: MAIN SCENE RENDER PASS (Swapchain or Offscreen)
        // ==============================================
        const cur_w = sapp.width();
        const cur_h = sapp.height();

        if (self.post_process.enabled) {
            if (self.offscreen_color_image.id == 0 or self.offscreen_width != cur_w or self.offscreen_height != cur_h) {
                self.resizeOffscreen(cur_w, cur_h);
            }
            var offscreen_action = sg.PassAction{};
            offscreen_action.colors[0] = .{
                .load_action = .CLEAR,
                .store_action = if (self.offscreen_sample_count > 1) .DONTCARE else .STORE,
                .clear_value = .{
                    .r = self.clear_color.r,
                    .g = self.clear_color.g,
                    .b = self.clear_color.b,
                    .a = self.clear_color.a,
                },
            };
            var offscreen_atts = sg.Attachments{};
            offscreen_atts.colors[0] = self.offscreen_color_att_view;
            if (self.offscreen_sample_count > 1) {
                offscreen_atts.resolves[0] = self.offscreen_resolve_att_view;
            }
            offscreen_atts.depth_stencil = self.offscreen_depth_att_view;

            sg.beginPass(.{
                .action = offscreen_action,
                .attachments = offscreen_atts,
            });
        } else {
            self.pass_action.colors[0] = .{
                .load_action = .CLEAR,
                .clear_value = .{
                    .r = self.clear_color.r,
                    .g = self.clear_color.g,
                    .b = self.clear_color.b,
                    .a = self.clear_color.a,
                },
            };

            sg.beginPass(.{
                .action = self.pass_action,
                .swapchain = sglue.swapchain(),
            });
        }

        var current_pipeline: sg.Pipeline = .{};

        // Render instanced meshes
        for (self.meshes.items) |mesh| {
            if (mesh.instances.items.len == 0 or mesh.visible_instance_count == 0) continue;

            const pip = if (mesh.index_type == .UINT32) self.pipeline_instanced_u32 else self.pipeline_instanced_u16;
            if (pip.id != current_pipeline.id) {
                sg.applyPipeline(pip);
                current_pipeline = pip;
                self.stats.pipeline_switches += 1;
            }

            const vs_params = inst_shd.VsParams{
                .view_proj = view_proj,
                .light_view_proj = light_view_proj,
            };
            const mat = if (mesh.material) |m| switch (m) {
                .standard => |s| s,
                else => &self.default_material,
            } else &self.default_material;

            const shadow_intensity_val: f32 = if (self.enable_shadows and mesh.receive_shadows) self.shadow_intensity else 0.0;
            const fs_params = inst_shd.FsParams{
                .light_dir = .{ self.light.direction.x, self.light.direction.y, self.light.direction.z, 0.0 },
                .light_color = .{ self.light.diffuse.r, self.light.diffuse.g, self.light.diffuse.b, self.light.intensity },
                .ambient_color = .{ self.light.ground_color.r, self.light.ground_color.g, self.light.ground_color.b, 1.0 },
                .diffuse_color = mat.getDiffuseColor4(),
                .shadow_params = .{ self.shadow_bias, shadow_intensity_val, 0.0, 0.0 },
                .light_counts = light_counts,
                .point_pos_range = point_pos_range,
                .point_color_int = point_color_int,
                .spot_pos_range = spot_pos_range,
                .spot_dir_inner = spot_dir_inner,
                .spot_color_outer = spot_color_outer,
                .spot_intensity = spot_intensity,
            };

            const tex = if (mat.diffuse_texture) |t| t else self.default_white_texture;
            var bind = sg.Bindings{};
            bind.vertex_buffers[0] = mesh.vertex_buffer;
            bind.vertex_buffers[1] = mesh.instance_buffer;
            bind.index_buffer = mesh.index_buffer;
            bind.views[inst_shd.VIEW_diffuse_tex] = tex.view;
            bind.samplers[inst_shd.SMP_smp] = tex.sampler;
            bind.views[inst_shd.VIEW_shadow_tex] = self.shadow_texture_view;
            bind.samplers[inst_shd.SMP_shadow_smp] = self.shadow_sampler;

            sg.applyBindings(bind);
            sg.applyUniforms(inst_shd.UB_vs_params, sg.asRange(&vs_params));
            sg.applyUniforms(inst_shd.UB_fs_params, sg.asRange(&fs_params));
            sg.draw(0, mesh.index_count, mesh.visible_instance_count);

            self.stats.draw_calls += 1;
            self.stats.triangles += (mesh.index_count / 3) * mesh.visible_instance_count;
        }

        // Sort non-instanced meshes
        std.mem.sort(RenderMeshItem, self.render_queue.items, {}, sortRenderItems);

        // Render sorted non-instanced meshes
        for (self.render_queue.items) |item| {
            const mesh = item.mesh;
            const model = item.model;
            const mvp = Mat4.mul(view_proj, model);
            const shadow_intensity_val: f32 = if (self.enable_shadows and mesh.receive_shadows) self.shadow_intensity else 0.0;

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
                    .light_view_proj = light_view_proj,
                };

                const fs_params = pbr_shd.FsParams{
                    .eye_pos = .{ eye.x, eye.y, eye.z, 1.0 },
                    .light_dir = .{ self.light.direction.x, self.light.direction.y, self.light.direction.z, 0.0 },
                    .light_color = .{ self.light.diffuse.r, self.light.diffuse.g, self.light.diffuse.b, self.light.intensity },
                    .ambient_color = .{ self.light.ground_color.r, self.light.ground_color.g, self.light.ground_color.b, 1.0 },
                    .base_color_factor = pbr_mat.getAlbedoColor4(),
                    .pbr_factors = .{
                        pbr_mat.metallic,
                        pbr_mat.roughness,
                        pbr_mat.occlusion_strength,
                        pbr_mat.environment_intensity * self.ibl_intensity,
                    },
                    .emissive_factor = pbr_mat.getEmissiveColor4(),
                    .shadow_params = .{ self.shadow_bias, shadow_intensity_val, 0.0, 0.0 },
                    .light_counts = light_counts,
                    .point_pos_range = point_pos_range,
                    .point_color_int = point_color_int,
                    .spot_pos_range = spot_pos_range,
                    .spot_dir_inner = spot_dir_inner,
                    .spot_color_outer = spot_color_outer,
                    .spot_intensity = spot_intensity,
                };

                const albedo_tex = if (pbr_mat.albedo_texture) |t| t else self.default_white_texture;
                const normal_tex = if (pbr_mat.normal_texture) |t| t else self.default_normal_texture;
                const mr_tex = if (pbr_mat.metallic_roughness_texture) |t| t else self.default_white_texture;
                const emissive_tex = if (pbr_mat.emissive_texture) |t| t else self.default_white_texture;
                const occlusion_tex = if (pbr_mat.occlusion_texture) |t| t else self.default_white_texture;
                const env_cube = if (pbr_mat.environment_texture) |c|
                    c
                else if (self.skybox_texture) |c|
                    c
                else
                    self.default_cube_texture;

                bind.views[pbr_shd.VIEW_albedo_tex] = albedo_tex.view;
                bind.views[pbr_shd.VIEW_normal_tex] = normal_tex.view;
                bind.views[pbr_shd.VIEW_metallic_roughness_tex] = mr_tex.view;
                bind.views[pbr_shd.VIEW_emissive_tex] = emissive_tex.view;
                bind.views[pbr_shd.VIEW_occlusion_tex] = occlusion_tex.view;
                bind.views[pbr_shd.VIEW_shadow_tex] = self.shadow_texture_view;
                bind.views[pbr_shd.VIEW_env_tex] = env_cube.view;
                bind.samplers[pbr_shd.SMP_smp] = albedo_tex.sampler;
                bind.samplers[pbr_shd.SMP_shadow_smp] = self.shadow_sampler;
                bind.samplers[pbr_shd.SMP_env_smp] = env_cube.sampler;

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
                    .light_view_proj = light_view_proj,
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
                    .shadow_params = .{ self.shadow_bias, shadow_intensity_val, 0.0, 0.0 },
                    .light_counts = light_counts,
                    .point_pos_range = point_pos_range,
                    .point_color_int = point_color_int,
                    .spot_pos_range = spot_pos_range,
                    .spot_dir_inner = spot_dir_inner,
                    .spot_color_outer = spot_color_outer,
                    .spot_intensity = spot_intensity,
                };

                const tex = if (mat.diffuse_texture) |t| t else self.default_white_texture;
                bind.views[shd.VIEW_diffuse_tex] = tex.view;
                bind.samplers[shd.SMP_smp] = tex.sampler;
                bind.views[shd.VIEW_shadow_tex] = self.shadow_texture_view;
                bind.samplers[shd.SMP_shadow_smp] = self.shadow_sampler;

                sg.applyBindings(bind);
                sg.applyUniforms(shd.UB_vs_params, sg.asRange(&vs_params));
                sg.applyUniforms(shd.UB_fs_params, sg.asRange(&fs_params));
                sg.draw(0, mesh.index_count, 1);

                self.stats.draw_calls += 1;
                self.stats.triangles += mesh.index_count / 3;
            }
        }

        if (self.skybox_enabled and self.skybox_texture != null and self.active_camera != null) {
            self.renderSkybox(self.active_camera.?, aspect);
        }

        if (self.active_camera != null and self.particle_systems.items.len > 0) {
            self.renderParticles(self.active_camera.?, aspect);
        }

        sg.endPass();

        if (self.post_process.enabled) {
            // ==============================================
            // PASS 3: FULLSCREEN POST-PROCESSING PASS
            // ==============================================
            var swap_action = sg.PassAction{};
            swap_action.colors[0] = .{
                .load_action = .DONTCARE,
            };
            sg.beginPass(.{
                .action = swap_action,
                .swapchain = sglue.swapchain(),
            });

            sg.applyPipeline(self.postprocess_pipeline);
            var post_bind = sg.Bindings{};
            post_bind.vertex_buffers[0] = self.postprocess_quad_vb;
            post_bind.index_buffer = self.postprocess_quad_ib;
            post_bind.views[post_shd.VIEW_scene_tex] = self.offscreen_resolve_tex_view;
            post_bind.samplers[post_shd.SMP_smp] = self.postprocess_sampler;
            sg.applyBindings(post_bind);

            const pp_params = post_shd.FsParams{
                .params1 = .{
                    self.post_process.exposure,
                    self.post_process.bloom_threshold,
                    self.post_process.bloom_intensity,
                    self.post_process.bloom_radius,
                },
                .params2 = .{
                    self.post_process.vignette_intensity,
                    self.post_process.vignette_radius,
                    self.post_process.saturation,
                    self.post_process.contrast,
                },
                .params3 = .{
                    @floatFromInt(@intFromEnum(self.post_process.tonemapping)),
                    self.post_process.chromatic_aberration,
                    if (self.post_process.bloom_enabled) 1.0 else 0.0,
                    if (self.post_process.vignette_enabled) 1.0 else 0.0,
                },
                .resolution = .{
                    @floatFromInt(cur_w),
                    @floatFromInt(cur_h),
                    1.0 / @as(f32, @floatFromInt(cur_w)),
                    1.0 / @as(f32, @floatFromInt(cur_h)),
                },
            };
            sg.applyUniforms(post_shd.UB_fs_params, sg.asRange(&pp_params));
            sg.draw(0, 6, 1);
            self.stats.draw_calls += 1;
            self.stats.triangles += 2;

            sg.endPass();
        }

        sg.commit();
    }

    fn renderSkybox(self: *Scene, camera: ArcRotateCamera, aspect: f32) void {
        const cube = self.skybox_texture orelse return;

        const view_no_trans = camera.getViewMatrix().removeTranslation();
        const view_proj = Mat4.mul(camera.getProjectionMatrix(aspect), view_no_trans);

        sg.applyPipeline(self.skybox_pipeline);

        var bind = sg.Bindings{};
        bind.vertex_buffers[0] = self.skybox_mesh_vb;
        bind.index_buffer = self.skybox_mesh_ib;
        bind.views[sky_shd.VIEW_sky_tex] = cube.view;
        bind.samplers[sky_shd.SMP_smp] = cube.sampler;
        sg.applyBindings(bind);

        const vs_params = sky_shd.VsParams{
            .view_proj = view_proj,
        };
        const fs_params = sky_shd.FsParams{
            .params = .{ self.skybox_exposure, 0.0, 0.0, 0.0 },
        };
        sg.applyUniforms(sky_shd.UB_vs_params, sg.asRange(&vs_params));
        sg.applyUniforms(sky_shd.UB_fs_params, sg.asRange(&fs_params));
        sg.draw(0, 36, 1);

        self.stats.draw_calls += 1;
        self.stats.triangles += 12;
    }

    fn renderParticles(self: *Scene, camera: ArcRotateCamera, aspect: f32) void {
        const view = camera.getViewMatrix();
        const proj = camera.getProjectionMatrix(aspect);
        const view_proj = Mat4.mul(proj, view);

        const cam_right = [4]f32{ view.m[0], view.m[4], view.m[8], 0.0 };
        const cam_up = [4]f32{ view.m[1], view.m[5], view.m[9], 0.0 };

        const vs_params = part_shd.VsParams{
            .view_proj = view_proj,
            .camera_right = cam_right,
            .camera_up = cam_up,
        };

        var current_pip_id: u32 = 0;

        for (self.particle_systems.items) |ps| {
            if (ps.active_count == 0) continue;

            const pip = if (ps.blend_mode == .additive) self.particle_pipeline_additive else self.particle_pipeline_alphablend;
            if (pip.id != current_pip_id) {
                sg.applyPipeline(pip);
                current_pip_id = pip.id;
                self.stats.pipeline_switches += 1;
            }

            const tex = if (ps.texture) |t| t else self.default_particle_texture;

            var bind = sg.Bindings{};
            bind.vertex_buffers[0] = self.particle_quad_vb;
            bind.vertex_buffers[1] = ps.instance_buffer;
            bind.index_buffer = self.particle_quad_ib;
            bind.views[part_shd.VIEW_particle_tex] = tex.view;
            bind.samplers[part_shd.SMP_smp] = tex.sampler;

            sg.applyBindings(bind);
            sg.applyUniforms(part_shd.UB_vs_params, sg.asRange(&vs_params));
            sg.draw(0, 6, @intCast(ps.active_count));

            self.stats.draw_calls += 1;
            self.stats.triangles += 2 * @as(u32, @intCast(ps.active_count));
        }
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

        sg.destroyPipeline(self.skybox_pipeline);
        sg.destroyBuffer(self.skybox_mesh_vb);
        sg.destroyBuffer(self.skybox_mesh_ib);

        sg.destroyPipeline(self.shadow_pipeline_u16);
        sg.destroyPipeline(self.shadow_pipeline_u32);
        sg.destroyPipeline(self.shadow_inst_pipeline_u16);
        sg.destroyPipeline(self.shadow_inst_pipeline_u32);
        sg.destroyView(self.shadow_attachment_view);
        sg.destroyView(self.shadow_texture_view);
        sg.destroySampler(self.shadow_sampler);
        sg.destroyImage(self.shadow_image);

        if (self.offscreen_color_image.id != 0) {
            sg.destroyImage(self.offscreen_color_image);
            sg.destroyView(self.offscreen_color_att_view);
            if (self.offscreen_resolve_image.id != 0) {
                sg.destroyImage(self.offscreen_resolve_image);
                sg.destroyView(self.offscreen_resolve_att_view);
                sg.destroyView(self.offscreen_resolve_tex_view);
            } else {
                sg.destroyView(self.offscreen_resolve_tex_view);
            }
            sg.destroyImage(self.offscreen_depth_image);
            sg.destroyView(self.offscreen_depth_att_view);
        }
        sg.destroyPipeline(self.postprocess_pipeline);
        sg.destroySampler(self.postprocess_sampler);
        sg.destroyBuffer(self.postprocess_quad_vb);
        sg.destroyBuffer(self.postprocess_quad_ib);

        for (self.particle_systems.items) |ps| {
            ps.deinit();
            self.allocator.destroy(ps);
        }
        self.particle_systems.deinit(self.allocator);
        self.default_particle_texture.deinit();
        sg.destroyPipeline(self.particle_pipeline_additive);
        sg.destroyPipeline(self.particle_pipeline_alphablend);
        sg.destroyBuffer(self.particle_quad_vb);
        sg.destroyBuffer(self.particle_quad_ib);
    }

};
