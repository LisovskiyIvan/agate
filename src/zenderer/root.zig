pub const math = @import("math");
pub const Vec2 = math.Vec2;
pub const Vec3 = math.Vec3;
pub const Vec4 = math.Vec4;
pub const Mat4 = math.Mat4;
pub const Color3 = math.Color3;
pub const Color4 = math.Color4;
pub const BoundingBox = math.BoundingBox;
pub const Frustum = math.Frustum;

pub const camera = @import("camera.zig");
pub const ArcRotateCamera = camera.ArcRotateCamera;

pub const lights = @import("lights.zig");
pub const HemisphericLight = lights.HemisphericLight;
pub const DirectionalLight = lights.DirectionalLight;
pub const PointLight = lights.PointLight;
pub const PointLightOptions = lights.PointLightOptions;
pub const SpotLight = lights.SpotLight;
pub const SpotLightOptions = lights.SpotLightOptions;

pub const material = @import("material.zig");
pub const StandardMaterial = material.StandardMaterial;
pub const PBRMaterial = material.PBRMaterial;
pub const Material = material.Material;

pub const texture = @import("texture.zig");
pub const Texture = texture.Texture;
pub const CubeTexture = texture.CubeTexture;
pub const SkyboxConfig = texture.SkyboxConfig;

pub const mesh = @import("mesh.zig");
pub const Mesh = mesh.Mesh;
pub const InstancedMesh = mesh.InstancedMesh;
pub const CullingStrategy = mesh.CullingStrategy;
pub const Vertex = mesh.Vertex;
pub const MeshBuilder = mesh.MeshBuilder;
pub const BoxOptions = mesh.BoxOptions;

pub const scene = @import("scene.zig");
pub const Scene = scene.Scene;

pub const loader = @import("loader/scene_loader.zig");
pub const SceneLoader = loader.SceneLoader;

pub const postprocess = @import("postprocess.zig");
pub const PostProcessConfig = postprocess.PostProcessConfig;
pub const TonemappingType = postprocess.TonemappingType;

pub const sokol = @import("sokol");
