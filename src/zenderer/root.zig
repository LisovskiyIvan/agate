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
pub const BoneAttachment = mesh.BoneAttachment;
pub const CullingStrategy = mesh.CullingStrategy;
pub const Vertex = mesh.Vertex;
pub const MeshBuilder = mesh.MeshBuilder;
pub const BoxOptions = mesh.BoxOptions;
pub const SphereOptions = mesh.SphereOptions;
pub const CylinderOptions = mesh.CylinderOptions;
pub const CapsuleOptions = mesh.CapsuleOptions;
pub const GroundOptions = mesh.GroundOptions;

pub const scene = @import("scene.zig");
pub const Scene = scene.Scene;

pub const loader = @import("loader/scene_loader.zig");
pub const SceneLoader = loader.SceneLoader;

pub const postprocess = @import("postprocess.zig");
pub const PostProcessConfig = postprocess.PostProcessConfig;
pub const TonemappingType = postprocess.TonemappingType;

pub const particles = @import("particles.zig");
pub const ParticleSystem = particles.ParticleSystem;
pub const ParticleBlendMode = particles.ParticleBlendMode;
pub const Particle = particles.Particle;

pub const Ray = math.Ray;
pub const RayHit = math.RayHit;
pub const TriangleHit = math.TriangleHit;

pub const physics = @import("physics.zig");
pub const RigidBody = physics.RigidBody;
pub const PhysicsWorld = physics.PhysicsWorld;
pub const PickingInfo = physics.PickingInfo;
pub const ColliderType = physics.ColliderType;
pub const JointId = physics.JointId;
pub const DistanceJointOptions = physics.DistanceJointOptions;
pub const SphericalJointOptions = physics.SphericalJointOptions;

pub const ui = @import("ui.zig");
pub const UICanvas = ui.UICanvas;
pub const UIVertex = ui.UIVertex;

pub const passes = @import("passes/mod.zig");

pub const animation = struct {
    pub const skeleton = @import("animation/skeleton.zig");
    pub const anim = @import("animation/animation.zig");
    pub const Skeleton = skeleton.Skeleton;
    pub const Bone = skeleton.Bone;
    pub const MAX_BONES = skeleton.MAX_BONES;
    pub const AnimationGroup = anim.AnimationGroup;
    pub const AnimationChannel = anim.AnimationChannel;
    pub const AnimationSampler = anim.AnimationSampler;
    pub const AnimationPath = anim.AnimationPath;
    pub const AnimationInterpolation = anim.AnimationInterpolation;
    pub const evaluateSkeleton = anim.evaluateSkeleton;
};
pub const Skeleton = animation.Skeleton;
pub const Bone = animation.Bone;
pub const AnimationGroup = animation.AnimationGroup;
pub const AnimationChannel = animation.AnimationChannel;
pub const AnimationSampler = animation.AnimationSampler;
pub const AnimationPath = animation.AnimationPath;
pub const AnimationInterpolation = animation.AnimationInterpolation;
pub const evaluateSkeleton = animation.evaluateSkeleton;

pub const sokol = @import("sokol");
