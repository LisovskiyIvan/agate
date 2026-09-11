pub const math = @import("math");
pub const Vec2 = math.Vec2;
pub const Vec3 = math.Vec3;
pub const Vec4 = math.Vec4;
pub const Mat4 = math.Mat4;
pub const Quat = math.Quat;
pub const Color3 = math.Color3;
pub const Color4 = math.Color4;
pub const BoundingBox = math.BoundingBox;
pub const Frustum = math.Frustum;
pub const FrustumPlane = math.FrustumPlane;
pub const lerp = math.lerp;

pub const camera = @import("camera.zig");
pub const ArcRotateCameraOptions = camera.ArcRotateCameraOptions;
pub const ArcRotateCamera = camera.ArcRotateCamera;
pub const Camera = camera.Camera;
pub const FreeCamera = camera.FreeCamera;
pub const FreeCameraOptions = camera.FreeCameraOptions;
pub const FollowCamera = camera.FollowCamera;
pub const FollowCameraOptions = camera.FollowCameraOptions;

pub const lights = @import("lights.zig");
pub const HemisphericLightOptions = lights.HemisphericLightOptions;
pub const HemisphericLight = lights.HemisphericLight;
pub const DirectionalLight = lights.DirectionalLight;
pub const DirectionalLightOptions = lights.DirectionalLightOptions;
pub const PointLight = lights.PointLight;
pub const PointLightOptions = lights.PointLightOptions;
pub const SpotLight = lights.SpotLight;
pub const SpotLightOptions = lights.SpotLightOptions;
pub const resolveSunDirection = lights.resolveSunDirection;
pub const resolveSunColor = lights.resolveSunColor;
pub const resolveSunIntensity = lights.resolveSunIntensity;

pub const material = @import("material.zig");
pub const StandardMaterial = material.StandardMaterial;
pub const PBRMaterial = material.PBRMaterial;
pub const Material = material.Material;
pub const AlphaMode = material.AlphaMode;

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
pub const computeTangents = mesh.computeTangents;
pub const BoxOptions = mesh.BoxOptions;
pub const SphereOptions = mesh.SphereOptions;
pub const CylinderOptions = mesh.CylinderOptions;
pub const CapsuleOptions = mesh.CapsuleOptions;
pub const GroundOptions = mesh.GroundOptions;
pub const TerrainOptions = mesh.TerrainOptions;
pub const TorusOptions = mesh.TorusOptions;
pub const TorusKnotOptions = mesh.TorusKnotOptions;
pub const DiscOptions = mesh.DiscOptions;
pub const RibbonOptions = mesh.RibbonOptions;
pub const LatheOptions = mesh.LatheOptions;
pub const PlaneOptions = mesh.PlaneOptions;
pub const TubeOptions = mesh.TubeOptions;
pub const LinesOptions = mesh.LinesOptions;
pub const ExtrudeOptions = mesh.ExtrudeOptions;

pub const scene = @import("scene.zig");
pub const Scene = scene.Scene;
pub const SceneStats = scene.SceneStats;
pub const RenderMeshItem = scene.RenderMeshItem;

pub const loader = @import("loader/scene_loader.zig");
pub const SceneLoader = loader.SceneLoader;

pub const postprocess = @import("postprocess.zig");
pub const PostProcessConfig = postprocess.PostProcessConfig;
pub const TonemappingType = postprocess.TonemappingType;

pub const ssao = @import("ssao.zig");
pub const SSAOConfig = ssao.SSAOConfig;

pub const particles = @import("particles.zig");
pub const ParticleSystem = particles.ParticleSystem;
pub const ParticleBlendMode = particles.ParticleBlendMode;
pub const Particle = particles.Particle;
pub const ParticleInstanceData = particles.ParticleInstanceData;

pub const Ray = math.Ray;
pub const RayHit = math.RayHit;
pub const TriangleHit = math.TriangleHit;

pub const physics = @import("physics.zig");
pub const RigidBody = physics.RigidBody;
pub const DebugLine = physics.DebugLine;
pub const PhysicsWorld = physics.PhysicsWorld;
pub const PickingInfo = physics.PickingInfo;
pub const ColliderType = physics.ColliderType;
pub const JointId = physics.JointId;
pub const DistanceJointOptions = physics.DistanceJointOptions;
pub const SphericalJointOptions = physics.SphericalJointOptions;
pub const RevoluteJointOptions = physics.RevoluteJointOptions;
pub const WheelJointOptions = physics.WheelJointOptions;
pub const PrismaticJointOptions = physics.PrismaticJointOptions;
pub const MotorJointOptions = physics.MotorJointOptions;
pub const WeldJointOptions = physics.WeldJointOptions;
pub const ParallelJointOptions = physics.ParallelJointOptions;
pub const HeightFieldOptions = physics.HeightFieldOptions;
pub const PhysicsRayHit = physics.PhysicsRayHit;
pub const CollisionFilter = physics.CollisionFilter;
pub const BodyOptions = physics.BodyOptions;
pub const ChildShape = physics.ChildShape;
pub const ChildShapeOptions = physics.ChildShapeOptions;
pub const CharacterController = physics.CharacterController;
pub const Rope = physics.Rope;
pub const RopeOptions = physics.RopeOptions;
pub const SensorEvent = physics.SensorEvent;
pub const ContactEvent = physics.ContactEvent;
pub const ContactHitEvent = physics.ContactHitEvent;

pub const ui = @import("ui.zig");
pub const UICanvas = ui.UICanvas;
pub const UIVertex = ui.UIVertex;
pub const GlyphUV = ui.GlyphUV;
pub const getGlyphUV = ui.getGlyphUV;

pub const passes = @import("passes/mod.zig");

pub const animation = struct {
    pub const skeleton = @import("animation/skeleton.zig");
    pub const anim = @import("animation/animation.zig");
    pub const easing = @import("animation/easing.zig");
    pub const Skeleton = skeleton.Skeleton;
    pub const Bone = skeleton.Bone;
    pub const MAX_BONES = skeleton.MAX_BONES;
    pub const AnimationGroup = anim.AnimationGroup;
    pub const AnimationChannel = anim.AnimationChannel;
    pub const AnimationSampler = anim.AnimationSampler;
    pub const AnimationPath = anim.AnimationPath;
    pub const AnimationInterpolation = anim.AnimationInterpolation;
    pub const NodeChannel = anim.NodeChannel;
    pub const NodeTarget = anim.NodeTarget;
    pub const EasingType = easing.EasingType;
    pub const easing_names = easing.easing_names;
    pub const evaluateEasing = easing.evaluate;
    pub const easingName = easing.easingName;
    pub const easingFromName = easing.easingFromName;
    pub const evaluateSkeleton = anim.evaluateSkeleton;
};
pub const Skeleton = animation.Skeleton;
pub const Bone = animation.Bone;
pub const MAX_BONES = animation.MAX_BONES;
pub const AnimationGroup = animation.AnimationGroup;
pub const AnimationChannel = animation.AnimationChannel;
pub const AnimationSampler = animation.AnimationSampler;
pub const AnimationPath = animation.AnimationPath;
pub const AnimationInterpolation = animation.AnimationInterpolation;
pub const NodeChannel = animation.NodeChannel;
pub const NodeTarget = animation.NodeTarget;
pub const EasingType = animation.EasingType;
pub const easing_names = animation.easing_names;
pub const evaluateEasing = animation.evaluateEasing;
pub const easingName = animation.easingName;
pub const easingFromName = animation.easingFromName;
pub const evaluateSkeleton = animation.evaluateSkeleton;

pub const Ragdoll = @import("ragdoll.zig").Ragdoll;
pub const RagdollOptions = @import("ragdoll.zig").RagdollOptions;
pub const RagdollPart = @import("ragdoll.zig").RagdollPart;
pub const RaycastVehicle = @import("vehicle.zig").RaycastVehicle;
pub const VehicleOptions = @import("vehicle.zig").VehicleOptions;

pub const audio = @import("audio.zig");
pub const AudioEngine = audio.AudioEngine;
pub const AudioClip = audio.AudioClip;
pub const VoiceKind = audio.VoiceKind;
pub const Voice = audio.Voice;

pub const sokol = @import("sokol");
