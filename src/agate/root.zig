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
pub const TargetCamera = camera.TargetCamera;
pub const TargetCameraOptions = camera.TargetCameraOptions;
pub const FlyCamera = camera.FlyCamera;
pub const FlyCameraOptions = camera.FlyCameraOptions;

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
pub const LODLevel = mesh.LODLevel;
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
pub const PolygonOptions = mesh.PolygonOptions;
pub const PolygonSideOrientation = mesh.PolygonSideOrientation;
pub const PolygonPlane = mesh.PolygonPlane;
pub const buildPolygonData = mesh.buildPolygonData;
pub const TrailOptions = mesh.TrailOptions;
pub const TrailMesh = mesh.TrailMesh;
pub const TrailNode = mesh.TrailNode;
pub const DecalOptions = mesh.DecalOptions;
pub const buildDecalData = mesh.buildDecalData;
pub const createDecal = mesh.createDecal;
pub const DecalProjector = mesh.DecalProjector;
pub const DecalManager = mesh.DecalManager;
pub const DecalSpawnOptions = mesh.DecalSpawnOptions;
pub const DecalInstance = mesh.DecalInstance;
pub const SkinJointWeight = mesh.SkinJointWeight;
pub const barycentric = mesh.barycentric;
pub const blendSkinWeights = mesh.blendSkinWeights;
pub const MorphTarget = mesh.MorphTarget;
pub const MAX_MORPH_TARGETS = mesh.MAX_MORPH_TARGETS;
pub const CSG = mesh.CSG;
pub const CSGVertex = mesh.CSGVertex;
pub const CSGPlane = mesh.CSGPlane;
pub const CSGPolygon = mesh.CSGPolygon;
pub const CSGNode = mesh.CSGNode;

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
pub const ScrollState = ui.ScrollState;
pub const TextInputState = ui.TextInputState;

pub const passes = @import("passes/mod.zig");
pub const DebugPass = passes.DebugPass;
pub const BloomPass = passes.BloomPass;
pub const OutlinePass = passes.OutlinePass;

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
    pub const MorphWeightsTarget = anim.MorphWeightsTarget;
    pub const AnimationEvent = anim.AnimationEvent;
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
pub const MorphWeightsTarget = animation.MorphWeightsTarget;
pub const AnimationEvent = animation.AnimationEvent;
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

pub const serialization = @import("serialization.zig");
pub const SceneState = serialization.SceneState;
pub const captureSceneState = serialization.capture;
pub const restoreSceneState = serialization.restore;
pub const serializeSceneState = serialization.serializeAlloc;
pub const deserializeSceneState = serialization.deserializeAlloc;
pub const saveSceneStateFile = serialization.saveFile;
pub const loadSceneStateFile = serialization.loadFile;

pub const loader_lights = @import("loader/lights.zig");
pub const obj_loader = @import("loader/obj.zig");
pub const stl_loader = @import("loader/stl.zig");
pub const ObjData = obj_loader.ObjData;
pub const StlData = stl_loader.StlData;
pub const parseObj = obj_loader.parse;
pub const parseStl = stl_loader.parse;
pub const appendObjToScene = obj_loader.appendToScene;
pub const appendStlToScene = stl_loader.appendToScene;
pub const loadGltfLights = loader_lights.loadLights;
pub const loadGltfCameras = loader_lights.loadCameras;

pub const ply_loader = @import("loader/ply.zig");
pub const PlyData = ply_loader.PlyData;
pub const parsePly = ply_loader.parse;
pub const appendPlyToScene = ply_loader.appendToScene;

pub const export_obj = @import("export/obj.zig");
pub const export_stl = @import("export/stl.zig");
pub const ObjExportOptions = export_obj.ObjExportOptions;
pub const StlExportOptions = export_stl.StlExportOptions;
pub const writeObjAlloc = export_obj.writeObjAlloc;
pub const writeMtlAlloc = export_obj.writeMtlAlloc;
pub const writeStlAlloc = export_stl.writeStlAlloc;
pub const writeStlAsciiAlloc = export_stl.writeStlAsciiAlloc;
pub const writeStlBinaryAlloc = export_stl.writeStlBinaryAlloc;

pub const ai = @import("ai.zig");
pub const NavNode = ai.NavNode;
pub const NavMesh = ai.NavMesh;
pub const Portal = ai.Portal;
pub const triArea2D = ai.triArea2D;
pub const stringPull = ai.stringPull;
pub const Pathfinding = ai.Pathfinding;
pub const NavAgent = ai.NavAgent;

pub const visibility = @import("visibility/mod.zig");
pub const HiZBuffer = visibility.HiZBuffer;
pub const SoftwareRasterizer = visibility.SoftwareRasterizer;
pub const OcclusionCuller = visibility.OcclusionCuller;

pub const sokol = @import("sokol");

test {
    _ = @import("ai.zig");
    _ = @import("mesh/csg_tests.zig");
    _ = @import("visibility/mod.zig");
}
