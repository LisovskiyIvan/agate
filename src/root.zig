//! agate: a Babylon.js-style 3D engine in Zig on top of sokol.
//!
//! This file is the library facade: everything public is reachable from here
//! (`@import("agate")`), while module paths (`agate.mesh.Mesh`,
//! `agate.scene.Scene`, ...) stay importable for finer-grained access.
//!
//! Naming conventions used across the API:
//! - Value constructors: `Type.new(...)` (math) and `Type.init(name, options)`
//!   (plain structs: materials, cameras, lights).
//! - Scene/GPU-owning entities: `create*` (MeshBuilder.createBox,
//!   scene.createParticleSystem); sokol-object wrappers use `make*` to mirror
//!   sg.makePipeline (`compute.makePipeline`).
//! - Pure data assembly without GPU side effects: `build*Data` returning
//!   `GeometryData`.
//! - Option structs: `*Options` for user-tweakable knobs (all fields
//!   defaulted); `*Desc` for registration descriptors with required fields
//!   (shader_material.RuntimeDesc, mirroring sg.*Desc); `*Params` for
//!   computed per-frame data packs (vat.VatSampleParams).

pub const math = @import("math");
pub const Vec2 = math.Vec2;
pub const Vec3 = math.Vec3;
pub const Mat4 = math.Mat4;
pub const Quat = math.Quat;
pub const Color3 = math.Color3;
pub const Color4 = math.Color4;
pub const BoundingBox = math.BoundingBox;
pub const Frustum = math.Frustum;

pub const camera = @import("camera.zig");
pub const Viewport = camera.Viewport;
pub const ArcRotateCamera = camera.ArcRotateCamera;
pub const Camera = camera.Camera;
pub const FreeCamera = camera.FreeCamera;
pub const FollowCamera = camera.FollowCamera;
pub const TargetCamera = camera.TargetCamera;
pub const FlyCamera = camera.FlyCamera;
pub const CameraRig = camera.CameraRig;

pub const lights = @import("lights.zig");
pub const HemisphericLight = lights.HemisphericLight;
pub const DirectionalLight = lights.DirectionalLight;
pub const PointLight = lights.PointLight;
pub const SpotLight = lights.SpotLight;
pub const sunDirectionFromAngles = lights.sunDirectionFromAngles;
pub const colorTemperatureToRgb = lights.colorTemperatureToRgb;
pub const ClusteredPointLight = lights.ClusteredPointLight;
pub const AreaLight = lights.AreaLight;

pub const material = @import("material.zig");
pub const PBRMaterial = material.PBRMaterial;
pub const Material = material.Material;
pub const ShaderMaterial = material.ShaderMaterial;

// Custom shader materials: build-time hook materials (build.zig
// `user_shader_materials`) and runtime-registered sources. See
// shader_material.zig docs for the registration/uniform model.
pub const shader_material = @import("shader_material.zig");

// Named material presets (Sky/Gradient/Grid/TriPlanar) as reusable
// ShaderMaterial constructors over the hook registrations above. See
// material_library.zig docs for the scene-wiring pattern.
pub const material_library = @import("material_library.zig");

// Typed material graphs that compile to ShaderMaterial hook snippets
// (graph parameters become runtime uniforms — no engine rebuild). See
// node_material.zig docs for the bake-once / drive-uniforms pattern.
pub const node_material = @import("node_material.zig");

pub const texture = @import("texture.zig");
/// TrueType (glyf-based) font parser, rasterizer and CPU glyph atlas plus
/// the UI-facing TtfFont handle. UI text switches via
/// UICanvas.setFontTtf; see ttf.zig for the supported/rejected matrix.
/// Reach the module path as `agate.ttf` (e.g. `agate.ttf.TtfFont`).
pub const ttf = @import("ttf.zig");
pub const Texture = texture.Texture;
pub const CubeTexture = texture.CubeTexture;
pub const SkyboxOptions = texture.SkyboxOptions;

// Render-to-texture target v1 (color+depth attachments, sampling views,
// optional prepared-scene capture). See render_target.zig.
pub const render_target = @import("render_target.zig");
pub const RenderTarget = render_target.RenderTarget;
pub const RenderTargetDesc = render_target.RenderTargetDesc;

pub const mesh = @import("mesh.zig");
pub const Mesh = mesh.Mesh;
pub const InstancedMesh = mesh.InstancedMesh;
pub const Vertex = mesh.Vertex;
pub const GeometryData = mesh.GeometryData;
pub const uploadGeometry = mesh.uploadGeometry;
pub const buildTorusKnotData = mesh.buildTorusKnotData;
pub const MeshBuilder = mesh.MeshBuilder;
pub const TrailMesh = mesh.TrailMesh;
pub const DecalProjector = mesh.DecalProjector;
pub const CSG = mesh.CSG;
pub const VatData = mesh.VatData;
pub const VatBaker = mesh.VatBaker;
pub const VatPlayer = mesh.VatPlayer;
pub const GreasedLineMesh = mesh.GreasedLineMesh;
pub const simplifyGeometry = mesh.simplifyGeometry;
pub const generateLODLevels = mesh.generateLODLevels;

pub const scene = @import("scene.zig");
pub const Scene = scene.Scene;
pub const SceneStats = scene.SceneStats;
pub const RenderMeshItem = scene.RenderMeshItem;
pub const sortRenderItems = scene.sortRenderItems;
pub const RenderQueues = scene.RenderQueues;
pub const sortOpaqueItemsIndexed = scene.sortOpaqueItemsIndexed;
pub const ReflectionProbe = scene.ReflectionProbe;
pub const Ui3dPanel = scene.Ui3dPanel;
pub const HighlightLayer = scene.HighlightLayer;
pub const HighlightEntry = scene.HighlightEntry;
pub const HighlightOptions = scene.HighlightOptions;

pub const loader = @import("loader/scene_loader.zig");
pub const SceneLoader = loader.SceneLoader;
pub const LoadTimings = loader.LoadTimings;

pub const postprocess = @import("postprocess.zig");

pub const jobs = @import("jobs.zig");
// User-configurable simulation threading (phase 1: dedicated physics
// stepping via ThreadingOptions; see docs/threading.md).
pub const threading = @import("threading.zig");
pub const ThreadingOptions = threading.ThreadingOptions;

/// Thin engine-owned frame lifecycle facade (update-vs-begin phase mutex,
/// worker start/stop, producer claim->build->stageUi->publish one-liner,
/// staged begin/finish/cancel with the smallest safe exclusion boundary,
/// serial legacy prepare, reuse fallback, truthful counters). Threaded
/// apps (agate demo, sandbox) compose their frame loop from this instead
/// of reproducing the ordering by hand.
pub const runtime = @import("runtime.zig");
pub const Runtime = runtime.Runtime;
/// Headless CPU-only scene fixture (no sg context needed): the same minimal
/// scene the unit suite drives the staged pipeline with (staged build /
/// begin / finish / reuse / retire epochs are real; GPU uploads and draws
/// degrade to their documented coherent-empty no-ops). For headless tooling
/// only (chaos runner CI mode) — windowed apps keep `Scene.init`. Teardown
/// mirrors the unit suite (drain retires, then deinit each owned list;
/// see examples/chaos_runner.zig) — `Scene.deinit` assumes full-init GPU
/// resources and must not be called on a fixture scene.
pub const testing = @import("testing.zig");
pub const app = @import("app.zig");
pub const App = app.App;
pub const handoff = @import("handoff.zig");
pub const Handoff = handoff.Handoff;
/// Graphics-context thread marker: apps call `markContextThread()` in their
/// init callback so engine paths can detect off-context GPU touches.
pub const gpu_thread = @import("gpu_thread.zig");
pub const particles = @import("particles.zig");
pub const ParticleSystem = particles.ParticleSystem;
pub const SimulationMode = particles.SimulationMode;
pub const CollisionMode = particles.CollisionMode;

/// Compute-pass support (see compute.zig for the backend matrix).
pub const compute = @import("compute.zig");

pub const Ray = math.Ray;
pub const picking = @import("scene/picking.zig");

pub const physics = @import("physics.zig");
pub const RigidBody = physics.RigidBody;
pub const PhysicsWorld = physics.PhysicsWorld;
pub const ColliderType = physics.ColliderType;
pub const JointId = physics.JointId;
pub const CharacterController = physics.CharacterController;
pub const Rope = physics.Rope;

pub const ui = @import("ui.zig");
pub const UICanvas = ui.UICanvas;
pub const ScrollState = ui.ScrollState;
pub const TextInputState = ui.TextInputState;
pub const LayoutStack = ui.LayoutStack;

// Layout system: flexible dimensions, anchors, docking, flexbox & CSS grid solvers
pub const UIEdges = ui.UIEdges;
// Style system phase 2: CSS theming + style transitions.
pub const CssTheme = ui.CssTheme;
pub const parseCss = ui.parseCss;

pub const passes = @import("passes/mod.zig");
pub const outline_pass = passes.outline_pass; // file-level: shouldOutlineMesh etc.

pub const animation = struct {
    pub const skeleton = @import("animation/skeleton.zig");
    pub const anim = @import("animation/animation.zig");
    pub const easing = @import("animation/easing.zig");
    pub const retarget = @import("animation/retarget.zig");
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
    pub const TranslationMode = retarget.TranslationMode;
    pub const BoneMap = retarget.BoneMap;
    pub const RetargetOptions = retarget.RetargetOptions;
    pub const retargetAnimationGroup = retarget.retargetAnimationGroup;
    pub const boneRestLength = retarget.boneRestLength;
    pub const translationScaleFactor = retarget.translationScaleFactor;
};
pub const Skeleton = animation.Skeleton;
pub const AnimationGroup = animation.AnimationGroup;
pub const AnimationEvent = animation.AnimationEvent;
pub const evaluateSkeleton = animation.evaluateSkeleton;
pub const BoneMap = animation.BoneMap;
pub const retargetAnimationGroup = animation.retargetAnimationGroup;

pub const Ragdoll = @import("ragdoll.zig").Ragdoll;
pub const RaycastVehicle = @import("vehicle.zig").RaycastVehicle;

pub const audio = @import("audio.zig");
pub const AudioEngine = audio.AudioEngine;
pub const AudioClip = audio.AudioClip;
pub const BusId = audio.BusId;
pub const invalid_bus = audio.invalid_bus;
pub const StreamOptions = audio.StreamOptions;
pub const AudioStream = audio.AudioStream;

pub const serialization = @import("serialization.zig");
pub const captureSceneState = serialization.capture;
pub const restoreSceneState = serialization.restore;
pub const serializeSceneState = serialization.serializeAlloc;
pub const deserializeSceneState = serialization.deserializeAlloc;
pub const AsyncSaveTask = serialization.AsyncSaveTask;
pub const AsyncLoadTask = serialization.AsyncLoadTask;

pub const obj_loader = @import("loader/obj.zig");
pub const stl_loader = @import("loader/stl.zig");
pub const parseObj = obj_loader.parse;
pub const parseStl = stl_loader.parse;
pub const parsePly = ply_loader.parse;
pub const appendObjToScene = obj_loader.appendToScene;
pub const appendStlToScene = stl_loader.appendToScene;

pub const ply_loader = @import("loader/ply.zig");
pub const appendPlyToScene = ply_loader.appendToScene;

pub const export_obj = @import("export/obj.zig");
pub const export_stl = @import("export/stl.zig");
pub const export_glb = @import("export/glb.zig");
pub const writeObjAlloc = export_obj.writeObjAlloc;
pub const writeMtlAlloc = export_obj.writeMtlAlloc;
pub const writeStlBinaryAlloc = export_stl.writeStlBinaryAlloc;
pub const writeGlbAlloc = export_glb.writeGlbAlloc;

pub const assets = @import("assets.zig");
pub const asset_manager = @import("asset_manager.zig");
pub const AssetManager = asset_manager.AssetManager;
pub const AssetTask = asset_manager.AssetTask;
pub const AssetCache = asset_manager.AssetCache;

pub const ai = @import("ai.zig");
pub const NavMesh = ai.NavMesh;
pub const Pathfinding = ai.Pathfinding;
pub const NavAgent = ai.NavAgent;
pub const Crowd = ai.Crowd;

pub const visibility = @import("visibility/mod.zig");

/// PBD cloth soft bodies (wave 29, v1): solver + mesh coupling + scene layer.
pub const softbody = @import("softbody.zig");
pub const SoftBody = softbody.SoftBody;
pub const ClothOptions = softbody.ClothOptions;

pub const profiler = @import("profiler.zig");

/// Opt-in GPU frame/phase timestamps, capabilities and completed samples.
pub const gpu_timing = @import("gpu_timing.zig");

pub const sokol = @import("sokol");

test {
    _ = @import("tests.zig");
}
