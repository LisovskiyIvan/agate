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
pub const Viewport = camera.Viewport;
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
pub const CameraEntry = @import("scene.zig").CameraEntry;
pub const CameraRig = camera.CameraRig;
pub const CameraRigMode = camera.CameraRigMode;
pub const StereoConvergenceMode = camera.StereoConvergenceMode;
pub const CameraRigSlot = camera.CameraRigSlot;
pub const MAX_RIG_SLOTS = camera.MAX_RIG_SLOTS;

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
pub const sunDirectionFromAngles = lights.sunDirectionFromAngles;
pub const colorTemperatureToRgb = lights.colorTemperatureToRgb;
pub const ClusteredPointLightOptions = lights.ClusteredPointLightOptions;
pub const ClusteredPointLight = lights.ClusteredPointLight;
pub const max_clustered_lights = lights.max_clustered_lights;
pub const AreaLightOptions = lights.AreaLightOptions;
pub const AreaLight = lights.AreaLight;
pub const max_area_lights = lights.max_area_lights;

pub const material = @import("material.zig");
pub const StandardMaterial = material.StandardMaterial;
pub const PBRMaterial = material.PBRMaterial;
pub const Material = material.Material;
pub const AlphaMode = material.AlphaMode;
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
/// KTX2 container reader: uncompressed LDR subset, block-compressed
/// BC1/BC2/BC3/BC7 / ETC2 RGBA8 / ASTC 4x4 upload-without-transcoding, and Basis
/// Universal transcoding (ETC1S/BasisLZ + UASTC LDR 4x4 → BC7/ASTC/ETC2/RGBA32).
/// Texture.decodeMemory/decodeImageMemory route KTX2 payloads here
/// automatically by magic sniff.
pub const ktx2 = @import("ktx2.zig");
/// DDS container reader: block-compressed BC1/BC2/BC3/BC7 subset (legacy
/// DXT1/DXT3/DXT5 fourCC plus the DX10 header), decoded into the same
/// RawBlockTexture the KTX2 block path uploads. Routed automatically by
/// Texture.decodeImageMemory; standalone uploads via
/// Texture.fromDdsMemory/fromDdsFile (glTF cannot reference .dds).
pub const dds = @import("dds.zig");
/// OpenEXR scanline reader: HALF/FLOAT channels (NONE/RLE/ZIPS/ZIP),
/// decoded to half-float RGBA for the HDR upload path. Routed automatically
/// by Texture.decodeHDRMemory (magic sniff); standalone uploads via
/// Texture.fromExrMemory/fromExrFile.
pub const exr = @import("exr.zig");
/// TrueType (glyf-based) font parser, rasterizer and CPU glyph atlas plus
/// the UI-facing TtfFont handle. UI text switches via
/// UICanvas.setFontTtf; see ttf.zig for the supported/rejected matrix.
pub const ttf = @import("ttf.zig");
/// UI-facing TrueType font handle (same declaration as ui.TtfFont;
/// install on a canvas with UICanvas.setFontTtf).
pub const TtfFont = ttf.TtfFont;
pub const TtfError = ttf.TtfError;
pub const GlyphInfo = ttf.GlyphInfo;
/// KTX2 decode companions: Texture.decodeImageMemory routes KTX2 payloads
/// here automatically; these names cover standalone use.
pub const Ktx2DecodeError = ktx2.DecodeError;
pub const Ktx2DecodeOptions = ktx2.DecodeOptions;
pub const Ktx2BasisDecodeOptions = ktx2.BasisDecodeOptions;
pub const Ktx2BasisInfo = ktx2.BasisInfo;
pub const Ktx2BasisKind = ktx2.BasisKind;
pub const Ktx2BasisTarget = ktx2.BasisTarget;
pub const RawBlockTexture = ktx2.RawBlockTexture;
/// DDS decode companions (Texture.fromDdsMemory wraps decodeBlock2D).
pub const DdsDecodeError = dds.DecodeError;
pub const DdsDecodeOptions = dds.DecodeOptions;
/// OpenEXR decode companions (Texture.decodeHDRMemory wraps decode).
pub const ExrDecodeError = exr.DecodeError;
pub const ExrDecoded = exr.Decoded;
pub const Texture = texture.Texture;
pub const CubeTexture = texture.CubeTexture;
pub const SkyboxOptions = texture.SkyboxOptions;

pub const mesh = @import("mesh.zig");
pub const Mesh = mesh.Mesh;
pub const InstancedMesh = mesh.InstancedMesh;
pub const BoneAttachment = mesh.BoneAttachment;
pub const LODLevel = mesh.LODLevel;
pub const CullingStrategy = mesh.CullingStrategy;
pub const Vertex = mesh.Vertex;
pub const GeometryData = mesh.GeometryData;
pub const uploadGeometry = mesh.uploadGeometry;
pub const buildTorusKnotData = mesh.buildTorusKnotData;
pub const MeshBuilder = mesh.MeshBuilder;
pub const computeNormals = mesh.computeNormals;
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
pub const MorphMode = mesh.MorphMode;
pub const CSG = mesh.CSG;
pub const CSGVertex = mesh.CSGVertex;
pub const CSGPlane = mesh.CSGPlane;
pub const CSGPolygon = mesh.CSGPolygon;
pub const CSGNode = mesh.CSGNode;
pub const vat = mesh.vat;
pub const VatData = mesh.VatData;
pub const VatBaker = mesh.VatBaker;
pub const VatPlayer = mesh.VatPlayer;
pub const VatConfig = mesh.VatConfig;
pub const VatLayout = mesh.VatLayout;
pub const VatSampleParams = mesh.VatSampleParams;
pub const greased_line = mesh.greased_line;
pub const GreasedLineOptions = mesh.GreasedLineOptions;
pub const GreasedLineUVMode = mesh.GreasedLineUVMode;
pub const GreasedLineColorMode = mesh.GreasedLineColorMode;
pub const GreasedLineMesh = mesh.GreasedLineMesh;
pub const buildGreasedLineData = mesh.buildGreasedLineData;
pub const simplify = mesh.simplify;
pub const Quadric3D = mesh.Quadric3D;
pub const SimplifyOptions = mesh.SimplifyOptions;
pub const LODLevelSpec = mesh.LODLevelSpec;
pub const simplifyGeometry = mesh.simplifyGeometry;
pub const simplifyMesh = mesh.simplifyMesh;
pub const generateLODLevels = mesh.generateLODLevels;

pub const scene = @import("scene.zig");
pub const Scene = scene.Scene;
pub const SceneStats = scene.SceneStats;
pub const RenderMeshItem = scene.RenderMeshItem;
pub const ReflectionProbe = scene.ReflectionProbe;
pub const ReflectionProbeOptions = scene.ReflectionProbeOptions;
pub const Ui3dPanel = scene.Ui3dPanel;
pub const Ui3dPanelOptions = scene.Ui3dPanelOptions;
pub const Ui3dFaceMode = scene.Ui3dFaceMode;
pub const Ui3dPickHit = scene.Ui3dPickHit;
pub const HighlightLayer = scene.HighlightLayer;
pub const HighlightEntry = scene.HighlightEntry;
pub const HighlightOptions = scene.HighlightOptions;
pub const SceneFrameSnapshot = scene.SceneFrameSnapshot;
pub const CameraSnapshot = scene.CameraSnapshot;
pub const AllocatorConfig = scene.AllocatorConfig;

pub const tags = @import("tags.zig");
pub const TagSet = tags.TagSet;
pub const TagQuery = tags.TagQuery;

pub const loader = @import("loader/scene_loader.zig");
pub const SceneLoader = loader.SceneLoader;
pub const c = @import("c.zig").c;

pub const postprocess = @import("postprocess.zig");
pub const PostProcessOptions = postprocess.PostProcessOptions;
pub const TonemappingType = postprocess.TonemappingType;
pub const LutFormat = postprocess.LutFormat;
pub const ShaftResolution = postprocess.ShaftResolution;

pub const ssao = @import("ssao.zig");
pub const SSAOOptions = ssao.SSAOOptions;

pub const jobs = @import("jobs.zig");
/// Thin engine-owned frame lifecycle facade (update-vs-begin phase mutex,
/// worker start/stop, producer claim->build->stageUi->publish one-liner,
/// staged begin/finish/cancel with the smallest safe exclusion boundary,
/// serial legacy prepare, reuse fallback, truthful counters). Threaded
/// apps (agate demo, sandbox) compose their frame loop from this instead
/// of reproducing the ordering by hand.
pub const runtime = @import("runtime.zig");
pub const Runtime = runtime.Runtime;
pub const RuntimeMetrics = runtime.Metrics;
pub const RuntimeBeginResult = runtime.BeginResult;
pub const RuntimeFrameResult = runtime.FrameResult;
pub const handoff = @import("handoff.zig");
pub const Handoff = handoff.Handoff;
/// Graphics-context thread marker: apps call `markContextThread()` in their
/// init callback so engine paths can detect off-context GPU touches.
pub const gpu_thread = @import("gpu_thread.zig");
pub const particles = @import("particles.zig");
pub const ParticleSystem = particles.ParticleSystem;
pub const ParticleBlendMode = particles.ParticleBlendMode;
pub const Particle = particles.Particle;
pub const ParticleInstanceData = particles.ParticleInstanceData;
pub const SimulationMode = particles.SimulationMode;
pub const UpdateError = particles.UpdateError;
pub const ComputeModeError = particles.ComputeModeError;
pub const ComputeParticleState = particles.ComputeParticleState;
pub const SubEmitter = particles.SubEmitter;
pub const SubEmitterTrigger = particles.SubEmitterTrigger;
pub const FlowSpace = particles.FlowSpace;
pub const FlowWrap = particles.FlowWrap;
pub const CollisionMode = particles.CollisionMode;
pub const CollisionError = particles.CollisionError;
pub const ParticleSphereCollider = particles.ParticleSphereCollider;
pub const ParticleBoxCollider = particles.ParticleBoxCollider;
pub const ParticlePlaneCollider = particles.ParticlePlaneCollider;
pub const max_box_colliders = particles.max_box_colliders;
pub const max_plane_colliders = particles.max_plane_colliders;
pub const GpuParticleSlot = particles.GpuParticleSlot;

/// Compute-pass support (see compute.zig for the backend matrix).
pub const compute = @import("compute.zig");

pub const Ray = math.Ray;
pub const RayHit = math.RayHit;
pub const TriangleHit = math.TriangleHit;

pub const physics = @import("physics.zig");
pub const RigidBody = physics.RigidBody;
pub const DebugLine = physics.DebugLine;
pub const PhysicsWorld = physics.PhysicsWorld;
pub const PhysicsProfile = physics.PhysicsProfile;
pub const PhysicsCounters = physics.PhysicsCounters;
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
pub const UIState = ui.UIState;
pub const UIStyle = ui.UIStyle;
pub const UIStyleOverride = ui.UIStyleOverride;
pub const UIStyleSet = ui.UIStyleSet;
pub const UIStyleClass = ui.UIStyleClass;
pub const UIStyleKind = ui.UIStyleKind;
pub const UIStyleRequest = ui.UIStyleRequest;
pub const UIStyledOptions = ui.UIStyledOptions;
pub const UIBoxStyle = ui.UIBoxStyle;
pub const UITheme = ui.UITheme;
pub const UIGradient = ui.UIGradient;
pub const UIShadow = ui.UIShadow;
pub const LayoutStack = ui.LayoutStack;
pub const layoutAlignOffset = ui.layoutAlignOffset;
pub const gridExtentSize = ui.gridExtentSize;
pub const gridExtentOffset = ui.gridExtentOffset;
pub const LayoutGridSpec = ui.LayoutGridSpec;
pub const LayoutFlowOptions = ui.LayoutFlowOptions;
pub const LayoutGridOptions = ui.LayoutGridOptions;
pub const LayoutAlign = ui.LayoutAlign;
pub const LayoutAlignCross = ui.LayoutAlignCross;
pub const LayoutFlexOptions = ui.LayoutFlexOptions;
pub const LayoutLabelOptions = ui.LayoutLabelOptions;
pub const LayoutButtonOptions = ui.LayoutButtonOptions;
pub const LayoutCheckboxOptions = ui.LayoutCheckboxOptions;
pub const LayoutSliderOptions = ui.LayoutSliderOptions;
pub const LayoutProgressOptions = ui.LayoutProgressOptions;
pub const LayoutDividerOptions = ui.LayoutDividerOptions;
pub const LayoutBadgeOptions = ui.LayoutBadgeOptions;

// Layout system: flexible dimensions, anchors, docking, flexbox & CSS grid solvers
pub const UISize = ui.UISize;
pub const UIEdges = ui.UIEdges;
pub const UIAnchor = ui.UIAnchor;
pub const UIDock = ui.UIDock;
pub const anchorRect = ui.anchorRect;
pub const dockRect = ui.dockRect;
pub const FlexDirection = ui.FlexDirection;
pub const JustifyContent = ui.JustifyContent;
pub const AlignItems = ui.AlignItems;
pub const LayoutItem = ui.LayoutItem;
pub const solveFlex = ui.solveFlex;
pub const GridTrack = ui.GridTrack;
pub const solveGridTracks = ui.solveGridTracks;
pub const AdvancedGridSpec = ui.AdvancedGridSpec;
// Style system phase 2: CSS theming + style transitions.
pub const TransitionOptions = ui.TransitionOptions;
pub const UIStyleTransition = ui.UIStyleTransition;
pub const max_style_transitions = ui.max_style_transitions;
pub const lerpStyle = ui.lerpStyle;
pub const styleEql = ui.styleEql;
pub const CssTheme = ui.CssTheme;
pub const CssClass = ui.CssClass;
pub const CssDiag = ui.CssDiag;
pub const CssDiagKind = ui.CssDiagKind;
pub const parseCss = ui.parseCss;
pub const loadThemeFile = ui.loadThemeFile;

pub const passes = @import("passes/mod.zig");
pub const DebugPass = passes.DebugPass;
pub const BloomPass = passes.BloomPass;
pub const HighlightPass = passes.HighlightPass;
pub const OutlinePass = passes.OutlinePass;
pub const outline_pass = passes.outline_pass; // file-level: shouldOutlineMesh etc.
pub const highlight_pass = passes.highlight_pass; // file-level: makeHighlightDrawItem etc.

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
pub const TranslationMode = animation.TranslationMode;
pub const BoneMap = animation.BoneMap;
pub const RetargetOptions = animation.RetargetOptions;
pub const retargetAnimationGroup = animation.retargetAnimationGroup;
pub const boneRestLength = animation.boneRestLength;
pub const translationScaleFactor = animation.translationScaleFactor;

pub const Ragdoll = @import("ragdoll.zig").Ragdoll;
pub const RagdollOptions = @import("ragdoll.zig").RagdollOptions;
pub const RagdollPart = @import("ragdoll.zig").RagdollPart;
pub const RaycastVehicle = @import("vehicle.zig").RaycastVehicle;
pub const VehicleOptions = @import("vehicle.zig").VehicleOptions;

pub const audio = @import("audio.zig");
pub const AudioEngine = audio.AudioEngine;
pub const AudioConfig = audio.AudioConfig;
pub const AudioEngineConfig = audio.AudioEngineConfig;
pub const default_max_buses = audio.default_max_buses;
pub const max_bus_capacity = audio.max_bus_capacity;
pub const max_buses = audio.max_buses;
pub const AudioClip = audio.AudioClip;
pub const VoiceKind = audio.VoiceKind;
pub const Voice = audio.Voice;
pub const AudioBus = audio.AudioBus;
pub const BusId = audio.BusId;
pub const invalid_bus = audio.invalid_bus;
pub const BusConfig = audio.BusConfig;
pub const BusAttenuation = audio.BusAttenuation;
pub const AttenuationModel = audio.AttenuationModel;
pub const BiquadFilterType = audio.BiquadFilterType;
pub const BusFilterConfig = audio.BusFilterConfig;
pub const BusReverbConfig = audio.BusReverbConfig;
pub const BiquadFilter = audio.BiquadFilter;
pub const ReverbProcessor = audio.ReverbProcessor;
pub const AudioOcclusionConfig = audio.AudioOcclusionConfig;
pub const RaycastFn = audio.RaycastFn;
pub const evaluateRaycastOcclusion = audio.evaluateRaycastOcclusion;
pub const AudioOcclusionTracker = audio.AudioOcclusionTracker;
pub const AudioEmitter = audio.AudioEmitter;
pub const StreamFormat = audio.StreamFormat;
pub const StreamState = audio.StreamState;
pub const StreamError = audio.StreamError;
pub const StreamOptions = audio.StreamOptions;
pub const PlaySoundOptions = audio.PlaySoundOptions;
pub const AudioStream = audio.AudioStream;

pub const serialization = @import("serialization.zig");
pub const SceneState = serialization.SceneState;
pub const GameProperty = serialization.GameProperty;
pub const captureSceneState = serialization.capture;
pub const restoreSceneState = serialization.restore;
pub const serializeSceneState = serialization.serializeAlloc;
pub const deserializeSceneState = serialization.deserializeAlloc;
pub const saveSceneStateFile = serialization.saveFile;
pub const loadSceneStateFile = serialization.loadFile;
pub const saveSceneStateFileAsync = serialization.saveFileAsync;
pub const loadSceneStateFileAsync = serialization.loadFileAsync;
pub const AsyncSaveTask = serialization.AsyncSaveTask;
pub const AsyncLoadTask = serialization.AsyncLoadTask;

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
pub const export_glb = @import("export/glb.zig");
pub const ObjExportOptions = export_obj.ObjExportOptions;
pub const StlExportOptions = export_stl.StlExportOptions;
pub const GlbExportOptions = export_glb.GlbExportOptions;
pub const writeObjAlloc = export_obj.writeObjAlloc;
pub const writeMtlAlloc = export_obj.writeMtlAlloc;
pub const writeStlAlloc = export_stl.writeStlAlloc;
pub const writeStlAsciiAlloc = export_stl.writeStlAsciiAlloc;
pub const writeStlBinaryAlloc = export_stl.writeStlBinaryAlloc;
pub const writeGlbAlloc = export_glb.writeGlbAlloc;
pub const writeGlbMeshAlloc = export_glb.writeGlbMeshAlloc;

pub const assets = @import("assets.zig");
pub const AsyncTexturePipeline = assets.AsyncTexturePipeline;
pub const asset_manager = @import("asset_manager.zig");
pub const AssetManager = asset_manager.AssetManager;
pub const AssetTask = asset_manager.AssetTask;
pub const AssetCache = asset_manager.AssetCache;
pub const TaskState = asset_manager.TaskState;
pub const TaskType = asset_manager.TaskType;

pub const ai = @import("ai.zig");
pub const NavNode = ai.NavNode;
pub const NavMesh = ai.NavMesh;
pub const Portal = ai.Portal;
pub const triArea2D = ai.triArea2D;
pub const stringPull = ai.stringPull;
pub const Pathfinding = ai.Pathfinding;
pub const NavAgent = ai.NavAgent;
pub const Crowd = ai.Crowd;
pub const CrowdAgent = ai.CrowdAgent;
pub const CrowdAgentParams = ai.CrowdAgentParams;

pub const visibility = @import("visibility/mod.zig");
pub const HiZBuffer = visibility.HiZBuffer;
pub const SoftwareRasterizer = visibility.SoftwareRasterizer;
pub const OcclusionCuller = visibility.OcclusionCuller;

/// PBD cloth soft bodies (wave 29, v1): solver + mesh coupling + scene layer.
pub const softbody = @import("softbody.zig");
pub const SoftBody = softbody.SoftBody;
pub const SoftBodyLayer = softbody.SoftBodyLayer;
pub const SoftBodyError = softbody.SoftBodyError;
pub const Cloth = softbody.Cloth;
pub const ClothOptions = softbody.ClothOptions;
pub const SphereCollider = softbody.SphereCollider;

pub const profiler = @import("profiler.zig");
pub const Profiler = profiler.Profiler;
pub const FrameRecord = profiler.FrameRecord;
pub const MemorySnapshot = profiler.MemorySnapshot;
pub const SessionSummary = profiler.SessionSummary;
pub const DiagnosticFinding = profiler.DiagnosticFinding;
pub const DiagnosticSeverity = profiler.DiagnosticSeverity;
pub const ReportWriteTask = profiler.ReportWriteTask;
pub const ReportFiles = profiler.ReportFiles;

/// GPU frame timings v1 (vendored sokol patch, default off).
pub const gpu_timing = @import("gpu_timing.zig");

pub const sokol = @import("sokol");

test {
    _ = @import("tests.zig");
}

// Every root re-export added for recent Scene APIs: reference each symbol
// so future removals/renames break loudly here instead of downstream.
test "root re-exports cover recent Scene APIs" {
    _ = ClusteredPointLightOptions;
    _ = ClusteredPointLight;
    _ = max_clustered_lights;
    _ = AreaLightOptions;
    _ = AreaLight;
    _ = max_area_lights;
    _ = sunDirectionFromAngles;
    _ = colorTemperatureToRgb;
    _ = ReflectionProbe;
    _ = ReflectionProbeOptions;
    _ = Ui3dPanel;
    _ = Ui3dPanelOptions;
    _ = Ui3dFaceMode;
    _ = Ui3dPickHit;
    _ = HighlightLayer;
    _ = HighlightEntry;
    _ = HighlightOptions;
    _ = HighlightPass;
    _ = highlight_pass;
    _ = SceneFrameSnapshot;
    _ = CameraSnapshot;
    _ = UpdateError;
    _ = ComputeModeError;
    _ = ComputeParticleState;
    _ = SubEmitter;
    _ = SubEmitterTrigger;
    _ = FlowSpace;
    _ = FlowWrap;
    _ = CollisionMode;
    _ = CollisionError;
    _ = ParticleSphereCollider;
    _ = ParticleBoxCollider;
    _ = ParticlePlaneCollider;
    _ = max_box_colliders;
    _ = max_plane_colliders;
    _ = GpuParticleSlot;
    _ = TtfFont;
    _ = TtfError;
    _ = GlyphInfo;
    _ = Ktx2DecodeError;
    _ = Ktx2DecodeOptions;
    _ = RawBlockTexture;
    _ = DdsDecodeError;
    _ = DdsDecodeOptions;
    _ = ExrDecodeError;
    _ = ExrDecoded;
    _ = LutFormat;
    _ = layoutAlignOffset;
    _ = gridExtentSize;
    _ = gridExtentOffset;
    _ = GlbExportOptions;
    _ = writeGlbAlloc;
    _ = writeGlbMeshAlloc;
    _ = AssetManager;
    _ = AssetTask;
    _ = AssetCache;
    _ = Crowd;
    _ = CrowdAgent;
    _ = CrowdAgentParams;
    _ = CameraRig;
    _ = CameraRigMode;
    _ = StereoConvergenceMode;
    _ = CameraRigSlot;
    _ = TagSet;
    _ = TagQuery;
    _ = Handoff;
    _ = AllocatorConfig;
}

test "root re-exports cover the runtime frame facade" {
    _ = Runtime;
    _ = RuntimeMetrics;
    _ = RuntimeBeginResult;
    _ = RuntimeFrameResult;
    _ = runtime.BeginResult;
    _ = runtime.FrameResult;
}
