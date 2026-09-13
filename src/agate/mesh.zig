const std = @import("std");

pub const types = @import("mesh/types.zig");
pub const Vertex = types.Vertex;
pub const CullingStrategy = types.CullingStrategy;
pub const MAX_MORPH_TARGETS = types.MAX_MORPH_TARGETS;
pub const MorphTarget = types.MorphTarget;
pub const MorphMode = types.MorphMode;
pub const InstancedMesh = types.InstancedMesh;
pub const BoneAttachment = types.BoneAttachment;
pub const GeometryData = types.GeometryData;
pub const LODLevel = types.LODLevel;
pub const SkinJointWeight = types.SkinJointWeight;

pub const tangents = @import("mesh/tangents.zig");
pub const computeTangents = tangents.computeTangents;
pub const pickOrthogonal = tangents.pickOrthogonal;

pub const mesh_impl = @import("mesh/mesh.zig");
pub const Mesh = mesh_impl.Mesh;
pub const uploadGeometry = mesh_impl.uploadGeometry;

/// GPU morph blending: delta-texture packing plus the pure Zig mirror of
/// the vertex-shader blend (opt-in via Mesh.morph_mode == .gpu).
pub const morph_gpu = @import("mesh/morph_gpu.zig");

pub const builders = @import("mesh/builders.zig");
pub const BoxOptions = builders.BoxOptions;
pub const SphereOptions = builders.SphereOptions;
pub const GroundOptions = builders.GroundOptions;
pub const TerrainOptions = builders.TerrainOptions;
pub const CylinderOptions = builders.CylinderOptions;
pub const CapsuleOptions = builders.CapsuleOptions;
pub const TorusOptions = builders.TorusOptions;
pub const TorusKnotOptions = builders.TorusKnotOptions;
pub const DiscOptions = builders.DiscOptions;
pub const RibbonOptions = builders.RibbonOptions;
pub const LatheOptions = builders.LatheOptions;
pub const PlaneOptions = builders.PlaneOptions;
pub const TubeOptions = builders.TubeOptions;
pub const LinesOptions = builders.LinesOptions;
pub const ExtrudeOptions = builders.ExtrudeOptions;
pub const PolygonSideOrientation = builders.PolygonSideOrientation;
pub const PolygonPlane = builders.PolygonPlane;
pub const PolygonOptions = builders.PolygonOptions;

// Quad/seed helpers shared by the builders and used by mesh tests.
pub const storeQuad = builders.storeQuad;
pub const appendGridQuad = builders.appendGridQuad;
pub const appendGridQuadFlipped = builders.appendGridQuadFlipped;
pub const resolveFrameSeed = builders.resolveFrameSeed;

// Pure data builders: assemble a GeometryData without touching the GPU.
// (Box/Ground/Terrain/Sphere/Cylinder/Capsule variants are internal to
// mesh/builders.zig; MeshBuilder wraps them in create* factories.)
pub const buildPlaneData = builders.buildPlaneData;
pub const buildTorusData = builders.buildTorusData;
pub const buildTorusKnotData = builders.buildTorusKnotData;
pub const buildDiscData = builders.buildDiscData;
pub const buildRibbonData = builders.buildRibbonData;
pub const buildLatheData = builders.buildLatheData;
pub const buildTubeData = builders.buildTubeData;
pub const buildLinesData = builders.buildLinesData;
pub const buildExtrudeData = builders.buildExtrudeData;
pub const buildPolygonData = builders.buildPolygonData;

pub const builder = @import("mesh/builder.zig");
pub const MeshBuilder = builder.MeshBuilder;

pub const decal = @import("mesh/decal.zig");
pub const DecalOptions = decal.DecalOptions;
pub const buildDecalData = decal.buildDecalData;
pub const createDecal = decal.createDecal;
pub const DecalProjector = decal.DecalProjector;
pub const DecalSpawnOptions = decal.DecalSpawnOptions;
pub const DecalInstance = decal.DecalInstance;
pub const DecalManager = decal.DecalManager;
pub const barycentric = decal.barycentric;
pub const blendSkinWeights = decal.blendSkinWeights;

pub const trail = @import("mesh/trail.zig");
pub const TrailOptions = trail.TrailOptions;
pub const TrailMesh = trail.TrailMesh;
pub const TrailNode = trail.TrailNode;

pub const csg = @import("mesh/csg.zig");
pub const CSG = csg.CSG;
pub const CSGVertex = csg.CSGVertex;
pub const CSGPlane = csg.CSGPlane;
pub const CSGPolygon = csg.CSGPolygon;
pub const CSGNode = csg.CSGNode;

test {
    _ = @import("mesh/tests.zig");
    _ = @import("mesh/csg_tests.zig");
}
