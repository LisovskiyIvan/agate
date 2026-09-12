const std = @import("std");

pub const types = @import("mesh/types.zig");
pub const Vertex = types.Vertex;
pub const CullingStrategy = types.CullingStrategy;
pub const MAX_MORPH_TARGETS = types.MAX_MORPH_TARGETS;
pub const MorphTarget = types.MorphTarget;
pub const InstancedMesh = types.InstancedMesh;
pub const BoneAttachment = types.BoneAttachment;
pub const GeometryData = types.GeometryData;
pub const LODLevel = types.LODLevel;

pub const tangents = @import("mesh/tangents.zig");
pub const computeTangents = tangents.computeTangents;
pub const pickOrthogonal = tangents.pickOrthogonal;
pub const orthogonal_dot_threshold = tangents.orthogonal_dot_threshold;

pub const mesh_impl = @import("mesh/mesh.zig");
pub const Mesh = mesh_impl.Mesh;
pub const uploadGeometry = mesh_impl.uploadGeometry;

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

pub const TrigEntry = builders.TrigEntry;
pub const trigEntry = builders.trigEntry;
pub const buildTrigTable = builders.buildTrigTable;
pub const storeQuad = builders.storeQuad;
pub const storeQuadFlipped = builders.storeQuadFlipped;
pub const appendGridQuad = builders.appendGridQuad;
pub const appendGridQuadFlipped = builders.appendGridQuadFlipped;
pub const resolveFrameSeed = builders.resolveFrameSeed;

pub const buildBoxData = builders.buildBoxData;
pub const buildGroundData = builders.buildGroundData;
pub const buildTerrainData = builders.buildTerrainData;
pub const buildSphereData = builders.buildSphereData;
pub const buildCylinderData = builders.buildCylinderData;
pub const buildCapsuleData = builders.buildCapsuleData;
pub const buildPlaneData = builders.buildPlaneData;
pub const buildTorusData = builders.buildTorusData;
pub const buildTorusKnotData = builders.buildTorusKnotData;
pub const buildDiscData = builders.buildDiscData;
pub const buildRibbonData = builders.buildRibbonData;
pub const buildLatheData = builders.buildLatheData;
pub const buildTubeData = builders.buildTubeData;
pub const buildLinesData = builders.buildLinesData;
pub const buildExtrudeData = builders.buildExtrudeData;

pub const builder = @import("mesh/builder.zig");
pub const MeshBuilder = builder.MeshBuilder;

pub const decal = @import("mesh/decal.zig");
pub const DecalOptions = decal.DecalOptions;
pub const buildDecalData = decal.buildDecalData;
pub const createDecal = decal.createDecal;

test {
    _ = @import("mesh/tests.zig");
}
