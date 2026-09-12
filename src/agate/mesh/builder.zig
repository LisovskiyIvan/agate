const std = @import("std");
const Scene = @import("../scene.zig").Scene;
const Mesh = @import("mesh.zig").Mesh;
const uploadGeometry = @import("mesh.zig").uploadGeometry;

const builders = @import("builders.zig");
const decal = @import("decal.zig");
const trail = @import("trail.zig");
const csg = @import("csg.zig");
pub const DecalOptions = decal.DecalOptions;
pub const buildDecalData = decal.buildDecalData;
pub const TrailOptions = trail.TrailOptions;
pub const TrailMesh = trail.TrailMesh;
pub const TrailNode = trail.TrailNode;
pub const CSG = csg.CSG;
pub const CSGVertex = csg.CSGVertex;
pub const CSGPlane = csg.CSGPlane;
pub const CSGPolygon = csg.CSGPolygon;
pub const CSGNode = csg.CSGNode;
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
pub const PolygonOptions = builders.PolygonOptions;
pub const PolygonSideOrientation = builders.PolygonSideOrientation;
pub const PolygonPlane = builders.PolygonPlane;
pub const buildPolygonData = builders.buildPolygonData;

pub const MeshBuilder = struct {
    pub fn createBox(scene: *Scene, name: []const u8, options: BoxOptions) !*Mesh {
        var data = try builders.buildBoxData(scene.allocator, options);
        defer data.deinit(scene.allocator);
        return uploadGeometry(scene, name, data);
    }

    pub fn createGround(scene: *Scene, name: []const u8, options: GroundOptions) !*Mesh {
        var data = try builders.buildGroundData(scene.allocator, options);
        defer data.deinit(scene.allocator);
        return uploadGeometry(scene, name, data);
    }

    /// Builds a height-map grid mesh. `heights` is row-major:
    /// index = row * count_x + column, matching the Box3D height field layout.
    /// The local origin is the (0, min_height, 0) corner.
    pub fn createTerrain(
        scene: *Scene,
        name: []const u8,
        heights: []const f32,
        count_x: u32,
        count_z: u32,
        options: TerrainOptions,
    ) !*Mesh {
        var data = try builders.buildTerrainData(scene.allocator, heights, count_x, count_z, options);
        defer data.deinit(scene.allocator);
        return uploadGeometry(scene, name, data);
    }

    pub fn createSphere(scene: *Scene, name: []const u8, options: SphereOptions) !*Mesh {
        var data = try builders.buildSphereData(scene.allocator, options);
        defer data.deinit(scene.allocator);
        return uploadGeometry(scene, name, data);
    }

    pub fn createCylinder(scene: *Scene, name: []const u8, options: CylinderOptions) !*Mesh {
        var data = try builders.buildCylinderData(scene.allocator, options);
        defer data.deinit(scene.allocator);
        return uploadGeometry(scene, name, data);
    }

    pub fn createCapsule(scene: *Scene, name: []const u8, options: CapsuleOptions) !*Mesh {
        var data = try builders.buildCapsuleData(scene.allocator, options);
        defer data.deinit(scene.allocator);
        return uploadGeometry(scene, name, data);
    }

    pub fn createTorus(scene: *Scene, name: []const u8, options: TorusOptions) !*Mesh {
        var data = try builders.buildTorusData(scene.allocator, options);
        defer data.deinit(scene.allocator);
        return uploadGeometry(scene, name, data);
    }

    pub fn createTorusKnot(scene: *Scene, name: []const u8, options: TorusKnotOptions) !*Mesh {
        var data = try builders.buildTorusKnotData(scene.allocator, options);
        defer data.deinit(scene.allocator);
        return uploadGeometry(scene, name, data);
    }

    pub fn createDisc(scene: *Scene, name: []const u8, options: DiscOptions) !*Mesh {
        var data = try builders.buildDiscData(scene.allocator, options);
        defer data.deinit(scene.allocator);
        return uploadGeometry(scene, name, data);
    }

    pub fn createRibbon(scene: *Scene, name: []const u8, options: RibbonOptions) !*Mesh {
        var data = try builders.buildRibbonData(scene.allocator, options);
        defer data.deinit(scene.allocator);
        return uploadGeometry(scene, name, data);
    }

    pub fn createLathe(scene: *Scene, name: []const u8, options: LatheOptions) !*Mesh {
        var data = try builders.buildLatheData(scene.allocator, options);
        defer data.deinit(scene.allocator);
        return uploadGeometry(scene, name, data);
    }

    pub fn createPlane(scene: *Scene, name: []const u8, options: PlaneOptions) !*Mesh {
        var data = try builders.buildPlaneData(scene.allocator, options);
        defer data.deinit(scene.allocator);
        return uploadGeometry(scene, name, data);
    }

    pub fn createTube(scene: *Scene, name: []const u8, options: TubeOptions) !*Mesh {
        var data = try builders.buildTubeData(scene.allocator, options);
        defer data.deinit(scene.allocator);
        return uploadGeometry(scene, name, data);
    }

    pub fn createLines(scene: *Scene, name: []const u8, options: LinesOptions) !*Mesh {
        var data = try builders.buildLinesData(scene.allocator, options);
        defer data.deinit(scene.allocator);
        return uploadGeometry(scene, name, data);
    }

    pub fn createExtrude(scene: *Scene, name: []const u8, options: ExtrudeOptions) !*Mesh {
        var data = try builders.buildExtrudeData(scene.allocator, options);
        defer data.deinit(scene.allocator);
        return uploadGeometry(scene, name, data);
    }

    pub fn createPolygon(scene: *Scene, name: []const u8, options: PolygonOptions) !*Mesh {
        var data = try builders.buildPolygonData(scene.allocator, options);
        defer data.deinit(scene.allocator);
        return uploadGeometry(scene, name, data);
    }

    pub fn createTrail(scene: *Scene, name: []const u8, options: TrailOptions) !*TrailMesh {
        return scene.createTrailMesh(name, options);
    }

    pub fn createDecal(scene: *Scene, name: []const u8, target_mesh: *const Mesh, options: DecalOptions) !*Mesh {
        return decal.createDecal(scene, name, target_mesh, options);
    }

    pub fn createCSG(scene: *Scene, name: []const u8, csg_solid: *const CSG) !*Mesh {
        return csg_solid.toMesh(scene, name);
    }
};
