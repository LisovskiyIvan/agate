const std = @import("std");
const sokol = @import("sokol");
const sapp = sokol.app;

const math = @import("math");
const Vec3 = math.Vec3;
const Ray = math.Ray;
const RayHit = math.RayHit;
const Camera = @import("../camera.zig").Camera;
const Mesh = @import("../mesh.zig").Mesh;
const physics = @import("../physics.zig");
const PickingInfo = physics.PickingInfo;
const PhysicsWorld = physics.PhysicsWorld;

/// CPU mouse picking. Free functions so the logic stays testable without
/// scene.zig; Scene passes its mesh list and optional physics world.
pub fn createPickingRay(cam: Camera, screen_x: f32, screen_y: f32) Ray {
    const w = sapp.widthf();
    const h = sapp.heightf();
    if (w <= 0.0 or h <= 0.0) return Ray.new(cam.getPosition(), Vec3.forward);

    const aspect = w / h;
    const vp = cam.getViewProjection(aspect);
    const inv_vp = vp.invert() orelse return Ray.new(cam.getPosition(), Vec3.forward);

    const ndc_x = (2.0 * screen_x) / w - 1.0;
    const ndc_y = 1.0 - (2.0 * screen_y) / h;

    const near_pt = inv_vp.transformPoint(Vec3.new(ndc_x, ndc_y, 0.0));
    const far_pt = inv_vp.transformPoint(Vec3.new(ndc_x, ndc_y, 1.0));
    const dir = far_pt.sub(near_pt).normalize();

    return Ray.new(near_pt, dir);
}

/// Closest hit across visible, non-instanced, non-decal meshes. A sphere
/// collider on the mesh's rigid body replaces the AABB test (tighter fit).
pub fn pickWithRay(meshes: []const *Mesh, world: ?*PhysicsWorld, r: Ray) PickingInfo {
    var closest_dist: f32 = std.math.inf(f32);
    var best_hit: ?RayHit = null;
    var best_mesh: ?*Mesh = null;

    for (meshes) |mesh| {
        if (!mesh.is_visible or mesh.is_lod_child or mesh.is_decal) continue;

        if (mesh.instances.items.len > 0) {
            continue;
        }

        const model = mesh.getWorldMatrix();
        const world_aabb = mesh.local_bounding_box.transform(model);

        if (if (world) |pw| pw.findBody(mesh) else null) |body| {
            if (body.collider == .sphere) {
                const radius = body.sphere_radius * mesh.scaling.x;
                if (r.intersectsSphereNormal(mesh.position, radius)) |hit| {
                    if (hit.distance < closest_dist) {
                        closest_dist = hit.distance;
                        best_hit = hit;
                        best_mesh = mesh;
                    }
                }
                continue;
            }
        }

        if (r.intersectsAABBNormal(world_aabb)) |hit| {
            if (hit.distance < closest_dist) {
                closest_dist = hit.distance;
                best_hit = hit;
                best_mesh = mesh;
            }
        }
    }

    if (best_hit) |hit| {
        return PickingInfo{
            .hit = true,
            .distance = hit.distance,
            .picked_point = hit.point,
            .picked_normal = hit.normal,
            .picked_mesh = best_mesh,
        };
    }

    return PickingInfo{};
}
