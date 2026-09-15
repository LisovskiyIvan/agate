const std = @import("std");
const sokol = @import("sokol");
const sapp = sokol.app;

const math = @import("math");
const Vec3 = math.Vec3;
const Mat4 = math.Mat4;
const Ray = math.Ray;
const RayHit = math.RayHit;
const Camera = @import("../camera.zig").Camera;
const Viewport = @import("../camera.zig").Viewport;
const Mesh = @import("../mesh.zig").Mesh;
const InstancedMesh = @import("../mesh.zig").InstancedMesh;
const physics = @import("../physics.zig");
const PickingInfo = physics.PickingInfo;
const PhysicsWorld = physics.PhysicsWorld;

/// Coordinate convention: `screen_x`/`screen_y` and the framebuffer size are
/// in the same pixel space as `sapp.widthf()`/`sapp.heightf()` (sokol reports
/// absolute mouse coordinates in framebuffer pixels, including high-DPI
/// mode; origin top-left, y down). Viewport `y` follows the same convention as
/// `Viewport.toPixelRect` + `sg.applyViewport(..., origin_top_left=true)`,
/// and the rect itself uses the same policy (round to whole pixels, minimum
/// 1px extent), so containment, ray aspect, and NDC all match the rect
/// rendering actually draws.
///
/// Pixel rectangle for a normalized viewport, mirroring
/// `Viewport.toPixelRect` exactly (same rounding, same 1px minimum) so
/// picking agrees with the viewport/scissor rects and aspect the render
/// path derives from the rounded rect.
fn viewportPixelRect(viewport: Viewport, fb_w: f32, fb_h: f32) struct { x: f32, y: f32, w: f32, h: f32 } {
    return .{
        .x = @round(viewport.x * fb_w),
        .y = @round(viewport.y * fb_h),
        .w = @max(1.0, @round(viewport.width * fb_w)),
        .h = @max(1.0, @round(viewport.height * fb_h)),
    };
}

/// Viewport-aware picking ray: unprojects the cursor relative to the given
/// viewport rect (not the whole window) using the viewport's own aspect, so
/// PIP/sub-view renders pick where they draw. Falls back to a forward ray
/// from the camera position for degenerate sizes.
pub fn createPickingRayViewport(cam: Camera, screen_x: f32, screen_y: f32, fb_w: f32, fb_h: f32, viewport: Viewport) Ray {
    if (fb_w <= 0.0 or fb_h <= 0.0) return Ray.new(cam.getPosition(), Vec3.forward);
    const r = viewportPixelRect(viewport, fb_w, fb_h);

    const aspect = r.w / r.h;
    const vp = cam.getViewProjection(aspect);
    const inv_vp = vp.invert() orelse return Ray.new(cam.getPosition(), Vec3.forward);

    const ndc_x = (2.0 * (screen_x - r.x)) / r.w - 1.0;
    const ndc_y = 1.0 - (2.0 * (screen_y - r.y)) / r.h;

    const near_pt = inv_vp.transformPoint(Vec3.new(ndc_x, ndc_y, 0.0));
    const far_pt = inv_vp.transformPoint(Vec3.new(ndc_x, ndc_y, 1.0));
    const dir = far_pt.sub(near_pt).normalize();

    return Ray.new(near_pt, dir);
}

/// True when the cursor lies inside the viewport's pixel rect.
pub fn viewportContainsPoint(viewport: Viewport, screen_x: f32, screen_y: f32, fb_w: f32, fb_h: f32) bool {
    if (fb_w <= 0.0 or fb_h <= 0.0) return false;
    const r = viewportPixelRect(viewport, fb_w, fb_h);
    return screen_x >= r.x and screen_x < r.x + r.w and screen_y >= r.y and screen_y < r.y + r.h;
}

/// Picks the visually relevant camera entry under the cursor: entries are
/// any slice whose elements expose `.viewport: Viewport` and
/// `.enabled: bool` (Scene passes its CameraEntry list directly). Draw order
/// is the Scene multi-camera order — the active camera first, then the rest
/// in index order — and the LAST enabled entry containing the cursor wins
/// (topmost overlay). Returns null when no enabled entry covers the cursor.
pub fn selectPickIndex(entries: anytype, active_idx: ?usize, screen_x: f32, screen_y: f32, fb_w: f32, fb_h: f32) ?usize {
    const n = entries.len;
    var best: ?usize = null;
    if (active_idx) |ai| {
        if (ai < n and entries[ai].enabled and viewportContainsPoint(entries[ai].viewport, screen_x, screen_y, fb_w, fb_h)) {
            best = ai;
        }
    }
    for (0..n) |i| {
        if (active_idx != null and i == active_idx.?) continue;
        if (!entries[i].enabled) continue;
        if (viewportContainsPoint(entries[i].viewport, screen_x, screen_y, fb_w, fb_h)) {
            best = i;
        }
    }
    return best;
}

/// CPU mouse picking. Free functions so the logic stays testable without
/// scene.zig; Scene passes its mesh list and optional physics world.
pub fn createPickingRay(cam: Camera, screen_x: f32, screen_y: f32) Ray {
    const w = sapp.widthf();
    const h = sapp.heightf();
    // Fullscreen viewport: identical to the legacy whole-window formula.
    return createPickingRayViewport(cam, screen_x, screen_y, w, h, .{});
}

/// Conservative sphere fallback for singular/degenerate world transforms:
/// a world-space sphere at the model translation whose radius is scaled by
/// the Frobenius norm of the linear part. The Frobenius norm upper-bounds
/// the largest singular value, so this can only over-hit, never under-hit.
fn conservativeSphereHit(r: Ray, model: Mat4, local_radius: f32) ?RayHit {
    const sx2 = model.m[0] * model.m[0] + model.m[1] * model.m[1] + model.m[2] * model.m[2];
    const sy2 = model.m[4] * model.m[4] + model.m[5] * model.m[5] + model.m[6] * model.m[6];
    const sz2 = model.m[8] * model.m[8] + model.m[9] * model.m[9] + model.m[10] * model.m[10];
    return r.intersectsSphereNormal(model.getTranslation(), local_radius * @sqrt(sx2 + sy2 + sz2));
}

/// Exact local-space sphere test shared by the plain-mesh and instance
/// paths: maps the world ray into the mesh's local space with the inverse
/// world matrix and intersects the local sphere (centered at the local
/// origin, radius = collider radius). The map is affine and the world
/// direction is unit length, so the ray parameter t equals the world-space
/// distance — no local re-normalization, which would corrupt the parameter
/// under non-uniform scale. Follows parent chains, bone attachment,
/// rotation and scale exactly. Singular transforms (|det| < 1e-8) and
/// numerically collapsed local directions use the conservative fallback.
fn sphereHit(r: Ray, model: Mat4, radius: f32) ?RayHit {
    const im = model.invert() orelse
        return conservativeSphereHit(r, model, radius);
    const o = r.origin;
    const d = r.direction;
    const lo = im.transformPoint(o);
    const ld = Vec3.new(
        im.m[0] * d.x + im.m[4] * d.y + im.m[8] * d.z,
        im.m[1] * d.x + im.m[5] * d.y + im.m[9] * d.z,
        im.m[2] * d.x + im.m[6] * d.y + im.m[10] * d.z,
    );
    const a = ld.lengthSq();
    if (a <= 1e-12) return conservativeSphereHit(r, model, radius);
    const b = lo.dot(ld);
    const cc = lo.lengthSq() - radius * radius;
    const disc = b * b - a * cc;
    if (disc < 0.0) return null;
    const sq = @sqrt(disc);
    var t = (-b - sq) / a;
    if (t < 0.0) t = (-b + sq) / a;
    if (t < 0.0) return null;
    // Local surface normal -> world via the inverse-transpose of the
    // linear part.
    const n_loc = lo.add(ld.scale(t)).normalize();
    const n_world = Vec3.new(
        im.m[0] * n_loc.x + im.m[1] * n_loc.y + im.m[2] * n_loc.z,
        im.m[4] * n_loc.x + im.m[5] * n_loc.y + im.m[6] * n_loc.z,
        im.m[8] * n_loc.x + im.m[9] * n_loc.y + im.m[10] * n_loc.z,
    ).normalize();
    return RayHit{
        .distance = t,
        .point = r.getPoint(t),
        .normal = n_world,
    };
}

/// Closest hit across visible, non-decal meshes, including every visible
/// instance of instance-bearing meshes. A sphere collider on the source
/// mesh's rigid body replaces the AABB test (tighter fit) for plain meshes
/// and instances alike.
///
/// Instance picking uses the drawn model matrix: each candidate is refreshed
/// with the same CPU-only `updateCachedTransforms()` the render path calls,
/// and the ray tests `cached_bounding_box` / the source-local sphere in that
/// exact space. The cached matrix is `TRS(instance) * source.base_matrix` —
/// the source mesh's own TRS and parent chain never reach the instanced draw,
/// so they must not move picks either. Resolution is AABB-level, matching the
/// mesh-level AABB path (no triangle tests exist in picking). The nearest hit
/// across meshes AND instances wins via the shared `closest_dist`;
/// `picked_instance` reports the raw `instances.items` index (invisible slots
/// included) and stays null for plain meshes.
///
/// Threading: instance picking refreshes per-instance transform caches in
/// place, so callers must hold phase ownership (the same `phase_mutex` the
/// render and simulation phases serialize on) — calling Scene.pick from an
/// unrelated thread would race the render-side instance staging.
pub fn pickWithRay(meshes: []const *Mesh, world: ?*PhysicsWorld, r: Ray) PickingInfo {
    var closest_dist: f32 = std.math.inf(f32);
    var best_hit: ?RayHit = null;
    var best_mesh: ?*Mesh = null;
    var best_instance: ?usize = null;

    for (meshes) |mesh| {
        if (mesh.is_lod_child or mesh.is_decal) continue;
        // Deferred-creation meshes have no buffers yet and are skipped by
        // every render path; picking must not see them before they draw.
        if (mesh.gpu_pending) continue;

        if (mesh.instances.items.len > 0) {
            // Instanced draws are gated per instance, not by the source
            // template's own is_visible flag: a hidden source with visible
            // instances still renders, so it must still be pickable.
            var sphere_radius: ?f32 = null;
            if (world) |pw| {
                if (pw.findBody(mesh)) |body| {
                    if (body.collider == .sphere) sphere_radius = body.sphere_radius;
                }
            }
            for (mesh.instances.items, 0..) |inst, idx| {
                if (!inst.is_visible) continue;
                // Same CPU-only refresh the render path performs; the cached
                // matrix is exactly the model matrix the instanced draw
                // uploads, so picks land where instances draw.
                inst.updateCachedTransforms();
                if (sphere_radius) |radius| {
                    if (sphereHit(r, inst.cached_world_matrix, radius)) |hit| {
                        if (hit.distance < closest_dist) {
                            closest_dist = hit.distance;
                            best_hit = hit;
                            best_mesh = mesh;
                            best_instance = idx;
                        }
                    }
                    continue;
                }
                if (r.intersectsAABBNormal(inst.cached_bounding_box)) |hit| {
                    if (hit.distance < closest_dist) {
                        closest_dist = hit.distance;
                        best_hit = hit;
                        best_mesh = mesh;
                        best_instance = idx;
                    }
                }
            }
            continue;
        }

        if (!mesh.is_visible) continue;

        const model = mesh.getWorldMatrix();
        const world_aabb = mesh.local_bounding_box.transform(model);

        if (if (world) |pw| pw.findBody(mesh) else null) |body| {
            if (body.collider == .sphere) {
                if (sphereHit(r, model, body.sphere_radius)) |hit| {
                    if (hit.distance < closest_dist) {
                        closest_dist = hit.distance;
                        best_hit = hit;
                        best_mesh = mesh;
                        best_instance = null;
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
                best_instance = null;
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
            .picked_instance = best_instance,
        };
    }

    return PickingInfo{};
}

test "viewport ray centers on the viewport, not the window" {
    const FreeCamera = @import("../camera.zig").FreeCamera;
    const cam: Camera = .{ .free = FreeCamera.init("test", .{ .position = Vec3.new(0, 0, 5) }) };
    const fb_w: f32 = 800.0;
    const fb_h: f32 = 600.0;
    // Right-half PIP viewport: its center is at x=600 in window pixels.
    const pip = Viewport{ .x = 0.5, .y = 0.0, .width = 0.5, .height = 1.0 };
    const r = createPickingRayViewport(cam, 600.0, 300.0, fb_w, fb_h, pip);
    // Viewport center looks straight ahead (-Z for the default free camera).
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), r.direction.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), r.direction.y, 1e-4);
    try std.testing.expect(r.direction.z < -0.9);
    // Same cursor through the whole-window formula points off-center instead.
    const whole = createPickingRayViewport(cam, 600.0, 300.0, fb_w, fb_h, .{});
    try std.testing.expect(whole.direction.x > 0.1);
    // Containment follows the pixel rect (right half only).
    try std.testing.expect(viewportContainsPoint(pip, 600.0, 300.0, fb_w, fb_h));
    try std.testing.expect(!viewportContainsPoint(pip, 200.0, 300.0, fb_w, fb_h));
}

test "overlay camera wins under the cursor, fallback is null" {
    const entries = [_]struct { viewport: Viewport, enabled: bool }{
        .{ .viewport = .{}, .enabled = true },
        .{ .viewport = .{ .x = 0.5, .y = 0.0, .width = 0.5, .height = 1.0 }, .enabled = true },
    };
    // Cursor inside the PIP overlay: topmost (index 1) wins over fullscreen primary.
    try std.testing.expectEqual(@as(?usize, 1), selectPickIndex(&entries, 0, 600.0, 300.0, 800.0, 600.0));
    // Cursor in the left half: only the primary covers it.
    try std.testing.expectEqual(@as(?usize, 0), selectPickIndex(&entries, 0, 200.0, 300.0, 800.0, 600.0));
    // Disabled overlay falls through to the primary.
    const disabled = [_]struct { viewport: Viewport, enabled: bool }{
        .{ .viewport = .{}, .enabled = true },
        .{ .viewport = .{ .x = 0.5, .y = 0.0, .width = 0.5, .height = 1.0 }, .enabled = false },
    };
    try std.testing.expectEqual(@as(?usize, 0), selectPickIndex(&disabled, 0, 600.0, 300.0, 800.0, 600.0));
    // No coverage at all: sensible null fallback (Scene maps this to PickingInfo{}).
    const halves = [_]struct { viewport: Viewport, enabled: bool }{
        .{ .viewport = .{ .x = 0.0, .y = 0.0, .width = 0.5, .height = 1.0 }, .enabled = true },
        .{ .viewport = .{ .x = 0.5, .y = 0.0, .width = 0.5, .height = 0.5 }, .enabled = true },
    };
    try std.testing.expectEqual(@as(?usize, null), selectPickIndex(&halves, 0, 600.0, 500.0, 800.0, 600.0));
}

test "picking containment matches the rounded toPixelRect edges" {
    // 800/3 = 266.67: rendering rounds the rect to x=267 w=267, so picking
    // must use the same edges (unrounded math would admit 266.9 and drop
    // 533.5, disagreeing with the drawn viewport by a pixel).
    const vp = Viewport{ .x = 1.0 / 3.0, .y = 0.0, .width = 1.0 / 3.0, .height = 1.0 };
    const rect = vp.toPixelRect(800, 600);
    try std.testing.expectEqual(@as(i32, 267), rect.x);
    try std.testing.expectEqual(@as(i32, 267), rect.width);
    try std.testing.expect(!viewportContainsPoint(vp, 266.9, 300.0, 800.0, 600.0));
    try std.testing.expect(viewportContainsPoint(vp, 267.0, 300.0, 800.0, 600.0));
    try std.testing.expect(viewportContainsPoint(vp, 533.5, 300.0, 800.0, 600.0));
    try std.testing.expect(!viewportContainsPoint(vp, 534.0, 300.0, 800.0, 600.0));
}

test "sphere picking follows the world transform for parented meshes" {
    var pw = PhysicsWorld.init(std.testing.allocator);
    defer pw.deinit();

    var parent = Mesh{
        .name = "parent",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = Vec3.new(10, 0, 0),
    };
    var child = Mesh{
        .name = "child",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .parent = &parent,
    };
    const body = try pw.createBody(&child, .sphere, 0.0);
    body.sphere_radius = 1.0;
    var list = [_]*Mesh{&child};

    // Ray through the world position hits (the old local-space code tested
    // mesh.position = origin and missed).
    const hit_info = pickWithRay(&list, &pw, Ray.new(Vec3.new(10, 0, -5), Vec3.new(0, 0, 1)));
    try std.testing.expect(hit_info.hit);
    try std.testing.expectEqual(&child, hit_info.picked_mesh.?);

    // Ray through the old local-space center misses.
    try std.testing.expect(!pickWithRay(&list, &pw, Ray.new(Vec3.new(0, 0, -5), Vec3.new(0, 0, 1))).hit);
}

test "sphere picking follows rotated parents" {
    var pw = PhysicsWorld.init(std.testing.allocator);
    defer pw.deinit();

    var parent = Mesh{
        .name = "parent",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .rotation = Vec3.new(0, 0, 90),
    };
    var child = Mesh{
        .name = "child",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = Vec3.new(1, 0, 0),
        .parent = &parent,
    };
    const body = try pw.createBody(&child, .sphere, 0.0);
    body.sphere_radius = 1.0;
    var list = [_]*Mesh{&child};

    // Rz(90deg) * (1,0,0) = (0,1,0): the rotated world center hits.
    const hit_info = pickWithRay(&list, &pw, Ray.new(Vec3.new(0, 1, -5), Vec3.new(0, 0, 1)));
    try std.testing.expect(hit_info.hit);
    try std.testing.expectEqual(&child, hit_info.picked_mesh.?);
    // The unrotated local offset misses.
    try std.testing.expect(!pickWithRay(&list, &pw, Ray.new(Vec3.new(1, 0, -5), Vec3.new(0, 0, 1))).hit);
}

test "sphere picking intersects the true scaled ellipsoid" {
    var pw = PhysicsWorld.init(std.testing.allocator);
    defer pw.deinit();

    var mesh = Mesh{
        .name = "scaled",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = Vec3.new(10, 0, 0),
        .scaling = Vec3.new(3, 1, 1),
    };
    const body = try pw.createBody(&mesh, .sphere, 0.0);
    body.sphere_radius = 1.0;
    var list = [_]*Mesh{&mesh};

    // Exact local-space test: scaling (3,1,1) maps the unit sphere to the
    // ellipsoid with semi-axes (3,1,1). y-offset 2 is outside the y
    // semi-axis and must miss (a world-space sphere approximation would
    // false-hit here); x-offset 2.5 is inside the x semi-axis and hits.
    try std.testing.expect(!pickWithRay(&list, &pw, Ray.new(Vec3.new(10, 2, -5), Vec3.new(0, 0, 1))).hit);
    try std.testing.expect(pickWithRay(&list, &pw, Ray.new(Vec3.new(12.5, 0, -5), Vec3.new(0, 0, 1))).hit);
    // Beyond the x semi-axis still misses.
    try std.testing.expect(!pickWithRay(&list, &pw, Ray.new(Vec3.new(14, 0, -5), Vec3.new(0, 0, 1))).hit);
}

test "sphere picking composes rotation with non-uniform parent scale" {
    var pw = PhysicsWorld.init(std.testing.allocator);
    defer pw.deinit();

    // Parent scale (3,1,1) + child Rz(45°): the world linear part is S·R
    // with singular values (3,1,1), so the ellipsoid's extent along world
    // y is still 1 while a max-column world radius would be sqrt(5) ≈ 2.24
    // and false-hit at y-offset 2 (the under-hit class this test locks).
    var parent = Mesh{
        .name = "scaled_parent",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .scaling = Vec3.new(3, 1, 1),
    };
    var child = Mesh{
        .name = "rotated_child",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = Vec3.new(1, 0, 0),
        .rotation = Vec3.new(0, 0, 45),
        .parent = &parent,
    };
    const body = try pw.createBody(&child, .sphere, 0.0);
    body.sphere_radius = 1.0;
    var list = [_]*Mesh{&child};

    const center = child.getWorldMatrix().getTranslation();
    // Ray through the transformed center hits.
    try std.testing.expect(pickWithRay(&list, &pw, Ray.new(center.add(Vec3.new(0, 0, -5)), Vec3.new(0, 0, 1))).hit);
    // Extent along world y is 1: offset 2 is outside and must miss.
    try std.testing.expect(!pickWithRay(&list, &pw, Ray.new(center.add(Vec3.new(0, 2, -5)), Vec3.new(0, 0, 1))).hit);
}

test "sphere picking normal uses the inverse-transpose under rotation" {
    var pw = PhysicsWorld.init(std.testing.allocator);
    defer pw.deinit();

    // Rz(90°): a ray offset in x hits off-center, so the normal has both x
    // and z components. The world normal must be the local normal mapped by
    // the inverse-transpose of the linear part — (0.5, 0, -0.866) here.
    // Mapping by the matrix itself (a common mix-up: rows vs columns of the
    // column-major inverse) would flip the x sign to -0.5.
    var mesh = Mesh{
        .name = "rot",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .rotation = Vec3.new(0, 0, 90),
    };
    const body = try pw.createBody(&mesh, .sphere, 0.0);
    body.sphere_radius = 1.0;
    var list = [_]*Mesh{&mesh};

    const hit_info = pickWithRay(&list, &pw, Ray.new(Vec3.new(0.5, 0, -5), Vec3.new(0, 0, 1)));
    try std.testing.expect(hit_info.hit);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), hit_info.picked_normal.x, 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), hit_info.picked_normal.y, 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, -0.86603), hit_info.picked_normal.z, 1e-3);
}

test "picking ray with degenerate framebuffer returns a defined forward ray" {
    const FreeCamera = @import("../camera.zig").FreeCamera;
    const cam: Camera = .{ .free = FreeCamera.init("test", .{ .position = Vec3.new(0, 0, 5) }) };
    const origin = cam.getPosition();
    const dims = [_][2]f32{ .{ 0, 600 }, .{ 800, 0 }, .{ -1, 600 }, .{ 800, -2 } };
    for (dims) |d| {
        const rr = createPickingRayViewport(cam, 100.0, 100.0, d[0], d[1], .{});
        try std.testing.expectEqual(origin.x, rr.origin.x);
        try std.testing.expectEqual(origin.y, rr.origin.y);
        try std.testing.expectEqual(origin.z, rr.origin.z);
        try std.testing.expectApproxEqAbs(Vec3.forward.x, rr.direction.x, 1e-6);
        try std.testing.expectApproxEqAbs(Vec3.forward.y, rr.direction.y, 1e-6);
        try std.testing.expectApproxEqAbs(Vec3.forward.z, rr.direction.z, 1e-6);
    }
    // Degenerate containment stays a defined false, never a crash.
    try std.testing.expect(!viewportContainsPoint(.{}, 100.0, 100.0, 0.0, 600.0));
    try std.testing.expect(!viewportContainsPoint(.{}, 100.0, 100.0, 800.0, 0.0));
}

test "instance picking hits an offset instance and reports its index" {
    var src = Mesh{
        .name = "src",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .local_bounding_box = math.BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    var inst0 = InstancedMesh{ .name = "i0", .source_mesh = &src, .position = Vec3.new(5, 0, 0) };
    var ptrs = [_]*InstancedMesh{&inst0};
    src.instances = .{ .items = ptrs[0..], .capacity = 1 };
    var list = [_]*Mesh{&src};

    const hit_info = pickWithRay(&list, null, Ray.new(Vec3.new(5, 0, -5), Vec3.new(0, 0, 1)));
    try std.testing.expect(hit_info.hit);
    try std.testing.expectEqual(&src, hit_info.picked_mesh.?);
    try std.testing.expectEqual(@as(?usize, 0), hit_info.picked_instance);

    // The source origin holds no instance, so a ray through it misses even
    // though the source mesh itself is under the cursor.
    try std.testing.expect(!pickWithRay(&list, null, Ray.new(Vec3.new(0, 0, -5), Vec3.new(0, 0, 1))).hit);
}

test "instance picking reports the nearer of two instances" {
    var src = Mesh{
        .name = "src",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .local_bounding_box = math.BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    var inst0 = InstancedMesh{ .name = "i0", .source_mesh = &src, .position = Vec3.new(0, 0, 0) };
    var inst1 = InstancedMesh{ .name = "i1", .source_mesh = &src, .position = Vec3.new(0, 0, 5) };
    var ptrs = [_]*InstancedMesh{ &inst0, &inst1 };
    src.instances = .{ .items = ptrs[0..], .capacity = 2 };
    var list = [_]*Mesh{&src};

    const from_front = pickWithRay(&list, null, Ray.new(Vec3.new(0, 0, -5), Vec3.new(0, 0, 1)));
    try std.testing.expect(from_front.hit);
    try std.testing.expectEqual(@as(?usize, 0), from_front.picked_instance);

    const from_back = pickWithRay(&list, null, Ray.new(Vec3.new(0, 0, 10), Vec3.new(0, 0, -1)));
    try std.testing.expect(from_back.hit);
    try std.testing.expectEqual(@as(?usize, 1), from_back.picked_instance);
}

test "instance picking skips invisible instances" {
    var src = Mesh{
        .name = "src",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .local_bounding_box = math.BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    var inst0 = InstancedMesh{ .name = "i0", .source_mesh = &src, .position = Vec3.new(0, 0, 0), .is_visible = false };
    var inst1 = InstancedMesh{ .name = "i1", .source_mesh = &src, .position = Vec3.new(0, 0, 5) };
    var ptrs = [_]*InstancedMesh{ &inst0, &inst1 };
    src.instances = .{ .items = ptrs[0..], .capacity = 2 };
    var list = [_]*Mesh{&src};

    // The nearer instance is invisible, so the farther visible one wins.
    const hit_info = pickWithRay(&list, null, Ray.new(Vec3.new(0, 0, -5), Vec3.new(0, 0, 1)));
    try std.testing.expect(hit_info.hit);
    try std.testing.expectEqual(@as(?usize, 1), hit_info.picked_instance);

    // All instances invisible behaves like a miss.
    inst1.is_visible = false;
    try std.testing.expect(!pickWithRay(&list, null, Ray.new(Vec3.new(0, 0, -5), Vec3.new(0, 0, 1))).hit);
}

test "instance picking matches the drawn instance matrix" {
    // The instanced draw path uses TRS(instance) * source.base_matrix as the
    // model matrix; the source mesh's own TRS and parent chain never reach
    // the draw. Picking must mirror that: a parent on the source does NOT
    // move instances, and picks land where instances actually render.
    var parent = Mesh{
        .name = "parent",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = Vec3.new(10, 0, 0),
    };
    var src = Mesh{
        .name = "src",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .parent = &parent,
        .local_bounding_box = math.BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    var inst0 = InstancedMesh{ .name = "i0", .source_mesh = &src };
    var ptrs = [_]*InstancedMesh{&inst0};
    src.instances = .{ .items = ptrs[0..], .capacity = 1 };
    var list = [_]*Mesh{&src};

    // Instance at its own local origin draws (and picks) at (0,0,0)...
    const hit_info = pickWithRay(&list, null, Ray.new(Vec3.new(0, 0, -5), Vec3.new(0, 0, 1)));
    try std.testing.expect(hit_info.hit);
    try std.testing.expectEqual(@as(?usize, 0), hit_info.picked_instance);
    // ...not at the source parent's position (the draw ignores it).
    try std.testing.expect(!pickWithRay(&list, null, Ray.new(Vec3.new(10, 0, -5), Vec3.new(0, 0, 1))).hit);
}

test "instance picking sees instances of a hidden source" {
    // The instanced draw ignores the source template's own is_visible flag
    // (only per-instance visibility gates it), so a hidden source with a
    // visible instance must still be pickable — exactly where it draws.
    var src = Mesh{
        .name = "src",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .is_visible = false,
        .local_bounding_box = math.BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    var inst0 = InstancedMesh{ .name = "i0", .source_mesh = &src, .position = Vec3.new(3, 0, 0) };
    var ptrs = [_]*InstancedMesh{&inst0};
    src.instances = .{ .items = ptrs[0..], .capacity = 1 };
    var list = [_]*Mesh{&src};

    const hit_info = pickWithRay(&list, null, Ray.new(Vec3.new(3, 0, -5), Vec3.new(0, 0, 1)));
    try std.testing.expect(hit_info.hit);
    try std.testing.expectEqual(@as(?usize, 0), hit_info.picked_instance);
}

test "plain mesh picking still reports a null instance index" {
    var src = Mesh{
        .name = "src",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .local_bounding_box = math.BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    var inst0 = InstancedMesh{ .name = "i0", .source_mesh = &src, .position = Vec3.new(0, 0, 0) };
    var ptrs = [_]*InstancedMesh{&inst0};
    src.instances = .{ .items = ptrs[0..], .capacity = 1 };
    var plain = Mesh{
        .name = "plain",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = Vec3.new(0, 0, -4),
        .local_bounding_box = math.BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    var list = [_]*Mesh{ &src, &plain };

    // Plain box spans z in [-5,-3], instance box spans [-1,1]: from z=-10 the
    // plain mesh is nearer and must win with a null instance index.
    const plain_wins = pickWithRay(&list, null, Ray.new(Vec3.new(0, 0, -10), Vec3.new(0, 0, 1)));
    try std.testing.expect(plain_wins.hit);
    try std.testing.expectEqual(&plain, plain_wins.picked_mesh.?);
    try std.testing.expectEqual(@as(?usize, null), plain_wins.picked_instance);

    // From z=-2 going +z the plain box [-5,-3] is behind the origin, so the
    // instance wins and its index is reported.
    const inst_wins = pickWithRay(&list, null, Ray.new(Vec3.new(0, 0, -2), Vec3.new(0, 0, 1)));
    try std.testing.expect(inst_wins.hit);
    try std.testing.expectEqual(&src, inst_wins.picked_mesh.?);
    try std.testing.expectEqual(@as(?usize, 0), inst_wins.picked_instance);

    // A lone plain mesh hit also reports null.
    var lone = [_]*Mesh{&plain};
    const lone_hit = pickWithRay(&lone, null, Ray.new(Vec3.new(0, 0, -10), Vec3.new(0, 0, 1)));
    try std.testing.expect(lone_hit.hit);
    try std.testing.expectEqual(@as(?usize, null), lone_hit.picked_instance);
}
