const std = @import("std");

const c = @import("../c.zig").c;
const math = @import("math");
const Vec3 = math.Vec3;
const Mat4 = math.Mat4;
const Color3 = math.Color3;

const Scene = @import("../scene.zig").Scene;
const FreeCamera = @import("../camera.zig").FreeCamera;

// glTF light/camera import (KHR_lights_punctual + node.camera).
//
// Mapping notes:
// - A punctual light / camera is attached to a glTF *node*; the world matrix
//   comes from cgltf_node_transform_world, composed with the caller-provided
//   parent_world (scene_loader passes Mat4.identity, so world == node world).
// - glTF lights and cameras look down their local -Z axis. The beam/travel
//   direction is therefore world.transformDirection(-Z).
// - Engine DirectionalLight.direction points TOWARD the sun (see cascades:
//   light_eye = center + dir * dist), so a glTF directional light maps to the
//   NEGATED travel direction. SpotLight.direction is the beam travel direction
//   (shader uses dot(-L, s_dir)), so it maps to -Z unnegated.
// - The engine owns up to 4 directional lights (createDirectionalLight for
//   the shadow-casting primary, addDirectionalLight for 3 shadowless fills)
//   and has a single ?Camera active_camera slot, hence the "first wins"
//   rules below (directional: first four win).

pub const default_point_range: f32 = 10.0; // mirrors PointLightOptions.range
pub const default_spot_range: f32 = 15.0; // mirrors SpotLightOptions.range
pub const default_camera_fov_deg: f32 = 60.0; // mirrors FreeCameraOptions.fov_deg
pub const default_camera_near: f32 = 0.1;
pub const default_camera_far: f32 = 100.0;

/// glTF light color -> engine color (passthrough, no colorspace conversion).
pub fn lightColor(light: *const c.cgltf_light) Color3 {
    return Color3.new(light.color[0], light.color[1], light.color[2]);
}

/// glTF intensity -> engine intensity multiplier. Negative values are clamped
/// to zero; unit semantics stay engine-side (no candela/lux conversion).
pub fn lightIntensity(light: *const c.cgltf_light) f32 {
    return @max(0.0, light.intensity);
}

/// glTF range == 0 means "infinite"; map it to the engine default so the
/// shader attenuation (dist >= range -> skip) stays meaningful.
pub fn rangeOrDefault(range: f32, fallback: f32) f32 {
    if (range > 0.0) return range;
    return fallback;
}

pub const SpotConeDeg = struct {
    inner_deg: f32,
    outer_deg: f32,
};

pub fn radToDeg(rad: f32) f32 {
    return rad * 180.0 / std.math.pi;
}

/// Spot cone angles: cgltf stores radians, the engine stores degrees
/// (SpotLight.inner/outer_angle_deg, converted to cos at uniform pack time).
pub fn spotConeDeg(light: *const c.cgltf_light) SpotConeDeg {
    return .{
        .inner_deg = @max(0.0, radToDeg(light.spot_inner_cone_angle)),
        .outer_deg = @max(0.0, radToDeg(light.spot_outer_cone_angle)),
    };
}

/// Beam/travel direction of a node-attached light or camera: local -Z axis.
pub fn forwardFromWorld(world: Mat4) Vec3 {
    return world.transformDirection(Vec3.new(0.0, 0.0, -1.0));
}

/// Full node world matrix composed with the loader parent transform.
pub fn nodeWorldMatrix(node: *const c.cgltf_node, parent_world: Mat4) Mat4 {
    var raw: [16]f32 = undefined;
    c.cgltf_node_transform_world(node, &raw);
    return parent_world.mul(Mat4{ .m = raw });
}

/// Inverse of FreeCamera.getForward: look direction -> euler degrees
/// (x = pitch, positive up; y = yaw; yaw 0 + pitch 0 faces -Z).
/// Yaw is undefined at the poles; atan2(0, 0) yields 0 there, which still
/// round-trips through getForward.
pub fn lookDirectionToEuler(dir: Vec3) Vec3 {
    const n = dir.normalize();
    const pitch_rad = std.math.asin(std.math.clamp(n.y, -1.0, 1.0));
    const yaw_rad = std.math.atan2(-n.x, -n.z);
    return Vec3.new(
        pitch_rad * 180.0 / std.math.pi,
        yaw_rad * 180.0 / std.math.pi,
        0.0,
    );
}

/// Forward mirror of FreeCamera.getForward for the same euler convention.
/// Kept here (instead of importing camera.zig logic) so tests stay pure.
pub fn forwardFromEuler(euler_deg: Vec3) Vec3 {
    const yaw_rad = euler_deg.y * std.math.pi / 180.0;
    const pitch_rad = euler_deg.x * std.math.pi / 180.0;
    const cp = @cos(pitch_rad);
    const fwd = Vec3.new(-@sin(yaw_rad) * cp, @sin(pitch_rad), -@cos(yaw_rad) * cp);
    if (fwd.length() > 0.0001) return fwd.normalize();
    return Vec3.new(0.0, 0.0, -1.0);
}

/// Perspective glTF field of view (radians, Y) -> engine degrees.
pub fn perspectiveFovDeg(yfov_rad: f32) f32 {
    const deg = radToDeg(yfov_rad);
    if (deg > 0.0) return deg;
    return default_camera_fov_deg;
}

/// Name priority: node name, then light/camera name, then indexed fallback.
/// Empty names count as missing.
pub fn resolveLightName(node_name: ?[]const u8, light_name: ?[]const u8, index: usize, buf: []u8) []const u8 {
    if (node_name) |n| if (n.len > 0) return n;
    if (light_name) |n| if (n.len > 0) return n;
    return std.fmt.bufPrint(buf, "gltf_light_{d}", .{index}) catch "gltf_light";
}

pub fn resolveCameraName(node_name: ?[]const u8, camera_name: ?[]const u8, index: usize, buf: []u8) []const u8 {
    if (node_name) |n| if (n.len > 0) return n;
    if (camera_name) |n| if (n.len > 0) return n;
    return std.fmt.bufPrint(buf, "gltf_camera_{d}", .{index}) catch "gltf_camera";
}

fn nodeName(node: *const c.cgltf_node) ?[]const u8 {
    if (node.name != null) {
        const span = std.mem.span(node.name);
        if (span.len > 0) return span;
    }
    return null;
}

fn lightObjectName(light: *const c.cgltf_light) ?[]const u8 {
    if (light.name != null) {
        const span = std.mem.span(light.name);
        if (span.len > 0) return span;
    }
    return null;
}

fn cameraObjectName(cam: *const c.cgltf_camera) ?[]const u8 {
    if (cam.name != null) {
        const span = std.mem.span(cam.name);
        if (span.len > 0) return span;
    }
    return null;
}

/// Imports KHR_lights_punctual lights attached to glTF nodes:
/// directional -> Scene.createDirectionalLight for the first, then
/// Scene.addDirectionalLight fills (up to 4 suns total, extras skipped),
/// point -> Scene.createPointLight, spot -> Scene.createSpotLight.
/// Names are duped into the scene allocator (glTF strings die with cgltf_free).
///
/// Multi-directional limit: the engine holds 1 primary sun + 3 fills; if the
/// scene already has a directional light it becomes the primary only when
/// the slot is free, further glTF directionals become fills, and lights
/// past the fourth are skipped (their duped names freed).
pub fn loadLights(scene: *Scene, gltf: *const c.cgltf_data, parent_world: Mat4) !void {
    if (gltf.nodes_count == 0) return;
    var have_directional = scene.lights.directional != null;
    for (0..gltf.nodes_count) |node_idx| {
        const node: *const c.cgltf_node = @ptrCast(&gltf.nodes[node_idx]);
        const light_ptr = node.light orelse continue;
        const light: *const c.cgltf_light = light_ptr;

        const world = nodeWorldMatrix(node, parent_world);
        var name_buf: [64]u8 = undefined;
        const base_name = resolveLightName(
            nodeName(node),
            lightObjectName(light),
            node_idx,
            &name_buf,
        );
        const owned_name = try scene.allocator.dupe(u8, base_name);

        switch (light.type) {
            c.cgltf_light_type_directional => {
                // Engine keeps up to 4 suns: the first becomes the
                // shadow-casting primary, the next three shadowless fills;
                // further lights are skipped (duped names freed).
                // Engine direction points TOWARD the sun: negate the beam.
                const to_sun = forwardFromWorld(world).scale(-1.0);
                if (!have_directional) {
                    const dl = scene.createDirectionalLight(owned_name, .{
                        .direction = to_sun,
                        .diffuse = lightColor(light),
                        .intensity = lightIntensity(light),
                    }) catch |err| {
                        scene.allocator.free(owned_name);
                        return err;
                    };
                    dl.owns_name = true;
                    have_directional = true;
                } else {
                    const fill = scene.addDirectionalLight(owned_name, .{
                        .direction = to_sun,
                        .diffuse = lightColor(light),
                        .intensity = lightIntensity(light),
                    }) catch |err| {
                        scene.allocator.free(owned_name);
                        if (err == error.TooManyDirectionalLights) continue;
                        return err;
                    };
                    fill.owns_name = true;
                }
            },
            c.cgltf_light_type_point => {
                const pl = scene.createPointLight(owned_name, .{
                    .position = world.getTranslation(),
                    .color = lightColor(light),
                    .intensity = lightIntensity(light),
                    .range = rangeOrDefault(light.range, default_point_range),
                }) catch |err| {
                    scene.allocator.free(owned_name);
                    return err;
                };
                pl.owns_name = true;
            },
            c.cgltf_light_type_spot => {
                const cone = spotConeDeg(light);
                const sl = scene.createSpotLight(owned_name, .{
                    .position = world.getTranslation(),
                    .direction = forwardFromWorld(world),
                    .color = lightColor(light),
                    .intensity = lightIntensity(light),
                    .range = rangeOrDefault(light.range, default_spot_range),
                    .inner_angle_deg = cone.inner_deg,
                    .outer_angle_deg = cone.outer_deg,
                }) catch |err| {
                    scene.allocator.free(owned_name);
                    return err;
                };
                sl.owns_name = true;
            },
            else => {
                // Invalid/unknown type: nothing to map; drop the duped name.
                scene.allocator.free(owned_name);
            },
        }
    }
}

/// Imports glTF cameras attached to nodes as FreeCamera values.
/// Perspective maps position + look euler + fov/near/far directly.
/// Orthographic is approximate: the engine FreeCamera is perspective-only,
/// so pose + near/far are kept and fov falls back to 60 degrees.
/// The first imported camera becomes scene.active_camera ONLY when the scene
/// has no active camera yet (never overwrites a user camera). The scene has
/// no camera list, so non-first cameras exist only as the returned count.
/// Returns the number of imported cameras.
pub fn loadCameras(scene: *Scene, gltf: *const c.cgltf_data, parent_world: Mat4) !usize {
    if (gltf.nodes_count == 0) return 0;
    var count: usize = 0;
    for (0..gltf.nodes_count) |node_idx| {
        const node: *const c.cgltf_node = @ptrCast(&gltf.nodes[node_idx]);
        const cam_ptr = node.camera orelse continue;
        const cam: *const c.cgltf_camera = cam_ptr;

        const world = nodeWorldMatrix(node, parent_world);
        const rotation = lookDirectionToEuler(forwardFromWorld(world));

        var fov_deg = default_camera_fov_deg;
        var near: f32 = default_camera_near;
        var far: f32 = default_camera_far;
        if (cam.type == c.cgltf_camera_type_perspective) {
            fov_deg = perspectiveFovDeg(cam.data.perspective.yfov);
            near = cam.data.perspective.znear;
            if (cam.data.perspective.has_zfar != 0) far = cam.data.perspective.zfar;
        } else if (cam.type == c.cgltf_camera_type_orthographic) {
            // Approximate mapping (see doc comment): keep clip planes.
            near = cam.data.orthographic.znear;
            far = cam.data.orthographic.zfar;
        }
        if (!(near > 0.0)) near = default_camera_near;
        if (!(far > near)) far = default_camera_far;

        var name_buf: [64]u8 = undefined;
        const base_name = resolveCameraName(
            nodeName(node),
            cameraObjectName(cam),
            node_idx,
            &name_buf,
        );
        const owned_name = try scene.allocator.dupe(u8, base_name);

        const free_cam = FreeCamera.init(owned_name, .{
            .position = world.getTranslation(),
            .rotation = rotation,
            .fov_deg = fov_deg,
            .near = near,
            .far = far,
        });
        // The scene has a single active_camera slot: the first import wins
        // and owns its name; later names are freed with the dropped values.
        if (scene.active_camera == null) {
            scene.setActiveCamera(.{ .free = free_cam }, owned_name);
        } else {
            scene.allocator.free(owned_name);
        }
        count += 1;
    }
    return count;
}

test "light color and intensity map through, range falls back" {
    var l = std.mem.zeroes(c.cgltf_light);
    l.color = .{ 1.0, 0.5, 0.25 };
    l.intensity = 2.0;
    l.range = 0.0;

    const col = lightColor(&l);
    try std.testing.expectApproxEqAbs(col.r, 1.0, 1e-6);
    try std.testing.expectApproxEqAbs(col.g, 0.5, 1e-6);
    try std.testing.expectApproxEqAbs(col.b, 0.25, 1e-6);
    try std.testing.expectApproxEqAbs(lightIntensity(&l), 2.0, 1e-6);
    try std.testing.expectApproxEqAbs(rangeOrDefault(l.range, default_point_range), 10.0, 1e-6);

    l.range = 25.0;
    try std.testing.expectApproxEqAbs(rangeOrDefault(l.range, default_point_range), 25.0, 1e-6);

    l.intensity = -3.0;
    try std.testing.expectApproxEqAbs(lightIntensity(&l), 0.0, 1e-6);
}

test "spot cone converts radians to engine degrees" {
    var l = std.mem.zeroes(c.cgltf_light);
    l.spot_inner_cone_angle = 0.2;
    l.spot_outer_cone_angle = 0.5;
    const cone = spotConeDeg(&l);
    // Engine SpotLight stores degrees (cos applied at uniform pack time).
    try std.testing.expectApproxEqAbs(cone.inner_deg, 0.2 * 180.0 / std.math.pi, 1e-4);
    try std.testing.expectApproxEqAbs(cone.outer_deg, 0.5 * 180.0 / std.math.pi, 1e-4);
    try std.testing.expect(cone.inner_deg < cone.outer_deg);
}

test "look direction round-trips through euler degrees" {
    const dirs = [_]Vec3{
        Vec3.new(0.0, 0.0, -1.0),
        Vec3.new(1.0, 0.0, 0.0),
        Vec3.new(0.0, 0.0, 1.0),
        Vec3.new(1.0, -0.5, -2.0).normalize(),
        Vec3.new(-0.3, 0.8, -0.5).normalize(),
        Vec3.new(0.0, 1.0, 0.0),
    };
    for (dirs) |d| {
        const back = forwardFromEuler(lookDirectionToEuler(d));
        try std.testing.expectApproxEqAbs(back.x, d.x, 1e-5);
        try std.testing.expectApproxEqAbs(back.y, d.y, 1e-5);
        try std.testing.expectApproxEqAbs(back.z, d.z, 1e-5);
    }
}

test "light and camera names prefer node, then object, then indexed fallback" {
    var buf: [64]u8 = undefined;
    try std.testing.expectEqualStrings(
        "node_name",
        resolveLightName("node_name", "light_name", 3, &buf),
    );
    try std.testing.expectEqualStrings(
        "light_name",
        resolveLightName(null, "light_name", 3, &buf),
    );
    try std.testing.expectEqualStrings(
        "gltf_light_7",
        resolveLightName(null, null, 7, &buf),
    );
    try std.testing.expectEqualStrings(
        "gltf_light_0",
        resolveLightName("", "", 0, &buf),
    );
    try std.testing.expectEqualStrings(
        "gltf_camera_2",
        resolveCameraName(null, null, 2, &buf),
    );
}
