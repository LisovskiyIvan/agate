const std = @import("std");
const c = @import("../c.zig").c;
const math = @import("math");
const Vec3 = math.Vec3;
const lights = @import("lights.zig");
const lightColor = lights.lightColor;
const lightIntensity = lights.lightIntensity;
const rangeOrDefault = lights.rangeOrDefault;
const default_point_range = lights.default_point_range;
const spotConeDeg = lights.spotConeDeg;
const lookDirectionToEuler = lights.lookDirectionToEuler;
const forwardFromEuler = lights.forwardFromEuler;
const resolveLightName = lights.resolveLightName;
const resolveCameraName = lights.resolveCameraName;

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
