const std = @import("std");
const Vec3 = @import("vec.zig").Vec3;
const Mat4 = @import("mat4.zig").Mat4;
const BoundingBox = @import("bounding_box.zig").BoundingBox;

pub const RayHit = struct {
    distance: f32,
    point: Vec3,
    normal: Vec3,
};

pub const TriangleHit = struct {
    distance: f32,
    point: Vec3,
    normal: Vec3,
    u: f32,
    v: f32,
};

pub const Ray = struct {
    origin: Vec3,
    direction: Vec3,

    pub fn new(origin: Vec3, direction: Vec3) Ray {
        return .{
            .origin = origin,
            .direction = direction.normalize(),
        };
    }

    pub fn getPoint(self: Ray, t: f32) Vec3 {
        return self.origin.add(self.direction.scale(t));
    }

    pub fn transform(self: Ray, m: Mat4) Ray {
        return .{
            .origin = m.transformPoint(self.origin),
            .direction = m.transformDirection(self.direction),
        };
    }

    /// Fast branchless Kay-Kajiya / Smits slab test for Ray-AABB intersection.
    /// Returns minimum positive distance t >= 0, or null if no hit.
    pub fn intersectsAABB(self: Ray, box: BoundingBox) ?f32 {
        var tmin: f32 = 0.0;
        var tmax: f32 = std.math.floatMax(f32);

        // X slab
        if (@abs(self.direction.x) < 1e-7) {
            if (self.origin.x < box.min.x or self.origin.x > box.max.x) return null;
        } else {
            const inv_d = 1.0 / self.direction.x;
            var t1 = (box.min.x - self.origin.x) * inv_d;
            var t2 = (box.max.x - self.origin.x) * inv_d;
            if (t1 > t2) std.mem.swap(f32, &t1, &t2);
            tmin = @max(tmin, t1);
            tmax = @min(tmax, t2);
            if (tmin > tmax) return null;
        }

        // Y slab
        if (@abs(self.direction.y) < 1e-7) {
            if (self.origin.y < box.min.y or self.origin.y > box.max.y) return null;
        } else {
            const inv_d = 1.0 / self.direction.y;
            var t1 = (box.min.y - self.origin.y) * inv_d;
            var t2 = (box.max.y - self.origin.y) * inv_d;
            if (t1 > t2) std.mem.swap(f32, &t1, &t2);
            tmin = @max(tmin, t1);
            tmax = @min(tmax, t2);
            if (tmin > tmax) return null;
        }

        // Z slab
        if (@abs(self.direction.z) < 1e-7) {
            if (self.origin.z < box.min.z or self.origin.z > box.max.z) return null;
        } else {
            const inv_d = 1.0 / self.direction.z;
            var t1 = (box.min.z - self.origin.z) * inv_d;
            var t2 = (box.max.z - self.origin.z) * inv_d;
            if (t1 > t2) std.mem.swap(f32, &t1, &t2);
            tmin = @max(tmin, t1);
            tmax = @min(tmax, t2);
            if (tmin > tmax) return null;
        }

        return tmin;
    }

    /// Slab test returning hit distance, intersection point, and surface normal.
    pub fn intersectsAABBNormal(self: Ray, box: BoundingBox) ?RayHit {
        var tmin: f32 = 0.0;
        var tmax: f32 = std.math.floatMax(f32);
        var hit_axis: usize = 0; // 0=X, 1=Y, 2=Z
        var hit_sign: f32 = -1.0;

        // X slab
        if (@abs(self.direction.x) < 1e-7) {
            if (self.origin.x < box.min.x or self.origin.x > box.max.x) return null;
        } else {
            const inv_d = 1.0 / self.direction.x;
            var t1 = (box.min.x - self.origin.x) * inv_d;
            var t2 = (box.max.x - self.origin.x) * inv_d;
            var sign: f32 = -1.0;
            if (t1 > t2) {
                std.mem.swap(f32, &t1, &t2);
                sign = 1.0;
            }
            if (t1 > tmin) {
                tmin = t1;
                hit_axis = 0;
                hit_sign = sign;
            }
            tmax = @min(tmax, t2);
            if (tmin > tmax) return null;
        }

        // Y slab
        if (@abs(self.direction.y) < 1e-7) {
            if (self.origin.y < box.min.y or self.origin.y > box.max.y) return null;
        } else {
            const inv_d = 1.0 / self.direction.y;
            var t1 = (box.min.y - self.origin.y) * inv_d;
            var t2 = (box.max.y - self.origin.y) * inv_d;
            var sign: f32 = -1.0;
            if (t1 > t2) {
                std.mem.swap(f32, &t1, &t2);
                sign = 1.0;
            }
            if (t1 > tmin) {
                tmin = t1;
                hit_axis = 1;
                hit_sign = sign;
            }
            tmax = @min(tmax, t2);
            if (tmin > tmax) return null;
        }

        // Z slab
        if (@abs(self.direction.z) < 1e-7) {
            if (self.origin.z < box.min.z or self.origin.z > box.max.z) return null;
        } else {
            const inv_d = 1.0 / self.direction.z;
            var t1 = (box.min.z - self.origin.z) * inv_d;
            var t2 = (box.max.z - self.origin.z) * inv_d;
            var sign: f32 = -1.0;
            if (t1 > t2) {
                std.mem.swap(f32, &t1, &t2);
                sign = 1.0;
            }
            if (t1 > tmin) {
                tmin = t1;
                hit_axis = 2;
                hit_sign = sign;
            }
            tmax = @min(tmax, t2);
            if (tmin > tmax) return null;
        }

        var normal = Vec3.zero;
        if (hit_axis == 0) normal.x = hit_sign;
        if (hit_axis == 1) normal.y = hit_sign;
        if (hit_axis == 2) normal.z = hit_sign;

        return RayHit{
            .distance = tmin,
            .point = self.getPoint(tmin),
            .normal = normal,
        };
    }

    /// Analytic Ray-Sphere intersection. Returns distance t >= 0 or null.
    pub fn intersectsSphere(self: Ray, center: Vec3, radius: f32) ?f32 {
        const oc = self.origin.sub(center);
        const b = oc.dot(self.direction);
        const c = oc.dot(oc) - radius * radius;

        if (c > 0.0 and b > 0.0) return null;

        const discr = b * b - c;
        if (discr < 0.0) return null;

        var t = -b - @sqrt(discr);
        if (t < 0.0) t = -b + @sqrt(discr);
        if (t < 0.0) return null;

        return t;
    }

    /// Analytic Ray-Sphere intersection with point and normal.
    pub fn intersectsSphereNormal(self: Ray, center: Vec3, radius: f32) ?RayHit {
        if (self.intersectsSphere(center, radius)) |dist| {
            const pt = self.getPoint(dist);
            const normal = pt.sub(center).normalize();
            return RayHit{
                .distance = dist,
                .point = pt,
                .normal = normal,
            };
        }
        return null;
    }

    /// Möller–Trumbore ray-triangle intersection algorithm.
    pub fn intersectsTriangle(self: Ray, v0: Vec3, v1: Vec3, v2: Vec3) ?TriangleHit {
        const edge1 = v1.sub(v0);
        const edge2 = v2.sub(v0);
        const pvec = self.direction.cross(edge2);
        const det = edge1.dot(pvec);

        if (@abs(det) < 1e-7) return null;
        const inv_det = 1.0 / det;

        const tvec = self.origin.sub(v0);
        const u = tvec.dot(pvec) * inv_det;
        if (u < 0.0 or u > 1.0) return null;

        const qvec = tvec.cross(edge1);
        const v = self.direction.dot(qvec) * inv_det;
        if (v < 0.0 or u + v > 1.0) return null;

        const t = edge2.dot(qvec) * inv_det;
        if (t < 1e-5) return null;

        const geom_normal = edge1.cross(edge2).normalize();
        const normal = if (det < 0.0) geom_normal.scale(-1.0) else geom_normal;

        return TriangleHit{
            .distance = t,
            .point = self.getPoint(t),
            .normal = normal,
            .u = u,
            .v = v,
        };
    }

    /// Ray-Plane intersection.
    pub fn intersectsPlane(self: Ray, plane_point: Vec3, plane_normal: Vec3) ?f32 {
        const denom = plane_normal.dot(self.direction);
        if (@abs(denom) < 1e-6) return null;
        const t = plane_point.sub(self.origin).dot(plane_normal) / denom;
        if (t < 0.0) return null;
        return t;
    }
};
