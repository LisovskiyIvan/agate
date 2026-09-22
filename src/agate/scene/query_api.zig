//! Scene query API: picking rays/picks, UI canvas accessors, projection,
//! event routing. Split out of `scene.zig` (facade). The audio
//! raycast adapter + `evaluateAudioOcclusion` stay on the owner
//! (core.zig): they cast `user_data` to the concrete `*Scene`.
//!
/// Anti-cycle rule (same as `audio/*`, `profiler/*`): every function takes
/// the scene as `anytype` (a `*Scene` from `core.zig` in practice) and this
/// module never imports `core.zig` or the `scene.zig` facade back.
/// Cross-leaf helpers consumed here are `pub` in their home module but are
/// deliberately NOT re-exported by the facade.
const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const Ray = math.Ray;
const RayHit = math.RayHit;
const sokol = @import("sokol");
const sapp = sokol.app;
const physics = @import("../physics.zig");
const PickingInfo = physics.PickingInfo;
const UICanvas = @import("../ui.zig").UICanvas;
const scene_picking = @import("picking.zig");
const mesh_mod = @import("../mesh.zig");
const Mesh = mesh_mod.Mesh;

/// Viewport-aware picking ray. In multi-camera mode the visually relevant
/// enabled camera under the cursor wins (draw overlay order: active first,
/// then the rest in index order, topmost covering entry). Falls back to a
/// forward ray from the origin when no camera covers the cursor (or no
/// camera exists); pick() maps the same case to PickingInfo{} (no hit).
/// Fullscreen viewports reduce to the legacy whole-window formula.
pub fn createPickingRay(self: anytype, screen_x: f32, screen_y: f32) Ray {
    const w = sapp.widthf();
    const h = sapp.heightf();
    if (w <= 0.0 or h <= 0.0) return Ray.new(Vec3.zero, Vec3.forward);
    if (self.enable_multi_camera and self.cameras.items.len > 0) {
        const idx = scene_picking.selectPickIndex(
            self.cameras.items,
            self.active_camera_index,
            screen_x,
            screen_y,
            w,
            h,
        ) orelse return Ray.new(Vec3.zero, Vec3.forward);
        const entry = self.cameras.items[idx];
        return scene_picking.createPickingRayViewport(entry.camera, screen_x, screen_y, w, h, entry.viewport);
    }
    if (self.active_camera) |cam| {
        const vp = if (self.active_camera_index) |ai| (if (ai < self.cameras.items.len)
            self.cameras.items[ai].viewport
        else
            cam.getViewport()) else cam.getViewport();
        if (!scene_picking.viewportContainsPoint(vp, screen_x, screen_y, w, h)) {
            return Ray.new(cam.getPosition(), Vec3.forward);
        }
        return scene_picking.createPickingRayViewport(cam, screen_x, screen_y, w, h, vp);
    }
    return Ray.new(Vec3.zero, Vec3.forward);
}

pub fn pickWithRay(self: anytype, r: Ray) PickingInfo {
    return scene_picking.pickWithRay(self.meshes.items, self.physics.getWorld(), r);
}

/// Raycasts into the scene and returns the closest hit mesh that matches the tag query.
pub fn pickWithRayTag(self: anytype, r: Ray, query_str: []const u8) PickingInfo {
    var closest_dist: f32 = std.math.inf(f32);
    var best_hit: ?RayHit = null;
    var best_mesh: ?*Mesh = null;

    for (self.meshes.items) |mesh| {
        if (!mesh.matchesTagQuery(query_str)) continue;
        if (mesh.is_lod_child or mesh.is_decal or mesh.gpu_pending) continue;

        const box = if (mesh.cached_frame == self.frame_id) mesh.cached_aabb else mesh.getWorldBoundingBox();
        if (r.intersectsAABBNormal(box)) |hit| {
            if (hit.distance < closest_dist) {
                closest_dist = hit.distance;
                best_hit = hit;
                best_mesh = mesh;
            }
        }
    }

    return .{
        .hit = best_mesh != null,
        .distance = if (best_mesh != null) closest_dist else 0.0,
        .picked_mesh = best_mesh,
        .picked_point = if (best_hit) |h| h.point else Vec3.zero,
        .picked_normal = if (best_hit) |h| h.normal else Vec3.up,
        .picked_instance = null,
    };
}

/// Viewport-aware pick: resolves the camera like createPickingRay but
/// returns PickingInfo{} (no hit) when no enabled camera covers the
/// cursor, instead of casting a fallback ray into the scene.
pub fn pick(self: anytype, screen_x: f32, screen_y: f32) PickingInfo {
    const w = sapp.widthf();
    const h = sapp.heightf();
    if (w <= 0.0 or h <= 0.0) return PickingInfo{};
    if (self.enable_multi_camera and self.cameras.items.len > 0) {
        const idx = scene_picking.selectPickIndex(
            self.cameras.items,
            self.active_camera_index,
            screen_x,
            screen_y,
            w,
            h,
        ) orelse return PickingInfo{};
        const entry = self.cameras.items[idx];
        const r = scene_picking.createPickingRayViewport(entry.camera, screen_x, screen_y, w, h, entry.viewport);
        return self.pickWithRay(r);
    }
    const r = self.createPickingRay(screen_x, screen_y);
    // Single-camera PIP: cursor outside the viewport is a clean miss.
    if (self.active_camera) |cam| {
        const vp = if (self.active_camera_index) |ai| (if (ai < self.cameras.items.len)
            self.cameras.items[ai].viewport
        else
            cam.getViewport()) else cam.getViewport();
        if (!scene_picking.viewportContainsPoint(vp, screen_x, screen_y, w, h)) {
            return PickingInfo{};
        }
    } else {
        return PickingInfo{};
    }
    return self.pickWithRay(r);
}

pub fn createUI(self: anytype) !*UICanvas {
    if (self.ui_canvas == null) {
        self.ui_canvas = try UICanvas.init(self.allocator);
    }
    return &self.ui_canvas.?;
}

pub fn getUI(self: anytype) ?*UICanvas {
    if (self.ui_canvas) |*u| return u;
    return null;
}

pub fn projectPoint(self: anytype, world_pos: Vec3) ?math.Vec2 {
    const cam = self.active_camera orelse return null;
    const w = sapp.widthf();
    const h = sapp.heightf();
    if (w <= 0.0 or h <= 0.0) return null;
    const vp = self.project.viewProjection(cam, w, h);
    return vp.projectPoint(world_pos, w, h);
}

pub fn handleEvent(self: anytype, ev: [*c]const sapp.Event) void {
    if (self.active_camera) |*cam| {
        cam.handleEvent(ev);
        if (self.active_camera_index) |idx| {
            if (idx < self.cameras.items.len) {
                self.cameras.items[idx].camera = cam.*;
            }
        }
    }
}
