//! glTF and scene file physics collider parser and spawner.
//! Supports:
//! 1. glTF OMI_physics_body / KHR_physics_rigid_bodies extensions
//! 2. glTF node and mesh extras (JSON fields `collider`, `shape`, `physics`, `mass`, etc.)
//! 3. Standard DCC/Unreal collision proxy naming conventions:
//!    - UCX_* (convex hull)
//!    - UBX_* (box)
//!    - USP_* (sphere)
//!    - UCP_* (capsule)
//!    - *_col, *-col, *_collider, *-collider
//!    - col_box_*, col_sphere_*, col_capsule_*, col_hull_*, col_mesh_*

const std = @import("std");
const c = @import("../c.zig").c;
const physics = @import("../physics.zig");
const ColliderType = physics.ColliderType;
const CollisionFilter = physics.CollisionFilter;
const PhysicsWorld = physics.PhysicsWorld;
const RigidBody = physics.RigidBody;
const Scene = @import("../scene.zig").Scene;
const Mesh = @import("../mesh.zig").Mesh;

pub const ColliderSpec = struct {
    collider_type: ColliderType = .box,
    mass: f32 = 0.0,
    is_sensor: bool = false,
    friction: f32 = 0.25,
    restitution: f32 = 0.6,
    filter: CollisionFilter = .{},
    /// True if this node was authored purely as a collision proxy (e.g. UCX_*, *_col).
    /// Proxy meshes have their visual rendering automatically disabled (`is_visible = false`).
    is_collision_proxy: bool = false,
};

fn parseColliderType(str: []const u8) ?ColliderType {
    if (std.ascii.eqlIgnoreCase(str, "box") or std.ascii.eqlIgnoreCase(str, "cuboid")) {
        return .box;
    } else if (std.ascii.eqlIgnoreCase(str, "sphere")) {
        return .sphere;
    } else if (std.ascii.eqlIgnoreCase(str, "capsule")) {
        return .capsule;
    } else if (std.ascii.eqlIgnoreCase(str, "hull") or std.ascii.eqlIgnoreCase(str, "convex")) {
        return .hull;
    } else if (std.ascii.eqlIgnoreCase(str, "mesh") or std.ascii.eqlIgnoreCase(str, "trimesh") or std.ascii.eqlIgnoreCase(str, "triangle_mesh")) {
        return .mesh;
    } else if (std.ascii.eqlIgnoreCase(str, "heightfield") or std.ascii.eqlIgnoreCase(str, "terrain")) {
        return .heightfield;
    }
    return null;
}

fn getJsonFloat(val: std.json.Value) ?f32 {
    return switch (val) {
        .float => |f| @floatCast(f),
        .integer => |i| @floatFromInt(i),
        else => null,
    };
}

fn getJsonBool(val: std.json.Value) ?bool {
    return switch (val) {
        .bool => |b| b,
        else => null,
    };
}

fn getJsonString(val: std.json.Value) ?[]const u8 {
    return switch (val) {
        .string => |s| s,
        else => null,
    };
}

/// Parses a collider specification from a JSON text block (from extras or extension data).
pub fn parseJsonCollider(allocator: std.mem.Allocator, json_text: []const u8) ?ColliderSpec {
    const trimmed = std.mem.trim(u8, json_text, " \t\r\n");
    if (trimmed.len < 2) return null;

    var parsed = std.json.parseFromSlice(std.json.Value, allocator, trimmed, .{}) catch return null;
    defer parsed.deinit();

    if (parsed.value != .object) return null;
    const root = parsed.value.object;

    var spec = ColliderSpec{};
    var found = false;

    // 1. Direct or nested collider / shape fields
    if (root.get("collider")) |col_val| {
        if (col_val == .string) {
            if (parseColliderType(col_val.string)) |ct| {
                spec.collider_type = ct;
                found = true;
            }
        } else if (col_val == .object) {
            if (col_val.object.get("shape") orelse col_val.object.get("type")) |shape_val| {
                if (getJsonString(shape_val)) |s| {
                    if (parseColliderType(s)) |ct| {
                        spec.collider_type = ct;
                        found = true;
                    }
                }
            }
        }
    }

    if (!found) {
        if (root.get("shape")) |shape_val| {
            if (getJsonString(shape_val)) |s| {
                if (parseColliderType(s)) |ct| {
                    spec.collider_type = ct;
                    found = true;
                }
            }
        }
    }

    // Check nested "physics" or "motion" object
    if (root.get("physics")) |phys_val| {
        if (phys_val == .object) {
            const pobj = phys_val.object;
            if (pobj.get("shape") orelse pobj.get("collider")) |ps| {
                if (getJsonString(ps)) |s| {
                    if (parseColliderType(s)) |ct| {
                        spec.collider_type = ct;
                        found = true;
                    }
                }
            }
            if (pobj.get("type")) |ptype| {
                if (getJsonString(ptype)) |s| {
                    if (std.ascii.eqlIgnoreCase(s, "static") or std.ascii.eqlIgnoreCase(s, "kinematic")) {
                        spec.mass = 0.0;
                    } else if (std.ascii.eqlIgnoreCase(s, "dynamic")) {
                        if (spec.mass <= 0.0) spec.mass = 1.0;
                        found = true;
                    }
                }
            }
            if (pobj.get("mass")) |m_val| {
                if (getJsonFloat(m_val)) |m| spec.mass = m;
            }
            if (pobj.get("friction")) |f_val| {
                if (getJsonFloat(f_val)) |f| spec.friction = f;
            }
            if (pobj.get("restitution")) |r_val| {
                if (getJsonFloat(r_val)) |r| spec.restitution = r;
            }
            if (pobj.get("is_sensor") orelse pobj.get("is_trigger")) |s_val| {
                if (getJsonBool(s_val)) |b| spec.is_sensor = b;
            }
        }
    }

    // Check OMI_physics_body format
    if (root.get("motion")) |mot_val| {
        if (mot_val == .object) {
            const mobj = mot_val.object;
            if (mobj.get("type")) |t_val| {
                if (getJsonString(t_val)) |s| {
                    if (std.ascii.eqlIgnoreCase(s, "static") or std.ascii.eqlIgnoreCase(s, "kinematic")) {
                        spec.mass = 0.0;
                    } else if (std.ascii.eqlIgnoreCase(s, "dynamic")) {
                        if (spec.mass <= 0.0) spec.mass = 1.0;
                    }
                }
            }
            if (mobj.get("mass")) |m_val| {
                if (getJsonFloat(m_val)) |m| spec.mass = m;
            }
        }
    }

    if (root.get("mass")) |m_val| {
        if (getJsonFloat(m_val)) |m| spec.mass = m;
    }
    if (root.get("friction")) |f_val| {
        if (getJsonFloat(f_val)) |f| spec.friction = f;
    }
    if (root.get("restitution")) |r_val| {
        if (getJsonFloat(r_val)) |r| spec.restitution = r;
    }
    if (root.get("is_sensor") orelse root.get("is_trigger")) |s_val| {
        if (getJsonBool(s_val)) |b| spec.is_sensor = b;
    }
    if (root.get("is_collision_proxy")) |p_val| {
        if (getJsonBool(p_val)) |b| spec.is_collision_proxy = b;
    }

    return if (found) spec else null;
}

/// Parses collider specification from a node name (Unreal / DCC conventions).
pub fn parseNameCollider(name: []const u8) ?ColliderSpec {
    // 1. Unreal Engine prefixes
    if (std.mem.startsWith(u8, name, "UCX_")) {
        return ColliderSpec{
            .collider_type = .hull,
            .mass = 0.0,
            .is_collision_proxy = true,
        };
    } else if (std.mem.startsWith(u8, name, "UBX_")) {
        return ColliderSpec{
            .collider_type = .box,
            .mass = 0.0,
            .is_collision_proxy = true,
        };
    } else if (std.mem.startsWith(u8, name, "USP_")) {
        return ColliderSpec{
            .collider_type = .sphere,
            .mass = 0.0,
            .is_collision_proxy = true,
        };
    } else if (std.mem.startsWith(u8, name, "UCP_")) {
        return ColliderSpec{
            .collider_type = .capsule,
            .mass = 0.0,
            .is_collision_proxy = true,
        };
    }

    // 2. col_* prefixes
    if (std.mem.startsWith(u8, name, "col_box_") or std.mem.startsWith(u8, name, "COL_BOX_")) {
        return ColliderSpec{ .collider_type = .box, .mass = 0.0, .is_collision_proxy = true };
    } else if (std.mem.startsWith(u8, name, "col_sphere_") or std.mem.startsWith(u8, name, "COL_SPHERE_")) {
        return ColliderSpec{ .collider_type = .sphere, .mass = 0.0, .is_collision_proxy = true };
    } else if (std.mem.startsWith(u8, name, "col_capsule_") or std.mem.startsWith(u8, name, "COL_CAPSULE_")) {
        return ColliderSpec{ .collider_type = .capsule, .mass = 0.0, .is_collision_proxy = true };
    } else if (std.mem.startsWith(u8, name, "col_hull_") or std.mem.startsWith(u8, name, "COL_HULL_")) {
        return ColliderSpec{ .collider_type = .hull, .mass = 0.0, .is_collision_proxy = true };
    } else if (std.mem.startsWith(u8, name, "col_mesh_") or std.mem.startsWith(u8, name, "COL_MESH_")) {
        return ColliderSpec{ .collider_type = .mesh, .mass = 0.0, .is_collision_proxy = true };
    }

    // 3. Suffix conventions: *_col, *-col, *_collider, *-collider
    const has_col_suffix = std.mem.endsWith(u8, name, "_col") or
        std.mem.endsWith(u8, name, "-col") or
        std.mem.endsWith(u8, name, "_COL") or
        std.mem.endsWith(u8, name, "-COL") or
        std.mem.endsWith(u8, name, "_collider") or
        std.mem.endsWith(u8, name, "-collider") or
        std.mem.endsWith(u8, name, "_COLLIDER") or
        std.mem.endsWith(u8, name, "-COLLIDER");

    if (has_col_suffix) {
        var ct: ColliderType = .mesh;
        var lower_buf: [128]u8 = undefined;
        const len = @min(name.len, lower_buf.len);
        for (name[0..len], 0..) |ch, i| {
            lower_buf[i] = std.ascii.toLower(ch);
        }
        const lower = lower_buf[0..len];

        if (std.mem.indexOf(u8, lower, "box") != null) {
            ct = .box;
        } else if (std.mem.indexOf(u8, lower, "sphere") != null) {
            ct = .sphere;
        } else if (std.mem.indexOf(u8, lower, "capsule") != null) {
            ct = .capsule;
        } else if (std.mem.indexOf(u8, lower, "hull") != null or std.mem.indexOf(u8, lower, "convex") != null) {
            ct = .hull;
        }

        return ColliderSpec{
            .collider_type = ct,
            .mass = 0.0,
            .is_collision_proxy = true,
        };
    }

    return null;
}

/// Discovers collider specification on a glTF node via extensions, extras, or naming.
pub fn parseNodeCollider(allocator: std.mem.Allocator, node_raw: [*c]const c.cgltf_node) ?ColliderSpec {
    const node: *const c.cgltf_node = node_raw orelse return null;
    // 1. Check extensions: OMI_physics_body / KHR_physics_rigid_bodies
    if (node.extensions_count > 0 and node.extensions != null) {
        for (0..node.extensions_count) |i| {
            const ext = &node.extensions[i];
            if (ext.name != null and ext.data != null) {
                const name = std.mem.span(ext.name);
                if (std.mem.eql(u8, name, "OMI_physics_body") or
                    std.mem.eql(u8, name, "KHR_physics_rigid_bodies"))
                {
                    if (parseJsonCollider(allocator, std.mem.span(ext.data))) |spec| {
                        return spec;
                    }
                }
            }
        }
    }

    // 2. Check node extras
    if (node.extras.data != null) {
        if (parseJsonCollider(allocator, std.mem.span(node.extras.data))) |spec| {
            return spec;
        }
    }

    // 3. Check mesh extras if present
    if (node.mesh != null and node.mesh.*.extras.data != null) {
        if (parseJsonCollider(allocator, std.mem.span(node.mesh.*.extras.data))) |spec| {
            return spec;
        }
    }

    // 4. Check node name
    if (node.name != null) {
        if (parseNameCollider(std.mem.span(node.name))) |spec| {
            return spec;
        }
    }

    // 5. Check mesh name
    if (node.mesh != null and node.mesh.*.name != null) {
        if (parseNameCollider(std.mem.span(node.mesh.*.name))) |spec| {
            return spec;
        }
    }

    return null;
}

/// Scans glTF nodes, hides collision proxies, and spawns RigidBody instances into physics_world.
pub fn loadColliders(
    scene: *Scene,
    gltf: *c.cgltf_data,
    spawned_meshes: []const *Mesh,
    node_mesh_start: []const usize,
    node_mesh_count: []const usize,
    physics_world: ?*PhysicsWorld,
    auto_colliders: bool,
    out_bodies: *std.ArrayList(*RigidBody),
) !void {
    if (gltf.nodes_count == 0) return;

    for (0..gltf.nodes_count) |node_idx| {
        const count = node_mesh_count[node_idx];
        if (count == 0) continue;
        const start = node_mesh_start[node_idx];
        const node = &gltf.nodes[node_idx];

        const spec = parseNodeCollider(scene.allocator, node) orelse continue;

        for (start..start + count) |mesh_idx| {
            if (mesh_idx >= spawned_meshes.len) break;
            const mesh = spawned_meshes[mesh_idx];

            if (spec.is_collision_proxy) {
                mesh.is_visible = false;
                mesh.cast_shadows = false;
                mesh.receive_shadows = false;
            }

            if (physics_world) |pw| {
                if (auto_colliders) {
                    // Box3D mesh colliders require mass == 0.0 (static)
                    const safe_mass = if (spec.collider_type == .mesh) 0.0 else spec.mass;
                    const body = try pw.createBodyWith(mesh, spec.collider_type, safe_mass, .{
                        .is_sensor = spec.is_sensor,
                        .filter = spec.filter,
                    });
                    body.friction = spec.friction;
                    body.restitution = spec.restitution;
                    try out_bodies.append(scene.allocator, body);
                }
            }
        }
    }
}
