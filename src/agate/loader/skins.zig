const std = @import("std");

const c = @import("../c.zig").c;
const math = @import("math");
const Vec3 = math.Vec3;
const Quat = math.Quat;
const Mat4 = math.Mat4;

const Scene = @import("../scene.zig").Scene;
const Skeleton = @import("../animation/skeleton.zig").Skeleton;
const gltf_util = @import("gltf_util.zig");

pub fn loadSkins(scene: *Scene, gltf: *c.cgltf_data, skeletons: []?*Skeleton) !void {
    for (0..gltf.skins_count) |skin_idx| {
        const s = &gltf.skins[skin_idx];
        const skel = try Skeleton.init(scene.allocator, s.joints_count);
        if (s.name != null) {
            skel.name = try scene.allocator.dupe(u8, std.mem.span(s.name));
        }

        // Inverse bind matrices
        if (s.inverse_bind_matrices) |ibm_acc| {
            for (0..s.joints_count) |ji| {
                var m_floats: [16]f32 = undefined;
                _ = c.cgltf_accessor_read_float(ibm_acc, ji, &m_floats, 16);
                skel.bones[ji].inverse_bind_matrix = Mat4{ .m = m_floats };
            }
        }

        // Find the mesh node that uses this skin to compute root_transform
        var mesh_node: ?*c.cgltf_node = null;
        for (0..gltf.nodes_count) |ni| {
            const n = &gltf.nodes[ni];
            if (n.skin == s and n.mesh != null) {
                mesh_node = n;
                break;
            }
        }

        var inv_mesh_w = Mat4.identity;
        if (mesh_node) |mn| {
            var mn_w: [16]f32 = undefined;
            c.cgltf_node_transform_world(mn, &mn_w);
            const mn_mat = Mat4{ .m = mn_w };
            if (mn_mat.invert()) |inv| {
                inv_mesh_w = inv;
            }
        }

        for (0..s.joints_count) |ji| {
            const j_node = s.joints[ji] orelse continue;
            const b = &skel.bones[ji];
            if (j_node[0].name != null) {
                b.name = try scene.allocator.dupe(u8, std.mem.span(j_node[0].name));
            }

            // Parent within skin.joints
            if (j_node[0].parent) |parent_node| {
                for (0..s.joints_count) |pi| {
                    if (s.joints[pi] == parent_node) {
                        b.parent_index = pi;
                        break;
                    }
                }
            }

            // If root joint (parent_index == null), compute root_transform
            if (b.parent_index == null) {
                var p_world = Mat4.identity;
                if (j_node[0].parent) |parent_node| {
                    var pw_floats: [16]f32 = undefined;
                    c.cgltf_node_transform_world(parent_node, &pw_floats);
                    p_world = Mat4{ .m = pw_floats };
                }
                skel.root_transform = inv_mesh_w.mul(p_world);
            }

            // Initial local TRS
            if (j_node[0].has_translation != 0) {
                b.local_position = Vec3.new(j_node[0].translation[0], j_node[0].translation[1], j_node[0].translation[2]);
            }
            if (j_node[0].has_rotation != 0) {
                b.local_rotation = (Quat{
                    .x = j_node[0].rotation[0],
                    .y = j_node[0].rotation[1],
                    .z = j_node[0].rotation[2],
                    .w = j_node[0].rotation[3],
                }).normalize();
            }
            if (j_node[0].has_scale != 0) {
                b.local_scale = Vec3.new(j_node[0].scale[0], j_node[0].scale[1], j_node[0].scale[2]);
            }
            b.bind_position = b.local_position;
            b.bind_rotation = b.local_rotation;
            b.bind_scale = b.local_scale;
        }

        skel.update();
        skeletons[skin_idx] = skel;
        try scene.skeletons.append(scene.allocator, skel);
    }
}

/// Marks every glTF node used as a skeleton joint. Node tracks that
/// target such joints of skinned meshes are skipped (the skeleton
/// path already drives them). Caller owns the returned slice.
pub fn buildJointMask(allocator: std.mem.Allocator, gltf: *c.cgltf_data) ![]bool {
    var is_joint_node = try allocator.alloc(bool, gltf.nodes_count);
    @memset(is_joint_node, false);
    for (0..gltf.skins_count) |si| {
        const skin_joints = &gltf.skins[si];
        for (0..skin_joints.joints_count) |ji| {
            if (skin_joints.joints[ji]) |joint| {
                if (gltf_util.gltfNodeIndex(gltf, joint)) |ni| is_joint_node[ni] = true;
            }
        }
    }
    return is_joint_node;
}
