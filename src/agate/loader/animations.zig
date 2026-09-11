const std = @import("std");

const c = @import("../c.zig").c;

const Scene = @import("../scene.zig").Scene;
const Mesh = @import("../mesh.zig").Mesh;
const Skeleton = @import("../animation/skeleton.zig").Skeleton;
const AnimationGroup = @import("../animation/animation.zig").AnimationGroup;
const AnimationChannel = @import("../animation/animation.zig").AnimationChannel;
const NodeChannel = @import("../animation/animation.zig").NodeChannel;
const NodeTarget = @import("../animation/animation.zig").NodeTarget;
const MorphWeightsTarget = @import("../animation/animation.zig").MorphWeightsTarget;
const MAX_MORPH_TARGETS = @import("../mesh.zig").MAX_MORPH_TARGETS;
const AnimationPath = @import("../animation/animation.zig").AnimationPath;
const gltf_util = @import("gltf_util.zig");

pub fn loadAnimations(
    scene: *Scene,
    gltf: *c.cgltf_data,
    skeletons: []?*Skeleton,
    spawned_meshes: *std.ArrayList(*Mesh),
    node_mesh_start: []usize,
    node_mesh_count: []usize,
    is_joint_node: []bool,
) !void {
    // 4. Parse animations (skeleton tracks + plain node tracks).
    // Group names come from gltf animation.name so UI code can enumerate
    // them via scene.animation_groups.
    for (0..gltf.animations_count) |anim_idx| {
        const src_anim = &gltf.animations[anim_idx];
        const anim_name = if (src_anim.name != null)
            try scene.allocator.dupe(u8, std.mem.span(src_anim.name))
        else
            try std.fmt.allocPrint(scene.allocator, "anim_{d}", .{anim_idx});

        // Find which skeleton this animation targets (joint channels only)
        var target_skel: ?*Skeleton = null;
        var target_skin_idx: ?usize = null;
        for (0..src_anim.channels_count) |ci| {
            const ch = &src_anim.channels[ci];
            if (ch.target_node == null) continue;
            for (0..gltf.skins_count) |si| {
                const s = &gltf.skins[si];
                for (0..s.joints_count) |ji| {
                    if (s.joints[ji] == ch.target_node) {
                        target_skel = skeletons[si];
                        target_skin_idx = si;
                        break;
                    }
                }
                if (target_skel != null) break;
            }
            if (target_skel != null) break;
        }

        // NOTE: no fallback to skeletons[0] here. Clips that only drive
        // plain nodes get a skeleton-less group below instead of an empty
        // bone track on an unrelated skeleton.
        var skin_ref: ?*c.cgltf_skin = null;
        if (target_skin_idx) |si| skin_ref = &gltf.skins[si];

        var channels_list = std.ArrayList(AnimationChannel).empty;
        var node_channels_list = std.ArrayList(NodeChannel).empty;
        var node_targets_list = std.ArrayList(NodeTarget).empty;
        var morph_targets_list = std.ArrayList(MorphWeightsTarget).empty;
        var max_duration: f32 = 0.0;

        // Dedupes NodeTargets when several channels drive one node.
        const node_target_for_node = try scene.allocator.alloc(?usize, gltf.nodes_count);
        defer scene.allocator.free(node_target_for_node);
        @memset(node_target_for_node, null);

        for (0..src_anim.channels_count) |ci| {
            const ch = &src_anim.channels[ci];
            if (ch.target_node == null or ch.sampler == null) continue;

            var bone_idx: ?usize = null;
            if (skin_ref) |skin| {
                for (0..skin.joints_count) |ji| {
                    if (skin.joints[ji] == ch.target_node) {
                        bone_idx = ji;
                        break;
                    }
                }
            }

            if (bone_idx) |b_idx| {
                const path_type: AnimationPath = switch (ch.target_path) {
                    c.cgltf_animation_path_type_translation => .translation,
                    c.cgltf_animation_path_type_rotation => .rotation,
                    c.cgltf_animation_path_type_scale => .scale,
                    c.cgltf_animation_path_type_weights => .weights,
                    else => continue,
                };

                const stride: usize = switch (path_type) {
                    .translation, .scale => 3,
                    .rotation => 4,
                    .weights => 1,
                };
                const samp_ptr: *const c.cgltf_animation_sampler = @ptrCast(ch.sampler.?);
                const samp_data = try gltf_util.readSampler(scene.allocator, @constCast(samp_ptr), stride) orelse continue;
                if (samp_data.timestamps[samp_data.timestamps.len - 1] > max_duration) {
                    max_duration = samp_data.timestamps[samp_data.timestamps.len - 1];
                }

                try channels_list.append(scene.allocator, .{
                    .bone_index = b_idx,
                    .target_path = path_type,
                    .sampler = .{
                        .timestamps = samp_data.timestamps,
                        .outputs = samp_data.outputs,
                        .interpolation = samp_data.interpolation,
                    },
                });
                continue;
            }

            // Node path: translation/rotation/scale of a non-joint node
            // drives the spawned meshes of that node; weights drives their
            // morph targets (one binding + channel copy per mesh, since
            // weights slices are per-mesh state).
            const path_type: AnimationPath = switch (ch.target_path) {
                c.cgltf_animation_path_type_translation => .translation,
                c.cgltf_animation_path_type_rotation => .rotation,
                c.cgltf_animation_path_type_scale => .scale,
                c.cgltf_animation_path_type_weights => .weights,
                else => continue,
            };
            const node_idx = gltf_util.gltfNodeIndex(gltf, @ptrCast(ch.target_node.?)) orelse continue;
            const range_start = node_mesh_start[node_idx];
            const range_count = node_mesh_count[node_idx];
            if (range_count == 0) continue; // joint-only or mesh-less node

            if (path_type == .weights) {
                // Target count comes from the glTF primitive: cgltf
                // guarantees every primitive of a mesh shares it. Spawned
                // meshes keep at most MAX_MORPH_TARGETS of them.
                const target_node: *const c.cgltf_node = @ptrCast(ch.target_node.?);
                if (target_node.mesh == null) continue;
                const src_mesh: *const c.cgltf_mesh = @ptrCast(target_node.mesh);
                if (src_mesh.primitives_count == 0) continue;
                const prim0: *const c.cgltf_primitive = @ptrCast(&src_mesh.primitives[0]);
                const want_count: usize = @min(prim0.targets_count, MAX_MORPH_TARGETS);
                if (want_count == 0) continue;

                const samp_ptr: *const c.cgltf_animation_sampler = @ptrCast(ch.sampler.?);
                if (samp_ptr.input == null or samp_ptr.output == null) continue;
                const input_acc: *const c.cgltf_accessor = @ptrCast(samp_ptr.input);
                const output_acc: *const c.cgltf_accessor = @ptrCast(samp_ptr.output);
                const key_count: usize = input_acc.count;
                if (key_count == 0) continue;
                // Mirrors samplerHasFrames(.weights): one value per target per key.
                if (output_acc.count < key_count * want_count) continue;

                const samp_data = try gltf_util.readSampler(scene.allocator, @constCast(samp_ptr), want_count) orelse continue;
                if (samp_data.timestamps[samp_data.timestamps.len - 1] > max_duration) {
                    max_duration = samp_data.timestamps[samp_data.timestamps.len - 1];
                }

                for (0..range_count) |mi| {
                    const mesh_obj = spawned_meshes.items[range_start + mi];
                    // NOTE: no is_joint_node skip here. The skeleton drives
                    // joint transforms, never morph weights.
                    if (mesh_obj.morph_weights.len < want_count) continue;

                    const rest = try scene.allocator.dupe(f32, mesh_obj.morph_weights[0..want_count]);
                    errdefer scene.allocator.free(rest);
                    const morph_idx = morph_targets_list.items.len;
                    try morph_targets_list.append(scene.allocator, .{
                        .weights = mesh_obj.morph_weights[0..want_count],
                        .rest_weights = rest,
                        .dirty = &mesh_obj.morph_dirty,
                    });

                    // One owned buffer copy per mesh: several primitives of
                    // a node share the track but must free independently.
                    const ts_copy = try scene.allocator.dupe(f32, samp_data.timestamps);
                    errdefer scene.allocator.free(ts_copy);
                    const out_copy = try scene.allocator.dupe(f32, samp_data.outputs);
                    errdefer scene.allocator.free(out_copy);
                    try node_channels_list.append(scene.allocator, .{
                        .target = morph_idx,
                        .target_path = .weights,
                        .sampler = .{
                            .timestamps = ts_copy,
                            .outputs = out_copy,
                            .interpolation = samp_data.interpolation,
                        },
                    });
                }

                scene.allocator.free(samp_data.timestamps);
                scene.allocator.free(samp_data.outputs);
                continue;
            }

            const stride: usize = switch (path_type) {
                .translation, .scale => 3,
                .rotation => 4,
                .weights => 1,
            };
            const samp_ptr = @as(*const c.cgltf_animation_sampler, @ptrCast(ch.sampler.?));
            const samp_data = try gltf_util.readSampler(scene.allocator, @constCast(samp_ptr), stride) orelse continue;
            if (samp_data.timestamps[samp_data.timestamps.len - 1] > max_duration) {
                max_duration = samp_data.timestamps[samp_data.timestamps.len - 1];
            }

            for (0..range_count) |mi| {
                const mesh_obj = spawned_meshes.items[range_start + mi];
                // A joint transform of a skinned mesh is already driven by
                // the skeleton path; applying the node track on top would
                // double-apply it, so such targets are skipped.
                if (is_joint_node[node_idx] and mesh_obj.skeleton != null) continue;

                var target_idx: usize = undefined;
                if (node_target_for_node[node_idx]) |existing| {
                    target_idx = existing;
                } else {
                    // Split the baked world matrix: the animated local TRS
                    // moves into the mesh fields while the static parent
                    // world stays in base_matrix, so TRS(t) * base ==
                    // world(t) and the rest pose renders unchanged.
                    // Limitation: an animated parent node is baked into
                    // the child base_matrix once, so nested animated
                    // hierarchies do not follow their parents at runtime.
                    const node_ptr: *const c.cgltf_node = @ptrCast(&gltf.nodes[node_idx]);
                    const trs = gltf_util.nodeLocalTRS(node_ptr);
                    mesh_obj.position = trs.pos;
                    mesh_obj.rotation = trs.rot.toEulerDeg();
                    mesh_obj.scaling = trs.scale;
                    mesh_obj.base_matrix = gltf_util.nodeParentWorld(node_ptr);
                    target_idx = node_targets_list.items.len;
                    try node_targets_list.append(scene.allocator, .{
                        .position = &mesh_obj.position,
                        .rotation_euler = &mesh_obj.rotation,
                        .scaling = &mesh_obj.scaling,
                        .rest_position = trs.pos,
                        .rest_rotation = trs.rot,
                        .rest_scale = trs.scale,
                    });
                    node_target_for_node[node_idx] = target_idx;
                }

                // One owned buffer copy per mesh: several primitives of a
                // node share the track but must free independently.
                const ts_copy = try scene.allocator.dupe(f32, samp_data.timestamps);
                errdefer scene.allocator.free(ts_copy);
                const out_copy = try scene.allocator.dupe(f32, samp_data.outputs);
                errdefer scene.allocator.free(out_copy);
                try node_channels_list.append(scene.allocator, .{
                    .target = target_idx,
                    .target_path = path_type,
                    .sampler = .{
                        .timestamps = ts_copy,
                        .outputs = out_copy,
                        .interpolation = samp_data.interpolation,
                    },
                });
            }

            scene.allocator.free(samp_data.timestamps);
            scene.allocator.free(samp_data.outputs);
        }

        const bone_slice = try channels_list.toOwnedSlice(scene.allocator);
        const node_ch_slice = try node_channels_list.toOwnedSlice(scene.allocator);
        const node_tg_slice = try node_targets_list.toOwnedSlice(scene.allocator);
        const morph_tg_slice = try morph_targets_list.toOwnedSlice(scene.allocator);
        if (bone_slice.len == 0 and node_ch_slice.len == 0 and morph_tg_slice.len == 0) {
            if (bone_slice.len > 0) scene.allocator.free(bone_slice);
            if (node_ch_slice.len > 0) scene.allocator.free(node_ch_slice);
            if (node_tg_slice.len > 0) scene.allocator.free(node_tg_slice);
            // morph_tg_slice is empty here: no rest_weights to free.
            if (morph_tg_slice.len > 0) scene.allocator.free(morph_tg_slice);
            scene.allocator.free(anim_name);
            continue;
        }

        const ag = try AnimationGroup.init(scene.allocator, anim_name, bone_slice, max_duration);
        scene.allocator.free(anim_name);
        ag.skeleton = target_skel; // null for node-only clips
        ag.node_channels = node_ch_slice;
        ag.node_targets = node_tg_slice;
        ag.morph_targets = morph_tg_slice;
        try scene.animation_groups.append(scene.allocator, ag);
    }
}
