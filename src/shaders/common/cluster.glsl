struct ClusterLight {
    vec4 pos_range;   // xyz: position, w: radius
    vec4 color_int;   // rgb: color, a: intensity
    vec4 dir_inner;   // xyz: direction, w: cos_inner (< -1.5 for point light)
    vec4 spot_params; // x: cos_outer, y: shadow_type (0: none, 1: point, 2: spot), z: shadow_slot (0 or 1), w: shadow_bias
};
layout(std430, binding = 12) readonly buffer ssbo_cluster_lights {
    ClusterLight cluster_lights[];
};
struct ClusterTile {
    uvec2 head; // x: index-list offset, y: light count
};
layout(std430, binding = 13) readonly buffer ssbo_cluster_tiles {
    ClusterTile cluster_tiles[];
};
struct ClusterIndex {
    uint light;
};
layout(std430, binding = 14) readonly buffer ssbo_cluster_indices {
    ClusterIndex cluster_indices[];
};
