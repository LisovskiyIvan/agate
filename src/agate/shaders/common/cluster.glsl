struct ClusterLight {
    vec4 pos_range; // xyz: position, w: radius
    vec4 color_int; // rgb: color, a: intensity
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
