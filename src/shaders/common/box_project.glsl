// Parallax-corrected reflection vector for interior reflection probes (Lagarde 2012).
// Given an uncorrected reflection ray R from fragment world position world_pos,
// with probe center probe_pos and box half-extent box_extent,
// intersects the ray with the axis-aligned bounding box and returns the corrected
// reflection direction from probe_pos toward the box intersection point.
// Returns R unmodified if box_extent <= 0.001 (unbounded probe).
vec3 boxProjectReflection(vec3 R, vec3 world_pos, vec3 probe_pos, float box_extent) {
    if (box_extent <= 0.001) {
        return R;
    }
    vec3 box_min = probe_pos - vec3(box_extent);
    vec3 box_max = probe_pos + vec3(box_extent);
    vec3 inv_R = vec3(
        abs(R.x) > 1e-6 ? 1.0 / R.x : (R.x >= 0.0 ? 1e6 : -1e6),
        abs(R.y) > 1e-6 ? 1.0 / R.y : (R.y >= 0.0 ? 1e6 : -1e6),
        abs(R.z) > 1e-6 ? 1.0 / R.z : (R.z >= 0.0 ? 1e6 : -1e6)
    );
    vec3 t_min = (box_min - world_pos) * inv_R;
    vec3 t_max = (box_max - world_pos) * inv_R;
    vec3 t_far = max(t_min, t_max);
    float t = min(t_far.x, min(t_far.y, t_far.z));
    if (t <= 0.0) {
        return R;
    }
    vec3 intersect_pos = world_pos + R * t;
    vec3 corrected = intersect_pos - probe_pos;
    float len = length(corrected);
    return len > 1e-6 ? (corrected / len) : R;
}
