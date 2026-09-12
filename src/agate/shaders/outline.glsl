// Inverse-hull outline/highlight shader for agate (renders inside the main
// pass, after opaque geometry, like DebugPass/SkyboxPass).
// The rim is a constant screen-space width by construction: the normal tip
// is projected next to the base vertex, the NDC delta is renormalized in
// pixel space, and the offset is scaled by clip depth, so the width does
// not shrink with camera distance. Back faces of the inflated hull are
// drawn (front faces culled); fragments covered by the original surface
// fail the LESS_EQUAL depth test, leaving only the silhouette rim.
@header const m = @import("math")
@ctype mat4 m.Mat4

@vs vs
@glsl_options fixup_clipspace
layout(binding = 0) uniform vs_params {
    mat4 mvp;
    mat4 model;
    vec4 color; // rgb: outline color, a: opacity (pipeline blends it)
    vec4 params; // x: width_px (clamped CPU-side), y: viewport_w, z: viewport_h, w: depth_bias (NDC)
};

in vec3 position;
in vec3 normal;

out vec4 v_color;

void main() {
    vec4 clip0 = mvp * vec4(position, 1.0);

    // Silhouette direction: project the normal tip next to the base vertex
    // and take the NDC delta. The object-space tip is exact for rigid model
    // matrices (rotation + translation + uniform scale, the outline scope);
    // non-uniform scale skews the direction slightly (documented limit).
    vec4 probe = mvp * vec4(position + normal, 1.0);
    float w0 = max(clip0.w, 1e-6);
    float w1 = max(probe.w, 1e-6);
    vec2 dir_ndc = probe.xy / w1 - clip0.xy / w0;

    // Renormalize in pixel space so the width is uniform on non-square
    // viewports, then convert back to NDC. Multiplying by clip depth keeps
    // the width constant in screen space at any camera distance.
    vec2 vp = max(params.yz, vec2(1.0));
    vec2 dir_px = dir_ndc * vp * 0.5;
    float len_px = max(length(dir_px), 1e-4);
    vec2 offset_ndc = dir_px / len_px * (params.x * 2.0 / vp);

    vec4 clip = clip0;
    clip.xy += offset_ndc * w0;
    clip.z -= params.w * w0;
    gl_Position = clip;
    v_color = color;
}
@end

@fs fs
in vec4 v_color;

out vec4 frag_color;

void main() {
    frag_color = v_color;
}
@end

@program outline vs fs
