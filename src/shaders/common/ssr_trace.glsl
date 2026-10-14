vec3 reconstructWorldPos(vec2 uv, float depth) {
    vec4 clip = vec4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    vec4 world = inv_view_proj * clip;
    return world.xyz / world.w;
}
vec3 reconstructWorldNormal(vec2 uv, vec3 world_pos) {
    vec2 texel = resolution.zw;
    float d_r = texture(sampler2D(depth_tex, depth_smp), uv + vec2(texel.x, 0.0)).r;
    float d_l = texture(sampler2D(depth_tex, depth_smp), uv - vec2(texel.x, 0.0)).r;
    float d_u = texture(sampler2D(depth_tex, depth_smp), uv - vec2(0.0, texel.y)).r;
    float d_d = texture(sampler2D(depth_tex, depth_smp), uv + vec2(0.0, texel.y)).r;

    vec3 p_r = reconstructWorldPos(uv + vec2(texel.x, 0.0), d_r);
    vec3 p_l = reconstructWorldPos(uv - vec2(texel.x, 0.0), d_l);
    vec3 p_u = reconstructWorldPos(uv - vec2(0.0, texel.y), d_u);
    vec3 p_d = reconstructWorldPos(uv + vec2(0.0, texel.y), d_d);

    // Choose the continuous side by VIEW depth, not world Z. For the
    // perspective cameras clip.w is view depth; VP's fourth row gives its
    // world-space axis without another reconstruction or any new uniform.
    // CPU mirror: postprocess/depth_normal.zig depthNormalFromNeighbors.
    vec3 view_forward = vec3(view_proj[0].w, view_proj[1].w, view_proj[2].w);
    vec3 dr = p_r - world_pos;
    vec3 dl = world_pos - p_l;
    vec3 du = p_u - world_pos;
    vec3 dd = world_pos - p_d;
    vec3 dx = (abs(dot(dr, view_forward)) < abs(dot(dl, view_forward))) ? dr : dl;
    vec3 dy = (abs(dot(du, view_forward)) < abs(dot(dd, view_forward))) ? du : dd;
    vec3 n = cross(dx, dy);
    float len_sq = dot(n, n);
    // Test the gradient angle, not the pixel footprint in world units.
    // The old absolute 1e-6 cutoff replaced ordinary wall normals with up,
    // allowing SSR to paint rectangular reflection bands onto verticals.
    // Invalid normals contribute no SSR/contact/SSGI, not a fictitious floor.
    // CPU mirror: postprocess/depth_normal.zig depthNormalFromGradients.
    float scale = dot(dx, dx) * dot(dy, dy);
    return (len_sq > max(scale * 1e-8, 1e-30)) ? (n * inversesqrt(len_sq)) : vec3(0.0);
}
