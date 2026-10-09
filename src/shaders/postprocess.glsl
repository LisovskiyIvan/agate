// Fullscreen Post-Processing Shader for agate
// Supports ACES Filmic & Reinhard Tone Mapping, Bloom, Vignette, Chromatic Aberration, Saturation & Contrast,
// White Balance, Sharpen, Film Grain
@header const m = @import("math")
@ctype mat4 m.Mat4

@vs vs
@glsl_options fixup_clipspace
// @include "common/fullscreen_vs.glsl"
@end

@fs fs
// Shared finite-radiance guard (fragment scope; parent enables the build
// include for this shader).
// @include "common/linear_output.glsl"
layout(binding = 0) uniform fs_params {
    vec4 params1; // x: exposure, y: bloom_threshold (pyramid prefilter; composite ignores), z: bloom_intensity, w: unused (parent sets 0)
    vec4 params2; // x: vignette_intensity, y: vignette_radius, z: saturation, w: contrast
    vec4 params3; // x: tonemapping (0=none, 1=ACES, 2=Reinhard), y: chromatic_aberration, z: bloom_available (1/0, parent packs enabled && view valid), w: vignette_enabled (1/0)
    vec4 ssao_params; // x: ssao_enabled (1/0), y: ssao_debug (1/0), z: ssao_intensity, w: spare
    vec4 fxaa_params; // x: fxaa_enabled (1/0), yzw: spare
    vec4 resolution; // xy: resolution, zw: texel size (1.0/width, 1.0/height)
    vec4 camera_params; // x: near_z, y: far_z, zw: unused (0, 0)
    vec4 camera_pos; // xyz: camera world pos, w: unused
    vec4 sun_dir; // xyz: sun direction (normalized), w: unused
    vec4 sun_color; // xyz: sun color, w: unused
    vec4 fog_params; // x: fog_enabled (1/0), y: fog_density, z: fog_height_falloff, w: fog_start_distance
    vec4 fog_color; // xyz: fog_color, w: fog_sun_scattering
    vec4 ssr_params; // x: ssr_enabled (1/0), y: ssr_intensity, z: ssr_thickness, w: ssr_max_distance
    vec4 ssr_params2; // x: raymarch steps, yzw: spare (0)
    vec4 params5; // x: sharpen_amount (0=off), y: grain_intensity (0=off), z: temperature [-1,1], w: tint [-1,1]
    vec4 dof_params; // x: dof_enabled (1/0), y: focus_distance, z: focus_range, w: max_blur_px
    vec4 glow_params; // x: glow_enabled (1/0), y: intensity, z/w: unused
    vec4 glow_tint; // xyz: glow color multiplier, w: unused
    vec4 highlight_params; // x: highlight_enabled (1/0), y: baked global scale (always 1.0: per-item intensity folds into the mask), z/w: unused
    vec4 grade_shadows; // xyz: shadows lift [-1,1], w: unused
    vec4 grade_midtones; // xyz: midtones lift [-1,1], w: unused
    vec4 grade_highlights; // xyz: highlights lift [-1,1], w: unused
    vec4 lut_params; // x: lut_enabled (1/0), y: lut_strength [0,1], z: lut size N, w: unused
    mat4 view_proj; // camera view-projection matrix
    mat4 inv_view_proj; // inverse view-projection matrix
    mat4 reproj_mat; // combined reprojection matrix (prev_view_proj * inv_view_proj)
    vec4 motion_blur_params; // x: motion_blur_enabled (1/0), y: intensity, z: max_blur_px, w: samples
    vec4 taa_params; // x: taa_enabled (1/0), y: history blend [0,1], z: clamp strength [0,1], w: sharpen amount [0,1]
    vec4 taa_state; // x: history_valid (1/0), y: capture_only (1/0), zw: unused
    vec4 shaft_params; // x: shaft_enabled (1/0), y: intensity, zw: unused
    vec4 contact_shadow_params; // x: enabled (1/0), y: intensity, z: distance, w: thickness
    vec4 contact_shadow_params2; // x: raymarch steps, yzw: spare
    vec4 ssgi_params; // x: enabled (1/0), y: intensity [0,1], z: radius (m), w: gather steps
    vec4 local_tonemap_params; // x: enabled (1/0), y: intensity, z: contrast, w: unused
    // APPENDED LAST (display transfer): x = manual display encode needed
    // (1 = UNORM swapchain: encode exact piecewise sRGB once at the very end;
    // 0 = sRGB target: output linear, hardware encodes). yzw unused. Packed
    // by the parent.
    vec4 output_params;
};

layout(binding = 0) uniform texture2D scene_tex;
layout(binding = 1) uniform texture2D ssao_tex;
@image_sample_type depth_tex unfilterable_float
layout(binding = 2) uniform texture2D depth_tex;
layout(binding = 3) uniform texture2D bloom_tex;
layout(binding = 4) uniform texture2D lut_tex;
layout(binding = 5) uniform texture2D history_tex;
layout(binding = 6) uniform texture2D glow_tex;
layout(binding = 7) uniform texture2D highlight_tex;
layout(binding = 8) uniform texture2D highlight_mask_tex;
layout(binding = 9) uniform texture2D shaft_tex;
layout(binding = 10) uniform texture2D velocity_tex;

layout(binding = 0) uniform sampler smp;
@sampler_type depth_smp nonfiltering
layout(binding = 1) uniform sampler depth_smp;
@sampler_type velocity_smp nonfiltering
layout(binding = 3) uniform sampler velocity_smp;
layout(binding = 2) uniform sampler lut_smp;

in vec2 v_uv;
out vec4 frag_color;

vec3 ACESFilm(vec3 x) {
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

vec3 Reinhard(vec3 x) {
    return x / (x + vec3(1.0));
}

vec3 UchimuraTonemap(vec3 x) {
    const float P = 1.0;
    const float a = 1.0;
    const float m = 0.22;
    const float l = 0.4;
    const float c = 1.33;
    const float b = 0.0;
    float l0 = ((P - m) * l) / a;
    float S0 = m + l0;
    float S1 = m + a * l0;
    float C2 = (a * P) / (P - S1);
    float cp = -C2 / P;

    vec3 w0 = vec3(1.0) - smoothstep(vec3(0.0), vec3(m), x);
    vec3 w2 = step(vec3(m + l0), x);
    vec3 w1 = vec3(1.0) - w0 - w2;

    vec3 T = m * pow(max(x / m, vec3(0.0)), vec3(c)) + b;
    vec3 S = P - (P - S1) * exp(cp * (x - S0));
    vec3 L = m + a * (x - m);

    return clamp(T * w0 + L * w1 + S * w2, 0.0, 1.0);
}

vec3 AgXTonemap(vec3 color) {
    if (color.r <= 0.0 && color.g <= 0.0 && color.b <= 0.0) return vec3(0.0);

    const mat3 agx_mat = mat3(
        0.842479062253094, 0.0423282422610123, 0.0423756549057051,
        0.0784335996993431, 0.878468636469772, 0.0784336099914461,
        0.0792237451477422, 0.0791661274605434, 0.879142973798673
    );
    const mat3 agx_mat_inv = mat3(
        1.19687900512017, -0.0528968517590771, -0.0529716355084725,
        -0.0980208811401368, 1.15190312990417, -0.0980434501171241,
        -0.0990297440797205, -0.098961176813784, 1.15107367264185
    );
    color = max(color, vec3(1e-6));
    color = agx_mat * color;

    const float min_ev = -10.0;
    const float max_ev = 6.5;
    color = clamp((log2(max(color, vec3(1e-6))) - min_ev) / (max_ev - min_ev), 0.0, 1.0);

    vec3 x2 = color * color;
    vec3 x4 = x2 * x2;
    color = + 15.5 * x4 * color
            - 40.14 * x4
            + 31.96 * x2 * color
            - 6.868 * x2
            + 0.4298 * color
            + 0.1191;
    color = clamp((color - vec3(0.1191)) / (1.0 - 0.1191), 0.0, 1.0);
    color = agx_mat_inv * color;
    return clamp(color, 0.0, 1.0);
}

vec3 PBRNeutralTonemap(vec3 color) {
    const float startCompression = 0.8 - 0.04;
    const float desaturation = 0.15;
    float x = min(color.r, min(color.g, color.b));
    float offset = (x < 0.08) ? (x - 6.25 * x * x) : 0.04;
    color = max(color - offset, vec3(0.0));
    float peak = max(color.r, max(color.g, color.b));
    if (peak < startCompression) return color;
    const float d = 1.0 - startCompression;
    float newPeak = 1.0 - d * d / (peak + d - startCompression);
    color *= newPeak / peak;
    float g = 1.0 - 1.0 / (desaturation * (peak - newPeak) + 1.0);
    return clamp(mix(color, newPeak * vec3(1.0), g), 0.0, 1.0);
}

vec3 applyLocalTonemapping(vec3 color, vec2 uv) {
    if (local_tonemap_params.x < 0.5) return color;
    float intensity = local_tonemap_params.y;
    if (intensity <= 0.001) return color;
    float contrast = local_tonemap_params.z;

    float local_luma;
    if (params3.z > 0.5) {
        vec3 bloom_sample = texture(sampler2D(bloom_tex, smp), uv).rgb;
        local_luma = dot(bloom_sample, vec3(0.2126, 0.7152, 0.0722));
    } else {
        vec2 off = resolution.zw * 24.0;
        float l0 = dot(texture(sampler2D(scene_tex, smp), uv).rgb, vec3(0.2126, 0.7152, 0.0722));
        float l1 = dot(texture(sampler2D(scene_tex, smp), uv + vec2(off.x, off.y)).rgb, vec3(0.2126, 0.7152, 0.0722));
        float l2 = dot(texture(sampler2D(scene_tex, smp), uv - vec2(off.x, off.y)).rgb, vec3(0.2126, 0.7152, 0.0722));
        float l3 = dot(texture(sampler2D(scene_tex, smp), uv + vec2(-off.x, off.y)).rgb, vec3(0.2126, 0.7152, 0.0722));
        float l4 = dot(texture(sampler2D(scene_tex, smp), uv + vec2(off.x, -off.y)).rgb, vec3(0.2126, 0.7152, 0.0722));
        local_luma = (l0 + l1 + l2 + l3 + l4) * 0.2;
    }

    float pixel_luma = dot(color, vec3(0.2126, 0.7152, 0.0722));
    float denom = max(0.18 + local_luma, 1e-4);
    float numer = max(0.18 + pixel_luma, 1e-4);
    float scale = pow(numer / denom, contrast);
    scale = clamp(scale, 0.25, 4.0);

    vec3 adapted_color = color * scale;
    return mix(color, adapted_color, intensity);
}

vec3 reconstructWorldPos(vec2 uv, float depth) {
    vec4 clip = vec4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    vec4 world = inv_view_proj * clip;
    return world.xyz / world.w;
}

vec2 reprojectClipToPrevUv(vec4 clip) {
    vec4 prev_clip = reproj_mat * clip;
    if (prev_clip.w <= 0.0001) return vec2(-1.0);
    vec2 prev_ndc = prev_clip.xy / prev_clip.w;
    return vec2(prev_ndc.x * 0.5 + 0.5, 0.5 - prev_ndc.y * 0.5);
}

vec2 projectWorldToUv(vec3 world_pos) {
    vec4 clip = view_proj * vec4(world_pos, 1.0);
    if (clip.w <= 0.0001) return vec2(-1.0);
    vec3 ndc = clip.xyz / clip.w;
    return vec2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
}

vec3 projectWorldToUvDepth(vec3 world_pos) {
    vec4 clip = view_proj * vec4(world_pos, 1.0);
    if (clip.w <= 0.0001) return vec3(-1.0, -1.0, 1.0);
    vec3 ndc = clip.xyz / clip.w;
    return vec3(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5, ndc.z);
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

    vec3 dx = (abs(p_r.z - world_pos.z) < abs(p_l.z - world_pos.z)) ? (p_r - world_pos) : (world_pos - p_l);
    vec3 dy = (abs(p_u.z - world_pos.z) < abs(p_d.z - world_pos.z)) ? (p_u - world_pos) : (world_pos - p_d);
    vec3 n = cross(dx, dy);
    float len_sq = dot(n, n);
    return (len_sq > 1e-6) ? (n * inversesqrt(len_sq)) : vec3(0.0, 1.0, 0.0);
}

vec3 applySSR(vec3 scene_color, vec2 uv, float raw_depth) {
    if (ssr_params.x < 0.5) return scene_color;
    if (raw_depth >= 0.9999) return scene_color;

    vec3 world_pos = reconstructWorldPos(uv, raw_depth);
    vec3 V = normalize(world_pos - camera_pos.xyz);
    vec3 N = reconstructWorldNormal(uv, world_pos);

    // Only reflect on upward horizontal surfaces (floor/ground)
    if (N.y < 0.45) return scene_color;

    vec3 R = reflect(V, N);
    if (R.y < 0.02) return scene_color;

    float max_dist = ssr_params.w;
    float thickness = ssr_params.z;
    int ssr_steps = int(ssr_params2.x);
    if (ssr_steps < 4) ssr_steps = 16;
    float step_size = max_dist / float(ssr_steps);

    vec3 ray_start = world_pos + N * 0.08;
    vec3 ray_dir_step = R * step_size;

    vec4 clip_start = view_proj * vec4(ray_start, 1.0);
    vec4 clip_step = view_proj * vec4(ray_dir_step, 0.0);

    for (int i = 1; i <= ssr_steps; i++) {
        vec4 march_clip = clip_start + clip_step * float(i);
        if (march_clip.w <= 0.0001) break;

        vec3 ndc = march_clip.xyz / march_clip.w;
        vec2 march_uv = vec2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
        if (march_uv.x < 0.01 || march_uv.x > 0.99 || march_uv.y < 0.01 || march_uv.y > 0.99) {
            break;
        }

        float scene_d = texture(sampler2D(depth_tex, depth_smp), march_uv).r;
        if (scene_d >= 0.9999) continue;
        if (ndc.z < scene_d) continue;

        vec3 ray_pos = ray_start + ray_dir_step * float(i);
        vec3 scene_pos = reconstructWorldPos(march_uv, scene_d);

        float ray_cam_dist = length(ray_pos - camera_pos.xyz);
        float scene_cam_dist = length(scene_pos - camera_pos.xyz);
        float depth_diff = ray_cam_dist - scene_cam_dist;
        float dist_to_surface = length(ray_pos - scene_pos);

        if (depth_diff >= 0.0 && dist_to_surface < thickness) {
            float edge_dist_x = min(march_uv.x, 1.0 - march_uv.x);
            float edge_dist_y = min(march_uv.y, 1.0 - march_uv.y);
            float edge_fade = clamp(min(edge_dist_x, edge_dist_y) * 10.0, 0.0, 1.0);

            float dist_fade = 1.0 - (float(i - 1) / float(ssr_steps));
            dist_fade *= dist_fade;

            float fresnel = 0.04 + 0.96 * pow(1.0 - max(0.0, dot(-V, N)), 5.0);
            fresnel = clamp(fresnel * 1.8, 0.15, 0.9);

            vec3 refl_color = texture(sampler2D(scene_tex, smp), march_uv).rgb;
            scene_color = mix(scene_color, refl_color, ssr_params.y * edge_fade * dist_fade * fresnel);
            break;
        }
    }

    return scene_color;
}

vec3 applyContactShadows(vec3 scene_color, vec2 uv, float raw_depth) {
    if (contact_shadow_params.x < 0.5) return scene_color;
    if (raw_depth >= 0.9999) return scene_color;

    float intensity = contact_shadow_params.y;
    if (intensity <= 0.001) return scene_color;

    vec3 L = sun_dir.xyz;
    if (dot(L, L) < 0.1) return scene_color;

    vec3 world_pos = reconstructWorldPos(uv, raw_depth);
    vec3 N = reconstructWorldNormal(uv, world_pos);

    float NdotL = dot(N, L);
    if (NdotL <= 0.0) return scene_color;

    float max_dist = contact_shadow_params.z;
    float thickness = contact_shadow_params.w;
    int steps = int(contact_shadow_params2.x);
    if (steps < 4) steps = 12;

    float step_size = max_dist / float(steps);
    // Ray offset from surface to avoid self-shadowing acne
    vec3 ray_start = world_pos + N * 0.015 + L * 0.008;
    vec3 ray_dir_step = L * step_size;

    vec4 clip_start = view_proj * vec4(ray_start, 1.0);
    vec4 clip_step = view_proj * vec4(ray_dir_step, 0.0);

    float jitter = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
    float occlusion = 0.0;

    for (int i = 1; i <= steps; i++) {
        float t = float(i) - (1.0 - jitter);
        vec4 march_clip = clip_start + clip_step * t;
        if (march_clip.w <= 0.0001) break;

        vec3 ndc = march_clip.xyz / march_clip.w;
        vec2 march_uv = vec2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);

        if (march_uv.x < 0.005 || march_uv.x > 0.995 || march_uv.y < 0.005 || march_uv.y > 0.995) {
            break;
        }

        float scene_d = texture(sampler2D(depth_tex, depth_smp), march_uv).r;
        if (scene_d >= 0.9999) continue;

        if (ndc.z < scene_d) continue;

        vec3 ray_pos = ray_start + ray_dir_step * t;
        vec3 scene_pos = reconstructWorldPos(march_uv, scene_d);

        float ray_cam_dist = length(ray_pos - camera_pos.xyz);
        float scene_cam_dist = length(scene_pos - camera_pos.xyz);
        float depth_diff = ray_cam_dist - scene_cam_dist;

        if (depth_diff >= 0.0 && depth_diff < thickness) {
            vec2 edge_dist = min(march_uv, vec2(1.0) - march_uv);
            float edge_fade = clamp(min(edge_dist.x, edge_dist.y) * 20.0, 0.0, 1.0);

            float dist_ratio = t / float(steps);
            float dist_fade = 1.0 - dist_ratio * dist_ratio;
            float hit_factor = 1.0 - (depth_diff / thickness);

            occlusion = hit_factor * dist_fade * edge_fade;
            break;
        }
    }

    if (occlusion > 0.0) {
        float shadow_attenuation = clamp(1.0 - occlusion * intensity * NdotL, 0.0, 1.0);
        scene_color *= shadow_attenuation;
    }

    return scene_color;
}

// Screen-space one-bounce diffuse GI (v1 color bleed): jittered disk
// gather; each depth-hit neighbor inside the world-space influence radius
// contributes its HDR radiance weighted by hemisphere cosine and squared
// linear falloff (CPU mirror: postprocess/ssgi.zig ssgiWeight). The
// weighted sum normalizes into a bleed color scaled by gather coverage
// and intensity, then ADDS to the lit color (forward composite carries no
// albedo GBuffer — the lit color already encodes it). Per-pixel jitter
// hands the denoise to TAA; a per-sample luma cap keeps HDR fireflies
// out. Sensitive to depth-only neighbors: thin geometry without radiance
// data contributes nothing (depth-gated like SSR).
const float SSGI_LUMA_CAP = 8.0; // per-sample HDR firefly cap (CPU mirror: ssgi.SSGI_LUMA_CAP)

vec3 applySSGI(vec3 scene_color, vec2 uv, float raw_depth) {
    if (raw_depth >= 0.9999) return scene_color;
    float intensity = ssgi_params.y;
    if (intensity <= 0.001) return scene_color;
    float radius = max(ssgi_params.z, 0.05);
    int steps = int(ssgi_params.w);
    if (steps < 4) steps = 8;

    vec3 world_pos = reconstructWorldPos(uv, raw_depth);
    vec3 N = reconstructWorldNormal(uv, world_pos);

    // Roughly constant WORLD radius: scale the pixel disk by camera
    // distance; clamp so the gather stays local (2..48 px).
    float cam_dist = max(length(world_pos - camera_pos.xyz), 0.05);
    float px_radius = clamp((radius / cam_dist) * resolution.y * 0.25, 2.0, 48.0);

    float jitter = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
    vec3 bleed = vec3(0.0);
    float total_w = 0.0;
    for (int i = 0; i < steps; i++) {
        float fi = float(i);
        float ang = 6.2831853 * (fi + jitter) / float(steps);
        float r = px_radius * sqrt((fi + 0.5) / float(steps));
        vec2 s_uv = uv + vec2(cos(ang), sin(ang)) * r * resolution.zw;
        if (s_uv.x <= 0.002 || s_uv.x >= 0.998 || s_uv.y <= 0.002 || s_uv.y >= 0.998) continue;

        float s_d = texture(sampler2D(depth_tex, depth_smp), s_uv).r;
        if (s_d >= 0.9999) continue;
        vec3 s_pos = reconstructWorldPos(s_uv, s_d);
        vec3 to_s = s_pos - world_pos;
        float dist = length(to_s);
        if (dist >= radius) continue;
        float cos_nd = dot(N, to_s / max(dist, 1e-4));
        if (cos_nd <= 0.0) continue;

        vec3 s_col = texture(sampler2D(scene_tex, smp), s_uv).rgb;
        float falloff = 1.0 - dist / radius;
        float w = cos_nd * falloff * falloff;
        bleed += min(s_col, vec3(SSGI_LUMA_CAP)) * w;
        total_w += w;
    }
    if (total_w <= 0.0001) return scene_color;

    vec3 bleed_color = bleed / total_w;
    float coverage = clamp(total_w / float(steps) * 3.0, 0.0, 1.0);
    return scene_color + bleed_color * (coverage * intensity);
}

vec3 applyAtmosphericFog(vec3 scene_color, vec2 uv, float raw_depth) {
    if (fog_params.x < 0.5) return scene_color;

    vec3 world_pos = reconstructWorldPos(uv, raw_depth);
    vec3 to_pixel = world_pos - camera_pos.xyz;
    float dist = length(to_pixel);
    vec3 ray_dir = (dist > 0.0001) ? (to_pixel / dist) : vec3(0.0, 0.0, 1.0);

    // Directional Sun Inscattering (atmospheric Mie glow)
    float sun_dot = max(0.0, dot(ray_dir, sun_dir.xyz));
    float sun_inscatter = pow(sun_dot, 8.0) * fog_color.w;
    vec3 current_fog_color = mix(fog_color.rgb, sun_color.rgb * 1.5, sun_inscatter);

    // Skybox horizon haze
    if (raw_depth >= 0.9999) {
        float horizon_haze = clamp(1.0 - abs(ray_dir.y) * 4.0, 0.0, 1.0);
        float sky_fog = horizon_haze * clamp(fog_params.y * 15.0, 0.0, 0.7);
        return mix(scene_color, current_fog_color, sky_fog);
    }

    // Effective distance beyond start distance
    float fog_start = fog_params.w;
    float eff_dist = max(0.0, dist - fog_start);

    // Exponential Height Falloff
    float falloff = fog_params.z;
    float delta_y = world_pos.y - camera_pos.y;
    float height_density = (abs(delta_y) > 0.001)
        ? (exp(-camera_pos.y * falloff) - exp(-world_pos.y * falloff)) / (delta_y * falloff)
        : exp(-camera_pos.y * falloff);
    height_density = clamp(height_density, 0.0, 5.0);

    // Beer-Lambert transmittance: extinction = 1.0 - exp(-optical_depth)
    float optical_depth = eff_dist * fog_params.y * height_density;
    float fog_amount = clamp(1.0 - exp(-optical_depth), 0.0, 1.0);
    return mix(scene_color, current_fog_color, fog_amount);
}

// Camera & Object Motion Blur: gathers samples along screen velocity derived from velocity buffer or depth reprojection
vec3 applyMotionBlur(vec3 color, vec2 uv, float depth) {
    if (motion_blur_params.x < 0.5) return color;
    if (depth >= 1.0) return color;

    vec4 vel_sample = texture(sampler2D(velocity_tex, velocity_smp), uv);
    vec2 velocity;
    if (vel_sample.a > 0.5) {
        velocity = vel_sample.xy * motion_blur_params.y;
    } else {
        vec4 clip = vec4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
        vec2 prev_uv = reprojectClipToPrevUv(clip);
        if (prev_uv.x < 0.0) return color;
        velocity = (uv - prev_uv) * motion_blur_params.y;
    }
    float max_blur = motion_blur_params.z * resolution.z;
    float speed = length(velocity);
    if (speed > max_blur) {
        velocity = velocity * (max_blur / speed);
    }
    // Sub-pixel threshold: don't blur when movement is sub-pixel (saves full loop on almost-static areas)
    if (speed < resolution.z * 0.75) return color;

    int samples = int(motion_blur_params.w);
    if (samples < 2) samples = 8;

    vec3 acc = color;
    for (int i = 1; i < samples; ++i) {
        float t = float(i) / float(samples - 1) - 0.5;
        vec2 sample_uv = clamp(uv + velocity * t, vec2(0.001), vec2(0.999));
        acc += texture(sampler2D(scene_tex, smp), sample_uv).rgb;
    }
    return acc * (1.0 / float(samples));
}

// Add the raymarched shaft radiance before exposure and tone mapping. This
// keeps the shaft contribution in the same linear HDR domain as scene color.
vec3 addShaftRadiance(vec3 color, vec2 uv) {
    if (shaft_params.x > 0.5 && shaft_params.y > 0.001) {
        color += texture(sampler2D(shaft_tex, smp), uv).rgb * shaft_params.y;
    }
    return color;
}

// Sample linear-radiance scene color: chromatic aberration, then the
// depth-dependent passes (motion blur, SSR, SSAO, atmospheric fog).
vec3 sampleScene(vec2 uv) {
    vec3 base_color;
    float ca = params3.y;
    if (ca > 0.00001) {
        vec2 dist_from_center = uv - 0.5;
        vec2 ca_offset = dist_from_center * ca;
        float r = texture(sampler2D(scene_tex, smp), uv + ca_offset).r;
        float g = texture(sampler2D(scene_tex, smp), uv).g;
        float b = texture(sampler2D(scene_tex, smp), uv - ca_offset).b;
        base_color = vec3(r, g, b);
    } else {
        base_color = texture(sampler2D(scene_tex, smp), uv).rgb;
    }

    vec3 color = base_color;

    // Depth-dependent passes: Motion Blur, Contact Shadows, SSR, SSAO, and Atmospheric Fog
    float raw_depth = 0.0;
    bool needs_depth = (ssr_params.x > 0.5 || fog_params.x > 0.5 || motion_blur_params.x > 0.5 || contact_shadow_params.x > 0.5 || ssgi_params.x > 0.5);
    if (needs_depth) {
        raw_depth = texture(sampler2D(depth_tex, depth_smp), uv).r;
    }

    // Camera Motion Blur: gathers HDR scene samples along screen velocity
    if (motion_blur_params.x > 0.5) {
        color = applyMotionBlur(color, uv, raw_depth);
    }

    // Screen-Space Contact Shadows & Local Occlusion
    if (contact_shadow_params.x > 0.5) {
        color = applyContactShadows(color, uv, raw_depth);
    }

    // Screen-Space Reflections (SSR)
    if (ssr_params.x > 0.5) {
        color = applySSR(color, uv, raw_depth);
    }

    // SSAO Occlusion: applied to the resolved scene color so motion blur never washes out or erases AO
    if (ssao_params.x > 0.5) {
        float ao = clamp(texture(sampler2D(ssao_tex, smp), uv).r, 0.0, 1.0);
        float ao_factor = clamp(1.0 - (1.0 - ao) * ssao_params.z, 0.0, 1.0);
        color *= ao_factor;
    }

    // Screen-Space GI (v1 color bleed): additive one-bounce indirect
    // before fog so aerial perspective covers it like direct light.
    if (ssgi_params.x > 0.5) {
        color = applySSGI(color, uv, raw_depth);
    }

    // Atmospheric Depth & Height Fog (in-scattering over the occluded surface)
    if (fog_params.x > 0.5) {
        color = applyAtmosphericFog(color, uv, raw_depth);
    }

    return color;
}

float rgbToLuma(vec3 c) {
    return dot(c, vec3(0.299, 0.587, 0.114));
}

// Sin-free integer-style hash (David Hoskins) for film grain.
// Stable per screen-space pixel; static (no time input).
float grainHash(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// Fast tonemapped tap for perceptual proxies (sharpen): single scene fetch
// + shaft + exposure + the existing Reinhard/ACES. Skips re-running SSR,
// fog and chromatic dispersion on neighbor taps.
vec3 sampleSceneFastLDR(vec2 uv) {
    vec3 color = addShaftRadiance(texture(sampler2D(scene_tex, smp), uv).rgb, uv) * params1.x; // Exposure
    float tonemap_mode = params3.x;
    if (tonemap_mode > 1.5) {
        color = Reinhard(color);
    } else if (tonemap_mode > 0.5) {
        color = ACESFilm(color);
    }
    return clamp(color, 0.0, 1.0);
}

// Fast luma approximation for FXAA edge classification: clamped
// tonemapped luma as a perceptual proxy (no per-tap SSR cost).
float sampleLumaFast(vec2 uv) {
    vec3 c = addShaftRadiance(texture(sampler2D(scene_tex, smp), uv).rgb, uv);
    float luma_hdr = dot(c, vec3(0.299, 0.587, 0.114)) * params1.x;
    return luma_hdr / (luma_hdr + 1.0);
}

// FXAA 3.11 Quality Anti-Aliasing
#define FXAA_EDGE_THRESHOLD_MIN 0.0312
#define FXAA_EDGE_THRESHOLD     0.125
#define FXAA_SUBPIX_CAP         0.75
#define FXAA_SEARCH_STEPS       10

float dofLinearize(float d) {
    float near = camera_params.x;
    float far = camera_params.y;
    return (near * far) / max(far - d * (far - near), 0.0001);
}

// Parametric zone grade: per-channel lifts weighted by luminance zones.
// Mirrors applyGrade in postprocess.zig (same luma weights and ramps).
vec3 applyColorCurves(vec3 c) {
    float l = dot(c, vec3(0.2126, 0.7152, 0.0722));
    float w_s = clamp(1.0 - l * 2.0, 0.0, 1.0);
    float w_h = clamp((l - 0.5) * 2.0, 0.0, 1.0);
    float w_m = clamp(1.0 - abs(l - 0.5) * 2.0, 0.0, 1.0);
    return c + grade_shadows.xyz * w_s + grade_midtones.xyz * w_m + grade_highlights.xyz * w_h;
}

// Texture LUT color grade on a 2D strip (N*N wide, N tall; N = cube edge
// in lut_params.z). Mirrors lutStripUv/applyLutStrip in postprocess.zig,
// which document the full uv derivation. Blue picks the layer with a
// half-texel-inset t = b*(N-1); hardware bilinear interpolates r/g inside
// one layer, and the manual mix of layers floor(t) and floor(t)+1 by the
// fraction f adds the third (trilinear) axis. The graded color then blends
// back toward the curve-graded input by lut_strength, so strength 0 is
// an exact no-op. All zeros in lut_params (no LUT bound) short-circuit to
// the unchanged pre-LUT path.
vec3 applyLut(vec3 color) {
    if (lut_params.x < 0.5) return color;
    float n = lut_params.z;
    float t = clamp(color.b, 0.0, 1.0) * (n - 1.0);
    float k = floor(t);
    float f = t - k;
    float u_slice = (clamp(color.r, 0.0, 1.0) * (n - 1.0) + 0.5) / n;
    float v_slice = (clamp(color.g, 0.0, 1.0) * (n - 1.0) + 0.5) / n;
    float k1 = min(k + 1.0, n - 1.0);
    vec2 uv0 = vec2((k + u_slice) / n, v_slice);
    vec2 uv1 = vec2((k1 + u_slice) / n, v_slice);
    vec3 c0 = texture(sampler2D(lut_tex, lut_smp), uv0).rgb;
    vec3 c1 = texture(sampler2D(lut_tex, lut_smp), uv1).rgb;
    return mix(color, mix(c0, c1, f), clamp(lut_params.y, 0.0, 1.0));
}

// Exact IEC 61966-2-1 sRGB encode for the UNORM display path. Input is
// finite-bound first (NaN/Inf lanes map to 0 via boundRadiance), so the pow
// below can never see a non-finite base; output stays in 0..1 (the caller
// clamps the graded color before encoding).
vec3 linearToSrgb(vec3 c) {
    c = boundRadiance(c);
    vec3 lo = c * 12.92;
    vec3 hi = vec3(1.055) * pow(c, vec3(1.0 / 2.4)) - vec3(0.055);
    return mix(hi, lo, vec3(lessThanEqual(c, vec3(0.0031308))));
}

// Fast radiance tap for the TAA neighborhood: scene radiance + shaft, no
// exposure/tonemap (mirrors the full sampleScene minus its depth passes).
// Finite-bound at the tap so a poisoned texel (NaN/Inf) maps to 0 before it
// can widen the neighborhood box to Inf and let poisoned history through.
vec3 sampleRadianceFast(vec2 uv) {
    return boundRadiance(addShaftRadiance(texture(sampler2D(scene_tex, smp), uv).rgb, uv));
}

// TAA neighborhood: 3x3 box over linear radiance taps (values can exceed
// 1, so no 0..1 clamp here) with variance bounding. Mirrors taaVarianceBounds/
// taaNeighborhoodAvg in postprocess.zig.
void taaNeighborhood(vec2 uv, vec3 center, out vec3 box_min, out vec3 box_max, out vec3 avg) {
    vec2 texel = resolution.zw;
    box_min = center;
    box_max = center;
    vec3 sum = center;
    vec3 sum_sq = center * center;
    vec3 t;
    t = sampleRadianceFast(uv + vec2(-texel.x, -texel.y)); box_min = min(box_min, t); box_max = max(box_max, t); sum += t; sum_sq += t * t;
    t = sampleRadianceFast(uv + vec2(0.0, -texel.y)); box_min = min(box_min, t); box_max = max(box_max, t); sum += t; sum_sq += t * t;
    t = sampleRadianceFast(uv + vec2(texel.x, -texel.y)); box_min = min(box_min, t); box_max = max(box_max, t); sum += t; sum_sq += t * t;
    t = sampleRadianceFast(uv + vec2(-texel.x, 0.0)); box_min = min(box_min, t); box_max = max(box_max, t); sum += t; sum_sq += t * t;
    t = sampleRadianceFast(uv + vec2(texel.x, 0.0)); box_min = min(box_min, t); box_max = max(box_max, t); sum += t; sum_sq += t * t;
    t = sampleRadianceFast(uv + vec2(-texel.x, texel.y)); box_min = min(box_min, t); box_max = max(box_max, t); sum += t; sum_sq += t * t;
    t = sampleRadianceFast(uv + vec2(0.0, texel.y)); box_min = min(box_min, t); box_max = max(box_max, t); sum += t; sum_sq += t * t;
    t = sampleRadianceFast(uv + vec2(texel.x, texel.y)); box_min = min(box_min, t); box_max = max(box_max, t); sum += t; sum_sq += t * t;
    avg = sum / 9.0;
    vec3 sigma = sqrt(max(sum_sq / 9.0 - avg * avg, vec3(0.0)));
    box_min = max(box_min, avg - 1.25 * sigma);
    box_max = min(box_max, avg + 1.25 * sigma);
}

// TAA resolve in pre-exposure radiance: per-object velocity buffer with
// depth-reprojected camera velocity fallback; the history holds radiance (parent
// provides the RGBA16F capture target). Mirrors taaResolvePixel in
// postprocess.zig (bounds + clamp + blend + sharpen). Disabled (or no valid
// history yet) returns `current` before any history/depth sampling.
vec3 applyTAA(vec3 current, vec2 uv) {
    if (taa_params.x < 0.5) return current;
    if (taa_state.x < 0.5) return current;
    float raw_depth = texture(sampler2D(depth_tex, depth_smp), uv).r;
    if (raw_depth >= 0.9999) return current;

    // Closest depth search in 3x3 cross to prevent silhouette edge ghosting
    vec2 texel = resolution.zw;
    vec2 best_offset = vec2(0.0);
    float best_depth = raw_depth;
    float d;
    d = texture(sampler2D(depth_tex, depth_smp), uv + vec2(-texel.x, 0.0)).r;
    if (d < best_depth) { best_depth = d; best_offset = vec2(-texel.x, 0.0); }
    d = texture(sampler2D(depth_tex, depth_smp), uv + vec2(texel.x, 0.0)).r;
    if (d < best_depth) { best_depth = d; best_offset = vec2(texel.x, 0.0); }
    d = texture(sampler2D(depth_tex, depth_smp), uv + vec2(0.0, -texel.y)).r;
    if (d < best_depth) { best_depth = d; best_offset = vec2(0.0, -texel.y); }
    d = texture(sampler2D(depth_tex, depth_smp), uv + vec2(0.0, texel.y)).r;
    if (d < best_depth) { best_depth = d; best_offset = vec2(0.0, texel.y); }

    vec4 vel_sample = texture(sampler2D(velocity_tex, velocity_smp), uv + best_offset);
    vec2 prev_uv;
    if (vel_sample.a > 0.5) {
        prev_uv = uv - vel_sample.xy;
    } else {
        vec4 clip = vec4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, best_depth, 1.0);
        prev_uv = reprojectClipToPrevUv(clip);
    }
    if (prev_uv.x < 0.001 || prev_uv.x > 0.999 || prev_uv.y < 0.001 || prev_uv.y > 0.999) return current;
    vec3 hist = boundRadiance(texture(sampler2D(history_tex, smp), prev_uv).rgb);
    vec3 box_min;
    vec3 box_max;
    vec3 avg;
    taaNeighborhood(uv, current, box_min, box_max, avg);

    // Occlusion/disocclusion rejection: calculate divergence of history from variance box
    vec3 span = max(box_max - box_min, vec3(1e-4));
    vec3 dist_min = max(box_min - hist, vec3(0.0)) / span;
    vec3 dist_max = max(hist - box_max, vec3(0.0)) / span;
    float max_dist = max(max(dist_min.x, dist_max.x), max(max(dist_min.y, dist_max.y), max(dist_min.z, dist_max.z)));
    float reject_weight = 1.0 / (1.0 + max_dist * max_dist * 4.0);

    float clamp_strength = clamp(taa_params.z, 0.0, 1.0);
    vec3 hist_clamped = mix(hist, clamp(hist, box_min, box_max), clamp_strength);
    float blend = clamp(taa_params.y, 0.0, 1.0) * reject_weight;
    vec3 outc = mix(current, hist_clamped, blend);
    float sharp = clamp(taa_params.w, 0.0, 1.0);
    if (sharp > 0.0001) {
        outc = clamp(outc + (current - avg) * sharp, box_min, box_max);
    }
    return boundRadiance(outc);
}

// Full linear resolve at one uv: existing depth math (motion/SSR/SSAO/fog
// via sampleScene) + shaft, bounded to finite half. One full-cost
// sample; edge/search taps stay on the cheap proxies.
vec3 sampleRadiance(vec2 uv) {
    return boundRadiance(addShaftRadiance(sampleScene(uv), uv));
}

// FXAA 3.11 resolve in linear radiance (honors fxaa_params.x). Edge
// classification reuses the exact FXAA math, offsets and weights with the
// cheap clamped-tonemapped-luma proxy taps (sampleLumaFast, no per-tap SSR
// cost); the actual filter resolves LINEAR radiance at the walked uv, so
// the output stays pre-tonemap and is never double-toned. OFF returns the
// unmodified center resolve.
vec3 applyFXAA(vec2 uv, vec2 rcpFrame) {
    vec3 centerLin = sampleRadiance(uv);
    float lumaCenter = sampleLumaFast(uv);

    float lumaDown  = sampleLumaFast(uv + vec2(0.0, -rcpFrame.y));
    float lumaUp    = sampleLumaFast(uv + vec2(0.0,  rcpFrame.y));
    float lumaLeft  = sampleLumaFast(uv + vec2(-rcpFrame.x, 0.0));
    float lumaRight = sampleLumaFast(uv + vec2( rcpFrame.x, 0.0));

    float lumaMin = min(lumaCenter, min(min(lumaDown, lumaUp), min(lumaLeft, lumaRight)));
    float lumaMax = max(lumaCenter, max(max(lumaDown, lumaUp), max(lumaLeft, lumaRight)));
    float lumaRange = lumaMax - lumaMin;

    if (lumaRange < max(FXAA_EDGE_THRESHOLD_MIN, lumaMax * FXAA_EDGE_THRESHOLD)) {
        return centerLin;
    }

    float lumaDownLeft  = sampleLumaFast(uv + vec2(-rcpFrame.x, -rcpFrame.y));
    float lumaUpRight   = sampleLumaFast(uv + vec2( rcpFrame.x,  rcpFrame.y));
    float lumaUpLeft    = sampleLumaFast(uv + vec2(-rcpFrame.x,  rcpFrame.y));
    float lumaDownRight = sampleLumaFast(uv + vec2( rcpFrame.x, -rcpFrame.y));

    float lumaDownUp = lumaDown + lumaUp;
    float lumaLeftRight = lumaLeft + lumaRight;

    float lumaLeftCorners = lumaDownLeft + lumaUpLeft;
    float lumaDownCorners = lumaDownLeft + lumaDownRight;
    float lumaRightCorners = lumaDownRight + lumaUpRight;
    float lumaUpCorners = lumaUpRight + lumaUpLeft;

    float edgeHorizontal = abs(-2.0 * lumaLeft + lumaLeftCorners) +
                           abs(-2.0 * lumaCenter + lumaDownUp) * 2.0 +
                           abs(-2.0 * lumaRight + lumaRightCorners);
    float edgeVertical   = abs(-2.0 * lumaUp + lumaUpCorners) +
                           abs(-2.0 * lumaCenter + lumaLeftRight) * 2.0 +
                           abs(-2.0 * lumaDown + lumaDownCorners);

    bool isHorizontal = (edgeHorizontal >= edgeVertical);

    float luma1 = isHorizontal ? lumaDown : lumaLeft;
    float luma2 = isHorizontal ? lumaUp : lumaRight;
    float gradient1 = abs(luma1 - lumaCenter);
    float gradient2 = abs(luma2 - lumaCenter);

    bool is1Steeper = gradient1 >= gradient2;
    float gradientScaled = 0.25 * max(gradient1, gradient2);

    float stepLength = isHorizontal ? rcpFrame.y : rcpFrame.x;
    float lumaLocalAverage = 0.0;

    if (is1Steeper) {
        stepLength = -stepLength;
        lumaLocalAverage = 0.5 * (luma1 + lumaCenter);
    } else {
        lumaLocalAverage = 0.5 * (luma2 + lumaCenter);
    }

    vec2 currentUv = uv;
    if (isHorizontal) {
        currentUv.y += stepLength * 0.5;
    } else {
        currentUv.x += stepLength * 0.5;
    }

    vec2 offset = isHorizontal ? vec2(rcpFrame.x, 0.0) : vec2(0.0, rcpFrame.y);
    vec2 uv1 = currentUv - offset;
    vec2 uv2 = currentUv + offset;

    float lumaEnd1 = sampleLumaFast(uv1) - lumaLocalAverage;
    float lumaEnd2 = sampleLumaFast(uv2) - lumaLocalAverage;

    bool reached1 = abs(lumaEnd1) >= gradientScaled;
    bool reached2 = abs(lumaEnd2) >= gradientScaled;

    if (!reached1) uv1 -= offset;
    if (!reached2) uv2 += offset;

    for (int i = 2; i < FXAA_SEARCH_STEPS; i++) {
        if (!reached1) {
            lumaEnd1 = sampleLumaFast(uv1) - lumaLocalAverage;
            reached1 = abs(lumaEnd1) >= gradientScaled;
        }
        if (!reached2) {
            lumaEnd2 = sampleLumaFast(uv2) - lumaLocalAverage;
            reached2 = abs(lumaEnd2) >= gradientScaled;
        }
        if (reached1 && reached2) break;
        if (!reached1) uv1 -= offset;
        if (!reached2) uv2 += offset;
    }

    float distance1 = isHorizontal ? (uv.x - uv1.x) : (uv.y - uv1.y);
    float distance2 = isHorizontal ? (uv2.x - uv.x) : (uv2.y - uv.y);

    bool isDirection1 = distance1 < distance2;
    float distanceFinal = min(distance1, distance2);
    float edgeThickness = distance1 + distance2;

    float lumaNearEnd = isDirection1 ? lumaEnd1 : lumaEnd2;
    bool isOpposite = (lumaNearEnd < 0.0) != ((lumaCenter - lumaLocalAverage) < 0.0);

    float pixelOffset = -distanceFinal / edgeThickness + 0.5;
    float finalEdgeOffset = isOpposite ? pixelOffset : 0.0;

    float lumaAverageCorners = lumaLeftCorners + lumaRightCorners;
    float subpixLuma = (2.0 * (lumaDownUp + lumaLeftRight) + lumaAverageCorners) * (1.0 / 12.0);
    float subpixRange = abs(subpixLuma - lumaCenter);
    float subpixFactor = clamp(subpixRange / lumaRange, 0.0, 1.0);
    float subpixBlend = (-2.0 * subpixFactor + 3.0) * subpixFactor * subpixFactor;
    float subpixBlendFinal = subpixBlend * subpixBlend * FXAA_SUBPIX_CAP;

    float finalOffset = max(finalEdgeOffset, subpixBlendFinal);

    vec2 finalUv = uv;
    if (isHorizontal) {
        finalUv.y += finalOffset * stepLength;
    } else {
        finalUv.x += finalOffset * stepLength;
    }

    return sampleRadiance(finalUv);
}

// DoF in linear radiance: reuses the existing CoC + golden-spiral gather
// offsets, but gathers raw linear radiance (scene + shaft, no per-tap
// SSR/fog cost) around a linear center, BEFORE bloom/glow adds + tonemap.
vec3 applyDoF(vec3 color, vec2 uv) {
    if (dof_params.x < 0.5) return color;
    float raw = texture(sampler2D(depth_tex, depth_smp), uv).r;
    float lin = (raw >= 0.9999) ? camera_params.y : dofLinearize(raw);
    float fr = max(dof_params.z, 0.0001);
    float coc = clamp(abs(lin - dof_params.y) / fr, 0.0, 1.0) * max(dof_params.w, 0.0);
    if (coc < 0.5) return color;

    const int DOF_TAPS = 14;
    // Precalculated 2D rotator for golden angle (2.3999632 rad):
    // cos(2.3999632) ~= -0.7373688, sin(2.3999632) ~= 0.6754904
    const vec2 rot_step = vec2(-0.73736882, 0.67549038);
    vec2 rot = vec2(1.0, 0.0);

    vec2 texel = resolution.zw;
    vec3 acc = color;
    float wsum = 1.0;
    for (int i = 0; i < DOF_TAPS; i++) {
        float fi = float(i);
        float rr = (fi + 0.5) / float(DOF_TAPS) * coc;
        vec2 off = rot * (rr * texel);
        vec2 tap_uv = uv + off;
        acc += boundRadiance(addShaftRadiance(texture(sampler2D(scene_tex, smp), tap_uv).rgb, tap_uv));
        wsum += 1.0;
        rot = vec2(rot.x * rot_step.x - rot.y * rot_step.y, rot.x * rot_step.y + rot.y * rot_step.x);
    }
    return boundRadiance(acc / wsum);
}

// Pre-tonemap radiance: FXAA when fxaa_params.x is on (flag honored,
// never silently off), then TAA. Chromatic/SSR/fog reuse their existing
// math in this linear space via sampleScene.
vec3 resolveRadiance(vec2 uv) {
    vec3 hdr;
    if (fxaa_params.x > 0.5) {
        hdr = applyFXAA(uv, resolution.zw);
    } else {
        hdr = sampleRadiance(uv);
    }
    hdr = applyTAA(hdr, uv);
    return boundRadiance(hdr);
}

// Bloom add pre-tonemap: the single bloom-pyramid composite (bloom_tex),
// sampled iff bloom is available (params3.z, parent-packed) and intensity
// is above zero. Always added before exposure/tonemap.
vec3 bloomAdd(vec2 uv) {
    if (params3.z > 0.5 && params1.z > 0.001) {
        return texture(sampler2D(bloom_tex, smp), uv).rgb * params1.z;
    }
    return vec3(0.0);
}

void main() {
    vec2 uv = v_uv;

    // SSAO debug: diagnostic gray, no exposure/tonemap/encode. A
    // capture-only re-entry stores this gray (history then reprojects gray;
    // acceptable for a diagnostic view).
    if (ssao_params.y > 0.5) {
        float ao_dbg = texture(sampler2D(ssao_tex, smp), uv).r;
        frag_color = vec4(ao_dbg, ao_dbg, ao_dbg, 1.0);
        return;
    }

    // Linear-radiance chain: FXAA-or-resolve + TAA in radiance, DoF in
    // linear BEFORE bloom/glow adds, exactly one exposure + one tonemap +
    // one display encode.
    vec3 color = resolveRadiance(uv);
    // History capture stores pre-exposure, pre-tonemap radiance (finite-half
    // bound only, no 0..1 clamp, no encode): the next frame's applyTAA reads
    // radiance, never display-encoded color.
    if (taa_state.y > 0.5) {
        frag_color = vec4(boundRadiance(color), 1.0);
        return;
    }
    color = applyDoF(color, uv);
    color = boundRadiance(color + bloomAdd(uv));
    // Glow layer v1 (global halo, independent of bloom): threshold-extracted
    // + separable-blurred glow_tex added with intensity * tint. Disabled (or
    // intensity ~0) returns before sampling. Composites AFTER bloom so
    // either toggle leaves the other's contribution unchanged; before
    // exposure/tonemap so the halo grades with the scene.
    if (glow_params.x > 0.5 && glow_params.y > 0.001) {
        color += texture(sampler2D(glow_tex, smp), uv).rgb * (glow_params.y * glow_tint.xyz);
    }
    // Highlight layer v1 (per-mesh inner glow): the raw per-item mask
    // (color x intensity folded at draw, frame-max sigma blur) minus its
    // blurred halo, floored at zero per channel and doubled, added with
    // the baked global scale. Interior pixels have mask ~= blurred, so
    // they contribute ~0; the silhouette edge (blur ~= half coverage)
    // restores the full mask color; outside the mesh the raw mask is 0
    // and the blurred spill clamps to 0 — inner-only glow, no
    // out-of-mesh halo. The difference form (rather than
    // mask * (1 - blurred)) is exact under the folded intensity: on a
    // binary coverage mask both coincide, but only max(mask - blurred, 0)
    // reaches 0 in the interior for any intensity. Mirrors
    // highlightInnerGlow/highlightComposite in postprocess.zig (same
    // per-channel math, same x2 gain). Disabled returns before sampling
    // EITHER texture. Composites AFTER glow so either toggle leaves the
    // other's contribution unchanged; before exposure/tonemap so per-mesh
    // colors grade with the scene.
    if (highlight_params.x > 0.5) {
        vec3 hl_raw = texture(sampler2D(highlight_mask_tex, smp), uv).rgb;
        vec3 hl_blurred = texture(sampler2D(highlight_tex, smp), uv).rgb;
        vec3 hl_inner = max(hl_raw - hl_blurred, vec3(0.0)) * 2.0;
        color += hl_inner * highlight_params.y;
    }
    color = boundRadiance(color);
    color *= params1.x; // Exposure (once)
    color = boundRadiance(color); // Bound AFTER exposure too (matches CPU tonemap)

    // Local Tonemapping & Contrast Adaptation (compresses dynamic range while preserving local details)
    color = applyLocalTonemapping(color, uv);
    color = boundRadiance(color);

    float tonemap_mode = params3.x;
    if (tonemap_mode > 4.5) {
        color = PBRNeutralTonemap(color);
    } else if (tonemap_mode > 3.5) {
        color = AgXTonemap(color);
    } else if (tonemap_mode > 2.5) {
        color = UchimuraTonemap(color);
    } else if (tonemap_mode > 1.5) {
        color = Reinhard(color);
    } else if (tonemap_mode > 0.5) {
        color = ACESFilm(color);
    }
    // Display-referred tail (post-tonemap; DoF already ran in linear above).
    // applyColorCurves/applyLut are the shared fns.
    // White Balance (post-tonemap channel gains, 0 = neutral)
    float wb_temp = params5.z;
    float wb_tint = params5.w;
    if (abs(wb_temp) > 0.0001 || abs(wb_tint) > 0.0001) {
        // Positive temperature warms (boosts red, cuts blue), negative cools.
        // Positive tint pushes magenta (cuts green), negative pushes green.
        vec3 wb_gains = vec3(1.0 + wb_temp * 0.20, 1.0 - wb_tint * 0.12, 1.0 - wb_temp * 0.20);
        color = clamp(color * wb_gains, 0.0, 1.0);
    }
    // Sharpen (unsharp mask, 5-tap cross kernel over tonemapped taps)
    float sharpen_amt = params5.x;
    if (sharpen_amt > 0.0001) {
        // Neighbor taps reuse the fast tonemapped proxy (skips SSR/fog
        // re-evaluation, same approximation FXAA itself uses for edge taps).
        vec2 texel = resolution.zw;
        vec3 tap_up = sampleSceneFastLDR(uv + vec2(0.0, texel.y));
        vec3 tap_down = sampleSceneFastLDR(uv - vec2(0.0, texel.y));
        vec3 tap_left = sampleSceneFastLDR(uv - vec2(texel.x, 0.0));
        vec3 tap_right = sampleSceneFastLDR(uv + vec2(texel.x, 0.0));
        vec3 blur = (tap_up + tap_down + tap_left + tap_right) * 0.25;
        // Clamp to the center+taps neighborhood so low amounts cannot ring.
        vec3 n_min = min(min(tap_up, tap_down), min(min(tap_left, tap_right), color));
        vec3 n_max = max(max(tap_up, tap_down), max(max(tap_left, tap_right), color));
        color = clamp(color + (color - blur) * sharpen_amt, n_min, n_max);
    }
    // Contrast
    float contrast = params2.w;
    if (abs(contrast - 1.0) > 0.001) {
        color = (color - vec3(0.5)) * contrast + vec3(0.5);
    }
    // Saturation
    float saturation = params2.z;
    if (abs(saturation - 1.0) > 0.001) {
        float luma = dot(color, vec3(0.2126, 0.7152, 0.0722));
        color = mix(vec3(luma), color, saturation);
    }
    // Parametric color curves (shadows/midtones/highlights lifts)
    color = applyColorCurves(color);
    // Texture LUT grade, sampled after the curves so the LUT authors the
    // final look on top of the parametric grade.
    color = applyLut(color);
    // Vignette
    if (params3.w > 0.5 && params2.x > 0.001) {
        vec2 v_coord = uv * (vec2(1.0) - uv.yx);
        float vig = v_coord.x * v_coord.y * 15.0;
        vig = clamp(pow(vig, params2.y * 0.5), 0.0, 1.0);
        color = mix(color * vig, color, 1.0 - params2.x);
    }
    // Film Grain (static screen-space hash, last so FXAA never sees the noise
    // and sharpen never amplifies it; applied after vignette so grain stays uniform)
    float grain_amt = params5.y;
    if (grain_amt > 0.00001) {
        vec2 grain_pixel = floor(v_uv * resolution.xy);
        float grain_n = grainHash(grain_pixel) - 0.5;
        // Luminance mask: full strength in shadows/midtones, tapered in highlights.
        float grain_luma = rgbToLuma(color);
        float lum_mask = clamp(1.0 - grain_luma * 1.2, 0.15, 1.0);
        color += grain_n * grain_amt * 2.0 * lum_mask;
    }
    color = clamp(color, 0.0, 1.0);
    // Exactly one display transfer: manual exact piecewise encode for UNORM,
    // linear for sRGB targets (hardware encodes).
    if (output_params.x > 0.5) {
        color = linearToSrgb(color);
    }
    frag_color = vec4(color, 1.0);
}
@end

@program postprocess vs fs
