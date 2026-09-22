// Fullscreen Post-Processing Shader for agate
// Supports ACES Filmic & Reinhard Tone Mapping, Bloom, Vignette, Chromatic Aberration, Saturation & Contrast,
// White Balance, Sharpen, Film Grain
@header const m = @import("math")
@ctype mat4 m.Mat4

@vs vs
@glsl_options fixup_clipspace
in vec2 position;
in vec2 texcoord0;

out vec2 v_uv;

void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    #if !SOKOL_GLSL
        v_uv = vec2(texcoord0.x, 1.0 - texcoord0.y);
    #else
        v_uv = texcoord0;
    #endif
}
@end

@fs fs
layout(binding = 0) uniform fs_params {
    vec4 params1; // x: exposure, y: bloom_threshold, z: bloom_intensity, w: bloom_radius
    vec4 params2; // x: vignette_intensity, y: vignette_radius, z: saturation, w: contrast
    vec4 params3; // x: tonemapping (0=none, 1=ACES, 2=Reinhard), y: chromatic_aberration, z: bloom_enabled (1/0), w: vignette_enabled (1/0)
    vec4 params4; // x: ssao_enabled (1/0), y: ssao_debug (1/0), z: ssao_intensity, w: fxaa_enabled (1/0)
    vec4 resolution; // xy: resolution, zw: texel size (1.0/width, 1.0/height)
    vec4 camera_params; // x: near_z, y: far_z, z/w: unused
    vec4 camera_pos; // xyz: camera world pos, w: unused
    vec4 sun_dir; // xyz: sun direction (normalized), w: unused
    vec4 sun_color; // xyz: sun color, w: unused
    vec4 fog_params; // x: fog_enabled (1/0), y: fog_density, z: fog_height_falloff, w: fog_start_distance
    vec4 fog_color; // xyz: fog_color, w: fog_sun_scattering
    vec4 ssr_params; // x: ssr_enabled (1/0), y: ssr_intensity, z: ssr_thickness, w: ssr_max_distance
    vec4 params5; // x: sharpen_amount (0=off), y: grain_intensity (0=off), z: temperature [-1,1], w: tint [-1,1]
    vec4 dof_params; // x: dof_enabled (1/0), y: focus_distance, z: focus_range, w: max_blur_px
    vec4 bloom_pyramid; // x: pyramid_enabled (1/0), y: mips, z/w: unused
    vec4 glow_params; // x: glow_enabled (1/0), y: intensity, z/w: unused
    vec4 glow_tint; // xyz: glow color multiplier, w: unused
    vec4 highlight_params; // x: highlight_enabled (1/0), y: baked global scale (always 1.0: per-item intensity folds into the mask), z/w: unused
    vec4 grade_shadows; // xyz: shadows lift [-1,1], w: unused
    vec4 grade_midtones; // xyz: midtones lift [-1,1], w: unused
    vec4 grade_highlights; // xyz: highlights lift [-1,1], w: unused
    vec4 lut_params; // x: lut_enabled (1/0), y: lut_strength [0,1], z: lut size N, w: unused
    mat4 view_proj; // camera view-projection matrix
    mat4 inv_view_proj; // inverse view-projection matrix
    mat4 prev_view_proj; // previous frame view-projection matrix
    vec4 motion_blur_params; // x: motion_blur_enabled (1/0), y: intensity, z: max_blur_px, w: unused
    vec4 taa_params; // x: taa_enabled (1/0), y: history blend [0,1], z: clamp strength [0,1], w: sharpen amount [0,1]
    vec4 taa_state; // x: history_valid (1/0), y: capture_only (1/0), zw: unused
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

layout(binding = 0) uniform sampler smp;
@sampler_type depth_smp nonfiltering
layout(binding = 1) uniform sampler depth_smp;
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

vec3 extractBright(vec3 c, float thresh) {
    float luma = dot(c, vec3(0.2126, 0.7152, 0.0722));
    float factor = max(0.0, luma - thresh);
    return c * (factor / max(luma, 0.0001));
}

vec3 reconstructWorldPos(vec2 uv, float depth) {
    vec4 clip = vec4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    vec4 world = inv_view_proj * clip;
    return world.xyz / world.w;
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

vec3 applySSR(vec3 scene_color, vec2 uv, float raw_depth) {
    if (ssr_params.x < 0.5) return scene_color;
    if (raw_depth >= 0.9999) return scene_color;

    vec3 world_pos = reconstructWorldPos(uv, raw_depth);
    vec3 V = normalize(world_pos - camera_pos.xyz);

    // Reconstruct world normal from depth buffer
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
    vec3 N = normalize(cross(dx, dy));

    // Only reflect on upward horizontal surfaces (floor/ground)
    if (N.y < 0.45) return scene_color;

    vec3 R = reflect(V, N);
    if (R.y < 0.02) return scene_color;

    float max_dist = ssr_params.w;
    float thickness = ssr_params.z;
    const int SSR_STEPS = 16;
    float step_size = max_dist / float(SSR_STEPS);

    vec3 ray_pos = world_pos + N * 0.08;

    for (int i = 0; i < SSR_STEPS; i++) {
        ray_pos += R * step_size;

        vec3 march_proj = projectWorldToUvDepth(ray_pos);
        vec2 march_uv = march_proj.xy;
        if (march_uv.x < 0.01 || march_uv.x > 0.99 || march_uv.y < 0.01 || march_uv.y > 0.99) {
            break;
        }

        float scene_d = texture(sampler2D(depth_tex, depth_smp), march_uv).r;
        if (scene_d >= 0.9999) continue;
        if (march_proj.z < scene_d) continue;

        vec3 scene_pos = reconstructWorldPos(march_uv, scene_d);

        float ray_cam_dist = length(ray_pos - camera_pos.xyz);
        float scene_cam_dist = length(scene_pos - camera_pos.xyz);
        float depth_diff = ray_cam_dist - scene_cam_dist;
        float dist_to_surface = length(ray_pos - scene_pos);

        if (depth_diff >= 0.0 && dist_to_surface < thickness) {
            float edge_dist_x = min(march_uv.x, 1.0 - march_uv.x);
            float edge_dist_y = min(march_uv.y, 1.0 - march_uv.y);
            float edge_fade = clamp(min(edge_dist_x, edge_dist_y) * 10.0, 0.0, 1.0);

            float dist_fade = 1.0 - (float(i) / float(SSR_STEPS));
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

    // Distance attenuation
    float fog_start = fog_params.w;
    float eff_dist = max(0.0, dist - fog_start);
    float dist_factor = 1.0 - exp(-eff_dist * fog_params.y);

    // Exponential Height Falloff
    float falloff = fog_params.z;
    float delta_y = world_pos.y - camera_pos.y;
    float height_density = (abs(delta_y) > 0.001)
        ? (exp(-camera_pos.y * falloff) - exp(-world_pos.y * falloff)) / (delta_y * falloff)
        : exp(-camera_pos.y * falloff);
    height_density = clamp(height_density, 0.0, 5.0);

    float fog_amount = clamp(dist_factor * height_density, 0.0, 1.0);
    return mix(scene_color, current_fog_color, fog_amount);
}

// Camera Motion Blur: gathers samples along screen velocity derived from depth reprojection
vec3 applyMotionBlur(vec3 color, vec2 uv, float depth) {
    if (motion_blur_params.x < 0.5) return color;
    if (depth >= 1.0) return color;

    vec3 world_pos = reconstructWorldPos(uv, depth);
    vec4 prev_clip = prev_view_proj * vec4(world_pos, 1.0);
    if (prev_clip.w <= 0.0001) return color;
    vec2 prev_ndc = prev_clip.xy / prev_clip.w;
    vec2 prev_uv = vec2(prev_ndc.x * 0.5 + 0.5, 0.5 - prev_ndc.y * 0.5);

    vec2 velocity = (uv - prev_uv) * motion_blur_params.y;
    float max_blur = motion_blur_params.z * resolution.z;
    float speed = length(velocity);
    if (speed > max_blur) {
        velocity = velocity * (max_blur / speed);
    }
    if (speed < 0.0002) return color;

    vec3 acc = color;
    for (int i = 1; i < 8; ++i) {
        float t = float(i) / 7.0 - 0.5;
        vec2 sample_uv = clamp(uv + velocity * t, vec2(0.001), vec2(0.999));
        acc += texture(sampler2D(scene_tex, smp), sample_uv).rgb;
    }
    return acc * 0.125;
}

// Sample scene HDR color, apply chromatic aberration, SSAO, SSR, Fog, and Motion Blur
vec3 sampleSceneRaw(vec2 uv) {
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

    // SSAO Occlusion
    if (params4.x > 0.5) {
        float ao = clamp(texture(sampler2D(ssao_tex, smp), uv).r, 0.0, 1.0);
        float ao_factor = clamp(1.0 - (1.0 - ao) * params4.z, 0.0, 1.0);
        color *= ao_factor;
    }

    // Depth-dependent passes: SSR, Atmospheric Fog, and Motion Blur
    float raw_depth = texture(sampler2D(depth_tex, depth_smp), uv).r;

    // Screen-Space Reflections (SSR)
    color = applySSR(color, uv, raw_depth);

    // Atmospheric Depth & Height Fog
    color = applyAtmosphericFog(color, uv, raw_depth);

    // Camera Motion Blur
    color = applyMotionBlur(color, uv, raw_depth);

    return color;
}

// Sample tonemapped LDR color for perceptual FXAA edge detection
vec3 sampleSceneLDR(vec2 uv) {
    vec3 color = sampleSceneRaw(uv);
    color *= params1.x; // Exposure

    float tonemap_mode = params3.x;
    if (tonemap_mode > 1.5) {
        color = Reinhard(color);
    } else if (tonemap_mode > 0.5) {
        color = ACESFilm(color);
    }
    return clamp(color, 0.0, 1.0);
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

// Fast LDR sample for FXAA edge detection and tangent searching
// Includes scene_tex, chromatic aberration, and SSAO (from pre-rendered ssao_tex)
// Skips heavy raymarching SSR and atmospheric fog on neighbor taps
vec3 sampleSceneFastLDR(vec2 uv) {
    vec3 color;
    float ca = params3.y;
    if (ca > 0.00001) {
        vec2 dist_from_center = uv - 0.5;
        vec2 ca_offset = dist_from_center * ca;
        float r = texture(sampler2D(scene_tex, smp), uv + ca_offset).r;
        float g = texture(sampler2D(scene_tex, smp), uv).g;
        float b = texture(sampler2D(scene_tex, smp), uv - ca_offset).b;
        color = vec3(r, g, b);
    } else {
        color = texture(sampler2D(scene_tex, smp), uv).rgb;
    }

    // SSAO Occlusion (fast texture fetch from pre-rendered pass)
    if (params4.x > 0.5) {
        float ao = clamp(texture(sampler2D(ssao_tex, smp), uv).r, 0.0, 1.0);
        float ao_factor = clamp(1.0 - (1.0 - ao) * params4.z, 0.0, 1.0);
        color *= ao_factor;
    }

    color *= params1.x; // Exposure
    float tonemap_mode = params3.x;
    if (tonemap_mode > 1.5) {
        color = Reinhard(color);
    } else if (tonemap_mode > 0.5) {
        color = ACESFilm(color);
    }
    return clamp(color, 0.0, 1.0);
}

// Fast luma approximation for FXAA edge detection and tangent walking:
// skips re-running tonemapping polynomials on every neighbor tap.
float sampleLumaFast(vec2 uv) {
    vec3 c = texture(sampler2D(scene_tex, smp), uv).rgb;
    float luma_hdr = dot(c, vec3(0.299, 0.587, 0.114)) * params1.x;
    return luma_hdr / (luma_hdr + 1.0);
}

// Temporal Anti-Aliasing neighborhood: 3x3 box (min/max/average) over the
// tonemapped-LDR center + 8 fast-LDR taps (the same neighbor approximation
// FXAA and sharpen use). Mirrors taaNeighborhoodBounds/taaNeighborhoodAvg
// in postprocess.zig.
void taaNeighborhood(vec2 uv, vec3 center, out vec3 box_min, out vec3 box_max, out vec3 avg) {
    vec2 texel = resolution.zw;
    box_min = center;
    box_max = center;
    vec3 sum = center;
    vec3 t;
    t = sampleSceneFastLDR(uv + vec2(-texel.x, -texel.y)); box_min = min(box_min, t); box_max = max(box_max, t); sum += t;
    t = sampleSceneFastLDR(uv + vec2(0.0, -texel.y)); box_min = min(box_min, t); box_max = max(box_max, t); sum += t;
    t = sampleSceneFastLDR(uv + vec2(texel.x, -texel.y)); box_min = min(box_min, t); box_max = max(box_max, t); sum += t;
    t = sampleSceneFastLDR(uv + vec2(-texel.x, 0.0)); box_min = min(box_min, t); box_max = max(box_max, t); sum += t;
    t = sampleSceneFastLDR(uv + vec2(texel.x, 0.0)); box_min = min(box_min, t); box_max = max(box_max, t); sum += t;
    t = sampleSceneFastLDR(uv + vec2(-texel.x, texel.y)); box_min = min(box_min, t); box_max = max(box_max, t); sum += t;
    t = sampleSceneFastLDR(uv + vec2(0.0, texel.y)); box_min = min(box_min, t); box_max = max(box_max, t); sum += t;
    t = sampleSceneFastLDR(uv + vec2(texel.x, texel.y)); box_min = min(box_min, t); box_max = max(box_max, t); sum += t;
    avg = sum / 9.0;
}

// Temporal Anti-Aliasing resolve on the tonemapped LDR image. Velocity comes
// from depth reprojection with the (jittered) current/prev view-projection —
// exactly the applyMotionBlur math — and the history sample is bilinear
// (smp). Mirrors taaResolvePixel in postprocess.zig (bounds + clamp + blend
// + sharpen). Disabled (or no valid history yet) returns `current` before
// any history/depth sampling, so the off path stays bit-identical.
vec3 applyTAA(vec3 current, vec2 uv) {
    if (taa_params.x < 0.5) return current;
    if (taa_state.x < 0.5) return current;
    float raw_depth = texture(sampler2D(depth_tex, depth_smp), uv).r;
    if (raw_depth >= 0.9999) return current;
    vec3 world_pos = reconstructWorldPos(uv, raw_depth);
    vec4 prev_clip = prev_view_proj * vec4(world_pos, 1.0);
    if (prev_clip.w <= 0.0001) return current;
    vec2 prev_ndc = prev_clip.xy / prev_clip.w;
    vec2 prev_uv = vec2(prev_ndc.x * 0.5 + 0.5, 0.5 - prev_ndc.y * 0.5);
    if (prev_uv.x < 0.001 || prev_uv.x > 0.999 || prev_uv.y < 0.001 || prev_uv.y > 0.999) return current;
    vec3 hist = texture(sampler2D(history_tex, smp), prev_uv).rgb;
    vec3 box_min;
    vec3 box_max;
    vec3 avg;
    taaNeighborhood(uv, current, box_min, box_max, avg);
    float clamp_strength = clamp(taa_params.z, 0.0, 1.0);
    vec3 hist_clamped = mix(hist, clamp(hist, box_min, box_max), clamp_strength);
    float blend = clamp(taa_params.y, 0.0, 1.0);
    vec3 outc = mix(current, hist_clamped, blend);
    float sharp = clamp(taa_params.w, 0.0, 1.0);
    if (sharp > 0.0001) {
        outc = clamp(outc + (current - avg) * sharp, box_min, box_max);
    }
    return outc;
}

// FXAA 3.11 Quality Anti-Aliasing
#define FXAA_EDGE_THRESHOLD_MIN 0.0312
#define FXAA_EDGE_THRESHOLD     0.125
#define FXAA_SUBPIX_CAP         0.75
#define FXAA_SEARCH_STEPS       10

vec3 applyFXAA(vec2 uv, vec2 rcpFrame) {
    vec3 colorCenter = sampleSceneLDR(uv);
    float lumaCenter = rgbToLuma(colorCenter);

    // 4 cross neighbors (fast scalar luminance)
    float lumaDown  = sampleLumaFast(uv + vec2(0.0, -rcpFrame.y));
    float lumaUp    = sampleLumaFast(uv + vec2(0.0,  rcpFrame.y));
    float lumaLeft  = sampleLumaFast(uv + vec2(-rcpFrame.x, 0.0));
    float lumaRight = sampleLumaFast(uv + vec2( rcpFrame.x, 0.0));

    float lumaMin = min(lumaCenter, min(min(lumaDown, lumaUp), min(lumaLeft, lumaRight)));
    float lumaMax = max(lumaCenter, max(max(lumaDown, lumaUp), max(lumaLeft, lumaRight)));
    float lumaRange = lumaMax - lumaMin;

    // Early exit if contrast is below threshold
    if (lumaRange < max(FXAA_EDGE_THRESHOLD_MIN, lumaMax * FXAA_EDGE_THRESHOLD)) {
        return colorCenter;
    }

    // 4 corner neighbors
    float lumaDownLeft  = sampleLumaFast(uv + vec2(-rcpFrame.x, -rcpFrame.y));
    float lumaUpRight   = sampleLumaFast(uv + vec2( rcpFrame.x,  rcpFrame.y));
    float lumaUpLeft    = sampleLumaFast(uv + vec2(-rcpFrame.x,  rcpFrame.y));
    float lumaDownRight = sampleLumaFast(uv + vec2( rcpFrame.x, -rcpFrame.y));

    // Edge orientation detection (horizontal vs vertical)
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

    // Select gradient perpendicular to edge
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

    // Search along edge tangent
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

    // Distance to edge ends
    float distance1 = isHorizontal ? (uv.x - uv1.x) : (uv.y - uv1.y);
    float distance2 = isHorizontal ? (uv2.x - uv.x) : (uv2.y - uv.y);

    bool isDirection1 = distance1 < distance2;
    float distanceFinal = min(distance1, distance2);
    float edgeThickness = distance1 + distance2;

    float lumaNearEnd = isDirection1 ? lumaEnd1 : lumaEnd2;
    bool isOpposite = (lumaNearEnd < 0.0) != ((lumaCenter - lumaLocalAverage) < 0.0);

    float pixelOffset = -distanceFinal / edgeThickness + 0.5;
    float finalEdgeOffset = isOpposite ? pixelOffset : 0.0;

    // Subpixel antialiasing
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

    return sampleSceneLDR(finalUv);
}

float dofLinearize(float d) {
    float near = camera_params.x;
    float far = camera_params.y;
    return (near * far) / max(far - d * (far - near), 0.0001);
}

// Depth of Field: gather blur with a golden-angle spiral over the tonemapped
// color. Pixels inside the focal plane return early with zero extra taps.
// Sky (cleared depth) is treated as far plane distance.
vec3 applyDoF(vec3 color, vec2 uv) {
    if (dof_params.x < 0.5) return color;
    float raw = texture(sampler2D(depth_tex, depth_smp), uv).r;
    float lin = (raw >= 0.9999) ? camera_params.y : dofLinearize(raw);
    float fr = max(dof_params.z, 0.0001);
    float coc = clamp(abs(lin - dof_params.y) / fr, 0.0, 1.0) * max(dof_params.w, 0.0);
    // In focus: no extra samples.
    if (coc < 0.5) return color;

    const int DOF_TAPS = 14;
    const float GOLDEN_ANGLE = 2.3999632;
    vec2 texel = resolution.zw;
    vec3 acc = color;
    float wsum = 1.0;
    for (int i = 0; i < DOF_TAPS; i++) {
        float fi = float(i);
        float ang = fi * GOLDEN_ANGLE;
        float rr = (fi + 0.5) / float(DOF_TAPS) * coc;
        vec2 off = vec2(cos(ang), sin(ang)) * rr * texel;
        // Fast LDR path reuses the FXAA neighbor approximation (skips
        // SSR/fog re-evaluation on taps).
        acc += sampleSceneFastLDR(uv + off);
        wsum += 1.0;
    }
    return acc / wsum;
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

void main() {
    vec2 uv = v_uv;

    // SSAO Debug view early exit
    if (params4.y > 0.5) {
        float ao_dbg = texture(sampler2D(ssao_tex, smp), uv).r;
        frag_color = vec4(ao_dbg, ao_dbg, ao_dbg, 1.0);
        return;
    }

    // Anti-Aliasing (FXAA 3.11) or Direct Tonemapped Sample
    vec3 color;
    if (params4.w > 0.5) {
        color = applyFXAA(uv, resolution.zw);
    } else {
        color = sampleSceneLDR(uv);
    }

    // Temporal Anti-Aliasing (tonemapped-LDR resolve against the reprojected
    // history; same depth + current/prev VP velocity math as motion blur).
    // Disabled (or no valid history yet) returns `color` before any
    // history/depth sampling, so the off path is bit-identical to pre-TAA.
    color = applyTAA(color, uv);
    // History capture draws re-enter with taa_state.y = 1 and store exactly
    // this post-TAA color for the next frame (PostFXStack.renderChain).
    if (taa_state.y > 0.5) {
        frag_color = vec4(clamp(color, 0.0, 1.0), 1.0);
        return;
    }

    // Depth of Field (gather blur on the tonemapped image)
    color = applyDoF(color, uv);

    // White Balance (post-tonemap channel gains, 0 = neutral)
    float wb_temp = params5.z;
    float wb_tint = params5.w;
    if (abs(wb_temp) > 0.0001 || abs(wb_tint) > 0.0001) {
        // Positive temperature warms (boosts red, cuts blue), negative cools.
        // Positive tint pushes magenta (cuts green), negative pushes green.
        vec3 wb_gains = vec3(1.0 + wb_temp * 0.20, 1.0 - wb_tint * 0.12, 1.0 - wb_temp * 0.20);
        color = clamp(color * wb_gains, 0.0, 1.0);
    }

    // Sharpen (unsharp mask on tonemapped LDR, 5-tap cross kernel)
    float sharpen_amt = params5.x;
    if (sharpen_amt > 0.0001) {
        // Neighbor taps reuse the fast LDR path (skips SSR/fog re-evaluation,
        // same approximation FXAA itself uses for its neighbor taps).
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

    // Bloom glow pass: high-quality pyramid composite when BloomPass fed
    // bloom_tex, otherwise the legacy single-shader multi-tap fallback.
    if (bloom_pyramid.x > 0.5 && params1.z > 0.001) {
        vec3 glow = texture(sampler2D(bloom_tex, smp), uv).rgb;
        vec3 bloom_scaled = glow * params1.z;
        color += bloom_scaled;
    } else if (params3.z > 0.5 && params1.z > 0.001) {
        float thresh = params1.y;
        vec2 texel = resolution.zw * params1.w;
        vec3 bloom = vec3(0.0);

        bloom += extractBright(texture(sampler2D(scene_tex, smp), uv + vec2(-1.0, -1.0) * texel).rgb, thresh) * 0.0625;
        bloom += extractBright(texture(sampler2D(scene_tex, smp), uv + vec2( 0.0, -1.0) * texel).rgb, thresh) * 0.1250;
        bloom += extractBright(texture(sampler2D(scene_tex, smp), uv + vec2( 1.0, -1.0) * texel).rgb, thresh) * 0.0625;
        bloom += extractBright(texture(sampler2D(scene_tex, smp), uv + vec2(-1.0,  0.0) * texel).rgb, thresh) * 0.1250;
        bloom += extractBright(texture(sampler2D(scene_tex, smp), uv + vec2( 0.0,  0.0) * texel).rgb, thresh) * 0.2500;
        bloom += extractBright(texture(sampler2D(scene_tex, smp), uv + vec2( 1.0,  0.0) * texel).rgb, thresh) * 0.1250;
        bloom += extractBright(texture(sampler2D(scene_tex, smp), uv + vec2(-1.0,  1.0) * texel).rgb, thresh) * 0.0625;
        bloom += extractBright(texture(sampler2D(scene_tex, smp), uv + vec2( 0.0,  1.0) * texel).rgb, thresh) * 0.1250;
        bloom += extractBright(texture(sampler2D(scene_tex, smp), uv + vec2( 1.0,  1.0) * texel).rgb, thresh) * 0.0625;

        vec2 texel2 = texel * 2.5;
        bloom += extractBright(texture(sampler2D(scene_tex, smp), uv + vec2(-1.0,  0.0) * texel2).rgb, thresh) * 0.1000;
        bloom += extractBright(texture(sampler2D(scene_tex, smp), uv + vec2( 1.0,  0.0) * texel2).rgb, thresh) * 0.1000;
        bloom += extractBright(texture(sampler2D(scene_tex, smp), uv + vec2( 0.0, -1.0) * texel2).rgb, thresh) * 0.1000;
        bloom += extractBright(texture(sampler2D(scene_tex, smp), uv + vec2( 0.0,  1.0) * texel2).rgb, thresh) * 0.1000;

        vec3 bloom_scaled = bloom * params1.z;
        color += bloom_scaled;
    }

    // Glow layer v1 (global halo, independent of bloom): threshold-extracted
    // + separable-blurred glow_tex added with intensity * tint. Disabled (or
    // intensity ~0) returns before sampling, so the off path is bit-identical
    // to pre-glow. Composites AFTER bloom so either toggle leaves the other's
    // contribution unchanged; before the grading chain so the halo grades
    // with the same LDR the bloom halo uses.
    if (glow_params.x > 0.5 && glow_params.y > 0.001) {
        vec3 glow = texture(sampler2D(glow_tex, smp), uv).rgb;
        color += glow * (glow_params.y * glow_tint.xyz);
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
    // EITHER texture, so the off path is bit-identical to pre-highlight.
    // Composites AFTER glow so either toggle leaves the other's
    // contribution unchanged; before the grading chain so per-mesh colors
    // grade with the same LDR the bloom/glow halos use.
    if (highlight_params.x > 0.5) {
        vec3 hl_raw = texture(sampler2D(highlight_mask_tex, smp), uv).rgb;
        vec3 hl_blurred = texture(sampler2D(highlight_tex, smp), uv).rgb;
        vec3 hl_inner = max(hl_raw - hl_blurred, vec3(0.0)) * 2.0;
        color += hl_inner * highlight_params.y;
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

    frag_color = vec4(clamp(color, 0.0, 1.0), 1.0);
}
@end

@program postprocess vs fs
