// Fullscreen Post-Processing Shader for zenderer
// Supports ACES Filmic & Reinhard Tone Mapping, Bloom, Vignette, Chromatic Aberration, Saturation & Contrast
@header const m = @import("math")

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
};

layout(binding = 0) uniform texture2D scene_tex;
layout(binding = 1) uniform texture2D ssao_tex;
layout(binding = 0) uniform sampler smp;

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

// Sample scene HDR color, apply chromatic aberration and SSAO
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

// FXAA 3.11 Quality Anti-Aliasing
#define FXAA_EDGE_THRESHOLD_MIN 0.0312
#define FXAA_EDGE_THRESHOLD     0.125
#define FXAA_SUBPIX_CAP         0.75
#define FXAA_SEARCH_STEPS       10

vec3 applyFXAA(vec2 uv, vec2 rcpFrame) {
    vec3 colorCenter = sampleSceneLDR(uv);
    float lumaCenter = rgbToLuma(colorCenter);

    // 4 cross neighbors
    float lumaDown  = rgbToLuma(sampleSceneLDR(uv + vec2(0.0, -rcpFrame.y)));
    float lumaUp    = rgbToLuma(sampleSceneLDR(uv + vec2(0.0,  rcpFrame.y)));
    float lumaLeft  = rgbToLuma(sampleSceneLDR(uv + vec2(-rcpFrame.x, 0.0)));
    float lumaRight = rgbToLuma(sampleSceneLDR(uv + vec2( rcpFrame.x, 0.0)));

    float lumaMin = min(lumaCenter, min(min(lumaDown, lumaUp), min(lumaLeft, lumaRight)));
    float lumaMax = max(lumaCenter, max(max(lumaDown, lumaUp), max(lumaLeft, lumaRight)));
    float lumaRange = lumaMax - lumaMin;

    // Early exit if contrast is below threshold
    if (lumaRange < max(FXAA_EDGE_THRESHOLD_MIN, lumaMax * FXAA_EDGE_THRESHOLD)) {
        return colorCenter;
    }

    // 4 corner neighbors
    float lumaDownLeft  = rgbToLuma(sampleSceneLDR(uv + vec2(-rcpFrame.x, -rcpFrame.y)));
    float lumaUpRight   = rgbToLuma(sampleSceneLDR(uv + vec2( rcpFrame.x,  rcpFrame.y)));
    float lumaUpLeft    = rgbToLuma(sampleSceneLDR(uv + vec2(-rcpFrame.x,  rcpFrame.y)));
    float lumaDownRight = rgbToLuma(sampleSceneLDR(uv + vec2( rcpFrame.x, -rcpFrame.y)));

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

    float lumaEnd1 = rgbToLuma(sampleSceneLDR(uv1)) - lumaLocalAverage;
    float lumaEnd2 = rgbToLuma(sampleSceneLDR(uv2)) - lumaLocalAverage;

    bool reached1 = abs(lumaEnd1) >= gradientScaled;
    bool reached2 = abs(lumaEnd2) >= gradientScaled;

    if (!reached1) uv1 -= offset;
    if (!reached2) uv2 += offset;

    for (int i = 2; i < FXAA_SEARCH_STEPS; i++) {
        if (!reached1) {
            lumaEnd1 = rgbToLuma(sampleSceneLDR(uv1)) - lumaLocalAverage;
            reached1 = abs(lumaEnd1) >= gradientScaled;
        }
        if (!reached2) {
            lumaEnd2 = rgbToLuma(sampleSceneLDR(uv2)) - lumaLocalAverage;
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

    // Bloom glow pass (multi-tap bright pass blur)
    if (params3.z > 0.5 && params1.z > 0.001) {
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

        vec3 bloom_scaled = bloom * params1.z * params1.x;
        float tonemap_mode = params3.x;
        if (tonemap_mode > 1.5) {
            bloom_scaled = Reinhard(bloom_scaled);
        } else if (tonemap_mode > 0.5) {
            bloom_scaled = ACESFilm(bloom_scaled);
        }
        color += bloom_scaled;
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

    // Vignette
    if (params3.w > 0.5 && params2.x > 0.001) {
        vec2 v_coord = uv * (vec2(1.0) - uv.yx);
        float vig = v_coord.x * v_coord.y * 15.0;
        vig = clamp(pow(vig, params2.y * 0.5), 0.0, 1.0);
        color = mix(color * vig, color, 1.0 - params2.x);
    }

    frag_color = vec4(clamp(color, 0.0, 1.0), 1.0);
}
@end

@program postprocess vs fs
