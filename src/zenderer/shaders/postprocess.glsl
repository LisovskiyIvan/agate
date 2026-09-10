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
    vec4 params4; // x: ssao_enabled (1/0), y: ssao_debug (1/0), z: ssao_intensity, w: unused
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

void main() {
    vec2 uv = v_uv;

    // Chromatic Aberration
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

        color += bloom * params1.z;
    }

    // Exposure
    color *= params1.x;

    // SSAO Occlusion & Debug view
    if (params4.y > 0.5) {
        float ao_dbg = texture(sampler2D(ssao_tex, smp), uv).r;
        frag_color = vec4(ao_dbg, ao_dbg, ao_dbg, 1.0);
        return;
    }
    if (params4.x > 0.5) {
        float ao = clamp(texture(sampler2D(ssao_tex, smp), uv).r, 0.0, 1.0);
        float ao_factor = clamp(1.0 - (1.0 - ao) * params4.z, 0.0, 1.0);
        color *= ao_factor;
    }

    // Tone Mapping
    float tonemap_mode = params3.x;
    if (tonemap_mode > 1.5) {
        color = Reinhard(color);
    } else if (tonemap_mode > 0.5) {
        color = ACESFilm(color);
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
