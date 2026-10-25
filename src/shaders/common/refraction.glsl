// Screen-space slab refraction v2:
// - Projected slab exit point with 5-tap Gaussian blur for rough glass.
// - Offscreen recovery: seamless transition to local parallax-corrected IBL probes
//   (probe_tex/probe2_tex with boxProjectReflection) and global environment map (env_tex).
// - Smooth screen-edge vignette falloff (8% margin) eliminates edge clamping/stretching.
// - Total Internal Reflection (TIR) fallback.
// - Chromatic dispersion: wavelength-dependent IOR separation (red/green/blue rays)
//   controlled by refraction_factors.w (dispersion factor).

vec3 sampleRefractionEnv(vec3 dir, float roughness) {
    float max_lod = 7.0;
    float spec_max_lod = max_lod - 1.0;
    float lod = roughness * spec_max_lod;
    vec3 result = vec3(0.0);

    float w0 = (probe_params.x > 0.5) ? (probe_params.w > 0.0001 ? probe_params.w : 1.0) : 0.0;
    float w1 = (probe2_params.x > 0.5) ? probe2_params.w : 0.0;
    float w_env = max(0.0, 1.0 - w0 - w1);

    vec3 R0 = (w0 > 0.001) ? boxProjectReflection(dir, v_world_pos, probe_box[0].xyz, probe_box[0].w) : dir;
    vec3 R1 = (w1 > 0.001) ? boxProjectReflection(dir, v_world_pos, probe_box[1].xyz, probe_box[1].w) : dir;

    if (w0 > 0.001) {
        result += textureLod(samplerCube(probe_tex, probe_smp), R0, clamp(lod, 0.0, probe_params.z - 1.0)).rgb * probe_params.y * w0;
    }
    if (w1 > 0.001) {
        result += textureLod(samplerCube(probe2_tex, probe_smp), R1, clamp(lod, 0.0, probe2_params.z - 1.0)).rgb * probe2_params.y * w1;
    }
    if (w_env > 0.001) {
        result += textureLod(samplerCube(env_tex, env_smp), dir, lod).rgb * w_env;
    }
    return result;
}

vec3 sampleRefractionRay(vec3 N, vec3 V, float ior, float roughness) {
    vec3 ray = refract(-V, N, 1.0 / ior);
    bool tir = dot(ray, ray) < 1e-4;
    if (tir) {
        ray = reflect(-V, N);
    }

    vec3 exitPoint = v_world_pos + ray * (refraction_factors.y / max(dot(-ray, N), 0.1));
    vec4 clip = refraction_view_proj * vec4(exitPoint, 1.0);

    float screen_weight = 0.0;
    vec2 uv = vec2(0.0);
    if (!tir && clip.w > 1e-4) {
        vec2 screen_coord = clip.xy / clip.w * 0.5 + 0.5;
        #if !SOKOL_GLSL
            screen_coord.y = 1.0 - screen_coord.y;
        #endif
        vec2 texel = 1.0 / refraction_capture.yz;
        vec2 edge = smoothstep(vec2(0.0), vec2(0.08), screen_coord) * smoothstep(vec2(0.0), vec2(0.08), 1.0 - screen_coord);
        screen_weight = edge.x * edge.y;
        uv = clamp(screen_coord, texel * 0.5, vec2(1.0) - texel * 0.5);
    }

    vec3 screen_col = vec3(0.0);
    if (screen_weight > 0.001) {
        vec2 texel = 1.0 / refraction_capture.yz;
        vec2 radius = texel * (roughness * roughness * 4.0);
        screen_col = texture(sampler2D(refraction_tex, refraction_smp), uv).rgb * 0.5;
        screen_col += texture(sampler2D(refraction_tex, refraction_smp), uv + vec2(radius.x, 0.0)).rgb * 0.125;
        screen_col += texture(sampler2D(refraction_tex, refraction_smp), uv - vec2(radius.x, 0.0)).rgb * 0.125;
        screen_col += texture(sampler2D(refraction_tex, refraction_smp), uv + vec2(0.0, radius.y)).rgb * 0.125;
        screen_col += texture(sampler2D(refraction_tex, refraction_smp), uv - vec2(0.0, radius.y)).rgb * 0.125;
    }

    vec3 env_col = vec3(0.0);
    if (screen_weight < 0.999) {
        env_col = sampleRefractionEnv(ray, roughness);
    }

    return mix(env_col, screen_col, screen_weight);
}

vec3 refractedBackground(vec3 N, vec3 V, float roughness) {
    float ior = max(1.001, refraction_factors.z);
    float disp = refraction_factors.w;

    vec3 background;
    if (disp > 0.001) {
        float ior_r = max(1.001, ior * (1.0 - disp * 0.04));
        float ior_g = ior;
        float ior_b = ior * (1.0 + disp * 0.04);
        float r = sampleRefractionRay(N, V, ior_r, roughness).r;
        float g = sampleRefractionRay(N, V, ior_g, roughness).g;
        float b = sampleRefractionRay(N, V, ior_b, roughness).b;
        background = vec3(r, g, b);
    } else {
        background = sampleRefractionRay(N, V, ior, roughness);
    }

    float f0 = (ior - 1.0) / (ior + 1.0);
    f0 *= f0;
    float fresnel = f0 + (1.0 - f0) * pow(1.0 - max(dot(N, V), 0.0), 5.0);
    return background * (1.0 - fresnel);
}
