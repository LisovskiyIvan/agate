// Half-resolution linear opaque capture, projected slab exit point.
// No screen-space raymarch, offscreen recovery or transparent recursion.
vec3 refractedBackground(vec3 N, vec3 V, float roughness) {
    float ior = refraction_factors.z;
    vec3 ray = refract(-V, N, 1.0 / ior);
    vec3 exitPoint = v_world_pos + ray * (refraction_factors.y / max(dot(-ray, N), 0.1));
    vec4 clip = refraction_view_proj * vec4(exitPoint, 1.0);
    if (clip.w <= 0.0001) return vec3(0.0);
    vec2 uv = clip.xy / clip.w * 0.5 + 0.5;
    #if !SOKOL_GLSL
        uv.y = 1.0 - uv.y;
    #endif
    vec2 texel = 1.0 / refraction_capture.yz;
    uv = clamp(uv, texel * 0.5, vec2(1.0) - texel * 0.5);
    vec2 radius = texel * (roughness * roughness * 4.0);
    vec3 background = texture(sampler2D(refraction_tex, refraction_smp), uv).rgb * 0.5;
    background += texture(sampler2D(refraction_tex, refraction_smp), uv + vec2(radius.x, 0.0)).rgb * 0.125;
    background += texture(sampler2D(refraction_tex, refraction_smp), uv - vec2(radius.x, 0.0)).rgb * 0.125;
    background += texture(sampler2D(refraction_tex, refraction_smp), uv + vec2(0.0, radius.y)).rgb * 0.125;
    background += texture(sampler2D(refraction_tex, refraction_smp), uv - vec2(0.0, radius.y)).rgb * 0.125;
    float f0 = (ior - 1.0) / (ior + 1.0);
    f0 *= f0;
    float fresnel = f0 + (1.0 - f0) * pow(1.0 - max(dot(N, V), 0.0), 5.0);
    return background * (1.0 - fresnel);
}
