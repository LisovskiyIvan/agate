// Hemispheric light — Babylon.js `HemisphericLight` model (irradiance AND the
// specular lobe its PBR adds for `HEMILIGHT`).
//
// Babylon treats the hemispheric light as a real light, not a flat ambient
// term: its irradiance interpolates between `groundColor` (surface facing
// away from the light) and `diffuse * intensity` (surface facing the light)
// by `0.5 + 0.5 * dot(N, direction)`:
//
//     irradiance = mix(groundColor * intensity, diffuse * intensity,
//                      0.5 + 0.5 * N·L)
//
// (`HemisphericLight.transferToEffect` uploads
// `vLightGround = groundColor.scale(intensity)`, so BOTH ends of the mix are
// intensity-scaled — the ground end used to ride the unscaled
// `ambient_color.rgb` lane here.)
//
// Historically agate ignored `direction`, `diffuse` and `intensity` here and
// used `groundColor` as a constant ambient, which made every lit surface
// darker and tinted than the Babylon original (measured on the bench scene:
// the ground plane came out 0.39 linear vs Babylon's 0.60 with the same
// two lights).
//
// Uniform lanes (see `scene/uniforms.zig`, appended last so no existing
// offset shifts):
//   ambient_color.rgb  = groundColor (Babylon's vLightGround is this lane
//                        scaled by intensity — done in hemiIrradiance below,
//                        so the lane itself keeps carrying the raw colour)
//   hemi_dir_intensity = (direction.xyz, intensity), direction normalized
//   hemi_diffuse.rgb   = diffuse
vec3 hemiIrradiance(vec3 n) {
    float intensity = hemi_dir_intensity.w;
    float mix_factor = dot(n, hemi_dir_intensity.xyz) * 0.5 + 0.5;
    return mix(ambient_color.rgb * intensity, hemi_diffuse.rgb * intensity, mix_factor);
}

// The hemispheric SPECULAR lobe — Babylon's PBR gives a HEMILIGHT one and
// agate did not, which was the whole metal-path residual on the bench.
//
// Babylon's light loop is light-type agnostic inside `#ifdef SPECULARTERM`
// (`lightFragment`, shipped bundle 9.28.0, byte-identical in the WGSL and
// GLSL copies of the chunk):
//
//     #elif defined(HEMILIGHT{X})
//     preInfo=computeHemisphericPreLightingInfo(light{X}.vLightData,viewDirectionW,normalW);
//     ...
//     preInfo.attenuation=1.0;          // no distance falloff
//     #if defined(HEMILIGHT{X})
//     preInfo.roughness=roughness;      // no light-radius remap either
//     ...
//     info.specular=computeSpecularLighting(preInfo,N,specularEnvironmentR0,
//                     coloredFresnel,AARoughnessFactor,light.vLightDiffuse.rgb);
//
// with `computeHemisphericPreLightingInfo` giving
//     NdotL = saturateEps(dot(N, L) * 0.5 + 0.5)   // wrapped, NOT max(.,0)
//     L     = normalize(lightData.xyz)
//     H     = normalize(V + L), VdotH = saturate(dot(V, H))
// and `computeSpecularLighting` returning
//     F(VdotH) * D(NdotH, a) * Vis(NdotL, NdotV, a) * attenuation * NdotL * lightColor
// i.e. exactly the same Cook-Torrance term as the sun, for the hemispheric
// light. Three details matter:
//
//   * the light colour is `vLightDiffuse` (= diffuse * intensity), NOT
//     `vLightSpecular` — in 9.28.0 `vLightSpecular` is read only by the
//     AREALIGHT branches (`computeAreaSpecularLighting`), which is why
//     `hemi.specular = 0` in the bench changes no pixel at all;
//   * `NdotL` is the wrapped hemispheric value, so the lobe never vanishes on
//     the dark side — it decays to the 1e-7 `Epsilon` floor of `saturateEps`
//     instead of to 0;
//   * `F0`/`R90` are the material's `colorReflectanceF0` =
//     `mix(0.04, albedo, metallic)` and `colorReflectanceF90` =
//     `mix(specularWeight * f90Scale, 1, metallic)` (Babylon
//     `reflectivityBlock`). glTF's defaults (ior 1.5 -> f90Scale 1,
//     specularWeight 1) make R90 = 1, i.e. plain Schlick —
//     `fresnelGrazingReflectance` only applies under `ALPHAFRESNEL`. That is
//     what `fresnelSchlick` already computes here.
//
// Two consequences that the caller must keep:
//   * there is NO `(1 - metallic)` on this term (that factor belongs to the
//     irradiance path only): with F0 = mix(0.04, albedo, metallic) the lobe
//     is Fresnel-weighted by construction, so it is strong on metals and a
//     4% one on dielectrics — a dielectric is not disturbed by adding it;
//   * `intensity = 0` zeroes it exactly (the light colour is a plain product).
//
// Not multiplied by ambient occlusion: Babylon applies
// `aoOut.ambientOcclusionColor` to `finalDiffuse` and to the IBL
// irradiance, never to `finalSpecular`, and the analytic lobe here is part of
// that specular sum.
vec3 hemiSpecular(vec3 N, vec3 V, float NdotV, float roughness, vec3 F0) {
    // `ambient_color`/`hemi_dir_intensity` are frame lanes a caller can leave
    // zeroed (a state built without a scene): keep `normalize` finite the way
    // Babylon's `Vector3.Normalize` is (zero length -> zero vector).
    vec3 dir = hemi_dir_intensity.xyz;
    float dir_len2 = dot(dir, dir);
    vec3 L = dir_len2 > 0.0 ? dir * inversesqrt(dir_len2) : vec3(0.0, 1.0, 0.0);
    vec3 H = normalize(V + L);
    // Babylon clamps to [Epsilon = 1e-7, 1] (`saturateEps`), not to [0, 1].
    float NdotL = clamp(dot(N, L) * 0.5 + 0.5, 0.0000001, 1.0);
    float VdotH = clamp(dot(V, H), 0.0, 1.0);
    float NDF = distributionGGX(N, H, roughness);
    float Vis = smithVisibilityGGXCorrelated(NdotL, NdotV, roughness);
    vec3 F = fresnelSchlick(VdotH, F0);
    return NDF * Vis * F * NdotL * (hemi_diffuse.rgb * hemi_dir_intensity.w);
}
