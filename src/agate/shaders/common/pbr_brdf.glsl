// Babylon's dielectric F0 for the metallic workflow — and the constant its
// LEGACY energy conservation is built from. `PBRBaseMaterial` uploads
// `vReflectivityColor.a = ((ior - 1) / (ior + 1))^2` (the exact normal-
// incidence Fresnel reflectance of the IOR: 0.04 at the glTF default 1.5),
// and `reflectivityBlock` uses it twice:
//
//     dielectricColorF0 = vec3(dielectricF0 * surfaceReflectivityColor)
//     surfaceAlbedo     = baseColor * (1 - dielectricF0 * surfaceReflectivityColor)
//                                    * (1 - metallic)      // LEGACY path only
//
// `surfaceReflectivityColor` is `metallicReflectanceFactors.rgb`. A glTF
// material without `KHR_materials_specular` leaves it white and
// `metallicF0Factor` at 1, so both expressions reduce to this constant; agate
// does not support that extension, so the value is not a uniform here either.
const float dielectric_f0 = 0.04;

// Babylon's roughness remap: alphaG = roughness^2 + MINIMUMVARIANCE
// (`convertRoughnessToAverageSlope`, MINIMUMVARIANCE = 0.0005). The epsilon
// keeps the slope non-zero at roughness 0 so the visibility division below
// stays finite.
float convertRoughnessToAverageSlope(float roughness) {
    return roughness * roughness + 0.0005;
}

// Babylon's specular anti-aliasing (`SPECULARAA` → `getAARoughnessFactors`).
// A sub-pixel-wide specular lobe cannot be resolved by point-sampled normals,
// so Babylon widens the specular roughness by how fast the SHADING normal
// changes across the screen (the geometric curvature the pixel footprint
// covers); the lobe then integrates roughly what the pixel would have seen.
//
// Extracted from the shipped bundle (babylonjs 9.28.0, `pbrBRDFFunctions`;
// the WGSL and GLSL copies are character-for-character equivalent):
//
//     vec2 getAARoughnessFactors(vec3 normalVector) {
//     #ifdef SPECULARAA
//         vec3 nDfdx = dFdx(normalVector.xyz);
//         vec3 nDfdy = dFdy(normalVector.xyz);
//         float slopeSquare = max(dot(nDfdx, nDfdx), dot(nDfdy, nDfdy));
//         float geometricRoughnessFactor = pow(saturate(slopeSquare), 0.333);
//         float geometricAlphaGFactor   = sqrt(slopeSquare);
//         geometricAlphaGFactor *= 0.75;
//         return vec2(geometricRoughnessFactor, geometricAlphaGFactor);
//     #else
//         return vec2(0.);
//     #endif
//     }
//
// `.x` is consumed inside `computeSpecularLighting` — shared by the sun, the
// point/spot/area/clustered lights AND the hemispheric lobe — as
//
//     float roughness = max(info.roughness, geometricRoughnessFactor);
//
// i.e. it can only ever RAISE the roughness. `.y` is only used for the
// reflection LOD (Babylon's IBL `alphaG += AARoughnessFactors.y`), which agate
// maps differently (see the IBL block: `lod = roughness * max_lod`), so this
// port carries `.x` and documents `.y` as not ported.
//
// Not an unconditional term: Babylon compiles it in only for materials with
// `enableSpecularAntiAliasing` (`PBRBaseMaterial.isReadyForSubMesh`:
// `setValue("SPECULARAA", caps.standardDerivatives && enableSpecularAntiAliasing)`).
// Its default is FALSE for a hand-built `PBRMaterial` — the bench's ground
// plane renders with it off — while the glTF loading adapters turn it on for
// every material they create (`babylonjs.loaders.js`, PBRMaterialLoadingAdapter
// constructor: `this._material.enableSpecularAntiAliasing = true`), which is
// why the helmet and the fox carry it. The flag arrives in
// `channel_selectors.w` (0 = off; see MaterialDrawRecord.channel_selectors).
float aaRoughnessFactor(vec3 normal_vector) {
    if (channel_selectors.w < 0.5) return 0.0;
    vec3 n_dx = dFdx(normal_vector);
    vec3 n_dy = dFdy(normal_vector);
    float slope_square = max(dot(n_dx, n_dx), dot(n_dy, n_dy));
    return pow(clamp(slope_square, 0.0, 1.0), 0.333);
}

float distributionGGX(vec3 N, vec3 H, float roughness) {
    float a = convertRoughnessToAverageSlope(roughness);
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;
    float num = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;
    return num / max(denom, 0.0000001);
}

// Height-correlated Smith masking-shadowing — Babylon's
// `smithVisibility_GGXCorrelated`. This is the physically-derived model
// (Heitz 2014) used by Babylon.js, Filament and the glTF sample viewer; it
// replaces the Schlick-GGX approximation with k = (r+1)^2/8, which is
// Lazarov's *IBL* fit and over-darkens grazing angles on direct light.
//
// The returned visibility already contains the 1/(4*NdotL*NdotV) factor of
// the Cook-Torrance denominator, so the specular is
//     f_spec = D * Vis * F        (the caller applies NdotL)
// matching Babylon's `specTerm = fresnel * distribution * smithVisibility`.
float smithVisibilityGGXCorrelated(float NdotL, float NdotV, float roughness) {
    float alphaG = convertRoughnessToAverageSlope(roughness);
    float a2 = alphaG * alphaG;
    float ggx_v = NdotL * sqrt(NdotV * NdotV * (1.0 - a2) + a2);
    float ggx_l = NdotV * sqrt(NdotL * NdotL * (1.0 - a2) + a2);
    return 0.5 / max(ggx_v + ggx_l, 1e-6);
}

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness) {
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

vec2 envBRDFApprox(float roughness, float NoV) {
    const vec4 c0 = vec4(-1.0, -0.0275, -0.572, 0.022);
    const vec4 c1 = vec4(1.0, 0.0425, 1.04, -0.04);
    vec4 r = roughness * c0 + c1;
    float a004 = min(r.x * r.x, exp2(-9.28 * NoV)) * r.x + r.y;
    vec2 AB = vec2(-1.04, 1.04) * a004 + r.zw;
    return AB;
}

// Clearcoat + sheen helpers (scalar/color only, Babylon parity). The coat is
// a second GGX specular lobe with its own roughness and a dielectric
// F0 = 0.04 tinted by clearcoat_color; the base specular (direct + IBL) is
// attenuated by (1 - F_cc) for energy conservation while diffuse passes
// through. Sheen is an additive fabric lobe: Charlie distribution
// (Estevez-Kulla) + Neubelt visibility with an explicit grazing-angle
// weight on the IBL path (Karis-style); the direct path gets its grazing
// response from the Neubelt visibility. Intensity 0 zeroes every term, so
// disabled materials shade bit-identically to before.
vec3 clearcoatF0() {
    return vec3(0.04) * clearcoat_color.rgb;
}

float sheenDistributionCharlie(float roughness, float NoH) {
    float alpha = max(roughness * roughness, 0.0001);
    float inv_alpha = 1.0 / alpha;
    float cos2h = NoH * NoH;
    float sin2h = max(1.0 - cos2h, 0.0078125);
    return (2.0 + inv_alpha) * pow(sin2h, inv_alpha * 0.5) / (2.0 * PI);
}

float sheenVisibilityNeubelt(float NoV, float NoL) {
    return clamp(1.0 / (4.0 * (NoL + NoV - NoL * NoV)), 0.0, 1.0);
}

// Anisotropy v1 (Heitz-style GGX): stretches the base-lobe NDF along the
// tangent frame; intensity 0 early-returns to the legacy isotropic NDF
// (bit-identical), geometry G stays isotropic and IBL stays isotropic
// (v1 scope, documented). CPU mirror: material.anisotropyAxes.
float anisoNDF(vec3 N, vec3 T, vec3 B, vec3 H, float roughness) {
    float aniso = clamp(anisotropy_factors.x, 0.0, 1.0);
    if (aniso <= 0.0) return distributionGGX(N, H, roughness);
    float rough2 = roughness * roughness;
    float aspect = sqrt(max(1.0 - 0.9 * aniso, 0.01));
    float ax = max(rough2 / aspect, 0.001);
    float ay = max(rough2 * aspect, 0.001);
    float TH = dot(T, H);
    float BH = dot(B, H);
    float NH = max(dot(N, H), 0.0);
    float denom = (TH * TH) / (ax * ax) + (BH * BH) / (ay * ay) + NH * NH;
    return 1.0 / max(PI * ax * ay * denom * denom, 0.0000001);
}

// Combined coat + sheen add-on for one punctual light. The caller multiplies
// `additive` by radiance * NdotL * shadow and scales its own BASE specular
// by `base_atten` (diffuse/kD untouched). Safe at NdotL = 0: the 0.0001
// guard matches the base lobe and the caller zeroes the contribution.
void coatSheenLight(vec3 N, vec3 V, vec3 L, vec3 H, float NdotV, float NdotL,
    float cc_rough, float cc_intensity, vec3 cc_F0,
    float sheen_rough, float sheen_intensity, vec3 sheen_tint,
    out vec3 base_atten, out vec3 additive) {
    float cc_NDF = distributionGGX(N, H, cc_rough);
    float cc_Vis = smithVisibilityGGXCorrelated(NdotL, NdotV, cc_rough);
    vec3 cc_F = fresnelSchlick(clamp(dot(H, V), 0.0, 1.0), cc_F0) * cc_intensity;
    base_atten = vec3(1.0) - cc_F;
    vec3 cc_spec = cc_NDF * cc_Vis * cc_F;
    float sheenD = sheenDistributionCharlie(sheen_rough, max(dot(N, H), 0.0));
    float sheenV = sheenVisibilityNeubelt(NdotV, NdotL);
    vec3 sheen_term = sheen_tint * (sheenD * sheenV) * sheen_intensity;
    additive = cc_spec + sheen_term;
}
