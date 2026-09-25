float distributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;
    float num = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;
    return num / max(denom, 0.0000001);
}

float geometrySchlickGGX(float NdotV, float roughness) {
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

float geometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = geometrySchlickGGX(NdotV, roughness);
    float ggx1 = geometrySchlickGGX(NdotL, roughness);
    return ggx1 * ggx2;
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
    float cc_G = geometrySmith(N, V, L, cc_rough);
    vec3 cc_F = fresnelSchlick(clamp(dot(H, V), 0.0, 1.0), cc_F0) * cc_intensity;
    base_atten = vec3(1.0) - cc_F;
    vec3 cc_spec = (cc_NDF * cc_G * cc_F) / (4.0 * NdotV * NdotL + 0.0001);
    float sheenD = sheenDistributionCharlie(sheen_rough, max(dot(N, H), 0.0));
    float sheenV = sheenVisibilityNeubelt(NdotV, NdotL);
    vec3 sheen_term = sheen_tint * (sheenD * sheenV) * sheen_intensity;
    additive = cc_spec + sheen_term;
}
