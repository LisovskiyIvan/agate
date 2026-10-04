// Instanced Cook-Torrance PBR (Physically Based Rendering) with Multi-Lights for agate
@header const m = @import("math")
@ctype mat4 m.Mat4

@vs vs
@glsl_options fixup_clipspace
layout(binding = 0) uniform vs_params {
    mat4 view_proj;
};

// Per-vertex attributes (Buffer 0)
in vec3 position;
in vec3 normal;
in vec4 tangent;
in vec4 color0;
in vec2 texcoord0;
in vec2 texcoord1;

// Per-instance attributes (Buffer 1)
in vec4 inst_mat0;
in vec4 inst_mat1;
in vec4 inst_mat2;
in vec4 inst_mat3;

out vec3 v_world_pos;
out vec3 v_normal;
out vec3 v_tangent;
out vec3 v_bitangent;
out vec4 v_color;
out vec2 v_uv;
out vec2 v_uv1;

void main() {
    mat4 model = mat4(inst_mat0, inst_mat1, inst_mat2, inst_mat3);
    vec4 world_pos = model * vec4(position, 1.0);
    v_world_pos = world_pos.xyz;
    gl_Position = view_proj * world_pos;

    vec3 N = normalize(mat3(model) * normal);
    vec3 T = normalize(mat3(model) * tangent.xyz);
    // Gram-Schmidt orthogonalization
    T = normalize(T - dot(T, N) * N);
    vec3 B = cross(N, T) * tangent.w;

    v_normal = N;
    v_tangent = T;
    v_bitangent = B;
    v_color = color0;
    v_uv = texcoord0;
    v_uv1 = texcoord1;
}
@end

@fs fs
// Shadow atlas resolution. Must match ShadowPass.SHADOW_ATLAS_SIZE in
// passes/shadow_pass.zig (sokol-shdc --defines cannot carry a value).
#ifndef SHADOW_ATLAS_SIZE
#define SHADOW_ATLAS_SIZE 2048.0
#endif
layout(binding = 1) uniform fs_params {
    vec4 eye_pos;
    vec4 light_dir;
    vec4 light_color;
    vec4 ambient_color;
    vec4 base_color_factor;
    vec4 pbr_factors; // x: metallic, y: roughness, z: occlusion_strength, w: ibl_intensity
    vec4 emissive_factor; // xyz: emissive color, w: unused
    vec4 shadow_params; // x: bias, y: intensity, z: normal_bias, w: filter_radius
    vec4 shadow_splits; // x: split0, y: split1, z: split2, w: split3
    mat4 cascade_view_proj[4]; // 4 cascade light view-projection matrices
    vec4 cascade_debug; // x: debug_cascades, y/z/w: unused
    vec4 light_counts; // x: num_point_lights, y: num_spot_lights, z/w: unused
    vec4 point_pos_range[4];
    vec4 point_color_int[4];
    vec4 spot_pos_range[2];
    vec4 spot_dir_inner[2];
    vec4 spot_color_outer[2];
    vec4 spot_intensity[2];
    mat4 spot_view_proj[2];
    vec4 spot_shadow_params[2]; // x: cast_shadows (0/1), y: bias, z: normal_bias, w: unused
    // APPENDED LAST: existing offsets above must not shift for existing bindings.
    float alpha_cutoff; // cutout threshold; 0.0 disables the alpha test
    float normal_scale; // normal map xy scale (glTF normalTexture.scale)
    // APPENDED LAST (wave/ktx2): per-slot KHR_texture_transform UV maps and
    // manual channel selection. Slot order follows the texture bindings:
    // 0 albedo, 1 normal, 2 metallic-roughness, 3 emissive, 4 occlusion.
    vec4 uv_matrix[5]; // per slot: rotation*scale rows [m00, m01, m10, m11]
    vec4 uv_offset[5]; // per slot: xy offset, zw unused
    vec4 channel_selectors; // x occlusion, y roughness, z metallic (lane index),
                            // w: specular anti-aliasing flag (0/1 — Babylon's
                            // SPECULARAA; see aaRoughnessFactor)
    // APPENDED LAST (clearcoat/sheen): coat + fabric lobes (scalar/color
    // plus optional R-mask / rgb-tint maps in pbr-layers v1, bindings 15/16).
    // Intensity 0 disables the lobe:
    // every added shader term scales by its intensity and the base-specular
    // attenuation becomes exactly (1 - 0), so disabled materials render
    // bit-identically to before. Appended last so no existing offset shifts.
    vec4 clearcoat_factors; // x: intensity (0 = off), y: roughness, z/w: unused
    vec4 clearcoat_color; // rgb: coat tint (white = untinted), w: unused
    vec4 sheen_factors; // x: intensity (0 = off), y: roughness, z/w: unused
    vec4 sheen_color; // rgb: fabric tint (white = untinted), w: unused
    // APPENDED LAST (point shadows): 2 shadow slots x 6 cube faces
    // (+X,-X,+Y,-Y,+Z,-Z). Zeroed by default: the shader early-outs with
    // zero cost when no point light casts shadows.
    mat4 point_view_proj[12];
    vec4 point_shadow_params[4]; // per point-light slot: x: shadow slot+1 (0 = none), y: bias, z: normal_bias, w: unused
    // APPENDED LAST (multi-directional): up to 4 suns; slot 0 mirrors the
    // primary (light_dir/light_color, the only shadow caster), slots 1..3
    // are shadowless fills. Unused/disabled slots are zeroed (intensity 0
    // skips in the fill loop below).
    vec4 directional_dir[4];
    vec4 directional_color_int[4]; // rgb: color, a: intensity
    // APPENDED LAST (reflection probes, wave 25): per-draw probe state.
    // x: enabled (0/1), y: probe intensity, z: probe max lod, w: unused.
    // Zeroed when no probe applies: the shader then takes the no-probe
    // ambient/IBL path bit-identically. Appended last so no offset shifts.
    // (Instanced draws always upload zero here — probes skip instanced
    // batches in v1 — but the lane must exist for layout parity with the
    // regular/skinned PBR FsParams.)
    vec4 probe_params;
    // APPENDED LAST (rect area lights, wave 26, v1): up to 2 rects in
    // creation order. area_center_int xyz = rect center, w = intensity
    // (0 when disabled/unused — the loop below skips, so zero lights
    // render bit-identically); area_right/area_up = half-extent vectors;
    // area_color rgb = emitted color. Appended last so no offset shifts.
    // No area-light shadows in v1 (unshadowed by design).
    vec4 area_center_int[2];
    vec4 area_right[2];
    vec4 area_up[2];
    vec4 area_color[2];
    // APPENDED LAST (clustered forward lights, wave 30, v1): tile-grid
    // descriptor for the storage-buffer tile walk below (the light data
    // itself is never a uniform). clustered_params = (tiles_x, tiles_y,
    // staged light count, gpu-live 0/1); clustered_viewport = (screen_w,
    // screen_h, tile_size px, unused). Zeroed with an empty pool (and w =
    // 0 until a live GPU upload lands), so the loop gates off
    // bit-identically. Appended last so no offset shifts.
    vec4 clustered_params;
    vec4 clustered_viewport;
    // APPENDED LAST (pbr-layers v1): anisotropy / transmission / SSS.
    // All zeroed when unused: every new term gates on its factor (0 = off),
    // so unlayered materials shade bit-identically. Appended last so no
    // existing offset shifts.
    vec4 anisotropy_factors; // x: intensity (0 = isotropic/off), y: tangent-plane rotation (rad), z/w: unused
    vec4 transmission_factors; // x: factor (0 = off), y/z/w: unused
    vec4 transmission_color; // rgb: throughput tint (white = untinted), w: unused
    vec4 sss_factors; // x: strength (0 = off), y/z/w: unused
    vec4 sss_color; // rgb: scatter tint (white = untinted), w: unused
    // APPENDED LAST (hemispheric light model): Babylon's HemisphericLight is
    // a light, not a flat ambient — its irradiance interpolates between
    // `ambient_color.rgb` (groundColor) and `hemi_diffuse.rgb * w` by
    // `0.5 + 0.5 * dot(N, hemi_dir_intensity.xyz)`. See common/hemi.glsl.
    vec4 hemi_dir_intensity; // xyz: direction toward the light (normalized), w: intensity
    vec4 hemi_diffuse; // rgb: diffuse color, a: unused
    vec4 clearcoat_uv_matrix;
    vec4 clearcoat_uv_offset;
    vec4 sheen_uv_matrix;
    vec4 sheen_uv_offset;
    vec4 refraction_factors;
    mat4 refraction_view_proj;
    vec4 refraction_capture;
};

layout(binding = 0) uniform texture2D albedo_tex;
layout(binding = 1) uniform texture2D normal_tex;
layout(binding = 2) uniform texture2D metallic_roughness_tex;
layout(binding = 3) uniform texture2D emissive_tex;
layout(binding = 4) uniform texture2D occlusion_tex;
layout(binding = 5) uniform texture2D shadow_tex;
@image_sample_type shadow_depth_tex unfilterable_float
layout(binding = 7) uniform texture2D shadow_depth_tex; // same atlas view as shadow_tex, raw-depth reads for PCSS
layout(binding = 6) uniform textureCube env_tex;
layout(binding = 8) uniform texture2D spot_shadow_tex;
// Point shadow atlas (passes/shadow_pass.zig POINT_SHADOW_* layout).
// Binding 10: fs uses 0..8, the instanced vs uses no textures — same slot
// as the other PBR-family shaders. Sampled through shadow_smp.
layout(binding = 10) uniform texture2D point_shadow_tex;
layout(binding = 17) uniform texture2D brdf_lut_tex; // Babylon env-BRDF lookup (coloredEnergyConservationFactor)
layout(binding = 0) uniform sampler smp; // color slot: albedo (its own sampler)
// Data slots (normal / metallic-roughness / occlusion / emissive) sample
// through data_smp so their textures keep their OWN filter/wrap settings
// instead of inheriting the albedo sampler.
layout(binding = 5) uniform sampler data_smp;
layout(binding = 1) uniform sampler shadow_smp;
layout(binding = 2) uniform sampler env_smp;
@sampler_type depth_smp nonfiltering
layout(binding = 3) uniform sampler depth_smp;
// Reflection probe cube (wave 25): binding 11 is the next free texture
// slot in the shared pool (fs uses 0..8 and 10, the instanced vs uses no
// textures), binding 6 the next free sampler slot. Instanced draws bind
// the default cube with zeroed params (no-probe path); the slots must still
// exist for layout parity with the regular PBR family.
layout(binding = 11) uniform textureCube probe_tex;
layout(binding = 6) uniform sampler probe_smp;
layout(binding = 7) uniform sampler brdf_lut_smp;
// PBR layers v1: coat/fabric masks sampled through data_smp (each texture
// keeps its own filter/wrap, like the other data slots — no new sampler).
// Unset slots bind the default white texture: mask 1 / tint 1 = identity,
// so the scalar path stays bit-identical. Bindings 15/16 are free view
// slots in the shared pool (fs uses 0..8, 10, 11; vs uses 9 for morph;
// 12..14 are storage buffers, a separate array).
layout(binding = 15) uniform texture2D clearcoat_tex;
layout(binding = 16) uniform texture2D sheen_tex;
layout(binding = 18) uniform texture2D refraction_tex;
layout(binding = 8) uniform sampler refraction_smp;
// Clustered forward lights (wave 30, v1): storage buffers for the tile
// walk. Bindings 12..14 are free view slots in the shared pool of every
// forward shader (this family uses fs 0..8, 10, 11, vs no textures).
// Layouts mirror scene/clustered_lights.zig (ClusterLightGpu = 2x vec4,
// ClusterTileGpu = uvec2, indices = u32); every buffer block holds exactly
// one flexible array of a struct (sokol-shdc requirement). The draw always
// binds something valid here (real tile buffers when a GPU upload landed,
// the shared 16-byte dummy otherwise); the loop below only reads them when
// clustered_params gates it on.
// @include "common/cluster.glsl"

in vec3 v_world_pos;
in vec3 v_normal;
in vec3 v_tangent;
in vec3 v_bitangent;
in vec4 v_color;
in vec2 v_uv;
in vec2 v_uv1;

out vec4 frag_color;
// @include "common/refraction.glsl"

// PI comes from common/pbr_brdf.glsl (single source for the BRDF chunk).

const vec2 POISSON_DISK[16] = vec2[](
    vec2(-0.94201624, -0.39906216),
    vec2( 0.94558609, -0.76890725),
    vec2(-0.09418410, -0.92938870),
    vec2( 0.34495938,  0.29387760),
    vec2(-0.91588581,  0.45771432),
    vec2(-0.81544232, -0.87912464),
    vec2(-0.38277543,  0.27676845),
    vec2( 0.97484398,  0.75648379),
    vec2( 0.44323325, -0.97511554),
    vec2( 0.53742981, -0.47373420),
    vec2(-0.26496911, -0.41893023),
    vec2( 0.79197514,  0.19090188),
    vec2(-0.24188840,  0.99706507),
    vec2(-0.81409955,  0.91437590),
    vec2( 0.19984126,  0.78641367),
    vec2( 0.14383161, -0.14100790)
);

const vec2 CASCADE_OFFSETS[4] = vec2[](
    vec2(0.0, 0.0),
    vec2(0.5, 0.0),
    vec2(0.0, 0.5),
    vec2(0.5, 0.5)
);

// PCSS (percentage-closer soft shadows) for the 4-cascade sun atlas.
// Blocker search reads raw depths from shadow_depth_tex (same atlas view as
// shadow_tex, regular sampler): PCSS_BLOCKER_SAMPLES Poisson taps inside
// pcss_blocker_radius, average blocker depth -> penumbra ->
// (d_receiver - d_blocker) / d_blocker * light_size, clamped to
// [min_penumbra, max_penumbra]; the fixed-radius Poisson PCF then runs with the
// penumbra as its disk radius. Params ride free uniform lanes (fs_params
// layout unchanged):
//   cascade_debug.y = pcss_enabled (0.0/1.0)
//   cascade_debug.z = pcss_light_size
//   cascade_debug.w = pcss_blocker_radius (atlas-UV search radius)
//   light_counts.z  = pcss_min_penumbra (atlas-UV clamp)
//   light_counts.w  = pcss_max_penumbra (atlas-UV clamp)
// Disabled (y <= 0.5): fixed-radius 16x/8x Poisson PCF, bit-identical.
// Counts mirror scene/shadow_pcss.zig (blocker_sample_count).
#define PCSS_BLOCKER_SAMPLES 12

// @include "common/shadow_pcf.glsl"

// @include "common/pbr_brdf.glsl"

// KHR_texture_transform: uv' = matrix * uv + offset. Identity uniforms make
// this a no-op, so materials without the extension sample unchanged.
// @include "common/uv_apply.glsl"

// Manual channel selection (no glTF counterpart; see Channel in
// material.zig): picks one RGBA lane by uniform index through constant
// branches — SPIRV-Cross cannot flatten dynamic component indexing for
// HLSL5 targets, same constraint as morphWeight.
// @include "common/channel_select.glsl"
// @include "common/hemi.glsl"
// @include "common/hemi_pbr.glsl"
// @include "common/specular_aa.glsl"
// @include "common/linear_output.glsl"

void main() {
    vec4 albedo_tex_val = texture(sampler2D(albedo_tex, smp), uvApply(uv_matrix[0], uv_offset[0], v_uv));
    vec4 albedo_rgba = v_color * base_color_factor * albedo_tex_val;
    // Alpha test (cutout): cutout materials discard sub-cutoff fragments
    // before any lighting work. Opaque/blend materials upload 0.0, so this
    // never fires for them (alpha is always >= 0.0).
    if (albedo_rgba.a < alpha_cutoff) discard;

    // Unlit mode: bypass all lighting, shadows, and IBL
    if (uv_offset[0].z > 0.5) {
        vec4 emissive_sample = texture(sampler2D(emissive_tex, data_smp), uvApply(uv_matrix[3], uv_offset[3], v_uv));
        vec3 emissive = emissive_factor.rgb * emissive_sample.rgb;
        frag_color = linearOutputColor(albedo_rgba.rgb + emissive, albedo_rgba.a);
        return;
    }
    vec3 albedo = albedo_rgba.rgb;

    vec4 mr_sample = texture(sampler2D(metallic_roughness_tex, data_smp), uvApply(uv_matrix[2], uv_offset[2], v_uv));
    float metallic = clamp(pbr_factors.x * channelSelect(mr_sample, channel_selectors.z), 0.0, 1.0);
    float roughness = clamp(pbr_factors.y * channelSelect(mr_sample, channel_selectors.y), 0.04, 1.0);

    // Normal mapping with TBN matrix; xy scaled by normal_scale (z stays
    // unsigned so the TBN projection keeps the hemisphere).
    vec3 map_n = texture(sampler2D(normal_tex, data_smp), uvApply(uv_matrix[1], uv_offset[1], v_uv)).xyz * 2.0 - 1.0;
    map_n.xy *= normal_scale;
    mat3 TBN = mat3(normalize(v_tangent), normalize(v_bitangent), normalize(v_normal));
    vec3 N = normalize(TBN * map_n);

    // Babylon's `TWOSIDEDLIGHTING` (pbrBlockNormalFinal): with back-face
    // culling OFF and the material flag ON, the shading normal is flipped on
    // back faces so the inside of a double-sided surface is lit like its
    // outside. The flag rides `emissive_factor.w` (unused by Babylon's own
    // vec3 emissive upload). Applied AFTER the normal map, like Babylon.
    if (emissive_factor.w > 0.5 && !gl_FrontFacing) N = -N;

    vec3 V = normalize(eye_pos.xyz - v_world_pos);
    float NdotV = max(dot(N, V), 0.0001);

    // Babylon's specular anti-aliasing (SPECULARAA — see aaRoughnessFactor in
    // common/pbr_brdf.glsl): `computeSpecularLighting` evaluates its lobes at
    //    roughness = max(info.roughness, AARoughnessFactors.x)
    // so the screen-space normal variation can only RAISE the specular
    // roughness. `aa_rough` is exactly 0 with the material flag off, which
    // makes `spec_roughness == roughness` and every lobe bit-identical to the
    // pre-SPECULARAA shading. The IBL block below deliberately keeps the raw
    // `roughness` (Babylon feeds it its own `alphaG`, not this value).
    float aa_rough = aaRoughnessFactor(N);
    float spec_roughness = max(roughness, aa_rough);

    vec3 F0 = vec3(dielectric_f0);
    F0 = mix(F0, albedo, metallic);
    // Babylon `coloredEnergyConservationFactor` (MS_BRDF_ENERGY_CONSERVATION):
    // the env-BRDF lookup at vec2(NdotV, perceptualRoughness) scales the whole
    // analytic specular sum and the specular IBL. Babylon samples the lookup
    // with the RAW perceptual roughness (before SPECULARAA), and the engine
    // uploads the LUT linearised, so `.y` is already the linear value.
    vec2 environment_brdf = texture(sampler2D(brdf_lut_tex, brdf_lut_smp), vec2(NdotV, roughness)).xy;
    vec3 spec_ec = specEnergyConservation(environment_brdf.y, F0);

    // Energy-conserved diffuse albedo (Babylon `reflectivityBlock`). Under the
    // `LEGACY_SPECULAR_ENERGY_CONSERVATION` define — which PBRBaseMaterial sets
    // unconditionally — Babylon feeds
    //     surfaceAlbedo = baseColor * (1 - dielectricF0 * surfaceReflectivityColor)
    //                              * (1 - metallic)
    // to EVERY diffuse term (`finalDiffuse = diffuseBase * surfaceAlbedo`,
    // `finalIrradiance *= surfaceAlbedo`), while the specular keeps the raw
    // `baseColor` in `F0` above. glTF without KHR_materials_specular (all agate
    // supports) leaves `surfaceReflectivityColor` white, so the factor is the
    // constant `1 - dielectric_f0`; the `(1 - metallic)` half is already
    // applied per site (in `kD`, in the hemispheric term). Metals are
    // unaffected (their diffuse is gated to zero anyway); dielectrics and the
    // ground lose the 4%.
    vec3 diffuse_albedo = albedo * (1.0 - dielectric_f0);

    // Clearcoat + sheen factors (scalar/color only, no textures this wave).
    // Intensity 0 disables the lobe while keeping every direct term exact.
    // Babylon's computeClearCoatLighting / computeSheenLighting apply the
    // same max(roughness, AARoughnessFactors.x) to the coat / sheen lobe.
    float cc_rough = max(clamp(clearcoat_factors.y, 0.03, 1.0), aa_rough);
    float cc_intensity = clamp(clearcoat_factors.x, 0.0, 1.0);
    vec3 cc_F0 = clearcoatF0();
    // Babylon's computeClearCoatLighting / computeSheenLighting apply the
    // same max(roughness, AARoughnessFactors.x) to the coat / sheen lobe.
    float sheen_rough = max(clamp(sheen_factors.y, 0.07, 1.0), aa_rough);
    float sheen_intensity = clamp(sheen_factors.x, 0.0, 1.0);

    // PBR layers v1: coat/fabric masks sampled with the ALBEDO uv transform
    // (no per-slot coat transform yet — v1 scope). White fallback = 1, so
    // unset slots keep the scalar path exact; the branch only skips the
    // fetch when the lobe is off (mask * 0 == 0 either way).
    vec2 coat_uv = uvApply(clearcoat_uv_matrix, clearcoat_uv_offset, v_uv);
    float cc_mask = 1.0;
    if (cc_intensity > 0.0) {
        cc_mask = texture(sampler2D(clearcoat_tex, data_smp), coat_uv).r;
    }
    cc_intensity *= cc_mask;
    vec3 sheen_tint = sheen_color.rgb;
    if (sheen_intensity > 0.0) {
        sheen_tint *= texture(sampler2D(sheen_tex, data_smp), uvApply(sheen_uv_matrix, sheen_uv_offset, v_uv)).rgb;
    }

    // Anisotropy v1 frame: the existing TBN varyings (vertex tangent
    // attribute, Gram-Schmidt orthogonalized in vs) — NO new attribute.
    // Meshes without authored tangents carry the loader/builder fallback
    // (+X, see loader/mesh_spawn.zig): documented approximation.
    vec3 aniso_T = normalize(v_tangent);
    vec3 aniso_B = normalize(v_bitangent);
    float aniso_rot = anisotropy_factors.y;
    if (clamp(anisotropy_factors.x, 0.0, 1.0) > 0.0 && aniso_rot != 0.0) {
        float aniso_cr = cos(aniso_rot);
        float aniso_sr = sin(aniso_rot);
        vec3 aniso_rT = aniso_T * aniso_cr + aniso_B * aniso_sr;
        aniso_B = aniso_B * aniso_cr - aniso_T * aniso_sr;
        aniso_T = aniso_rT;
    }

    // Transmission v1: thin-slab approx (no refraction target — non-goal).
    // Diffuse throughput scales by (1 - factor) AFTER F0 so metals keep
    // their F0; the additive back-light term lands in the post-Lo block.
    // factor 0 skips: unlayered albedo bit-identical.
    vec3 orig_albedo = albedo;
    float transm_factor = clamp(transmission_factors.x, 0.0, 1.0);
    bool refracting = refraction_factors.x > 0.5 && refraction_capture.x > 0.5;
    if (transm_factor > 0.0) {
        albedo *= (1.0 - transm_factor);
        // keep the energy-conserved copy in step (it feeds every diffuse site)
        diffuse_albedo *= (1.0 - transm_factor);
    }
    float sss_strength = clamp(sss_factors.x, 0.0, 1.0);

    // 1. Primary Directional Light (with shadow mapping)
    vec3 L = light_dir.xyz;
    vec3 H = normalize(V + L);
    float NdotL = max(dot(N, L), 0.0);

    float NDF = anisoNDF(N, aniso_T, aniso_B, H, spec_roughness);
    float Vis = smithVisibilityGGXCorrelated(NdotL, NdotV, spec_roughness);
    vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);

    vec3 specular = NDF * Vis * F;
    // No (1 - F) Fresnel factor here (nor on the other analytic diffuse
    // sites below): Babylon's punctual diffuse is `diffuseBase *
    // surfaceAlbedo` with `diffuseBase` from `computeDiffuseLighting`
    // (diffuseTerm * attenuation * NdotL * lightColor, no Fresnel) and
    // `surfaceAlbedo` carrying only the (1 - metallic) gate — verified
    // against babylon.js 9.28.0. An extra (1 - F) darkened fractional-metal
    // diffuse by up to ~30% (F0 ~ 0.3 at metallic 0.64); it is invisible at
    // metallic 0/1, which is why scalar probes never caught it.
    vec3 kD = vec3(1.0 - metallic);

    // Coat + sheen add-on for the sun (base_atten scales the BASE specular
    // only — diffuse passes through unattenuated).
    vec3 sun_atten;
    vec3 sun_additive;
    coatSheenLight(N, V, L, H, NdotV, NdotL, cc_rough, cc_intensity, cc_F0, sheen_rough, sheen_intensity, sheen_tint, sun_atten, sun_additive);

    vec3 debug_tint = vec3(0.0);
    float shadow = calculateShadow(v_world_pos, N, L, debug_tint);
    vec3 radiance = light_color.rgb * light_color.a;
    vec3 Lo = (kD * diffuse_albedo / PI + specular * sun_atten * spec_ec + sun_additive) * radiance * NdotL * (1.0 - shadow);

    // Extra directional fills (slots 1..3, no shadows): the same
    // Cook-Torrance lobe as the sun (coat + sheen included), no shadow
    // term. Zero intensity (disabled/unused) skips, so a single sun
    // shades bit-identically.
    for (int i = 1; i < 4; i++) {
        vec3 d_dir = directional_dir[i].xyz;
        vec3 d_col = directional_color_int[i].rgb;
        float d_int = directional_color_int[i].w;
        if (d_int <= 0.0) continue;
        float d_NdotL = max(dot(N, d_dir), 0.0);
        if (d_NdotL <= 0.0) continue;
        vec3 d_H = normalize(V + d_dir);
        float d_NDF = anisoNDF(N, aniso_T, aniso_B, d_H, spec_roughness);
        float d_Vis = smithVisibilityGGXCorrelated(d_NdotL, NdotV, spec_roughness);
        vec3 d_F = fresnelSchlick(max(dot(d_H, V), 0.0), F0);
        vec3 d_spec = d_NDF * d_Vis * d_F;
        vec3 d_kD = vec3(1.0 - metallic);
        vec3 d_rad = d_col * d_int;
        vec3 d_atten;
        vec3 d_additive;
        coatSheenLight(N, V, d_dir, d_H, NdotV, d_NdotL, cc_rough, cc_intensity, cc_F0, sheen_rough, sheen_intensity, sheen_tint, d_atten, d_additive);
        Lo += (d_kD * diffuse_albedo / PI + d_spec * d_atten * spec_ec + d_additive) * d_rad * d_NdotL;
    }

    // 2. Point Lights (up to 4)
    int num_points = int(light_counts.x);
    for (int i = 0; i < 4; i++) {
        if (i >= num_points) break;
        vec3 p_pos = point_pos_range[i].xyz;
        float p_range = point_pos_range[i].w;
        vec3 p_col = point_color_int[i].rgb;
        float p_int = point_color_int[i].w;

        vec3 p_to_light = p_pos - v_world_pos;
        float dist = length(p_to_light);
        if (dist >= p_range || dist < 0.0001) continue;

        vec3 p_L = p_to_light / dist;
        vec3 p_H = normalize(V + p_L);

        // Windowed inverse-square attenuation (smooth cutoff at p_range)
        float d_norm = dist / p_range;
        float factor = clamp(1.0 - d_norm * d_norm * d_norm * d_norm, 0.0, 1.0);
        float att = (factor * factor) / (dist * dist + 1.0);

        float p_NdotL = max(dot(N, p_L), 0.0);
        if (p_NdotL > 0.0) {
            float p_NDF = anisoNDF(N, aniso_T, aniso_B, p_H, spec_roughness);
            float p_Vis = smithVisibilityGGXCorrelated(p_NdotL, NdotV, spec_roughness);
            vec3 p_F = fresnelSchlick(max(dot(p_H, V), 0.0), F0);

            vec3 p_spec = p_NDF * p_Vis * p_F;
            vec3 p_kD = vec3(1.0 - metallic);
            vec3 p_rad = p_col * (p_int * att);
            vec3 p_atten;
            vec3 p_additive;
            coatSheenLight(N, V, p_L, p_H, NdotV, p_NdotL, cc_rough, cc_intensity, cc_F0, sheen_rough, sheen_intensity, sheen_tint, p_atten, p_additive);

            float point_shadow = calculatePointShadow(i, v_world_pos, N, p_L);
            Lo += (p_kD * diffuse_albedo / PI + p_spec * p_atten * spec_ec + p_additive) * p_rad * p_NdotL * (1.0 - point_shadow);
        }
        if (transm_factor > 0.0 && !refracting) {
            float p_transm_back = max(dot(-N, p_L), 0.0);
            if (p_transm_back > 0.0) {
                vec3 p_rad = p_col * (p_int * att);
                Lo += transmission_color.rgb * transm_factor * orig_albedo / PI * p_rad * p_transm_back;
            }
        }
    }

    // 3. Spot Lights (up to 2)
    int num_spots = int(light_counts.y);
    for (int i = 0; i < 2; i++) {
        if (i >= num_spots) break;
        vec3 s_pos = spot_pos_range[i].xyz;
        float s_range = spot_pos_range[i].w;
        vec3 s_dir = spot_dir_inner[i].xyz;
        float cos_inner = spot_dir_inner[i].w;
        vec3 s_col = spot_color_outer[i].rgb;
        float cos_outer = spot_color_outer[i].w;
        float s_int = spot_intensity[i].x;

        vec3 s_to_light = s_pos - v_world_pos;
        float dist = length(s_to_light);
        if (dist >= s_range || dist < 0.0001) continue;

        vec3 s_L = s_to_light / dist;
        vec3 s_H = normalize(V + s_L);

        // Distance attenuation
        float d_norm = dist / s_range;
        float factor = clamp(1.0 - d_norm * d_norm * d_norm * d_norm, 0.0, 1.0);
        float dist_att = (factor * factor) / (dist * dist + 1.0);

        // Cone attenuation
        float cos_angle = dot(-s_L, s_dir);
        float cone_att = clamp((cos_angle - cos_outer) / max(cos_inner - cos_outer, 0.0001), 0.0, 1.0);
        cone_att *= cone_att;

        float total_att = dist_att * cone_att;
        if (total_att <= 0.0) continue;

        float s_NdotL = max(dot(N, s_L), 0.0);
        if (s_NdotL > 0.0) {
            float s_NDF = anisoNDF(N, aniso_T, aniso_B, s_H, spec_roughness);
            float s_Vis = smithVisibilityGGXCorrelated(s_NdotL, NdotV, spec_roughness);
            vec3 s_F = fresnelSchlick(max(dot(s_H, V), 0.0), F0);

            vec3 s_spec = s_NDF * s_Vis * s_F;
            vec3 s_kD = vec3(1.0 - metallic);
            vec3 s_rad = s_col * (s_int * total_att);
            vec3 s_atten;
            vec3 s_additive;
            coatSheenLight(N, V, s_L, s_H, NdotV, s_NdotL, cc_rough, cc_intensity, cc_F0, sheen_rough, sheen_intensity, sheen_tint, s_atten, s_additive);

            float spot_shadow = calculateSpotShadow(i, v_world_pos, N, s_L);
            Lo += (s_kD * diffuse_albedo / PI + s_spec * s_atten * spec_ec + s_additive) * s_rad * s_NdotL * (1.0 - spot_shadow);
        }
    }

    // Rect area lights (up to 2, no shadows): zero-intensity lanes skip,
    // so zero lights add nothing bit-identically. Same Cook-Torrance lobe
    // as the other punctual lights, driven by the closest-point
    // representative direction (no LTC in v1 — see areaLightFactor).
    for (int i = 0; i < 2; i++) {
        vec3 a_L;
        float a_NdotL;
        float a_factor = areaLightFactor(v_world_pos, N, i, a_L, a_NdotL);
        if (a_factor <= 0.0) continue;
        vec3 a_H = normalize(V + a_L);
        float a_NDF = anisoNDF(N, aniso_T, aniso_B, a_H, spec_roughness);
        float a_Vis = smithVisibilityGGXCorrelated(a_NdotL, NdotV, spec_roughness);
        vec3 a_F = fresnelSchlick(max(dot(a_H, V), 0.0), F0);
        vec3 a_spec = a_NDF * a_Vis * a_F;
        vec3 a_kD = vec3(1.0 - metallic);
        vec3 a_rad = area_color[i].rgb * a_factor;
        vec3 a_atten;
        vec3 a_additive;
        coatSheenLight(N, V, a_L, a_H, NdotV, a_NdotL, cc_rough, cc_intensity, cc_F0, sheen_rough, sheen_intensity, sheen_tint, a_atten, a_additive);
        Lo += (a_kD * diffuse_albedo / PI + a_spec * a_atten * spec_ec + a_additive) * a_rad * a_NdotL;
    }

    // Clustered forward point lights (up to 64, no shadows): pixel -> tile
    // -> tile light indices -> the same Cook-Torrance lobe as the direct
    // lanes above (coat + sheen included, no shadow term). Gated on a live
    // tile grid (tiles_x/y > 0), a non-empty staged pool (count > 0) and a
    // landed GPU upload (w > 0.5); otherwise skipped entirely, so the empty
    // pool shades bit-identically. Tile id follows the CPU build
    // (scene/clustered_lights.zig tileNdcRect, bottom-left origin); Metal
    // and D3D run window-y down, hence the !SOKOL_GLSL flip. The tile
    // lookup is byte-identical in all five forward shaders (only the lobe
    // tail mirrors the host shader); sokol-shdc has no include mechanism,
    // so the duplication is deliberate and documented.
    if (clustered_params.x > 0.5 && clustered_params.y > 0.5 && clustered_params.z > 0.5 && clustered_params.w > 0.5) {
        vec2 c_px = gl_FragCoord.xy;
        #if !SOKOL_GLSL
            c_px.y = clustered_viewport.y - c_px.y;
        #endif
        int c_tx = clamp(int(c_px.x / clustered_viewport.z), 0, int(clustered_params.x) - 1);
        int c_ty = clamp(int(c_px.y / clustered_viewport.z), 0, int(clustered_params.y) - 1);
        uvec2 c_head = cluster_tiles[uint(c_ty) * uint(clustered_params.x) + uint(c_tx)].head;
        for (uint c_k = 0u; c_k < c_head.y; c_k++) {
            uint c_li = cluster_indices[c_head.x + c_k].light;
            vec3 c_pos = cluster_lights[c_li].pos_range.xyz;
            float c_range = cluster_lights[c_li].pos_range.w;
            vec3 c_col = cluster_lights[c_li].color_int.rgb;
            float c_int = cluster_lights[c_li].color_int.w;

            vec3 c_to_light = c_pos - v_world_pos;
            float c_dist = length(c_to_light);
            if (c_dist >= c_range || c_dist < 0.0001) continue;

            vec3 c_L = c_to_light / c_dist;
            vec3 c_H = normalize(V + c_L);

            float c_d_norm = c_dist / c_range;
            float c_factor = clamp(1.0 - c_d_norm * c_d_norm * c_d_norm * c_d_norm, 0.0, 1.0);
            float c_att = (c_factor * c_factor) / (c_dist * c_dist + 1.0);

            float c_NdotL = max(dot(N, c_L), 0.0);
            if (c_NdotL > 0.0) {
                float c_NDF = anisoNDF(N, aniso_T, aniso_B, c_H, spec_roughness);
                float c_Vis = smithVisibilityGGXCorrelated(c_NdotL, NdotV, spec_roughness);
                vec3 c_F = fresnelSchlick(max(dot(c_H, V), 0.0), F0);
                vec3 c_spec = c_NDF * c_Vis * c_F;
                vec3 c_kD = vec3(1.0 - metallic);
                vec3 c_rad = c_col * (c_int * c_att);
                vec3 c_atten;
                vec3 c_additive;
                coatSheenLight(N, V, c_L, c_H, NdotV, c_NdotL, cc_rough, cc_intensity, cc_F0, sheen_rough, sheen_intensity, sheen_tint, c_atten, c_additive);
                Lo += (c_kD * diffuse_albedo / PI + c_spec * c_atten * spec_ec + c_additive) * c_rad * c_NdotL;
            }
        }
    }

    // Transmission v1 + SSS v1 post terms (sun + ambient driven; point /
    // spot / area / clustered punctuals do NOT contribute — v1 scope).
    // Both gate on a uniform branch: 0 adds exactly nothing bit-identical.
    if (transm_factor > 0.0 && !refracting) {
        float transm_back = clamp(dot(-N, L) * 0.5 + 0.5, 0.0, 1.0);
        vec3 transm_irr = light_color.rgb * light_color.a * transm_back + hemiIrradiance(N) * 0.5;
        Lo += transmission_color.rgb * transm_factor * orig_albedo * transm_irr;
    }
    if (sss_strength > 0.0) {
        // Wrap mirror: material.wrapNdotL (unit-tested on CPU).
        float sss_wrap = sss_strength * 0.5;
        float wrap_nl = clamp((dot(N, L) + sss_wrap) / (1.0 + sss_wrap), 0.0, 1.0);
        float back_scatter = pow(clamp(dot(V, -L), 0.0, 1.0), 2.0);
        vec3 sss_irr = light_color.rgb * light_color.a * (wrap_nl * 0.6 + back_scatter * 0.4) + hemiIrradiance(N) * 0.25;
        Lo += sss_color.rgb * sss_strength * albedo * sss_irr;
    }

    // Ambient Occlusion
    float ao_sample = channelSelect(texture(sampler2D(occlusion_tex, data_smp), uvApply(uv_matrix[4], uv_offset[4], v_uv)), channel_selectors.x);
    float ao = 1.0 + pbr_factors.z * (ao_sample - 1.0);

    // Image-Based Lighting (IBL): two cube fetches + BRDF fit skipped when off.
    // Reflection probe (wave 25): same substitution as the regular PBR
    // shader (probe cube replaces env_tex when probe_params.x > 0.5).
    // Instanced draws always upload zero here, so this stays on the env-map branch.
    vec3 ibl = vec3(0.0);
    float ibl_intensity = pbr_factors.w;
    if (ibl_intensity > 0.001) {
        vec3 R = reflect(-V, N);
        float max_lod = 7.0;
        float lod = roughness * max_lod;
        vec3 prefiltered_spec;
        vec3 irradiance;
        if (probe_params.x > 0.5) {
            prefiltered_spec = textureLod(samplerCube(probe_tex, probe_smp), R, clamp(lod, 0.0, probe_params.z)).rgb * probe_params.y;
            irradiance = textureLod(samplerCube(probe_tex, probe_smp), N, probe_params.z).rgb * probe_params.y;
        } else {
            prefiltered_spec = textureLod(samplerCube(env_tex, env_smp), R, lod).rgb;
            irradiance = textureLod(samplerCube(env_tex, env_smp), N, max_lod).rgb;
        }

        vec3 reflectance_ibl = environmentReflectance(F0, environment_brdf);
        vec3 specular_ibl = prefiltered_spec * reflectance_ibl * spec_ec;

        // Clearcoat IBL: own roughness lobe; its fresnel attenuates the base
        // specular IBL (energy conservation). Sheen IBL: grazing-weighted
        // share of the diffuse irradiance. Both gated on intensity so the
        // disabled path keeps the direct IBL bit-identical (and skips the
        // extra cube fetch).
        vec3 cc_F_ibl = fresnelSchlickRoughness(NdotV, cc_F0, cc_rough) * cc_intensity;
        vec3 cc_spec_ibl = vec3(0.0);
        if (cc_intensity > 0.001) {
            float cc_lod = cc_rough * max_lod;
            vec3 cc_prefiltered;
            if (probe_params.x > 0.5) {
                cc_prefiltered = textureLod(samplerCube(probe_tex, probe_smp), R, clamp(cc_lod, 0.0, probe_params.z)).rgb * probe_params.y;
            } else {
                cc_prefiltered = textureLod(samplerCube(env_tex, env_smp), R, cc_lod).rgb;
            }
            vec2 cc_brdf = texture(sampler2D(brdf_lut_tex, brdf_lut_smp), vec2(NdotV, cc_rough)).xy;
            cc_spec_ibl = cc_prefiltered * environmentReflectance(cc_F0, cc_brdf) * cc_intensity;
        }
        specular_ibl = specular_ibl * (vec3(1.0) - cc_F_ibl) + cc_spec_ibl;

        vec3 kD_ibl = clamp(vec3(1.0) - reflectance_ibl * spec_ec, 0.0, 1.0) * (1.0 - metallic);
        vec3 diffuse_ibl = kD_ibl * irradiance * diffuse_albedo;
        float sheen_grazing = pow(clamp(1.0 - NdotV, 0.0, 1.0), 5.0);
        vec3 sheen_ibl = sheen_tint * sheen_intensity * irradiance * sheen_grazing;

        ibl = (diffuse_ibl + specular_ibl) * (ibl_intensity * ao) + sheen_ibl * (ibl_intensity * ao);
    }

    // Directional ambient base
    // Hemispheric base (Babylon model — see common/hemi.glsl). Metals have
    // no diffuse lobe: Babylon's `diffuseColor = albedo * (1 - metallic)`,
    // so a full metal takes only the specular path (an earlier revision fed
    // the hemispheric term to metals too, washing them out).
    // Ambient occlusion does NOT touch this term. Babylon's hemispheric light
    // is an analytic light: its diffuse lands in `diffuseBase` and therefore in
    // `finalDiffuse`, which is occluded only by
    //     ambientOcclusionForDirectDiffuse = mix(vec3(1), ao,
    //                                            ambientTextureImpactOnAnalyticalLights)
    // (`pbrBlockFinalUnlitComponents`) and
    // `PBRMaterial.DEFAULT_AO_ON_ANALYTICAL_LIGHTS == 0`. The occlusion map
    // reaches `finalAmbient` (= vAmbientColor * surfaceAlbedo, black by default)
    // and the IBL irradiance only — both handled above/below. The bench measures
    // the difference: see PROBE.md §10.
    vec3 ambient = hemiIrradiance(N) * diffuse_albedo * (1.0 - metallic);

    // Hemispheric SPECULAR (Babylon model — see common/hemi.glsl). Babylon's
    // PBR adds a full GGX lobe for a HEMILIGHT, coloured by the light's
    // `vLightDiffuse` (not `vLightSpecular`) and Fresnel-weighted by the
    // material F0. For a metal that lobe is the ONLY hemispheric term left
    // (the irradiance above is `(1 - metallic)`-gated), which is exactly the
    // metal-path residual the bench measured. NOT multiplied by `ao`:
    // Babylon occludes `finalDiffuse` and the IBL irradiance, never its
    // analytic `finalSpecular`. Evaluated once per fragment, not per light.
    Lo += (hemiSpecular(N, V, NdotV, spec_roughness, F0) * spec_ec);

    // Emissive
    vec4 emissive_sample = texture(sampler2D(emissive_tex, data_smp), uvApply(uv_matrix[3], uv_offset[3], v_uv));
    vec3 emissive = emissive_factor.rgb * emissive_sample.rgb;

    vec3 final_color = ambient + ibl + Lo + emissive + debug_tint;
    if (refracting) final_color += refractedBackground(N, V, roughness) * transm_factor * (1.0 - metallic) * transmission_color.rgb;

    frag_color = linearOutputColor(final_color, albedo_rgba.a);
}
@end

@program instanced_pbr vs fs
