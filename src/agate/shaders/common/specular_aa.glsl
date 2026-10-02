// agate's specular anti-aliasing (Babylon's `SPECULARAA` ->
// `getAARoughnessFactors`). Split out of common/pbr_brdf.glsl so that chunk
// stays pure math: this helper reads the per-draw `channel_selectors.w` lane,
// which a shader without that lane (the standard-material family) cannot
// compile.
//
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
