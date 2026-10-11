// Shared vertex-stage morph-target helpers (GPU morph path).
//
// Included by the morph-capable PBR vertex shaders (pbr, skinned_pbr)
// via `the morph_helpers include`. Single source so the
// blend loop stays bit-identical across the family by construction
// (instanced_pbr has no morph path and does not include this).
//
// Contract:
// - Reads the caller's vs_morph block (morph_weights0/1, morph_params)
//   plus morph_tex + morph_smp (RGBA32F strip, NEAREST). The block
//   declaration stays per-shader — pbr uses UB 2, skinned_pbr UB 3
//   (different free slots in the shared cross-stage pool); only the pure
//   helpers live here.
// - Deforms bind-pose attributes in place: skinned callers invoke
//   applyMorphDeltas BEFORE the skin matrix (glTF: morphs deform the bind
//   pose, skinning then transforms), rigid callers before the model
//   transform. Mirrors morph_gpu.blendDeltas (mesh/morph_gpu.zig).
// One RGBA32F texel (xyz) at strip position idx; exact texel centers with
// NEAREST filtering, so no filtering support is needed for float textures.
vec3 morphTexel(float idx) {
    float u = (mod(idx, morph_params.y) + 0.5) / morph_params.y;
    float v = (floor(idx / morph_params.y) + 0.5) / morph_params.z;
    return texture(sampler2D(morph_tex, morph_smp), vec2(u, v)).xyz;
}

// Target weight lookup with constant vec4 lanes: SPIRV-Cross cannot
// flatten dynamic component indexing (morph_weights0[t]) for HLSL5
// targets, so the lane is selected via constant branches.
float morphWeight(int t) {
    if (t == 0) return morph_weights0.x;
    if (t == 1) return morph_weights0.y;
    if (t == 2) return morph_weights0.z;
    if (t == 3) return morph_weights0.w;
    if (t == 4) return morph_weights1.x;
    if (t == 5) return morph_weights1.y;
    if (t == 6) return morph_weights1.z;
    return morph_weights1.w;
}

// Mirrors morph_gpu.blendDeltas (mesh/morph_gpu.zig): per-vertex
// base + sum(weight * delta) accumulation in the same order.
void applyMorphDeltas(inout vec3 pos, inout vec3 nrm, inout vec3 tan_xyz, int vertex_id) {
    if (morph_params.x < 0.5) return;
    float base = float(vertex_id) * 24.0;
    for (int t = 0; t < 8; t++) {
        float w = morphWeight(t);
        if (w == 0.0) continue;
        float idx = base + float(t) * 3.0;
        pos += w * morphTexel(idx);
        nrm += w * morphTexel(idx + 1.0);
        tan_xyz += w * morphTexel(idx + 2.0);
    }
}
