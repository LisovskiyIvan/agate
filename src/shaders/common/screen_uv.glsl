// Screen-space UV <-> NDC helpers with the backend orientation rule in ONE
// place. The scene-side shaders (ssao.glsl, common/shadow_pcf.glsl,
// common/refraction.glsl, volumetric_raymarch.glsl) always flipped the NDC
// y for non-GL backends; the post-effect family used to hardcode the
// non-GL variant, which mirrored every depth-based post effect on the
// Linux/OpenGL 4.3 backend (grading/bloom are uv-space and unaffected).
//
// Conventions:
//   - Projection matrices are built CPU-side (GL-style, NDC y = +1 up) for
//     EVERY backend; sokol-shdc's fixup_clipspace flips only the rasterizer.
//   - v_uv comes from common/fullscreen_vs.glsl: top-left origin on
//     Metal/D3D/WebGPU (v = 0 at the top of the screen) and bottom-left
//     origin on GL (v = 0 at the bottom).
// So NDC -> uv needs the flip on non-GL, and uv -> NDC needs it on non-GL
// too (where v grows downward).
//
// SOKOL_GLSL is predefined by sokol-shdc for the glsl430 output leg (the
// Linux/GL 4.3 runtime; verified against the vendored shdc binary).

vec2 agateNdcToUv(vec2 ndc) {
    vec2 uv = ndc * 0.5 + 0.5;
    #if !SOKOL_GLSL
        uv.y = 1.0 - uv.y;
    #endif
    return uv;
}

vec2 agateUvToNdc(vec2 uv) {
    vec2 ndc = uv * 2.0 - 1.0;
    #if !SOKOL_GLSL
        ndc.y = -ndc.y;
    #endif
    return ndc;
}
