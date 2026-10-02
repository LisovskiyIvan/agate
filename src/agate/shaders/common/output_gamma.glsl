// Babylon's OUTPUT STAGE — the `applyImageProcessing` -> `toGammaSpace` pair
// that ends every Babylon PBR / standard fragment shader.
//
// Babylon encodes the shaded value with a plain power and renders into a
// backbuffer that is NOT sRGB, so the encode is part of the shaded result: it
// happens BEFORE blending and before the MSAA resolve (verified in the shipped
// 9.28.0 bundle's WGSL for the WebGPU bench path:
// `finalColor=applyImageProcessing(finalColor);` with
// `applyImageProcessing` = `toGammaSpaceVec3(rgb)` + `saturateVec3`, and
// `toGammaSpaceVec3` = `pow(color, vec3f(GammaEncodePowerApprox))` where
// `GammaEncodePowerApprox = 1/LinearEncodePowerApprox` = 1/2.2).
//
// agate's own convention is the opposite one: shaders write LINEAR color and
// the sRGB swapchain (`sapp_desc.srgb = true`) makes the hardware apply the
// exact piecewise sRGB curve. The two curves are not the same function:
//
//   linear 0.002  ->  sRGB 0.0258 (7/255)   vs  pow(1/2.2) 0.0593 (15/255)
//   linear 0.5    ->  sRGB 0.7354 (188/255) vs  pow(1/2.2) 0.7297 (186/255)
//
// i.e. tens of percent apart in deep shadow and ~1% apart in the midtones —
// with a high-frequency image that is a per-pixel difference, which is why it
// dominated the bench residual (bench/PROBE.md §10.13: matching the curve
// takes the `mronly` probe from 8.80% to 1.63% of pixels over |Δ|>8).
//
// `Scene.output_gamma` (lane `output_params.x`) selects it for a frame, so an
// app whose backbuffer IS sRGB can keep the hardware path (flag off). Babylon
// itself has no such switch — there the encode is unconditional — which is
// why the flag defaults to ON.
vec4 babylonOutputColor(vec3 rgb, float alpha) {
    rgb = max(rgb, vec3(0.0));
    if (output_params.x > 0.5) {
        rgb = clamp(pow(rgb, vec3(1.0 / 2.2)), 0.0, 1.0);
    }
    return vec4(rgb, alpha);
}
