// Linear-radiance output for every scene render/capture shader.
// Scene shaders write LINEAR radiance; the single display transfer runs once
// in the postprocess output stage (postprocess.glsl).
//
// `boundRadiance` is the ONE finite-radiance guard every bound uses (forward
// outputs, postprocess composite/TAA taps, bloom prefilter, sky/particles):
// non-finite lanes (NaN/Inf from 0/0 or overflow upstream) map to 0 via an
// IEEE-754 exponent check + boolean mix — an exact select with no NaN
// arithmetic and no backend-specific isnan builtin — then the value
// clamps to 0..65504 so >1 radiance survives to the composite. Mirrors
// hdr.boundHdr (sanitizeFinite maps invalid to 0): Inf clamps to 0 here,
// never to half max.
vec3 boundRadiance(vec3 c) {
    uvec3 exponent = floatBitsToUint(c) & uvec3(0x7f800000u);
    bvec3 invalid = equal(exponent, uvec3(0x7f800000u));
    vec3 finite = mix(c, vec3(0.0), invalid);
    return clamp(finite, vec3(0.0), vec3(65504.0));
}
// `linearOutputColor` bounds RGB to finite half range (0..65504, non-finite
// to 0) so >1 radiance survives to the composite. It never gamma-encodes
// and never clamps to 1; blending and MSAA resolve therefore see linear
// values.
vec4 linearOutputColor(vec3 rgb, float alpha) {
    return vec4(boundRadiance(rgb), alpha);
}
