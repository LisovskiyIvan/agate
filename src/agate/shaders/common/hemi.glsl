// Hemispheric light irradiance — Babylon.js `HemisphericLight` model.
//
// Babylon treats the hemispheric light as a real light, not a flat ambient
// term: its irradiance interpolates between `groundColor` (surface facing
// away from the light) and `diffuse * intensity` (surface facing the light)
// by `0.5 + 0.5 * dot(N, direction)`:
//
//     irradiance = mix(groundColor, diffuse * intensity, 0.5 + 0.5 * N·L)
//
// Historically agate ignored `direction`, `diffuse` and `intensity` here and
// used `groundColor` as a constant ambient, which made every lit surface
// darker and tinted than the Babylon original (measured on the bench scene:
// the ground plane came out 0.39 linear vs Babylon's 0.60 with the same
// two lights).
//
// Uniform lanes (see `scene/uniforms.zig`, appended last so no existing
// offset shifts):
//   ambient_color.rgb  = groundColor
//   hemi_dir_intensity = (direction.xyz, intensity), direction normalized
//   hemi_diffuse.rgb   = diffuse
vec3 hemiIrradiance(vec3 n) {
    float mix_factor = dot(n, hemi_dir_intensity.xyz) * 0.5 + 0.5;
    return mix(ambient_color.rgb, hemi_diffuse.rgb * hemi_dir_intensity.w, mix_factor);
}
