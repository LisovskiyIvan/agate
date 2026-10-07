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
