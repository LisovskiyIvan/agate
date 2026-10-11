// Shared forward per-material/object uniform block (wave: fs_params split).
//
// Included by the three PBR forward shaders (pbr/skinned_pbr/instanced_pbr)
// via `the frame_mat include`. Single source so the layout
// stays bit-identical across the family by construction.
//
// Contract:
// - Wrapped by each shader with UB binding 5: free in every forward shader's shared UB pool
//   (pbr: 0 vs_params, 1 fs_view, 2 vs_morph; skinned: +1 vs_skin, 3
//   vs_morph; hook user blocks: 3 fs + 4 vs). UB pool is separate from
//   the texture/sampler pools, so overlapping numbers there are fine.
// - Members are the per-material/object subset of the legacy fs_params
//   block, in the legacy relative order: material factors, UV/channel
//   packing, coat/fabric + anisotropy/transmission/SSS layers, the
//   refraction MATERIAL gate (refraction_factors; the camera pair lives
//   in the view block), and the per-draw probe state (probe_params,
//   probe2_params, probe_box).
// - Uploaded per draw (values differ per material/object); the view
//   block (common/frame_view.glsl, UB 1) is uploaded once per
//   contiguous pipeline/shadow-policy run (see scene/draw.zig
//   ViewUploadCache).
    vec4 base_color_factor;
    vec4 pbr_factors; // x: metallic, y: roughness, z: occlusion_strength, w: ibl_intensity
    vec4 emissive_factor; // xyz: emissive color, w: two-sided-lighting flag
    float alpha_cutoff; // cutout threshold; 0.0 disables the alpha test
    float normal_scale; // normal map xy scale (glTF normalTexture.scale)
    vec4 uv_matrix[5]; // per slot: rotation*scale rows [m00, m01, m10, m11]
    vec4 uv_offset[5]; // per slot: xy offset, z: unlit flag, w unused
    vec4 channel_selectors; // x occlusion, y roughness, z metallic (lane index), w: specular-AA flag
    vec4 clearcoat_factors; // x: intensity (0 = off), y: roughness, z/w: unused
    vec4 clearcoat_color; // rgb: coat tint (white = untinted), w: unused
    vec4 sheen_factors; // x: intensity (0 = off), y: roughness, z/w: unused
    vec4 sheen_color; // rgb: fabric tint (white = untinted), w: unused
    vec4 anisotropy_factors; // x: intensity (0 = isotropic/off), y: tangent-plane rotation (rad), z/w: unused
    vec4 transmission_factors; // x: factor (0 = off), y/z/w: unused
    vec4 transmission_color; // rgb: throughput tint (white = untinted), w: unused
    vec4 sss_factors; // x: strength (0 = off), y/z/w: unused
    vec4 sss_color; // rgb: scatter tint (white = untinted), w: unused
    vec4 clearcoat_uv_matrix;
    vec4 clearcoat_uv_offset;
    vec4 sheen_uv_matrix;
    vec4 sheen_uv_offset;
    vec4 refraction_factors;
    vec4 probe_params;
    vec4 probe2_params;
    vec4 probe_box[2];
