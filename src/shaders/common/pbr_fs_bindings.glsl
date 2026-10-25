// Shared forward fragment texture/sampler view bindings (PBR family).
//
// Included by the three PBR forward fragment shaders (pbr, skinned_pbr,
// instanced_pbr) via `the pbr_fs_bindings include`. Single
// source so the view/sampler layout stays bit-identical across the family
// by construction.
//
// Contract:
// - Declarations AND binding numbers are load-bearing: every view/sampler
//   slot must match the resources the draw binds 1:1 (see
//   scene/draw.zig). The slot pool is shared across stages: rigid and
//   skinned vs use texture 9 for the morph strip (instanced vs uses no
//   textures), fs uses 0..8, 10, 11, 15..21, and the clustered storage
//   buffers (common/cluster.glsl) use 12..14.
// - Slot comments below name the per-variant vs occupants explicitly, so a
//   future vs occupant change updates the accounting in ONE place.
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
@image_sample_type spot_shadow_depth_tex unfilterable_float
layout(binding = 20) uniform texture2D spot_shadow_depth_tex; // same atlas view as spot_shadow_tex, raw-depth reads for spot PCSS
// Point shadow atlas (passes/shadow_pass.zig POINT_SHADOW_* layout).
// Binding 10: fs uses 0..8 — next free slot in the shared pool (rigid and
// skinned vs use texture 9 for morph, instanced vs uses no textures); same
// slot across the PBR family. Sampled through the shared shadow_smp
// compare sampler.
layout(binding = 10) uniform texture2D point_shadow_tex;
@image_sample_type point_shadow_depth_tex unfilterable_float
layout(binding = 21) uniform texture2D point_shadow_depth_tex; // same atlas view as point_shadow_tex, raw-depth reads for point PCSS (binding 21: next free view slot — fs uses 0..8, 10, 11, 15..20; rigid/skinned vs uses 9, storage uses 12..14)
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
// slot in the shared pool (fs uses 0..8 and 10; rigid/skinned vs uses 9
// for morph, instanced vs uses no textures), binding 6 the next free
// sampler slot (fs uses 0..3 and 5; rigid/skinned vs uses 4).
// The draw always binds something valid here (probe cube or default cube);
// the shader only samples it when probe_params.x > 0.5. Instanced draws
// always upload zeroed params (no-probe path); the slots must still exist
// for layout parity with the regular PBR family.
layout(binding = 11) uniform textureCube probe_tex;
layout(binding = 19) uniform textureCube probe2_tex;
layout(binding = 6) uniform sampler probe_smp;
layout(binding = 7) uniform sampler brdf_lut_smp;
// PBR layers v1: coat/fabric/anisotropy masks sampled through data_smp (each texture
// keeps its own filter/wrap, like the other data slots — no new sampler).
// Unset slots bind the default white texture: mask 1 / tint 1 = identity,
// so the scalar path stays bit-identical. Bindings 15/16/22 are free view
// slots in the shared pool (fs uses 0..8, 10, 11, 15..21; rigid/skinned vs uses 9
// for morph, instanced vs uses no textures; 12..14 are storage buffers, a
// separate array).
layout(binding = 15) uniform texture2D clearcoat_tex;
layout(binding = 16) uniform texture2D sheen_tex;
layout(binding = 22) uniform texture2D anisotropy_tex;
layout(binding = 18) uniform texture2D refraction_tex;
layout(binding = 8) uniform sampler refraction_smp;
// Clustered forward lights (wave 30, v1): storage buffers for the tile
// walk. Bindings 12..14 are free view slots in the shared pool of every
// forward shader (this family uses fs 0..8, 10, 11; rigid/skinned vs uses
// 9 for morph, instanced vs uses no textures).
// Layouts mirror scene/clustered_lights.zig (ClusterLightGpu = 4x vec4,
// ClusterTileGpu = uvec2, indices = u32); every buffer block holds exactly
// one flexible array of a struct (sokol-shdc requirement). The draw always
// binds something valid here (real tile buffers when a GPU upload landed,
// the shared 16-byte dummy otherwise); the loop below only reads them when
// clustered_params gates it on.
