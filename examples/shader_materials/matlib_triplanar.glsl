// Material library preset: "matlib_triplanar".
// Registered in build.zig's `user_shader_materials` table; construct via:
//   var sm = agate.material_library.triPlanar("tri", .{}) orelse return null;
//
// Procedural three-axis blend (no textures required): each world axis gets
// its own tint (u_tri_x/u_tri_y/u_tri_z), weighted by pow(abs(N), sharp).
// The material's primary texture slot (ShaderMaterial.texture, sampled here
// with world-space planar UVs) optionally modulates the blend — with no
// texture bound the white fallback keeps the flat axis colors, so the
// texture is strictly opt-in through the existing slot. The v_uv-sampled
// texel already folded into `base` is replaced; v_color/diffuse_color are
// re-applied so the tint contract matches every other material.
//
// Params are packed into the generated sm_user_params UB (8x vec4) and
// editable at runtime through ShaderMaterial.setUniform(name, value).

// base: standard

// @param u_tri_x vec4 = 1.0 0.35 0.35 1.0
// @param u_tri_y vec4 = 0.35 1.0 0.35 1.0
// @param u_tri_z vec4 = 0.35 0.35 1.0 1.0
// @param u_tri_scale float = 0.5
// @param u_tri_sharp float = 2.0

// @hook(albedo)
vec3 tri_w = pow(abs(N), vec3(u_tri_sharp));
tri_w /= max(tri_w.x + tri_w.y + tri_w.z, 0.0001);
vec3 tri_tx = texture(sampler2D(diffuse_tex, smp), v_world_pos.yz * u_tri_scale).rgb;
vec3 tri_ty = texture(sampler2D(diffuse_tex, smp), v_world_pos.xz * u_tri_scale).rgb;
vec3 tri_tz = texture(sampler2D(diffuse_tex, smp), v_world_pos.xy * u_tri_scale).rgb;
vec3 tri_col = tri_tx * u_tri_x.rgb * tri_w.x + tri_ty * u_tri_y.rgb * tri_w.y + tri_tz * u_tri_z.rgb * tri_w.z;
base.rgb = tri_col * v_color.rgb * diffuse_color.rgb;
