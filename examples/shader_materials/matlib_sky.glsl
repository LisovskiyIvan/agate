// Material library preset: "matlib_sky".
// Registered in build.zig's `user_shader_materials` table; construct via:
//   var sm = agate.material_library.sky("sky", .{}) orelse return null;
//
// Vertical three-stop sky gradient driven by the normalized world-space
// direction (works on any dome/sphere centered near the origin). The albedo
// hook paints the gradient; the post_lighting hook replaces the lit result
// with the gradient verbatim, so the sky is effectively unlit (no sun bleed
// on the dome) while staying on the standard pipeline (shadows/morphs/alpha
// modes keep working like any ShaderMaterial).
//
// Params are packed into the generated sm_user_params UB (8x vec4) and
// editable at runtime through ShaderMaterial.setUniform(name, value).

// base: standard

// @param u_sky_top vec4 = 0.20 0.45 0.80 1.0
// @param u_sky_horizon vec4 = 0.75 0.85 0.95 1.0
// @param u_sky_bottom vec4 = 0.10 0.10 0.12 1.0
// @param u_sky_exponent float = 1.5

// @hook(albedo)
vec3 sky_dir = normalize(v_world_pos);
float sky_up = clamp(sky_dir.y, 0.0, 1.0);
float sky_down = clamp(-sky_dir.y, 0.0, 1.0);
vec3 sky_col = mix(u_sky_horizon.rgb, u_sky_top.rgb, pow(sky_up, u_sky_exponent));
sky_col = mix(sky_col, u_sky_bottom.rgb, pow(sky_down, u_sky_exponent));
base.rgb = sky_col;
base.a = 1.0;

// @hook(post_lighting)
final_rgb = base.rgb;
