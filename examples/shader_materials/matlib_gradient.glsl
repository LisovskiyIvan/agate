// Material library preset: "matlib_gradient".
// Registered in build.zig's `user_shader_materials` table; construct via:
//   var sm = agate.material_library.gradient("grad", .{}) orelse return null;
//
// Two-stop vertical gradient by world-space height, MULTIPLIED into the
// lit albedo (unlike matlib_sky, lighting still applies — this is a tint
// for ordinary surfaces, not a sky dome).
//
// Params are packed into the generated sm_user_params UB (8x vec4) and
// editable at runtime through ShaderMaterial.setUniform(name, value).

// base: standard

// @param u_grad_top vec4 = 0.90 0.95 1.00 1.0
// @param u_grad_bottom vec4 = 0.15 0.20 0.30 1.0
// @param u_grad_scale float = 0.5
// @param u_grad_offset float = 0.5

// @hook(albedo)
float grad_t = clamp(v_world_pos.y * u_grad_scale + u_grad_offset, 0.0, 1.0);
base.rgb *= mix(u_grad_bottom.rgb, u_grad_top.rgb, grad_t);
