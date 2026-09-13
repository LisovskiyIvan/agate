// Example agate shader material: "ramp_wave".
// Registered in build.zig's `user_shader_materials` table; override any mesh
// material with it via:
//   const mat = try scene.createShaderMaterial("wave", "ramp_wave");
//   mesh.material = .{ .shader_material = mat };
//
// Demonstrates all standard-template hooks:
//   vertex        — sine displacement along world Y
//   albedo        — two-color vertical gradient ramp
//   post_lighting — cheap fresnel rim glow
//
// Params are packed into the generated sm_user_params UB (8x vec4) and
// editable at runtime through ShaderMaterial.setUniform(name, value).

// base: standard

// @param u_wave_speed float = 4.0
// @param u_wave_height float = 0.12
// @param u_ramp_low vec4 = 0.10 0.30 0.55 1.0
// @param u_ramp_high vec4 = 0.95 0.75 0.35 1.0

// @hook(vertex)
float wave_phase = morphed_pos.x * 2.5 + morphed_pos.z * 1.5;
morphed_pos.y += sin(wave_phase * u_wave_speed) * u_wave_height;
morphed_nrm = normalize(morphed_nrm + vec3(0.0, cos(wave_phase * u_wave_speed) * u_wave_height * 2.5, 0.0));

// @hook(albedo)
float ramp_t = clamp(v_world_pos.y * 0.5 + 0.5, 0.0, 1.0);
vec3 ramp_color = mix(u_ramp_low.rgb, u_ramp_high.rgb, ramp_t * ramp_t);
base.rgb *= ramp_color;

// @hook(post_lighting)
vec3 rim_view_dir = normalize(eye_pos.xyz - v_world_pos);
float rim = pow(1.0 - max(dot(N, rim_view_dir), 0.0), 3.0);
final_rgb += u_ramp_high.rgb * rim * 0.6;
