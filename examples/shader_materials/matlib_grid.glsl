// Material library preset: "matlib_grid".
// Registered in build.zig's `user_shader_materials` table; construct via:
//   var sm = agate.material_library.grid("grid", .{}) orelse return null;
//
// Procedural world-space XZ grid (no textures required): square cells of
// 1/u_grid_scale world units, fixed-width anti-alias-free lines of
// u_grid_width cell units. Fully procedural — fwidth-free on purpose so the
// merged output stays valid under all three shdc slangs
// (glsl430/metal_macos/hlsl5) without derivative-precision surprises.
// Lighting still applies (the grid replaces the albedo, not the lobe).
//
// Params are packed into the generated sm_user_params UB (8x vec4) and
// editable at runtime through ShaderMaterial.setUniform(name, value).

// base: standard

// @param u_grid_line vec4 = 1.0 1.0 1.0 1.0
// @param u_grid_fill vec4 = 0.10 0.10 0.10 1.0
// @param u_grid_scale float = 1.0
// @param u_grid_width float = 0.05

// @hook(albedo)
vec2 grid_f = fract(v_world_pos.xz * u_grid_scale);
vec2 grid_dist = min(grid_f, 1.0 - grid_f);
float grid_m = min(grid_dist.x, grid_dist.y);
float grid_line = 1.0 - smoothstep(0.0, u_grid_width, grid_m);
base.rgb = mix(u_grid_fill.rgb, u_grid_line.rgb, grid_line);
