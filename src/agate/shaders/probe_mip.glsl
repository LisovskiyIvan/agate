// Reflection-probe mip prefilter (box approximation, wave 25).
// Renders one cube-face mip level from the previous level: every destination
// texel samples the source mip with LINEAR filtering at exact 2:1 texel
// centers, which is an exact 2x2 box average per texel. This is NOT a GGX
// importance-sampled prefilter (no per-roughness lobe shaping); the PBR
// shaders then pick textureLod(probe, R, roughness * max_lod) as a
// documented roughness approximation.
// Orientation note: sampling is by world-space direction (samplerCube maps
// the direction to the right face automatically), so the per-face UV->dir
// table below only needs to cover its own face's directions — in-plane
// rotation vs the capture cameras is irrelevant to the blur. The table must
// agree with scene/probe_layer.zig (faceDir order +X,-X,+Y,-Y,+Z,-Z).
@vs vs
@glsl_options fixup_clipspace
in vec2 position;
in vec2 texcoord0;

out vec2 v_uv;

void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    v_uv = texcoord0;
}
@end

@fs fs
layout(binding = 0) uniform fs_params {
    vec4 params; // x: face 0..5 (+X,-X,+Y,-Y,+Z,-Z), y: source mip lod, zw: unused
};

layout(binding = 0) uniform textureCube src_tex;
layout(binding = 0) uniform sampler smp;

in vec2 v_uv;
out vec4 frag_color;

vec3 probeFaceDir(float face, vec2 uv) {
    vec2 p = uv * 2.0 - 1.0;
    if (face < 0.5) return normalize(vec3(1.0, -p.y, -p.x)); // +X
    if (face < 1.5) return normalize(vec3(-1.0, -p.y, p.x)); // -X
    if (face < 2.5) return normalize(vec3(p.x, 1.0, p.y)); // +Y
    if (face < 3.5) return normalize(vec3(p.x, -1.0, -p.y)); // -Y
    if (face < 4.5) return normalize(vec3(p.x, -p.y, 1.0)); // +Z
    return normalize(vec3(-p.x, -p.y, -1.0)); // -Z
}

void main() {
    vec3 d = probeFaceDir(params.x, v_uv);
    vec3 c = textureLod(samplerCube(src_tex, smp), d, params.y).rgb;
    frag_color = vec4(c, 1.0);
}
@end

@program probe_mip vs fs
