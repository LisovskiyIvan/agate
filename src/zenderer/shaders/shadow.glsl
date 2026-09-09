// Depth-only shadow mapping shader for zenderer
@header const m = @import("math")
@ctype mat4 m.Mat4

@vs vs
@glsl_options fixup_clipspace
layout(binding = 0) uniform vs_params {
    mat4 mvp;
};

in vec3 position;

void main() {
    gl_Position = mvp * vec4(position, 1.0);
}
@end

@vs vs_inst
@glsl_options fixup_clipspace
layout(binding = 0) uniform vs_inst_params {
    mat4 light_view_proj;
};

in vec3 position;
in vec4 inst_mat0;
in vec4 inst_mat1;
in vec4 inst_mat2;
in vec4 inst_mat3;

void main() {
    mat4 model = mat4(inst_mat0, inst_mat1, inst_mat2, inst_mat3);
    gl_Position = light_view_proj * (model * vec4(position, 1.0));
}
@end

@vs vs_skinned
@glsl_options fixup_clipspace
layout(binding = 0) uniform vs_params {
    mat4 mvp;
};
layout(binding = 1) uniform vs_skin {
    mat4 bones[64];
};

in vec3 position;
in vec4 joints;
in vec4 weights;

void main() {
    ivec4 j = ivec4(joints);
    mat4 skin_mat = weights.x * bones[j.x] +
                    weights.y * bones[j.y] +
                    weights.z * bones[j.z] +
                    weights.w * bones[j.w];
    vec4 skinned_pos = skin_mat * vec4(position, 1.0);
    gl_Position = mvp * skinned_pos;
}
@end

@fs fs
void main() {
}
@end

@program shadow vs fs
@program shadow_instanced vs_inst fs
@program shadow_skinned vs_skinned fs
