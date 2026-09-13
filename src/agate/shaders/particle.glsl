// Instanced Camera-Facing Billboard Particle Shader for agate
@header const m = @import("math")
@ctype mat4 m.Mat4

@vs vs
@glsl_options fixup_clipspace
layout(binding = 0) uniform vs_params {
    mat4 view_proj;
    vec4 camera_right;
    vec4 camera_up;
};

in vec2 position;
in vec2 texcoord0;
in vec4 inst_pos_size;
in vec4 inst_color;
// xy = UV offset of the spritesheet cell, zw = UV scale (1/columns, 1/rows).
in vec4 inst_uv_rect;
// x = billboard rotation in radians (counter-clockwise in billboard plane).
in vec4 inst_rotation;

out vec2 v_uv;
out vec4 v_color;

void main() {
    float c = cos(inst_rotation.x);
    float s = sin(inst_rotation.x);
    vec2 rotated = vec2(c * position.x - s * position.y, s * position.x + c * position.y);
    vec3 world_pos = inst_pos_size.xyz +
        (camera_right.xyz * rotated.x + camera_up.xyz * rotated.y) * inst_pos_size.w;
    gl_Position = view_proj * vec4(world_pos, 1.0);
    v_uv = inst_uv_rect.xy + texcoord0 * inst_uv_rect.zw;
    v_color = inst_color;
}
@end

@fs fs
layout(binding = 0) uniform texture2D particle_tex;
layout(binding = 0) uniform sampler smp;

in vec2 v_uv;
in vec4 v_color;

out vec4 frag_color;

void main() {
    vec4 tex = texture(sampler2D(particle_tex, smp), v_uv);
    frag_color = tex * v_color;
}
@end

@program particle vs fs

// Stateless GPU particle simulation: each instance is a fixed spawn-slot
// (see GpuParticleSlot in particles.zig); position, fade, size, rotation and
// the spritesheet frame are evaluated analytically from the slot age
// t = (time - spawn_time) / lifetime. No per-frame state is read back or
// integrated, so dead slots (t < 0 or t >= 1) collapse into a degenerate
// off-screen triangle.
@vs vs_gpu
@glsl_options fixup_clipspace
layout(binding = 0) uniform vs_params {
    mat4 view_proj;
    vec4 camera_right;
    vec4 camera_up;
};
layout(binding = 1) uniform gpu_params {
    // x = seconds since the system's epoch clock, y = exponential drag k
    // (1/s, 0 = off), zw unused.
    vec4 time_drag;
    // xyz = world-space gravity acceleration, w unused.
    vec4 gravity;
    // x = spritesheet columns, y = rows, z = loops over lifetime, w unused.
    vec4 sprite;
};

in vec2 position;
in vec2 texcoord0;
// Mirror of GpuParticleSlot (particles.zig), five FLOAT4 attributes:
in vec4 gpu_slot0;  // xyz = spawn position, w = spawn time (epoch offset)
in vec4 gpu_slot1;  // xyz = initial velocity, w = lifetime (s)
in vec4 gpu_slot2;  // color start rgba
in vec4 gpu_slot3;  // color end rgba
in vec4 gpu_slot4;  // x = size start, y = size end, z = rotation start (rad), w = angular velocity (rad/s)

out vec2 v_uv;
out vec4 v_color;

void main() {
    // Mirrors particles.zig slotAge (lifetime clamp included).
    float age = time_drag.x - gpu_slot0.w;
    float t = age / max(gpu_slot1.w, 1.0e-4);
    if (t < 0.0 || t >= 1.0) {
        // Dead or unborn slot: constant off-screen vertex positions build a
        // zero-area triangle that never rasterizes (NaN-safe: no mix() with
        // garbage slot data happens on this path).
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        v_uv = vec2(0.0);
        v_color = vec4(0.0);
        return;
    }

    // Analytic integration, mirrors particles.zig analyticDragSpans/
    // analyticPosition: p = p0 + v0*s + g*s2 with
    //   drag off:  s = age, s2 = age^2 / 2
    //   drag k>0:  s = (1 - exp(-k*age)) / k, s2 = (age - s) / k
    float s;
    float s2;
    float k = time_drag.y;
    if (k > 1.0e-6) {
        s = (1.0 - exp(-k * age)) / k;
        s2 = (age - s) / k;
    } else {
        s = age;
        s2 = 0.5 * age * age;
    }
    vec3 center = gpu_slot0.xyz + gpu_slot1.xyz * s + gravity.xyz * s2;

    v_color = mix(gpu_slot2, gpu_slot3, t);
    float size = mix(gpu_slot4.x, gpu_slot4.y, t);

    // Spritesheet frame, mirrors particles.zig spritesheetFrameForAge /
    // spritesheetUvRect (frames left-to-right, bottom-to-top).
    float cols = max(sprite.x, 1.0);
    float rows = max(sprite.y, 1.0);
    float frames = cols * rows;
    if (frames > 1.5) {
        float frame = mod(floor(t * sprite.z * frames), frames);
        vec2 cell = vec2(mod(frame, cols), floor(frame / cols));
        v_uv = (texcoord0 + cell) / vec2(cols, rows);
    } else {
        v_uv = texcoord0;
    }

    // Billboard rotation: constant angular velocity integrates exactly to
    // rot = rot0 + w*age (trig is periodic, so no angle wrapping needed).
    float rot = gpu_slot4.z + gpu_slot4.w * age;
    float c = cos(rot);
    float sn = sin(rot);
    vec2 rotated = vec2(c * position.x - sn * position.y, sn * position.x + c * position.y);
    vec3 world_pos = center + (camera_right.xyz * rotated.x + camera_up.xyz * rotated.y) * size;
    gl_Position = view_proj * vec4(world_pos, 1.0);
}
@end

@program particle_gpu vs_gpu fs
