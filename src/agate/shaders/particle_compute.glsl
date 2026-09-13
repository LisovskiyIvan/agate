// Compute-simulated particle shader for agate.
//
// Unlike particle.glsl (stateless analytic vertex simulation), this file owns
// a stateful simulation: a compute shader (`@cs cs_simulate`) integrates a
// storage buffer of live particle state each frame, and the render vertex
// stage (`vs_compute`) reads the same buffer per instance (gl_InstanceIndex).
//
// This file is compiled in its own sokol-shdc invocation with slang
// glsl430/metal_macos/hlsl5: GLSL 410 (the engine's default Linux slang)
// cannot express compute stages. Compute therefore requires GL 4.3+ on the
// Linux GL path; availability is gated at runtime via
// sg.queryFeatures().compute (see compute.zig for the full matrix).
//
// The state buffer is written by the compute shader only; the CPU writes the
// spawn records (GpuParticleSlot ring, same as the .gpu path) into a second
// read-only buffer, so no buffer is written by both CPU and GPU.
@header const m = @import("math")
@ctype mat4 m.Mat4

// One live particle, 6 x vec4 (std430). Mirrored by
// particles.ComputeParticleState in particles.zig — keep in sync.
@block compute_common
struct pstate {
    vec4 pos_age;    // xyz = current world position, w = age (s); -1.0 = dead
    vec4 vel_rot;    // xyz = current velocity, w = rotation (rad)
    vec4 life_misc;  // x = lifetime (s), y = angular velocity (rad/s), zw unused
    vec4 col0;       // color start rgba
    vec4 col1;       // color end rgba
    vec4 size;       // x = size start, y = size end, zw unused
};
// CPU-written spawn record (mirror of GpuParticleSlot in particles.zig:
// 5 x vec4). Wrapped in a struct because shdc requires SSBO flexible arrays
// of a named struct type.
struct slot_record {
    vec4 data[5];
};
@end

// Compute shader: one invocation per slot. Three disjoint cases per slot:
//   1. slot was (re)spawned this frame (ring-window test against the spawn
//      cursor): initialize state from the CPU-written spawn record,
//   2. first frame after (re)creation of an unsealed state buffer
//      (init_all flag): stamp the slot as dead (-1 age),
//   3. otherwise integrate the live state one frame forward.
@cs cs_simulate
@include_block compute_common

layout(binding = 0) uniform cs_params {
    // x = dt (s), y = exponential drag k (1/s, 0 = off),
    // z = spawn ring start index, w = spawns this frame.
    vec4 sim;
    // x = num slots, y = init_all flag (0/1), zw unused.
    vec4 misc;
    // xyz = world gravity, w unused.
    vec4 gravity;
};

// CPU-written spawn ring (GpuParticleSlot records), read-only.
layout(binding = 0) readonly buffer cs_slots {
    slot_record slots[];
};
// Live state, read/write.
layout(binding = 1) buffer cs_state {
    pstate st[];
};

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= uint(misc.x)) {
        return;
    }
    // Spawn ring window: k = (idx - spawn_start) mod num_slots < spawn_count.
    uint rel = (idx + uint(misc.x) - uint(sim.z)) % uint(misc.x);
    if (uint(sim.w) > 0u && rel < uint(sim.w)) {
        vec4 s0 = slots[idx].data[0]; // xyz spawn pos, w spawn time (unused here)
        vec4 s1 = slots[idx].data[1]; // xyz velocity, w lifetime
        vec4 s2 = slots[idx].data[2]; // color start
        vec4 s3 = slots[idx].data[3]; // color end
        vec4 s4 = slots[idx].data[4]; // x/y size start/end, z rotation0, w angvel
        st[idx].pos_age = vec4(s0.xyz, 0.0);
        st[idx].vel_rot = vec4(s1.xyz, s4.z);
        st[idx].life_misc = vec4(max(s1.w, 1.0e-4), s4.w, 0.0, 0.0);
        st[idx].col0 = s2;
        st[idx].col1 = s3;
        st[idx].size = vec4(s4.xy, 0.0, 0.0);
    } else if (misc.y > 0.5) {
        // Uninitialized backing memory after buffer creation: dead slot.
        st[idx].pos_age = vec4(0.0, 0.0, 0.0, -1.0);
    } else {
        pstate p = st[idx];
        if (p.pos_age.w >= 0.0) {
            float dt = sim.x;
            float k = sim.y;
            if (k > 1.0e-6) {
                // Exact per-step solution of dv/dt = g - k*v (same closed
                // form as the analytic .gpu path, applied over dt):
                //   p' = p + v*s + g*s2,  v' = v*e + g*s
                //   e = exp(-k*dt), s = (1 - e)/k, s2 = (dt - s)/k
                float e = exp(-k * dt);
                float s = (1.0 - e) / k;
                float s2 = (dt - s) / k;
                p.pos_age.xyz += p.vel_rot.xyz * s + gravity.xyz * s2;
                p.vel_rot.xyz = p.vel_rot.xyz * e + gravity.xyz * s;
            } else {
                // Drag off: semi-implicit Euler, v += g*dt; p += v'*dt.
                p.vel_rot.xyz += gravity.xyz * dt;
                p.pos_age.xyz += p.vel_rot.xyz * dt;
            }
            p.pos_age.w += dt;
            // Rotation: constant angular velocity.
            p.vel_rot.w += p.life_misc.y * dt;
            if (p.pos_age.w >= p.life_misc.x) {
                // Dead marker; the render stage culls on negative age.
                p.pos_age.w = -1.0;
            }
            st[idx] = p;
        }
    }
}
@end
@program particle_compute_sim cs_simulate

// Render stage: billboard quads, one instance per storage-buffer slot.
@vs vs_compute
@include_block compute_common
@glsl_options fixup_clipspace
layout(binding = 0) uniform vs_params {
    mat4 view_proj;
    vec4 camera_right;
    vec4 camera_up;
    // x = spritesheet columns, y = rows, z = loops over lifetime, w unused.
    vec4 sprite;
};

layout(binding = 0) readonly buffer vs_state {
    pstate st[];
};

in vec2 position;
in vec2 texcoord0;

out vec2 v_uv;
out vec4 v_color;

void main() {
    pstate p = st[gl_InstanceIndex];
    float age = p.pos_age.w;
    // Dead/unborn slots collapse into a zero-area off-screen triangle
    // (same culling contract as the analytic path in particle.glsl).
    if (age < 0.0) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        v_uv = vec2(0.0);
        v_color = vec4(0.0);
        return;
    }
    float t = clamp(age / max(p.life_misc.x, 1.0e-4), 0.0, 1.0);
    v_color = mix(p.col0, p.col1, t);
    float size = mix(p.size.x, p.size.y, t);

    // Spritesheet frame: mirrors particles.zig spritesheetFrameForAge /
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

    // Billboard rotation.
    float c = cos(p.vel_rot.w);
    float sn = sin(p.vel_rot.w);
    vec2 rotated = vec2(c * position.x - sn * position.y, sn * position.x + c * position.y);
    vec3 world_pos = p.pos_age.xyz + (camera_right.xyz * rotated.x + camera_up.xyz * rotated.y) * size;
    gl_Position = view_proj * vec4(world_pos, 1.0);
}
@end

@fs fs
// NOTE: view bindslots are unique across stages within a program (shdc
// validates this): the texture takes slot 1 because the vs storage buffer
// owns slot 0. Samplers live in their own namespace.
layout(binding = 1) uniform texture2D particle_tex;
layout(binding = 0) uniform sampler smp;

in vec2 v_uv;
in vec4 v_color;

out vec4 frag_color;

void main() {
    vec4 tex = texture(sampler2D(particle_tex, smp), v_uv);
    frag_color = tex * v_color;
}
@end

@program particle_compute vs_compute fs
