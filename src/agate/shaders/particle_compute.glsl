// Stateful compute particle simulation (v1, agate wave 25).
//
// One index-exclusive invocation per particle slot: each thread touches only
// its own state/draw records, so the in-place update needs no barrier and no
// ping-pong buffer. SoA-of-vec4 layouts mirror particles.zig
// (ComputeParticleState / GpuParticleSlot / ParticleInstanceData) as
// vec4-only structs so std430 offsets are trivially 16-aligned.
//
// Spawns staged by the CPU update are uploaded to `cs_spawn` (prefix
// [0, spawn_count)) and claimed by the ring window
// [spawn_base, spawn_base + spawn_count) mod capacity: the wrapping cursor
// overwrites the oldest slot (ring semantics, same as the .gpu slot ring).
// Deaths (age >= life, or never-written life <= 0) collapse into a
// degenerate zero-size instance the billboard pipeline rasterizes as
// nothing; they are unobservable on CPU (no death events, no sub-emitters).
// Visual endpoints (color/size/rotation) persist in the per-slot state, so
// mid-life parameter changes affect only later spawns -- same rule as the
// .gpu slot ring.
@cs cs
layout(local_size_x = 64) in;

layout(binding = 0) uniform cs_params {
    // x = frame dt (s), y = exponential drag k (1/s, 0 = off), zw unused.
    vec4 dyn;
    // xyz = world-space gravity acceleration, w unused.
    vec4 grav;
    // x = spritesheet columns, y = rows, z = loops over lifetime, w unused.
    vec4 sheet;
    // x = capacity, y = spawn_base (ring cursor at frame start),
    // z = spawn_count this frame, w unused. Float-encoded (exact to 2^24);
    // converted with uint() on use (shdc uniform blocks allow only
    // float/int base types).
    vec4 addr;
};

// Per-particle state (mirror of ComputeParticleState, 6 x vec4 = 96 bytes).
struct CState {
    vec4 pos_age;   // xyz = position, w = age (s)
    vec4 vel_life;  // xyz = velocity, w = lifetime (s)
    vec4 rot_seed;  // x = rotation start (rad), y = angular velocity (rad/s),
                    // z = seed, w unused
    vec4 col0;      // color start (rgba)
    vec4 col1;      // color end (rgba)
    vec4 sizes;     // x = size start, y = size end, zw unused
};
layout(std430, binding = 0) buffer cs_state {
    CState state[];
};

// Staged spawns, prefix [0, spawn_count) (mirror of GpuParticleSlot,
// 5 x vec4 = 80 bytes): a.xyz = spawn position, b.xyz = velocity,
// b.w = lifetime; c/d = color start/end; e.x = size start, e.y = size end,
// e.z = rotation start (rad), e.w = angular velocity (rad/s). The spawn-time
// field (a.w, epoch seconds) is unused by the compute path: age integrates
// in state instead.
struct CSpawn {
    vec4 a;
    vec4 b;
    vec4 c;
    vec4 d;
    vec4 e;
};
layout(std430, binding = 1) buffer cs_spawn {
    CSpawn spawns[];
};

// Baked draw instances (mirror of ParticleInstanceData, 4 x vec4):
// a = pos_size, b = color, c = uv_offset_scale, d = rotation_misc.
// Consumed by the existing CPU billboard pipeline unchanged
// (blend/texture/soft-particle behavior identical); dead slots are all-zero
// (zero-area, rasterizes nothing).
struct COut {
    vec4 a;
    vec4 b;
    vec4 c;
    vec4 d;
};
layout(std430, binding = 2) buffer cs_draw {
    COut draws[];
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint cap = uint(addr.x);
    if (idx >= cap) {
        return;
    }
    float dt = dyn.x;
    uint base = uint(addr.y);
    uint count = uint(addr.z);

    vec3 pos;
    vec3 vel;
    float age;
    float life;
    vec4 col0;
    vec4 col1;
    vec2 sizes;
    float rot;

    // Ring-window claim: rel < count means this slot is (re)spawned this
    // frame from spawns[rel]; the window arithmetic wraps mod capacity.
    uint rel = (idx >= base) ? (idx - base) : (idx + cap - base);
    if (rel < count) {
        CSpawn s = spawns[rel];
        pos = s.a.xyz;
        vel = s.b.xyz;
        life = max(s.b.w, 1.0e-4);
        age = 0.0;
        rot = s.e.z;
        col0 = s.c;
        col1 = s.d;
        sizes = s.e.xy;
        state[idx].pos_age = vec4(pos, age);
        state[idx].vel_life = vec4(vel, life);
        state[idx].rot_seed = vec4(s.e.z, s.e.w, 0.0, 0.0);
        state[idx].col0 = col0;
        state[idx].col1 = col1;
        state[idx].sizes = vec4(sizes, vec2(0.0));
    } else {
        CState st = state[idx];
        life = st.vel_life.w;
        if (life <= 0.0) {
            draws[idx].a = vec4(0.0);
            draws[idx].b = vec4(0.0);
            draws[idx].c = vec4(0.0);
            draws[idx].d = vec4(0.0);
            return;
        }
        age = st.pos_age.w + dt;
        if (age >= life) {
            // Sticky dead: park the age at life so the slot stays culled
            // until the ring respawns it.
            state[idx].pos_age.w = life;
            draws[idx].a = vec4(0.0);
            draws[idx].b = vec4(0.0);
            draws[idx].c = vec4(0.0);
            draws[idx].d = vec4(0.0);
            return;
        }
        vel = st.vel_life.xyz;
        float k = dyn.y;
        if (k > 1.0e-6) {
            vel *= exp(-k * dt);
        }
        vel += grav.xyz * dt;
        pos = st.pos_age.xyz + vel * dt;
        rot = st.rot_seed.x + st.rot_seed.y * age;
        col0 = st.col0;
        col1 = st.col1;
        sizes = st.sizes.xy;
        state[idx].pos_age = vec4(pos, age);
        state[idx].vel_life = vec4(vel, life);
    }

    // Baked visuals, mirrors particles.zig fillRange / sprite math and the
    // .gpu vertex shader: color/size lerp by t, spritesheet cell
    // left-to-right bottom-to-top, exact rotation (no wrapping needed).
    float t = clamp(age / life, 0.0, 1.0);
    vec4 color = mix(col0, col1, t);
    float size = mix(sizes.x, sizes.y, t);
    float cols = max(sheet.x, 1.0);
    float rows = max(sheet.y, 1.0);
    float frames = cols * rows;
    vec4 uv;
    if (frames > 1.5) {
        float frame = mod(floor(t * sheet.z * frames), frames);
        vec2 cell = vec2(mod(frame, cols), floor(frame / cols));
        uv = vec4(cell / vec2(cols, rows), 1.0 / vec2(cols, rows));
    } else {
        uv = vec4(0.0, 0.0, 1.0, 1.0);
    }
    draws[idx].a = vec4(pos, size);
    draws[idx].b = color;
    draws[idx].c = uv;
    draws[idx].d = vec4(rot, 0.0, 0.0, 0.0);
}
@end

@program particle_compute cs
