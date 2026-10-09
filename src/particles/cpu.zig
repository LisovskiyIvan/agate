//! CPU simulation core: emission integration, compaction and instance fill.
//! Split out of `particles.zig` (facade).
//!
//! `updateCpu` takes the system as `anytype` (a `*ParticleSystem` from
//! `system.zig` in practice) so this module never imports `system.zig` or the
//! facade back — same discipline as `profiler/*`. Phase payloads
//! (`IntegrateCtx` / `FillCtx`) are index-exclusive, hence worker-count
//! invariant (pinned by the moved test below). Emission stays serial up front
//! via the owner's `emitOne`; deaths reach `subemitters` directly (sibling
//! import, no cycle).

const std = @import("std");
const math = @import("math");
const jobs = @import("../jobs.zig");

const types = @import("types.zig");
const flow = @import("flow.zig");
const collisions = @import("collisions.zig");
const sampling = @import("sampling.zig");
const subemitters = @import("subemitters.zig");

const Vec3 = math.Vec3;
const Mat4 = math.Mat4;
const Color4 = math.Color4;
const FlowCtx = flow.FlowCtx;
const DeathEvent = subemitters.DeathEvent;
const max_sub_emitter_spawns_per_tick = types.max_sub_emitter_spawns_per_tick;

// Local aliases so the moved body stays byte-identical.
const flowUvForPosition = flow.flowUvForPosition;
const sampleFlowPixels = flow.sampleFlowPixels;
const activeFlowCtx = flow.activeFlowCtx;
const activeCollisionCtx = collisions.activeCollisionCtx;
const collideParticle = collisions.collideParticle;
const fireSubEmitters = subemitters.fireSubEmitters;
const normalizeAngleDeg = sampling.normalizeAngleDeg;
const rotationToRadians = sampling.rotationToRadians;
const spritesheetFrameForAge = sampling.spritesheetFrameForAge;
const spritesheetUvRect = sampling.spritesheetUvRect;
const localToWorld = sampling.localToWorld;
const worldScaleFactor = sampling.worldScaleFactor;

/// CPU simulation step, structured as three phases so the heavy work is
/// data-parallel while the results stay bit-identical to the legacy
/// fused loop for any worker count:
///   A (parallel)  — age + integrate every live slot exactly once,
///                   recording survival in `alive_scratch`;
///   B (serial)    — legacy swap-compaction of dead slots;
///   B2 (serial)   — on-death sub-emitter spawns, only when sub-emitters
///                   are attached AND deaths were recorded (skipped
///                   entirely otherwise: no PRNG draws, no state touched);
///   C (parallel)  — fill render instance data for the compacted range.
/// Each slot is touched only through its own index in A and C, so no
/// locks are needed; emission (PRNG-driven) stays serial up front.
pub fn updateCpu(self: anytype, dt: f32) void {
    if (self.is_emitting and self.emit_rate > 0.0) {
        self.emit_accumulator += dt * self.emit_rate;
        while (self.emit_accumulator >= 1.0 and self.active_count < self.capacity) {
            self.emitOne();
            self.emit_accumulator -= 1.0;
        }
    }

    const emitter_matrix = self.resolveEmitterMatrix();
    const emitter_scale = if (emitter_matrix) |m| worldScaleFactor(m) else 1.0;
    const grav_dt = self.gravity.scale(dt);

    const pool = self.thread_pool orelse jobs.global;

    // Phase A: integrate. Each worker owns its index range exclusively.
    // The flow snapshot is taken once per tick (read-only thereafter).
    const PhaseA = Phase(@TypeOf(self));
    var ictx = PhaseA.IntegrateCtx{ .ps = self, .grav_dt = grav_dt, .dt = dt, .flow = activeFlowCtx(self), .coll = activeCollisionCtx(self) };
    jobs.parallelFor(pool, PhaseA.IntegrateCtx, &ictx, PhaseA.integrateRange, self.active_count);

    // Phase B: compact. Same swap-with-last recycling as the legacy
    // loop (dead slot at i replaced by the last integrated slot). When
    // sub-emitters are attached, deaths are additionally recorded (up to
    // the per-tick bound) for the serial fire pass below; the recording
    // touches no simulation state, so the legacy path is untouched when
    // sub_emitter_count == 0.
    const alive = self.alive_scratch;
    var deaths: [max_sub_emitter_spawns_per_tick]DeathEvent = undefined;
    var death_count: usize = 0;
    const want_deaths = self.sub_emitter_count > 0;
    var i: usize = 0;
    while (i < self.active_count) {
        if (alive[i] == 0) {
            if (want_deaths and death_count < deaths.len) {
                const d = self.particles[i];
                deaths[death_count] = .{ .position = d.position, .velocity = d.velocity, .depth = d.sub_depth };
                death_count += 1;
            }
            self.active_count -= 1;
            if (i < self.active_count) {
                self.particles[i] = self.particles[self.active_count];
                alive[i] = alive[self.active_count];
                continue;
            } else {
                break;
            }
        }
        i += 1;
    }

    // Phase B2: on-death sub-emitters (serial, deterministic death order).
    // Post-compaction so self-emitter appends are safe; pre-fill so
    // same-tick children get valid instance data.
    if (death_count > 0) fireSubEmitters(self, deaths[0..death_count]);
    self.sub_tick += 1;

    // Phase C: render-data fill over the compacted range.
    const PhaseC = Phase(@TypeOf(self));
    var fctx = PhaseC.FillCtx{
        .ps = self,
        .emitter_matrix = emitter_matrix,
        .emitter_scale = emitter_scale,
    };
    jobs.parallelFor(pool, PhaseC.FillCtx, &fctx, PhaseC.fillRange, self.active_count);
}

/// Phase payload namespace, generic over the system pointer type (in practice
/// `*ParticleSystem` from `system.zig`). Zig structs cannot hold `anytype`
/// fields and a generic range function cannot coerce to the concrete
/// `fn (*Ctx, usize, usize)` that `jobs.parallelFor` takes (verified), so the
/// payloads and their range functions are instantiated per caller through
/// this namespace instead of naming the owner type — which would be a
/// leaf-to-owner import cycle. Bodies are identical to the pre-split loop.
fn Phase(comptime System: type) type {
    return struct {
        /// Phase-A payload: per-slot integration. A slot touches only
        /// `particles[i]` and `alive_scratch[i]`, so workers own disjoint
        /// ranges without locks.
        pub const IntegrateCtx = struct {
            ps: System,
            grav_dt: Vec3,
            dt: f32,
            /// Null when the field is disarmed: the loop then executes the legacy
            /// instruction stream bit-for-bit (one predictable branch per slot, no
            /// extra FP ops, no memory traffic).
            flow: ?FlowCtx = null,
            /// Null when collisions are disarmed (mode `.none`, or no
            /// geometry): same bit-for-bit legacy discipline as `flow`
            /// above (one cached branch per slot). Armed: one sphere loop
            /// (cap 8) + one ground check per live particle; `.kill` marks
            /// the slot dead (compacts away like an age death downstream).
            coll: ?collisions.CollisionCtx = null,
        };

        pub fn integrateRange(ctx: *IntegrateCtx, start: usize, end: usize) void {
            const ps = ctx.ps;
            for (start..end) |i| {
                const p = &ps.particles[i];
                p.age += ctx.dt;
                if (p.age >= p.lifetime) {
                    ps.alive_scratch[i] = 0;
                    continue;
                }
                ps.alive_scratch[i] = 1;
                // Physics update (hoisted gravity delta + scaled velocity).
                // In local_space mode gravity/velocity integrate in emitter-local
                // units; the world transform applies at instance-fill time.
                p.velocity = p.velocity.add(ctx.grav_dt);
                // Flow field (CPU-only force): sampled at the pre-move position,
                // applied as acceleration. Skipped entirely when disarmed.
                if (ctx.flow) |f| {
                    const uv = flowUvForPosition(f.space, p.position, f.emitter, f.scale, f.scroll);
                    const push = sampleFlowPixels(f.pixels, f.width, f.height, uv, f.wrap);
                    p.velocity = p.velocity.add(push.scale(f.strength * ctx.dt));
                }
                p.position = p.position.add(p.velocity.scale(ctx.dt));
                // Collisions (CPU-only; see collisions.zig): resolved against
                // the post-move position in stored simulation coordinates.
                // Skipped entirely when disarmed. `.kill` marks the slot dead
                // (the serial compaction below recycles it like an age death,
                // including sub-emitter death events); `.bounce` corrects the
                // position/velocity in place. Pure per slot: no PRNG, no
                // cross-slot state, hence worker-count invariant.
                if (ctx.coll) |c| {
                    const r = collideParticle(p.position, p.velocity, c);
                    if (r.killed) {
                        ps.alive_scratch[i] = 0;
                        continue;
                    }
                    p.position = r.pos;
                    p.velocity = r.vel;
                }
                p.rotation = normalizeAngleDeg(p.rotation + p.angular_velocity * ctx.dt);
            }
        }

        /// Phase-C payload: render instance fill. Reads `particles[i]` (already
        /// compacted), writes `instances[i]` — again index-exclusive.
        pub const FillCtx = struct {
            ps: System,
            emitter_matrix: ?Mat4,
            emitter_scale: f32,
        };

        pub fn fillRange(ctx: *FillCtx, start: usize, end: usize) void {
            const ps = ctx.ps;
            for (start..end) |i| {
                const p = &ps.particles[i];
                const t = p.age / p.lifetime;
                const current_size = p.size + (p.size_end - p.size) * t;
                const current_color = Color4.lerp(p.color, p.color_end, t);
                const frame = spritesheetFrameForAge(
                    p.age,
                    p.lifetime,
                    ps.spritesheet_columns,
                    ps.spritesheet_rows,
                    ps.spritesheet_loops,
                );
                const uv = spritesheetUvRect(frame, ps.spritesheet_columns, ps.spritesheet_rows);

                var render_pos = p.position;
                var render_size = current_size;
                if (ctx.emitter_matrix) |m| {
                    render_pos = localToWorld(m, p.position);
                    render_size = current_size * ctx.emitter_scale;
                }

                ps.instances[i] = .{
                    .pos_size = .{ render_pos.x, render_pos.y, render_pos.z, render_size },
                    .color = current_color.toArray(),
                    .uv_offset_scale = uv,
                    .rotation_misc = .{ rotationToRadians(p.rotation), 0.0, 0.0, 0.0 },
                };
            }
        }
    };
}
