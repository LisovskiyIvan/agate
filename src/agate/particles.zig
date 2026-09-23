//! Facade for the particle modules. The `particles.zig` particle system was
//! split into focused leaves under `particles/` following the repo pattern
//! (free functions + thin forwarders; Zig 0.16 has no usingnamespace; see
//! `profiler.zig`, `scene/render_queue.zig`, `texture.zig`, `camera.zig`):
//!
//! - `particles/types.zig` — shared vocabulary: modes (`SimulationMode` with
//!   the `.cpu`/`.gpu`/`.compute` feature-matrix contracts), errors, slot and
//!   state layouts (`GpuParticleSlot`, `ComputeParticleState`), the pure
//!   trajectory mirror (`slotAge`, `analyticPosition`), `Particle`,
//!   `ParticleInstanceData` and the sub-emitter bounds. Leaf: no sibling
//!   imports.
//! - `particles/sampling.zig` — spawn sampler (`sampleSpawn`, shared verbatim
//!   by all three modes) plus the pure visual helpers (spritesheet,
//!   rotation, `localToWorld`). Imports `math` only.
//! - `particles/system.zig` — owns the `ParticleSystem` type: fields, the
//!   trivial lifecycle (`init`/`deinit`/`start`/`stop`/`reset`), the small
//!   accessors and thin forwarders into the siblings below, so every call
//!   site keeps working unchanged. Also hosts the `SubEmitter` rule type
//!   (its `system` field pins it to the owner) and the shared test helpers.
//! - `particles/cpu.zig` — `updateCpu` (three-phase parallel integrate,
//!   serial compact, parallel instance fill). Imports `flow` + `sampling` +
//!   `subemitters`.
//! - `particles/gpu.zig` — the stateless `.gpu` path (slot-ring staging and
//!   upload). Imports `sampling` + `flow`.
//! - `particles/flow.zig` — flow-field texture arming and CPU sampling.
//!   Imports `types` only.
//! - `particles/collisions.zig` — CPU-only particle-vs-geometry collisions
//!   (static sphere colliders + ground plane, `.kill`/`.bounce` response).
//!   Imports `types` only.
//! - `particles/subemitters.zig` — on-death child-spawn pass (`emitChild`,
//!   `fireSubEmitters`). Imports `sampling` + `gpu` + `compute_mode`.
//! - `particles/compute_mode.zig` — the stateful `.compute` path (staging,
//!   flush, dispatch, retire). Imports `sampling` + `flow`.
//!
//! Everything that was public before the split is re-exported here
//! unchanged; consumers (`scene.zig`, `scene/particle_layer.zig`,
//! `passes/particle_pass.zig`, `root.zig`) see the same API as when
//! everything lived in this file.
//!
//! Documented anti-cycle rule: leaves must never import this facade —
//! importing it back would make the re-exports depend on their own
//! consumers. The method bodies take the system as `anytype` (same
//! discipline as `profiler/*` taking a generic profiler), so library code has
//! no leaf-to-owner edge at all; moved tests reach `system.zig` helpers
//! through block-scoped imports that exist only in test builds.
//! Cross-leaf helpers (`sampling.sampleSpawn`, `gpu.pushGpuSlot`,
//! `compute_mode.pushComputeSpawn`, `flow.activeFlowCtx`) are `pub` in their
//! home module for the sibling that needs them but are deliberately NOT
//! re-exported here, so the public surface is identical to the pre-split
//! file.
const types = @import("particles/types.zig");
const sampling = @import("particles/sampling.zig");
const system = @import("particles/system.zig");
const flow = @import("particles/flow.zig");
const collisions = @import("particles/collisions.zig");

// Shared vocabulary (lives in particles/types.zig).
pub const ParticleBlendMode = types.ParticleBlendMode;
pub const FlowSpace = types.FlowSpace;
pub const FlowWrap = types.FlowWrap;
pub const SimulationMode = types.SimulationMode;
pub const UpdateError = types.UpdateError;
pub const ComputeModeError = types.ComputeModeError;
pub const ComputeParticleState = types.ComputeParticleState;
pub const compute_workgroup_size = types.compute_workgroup_size;
pub const GpuParticleSlot = types.GpuParticleSlot;
pub const SlotAge = types.SlotAge;
pub const slotAge = types.slotAge;
pub const DragSpans = types.DragSpans;
pub const analyticDragSpans = types.analyticDragSpans;
pub const analyticPosition = types.analyticPosition;
pub const Particle = types.Particle;
pub const ParticleInstanceData = types.ParticleInstanceData;
pub const max_sub_emitters = types.max_sub_emitters;
pub const max_sub_emitter_depth = types.max_sub_emitter_depth;
pub const max_sub_emitter_spawns_per_tick = types.max_sub_emitter_spawns_per_tick;

// Spawn sampler and pure visual helpers (live in particles/sampling.zig).
pub const normalizeAngleDeg = sampling.normalizeAngleDeg;
pub const rotationToRadians = sampling.rotationToRadians;
pub const spritesheetFrameCount = sampling.spritesheetFrameCount;
pub const spritesheetFrameForAge = sampling.spritesheetFrameForAge;
pub const spritesheetUvRect = sampling.spritesheetUvRect;
pub const localToWorld = sampling.localToWorld;
pub const worldScaleFactor = sampling.worldScaleFactor;

// Particle system owner (lives in particles/system.zig).
pub const SubEmitterTrigger = system.SubEmitterTrigger;
pub const SubEmitter = system.SubEmitter;
pub const ParticleSystem = system.ParticleSystem;

// Flow-field sampling (lives in particles/flow.zig).
pub const flowUvForPosition = flow.flowUvForPosition;
pub const sampleFlowPixels = flow.sampleFlowPixels;

// CPU-only collisions (live in particles/collisions.zig).
pub const CollisionMode = collisions.CollisionMode;
pub const CollisionError = collisions.CollisionError;
pub const ParticleSphereCollider = collisions.ParticleSphereCollider;
pub const max_sphere_colliders = collisions.max_sphere_colliders;
pub const ParticleBoxCollider = collisions.ParticleBoxCollider;
pub const max_box_colliders = collisions.max_box_colliders;
pub const ParticlePlaneCollider = collisions.ParticlePlaneCollider;
pub const max_plane_colliders = collisions.max_plane_colliders;
