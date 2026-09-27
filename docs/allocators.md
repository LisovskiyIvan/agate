# Scene allocator domains (`AllocatorConfig`)

Optional per-domain allocator configuration for advanced users
(`scene/core.zig`: `AllocatorConfig`, `Scene.initWithAllocators`,
`Scene.initIntoWithAllocators`).

## Default: byte-identical to today

`Scene.init(allocator)` / `Scene.initInto(allocator)` set every domain to
the single allocator. `initWithAllocators` with all-null domains resolves
identically. The resolved allocators are stored as public fields next to
`allocator` (`render_allocator`, `sim_allocator`, `io_allocator`; each may
equal `allocator`).

## Routing (per consumer, with proof)

Only consumers where every allocation AND every free of the object use one
stored allocator are routed. Everything else stays on `core`.

| Consumer | Domain | Alloc site | Free site | Why safe |
|---|---|---|---|---|
| `UploadQueue` (`uploads`) | `io` | `UploadQueue.init(io, 2)` (core.zig) stores it; slots/paths/decode bytes via stored | `UploadQueue.deinit` via stored (core.zig lifecycle) | Self-contained: struct + runner + slots all use the stored allocator |
| file-I/O runner (`io_runner`) | `io` | `TaskRunner.init(io, 1)` stores it (threads/queue) | `TaskRunner.deinit` via stored | Same self-containment. Note: task BODIES keep their call-site allocator (core, e.g. serialization capture in lifecycle) — only runner internals use `io` |
| particle systems + prepared/build frames | `sim` | `ParticleLayer.create` → `ParticleSystem.init(sim)` stores it (arrays, struct); `buildCapture`/`captureFrame`/`latchSlotFrame` fund `build_frame`/`frame` with sim | `ParticleSystem.deinit` (stored) + `ParticleLayer.deinit(sim)` | No per-system destroy exists; the layer owns systems until deinit. Exception: `stageIntoSlot` keeps `core` — its `out` (`back.particle_draws`) is a core-owned P7 slot list freed in `FrameDraws.deinit` |
| nav meshes + agents | `sim` | `NavLayer.createMesh*` / `createAgent` store/pass sim (`NavMesh`/`NavAgent` free through the stored allocator; waypoint paths from `setDestination`/`findPath` likewise) | `agent.deinit` (stored) + struct destroy and `NavLayer.deinit(sim)` | Layer owns both lists until deinit |
| clustered CPU tile scratch | `render` | `rebuildCpuForSlot(render)` grows `cpu_lights/headers/indices` (view_render, context thread) | `ClusteredGpuCache.deinit(render)` | Both context-thread render path. Split object: `upload`/`retireBuffers` stay `core` — their allocator funds appends into the core-owned `gpu_retire` queue (growth retire + add/remove retire), so they must match `GpuRetireQueue.deinit(core)` |

### Deliberately left on `core` (with reason)

- **Shadow system** (`ShadowSystem.init`): `ShadowPass.prepareInto` funds
  the core-owned P7 slot payloads (`back.shadow` in queue_builder.zig:176)
  with the pass's stored allocator, while `FrameDraws.deinit(core)` frees
  them. Routing the stored allocator to `render` would free slot memory
  across domains. Stays `core` (no per-call allocator params allowed).
- **Soft bodies** (`SoftBodyLayer`): `create` builds cloth solver memory
  AND the scene-registry mesh/material in one intertwined rollback chain
  (`scene.allocator` throughout softbody.zig `create`); `removeForMesh`
  runs inside `Scene.destroyMesh` (registry.zig:120) on BOTH the
  context-thread sync path and the game/off-context path, and `extractAt`
  hands the mesh back to the caller for core-domain retirement. Splitting
  solver memory to `sim` while mesh/material/retire stay `core` cannot be
  proven free of cross-domain frees across all three teardown paths
  (create rollback, `extractAt`, `removeForMesh`), so it stays `core`.
- **`clustered.upload` / `retireBuffers`**: stay `core` (retire-queue
  appends, see table). Only the CPU scratch is `render`.
- **`particles.stageIntoSlot`**: stays `core` (slot-owned `out`, see table).
- **Everything else** (meshes, materials, registries, draws slots,
  physics, trails, decals, UI frame, profiler, snapshots): `core`.
- **`audio`/`physics` are NOT part of the config**: app-owned modules that
  already take their own allocator (`physics` scene integration keeps
  using `core`).

## Invariants

1. **Default**: `init(allocator)` behaves exactly as today (all domains =
   `allocator`; verified by the default-equality test).
2. **Same-domain free**: each domain allocator allocates AND frees its own
   objects; never free across domains (verified by the tracking-allocator
   routing tests, which fail on foreign frees and on leaks).
3. **Thread-safety per domain**:
   - `core`: MUST be thread-safe when workers are used (existing
     API.md rule: prepare/render lazy caches and jobs workers allocate
     from it).
   - `io`: MUST be thread-safe, always (decode workers allocate decode
     buffers with it; `io_runner` threads run tasks; `requestFile`
     appends from game/context threads under a spinlock).
   - `sim`: MUST be thread-safe. Game-side `update`/`buildCapture`
     (particle CPU growth, nav pathfinding) and context-side
     `captureFrame`/`latchSlotFrame` + `deinit` all allocate/free with
     it, and under `--concurrent-build` the game build overlaps the
     prepare latch.
   - `render`: context-thread only (proved: `rebuildCpuForSlot` runs in
     `view_render` on the context thread; `ClusteredGpuCache.deinit`
     runs in context-only `Scene.deinit`). A non-thread-safe allocator
     is acceptable for `render` ONLY as long as every touch stays on the
     context thread; any future worker use would require thread-safety.
4. **Lifetime**: domain allocators must outlive `Scene.deinit` (deinit
   frees domain memory through them; use-after-free otherwise).
5. **Failure**: `UploadQueue.init` / `TaskRunner.init` failures degrade to
   null (`catch null`, sync fallbacks) — a failing `io` allocator still
   initializes the scene and `deinit` succeeds.
