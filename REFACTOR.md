# Threading refactor: status & roadmap

Status as of 2026-09-15. Legend: [x] done, [~] partial, [ ] planned.

## Status at a glance

| Stage | Scope | Status |
|---|---|---|
| 1. Data-parallel CPU systems | job pool, particles, culling | [x] done |
| 2. Async assets | TaskRunner, UploadQueue, glTF async textures | [x] done |
| 3. Simulation/render decoupling | threads, non-blocking render, full payload | [x] done |

## What exists (as built)

### Primitives

- `jobs.Pool` — fork-join data-parallel work: calling thread + N workers
  forage chunks off an atomic cursor. Workers park on a pthread condvar
  (Zig 0.16 std.Thread ships no public condvar; `std.c` is wrapped
  directly — the engine links libc for sokol anyway). Single-producer
  (`forkJoin`), deterministic output order independent of scheduling.
- `jobs.SpscRing(T, N)` — lock-free single-producer/single-consumer ring,
  drop-newest when full, monotonic indices, power-of-two capacity.
- `jobs.TaskRunner` — fire-and-forget tasks on dedicated threads,
  FIFO, join-on-shutdown drains the queue; allocation failure runs the
  task inline rather than dropping it.
- `jobs.Mutex` — blocking pthread mutex for coarse phase ownership.
- `handoff.Handoff(T, slots)` — lock-free latest-wins mailbox:
  `claim`/`publish`/`takeLatest`, global publish sequence, stale frames
  never resurface, drops allowed. Single-publisher by contract.
- `assets.UploadQueue` + `PendingTexture` — decode off-thread
  (`Texture.decodeFile/decodeMemory` are documented GPU-free and
  thread-safe), upload on the sg-context thread, optional live
  `target: *?Texture` slot patching (materials pick textures up with no
  re-wiring).

### Stage 1 — data-parallel systems [x]

- `Scene.update(dt)` — single game-side entry point, canonical order:
  camera -> lights -> physics -> animations -> particles -> nav agents ->
  trails -> decals. Trails and nav agents folded in after verifying they
  receive the same clamped real-seconds dt through the umbrella that the
  app used to pass them.
- CPU particle integration is three-phase: integrate (parallel) ->
  legacy swap-compaction (serial) -> instance fill (parallel).
  Bit-identical output for any worker count (pinned by a 16k-particle
  byte-equality test).
- Frustum culling: `cullNonInstancedMesh` (pure per-mesh test) +
  chunked parallel pass; chunks partition the mesh list in fixed order
  and merge in chunk order, so queues and stats match the serial loop
  exactly (3000-mesh equivalence test). Active above
  `FrameCullContext.parallel_min_meshes` (default 1024). Instance-bearing
  meshes stay serial (they own sg buffer uploads).
- Update-side `sg.*` calls eliminated: particles, trails, and mesh
  morphs stage CPU data and set dirty flags; `Scene.flushPendingGpuUploads()`
  (render start) performs the uploads. The update phase is free of sg.*
  calls.

### Stage 2 — async assets [x]

- `Scene.updateLights(dt)` packs point/spot lights (including the
  incumbency-hysteresis fade simulation) during the update phase and
  publishes through `Scene.light_handoff`
  (`handoff.Handoff(FramePack, 2)`); render takes the newest pack.
  First state group fully thread-ready (pinned by a fade-in test).
- `LoadOptions.async_textures` — glTF images decode on the UploadQueue
  instead of blocking `appendGlb`; materials start with null slots
  (default-white fallback) and patch in via `AsyncTexCtx` targets.
  Measured (Debug): DamagedHelmet critical path 169 -> 10 ms, Fox
  41 -> 11 ms; one drain observed patching 7 textures.
- Sync mode (default elsewhere) is unchanged.
- Serialization save off-thread: [ ] blocked on a scene quiesce/snapshot
  story (state holds pointers; capture-then-marshal races the main
  thread).

### Stage 3 — simulation/render decoupling [~]

Shipped (engine demo & sandbox, threaded by default; `--no-threads` falls
back to inline simulation):

- Game thread loop: `simulate(dt)` = input drain (SpscRing) + demo state
  + `Scene.update(dt)`, paced ~1 kHz on its own sokol-time clock.
- sapp thread: windowing + `Scene.render()` of the newest state.
- Phase ownership: `jobs.Mutex` held for the whole update phase and the
  whole render phase — the two never overlap. ESC on the game thread sets
  a quit flag the sapp thread observes (sapp stays single-threaded).
- **Sandbox UI & threading**: raw `sapp.Event` input events cross into the
  game thread via `jobs.SpscRing(sapp.Event, 512)`. Event consumption and
  UI event dispatch (`sandbox_ui.handleEvent`) run on the game thread
  under phase ownership. Because UI callbacks execute on the game thread,
  they mutate `scene` and `sb_scene` state directly with zero mutation-queue
  boilerplate. `sandbox_ui.renderUI` runs on the sapp thread under the
  phase mutex, safely generating vertex data for `Scene.render()`.
- Update phase is free of `sg.*`; all GPU pushes happen at render start
  (`flushPendingGpuUploads`) or during draws.
- Thread-safety verified by: two-thread lost-update mutex test
  (20000/20000), 1500-frame threaded smoke, handoff threaded stress test,
  SpscRing cross-thread order test.

Known limitations of the shipped split:

- **Coarse payload.** Phase ownership means an update spike delays render
  for its duration. Strictly non-blocking render needs the full per-item
  payload (below).

## What remains (TODO, in priority order)

1. [x] **Full per-item frame payload / strictly non-blocking render.**
   Converted all read-during-render state into published records and snapshots:
   - per drawn item: model Mat4, compact `MaterialDrawRecord` (~120 B: factors,
     uv transforms, cutoff, texture views and samplers), `receive_shadows`,
   - skeleton skin palettes (`[2][MAX_BONES]Mat4` double-buffered with atomic
     release/acquire slots),
   - morph weights and texture view packaged into `RenderMeshItem`,
   - `CameraSnapshot` and `SceneFrameSnapshot` capturing derived cameras,
     sun direction/color/intensity, CSM cascades, light packs, skybox,
     clear color, MSAA, outline, SSAO, and post-process options.
   `Scene.prepareFrame()` consumes the snapshot under the brief handoff lock (<0.05 ms),
   and `Scene.render()` executes all GPU passes completely outside `phase_mutex`.
2. [x] **Sandbox joins the split.** Raw sapp event ring buffer feeding
   game-thread UI event handling, state mutations directly on the game
   thread, `flushPendingGpuUploads` for morph targets, `threaded = true`
   by default.
3. [x] **Serialization save off-thread.** Implemented `AsyncSaveTask` and `AsyncLoadTask`
   with snapshot semantics via `jobs.TaskRunner`. `Scene.saveStateFileAsync`
   captures `SceneState` snapshot (<0.1 ms) on the game thread and dispatches
   binary serialization and disk I/O to background worker threads without stalling
   simulation or render loops. `Scene.loadStateFileAsync` reads and deserializes
   off-thread, ready for fast in-place `restore()`. Fully integrated into
   `sandbox` showcase with non-blocking UI status updates and automated testing.
4. [ ] **Granularity refinements** (optional, as scenes grow):
   lower `parallel_min_meshes` (1024) or make it adaptive; parallel
   shadow-pass binning; parallel instanced-path transform staging.
5. [ ] **GPU-side follow-ups** (optional): async compute is available on
   Metal/D3D12/WebGPU but sokol does not expose queues — revisit only if
   a compute-heavy workload demands it.

## Hard constraints (unchanged)

- **sokol_gfx is not thread-safe.** Every `sg.*` call happens on the
  context thread (today: main/sapp thread). `SOKOL_THREAD_SAFETY` is a
  mutex, not parallelism.
- **sapp callbacks are main-thread.** Input is produced there and crosses
  into the game side via `jobs.SpscRing`.
- The swapchain is acquired through `sglue` inside the frame callback, so
  render stays inside `sapp_run`; decoupling moves simulation off, not
  rendering.

## Rules for threaded code in agate

1. `sg.*` only from the context-owning thread; updates stage + flag, the
   render side flushes (`flushPendingGpuUploads` pattern).
2. Jobs are CPU-only, own disjoint index ranges, never nest
   `parallelFor`; no sg, no shared mutable state.
3. Determinism: per-entity work must not depend on chunking — pinned by
   tests (byte-equality, queue equivalence), not convention.
4. No silent fallbacks: missing pool/queue degrades to *serial execution*
   (a scheduling detail); allocation failures still surface as errors.
5. New shared state must be classified at design time per the audit
   taxonomy: COPY into payload / ALTERNATE-OWNED by one phase /
   GPU_ONLY handle / READONLY_AFTER_LOAD. The audit (2026-09) is the
   reference map; re-run it after touching render reads.

## What other engines do (reference)

- **Unreal**: Game -> Render -> RHI threads, one frame latency between
  each; task graph for systems. Stage 3's end state, most explicit form.
- **Unity**: main + render thread + work-stealing job system (stage 1).
- **id Tech (Doom Eternal)**: everything jobified; dedicated render
  thread only submits.
- **Naughty Dog (GDC 2015)**: fiber-based jobs, whole frame as one job
  DAG — end-state inspiration, not a starting point.
- **Frostbite**: frame graph + jobs; dropped the dedicated render thread
  in favor of jobs (requires owning the whole submission model).
- **Godot / bgfx**: thread-safe command queue consumed by the render
  thread — closest to what a full-payload agate looks like.
