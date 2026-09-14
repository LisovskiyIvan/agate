# Threading refactor: single-threaded → multi-threaded

Status legend: [x] done, [~] in progress, [ ] planned.

## Current state

The whole frame runs in the `sapp` frame callback on the main thread:
input events → `Scene.update*()` → `Scene.render()` → `sg.commit()`, strictly
sequential (`src/main.zig`, `scene.zig`). No engine subsystem spawns threads.

## Hard constraints

- **sokol_gfx is not thread-safe.** Every `sg.*` call must come from one
  thread (the GL/Metal context owner). `SOKOL_THREAD_SAFETY=1` exists but is
  a single global mutex: it serializes, it does not parallelize.
- **`sapp` callbacks are main-thread.** Windowing, input, swapchain stay there.
- Implicit main-thread assumptions: generation-tagged caches
  (`Mesh.cached_*`, `frame_id` in `render_queue.zig`), per-frame scratch
  buffers, single PRNGs per subsystem.

## The ladder (cheap → expensive)
### [x] Stage 1 — job pool + data-parallel systems (landed)
Fork-join pool for pure-CPU work. No `sg.*`, no shared mutable state; each
job owns a disjoint index range.

- `src/agate/jobs.zig` — `Pool` (N workers + calling thread forage a chunked
  atomic cursor), `parallelFor(?*Pool, ctx, fn, len)` with serial fallback
  (null pool / `min_len_for_workers`); workers park on a pthread condvar.
  Process-global handle `jobs.global` for engine code; explicit pools for
  tests. Landmark: Zig 0.16 `std.Thread` ships no public condvar, so the
  pool wraps `std.c` directly (the engine links libc for sokol anyway).
- [x] CPU particle integration (`particles.zig updateCpu`) split into
  **integrate (parallel) → swap-compaction (serial, legacy-exact) →
  instance fill (parallel)**. Per-slot work is order-independent, so results
  are bit-identical for any worker count (pinned by a determinism test).
  Pool is injected per system (`thread_pool`), falling back to
  `jobs.global`, then to inline execution.
- [x] Frustum culling (`render_queue.buildFrameQueues`): split into
  `cullNonInstancedMesh` (pure per-mesh test producing a record) +
  `submitInstancedMesh` (serial; owns all sg buffer work) + a chunked
  parallel pass. Chunks partition the mesh list in fixed order and merge
  back in chunk order, so queues and stats are identical to the serial
  loop — pinned by a 3000-mesh equivalence test. Instance-bearing meshes
  keep the serial path (sg calls inside); world-matrix caches are warmed
  serially before the parallel pass (shared parents would race otherwise).
  Active only above `FrameCullContext.parallel_min_meshes` (default 1024)
  with a pool attached — current demo scenes sit below it by design.
- [x] Trails / decals / nav agents audit (2026-09): `NavAgent.update` reads
  only its own state (waypoints/position/velocity — no neighbor coupling),
  `TrailMesh.update` touches only its own segment history, `DecalLayer`
  ticks a single manager. Parallel-safe in shape; not routed through the
  pool yet — real scenes hold single-digit entities.

### [x] Stage 2 — async assets & uploads (loader integration landed)

- [x] `jobs.TaskRunner` — fire-and-forget tasks on dedicated threads;
  deliberately separate from `Pool.forkJoin` (forkJoin spins its callers,
  so long tasks must never share its workers). Shutdown drains: join
  guarantees every posted task finished writing.
- [x] Loader integration: `LoadOptions.async_textures` (opt-in; sandbox
  demos use it). Sync mode unchanged — fork-join predecode, single hitch.
  Async mode skips the predecode; materials start with null texture slots
  (the engine's existing default-white fallback renders) and
  `AsyncTexCtx` registers each slot as an `UploadQueue` patch target —
  one background decode per image+srgb pair, embedded bytes copied at
  request time (cgltf data dies at appendGlb return). `Scene` owns the
  queue: drained at the top of `render()` so patched textures draw the
  same frame; deinit'd FIRST in `Scene.deinit` so in-flight decodes join
  before any targeted material is freed. Emissive white-override now
  keys on "texture requested" so async loads keep the factor-hack
  semantics. Measured (Debug): DamagedHelmet critical-path 169 → 10 ms,
  Fox 41 → 11 ms; sandbox frame loop verified uploading +7 textures in
  one drain. NOTE: this path is only fully covered by app smoke — tests
  have no GL context and cannot reach the uploaded state.
- [ ] Serialization save off-thread: needs a scene-quiesce or snapshot
  story first (state holds pointers; plain capture-then-marshal is unsafe
  while the main thread mutates).

### [ ] Stage 3 — game/render thread split ("non-blocking render")

Render thread owns the graphics context: `sg.setup` happens there and **all**
`sg.*` calls (create/update/destroy/commit) move to it. Main thread:
input + `Scene.update` → capture per-frame uniform/draw snapshots
(`scene/uniforms.zig` structs are already close to what needs capturing) →
SPSC handoff. Render thread consumes the latest complete frame.

Costs to accept explicitly: +1 frame input→pixel latency; state snapshot
discipline (double-buffer whatever update mutates); strict
create/destroy-on-render-thread ownership. Benefits: update spikes
(serialization, streaming, 100k+ particles) never starve submission.

Not available in sokol: parallel command recording (Vulkan/DX12 secondary
command buffers). "Multithreaded render" here always means "render on its
own thread", never "render recorded by many threads".

## What other engines do (reference)

- **Unreal**: Game → Render → RHI threads, one frame latency between each;
  task graph for systems. Stage 3 in its most explicit form.
- **Unity**: main + render thread + work-stealing job system
  (`IJobParallelFor` ≈ stage 1).
- **id Tech (Doom Eternal)**: everything jobified; dedicated render thread
  only submits.
- **Naughty Dog (GDC 2015)**: fiber-based jobs, whole frame is one job DAG —
  end-state inspiration, not a starting point.
- **Frostbite**: frame graph + jobs; dropped the dedicated render thread in
  favor of jobs (requires owning the whole submission model — we do not).
- **Godot / bgfx**: thread-safe command queue consumed by the render thread —
  closest to what stage 3 looks like under sokol.

## Rules for any threaded code in agate

1. `sg.*` only from the context-owning thread (today: main; stage 3: render).
2. Jobs are CPU-only, own disjoint ranges, never nest `parallelFor`.
3. Determinism: per-entity integration must not depend on chunking —
   pinned by tests, not by convention.
4. No silent fallbacks: a missing pool degrades to *serial execution*, which
   is a scheduling detail, not a behavioral one. Allocation/OOM still errors.
