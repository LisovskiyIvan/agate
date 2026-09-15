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

### [~] Stage 3 — simulation/render decoupling (groundwork landed)

Sokol reality check (design driver): the swapchain is acquired through
`sglue`/`sapp_swapchain`, which is only valid inside the sapp frame
callback, so the render side must stay inside `sapp_run`. The achievable
split is therefore NOT "render on a separate thread" but:

- **sapp thread (window + render)**: frame callback drains input, renders
  the latest published state, commits. Never blocked by simulation spikes.
- **game thread**: `Scene.update*` at its own cadence, publishing a frame
  payload through `Handoff` (latest-wins; drops allowed, never queues).
- **Input** crosses threads through `jobs.SpscRing` (events arrive on the
  sapp thread; the update side drains them).

Costs to accept explicitly: +1 frame input→pixel latency; the frame payload
must own everything render reads (or render keeps reading engine state with
ownership alternating through the handoff — requires identifying every
mutable field render touches: mesh transforms, materials, particles, UI).
That audit is the real work of this stage; the primitives below are its
mechanical half.

Landed groundwork:

- [x] `jobs.SpscRing(T, N)` — lock-free SPSC ring, drop-newest when full,
  monotonic indices, power-of-two capacity; same-thread safe so apps can
  adopt it before any thread crosses. Engine `main.zig` input already
  flows through it (drained at the top of `frame()`).
- [x] `handoff.Handoff(T, slots)` — lock-free latest-wins N-slot mailbox:
  `claim` CASes a free slot, `publish` stamps a global sequence,
  `takeLatest` copies the newest payload and releases stale slots. Fixed
  during testing: the sequence MUST come from a global counter — per-slot
  seqs let stale frames resurface; and multi-publisher is unsupported by
  design (counter assignment happens before the meta store, so "newest"
  would be ill-defined). Verified with a threaded publisher/consumer
  stress test (untorn checksums, strict delivery order, drops allowed).

Remaining slices, in order:

1. [x] Frame payload audit (see above).
2. [x] Update-phase consolidation: `Scene.update(dt)` is the single
   game-side entry point — camera, lights (`updateLights`, hysteresis
   fades included), physics, animations, particles, **nav agents**,
   **trails**, decals in canonical order. Engine and sandbox frame loops
   both drive it. Nav agents and trails were folded in after verifying
   they receive the same clamped real-seconds dt the umbrella passes
   (the sandbox had been feeding them exactly that value), and their
   updates are self-only CPU work. The light pack travels through
   `Scene.light_handoff` (`handoff.Handoff(FramePack, 2)`): update
   publishes, render take-latest — the first state group is already
   thread-ready; when the split lands, lights cross the boundary with
   zero code change. The consumed copy (`light_pack`) stands between
   publishes, which covers the PIP multi-render-per-update pattern
   (pinned by test). Deferred to split day: trail and particle updates
   end with sg.updateBuffer uploads — those move to a render-side flush
   (pattern already exists in particles.flushGpuUpload).
3. [ ] Game thread: run the update umbrella on a worker publishing
   through `Handoff` (same-thread mode first — publish inline, consume
   immediately — to prove no drift), then the actual split behind a flag.
   Serialization of engine systems is expected to surface races — fix
   them at the audit level, not with locks.

#### Frame-payload audit (done 2026-09)

Full sweep of every CPU read in `Scene.render()` (core path + all passes).
Key findings, condensed:

Render-side hidden mutations today (must stay render-owned or move):
- `LightRig.packFrame(..., dt)` runs light-selection **hysteresis fades
  inside render** (light_selection.zig update) — simulation living in the
  render pass; move to the publish step.
- `worldMatrixCached` writes `mesh.cached_matrix/cached_aabb/cached_frame`;
  `submitInstancedMesh` writes InstancedMesh caches + instance buffers;
  `ShadowPass.binned_meshes`/`spot_needs_clear`; `Scene.stats` reset+counters;
  `frame_id` bump; lazy GPU resources (`ensureForwardMsaa`,
  `ShaderMaterialCache.getOrCreate`, sky/postfx MSAA twins); `WarnOnce`.
- `UploadQueue.drain` patches material texture slots at render top (stage 2).
- Update side issues sg calls today: `particles.update` (instance buffer
  upload), `TrailMesh.update` (index buffer upload) — must move under the
  render side on a split.
- `DecalManager.update` destroys meshes/materials in `scene.meshes` —
  container-level hazard.

True shared-mutable (update writes AND render reads/writes same field):
mesh/instance caches; `morph_weights` (draw-time read); skeleton
`skin_matrices` (draw-time upload); material scalars+slots; light
hysteresis; container arrays frozen only by single-thread discipline.

Payload split decision (summary): COPY per frame — per-camera derived
{view_proj, eye, viewport, mask, clear}, sun pack, `FramePack`, shadow
config+cascades, post/ssao/msaa/sky config, per-item draw records (model,
flags, material scalars ~120B, texture handles), skin palette (4KB/skeleton),
morph weights, stats double-buffer. ALTERNATE (render-owned): queues,
matrix/instance caches, occlusion culler + HiZ, hysteresis, frame_id, lazy
GPU caches. GPU_ONLY: all sg handles. RO_AFTER_LOAD: topology, base_matrix,
parents, LODs, cpu geometry.

Two semantics bugs the audit surfaced, FIXED:
1. `worldMatrixCached` ignored `attach_bone` while `Mesh.getWorldMatrix`
   honored it — bone-attached meshes rendered at different positions in
   main vs shadow passes. The cached path now mirrors getWorldMatrix
   exactly (bone branch replaces the parent chain when the host skeleton
   is live; dead skeleton falls through to parent).
2. `frame_id` was bumped before the shadow pass, whose cache checks
   therefore always missed and recomputed every world matrix twice per
   frame. Shadow fallbacks now use `worldMatrixCached`/`worldAABBCached`:
   shadow warms the cache, the main pass queue build hits it, and bone
   semantics are identical across passes. Nothing in engine or sandbox
   sets attach_bone yet, so the fix is inert today — it closes the trap.

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
