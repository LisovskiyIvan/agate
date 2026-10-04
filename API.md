# Agate public API — Scene ownership guide

Scene creation, resource ownership, named-object lookup, and thread boundaries.

> Module-by-module documentation (Russian; full API references, threading and
> ownership notes, examples): **[docs/README.md](./docs/README.md)**.

## Scene creation

```zig
// Run after sokol.gfx.setup, on the marked graphics-context thread.
agate.gpu_thread.markContextThread(); // once, in the sokol init callback, before spawning the game thread
var scene: agate.Scene = undefined;
scene.initInto(allocator);
defer scene.deinit();
```

- `markContextThread` is strict and always live in every build mode
  (including ReleaseFast/ReleaseSmall): with a live `sg` context every
  non-marked thread fails `isOnContextThread` and every off-owner GPU
  touch panics. Unmarked + headless is a CPU-phase inline decision, not
  GPU authorization. With no marked thread (unit tests, tools) every
  thread counts as the context thread only while headless.

- Alternatively, use `var scene = agate.Scene.init(allocator)`; do not
  initialize the same Scene twice. Keep its address stable once content
  holds references to it.
- The allocator MUST be thread-safe when using workers (e.g. GPA with `.thread_safe = true`):
  prepare/render lazy caches and jobs workers allocate from it.
- `deinit` is context-thread only and drains everything: the retire queue
  (including uncompleted epochs), physics, meshes, materials, trails,
  particles, passes, and the async upload/file-I/O runners (joined first,
  so in-flight decodes finish before their target materials are freed).

## Allocator domains (optional)

Advanced users may split allocation by domain via `AllocatorConfig`
(`Scene.initWithAllocators` / `Scene.initIntoWithAllocators`); the
default (`init` / `initInto`) uses one allocator for everything and is
unchanged. Rules, per-consumer routing, thread-safety and lifetime
requirements: **[docs/allocators.md](./docs/allocators.md)**.

## Resources and lifetimes

### Meshes

- Create: `MeshBuilder.*` constructors, `uploadGeometry`, CSG/simplify
  builders, glTF/OBJ/PLY/STL loaders. All append to `scene.meshes`.
- Destroy: `scene.destroyMesh(mesh)`.
  - Unlinks from `scene.meshes` plus every cross-mesh referent first:
    outline list, highlights, soft bodies, physics bodies, parent/child
    and bone-attachment links, LOD bands, morph-target bindings,
    decal-manager instances.
  - GPU teardown is context-thread only. On the context thread the
    buffers are destroyed inline; off-context the mesh is unlinked now
    and retired into `gpu_retire` (epoch-stamped, no `sg.*`, no frees).
    The next render-start flush (`flushPendingGpuUploads`, inside staged
    `finishStagedPrepare`) destroys due entries after their epoch completes;
    `deinit` destroys everything, including uncompleted epochs.
  - Never call `sg.destroy*` on a mesh yourself; never `destroy` the
    struct directly — always go through `destroyMesh`.
  - Context-thread destruction must not invalidate a still-pending
    prepared frame. Game-side destruction under the update exclusion
    uses epoch retirement instead of freeing in-flight GPU handles.
- Off-context removal without free: `scene.removeMesh(mesh)` returns
  true when the mesh was registered and unlinks it from `scene.meshes`
  only — no referent scrub, no GPU retire, no frees. Ownership transfers
  to the caller, who must finish the teardown itself: `mesh.deinit` +
  allocator destroy on the context thread, or `gpu_retire.retireMesh`
  off-context for the next render-start flush (as `removeSoftBodyCloth`
  does). Prefer `destroyMesh` unless deliberately taking over lifetime
  management (e.g. error rollback after a manual `scene.meshes.append`,
  as the soft-body creator does).
- Off-context creation is CPU-only and deferred: with no context thread
  (or no valid `sg` context) the mesh keeps CPU mirrors plus
  `pending_vertices` (`gpu_pending = true`) and `finishGpuUpload` builds
  the buffers on the next context-thread flush. `Mesh.deinit` frees the
  retained copies either way, so a mesh that dies pending never leaks.

### Mesh names (`name` / `owns_name`)

- The name is a plain `[]const u8`; `owns_name` records whether it was
  allocated from the Scene allocator (`Mesh.deinit` frees it iff set and
  non-empty). Zig struct fields have no access modifiers — the flag is a
  documented convention, not enforcement.
- Borrowed (`owns_name == false`): string literals and builder/loader
  inputs held by the caller. Owned (`owns_name == true`): adopted dupes
  (loaders dupe; callers adopting a heap name, e.g. an `allocPrint`ed
  segment name passed to a builder, set the flag).
- Rename only via `scene.renameMesh(mesh, new_name) !void`:
  - `new_name` is borrowed; the Scene keeps an internal copy.
  - Copy-before-free: aliased input (`renameMesh(m, m.name)`, or a
    subslice) is safe.
  - Atomic on OOM: the old name and flag are untouched.
  - Always owned after success; `getMeshByName` resolves the new name.

### Materials

- Create: `scene.createStandardMaterial(name)`,
  `scene.createPBRMaterial(name)`, `scene.createShaderMaterial(name,
  shader_name)` (null on an unknown shader or allocation failure).
- Destroy: `scene.destroyPBRMaterial(mat)` unlinks and frees the CPU
  material immediately with no GPU retire — prepared draw records carry
  GPU handle values, never CPU material references, and materials own no
  GPU objects needing deferred teardown.
  Detach live mesh references first and let pending asynchronous uploads
  targeting the material finish before destroying it.
- Standard/shader materials remain Scene-owned until `Scene.deinit`;
  there is no individual destroy API for them.
- Material names are borrowed: `StandardMaterial.init(name)` /
  `PBRMaterial.init(name)` (and the `create*Material` wrappers) store
  the slice as-is — no dupe, no `owns_name` flag, nothing freed at
  destroy. The name storage must outlive the material (string literals
  or caller-owned buffers kept alive). Contrast mesh names, which are
  owned iff `owns_name` and renamed only via `scene.renameMesh`.
- Texture handles assigned to materials are borrowed. Keep their owner
  alive through every prepared frame that can reference them; a copied
  handle is not a retained resource. Async texture uploads use the queue
  drained in `beginPrepare`, not during draw submission.

### Particle systems

- Create: `scene.createParticleSystem(name, capacity)` (off-context
  creation defers GPU buffers to the context-side flush). Step: `scene.updateParticles(dt)`
  (CPU staging only; GPU-path errors surface as `UpdateError`, never a
  silent CPU downgrade).
- Scene-owned until `deinit` (context-thread only, like any direct
  `ps.deinit()`): there is deliberately NO `destroyParticleSystem`.
  `ParticleSystem.deinit` issues `sg.destroy*` inline (instance/slot and
  compute buffers, compute views/pipeline/shader, owned textures), which
  is illegal off-context, and the retire queue has no particle entry
  kind; mid-life removal would also have to scrub sub-emitter
  back-references and the prepared/build frames that borrow its handle
  ids. Create systems sparingly and reuse them (`start`/`stop`/`reset`).

### Trails

- Create: `scene.createTrailMesh(name, options)` (off-context defers
  buffer creation to the render-side flush). Step:
  `scene.updateTrails(dt)` (CPU staging only).
- Destroy: `scene.destroyTrailMesh(trail)` — unlinks the trail, destroys
  the linked scene mesh through `destroyMesh` (same epoch-retire
  contract), clears other trails targeting that mesh, then frees the
  trail's CPU staging and the struct. `TrailMesh.deinit` itself is
  CPU-only and never touches the linked mesh.

## Named-object lookup

- `scene.getMeshByName(name)` — first mesh with an equal name, else null.
- Tag/query search: `getMeshesByTag` / `getMeshesByQuery` (owned result
  lists), `countMeshesByTag` / `countMeshesByQuery`, `findFirstMeshByTag`
  / `findFirstMeshByQuery`. Mesh tags: `addTag(s)` / `removeTag` /
  `hasTag` / `matchesTagQuery`.

## AssetManager

`add*Task` returns a stable pointer until `reset`/`deinit`, including when
callbacks enqueue more tasks. Names and paths are borrowed for that lifetime;
callbacks may enqueue, but must not reset or destroy their manager.

- Mesh tasks with a Scene append real meshes. Scene-less OBJ/STL/PLY tasks
  retain CPU `GeometryData` in `task.mesh_result`. `takeMeshGeometry()`
  transfers it to the caller, who frees it with the manager's allocator.
  Scene-less GLB/GLTF tasks fail with `MeshRequiresScene`.
- Text/binary/texture results are borrowed from the cache when caching is
  enabled; otherwise the task owns them until reset. Cache replacement or
  clearing invalidates borrowed results. Do not clear textures while a
  prepared frame still uses their GPU handles.
- `loadStep(max_tasks)` is synchronous and limits **task count**, not elapsed
  time. Do not treat it as a frame-time budget. Use the existing `UploadQueue`
  or the loader's `.async_textures` option for off-thread texture decoding;
  upload still happens on the graphics context. These do not offload mesh parsing.
- A manager is single-caller, not internally synchronized. GPU texture
  creation and destruction require the context thread. Headless texture tasks
  validate decoding only and return no GPU texture.

## Thread ownership

- Game (update) thread, under update-vs-prepare exclusion: mutate live
  meshes/materials/transforms/tags, create content (deferred paths above),
  `destroyMesh`/`destroyTrailMesh`/`renameMesh` (pure CPU plus the retire
  path), step decals/particles/trails/physics/animations, publish frame
  snapshots and prepared builds.
- Context thread (staged prepare + render, sequential): `buildPreparedFrame`
  (producer-side frozen build), `beginStagedPrepare`/`finishStagedPrepare`,
  `render`/`renderReuse`, flushes, all `sg.*` creation/destruction,
  `Scene.deinit`, `resizeOffscreen`. Workers and the update side never
  touch the frame snapshot, render passes, or render-owned caches.
- Rule of thumb: if it touches `sg.*`, it runs on the context thread or
  through `gpu_retire`. `gpu_thread.isOnContextThread()` is the branch;
  with no marked thread (unit tests, tools) every thread counts as the
  context thread. Update-side code never writes `scene.stats` directly —
  timings cross via `recordUpdateTime`/`recordPhysicsTime` and are
  transferred by the finished staged prepare.

## Runtime facade (threaded frame lifecycle)

`agate.Runtime` owns the update-vs-prepare choreography so apps do not
reproduce the claim/build/stage/publish and begin/finish/render ordering
by hand. The host still owns the window, sokol setup, its `simulate`
body, render instrumentation, and save/quit policy.

```zig
var rt = agate.Runtime.init();
defer rt.deinit(); // joins the worker before Scene.deinit

fn gameTick(ctx: *Tick) void { /* simulate only; no build here */ }
fn gameLoop() void {
    while (rt.shouldRun()) {
        _ = rt.update(&scene, &tick, gameTick); // simulate + build (lock-free by default)
    }
}
```

- Simple path (the agate demo): `update(scene, ctx, tick)` runs `tick`,
  then `claim -> build -> stageUi -> publish`. `renderFrame(scene)`
  returns `prepared`, `reused`, `skipped` (nothing consumable yet), or
  `busy` (exclusion mode only: acquisition budget exceeded; a consumable
  front is still re-presented). The first frames skip until the first
  build is ready.
- Lock-free by default: `beginPrepare`/`renderFrame` take no phase mutex.
  Safety comes from the fresh-build contract — the producer froze every
  upload payload into the slot (`stageUploads`/`stageUi`), the game-side
  commit consumes the live flags/scalars, and host live-state reads travel
  frozen in `BuildClaim.stageHostBytes` (read back as
  `PrepareClaim.host_bytes`). `setProducerExclusion(true)` restores the
  previous mutex window (`beginPrepareWith` then runs host live-state reads
  inside it; bounded by `setLockWaitNs`).
- Advanced path (instrumented hosts, e.g. Sandbox): `gameLock`/
  `gameUnlock`, `produceBuild`/`produceBuildWithHostBytes`,
  `beginPrepare`/`beginPrepareWith` (return `BeginResult`
  `{ claim, busy, wait_ns, held_ns }`: null claim + `busy == false` means
  no fresh build — reuse or skip; `busy == true` means the exclusion
  budget was exceeded), `finishPrepare`/`cancelPrepare`,
  `reuseIfConsumable`, and `tryRunLocked` (never blocks;
  false means the caller keeps its previous snapshot/title).
- Claim contract: a successful begin must pair with exactly one
  `finishPrepare` or `cancelPrepare`. `finishPrepare` and
  `render`/`renderReuse` are context-owned and overlap the next producer
  tick.
- Still mutex-dependent by design: the `--no-concurrent-build`/
  `--no-threads` diagnostics (explicit produce + begin/finish inline
  under `gameLock`/exclusion instead of a helper), profiler control and
  report file IO (bounded window only when a request is pending), and
  registry add/remove while a latch is in flight (an app-contract
  violation, kept coherent by the commit guards).
- Worker lifecycle: `spawnWorker(entry)` starts the game loop,
  `shouldRun()` gates it, `quiesce()` is idempotent, and `Runtime.deinit`
  quiesces then tears down the mutex. `spawnWorker` must not be called
  twice.
- `RuntimeMetrics` counts what each call did (`producer_builds`,
  `producer_skips`, `begins`, `begin_empty`, `begin_busy`, `finishes`,
  `cancels`, `reuses`, `skipped_presents`). Hosts that
  keep bespoke legacy branches bump the counter for the event they
  perform, so totals stay truthful.
- Serial diagnostic (no worker): run the explicit frozen-frame sequence
  inline — `produceBuild` (game side, under `gameLock`), then
  `beginPrepare` + `finishPrepare`/`cancelPrepare`, then
  `render`/`renderReuse`. `render` never auto-prepares: with no
  consumable frame it drops the present (first frames skip until the
  first build is ready). `Scene.buildPreparedFrame() bool` is the
  fully-claimed helper for the same sequence (`tryClaimBuildSlot` +
  `build` + `stageUi` + `publish`); it returns false when every
  non-front slot is pinned/claimed (latest-wins skip — the context
  reuses the last front).

> Breaking API (historic names, not instructions): `Scene.prepareFrame`,
> `Runtime.prepareSerial`, and the `FrameDraws` back-slot accessors are
> removed. `PrepareClaim` is `{ token_id, back_idx, build_seq, host_bytes }`.

The contracts above still describe the `Scene` itself: what the game
thread may do, what the context thread owns, and when destruction
actually happens (retire flush vs `Scene.deinit`).
