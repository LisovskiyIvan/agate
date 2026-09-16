# Threading refactor: status & roadmap

Status as of 2026-09-15. Legend: [x] done, [~] partial, [ ] planned.

## Status at a glance

| Stage | Scope | Status |
|---|---|---|
| 0. Thread affinity & GPU ownership | marker, deferred create/update/destroy, off-context loads | [x] done |
| 1. Data-parallel CPU systems | job pool, particles, culling | [x] done |
| 2. Async assets | TaskRunner, UploadQueue, glTF async textures | [x] done |
| 3. Simulation/render decoupling | threads, lock-free render, full per-item payload | [x] done |

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
  never resurface, drops allowed. Single-publisher by contract; a
  publisher-side `releasePublished()` drains saturated slots before
  republishing so the newest state always wins.
- `gpu_thread` — graphics-context thread marker: `markContextThread()`
  (called from the apps' sokol init), `isOnContextThread()` for call
  sites that must defer GPU work, `assertOnContextThread()` as the
  Debug + ReleaseSafe tripwire.
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
  `FrameCullContext.parallel_min_meshes` (default 128). Instance-bearing
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
- Serialization save/load runs off-thread after an owned `SceneState` capture;
  restore is applied explicitly on the game thread.

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
- **Thread affinity marker.** Both apps call
  `gpu_thread.markContextThread()` in their sokol init; engine paths that
  may run on the game thread branch on `isOnContextThread()` and
  `assertOnContextThread()` catches misplaced `sg.*` in Debug and
  ReleaseSafe. Off-context sync texture creation is rejected loudly;
  async decode/upload is the legal path.
- **Deferred creation and destruction.** `Scene.destroyMesh` unlinks
  immediately and queues off-context teardown (`pending_gpu_destroys` +
  allocation-free `[8]` overflow, err-logged leak only under pathological
  OOM), drained at render start and in `Scene.deinit`. `uploadGeometry`
  and the glTF loader build CPU-only meshes off-context
  (`gpu_pending` + `pending_vertices`), with `pending_dynamic_update`
  (CPU-morph empty dynamic buffer) and `morph_upload_pending` (delta
  texture) finished by `Mesh.finishGpuUpload`. Particle systems and
  trail meshes defer instance/vertex/index buffer creation the same way.
- **Off-context GLB loading.** `.async_textures` queues image decode on
  `Scene.uploads`; an off-context load auto-degrades to async textures
  when the queue exists (sync texture creation would assert). The
  sandbox `--test-async-load` harness pins the full path
  (`gpu_pending=1 -> 0`, material texture patched by the drain).
- **Mailbox newest-wins.** Saturated frame/light handoffs drain stale
  published slots before republishing (`Handoff.releasePublished`), so
  the newest complete state always wins (100/200/300 regression test).
- **Destroy referents.** `destroyMesh` neutralizes every cross-reference
  before freeing: physics body, children's `parent`, bone attachments,
  LOD entries, decal-manager instances (and their materials), animation
  morph bindings (weights slice + dirty pointer tombstoned so channel
  indices stay valid).

Known limitations of the shipped split:

- **Coarse payload.** Phase ownership means an update spike delays render
  for its duration. Strictly non-blocking render needs the full per-item
  payload (below).
- **Per-instance transparency sorting (OIT)** implemented: transparent instanced
  batches and instanced decals sort instance matrices back-to-front relative to camera eye
  in `submitInstancedMesh` with deterministic tie-breaking. Instanced picking is
  implemented: `pickWithRay` tests every visible instance in the drawn space
  (`cached_world_matrix`, refreshed with the same call the render path uses)
  and reports `PickingInfo.picked_instance`.
- **Uploads and I/O are paced and split**: `UploadQueue.drainBudget` uploads
  at most `Scene.upload_budget_per_frame` (4) textures per frame without ever
  holding the spinlock across GPU work (stack-batched collection), and async
  save/load runs on a dedicated `Scene.io_runner` instead of the texture
  decode runner. `SceneStats` exposes phase timings and upload counters
  (`--stats`).

## What remains (TODO, in priority order)

1. [x] **Full per-item frame payload / strictly non-blocking render.**
   Implemented full per-item payload: `RenderMeshItem`, `RenderInstancedBatch`,
   `ShadowDrawItem`, and `OutlineDrawItem` carry all needed GPU buffer handles
   (`vertex_buffer`, `index_buffer`, `instance_buffer`, etc.), material draw
   records, and model matrices. Zero `*Mesh` reads/dereferences occur during
   `Scene.render()` execution.
   Queue construction, shadow binning (`ShadowPass.prepare`), outline collection,
   and upload queue drains execute strictly during `prepareFrame()` under `phase_mutex`.
   In `main.zig` (both `agate` and `sandbox`), `phase_mutex` is shrunk to cover
   only `prepareFrame()` (+ UI generation in sandbox), allowing `Scene.render()` to run
   100% lock-free concurrently with the ~1 kHz simulation loop. Verified under
   multi-thousand mesh creation/destruction churn stress tests without data races or asserts.
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
4. [x] **Granularity refinements.**
   - Lowered default `parallel_min_meshes` from 1024 to 128 (and adaptive `(workers + 1) * 32` when set to 0),
     with zero-allocation worker culling (`initCapacity(span)` on caller thread + `appendAssumeCapacity` in workers).
   - Parallel shadow-pass bucket binning (`ShadowPass.binMeshes` with chunk counting, prefix sums, and lock-free parallel scatter).
   - Parallel instanced-path transform staging in `submitInstancedMesh` (`ParallelInstanceStage` with parallel TRS updates, chunk AABB reductions, and parallel matrix scattering).
   - All parallel paths verified bit-identical and deterministic with comprehensive unit tests and live sandbox runs.
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

1. `sg.*` only from the context-owning thread. Creation, upload and
   destruction follow one shape: the game side stages CPU data or unlinks
   objects and sets a flag (`gpu_pending`, `*_pending`,
   `pending_gpu_destroys`), the render side finishes it
   (`flushPendingGpuUploads` / `drainPendingGpuDestroys`). `gpu_thread`
   asserts the thread in Debug + ReleaseSafe; off-context sync texture
   creation is rejected loudly, async decode/upload is the legal path.
2. Jobs are CPU-only, own disjoint index ranges, never nest
   `parallelFor`; no sg, no shared mutable state.
3. Determinism: per-entity work must not depend on chunking — pinned by
   tests (byte-equality, queue equivalence), not convention. Parallel and
   serial queue paths must produce identical `transparent_order`, so the
   tie-break derives from the source mesh index, never insertion order.
4. No silent fallbacks: missing pool/queue degrades to *serial execution*
   (a scheduling detail); allocation failures still surface as errors
   (e.g. `GpuBufferAllocationFailed` instead of dead buffer handles).
5. Deleting an object must neutralize its referents before the memory is
   freed (physics body, hierarchy links, LOD entries, decals, animation
   morph bindings); failing to do so is a bug, not a limitation.
6. New shared state must be classified at design time per the audit
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

## Аудит 2026-09-15: состояние, ошибки, куда двигаться

Дата: 2026-09-15. Первичный метод: статический аудит кода; затем исправления,
независимый review и проверки ниже. Пути относительно `agate/`, кроме `sandbox/`.
Вывод: широта возможностей опережает надёжность их совместной работы.
Следующий шаг — стабилизация потоковой модели, ресурсов и рендера, а не новые эффекты.

### 1. Критично: simulation/render разделены на потоки, но данные не разделены — исправлено стабилизацией

- `src/main.zig:205-212`, `sandbox/src/main.zig:359-364`: mutex защищает только
  `prepareFrame()`, затем `scene.render()` работает без него.
- Рендер читает живую сцену: `src/agate/scene.zig:780-800`
  (`self.meshes.items`), `src/agate/scene/render_queue.zig:161-183`
  (`worldMatrixCached`), игровой поток параллельно пишет TRS
  (`src/main.zig:172-173`).
- `SceneFrameSnapshot` содержит камеры/свет/настройки, но не независимый снимок
  мешей и материалов: `src/agate/scene/snapshot.zig:41-85`.
- Оба потока используют `jobs.global`, хотя `Pool.forkJoin()` single-producer:
  `src/agate/jobs.zig:140`. Возможны потеря dispatch и зависание.
- Действие: сначала вернуть mutex на весь `prepareFrame() + render()` в обоих
  приложениях; убрать утверждение о завершённом независимом рендере; настоящий
  concurrent render делать отдельной задачей с неизменяемыми render-records.
- Статус: mutex снова охватывает полный render в `agate` и `sandbox`; общий
  single-producer pool больше не вызывается двумя producer одновременно.
- Проверка: CPU-частицы выше порога 4096 + сотни мешей в рендере; анимации,
  изменение материалов, создание/удаление объектов. Smoke с кубом недостаточен.

### 2. Высокий приоритет: ошибочный `free` при загрузке кубмапы — исправлено

- `CubeTexture.fromFiles()`: `src/agate/texture.zig:944-974`.
- `face_data` стартует как `undefined`; при неквадратной грани или несовпадении
  размеров освобождается `0..i+1`, хотя `face_data[i]` присваивается позже.
- Дополнительно: отказ `dupeZ()` на следующей грани оставляет ранее загруженные
  изображения без очистки.
- Действие: счётчик успешно загруженных граней + единый `defer`; регистрировать
  `data` до последующих проверок.
- Статус: введены `loaded` + единый sized-defer; текущая грань освобождается
  явно при ошибке валидации. Заодно закрыто переполнение RGBA8 size arithmetic.
- Проверка: первая неквадратная грань; несовпадение размеров на 2-й и 6-й;
  отсутствующий файл после успешных загрузок; отказ аллокатора. Одного
  `std.testing.allocator` недостаточно: освобождение идёт через `stbi_image_free`.

### 3. Высокий приоритет: частичный отказ `spawn` может повесить инициализацию — исправлено

- `Pool.init()`: `src/agate/jobs.zig:106-110`.
- `TaskRunner.init()`: `src/agate/jobs.zig:298-302`.
- При отказе позднего `spawn` `errdefer` делает `join()` без `quit`/`broadcast`,
  worker ждёт работу, инициализатор ждёт worker.
- Действие: на частичном отказе выполнить `quit → broadcast → join` только
  созданных потоков, затем освободить ресурсы.
- Статус: оба init-пути сигналят shutdown, join-ят созданный prefix и уничтожают
  pthread mutex/cond на всех error-path после инициализации `self`.
- Проверка: управляемый отказ второго/третьего `spawn`; должен вернуться error,
  а не зависнуть.

### 4. Анимация: усечённые bone-треки без bounds-check — исправлено

- LINEAR/STEP-пути: `src/agate/animation/sampler.zig:138-163`,
  `src/agate/animation/sampler.zig:173-220`.
- Node-пути guard есть через `samplerHasFrames()`, bone-пути вызывают сэмплер
  напрямую: `src/agate/animation/group.zig:516-537`.
- Действие: применить существующую проверку к bone LINEAR/STEP; поведение cubic
  не менять.
- Статус: malformed LINEAR/STEP пропускаются с сохранением позы; безопасный
  прежний fallback CUBICSPLINE сохранён и закреплён тестами.
- Проверка: bone-аналог `NodeChannel invalid targets and samplers never crash`
  (`src/agate/animation/tests.zig:389`), включая `applyAtTime` и блендинг скелета.

### 5. Прозрачность сортируется не в общем порядке — исправлено

- `src/agate/scene.zig:803-844`: сначала все прозрачные regular, затем все
  прозрачные instanced без сортировки по расстоянию
  (`src/agate/scene/render_queue.zig:438-439`).
- Действие: общая прозрачная очередь либо слияние двух очередей по distance-key.
- Статус: введён единый retained order для regular/instanced batches со строгой
  back-to-front сортировкой; instanced decals согласованы с blend/cull-off draw.
- Проверка: regular/instanced на 15/10/5 м, порядок `15 → 10 → 5`.
- Не путать с OIT и per-instance сортировкой внутри draw call — отдельные лимиты.

### 6. Picking не соответствует viewport камеры — исправлено

- `createPickingRay()` всегда использует размер окна:
  `src/agate/scene/picking.zig:17-27`; рендер строит проекцию по viewport:
  `src/agate/scene.zig:893-905`.
- Действие: общий расчёт viewport для рендера/picking/projection; выбор камеры
  под курсором для PIP.
- Статус: ray использует тот же округлённый pixel rect/aspect, а PIP выбирает
  верхнюю включённую камеру под курсором. Instanced picking остаётся backlog.
- Проверка: центр/углы смещённого viewport, клик вне него, перекрывающиеся камеры.
- Отсутствие picking инстансов задокументировано в `picking.zig:36` — недостающая
  возможность, а не скрытый баг.

### 7. Пакет не самодостаточен для внешнего подключения — исправлено в manifest/build graph

- `.paths` включает только build-файлы и `src`: `build.zig.zon:40-47`.
- Сборка требует `examples/shader_materials/ramp_wave.glsl` (`build.zig:42-44`)
  и `tools/test_runner.zig` (`build.zig:368`).
- `sandbox/build.zig.zon:7-9` использует `.path = "../agate"` и маскирует проблему.
- Действие: включить нужные файлы в `.paths`; проверить упакованный пакет
  отдельным потребителем. Обычной сборке убрать скрытую перезапись
  `src/agate/tests.zig` (`build.zig:115-123`): вынести регенерацию в явный шаг.
- Статус: `.paths` включает test runner и каталог shader snippets; обычный build
  не пишет source tree; `zig build update-tests` — единственный явный regen,
  `zig build test` проверяет stale registry. `run` теперь передаёт `b.args`.

### Проверено и отведено как ложные срабатывания

- `UploadQueue.requestMemory`: утечки нет, владение переходит только при успехе,
  caller освобождает при ошибке (`src/agate/assets.zig:175-199`,
  `src/agate/loader/materials.zig:249-253`). Осталась только неоднозначность
  комментария о владении.
- `applyTorqueImpulse`: Box3D сам будит тело в `b3Body_SetAngularVelocity`
  (`src/agate/c/box3d/src/body.c:1178-1211`). Осталось недокументированное
  упрощение mass вместо inertia tensor.

### Чего не хватает

1. [~] Матрица интеграционных сцен: unit-тестами закрыты serial/parallel
   очереди и типы очередей, instanced picking и OOM-пути; runtime-покрытие —
   `--test-pip` (PIP), `--msaa`, `--particles`, `--test-decal`,
   `--test-async-load` и `--test-stress` (200+ мешей, instanced-группа,
   >4096 частиц, create/destroy-хворь на игровом потоке, 240 кадров).
   Visual regression отсутствует.
2. [~] README и CI по-прежнему отсутствуют; проверка fetched/archive-пакета
   не закрыта (`zig fetch` зависает в этой среде); устаревшие статусы
   `roadmap.md` выправлены 2026-09-15.
3. [x] Контракты владения и отложенное освобождение ресурсов: affinity-маркер,
   отложенные create/update/destroy, чистка referent'ов при удалении и
   OOM-политики без off-context `sg.*`. Осталось: полный per-item payload и
   instanced picking.
4. [x] Сохранение игры: устойчивые entity ID (`u64 id`), связи объектов (parent-child иерархия),
   пользовательские свойства (`SceneState.game_properties: []GameProperty`) и версионирование
   формата (v3 с сохранением полной обратной совместимости с v2).
5. [~] Метрики: `SceneStats` отдаёт `update_ms`/`prepare_ms`/`shadow_ms`/
   `main_ms`/`post_ms` и `uploaded_textures_frame`/`uploaded_bytes_frame`
   (текстуры; байты буферных аплоадов — следующая волна); agate печатает
   строку по `--stats`. Осталось: ожидание mutex/jobs, p95/p99, память
   текстур/геометрии.
6. Asset pipeline раньше новых эффектов: mip-цепочки, сжатые текстуры, кэш импорта.

### Порядок работ

1. [x] Стабилизация: потоки, `spawn`, кубмапы, bounds-check анимаций.
2. [x] Владение ресурсами: affinity-маркер, отложенные create/update/destroy,
   чистка referent'ов, OOM-политики, off-context glTF-загрузки.
3. [~] Воспроизводимость: package paths, test-registry и `zig build fmt`
   готовы; README/CI и archive-consumer check остаются.
4. [~] Согласованность рендера: прозрачная очередь и viewport-aware picking
   готовы; visual-regression сцены остаются.
5. [ ] Одна небольшая законченная игра как проверка движка.
6. [x] Оптимизация только по профилю: полный per-item payload (см. #1 выше),
   per-instance сортировка прозрачности (OIT), встроенный profiler / flight recorder,
   снапшоты памяти VRAM/CPU, байтовые метрики аплоадов.

### Выполненные проверки

```sh
# В agate/: успешно (2026-09-15)
zig build fmt                          # чисто (ранее падал на 7 файлах)
zig build update-tests                 # повторный запуск идемпотентен
zig build test                         # 584/584
zig build test -Doptimize=ReleaseSafe  # 584/584
zig build
zig build run -- --frames 30 [--no-threads | --msaa 4]

# В sandbox/: успешно
zig build
zig build run -- --frames 120
zig build run -- --test-pip            # PIP: вкл/выкл мультикамеры
zig build run -- --test-save           # async save/load состояния
zig build run -- --test-decal          # создание декали из игрового потока
zig build run -- --test-async-load     # off-context GLB + async-текстуры
zig build bench -Doptimize=ReleaseFast
```

`zig build fmt` зелёный: семь ранее неформатированных файлов отформатированы
2026-09-15. Sandbox dependency-build не изменяет `src/agate/tests.zig`.
Отдельный `zig fetch .` зависает в этой среде, поэтому fetched/archive package,
visual regression и нагрузочная матрица остаются незакрытыми проверками.

## Стабилизационные волны (2026-09-15)

Четыре волны после аудита. Коммиты: `c64e8be`, `8e27ab7`, `38f9a9e`, `222722b`
(+ форматирование `0e8d413`) в agate и `e977606`, `605045a` в sandbox.
Тестов стало 563 → 584.

### Волна 1 — стабилизация (`c64e8be`)
- mutex снова охватывает `prepareFrame() + render()` до появления полного
  per-item payload; частичный отказ spawn завершает потоки корректно;
- кубмапы: единый sized-defer, checked size arithmetic; bounds-check
  LINEAR/STEP bone-треков;
- общая back-to-front очередь прозрачности (regular + instanced, decals);
- viewport-aware picking (ray, containment, PIP-камера);
- `.paths` пакета, явный `zig build update-tests`, передача аргументов в run.

### Волна 2 — потоковая модель (`8e27ab7`, sandbox `e977606`)
- `gpu_thread` маркер и ассерты графического потока;
- `Scene.destroyMesh` откладывает teardown; `uploadGeometry` строит CPU-only
  меши (`gpu_pending`); частицы и trail откладывают буферы;
- насыщенные mailbox'ы дренируют stale-слоты и перепубликуют (100/200/300);
- parallel culling: skip инстансов, OOM-fallback в serial, единый tie-break
  по индексу меша, guard на tail-чанки;
- sphere picking через обратную матрицу (точный для родительских цепочек,
  поворотов и неравномерного масштаба);
- `buildRaw` проверяет размеры; KTX2 маппит новые ошибки;
- sandbox: mouse-mailbox вместо флуда кольца, `--test-decal`.

### Волна 3 — жизненный цикл ресурсов (`38f9a9e`)
- `destroyMesh` нейтрализует referent'ы: тело физики, `parent`,
  `attach_bone`, LOD-записи, декаль-инстансы и morph-привязки анимаций
  (tombstone, чтобы индексы каналов остались валидными);
- очередь разрушения при OOM не трогает `sg.*` вне контекста (фиксированный
  overflow, err-logged leak только при полном исчерпании);
- glTF `parsePrimitive` вне контекста: без `sg.*`, флаги
  `pending_dynamic_update` / `morph_upload_pending`, полный `errdefer`;
- `uploadGeometry`: единый errdefer, `GpuBufferAllocationFailed` вместо
  мёртвых handles; morph delta валидация, лог первой ошибки, базовая поза
  в кадре 1 для отложенных CPU-морфов.

### Волна 4 — off-context загрузка (`222722b`)
- синхронное создание текстур требует графический поток (assert в точках
  `Texture.fromRaw/fromMemory/fromFile`); off-context GLB-загрузка
  автоматически переходит на async-текстуры при наличии `scene.uploads`;
- `spawnMeshes` больше не требует контекст; исправлена утечка имени при OOM;
- `Scene.deinit` разрушает физику до мешей (тела держат raw `mesh`-указатели);
- `--test-async-load`: `gpu_pending=1 → 0`, `textured=1` — полный off-context
  путь подтверждён runtime-харнессом;
- `zig build fmt` впервые зелёный (7 файлов отформатированы).

### Волна 5 — оставшиеся пункты плана
- instanced picking: `pickWithRay` тестирует каждый видимый инстанс в
  нарисованном пространстве (`cached_world_matrix`/`cached_bounding_box`
  после `updateCachedTransforms`); скрытый источник с видимыми инстансами
  пикается так же, как рисуется; `PickingInfo.picked_instance` — сырой
  индекс; `gpu_pending`-меши не пикаются;
- `UploadQueue`: стековый чанк вместо аллокации под спинлоком, бюджет
  `upload_budget_per_frame`, счётчики текстуры/байт;
- `Scene.io_runner`: отдельный `TaskRunner` для async save/load — декод
  текстур больше не голодает на файловом I/O;
- метрики фаз (`update/prepare/shadow/main/post`) и аплоадов в `SceneStats`,
  `--stats` в agate;
- sandbox `--test-stress`: 200+ мешей, instanced-группа 1×64, CPU-частицы
  cap 5000, churn create/destroy + декали на игровом потоке, 240 кадров,
  `errors=0`, без графических ассертов.

### Волна 6 — OIT, multi-producer forkJoin, дедупликация текстур и сохранения
- **Per-instance transparency sorting (OIT)**: в `submitInstancedMesh` для прозрачных
  инстансированных мешей (`materialIsTransparent` или `is_decal`) матрицы инстансов
  сортируются строго back-to-front относительно `ctx.eye` с детерминированным
  тай-брейком;
- **Multi-producer jobs.Pool.forkJoin**: добавлен `dispatch_mutex: Mutex` в `jobs.Pool`,
  обеспечивающий потокобезопасность при одновременных диспатчах из игрового и
  рендеринг-потоков без повреждения mailbox воркеров;
- **Дедупликация текстур в UploadQueue**: добавлены `findFile` и `getOrRequestFile`,
  предотвращающие дублирование декодирования и загрузки одинаковых текстур; слоты
  уже загруженных текстур патчатся немедленно;
- **Персистентность сохранений сцены (формат v3)**:
  - Устойчивые идентификаторы сущностей (`Mesh.id: u64`);
  - Сохранение и восстановление иерархии `parent`/`child` мешей;
  - Произвольные строковые свойства игры (`SceneState.game_properties`, `setGameProperty`, `getGameProperty`);
  - Полная обратная совместимость со старыми файлами формата v2.

### Волна 7 — встроенный профилировщик и снапшоты памяти (Profiler / Flight Recorder)
- **Встроенный модуль Profiler**:
  - Запись кадровых метрик в реальном времени с разбиением по фазам (`Update`, `Prepare`, `Shadow Pass`, `Main Pass`, `PostFX`), подсчетом draw calls, полигонов, смен шейдерных пайплайнов и объема загрузок VRAM;
  - Снапшоты памяти (`MemorySnapshot`): полный учет распределения памяти CPU и VRAM GPU (текстуры, меши/буферы, таргеты рендера и теневые карты);
  - Экспорт в интерактивный HTML-отчет (темная тема, интерактивный график-таймлайн SVG со стаком фаз, top spike frames, таблицы VRAM ассетов);
  - Экспорт в подробный Markdown-отчет (`.md`);
  - Экспорт в Chrome Trace Event JSON (`.json`) для анализа в `chrome://tracing` и `ui.perfetto.dev`;
  - Автоматическая диагностика узких мест («Что не так»): автоматический анализ просадок FPS, длинных фаз рендера, избытка draw calls, частых переключений пайплайнов, тяжелых несжатых текстур 2K+ и высокого потребления VRAM с выдачей конкретных рекомендаций;
  - Интеграция в Scene API (`scene.startProfiling()`, `scene.stopProfiling()`, `scene.saveProfileReports("...")`);
  - Интеграция в демо agate и sandbox: горячая клавиша `F8` для включения/выключения записи на лету и CLI-флаг `--profile`;
  - Число тестов выросло с 601 до 605 (все проходят в Debug и ReleaseSafe без утечек памяти).

### Волна 8 — строго неблокирующий рендер и полный per-item payload
- **Автономный per-item payload рендера**:
  - `RenderMeshItem` и `RenderInstancedBatch` инкапсулируют все GPU буферы (`vertex_buffer`, `index_buffer`, `instance_buffer`), параметры материалов (`MaterialDrawRecord`), morph/skin данные и матрицу трансформации;
  - Полностью исключены любые разыменования указателей `*Mesh` во время выполнения `Scene.render()`;
  - `ShadowDrawItem`: выделена отдельная структура для теневого прохода, теневые бакеты формируются в `ShadowPass.prepare()` на фазе `prepareFrame()`, а `ShadowPass.renderPrepared()` рисует исключительно по сформированным элементам;
  - `OutlineDrawItem`: предварительный сбор контуров выделения в `prepareFrame()`; `renderOutlineItems` рисует контуры без обращения к живым мешам сцены.
- **Разделение фаз подготовки и отрисовки**:
  - Очереди отрисовки для всех камер (`self.queues`, `self.view_queues`) строятся строго в `prepareFrame()` через `prepareViewQueues`;
  - Сброс статистики кадра, инкремент `frame_id`, обработка аплоадов и очистка очередей перенесены в `prepareFrame()`;
  - `Scene.render()` выполняет только отправку draw-call'ов в Sokol GFX на основе готовых иммутабельных структур.
- **Сужение phase_mutex до prepareFrame()**:
  - В `agate` и `sandbox` `phase_mutex` захватывается только на время `prepareFrame()` (плюс генерация UI-вершин в sandbox);
  - `Scene.render()` работает 100% lock-free и не блокирует параллельный цикл симуляции (~1 кГц) на игровом потоке;
  - Проверено стресс-тестом на 240 кадров (`--test-stress`: 4160 созданий и 3895 удалений мешей на игровом потоке во время concurrent render'а, 0 ошибок, 0 ассертов);
  - Все 605 тестов стабильно проходят в Debug и ReleaseSafe.

### Инварианты (закреплены ассертами и тестами)
1. `sg.*` — только на графическом потоке; иначе отложить и завершить на
   render-стороне.
2. Ошибка аллокатора не публикует невалидные GPU-handles и не течёт.
3. Удаление объекта разрывает все ссылки на него до освобождения памяти.
4. Mailbox всегда отдаёт самое новое состояние; устаревшие кадры отбрасываются.
5. Тест-реестр — источник правды для `zig build test`; новый файл требует
   `zig build update-tests`.
