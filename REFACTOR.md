# Threading refactor: status & roadmap

Status as of 2026-09-15. Legend: [x] done, [~] partial, [ ] planned.

## Status at a glance

| Stage | Scope | Status |
|---|---|---|
| 1. Data-parallel CPU systems | job pool, particles, culling | [x] done |
| 2. Async assets | TaskRunner, UploadQueue, glTF async textures | [x] done |
| 3. Simulation/render decoupling | threads, coarse phase ownership, partial payload | [~] safe; non-blocking pending |

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

Known limitations of the shipped split:

- **Coarse payload.** Phase ownership means an update spike delays render
  for its duration. Strictly non-blocking render needs the full per-item
  payload (below).

## What remains (TODO, in priority order)

1. [ ] **Full per-item frame payload / strictly non-blocking render.**
   Partial groundwork exists: `MaterialDrawRecord`, double-buffered skin
   palettes, morph data in `RenderMeshItem`, plus `CameraSnapshot` and
   `SceneFrameSnapshot` for camera/light/pass state. It is not a complete
   render snapshot: queue construction, shadows, instancing and draw paths
   still read live `Mesh`/`Material` state and write transform caches.
   Therefore `phase_mutex` intentionally spans `prepareFrame() + render()`.
   Unlock render only after every pass consumes immutable per-frame records
   with explicit resource lifetime guarantees.
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

1. Матрица интеграционных сцен: threaded/serial, regular/instanced/skinned/morph,
   opaque/cutout/blend/double-sided, одна камера/PIP, post/MSAA,
   загрузка/изменение/удаление ресурсов.
2. README, CI и проверка реально fetched/archive-пакета; в `roadmap.md` есть устаревшие статусы
   (например, DoF/MSAA помечены отсутствующими при наличии кода).
3. Контракты владения и отложенное освобождение ресурсов до дальнейшего
   распараллеливания.
4. Сохранение игры: устойчивые ID, связи объектов, игровое состояние, версия
   формата. Текущий serializer намеренно хранит только состояние существующей
   сцены (`src/agate/serialization.zig:55-61`).
5. Метрики: CPU simulation/render submission отдельно, ожидание mutex/jobs,
   p95/p99, upload bytes/frame, память текстур/геометрии.
6. Asset pipeline раньше новых эффектов: mip-цепочки, сжатые текстуры, кэш импорта.

### Порядок работ

1. [x] Стабилизация: потоки, `spawn`, кубмапы, bounds-check анимаций.
2. [~] Воспроизводимость: package paths и test-registry готовы; README/CI и
   archive-consumer check остаются.
3. [~] Согласованность рендера: прозрачная очередь и viewport-aware picking
   готовы; visual-regression сцены остаются.
4. Одна небольшая законченная игра как проверка движка.
5. Оптимизация только по профилю.

### Выполненные проверки

```sh
# В agate/: успешно
zig build update-tests                 # повторный запуск идемпотентен
zig build test                         # 563/563
zig build test -Doptimize=ReleaseSafe # 563/563
zig build
zig build -Doptimize=ReleaseSafe
zig build run -- --frames 30
zig build run -- --frames 30 --no-threads
zig build run -- --frames 30 --msaa 4

# В sandbox/: успешно
zig build
zig build run -- --frames 120
zig build run -- --test-pip
zig build run -- --test-save
zig build bench -Doptimize=ReleaseFast
```

Изменённые Zig-файлы проходят `zig fmt --check`. Общий `zig build fmt` пока
падает на семи неизменённых ранее неформатированных файлах; они не затрагивались.
Sandbox dependency-build не изменяет `src/agate/tests.zig`. Отдельный
`zig fetch .` дважды завис без вывода, поэтому fetched/archive package и ручная
визуальная проверка прозрачности остаются незакрытыми проверками.
