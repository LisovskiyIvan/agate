# Архитектура движка Agate

> Путь: src/ (root.zig — фасад) · Импорт: agate.X (root.zig) · Потоки: game producer / context render / audio / io_runner

## Что это

Agate — Babylon.js-style 3D-движок на Zig поверх sokol (`sokol.app` + `sokol.gfx` + `sokol.glue`). `src/root.zig` — фасад библиотеки: всё публичное достижимо через `@import("agate")`, модульные пути (`agate.mesh.Mesh`, `agate.scene.Scene`) остаются для точечного доступа. Соглашения имён: `Type.new(...)`/`Type.init(name, options)` для значений, `create*` для Scene/GPU-владеющих сущностей, `make*` для sokol-обёрток в духе `sg.makePipeline`, `build*Data → GeometryData` для чистой сборки без GPU-эффектов, `*Options` для дефолтных ручек, `*Desc` для регистрационных дескрипторов с обязательными полями, `*Params` для вычисленных покадровых паков.

`scene.zig` — оркестратор, разбитый на владельца (`scene/core.zig`, тип `Scene`) и листья `scene/` (free-функции + форвардеры; листья берут сцену как `anytype` и никогда не импортируют фасад назад — то же правило, что в `audio/*` и `profiler/*`). Живые примеры-приложения — `examples/` (RTT/gpu-timing/HDR, включая веб-варианты) и интерактивный sandbox в отдельном репозитории `../sandbox`; шейдеры компилируются в build-time через sokol-shdc.

## Быстрый старт

```zig
const agate = @import("agate");

var gpa = std.heap.DebugAllocator(.{ .thread_safe = true }){};
var scene: agate.Scene = undefined;
scene.initInto(gpa.allocator());
defer scene.deinit();

// Слои снизу вверх: math → mesh/material → scene → runtime.
scene.active_camera = .{ .arc_rotate = agate.ArcRotateCamera.init("cam", .{}) };
const mat = try scene.createPBRMaterial("wall");
const box = try agate.MeshBuilder.createBox(&scene, "box", .{});
box.material = .{ .pbr = mat };

// Кадр: game пишет, context готовит и рисует (см. ./runtime.md).
var runtime: agate.Runtime = agate.Runtime.init();
defer runtime.deinit();
// game: _ = runtime.update(&scene, &tick_ctx, Tick.run); // simulate + producer build
// context: switch (runtime.renderFrame(&scene)) { .prepared, .reused, .skipped, .busy }
```

Импорт всегда `const agate = @import("agate");` (модуль `agate` собирается в `build.zig` из `src/root.zig` + сгенерированных шейдер-модулей + `math` + `shader_material_registry`).

## API

Здесь нет единого API — это карта. Слои движка:

| Слой | Модули | Роль |
|---|---|---|
| Фундамент | `math`, `tags`, `jobs`, `observable`, `gpu_thread`, `gpu_timing`, `gpu_upload_meter`, `c` | Векторы/матрицы/фрустум; fork-join пул; события; маркер контекстного потока; GPU-тайминги; счётчик аплоадов; C-декодеры |
| Данные и ресурсы | `mesh`, `material`, `node_material`, `shader_material`, `material_library`, `texture`, `ktx2`, `dds`, `exr`, `ttf`, `compute` | Геометрия/CPU-зеркала; материалы; hooked шейдер-материалы; контейнеры текстур; compute-поддержка |
| Загрузка/выгрузка | `loader/*`, `export/*`, `assets`, `asset_manager` | glTF/OBJ/PLY/STL; OBJ/STL/GLB-экспорт; async-текстуры; менеджер задач |
| Симуляция | `physics`, `physics_mesh`, `ragdoll`, `vehicle`, `softbody`, `animation/*`, `ai`, `particles` | Box3D-миры/тела/джойнты; скелеты/ретаргет; navmesh/crowd; CPU/GPU-частицы |
| Сцена | `scene`, `scene/*`, `camera`, `lights`, `visibility/*` | Реестры, камеры, свет, очереди, кадры, пиккинг, снапшоты |
| Кадр и рендер | `runtime`, `handoff`, `passes/*`, `postprocess`, `ssao` | Жизненный цикл кадра; newest-wins mailboxes; проходы; пост-эффекты |
| Окружение | `audio`, `ui`, `profiler`, `serialization` | Аудио-движок/шины/стримы; canvas-виджеты; профайлер; capture/restore |

### Модель потоков

| Поток | Владеет | Никогда не трогает |
|---|---|---|
| Game producer (один, обязателен) | `simulate`, живые регистры, mailboxes, `pending_update_ms`, claim→build→stage→publish | `stats`, `Profiler`, prepared payloads, `sg.*` |
| Context render (sokol-app колбэк) | `begin/finish/prepare/render`, `stats`, GPU-объекты, retire-flush | Живые game-регистры (только замороженные слоты) |
| Audio | `AudioEngine`, шины, голоса, стримы | Графические очереди |
| io_runner (`jobs.TaskRunner`, 1 поток) | Декоды текстур, save/load кодирование + file IO | Живую сцену (только снапшоты) |
| Jobs-воркеры (`jobs.Pool`, до 8) | CPU-чанки `parallelFor` (курсор-атомик) | `sg.*`, shared state, вложенный `parallelFor` |

Фазовый мьютекс (`Runtime.mutex`, `jobs.Mutex`): update-vs-begin exclusion. По умолчанию staged begin идёт lock-free (см. ниже); `setProducerExclusion(true)` — только mutex-диагностика тех же данных/алгоритма. `finish`/`render` перекрываются со следующим update. `sg.*` — только контекстный поток (`gpu_thread.markContextThread` один раз до `Scene.initInto`; `assertOnContextThread` на входах render-фазы; headless без маркера — только CPU-cleanup).

### Lock-free staged prepare (одной страницей; детали — `./frame-pipeline.md`)

Слоты (`FrameDraws`, тройной буфер переменных списков с pin/lease): продюсер `tryClaimBuildSlot → build → stageUi → stageHostBytes → publish` (или `Scene.buildPreparedFrame`) замораживает payloads в слот и релизит поколение (`build_seq`); контекст `beginStagedPrepare` лэтчит свежее поколение без мьютекса (только slot-owned + context-owned чтения), `finishStagedPrepare` публикует front, `render` рисует подготовленное и никогда не готовит свежий кадр сам, `renderReuse` перепрезентует валидный front при пустом begin. Живые dirty-флаги/скаляры потребляет game-side commit, host live-reads едут замороженными `host_bytes` в claim. Пустой begin — никогда не live-fallback: reuse/skip. Каждый begin — ровно один finish/cancel (one-shot, живой GPU-владелец).

### Владение GPU-ресурсами (отложенное создание/уничтожение, retire)

- Создание вне контекста — CPU-only + deferred: mesh держит CPU-зеркала + `pending_vertices` (`gpu_pending = true`), `finishGpuUpload` добилдивает буферы на следующем context-flush (`flushPendingGpuUploads` в начале кадра).
- Уничтожение вне контекста — unlink сейчас + epoch-stamped retire в `gpu_retire` (`retireMesh`/`retireBuffer` с любого потока, без `sg.*` и free); контекстный flush на render-start уничтожает due-записи после завершения их эпохи; `deinit` дренирует всё включая незавершённые эпохи. Удаление никогда не инвалидирует pending prepared-кадр (game-side destroy под update-исключением идёт через retire, не free in-flight handles).
- Новым kind'ам записей — только в `GpuRetireQueue`, новых очередей в `Scene` не заводить (tripwire).
- PBR-дедуп views по id; у частиц осознанно нет destroy (см. `./scene.md`, `./particles.md`).

### Сборка и инструменты (build.zig)

- Модуль `agate`: `root.zig` + N сгенерированных шейдер-модулей (`pbr`, `skinned_pbr`, `instanced_pbr`, `shadow`, `msaa_depth`, `velocity`, `skybox`, `postprocess`, `particle`, `particle_compute`, `ui`, `ssao`, `ssao_blur`, `debug`, `bloom_down/up`, `glow_extract/blur`, `volumetric_raymarch/blur`, `outline`, `probe_mip`, `ui3d_panel`, `depth_pyramid`) + `math` + `shader_material_registry`. Единый slang для всех ног — `engine_shader_slang` (`glsl430` + `hlsl5` + `metal_macos` + `wgsl`; формат floor — Metal/WebGPU/D3D11/GL 4.3, без glsl410/WebGL-фолбэка — SSBO-блокам нужен GLSL 4.30+); `// @include` раскрываются хост-препроходом `expand_shader_includes`.
- Shader-material registry: таблица `user_shader_materials` (hook-уровень: имя + snippet + base `standard/pbr`) → merge-tool встраивает сниппет в базовый шаблон по hook-маркерам → shdc (тот же `engine_shader_slang`) → генерированный `shader_material_registry` (имя/key-Wyhash/base/UB-индексы/params/`make_shader`). Плюс user-owned путь `compileUserShader` для downstream-проектов без правок движка (тот же sokol-инстанс через `dep_agate`).
- C/C++: `c_impl.c` (stb, `-DSTBI_NEON` на aarch64), Box3D v0.1.0 (C17), meshoptimizer decoder-subset (C++, без исключений/RTTI), BasisU transcoder + zstd (transcode KTX2 в `ktx2.zig`).
- Тестовый реестр `src/tests.zig` — GENERATED обходом дерева (`zig build update-tests`, затем `zig build test` с CheckFile-гейтом; обычные сборки реестр не переписывают). Fuzz: вендорный раннер `tools/test_runner.zig`, `zig build test --fuzz[=limit]`; C-флаги гасят sancov-инструментацию (`no_sancov`).
- Примеры `examples/` (`zig build run-rtt / run-gpu-timing / run-hdr / run-runtime-worker`, нужны GPU/дисплей): RTT+refraction smoke, GPU timing/lifecycle gate, HDR-showcase и threaded staged-фрейм (`Runtime.spawnWorker` — lock-free producer на игровом потоке; wasm/single-thread деградирует в inline serial). У gpu-timing и hdr есть веб-варианты (`*_web.zig`). Интерактивное демо всего движка — отдельный репозиторий `../sandbox` (зоны, физика, аудио, UI, save/load).

### Веб-таргет (wasm32-emscripten + WebGPU)

Экспериментальный сборочный таргет: движок и примеры собираются под `wasm32-emscripten` и рисуют через WebGPU (бэкенд sokol WGPU). Это инструмент для веб-сравнения с Babylon.js, а не продуктовая веб-платформа.

```sh
# в agate/examples/: gpu-timing и hdr-showcase имеют *_web.zig варианты
zig build -Dtarget=wasm32-emscripten -Doptimize=ReleaseFast
```

- `build.zig`: `is_web = target.result.cpu.arch.isWasm()`; WebGPU форсируется на вебе (`.wgpu = is_web`), опции `-Dwgpu` нет — stray-флаг это unknown-option error, нативный WebGPU не поддерживается (натив — sokol auto backend: macOS Metal, Windows D3D11, Linux GL). Для wasm добавляются system-include-пути emsdk (включая webgpu-порты при WGPU); C-флаги Box3D/вендоров зависят от `is_web`. `pub fn getEmsdk(dep_agate)` отдаёт downstream-сборкам тот же emsdk; при `-Doptimize=Debug` на wasm C-часть собирается `-O2`.
- Downstream-линковка (как в `../sandbox/build.zig`): статическая библиотека через `agate_build.sokol.emLinkStep` (`use_webgpu = true`, emmalloc, preload-file с ассетами, shell из `agate_build.sokolShellPath(dep_agate)`), `emRunStep` даёт `zig build run` в браузере.
- Уже работает: WebGPU-бэкенд через sokol WGPU, примеры и sandbox собираются и запускаются в браузере, 32-битная wasm-совместимость.
- Не входит: JS/TS API, DOM/HTML, npm, WebXR (см. `../roadmap.md`, раздел 🟡/🚫).

### Тесты и гейты

- `zig build test` — юнит-реестр (все `test`-блоки дерева; math отдельно) + `zig fmt --check` гейт (`zig build fmt`). Гейт зелёный = exit 0; строка `failed command` в логе раннера — предсуществующий шум harness (см. `../MEASUREMENTS.md`).
- Headless-прогон без GPU/окна: `agate.App.runHeadless` (юнит-тесты движка уже идут headless: `sg.isvalid()` early-out в проходах).
- Headed GPU-smoke: `examples/` — `run-rtt` (render target + refraction), `run-gpu-timing` (timing/lifecycle gate из `../MEASUREMENTS.md`), `run-hdr` (HDR-студия). Perf-гейты (bench-threads CPU + gpu-timing GPU, правило «только back-to-back внутри сессии») — в sandbox, см. `../MEASUREMENTS.md`.
- Сериализация/экспорт: fuzz-корпуса (`serialization_fuzz.zig`), round-trip тесты.

### Карта модулей

| Doc | Файлы | Что это |
|---|---|---|
| `./architecture.md` | `root.zig`, `build.zig`, `examples/` | Эта карта: слои, потоки, владение, сборка, гейты |
| `./runtime.md` | `runtime.zig`, `handoff.zig`, `gpu_thread.zig`, `jobs.zig`, `observable.zig`, `gpu_timing.zig`, `gpu_upload_meter.zig` | Жизненный цикл кадра, mailbox newest-wins, affinity-маркер, job pool, события, GPU-тайминги |
| `./scene.md` | `scene.zig`, `scene/lifecycle.zig`, `content.zig`, `registry.zig`, `stats.zig`, `sim_api.zig`, `query_api.zig`, `lights_api.zig`, `profile_api.zig`, `snapshot.zig`, `cameras.zig`, `attachments.zig` | Реестры контента, update-проходы, pick/query, API-фасады |
| `./frame-pipeline.md` | `scene/frame_api.zig`, `frame_build.zig`, `frame_prepare.zig`, `frame_render.zig`, `frame_draws.zig`, `upload_packets.zig`, `instance_staging.zig`, `gpu_retire.zig` | Слоты, freeze→commit, host_bytes, staged prepare, retire-эпохи |
| `./render-pipeline.md` | `scene/render_queue.zig`, `render_queue/`, `queue_builder.zig`, `draw.zig`, `pipelines.zig`, `forward_pipelines.zig`, `view_render.zig`, `frame_draws.zig`, `project_cache.zig`, `uniforms.zig`, `msaa.zig`, `viewport_clear.zig` | Очереди, вьюхи, пайплайны, MSAA, юниформы |
| `./scene-layers.md` | `scene/sky_layer.zig`, `probe_layer.zig`, `probe_render.zig`, `decal_layer.zig`, `trail_layer.zig`, `nav_layer.zig`, `particle_layer.zig`, `physics_layer.zig`, `animation_runtime.zig`, `gui3d_layer.zig`, `highlight_layer.zig`, `light_rig.zig`, `light_selection.zig`, `clustered_lights.zig`, `shadow_system.zig`, `shadow_pcss.zig`, `cascades.zig`, `postfx_stack.zig`, `draw.zig`, `patch_instance_refs.zig` | Слои сцены: небо, пробы, декали, трейлы, тени, пост-стек |
| `./mesh.md` | `mesh.zig`, `mesh/*`, `physics_mesh.zig` | Меши, билдеры, VAT, morph/skin, LOD/simplify, greased-line/trail |
| `./material.md` | `material.zig`, `material_library.zig`, `node_material.zig`, `shader_material.zig`, `shader_material/*` | Standard/PBR/shader-материалы, hook-merge, граф→сниппет, пресеты |
| `./lights.md` | `lights.zig`, `lights/*` | Источники света, солнце, кластеризация, area-lights |
| `./cameras.md` | `camera.zig`, `camera/*` | Free/arc-rotate/follow/target/fly, rig, проекции |
| `./texture.md` | `texture.zig`, `texture/*`, `ktx2.zig`, `dds.zig`, `exr.zig` | Текстуры, кубы, skybox, KTX2/DDS/EXR-декодеры |
| `./loader.md` | `loader/*` | glTF/OBJ/PLY/STL загрузчики и scene-append |
| `./export.md` | `export/*` | OBJ/STL/GLB-экспорт (`write*Alloc`) |
| `./assets.md` | `assets.zig`, `assets/*`, `asset_manager.zig` | Async-текстуры, менеджер задач, кэш |
| `./physics.md` | `physics.zig`, `physics/*`, `ragdoll.zig`, `vehicle.zig` | Миры, тела, джойнты, персонажи, рэгдолл, транспорт |
| `./particles.md` | `particles.zig`, `particles/*` | CPU/GPU-частицы, саб-эмиттеры, коллайдеры, потоки |
| `./animation.md` | `animation/*` | Скелеты, клипы, easing, ретаргет, VAT-плеер |
| `./ai.md` | `ai.zig`, `ai/*` | Navmesh, string-pull, агенты, crowd |
| `./math.md` | `math.zig`, `math/*` | Векторы, матрицы, кватернионы, фрустум, лучи |
| `./audio.md` | `audio.zig`, `audio/*` | Движок, клипы, голоса, шины, фильтры, реверб, стримы, окклюзия |
| `./ui.md` | `ui.zig`, `ui/*`, `ttf.zig`, `ttf/*` | Canvas, лейаут/flex/grid, стили/CSS, шрифты, глифы |
| `./profiler.md` | `profiler.zig`, `profiler/*` | Кадровые записи, отчёты, memory snapshot, диагностика |
| `./serialization.md` | `serialization.zig`, `serialization/*` | Capture/restore, sync/async save/load |
| `./visibility.md` | `visibility/*` | HiZ, software-растеризатор, occlusion culler |
| `./shaders.md` | `shaders/*.glsl`, `shaders/common/*`, `compute.zig` | GLSL-шаблоны, общие чанки, compute-матрица |
| `./postprocess.md` | `postprocess.zig`, `postprocess/*` | Тонемэппинг, LUT, bloom/glow/volumetric knobs |
| `./passes.md` | `passes/*` | Debug/bloom/highlight/outline проходы |

## Потоки и владение

См. таблицу потоков выше и `./runtime.md`. Коротко: один game-продюсер пишет живое; контекст готовит/рисует замороженное; аудио и io_runner изолированы; jobs-воркеры — чистые CPU-чанки. GPU-объекты создаются/уничтожаются только на контексте (deferred + retire-очередь с эпохами). `stats`/`Profiler` — context-owned; update-сторона пишет только staged атомики и mailboxes.

## Ошибки и краевые случаи

- Нарушение контракта потоков (game пишет `stats`, воркер зовёт `sg.*`, registry-mutate поперёк latch) — баг приложения; tripwire'ы (`assertOnContextThread`, commit guards) делают его громким, а не тихим.
- Два продюсера, вложенный `parallelFor`, `sg.*` в job — запрещены контрактом (см. `./runtime.md`).
- Headless (нет `sg.setup`): все GPU-пути fail-closed (тайминги 0, аплоады CPU-only + deferred, render early-out после эпохи).
- OOM в реестрах/слотах — fail-closed (null/skip/drain, счётчики дропов: `SceneStats.build_oom_drops` для очередей/аплоадов), не паника. Нарушение пейринга claim-токенов (stale/double finish) — warn-лог + защитный release активного claim: кадр теряется, конвейер не клинится.

## Производительность

- Горячий путь кадра: O(видимые меши) в queue-build (параллелится пулом выше 4096), O(слоты) в handoff-скане, O(1) в mailbox-операциях; поиск по имени/тегу — линейный, вне горячего пути.
- Текстурный стриминг троттлится бюджетом 8 MiB/кадр (`uploaded_bytes_frame`); динамические буферные апдейты — uncounted (`updated_bytes_frame`).
- GPU-тайминги default OFF; профайлер пишет только в конце `render`; сериализация/отчёты — на io_runner.

## Смотрите также

- `./runtime.md` — фасад кадра и примитивы потоков
- `./scene.md` — контент и фасады сцены
- `./frame-pipeline.md` — стадии кадра детально
- `./render-pipeline.md` — очереди и проходы
- `./scene-layers.md` — слои сцены
- `./math.md` — математический фундамент
- `./profiler.md` — измерения и гейты
- `./serialization.md` — сейвы и восстановление
- `./assets.md` — фоновые загрузки и кэш задач
