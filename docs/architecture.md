# Архитектура движка Agate

> Путь: src/agate/ (root.zig — фасад) · Импорт: agate.X (root.zig) · Потоки: game producer / context render / audio / io_runner

## Что это

Agate — Babylon.js-style 3D-движок на Zig поверх sokol (`sokol.app` + `sokol.gfx` + `sokol.glue`). `src/agate/root.zig` — фасад библиотеки: всё публичное достижимо через `@import("agate")`, модульные пути (`agate.mesh.Mesh`, `agate.scene.Scene`) остаются для точечного доступа. Соглашения имён: `Type.new(...)`/`Type.init(name, options)` для значений, `create*` для Scene/GPU-владеющих сущностей, `make*` для sokol-обёрток в духе `sg.makePipeline`, `build*Data → GeometryData` для чистой сборки без GPU-эффектов, `*Options` для дефолтных ручек, `*Desc` для регистрационных дескрипторов с обязательными полями, `*Params` для вычисленных покадровых паков.

`scene.zig` — оркестратор, разбитый на владельца (`scene/core.zig`, тип `Scene`) и листья `scene/` (free-функции + форвардеры; листья берут сцену как `anytype` и никогда не импортируют фасад назад — то же правило, что в `audio/*` и `profiler/*`). Демо — `src/main.zig` (бинарник `agate`), шейдеры компилируются в build-time через sokol-shdc.

## Быстрый старт

```zig
const agate = @import("agate");

var gpa = std.heap.DebugAllocator(.{ .thread_safe = true }){};
var scene: agate.Scene = undefined;
scene.initInto(gpa.allocator());
defer scene.deinit();

// Слои снизу вверх: math → mesh/material → scene → runtime.
var cam = agate.ArcRotateCamera.new(...);
try scene.addCamera(.{ .name = "main", .camera = .{ .arc_rotate = cam } });
const mat = try scene.createStandardMaterial("wall");
const box = try agate.MeshBuilder.createBox(&scene, "box", .{});
box.material = .{ .standard = mat };

// Кадр: game пишет, context готовит и рисует (см. ./runtime.md).
var runtime: agate.Runtime = agate.Runtime.init();
defer runtime.deinit();
scene.publishFrameSnapshot(aspect, w, h);
_ = runtime.produceBuild(&scene);
const begun = runtime.beginPrepare(&scene);
if (begun.claim) |c| { runtime.finishPrepare(&scene, c); scene.render(); }
```

Импорт всегда `const agate = @import("agate");` (модуль `agate` собирается в `build.zig` из `src/agate/root.zig` + сгенерированных шейдер-модулей + `math` + `shader_material_registry`).

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

Фазовый мьютекс (`Runtime.mutex`, `jobs.Mutex`): update-vs-begin exclusion. По умолчанию staged begin идёт lock-free (см. ниже); `setProducerExclusion(true)` возвращает окно исключения; `prepareSerial` лочит всегда. `finish`/`render` перекрываются со следующим update. `sg.*` — только контекстный поток (`gpu_thread.markContextThread` один раз в init до спавна; `assertOnContextThread` на входах render-фазы; тесты без маркера — синхронный fallback).

### Lock-free staged prepare (одной страницей; детали — `./frame-pipeline.md`)

Слоты (`FrameDraws`, тройной буфер переменных списков с pin/lease): продюсер `tryClaimBuildSlot → build → stageUi → stageHostBytes → publish` замораживает payloads в слот и релизит поколение (`build_seq`); контекст `beginStagedPrepare` лэтчит свежее поколение без мьютекса (только slot-owned + context-owned чтения), `finishStagedPrepare` публикует front, `render` рисует, `renderReuse` перепрезентует front при пустом begin. Живые dirty-флаги/скаляры потребляет game-side commit, `Scene.lock_free_prepare` гасит UI live-fallbacks, host live-reads едут замороженными `host_bytes` в claim. Пустой begin — никогда не live-fallback: reuse/skip. Каждый begin — ровно один finish/cancel.

### Владение GPU-ресурсами (отложенное создание/уничтожение, retire)

- Создание вне контекста — CPU-only + deferred: mesh держит CPU-зеркала + `pending_vertices` (`gpu_pending = true`), `finishGpuUpload` добилдивает буферы на следующем context-flush (`flushPendingGpuUploads` в начале кадра).
- Уничтожение вне контекста — unlink сейчас + epoch-stamped retire в `gpu_retire` (`retireMesh`/`retireBuffer` с любого потока, без `sg.*` и free); контекстный flush на render-start уничтожает due-записи после завершения их эпохи; `deinit` дренирует всё включая незавершённые эпохи. Удаление никогда не инвалидирует pending prepared-кадр (game-side destroy под update-исключением идёт через retire, не free in-flight handles).
- Новым kind'ам записей — только в `GpuRetireQueue`, новых очередей в `Scene` не заводить (tripwire).
- PBR-дедуп views по id; у частиц осознанно нет destroy (см. `./scene.md`, `./particles.md`).

### Сборка и инструменты (build.zig)

- Модуль `agate`: `root.zig` + N сгенерированных шейдер-модулей (`standard`, `pbr`, `skinned_pbr`, `instanced`, `instanced_pbr`, `shadow`, `msaa_depth`, `skybox`, `postprocess`, `particle`, `particle_compute`, `ui`, `ssao`, `ssao_blur`, `debug`, `bloom_down/up`, `glow_extract/blur`, `volumetric_raymarch/blur`, `outline`, `probe_mip`, `ui3d_panel`) + `math` + `shader_material_registry`. Slang по умолчанию `glsl410/metal_macos/hlsl5`; forward-шейдеры со storage-блоками и compute — `glsl430/...`; `// @include` раскрываются хост-препроходом `expand_shader_includes`.
- Shader-material registry: таблица `user_shader_materials` (hook-уровень: имя + snippet + base `standard/pbr`) → merge-tool встраивает сниппет в базовый шаблон по hook-маркерам → shdc (`glsl430/metal_macos/hlsl5`) → генерированный `shader_material_registry` (имя/key-Wyhash/base/UB-индексы/params/`make_shader`). Плюс user-owned путь `compileUserShader` для downstream-проектов без правок движка (тот же sokol-инстанс через `dep_agate`).
- C/C++: `c_impl.c` (stb, `-DSTBI_NEON` на aarch64), Box3D v0.1.0 (C17), meshoptimizer decoder-subset (C++, без исключений/RTTI), BasisU transcoder + zstd (transcode KTX2 в `ktx2.zig`).
- Тестовый реестр `src/agate/tests.zig` — GENERATED обходом дерева (`zig build update-tests`, затем `zig build test` с CheckFile-гейтом; обычные сборки реестр не переписывают). Fuzz: вендорный раннер `tools/test_runner.zig`, `zig build test --fuzz[=limit]`; C-флаги гасят sancov-инструментацию (`no_sancov`).
- Демо-бинарник `agate` (`src/main.zig`, `zig build run`): threaded game/context, CLI `--frames/--particles/--msaa/--stats`; headless-smoke `agate --frames 120 --msaa 4` без sokol validation errors.

### Веб-таргет (wasm32-emscripten + WebGPU)

Экспериментальный сборочный таргет (`b54861e`, 30.09.2026): движок, sandbox и бенч собираются под `wasm32-emscripten` и рисуют через WebGPU (бэкенд sokol WGPU). Это инструмент для веб-сравнения с Babylon.js, а не продуктовая веб-платформа.

```sh
# sandbox/ — сборка в sandbox/zig-out/web/ (sandbox.html/js/wasm/data)
zig build -Dtarget=wasm32-emscripten -Doptimize=ReleaseFast
zig build -Dtarget=wasm32-emscripten -Dweb-debug   # оставить Debug (иначе Debug флорится в ReleaseFast)
```

- `build.zig` (агент): `is_web = target.result.cpu.arch.isWasm()`; `opt_wgpu = -Dwgpu orelse is_web` (строка ~275–276). Зависимость sokol подключается с `.wgpu = is_web` — тот же флаг прокидывается в `compileUserShader` для downstream-шейдеров (`build.zig` ~129–133). Для wasm добавляются system-include-пути emsdk: `upstream/emscripten/cache/sysroot/include`, `.../include/c++/v1` и, при WGPU, `.../cache/ports/emdawnwebgpu/emdawnwebgpu_pkg/webgpu/include` (~425–430); C-флаги Box3D/вендоров зависят от `is_web` (~520). `pub fn getEmsdk(dep_agate)` (~150) отдаёт downstream-сборкам тот же emsdk; при `-Doptimize=Debug` на wasm C-часть собирается `-O2` (нативный Debug оставляет `-O0`).
- `sandbox/build.zig`: на веб-таргете root-модуль — `src/web_main.zig` вместо `main.zig`, статическая библиотека линкуется через `agate_build.sokol.emLinkStep` (`use_webgpu = true`, `use_webgl2 = false`, `use_emmalloc = true`, `use_filesystem = true`, `shell_file_path = vendor/sokol/src/sokol/web/shell.html`, `--preload-file assets@assets`, `-sSTACK_SIZE=1MB`, `-sINITIAL_MEMORY=128MB`, `-sALLOW_MEMORY_GROWTH=1`); `emRunStep` даёт `zig build run` в браузере. Без `-Dweb-debug` Debug-сборка на wasm флорится в `ReleaseFast` (неоптимизированный был бы с `-O0` + safety-checks + `SAFE_HEAP`, что даёт неприемлемый FPS).
- Уже работает: WebGPU-бэкенд через sokol WGPU, sandbox и бенч собираются и запускаются в браузере, 32-битная wasm-совместимость.
- Не входит: JS/TS API, DOM/HTML, npm, WebXR (см. `../roadmap.md`, раздел 🟡/🚫).

### Тесты и гейты

- `zig build test` — юнит-реестр (все `test`-блоки дерева; math отдельно) + `zig fmt --check` гейт (`zig build fmt`).
- GPU-гейт `test-gpu` (8 legs — пишут/держат другие агенты, см. `./render-pipeline.md`): headed-прогоны проходов/пайплайнов.
- Bench: `scene.stats` + `Profiler`/`SessionSummary` + `AGATE_GPU_TIMINGS=1` для GPU-чисел; `--stats` каждые 120 кадров в демо.
- Сериализация/экспорт: fuzz-корпуса (`serialization_fuzz.zig`), round-trip тесты.

### Карта модулей

| Doc | Файлы | Что это |
|---|---|---|
| `./architecture.md` | `root.zig`, `build.zig`, `src/main.zig` | Эта карта: слои, потоки, владение, сборка, гейты |
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

- Нарушение контракта потоков (game пишет `stats`, воркер зовёт `sg.*`, registry-mutate поперёк latch) — баг приложения; tripwire'ы (`assertOnContextThread`, `Scene.lock_free_prepare`, commit guards) делают его громким, а не тихим.
- Два продюсера, вложенный `parallelFor`, `sg.*` в job — запрещены контрактом (см. `./runtime.md`).
- Headless (нет `sg.setup`): все GPU-пути fail-closed (тайминги 0, аплоады CPU-only + deferred, render early-out после эпохи).
- OOM в реестрах/слотах — fail-closed (null/skip/drain, счётчики дропов), не паника, кроме debug-ассертов на двойные терминалы claim.

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
