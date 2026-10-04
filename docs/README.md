# Документация модулей Agate

По одному файлу на семейство модулей: назначение, реальный API (сигнатуры сверены
с исходниками), потоки и владение, ошибки и краевые случаи, примеры. Проза на
русском, идентификаторы в английском. Обзор движка целиком — [architecture.md](./architecture.md);
краткий публичный API — [../API.md](../API.md); статус фич — [../roadmap.md](../roadmap.md).

## Обзор

| Файл | О чём |
|---|---|
| [architecture.md](./architecture.md) | Слои движка, модель потоков (game/context/audio/io_runner), lock-free staged prepare одной страницей, владение GPU-ресурсами, сборка, тесты и гейты, карта модулей |
| [graphics-roadmap.md](./graphics-roadmap.md) | План качества/перфа: HDR и IBL сначала, предпосылки GI/temporal/indirect, гейты и стоп-условия |
| [modernization-audit.md](./modernization-audit.md) | Аудит устаревших путей/API: один HDR-renderer и staged-протокол, кандидаты на удаление, зависимости и проверки |

## Ядро и потоки

| Файл | О чём |
|---|---|
| [runtime.md](./runtime.md) | `Runtime` (фасад кадра, `beginPrepare*`/`produceBuild*`, `setProducerExclusion`, метрики), `Handoff`, affinity-маркер, job pool (`forkJoin`/`parallelFor`), наблюдаемость |
| [gpu-timing.md](./gpu-timing.md) | Opt-in Metal/WebGPU timestamps: capabilities, availability, submission ids, lifecycle и живой GPU-гейт (только optional `Sample`, legacy `poll*Ms` удалены) |
| [scene.md](./scene.md) | Фасад `Scene`: registry/content, lifecycle, камеры сцены, API-фасады (sim/query/lights/profile), снимки и статистика |
| [frame-pipeline.md](./frame-pipeline.md) | Многопоточный кадр: 3-слотовый протокол (claim→build→stage→publish, pin/lease), lock-free контракт, `host_bytes`, режимы staged/mutex/serial |
| [render-pipeline.md](./render-pipeline.md) | Cull/сортировка/биннинг, zero-dereference draw items, инстансинг (4 фазы), фабрика пайплайнов, `FrameContext`, PostFX-цепочка, тени CSM/spot/point |
| [scene-layers.md](./scene-layers.md) | Слои сцены (decal/gui3d/highlight/nav/particle/physics/probe/sky/trail), clustered lights, top-K выбор света, `LightRig`, параллельная анимация |

## Геометрия и материалы

| Файл | О чём |
|---|---|
| [mesh.md](./mesh.md) | `Mesh`/`InstancedMesh`/LOD, `GeometryData` + `uploadGeometry`, все построители, CSG, декали, GreasedLine, TrailMesh, морфы (.cpu/.gpu), QEM-упрощение, VAT |
| [material.md](./material.md) | Standard/PBR (UV0/UV1, clearcoat/sheen/aniso/transmission/SSS, opt-in refraction), unlit, alpha-режимы, shader-материалы, node-материалы (граф), библиотека пресетов |
| [lights.md](./lights.md) | 7 типов света, лимиты и тени по типам, sun-резолверы, цветовая температура |
| [cameras.md](./cameras.md) | 5 камер с инерцией, `Camera`-union, viewport'ы, риги (dual/quad/CAD/stereo VR), PIP |
| [texture.md](./texture.md) | PNG/JPEG/HDR/EXR/DDS/KTX2+Basis, мипмапы, sRGB, cube-текстуры, capacities |
| [render-target.md](./render-target.md) | Render-to-texture: color/depth/MSAA resolve, borrowed Texture, resize, GPU smoke |
| [loading-performance.md](./loading-performance.md) | Опциональные замеры glTF по стадиям, побайтово совместимое ускорение sRGB+mips |
| [loader.md](./loader.md) | glTF/GLB-конвейер + OBJ/STL/PLY/meshopt, `LoadOptions` (в т.ч. `morph_mode`), async-загрузка |
| [export.md](./export.md) | Экспорт OBJ+MTL, STL, GLB |
| [assets.md](./assets.md) | `UploadQueue` (дедупликация, покадровые бюджеты), `AssetManager`, `io_runner` |

## Симуляция

| Файл | О чём |
|---|---|
| [physics.md](./physics.md) | Box3D: тела/compound, суставы, character/rope, события, запросы, ragdoll/vehicle, debug-линии, sleeping |
| [particles.md](./particles.md) | CPU/GPU/compute режимы, коллизии, flow-поля, sub-emitters, staged-интеграция, лимиты |
| [animation.md](./animation.md) | Скелеты (64 кости), клипы/блендинг/события, Hermite-сэмплеры, ретаргетинг, node-TRS, параллельная оценка |
| [ai.md](./ai.md) | NavMesh, A* + funnel, `NavAgent`, толпа ORCA |
| [math.md](./math.md) | SIMD `@Vector`-типы, TRS-сборка, 4-wide culling, детерминизм |

## Платформенные сервисы

| Файл | О чём |
|---|---|
| [audio.md](./audio.md) | Синтез, шины-DAG, DSP (biquad/Freeverb), окклюзия, стриминг OGG/MP3/WAV, 0 аллокаций в аудиопотоке |
| [ui.md](./ui.md) | Canvas/виджеты, Flexbox/Grid/anchors/docking, темы и переходы, SDF+TrueType текст, 3D-панели |
| [profiler.md](./profiler.md) | Фазовые метрики, отчёты HTML/MD/Chrome-trace (enqueue-only окно), `MemorySnapshot` с latched-ответом |
| [serialization.md](./serialization.md) | AGSC v1–v3, save/load память/файл/async, custom properties, что сохраняется |
| [visibility.md](./visibility.md) | CPU Hi-Z окклюзия: пирамида, растеризатор окклюдеров, `OcclusionCuller` |
| [shaders.md](./shaders.md) | Компиляция sokol-shdc, `// @include`-чанки `shaders/common/`, shader-материалы и hook-точки, drift-тесты |
| [postprocess.md](./postprocess.md) | Единая HDR-цепочка, параметры/клампы, порядок composite-стека |
| [passes.md](./passes.md) | GPU-пассы: что рендерят, таргеты, включение (bloom — единая HDR-пирамида) |

## Соглашения

- Имена: `Type.new`/`Type.init(name, options)` — значения; `create*` — Scene/GPU-владеющие;
  `build*Data → GeometryData` — чистая CPU-сборка; `*Options` — дефолтные ручки;
  `*Desc` — дескрипторы с обязательными полями; `*Params` — покадровые паки.
- Потоки: **game** (producer, симуляция + build), **context** (sapp/render, begin/finish/render,
  все `sg.*`), **audio** (0 аллокаций/блокировок), **io_runner** (файловый IO).
  Подробности — [architecture.md](./architecture.md).
