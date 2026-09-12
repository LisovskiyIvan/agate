# Agate Roadmap vs Babylon.js: что сделано, что нет, что не планируется

> Это одновременно карта возможностей и очередь работ: всё из раздела **❌** — кандидаты в реализацию, **🚫** — вне области нативного движка.

> Дата: 10.09.2026.
> **Agate** — нативный десктопный движок: Zig 0.16, sokol (app/gfx/glue/audio/time), встроенные C-библиотеки cgltf, stb_image и физический движок Box3D v0.1.0. Forward-рендер, шейдеры компилируются под GL 4.1 (Linux), Metal (macOS), D3D11/HLSL5 (Windows).
> **Babylon.js** — 9.x (2026): WebGL2/WebGPU, TypeScript, браузер + Babylon Native/Node.js.
>
> Agate **не рассчитан на веб**: браузерных и JS-зависимых возможностей Babylon в нём нет и не планируется. Всё остальное, чего пока нет, — потенциальный бэклог, а не приговор.

**Легенда**

| Знак | Значение |
|------|----------|
| ✅ | Реализовано |
| 🟡 | Реализовано частично / упрощённо |
| ❌ | Не реализовано (решение о добавлении не принято) |
| 🚫 | Не планируется: завязано на веб / браузер / JS-экосистему |

---

## Краткая сводка

| Направление (аналог в Babylon.js) | Agate | Статус |
|---|---|---|
| Ядро: сцена, граф, трансформы, математика | Scene, Mesh, SIMD-математика | ✅ |
| Рендер | Forward, 8 пайплайнов, opaque, сортировка по пайплайну/текстуре/дистанции | ✅ |
| Frustum culling | AABB + SIMD 4-wide | ✅ |
| Инстансинг | InstancedMesh + GPU-пайплайн | ✅ |
| Камеры | ArcRotate + Free + Fly + Follow + Target, объединяющая union Camera | 🟡 |
| Свет | Hemispheric + Directional (солнце) + до 4 Point + до 2 Spot | 🟡 |
| Тени | 4-каскадный CSM для солнца, 16× Poisson PCF | 🟡 |
| Материал Standard | Diffuse-цвет/текстура | ✅ |
| Материал PBR (metallic-roughness) | Albedo/Normal/MR/Emissive/AO + IBL | 🟡 |
| OpenPBR, clearcoat, sheen, transmission | — | ❌ |
| Текстуры 2D | PNG/JPEG + HDR (Radiance) через stb_image, RGBA8/RGBA16F, CPU-мипмапы | 🟡 |
| HDR/EXR/DDS/KTX/Basis, сжатие, видеотекстуры | — | ❌ |
| Cube / Skybox / IBL | CubeTexture, equirect → cube, процедурное небо | ✅ |
| Постобработка | ACES/Reinhard, bloom, виньетка, CA, sharpen, grain, white balance, FXAA, fog, SSR, SSAO | 🟡 |
| DoF, motion blur, TAA, MSAA, LUT-цветокоррекция | — | ❌ |
| Частицы | CPU-симуляция + GPU-инстансы, additive/alpha, local space, спрайт-листы, поворот | 🟡 |
| GPU-симуляция, sub-emitters, flow maps | — | ❌ |
| Анимация | Скелетная (до 64 костей, GPU skinning, блендинг/crossfade) + node-анимации glTF TRS + easing | 🟡 |
| События анимаций, ретаргетинг | — | ❌ |
| Меш-билдеры | Box, Sphere, Cylinder, Capsule, Ground, Terrain, Torus, TorusKnot, Disc, Ribbon, Lathe, Plane, Tube, Extrude, Lines | 🟡 |
| LOD & Декали | Mesh.addLODLevel / getLOD / getLODForCamera + Sutherland-Hodgman Decal Projector | ✅ |
| Polygon / CSG / Упрощение мешей | — | ❌ |
| glTF/GLB | PBR, сэмплеры, скины, анимации, морфы, свет/камеры (KHR_lights_punctual), внешние URI | 🟡 |
| Draco/meshopt/KTX2, экспорт | — | ❌ |
| Физика | Box3D: коллайдеры, compound, суставы, character, rope, события, запросы AABB/сфера/точка, ragdoll/vehicle-хелперы | ✅ |
| Soft body | — | ❌ |
| Debug-рендер физики | генерация линий коллайдеров (`appendDebugLines`) + 3D-пасс линий (depth-tested) | ✅ |
| UI | Экранный canvas, SDF-текст, кнопки/панели, checkbox, slider, dropdown, скролл, text input | 🟡 |
| Layout-контейнеры, 3D GUI | — | ❌ |
| Аудио | Процедурный синтез + WAV-файлы, 24 голоса, панорама/затухание | 🟡 |
| mp3/ogg, стриминг, шины, эффекты | — | ❌ |
| Пикинг | CPU-луч (AABB/сфера/треугольник), raycast в физике | ✅ |
| Сериализация сцены (бинарный AGSC: TRS/материалы/свет/камера/post FX), экспорт | ✅ |
| Навигация/crowd/pathfinding | — | ❌ |
| Сеть/multiplayer | — | ❌ |
| Frame graph, clustered lighting, volumetric, Gaussian splatting | — | ❌ |
| Large world rendering, geospatial | — | ❌ |
| Тесты/бенчмарки | 310 unit-тестов, `zig build test`, `sandbox --bench` | ✅ |
| Inspector, Playground, NME, редакторы частиц/GUI | — | 🚫 |
| WebGL/WebGPU, DOM/HTML, JS/TS API, npm | — | 🚫 |
| WebXR (VR/AR), WebAudio, Web Workers, CDN | — | 🚫 |
| Node.js/NullEngine, серверный headless-рендер | — | 🚫 |
| Babylon Native / React Native (JS-рантайм-мосты) | — | 🚫 |

---

## Журнал реализации (итерация 10.09.2026)

Выполнено и проверено: `zig build test` (244 тестов), `zig build` (agate), `zig build` + runtime smoke в sandbox (Metal, 10–30 кадров, с принудительно включённым пост-процессом и оверлеем физики).

| Фича | Файлы | Статус |
|---|---|---|
| Mesh builders: Torus, TorusKnot, Disc, Ribbon, Lathe | `mesh.zig`, `root.zig` | ✅ |
| Пост-эффекты: sharpen, film grain, white balance | `postprocess.zig`, `postprocess_pass.zig`, `postprocess.glsl` | ✅ |
| WAV-плеер: `AudioClip` + `playClip` (8/16/24/32-bit, float32, mono/stereo, loop/rate) | `audio.zig` | ✅ |
| UI: checkbox, slider, divider, arrow | `ui.zig` | ✅ |
| Камеры: FreeCamera, FollowCamera, union `Camera`, `Scene.updateCamera` | `camera.zig`, `scene.zig`, passes | ✅ |
| Физика: `queryAABB` / `querySphere` / `queryPoint` (+filter), `spherecast` | `physics.zig` | ✅ |
| `DirectionalLight` в API сцены (солнце для CSM и освещения) | `lights.zig`, `scene.zig` | ✅ |
| Физика: debug-линии коллайдеров `appendDebugLines` | `physics.zig` | ✅ (рендера нет) |

Следующие кандидаты: морф-таргеты, alpha-test (cutout)/double-sided, mp3/ogg, soft body, сериализация сцены, GPU bloom/SSR, прозрачный инстансинг.

### Волна 2: фичи из roadmap (10.09.2026)

Реализовано пятью параллельными агентами, интеграция и проверка — централизованно. `zig build test` — 149 тестов (+43), `zig build` agate/sandbox, runtime smoke без утечек.

| Фича | Файлы | Статус |
|---|---|---|
| Меш-билдеры Plane, Tube (parallel transport), Extrude (ear clipping), Lines | `mesh.zig` | ✅ |
| Node-анимации glTF (TRS) + easing (10 кривых) | `animation/`, `loader/`, `easing.zig` | ✅ |
| Прозрачность: `AlphaMode.blend`, blend-пайплайны, back-to-front очередь | `material.zig`, `scene.zig` | ✅ |
| HDR-текстуры Radiance → RGBA16F, equirect→cube, env у PBR-материала | `texture.zig`, `scene.zig` | ✅ |
| Ragdoll- и RaycastVehicle-хелперы | `ragdoll.zig`, `vehicle.zig` | ✅ |

Sandbox: всё UI сведено в **одну панель** (HUD + «NEW FEATURES WAVE 2» + CONTROLS); в сцену добавлены Plane/Tube/Lines/Extrude, стеклянная сфера (blend-очередь), HDR-env сфера с процедурной equirect-панорамой, node-анимация тора с easing.

### Рефакторинг-волна (10.09.2026)

- `scene.zig`: data-driven pipeline-фабрика (16 пайплайнов из таблицы) + единый `FrameUniforms` (убрано троирование ~30 полей); удалено мёртвое поле `default_black_texture`.
- `root.zig`: устранено дублирование экспортов `animation`, добавлены забытые публичные типы (`HemisphericLightOptions`, `ArcRotateCameraOptions`, `FrustumPlane`, `ParticleInstanceData`, `GlyphUV`, `easing_names`, `SceneStats` и др.).
- `mesh.zig`: общие `storeQuad/appendGridQuad` (14 билдеров), `buildTrigTable`, `pickOrthogonal`/`resolveFrameSeed`; Disc переведён на общую таблицу.
- `ragdoll.zig`/`vehicle.zig`: общий `physics_mesh.zig` (create/free/sync/teardown), удалены мёртвые `owns_meshes` и неиспользуемые параметры `world`.
- `texture.zig`: общий `boxDownsampleU8` для 2D и cube-мипов (+NPOT-безопасность), проверки переполнения checkerboard, валидация `face_size`; `fromEquirectangularFile` мигрирован на `std.Io` (в Zig 0.16 `std.fs.cwd` удалён — латентный баг пойман агентом).
- `audio.zig`: векторные WAV-пути u8/i24/i32, потоковый `fromWavFile` (пик памяти файл+f32 → f32+64КБ), общие декодеры.
- `build.zig`: таблица 11 шейдерных модулей, шаг `zig build fmt`, макрос `SHADOW_ATLAS_SIZE` в шейдерах + `pub const SHADOW_ATLAS_SIZE` в `shadow_pass.zig` (sokol-shdc `#include` не поддерживает).
- Итог: 161 тест, `zig fmt --check src` чистый, runtime smoke без утечек; перф без регрессий (~2.6 мс).

### Распил структуры (10.09.2026)

Публичный API сохранён полностью (сверено списками `pub` до/после: 73 метода `PhysicsWorld`, 34 метода/поля `Scene`, 26 публичных типов physics, `SceneLoader.appendGlb/appendGltf`).

| Было | Стало | Новые модули |
|---|---|---|
| `physics.zig` 4823 строки (логика 2908 + тесты 1915) | 3492 (логика 1577 + тесты) | `physics/{convert,types,body,debug_geo,queries,joints,debug,events}.zig` |
| `loader/scene_loader.zig` 1058 | 96 (оркестратор) | `loader/{gltf_util,materials,skins,mesh_spawn,animations}.zig` |
| `scene.zig` 1902 | 1477 (логика 1217 + тесты) | `scene/{render_queue,pipelines,cascades,uniforms,projection,draw}.zig` |

Ограничение Zig 0.16: `usingnamespace` отсутствует, mixin-паттерн не работает. Методы вынесены свободными generic-функциями (`world: anytype` / `scene: anytype`) + тонкими форвардинг-обёртками в исходном файле; C-колбэки запросов — фабрики `OverlapQuery(comptime World)`/`SphereCastQuery(World)` без цикла импортов. В `physics.zig` остались `CharacterController`, `Rope` и ядро `PhysicsWorld` (цикл на приватных хелперах). Проверки: 161 тест, `zig build` agate/sandbox, runtime smoke без утечек, `--bench` зелёный.

### Волна 3: фичи (10.09.2026)

| Фича | Файлы | Статус |
|---|---|---|
| Морф-таргеты glTF (до 8, CPU-блендинг с dirty-tracking, weights-каналы + easing) | `mesh.zig`, `animation.zig`, `loader/{mesh_spawn,animations}.zig`, `scene.zig` | ✅ |
| Сериализация состояния сцены (бинарный `AGSC` v1: TRS/материалы/свет/камера/post FX; save/load в память и файл) | `serialization.zig` | ✅ |
| UI: dropdown, скролл (`ScrollState`), text input (`TextInputState`, UTF-8-редактирование) | `ui.zig` | ✅ |
| glTF-свет (`KHR_lights_punctual`) и камеры | `loader/lights.zig`, `scene_loader.zig` | ✅ |
| Частицы: локальное пространство эмиттера, спрайт-листы, поворот/angular velocity | `particles.zig`, `passes/particle_pass.zig`, `shaders/particle.glsl` | ✅ |

Sandbox: dropdown камеры, кнопки/клавиши Save State [7] / Load State [8] (in-memory снимок), обновлённый CONTROLS. Итог: 244 теста, `zig build test`/agate/sandbox зелёные, runtime smoke без утечек (~2.6 мс, init ~390 мс).
Sandbox-демо (все новые фичи выведены в интерфейс): галерея из 5 новых примитивов; панель New Features с чекбоксами Sharpen/Grain и слайдерами Sharpen/Grain/Temperature/Tint; кнопки Camera/WAV/Lines; клавиши `4` — смена камеры Arc→Free→Follow, `5` — debug-линии физики + счётчик `querySphere`, `6` — проигрывание WAV из памяти. DirectionalLight включён в сцену.

Оптимизация движка (замерено): кэш view-projection в `Scene.projectPoint` + hoist солнца, резерв `appendDebugLines` (без роста списка) + опциональный `debug_circle_segments`, единичный `sqrt` в `ui.drawLine` и запас буфера UI, блочный микшер аудио с предвычисленными гейнами + быстрый декод WAV, предвычисление trig-таблиц в билдерах. Оверлей физики: 4.08–4.49 → 3.13–3.24 мс/кадр (~25%), база без оверлея не изменилась (~2.4 мс, ~420 FPS). Найден и исправлен leak в `uploadGeometry` (затирание `cpu_positions/cpu_indices`). Отложено: GPU-оптимизации bloom/SSR (нужна визуальная проверка).


### Волна 4: фичи (10.09.2026, вечер)

| Фича | Файлы | Статус |
|---|---|---|
| Alpha-test (cutout) + double-sided (cull-off твины, +16 пайплайнов) | `material.zig`, `scene/{pipelines,uniforms,draw,render_queue}.zig`, шейдеры | ✅ |
| Камеры `TargetCamera` (look-at со сглаживанием) и `FlyCamera` (крен) + union | `camera.zig`, `scene/projection.zig` | ✅ |
| Анимация: cubic-spline Hermite (TRS/quat/weights) + события/колбэки | `animation.zig`, `loader/{gltf_util,animations}.zig` | ✅ |
| Debug-рендер физики: 3D-пасс линий (depth-test, без записи) | `passes/debug_pass.zig`, `passes/mod.zig`, `shaders/debug.glsl`, `scene.zig` | ✅ |
| Импорт OBJ и STL (ASCII/бинарный, дедуп, триангуляция, нормали) | `loader/obj.zig`, `loader/stl.zig` | ✅ |
| Сериализация v2: target/fly-камеры, cutout/cutoff/double_sided; glTF alphaMode/doubleSided | `serialization.zig`, `loader/materials.zig` | ✅ |

Sandbox: debug-линии рисует движковый 3D-пасс (UI-проекция удалена), в dropdown камер добавлены Fly/Target, OBJ-пирамида в галерее, Save/Load работает с новым форматом v2. Итог: 244 теста (+49), сборки/бенч/смоук зелёные (~2.6 мс, init ~380 мс).

### Волна 5: sandbox-витрина всех фич (11.09.2026)

Инвентаризация «что уже есть в движке, но ни разу не показано в sandbox» + закрытие пробелов. Новые демо-острова живут в `sandbox/src/sandbox_showcase.zig`:

| Фича | Что добавлено в sandbox |
|---|---|
| Морф-таргеты | `AnimatedMorphCube.glb` (2 blend shape), слайдер Morph, кнопка ручного/клипового режима |
| События анимации | 4 маркера на морф-клипе (`setEvents`), burst частиц + лог по каждому срабатыванию |
| Cubic-spline | `InterpolationTest.glb` (STEP/LINEAR/CUBICSPLINE-ряды), счётчик cubic-сэмплеров в HUD |
| glTF-свет | `LightsPunctualLamp.glb` (KHR_lights_punctual, 5 point-светов с клампом интенсивности) |
| Cutout + double-sided | Процедурный «папоротник» (alpha-test текстура) + чекер-флаг с двусторонним пайплайном |
| Частицы | Локальный thruster на машине, alpha-blend дым, 4×4 spritesheet-взрыв + angular velocity |
| STL | `slotted_disk.stl` (ASCII) и `pr2_head_pan.stl` (binary) через `appendStlToScene` |
| WAV из файла | `ambient_loop.wav`, лупающая позиционная петля (`fromWavFile` + pan) |
| Spatial queries | AABB / point / spherecast вокруг игрока (счётчики в HUD) |
| Сериализация в файл | `saveSceneStateFile` / `loadSceneStateFile` ([9]/[0]) рядом с in-memory [7]/[8] |
| Постобработка | Чекбоксы Bloom/Vignette/Chromatic + слайдер Exposure к прежним Sharpen/Grain/Temp/Tint |
| UI-контролы | Скроллируемый event-log (ScrollState + колесо), текстовый фильтр (TextInputState + CHAR), progress bar |
| Skinning | Второй персонаж `CesiumMan.glb` (ассет уже лежал в repo, но не использовался) |

Попутно найдены и исправлены два движковых бага (видны только на runtime-пути glTF):
- **Морфы паниковали при первом блендинге**: glTF-меши с morph targets получали immutable vertex buffer, а `applyMorphs` вызывает `sg_update_buffer`. Теперь такие меши создаются с `dynamic_update`, а базовая геометрия заливается первым `applyMorphs` до рендера (`loader/mesh_spawn.zig`).
- **Утечка имён glTF-светов**: `loadLights` дублировал имена в арену сцены, но `Scene.deinit` их не освобождал. Добавлен `owns_name` у `PointLight`/`SpotLight`/`DirectionalLight` и освобождение при deinit/replace (`lights.zig`, `scene.zig`, `loader/lights.zig`).

Плюс `--ui-scale N` (CLI) — глобальный масштаб HUD для маленьких окон/скриншотов. Ассеты: `AnimatedMorphCube.glb`, `InterpolationTest.glb`, `LightsPunctualLamp.glb` (Khronos glTF-Sample-Assets), `slotted_disk.stl`/`pr2_head_pan.stl` (three.js examples), `ambient_loop.wav` (сгенерирован). Проверки: 244 теста, `zig build` agate/sandbox, smoke без утечек и паник (~2.9 мс, init ~612 мс — +230 мс из-за 6 текстур лампы 2048²).


### Волна 6: качество рендера (11.09.2026)

Шесть параллельных агентов, интеграция и проверка — централизованно. `zig build test` — **310 тестов (+66)**.

| Фича | Файлы | Статус |
|---|---|---|
| Bloom с мип-пирамидой (Karis downsample + tent upsample, 3–7 мипов) | `passes/bloom_pass.zig`, `shaders/bloom_down.glsl`, `shaders/bloom_up.glsl`, `postprocess.zig` | ✅ |
| DoF (CoC по глубине, 14 golden-angle taps) + цветовые curves (shadows/midtones/highlights) | `shaders/postprocess.glsl`, `postprocess.zig` | ✅ |
| Outline-слой (inverse hull, cull front, ширина в пикселях) | `passes/outline_pass.zig`, `shaders/outline.glsl` | ✅ |
| PCSS для CSM (12 blocker-семплов + variable penumbra, legacy PCF при выключенном) | 4 forward-шейдера, `scene/shadow_pcss.zig`, `scene/uniforms.zig` | ✅ |
| Выбор 4 point + 2 spot светов по значимости `intensity×range/(1+d²)` | `scene/light_selection.zig` | ✅ |
| PLY-импорт (ASCII + binary LE/BE, цвета/UV/нормали, fan-триангуляция, пропуск unknown) | `loader/ply.zig` | ✅ |
| Экспорт OBJ+MTL и STL (ASCII/binary, world-transform, видимость) | `export/obj.zig`, `export/stl.zig` | ✅ |

Sandbox: PLY-октаэдр в галерее; клавиши `[;]` bloom-пирамида, `[']` DoF, `[\]` PCSS, `[/]` outline на последнем pick, `[=]` экспорт `scene.obj`/`scene.mtl`, `[-]` экспорт `scene.stl`; строка `W6: B/D/P/O` в сайдбаре. UX-проход: dropdown камеры рисуется верхним z-слоем (раньше его перекрывали кнопки), сайдбар разбит на блоки VIEW/DEMO/WAVE 6 FX с воздухом, новые острова сцены раздвинуты и подписаны world-space-метками (Morph/Cubic/PLY/STL/lamp/foliage/flag/Cesium/thruster/smoke) + зоны GALLERY EAST, ALPHA GARDEN, LIGHT LAB, CHARACTER STAGE.

Найденные при интеграции дефекты агентских заготовок (исправлено родителем):
- raw-depth текстуры в forward-шейдерах надо объявлять `@image_sample_type … unfilterable_float` + `@sampler_type … nonfiltering`, иначе валидация sokol падает (`filterable image expected` / `NONFILTERING sampler required`); добавлен отдельный nonfiltering `depth_sampler` в `ShadowPass` и биндинг в `scene/draw.zig`.
- `OutlinePass.resize` не вызывался — viewport оставался 1×1 и контур не рисовался; resize подключён в `Scene.render` (и `BloomPass.resize` рядом).
- `gltf_util.readSampler` неверно читал морф-веса: у SCALAR-аксессора на ключ приходится `stride` элементов (по одному на morph target), а код читал один элемент на ключ — таргеты алиасились и вес «дрожал» (0 → ramp → 0). Теперь скалярные элементы читаются поштучно; регрессионный тест `readSampler reads interleaved morph weights per target` в `loader/gltf_util.zig`.
- В sandbox STEP-ряды `InterpolationTest` поставлены на паузу по умолчанию (ступенчатая интерполяция читается как дрожание кубов; LINEAR/CUBIC играют, STEP остаются в сцене).

Отложено осознанно: MSAA (в этой версии sokol нет depth-resolve, а постобработке нужна depth-текстура — нужен отдельный дизайн), LUT-текстура (сейчас параметрические curves), тени от point/spot, motion blur/TAA.

### Волна 7: модульность, lock-free аудио и оптимизация hot-path (12.09.2026)

Комплексный профилировочный аудит движка: устранение избыточных вычислений в кадре и за кадром, декомпозиция монолитов и переход на lock-free примитивы.

| Направление | Файлы | Оптимизация | Статус |
|---|---|---|---|
| Модуляризация мешей | `mesh/{types,tangents,builders,mesh,builder}.zig`, `mesh.zig` | Монолит 3219 строк разбит на 5 изолированных модулей; чистые CPU-билдеры `build*Data`, фасад 68 строк | ✅ |
| Унификация лоадеров | `loader/{ply,stl,obj}.zig` | Переход на `uploadGeometry` и `GeometryData`, удаление дублирования буферов sokol | ✅ |
| SIMD морф-таргеты | `mesh/mesh.zig` | Многопроходный цикл (до 24 проходов по вершинам) заменён на 1 проход с 4-wide `@Vector(4, f32)` SIMD | ✅ |
| Pre-binning теней | `passes/shadow_pass.zig` | Исключён цикл $4 \times 6 \times N$; линейная сортировка подсчётом по пайплайнам, пропуск пустых бакетов | ✅ |
| Динамический буфер UI | `ui.zig` | Динамическое геометрическое перевыделение GPU-буферов при переполнении вместо краша | ✅ |
| Lock-free аудио | `audio.zig` | Полное удаление `SpinLock`; lock-free SPSC командное кольцо, атомики громкости/mute, нулевые блокировки в аудиопотоке | ✅ |
| Hoist весов анимации | `animation/eval.zig` | Вынос вычисления `alpha` и нормализации весов из цикла по костям (`for (skel.bones)`) | ✅ |
| Single-pass выбор света | `scene/light_selection.zig` | Алгоритм $\mathcal{O}(K \cdot N^2)$ заменён на однопроходный $\mathcal{O}(N \cdot K)$ top-K с вычислением score ровно 1 раз на свет | ✅ |
| Кэш инстансов & Wyhash | `mesh/types.zig`, `scene.zig` | Кэширование TRS/AABB в `InstancedMesh` (пропуск sin/cos и 8-corner AABB у статичных копий) + замена побайтового FNV-1a на Wyhash | ✅ |
| Очереди инстансинга | `scene.zig` | Предварительная фильтрация в Phase 0 исключила двойной холостой прогон всех мешей сцены в draw loop | ✅ |
| Спящие тела в физике | `physics.zig`, `physics/body.zig` | Пропуск `pullBody` (тригонометрия Quat→Euler, скорость, контактные манифолды) для спящих тел (`!is_awake`) | ✅ |
| Кэш матриц SSAO | `passes/ssao_pass.zig` | Кэширование проекции и обращения матрицы `inv_proj` при неизменных параметрах камеры | ✅ |

Проверки: 310+ тестов, `zig build test` (agate) и `zig build` (sandbox) проходят за 1-2 сек без ошибок и предупреждений.

### Волна 8: LOD (Level of Detail) и система декалей (12.09.2026)

| Фича | Файлы | Описание | Статус |
|---|---|---|---|
| Система LOD | `mesh/types.zig`, `mesh/mesh.zig`, `scene.zig`, `passes/shadow_pass.zig` | Сортированные дистанционные уровни детальности (`LODLevel`), автоматический выбор меша по расстоянию до камеры, поддержка дистанционного куллинга (`mesh: null`), исключение `is_lod_child` из теневых пассов и рейкаста, синхронизация трансформов и материалов | ✅ |
| Система декалей | `mesh/decal.zig`, `mesh/builder.zig`, `mesh.zig`, `root.zig` | Проектор ориентированного куба (OBB) на целевой меш произвольной формы, алгоритм отсечения многоугольников Sutherland-Hodgman по 6 плоскостям, depth bias против z-fighting, backface culling, вычисление UV, касательных (tangents) и нормалей | ✅ |

Проверки: 315 unit-тестов, `zig build test` (agate) и `zig build` (sandbox) проходят чисто без ошибок и предупреждений.


---

## ✅ Что сделано

### Ядро и платформа

* Нативное приложение на sokol: macOS (Metal), Windows (D3D11), Linux (GL).
* Scene graph: иерархия `Mesh.parent`, TRS-трансформы, ленивый пересчёт world-матриц за кадр (`scene.zig: worldMatrixCached`).
* Математика: `Vec2/3/4`, `Mat4` (SIMD-перемножение), `Quat` (slerp/nlerp), `Color3/4`, `BoundingBox`, `Frustum`, `Ray`.
* Статистика кадра: меши, отсечённые, draw calls, треугольники, переключения пайплайнов (`SceneStats`).
* 310 unit-тестов в библиотеке, отдельный sandbox с бенчмарками (`zig build test`, флаг `--bench`).

### Рендеринг

* Forward-рендер, шейдеры cross-compile через sokol-shdc (GLSL410/Metal/HLSL5).
* 8 пайплайнов: Standard, PBR, Instanced, Skinned PBR — каждый под u16/u32 индексы.
* Сортировка очереди: сначала непрозрачные Standard/PBR группами по текстуре, front-to-back для early-Z.
* Frustum culling AABB, включая SIMD-батч по 4 инстанса (`Frustum.intersectsAABB4`).
* GPU-инстансинг: динамический instance-буфер с дедупликацией загрузок по хэшу.
* Offscreen-буфер для постобработки с depth-текстурой; UI-оверлей поверх.
* Документированные проходы: shadow → main → skybox → particles → SSAO → post-process → UI (`passes/`).

### Камеры

* `ArcRotateCamera` — орбита мышью, зум колесом, лимиты радиуса/угла.
* `FreeCamera` — движение WASD/стрелками + Space/Ctrl, обзор ЛКМ-драгом, скорость/чувствительность, `update(dt)`.
* `FollowCamera` — следование за мешем или точкой с радиусом, высотой, смещением угла и сглаживанием.
* `Camera` — union-абстракция (arc_rotate/free/follow), `Scene.updateCamera(dt)`; проходы и пикинг работают с любым типом.

### Свет и тени

* `HemisphericLight` (небо + ground color) — всегда одна.
* `DirectionalLight` — `Scene.createDirectionalLight`; управляет солнцем (направление, цвет, интенсивность, 4-каскадный CSM), hemi остаётся ambient.
* До 4 `PointLight` с range/интенсивностью и до 2 `SpotLight` (inner/outer cone) — per-pixel затухание.
* Тени: 4-каскадный CSM (атлас 2048², 4 × 1024²), 16-выборок Poisson PCF, depth bias + normal bias, мягкость, fade дальнего каскада, debug-режим каскадов.
* `mesh.cast_shadows` / `mesh.receive_shadows` на каждый меш; скелетные меши тоже отбрасывают тени (skinned shadow-пайплайн).

### Материалы и IBL

* `StandardMaterial`: diffuse color + текстура.
* `PBRMaterial` (metallic-roughness, Cook-Torrance): albedo, normal, metallic-roughness, emissive, occlusion (сила), alpha, цветовые факторы, environment intensity.
* IBL от skybox-кубмапы, exposure, выбор текстуры отражений.
* Дефолтные 1×1 текстуры (white/black/flat normal/cube) — PBR работает без ассетов.

### Текстуры, небо, окружение

* 2D-текстуры: декод PNG/JPEG/… через stb_image, RGBA8, настройки wrap/min/mag, полная CPU-цепочка мипмапов (box-filter), процедурные checkerboard и particle-dot.
* CubeTexture: 6 граней, дефолтная 1×1, процедурный skybox-градиент, развёртка equirectangular-панорамы в куб, загрузка граней из файлов.
* Асинхронный декод картинок glTF на worker-потоках с fallback на синхронный путь.

### Геометрия

* `Mesh` + `InstancedMesh`, дочерние меши, `BoneAttachment` (крепление к кости).
* `MeshBuilder`: Box (с цветами граней), Sphere, Cylinder, Capsule, Ground (subdivisions), Terrain (heightmap), Torus, TorusKnot, Disc, Ribbon (набор путей), Lathe (профиль вращения).
* Генерация касательных (`computeTangents`), u16/u32 индексы, локальный AABB, опциональное хранение CPU-геометрии для физики (hull/mesh-коллайдеры).
* Вершинные цвета (color0) поддерживаются Standard/PBR шейдерами.

### Анимация

* Скелеты до 64 костей (`MAX_BONES`), GPU-скиннинг через uniform matrix palette.
* glTF-скины и каналы translation/rotation/scale; линейная и step-интерполяция.
* `AnimationGroup`: play/pause/stop, loop, `playRange`, скорость (в т.ч. отрицательная), вес.
* Блендинг базовых клипов (1, 2 и N клипов), аддитивные слои, fadeTo/fadeIn/fadeOut/crossFadeTo.
* Крепления (sockets) мешей к костям, анимированные тени для скелетов.

### Частицы

* `ParticleSystem`: CPU-симуляция, рендер GPU-инстансами.
* Режимы additive / alpha-blend, текстура частицы (есть процедурный dot).
* Эмиттер-бокс, emit rate, burst, гравитация, время жизни, интерполяция цвета и размера start→end.

### Постобработка

* Tonemapping: ACES или Reinhard, exposure.
* Bloom (bright pass + гало), виньетка, saturation/contrast, chromatic aberration, sharpen (unsharp mask), film grain, white balance (temperature/tint).
* FXAA 3.11.
* Дистанционный + высотный fog с подмешиванием цвета солнца (sun scattering).
* SSR: 16 шагов screen-space марша, fresnel, edge fade, настраиваемая интенсивность/толщина/дистанция.
* SSAO: depth-based выборки + bilateral blur, debug-режим, интенсивность/power/radius.
* У каждого эффекта есть вкл/выкл и параметры в `PostProcessConfig` / `SSAOConfig`.

### Физика (Box3D v0.1.0)

* Тела: static/dynamic/kinematic, массы, restitution, friction, linear/angular damping, гравитация.
* Коллайдеры: box, sphere, capsule, convex hull, triangle mesh (static), heightfield.
* Compound-коллайдеры: дополнительные box/sphere/capsule/hull child shapes с фильтрами и событиями.
* Суставы: distance, spherical, revolute (+limits/motor), wheel (+spin/steering), prismatic (+limits/motor), motor, weld, parallel.
* Character controller (кинематический mover), rope (цепочка сегментов, резка/ремонт).
* Сенсоры и события: sensor events, contact events, contact hit events.
* Raycast с фильтрами, `applyImpulse` / `applyTorqueImpulse` / `applyForce`, radial explosion (`applyExplosion`).
* Пространственные запросы: `queryAABB` / `querySphere` / `queryPoint` (+ `*WithFilter`) и `spherecast`.
* Debug-геометрия: `appendDebugLines` / `DebugLine` — каркасы коллайдеров (box/sphere/capsule/compound/AABB) для внешнего отладчика.
* Фиксированный шаг 1/60 c аккумулятором (Fix Your Timestep).

### UI

* `UICanvas`: экранный immediate-mode рендер квадов.
* Примитивы: rect, outline, panel, прогресс-бар, кнопка, бейдж, checkbox, slider, divider, arrow, line (`drawLine`), SDF-текст (обычный/жирный/с обводкой), `measureText`, hit-test.
* SDF-шрифт зашит в движок (SDF-атлас), масштабируется без потери чёткости.
* `Scene.projectPoint` — мировые точки в экранные (используется sandbox для 3D-подписей над объектами).

### Аудио

* Процедурный синтез на sokol.audio: thump, noise burst, blip — без аудиоассетов.
* `AudioClip.fromWavMemory` / `fromWavFile` — WAV PCM 8/16/24/32-bit и float32, mono/stereo, ресемплинг; `AudioEngine.playClip` с loop/rate/позиционированием.
* 24 голоса с вытеснением, мастер-громкость, mute.
* Позиционирование: затухание по дистанции (30 м), панорама по вектору слушателя.
* Потокобезопасный микс в audio-callback, ядро микса покрыто тестами.

### Пикинг и ввод

* `Scene.pick` / `pickWithRay` — CPU-луч по AABB/сферам видимых мешей, с учётом sphere-коллайдеров.
* `PhysicsWorld.raycast` — точный луч по физическим формам, включая heightfield/hull/mesh.
* `ArcRotateCamera` — ЛКМ-орбита, колесо-зум, лимиты радиуса/угла; `FreeCamera` и `FollowCamera` — см. раздел «Камеры».
* События окна/мыши/клавиатуры через sokol; sandbox показывает полноценное управление.

---

## 🟡 Что сделано частично

| Возможность Babylon.js | В Agate есть | Чего не хватает |
|---|---|---|
| Камеры (Universal/Free/Follow/Target/Fly/VR, мультикамера, viewports) | ArcRotate + Free + Fly + Follow + Target + union Camera | Камера-ригов, мультикамеры и viewport'ов, touch/pinch, инерции |
| Свет (Directional, RectArea, тысячи источников, clustered) | 1 hemi (ambient) + 1 directional (солнце) + 4 point + 2 spot (выбор лучших по камере) | Area-света, кластерного освещения, light probes, нескольких directional |
| Тени (PCF/PCSS/Blur/Contact hardening для всех источников) | CSM для directional, Poisson PCF + PCSS | Теней от point/spot, ESM, каскадных настроек per-light |
| PBR (OpenPBR, clearcoat, sheen, anisotropy, transmission, SSS) | metallic-roughness + IBL | Расширенных слоёв PBR, OpenPBR, unlit-режима |
| Прозрачность | Все alpha-режимы (opaque/cutout/blend) + double-sided (cull-off пайплайны), back-to-front очередь | Сортировки прозрачных инстансов; back-face освещение по геометрическим нормалям |
| Текстуры (EXR/DDS/KTX/Basis, сжатие, видео, anisotropy) | PNG/JPEG RGBA8 + HDR Radiance RGBA16F, equirect→cube, мипмапы, wrap/filter | EXR/сжатых форматов, видеотекстур, анизотропии, render-target/reflection probe текстур |
| Постобработка (DoF, motion blur, TAA, MSAA, glow/highlight, LUT) | ACES/Reinhard, bloom с мип-пирамидой, DoF, цветовые curves, outline-слой, виньетка, CA, FXAA, fog, SSR, SSAO, sharpen, grain, white balance | Motion blur, TAA, MSAA, LUT-текстур, glow/highlight; MSAA выключен (sample_count=1) |
| Анимация (retargeting, GPU-морфы) | Скелетная + node-анимации, морф-таргеты, cubic-spline (Hermite), события/колбэки, easing | GPU-морфов, ретаргетинга, редактора |
| Частицы (GPU-симуляция, sub-emitters, flow maps, spritesheet) | CPU-симуляция + GPU-рендер, спрайт-листы, локальное пространство | GPU-симуляции, sub-emitters, flow maps, коллизий с физикой |
| Меш-билдеры и геометрия (CSG2, LOD, упрощение, decals, GreasedLine) | 15 примитивов + terrain + LOD + Decals | Polygon/N-gon, CSG, упрощения |
| glTF (Draco/meshopt/KTX2, расширения, экспорт) | GLB/GLTF, PBR-текстуры, скины, анимации, морф-таргеты, KHR_lights_punctual-свет, камеры | Draco/meshopt/KTX2, glTF-экспорта |
| Физика (Havok: ragdoll/vehicle/soft body, инспектор) | Box3D + суставы, character, rope, запросы, ragdoll/vehicle-хелперы, debug-линии | Soft body, рендера debug-линий (данные уже генерируются) |
| UI/GUI (полный набор контролов, layout, 3D GUI, редактор) | Immediate-mode примитивы + SDF-текст + checkbox/slider | Инпутов, скроллов, dropdown, гридов/layout, 3D-виджетов, загрузки шрифтов, фокуса/состояния |
| Аудио (файлы, стриминг, шины, эффекты, doppler) | Процедурный синтез + WAV-файлы, позиционирование | mp3/ogg, стриминга, шин/эффектов, doppler/окклюзии |
| Материалы (NodeMaterial, ShaderMaterial, библиотека материалов) | Standard + PBR | Пользовательских шейдеров без правки движка, нодовых материалов, библиотеки (Sky/Gradient/Grid/TriPlanar/…) |
| Инструменты разработчика (Inspector, отладочные оверлеи) | `SceneStats`, debug-режимы SSAO/каскадов, `appendDebugLines` | Инспектора сцены, профилировщика, редактирования на лету |

---

## ❌ Чего нет (потенциальный бэклог, не веб-специфика)

**Камеры и ввод**
* Камера-риги, мультикамеры и viewports, инерция/сглаживание ввода.
* Тач-управление, геймпад, виртуальные джойстики; встроенное управление персонажем (кроме physics character controller).

**Свет и тени**
* Несколько directional-светов одновременно.
* Тени от point/spot, PCSS/contact hardening, ESM, blur-exponential.
* Area (rect) свет, light probes, динамический IBL, кластерное освещение (сотни источников), объёмный свет/атмосфера.

**Материалы и текстуры**
* OpenPBR, clearcoat, sheen, anisotropic, transmission, subsurface.
* NodeMaterial/ShaderMaterial (кастомные шейдеры без пересборки движка), библиотека материалов.
* EXR/DDS/KTX/Basis, сжатые форматы, видеотекстуры, render-to-texture, reflection/refraction probes, кубмапы-зонды.
* Сортировка прозрачных инстансов; back-face освещение по геометрическим нормалям.

**Постобработка и эффекты**
* Depth of Field, motion blur, TAA, MSAA/SSAA, LUT/color curves, bloom с мип-пирамидой, SSR/SSAO более высокого качества.
* Glow layer, highlight layer, outline renderer, lens flares, snapshot-рендер.

**Геометрия**
* Polygon/N-gon-билдеры, толстые GreasedLine-линии, trail.
* CSG/CSG2, mesh simplification, инстансинг с per-instance материалами (PBR-инстансинг отсутствует, instancing только для Standard).
* Blend shapes с GPU-скиннингом (сейчас CPU-блендинг морфов), морфы >8 таргетов.

**Анимация**
* Animation retargeting, ретаргетинг скелетов, редактор анимаций.

**Частицы**
* GPU-симуляция, sub-emitters, flow maps, нодовый редактор частиц, коллизии с физикой.

**Физика**
* Soft bodies; импорт коллайдеров из файлов сцен.

**UI/GUI**
* Grid/layout-контейнеры, привязки/анимации UI, 3D-GUI, загрузка TTF/OTF-шрифтов и Unicode (сейчас зашитый SDF-атлас, ASCII).

**Аудио**
* mp3/ogg (WAV уже поддержан), стриминг, музыкальные циклы, шины/эффекты, doppler, окклюзия.

**Ассеты и данные**
* Экспорт glTF, AssetManager с прогрессом и кэшем.
* Draco/meshopt/KTX2, 3D Tiles.

**Архитектура рендера**
* Frame graph / node render graph, кастомные rendering pipelines, compute-шейдеры.
* Occlusion queries, GPU culling, large world rendering (floating origin).
* Realtime ray tracing/Gaussian splatting (в Babylon 9 тоже отдельные подсистемы).

**Прочее**
* Навигация (navmesh/Recast), crowd simulation, pathfinding.
* Сеть/multiplayer, репликация, WebSocket/WebRTC.
* Behaviors/Actions/Observables как API-слой, теги объектов, smart filters, flow graph.
* Локализация.

---

## 🚫 Что не планируется (Agate — нативный движок, не для веба)

**Платформа и API**
* WebGL/WebGPU-рендер, HTML-канвас, DOM, CSS-интеграция.
* JavaScript/TypeScript API, npm-пакеты, ESM/tree-shaking, сборка под браузер, WASM-таргет.
* Web Workers, browser storage, CDN, асинхронная загрузка браузерными механизмами.

**Инструменты экосистемы Babylon.js**
* Playground, Sandbox (браузерный), Spector.js.
* Inspector (браузерный), Node Material Editor, Node Particle Editor, Node Render Graph Editor, GUI Editor, NME.
* Babylon.js Viewer (HTML-компонент), встраивание на веб-страницу.

**XR и веб-медиа**
* WebXR (VR/AR/immersive), WebXR Layers, hand tracking, WebAudio-движок, видеотекстуры из браузера.
* Babylon Native / React Native — JS-рантайм-мосты (Agate — нативный Zig-движок без JS-слоя).

**Серверный/облачный рантайм**
* Node.js NullEngine, серверный headless-рендер, облачные пайплайны на JS.
* Geospatial/3D Tiles/Cesium-интеграция — привязана к веб-платформе и web-картографии.

> Замечание: пункты из раздела ❌ не следует автоматически считать «не планируется». Это фичи, которых в коде пока нет; какие из них попадут в роадмап Agate — вопрос приоритетов. Раздел 🚫 — принципиально вне области нативного движка.

---

## Источники

* Agate: `agate/src/agate/` (`root.zig`, `scene.zig`, `mesh.zig`, `material.zig`, `texture.zig`, `lights.zig`, `camera.zig`, `particles.zig`, `postprocess.zig`, `ssao.zig`, `ui.zig`, `audio.zig`, `physics.zig`, `animation/`, `loader/`, `passes/`, `shaders/`), `agate/build.zig`, `sandbox/`.
* Babylon.js: официальные docs/features и релиз 9.x (2026) — WebGPU/WebGL2, PBR/OpenPBR, Node Material, GUI, физика Havok, WebXR, frame graph, clustered lighting, volumetric lighting, Gaussian splatting, geospatial, Inspector v2 и др.
