# Agate Roadmap vs Babylon.js: что сделано, что нет, что не планируется

> Это одновременно карта возможностей и очередь работ: всё из раздела **❌** — кандидаты в реализацию, **🚫** — вне области нативного движка.

> Дата: 10.09.2026 (обновлено 22.09.2026).
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
| Потоки и владение GPU | game/render threads, affinity-маркер, неблокирующий render без разыменования мешей, async-ассеты; 3-слотовая ротация prepared-фреймов + consumer pin/lease + slot-owned snapshot; latch live-touch-free (slot records + `commitPublishedRecords`, UI packet handles); concurrent-build primitive landed (`tryClaimBuildSlot`/`BuildClaim`, `releaseHandoff` без flip); atomic handoff edge done (seq words + `build_slot` — `std.atomic.Value`, release/acquire edge); prepareFrame locked-claim + `build_stats` через слот payload; lock-free ADOPTED behind experimental `--concurrent-build` (default OFF; sandbox claim/publish landed, default bit-identical); остаток — burn-in затем default flip + freeze-then-latch, затем снятие phase-mutex | 🟡 |
| Рендер | Forward, 8 пайплайнов, opaque/blend/cutout, per-instance OIT-сортировка, сортировка по пайплайну/текстуре/дистанции | ✅ |
| Frustum culling | AABB + SIMD 4-wide | ✅ |
| Occlusion culling | CPU Hierarchical Z-Buffer (Hi-Z), 9-уровневая пирамида, O(1) AABB-тест, 0 GPU stall/pop-in | ✅ |
| Инстансинг | InstancedMesh + GPU-пайплайны (Standard + Cook-Torrance PBR + IBL + Shadows) | ✅ |
| Камеры | ArcRotate + Free + Fly + Follow + Target, union Camera, мультикамера/PIP, камера-риги (CameraRig: dual, quad, CAD, stereoscopic VR 3D), инерция/сглаживание ввода | ✅ |
| Свет | Hemispheric + до 4 Directional (солнце с CSM + до 3 shadowless fill) + до 4 Point + до 2 Spot + до 2 RectArea (closest-point approximation, без теней) + clustered до 64 point (tile-based forward+, без теней в v1, 2D columns задокументированы) | 🟡 |
| Тени | 4-каскадный CSM для солнца + перспективные тени SpotLight (до 2 прожекторов, 4-tap PCF) + тени PointLight (до 2, 2D-атлас 1536×512, 4-tap PCF, OFF по умолчанию) | ✅ |
| Reflection probes | До 4 зондов, on-demand capture (128px RGBA8-куб + 8 мипов, 6 face-проходов); PBR×3 заменяет IBL-источник, Standard ambient — из coarsest mip; nearest enabled+captured в радиусе, без блендинга | ✅ |
| Материал Standard | Diffuse-цвет/текстура + Unlit-режим | ✅ |
| Материал PBR (metallic-roughness) | Albedo/Normal/MR/Emissive/AO + IBL + Unlit-режим | ✅ |
| OpenPBR, clearcoat, sheen, transmission | PBR clearcoat + sheen (scalar/color + маски/тинт-текстуры), anisotropy v1, thin-film transmission v1 (без refraction RT), SSS v1 (wrap+back-scatter); без OpenPBR | 🟡 |
| Текстуры 2D | PNG/JPEG + HDR (Radiance) через stb_image, RGBA8/RGBA16F, CPU-мипмапы | 🟡 |
| HDR/EXR/DDS, сжатие (Basis/BC/ETC/ASTC), видеотекстуры | KTX2 LDR (мипы, cube, sRGB) + BC7-батч моделей (DamagedHelmet, Lamp, CesiumMan, Fox), HDR Radiance + EXR scanline (HALF/FLOAT, NONE/RLE/ZIPS/ZIP, strict `Texture.fromExrFile/fromExrMemory`) + DDS BC1/BC2/BC3/BC7 (мипы, `Texture.fromDdsFile/fromDdsMemory`) | 🟡 |
| Cube / Skybox / IBL | CubeTexture, equirect → cube, процедурное небо | ✅ |
| Постобработка | ACES/Reinhard, bloom, glow layer (threshold + separable blur + additive, default off), highlight layer (per-mesh inner glow: маска-RT + blur + additive, cap 8, default off), виньетка, CA, sharpen, grain, white balance, FXAA, fog, SSR, SSAO, camera motion blur, TAA (default off) | 🟡 |
| DoF, motion blur, TAA, MSAA, LUT-цветокоррекция | DoF, camera motion blur, TAA (jitter+reprojection+clamp, default off, под MSAA off), цветовые curves и LUT-стрип (2D strip + API) есть; MSAA main target + depth-prepass v1 (`Scene.msaa_depth_prepass`, default off) кормит постэффекты 1x-глубиной | 🟡 |
| Частицы | CPU-симуляция + GPU-инстансы, additive/alpha, local space, спрайт-листы, поворот, sub-emitters (on-death, SplitMix), flow maps (`setFlowMap`), коллизии CPU-частиц (сферы cap 8 + ground plane, kill/bounce), stateful compute-режим (см. GPU-симуляция) | 🟡 |
| GPU-симуляция | Stateful compute-режим частиц (`SimulationMode.compute`, in-place state без ping-pong; гейт `computeAvailable()` + `error.ComputeUnsupported`, без тихого fallback; non-goals: sub-emitter deaths, flow maps, сортировка, коллизии) | ✅ |
| Анимация | Скелетная (до 64 костей, GPU skinning, блендинг/crossfade) + node-анимации glTF TRS + easing | 🟡 |
| События анимаций, ретаргетинг | События/колбэки + ретаргетинг скелетов (name/index/bone_map, rotation_only) | ✅ |
| Меш-билдеры | Box, Sphere, Cylinder, Capsule, Ground, Terrain, Torus, TorusKnot, Disc, Ribbon, Lathe, Plane, Tube, Extrude, Lines, Polygon, TrailMesh | ✅ |
| LOD & Декали | Mesh.addLODLevel / getLOD / getLODForCamera + Sutherland-Hodgman Decal Projector | ✅ |
| CSG (Конструктивная блочная геометрия) | BSP-дерево (splitPolygon, invert, clipTo), Union, Subtract, Intersect, MeshBuilder/Scene интеграция | ✅ |
| Упрощение мешей (Mesh simplification) & Greased Lines | Garland-Heckbert QEM edge-collapse + автоматическая генерация LOD-уровней; GreasedLine: ribbon/billboard толстые 3D-линии, miter joints, multi-path, per-vertex width/color, UV/dashed modes | ✅ |
| glTF/GLB | PBR, сэмплеры, скины, анимации, морфы, свет/камеры (KHR_lights_punctual), квантование (KHR_mesh_quantization), авто-нормали | 🟡 |
| Draco/meshopt, KTX2-транскодинг (Basis), экспорт | KTX2-контейнер (LDR + BC7-батч через `sandbox/tools/convert_ktx2.sh`) в glTF-загрузке, `KHR_texture_basisu` в whitelist `extensionsRequired` | 🟡 |
| Физика | Box3D: коллайдеры, compound, суставы, character, rope, события, запросы AABB/сфера/точка, ragdoll/vehicle-хелперы | ✅ |
| Soft body | PBD cloth v1 (Verlet 2–64, Jacobi constraints, pinned, sphere/floor коллайдеры, cap 4; non-goals: self-collision, tearing, fluids, box3d coupling, GPU sim, cloth-cloth, persistence) | ✅ |
| Debug-рендер физики | генерация линий коллайдеров (`appendDebugLines`) + 3D-пасс линий (depth-tested) | ✅ |
| UI | Экранный canvas, SDF-текст + TrueType (glyf-парсер, cmap 4/12, композитные глифы, kern fmt0, scanline-растеризатор, атлас), кнопки/панели, checkbox, slider, dropdown, скролл, text input, CSS-темы, анимации переходов + 3D world-space панели (до 4, pick+inject, render-on-demand) | ✅ |
| Layout-контейнеры, Flex/Grid UI | LayoutStack: HStack, VStack, Flexbox, CSS Grid (fr/px/%), 9-point Anchors, Docking (top/bottom/left/right/fill), Spacers, Spans | ✅ |
| Аудио | Процедурный синтез + WAV-файлы, 24 голоса, динамический реестр шин, DAG-иерархия, затухание (linear/inv/exp), Doppler, DSP-фильтры (biquad IIR), стерео-реверберация Freeverb, звуковая окклюзия геометрией/физикой (multi-tap raycast, LPF muffling), OGG/MP3/WAV потоковый стриминг с диска/памяти, SPSC lock-free кольцевые буферы и кроссфейдинг музыки | ✅ |
| mp3/ogg, стриминг, шины, эффекты | OGG Vorbis (`stb_vorbis`), MP3 (`dr_mp3`), WAV стриминг с диска и памяти, SPSC lock-free ring buffer, gapless loop, crossfade, динамические шины (DAG-дерево, biquad low/high/band/notch, Freeverb reverb, окклюзия геометрией) | ✅ |
| Пикинг и теги объектов | CPU-луч (AABB/сфера/треугольник), raycast в физике, точный raycast по инстансам (InstancedMesh), теги объектов (TagSet) и булевы смарт-фильтры (TagQuery: and/or/not/parentheses), Scene.pickWithRayTag | ✅ |
| Сериализация сцены (бинарный AGSC v1-v3: TRS/материалы/свет/камера/post FX/entity IDs/custom properties), экспорт | ✅ |
| Навигация/crowd/pathfinding | NavMesh (dual-graph, slope filter, grid builder), A* поиск, Funnel (string-pulling), NavAgent | ✅ |
| Сеть/multiplayer | — | ❌ |
| Frame graph, volumetric, Gaussian splatting | — | ❌ |
| Large world rendering, geospatial | — | ❌ |
| Тесты/бенчмарки | 1173 unit-тестов, встроенный профилировщик (HTML/JSON trace), `zig build test`, `zig build fmt`, `sandbox --bench` | ✅ |
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

Отложено осознанно: MSAA (в этой версии sokol нет depth-resolve — закрыто волной 39: single-sample depth-prepass `Scene.msaa_depth_prepass`, default off, см. таблицу постобработки), LUT-текстура (сейчас параметрические curves), тени от point/spot, motion blur/TAA.

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

### Волна 9: Перспективные тени прожекторов (SpotLight Shadows) (12.09.2026)

| Фича | Файлы | Описание | Статус |
|---|---|---|---|
| Математика и проекции SpotLight | `lights.zig` | Перспективная матрица отсечения (`fov = 2 * outer_angle_deg`, `aspect = 1.0`, `near/far = range`), lookAt-матрица вида, настраиваемые `shadow_bias` и `shadow_normal_bias` | ✅ |
| Атлас теней прожекторов | `passes/shadow_pass.zig` | Depth-атлас 1024×512 (2 тайла 512×512 для Spot 0 и Spot 1), рендеринг геометрии через переиспользуемые бакеты пайплайнов (Standard, Instanced, Skinned), пропуск при отсутствии источников | ✅ |
| Шейдерная фильтрация PCF | `shaders/{standard,pbr,instanced,skinned_pbr}.glsl` | Вычисление перспективных координат в атласе, 4-tap PCF фильтрация с защитой от выхода за границы тайлов, нормальный сдвиг поверхности (normal bias) против теневых артефактов (shadow acne) | ✅ |
| Интеграция и интерактивное демо | `scene.zig`, `scene/draw.zig`, `sandbox_scene.zig`, `sandbox_ui.zig` | Динамический качающийся прожектор над сценой с персонажами (Fox и CesiumMan), отбрасывание честных теней от анимированных скелетных моделей на пол, UI-кнопка переключения `SpotShd: ON/OFF` | ✅ |

### Волна 10: PBR-инстансинг (Cook-Torrance Instanced PBR) (12.09.2026)

| Фича | Файлы | Описание | Статус |
|---|---|---|---|
| Шейдер `instanced_pbr.glsl` | `shaders/instanced_pbr.glsl`, `build.zig` | Полный Cook-Torrance PBR для инстансированных мешей: поинстансные матрицы (`ATTR_inst_mat0..3`), TBN-базис нормалей, GGX NDF, Smith geometry, Fresnel-Schlick, 4-каскадный CSM (PCSS/PCF), SpotLight тени (4-tap PCF), IBL-отражения кубмапы неба, альфа-тест cutout | ✅ |
| Пайплайны PBR-инстансинга | `scene/pipelines.zig` | Семейство `instanced_pbr`, Buffer 0 (`Vertex` 80 байт: pos, norm, tan, col, uv), Buffer 1 (`Mat4` 64 байт: `step_func = .PER_INSTANCE`), 4 базовых пайплайна (u16/u32 opaque/blend) + 4 double-sided cull-off твина | ✅ |
| Интеграция в Scene & Draw | `scene.zig`, `scene/draw.zig` | Автоматический выбор PBR-инстанс пайплайнов при `mesh.material == .pbr`, биндинг текстур albedo/normal/metallic-roughness/emissive/occlusion/env cubemap/shadow maps, передача параметров в UB_vs_params и UB_fs_params | ✅ |
| Sandbox-витрина | `sandbox_showcase.zig`, `sandbox_ui.zig` | Интерактивная витрина «Instanced PBR Grid» (64 медных/золотых сферы 8×8 с плавной синусоидальной волной над темным пьедесталом, честные CSM-тени и IBL-блики в 1 draw call), плавающий HUD-бейдж и отображение в debug-панели | ✅ |

Проверки: 320 unit-тестов, `zig build test` (agate) и `zig build` (sandbox) проходят чисто без ошибок и предупреждений, 340+ FPS в runtime.

### Волна 11: Polygon Builder с отверстиями и динамические шлейфы TrailMesh (12.09.2026)

| Фича | Файлы | Описание | Статус |
|---|---|---|---|
| Polygon Builder с Ear-Clipping | `mesh/builders.zig`, `mesh/builder.zig`, `mesh.zig`, `root.zig` | Генератор 2D и 3D экструдированных полигонов (`PolygonOptions`, `PolygonPlane`, `PolygonSideOrientation`): автоматическая нормализация обхода (CCW контур, CW отверстия), лучевой разрез и сшивка произвольного числа отверстий (Hole Bridging), устойчивая ear-clipping триангуляция, генерация боковых граней призмы с расчетными тангентами и нормалями | ✅ |
| Динамический TrailMesh | `mesh/trail.zig`, `mesh/builder.zig`, `mesh.zig`, `root.zig` | Шлейфы движения за объектами (`TrailMesh`, `TrailOptions`): кольцевой буфер узлов с фильтрацией минимальной дистанции и старением/затуханием, billboard-ориентация квад-стрипа лицом к камере, сужение ширины (taper), интерполяция цвета вершины по времени жизни, динамическое GPU-обновление вершинного и индексного буферов (`sg.updateBuffer`) | ✅ |
| Интеграция со сценой | `scene.zig`, `root.zig` | Регистрация и управление шлейфами сцены (`Scene.createTrailMesh`, `Scene.updateTrails(dt)`), автоматическое освобождение GPU-ресурсов в `Scene.deinit`, экспорт фасадов в `root.zig` | ✅ |
| Sandbox-витрина и тесты | `sandbox_showcase.zig`, `sandbox_scene.zig`, `sandbox_ui.zig`, `mesh/tests.zig` | Золотая 3D 8-конечная звезда с квадратным отверстием на подиуме, спаренные шлейфы за спорткаром (`car_trail_left`, `car_trail_right`) и огненный шлейф за летающей сферой света (`orb_trail`), 6 новых unit-тестов для выпуклых/вогнутых полигонов, отверстий, экструзии и старения шлейфов | ✅ |

Проверки: 326 unit-тестов, `zig build test` (agate) и `zig build` (sandbox) проходят чисто без предупреждений, >350 FPS в runtime.

### Волна 12: AI-навигация (NavMesh, Funnel String-Pulling, A* Pathfinding, NavAgent) (12.09.2026)

| Фича | Файлы | Описание | Статус |
|---|---|---|---|
| NavMesh и генераторы | `ai/navmesh.zig`, `ai.zig`, `root.zig` | 3D навигационная полигональная сетка (`NavMesh`, `NavNode`): дуальный граф связности смежных треугольников через ребра (`neighbors`), фильтрация крутых уклонов (`max_slope_deg`), ориентированное определение принадлежности 2D/3D точек (`containsPointXZ`), автоматический построитель прямоугольных сеток с препятствиями (`buildGrid`) с защитой от касания границ (`obstacle_boxes`), проекция и клампинг высоты (`clampToMesh`) | ✅ |
| Funnel Algorithm (String-Pulling) | `ai/funnel.zig`, `ai.zig`, `root.zig` | Алгоритм натяжения струны сквозь порталы коридора треугольников: определение левых и правых порталов ребер, 2D ориентированная площадь (`triArea2D`), схлопывание коридора в минимальную гладкую ломаную траекторию без зигзагов | ✅ |
| A* Pathfinding | `ai/pathfinding.zig`, `ai.zig`, `root.zig` | Эвристический поиск пути по дуальному графу центроидов узлов на `std.PriorityQueue`: нахождение стартового и целевого узлов, расчет коридора смежных треугольников, извлечение общих порталов и вызов Funnel алгоритма для получения 3D polyline пути | ✅ |
| NavAgent и интеграция со сценой | `ai/agent.zig`, `scene.zig`, `root.zig` | Автономный навигационный агент (`NavAgent`): следование по вейпоинтам с настраиваемой скоростью (`speed`), дистанцией прибытия (`stopping_distance`), плавным угловым подруливанием (`rotation_speed`, `yaw`), привязкой к высоте меша (`snap_to_mesh`), методами `setDestination`, `teleport`, автоматическое обновление агентов в `Scene.updateNavAgents(dt)` и очистка в `Scene.deinit` | ✅ |
| Sandbox-витрина и тесты | `sandbox_showcase.zig`, `sandbox_scene.zig`, `sandbox_ui.zig`, `main.zig`, `ai/tests.zig` | Навигационная арена во внутреннем дворе замка с тремя гранитными обелисками-препятствиями, автономный парящий дрон-компаньон со шлейфом и подсветкой, режимы патрулирования и преследования игрока по клавише `[[]`, HUD-статус в demo-панели и шпаргалке управления, 8 unit-тестов (смежность, препятствия, обход углов A*, Funnel, телепорт) | ✅ |

Проверки: 334+ unit-тестов, `zig build test` (agate) и `zig build` (sandbox) проходят чисто без предупреждений, >360 FPS в runtime.

### Волна 13: CSG (Constructive Solid Geometry) булевы операции (12.09.2026)

| Фича | Файлы | Описание | Статус |
|---|---|---|---|
| BSP-дерево и полигональное разбиение | `mesh/csg.zig`, `mesh.zig`, `root.zig` | Ядро BSP-дерева (`CSGNode`, `CSGPlane`, `CSGPolygon`, `CSGVertex`): разбиение выпуклых полигонов произвольной плоскостью (`splitPolygon`), классификация вершин (FRONT, BACK, COPLANAR, SPANNING) с эпсилон-допуском $10^{-5}$, рекурсивная инверсия твердотельного объема (`invert`), фильтрация и отсечение геометрии (`clipPolygons`, `clipTo`, `build`) | ✅ |
| Булевы операции над твердыми телами | `mesh/csg.zig`, `root.zig` | Симметричные алгоритмы булевой геометрии на базе BSP: объединение (`unionWith`), вычитание (`subtract`), пересечение (`intersect`), поддержка цепочек операций и защита от непересекающихся/вырожденных тел | ✅ |
| Конвертация геометрии и мешей | `mesh/csg.zig`, `mesh/builder.zig`, `scene.zig`, `root.zig` | Конструкторы `CSG.fromBox`, `CSG.fromSphere`, `CSG.fromCylinder`, `CSG.fromGeometryData`, `CSG.fromMesh`, обратная триангуляция выпуклых $N$-гонов веером (`toGeometryData`), пересчет касательных (tangents) и AABB, создание и загрузка GPU-меша (`Scene.createCSGMesh`) | ✅ |
| Sandbox-витрина и тесты | `sandbox_showcase.zig`, `sandbox_ui.zig`, `mesh/csg_tests.zig` | Культовая CAD-скульптура в галерее на обсидиановом пьедестале: скругленный куб (Box ∩ Sphere) со сквозными перфорациями по всем 3 осям (- 3×Cylinder), вращение в реальном времени, PBR-материал «розовое золото», 7 unit-тестов булевых операций и BSP-плоскостей | ✅ |

Проверки: 341+ unit-тестов, `zig build test` (agate) и `zig build` (sandbox) проходят чисто без предупреждений, >360 FPS в runtime.

### Волна 14: Иерархическое программное отсечение невидимой геометрии (Hierarchical Software Occlusion Culling / Hi-Z) (12.09.2026)

| Фича | Файлы | Описание | Статус |
|---|---|---|---|
| Иерархический Z-буфер (Hi-Z) | `visibility/hiz_buffer.zig`, `visibility/mod.zig`, `root.zig` | 9-уровневая консервативная пирамида глубин $256 \times 128 \dots 1 \times 1$ (175 КБ, 100% помещается в L1/L2 CPU-кэш): сверхбыстрый downsample с операцией $\max$, исключающей ложные отсечения (zero false-positives), проекция 8 вершин AABB и $O(1)$ тест видимости по $2 \times 2$ области соответствующего мип-уровня | ✅ |
| Программный растеризатор окклюдеров | `visibility/rasterizer.zig`, `root.zig` | Высокопроизводительный CPU-растеризатор треугольников и ориентированных OBB-боксов: отсечение ближней плоскостью камеры (near-plane clipping $w \ge 0.001$), 2D backface culling, субпиксельные барицентрические координаты, субмиллисекундная растеризация окклюдеров | ✅ |
| Высокоуровневый OcclusionCuller & интеграция | `visibility/culler.zig`, `scene.zig`, `mesh/types.zig`, `mesh/mesh.zig` | Менеджер отсечения сцены (`OcclusionCuller`): регистрация окклюдеров (`mesh.is_occluder`), автоматическая растеризация геометрии/боксов в Phase -1, отсечение скрытых за стенами мешей и инстансов до отправки на GPU, интеграция со `SceneStats` (`occluded_meshes`, `occluders_count`, `occluder_triangles`) | ✅ |
| Sandbox-лаборатория и тесты | `sandbox_showcase.zig`, `sandbox_ui.zig`, `main.zig`, `visibility/tests.zig` | Интерактивная витрина «Occlusion Lab»: циклопическая крепостная стена-окклюдер и скрытая сокровищница из 32 рубиновых кристаллов за ней; мгновенное отсечение 32 мешей без малейшего pop-in, горячая клавиша `[]]` и строка в HUD, 6 unit-тестов математики, консервативности и бенчмарк 1000 AABB запросов | ✅ |

Проверки: 347+ unit-тестов, `zig build test` (agate) и `zig build` (sandbox) проходят чисто без предупреждений, >350 FPS в runtime.

### Волна 15: стабилизация потоков и владения ресурсами (15.09.2026)

Четыре волны после аудита `REFACTOR.md` (детали и инварианты — там же).
Коммиты agate: `c64e8be` (стабилизация), `8e27ab7` (потоковая модель),
`38f9a9e` (жизненный цикл), `222722b` + `0e8d413` (off-context загрузки,
форматирование); sandbox: `e977606`, `605045a`.

| Направление | Файлы | Описание | Статус |
|---|---|---|---|
| Affinity-маркер графического потока | `gpu_thread.zig`, оба `main.zig` | `markContextThread` / `isOnContextThread` / `assertOnContextThread` (Debug+ReleaseSafe); приложения маркируют sapp-поток | ✅ |
| Отложенное уничтожение | `scene.zig`, `mesh/mesh.zig` | `destroyMesh` вне контекста снимает меш и складывает в очередь (+фиксированный overflow на OOM), дренаж на render-start и в `deinit` | ✅ |
| Отложенное создание | `mesh/mesh.zig`, `loader/mesh_spawn.zig`, `morph_gpu.zig` | `uploadGeometry` и glTF-примитивы строятся CPU-only (`gpu_pending`, `pending_dynamic_update`, `morph_upload_pending`), буферы собирает `finishGpuUpload` | ✅ |
| Частицы и trail | `particles.zig`, `mesh/trail.zig` | буферы создаются на flush-стороне; частичное создание уничтожает недоделанный handle | ✅ |
| Очистка referent'ов | `scene.zig` | тело физики, `parent`, `attach_bone`, LOD-записи, декаль-инстансы и morph-привязки анимаций разрываются до освобождения | ✅ |
| Mailbox newest-wins | `handoff.zig`, `scene.zig` | `releasePublished` + перепубликация при насыщении: свежий кадр и light pack всегда побеждают | ✅ |
| Очереди рендера | `scene/render_queue.zig` | parallel cull skip инстансов, OOM-fallback в serial, единый tie-break по индексу меша, guard tail-чанков | ✅ |
| Picking | `scene/picking.zig` | sphere-тест в локальном пространстве через обратную матрицу (точно для parented/rotated/non-uniform) | ✅ |
| Текстуры | `texture.zig`, `ktx2.zig`, `loader/materials.zig` | checked-размеры `buildRaw`, KTX2 error-маппинг, off-context assert и авто-async для GLB | ✅ |
| Инструменты | `sandbox/main.zig` | `--test-decal`, `--test-async-load`, mouse-mailbox вместо флуда кольца | ✅ |

Проверки волны: 584/584 (Debug + ReleaseSafe), `zig build fmt` зелёный,
smoke-набор agate и sandbox (включая `--test-decal` и `--test-async-load`) чистый.

### Волна 16: Инстанс-пикинг, профилировщик, OIT-сортировка и формат сцены v3 (15–16.09.2026)

Коммиты: `b5cb6d7`, `4170bfc`.

| Направление | Файлы | Описание | Статус |
|---|---|---|---|
| Точный инстанс-пикинг | `scene/picking.zig`, `scene.zig` | `pickWithRay` проверяет каждый видимый инстанс в пространстве отрисовки с корректным inverse-transpose нормалей и возвратом индекса `picked_instance` | ✅ |
| Дозирование GPU-загрузок | `assets.zig`, `scene.zig` | `UploadQueue.drainCounted` с покадровым бюджетом (`upload_budget_per_frame = 4`), выделенный поток `io_runner` под сохранение/загрузку | ✅ |
| Встроенный профилировщик | `profiler.zig`, `main.zig`, `root.zig` | Замер фаз (Update, Prepare, Shadow, Main, PostFX), эвристическая диагностика боттлнеков с рекомендациями, генерация HTML с интерактивным SVG-таймлайном, Markdown и Chrome Trace JSON (`chrome://tracing`) | ✅ |
| Снимки памяти | `profiler.zig`, `mesh/mesh.zig` | `MemorySnapshot`: раздельный учёт памяти CPU-геометрии, общих аллокаций и видеопамяти VRAM (буферы и текстуры GPU) | ✅ |
| OIT-сортировка прозрачности | `scene/render_queue.zig` | Поинстансная back-to-front сортировка прозрачных инстансов внутри `submitInstancedMesh` | ✅ |
| Сериализация сцены v3 | `serialization.zig` | Бинарный формат v3: поддержка Entity ID, иерархии нод, кастомных игровых свойств с обратной совместимостью с v1 и v2 | ✅ |
| Дедупликация текстур | `assets.zig` | Проверка существующих текстур в очереди (`findFile`, `getOrRequestFile`) для предотвращения повторного декодирования | ✅ |

### Волна 17: Полностью неблокирующий рендер без разыменования мешей (16.09.2026)

Коммит: `ea5ee2e`.

| Направление | Файлы | Описание | Статус |
|---|---|---|---|
| Изоляция данных рендера | `scene/render_queue.zig`, `scene.zig` | Упаковка всех GPU-хэндлов, параметров материалов и матриц в `RenderMeshItem` и `RenderInstancedBatch` в `prepareFrame()` | ✅ |
| Zero-dereference render | `scene/draw.zig`, `scene.zig` | Полное исключение чтения и разыменования указателей `*Mesh` во время `Scene.render()` | ✅ |
| Lock-free shadow & outline | `passes/shadow_pass.zig`, `passes/outline_pass.zig` | Автономные `ShadowDrawItem` и `OutlineDrawItem` со своими предвычисленными бинами и матрицами | ✅ |
| Сужение мьютекса фаз | `scene.zig`, `main.zig` | `phase_mutex` удерживается только во время `prepareFrame()`; `Scene.render()` исполняется параллельно и lock-free относительно игрового цикла | ✅ |

### Волна 18: Квантование glTF, генерация нормалей, unlit-материалы и Camera Motion Blur (16.09.2026)

Коммиты: `4d0aeb7`, `9624277`.

| Направление | Файлы | Описание | Статус |
|---|---|---|---|
| glTF KHR_mesh_quantization | `loader/mesh_spawn.zig`, `mesh.zig` | Декодирование нормализованных 8- и 16-битных целочисленных атрибутов (позиции, нормали, UV) | ✅ |
| Автогенерация нормалей glTF | `mesh/tangents.zig`, `loader/scene_loader.zig` | `computeMissingNormals`: взвешенный по площади расчёт нормалей граней для мешей без атрибута нормалей | ✅ |
| Оптимизация биннинга теней | `passes/shadow_pass.zig`, `scene/uniforms.zig` | Корректные границы бинов теней, предвычисление uniform-буферов | ✅ |
| Unlit-режим материалов | `material.zig`, `loader/materials.zig`, шейдеры | Флаг `Material.unlit = true` отключает расчёт света и теней в Standard и Cook-Torrance PBR шейдерах для стилизованной графики и UI | ✅ |
| Camera Motion Blur | `postprocess.zig`, `passes/postprocess_pass.zig`, `shaders/postprocess.glsl` | Полноэкранный эффект размытия движения камеры по delta VP-матрице с настраиваемым числом выборок (до 16) и интенсивностью | ✅ |

### Волна 19: Пользовательская аудио-система: динамические DAG-шины, затухание и эффект Доплера (16.09.2026)

Коммиты: `9624277`, `3aad9ff`, `bebad8f`, `691cf09`.

| Направление | Файлы | Описание | Статус |
|---|---|---|---|
| Открытый реестр шин | `audio.zig`, `root.zig` | Открытый тип `BusId = enum(u8) { _, pub const invalid }` — движок не навязывает enum, пользователь сам создаёт мастер-, SFX-, музыку или любые другие шины | ✅ |
| Прямой роутинг звуков | `audio.zig` | Воспроизведение без шины (`bus: ?BusId = null`) направляет голос или клип напрямую на мастер-выход | ✅ |
| DAG-иерархия шин | `audio.zig` | Дерево шин: каскадное наследование эффективной громкости и mute с защитой от циклов (`getBusEffectiveVolume`), автоматический reparenting дочерних шин при уничтожении родителя | ✅ |
| Безопасная емкость без аллокаций | `audio.zig`, `root.zig` | Фиксированный статический потолок до 128 шин с нулевыми аллокациями в аудиопотоке, настройка активной ёмкости при инициализации (`AudioConfig{ .max_buses = 32 }`) | ✅ |
| 3D Spatial Audio & затухание | `audio.zig` | Модели `linear`, `inverse`, `exponential` с параметрами `min_distance`, `max_distance`, `rolloff`, переключение spatial/non-spatial на лету (`setBusSpatial`) | ✅ |
| 3D Эффект Доплера | `audio.zig` | Расчёт изменения частоты по векторам скоростей слушателя и источников звука (`setListenerVelocity`, скорость голоса/клипа, `doppler_factor`) | ✅ |
| Интеграция с синтезатором и сценой | `audio.zig`, `sandbox_scene.zig`, `sandbox_ui.zig` | Хелперы `playImpactOn`, `playExplosionOn`, `playBlipOn`, полное управление шинами в UI и сцене sandbox | ✅ |

Проверки: 613/613 unit-тестов, `zig build test` (agate) и `zig build` (sandbox) проходят за ~1 сек, >200 FPS в runtime.

### Волна 20: Audio DSP-эффекты и фильтры на шинах (Biquad IIR, Freeverb Reverb, DAG-микширование) (16.09.2026)

| Направление | Файлы | Описание | Статус |
|---|---|---|---|
| Biquad IIR фильтры | `audio/dsp.zig`, `audio.zig`, `root.zig` | 2-й порядок Robert Bristow-Johnson (Cookbook) в форме Transposed Direct Form II: lowpass, highpass, bandpass, notch; защита от denormals, in-place обработка блоками | ✅ |
| Freeverb стерео-ревербератор | `audio/dsp.zig`, `audio.zig`, `root.zig` | Алгоритмический стерео-ревербератор: пул из 8 параллельных гребенчатых (LBCF) и 4 последовательных аллпасс (APF) фильтров на канал, кольцевые буферы со степенями двойки (`& 2047`, `& 1023`), масштабирование задержек под частоту дискретизации | ✅ |
| DAG-микширование блоками (chunks) | `audio.zig` | Обработка звука блоками по 64 фрейма (128 сэмплов = 512 байт на шину, L1 cache-friendly), топологическая сортировка шин в аудиопотоке (листья -> родители -> корни), каскадирование эффектов и громкости без аллокаций | ✅ |
| Пресеты и управление на лету | `audio.zig` | Геймплейные пресеты (`setBusUnderwater`, `setBusMuffled`, `setBusTelephone`, `setBusCaveReverb`, `setBusRoomReverb`), атомарное изменение среза (`cutoff`) и резонанса (`q`) без щелчков | ✅ |
| Интеграция с UI и сценой sandbox | `sandbox_scene.zig`, `sandbox_ui.zig`, `main.zig` | Хоткеи F9 (Underwater Lowpass), F10 (Cave Reverb) и интерактивные кнопки в HUD | ✅ |
| Полная безопасность реального времени | `audio.zig`, `audio/dsp.zig` | 0 динамических аллокаций в аудиопотоке sokol-audio, lock-free атомарные параметры, автоматический tail-процессинг реверберации после завершения голосов | ✅ |

Проверки: 621/621 unit-тест, `zig build test` (agate) и `zig build` (sandbox) проходят без ошибок.

### Волна 21: Акустическая окклюзия звука геометрией и физикой (16.09.2026)

| Направление | Файлы | Описание | Статус |
|---|---|---|---|
| Конфигурация окклюзии | `audio/occlusion.zig`, `audio.zig`, `root.zig` | `AudioOcclusionConfig` с гибкими параметрами: `min_volume` (0.25), `min_cutoff` (500 Гц), `max_cutoff` (20000 Гц), `num_rays` (1..5), `spread_radius` (0.6 м), `smooth_time` (0.15 с) | ✅ |
| Multi-tap Raycast & Дифракция | `audio/occlusion.zig` | Трассировка лучей окклюзии (1 прямой луч или 5-лучевой ортогональный дифракционный паттерн вокруг источника звука для реалистичного огибания углов и препятствий) | ✅ |
| Временное сглаживание (Temporal Smoothing) | `audio/occlusion.zig` | `AudioOcclusionTracker`: экспоненциальное фильтрование окклюзии без щелчков (`1 - exp(-dt / smooth_time)`), мгновенное или плавное применение | ✅ |
| AudioEmitter компонент | `audio/occlusion.zig`, `root.zig` | Высокоуровневая структура 3D-источника звука с позицией, скоростью, шиной, трекером окклюзии и методом `update(dt, listener_pos, raycast_fn, user_data)` | ✅ |
| Окклюзия процедурных голосов и WAV-клипов | `audio.zig` | Поддержка `occlusion` и `occlusion_config` в `PlayOptions` и `ClipPlayOptions`; затухание громкости и срез частот (однополюсный IIR-фильтр низких частот в реальном времени для WAV-сэмплов) | ✅ |
| Окклюзия на аудиошинах | `audio.zig` | Выделенные IIR biquad lowpass фильтры `bus_occlusion_filters` на каждой шине; `setBusOcclusion`, `updateBusOcclusion`, `updateBusOcclusionWithRaycast`; независимость от художественных EQ/фильтров | ✅ |
| Адаптеры физики и сцены | `physics/world.zig`, `scene.zig` | `evaluateAudioOcclusion` и `audioRaycastAdapter` для быстрой проверки препятствий через физические коллайдеры `PhysicsWorld` и полигональные меши `Scene.pickWithRay` | ✅ |
| Тесты и потокобезопасность | `audio/tests.zig`, `audio/occlusion.zig` | 8 новых модульных тестов, атомарные значения окклюзии (`@bitCast(f32)`), 0 аллокаций в аудиопотоке | ✅ |

Проверки: 629/629 unit-тестов, `zig build test` (agate) и `zig build` (sandbox) проходят без ошибок.

### Волна 22: Воспроизведение звуков и потоковый стриминг playSound (OGG, MP3, WAV), кроссфейд и управление стримами (17.09.2026)

| Направление | Файлы | Описание | Статус |
|---|---|---|---|
| Форматы и декодеры | `audio/stream.zig`, `audio.zig`, `root.zig` | Инкрементальное декодирование чанками для OGG Vorbis (`stb_vorbis`), MP3 (`dr_mp3`) и WAV (RIFF/PCM 8/16/24/32-bit и IEEE float32) из файлов и памяти | ✅ |
| SPSC Lock-free Ring Buffer | `audio/stream.zig` | Безопасная передача аудиоданных из фонового/главного потока в аудиопоток sokol-audio; бинарная маска степеней двойки (65536 стерео-фреймов), 0 динамических аллокаций и 0 I/O в аудиоколлбэке | ✅ |
| Бесшовный лупинг и ресемплинг | `audio/stream.zig` | Gapless looping при достижении конца трека, линейная интерполяция/ресемплинг произвольных sample rate (22.05, 44.1, 48 кГц и др.) к целевой частоте дискретизации движка | ✅ |
| Унифицированный API `playSound` | `audio.zig`, `audio/stream.zig` | `playSound` (файл), `playSoundFromMemory` (память), `playSoundOnce` / `playSoundOnceFromMemory` с автоматическим освобождением `auto_destroy` по завершении; потолок увеличен до 32 одновременных стримов (`max_streams = 32`) | ✅ |
| Управление звуковыми потоками | `audio.zig`, `audio/stream.zig` | `stopSound`, `pauseSound`, `resumeSound`, `setSoundVolume`, `stopAllSounds` с опциональным плавным фейд-аутом (`fade_duration`), точный сик (`seekToSeconds`/`seekToFrame`) | ✅ |
| Плавный фейдинг и кроссфейд | `audio/stream.zig`, `audio.zig` | Встроенный `fadeTo` (fade-in / fade-out с авто-остановкой), универсальный кроссфейдинг `crossfadeSound` / `crossfadeSoundFromMemory` с одновременным затуханием старого звука и нарастанием нового | ✅ |
| Интеграция с DAG-шинами и DSP | `audio.zig` | Стримы микшируются прямо в чанки назначенных шин (`bus_chunks[idx]`), автоматически наследуя все эффекты шин: Biquad IIR EQ/фильтры, глушение окклюзией и стерео-реверберацию Freeverb | ✅ |
| Модульные тесты | `audio/tests.zig`, `tests.zig` | 10 модульных тестов: OGG/MP3/WAV воспроизведение, авто-освобождение памяти по завершении, бесшовный лупинг, кроссфейд, пауза/возобновление/громкость, маршрутизация в шину с IIR LPF | ✅ |

Проверки: 639/639 unit-тестов, `zig build test` (agate), `zig fmt --check src/` и `zig build` (sandbox) проходят без ошибок.

### Волна 23: UI Layout Containers, Flexbox, CSS Grid и 9-точечное позиционирование (17.09.2026)

| Направление | Файлы | Описание | Статус |
|---|---|---|---|
| Гибкие размеры `UISize` | `ui/layout.zig`, `ui.zig`, `root.zig` | `UISize`: `.fixed(px)`, `.percent(%)`, `.flex(weight)` / `.fill`, `.auto` с чистым разрешением `resolve(available, auto_content)` | ✅ |
| Инденты и отступы `UIEdges` | `ui/layout.zig`, `ui.zig`, `root.zig` | 4-сторонние отступы `UIEdges`: `.all(v)`, `.symmetric(h, v)`, `.trbl(t, r, b, l)`, `.horizontal(h)`, `.vertical(v)`, операции `inset` и `outset` прямоугольников | ✅ |
| 9-точечные якоря `UIAnchor` | `ui/layout.zig`, `ui.zig`, `root.zig` | `UIAnchor` (top_left, top_center, top_right, center_left, center, center_right, bottom_left, bottom_center, bottom_right) с автоматическим учётом полей `margin` через `anchorRect` | ✅ |
| Докинг панелей `UIDock` | `ui/layout.zig`, `ui.zig`, `root.zig` | `UIDock` (top, bottom, left, right, fill): последовательное «вырезание» экранного пространства под тулбары, сайдбары и статус-бары через `dockRect` с мутацией оставшегося прямоугольника | ✅ |
| Flexbox Solver `solveFlex` | `ui/layout.zig`, `ui.zig`, `root.zig` | 1D Flexbox солвер без аллокаций памяти: row/column, реверс, распределение `flex`-весов, `JustifyContent` (start, center, end, space_between, space_around, space_evenly), `AlignItems` (start, center, end, stretch) и `align_self` | ✅ |
| CSS Grid Solver `solveGridTracks` | `ui/layout.zig`, `ui.zig`, `root.zig` | Многоколоночный и многострочный солвер треков: поддержка пикселей (`.px`), процентов (`.percent`) и долей (`.fr`), многоячеечные спаны `placeGridSpan` (colspan / rowspan) в `AdvancedGridSpec` | ✅ |
| Интеграция с `LayoutStack` | `ui.zig`, `root.zig` | Методы `beginFlex`, `beginHStack`, `beginVStack`, `beginGrid` с `UIEdges`-паддингом, `placeSize`, `placeFlex`, `spacer`, `spacerWeight`, `placeGridSpan`, `anchor`, `dock` | ✅ |
| Immediate-mode виджеты | `ui.zig`, `root.zig` | Размещение виджетов прямо на `LayoutStack` в 1 строчку: `label`, `button` (возвращает `bool` клика), `checkbox`, `slider`, `progressBar`, `divider`, `badge` с автоматическим mouse input canvas (`setInput`) | ✅ |
| Модульные тесты | `ui/layout.zig`, `ui.zig` | 14 новых unit-тестов: размеры UISize, UIEdges inset/outset, 9-точечные якоря, докинг, Flexbox flow, CSS grid tracks & spans, LayoutStack padding/spacer/flex/grid/anchor/dock и immediate-mode виджеты | ✅ |

Проверки: 653/653 unit-тестов, `zig build test` (agate), `zig fmt --check src/` и `zig build` (sandbox) проходят без ошибок.

### Волна 24: Greased Lines и QEM-упрощение мешей с автоматической генерацией LOD (17.09.2026)

| Направление | Файлы | Описание | Статус |
|---|---|---|---|
| Greased Lines Builder & Mesh | `mesh/greased_line.zig`, `mesh.zig`, `scene.zig`, `root.zig` | `GreasedLineOptions`, `buildGreasedLineData`: 3D-полилинии с настраиваемой шириной (константная или per-vertex `widths`), сглаживание стыков (miter joints с `miter_limit`), единая геометрия для multi-path линий, поддержка замкнутых петель (`closed`), UV-режимы (`.relative`, `.unit_length`) для пунктиров/дашей (`dash_ratio`, `dash_length`), режимы цветов (`.single`, `.per_vertex`, `.gradient`). `GreasedLineMesh`: динамический меш с real-time обновлением точек (`setPoints`), толщины (`setWidth`), цвета (`setColor`) и ориентации на камеру (`update(camera_pos)`). | ✅ |
| Garland-Heckbert QEM Quadric3D | `mesh/simplify.zig`, `mesh.zig`, `root.zig` | Симметричная 4x4 матрица ошибки `Quadric3D` (10 float параметров): построение из уравнений плоскостей `fromPlane` с весом площади треугольников, сложение `add`, вычисление квадратичной ошибки `evaluate(p)`, точное аналитическое решение `solveOptimal(p0, p1)` с проверкой обусловленности/детерминанта матрицы и робастным фоллбэком на граничные и среднюю точку. | ✅ |
| Децимация мешей `simplifyGeometry` | `mesh/simplify.zig`, `mesh.zig`, `scene.zig`, `root.zig` | QEM-децимация геометрии: вычисление топологических ребер, взвешивание плоскостей по площади треугольников, обнаружение границ сетки и наложение штрафных квадрик (`border_penalty: 500.0`) для сохранения контуров и силуэтов (`preserve_border`), проверка переворота нормалей (`prevent_normal_flips`), интерполяция UV, цветов вершин и тангенсов (`preserve_attributes`), пересчет сглаженных нормалей surviving граней. | ✅ |
| `Mesh.toGeometryData` & `simplifyMesh` | `mesh/mesh.zig`, `mesh/simplify.zig`, `scene.zig`, `root.zig` | Извлечение CPU-геометрии из `Mesh` через `toGeometryData` (с поддержкой `pending_vertices`, `morph_base` и `cpu_positions`), генерация упрощенного меша `Scene.simplifyMesh` и `MeshBuilder.simplifyMesh`. | ✅ |
| Автоматическая генерация LOD | `mesh/simplify.zig`, `scene.zig`, `root.zig` | `generateLODLevels`: пакетная генерация уровней детализации по спецификациям `[]const LODLevelSpec` (дистанция, целевое соотношение треугольников, опции упрощения) с автоматической регистрацией в `source_mesh.addLODLevel`. | ✅ |
| Модульные тесты | `mesh/tests.zig` | 9 новых тестов: GreasedLine single path ribbon vertices/indices, multi-path per-vertex widths/colors/dash, closed loop closure, GreasedLineMesh lifecycle/updates, Quadric3D plane accumulation and analytical evaluation, simplifyGeometry box decimation, preserve_border retention on open planes, Scene.simplifyMesh with toGeometryData, Scene.generateLODLevels distance bracket switching. | ✅ |

Проверки: 662/662 unit-тестов, `zig build test` (agate), `zig fmt --check src/` и `zig build` (sandbox) проходят без ошибок.

---

## ✅ Что сделано

### Ядро и платформа

* Нативное приложение на sokol: macOS (Metal), Windows (D3D11), Linux (GL).
* Scene graph: иерархия `Mesh.parent`, TRS-трансформы, ленивый пересчёт world-матриц за кадр (`scene.zig: worldMatrixCached`).
* Математика: `Vec2/3/4`, `Mat4` (SIMD-перемножение), `Quat` (slerp/nlerp), `Color3/4`, `BoundingBox`, `Frustum`, `Ray`.
* Статистика кадра: меши, отсечённые, draw calls, треугольники, переключения пайплайнов, тайминги фаз (`SceneStats`).
* Многопоточность: game/render threads под узким phase mutex (`prepareFrame`), `jobs.Pool`/`TaskRunner`, lock-free `Handoff`/`SpscRing`, affinity-маркер `gpu_thread`, отложенные создание/обновление/уничтожение GPU-ресурсов вне графического потока.
* Встроенный профилировщик (`profiler.zig`): точный замер фаз кадра (Update, Prepare, Shadow, Main, PostFX), автоматическая эвристическая диагностика боттлнеков с подсказками, экспорт в интерактивный HTML (SVG-таймлайн), Markdown и Chrome Trace Event JSON (`chrome://tracing`).
* Мониторинг памяти (`MemorySnapshot`): раздельный учёт памяти геометрии на CPU, общих аллокаций рантайма и видеопамяти VRAM для GPU-буферов и текстур.
* Сериализация состояния сцены v3 (`serialization.zig`): сохранение и загрузка сущностей с постоянными Entity ID, графом иерархии нод и произвольными игровыми свойствами (полная совместимость с версиями v1 и v2).
* Система тегов объектов и булевых смарт-фильтров (`tags.zig`): `TagSet` (множество строковых тегов с регистронезависимым поиском, дедупликацией и разбором списков через разделители), `TagQuery` (парсер и AST-оценщик булевых выражений: `&`/`&&`/`and`, `|`/`||`/`or`, `!`/`not`, круглые скобки, неявный AND), нативная интеграция в `Mesh` (`tags`, `addTag`, `addTags`, `removeTag`, `hasTag`, `matchesTagQuery`) и `Scene` (`getMeshesByTag`, `getMeshesByQuery`, `countMeshesByTag`, `countMeshesByQuery`, `findFirstMeshByTag`, `findFirstMeshByQuery`, `pickWithRayTag`).
* Дозирование загрузок и асинхронный I/O: покадровый лимит загрузки текстур на GPU (`upload_budget_per_frame = 4`), дедупликация файлов в очереди `UploadQueue`, отдельный поток `io_runner` под сохранение и загрузку сцен.
* 1078 unit-тестов в библиотеке, отдельный sandbox с бенчмарками (`zig build test`, `zig build fmt`, флаг `--bench`).

### Рендеринг

* Forward-рендер, шейдеры cross-compile через sokol-shdc (GLSL410/Metal/HLSL5).
* 8 пайплайнов: Standard, PBR, Instanced, Skinned PBR — каждый под u16/u32 индексы + double-sided твины.
* Полностью неблокирующий рендеринг: zero-dereference рендер (`Scene.render` исполняется без захвата блокировок симуляции и не разыменовывает указатели `*Mesh`); все GPU-хэндлы, матрицы и дескрипторы материалов упаковываются в фазе `prepareFrame()` в изолированные структуры `RenderMeshItem`, `RenderInstancedBatch`, `ShadowDrawItem`, `OutlineDrawItem`.
* Сортировка очереди: непрозрачные Standard/PBR группами по текстуре, front-to-back для early-Z; back-to-front для прозрачных мешей.
* OIT-сортировка прозрачных инстансов: поинстансная сортировка back-to-front внутри батча `submitInstancedMesh` перед заливкой в GPU instance-буфер.
* Frustum culling AABB, включая SIMD-батч по 4 инстанса (`Frustum.intersectsAABB4`).
* Occlusion culling: CPU Hierarchical Z-Buffer (Hi-Z), 9-уровневая консервативная пирамида глубин, O(1) AABB-тест, 0 GPU stall/pop-in.
* GPU-инстансинг: динамический instance-буфер с дедупликацией загрузок по хэшу (Standard + Cook-Torrance PBR + CSM + Spot shadows).
* Offscreen-буфер для постобработки с depth-текстурой; UI-оверлей поверх.
* Документированные проходы: shadow → main → skybox → particles → SSAO → post-process → UI (`passes/`).

### Камеры

* `ArcRotateCamera` — орбита мышью, зум колесом, лимиты радиуса/угла.
* `FreeCamera` — движение WASD/стрелками + Space/Ctrl, обзор ЛКМ-драгом, скорость/чувствительность, `update(dt)`.
* `FollowCamera` — следование за мешем или точкой с радиусом, высотой, смещением угла и сглаживанием.
* `Camera` — union-абстракция (arc_rotate/free/follow/target/fly), `Scene.updateCamera(dt)`; проходы и пикинг работают с любым типом.
* Мультикамера / PIP (Picture-in-Picture) с раздельными viewport'ами.
* `CameraRig` — система камера-ригов и пресетов мультикамер (`CameraRigMode`: `single`, `dual_horizontal`, `dual_vertical`, `quad_view`, `setupCadQuadView` с 4 ортогональными/перспективными проекциями, `pip`, `stereoscopic_side_by_side`, `stereoscopic_over_under` с настраиваемым IPD 64мм и convergence `parallel`/`toe_in`, `custom` с локальными смещениями `local_offset` и `look_at_target`, синхронизация с `Scene` через `applyToScene`/`syncToScene`).

### Свет и тени

* `HemisphericLight` (небо + ground color) — всегда одна.
* `DirectionalLight` — `Scene.createDirectionalLight` (солнце: направление, цвет, интенсивность, 4-каскадный CSM) + до 3 shadowless fill через `Scene.addDirectionalLight` (всего до 4, `is_enabled`), hemi остаётся ambient.
* До 4 `PointLight` с range/интенсивностью и до 2 `SpotLight` (inner/outer cone) — per-pixel затухание, выбор значимых источников в камере за 1 проход.
* Clustered point lights до 64 (`Scene.addClusteredPointLight`, cap 64 с `error.TooManyClusteredLights`, values-only, session-local): tile-based forward+ (64×64 px window-grid tiles per view на context-потоке из staged snapshot; PIP независим; лаг 1 кадр), SSBO-блоки 12/13/14 во всех пяти forward-шейдерах (пустой pool — бит-идентичный legacy-путь), без теней в v1, 2D-column over-inclusion задокументирован (near/far игнорируются), без per-tile cap (worst-case ~0.5 МБ при 4K/64 — metered, unthrottled); forward-шейдеры glsl430 ради SSBO (Metal/D3D legs без изменений, Linux-GL требует 4.3+).
* До 2 `RectAreaLight` (`Scene.addAreaLight`, half-extent `right`/`up`, cap 2 с `error.TooManyAreaLights`) — документированная closest-point-on-rect аппроксимация (НЕ LTC: жёстче края у больших/близких rect, без rect-shape specular анизотропии, БЕЗ теней в v1); API-only, session-local.
* Тени: 4-каскадный CSM (атлас 2048², 4 × 1024²), 16-выборок Poisson PCF / переменная полутень PCSS, depth bias + normal bias, мягкость, fade дальнего каскада.
* Перспективные тени SpotLight: depth-атлас 1024×512 (до 2 прожекторов), 4-tap PCF-фильтрация.
* `mesh.cast_shadows` / `mesh.receive_shadows` на каждый меш; скелетные меши тоже отбрасывают тени (skinned shadow-пайплайн).
* Reflection probes (до 4, `Scene.addReflectionProbe`, on-demand `captureReflectionProbe`: 128px RGBA8-куб + 8 мипов, 6 face-проходов; PBR×3 заменяет IBL-источник, Standard ambient — из coarsest mip; выбор ближайшего enabled+captured в радиусе, без блендинга; документированные приближения: exact 2×2 box вместо GGX-префильтрации, лаг 1 кадр, инстансы — legacy).

### Материалы и IBL

* `StandardMaterial`: diffuse color + текстура, поддержка флага `unlit`.
* `PBRMaterial` (metallic-roughness, Cook-Torrance): albedo, normal, metallic-roughness, emissive, occlusion (сила), alpha, цветовые факторы, environment intensity, per-slot UV-трансформы (KHR_texture_transform), выбор каналов AO/roughness/metallic.
* Unlit-режим (`Material.unlit = true`): полный обход расчётов освещения и теней в PBR и Standard шейдерах для стилизованной геометрии, спецэффектов и элементов интерфейса.
* IBL от skybox-кубмапы, exposure, выбор текстуры отражений.
* Дефолтные 1×1 текстуры (white/black/flat normal/cube) — PBR работает без ассетов.

### Текстуры, небо, окружение

* 2D-текстуры: декод PNG/JPEG/… через stb_image, RGBA8, настройки wrap/min/mag, полная CPU-цепочка мипмапов (box-filter), процедурные checkerboard и particle-dot. OpenEXR scanline (HALF/FLOAT; NONE/RLE/ZIPS/ZIP; строгий API `Texture.fromExrFile/fromExrMemory` → RGBA16F, без silent fallback).
* CubeTexture: 6 граней, дефолтная 1×1, процедурный skybox-градиент, развёртка equirectangular-панорамы в куб, загрузка граней из файлов.
* Асинхронный декод картинок glTF на worker-потоках с fallback на синхронный путь и дедупликацией в очереди загрузок.

### Геометрия

* `Mesh` + `InstancedMesh`, дочерние меши, `BoneAttachment` (крепление к кости).
* `MeshBuilder`: Box (с цветами граней), Sphere, Cylinder, Capsule, Ground (subdivisions), Terrain (heightmap), Torus, TorusKnot, Disc, Ribbon (набор путей), Lathe (профиль вращения), Plane, Tube, Extrude, Lines, Polygon (с отверстиями и ear-clipping), TrailMesh.
* CSG (Constructive Solid Geometry): BSP-дерево, булевы операции (Union, Subtract, Intersect).
* LOD: иерархические дистанционные уровни детальности мешей (`addLODLevel`, `getLODForCamera`) с автоматическим переключением.
* Декали: проектор ориентированного куба (OBB) с отсечением по Sutherland-Hodgman, расчётом нормалей и tangents.
* Автоматическая генерация нормалей (`computeMissingNormals`): расчёт взвешенных по площади нормалей граней при их отсутствии в файле модели.
* Поддержка glTF `KHR_mesh_quantization`: декодирование 8/16-битных нормализованных целочисленных вершинных атрибутов.
* Генерация касательных (`computeTangents`), u16/u32 индексы, локальный AABB, хранение CPU-геометрии для физики.
* Вершинные цвета (color0) поддерживаются Standard/PBR шейдерами.

### Анимация

* Скелеты до 64 костей (`MAX_BONES`), GPU-скиннинг через uniform matrix palette.
* glTF-скины и каналы translation/rotation/scale; линейная, step и Hermite cubic-spline интерполяция.
* `AnimationGroup`: play/pause/stop, loop, `playRange`, скорость (в т.ч. отрицательная), вес.
* Блендинг базовых клипов (1, 2 и N клипов), аддитивные слои, fadeTo/fadeIn/fadeOut/crossFadeTo.
* События и таймлайн-маркеры анимаций с колбэками.
* Морф-таргеты glTF (до 8 targets) с CPU/GPU-блендингом и синхронизацией каналов весов.
* Крепления (sockets) мешей к костям, анимированные тени для скелетов.

### Частицы

* `ParticleSystem`: CPU-симуляция, рендер GPU-инстансами.
* Режимы additive / alpha-blend, текстура частицы (есть процедурный dot), спрайт-листы (spritesheets) и угловое вращение.
* Локальное пространство эмиттера, эмиттер-бокс, emit rate, burst, гравитация, время жизни, интерполяция цвета и размера start→end.
* Коллизии CPU-частиц v1: сферы (cap 8, `error.TooManyColliders`) + ground plane, режимы `.kill`/`.bounce` (restitution/friction), детерминизм без общего PRNG (60×1/60 ≡ 30×1/30), `error.CollisionNeedsCpu` для не-CPU систем, default `.none` = бит-идентичная симуляция.
* Stateful GPU-симуляция (`SimulationMode.compute` на том же `ParticleSystem`: CPU spawn/emitter/lifetime-семантика shared verbatim, in-place state в storage-буфере без ping-pong, draw через существующий billboard-пайплайн; гейт `computeAvailable()` + `error.ComputeUnsupported` без тихого fallback; non-goals: sub-emitter deaths, flow maps, сортировка, коллизии, CPU-детерминизм).

### Постобработка

* Tonemapping: ACES или Reinhard, exposure.
* Bloom (bright pass + Karis downsample + tent upsample, мип-пирамида 3–7 мипов), glow layer (threshold-экстракция + separable Gaussian blur + аддитивная композиция, после bloom до grading, `InvalidGlowOptions`-валидация, default off = бит-идентичность), виньетка, saturation/contrast, chromatic aberration, sharpen (unsharp mask), film grain, white balance (temperature/tint).
* Camera Motion Blur: шейдерное размытие движения камеры по матрицам вида-проекции текущего и предыдущего кадров (`prev_view_proj`), настраиваемое число выборок (до 16) и интенсивность.
* DoF (Depth of Field) по глубине (14 golden-angle taps CoC), параметрические цветовые curves (shadows/midtones/highlights).
* Outline-слой (inverse hull, контур объектов с настраиваемой шириной и цветом).
* Highlight layer v1 (per-mesh colored inner glow, паритет Babylon.js `HighlightLayer`): cap 8 подсвеченных мешей (`error.TooManyHighlights`), per-mesh цвет/радиус/интенсивность (`HighlightOptions`), half-res маска-RT → separable Gaussian blur (общий `glow_blur`) → additive-композит после glow-блока до grading; staged `HighlightDrawItem` на prepare (никаких live `Mesh` в `Scene.render`), default off = бит-идентичность; non-goals v1: skinned-меши fail-closed, только template-proxy для инстансов, без depth (свечение просвечивает foreground), alpha-cutout рисует квад.
* FXAA 3.11.
* Дистанционный + высотный fog с подмешиванием цвета солнца (sun scattering).
* SSR: 16 шагов screen-space марша, fresnel, edge fade, настраиваемая интенсивность/толщина/дистанция.
* SSAO: depth-based выборки + bilateral blur, debug-режим, интенсивность/power/radius.
* У каждого эффекта есть вкл/выкл и параметры в `PostProcessOptions` / `SSAOOptions`.

### Физика (Box3D v0.1.0)

* Тела: static/dynamic/kinematic, массы, restitution, friction, linear/angular damping, гравитация, спящий режим (`!is_awake`).
* Коллайдеры: box, sphere, capsule, convex hull, triangle mesh (static), heightfield.
* Compound-коллайдеры: дополнительные box/sphere/capsule/hull child shapes с фильтрами и событиями.
* Суставы: distance, spherical, revolute (+limits/motor), wheel (+spin/steering), prismatic (+limits/motor), motor, weld, parallel.
* Character controller (кинематический mover), rope (цепочка сегментов, резка/ремонт), ragdoll и vehicle-хелперы.
* Сенсоры и события: sensor events, contact events, contact hit events.
* Raycast с фильтрами, `applyImpulse` / `applyTorqueImpulse` / `applyForce`, radial explosion (`applyExplosion`).
* Пространственные запросы: `queryAABB` / `querySphere` / `queryPoint` (+ `*WithFilter`) и `spherecast`.
* Debug-геометрия: `appendDebugLines` / `DebugLine` + 3D-пасс линий физических форм с depth-test.
* Фиксированный шаг 1/60 c аккумулятором (Fix Your Timestep).

### UI

* `UICanvas`: экранный immediate-mode рендер квадов с геометрическим динамическим перевыделением GPU-буферов.
* Примитивы: rect, outline, panel, прогресс-бар, кнопка, бейдж, checkbox, slider, dropdown, скролл (`ScrollState`), поле ввода текста (`TextInputState`, UTF-8 редактирование), divider, arrow, line (`drawLine`), SDF-текст (обычный/жирный/с обводкой), `measureText`, hit-test.
* SDF-шрифт зашит в движок (SDF-атлас), масштабируется без потери чёткости, CLI-флаг `--ui-scale N`.
* TrueType-шрифты (`ttf.zig`: glyf-парсер head/maxp/cmap fmt4+fmt12/hhea/hmtx/loca, простые + композитные глифы ≤8, kern fmt0; scanline coverage-растеризатор, 512px RGBA8 shelf-атлас; явный error set без silent fallback; `UICanvas.setFontTtf/clearFontTtf/hasTtfFont/measureTextCurrent`, битмапный шрифт — дефолт; вне скоупа: хинтинг, лигатуры/шейпинг/RTL, субпиксель, цветные шрифты).
* 3D world-space GUI-панели (`Scene.addUi3dPanel`, до 4, `TooManyUi3dPanels`/`InvalidUi3dPanelSize`; `pickUi3dPanel` лучом из staged view_proj inverse + `injectPointer`/`injectRelease` в canvas input; dirty-driven render-on-demand, max 1 capture/frame, offscreen RT + private VB/IB, unlit double-sided quad после transparent queue; headless fail-closed, retire-safe).
* `Scene.projectPoint` — мировые точки в экранные (используется для 3D-подписей над объектами).

### Аудио

* Процедурный синтез на sokol.audio: thump, noise burst, blip — без внешних аудиоассетов.
* `AudioClip.fromWavMemory` / `fromWavFile` — WAV PCM 8/16/24/32-bit и float32, mono/stereo, потоковый ресемплинг; `AudioEngine.playClip` с loop/rate/позиционированием.
* Воспроизведение и потоковый стриминг аудио с диска и памяти (`AudioStream`, `playSound`, `playSoundFromMemory`, `playSoundOnce`, `playSoundOnceFromMemory`): фоновое инкрементальное декодирование OGG Vorbis (`stb_vorbis`), MP3 (`dr_mp3`) и WAV, SPSC lock-free кольцевой буфер (65536 стерео-фреймов), автоматическое освобождение памяти (`auto_destroy`), 0 динамических аллокаций и 0 I/O в аудиоколлбэке sokol-audio.
* Управление звуковыми потоками и кроссфейдинг: `stopSound`, `pauseSound`, `resumeSound`, `setSoundVolume`, `stopAllSounds`, `crossfadeSound` / `crossfadeSoundFromMemory`, плавный `fadeTo` с авто-остановкой, бесшовный лупинг (gapless loop) и точный сик (`seekToSeconds`/`seekToFrame`).
* Динамический открытый реестр шин (`BusId = enum(u8) { _, pub const invalid }`): пользователь сам объявляет любые шины (Master, SFX, Music, Ambient, UI, Weapons и др.).
* Прямой роутинг по умолчанию: воспроизведение звуков без шины (`bus: ?BusId = null`) направляет поток напрямую на мастер-выход.
* DAG-дерево шин (Parent-Child): каскадное наследование эффективной громкости и mute с защитой от циклов (`getBusEffectiveVolume`), авто-переподключение дочерних шин при удалении родителя.
* Статическая память без аллокаций: потолок 128 шин (`max_bus_capacity`), конфигурируемый активный лимит при старте через `AudioConfig{ .max_buses = 32 }` — 100% real-time safety в потоке аудио.
* 3D Spatial Audio & затухание: модели `linear`, `inverse`, `exponential` с параметрами `min_distance`, `max_distance`, `rolloff`. Динамическое переключение spatial/non-spatial на лету (`setBusSpatial`).
* 3D Эффект Доплера: расчёт сдвига высоты тона по взаимным скоростям слушателя и источников (`setListenerVelocity`, скорость эмиттера, `doppler_factor`).
* DSP-фильтры (Biquad IIR) и реверберация (Freeverb): lowpass, highpass, bandpass, notch, stereo reverb, геймплейные пресеты (`setBusUnderwater`, `setBusMuffled`, `setBusCaveReverb` и др.).
* Акустическая окклюзия геометрией/физикой: `evaluateAudioOcclusion`, multi-tap raycast, сглаживание фильтром, выделенные фильтры окклюзии на шинах.
* Хелперы синтезатора с привязкой к шинам: `playImpactOn(bus, pos, speed)`, `playExplosionOn(bus, pos, size)`, `playBlipOn(bus, freq)`.
* 24 аппаратных голоса с вытеснением + 32 одновременных фоновых стрима (`max_streams = 32`), lock-free SPSC кольца команд, атомики громкости/mute.

### Пикинг и ввод

* `Scene.pick` / `pickWithRay` — CPU-луч по AABB/сферам видимых мешей, точный луч по физическим телам (`PhysicsWorld.raycast`).
* Точный инстанс-пикинг: проверка каждого видимого инстанса `InstancedMesh` в пространстве отрисовки с корректным inverse-transpose нормалей и возвратом индекса инстанса (`picked_instance`).
* `ArcRotateCamera` — ЛКМ-орбита, колесо-зум, лимиты радиуса/угла; `FreeCamera`, `FollowCamera`, `FlyCamera`, `TargetCamera` — см. раздел «Камеры».
* События окна/мыши/клавиатуры через sokol; sandbox демонстрирует комплексное управление с hotkeys и HUD.

---

## 🟡 Что сделано частично

| Возможность Babylon.js | В Agate есть | Чего не хватает |
|---|---|---|
| Камеры (Universal/Free/Follow/Target/Fly/VR, мультикамера, viewports, риги) | ArcRotate + Free + Fly + Follow + Target + union Camera, мультикамера/viewport'ы (PIP), инерция/сглаживание вращения/зума (ArcRotate, Free, Fly), камера-риги (CameraRig) с пресетами (dual, quad, CAD, PIP, stereoscopic 3D VR) | touch/pinch, геймпад |
| Свет (Directional, RectArea, тысячи источников, clustered) | 1 hemi (ambient) + до 4 directional (1 солнце с CSM + до 3 shadowless fill) + 4 point + 2 spot (выбор лучших по камере) + до 2 rect area (closest-point approximation, без теней, API-only) + clustered до 64 point (tile-based forward+, без теней в v1) | Кластерного освещения сверх 64 / с тенями |
| Тени (PCF/PCSS/Blur/Contact hardening для всех источников) | CSM для directional, Poisson PCF + PCSS, перспективные тени SpotLight, тени PointLight (до 2, 2D-атлас, 4-tap PCF) | ESM, каскадных настроек per-light |
| PBR (OpenPBR, clearcoat, sheen, anisotropy, transmission, SSS) | metallic-roughness + IBL, unlit-режим, clearcoat + sheen (scalar/color + текстуры масок/тинта), импорт `KHR_materials_clearcoat/sheen` из glTF, anisotropy (GGX-stretch, 0 = legacy), thin-film transmission (без refraction RT), SSS v1 (wrap-diffuse + back-scatter) | OpenPBR, полный refraction (IOR/thickness), физической SSS/BSSRDF, анизотропных roughness-карт |
| Прозрачность | Все alpha-режимы (opaque/cutout/blend) + double-sided (cull-off пайплайны), единый back-to-front порядок regular+instanced, per-instance сортировка прозрачных инстансов (OIT) | back-face освещение по геометрическим нормалям, пиксельный WBOIT |
| Текстуры (EXR/DDS/KTX/Basis, сжатие, видео) | PNG/JPEG RGBA8 + HDR Radiance RGBA16F, EXR scanline HALF/FLOAT (NONE/RLE/ZIPS/ZIP, strict API `Texture.fromExrFile/fromExrMemory`), equirect→cube, мипмапы, wrap/filter/anisotropy, KTX2 LDR (мипы/cube/sRGB) + BC7-батч моделей (`sandbox/tools/convert_ktx2.sh`), DDS BC1/BC2/BC3/BC7 (мипы) | KTX2-суперкомпрессии (нужен рантайм-транскодер) и прочие блочные форматы (ETC/ASTC), HDR-16F в KTX2, видеотекстуры, render-target/refraction probe текстуры |
| Постобработка (DoF, motion blur, TAA, MSAA, glow/highlight, LUT) | ACES/Reinhard, bloom с мип-пирамидой, glow layer (global v1: threshold-экстракция + separable blur + аддитивная композиция, независим от bloom, default off), highlight layer (per-mesh inner glow: маска-RT + blur + additive, цвет/радиус/интенсивность на меш, cap 8 `error.TooManyHighlights`, default off), DoF, camera motion blur, TAA (Halton-jitter, history ping-pong, 3×3 neighborhood clamp, default off), MSAA depth-prepass (PASS 1.7, single-sample depth-only проход, default off) — постглубина (SSAO/SSR/DoF/Fog/MotionBlur) под MSAA, цветовые curves, LUT-стрип (2D strip + `setColorGradingLut`/`lut_strength`), outline-слой, виньетка, CA, FXAA, fog, SSR, SSAO, sharpen, grain, white balance | TAA под MSAA (v1 non-goal); без гейта depth-эффекты при MSAA подавлены (prepass v1: 1x-глубина, ±1 пиксель на гранях, primary-only) |
| Анимация (retargeting, GPU-морфы) | Скелетная + node-анимации, морф-таргеты, cubic-spline (Hermite), события/колбэки, easing, ретаргетинг скелетов (name/index/bone_map) | GPU-морфов, редактора |
| Частицы (GPU-симуляция, sub-emitters, flow maps, spritesheet) | CPU-симуляция + GPU-рендер, спрайт-листы, локальное пространство, sub-emitters, flow maps, коллизии CPU-частиц v1 (сферы cap 8 + ground plane, kill/bounce, `error.CollisionNeedsCpu`), stateful compute-симуляция | Коллизий с мешами / rigid-body coupling, CCD, нодового редактора |
| Меш-билдеры и геометрия (CSG2, LOD, упрощение, decals, GreasedLine) | 16 примитивов + terrain + LOD + Decals + Polygon + TrailMesh + CSG + GreasedLine + QEM-упрощение мешей (decimation) | CSG2 |
| glTF (Draco/meshopt/KTX2, расширения, экспорт) | GLB/GLTF, EXT_meshopt_compression, KHR_mesh_quantization, автогенерация нормалей, PBR-текстуры (в т.ч. .ktx2), скины, анимации, морф-таргеты, KHR_lights_punctual-свет, камеры, KHR_texture_transform (texCoord0), KHR_materials_clearcoat/sheen, экспорт GLB (бинарный glTF 2.0) | Draco, KTX2-транскодинг (Basis), multi-UV (texCoord>0) |
| Физика (Havok: ragdoll/vehicle/soft body, инспектор) | Box3D + суставы, character, rope, запросы, ragdoll/vehicle-хелперы, debug-линии + PBD cloth v1 (cap 4, session-local) | Импорт коллайдеров из файлов; soft body за пределами PBD cloth v1 |
| UI/GUI (полный набор контролов, layout, 3D GUI, редактор) | Immediate-mode примитивы + SDF-текст + TrueType-шрифты + checkbox/slider/dropdown/скролл/text input + 3D world-space панели (до 4, pick+inject, render-on-demand) | Фокуса/состояния, редактора |
| Аудио (файлы, стриминг, шины, эффекты, doppler) | Процедурный синтез + WAV/OGG/MP3, потоковый стриминг с диска/памяти, SPSC lock-free кольцевые буферы, кроссфейд музыки, 24 голоса, динамический DAG шин, spatial/non-spatial, затухание (linear/inv/exp), Doppler, biquad IIR фильтры, Freeverb реверберация, звуковая окклюзия | Микро-чанковый асинхронный I/O менеджер фонового дискового кэширования для сотен одновременных дорожек |
| Материалы (NodeMaterial, ShaderMaterial, библиотека материалов) | Standard + PBR + ShaderMaterial (engine-hook + внешний shdc-путь: свой `.glsl` собирается build-API движка без правки его исходников) + библиотека материалов (Sky/Gradient/Grid/TriPlanar hook-пресеты + sandbox-витрина S25, скрыта по умолчанию) + NodeMaterial v1 (типизированный граф 10 нод → GLSL-codegen через hook-путь, параметры — runtime-уiformы, golden-тесты) | NodeMaterial v2: PBR-output/вершинные хуки, vec4-порты, больше текстурных слотов, сериализация графа, runtime-компиляция незарегистрированного графа; визуальный редактор графа |
| Инструменты разработчика (Inspector, отладочные оверлеи) | `SceneStats`, встроенный профилировщик фаз кадра (HTML/MD/Chrome Trace) + GPU frame time (Metal) и per-pass тайминги shadow/main/post (GL timer-query, за гейтом), снимки памяти CPU/GPU (MemorySnapshot), debug-режимы SSAO/каскадов, `appendDebugLines` | Интерактивного UI-инспектора сцены (in-game editor), редактирования на лету |

---

## ❌ Чего нет (потенциальный бэклог, не веб-специфика)

**Камеры и ввод**
* Тач-управление, геймпад, виртуальные джойстики; встроенное управление персонажем (кроме physics character controller).

**Свет и тени**
* Тени от point-светов сверх лимита (3+), PCSS/contact hardening для точечных источников, ESM, blur-exponential; тени rect area-светов (rect area без теней в v1).
* Динамический IBL, объёмный свет/атмосфера (rect area до 2 с closest-point approximation, clustered до 64 без теней в v1 и reflection probes до 4 с on-demand capture уже реализованы, см. ✅).

**Материалы и текстуры**
* OpenPBR, полный refraction (IOR/thickness), физическая SSS/BSSRDF, анизотропные roughness-карты (текстуры clearcoat/sheen, импорт `KHR_materials_clearcoat/sheen` из glTF, anisotropy, thin-film transmission и SSS v1 уже сделаны, см. 🟡).

* NodeMaterial v2 (визуального редактора графа, PBR-output/вершинных хуков, сериализации и runtime-компиляции незарегистрированного графа); v1 (граф → GLSL через hook-путь, runtime-параметры) уже сделан, см. 🟡. Библиотека материалов и ShaderMaterial с внешним shdc-путём сделаны, см. 🟡 (engine-hook + внешний `.glsl` через build-API).
* Рантайм KTX2-транскодинг суперкомпрессии (BasisLZ/Zstd; нужен basis_universal), ETC/ASTC, HDR-16F в KTX2, видеотекстуры, render-to-texture, refraction probes, кубмапы-зонды (DDS BC1/BC2/BC3/BC7 и офлайн BC7-батч моделей уже сделаны, см. 🟡; reflection probes уже реализованы, см. ✅).
* Back-face освещение по геометрическим нормалям (per-instance OIT сортировка прозрачных инстансов уже реализована).

**Постобработка и эффекты**
* SSAA (суперсэмплинг), lens flares, snapshot-рендер, SSR/SSAO более высокого качества (MSAA с depth-resolve через single-sample depth-prepass, default off, — волна 39; glow layer — global v1 — highlight layer — per-mesh inner glow v1, cap 8, без depth/скinned/инстанс-подсветки — и TAA через Halton-jitter + history + clamp, LUT-цветокоррекция через 2D-стрип и Camera Motion Blur уже реализованы).

**Геометрия**
* Инстансинг с per-instance материалами (PBR-инстансинг поддержан, per-instance material overrides отсутствуют; GreasedLine и QEM-упрощение мешей с автоматической генерацией LOD уже реализованы).
* Морфы >8 таргетов (GPU-блендинг через delta-текстуру реализован).

**Анимация**
* Редактор анимаций (ретаргетинг скелетов name/index/bone_map уже реализован).

**Частицы**
* Нодовый редактор частиц, коллизии с мешами / rigid-body coupling, CCD (коллизии CPU-частиц со сферами и ground plane — kill/bounce — уже реализованы в v1; stateful GPU-симуляция compute-режимом, CPU on-death sub-emitters и flow maps тоже, см. ✅).

**Физика**
* Импорт коллайдеров из файлов сцен (PBD cloth v1 done: cap 4, без self-collision/tearing/fluids/GPU/box3d-coupling/cloth-cloth/persistence).

**UI/GUI**
* Unicode-шейпинг поверх TTF (сам TTF — парсинг, растеризация, атлас — уже реализован, см. ✅; 3D world-space панели — тоже ✅; LayoutStack, HStack/VStack/Flexbox/CSS Grid, 9-точечные якоря, докинг панелей, анимации переходов и CSS-темы уже реализованы).

**Аудио**
* Высокоуровневый интерактивный секвенсер / FMOD-style нодовый звуковой граф (потоковый стриминг OGG/MP3/WAV, SPSC ring buffer, кроссфейд музыки, динамические шины, DAG-дерево, затухание, Doppler, biquad IIR-фильтры, Freeverb-реверберация и звуковая окклюзия уже реализованы).

**Ассеты и данные**
* Экспорт glTF (GLB бинарный glTF 2.0) и AssetManager с прогрессом и кэшем уже реализованы (см. ✅).
* Draco/meshopt, рантайм KTX2-транскодинг (Basis), 3D Tiles (офлайн BC7-конвертация моделей уже покрыта скриптом).

**Архитектура рендера**
* Frame graph / node render graph, кастомные rendering pipelines.
* GPU compute culling, large world rendering (floating origin) (Software Hi-Z Occlusion Culling реализован в Волне 14; compute-шейдеры есть через `compute.zig` с runtime-гейтом).
* Realtime ray tracing/Gaussian splatting (в Babylon 9 тоже отдельные подсистемы).

**Прочее**
* Crowd simulation (RVO2/ORCA 2D локальное избегание столкновений толпы) уже реализована (см. ✅).
* Сеть/multiplayer, репликация, WebSocket/WebRTC.
* Behaviors/Actions как API-слой, flow graph (Observables/EventBus, теги объектов и smart filters уже реализованы, см. ✅).
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
