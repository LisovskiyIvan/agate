# AUDIT — публичный API, слоп, оптимизация (wave/cleanup)

Дата: 2026-09-12. Базлайн: `zig build test` 451/451 passed.

Метод: ручной проход публичных поверхностей (root.zig, scene.zig, mesh.zig,
physics.zig, material.zig, texture.zig, ui.zig, audio.zig, particles.zig,
shader_material.zig, serialization.zig, compute.zig, camera.zig, lights.zig,
loader/*) + скрипт-скан всех module-level `pub` деклараций с проверкой
упоминаний по всему workspace: `src/`, `examples/`, `tools/` и `../sandbox`
(только grep, read-only). «Мёртвый» ниже = ни одного упоминания имени вне
файла-объявления (включая тесты и sandbox).

## 1. Конвенции именования — выводы

Глаголы создания. Разнобой `create*/init*/make*/build*/new*` оказался
**слоёным, а не хаотичным** — каждый суффикс живёт в своём слое:

| Слой | Конвенция | Примеры |
|---|---|---|
| Значения math | `Type.new(...)` | `Vec3.new`, `Color4.new` |
| Конструктор значения структуры | `init(name, options)` | `StandardMaterial.init`, камеры |
| Сущности сцены/GPU-ресурсы | `create*` | `MeshBuilder.createBox`, `scene.createParticleSystem` |
| Чистая сборка данных (без GPU) | `build*Data` | `buildSphereData -> GeometryData` |
| Обёртки над sokol GPU-объектами | `make*` (зеркалит `sg.makePipeline`) | `compute.makePipeline` |

Суффиксы опций. Доминанта — `*Options` (~60 типов). Отклонения:

| Тип | Решение |
|---|---|
| `PostProcessConfig`, `SSAOConfig`, `SkyboxConfig`, `TransitionConfig` | **переименовать в `*Options`**, оставить тонкий deprecated-алиас |
| `audio.PlayParams` | **переименовать в `PlayOptions`** (+ deprecated-алиас); `ClipPlayOptions` уже в конвенции |
| `particles.ComputeFrameParams` | оставить `*Params`: это вычисляемый пак данных кадра (метафора GPU-юниформов), не «опции пользователя» |
| `shader_material.RuntimeDesc` | оставить `*Desc`: дескриптор регистрации GPU-источников (метафора `sg.*Desc`), поля обязательные |
| `ui.LayoutGridSpec` vs `LayoutGridOptions` | оставить оба, задокументировать: `Options` = входные ручки `beginGrid`, `Spec` = рассчитанная геометрия сетки (возвращается `LayoutStack.grid()`, имеет методы) |

Варианты функций «просто + с опциями»: `appendGlb/appendGlbOptions` и
`createRigidBody/createRigidBodyWith` — два стиля для одной задачи. Оставлено
как есть (обратная совместимость, оба удобны); для нового API правило: одна
функция с options-параметром, дефолты в полях структуры.

## 2. Таблица находок

| # | Находка | Файл | Решение | Тип |
|---|---|---|---|---|
| 1 | `Scene.computeCascades` — мёртвый 转发 после декомпозиции (0 вызовов; подсистемы зовут `scene_cascades.computeCascades`) | scene.zig | удалить | remove |
| 2 | `Scene.materialIsTransparent` — мёртвый шим (0 внешних вызовов) | scene.zig | удалить | remove |
| 3 | `Scene.frameUniforms` — мёртвый шим (0 вызовов) | scene.zig | удалить | remove |
| 4 | `Scene.getPhysicsWorld` — мёртвый (0 вызовов; `enablePhysics` уже возвращает `*PhysicsWorld`) | scene.zig | удалить | remove |
| 5 | `Scene.FrameContext` / `Scene.FrameUniforms` — алиасы «для тестов и тулинга», 0 использований | scene.zig | удалить | remove |
| 6 | 20 дубль-реэкспортов типов, доступных из root (`ParticleSystem`, `PhysicsWorld`, `UICanvas`, `NavMesh`, …): путь `agate.scene.X` никем не используется (0 в src/examples/tools/sandbox) | scene.zig | удалить | remove |
| 7 | Реэкспорты внутренностей билдеров: `TrigEntry`, `trigEntry`, `buildTrigTable`, `storeQuadFlipped`, `buildBoxData`, `buildGroundData`, `buildTerrainData`, `buildSphereData`, `buildCylinderData`, `buildCapsuleData`, `orthogonal_dot_threshold` — 0 упоминаний фасадного пути | mesh.zig | удалить | remove |
| 8 | ~24 среднеслойных реэкспорта (`BoxOptions`…`CSGNode`, `buildPolygonData`, `buildDecalData`) в `mesh/builder.zig` — 0 использований (mesh.zig реэкспортирует те же типы напрямую из исходных модулей) | mesh/builder.zig | удалить | remove |
| 9 | 6 модульных дублей статических хелперов `UICanvas` (`dropdownHit`, `scrollClamp`, …) — задокументированы как алиасы, 0 вызовов + фасадный `animKeyHash` (0 внешних) | ui.zig | удалить | remove |
| 10 | `getSkinnedWorldPosition` — pub, используется только внутри файла | mesh/decal.zig | сделать непубличным | fix |
| 11 | `destroyDeltaResources` — pub, используется только внутри файла (Mesh.deinit держит свою копию из-за цикла импорта — комментарий сохранён) | mesh/morph_gpu.zig | сделать непубличным | fix |
| 12 | `PostProcessConfig` -> `PostProcessOptions` (+deprecated alias) | postprocess.zig, root.zig | rename | rename |
| 13 | `SSAOConfig` -> `SSAOOptions` (+deprecated alias) | ssao.zig, root.zig | rename | rename |
| 14 | `SkyboxConfig` -> `SkyboxOptions` (+deprecated alias) | texture.zig, root.zig | rename | rename |
| 15 | `TransitionConfig` -> `TransitionOptions` (+deprecated alias) | ui.zig (ui/types.zig), root.zig | rename | rename |
| 16 | `PlayParams` -> `PlayOptions` (+deprecated alias) | audio.zig, root.zig | rename | rename |
| 17 | `pub const appendGltf = appendGlb` — алиас без вызовов (sandbox зовёт `appendGlb`); загрузчик и так ест .gltf и .glb | loader/scene_loader.zig | удалить алиас, задокументировать форматы | remove |
| 18 | `FrameContext` (~830 байт) копируется по значению на каждый draw-вызов (`drawRegularItem`/`drawInstancedMesh`/`drawShaderMaterialItem`/`frameUniformsFor`) | scene/draw.zig, scene.zig | передавать `*const FrameContext` | fix (perf) |
| 19 | `Scene.resizeOffscreen` — 0 вызовов, но это единственный путь, ресайзящий SSAO-таргет вместе с остальными (`beginMainPass` ресайзит только postprocess+bloom+outline) | scene.zig | оставить, задокументировать как хук ресайза окна | doc |
| 20 | Отсутствуют док-комментарии: конвенции в root.zig, структуры опций postprocess/ssao/skybox/transition, `PlayOptions`, суффиксы `*Params`/`*Spec`/`*Desc`, ряд pub-функций particles/texture/audio | разные | добавить доки | doc |

## 3. Проверено и НЕ тронуто (намеренные конструкции, не слоп)

- `physics.zig`-фасад с реэкспортами — документирован, путь `agate.X` живой.
- `physics/joints.zig` свободные функции + методы `PhysicsWorld` — forwarding
  ради разрыва цикла импортов (тип мира живёт в нейтральном `world.zig`).
- Чистые Zig-зеркала GPU-математики (`postprocess.zig`: bloom/LUT/DOF,
  `scene/shadow_pcss.zig`, `mesh/morph_gpu.zig`, `particles.zig` spritesheet/
  analytic drag) — pub ради ин-файл тестов, осознанный паттерн.
- Легаси-фоллбэки: instant-swap света при `hysteresis_enabled = false`,
  CPU-режимы частиц (`.cpu`) и морфов (`.cpu`), legacy single-shader bloom —
  рабочие ветки, не мёртвый код.
- `MeshBuilder` как статический неймспейс фабрик — метафора Babylon.js.
- Широкие реэкспорты root.zig (включая пока не используемые sandbox:
  `serialization`, `visibility`, `audio`, `ai`, экспортёры) — это и есть
  публичная поверхность библиотеки.
- `Scene.post_process` / `Scene.ssao` плоские поля — sandbox читает/пишет
  напрямую; комментарий на месте.
- Плоские реэкспорты `ui.zig` (`UIStyle*`, `Css*`) — «один import path»
  задокументировано, используются.
- `serialization.zig` Entry-типы — схема формата файла, часть API.

## 4. ЛОМАЮЩИЕ изменения для sandbox

Все удаления ниже проверены grep'ом по `sandbox/src` (0 вхождений) — текущий
код sandbox не ломается ни одним из них. Список — для полноты картины и на
случай внешних потребителей.

Удалённые имена (были доступны, никем не использовались):

- `agate.scene.{PostProcessConfig, TonemappingType, SSAOConfig, ParticleSystem,
  ParticleBlendMode, Particle, PhysicsWorld, RigidBody, PickingInfo,
  ColliderType, UICanvas, UIVertex, DecalManager, TrailMesh, TrailOptions,
  DecalProjector, CSG, NavMesh, NavNode, NavAgent}` — брать из `agate.*`
- `agate.Scene.{computeCascades, materialIsTransparent, frameUniforms,
  getPhysicsWorld, FrameContext, FrameUniforms}` — мёртвые шимы/алиасы
  удалены; `resizeOffscreen` оставлен (единственный путь ресайза SSAO),
  задокументирован
- `agate.mesh.{TrigEntry, trigEntry, buildTrigTable, storeQuadFlipped,
  buildBoxData, buildGroundData, buildTerrainData, buildSphereData,
  buildCylinderData, buildCapsuleData, orthogonal_dot_threshold}` —
  внутренности `mesh/builders.zig` / `mesh/tangents.zig`
- все реэкспорты `agate.mesh.builder.{…}` (типы доступны как `agate.mesh.X`)
- `agate.ui.{dropdownItemHeight, dropdownItemRect, dropdownHit, scrollClamp,
  scrollOffsetForItem, scrollbarThumbRect, animKeyHash}` — методы
  `UICanvas.*` / `ui_types` остаются
- `agate.SceneLoader.appendGltf` — использовать `appendGlb` (ест оба формата)
- не-pub более: `mesh.decal.getSkinnedWorldPosition`,
  `mesh.morph_gpu.destroyDeltaResources`

Переименования (старое имя оставлено как deprecated-алиас, код sandbox
совместим; новые имена — канонические):

- `PostProcessConfig` -> `PostProcessOptions`
- `SSAOConfig` -> `SSAOOptions`
- `SkyboxConfig` -> `SkyboxOptions`
- `TransitionConfig` -> `TransitionOptions`
- `audio.PlayParams` -> `audio.PlayOptions` (`agate.PlayOptions`)

Смена сигнатуры (внутренний путь рендера, наружу не видна):
`scene/draw.zig` `drawRegularItem`/`drawInstancedMesh` принимают
`*const FrameContext` вместо копии.

## 5. Проверки (финал)

- `zig build test --summary all` -> 26/26 steps, **451/451 passed**, exit 0
- `zig build` -> 0 (демо-бинарь собирается)
- `zig build fmt` -> 0
- runtime-смоук `agate --frames N` в этой среде невыполним (нет оконной
  сессии); прогнать `sandbox --bench` после мержа.

## 6. Коммиты

| Хэш | Содержимое |
|---|---|
| `5d84046` | docs(audit): этот отчёт |
| `ee1f20a` | refactor: удаление мёртвой поверхности API + доки (−118/+82 строк, 14 файлов) |
| `cbd04ab` | refactor(api): *Options-переименования с deprecated-алиасами + `*const FrameContext` (18 файлов) |

## 7. Итог

- находок всего: 20 (remove: 9 групп, rename: 5, fix: 3, doc: 3)
- ломающих для текущего sandbox: 0 — все удалённые имена имеют 0 вхождений в
  sandbox/src, переименования сохранены deprecated-алиасами (и в корне, и в
  определяющих модулях)
- оптимизация: `*const FrameContext` вместо копии ~830 байт на каждый draw
  (drawRegularItem / drawInstancedMesh / drawShaderMaterialItem /
  buildFrameUniforms); измерить `sandbox --bench` после мержа
