# Сцена и контент

> Путь: src/agate/scene.zig, scene/lifecycle.zig, content.zig, registry.zig, stats.zig, sim_api.zig, query_api.zig, lights_api.zig, profile_api.zig, snapshot.zig, cameras.zig, attachments.zig · Импорт: agate.Scene, agate.SceneStats, agate.RenderMeshItem и др. (root.zig) · Потоки: game (update/build) / context (prepare/finish/render) / io_runner (save/load)

## Что это

`Scene` — оркестратор движка: плоские реестры контента (меши, материалы, анимации), состояние камер, кросскаттинг-конфиг/статистика, все render-подсистемы и frame-handoff mailboxes. `scene.zig` — фасад, реэкспортирующий публичное API без изменений; тела живут в листьях `scene/` (free-функции + тонкие форвардеры, т.к. в Zig 0.16 нет `usingnamespace`). Листья никогда не импортируют фасад назад — методы берут сцену как `anytype`.

Инварианты: меши/материалы — Scene-owned (создание аппендит в реестр, уничтожение — unlink + free через matching destroy); GPU-буферы мешей умирают только на контекстном потоке (иначе unlink + epoch-retire); статистика — context-owned (update-сторона пишет только `pending_*` + mailboxes, никогда `stats` напрямую); `sg.*` — только контекст.

## Быстрый старт

```zig
const agate = @import("agate");

// После markContextThread + sokol.gfx.setup, на помеченном контекстном потоке.
var scene: agate.Scene = undefined;
scene.initInto(allocator); // GPU-владелец помечен ДО сцены; headless без маркера — только CPU-cleanup
defer scene.deinit();

// Материал + меш (Scene-owned, реестр).
const mat = try scene.createStandardMaterial("wall");
const box = try agate.MeshBuilder.createBox(&scene, "box", .{});
box.material = .{ .standard = mat };

// Камера.
try scene.addCamera(.{ .name = "main", .camera = .{ .arc_rotate = cam } });
scene.switchCameraByName("main");

// Кадр, game-side (producer build BEFORE begin):
scene.updateCamera(dt);
scene.updateLights(dt);
scene.updatePhysics(dt);
scene.updateAnimations(dt);
try scene.updateParticles(dt);
scene.updateDecals(dt);
scene.publishFrameSnapshot(aspect, w, h);
_ = scene.buildPreparedFrame(); // tryClaimBuild → build → stageUi → publish, полный freeze

// Кадр, context-side (только свежий полный билд; render сам не готовит):
if (scene.beginStagedPrepare()) |claim| {
    scene.finishStagedPrepare(claim);
    scene.render();
} else scene.renderReuse(); // только при валидном front, иначе skip
```

Аллокатор обязан быть thread-safe при воркерах (GPA с `.thread_safe = true`): prepare/render ленивые кэши и jobs-воркеры аллоцируют из него. Адрес `Scene` держать стабильным, дважды не инициализировать (`initInto` или `init`).

## API

### Создание/удаление контента — `scene/registry.zig`, `scene/content.zig`

```zig
pub fn createStandardMaterial(self: anytype, name: []const u8) !*StandardMaterial;
pub fn createPBRMaterial(self: anytype, name: []const u8) !*PBRMaterial;
pub fn createShaderMaterial(self: anytype, name: []const u8, shader_name: []const u8) ?*ShaderMaterial;
pub fn destroyPBRMaterial(self: anytype, mat: *PBRMaterial) void;
pub fn removeMesh(self: anytype, mesh: *Mesh) bool;
pub fn destroyMesh(self: anytype, mesh: *Mesh) void;
pub fn renameMesh(self: anytype, mesh: *Mesh, new_name: []const u8) !void;
pub fn destroyTrailMesh(self: anytype, trail: *TrailMesh) void;
// Поиск:
pub fn getMeshByName(self: anytype, name: []const u8) ?*Mesh;
pub fn getMeshesByTag(self: anytype, allocator: std.mem.Allocator, tag_str: []const u8) !std.ArrayListUnmanaged(*Mesh);
pub fn getMeshesByQuery(self: anytype, allocator: std.mem.Allocator, query_str: []const u8) !std.ArrayListUnmanaged(*Mesh);
pub fn countMeshesByTag(self: anytype, tag_str: []const u8) usize;
pub fn countMeshesByQuery(self: anytype, query_str: []const u8) usize;
pub fn findFirstMeshByTag(self: anytype, tag_str: []const u8) ?*Mesh;
pub fn findFirstMeshByQuery(self: anytype, query_str: []const u8) ?*Mesh;
```

`createShaderMaterial` возвращает `null`, когда shader-имя не зарегистрировано (build.zig `user_shader_materials` или `registerRuntime`). `destroyPBRMaterial` безопасен под update||render без GPU-retire: prepared-записи несут GPU-handle значения, не CPU-refs, материалы GPU-объектов не владеют.

`destroyMesh` сначала нейтрализует все кросс-ссылки (outline, highlights, soft bodies, physics bodies, parent/child + bone attachments, LOD-полосы с сохранением порядка, morph-bindings с tombstone чтобы `NodeChannel.target` индексы остались валидны, decal-manager инстансы, trail follow-targets), затем: на контекстном потоке — inline `deinit` + free; вне его — unlink + `gpu_retire.retireMesh` до flush на render-start. Имена мешей — `[]const u8` с флагом `owns_name` (convention, не enforcement); мутировать только через `renameMesh` (копия до free — алиасинг `renameMesh(m, m.name)` безопасен; OOM атомарен; после успеха `owns_name` всегда true).

`content.zig` — teardown-хелперы реестров (вызываются из `deinit`): `deinitMeshes`, `deinitMaterials`, `deinitShaderMaterials`, `deinitPbrMaterials` (дедуп texture views по id), `deinitAnimations`. Отдельного `destroyParticleSystem` осознанно нет: `ParticleSystem.deinit` делает `sg.destroy*` inline (нелегально вне контекста), у retire-очереди нет particle-kind, плюс надо чистить sub-emitter backrefs и borrowed handle ids в prepared/build кадрах. Системы создавать редко и переиспользовать.

### Жизненный цикл — `scene/lifecycle.zig`

```zig
pub fn resizeOffscreen(self: anytype, width: i32, height: i32) void; // CONTEXT ONLY, asserted
pub fn setPostProcess(self: anytype, config: PostProcessOptions) void;
pub fn setSSAO(self: anytype, config: SSAOOptions) void;
pub fn saveStateFileAsync(self: anytype, path: []const u8) !*serialization.AsyncSaveTask;
pub fn loadStateFileAsync(self: anytype, path: []const u8) !*serialization.AsyncLoadTask;
pub fn setSkybox(self: anytype, cube: CubeTexture) void;
pub fn createDefaultSkybox(self: anytype, config: SkyboxOptions) !void;
pub fn ensureForwardMsaa(self: anytype, samples: i32) *ForwardPipelines; // ленивый twin
pub fn deinit(self: anytype) void; // context-only; дренирует всё включая незавершённые эпохи
```

Порядок `deinit`: profiler → uploads/io_runner (join декодов и файловых задач до teardown) → камеры → `viewport_clear` → decals → `gpu_retire` (всё, включая незавершённые эпохи) → physics (до мешей: bodies держат raw mesh-указатели) → softbodies → меши/материалы → lights → draws → trails/greased/nav → дефолтные текстуры → shadows/sky/probes/clustered/gui3d → forward(+msaa) → анимации → outline/postfx → particles → ui_canvas/ui_frame. Ошибки: async save/load без `io_runner` → `error.NoTaskRunner`; load-поллинг — `task.isDone()`, restore на game-потоке.

### Камеры — `scene/cameras.zig`

```zig
pub const CameraEntry = struct { ... }; // запись списка камер
pub fn addCamera(self: anytype, entry: CameraEntry) !usize;
pub fn removeCamera(self: anytype, index: usize) void;
pub fn getCamera(self: anytype, index: usize) ?*CameraEntry;
pub fn getCameraByName(self: anytype, name: []const u8) ?*CameraEntry;
pub fn switchCamera(self: anytype, index: usize) void;
pub fn switchCameraByName(self: anytype, name: []const u8) bool;
pub fn nextCamera(self: anytype) void;
pub fn prevCamera(self: anytype) void;
pub fn getActiveCameraIndex(self: anytype) ?usize;
pub fn getActiveCameraName(self: anytype) ?[]const u8;
pub fn getCameraCount(self: anytype) usize;
pub fn setActiveCamera(self: anytype, cam: ?Camera, owned_name: ?[]const u8) void;
pub fn updateCamera(self: anytype, dt: f32) void;
```

Детали типов камер (`FreeCamera`, `ArcRotateCamera`, `FollowCamera` и др.) — см. `./cameras.md`. Здесь только реестр/переключение/апдейт.

### Свет — `scene/lights_api.zig` (фасад; математика — `./lights.md`)

```zig
pub fn createHemisphericLight(self: anytype, name: []const u8, options: HemisphericLightOptions) HemisphericLight;
pub fn createPointLight(self: anytype, name: []const u8, options: PointLightOptions) !*PointLight;
pub fn createSpotLight(self: anytype, name: []const u8, options: SpotLightOptions) !*SpotLight;
pub fn createDirectionalLight(self: anytype, name: []const u8, options: DirectionalLightOptions) !*DirectionalLight;
pub fn addDirectionalLight(self: anytype, name: []const u8, options: DirectionalLightOptions) !*DirectionalLight;
pub fn setSunAngles(self: anytype, azimuth_rad: f32, elevation_rad: f32) void;
pub fn setSunColorTemperature(self: anytype, kelvin: f32) void;
pub fn addAreaLight(self: anytype, name: []const u8, options: AreaLightOptions) !*AreaLight;
pub fn removeAreaLight(self: anytype, index: usize) void;
pub fn getAreaLight(self: anytype, index: usize) ?*AreaLight;
pub fn areaLightCount(self: anytype) usize;
pub fn addClusteredPointLight(self: anytype, position: Vec3, options: ClusteredPointLightOptions) error{TooManyClusteredLights}!usize;
pub fn removeClusteredPointLight(self: anytype, index: usize) void;
pub fn getClusteredPointLight(self: anytype, index: usize) ?*ClusteredPointLight;
pub fn clusteredPointLightCount(self: anytype) usize;
pub fn updateLights(self: anytype, dt: f32) void; // update-фаза: packing light_pack
```

### Аттачменты — `scene/attachments.zig`

```zig
// Reflection probes:
pub fn addReflectionProbe(self: anytype, position: Vec3, options: ReflectionProbeOptions) error{TooManyReflectionProbes}!usize;
pub fn removeReflectionProbe(self: anytype, index: usize) void;
pub fn getReflectionProbe(self: anytype, index: usize) ?*ReflectionProbe;
pub fn reflectionProbeCount(self: anytype) usize;
pub fn captureReflectionProbe(self: anytype, index: usize) void;
pub fn captureDirtyReflectionProbes(self: anytype) void;
pub fn probeDirtyCount(self: anytype) usize;
// 3D-GUI панели:
pub fn addUi3dPanel(self: anytype, ...) usize; // см. Ui3dPanelOptions
pub fn removeUi3dPanel(self: anytype, index: usize) void;
pub fn getUi3dPanel(self: anytype, index: usize) ?*Ui3dPanel;
pub fn getUi3dPanelByName(self: anytype, name: []const u8) ?*Ui3dPanel;
pub fn ui3dPanelCount(self: anytype) usize;
pub fn markUi3dPanelDirty(self: anytype, index: usize) void;
pub fn markAllUi3dPanelsDirty(self: anytype) void;
pub fn ui3dDirtyCount(self: anytype) usize;
pub fn pickUi3dPanel(self: anytype, mouse_x: f32, mouse_y: f32) ?Ui3dPickHit;
// Highlights (per-mesh):
pub fn addHighlightMesh(self: anytype, mesh: *Mesh, options: HighlightOptions) error{TooManyHighlights, InvalidHighlightOptions}!usize;
pub fn removeHighlightMesh(self: anytype, index: usize) void;
pub fn clearHighlights(self: anytype) void;
pub fn getHighlightMesh(self: anytype, index: usize) ?*HighlightEntry;
pub fn highlightCount(self: anytype) usize;
// PBD cloth:
pub fn addSoftBodyCloth(self: anytype, name: []const u8, options: ClothOptions) SoftBodyError!*SoftBody;
pub fn removeSoftBodyCloth(self: anytype, index: usize) SoftBodyError!void;
pub fn getSoftBody(self: anytype, index: usize) ?*SoftBody;
pub fn getSoftBodyByName(self: anytype, name: []const u8) ?*SoftBody;
pub fn softBodyCount(self: anytype) usize;
pub fn updateSoftBodies(self: anytype, dt: f32) void;
```

Все — bounded optional attachments; пустые по умолчанию меняют ноль путей отрисовки.

### Симуляция/контент-билдеры — `scene/sim_api.zig`

```zig
pub fn getOrCreateDecalManager(self: anytype, max_decals: usize) *DecalManager;
pub fn updateDecals(self: anytype, dt: f32) void;
pub fn createParticleSystem(self: anytype, name: []const u8, capacity: usize) !*ParticleSystem;
pub fn updateParticles(self: anytype, dt: f32) particles.UpdateError!void;
pub fn createTrailMesh(self: anytype, name: []const u8, options: TrailOptions) !*TrailMesh;
pub fn updateTrails(self: anytype, dt: f32) void; // явный вызов, реальные секунды (не norm-dt)
pub fn createCSGMesh(self: anytype, name: []const u8, csg_solid: *const CSG) !*Mesh;
pub fn createGreasedLine(self: anytype, name: []const u8, options: GreasedLineOptions) !*Mesh;
pub fn createGreasedLineMesh(self: anytype, name: []const u8, options: GreasedLineOptions) !*GreasedLineMesh;
pub fn simplifyMesh(self: anytype, name: []const u8, source_mesh: *Mesh, options: SimplifyOptions) !*Mesh;
pub fn generateLODLevels(self: anytype, source_mesh: *Mesh, specs: []const LODLevelSpec) !void;
pub fn createNavMeshFromTriangles(self: anytype, ...) !...;
pub fn createNavMeshGrid(self: anytype, ...) !...;
pub fn createNavAgent(self: anytype, nav_mesh: *const NavMesh, start_pos: Vec3) !*NavAgent;
pub fn updateNavAgents(self: anytype, dt: f32) void; // явный вызов, реальные секунды
pub fn updateAnimations(self: anytype, dt: f32) void;
pub fn enablePhysics(self: anytype, gravity: ?Vec3) *PhysicsWorld;
pub fn getRigidBody(self: anytype, mesh: *const Mesh) ?*RigidBody;
pub fn createRigidBody(self: anytype, mesh: *Mesh, collider: ColliderType, mass: f32) !*RigidBody;
pub fn createRigidBodyWith(self: anytype, mesh: *Mesh, collider: ColliderType, mass: f32, options: BodyOptions) !*RigidBody;
pub fn updatePhysics(self: anytype, dt: f32) void;
```

Канонический порядок update (`scene/frame_api.zig update`): camera → lights → physics → animations → soft bodies → particles → decals, затем `publishFrameSnapshot`. `updateTrails`/`updateNavAgents` в `update` осознанно НЕ входят (им нужен real-seconds dt, а не 60fps-нормализованный) — приложения вызывают их явно под тем же update-vs-prepare исключением.

### Pick/query — `scene/query_api.zig`, `scene/picking.zig`

```zig
pub fn createPickingRay(self: anytype, screen_x: f32, screen_y: f32) Ray;
pub fn pickWithRay(self: anytype, r: Ray) PickingInfo;
pub fn pickWithRayTag(self: anytype, r: Ray, query_str: []const u8) PickingInfo;
pub fn pick(self: anytype, screen_x: f32, screen_y: f32) PickingInfo;
pub fn createUI(self: anytype) !*UICanvas;
pub fn getUI(self: anytype) ?*UICanvas;
pub fn projectPoint(self: anytype, world_pos: Vec3) ?math.Vec2;
pub fn handleEvent(self: anytype, ev: [*c]const sapp.Event) void;
```

### Снапшот и статистика — `scene/snapshot.zig`, `scene/stats.zig`

```zig
pub fn packFrameSnapshot(scene: anytype, aspect: f32, cur_w: i32, cur_h: i32) SceneFrameSnapshot;
pub fn publishFrameSnapshot(scene: anytype, aspect: f32, cur_w: i32, cur_h: i32) void;
// SceneStats — см. stats.zig: total/rendered/culled/occluded_meshes, draw_calls (+shadow/main/post),
// triangles, pipeline_switches, uploaded_textures_frame, uploaded_bytes_frame (только текстуры
// UploadQueue, бюджет 8 MiB), updated_bytes_frame (uncounted, через gpu_upload_meter),
// update_ms/physics_ms/prepare_ms/shadow_ms/main_ms/post_ms,
// gpu_frame_ms/gpu_shadow_ms/gpu_main_ms/gpu_post_ms.
pub fn mergeFrom(self: *SceneStats, other: *const SceneStats) void; // только счётчики; тайминги/аплоады не мержит
```

`publishFrameSnapshot` — хвост каждого update (в т.ч. внутри `Scene.update`).

### Профайлер-фасад — `scene/profile_api.zig`

```zig
pub fn startProfiling(self: anytype) void;
pub fn stopProfiling(self: anytype) void;
pub fn resetProfiling(self: anytype) void;
pub fn isProfiling(self: anytype) bool;
pub fn captureMemorySnapshot(self: anytype) !*const MemorySnapshot;
pub fn saveProfileReportHtml(self: anytype, path: []const u8) !void;
pub fn saveProfileReportMd(self: anytype, path: []const u8) !void;
pub fn saveProfileTraceJson(self: anytype, path: []const u8) !void;
pub fn saveProfileReports(self: anytype, base_path: []const u8) !void;
pub fn saveProfileReportsAsync(self: anytype, ...) !...;
```

Тонкие форвардеры в `profiler`; control/report — под ограниченным мьютексом только при pending request, encode + file IO — разблокированно на io_runner.

## Потоки и владение

- Game-side (`update`, симуляция, билдеры): живые регистры/канвас/трейлы/nav + mailboxes + `pending_update_ms`/`pending_physics_ms` (атомарные staged). `recordUpdateTime`/`recordPhysicsTime` — единственный легальный путь таймингов с игры; прямая запись `scene.stats.update_ms` запрещена (render читает конкурентно).
- Context-side (`beginStagedPrepare`+`finishStagedPrepare` (one-shot, живой GPU-владелец), `flushPendingGpuUploads`, `render`/`renderReuse`): потребляет слоты + context-owned кэши, пишет `stats`. `flushPendingGpuUploads` + `resizeOffscreen` + `deinit` — только контекст (asserted). `render` никогда не готовит свежий кадр сам.
- `update` может перекрывать `render`; `prepare` и `render` строго последовательны на контекстном потоке; update-vs-prepare исключены фазовым мьютексом только в диагностике (`producer_exclusion`), данные и алгоритм те же.
- GPU-владение: создание вне контекста — CPU-only + deferred (`pending_vertices`, `gpu_pending`), добилдится во flush; уничтожение вне контекста — unlink + retire с epoch-семантикой, добивка на render-start после завершения эпохи; `deinit` дренирует всё включая незавершённые эпохи.
- Профайлер целиком render-owned: `recordFrame` только в конце `render`; воркеры/игровой поток его не трогают.
- Сериализация: `saveStateFileAsync`/`loadStateFileAsync` — захват снапшота + file IO на `io_runner`, game/render потоки на диске не блокируются.

## Ошибки и краевые случаи

- `createShaderMaterial` → `null` при неизвестном shader-имени; `addClusteredPointLight` → `error.TooManyClusteredLights`; `addReflectionProbe` → `error.TooManyReflectionProbes`; `addHighlightMesh` → `error{TooManyHighlights, InvalidHighlightOptions}`; билдеры/реестры → `error.OutOfMemory` (`renameMesh` при OOM атомарен — старое имя цело).
- `updateParticles` → `particles.UpdateError`; compute-путь — `ComputeModeError`; коллизии — `CollisionError` (см. `./particles.md`).
- `destroyMesh` на mesh не из этой сцены — membership не перепроверяется (контракт вызывающего); двойной destroy запрещён (второй — unlink-miss + use-after-free).
- Registry add/remove поперёк in-flight latch — нарушение контракта приложения (commit guards держат когерентность, но так делать нельзя).
- `TrailMesh.update` после destroy цели — follow-target зануляется в `destroyMesh`, чтение retired storage исключено.
- `saveStateFileAsync`/`loadStateFileAsync` без `io_runner` → `error.NoTaskRunner`.
- Пустой кадр без камеры: render early-out после завершения эпохи (headless-тесты на этом строятся).

## Производительность

- Реестры плоские (`ArrayListUnmanaged`), поиск по имени/тегу — линейный скан; для горячих путей держать указатели, не дёргать `getMeshByName` покадрово. Tag-query парсится на каждый вызов `getMeshesByQuery` — переиспользовать распарсенный `TagQuery` вне кадра.
- `SceneStats` — обычные (не атомарные) поля: синхронизация фазовая, не поточечная; `mergeFrom` складывает только счётчики (`occluders_count`/`occluder_triangles` — присваиванием, mirror last-view-wins).
- LOD-полосы distance-sorted; удаление LOD-уровня — order-preserving `orderedRemove` (O(n) в полосе, не в сцене).
- `deinitPbrMaterials` дедупит texture views по id — shared views уничтожаются ровно один раз.

## Смотрите также

- `./runtime.md` — жизненный цикл кадра и потоки
- `./frame-pipeline.md` — build/prepare/render стадии детально
- `./render-pipeline.md` — очереди, вьюхи, проходы отрисовки
- `./mesh.md` — `Mesh`, `MeshBuilder`, VAT, morph/skin
- `./material.md` — standard/PBR/shader материалы
- `./lights.md` — источники света и кластеризация
- `./cameras.md` — типы камер
- `./particles.md` — системы частиц и коллайдеры
- `./physics.md` — тела, джойнты, контроллеры
- `./animation.md` — группы анимаций и скелеты
- `./ai.md` — navmesh, агенты, crowd
- `./texture.md` — текстуры и skybox
- `./ui.md` — `UICanvas` и снапшот-передача
- `./profiler.md` — профайлер и отчёты
- `./serialization.md` — capture/restore и async save/load
- `./visibility.md` — culling и occlusion
