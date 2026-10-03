# Меши

> Путь: src/agate/mesh.zig, src/agate/mesh/ · Импорт: agate.Mesh, agate.MeshBuilder, agate.GeometryData (root.zig) · Потоки: создание на любом потоке (отложенный GPU-путь), финализация и отрисовка только на context-потоке

## Что это

Модуль мешей покрывает весь жизненный цикл геометрии: CPU-представление (`GeometryData` — вершины `Vertex` + индексы `u32` + `BoundingBox`), загрузку в GPU-буферы sokol (`uploadGeometry`), владение мешем сценой (`Mesh`), инстансинг (`InstancedMesh`), уровни детализации (`LODLevel`), привязку к костям (`BoneAttachment`), морф-таргеты (blend shapes, режимы `.cpu`/`.gpu`), процедурные примитивы (`MeshBuilder` + `build*Data`), нормали/касательные, BSP-булевы операции (CSG), декали-проекторы, линии (`GreasedLineMesh`), шлейфы (`TrailMesh`), QEM-упрощение и запекание вертечной анимации (VAT).

Фасад `mesh.zig` реэкспортирует всё наружу; подмодули в `mesh/` никогда не импортируют фасад обратно (репозиторное правило против циклов). Чистые CPU-построители (`build*Data`) не трогают GPU и возвращают владеющий `GeometryData`, который затем публикуется через `uploadGeometry`.

## Быстрый старт

```zig
const agate = @import("agate");

// 1. Процедурный примитив: CPU-данные без GPU.
var box_geom: agate.GeometryData = try agate.builders.buildBoxData(
    allocator,
    .{ .width = 2.0, .height = 1.0, .depth = 1.0 },
);
defer box_geom.deinit(allocator);

// 2. Публикация в сцену (на context-потоке — сразу, иначе отложенно).
const mesh: *agate.Mesh = try agate.uploadGeometry(&scene, "box", box_geom);

// 3. Либо фабрика в один вызов:
const ground = try agate.MeshBuilder.createGround(&scene, "ground", .{
    .width = 20.0,
    .height = 20.0,
    .subdivisions = 4,
});
ground.material = agate.Material{ .pbr = &my_pbr };

// 4. Инстансинг: один GPU-меш, много трансформов.
const inst = try mesh.createInstance(&scene, "box#1");
inst.position = .{ .x = 3, .y = 0, .z = 0 };
inst.markDirty();

// 5. Удаление — только через сцену (чистит перекрёстные ссылки).
scene.destroyMesh(mesh);
```

## API

### Базовые типы (`mesh/types.zig`)

```zig
pub const Vertex = extern struct {
    position: [3]f32,
    normal: [3]f32,
    color: [4]f32,
    uv: [2]f32,
    tangent: [4]f32 = .{ 1, 0, 0, 1 },
    joints: [4]f32 = .{ 0, 0, 0, 0 },
    weights: [4]f32 = .{ 1, 0, 0, 0 },
    uv1: [2]f32 = .{ 0, 0 },
};
pub const CullingStrategy = enum { frustum, occlusion, always_render };
pub const MAX_MORPH_TARGETS: usize = 8;
pub const MorphMode = enum { cpu, gpu };
pub const MorphTarget = struct {
    position_deltas: [][3]f32 = &.{},
    normal_deltas: [][3]f32 = &.{},
    tangent_deltas: [][3]f32 = &.{}, // xyz; w касательной сохраняется
};
pub const GeometryData = struct {
    vertices: []Vertex,
    indices: []u32,
    bounds: BoundingBox,
    pub fn deinit(self: *GeometryData, allocator: std.mem.Allocator) void;
};
```

`Vertex` — `extern`, layout фиксирован под вершинный шейдер. `MorphTarget` содержит дельты, масштабируемые весами `Mesh.morph_weights`; пустой слайс означает «атрибут отсутствует». Лимит морфов — 8 на меш (лишние glTF-таргеты загрузчик отбрасывает).

### `Mesh` (`mesh/mesh.zig`) — ключевые поля и методы

```zig
pub fn uploadGeometry(scene: *Scene, name: []const u8, data: GeometryData) !*Mesh;
pub fn createInstance(self: *Mesh, scene: *Scene, name: []const u8) !*InstancedMesh;
pub fn ensureUid(self: *Mesh) u64;
pub fn setStandardMaterial(self: *Mesh, mat: *StandardMaterial) void;
pub fn setPBRMaterial(self: *Mesh, mat: *PBRMaterial) void;
pub fn attachToBone(self: *Mesh, host_mesh: *Mesh, bone_index: usize) void;
pub fn attachToBoneByName(self: *Mesh, host_mesh: *Mesh, bone_name: []const u8) !void;
pub fn detachFromBone(self: *Mesh) void;
pub fn addLODLevel(self: *Mesh, allocator: std.mem.Allocator, distance: f32, lod_mesh: ?*Mesh) !void;
pub fn getLOD(self: *const Mesh, distance: f32) ?*Mesh;
pub fn getLODSq(self: *const Mesh, distance_sq: f32) ?*Mesh;
pub fn getLODForCamera(self: *const Mesh, camera_pos: Vec3) ?*Mesh;
pub fn retainCpuGeometry(self: *Mesh, allocator, vertices: []const Vertex, indices: []const u16) !void;
pub fn retainCpuGeometryU32(self: *Mesh, allocator, vertices: []const Vertex, indices: []const u32) !void;
pub fn retainCpuSkin(self: *Mesh, allocator, vertices: []const Vertex) !void;
pub fn toGeometryData(self: *const Mesh, allocator: std.mem.Allocator) !GeometryData; // или error.NoCpuGeometry
pub fn hasMorphTargets(self: *const Mesh) bool;
pub fn setMorphWeight(self: *Mesh, index: usize, weight: f32) void;   // clamp [0,1], OOB игнорируется
pub fn setMorphWeights(self: *Mesh, weights: []const f32) void;
pub fn retainMorphBase(self: *Mesh, allocator, vertices: []const Vertex) !void;
pub fn applyMorphs(self: *Mesh) void;        // CPU-бленд; в .gpu режиме — no-op
pub fn flushGpuUploads(self: *Mesh) void;    // context-поток, начало кадра
pub fn finishGpuUpload(self: *Mesh, allocator: std.mem.Allocator) void;
pub fn getGpuMemoryBytes(self: *const Mesh) usize;
pub fn getCpuMemoryBytes(self: *const Mesh) usize;
pub fn deinit(self: *Mesh, allocator: std.mem.Allocator) void;
```

Важные поля: `position/rotation (градусы Эйлера)/scaling`, `base_matrix`, `material: ?Material`, `parent`, `skeleton`, `is_visible/cast_shadows/receive_shadows`, `culling_strategy`, `layer_mask`, `local_bounding_box`, CPU-зеркала `cpu_positions/cpu_indices/cpu_skin` (нужны физике и декалям), `instances`, `lod_levels`, `tags: TagSet`.

Контракт `uploadGeometry`: индексы сужаются до 16 бит, если вершин ≤ 65535 (`index_type` выбирается автоматически). Вне context-потока или без валидного sg-контекста меш создаётся в отложенном режиме (`gpu_pending = true`, вершины копируются в `pending_vertices`, CPU-зеркала заполняются), а `Scene.flushPendingGpuUploads` в начале кадра доводит буферы через `finishGpuUpload`. Немедленный путь при исчерпании пула sokol возвращает `error.GpuBufferAllocationFailed` и ничего не публикует. Имя меша заимствованное (`owns_name == false`); переименование — только через `Scene.renameMesh`.

`toGeometryData` требует CPU-зеркал, иначе `error.NoCpuGeometry`. `getWorldMatrix()` возвращает cached × parent-цепочку; `getWorldBoundingBox()` — трансформ локального AABB.

### Инстансинг и LOD

```zig
pub const InstancedMesh = struct { // mesh/mesh.zig
    name: []const u8,
    position/rotation/scaling: Vec3, // rotation — градусы
    is_visible/cast_shadows/receive_shadows: bool,
    culling_strategy: CullingStrategy = .frustum,
    layer_mask: u32 = 0xFFFFFFFF,
    source_mesh: *Mesh,
    pub fn markDirty(self: *InstancedMesh) void;
    pub fn getWorldMatrix(self: *InstancedMesh) Mat4;
    pub fn getWorldBoundingBox(self: *InstancedMesh) BoundingBox;
};
pub const LODLevel = struct { distance: f32, mesh: ?*Mesh };
pub const BoneAttachment = struct { host_mesh: *Mesh, bone_index: usize, offset_matrix: Mat4 };
```

`InstancedMesh` кэширует мировую матрицу и AABB, пересчёт — ленивый по dirty-флагу и сравнению TRS. Рендер-сторона публикует видимые матрицы раз в кадр в `instance_render` (этап `scene/instance_staging.zig`); чтение — через снапшоты кадра. LOD: `addLODLevel` добавляет пару (дистанция, меш), `getLOD/getLODSq/getLODForCamera` выбирают меш по расстоянию; `is_lod_child` помечает дочерние меши.

### Построители (`builder.zig`, `builders.zig`, `builders/`)

`MeshBuilder` — фабрики «сразу в сцену» (все возвращают `!*Mesh`, кроме `createTrail` → `!*TrailMesh`):

`createBox/createGround/createTerrain/createSphere/createCylinder/createCapsule/createTorus/createTorusKnot/createDisc/createRibbon/createLathe/createPlane/createTube/createLines/createExtrude/createPolygon/createDecal/createCSG/createGreasedLine`, плюс `createGreasedLineMesh`, `simplifyMesh`, `generateLODLevels`.

Чистые аналоги `build*Data(allocator, options) !GeometryData` (box/ground/terrain — внутренние для `builders.zig`, остальные реэкспортированы: `buildPlaneData`, `buildTorusData`, `buildTorusKnotData`, `buildDiscData`, `buildRibbonData`, `buildLatheData`, `buildTubeData`, `buildLinesData`, `buildExtrudeData`, `buildPolygonData`, `buildGreasedLineData`). Опции (`BoxOptions`, `SphereOptions`, `GroundOptions`, … `PolygonOptions`, `ExtrudeOptions`) живут в `builders/solids.zig`, `revolve.zig`, `sweep.zig`, `extrude.zig`, `polygon.zig`; общие квад-хелперы — `storeQuad/appendGridQuad/appendGridQuadFlipped/resolveFrameSeed` (`builders/common.zig`).

### Нормали и касательные (`tangents.zig`)

```zig
pub fn computeNormals(vertices: []Vertex, indices: ?[]const u32, indices16: ?[]const u16) void;
pub fn computeTangents(vertices: []Vertex, indices: ?[]const u32, indices16: ?[]const u16) void;
pub fn pickOrthogonal(n: Vec3) Vec3;
```

Пересчёт по месту, сложность O(V+T). Используются загрузчиками и CSG-выходом.

`mesh.tangents.computeTangentsForUv(..., tex_coord: u1)` выбирает UV0/UV1;
обычный `computeTangents` остаётся UV0. glTF-загрузчик выбирает набор normal map
при отсутствии авторских касательных. QEM и CSG копируют оба UV-набора и
интерполируют их при collapse/split (покрыто тестами). CSG из одних
`cpu_positions/cpu_indices` не имеет ни UV0, ни UV1 — оба остаются нулевыми.

### CSG (`csg.zig`)

BSP-булевы операции над твёрдыми телами: `CSG.unionWith/subtract/intersect(other) !CSG`, узлы `CSGNode` (`build/clipTo/clipPolygons/allPolygons/fromPolygons`), полигоны `CSGPolygon` (`init/clone/flip`), плоскости `CSGPlane` (`fromPoints/splitPolygon`, `EPSILON = 1e-5`). Результат публикуется через `MeshBuilder.createCSG(scene, name, &csg_solid)`. Сложность зависит от числа полигонов BSP-дерева; аллокации — на аллокаторе сцены.

### Декали (`decal.zig`)

```zig
pub const DecalOptions = struct { position/normal: Vec3, size: Vec3, angle: f32 = 0,
    cull_backfaces: bool = true, depth_bias: f32 = 0.004, parent_to_target: bool = false };
pub fn buildDecalData(allocator, target: *const Mesh, options: DecalOptions) !GeometryData;
pub fn createDecal(scene, name, target_mesh, options: DecalOptions) !*Mesh;
pub const DecalProjector = struct { pub fn toDecalOptions(self) DecolOptions;
    pub fn projectMesh(self, scene, name, target_mesh) !*Mesh;
    pub fn projectScene(self, scene, name) !?*Mesh; ... };
pub const DecalManager/DecalInstance/DecalSpawnOptions = ...; // пул долгоживущих декалей
pub fn barycentric(p/a/b/c: Vec3) [3]f32;
pub fn blendSkinWeights(...) ...;
```

Проектор вырезает геометрию целевого меша в боксе проектора (требует `cpu_positions/cpu_indices`). Скиннинг переносится через `blendSkinWeights`.

Декаль создаёт **собственную проекторную UV0**, не копирует текстурные
координаты поверхности. UV1 остаётся нулевым, как у остальных одноканальных
процедурных builders; материал декали должен выбирать UV0, если UV1 не
задан вручную. Это генерация нового mapping, не потеря импортированного UV1.

### Линии и шлейфы (`greased_line.zig`, `trail.zig`)

```zig
pub const GreasedLineUVMode = enum { relative, absolute, ... };
pub const GreasedLineColorMode = enum { single, gradient, ... };
pub const GreasedLineOptions = struct { points/paths: ..., width: f32 = 0.1, widths: ?[]const f32,
    color/color_end: Color4, colors: ?[]const Color4, color_mode, up: ?Vec3, camera_pos: ?Vec3,
    closed: bool, uv_mode, uv_scale, dash_ratio/dash_length/dash_offset: f32, miter_limit: f32 = 3.0 };
pub fn buildGreasedLineData(allocator, options: GreasedLineOptions) !GeometryData;
pub const GreasedLineMesh = struct { pub fn init(scene, name, options) !*GreasedLineMesh;
    pub fn setPoints/setWidth/setColor/update(camera_pos)/flushGpuUploads/deinit(...); };
pub const TrailOptions = struct { diameter: f32 = 0.35, segments: u32 = 64, lifetime: f32 = 1.2,
    min_distance: f32 = 0.04, taper: bool = true, color_start/color_end: Color4, auto_start: bool = true };
pub const TrailMesh = struct { pub fn init(scene, name, options) !*TrailMesh;
    pub fn setTarget/addNode/reset/update(dt, camera_pos)/flushGpuUploads/deinit(...); };
```

GreasedLine — камера-билбордированные ленты (статическая геометрия + динамический `GreasedLineMesh`); Trail — затухающая лента за движущейся целью (`setTarget`, узлы стареют по `lifetime`).

### Морфы: CPU-путь и GPU-путь (`morph_gpu.zig`)

```zig
pub const TEXELS_PER_VERTEX: u32 = MAX_MORPH_TARGETS * 3; // 24
pub const TEXTURE_MAX_WIDTH: u32 = 4096;
pub const MorphSlot = enum(u32) { position = 0, normal = 1, tangent = 2 };
pub fn textureSizeFor(vertex_count: usize) TextureSize;
pub fn texelIndex(vertex_index, target: usize, slot: MorphSlot) u32;
pub fn packDeltas(allocator, targets, vertex_count, size) ![]f32;
pub fn blendDeltas(...) BlendedDeltas;      // Zig-зеркало вершинного шейдера
pub fn packWeights(weights: []const f32) PackedWeights;
pub fn vsUniforms(mesh: *const Mesh) VsUniforms; // паникует без дельта-текстуры
pub fn supported() bool;                    // sg.isvalid + RGBA32F sample
pub fn uploadMorphDeltas(mesh: *Mesh, allocator) !void;
```

CPU-режим (по умолчанию): `setMorphWeight(s)` → `morph_dirty`, `applyMorphs()` блендит `staging = base + Σ w·Δ` (SIMD, нормали не перенормируются) и помечает `morph_upload_needed`; `flushGpuUploads()` на context-потоке делает `sg.updateBuffer`. GPU-режим (opt-in через `morph_mode = .gpu` или `LoadOptions.morph_mode`): базовый буфер статичен, дельты пакуются в RGBA32F-текстуру (`TEXELS_PER_VERTEX` текселов на вершину); бленд — в вершинном шейдере forward-контуров `standard/pbr/skinned_pbr`. Вне контекста загрузчик ставит `morph_upload_pending`, финализация — в `finishGpuUpload`. Рисование `.gpu`-меша без загруженной дельта-текстуры — паника (`vsUniforms`), тихого отката к base pose нет. Гейт возможностей — `supported()`.

### Упрощение и LOD-генерация (`simplify.zig`)

```zig
pub const SimplifyOptions = struct { target_ratio: f32 = 0.5, target_triangles: ?usize = null,
    max_error: f32 = 1.0, preserve_border: bool = true, preserve_attributes: bool = true,
    prevent_normal_flips: bool = true, border_penalty: f32 = 500.0 };
pub fn simplifyGeometry(allocator, data: *const GeometryData, options: SimplifyOptions) !GeometryData;
pub fn simplifyMesh(allocator, scene, name, source_mesh: *Mesh, options: SimplifyOptions) !*Mesh;
pub const LODLevelSpec = struct { distance: f32, ratio: f32, options: ?SimplifyOptions = null };
pub fn generateLODLevels(allocator, scene, source_mesh: *Mesh, specs: []const LODLevelSpec) !void;
```

QEM edge-collapse: `target_triangles` перекрывает `target_ratio`. `simplifyMesh` копирует материал и трансформ источника. `generateLODLevels` именует детей `{name}_lod{d}` и линкует через `addLODLevel`.

### VAT (`vat.zig`)

Запекание вертечной анимации в текстуры (толпы/инстансинг без скелетов):

```zig
pub const VatLayout = enum { grid, ... };
pub const VatConfig = struct { fps: f32 = 30, include_normals: bool = true, texture_max_width: u32 = 4096 };
pub const VatSampleParams = struct { frame0/frame1: u32, lerp_frac: f32, total_frames: f32 };
pub const VatData = struct { ... positions: []f32, normals: ?[]f32, position_image/view, normal_image/view ...;
    pub fn samplePosition/sampleNormal/samplePositionInterpolated/sampleNormalInterpolated(...) Vec3;
    pub fn texelCoord/texelFloatIndex(...) ...; pub fn uploadTextures(self) !void;
    pub fn calculateBounds(self) !void; pub fn deinit(self) void; };
pub const VatBaker = struct { pub fn bakeSkeletal(...) ...; pub fn bakeProcedural(...) ...; };
pub const VatPlayer = struct { pub fn init(vat) VatPlayer; pub fn update(dt)/seek/getSampleParams/currentFrame(...); };
```

## Потоки и владение

Владелец всех мешей — `Scene` (`scene.meshes`): `uploadGeometry` и `MeshBuilder.create*` сразу аппендят меш; освобождение — только `scene.destroyMesh` (чистит ссылки из слоёв, очередей и ретайр GPU через эпохи; вне context-потока — отвязка сейчас, GPU-теардаун позже). `Mesh.deinit` напрямую не вызывать для живых мешей сцены. `GeometryData` владеет слайсами, освобождается через `deinit`; `uploadGeometry` копирует данные в GPU и CPU-зеркала, входной `GeometryData` после вызова можно освободить. `InstancedMesh` владеет сцен-аллокатор (`instances`), удаляется вместе с мешем. Морф-слайсы, `cpu_*`, `pending_vertices` принадлежат scene-аллокатору и чистятся в `deinit`. Имя — заимствованное, если не установлено `owns_name`.

Поточность: чистые `build*Data`, `simplifyGeometry`, `packDeltas`, CSG — потокобезопасны (только аллокатор). `uploadGeometry`/`create*` с любого потока безопасны благодаря отложенному пути; `finishGpuUpload/flushGpuUploads/deinit` (sg.*) — только context-поток, их дёргает `Scene.flushPendingGpuUploads` в начале кадра. `applyMorphs/setMorphWeight` — игровые/анимационные потоки, сам бленд CPU и дешёвый; фактический `sg.updateBuffer` — только во flush.

## Ошибки и краевые случаи

| Ситуация | Поведение |
|---|---|
| Пула sokol нет (`makeBuffer id == 0`) | `error.GpuBufferAllocationFailed`, меш не публикуется, утечек нет |
| Нет sg-контекста / чужой поток | Отложенный путь (`gpu_pending`), ретрай во flush; тесты без контекста не трогают sg |
| `toGeometryData` без CPU-зеркал | `error.NoCpuGeometry` |
| `setMorphWeight` с OOB-индексом | Игнорируется, без паники |
| > 8 морф-таргетов в glTF | Лишние отбрасываются при импорте |
| `.gpu`-меш без дельта-текстуры | Паника в `vsUniforms` (громко, не base pose) |
| RGBA32F не поддерживается | `supported() == false`, `uploadMorphDeltas` — громкая ошибка, ретрай каждый flush (лог — один раз) |
| Вырожденный CSG / пустая геометрия | Пустой `CSG`/ошибка аллокатора; экспорт пропускает меши без `cpu_positions` |
| Переименование меша напрямую | Запрещено: только `Scene.renameMesh` (copy-before-free, OOM-атомарность) |

## Производительность

- Индексы u16 при ≤ 65535 вершин — вдвое меньше индексного трафика; выбор автоматический.
- CPU-морфы: один SIMD-проход по вершинам только по активным (вес ≠ 0) таргетам; `sg.updateBuffer` — только когда `morph_upload_needed`, объём учитывается в `gpu_upload_meter`.
- GPU-морфы: нулевая CPU-стоимость бленда в кадре (только веса в юниформах); цена — RGBA32F-текстура `width × height × 16` байт (см. `getGpuMemoryBytes`).
- Инстансинг: одна публикация матриц в кадр с Wyhash-дедупом аплоада (`hash` + `uploaded_count`).
- `getGpuMemoryBytes/getCpuMemoryBytes` — точная оценка для бюджетирования (вершины + индексы + дельта-текстура + инстанс-буфер).
- `simplifyGeometry` — тяжёлая офлайн-операция (QEM, куча рёбер), не вызывать в кадре; LOD генерировать на загрузке.

## Смотрите также

- `./material.md` — материалы, назначаемые мешам
- `./loader.md` — спавн мешей из glTF/OBJ/STL/PLY, `LoadOptions.morph_mode`
- `./export.md` — выгрузка мешей (читает `cpu_positions/cpu_indices`)
- `./animation.md` — скелеты, скиннинг, морф-веса из анимаций
- `./visibility.md` — отсечение, LOD-выбор в кадре
- `./physics.md` — коллайдеры из CPU-геометрии
- `./scene.md` — владение, `destroyMesh`, `flushPendingGpuUploads`
