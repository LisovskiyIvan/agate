# Загрузчик сцен

> Путь: src/agate/loader/ · Импорт: agate.SceneLoader, agate.parseObj/appendObjToScene, agate.parseStl/appendStlToScene, agate.parsePly/appendPlyToScene (root.zig) · Потоки: парсинг на любом потоке; sg-создание на context-потоке (вне его — принудительный async-режим текстур)

## Что это

Оркестратор импорта: `SceneLoader` (glTF `.glb`/`.gltf` через cgltf) плюс нативные читатели OBJ/STL/PLY и декодер meshopt. Конвейер glTF: парсинг → проверка `extensions_required` → `load buffers` → meshopt-распаковка → материалы (с параллельным предекодом изображений) → скины/скелеты → спавн мешей с нодовыми трансформами → анимации → punctual-источники и камеры. Подмодули: `gltf_util` (TRS/кватернионы/семплеры), `materials` (слоты, семплеры, async-контекст), `skins` (скелеты, joint-маска), `mesh_spawn` (примитивы, морфы, CPU-зеркала), `animations` (треки), `lights` (источники, камеры, конусы), `obj/stl/ply` (текстовые/слайсерные форматы), `meshopt` (EXT_meshopt_compression), `fixtures` (тестовые данные).

## Быстрый старт

```zig
const agate = @import("agate");

// glTF целиком: меши + материалы + скелеты + анимации + свет + камеры.
const meshes: []*agate.Mesh = try agate.SceneLoader.appendGlb(&scene, "assets/hero.glb");

// С опциями: GPU-морфы + фоновые текстуры.
const meshes2 = try agate.SceneLoader.appendGlbOptions(&scene, "assets/hero.glb", .{
    .morph_mode = .gpu,
    .async_textures = true,
});

// Простые форматы напрямую.
const obj_meshes = try agate.appendObjToScene(&scene, scene.allocator, "prop", obj_bytes);
const stl_meshes = try agate.appendStlToScene(&scene, scene.allocator, "part", stl_bytes);
const ply_meshes = try agate.appendPlyToScene(&scene, scene.allocator, "scan", ply_bytes);

// Только парсинг без сцены (валидация, инспекции).
var data = try agate.parseObj(allocator, obj_bytes);
defer data.deinit(allocator);
```

## API

### `SceneLoader` (`loader/scene_loader.zig`)

```zig
pub const SceneLoader = struct {
    pub const LoadOptions = struct {
        morph_mode: MorphMode = .cpu,   // бленд морфов: CPU-буфер или GPU дельта-текстура
        async_textures: bool = false,   // декод изображений на UploadQueue-воркерах
    };
    pub fn appendGlb(scene: *Scene, file_path: []const u8) ![]*Mesh; // == appendGlbOptions(..., .{})
    pub fn appendGlbOptions(scene: *Scene, file_path: []const u8, load_options: LoadOptions) ![]*Mesh;
};
pub fn isExtensionSupported(name: []const u8) bool; // "EXT_meshopt_compression" (+ базовый набор)
```

Возвращает слайс заспавненных мешей в порядке нод (владеет сцена). Ошибки: `GltfParseFailed`, `UnsupportedGltfExtension` (любое из `extensions_required` вне поддержки — отказ целиком по спецификации glTF 2.0), `GltfLoadBuffersFailed`, `GltfMeshoptDecodeFailed`, далее ошибки аллокатора/GPU. `.gltf` (JSON + внешний .bin) и `.glb` поддерживаются одинаково; внешние изображения грузятся относительно `base_dir` файла.

`morph_mode`: `.cpu` — историческое поведение (динамический вершинный буфер, `applyMorphs` в первом кадре); `.gpu` — статичный base-буфер + RGBA32F дельта-текстура (требует `morph_gpu.supported()`, иначе громкая ошибка; только forward standard/pbr/skinned контуры). Вне context-потока CPU-морфы получают отложенный динамический буфер (`pending_dynamic_update`), GPU-морфы — `morph_upload_pending` с финализацией во flush.

`async_textures`: предекод пропускается, изображения декодятся на воркерах `scene.uploads`, материалы стартуют с null-слотами (рендерится `default_white`) и патчатся через `scene.render()` drain. Вне графического потока async принудительно включается при наличии очереди — синхронное создание касается sg.* и громко упадёт. Без очереди — тихий фолбэк в синхронный режим.

### Подмодули glTF

```zig
// gltf_util.zig — математика нод и семплеров.
pub fn gltfNodeIndex(gltf: *c.cgltf_data, node: *c.cgltf_node) ?usize;
pub fn quatFromBasis(x/y/z: Vec3) Quat;
pub fn nodeLocalTRS(node: *const c.cgltf_node) struct { pos: Vec3, rot: Quat, scale: Vec3 };
pub fn nodeParentWorld(node: *const c.cgltf_node) Mat4;
pub const SamplerData = struct { ... };
pub fn readSampler(allocator, samp: *c.cgltf_animation_sampler, stride: usize) !?SamplerData;

// materials.zig — слоты, семплеры, async-контекст.
pub fn textureImage(tex) ?[*c]const c.cgltf_image;
pub fn colorSlotImageFlags(allocator, gltf) ![]bool;
pub fn decodeImagesInParallel(scene, gltf, decoded, base_dir: ?[]const u8) void; // fork-join предекод
pub fn applyGltfSampler(wrap_s/wrap_t/mag/min: c_int, opts: *Texture.Options) void;
pub fn uvTransformFromView(view: anytype) UvTransform; // KHR_texture_transform
pub const AsyncTexCtx = struct { pub fn init(scene, gltf, base_dir, queue) AsyncTexCtx;
    pub fn deinit(...); pub fn register(self, view, srgb: bool, slot: *?Texture) void; };
pub fn loadTextureSlot/loadTextureFromView(...) ...;
pub fn loadMaterials(scene, gltf, base_dir, materials, image_cache, decoded, async_ctx) !void;

// skins.zig — скелеты.
pub fn loadSkins(scene, gltf, skeletons: []?*Skeleton) !void;
pub fn buildJointMask(allocator, gltf) ![]bool; // ноды-джойнты пропускаются нодовой анимацией

// mesh_spawn.zig — геометрия.
pub fn spawnMeshes(scene, gltf, materials, skeletons, out, node_mesh_start/count, morph_mode: MorphMode) !void;
pub fn parsePrimitive(...) ...; // индексы, атрибуты, морфы (первые 8), скин, CPU-зеркала

// animations.zig — треки скелетов и нод.
pub fn loadAnimations(scene, gltf, skeletons, spawned_meshes, node_mesh_start/count, is_joint_node) !void;

// lights.zig — свет, камеры, математика конусов.
pub const default_point_range: f32 = 10.0;
pub const default_spot_range: f32 = 15.0;
pub const default_camera_fov_deg: f32 = 60.0;
pub const default_camera_near/far: f32 = 0.1 / 100.0;
pub fn lightColor/lightIntensity/rangeOrDefault/radToDeg/spotConeDeg(...) ...;
pub fn forwardFromWorld/forwardFromEuler/lookDirectionToEuler/perspectiveFovDeg(...) ...;
pub fn resolveLightName/resolveCameraName(...) ...;
pub fn loadLights(scene, gltf: *const c.cgltf_data, parent_world: Mat4) !void;   // KHR_lights_punctual
pub fn loadCameras(scene, gltf: *const c.cgltf_data, parent_world: Mat4) !usize; // → FreeCamera
```

Соответствия glTF: материалы → PBR (albedo/MR/normal/emissive/occlusion + `KHR_texture_transform` в `*_uv_transform`, семплеры в `Texture.Options`); скин → `Skeleton` + `cpu_skin`; морфы → `MorphTarget` (первые `MAX_MORPH_TARGETS`, пустые атрибуты = пустые слайсы); источники `KHR_lights_punctual` → Point/Spot/Directional (range/конусы по умолчанию выше при отсутствии); камеры → `FreeCamera`. Имена glTF heap-дублируются (`owns_name`, освобождает сцена).

### Простые форматы

```zig
// obj.zig
pub const max_triangles: usize = 10_000_000;
pub const ObjData = struct { pub fn vertexCount(self) usize; pub fn deinit(self, *ObjData, allocator) void; };
pub fn parse(allocator, bytes: []const u8) !ObjData;              // позиции/uv/нормали, ключ (pos,uv,nrm)
pub fn appendToScene(scene, allocator, name: []const u8, bytes: []const u8) ![]*Mesh;
// stl.zig
pub const max_facets: usize = 10_000_000;
pub const StlData = struct { ... };
pub fn parse(allocator, bytes: []const u8) !StlData;              // ASCII + binary, little-endian
pub fn appendToScene(scene, allocator, name, bytes) ![]*Mesh;
// ply.zig
pub const PlyData = struct { pub fn vertexCount/averageColor/deinit(...) ... };
pub fn parse(allocator, bytes: []const u8) !PlyData;              // ASCII/бинарный, цвета вершин
pub fn appendToScene(scene, allocator, name, bytes) ![]*Mesh;
// meshopt.zig — EXT_meshopt_compression (режимы и ограничения — в исходнике).
```

Лимиты anti-OOM: OBJ свыше 10M треугольников, STL свыше 10M фасеток — ошибка парсинга. OBJ-импортёр ключеит вершины тройкой (позиция, uv, нормаль); экспортный дедуп нормалей (см. `./export.md`) спроектирован под этот ключ, чтобы round-trip не раздувал вершины. STL без нормалей/цветов (фасеточные нормали вычисляются); PLY несёт вершинные цвета (`averageColor` для фолбэка материала).

## Потоки и владение

Парсинг (`parse*`, cgltf-разбор, meshopt) — чистый CPU, любой поток. Спавн мешей идёт через `uploadGeometry`: вне context-потока — отложенный путь с финализацией во flush. Текстуры: синхронный путь трогает sg inline (только context), async-путь — только очередь + патч слотов. Всё заспавненное владеет сцена (меши, материалы, скелеты, анимации, источники, камеры); `ObjData/PlyData/StlData` — временные, `deinit` на вызывающем. Возвращённый слайс `[]*Mesh` аллоцирован на аллокаторе сцены.

## Ошибки и краевые случаи

| Ситуация | Поведение |
|---|---|
| Неподдерживаемое `extensions_required` | `UnsupportedGltfExtension`, файл целиком отклоняется |
| Битый JSON/GLB, недоступный .bin | `GltfParseFailed` / `GltfLoadBuffersFailed` |
| Ошибка meshopt-распаковки | `GltfMeshoptDecodeFailed` |
| Морфов > 8 | Первые 8 загружаются, остальные отбрасываются (документировано) |
| `texCoord > 0` | Игнорируется (texcoord0, см. `./material.md`) |
| Отсутствующие range/углы/камеры | Дефолты `default_*` выше |
| OBJ > 10M треугольников / STL > 10M фасеток | Ошибка парсинга (anti-OOM) |
| Пустой STL (0 фасеток) | Импорт сообщает `NoGeometry` |
| Async без очереди | Тихий фолбэк в синхронный декод |

## Производительность

- Синхронный предекод изображений — fork-join (параллелизм по изображениям; один декод монолитен и не сплитится): один хитч загрузки вместо поп-ина текстур.
- Async-режим убирает хитч ценой временного `default_white` и pop-in по мере drain (бюджет — см. `./assets.md`).
- `image_cache` дедуплицирует изображения glTF (×2 слота под sRGB-варианты): одна картинка на albedo+emissive декодируется один раз.
- CPU-зеркала (`cpu_positions/indices/skin`) удерживаются всегда: цена памяти против физики/декалей/экспорта без повторного парсинга.

## Смотрите также

- `./mesh.md` — меши, морфы, `uploadGeometry`-контракт
- `./material.md` — слоты и семплеры glTF-материалов
- `./texture.md` — декод изображений, Basis, мипмапы
- `./animation.md` — скелеты и треки из glTF
- `./lights.md` — punctual-источники и их дефолты
- `./cameras.md` — glTF-камеры как `FreeCamera`
- `./assets.md` — `UploadQueue` для async-текстур
