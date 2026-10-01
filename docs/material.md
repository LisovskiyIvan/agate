# Материалы

> Путь: src/agate/material.zig, src/agate/material/, src/agate/node_material.zig, src/agate/material_library.zig · Импорт: agate.Material, agate.PBRMaterial, agate.StandardMaterial (root.zig) · Потоки: создание и мутация на любом потоке (plain data), чтение draw-контуром на context-потоке

## Что это

Материал описывает, как поверхность меша взаимодействует со светом. Ядро — три вида: `StandardMaterial` (простой диффузный, Babylon-паритет), `PBRMaterial` (металлик-рафнесс PBR с расширениями glTF и слоями clearcoat/sheen/anisotropy/transmission/subsurface), `ShaderMaterial` (пользовательский шейдер, см. `./shaders.md`). Полиморфная обёртка `Material` — union указателей (`standard | pbr | shader`), хранимый в `Mesh.material`. Рендер-данные кадра готовит `buildDrawRecord`/`buildShaderSnapshot` (`draw_record.zig`); процедурные графы компилирует `node_material.zig`; готовые пресеты шейдерных материалов даёт `material_library.zig`.

## Быстрый старт

```zig
const agate = @import("agate");

// PBR-материал на куче сцены (живёт, пока жив меш/сцена).
var mat = try scene.allocator.create(agate.PBRMaterial);
mat.* = agate.PBRMaterial.init("hero_armor");
mat.albedo_color = .{ .r = 0.8, .g = 0.2, .b = 0.1 };
mat.metallic = 0.9;
mat.roughness = 0.25;
mat.alpha_mode = .@"opaque";

mesh.setPBRMaterial(mat); // Mesh.material = .{ .pbr = mat }

// Прозрачное стекло: blend-режим + двусторонность.
var glass = try scene.allocator.create(agate.PBRMaterial);
glass.* = .{ .name = "glass", .alpha_mode = .blend, .alpha = 0.4, .double_sided = true };

// Шейдерный пресет из библиотеки.
var sky = agate.material_library.sky("sky_mat", .{});
mesh.material = .{ .shader = &sky };

// Проверка очереди без знания вида.
if (mesh.material) |m| {
    if (m.isTransparent()) { /* blend-очередь */ }
}
```

## API

### `AlphaMode` и общие предикаты (`material/types.zig`, `material/union.zig`)

```zig
pub const AlphaMode = enum { @"opaque", cutout, blend };
pub const Material = union(enum) {
    standard: *StandardMaterial,
    pbr: *PBRMaterial,
    shader: *ShaderMaterial,
    pub fn isTransparent(self: Material) bool; // true только для .blend
    pub fn isCutout(self: Material) bool;      // .cutout: альфа-тест в opaque-очереди
    pub fn isDoubleSided(self: Material) bool;
    pub fn isUnlit(self: Material) bool;
    pub fn setUnlit(self: *Material, unlit_val: bool) void;
    pub fn alphaCutoff(self: Material) f32;
    pub fn name(self: Material) []const u8;
    pub fn alpha(self: Material) f32;
    pub fn alphaMode(self: Material) AlphaMode;
    pub fn baseColor3(self: Material) Color3;
    pub fn primaryTexture(self: Material) ?Texture;
    pub fn tintColor4(self: Material) [4]f32;
};
pub fn coatParamsFor(mat: ?Material) ?CoatParams;
```

Семантика режимов: `opaque` — запись глубины, без блендинга; `cutout` — тот же opaque-контур, но фрагменты с alpha < `alpha_cutoff` отбрасываются (сортировки нет); `blend` — `SRC_ALPHA/ONE_MINUS_SRC_ALPHA`, тест глубины включён, запись выключена, отрисовка после всей непрозрачной геометрии с сортировкой сзади-вперёд.

### `StandardMaterial` (`material/standard.zig`)

```zig
pub const StandardMaterial = struct {
    name: []const u8 = "StandardMaterial",
    diffuse_color: Color3 = Color3.white,
    alpha: f32 = 1.0,
    alpha_mode: AlphaMode = .@"opaque",
    alpha_cutoff: f32 = 0.5,   // только для .cutout
    double_sided: bool = false, // выключает cull (отдельный twin-контур)
    unlit: bool = false,        // bypass освещения: base color + emissive
    emissive_color: Color3 = Color3.black,
    diffuse_texture: ?Texture = null,
    diffuse_uv_transform: UvTransform = .{},
    pub fn init(name: []const u8) StandardMaterial;
    pub fn getDiffuseColor4(self: StandardMaterial) [4]f32;
    pub fn isTransparent(self: StandardMaterial) bool;
    pub fn isCutout(self: StandardMaterial) bool;
};
```

Минимальный дешёвый материал для UI, дебага, неметаллических поверхностей без IBL. glTF никогда не порождает standard-материалы — только ручное создание.

### `PBRMaterial` (`material/pbr.zig`)

```zig
pub const PBRMaterial = struct {
    name: []const u8 = "PBRMaterial",
    albedo_color: Color3 = Color3.white,
    alpha: f32 = 1.0,
    alpha_mode: AlphaMode = .@"opaque",
    alpha_cutoff: f32 = 0.5,
    double_sided: bool = false,
    unlit: bool = false, // bypass: прямое albedo + emissive, без теней/SSAO/IBL
    ior: f32 = 1.5,
    metallic: f32 = 0.0,
    roughness: f32 = 0.5,
    albedo_texture: ?Texture = null,
    normal_texture: ?Texture = null,
    normal_scale: f32 = 1.0, // glTF normalTexture.scale; 0 = плоская геом. нормаль
    metallic_roughness_texture: ?Texture = null,
    emissive_texture: ?Texture = null,
    emissive_color: Color3 = Color3.black,
    occlusion_texture: ?Texture = null,
    occlusion_strength: f32 = 1.0,
    albedo_uv_transform / normal_uv_transform / metallic_roughness_uv_transform /
    emissive_uv_transform / occlusion_uv_transform: UvTransform = .{},
    occlusion_channel: Channel = .r, // glTF-умолчания: AO=R
    roughness_channel: Channel = .g,
    metallic_channel: Channel = .b,
    environment_texture: ?CubeTexture = null,
    environment_intensity: f32 = 1.0,
    clearcoat: Clearcoat = .{},
    sheen: Sheen = .{},
    anisotropy: Anisotropy = .{},
    transmission: Transmission = .{},
    subsurface: Subsurface = .{},
    pub fn init(name: []const u8) PBRMaterial;
    pub fn getAlbedoColor4(self: PBRMaterial) [4]f32;
    pub fn getEmissiveColor4(self: PBRMaterial) [4]f32;
    pub fn isTransparent(self: PBRMaterial) bool;
    pub fn isCutout(self: PBRMaterial) bool;
};
```

Текстуры и слоты:

| Слот | Назначение | sRGB | Примечание |
|---|---|---|---|
| `albedo_texture` | Базовый цвет | Да (конвертация до мипов) | `albedo_uv_transform` |
| `normal_texture` | Касательный нормалмап | Нет (linear) | Масштаб `normal_scale` |
| `metallic_roughness_texture` | ORM-пак (B=metal, G=rough) | Нет | Каналы переназначаются через `Channel` |
| `occlusion_texture` | AO, R-канал по умолчанию | Нет | Сила `occlusion_strength` |
| `emissive_texture` × `emissive_color` | Самосвечение | Да | Аддитивно поверх |
| `environment_texture` | IBL-куб | Да | Интенсивность `environment_intensity` |

`Channel` (`r/g/b/a`) — API для ручных материалов с не-glTF раскладкой ORM; glTF всегда использует фиксированные каналы. `UvTransform` — подмножество `KHR_texture_transform` (`uv' = R(rotation)·(scale·uv) + offset`, радианы, против часовой); второй UV-сет (`texCoord > 0`) не поддерживается — трансформ применяется к texcoord0.

Расширенные слои (все выключены по умолчанию, включение — скаляром, текстура сама по себе слой не включает; выключенный слой шейдит бит-идентично legacy):

| Слой | Гейт | Эффект |
|---|---|---|
| `Clearcoat{ intensity, roughness, color, mask_texture: ?Texture }` | `intensity == 0` → off | Диэлектрический лак: отдельный GGX-лоб, F0 = 0.04, база гасится на (1 − F_cc) |
| `Sheen{ color, intensity, roughness, color_texture: ?Texture }` | `intensity == 0` → off | Тканевый fuzz-лоб (Charlie + Neubelt), аддитивно |
| `Anisotropy{ intensity, rotation }` | `intensity == 0` → изотроп | Растяжение GGX-NDF вдоль касательных; `anisotropyAxes(roughness, intensity)` — CPU-зеркало |
| `Transmission{ factor, color, ior }` | `factor == 0` → off | Дешёвая тонкослойная аппроксимация БЕЗ рефракции (albedo × (1−factor) + аддитивный back-light); настоящее стекло — через `alpha_mode.blend` |
| `Subsurface{ strength, color }` | `strength == 0` → off | Wrap-диффуз + back-scatter от солнца/эмбиента; `wrapNdotL(ndotl, strength)` — CPU-зеркало |

`CoatParams` — render-side GPU-пак слоёв в side-table очереди (не в `MaterialDrawRecord` из-за лимита размера draw-записи); `coatParamsFor(mat)` собирает его из материала.

### `ShaderMaterial` (`material/shader_mat.zig`)

```zig
pub const ShaderMaterial = struct {
    pub fn init(name: []const u8) ShaderMaterial;
    pub fn initForShader(name: []const u8, material_name: []const u8) ?ShaderMaterial;
    pub fn resetUniformDefaults(self: *ShaderMaterial) void;
    pub fn setUniform(self: *ShaderMaterial, name: []const u8, value: shader_material.UniformValue) shader_material.SetUniformError!void;
    pub fn getTintColor4(self: ShaderMaterial) [4]f32;
    pub fn isTransparent(self: ShaderMaterial) bool;
    pub fn isCutout(self: ShaderMaterial) bool;
};
```

Привязка «имя → шейдер из реестра» + типизированные юниформы. Инструментарий пользовательских шейдеров (парсинг `param`-деклараций, мерж, хранение) — в `./shaders.md`.

### Draw-записи (`material/draw_record.zig`)

```zig
pub const MaterialDrawRecord = struct { ... };   // GPU-готовый срез материала на кадр
pub const ShaderDrawSnapshot = struct { ... };
pub fn buildDrawRecord(mat: ?Material, default_white: *const Texture, ...) ...;
pub fn buildShaderSnapshot(mat: ?Material, default_white: *const Texture) ?ShaderDrawSnapshot;
```

Собирают views текстур, факторы и очереди из живого материала; null-слоты подменяются `default_white`. Вызываются draw-контуром, не пользователем.

### `node_material.zig` — процедурные графы

```zig
pub const Type = enum { float, vec2, vec3 }; // тип выхода узла
pub const Kind = enum { const_float, const_color, uv, time, texture_sample,
    add, multiply, mix, sin, cos, output, clamp, step, smoothstep, pow,
    dot, length, normalize, fract, fresnel, panner }; // порты позиционные: connect(dst, port, src)
pub const max_param_len: usize = 48;
pub const time_param_name = "u_time";
pub const Graph = struct {
    pub fn init(allocator: std.mem.Allocator) Graph;
    pub fn deinit(self: *Graph) void;
    pub fn nodeCount(self: *const Graph) usize;
    pub fn addNode(self: *Graph, kind: Kind, opts: NodeOptions) Error!u32;
    pub fn connect(self: *Graph, dst: u32, port: u8, src: u32) void;
    pub fn validate(self: *const Graph, allocator: std.mem.Allocator) Error!void;
    pub fn compile(self: *const Graph, allocator: std.mem.Allocator, name: []const u8) Error!Compiled;
    pub fn serializeJson(self: *const Graph, allocator: std.mem.Allocator) ![]u8;
    pub fn deserializeJson(allocator: std.mem.Allocator, json_text: []const u8) !Graph;
};
pub const Compiled = struct { snippet: []u8, params: []merge.Param, pub fn deinit(...); };
```

Граф валидируется (циклы — ошибка), компилируется в hook-сниппет + таблицу параметров, совместимую с `shader_material.setUniform`. Ошибки: `error{ OutOfMemory, Cycle, TypeMismatch, InvalidJson, ... }` (см. `Error` в источнике).

### `material_library.zig` — пресеты

```zig
pub const sky_shader_name = "matlib_sky";
pub const gradient_shader_name = "matlib_gradient";
pub const grid_shader_name = "matlib_grid";
pub const triplanar_shader_name = "matlib_triplanar";
pub const PresetKind = enum { sky, gradient, grid, triplanar };
pub fn sky(material_name: []const u8, opts: SkyOptions) ?ShaderMaterial;
pub fn gradient(material_name: []const u8, opts: GradientOptions) ?ShaderMaterial;
pub fn grid(material_name: []const u8, opts: GridOptions) ?ShaderMaterial;
pub fn triPlanar(material_name: []const u8, opts: TriPlanarOptions) ?ShaderMaterial;
pub fn applySky/applyGradient/applyGrid/applyTriPlanar(sm: *ShaderMaterial, opts) ...!void;
pub fn presetForName(shader_name: []const u8) ?PresetInfo;
pub fn presetForKind(kind: PresetKind) PresetInfo;
// CPU-зеркала шейдерной математики: skyBlendT, gradientT, gridLineMask, triplanarWeights
```

Возвращают `null`, если шейдер не зарегистрирован. Опции — plain-структуры (`SkyOptions`, `GradientOptions`, `GridOptions`, `TriPlanarOptions`).

## Модель затенения PBR (паритет с Babylon.js)

Спекуляр считается как в Babylon.js: NDF Trowbridge-Reitz (GGX) от
`alphaG = roughness² + 0.0005` (`convertRoughnessToAverageSlope`), видимость —
height-correlated Smith `0.5 / (Gv + Gl)` (`smithVisibility_GGXCorrelated`,
Heitz 2014), итог `f_spec = D · Vis · F` (множитель `NdotL` применяет
вызывающий), диффуз — `albedo · (1 − metallic) / π`. Все формулы живут в
`shaders/common/pbr_brdf.glsl` и используются всеми тремя PBR-контурами
(`pbr`, `instanced_pbr`, `skinned_pbr`), включая clearcoat-лоб.

Ранее стояло приближение Schlick-GGX с `k = (r+1)²/8` (фит Лазарова, в UE4
он предназначен для IBL, а не для прямого света) и деление на
`4·NdotV·NdotL`. На скользящих углах оно давало до 1.8× меньше спекуляра,
чем модель Babylon, и синий оттенок на диэлектриках; на бенче земля
сходилась с эталоном только в пределах 10/255, а после перехода — 2-3/255.

## Потоки и владение

Материалы — plain data без внутренних мьютексов: создавать и мутировать можно на любом потоке до публикации в draw-контур; чтение в кадре — на context-потоке через снапшоты. `Material` хранит указатели — владение за вызывающим (обычно долгоживущие объекты на аллокаторе сцены; освобождение после `destroyMesh`/смены материала). Текстуры внутри материалов — владеющие GPU-хендлы `Texture` (см. `./texture.md`); асинхронная подмена слотов идёт через `UploadQueue`/`PendingTexture.addTarget`.

## Ошибки и краевые случаи

| Ситуация | Поведение |
|---|---|
| `setUniform` с неизвестным именем/типом | `shader_material.SetUniformError` (имя/тип mismatch) |
| `initForShader` для незарегистрированного шейдера | `null` |
| Текстура слоя (clearcoat/sheen) при нулевой интенсивности | Слой выключен, текстура игнорируется (null side-table) |
| `texCoord > 0` из glTF | Игнорируется, используется texcoord0 (задокументированное ограничение) |
| `normal_scale == 0` | Плоская геометрическая нормаль |
| `alpha_cutoff` при `opaque/blend` | Загружается 0.0, тест не срабатывает |
| Анизотропия без авторских касательных | Фолбэк +X (glTF-дефолт), не ошибка |
| Transmission/SSS + точечные/пятно/area/clustered источники | Не вносят вклад в transmitted/SSS-член (v1 скоуп: только солнце + эмбиент) |

## Производительность

- Выключенные слои бесплатны: шейдерные ветки дают бит-идентичный legacy-путь, side-table слоты занимают только активные draws.
- `unlit` пропускает весь световой расчёт (прямые источники, тени, SSAO, IBL) — самый дешёвый освещаемый контур.
- `cutout` дешевле `blend`: остаётся в opaque-очереди без сортировки.
- `double_sided` выбирает twin-контур без cull — только где нужно.
- ORM-пакование (одна текстура на occlusion/roughness/metallic) экономит сэмплы и память против трёх отдельных карт.

## Смотрите также

- `./shaders.md` — реестр шейдеров, юниформы, мерж параметров для `ShaderMaterial`
- `./mesh.md` — `Mesh.material`, назначение через `setPBRMaterial/setStandardMaterial`
- `./texture.md` — текстуры слотов, sRGB, мипмапы, блочные форматы
- `./lights.md` — источники, с которыми взаимодействует PBR
- `./loader.md` — импорт glTF-материалов, семплеры, `KHR_texture_transform`
