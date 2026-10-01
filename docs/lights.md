# Источники света

> Путь: src/agate/lights.zig, src/agate/lights/ · Импорт: agate.DirectionalLight, agate.PointLight, agate.SpotLight, agate.HemisphericLight, agate.AreaLight, agate.ClusteredPointLight (root.zig) · Потоки: создание/мутация на любом потоке, паковка в uniforms на context-потоке

## Что это

Световой модуль — семь листьев под фасадом `lights.zig`: фоновый `HemisphericLight`, солнце + заливки `DirectionalLight`, всенаправленный `PointLight` (опциональные куб-тени), конусный `SpotLight` (опциональная тень), прямоугольный `AreaLight` (без теней), пул из до 64 `ClusteredPointLight` для clustered-forward и солнечные резолверы/цветовая температура (`sun.zig`). Владеет источниками `LightRig` сцены (`scene/light_rig.zig`): жёсткие лимиты слотов, top-k отбор point/spot с гистерезисом, паковка uniform-массивов кадра. Листья импортируют только `math` (и `std` для spot/sun) и никогда — фасад.

## Быстрый старт

```zig
const agate = @import("agate");

// Солнце с тенью (слот 0) + заливка без теней (слот 1).
scene.light_rig.directional = agate.DirectionalLight.init("sun", .{
    .direction = agate.sunDirectionFromAngles(0.6, 0.9),
    .diffuse = agate.colorTemperatureToRgb(5600),
    .intensity = 3.0,
});
_ = try scene.light_rig.addDirectionalLight(scene.allocator, "fill", .{
    .direction = .{ .x = -0.5, .y = 0.3, .z = -0.5 },
    .intensity = 0.4,
});

// Точечный с тенью-кубом.
_ = try scene.light_rig.createPointLight(scene.allocator, "lamp", .{
    .position = .{ .x = 2, .y = 3, .z = 1 },
    .color = agate.math.Color3.white,
    .intensity = 20.0,
    .range = 12.0,
    .cast_shadows = true,
});

// Массовый пул без теней (value-тип, без имён).
_ = try scene.light_rig.addClusteredPointLight(.{ .x = 0, .y = 2, .z = 0 }, .{
    .intensity = 5.0,
    .radius = 8.0,
});
```

## API

### Лимиты слотов

| Источник | Лимит | Константа | Тени |
|---|---|---|---|
| Directional (солнце + заливки) | 4 (1 солнце + 3 заливки) | `max_directional_lights = 4`, `max_fill_directionals = 3` | Только солнце (CSM); заливки — всегда без теней |
| Point | 4 uniform-слота (top-k) | `LightRig.point_slots = 4` | Опционально, атлас; одновременно теневых — до `point_shadow_slots = 2` |
| Spot | 2 uniform-слота (top-k) | `LightRig.spot_slots = 2` | Опционально (`getShadowViewProj`) |
| Area (rect) | 2 | `max_area_lights = 2` | Нет (v1 без теней) |
| Clustered point | 64, вне legacy-слотов | `max_clustered_lights = 64` | Нет (unshadowed by design) |
| Hemispheric | 1 (фон сцены) | — | Нет (эмбиент) |

Превышение жёстких лимитов — громкие ошибки (`TooManyDirectionalLights`, `TooManyAreaLights`, `TooManyClusteredLights`), не тихий кламп. Заливки directional и area-источники сессионные: save/load их не персистит. Отключённые источники (`is_enabled = false`) пакуются нулевыми слотами и резолверами солнца считаются отсутствующими.

### `HemisphericLight` (`lights/hemispheric.zig`)

```zig
pub const HemisphericLightOptions = struct {
    direction: Vec3 = Vec3.up,
    diffuse: Color3 = Color3.white,
    ground_color: Color3 = Color3.black, // как HemisphericLight в Babylon
    intensity: f32 = 1.0,
};
pub const HemisphericLight = struct {
    name: []const u8 = "HemisphericLight",
    direction: Vec3, diffuse: Color3, ground_color: Color3, intensity: f32,
    pub fn init(name: []const u8, options: HemisphericLightOptions) HemisphericLight;
};
```

Небесно-земляной свет по модели Babylon.js `HemisphericLight`: вклад в
освещённость интерполируется между `ground_color` (поверхность отвернута от
света) и `diffuse * intensity` (поверхность смотрит на свет) по
`0.5 + 0.5 * dot(N, direction)`; итог умножается на альбедо и
`(1 - metallic)` (у металлов диффуза нет). Реализация —
`shaders/common/hemi.glsl`, юниформы `hemi_dir_intensity` / `hemi_diffuse`
(`scene/uniforms.zig`, дописаны в конец `fs_params`, чтобы не сдвинуть
существующие оффсеты). `ground_color` по-прежнему едет в слоте
`ambient_color.rgb`. Также legacy-фолбэк солнца, когда directional
отсутствует или выключен.

Полусферический свет даёт ещё и **спекулярный лепесток** (как в Babylon, где
`computeSpecularLighting` вызывается и для `HEMILIGHT`, причём цветом служит
`vLightDiffuse`, а не `vLightSpecular` — поэтому `hemi.specular` у Babylon
ни на что не влияет). Формула: `L = normalize(direction)`,
`NdotL = dot(N, L) * 0.5 + 0.5`, `attenuation = 1`, `roughness` материала,
`D · Vis · F(VdotH, F0)` без множителя `(1 − metallic)`; вклад строго нулевой
при `intensity = 0`. До этой правки весь полусферический вклад у металла
гас множителем `(1 − metallic)`, из-за чего металл у agate был темнее
Babylon на ~9/255 в среднем.

Диффузная часть дополнительно масштабируется ground-цветом на интенсивность
(`vLightGround = groundColor * intensity`, как в Babylon).

Ранее `direction`, `diffuse` и `intensity` игнорировались, а `ground_color`
работал плоским ambient: любая сцена выходила темнее и синее, чем в
Babylon.js (на бенче земля давала 0.39 linear вместо 0.60 при тех же двух
источниках).

### `DirectionalLight` (`lights/directional.zig`)

```zig
pub const DirectionalLightOptions = struct {
    direction: Vec3 = Vec3.new(0.5, 1.0, 0.5),
    diffuse: Color3 = Color3.white,
    intensity: f32 = 1.0,
};
pub const DirectionalLight = struct {
    name: []const u8 = "DirectionalLight",
    owns_name: bool = false, // имя из glTF-лоадера освобождает сцена
    direction: Vec3, diffuse: Color3, intensity: f32,
    is_enabled: bool = true,
    pub fn init(name: []const u8, options: DirectionalLightOptions) DirectionalLight; // нормализует direction
};
```

Слот 0 — солнце с CSM-тенью; `LightRig.addDirectionalLight` добавляет заливки в порядке создания (`directionalAt(i)`, `directionalCount()`).

### `PointLight` (`lights/point.zig`)

```zig
pub const PointLightOptions = struct {
    position: Vec3 = Vec3.zero,
    color: Color3 = Color3.white,
    intensity: f32 = 1.0,
    range: f32 = 10.0,
    cast_shadows: bool = false,
    shadow_bias: f32 = 0.002,
    shadow_normal_bias: f32 = 0.005,
    shadow_near: f32 = 0.1,
};
pub const PointLight = struct {
    ... range, is_enabled, cast_shadows, shadow_bias/normal_bias/near: f32,
    pub const shadow_face_count: usize = 6; // +X,-X,+Y,-Y,+Z,-Z
    pub fn init(name: []const u8, options: PointLightOptions) PointLight;
    pub fn getShadowFaceViewProj(self: PointLight, face: usize) Mat4; // 90°, aspect 1, near..range
};
```

Теневая матрица грани: перспектива 90° от позиции источника, far = max(range, near + 0.1). Раскладка тайлов атласа — в `passes/shadow_pass.zig` (`pointTileOrigin`).

### `SpotLight` (`lights/spot.zig`)

```zig
pub const SpotLightOptions = struct {
    position: Vec3 = Vec3.zero,
    direction: Vec3 = Vec3.new(0, -1, 0),
    color: Color3 = Color3.white,
    intensity: f32 = 1.0,
    range: f32 = 15.0,
    inner_angle_deg: f32 = 15.0,
    outer_angle_deg: f32 = 30.0,
    cast_shadows: bool = false,
    shadow_bias: f32 = 0.002,
    shadow_normal_bias: f32 = 0.005,
    shadow_near: f32 = 0.1,
};
pub const SpotLight = struct {
    ...
    pub fn init(name: []const u8, options: SpotLightOptions) SpotLight; // нормализует direction
    pub fn getShadowViewProj(self: SpotLight) Mat4; // fov = 2×outer, clamp [1,175]
};
```

При вырожденном direction — фолбэк (0,−1,0); при вертикальном направлении up переключается на (0,0,1), чтобы `lookAt` не вырождался.

### `AreaLight` (`lights/area.zig`)

```zig
pub const AreaLightOptions = struct {
    center: Vec3 = Vec3.zero,
    right: Vec3 = Vec3.new(0.5, 0, 0), // полуширина-вектор (+X)
    up: Vec3 = Vec3.new(0, 0.5, 0),    // полувысота-вектор (+Y)
    color: Color3 = Color3.white,
    intensity: f32 = 1.0,
    is_enabled: bool = true,
};
pub const AreaLight = struct {
    ...
    pub fn init(name: []const u8, options: AreaLightOptions) AreaLight;
    pub fn normal(self: AreaLight) Vec3; // normalize(right × up), zero при вырождении
    pub fn area(self: AreaLight) f32;    // 4·|right × up|
};
```

Прямоугольник задаётся центром и двумя полувекторами (угол = center ± right ± up). Вырожденный прямоугольник (нулевая площадь, параллельные оси) излучает ноль: `normal()` возвращает zero, шейдер гейтит по площади. Только API v1: в glTF прямоугольных источников нет (`KHR_lights_punctual` их не описывает), лоадер их не создаёт.

### `ClusteredPointLight` (`lights/clustered.zig`)

```zig
pub const ClusteredPointLightOptions = struct {
    color: Color3 = Color3.white,
    intensity: f32 = 1.0,
    radius: f32 = 10.0, // <= 0 пакуется как отсутствующий
    enabled: bool = true,
};
pub const ClusteredPointLight = struct {
    position: Vec3 = Vec3.zero, color: Color3, intensity: f32, radius: f32,
    is_enabled: bool = true,
    pub fn init(position: Vec3, options: ClusteredPointLightOptions) ClusteredPointLight;
};
```

Value-тип без имён/кучи/теней: `LightRig` держит фиксированный массив + счётчик, сцена — индексные add/remove/get/count. Едут вне legacy top-k линий (`point_slots`), тайлинг — в `scene/clustered_lights.zig`, GPU-билд читает staged `FramePack` (лаг 1 кадр).

### Солнце и цветовая температура (`lights/sun.zig`)

```zig
pub fn resolveSunDirection(directional_light: ?*const DirectionalLight, hemi: HemisphericLight) Vec3;
pub fn resolveSunColor(directional_light: ?*const DirectionalLight, hemi: HemisphericLight) Color3;
pub fn resolveSunIntensity(directional_light: ?*const DirectionalLight, hemi: HemisphericLight) f32;
pub fn sunDirectionFromAngles(azimuth_rad: f32, elevation_rad: f32) Vec3;
pub fn colorTemperatureToRgb(kelvin: f32) Color3;
```

Активный directional перекрывает hemispheric-фолбэк; выключенный считается отсутствующим. `sunDirectionFromAngles`: elevation от горизонта (0 = горизонт, π/2 = зенит), azimuth вокруг +Y (0 = +Z, π/2 = +X); вырождение возвращает `Vec3.up`. `colorTemperatureToRgb` — аппроксимация Планкиана Таннера Хелланда, вход клампится к [1000, 12000] К, выход — линейный RGB.

## Потоки и владение

Источники — plain-структуры; мутация полей (`position`, `intensity`, `is_enabled`) потокобезопасна в смысле данных (без атомиков, фазовое владение), чтение паковщиком — на context-потоке в prepare-фазе. Именованные источники живут в `LightRig` (создание через `create*/add*` с аллокатором; glTF-имена с `owns_name` освобождает сцена). Clustered-пул — значения в фиксированном массиве рига, без кучи.

## Ошибки и краевые случаи

| Ситуация | Поведение |
|---|---|
| 5-й directional / 3-й area / 65-й clustered | Жёсткая ошибка `TooMany*`, без замены |
| Выключенный directional | Резолверы солнца падают на hemispheric |
| Нулевой `direction` у directional | `resolveSunDirection` падает на hemispheric (защита `lengthSq > 1e-12`) |
| Вырожденный area-прямоугольник | Ноль излучения, не NaN |
| `radius <= 0` у clustered | Пакуется как отсутствующий (тайл-билд пропускает) |
| `cast_shadows` при выключенных тенях сцены | Теневые слоты не генерируются (`shadows_enabled` гейт) |
| Point/spot вне top-k | Не пакуются в uniforms; clustered-пул — без top-k, пакуются все включённые |

## Производительность

- Point/spot отбираются top-k по скору с гистерезисом (`light_selection.Hysteresis`) — без мерцания слотов на границе.
- Uniform-массивы фиксированного размера (4 point + 2 spot + fills + area + clustered-пул): стоимость шейдера константна, не зависит от числа источников в сцене.
- Clustered-forward: до 64 дополнительных точечных с 2D screen-tile отбраковкой — масштабируется на сценах с множеством локальных источников, где legacy-слоты не справляются.
- Тени — только где дёшево: CSM солнца, до 2 теневых point-атласов, spot-тени по запросу; area/clustered/заливки теней не имеют by design.

## Смотрите также

- `./material.md` — PBR-взаимодействие со светом, IBL
- `./loader.md` — `KHR_lights_punctual`, диапазоны и конусы по умолчанию
- `./scene.md` — `LightRig`, владение источниками
- `./render-pipeline.md` — паковка uniforms, теневые проходы
- `./passes.md` — shadow pass, тайлы point-атласа
