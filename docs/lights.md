# Источники света

> Путь: src/lights.zig, src/lights/ · Импорт: agate.DirectionalLight, agate.PointLight, agate.SpotLight, agate.HemisphericLight, agate.AreaLight, agate.ClusteredPointLight (root.zig) · Потоки: создание/мутация — game-поток под update-exclusion; паковка (updateLights/packFrame) — game-поток; рендер читает staged snapshot

## Что это

Световой модуль — семь листьев под фасадом `lights.zig`: фоновый `HemisphericLight`, солнце + заливки `DirectionalLight`, всенаправленный `PointLight` (опциональные куб-тени), конусный `SpotLight` (опциональная тень), прямоугольный `AreaLight` (без теней), clustered-пул (`ClusteredPointLight`/`ClusteredSpotLight`) и солнечные резолверы/цветовая температура (`sun.zig`). Владеет источниками `LightRig` сцены (`scene/light_rig.zig`): жёсткие лимиты слотов, единый SSBO локального света (до 64 point+spot+extra) и legacy top-k/hysteresis для probe-слоёв и выбора теневых кастеров. Листья импортируют только `math` (и `std` для spot/sun) и никогда — фасад.

## Быстрый старт

```zig
const agate = @import("agate");

// Солнце с тенью (слот 0) + заливка без теней (слот 1).
const sun = try scene.light_rig.createDirectionalLight(scene.allocator, "sun", .{
    .direction = agate.sunDirectionFromAngles(0.6, 0.9),
    .diffuse = agate.colorTemperatureToRgb(5600),
    .intensity = 3.0,
});
_ = sun; // scene.light_rig.directional уже указывает на него
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
    .shadow = .{ .cast_shadows = true },
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
| Point | 4 uniform-слота legacy/probe (top-k) + единый SSBO | `LightRig.point_slots = 4` | Опционально, атлас; одновременно теневых — до `point_shadow_slots = 2` |
| Spot | 2 uniform-слота legacy/probe (top-k) + единый SSBO | `LightRig.spot_slots = 2` | Опционально (`getShadowViewProj`) |
| Area (rect) | 2 | `max_area_lights = 2` | Нет (v1 без теней) |
| Clustered (SSBO) | 64 суммарно: point + spot + extra-пул | `max_clustered_lights = 64`, `max_clustered_spots = 32` | Extra-пул без теней; обычные point/spot — до 2+2 теневых |
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
(`scene/uniforms.zig`, едут в `fs_view`, view-глобальный блок сплита legacy
`fs_params`). `ground_color` по-прежнему едет в слоте
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

Value-тип без имён/кучи/теней: `LightRig` держит фиксированный массив + счётчик, сцена — индексные add/remove/get/count. Свет едет в едином SSBO (`FramePack.clustered_lights`, до 64 суммарно с point+spot), тайлинг — в `scene/clustered_lights.zig`, GPU-билд читает staged `FramePack` (лаг 1 кадр). Extra-пул теней не имеет; обычные point/spot — до 2+2 теневых.

### Солнце и цветовая температура (`lights/sun.zig`)

```zig
pub fn resolveSunDirection(directional_light: ?*const DirectionalLight, hemi: HemisphericLight) Vec3;
pub fn resolveSunColor(directional_light: ?*const DirectionalLight, hemi: HemisphericLight) Color3;
pub fn resolveSunIntensity(directional_light: ?*const DirectionalLight, hemi: HemisphericLight) f32;
pub fn sunDirectionFromAngles(azimuth_rad: f32, elevation_rad: f32) Vec3;
pub fn colorTemperatureToRgb(kelvin: f32) Color3;
```

Активный directional перекрывает hemispheric-фолбэк; выключенный считается отсутствующим. `sunDirectionFromAngles`: elevation от горизонта (0 = горизонт, π/2 = зенит), azimuth вокруг +Y (0 = +Z, π/2 = +X); вырождение возвращает `Vec3.up`. `colorTemperatureToRgb` — аппроксимация Планкиана Таннера Хелланда, вход клампится к [1000, 12000] К. Выход формулы — display-referred (sRGB-подобные) значения; `setSunColorTemperature` кладёт их в линейный пайплайн как есть (артистический компромисс, не строгая колориметрия).

## Потоки и владение

Источники — plain-структуры и изменяются **только на game-потоке** под update-vs-prepare exclusion: `create*/add*` пишут `ArrayList`/счётчики, поля мутируются свободно, но не во время `begin*`/render. `updateLights` (внутри `Scene.update`) пакует `FramePack` и публикует его через mailbox; context/render читает только staged snapshot. Именованные источники живут в `LightRig` (glTF-имена с `owns_name` освобождает сцена). Clustered-пул — значения в фиксированном массиве рига, без кучи.

## Ошибки и краевые случаи

| Ситуация | Поведение |
|---|---|
| 5-й directional / 3-й area / 65-й clustered | Жёсткая ошибка `TooMany*`, без замены |
| Выключенный directional | Резолверы солнца падают на hemispheric |
| Нулевой `direction` у directional | `resolveSunDirection` падает на hemispheric (защита `lengthSq > 1e-12`) |
| Вырожденный area-прямоугольник | Ноль излучения, не NaN |
| `radius <= 0` у clustered | Пакуется как отсутствующий (тайл-билд пропускает) |
| `cast_shadows` при выключенных тенях сцены | Теневые слоты не генерируются (`shadows_enabled` гейт) |
| Point/spot вне top-k | Legacy/probe-слоты их не содержат; main рендерит все включённые через единый SSBO (до 64), но тени выбираются только из top-k |
| >64 включённых локальных источников | Лишние молча не попадают в SSBO (`packUnifiedClustered` обрезает по 64); лимиты `add*` при этом не нарушаются |

## Производительность

- Top-k + гистерезис (`light_selection.Hysteresis`) работают в legacy/probe-слоях и при выборе теневых кастеров; main-картинку рендерит единый SSBO, поэтому яркость источника не зависит от `hysteresis_enabled`.
- Uniform-массивы legacy-слоёв фиксированы (4 point + 2 spot + fills + area); локальный свет в main едет через storage buffer и тайловый список.
- Clustered-forward: до 64 локальных источников (point+spot+extra) с 2D screen-tile отбраковкой 64px; при пустом списке шейдерный цикл закрыт.
- Тени — CSM солнца, до 2 point-атласов и 2 spot-страниц; extra clustered-пул и area-свет теней не имеют. Параметры теней point/spot (`*_shadow_params`) ключуются **индексом legacy-слота света**, а матрицы — рангом теневого слота; clustered-путь передаёт индекс через `ClusterLight.spot_params.w`.

## Смотрите также

- `./material.md` — PBR-взаимодействие со светом, IBL
- `./loader.md` — `KHR_lights_punctual`, диапазоны и конусы по умолчанию
- `./scene.md` — `LightRig`, владение источниками
- `./render-pipeline.md` — паковка uniforms, теневые проходы
- `./passes.md` — shadow pass, тайлы point-атласа
