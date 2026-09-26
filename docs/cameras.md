# Камеры

> Путь: src/agate/camera.zig, src/agate/camera/ · Импорт: agate.Camera, agate.CameraRig, agate.Viewport (root.zig) · Потоки: `handleEvent` на UI-потоке, `update(dt)` на потоке обновления, чтение матриц на context-потоке

## Что это

Пять типов камер под общим фасадом `camera.zig`: орбитальная `ArcRotateCamera`, WASD-камера `FreeCamera`, 6-DOF `FlyCamera`, догоняющая `FollowCamera` (следит за мешем), наблюдатель `TargetCamera` (сглаженное движение к цели). Полиморфная обёртка `Camera` — union пяти листьев с единым диспетчером. `Viewport` — нормализованный прямоугольник [0..1] для сплит-скринов и PIP. `CameraRig` — мультикамерные пресеты: dual/quad, PIP, стерео side-by-side/over-under для VR, произвольный custom-набор до 8 слотов.

## Быстрый старт

```zig
const agate = @import("agate");

// Орбитальная камера вокруг цели.
var orbit = agate.ArcRotateCamera.init("orbit", .{
    .target = .{ .x = 0, .y = 1, .z = 0 },
    .radius = 8.0,
    .beta = std.math.pi / 3.0,
});
var cam: agate.Camera = .{ .arc_rotate = orbit };
scene.addCamera(cam); // активная камера сцены

// Кадр: событие → update → чтение матриц рендером.
// handleEvent(&cam, ev); cam.update(dt);
// const vp = cam.getViewProjection(aspect);

// Догоняющая камера за мешем.
var chase = agate.FollowCamera.init("chase", .{});
chase.setTarget(hero_mesh);
var chase_cam: agate.Camera = .{ .follow = chase };

// Сплит-скрин ригом.
var rig = agate.CameraRig.initPreset(cam, .dual_horizontal);
```

## API

### Общее: `Viewport` и `Camera` (`camera/viewport.zig`, `camera/union.zig`)

```zig
pub const Viewport = struct {
    x: f32 = 0.0, y: f32 = 0.0, width: f32 = 1.0, height: f32 = 1.0,
    pub fn toPixelRect(self: Viewport, screen_w: i32, screen_h: i32) PixelRect;
    pub const PixelRect = struct { x/y/width/height: i32, pub fn aspect(self: PixelRect) f32; };
};
pub const Camera = union(enum) {
    arc_rotate: ArcRotateCamera, free: FreeCamera, fly: FlyCamera,
    follow: FollowCamera, target: TargetCamera,
    pub fn getName/getPosition/getForward/getRight/getUp(self: Camera) ...;
    pub fn getViewMatrix(self: Camera) Mat4;
    pub fn getProjectionMatrix(self: Camera, aspect: f32) Mat4;
    pub fn getViewProjection(self: Camera, aspect: f32) Mat4;
    pub fn handleEvent(self: *Camera, ev: [*c]const sokol.app.Event) void;
    pub fn update(self: *Camera, dt: f32) void;
    pub fn getNear/getFar/getFovDeg/getCullingMask/getViewport(...) ...;
    pub fn setCullingMask/setViewport/setPosition/setLookAt(...) ...;
};
```

Каждый конкретный тип дублирует тот же набор методов (`init(name, options)`, `handleEvent`, `update`, `get*`), union лишь диспетчеризует. `setLookAt(pos, target, up)` перевычисляет ориентацию; `culling_mask` пересекается с `layer_mask` мешей (см. `./visibility.md`).

### `ArcRotateCamera` — орбита (`camera/arc_rotate.zig`)

```zig
pub const ArcRotateCameraOptions = struct {
    alpha: f32 = 0.0, beta: f32 = std.math.pi / 3.0, radius: f32 = 5.0,
    target: Vec3 = Vec3.zero, fov_deg: f32 = 60, near: f32 = 0.1, far: f32 = 100,
    inertia: f32 = 0.0, culling_mask: u32 = 0xFFFFFFFF, viewport: Viewport = .{},
};
```

Управление: drag — вращение (`angular_sensitivity = 0.006`), колесо — зум (`wheel_precision = 0.5`). Лимиты: `lower/upper_radius_limit` (0.5/100), `lower/upper_beta_limit` (0.01/π−0.01, защита полюсов). Инерция Babylon-паритета: `inertia` 0.0 = мгновенно, ~0.9 = плавное затухание через `inertial_alpha/beta/radius_offset`. Позиция вычисляется из сферических координат вокруг `target`.

### `FreeCamera` — WASD + drag (`camera/free.zig`)

```zig
pub const FreeCameraOptions = struct {
    position: Vec3 = Vec3.zero, rotation: Vec3 = Vec3.zero,
    fov_deg: f32 = 60, near/far: f32, speed: f32 = 6.0,
    angular_sensitivity: f32 = 0.25, // градусов на пиксель
    inertia: f32 = 0.0, culling_mask, viewport, ...
};
```

Флаги движения `move_forward/back/left/right/up/down` + drag-поворот; `update(dt)` применяет скорость и инерцию (`inertial_rotation_x/y`). Стандартная камера редактора и glTF-импорта по умолчанию.

### `FlyCamera` — 6-DOF (`camera/fly.zig`)

Полётная камера с креном (Q/E, `roll_speed_deg = 90`) и бустом (`boost_multiplier = 4`): forward/right/up, свободная ориентация без полюсных лимитов. Опции — позиция, скорость, чувствительность, инерция. Используется для космических/полётных режимов, где `FreeCamera` с её плоским горизонтом не подходит.

### `FollowCamera` — преследование (`camera/follow.zig`)

```zig
pub const FollowCameraOptions = struct {
    target_position: Vec3 = Vec3.zero,
    radius: f32 = 5.0,          // дистанция до цели
    height_offset: f32 = 2.0,   // высота над целью
    rotation_offset_deg: f32 = 0.0,
    fov_deg: f32 = 60, near/far: f32,
    lerp_speed: f32 = 8.0,      // 0 = мгновенный snap
    culling_mask, viewport, ...
};
pub fn setTarget(self: *FollowCamera, mesh: ?*Mesh) void; // цель-меш (поле target_mesh)
```

Держит смещение от целевого меша (`*Mesh`, не позиция — следит за движением цели автоматически). `setTarget(null)` отвязывает. Сглаживание — пружиной/лерпом в `update(dt)`.

### `TargetCamera` — наблюдатель (`camera/target.zig`)

```zig
pub fn setTarget(self: *TargetCamera, target: Vec3) void;
pub fn setDesiredPosition(self: *TargetCamera, position: Vec3) void;
pub fn clearGoals(self: *TargetCamera) void;
```

Программная камера кат-сцен (`TargetCameraOptions`: `position/target/up`, `smoothing = 8.0`, 0 = мгновенный snap): фиксированный `lookAt`, движения мыши нет — только `setTarget/setDesiredPosition` + `update(dt)`, тянущий позицию и цель к заданным.

### `CameraRig` — риги (`camera/rig.zig`)

```zig
pub const MAX_RIG_SLOTS: usize = 8;
pub const CameraRigMode = enum { single, dual_horizontal, dual_vertical, quad_view, pip,
    stereoscopic_side_by_side, stereoscopic_over_under, custom };
pub const StereoConvergenceMode = enum { parallel, toe_in };
pub const CameraRigSlot = struct {
    name: []const u8 = "RigSlot", camera: Camera, viewport: Viewport = .{},
    clear_viewport: bool = true, clear_color: ?Color4 = null,
    culling_mask: u32 = 0xFFFFFFFF, enabled: bool = true,
    local_offset: Vec3 = Vec3.zero, // в локальном фрейме мастера: x=право, y=верх, z=вперёд
    look_at_target: ?Vec3 = null,
    sync_transform: bool = true, // тянуть позицию/ориентацию от мастера
};
pub const CameraRig = struct {
    mode: CameraRigMode = .single, master_camera: Camera,
    slots: [MAX_RIG_SLOTS]CameraRigSlot, slot_count: usize = 0,
    ipd: f32 = 0.064, convergence_distance: f32 = 2.0,
    stereo_convergence: StereoConvergenceMode = .parallel,
    pip_viewport: Viewport = .{ .x = 0.72, .y = 0.05, .width = 0.25, .height = 0.25 },
    pub fn init(master: Camera) CameraRig;
    pub fn initPreset(master: Camera, mode: CameraRigMode) CameraRig;
    pub fn setMode(self: *CameraRig, mode: CameraRigMode) void;
    pub fn setupSingle/setupDualHorizontal/setupDualVertical/setupQuadView/setupCadQuadView/setupPip/setupStereo(...) void;
    // setupDualHorizontal(secondary: ?Camera), setupQuadView(tr/bl/br: ?Camera),
    // setupPip(pip_cam: ?Camera, pip_vp: ?Viewport),
    // setupStereo(mode, ipd_opt: ?f32, convergence_opt: ?StereoConvergenceMode, convergence_dist_opt: ?f32),
    // setupCadQuadView(target: Vec3, distance: f32) — ортографический CAD-пресет.
};
```

Пресеты раскладок:

| Режим | Слоты | Назначение |
|---|---|---|
| `single` | 1 на весь экран | Обычный рендер |
| `dual_horizontal/dual_vertical` | 2 рядом / друг над другом | Сплит-скрин, редактор (вид + топ) |
| `quad_view` | 4 квадранта | Сплит на 4 (`setupQuadView(tr, bl, br)`); CAD-вариант — `setupCadQuadView(target, distance)` |
| `pip` | Мастер + инсет (дефолт правый верхний угол) | Картинка-в-картинке, зеркала, прицелы |
| `stereoscopic_side_by_side/over_under` | 2 со сдвигом `ipd` | VR/stereo-дисплеи |

Стерео: `parallel` — параллельные оси (HMD с оптической коллимацией), `toe_in` — схождение на `convergence_distance` (zero-parallax плоскость). Слоты с `sync_transform` следуют за мастером со своим `local_offset`; `look_at_target` заставляет слот смотреть в точку вместо копирования ориентации. `custom` — ручная сборка до 8 слотов.

## Потоки и владение

Камеры — value-типы, живут в сцене (`CameraEntry`) или локально в риге; указателей на кучу не держат (кроме имён-строк). `handleEvent` — только UI-поток (sokol events), `update(dt)` — поток обновления, чтение матриц — render/prepare на context-потоке. `FollowCamera` хранит сырой `*Mesh` цели: удаление меша при активной погоне требует `setTarget(null)` (иначе висячий указатель; сцена свои камеры чистит сама).

## Ошибки и краевые случаи

| Ситуация | Поведение |
|---|---|
| `beta` у полюсов (arc) | Кламп [0.01, π−0.01] |
| `radius` вне лимитов | Кламп [0.5, 100] |
| Высота экрана 0 в `aspect()` | Возвращает 1.0 |
| `FollowCamera` с удалённым мешем | Висячий указатель — отвязать вручную |
| `setMode(.custom)` | Сохраняет текущие слоты, режим помечается custom |
| glTF-камеры | Импортируются как `FreeCamera` (fov/near/far из файла, см. `./loader.md`) |

## Производительность

- `update(dt)` — O(1) на камеру, только тригонометрия; инерция не добавляет аллокаций.
- Мультикамера линейно умножает draw-пакеты: quad_view ≈ 4× Cull/Queue работы; PIP-дефолт (четверть экрана) дешевле за счёт меньшего fill rate, но Cull идёт полностью.
- `culling_mask` на слот/камеру + `layer_mask` мешей отсекают чужую геометрию до очередей — использовать в PIP/quad для чужих слоёв.
- Стерео-режимы дублируют shadow prepare один раз (тени общие), геометрию — дважды.

## Смотрите также

- `./scene.md` — активная камера, `CameraEntry`, владение
- `./visibility.md` — отсечение, `culling_mask` × `layer_mask`
- `./loader.md` — импорт glTF-камер
- `./render-pipeline.md` — мультивьюпортный рендер, PIP/clear semantics
- `./serialization.md` — персист камер (fills/area не персистятся, см. lights)
