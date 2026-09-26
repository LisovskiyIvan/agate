# Слои сцены

> Путь: `src/agate/scene/decal_layer.zig`, `gui3d_layer.zig`, `highlight_layer.zig`, `nav_layer.zig`, `particle_layer.zig`, `physics_layer.zig`, `probe_layer.zig`, `sky_layer.zig`, `trail_layer.zig`, `clustered_lights.zig`, `light_selection.zig`, `light_rig.zig`, `animation_runtime.zig` · Импорт: `agate.HighlightLayer`, `agate.HighlightEntry`, `agate.HighlightOptions`, `agate.Ui3dPanel`, `agate.ReflectionProbe` и др. (`root.zig`/`scene.zig`) · Потоки: симуляция — game; capture/build — game, latch/upload/draw — context.

## Что это

Слои — независимые подсистемы контента, подвешенные на `Scene`: каждый владеет своими объектами, тикает на game-стороне и отдаёт staged-представление в кадр. Инварианты общие: удаление GPU-ресурсов — только через `GpuRetireQueue` (эпохи context); draw-пути читают замороженное, а не live; лимиты слоёв — fail-closed (ошибка создания вместо частичного состояния).

## Быстрый старт

```zig
// Декали (ленивый менеджер):
const dm = scene.decals.getOrCreate(&scene, 64);
dm.spawn(...); // детали — mesh/decal.zig

// Подсветка:
try scene.highlights.add(mesh, .{ .color = .{ 0, 1, 0, 1 }, .intensity = 0.8 });

// 3D-панель:
const idx = try scene.gui3d.add(allocator, "panel", Vec3.new(0, 2, -3), .{
    .width = 2.0, .height = 1.2, .canvas_width = 1024, .canvas_height = 512,
});
scene.gui3d.markDirty(idx);

// Зонд отражений:
const probe = try scene.probes.add(Vec3.new(0, 2, 0), .{});
scene.probes.markDirty(probe);

// Свет через риг:
scene.lights.setSunAngles(az, el);
try scene.lights.createPointLight(allocator, "lamp", .{ .position = p, .intensity = 10 });

// Навигация / трейлы / небо:
const nav = try scene.nav.createMeshGrid(allocator, -10, 10, -10, 10, 0, 16, 16, &.{});
const agent = try scene.nav.createAgent(allocator, nav, start);
const trail = try scene.trails.create(&scene, allocator, "trail", .{...});
try scene.sky.createDefault(allocator, .{...});
```

## API

### Декали (`decal_layer.zig`)

```zig
pub const DecalLayer = struct {
    pub fn deinit(self: *DecalLayer) void
    pub fn getOrCreate(self: *DecalLayer, scene: anytype, max_decals: usize) *DecalManager
    pub fn update(self: *DecalLayer, dt: f32) void
};
```

Тонкий хостинг: сам `DecalManager` живёт в `mesh/decal.zig`, слою принадлежит опциональный экземпляр. Создание ленивое (`max_decals` — размер пула), `update` тикает распад/анимацию. Назначение — проецированные детали (пулевые отверстия, пятна); в очередях декали идут прозрачным проходом без записи глубины. Лимиты: размер пула фиксируется при создании.

### 3D-GUI (`gui3d_layer.zig`)

```zig
pub const max_panels = 4;
pub const max_captures_per_frame = 1;
pub const max_canvas_dimension: u32 = 2048;
pub const initial_cap_v: usize = 4096;
pub const initial_cap_i: usize = 6144;
pub const Ui3dFaceMode = enum { ... };
pub const Ui3dPanelOptions = struct { width, height, canvas_width, canvas_height, yaw_deg, face_mode, enabled, ... };
pub const Ui3dTarget = struct { pub fn deinit(self: *Ui3dTarget) void };
pub const PanelGpu = struct { ... };
pub const Ui3dPanel = struct {
    pub fn markDirty(self: *Ui3dPanel) void
    pub fn injectPointer(self: *Ui3dPanel, x: f32, y: f32, pressed: bool) void
    pub fn injectRelease(self: *Ui3dPanel) void
};
pub fn targetBytes(canvas_width: u32, canvas_height: u32) usize
pub fn panelRight(yaw_deg: f32) Vec3
pub fn panelNormal(yaw_deg: f32) Vec3
pub fn panelUp() Vec3
pub fn panelCorners(panel: *const Ui3dPanel) [4]Vec3
pub fn panelModel(panel: *const Ui3dPanel) Mat4
pub const PanelUvHit = struct { ... };
pub fn intersectRayPanel(ray: Ray, panel: *const Ui3dPanel) ?PanelUvHit
pub fn canvasPixelCoords(u: f32, v: f32, canvas_width: u32, canvas_height: u32) struct { x: f32, y: f32 }
pub const Ui3dPickHit = struct { ... };
pub const PanelVertex = struct { ... };
pub const Gui3dLayer = struct {
    pub fn add(self: *Gui3dLayer, allocator: std.mem.Allocator, name: []const u8, position: Vec3, options: Ui3dPanelOptions) error{ TooManyUi3dPanels, InvalidUi3dPanelSize, OutOfMemory }!usize
    pub fn remove(self: *Gui3dLayer, allocator: std.mem.Allocator, retire_queue: anytype, index: usize) void
    pub fn get(self: *Gui3dLayer, index: usize) ?*Ui3dPanel
    pub fn getByName(self: *Gui3dLayer, name: []const u8) ?*Ui3dPanel
    pub fn panelCount(self: *const Gui3dLayer) usize
    pub fn markDirty(self: *Gui3dLayer, index: usize) void
    pub fn markAllDirty(self: *Gui3dLayer) void
    pub fn nextDirtyIndex(self: *const Gui3dLayer) ?usize
    pub fn dirtyCount(self: *const Gui3dLayer) usize
    pub fn notifyCaptured(self: *Gui3dLayer, index: usize) void
    pub fn drawCount(self: *const Gui3dLayer) usize
    pub fn pick(self: *const Gui3dLayer, ray: Ray) ?Ui3dPickHit
    pub fn censusBytes(self: *const Gui3dLayer) usize
    pub fn ensureGpu(self: *Gui3dLayer, allocator: std.mem.Allocator, index: usize) bool
    pub fn capturePanel(self: *Gui3dLayer, allocator: std.mem.Allocator, retire_queue: anytype, index: usize) bool
    pub fn drawPanels(self: *Gui3dLayer, allocator: std.mem.Allocator, view_proj: Mat4, samples: i32, stats: *SceneStats) void
    pub fn deinit(self: *Gui3dLayer, allocator: std.mem.Allocator) void
};
```

Назначение — UI-канвасы, спроецированные в 3D (панели с yaw-ориентацией, UV-пикингом и вводом указателя). Лимиты: ≤4 панелей, размер мира ≤1024, канвас ≤2048, захват — не более 1 панели за кадр (PACED: `nextDirtyIndex` → `capturePanel` → `notifyCaptured`). Удаление — через retire-очередь (уничтожение на context-flush), со сдвигом индексов (индексы не кэшировать). `drawPanels` рисует из GPU-таргетов; `censusBytes` — учёт памяти (`targetBytes` на панель).

### Подсветка (`highlight_layer.zig`)

```zig
pub const max_highlights: usize = 8;
pub const HIGHLIGHT_COLOR_DEFAULT: [4]f32 = .{ 1.0, 1.0, 1.0, 1.0 };
pub const HIGHLIGHT_BLUR_DEFAULT: f32 = 4.0;
pub const HIGHLIGHT_INTENSITY_DEFAULT: f32 = 0.5;
pub const HighlightOptions = struct { color, blur, intensity, ... };
pub fn validateHighlightOptions(options: HighlightOptions) !void
pub const HighlightEntry = struct { mesh: *Mesh, options: HighlightOptions, ... };
pub const HighlightLayer = struct {
    pub fn add(self: *HighlightLayer, mesh: *Mesh, options: HighlightOptions) error{ TooManyHighlights, InvalidHighlightOptions }!usize
    pub fn remove(self: *HighlightLayer, index: usize) void
    pub fn removeForMesh(self: *HighlightLayer, mesh: *Mesh) void
    pub fn clear(self: *HighlightLayer) void
    pub fn get(self: *HighlightLayer, index: usize) ?*HighlightEntry
    pub fn highlightCount(self: *const HighlightLayer) usize
};
```

Назначение — выбор/подсветка до 8 мешей (PASS 2.85 mask/blur в `PostFXStack.renderChain`). Скиннед-меши на capture пропускаются (v1, нет skin store). `removeForMesh` — обязательная уборка при `destroyMesh` (иначе висячий `*Mesh`). Ошибки: переполнение и невалидные опции — создание запрещено, а не урезано.

### Навигация (`nav_layer.zig`)

```zig
pub const NavLayer = struct {
    pub fn deinit(self: *NavLayer, allocator: std.mem.Allocator) void
    pub fn createMeshFromTriangles(self: *NavLayer, allocator: std.mem.Allocator, positions: []const [3]f32, indices: []const u32, max_slope_rad: f32) !*NavMesh
    pub fn createMeshGrid(self: *NavLayer, allocator: std.mem.Allocator, min_x: f32, max_x: f32, min_z: f32, max_z: f32, elevation_y: f32, subdiv_x: usize, subdiv_z: usize, obstacles: []const BoundingBox) !*NavAgent
    pub fn createAgent(self: *NavLayer, allocator: std.mem.Allocator, nav_mesh: *const NavMesh, start_pos: Vec3) !*NavAgent
    pub fn updateAgents(self: *NavLayer, dt: f32) void
};
```

Назначение — статичные навмеши + агенты. `createMeshGrid` принимает AABB-препятствия; `updateAgents` тикает всех агентов (явный вызов из `Scene.update` не входит — приложение ведёт свой time base реальными секундами, см. `./frame-pipeline.md`). Лимиты: только память; владение — слой (destroy через аллокатор в `deinit`).

### Частицы (`particle_layer.zig`)

```zig
pub const ParticleDraw = passes.ParticlePass.ParticleDraw;
pub const ParticleLayer = struct {
    pub fn init() ParticleLayer
    pub fn deinit(self: *ParticleLayer, allocator: std.mem.Allocator) void
    pub fn create(self: *ParticleLayer, allocator: std.mem.Allocator, name: []const u8, capacity: usize) !*ParticleSystem
    pub fn update(self: *ParticleLayer, dt: f32) particles.UpdateError!void
    pub fn captureFrame(self: *ParticleLayer, allocator: std.mem.Allocator) void
    pub fn buildCapture(self: *ParticleLayer, allocator: std.mem.Allocator, seq: u64) void
    pub fn latchFrame(self: *ParticleLayer, allocator: std.mem.Allocator) void
    pub fn stageIntoSlot(self: *ParticleLayer, allocator: std.mem.Allocator, out: *std.ArrayListUnmanaged(ParticleDraw)) void
    pub fn latchSlotFrame(self: *ParticleLayer, allocator: std.mem.Allocator, draws: []const ParticleDraw) void
    pub fn renderPrepared(self: *ParticleLayer, camera: Camera, aspect: f32, samples: i32, stats: *SceneStats) void
    pub fn render(self: *ParticleLayer, camera: Camera, aspect: f32, samples: i32, stats: *SceneStats) void
};
```

Назначение — CPU/GPU/compute системы частиц с двойным представлением: `build_frame` (game) → `frame` (render-owned). `update` — только симуляция и dirty-флаги (sg-заливки позже на render-стороне). Staged-путь: `buildCapture(seq)` → `stageIntoSlot` (заморозка в слот кадра) → `latchSlotFrame` (потребление слотовой копии, не shared store). `reserve-once`: новее `latched_seq` — иначе исторический live-capture. Ноль активных систем — когерентный пустой кадр; OOM — fail-closed в пустой. MSAA-twin пасса — ленивый (compute всегда через 1x). `deinit` — только context (уничтожает sg-буферы инлайн).

### Физика (`physics_layer.zig`)

```zig
pub const PhysicsIntegration = struct {
    pub fn deinit(self: *PhysicsIntegration, allocator: std.mem.Allocator) void
    pub fn enable(self: *PhysicsIntegration, allocator: std.mem.Allocator, gravity: ?Vec3) *PhysicsWorld
    pub fn getWorld(self: *PhysicsIntegration) ?*PhysicsWorld
    pub fn step(self: *PhysicsIntegration, dt: f32) void
    pub fn captureDebug(self: *PhysicsIntegration, allocator: std.mem.Allocator) void
    pub fn buildDebug(self: *PhysicsIntegration, allocator: std.mem.Allocator, seq: u64) void
    pub fn latchDebug(self: *PhysicsIntegration, allocator: std.mem.Allocator) void
    pub fn stageIntoSlot(self: *PhysicsIntegration, allocator: std.mem.Allocator, lines: ..., visible: ...) void
    pub fn latchSlotDebug(self: *PhysicsIntegration, allocator: std.mem.Allocator, lines: ..., visible: ...) void
    pub fn uploadDebug(self: *PhysicsIntegration, allocator: std.mem.Allocator, samples: i32) void
    pub fn renderDebugPrepared(self: *PhysicsIntegration, view_proj: Mat4, samples: i32, stats: *SceneStats) void
    pub fn renderDebug(self: *PhysicsIntegration, allocator: std.mem.Allocator, view_proj: Mat4, samples: i32, stats: *SceneStats) void
};
```

Назначение — опциональный `PhysicsWorld` (ленивый `enable`, `null` gravity = дефолт) + отладочные линии. Тот же freeze/latch-рисунок, что у частиц: `buildDebug(seq)` → `stageIntoSlot` → `latchSlotDebug`. `uploadDebug` — один `updateBuffer` за prepare; `renderDebugPrepared` — без загрузок. `step` пишет `pending_physics_ms` для staged статистики.

### Зонды отражений (`probe_layer.zig`)

```zig
pub const max_probes = 4;
pub const face_resolution: i32 = 128;
pub const max_mips: u32 = 8;
pub const max_lod: f32 = 7.0;
pub const capture_near: f32 = 0.1;
pub const Face = enum(u3) { pos_x, neg_x, pos_y, neg_y, pos_z, neg_z };
pub fn faceDir(face: Face) Vec3
pub fn faceUp(face: Face) Vec3
pub fn faceView(face: Face, eye: Vec3) Mat4
pub fn faceViewProj(face: Face, eye: Vec3, near: f32, far: f32) Mat4
pub fn captureFar(radius: f32) f32
pub fn mipSize(mip: u32) i32
pub const ReflectionProbeOptions = struct { radius, ... };
pub const ProbeGpu = struct { pub fn deinit(self: *ProbeGpu) void };
pub fn targetBytes() usize
pub const ReflectionProbe = struct { position, radius, dirty, captured, ... };
pub const ProbeFrameEntry = struct { ... };
pub const FramePack = struct { ... };
pub const SelectedProbe = struct { ... };
pub fn selectProbe(entries: []const ProbeFrameEntry, pos: Vec3) ?SelectedProbe
pub const ProbeLayer = struct {
    pub fn add(self: *ProbeLayer, position: Vec3, options: ReflectionProbeOptions) error{TooManyReflectionProbes}!usize
    pub fn remove(self: *ProbeLayer, allocator: std.mem.Allocator, retire_queue: anytype, index: usize) void
    pub fn markDirty(self: *ProbeLayer, index: usize) void
    pub fn markAllDirty(self: *ProbeLayer) void
    pub fn nextDirtyIndex(self: *const ProbeLayer) ?usize
    pub fn dirtyCount(self: *const ProbeLayer) usize
    pub fn notifyCaptured(self: *ProbeLayer, index: usize) void
    pub fn packFrame(self: *const ProbeLayer) FramePack
    pub fn ensureGpu(self: *ProbeLayer, index: usize) bool
    pub fn deinit(self: *ProbeLayer) void
};
```

Назначение — до 4 cubemap-зондов IBL (128px грань, 8 мипов, near 0.1, far от радиуса). Захват пейсится (`captureDirtyProbes` — грязные по одному за кадр, грани + prefilter; см. `./render-pipeline.md`). Выборка в шейдере — ближайший по `selectProbe`. Удаление — через retire-очередь (пустой pre-capture таргет тоже ретайрится — единая дисциплина). Память — `targetBytes()` на зонд.

### Небо (`sky_layer.zig`)

```zig
pub const SkyboxLayer = struct {
    pub fn init() SkyboxLayer
    pub fn deinit(self: *SkyboxLayer) void
    pub fn setSkybox(self: *SkyboxLayer, cube: CubeTexture) void
    pub fn createDefault(self: *SkyboxLayer, allocator: std.mem.Allocator, config: SkyboxOptions) !void
    pub fn renderPrepared(self: *SkyboxLayer, ...) void
    pub fn render(self: *SkyboxLayer, camera: Camera, aspect: f32, fallback: CubeTexture, samples: i32, stats: *SceneStats) void
};
```

Назначение — skybox + IBL-вход (`sky_texture`, `ibl_intensity` — в снимке кадра). `createDefault` — процедурный куб по `SkyboxOptions`; `setSkybox` — свой куб. `render` с fallback-кубом (headless/до первой загрузки); `renderPrepared` — из заморожённого. Пустое небо — пропуск с нулевой стоимостью.

### Трейлы (`trail_layer.zig`)

```zig
pub const TrailLayer = struct {
    pub fn deinit(self: *TrailLayer, allocator: std.mem.Allocator) void
    pub fn create(self: *TrailLayer, scene: anytype, allocator: std.mem.Allocator, name: []const u8, options: TrailOptions) !*TrailMesh
    pub fn update(self: *TrailLayer, dt: f32, cam_pos: Vec3) void
};
```

Назначение — ленточные трейлы за движущимся якорем (`mesh/trail.zig: TrailMesh`, `TrailOptions`: длина, ширина, затухание). `create` требует `*Scene` (создание мешей), поэтому `scene: anytype`. `update` — явный, с позицией камеры (culling сегментов по дистанции; dt — реальные секунды, не нормализованный тик). Геометрия трейлов едет теми же очередями/пакетами (`trail_uploads` в слоте кадра).

### Кластерный свет (`clustered_lights.zig`)

```zig
pub const MAX_VIEW_SLOTS: usize = snapshot.MAX_CAMERAS;
pub const tile_size_px: u32 = 64;
pub const ClusteredLightStage = struct { ... };
pub const ClusterLightGpu = struct { ... };
pub const ClusterTileGpu = struct { ... };
pub const TileGrid = struct { tiles_x, tiles_y, ... };
pub fn tilesForViewport(screen_w: i32, screen_h: i32) TileGrid
pub const ViewRect = struct { ... };
pub fn tileNdcRect(tx: u32, ty: u32, view: ViewRect, win_w: i32, win_h: i32) [4]f32
pub fn buildTileLists(pos_range: []const [4]f32, color_int: []const [4]f32, count: usize, view_proj: Mat4, tiles_x: u32, tiles_y: u32, view: ViewRect, win_w: i32, win_h: i32, headers: [][2]u32, indices: []u32) usize
pub const ClusterBindingViews = struct { ... };
pub const ClusteredViewSlot = struct { ... };
pub const ClusteredGpuCache = struct {
    pub fn clampSlot(view_slot: usize) usize
    pub fn isLive(self: *const ClusteredGpuCache, view_slot: usize) bool
    pub fn rebuildCpu(...) void
    pub fn rebuildCpuForSlot(...) void
    pub fn ensureDummyViews(self: *ClusteredGpuCache) void
    pub fn upload(self: *ClusteredGpuCache, allocator: std.mem.Allocator, retire_queue: anytype, view_slot: usize) bool
    pub fn bindingViews(self: *const ClusteredGpuCache) ClusterBindingViews
    pub fn bindingViewsForSlot(self: *const ClusteredGpuCache, view_slot: usize) ClusterBindingViews
    pub fn retireBuffers(self: *ClusteredGpuCache, allocator: std.mem.Allocator, retire_queue: anytype) void
    pub fn deinit(self: *ClusteredGpuCache, allocator: std.mem.Allocator) void
};
```

Назначение — сотни точечных источников для forward-рендера: экран бьётся на тайлы 64px, каждый тайл хранит список света (`headers` + `indices`), свет — в storage-буферах, в юниформах только дескриптор (`clustered_params/viewport`). Каждый вид перестраивает тайлы и заливает только свой слот (`clampSlot`, ≤ `MAX_CAMERAS`); не залившийся вид биндит общий dummy. Пустой пул — шейдерный цикл закрыт с нулевой стоимостью. Сложность: O(тайлы × свет) CPU + одна заливка на вид.

### Выбор света (`light_selection.zig`)

```zig
pub fn scorePoint(light: *const PointLight, camera_pos: Vec3) f32
pub fn scoreSpot(light: *const SpotLight, camera_pos: Vec3) f32
pub fn selectPoint(lights_in: []const *PointLight, camera_pos: Vec3, out: []*PointLight) usize
pub fn selectSpot(lights_in: []const *SpotLight, camera_pos: Vec3, out: []*SpotLight) usize
pub fn selectTopKHysteresis(...) usize
pub fn Hysteresis(comptime T: type, comptime slots: usize, comptime scoreFn: fn (*const T, Vec3) f32) type
// Hysteresis(T).Self:
pub const Packed = struct { light: *T, slot: usize, factor: f32 };
pub fn reset(self: *Self) void
pub fn update(self: *Self, lights: []const *T, eye: Vec3, dt: f32, incumbency_bonus: f32, fade_time: f32, out: []Packed) usize
```

Назначение — упаковать неограниченный свет в 4 point + 2 spot uniform-слота без мерцания: score (яркость/дистанция/угол) + top-K + гистерезис (бонус действующим, fade вытесняемых через `factor`). Directional/area/clustered идут как есть (без отбора). Детерминировано при равных score (порядок создания).

### Риг света (`light_rig.zig`)

```zig
pub const LightRig = struct {
    pub fn init(name: []const u8, options: lights.HemisphericLightOptions) LightRig
    pub fn setHemispheric(self: *LightRig, name: []const u8, options: lights.HemisphericLightOptions) HemisphericLight
    pub fn createPointLight(self: *LightRig, allocator: std.mem.Allocator, name: []const u8, options: PointLightOptions) !*PointLight
    pub fn createSpotLight(self: *LightRig, allocator: std.mem.Allocator, name: []const u8, options: SpotLightOptions) !*SpotLight
    pub fn createDirectionalLight(self: *LightRig, allocator: std.mem.Allocator, name: []const u8, options: DirectionalLightOptions) !*DirectionalLight
    pub fn addDirectionalLight(self: *LightRig, allocator: std.mem.Allocator, name: []const u8, options: DirectionalLightOptions) !*DirectionalLight
    pub fn directionalCount(self: *const LightRig) usize
    pub fn directionalAt(self: *const LightRig, index: usize) ?*DirectionalLight
    pub fn addAreaLight(self: *LightRig, allocator: std.mem.Allocator, name: []const u8, options: AreaLightOptions) !*AreaLight
    pub fn removeAreaLight(self: *LightRig, allocator: std.mem.Allocator, index: usize) void
    pub fn getAreaLight(self: *const LightRig, index: usize) ?*AreaLight
    pub fn areaLightCount(self: *const LightRig) usize
    pub fn addClusteredPointLight(self: *LightRig, position: Vec3, options: ClusteredPointLightOptions) error{TooManyClusteredLights}!usize
    pub fn removeClusteredPointLight(self: *LightRig, index: usize) void
    pub fn getClusteredPointLight(self: *LightRig, index: usize) ?*ClusteredPointLight
    pub fn clusteredPointLightCount(self: *const LightRig) usize
    pub fn sunDirection(self: *const LightRig) Vec3
    pub fn sunColor(self: *const LightRig) Color3
    pub fn sunIntensity(self: *const LightRig) f32
    pub fn setSunAngles(self: *LightRig, azimuth_rad: f32, elevation_rad: f32) void
    pub fn setSunColorTemperature(self: *LightRig, kelvin: f32) void
    pub fn deinit(self: *LightRig, allocator: std.mem.Allocator) void
    pub const point_slots = 4;
    pub const spot_slots = 2;
    pub const point_shadow_slots = 2;
    pub const FramePack = struct { counts, directional_dir/color_int: [4][4]f32, point_pos_range/point_color_int: [4][4]f32, spot_pos_range/spot_dir_inner/spot_color_outer/spot_intensity: [2][4]f32, spot_view_proj: [2]Mat4, spot_shadow_params, spot_shadows, num_spot_shadows, point_view_proj: [12]Mat4, point_shadow_params, point_shadows, num_point_shadows, ... };
    pub fn packFrame(self: *LightRig, eye: Vec3, shadows_enabled: bool, dt: f32) FramePack
};
```

Назначение — владение всем светом + сборка `FramePack` за кадр: directionals verbatim (слот 0 — первичное солнце, 1..3 — fills), area/clustered verbatim, point/spot — через отбор с гистерезисом (опционально) с fade-масштабированием интенсивности. Теневые слоты: ≤2 спотов + ≤2 поинтов (6 граней на слот в `point_view_proj[12]`). Солнце: `setSunAngles` / `setSunColorTemperature` (кельвины → RGB), резолв preview — `resolveSunDirection/Color/Intensity`, `sunDirectionFromAngles`. Лимиты: uniform-слоты фиксированы; сверх — только через clustered-пул (`max_clustered_lights`).

### Анимация (`animation_runtime.zig`)

```zig
pub const min_skeletons_for_workers: usize = 4;
pub fn updateAnimations(animation_groups: []const *AnimationGroup, skeletons: []const *Skeleton, meshes: []const *Mesh, dt: f32) void
```

Назначение — продвижение skeletal-анимации за кадр: сначала сериально таймлайны групп (`ag.update(dt)` — время, события/колбэки, node-треки), затем блендинг base + additive клипов по скелетам (параллельно через воркеры при ≥4 скелетах, cap 16 клипов на сторону), затем CPU-блендинг morph-target'ов на помеченных мешах (`applyMorphs`, early-out когда чисто). Вызывается до построения очередей в `Scene.render`. Потоки: `collectActive` read-only по группам — безопасен в воркерах после `update`.

## Потоки и владение

| Слой | Game (update/build) | Context (latch/draw) |
|---|---|---|
| Декали, трейлы, нав, небо, highlights | `update`, создание/удаление | чтение staged (очереди, `highlight_items`) |
| Частицы, физика-debug | `update`, `buildCapture`/`buildDebug`, `stageIntoSlot` | `latchSlotFrame`/`latchSlotDebug`, `upload`, `renderPrepared` |
| Зонды, 3D-GUI | `markDirty`, `ensureGpu` (создание), `packFrame` | `captureDirtyProbes`, `capturePanel`, `drawPanels`, prefilter |
| Свет | `packFrame(eye, shadows, dt)` (game-билд) | `FrameContext` из staged пака, никогда live |
| Анимация | целиком game, до очередей | только готовые матрицы костей |

Удаление с GPU-ресурсами (`remove` панелей/зондов, destroy мешей с highlights) — всегда через retire-очередь: в `remove` только штамп эпохи + append (можно с любого потока), уничтожение — на context-flush. `deinit` слоёв с sg-ресурсами — только context.

## Ошибки и краевые случаи

- Лимиты — жёсткие ошибки создания: `TooManyUi3dPanels` (4), `TooManyReflectionProbes` (4), `TooManyHighlights` (8), `TooManyClusteredLights`, `InvalidUi3dPanelSize`, `InvalidHighlightOptions` — вместо тихого урезания.
- `remove`/`get` вне диапазона: no-op / `null` (контракт как `Scene.removeCamera`).
- Ноль систем/зондов/панелей/света: когерентный пустой кадр, шейдерные циклы закрыты нулевой стоимостью (слоты занулены).
- OOM в capture/stage: staged-wins в coherent-empty, stale не всплывает; retained capacity переиспользуется.
- `destroyMesh` обязан чистить референтов (`removeForMesh` у highlights и др. — см. `scene/registry.zig`).
- Registry add/remove в полёте latch — нарушение контракта приложения (commit-guard'ы держат когерентность, но запрещён).
- Скиннед-меши в highlights v1 — пропускаются на capture (нет skin store); compute-частицы всегда через 1x пасс.

## Производительность

- Highlights: ≤8 записей, маска только по primary-вьюпорту (PIP не кормят маску, v1).
- Зонды: ≤1 захвата за кадр (6 граней 128px + 8 мипов prefilter) — грязные очередь; `markAllDirty` после смены неба — растягивается на кадры.
- 3D-GUI: ≤1 capture за кадр; `initial_cap_v/i = 4096/6144` стартовых вершин/индексов на панель, рост — retained.
- Кластеры: тайлы 64px, O(тайлы × свет) CPU + 1 заливка на вид; пустой пул — бесплатно.
- Свет: отбор O(n log k) по score; гистерезис убирает мерцание слотов (fade вместо щелчков).
- Анимация: воркеры от 4 скелетов; morph — только грязные меши.
- Учёт: `censusBytes` (GUI), `targetBytes` (зонды), счётчики draw/пайплайнов в `SceneStats`.

## Смотрите также

- `./frame-pipeline.md` — freeze/latch/commit, слоты, retire-эпохи.
- `./render-pipeline.md` — очереди, инстансинг, тени, пост-цепочка.
- `./lights.md` — типы источников, опции, цветовая температура.
- `./particles.md` — системы частиц, CPU/GPU/compute режимы.
- `./physics.md` — `PhysicsWorld`, отладочная отрисовка.
- `./animation.md` — группы, клипы, скелеты, morph-target'ы.
- `./ai.md` — `NavMesh`/`NavAgent` детально.
- `./material.md` — декали и слои материалов.
- `./ui.md` — `UICanvas`, ввод, 2D-оверлей.
- `./texture.md` — cubemap'ы неба и зондов.
