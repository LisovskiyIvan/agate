# Частицы

> Путь: src/agate/particles.zig, src/agate/particles/*.zig, src/agate/compute.zig · Импорт: agate.ParticleSystem, agate.SimulationMode, agate.CollisionMode (root.zig) · Потоки: update-поток стейджит байты, context-поток выполняет sg-загрузки/диспатчи (flushGpuUploads)

## Что это

Система частиц с тремя драйверами симуляции на одном `ParticleSystem`: `.cpu` (по умолчанию, бит-в-бит legacy), `.gpu` (безсостоятельное кольцо spawn-слотов, интеграция в вершинном шейдере `particle_gpu`), `.compute` (v1, stateful GPU-интеграция в `shaders/particle_compute.glsl`). `particles.zig` — фасад; листья: `types.zig` (режимы, ошибки, layout), `sampling.zig` (спавн-сэмплер + визуальные хелперы), `system.zig` (владелец `ParticleSystem`), `cpu.zig` (трёхфазный update), `gpu.zig` (slot-ring), `compute_mode.zig` (стейджинг/диспатч), `flow.zig` (flow-поля), `collisions.zig` (CPU-коллизии), `subemitters.zig` (on-death дети). `compute.zig` — тонкая обёртка compute-проходов sokol-gfx (pipeline, storage views, `groupCount`).

Выбор режима — жёсткий контракт, никогда тихий фолбэк: несовместимая комбинация возвращает ошибку `UpdateError`. CPU-стороны (`local_space`, armed flow, armed collisions) отвергают `.gpu`/`.compute` явными ошибками.

## Быстрый старт

```zig
const ps = try scene.createParticleSystem("fire", 1024);
ps.emitter_position = agate.Vec3.new(0, 1, 0);
ps.emit_rate = 120.0;
ps.gravity = agate.Vec3.new(0, -3, 0);
ps.lifetime_min = 0.5;
ps.lifetime_max = 1.0;
ps.start(); // is_emitting = true

// Кадр (update-поток):
try ps.update(dt);
// Старт кадра рендера (context-поток):
ps.flushGpuUploads();
```

GPU-режим со взрывом и саб-эмиттером:

```zig
try ps.setSimulationMode(.gpu); // ошибка ComputeUnsupported/InvalidCapacity только для .compute
ps.burst(64);

// Дочерняя система дыма, стреляет по смерти родителя:
child.lifetime_min = 2.0; // долгие дети
parent.addSubEmitter(.{ .system = child, .probability = 1.0, .count = 2,
    .inherit_velocity = 0.5, .inherit_position = true, .spawn_radius = 0.2 });
```

Flow-поле и коллизии (только CPU):

```zig
// Процедурное поле 2x2 RGBA8: R=dir.x, G=dir.z, B=strength.
try ps.setFlowMap(tex_or_null, &pixels, w, h);
ps.flow_strength = 2.0;
ps.flow_space = .world_xz; // или .local_xz — поле едет с эмиттером

try ps.setCollisionMode(.bounce);
try ps.addSphereCollider(.{ .center = agate.Vec3.zero, .radius = 1.0 });
try ps.setGroundPlane(0.0);
```

## API

### Владелец и жизненный цикл

```zig
pub fn init(allocator: std.mem.Allocator, name: []const u8, capacity: usize) !*ParticleSystem
pub fn deinit(self: *ParticleSystem) void
pub fn start(self: *ParticleSystem) void
pub fn stop(self: *ParticleSystem) void
pub fn reset(self: *ParticleSystem) void
pub fn emitOne(self: *ParticleSystem) void
pub fn burst(self: *ParticleSystem, count: usize) void
pub fn update(self: *ParticleSystem, dt: f32) UpdateError!void
pub fn updateCpu(self: *ParticleSystem, dt: f32) void
pub fn updateGpu(self: *ParticleSystem, dt: f32) UpdateError!void
pub fn updateCompute(self: *ParticleSystem, dt: f32) UpdateError!void
pub fn flushGpuUploads(self: *ParticleSystem) void
pub fn resolveEmitterMatrix(self: *const ParticleSystem) ?Mat4
```

Поля эмиттера: `emitter_position`, `emitter_box_min/max` (спаун-бокс), `emit_rate`, `is_emitting`, `direction_min/max`, `speed_min/max`, `gravity`, `color_start/end`, `size_start/end`, `lifetime_min/max`, `rotation_min/max` (градусы), `angular_velocity_min/max` (град/с), спрайтшит `spritesheet_columns/rows/loops` (1x1 по умолчанию), `blend_mode: ParticleBlendMode (.additive/.alpha_blend)`, `drag` (только GPU/compute), `local_space: bool + emitter_mesh: ?*Mesh`.

Семантика: CPU `emitOne` дропает при полном буфере; GPU/compute кольца перерабатывают старейший слот. `burst` на CPU останавливается на capacity, на GPU — перезаписывает. `reset` переякоряет эпоху (`clock_seconds = 0`), чистит курсоры/high-water, стейджит очистку GPU-состояния для compute. `resolveEmitterMatrix` возвращает `null` в world-space (или без `emitter_mesh`); с `local_space + null mesh` координаты проходят как identity.

### Режимы и матрица фич

```zig
pub const SimulationMode = enum { cpu, gpu, compute };
pub const UpdateError = error{ LocalSpaceNeedsCpu, FlowMapNeedsCpu, CollisionNeedsCpu, OutOfMemory, ComputeUnsupported, InvalidCapacity };
pub const ComputeModeError = error{ ComputeUnsupported, InvalidCapacity };
pub fn setSimulationMode(self: *ParticleSystem, mode: SimulationMode) ComputeModeError!void
pub fn computeAvailable(self: *const ParticleSystem) bool
pub fn computeDispatchCount(self: *const ParticleSystem) u64
pub fn computeGroups(self: *const ParticleSystem) usize
pub fn buildComputeParams(self: *const ParticleSystem) pc_shd.CsParams
pub fn deinitComputeGpuObjects(self: *ParticleSystem) void
pub fn takeGpuBuffersForRetire(self: *ParticleSystem, out: []sg.Buffer) usize
```

| Фича | .cpu | .gpu | .compute |
|---|---|---|---|
| gravity | да (semi-implicit Euler) | да (аналитика) | да (Euler в диспатче) |
| drag (`drag`) | нет (by design) | да (точная экспонента) | да (per-step exp, НЕ аналитика) |
| lifetime / burst / emit_rate | да | да | да (CPU-стейджинг) |
| color/size lerp, спрайтшит, rotation | да | да | да (запекается в слот) |
| flow_map | да | `FlowMapNeedsCpu` | `FlowMapNeedsCpu` |
| sub-emitters | да (on-death) | родитель не стреляет | родитель не стреляет |
| local_space | да | `LocalSpaceNeedsCpu` | `LocalSpaceNeedsCpu` |
| collisions | да | `CollisionNeedsCpu` | `CollisionNeedsCpu` |
| noise / sorting | нет | нет | нет (non-goals) |

Интеграции различаются по построению: CPU — `v += g*dt; p += v*dt`; GPU — точная закрытая форма `p(t) = p0 + v0*s + g*(t−s)/k`, `s = (1−e^(−kt))/k` (при `k→0`: `s = t`, `s2 = t²/2`), закреплена golden-тестами `analyticPosition`. Совпадение CPU↔GPU побитово не гарантируется — только spawn-записи при том же сиде.

Compute v1 non-goals (явно): саб-эмиттеры от compute-смертей, flow, сортировка, коллизии, детерминированный CPU-replay, multi-view/PIP сверх обычного прохода, общий pipeline на системы (у каждой свой, создаётся один раз).

### Layout слотов и чистая математика

```zig
pub const GpuParticleSlot = extern struct { // 80 байт = 5 x vec4, зеркало particle_gpu в particle.glsl
    spawn_pos_time: [4]f32,   // xyz = spawn (world), w = время от эпохи clock_seconds
    velocity_lifetime: [4]f32,
    color_start: [4]f32, color_end: [4]f32,
    size_rotation: [4]f32,    // x = size0, y = size1, z = rot0 (рад), w = angvel (рад/с)
};
pub const ComputeParticleState = extern struct { // 96 байт = 6 x vec4, зеркало CState
    pos_age: [4]f32, vel_life: [4]f32, rot_seed: [4]f32,
    color_start: [4]f32, color_end: [4]f32, size_size: [4]f32,
};
pub const compute_workgroup_size: usize = 64; // = local_size_x в шейдере и compute.default_workgroup_size
pub fn slotAge(now: f32, spawn_time: f32, lifetime: f32) SlotAge // .{ t, alive }
pub const DragSpans = struct { s: f32, s2: f32 };
pub fn analyticDragSpans(drag: f32, time: f32) DragSpans
pub fn analyticPosition(spawn: Vec3, velocity: Vec3, gravity: Vec3, drag: f32, time: f32) Vec3
pub const Particle = struct { position, velocity, size, size_end, color: Color4, color_end: Color4,
    age, lifetime, rotation: f32 = 0, angular_velocity: f32 = 0, sub_depth: u8 = 0 };
pub const ParticleInstanceData = extern struct { pos_size: [4]f32, color: [4]f32,
    uv_offset_scale: [4]f32 = .{0,0,1,1}, rotation_misc: [4]f32 = .{0,0,0,0} };
```

`clock_seconds` — эпоха float32: разрешение деградирует (~0.24 мс после часа сессии), `reset` переякоряет. Compile-time `comptime` проверки пинят размеры (`GpuParticleSlot == 80`, `ComputeParticleState == 96`, `CsParams == 64`).

### Сэмплер и визуальные хелперы

```zig
pub fn sampleSpawn(self: anytype, rnd: std.Random) SpawnSample // порядок PRNG: box xyz, dir xyz, speed, lifetime, rotation, angvel — контракт
pub fn normalizeAngleDeg(angle: f32) f32       // в [0, 360)
pub fn rotationToRadians(angle_deg: f32) f32
pub fn spritesheetFrameCount(columns: u32, rows: u32) u32
pub fn spritesheetFrameForAge(age: f32, lifetime: f32, columns: u32, rows: u32, loops: f32) u32
pub fn spritesheetUvRect(frame: u32, columns: u32, rows: u32) [4]f32 // кадры слева-направо, снизу-вверх
pub fn localToWorld(matrix: Mat4, point: Vec3) Vec3
pub fn worldScaleFactor(matrix: Mat4) f32 // средний масштаб базиса; точен для uniform
```

Нулевые grid-измерения guard'ятся в 1. Кадр при `age == lifetime` заворачивается в 0 (частица там всё равно умирает).

### Саб-эмиттеры

```zig
pub const SubEmitterTrigger = enum { on_death };
pub const SubEmitter = struct { system: *ParticleSystem, trigger: SubEmitterTrigger = .on_death,
    probability: f32 = 1.0, count: u32 = 1, inherit_velocity: f32 = 0.5,
    inherit_position: bool = true, spawn_radius: f32 = 0.0 };
pub const max_sub_emitters: usize = 4;
pub const max_sub_emitter_depth: u8 = 4;
pub const max_sub_emitter_spawns_per_tick: usize = 256;
pub fn addSubEmitter(self: *ParticleSystem, sub: SubEmitter) void // assert при переполнении (Debug)
pub fn subEmitters(self: *const ParticleSystem) []const SubEmitter
```

Срабатывание — серийный проход после компактификации, до заливки инстансов (дети self-эмиттера валидны в тот же тик). Вероятность и джиттер — детерминированный SplitMix-хэш `(seed, tick, death_idx, emitter_idx)`, не общий PRNG: аттач саб-эмиттера не возмущает родительский поток, результат инвариантен к числу воркеров. `child.sub_depth = parent + 1`; смерти на `depth >= 4` не стреляют — циклы A→B→A гаснут. `.gpu`/`.compute` родитель не стреляет никогда (смерти на GPU ненаблюдаемы); `.gpu`/`.compute` ребёнок принимает спавны в кольцо, цепочки на нём останавливаются. Взрывная граница: ≤256 детей за тик на систему.

### Flow-поля

```zig
pub const FlowSpace = enum { world_xz, local_xz };
pub const FlowWrap = enum { repeat, clamp };
pub fn setFlowMap(self: *ParticleSystem, texture: ?Texture, pixels: []const u8, width: u32, height: u32) !void // error.InvalidDimensions
pub fn clearFlowMap(self: *ParticleSystem) void
pub fn sampleFlow(self: *const ParticleSystem, pos: Vec3) Vec3
pub fn flowUvForPosition(space: FlowSpace, pos: Vec3, emitter: Vec3, scale: Vec2, scroll: Vec2) Vec2
pub fn sampleFlowPixels(pixels: []const u8, width: u32, height: u32, uv: Vec2, wrap: FlowWrap) Vec3
```

Кодировка текселя (линейные данные, без sRGB): R = dir.x, G = dir.z в `[0,255]→[−1,1]`, B = сила `[0,1]`; A игнорируется; y всегда 0. Сэмплинг билинейный, угловое соглашение (uv 0/1 — центры угловых текселей); `.repeat` — fract (negative-safe), `.clamp` — растягивает край. Интеграция как ускорение: `v += decoded * tex_strength * flow_strength * dt`. CPU никогда не читает GPU-память: нужен owned `flow_pixels` (w*h*4, один alloc+memcpy); голый `flow_map` без копии — disarmed, молча игнорируется. Настройки: `flow_strength` (0 = disarmed), `flow_scale/scroll`. Валидация до мутации: при `InvalidDimensions` старое поле нетронуто.

### Коллизии

```zig
pub const CollisionMode = enum { none, kill, bounce };
pub const CollisionError = error{ CollisionNeedsCpu, TooManyColliders, InvalidOptions };
pub const ParticleSphereCollider = struct { center: Vec3 = ..., radius: f32 = 0.5, enabled: bool = true };
pub const ParticleBoxCollider = struct { center: Vec3 = ..., half_extents: Vec3 = ..., enabled: bool = true };
pub const ParticlePlaneCollider = struct { point: Vec3 = ..., normal: Vec3 = Vec3.up, enabled: bool = true };
pub const max_sphere_colliders: usize = 8; // root также реэкспортирует max_box_colliders = 8, max_plane_colliders = 4
pub fn addSphereCollider(self, collider: ParticleSphereCollider) CollisionError!void
pub fn addBoxCollider(self, collider: ParticleBoxCollider) CollisionError!void
pub fn addPlaneCollider(self, collider: ParticlePlaneCollider) CollisionError!void
pub fn addMeshAabbCollider(self, mesh_obj: anytype) CollisionError!void
pub fn clearSphereColliders / clearBoxColliders / clearPlaneColliders / clearColliders(self) void
pub fn setCollisionMode(self, mode: CollisionMode) CollisionError!void
pub fn setGroundPlane(self, height: f32) CollisionError!void
pub fn clearGroundPlane(self) void
```

Прямые записи `collision_restitution` (доля нормальной скорости, clamp [0,1]) и `collision_friction` (доля СОХРАНЯЕМОЙ тангенциальной скорости — конвенция softbody: 1 = скользко). Координаты коллайдеров — в stored simulation space (world или emitter-local при `local_space`, как flow). `.kill` компактится как возрастная смерть (может стрелять саб-эмиттерами). Discrete resolve, один проход (сферы → боксы → плоскости → земля), без CCD: быстрые частицы туннелируют. Чистые резолверы (`resolveSphereContact/resolveBoxContact/resolvePlaneContact/resolveGroundContact`, `collideParticle`) — pure, без PRNG, детерминированы. Включение на `.gpu`/`.compute` — `CollisionNeedsCpu` и на сеттерах, и на update (покрывает смену режима после arm'а).

### Compute-бэкенд (`compute.zig`)

```zig
pub const default_workgroup_size: usize = 64;
pub fn supported() bool // sg.queryFeatures().compute; нужен живой sg-контекст
pub fn backendSupportsCompute(backend: sg.Backend) bool // статическая матрица для тестов
pub fn groupCount(items: usize, local_size: usize) usize // ceil; 0 -> 0
pub fn makePipeline(shader: sg.Shader, label: [:0]const u8) sg.Pipeline // .compute = true
pub fn makeStorageView(buffer: sg.Buffer, label: [:0]const u8) sg.View
```

Матрица: Metal macOS/iOS/sim, D3D11, desktop GL 4.3+ (кроме macOS GL 4.1), GLES 3.1+, Web WebGPU — да; WebGL2, iOS GLES3, Vulkan (sokol WIP), DUMMY — нет. Шейдеры — отдельный shdc-вызов со compute-сленгами (glsl430/metal_macos/hlsl5), дефолтный glsl410 не подходит. `setSimulationMode(.compute)`: `InvalidCapacity` при capacity 0; `ComputeUnsupported` при живом контексте без compute или override=false; headless — стейджит, первый контекстный flush латчит поддержку, следующий update вернёт ошибку. Прямая запись поля `simulation_mode` валидацию пропускает (как `.gpu`) — предпочитать сеттер.

## Потоки и владение

Update-фаза (`update*`, `emitOne`, сеттеры) — только CPU-байты и флаги, никогда `sg.*`, безопасна вне контекста (включая off-context `createParticleSystem`: буферы deferred, `instance_buffer_pending`). Prepare-граница (`flushGpuUploads`, вызывается из `Scene.flushPendingGpuUploads` на старте рендера, context-поток): создание буферов/view/pipeline/shader, spawn-загрузки (`sg.updateBuffer` + `upload_meter.record`), compute-диспатч (`sg.beginPass(.{ .compute = true })`, `compute_dispatches +%= 1`). Headless flush — безопасный no-op с сохранением флагов для ретрая; на контексте без compute — латч `compute_known_unsupported`, флаги сбрасываются (без спин-ретраев). Тирадаун: буферы — через `takeGpuBuffersForRetire` (любой поток, sg-free, покрывает все 5 буферов всех режимов) в retire-очередь; view/pipeline/shader — `deinitComputeGpuObjects` строго на context-потоке. CPU-интеграция трёхфазная (parallel integrate → serial compact → parallel fill) через `jobs.parallelFor` с `thread_pool orelse jobs.global`; послотовая независимость даёт бит-идентичность при любом числе воркеров.

## Ошибки и краевые случаи

- `LocalSpaceNeedsCpu / FlowMapNeedsCpu / CollisionNeedsCpu` из `update(Gpu/Compute)` — чинить конфигурацию, режим не мутируется за спиной.
- `ComputeUnsupported` — бэкенд без compute (латч или override); `InvalidCapacity` — capacity 0; `OutOfMemory` — провал provisioning колец (без CPU-фолбэка).
- `TooManyColliders` (8/8/4), `InvalidOptions` (радиус ≤ 0/NaN, вырожденные боксы/нормали, NaN высота земли, невалидный AABB меша).
- `setFlowMap`: `InvalidDimensions` (0 dims, несовпадение длины, overflow) — старое поле нетронуто.
- Переполнение compute-стейджинга за кадр: bulk-eviction старейшей половины — стейдж всегда последнее окно `min(emitted, capacity)` в порядке эмиссии.
- `dt <= 0` в CPU: эмиссия по `emit_rate*dt` не стреляет, но старение/компактификация идут; `lifetime` зажимается снизу `1e-4`.
- `.gpu active_count` — верхняя граница (high-water), точный live-count только на GPU; шейдер схлопывает мёртвые слоты в вырожденный треугольник.

## Производительность

- CPU: O(живых) на тик; disarmed flow/collisions — по одной предсказуемой ветке на слот, бит-в-бит legacy, без лишних FP/mem. Alive-флаги в предвыделенном `alive_scratch` — ноль аллокаций в кадре. Подсистемы с нулём саб-эмиттеров не трогают PRNG/hashes смерти.
- GPU: O(эмитированных) на кадр, не O(частиц); загрузка — один `sg.updateBuffer` на непрерывный dirty-рандж, при wrap'е кольца — весь `[0, high_water)` префикс.
- Compute: диспатч покрывает `groupCount(high_water, 64)` групп; `compute_dt_accum` потребляется и обнуляется во flush; кадры без update ничего не диспатчат (кроме висячих саб-эмиттерных стейджей с dt 0).
- Память: `gpu_slots`/`compute_staging` provision'ятся лениво — CPU-only системы не платят; `flow_pixels` — w*h*4 однократно; коллайдеры/саб-эмиттеры — inline-фиксированные (`add*` не аллоцирует).

## Смотрите также

- `./scene.md` — `createParticleSystem`, particle-слой сцены.
- `./passes.md` — particle-проход рендера (billboard pipeline, bind compute draw-буфера).
- `./shaders.md` — `particle_gpu` / `particle_compute` шейдеры, `CsParams`.
- `./frame-pipeline.md` — update vs prepare vs draw границы, `flushGpuUploads`.
- `./profiler.md` — `upload_meter` учёт динамики.
