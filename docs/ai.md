# ИИ и навигация

> Путь: src/agate/ai.zig, src/agate/ai/*.zig · Импорт: agate.NavMesh, agate.Pathfinding, agate.NavAgent, agate.Crowd (root.zig) · Потоки: вся навигация однопоточная, владеет вызывающий поток (обычно update)

## Что это

Лёгкий навигационный стек без внешних зависимостей: `navmesh.zig` (треугольный NavMesh + dual-граф соседства), `pathfinding.zig` (A* по центроидам + порталы), `funnel.zig` (string-pulling, выпрямление коридора), `agent.zig` (одиночный агент со steering'ом), `crowd.zig` (толпа с ORCA-разведением в плоскости XZ). Всё детерминировано на CPU, аллокации только на построение пути; сам шаг агентов/толпы не аллоцирует.

Конвенция: треугольники — CCW при виде сверху, нормали развернуты вверх; движение — в плоскости XZ, высота Y интерполируется по плоскости треугольника.

## Быстрый старт

```zig
// Сетка 20x20 вокруг боксов препятствий:
var nav = try agate.NavMesh.buildGrid(allocator, -10, 10, -10, 10, 0.0, 20, 20, &obstacles);
defer nav.deinit();

// Или из геометрии уровня (стены с крутизной > max_slope выкидываются):
var nav2 = try agate.NavMesh.buildFromTriangles(allocator, positions, indices, std.math.pi * 0.25);
defer nav2.deinit();

// Путь одним вызовом (владелец — вызывающий, освободить!):
const path = try agate.Pathfinding.findPath(&nav, start, goal, allocator);
defer allocator.free(path);

// Агент с плавным рулением:
var agent = agate.NavAgent.init(allocator, &nav, start);
defer agent.deinit();
_ = try agent.setDestination(goal);
// в кадре:
agent.update(dt);
// agent.position, agent.yaw — применить к мешу.

// Толпа:
var crowd = agate.Crowd.init(allocator, &nav);
defer crowd.deinit();
const id = try crowd.addAgent(pos, .{});
_ = try crowd.setAgentDestination(id, goal);
crowd.update(dt);
```

## API

### NavMesh и dual-граф

```zig
pub const NavNode = struct { vertices: [3]Vec3, centroid: Vec3, normal: Vec3,
    neighbors: [3]?u32 = .{ null, null, null }, cost: f32 = 1.0, flags: u32 = 0 };
pub fn getEdge(self: NavNode, edge_index: usize) [2]Vec3 // рёбра: 0->v0v1, 1->v1v2, 2->v2v0
pub fn containsPointXZ(self: NavNode, pt: Vec3) bool     // 2D cross-тест + AABB early-out
pub fn getYAtXZ(self: NavNode, pt: Vec3) f32             // высота по уравнению плоскости
pub fn init(allocator: std.mem.Allocator, nodes: []NavNode, bounds: BoundingBox) NavMesh
pub fn deinit(self: *NavMesh) void
pub fn buildFromTriangles(allocator: std.mem.Allocator, positions: []const [3]f32, indices: []const u32, max_slope_rad: f32) !NavMesh
pub fn buildGrid(allocator: std.mem.Allocator, min_x: f32, max_x: f32, min_z: f32, max_z: f32,
    elevation_y: f32, subdiv_x: usize, subdiv_z: usize, obstacles: []const BoundingBox) !NavMesh
pub fn findNode(self: *const NavMesh, pt: Vec3) ?u32       // с допуском |Δy| < 3 м
pub fn findClosestNode(self: *const NavMesh, pt: Vec3) ?u32 // точное попадание, иначе ближайший центроид
pub fn clampToMesh(self: *const NavMesh, pt: Vec3) Vec3    // x/z как есть, y по поверхности
```

`buildFromTriangles` отбрасывает вырожденные треугольники (`len² < 1e-8`) и слишком крутые (`normal.y < cos(max_slope)`), разворачивает перевёрнутые нормалью вниз (меняет порядок вершин), считает центроид/границы и спаривает общие рёбра через квантованный (1 мм) хэш — так строится dual-граф: `neighbors[edge]` — индекс соседа или `null` (внешняя стена). Сложность O(треугольников), память — дуп узлов. `cost` (множитель A*) и `flags` (вода/опасность/прыжки) — пользовательские, билдер ставит `1.0/0`.

`buildGrid` режет плоскость на `subdiv_x × subdiv_z` квадов (минимум 1×1), выкидывает квады, пересекающие `obstacles` (AABB-тест с инсетом `0.05 * min(step)` чтобы касание гранью не блокировало), каждый квад — 2 CCW-треугольника, дальше общий `buildFromTriangles` с `max_slope = 0.4π`. Вне меша `findNode` — `null`, `findClosestNode` — ближайший центроид (включая Y), `clampToMesh` вне меша возвращает точку как есть.

### A*, funnel, string-pulling

```zig
pub fn findPath(navmesh: *const NavMesh, start_pos: Vec3, end_pos: Vec3, allocator: std.mem.Allocator) ![]Vec3
pub const Portal = struct { left: Vec3, right: Vec3 };
pub inline fn triArea2D(a: Vec3, b: Vec3, c: Vec3) f32 // знаковая площадь в XZ: > 0 — левый поворот
pub fn stringPull(allocator: std.mem.Allocator, start: Vec3, end: Vec3, portals: []const Portal) ![]Vec3
```

`findPath` — единственный аллоцирующий вызов кадра (путь владеет вызывающий). Шаги: ближайшие узлы старта/цели; один треугольник — `[start, end]`; иначе A* (`g` по дистанциям центроидов × `cost` соседа, эвристика — евклидово до цели, очередь `std.PriorityQueue`, массивы `g/came_from/closed` размера N — O(N) памяти на вызов); коридор разворачивается, между соседями ищется общее ребро, лево/право портала выбирается знаком `triArea2D(centroid_from, centroid_to, p0)`; финальный портал — точка цели; дальше `stringPull`. Нет коридора — `[start]` (один вейпоинт, не пусто!); пустой меш — пустой слайс. Сложность A* O(E log V), funnel O(порталов).

`stringPull` — классическая воронка: apex + левая/правая кромки сужаются по порталам, при схлопывании apex прыгает на кромку и воронка перезапускается; старт всегда первый вейпоинт, конец дописывается если отличается (> 1e-3). Пустые порталы — `[start, end]`.

### Одиночный агент

```zig
pub fn init(allocator: std.mem.Allocator, nav_mesh: *const NavMesh, start_pos: Vec3) NavAgent // старт clamp'ится к мешу
pub fn deinit(self: *NavAgent) void
pub fn setDestination(self: *NavAgent, target: Vec3) !bool // false: путь пуст (arrived=true)
pub fn stop(self: *NavAgent) void        // arrived, скорость 0, путь freed
pub fn teleport(self: *NavAgent, pos: Vec3) void // stop + clamp (если snap_to_mesh)
pub fn update(self: *NavAgent, dt: f32) void     // никогда не аллоцирует
```

Настройки: `speed = 2.0`, `acceleration = 4.0` (инерционное сглаживание `lerp(desired, min(accel*dt,1))`), `rotation_speed = 5.0` рад/с (поворот к курсу с заворотом угла в ±π), `waypoint_radius = 0.6` / `stopping_distance = 0.25` (последний вейпоинт строже), `slowdown_distance = 1.2` (торможение у цели, минимум 20% скорости), `elevation_speed = 10.0` (вертикальное сглаживание швов), `snap_to_mesh = true`, `yaw`. Движение — XZ + пропорциональный шаг по Y к вейпоинту; простой гасит скорость (`accel*2*dt`). Вейпоинт 0 — старт, движение сразу к 1. `update` с `dt <= 0` только гасит скорость.

### Толпа (ORCA)

```zig
pub const CrowdAgentParams = struct { radius: f32 = 0.5, height: f32 = 2.0, max_speed: f32 = 2.5,
    max_acceleration: f32 = 6.0, neighbor_dist: f32 = 8.0, max_neighbors: usize = 10,
    time_horizon: f32 = 2.0, time_horizon_obst: f32 = 1.0, stopping_distance: f32 = 0.3,
    waypoint_radius: f32 = 0.6, slowdown_distance: f32 = 1.2 };
pub fn init(allocator: std.mem.Allocator, nav_mesh: ?*const NavMesh) Crowd // null = прямые пути без меша
pub fn deinit(self: *Crowd) void
pub fn agentCount(self: *const Crowd) usize
pub fn addAgent(self: *Crowd, position: Vec3, params: CrowdAgentParams) !u32 // id с 1, монотонны
pub fn removeAgent(self: *Crowd, agent_id: u32) void   // swapRemove, молчит при miss
pub fn getAgent(self: *Crowd, agent_id: u32) ?*CrowdAgent
pub fn getAgentConst(self: *const Crowd, agent_id: u32) ?*const CrowdAgent
pub fn setAgentDestination(self: *Crowd, agent_id: u32, target: Vec3) !bool
pub fn setAgentVelocity(self: *Crowd, agent_id: u32, velocity: Vec3) void
pub fn update(self: *Crowd, dt: f32) void
```

Шаг толпы (4 фазы, O(агенты × соседи)): предпочтительные скорости по вейпоинтам (с торможением у цели) → ORCA полуплоскости на pairwise-соседях в `neighbor_dist`: вне коллизии — конус усечения на `time_horizon`, внутри — раздвижка проникновения на `1/dt`; каждая пара даёт линию, ответственность делится пополам (reciprocal) → LP2/LP3 (линейное программирование в круге `max_speed`, фолбэк `linearProgram3` при несовместности) → интеграция с лимитом ускорения, yaw по скорости (> 0.05), clamp к мешу. Жёсткие перекрытия раздвигаются самой ORCA-веткой «already colliding». `dt <= 0` или пусто — no-op.

Ограничения реализации: внутренний буфер скоростей `[64]` — агенты сверх 64 не интегрируются на этом шаге (считаются из первых 64); ORCA-линий на агента ≤ 32; препятствия как геометрия не учитываются (`time_horizon_obst` хранится, но в коде не используется — только агент-агент); `max_neighbors` хранится, но жёсткий кап — размер буфера линий. Статические препятствия — только через NavMesh-коридоры.

Настройка ORCA под задачу:

| Параметр | Эффект при увеличении |
|---|---|
| `neighbor_dist` | больше соседей → плавнее, но дороже (квадратично в радиусе) |
| `time_horizon` | раньше начинает уступать, шире дуга обхода |
| `radius` | больше личный зазор; сумма радиусов пары — дистанция контакта |
| `max_speed` | радиус LP-круга скоростей (жёсткий кап) |
| `max_acceleration` | скорость сходимости к ORCA-скорости за шаг (`max_accel * dt`) |

## Потоки и владение

Всё синхронно в вызывающем потоке; внутреннего пула и глобального состояния нет. `NavMesh` владеет слайсом узлов (`deinit` освобождает, `bounds` — копия). Пути (`findPath`, `setDestination`, `setAgentDestination`) — owned-слайсы: старый путь освобождается при новом назначении/`stop`/`deinit`/`removeAgent`; прямой путь толпы без меша — 2-точечный `[pos, target]`. Агент хранит `nav_mesh: *const NavMesh` — меш должен жить дольше агентов/толпы. `teleport`/`addAgent`/`clampToMesh` читают меш; конкурентных записей нет — для фоновых пересчётов держите отдельную копию меша.

## Ошибки и краевые случаи

- `findPath`: пустой меш → пустой слайс (не ошибка); miss узлов → пусто; нет коридора → `[start]`; OOM — единственная ошибка (`![]Vec3`).
- `setDestination` возвращает `false` и ставит `arrived` при пустом пути; старый путь при этом уже освобождён.
- Точки вне меша: старт/цель притягиваются к ближайшему узлу — путь может начаться в стороне от запрошенной точки; `teleport` с `snap_to_mesh = false` кладёт raw.
- Допуск высоты `findNode` 3 м: многоэтажки с перепадом < 3 м могут замаппиться не на тот этаж — режьте меши по этажам.
- Квант ребра 1 мм: T-junction'ы с зазором > 1 мм соседями не станут (стена вместо прохода).
- Толпа: `getAgent` miss → `null`; `removeAgent` miss — молча; `setAgentDestination` на miss → `false`; агенты `active = false` стоят, но участвуют как препятствия в ORCA соседей.
- Детерминизм: итерации по индексам, без хэш-мап в кадре — тот же `dt` даёт тот же результат; переменный `dt` меняет траекторию (явного фиксатора шага нет — шагайте константой снаружи при нужде).

## Производительность

- `containsPointXZ`: AABB early-out (4 сравнения) + 3 cross-продукта; `findNode` — линейный O(N) по узлам (без BVH — большие меши режьте на чанки или кэшируйте узел).
- A*: O(E log V) + O(N) scratch на вызов (3 массива + куча); коридор/порталы — линейны; funnel — линеен, без аллокаций кроме выхода.
- `NavAgent.update`: O(вейпоинтов通行) + O(1); `Crowd.update`: O(A × соседи в радиусе), LP на стековых буферах (32 линии), ноль аллокаций в кадре.
- Память пути: 2–N `Vec3`; длинные коридоры выпрямляются funnel'ом до нескольких точек.

## Смотрите также

- `./math.md` — `Vec3`, `BoundingBox.intersects` (buildGrid), `lerp`.
- `./physics.md` — тела и `CharacterController` (агенты физики не знают; стык — через позицию).
- `./scene.md` — `nav_layer` сцены, применение `yaw`/позиции к мешам.
- `./frame-pipeline.md` — куда ставить `agent.update/crowd.update` в кадре.
- `./profiler.md` — замер A* и ORCA на больших толпах.
