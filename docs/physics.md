# Физика

> Путь: src/agate/physics.zig, src/agate/physics/*.zig, src/agate/physics_mesh.zig, src/agate/ragdoll.zig, src/agate/vehicle.zig · Импорт: agate.PhysicsWorld, agate.RigidBody, agate.Ragdoll, agate.RaycastVehicle (root.zig) · Потоки: update-поток (владелец `PhysicsWorld`), render-поток только читает меши/DebugLine

## Что это

Модуль физики — тонкая Zig-обёртка над Box3D (через `src/agate/c.zig`). Мир `PhysicsWorld` владеет телами `RigidBody`, суставами `JointId` и очередями событий; каждый шаг синхронизирует движок с `Mesh.position/rotation/scaling` в обе стороны. Файл `physics.zig` — фасад, реэкспортирующий типы из `physics/types.zig`, `physics/body.zig` и `physics/world.zig`; сама логика разложена по листьям: `world.zig` (мир и тела), `body.zig` (тела и события), `queries.zig` (лучи и перекрытия), `joints.zig` (суставы), `character.zig` (кинематический контроллер), `rope.zig` (верёвки), `events.zig` (дренаж событий), `debug.zig` + `debug_geo.zig` (wireframe), `convert.zig` (конверсии единиц, internal). Поверх лежат готовые хелперы: `ragdoll.zig` (`Ragdoll`, 11 тел / 10 суставов) и `vehicle.zig` (`RaycastVehicle`, шасси + 4 колеса на wheel joints), а `physics_mesh.zig` (internal, не экспортируется из `root.zig`) владеет жизненным циклом их мешей.

Статика/динамика определяется массой: `mass <= 0` — статическое тело, иначе динамическое. Плотность шейпов пересчитывается из массы через объём (`shapeVolume` + `densityForMass`), сенсоры остаются безмассовыми. Шаг фиксированный: `step_h = 1/60`, до 4 подшагов за кадр, `substeps = 4` по умолчанию.

## Быстрый старт

```zig
const agate = @import("agate");

var world = agate.PhysicsWorld.init(allocator);
defer world.deinit();

// Земля-плоскость по умолчанию уже есть (ground_y = -1.2).
// Ящик 1x1x1 в точке (0, 2, 0):
const box = try agate.MeshBuilder.createBox(scene, "box", .{});
const body = try world.createBody(box, .box, 1.0);
body.friction = 0.5;

// Кадр:
world.step(dt); // тянет позиции из солвера в mesh.position/rotation

// Луч из камеры:
const hit = world.raycast(cam_pos, cam_dir, 100.0);
if (hit.hit) {
    if (hit.body) |b| b.applyImpulse(agate.Vec3.new(0, 5, 0));
}

// События контакта за последний шаг:
for (world.contact_events.items) |ev| {
    _ = ev; // .a, .b: ?*RigidBody, .began: bool
}
```

Кинематический персонаж без `RigidBody`:

```zig
var hero = agate.CharacterController.init(start_pos, 0.3, 1.7);
hero.move(&world, wish_dir, jump_pressed, dt);
// hero.position — точка ступней; hero.is_grounded — флаг опоры.
```

Рэгдолл и машина (меши опциональны — `scene = null` даёт physics-only режим):

```zig
var doll = try agate.Ragdoll.init(allocator, &world, scene_or_null, .{ .position = spawn });
defer doll.deinit(&world);
doll.applyImpulse(agate.Vec3.new(20, 5, 0));

var car = try agate.RaycastVehicle.init(allocator, &world, scene_or_null, .{});
defer car.deinit(&world);
car.setThrottle(1.0);
car.setSteering(0.2);
car.update(dt);
world.step(dt);
```

## API

### Мир: создание и шаг

```zig
pub fn init(allocator: std.mem.Allocator) PhysicsWorld
pub fn deinit(self: *PhysicsWorld) void
pub fn step(self: *PhysicsWorld, dt: f32) void
pub fn syncWorldParams(self: *PhysicsWorld) void
pub fn enableContinuous(self: *PhysicsWorld, flag: bool) void
pub fn isContinuousEnabled(self: *const PhysicsWorld) bool
pub fn getProfile(self: *const PhysicsWorld) PhysicsProfile
pub fn getCounters(self: *const PhysicsWorld) PhysicsCounters
```

Поля `PhysicsWorld`: `gravity: Vec3` (по умолчанию `(0, -9.81, 0)`), `ground_y: ?f32` (`-1.2`; `null` убирает плоскость), `substeps: u32` (`4`), `hit_event_threshold: f32` (`1.0` м/с), `debug_circle_segments: usize` (`24`), `bodies`, `joints`, `sensor_events`, `contact_events`, `contact_hit_events`. `step` при `dt <= 0.0001` только чистит события; кадры без тел пропускают солвер; накопление `acc` ограничено `0.1` с, отставание больше 4 подшагов сбрасывается.

| Метод | Сложность | Аллокации | Ошибки |
|---|---|---|---|
| `step(dt)` | O(тел + контактов) | нет (события — `catch {}`, дроп при OOM) | нет |
| `getProfile / getCounters` | O(1), прямой запрос Box3D | нет | нет |
| `syncWorldParams` | O(1), пересоздаёт землю при смене `ground_y` | нет | нет |

### Тела и compound-шейпы

```zig
pub fn createBody(self: *PhysicsWorld, mesh: *Mesh, collider: ColliderType, mass: f32) !*RigidBody
pub fn createBodyWith(self: *PhysicsWorld, mesh: *Mesh, collider: ColliderType, mass: f32, options: BodyOptions) !*RigidBody
pub fn createHeightField(self: *PhysicsWorld, mesh: *Mesh, heights: []const f32, count_x: u32, count_z: u32, options: HeightFieldOptions) !*RigidBody
pub fn removeBody(self: *PhysicsWorld, body: *RigidBody) void
pub fn findBody(self: *PhysicsWorld, mesh: *const Mesh) ?*RigidBody
pub fn addBoxShape(self: *PhysicsWorld, body: *RigidBody, half_extents: Vec3, options: ChildShapeOptions) !void
pub fn addSphereShape(self: *PhysicsWorld, body: *RigidBody, radius: f32, options: ChildShapeOptions) !void
pub fn addCapsuleShape(self: *PhysicsWorld, body: *RigidBody, half_height: f32, radius: f32, options: ChildShapeOptions) !void
pub fn addHullShape(self: *PhysicsWorld, body: *RigidBody, points: []const Vec3, options: ChildShapeOptions) !void
pub fn applyExplosion(self: *PhysicsWorld, epicenter: Vec3, radius: f32, max_impulse: f32, upward_modifier: f32) void
```

Типы (`physics/types.zig`):

```zig
pub const ColliderType = enum { box, sphere, capsule, hull, mesh, heightfield };
pub const HeightFieldOptions = struct { scale: Vec3 = Vec3.new(1,1,1), clockwise_winding: bool = false };
pub const CollisionFilter = struct { category_bits: u64 = 1, mask_bits: u64 = ~0, group_index: i32 = 0 };
pub const BodyOptions = struct { filter: CollisionFilter = .{}, is_sensor: bool = false,
    enable_sensor_events: bool = false, enable_contact_events: bool = false,
    enable_hit_events: bool = false, is_bullet: bool = false };
pub const ChildShapeOptions = struct { offset: Vec3 = Vec3.zero, friction: ?f32 = null,
    restitution: ?f32 = null, filter: ?CollisionFilter = null, is_sensor: ?bool = null,
    enable_sensor_events: ?bool = null, enable_contact_events: ?bool = null, enable_hit_events: ?bool = null };
pub const PickingInfo = struct { hit: bool = false, distance: f32 = 0.0, picked_point: Vec3 = ...,
    picked_normal: Vec3 = ..., picked_mesh: ?*Mesh = null, picked_instance: ?usize = null };
pub const DebugLine = struct { a: Vec3, b: Vec3, color: [3]f32 = .{ 0.1, 0.9, 0.3 } };
```

Правила коллайдеров: `hull` строится из `Mesh.cpu_positions` (нужно ≥ 4 точек); `mesh` — только статика (`mass == 0`, иначе `error.StaticColliderRequiresZeroMass`), данные ссылаются на `Mesh.cpu_positions/cpu_indices` напрямую; `heightfield` создаётся только через `createHeightField` (иначе `error.UseCreateHeightField`), сетка row-major `heights[row * count_x + col]`, `count_x/count_z >= 2`, scale строго положительный. Тонкие шейпы зажимаются снизу: `min_half_extent = 0.01`. Добавление child-шейпа перераспределяет плотность по всем шейпам, сохраняя массу тела.

`RigidBody` (`physics/body.zig`):

```zig
pub fn init(mesh: *Mesh, collider: ColliderType, mass: f32) RigidBody
pub fn shapeVolume(b: *const RigidBody, scale: Vec3) f32
pub fn densityForMass(mass: f32, volume: f32) f32
pub fn applyImpulse(self: *RigidBody, impulse: Vec3) void
pub fn applyTorqueImpulse(self: *RigidBody, torque_impulse: Vec3) void
pub fn applyForce(self: *RigidBody, force: Vec3, dt: f32) void
pub fn setBullet(self: *RigidBody, flag: bool) void
pub fn isBullet(self: *const RigidBody) bool
pub fn setMass(self: *RigidBody, mass: f32) void
pub fn worldToLocal(self: *const RigidBody, world_point: Vec3) Vec3
pub fn localToWorld(self: *const RigidBody, local_point: Vec3) Vec3
```

Поля состояния в живых: `velocity` (м/с), `angular_velocity` (град/с — внимание, Box3D внутри радианы, конверсия `deg2rad` автоматическая), `friction` (`0.25`), `restitution` (`0.6`), `linear_damping` (`0.015`), `angular_damping` (`0.04`), `use_gravity`, `enabled`, `is_grounded` (только чтение, считает `computeGrounded` по манифолдам с порогом нормали `up > 0.7` и `separation <= 0.01`). Прямая запись в поля подхватывается следующим `step` (push), солвер — в меши (pull). `applyForce` делит на массу и будит тело; `applyImpulse` на статике — no-op.

### Суставы

`JointId = c.b3JointId`. Все create-варианты есть в локальных (`local_anchor_a/b`) и мировых (`world_anchor`) координатах; мировые пересчитывают якоря через `worldToLocal`.

```zig
pub fn createDistanceJoint(self, body_a: *RigidBody, body_b: *RigidBody, local_anchor_a: Vec3, local_anchor_b: Vec3, options: DistanceJointOptions) !JointId
pub fn createDistanceJointWorld(self, body_a: *RigidBody, body_b: *RigidBody, world_anchor_a: Vec3, world_anchor_b: Vec3, options: DistanceJointOptions) !JointId
pub fn createSphericalJoint(...) !JointId
pub fn createSphericalJointWorld(self, body_a, body_b, world_anchor: Vec3, options: SphericalJointOptions) !JointId
pub fn createRevoluteJoint(...) !JointId
pub fn createRevoluteJointWorld(self, body_a, body_b, world_anchor: Vec3, options: RevoluteJointOptions) !JointId
pub fn setRevoluteMotor(self, joint_id: JointId, enabled: bool, motor_speed_rad: f32, max_motor_torque: f32) void
pub fn setRevoluteLimits(self, joint_id: JointId, lower_angle_rad: f32, upper_angle_rad: f32) void
pub fn revoluteAngleRad(self, joint_id: JointId) f32
pub fn revoluteAngleDeg(self, joint_id: JointId) f32
pub fn createWheelJoint(...) !JointId
pub fn createWheelJointWorld(self, body_a, body_b, world_anchor: Vec3, options: WheelJointOptions) !JointId
pub fn setWheelSpin(self, joint_id: JointId, enabled: bool, spin_speed_rad: f32, max_spin_torque: f32) void
pub fn setWheelSteering(self, joint_id: JointId, enabled: bool, target_angle_rad: f32, max_steering_torque: f32) void
pub fn wheelSpinSpeed(self, joint_id: JointId) f32
pub fn wheelSteeringAngle(self, joint_id: JointId) f32
pub fn createPrismaticJoint(...) !JointId
pub fn createPrismaticJointWorld(...) !JointId
pub fn setPrismaticMotor(self, joint_id: JointId, enabled: bool, motor_speed: f32, max_motor_force: f32) void
pub fn setPrismaticLimits(self, joint_id: JointId, lower_translation: f32, upper_translation: f32) void
pub fn prismaticTranslation(self, joint_id: JointId) f32
pub fn prismaticSpeed(self, joint_id: JointId) f32
pub fn createMotorJoint(...) !JointId
pub fn createMotorJointWorld(self, body_a, body_b, world_anchor: Vec3, options: MotorJointOptions) !JointId
pub fn setMotorLinearVelocity(self, joint_id: JointId, velocity: Vec3) void
pub fn setMotorAngularVelocity(self, joint_id: JointId, velocity_rad: Vec3) void
pub fn setMotorMaxVelocityForce(self, joint_id: JointId, max_force: f32) void
pub fn setMotorMaxVelocityTorque(self, joint_id: JointId, max_torque: f32) void
pub fn createWeldJoint(...) !JointId
pub fn createWeldJointWorld(...) !JointId
pub fn jointConstraintForce(self, joint_id: JointId) Vec3
pub fn jointConstraintTorque(self, joint_id: JointId) Vec3
pub fn createParallelJoint(...) !JointId
pub fn createParallelJointWorld(...) !JointId
pub fn setParallelSpring(self, joint_id: JointId, hertz: f32, damping_ratio: f32, max_torque: f32) void
pub fn isJointValid(_: *PhysicsWorld, joint_id: JointId) bool
pub fn destroyJoint(self: *PhysicsWorld, joint_id: JointId) void
```

Опции (дефолты важны):

| Опции | Ключевые поля |
|---|---|
| `DistanceJointOptions` | `length: ?f32` (null = автодистанция), `enable_spring/hertz=4/damping_ratio=0.5`, `min_length/max_length: ?f32`, `collide_connected=false` |
| `SphericalJointOptions` | пружина + `enable_cone_limit/cone_angle_rad`, `enable_twist_limit/lower/upper_twist_angle_rad` |
| `RevoluteJointOptions` | ось — z кадра A; `frame_a/frame_b: Quat`, spring, `enable_limit/lower/upper_angle_rad`, `enable_motor/motor_speed_rad/max_motor_torque` |
| `WheelJointOptions` | подвеска вдоль x кадра A, спин вокруг z кадра B; `suspension_hertz=7/damping=0.7/limits ±0.25`, spin motor, steering servo + limits |
| `MotorJointOptions` | velocity-мотор + pose-пружина; с нулями инертен (паттерн drag из SandboxScene) |
| `WeldJointOptions` | `linear/angular_hertz=0` (максимальная жёсткость); пара с `jointConstraintForce` = разрушаемые суставы |
| `ParallelJointOptions` | держит z-оси соосно; `hertz=4/damping_ratio=0.7/max_torque` |
| `PrismaticJointOptions` | слайдер вдоль x кадра A; spring/limit/motor в метрах |

Ось revolute — z кадра (world z при identity), prismatic — x кадра. `removeBody` автоматически чистит невалидные суставы из `world.joints`.

### Запросы

```zig
pub fn raycast(self: *PhysicsWorld, origin: Vec3, direction: Vec3, max_distance: f32) PhysicsRayHit
pub fn raycastWithFilter(self, origin: Vec3, direction: Vec3, max_distance: f32, filter: CollisionFilter) PhysicsRayHit
pub fn queryAABB(self, min: Vec3, max: Vec3, results: *std.ArrayListUnmanaged(*RigidBody)) !void
pub fn queryAABBWithFilter(self, min, max, filter, results) !void
pub fn querySphere(self, center: Vec3, radius: f32, results: *...) !void
pub fn querySphereWithFilter(self, center, radius, filter, results) !void
pub fn queryPoint(self, point: Vec3, results: *...) !void
pub fn queryPointWithFilter(self, point, filter, results) !void
pub fn spherecast(self: *PhysicsWorld, origin: Vec3, radius: f32, translation: Vec3) ?PhysicsRayHit
pub fn spherecastWithFilter(self, origin, radius, translation, filter) ?PhysicsRayHit
pub fn findBodyByShape(self: *PhysicsWorld, shape_id: c.b3ShapeId) ?*RigidBody
pub fn ownsShape(_: *PhysicsWorld, body: *RigidBody, shape_id: c.b3ShapeId) bool
```

`PhysicsRayHit = { hit, point, normal, distance, body: ?*RigidBody }`. `raycast` возвращает ближайший хит; при `max_distance <= 0` или вырожденном направлении — промах. `query*` дописывают (не чистят) каждое тело максимум один раз, пропускают `!enabled` и нетреканные шейпы (внутренняя земля). `queryPoint` — containment по мировому AABB шейпа (не точная поверхность). `spherecast` возвращает центр сферы в момент контакта (`point == origin + translation * t`), а не точку на цели. Все запросы O(перекрытий broadphase); единственная ошибка — `error.OutOfMemory` при append.

Плюс аудио-окклюзия через те же коллайдеры:

```zig
pub fn audioRaycastAdapter(origin: Vec3, direction: Vec3, max_distance: f32, user_data: ?*anyopaque) bool
pub fn evaluateAudioOcclusion(self: *PhysicsWorld, listener_pos: Vec3, emitter_pos: Vec3, config: audio.AudioOcclusionConfig) f32
```

### События

```zig
pub const SensorEvent = struct { sensor: ?*RigidBody, visitor: ?*RigidBody, began: bool };
pub const ContactEvent = struct { a: ?*RigidBody, b: ?*RigidBody, began: bool };
pub const ContactHitEvent = struct { a: ?*RigidBody, b: ?*RigidBody, point: Vec3, normal: Vec3, approach_speed: f32 };
pub fn clearEvents(self: *PhysicsWorld) void
```

Собираются внутри `step` (по подшагам через `drainEvents`), чистятся в начале следующего `step` — описывают только последний кадр. Сенсорные события требуют `enable_sensor_events` на ОБЕИХ сторонах (правило Box3D). Хиты — только при скорости сближения выше `hit_event_threshold`. `null` в паре — шейп без треканного тела (внутренняя земля).

### Персонаж, верёвка, debug

```zig
pub const CharacterController = struct {
    position: Vec3, velocity: Vec3, radius: f32 = 0.3, height: f32 = 1.5,
    move_speed: f32 = 4.5, jump_speed: f32 = 6.0, gravity: f32 = 18.0,
    slope_limit_cos: f32 = 0.7, is_grounded: bool = false, ground_normal: Vec3 = Vec3.up,
    step_height: f32 = 0.35, push_strength: f32 = 25.0, push_dynamic_bodies: bool = true, ...
};
pub fn init(position: Vec3, radius: f32, height: f32) CharacterController
pub fn move(self: *CharacterController, world: *PhysicsWorld, wish_dir: Vec3, jump_pressed: bool, dt: f32) void
```

Контроллер кинематический (mover API: collide + plane solve + velocity clip), НЕ тело: скользит вокруг динамических тел, невидим для сенсоров. `position` — точка ступней. `wish_dir.y` игнорируется, длина > 1 нормируется. Шаг ограничен `dt <= 1/30`. Есть climbing ступенек (`step_height`) и толкание динамических тел импульсом.

```zig
pub const RopeOptions = struct { collider: ColliderType = .capsule, segment_mass: f32 = 0.4,
    restitution: f32 = 0.1, friction: f32 = 0.4, max_stretch: f32 = 1.05, min_compress: ?f32 = null,
    spring: bool = false, spring_hertz: f32 = 3.0, spring_damping: f32 = 0.6,
    collide_connected: bool = false, pin_start/pin_end: ?*RigidBody = null,
    pin_start_anchor/pin_end_anchor: ?Vec3 = null, ... };
pub fn createRope(self: *PhysicsWorld, segment_meshes: []*Mesh, options: RopeOptions) !Rope
pub fn deinit(self: *Rope, world: *PhysicsWorld) void
pub fn detachAt(self: *Rope, world: *PhysicsWorld, index: usize) void
pub fn repair(self: *Rope, world: *PhysicsWorld) void
pub fn reset(self: *Rope, world: *PhysicsWorld) void
pub fn linkJointOptions(self: *const Rope) DistanceJointOptions
```

Верёвка владеет телами сегментов (пины внешние). `spacing` вычисляется из пролёта, минимум `0.05`; пустой список — `error.EmptyRope`. `detachAt` вне диапазона — no-op.

```zig
pub fn debugCircleSegments(self: *const PhysicsWorld) usize
pub fn appendDebugLines(self: *PhysicsWorld, allocator: std.mem.Allocator, out: *std.ArrayListUnmanaged(DebugLine)) !void
pub fn debugLineCount(self: *PhysicsWorld) usize
```

Цвета: динамика — зелёный, статика — белый, сенсоры — жёлтые (сенсор побеждает). Число линий: box 12, sphere `3*segs` (72 при 24), capsule `2*segs+4` (52), hull/mesh/heightfield — 12 (AABB). `appendDebugLines` резервирует точную ёмкость одним `ensureUnusedCapacity`, дальше только `appendAssumeCapacity`.

### Рэгдолл и машина

```zig
pub const RagdollPart = enum(u8) { pelvis, chest, head, upper_arm_l, upper_arm_r, lower_arm_l, lower_arm_r, upper_leg_l, upper_leg_r, lower_leg_l, lower_leg_r };
pub const part_count: usize = 11;
pub const joint_count: usize = 10;
pub const RagdollOptions = struct { position: Vec3 = ..., scale: f32 = 1.0, pelvis_size/chest_size/head_radius/... , pelvis_mass: f32 = 2.5, chest_mass: f32 = 4.0, ... friction: f32 = 0.6, restitution: f32 = 0.1 };
pub fn init(allocator: std.mem.Allocator, world: *PhysicsWorld, scene: ?*Scene, options: RagdollOptions) !Ragdoll
pub fn getPart(self: *Ragdoll, part: RagdollPart) *RigidBody
pub fn applyImpulse(self: *Ragdoll, impulse: Vec3) void
pub fn reset(self: *Ragdoll) void
pub fn syncMeshes(self: *Ragdoll) void
pub fn deinit(self: *Ragdoll, world: *PhysicsWorld) void
```

Суставы: 6 spherical (талия, шея, плечи, бёдра; cone/twist лимиты) + 4 revolute (локти/колени, лимиты в радианах). `reset` телепортирует в спавн-позу, подхватывается следующим `step`.

```zig
pub const wheel_count: usize = 4;
pub const VehicleOptions = struct { position: Vec3 = ..., chassis_size = (2.2, 0.5, 1.0), chassis_mass: f32 = 4.0,
    wheel_radius: f32 = 0.35, wheel_mass: f32 = 0.8, wheel_offsets: [4]Vec3 = ...,
    suspension_hertz: f32 = 8.0, suspension_damping: f32 = 0.7, suspension_travel: f32 = 0.2,
    drive_speed: f32 = 12.0, drive_torque: f32 = 30.0, max_steer_angle: f32 = 0.45, steer_torque: f32 = 25.0,
    brake_torque: f32 = 40.0, drive_front/drive_rear: bool = true, steer_front: bool = true, steer_rear: bool = false, ... };
pub fn init(allocator, world: *PhysicsWorld, scene: ?*Scene, options: VehicleOptions) !RaycastVehicle
pub fn chassisBody(self: *RaycastVehicle) *RigidBody
pub fn wheelBody(self: *RaycastVehicle, index: usize) *RigidBody
pub fn wheelJoint(self: *RaycastVehicle, index: usize) JointId
pub fn setThrottle(self: *RaycastVehicle, value: f32) void
pub fn setSteering(self: *RaycastVehicle, value: f32) void
pub fn setBrake(self: *RaycastVehicle, value: f32) void
pub fn update(self: *RaycastVehicle, dt: f32) void
pub fn syncMeshes(self: *RaycastVehicle) void
pub fn deinit(self: *RaycastVehicle, world: *PhysicsWorld) void
```

Передняя ось — `+X` (колёса со `wheel_offsets[i].x > 0` рулят при `steer_front`). Подвеска вдоль x кадра A, повёрнутого на `-90°` вокруг z (мировой `-Y`); отрицательный spin едет вперёд (`+X`). `update` вызывать каждый кадр до `world.step`. Deinit — строго до уничтожения сцены и мира (joint → тела → меши через `physics_mesh.teardown`).

## Потоки и владение

Владелец `PhysicsWorld` — update-поток: `step`, создание/удаление тел и суставов, чтение событий. Меши тел принадлежат сцене (или `physics_mesh` bare-режиму); `step` пишет в `mesh.position/rotation` напрямую — читать трансформы из render-потока во время шага нельзя, синхронизируйтесь кадровой границей (см. `./frame-pipeline.md`). `getRenderSkinMatrices`-подобного дабл-буфера здесь нет; для чтения из другого потока копируйте позиции после `step`. `pullBody` пропускает спящие тела (`!awake && !was_awake`) — их меши просто не трогаются. Debug-линии строятся в caller-буфер одним вызовом, без глобального состояния.

Владение: `createBody` аллоцирует `RigidBody` (мир владеет, `removeBody`/`deinit` освобождают); `mesh_data/heightfield_data` живут пока живёт тело; hull child-точки — owned-копия (`freeOwned`). Рэгдолл/машина/верёвка владеют своими телами; `Rope.deinit`/`Ragdoll.deinit`/`RaycastVehicle.deinit` обязательно до `Scene.deinit` и `world.deinit`.

## Ошибки и краевые случаи

- `error.OutOfMemory` — создание тел/шейпов/запросы (`query*` мапят OOM колбэка в ошибку).
- `error.MissingCollisionGeometry` — hull без ≥4 точек, mesh без позиций/индексов; `addHullShape` с <4 точек.
- `error.StaticColliderRequiresZeroMass` — `mesh`-коллайдер с `mass > 0`.
- `error.UseCreateHeightField` — `heightfield` через `createBody`.
- `error.InvalidHeightFieldDimensions / InvalidHeightFieldScale / PhysicsShapeCreationFailed / EmptyRope`.
- Смена `mesh.scaling` пересоздаёт шейпы (`recreateShape`); при ошибке старый коллайдер сохраняется.
- Смена статика↔динамика (`setMass` через 0) переключает `b3Body_SetType` и пересчитывает плотности.
- `transformMoved` толкает трансформ в солвер и будит тело; статика не будится.
- Угловые скорости в API — градусы/с, внутри — радианы/с; `pushBody`/`pullBody` канонизируют через солвер (углы Эйлера могут вернуться эквивалентными, но не побитово равными).
- `applyExplosion` с `radius <= 0` или `max_impulse <= 0` — no-op; статика и `!enabled` пропускаются.

## Производительность

- Фиксированный шаг `1/60`, `substeps = 4`, clamp кадра `0.1` с: физика не спиралит при просадках, но при >4 подшагов отставание сбрасывается (замедление вместо spiral of death).
- Change detection: параметры (трение, фильтр с `invokeContacts=true`, damping, gravity scale, скорости) пушатся в Box3D только при изменении.
- Sleeping нативный (Box3D islands): спящие тела не тянутся в меши, `was_awake` кэширует пробуждение.
- CCD: `enableContinuous` проксирует `b3World_EnableContinuous`; для быстрых снарядов дополнительно `BodyOptions.is_bullet`.
- Профиль: `getProfile` (step/pairs/collide/solve/bullets/transforms/sensors, мс) и `getCounters` (body/shape/contact/joint/island) — кормить в `./profiler.md`.
- Детерминизм: при фиксированном `dt` и том же порядке push интеграция повторяется; переменный `dt` накапливается через аккумулятор — для сетевого детерминизма шагайте константой снаружи.

## Смотрите также

- `./scene.md` — владение мешами и слоями (`physics_layer`).
- `./mesh.md` — `cpu_positions/cpu_indices` как источник hull/mesh коллайдеров.
- `./audio.md` — окклюзия через `evaluateAudioOcclusion`.
- `./profiler.md` — куда отдавать `PhysicsProfile/PhysicsCounters`.
- `./ai.md` — навигация поверх физики (агенты не толкают тела, кроме `CharacterController.push_dynamic_bodies`).
