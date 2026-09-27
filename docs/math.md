# Математика

> Путь: src/agate/math.zig, src/agate/math/*.zig · Импорт: agate.Vec3, agate.Mat4, agate.Quat, agate.BoundingBox, agate.Frustum, agate.Ray (root.zig) · Потоки: чистые value-типы, потокобезопасны (включая SIMD-хелперы)

## Что это

Фундаментальные value-типы движка отдельным build-модулем `math`: `vec.zig` (`Vec2/Vec3/Vec4`), `mat4.zig` (`Mat4`), `quat.zig` (`Quat`), `bounding_box.zig` (`BoundingBox`), `frustum.zig` (`Frustum/FrustumPlane`), `ray.zig` (`Ray/RayHit/TriangleHit`), `color.zig` (`Color3/Color4`), плюс `lerp(a, b, t)`. Соглашения: column-major матрицы (`m[col*4+row]`), Эйлер в градусах с порядком `R = Rz·Ry·Rx`, глубина клипа `[0, 1]` (фрустум Грибба–Хартманна под неё). SIMD `@Vector(4, f32)` используется точечно: `Vec4` арифметика, `Mat4.mul/transformPoint/transformDirection/projectPoint`, `Color4.lerp/scale`, фрустум-куллинг (скаляр + 4-wide).

## Быстрый старт

```zig
const m = agate.Mat4.fromRotationTranslationScale(pos, rot_deg_euler, scale);
const world_pt = m.transformPoint(local_pt);
const world_dir = m.transformDirection(local_dir);

// Камера и куллинг:
const vp = agate.Mat4.perspective(60.0, aspect, 0.1, 100.0).mul(
    agate.Mat4.lookAt(eye, target, agate.Vec3.up));
const frustum = agate.Frustum.fromViewProjection(vp);
if (frustum.intersectsAABB(mesh.getWorldBoundingBox())) {
    // рисовать
}

// Луч пикинга:
const ray = agate.Ray.new(origin, dir); // направление нормируется
if (ray.intersectsAABBNormal(box)) |hit| {
    // hit.distance, hit.point, hit.normal
}
```

## API

### Векторы

```zig
pub const Vec2 = extern struct { x: f32 = 0, y: f32 = 0,
    pub const zero, one; pub fn new(x: f32, y: f32) Vec2 };
pub const Vec3 = extern struct { x, y, z: f32,
    pub const zero, one, up, down, left, right, forward, backward;
    pub fn new(x: f32, y: f32, z: f32) Vec3
    pub fn add(a: Vec3, b: Vec3) Vec3
    pub fn sub(a: Vec3, b: Vec3) Vec3
    pub fn scale(v: Vec3, s: f32) Vec3
    pub fn dot(a: Vec3, b: Vec3) f32
    pub fn cross(a: Vec3, b: Vec3) Vec3
    pub fn lengthSq(v: Vec3) f32
    pub fn length(v: Vec3) f32
    pub fn normalize(v: Vec3) Vec3        // ноль -> ноль, без NaN
    pub fn lerp(a: Vec3, b: Vec3, t: f32) Vec3
    pub fn distance(a: Vec3, b: Vec3) f32
    pub fn distanceSq(a: Vec3, b: Vec3) f32
    pub inline fn toSimd(v: Vec3) @Vector(4, f32)   // w = 0
    pub inline fn fromSimd(v: @Vector(4, f32)) Vec3
    pub inline fn eql(a: Vec3, b: Vec3) bool         // точное ==
    pub inline fn toArray(v: Vec3) [3]f32 };
pub const Vec4 = extern struct { x, y, z, w: f32,
    pub const zero, one; pub fn new(x, y, z, w: f32) Vec4
    pub inline fn toSimd / fromSimd / toArray(v) ...;
    pub inline fn add / sub / scale / lerp ... };    // через @Vector
pub inline fn lerp(a: f32, b: f32, t: f32) f32 // math.zig
```

`Vec3` — скалярная арифметика (компилятор автовекторизует простые циклы, явного SIMD нет кроме конверсий); `Vec4`/`Color4` — явные `@Vector`-операции. `forward = +Z`, `left = −X`. Всё O(1), без аллокаций.

### Матрицы

```zig
pub const Mat4 = extern struct { m: [16]f32 }; // column-major: m[col*4+row]
pub const identity: Mat4;
pub fn mul(a: Mat4, b: Mat4) Mat4              // = mulSimd (NEON/AVX через @Vector)
pub fn mulScalar(a: Mat4, b: Mat4) Mat4        // скалярный эталон (паритет в тестах)
pub fn mulSimd(a: Mat4, b: Mat4) Mat4
pub fn perspective(fov_y_deg: f32, aspect: f32, near: f32, far: f32) Mat4
pub fn orthographic(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) Mat4
pub fn lookAt(eye: Vec3, target: Vec3, up_arg: Vec3) Mat4
pub fn translation(v: Vec3) Mat4
pub fn scaling(s: Vec3) Mat4
pub fn rotationX(deg: f32) Mat4
pub fn rotationY(deg: f32) Mat4
pub fn rotationZ(deg: f32) Mat4
pub fn fromRotationTranslationScale(pos: Vec3, rot_deg: Vec3, scale_v: Vec3) Mat4 // T·Rz·Ry·Rx·S
pub fn fromQuatTranslationScale(pos: Vec3, q_in: Quat, scale_v: Vec3) Mat4
pub fn removeTranslation(self: Mat4) Mat4
pub fn getTranslation(self: Mat4) Vec3
pub fn invert(self: Mat4) ?Mat4               // null при |det| < 1e-8
pub fn transformPoint(self: Mat4, p: Vec3) Vec3     // w=1 + перспективное деление (SIMD)
pub fn transformDirection(self: Mat4, d: Vec3) Vec3 // w=0, без трансляции + normalize (SIMD)
pub fn projectPoint(self: Mat4, p: Vec3, screen_w: f32, screen_h: f32) ?Vec2 // null за камерой (w <= 0.001)
```

`fromRotationTranslationScale` — аналитическая сборка `T·Rz·Ry·Rx·S`: замкнутые формы элементов `Rz·Ry·Rx` из синусов/косинусов (порядок фиксирован и совпадает с `Quat.fromEulerDeg`), скейл по колонкам, трансляция в последний столбец — без матричных умножений (5 `mul` композиции убраны), побитово совпадает со старым путём (порядок произведений/сумм и канонизация `−0 → +0` повторяют `mulSimd`). Мотивация: самая горячая per-entity операция (~78% бенчмарка transform-пайплайна), цель ~2×. Паритет с кватернионным путём закреплён тестом `m_euler ≈ m_quat`, ортонормальность — тестом analytic properties. `fromQuatTranslationScale` — та же сборка через нормализованный кватернион со скейлом по колонкам. `perspective` — под клип `[0, 1]` (m[10] = `far/(near−far)`); `projectPoint` маппит NDC в пиксели с top-left `(0,0)`. `invert` — классический adjugate/det. `transformDirection` нормирует результат (нулевой вектор даст ноль через `normalize`-гард).

### Кватернионы

```zig
pub const Quat = struct { x, y, z: f32 = 0, w: f32 = 1,
    pub const identity: Quat };
pub fn normalize(q: Quat) Quat        // вырожденный -> identity
pub fn mul(a: Quat, b: Quat) Quat     // Гамильтон; применяет b, затем a (column-vector)
pub fn fromEulerDeg(e: Vec3) Quat     // R = Rz(z)·Ry(y)·Rx(x)
pub fn toEulerDeg(q: Quat) Vec3       // обратно; полюс |pitch| ~ 90°: roll в yaw, стабильно
pub fn conjugate(q: Quat) Quat
pub fn invert(q: Quat) Quat           // = conjugate (для unit)
pub fn rotateVec(q: Quat, v: Vec3) Vec3 // Родригес, без матриц
pub fn nlerp(a: Quat, b: Quat, t: f32) Quat // короткий путь + normalize
pub fn slerp(a: Quat, b: Quat, t: f32) Quat // короткий путь; ~совпавшие -> nlerp
```

`slerp` фолбэчится в `nlerp` при `cos ≥ 0.9995` или `sin ≈ 0`. `toEulerDeg` у полюса форсирует roll = 0 и считает yaw как `atan2(−R01, R11)` — рендер не дёргается. Знак при блендинге выравнивается (dot < 0 → негировать) — всегда кратчайшая дуга.

### Боксы, фрустум, лучи, цвета

```zig
pub const BoundingBox = struct { min: Vec3, max: Vec3,
    pub const zero; pub fn init(min_v: Vec3, max_v: Vec3) BoundingBox
    pub fn isValid(self: BoundingBox) bool // неубывающий + ненулевой объём по одной оси
    pub fn center(self: BoundingBox) Vec3
    pub fn extents(self: BoundingBox) Vec3   // ПОЛУразмеры!
    pub fn corners(self: BoundingBox) [8]Vec3
    pub fn transform(self: BoundingBox, m: Mat4) BoundingBox // тугой AABB через |M|·e
    pub fn intersects(self: BoundingBox, other: BoundingBox) bool
    pub fn merge(self: BoundingBox, other: BoundingBox) BoundingBox
    pub fn containsPoint(self: BoundingBox, pt: Vec3) bool
    pub fn closestPoint(self: BoundingBox, pt: Vec3) Vec3 };
pub const FrustumPlane = struct { normal: Vec3, d: f32,
    pub fn init(nx, ny, nz, d: f32) FrustumPlane }; // нормирует коэффициенты
pub const Frustum = struct { planes: [6]FrustumPlane, // Left/Right/Bottom/Top/Near/Far
    pub fn fromViewProjection(vp: Mat4) Frustum // near = row 2, far = row 3 − row 2 (для [0,1])
    pub fn intersectsAABB(self: Frustum, aabb: BoundingBox) bool // center/extents + SIMD
    pub fn intersectsAABB4(self: Frustum, c_x, c_y, c_z, e_x, e_y, e_z: @Vector(4, f32)) @Vector(4, bool) };
pub const RayHit = struct { distance: f32, point: Vec3, normal: Vec3 };
pub const TriangleHit = struct { distance: f32, point: Vec3, normal: Vec3, u: f32, v: f32 };
pub const Ray = struct { origin: Vec3, direction: Vec3, // new() нормирует
    pub fn new(origin: Vec3, direction: Vec3) Ray
    pub fn getPoint(self: Ray, t: f32) Vec3
    pub fn transform(self: Ray, m: Mat4) Ray
    pub fn intersectsAABB(self: Ray, box: BoundingBox) ?f32
    pub fn intersectsAABBNormal(self: Ray, box: BoundingBox) ?RayHit
    pub fn intersectsSphere(self: Ray, center: Vec3, radius: f32) ?f32
    pub fn intersectsSphereNormal(self: Ray, center: Vec3, radius: f32) ?RayHit
    pub fn intersectsTriangle(self: Ray, v0: Vec3, v1: Vec3, v2: Vec3) ?TriangleHit
    pub fn intersectsPlane(self: Ray, plane_point: Vec3, plane_normal: Vec3) ?f32 };
pub const Color3 = extern struct { r, g, b: f32 = 1,
    pub const white, black, red, green, blue, yellow, gray;
    pub fn new(r, g, b: f32) Color3
    pub fn scale(self: Color3, s: f32) Color3
    pub fn toColor4(self: Color3, a: f32) Color4 };
pub const Color4 = extern struct { r, g, b, a: f32 = 1,
    pub const white, black, transparent;
    pub fn new(r, g, b, a: f32) Color4
    pub inline fn toSimd / fromSimd / toArray ...;
    pub inline fn lerp(c1: Color4, c2: Color4, t: f32) Color4 // SIMD
    pub inline fn scale(c: Color4, s: f32) Color4 };          // SIMD
```

| Метод | Алгоритм | Краевые случаи |
|---|---|---|
| `intersectsAABB(Normal)` | slab Kay–Kajiya/Smits, ветви на `\|d\| < 1e-7` | луч внутри — `tmin = 0`; нормаль — ось входа |
| `intersectsSphere` | аналитический (`b²−c`), ранний выход `c>0 && b>0` | касание/внутри — корректный `t ≥ 0` |
| `intersectsTriangle` | Мёллер–Трумбор, куллинга нет (`\|det\| < 1e-7` — miss) | `t < 1e-5` — miss; нормаль развёрнута к лучу |
| `intersectsPlane` | `t = (p−o)·n / (n·d)`, `\|denom\| < 1e-6` — miss | `t < 0` — miss |
| `intersectsAABB4` | 4 бокса за 6 плоскостей одним `@Vector(4)` проходом | бит-совпадение со скаляром (тест) |
| `BoundingBox.transform` | центр через M, экстенты через `\|M\|·e` | точен для TRS, консервативен для проективных |

## Потоки и владение

Всё — `extern struct`/copyable value-типы без аллокаций и глобального состояния: свободно шарить между потоками. `Frustum` строится на кадр из VP и дальше только читается job'ами куллинга; `intersectsAABB4` берёт SoA-пачки (центры/полуразмеры как 6 векторов) — готовить данные пачками по 4. `Mat4.invert`/`projectPoint` возвращают optional — обрабатывать `null` (сингуляр/за камерой), а не разворачивать вслепую.

## Ошибки и краевые случаи

- Ошибок (`error`) модуль не возвращает вообще — только optionals: `invert → null`, `projectPoint → null`, `intersects* → null`.
- `normalize` нуля — ноль (не NaN); `Quat.normalize` вырожденного — identity.
- `eql` — точное `==`, для epsilon-сравнений писать свою обёртку.
- `extents` — ПОЛУразмеры (центр ± extents); `zero`-бокс невалиден (`isValid == false`).
- `fromViewProjection` предполагает `[0, 1]` глубину движка; с чужой OpenGL-матрицей (`[−1, 1]`) near/far плоскости неверны.
- `toEulerDeg` у полюса теряет roll (by design, yaw несёт сумму) — round-trip `euler→quat→euler` там не тождественен.
- `transformDirection` нормирует: направление нулевой длины схлопнется в ноль через гард.

## Производительность

- `Mat4.mul` — всегда SIMD-вариант (4 `splat` + FMA на колонку); `mulScalar` оставлен эталоном для тестов паритета.
- `transformPoint/Direction/projectPoint` — векторные, без веток кроме `w`-коррекции.
- `intersectsAABB` — branchless slab, ранний выход по осям; `intersectsAABB4` — ~6 векторных итераций на 4 бокса вместо 24 скалярных тестов (путь `visibility` и instancing-куллинга).
- Нулевая нагрузка на аллокатор во всём модуле; типы `extern` — ABI-стабильны для GPU-зеркал (см. `./texture.md`, `./particles.md` про `GpuParticleSlot`).
- Детерминизм: чистые f32-операции; побитовая идентичность — в пределах одной платформы/флагов компиляции (fused multiply-add может отличаться между CPU — для сетевого детерминизма фиксируйте target).

## Смотрите также

- `./cameras.md` — `lookAt/perspective` в пайплайне камеры.
- `./visibility.md` — HiZ/растеризатор поверх `Frustum.intersectsAABB4`.
- `./mesh.md` — мировые матрицы и `getWorldBoundingBox` для куллинга.
- `./scene.md` — пикинг через `Ray`.
- `./animation.md` — `Quat.slerp`, `fromQuatTranslationScale` в скиннинге.
