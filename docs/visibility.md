# Видимость (CPU occlusion culling)

> Путь: src/agate/visibility/ · Импорт: agate.visibility (HiZBuffer, SoftwareRasterizer, OcclusionCuller) · Потоки: главный/рендер-поток (CPU, синхронно в prepare).

## Что это

Модуль `visibility` — CPU-окклюзия для отсечения невидимых объектов до отправки на GPU: крупные «окклюдеры» (стены, здания) растеризуются программно в компактный буфер глубины, над ним строится Hi-Z пирамида (max-пирамида, как Hi-Z в GPU-окклюзии), затем AABB каждого кандидата проверяется против пирамиды — полностью закрытые отбрасываются. Всё консервативно: ложно-положительное «видим» допустимо (лишний draw call), ложное «невидим» — нет (пропавший объект).

Состав (`mod.zig` реэкспортирует все три типа):

| Файл | Тип | Роль |
|---|---|---|
| `hiz_buffer.zig` | `HiZBuffer` | буфер глубины 256×128 + 9 мипов пирамиды, тесты AABB |
| `rasterizer.zig` | `SoftwareRasterizer` | программная растеризация треугольников/боксов (namespace, без состояния) |
| `culler.zig` | `OcclusionCuller` | фасад кадра: окклюдеры → пирамида → запросы |
| `tests.zig` | — | модульные тесты (стена закрывает/не закрывает, пирамида, боксы) |

Модуль намеренно CPU-only и dependency-light: `hiz_buffer` зависит только от математики, `rasterizer` — от буфера, `culler` — от обоих. Интеграция со сценой — через вызовы `beginFrame`/`rasterizeOccluder*`/`endOccluders`/`isOccluded` из prepare-фазы (см. `./frame-pipeline.md`).

## Быстрый старт

```zig
const agate = @import("agate");

var culler = agate.visibility.OcclusionCuller.init();

// Начало кадра: камера известна.
culler.beginFrame(camera_view_proj);

// Окклюдеры: крупные статичные объекты (стены, корпуса).
// Бокс — дёшево (12 треугольников), меш — точно (реальные треугольники).
culler.rasterizeOccluderBox(wall_aabb, wall_world_mat);
culler.rasterizeOccluderMesh(positions, indices, mesh_aabb, mesh_world_mat);

// Финализация: построить Hi-Z пирамиду.
culler.endOccluders();

// Запросы: отбросить закрытое, остальное — в очередь рендера.
for (candidates) |obj| {
    if (culler.isOccluded(obj.world_aabb)) {
        culled += 1;
        continue;
    }
    render_queue.push(obj);
}
```

Эвристика выбора окклюдеров: 5–20 крупнейших ближних к камере статичных объектов. Мелочь в окклюдеры не давать — растеризация дороже выгоды.

## API

### HiZBuffer (`visibility/hiz_buffer.zig`)

```zig
pub const HiZBuffer = struct {
    pub const WIDTH: u32 = 256;
    pub const HEIGHT: u32 = 128;
    pub const NUM_MIPS: usize = 9;
    pub const TOTAL_FLOATS: usize = 43691; // сумма всех мипов
    pub const MipInfo = struct { ... };    // width/height/offset мипа
    pub const MIP_TABLE: [NUM_MIPS]MipInfo = ...;

    pub fn init() HiZBuffer
    pub fn clear(self: *HiZBuffer, depth: f32) void
    pub fn buildPyramid(self: *HiZBuffer) void
    pub fn testAABB(self: *const HiZBuffer, view_proj: Mat4, aabb: BoundingBox) bool
};
```

Буфер — плоский массив `TOTAL_FLOATS` f32 (256×128 = 32768 + 128×64 + … + 1×1 = 43691; ~170 KiB). `clear(1.0)` заливает дальней плоскостью. `buildPyramid` сворачивает max-фильтром 2×2 сверху вниз (O(пиксели), ~44K операций — дёшево). `testAABB` проецирует 8 углов бокса, выбирает мип по экранному размеру проекции и сравнивает глубину бокса с max-глубиной покрытия: возвращает `true` только если бокс целиком дальше записанной глубины во всех покрытых текселях. Пустой буфер (без окклюдеров) — всегда `false`.

Консервативность: max-пирамида + целочисленное покрытие текселей с округлением наружу гарантируют отсутствие ложных отсечений; ценой — объекты «на грани» считаются видимыми.

Выбор мипа в `testAABB` — по экранному размеру проекции бокса: крупные близкие объекты тестируются против детальных верхних мипов, мелкие дальние — против грубых нижних, где один тексел покрывает большую область. Это даёт O(1) запрос независимо от размера объекта: вместо сканирования всех покрытых текселей базового уровня читается несколько текселей выбранного мипа. Пирамида max-типа хранит для каждого текселя максимальную (самую дальнюю) глубину покрытия — объект невидим, только если его ближняя грань дальше max-глубины во ВСЕХ покрытых текселях.

### SoftwareRasterizer (`visibility/rasterizer.zig`)

```zig
pub const SoftwareRasterizer = struct {
    pub const ClipVertex = struct { ... }; // вершина после проекции + w для клиппинга
    pub fn rasterizeTriangle(hiz: *HiZBuffer, view_proj: Mat4, p0: Vec3, p1: Vec3, p2: Vec3, front_only: bool) u32
    pub fn rasterizeBox(hiz: *HiZBuffer, view_proj: Mat4, aabb: BoundingBox, world_mat: Mat4) u32
    pub fn rasterizeTriangles(hiz: *HiZBuffer, view_proj: Mat4, positions: []const Vec3, indices: []const u32, world_mat: Mat4) u32
};
```

Конвейер треугольника: transform в клип → отсев за дальней/боковыми плоскостями → перспективное деление → запись min-глубины в покрытые тексели базового мипа (пирамида строится позже одним `buildPyramid`). Возвращаемое значение — число задетых треугольников (для статистики). `front_only = true` у боксов включает backface-culling (6 видимых граней из 12 треугольников достаточно для замкнутого бокса). Вырожденные треугольники (нулевая площадь, за камерой целиком) дают 0 без записи. Сложность — O(треугольники × площадь в текселях при 256×128).

### OcclusionCuller (`visibility/culler.zig`)

```zig
pub const OcclusionCuller = struct {
    hiz: HiZBuffer,
    view_proj: Mat4 = Mat4.identity,
    occluder_count: u32 = 0,
    triangles_rasterized: u32 = 0,

    pub fn init() OcclusionCuller
    pub fn beginFrame(self: *OcclusionCuller, view_proj: Mat4) void
    pub fn rasterizeOccluderBox(self: *OcclusionCuller, aabb: BoundingBox, world_mat: Mat4) void
    pub fn rasterizeOccluderMesh(self: *OcclusionCuller, positions: []const Vec3, indices: []const u32, local_aabb: BoundingBox, world_mat: Mat4) void
    pub fn endOccluders(self: *OcclusionCuller) void
    pub inline fn isOccluded(self: *const OcclusionCuller, world_aabb: BoundingBox) bool
};
```

`beginFrame` сбрасывает буфер (`clear(1.0)`) и счётчики. `rasterizeOccluderMesh` при пустой геометрии (`positions.len < 3 or indices.len < 3`) молча пропускается — это штатный фолбэк «геометрии нет», а не ошибка (тогда вызывайте бокс-версию). `endOccluders` строит пирамиду только если окклюдеры были (`occluder_count > 0`), иначе `isOccluded` всегда `false` по быстрому пути. Поля `occluder_count`/`triangles_rasterized` — публичная статистика кадра (их же забирает профайлер в `culled_objects`).

## Интеграция со Scene и статистика

В prepare-фазе сцена держит один `OcclusionCuller` на кадр (поле сцены или локал кадра — пирамида одноразовая, хранить между кадрами смысла нет):

```zig
// Псевдопорядок в prepare (см. ./frame-pipeline.md):
culler.beginFrame(view_proj);
// 1. Выбрать окклюдеры: крупнейшие статичные меши в фрустуме,
//    отсортированные по близости к камере (первые N, N ≈ 5..20).
for (occluders) |occ| culler.rasterizeOccluderBox(occ.aabb, occ.world);
culler.endOccluders();
// 2. Frustum-cull, затем Hi-Z тест выживших:
for (meshes) |m| {
    if (frustumCulled(m)) continue;
    if (culler.isOccluded(m.world_aabb)) { stats.culled_objects += 1; continue; }
    queue.push(m);
}
```

Что смотреть в профайлере (`./profiler.md`): `culled_objects` вырос, `rendered_meshes`/`draw_calls` упали при тех же `triangles` видимого — каллинг окупается. Если `culled_objects == 0` при включённом каллере — либо сцена открытая (нечего закрывать), либо окклюдеры не выбираются (проверьте `occluder_count`), либо пирамида строится после запросов (нарушен порядок `endOccluders` → `isOccluded`).

## Потоки и владение

- Все три типа — plain value-structs без аллокаций и без владения: `HiZBuffer` держит массив внутри себя (170 KiB на стеке/внутри `OcclusionCuller` — не создавайте в тесноте стека рекурсивных функций, храните в сцене/кадре), слайсы позиций/индексов — заимствованы на время вызова.
- Вызовы строго в одном потоке (prepare): `beginFrame → rasterize* → endOccluders → isOccluded*`. `isOccluded` до `endOccluders` тестирует против пустой (cleared) пирамиды — всегда `false`, ошибки нет, но и пользы нет.
- `view_proj` копируется в каллере (`beginFrame`), матрица сцены дальше может меняться свободно. `world_mat` окклюдеров читается только внутри вызова.
- Многокадрового состояния нет: каждый кадр начинается с `beginFrame`, переносить каллер между кадрами нельзя (камера уехала — пирамида врёт).

## Ошибки и краевые случаи

- Ошибок как значений нет — все функции total: невалидный AABB (`!aabb.isValid()`) даёт 0 треугольников; пустая геометрия — пропуск; ноль окклюдеров — `isOccluded == false`.
- Объект, пересекающий near-плоскость, считается видимым (клиппинг против near не отсекает — консервативно).
- Очень близкие окклюдеры (стена перед носом камеры) заливают весь буфер — после этого всё «окклюжено», кроме пересекающего near. Это корректно, но вырождает culling в frustum-culling; отодвигайте порог выбора окклюдеров от камеры.
- Динамические окклюдеры (двери, машины) — легальны, но пирамида перестраивается каждый кадр заново: держите их в меньшинстве, основу должны составлять статики.
- Прозрачные объекты — не окклюдеры (сквозь них видно) и спорные окклюди: тестировать AABB прозрачных можно, но popping при растворении заметнее — обычно их исключают из запросов.
- Сцены без крупных объектов (пустырь, небо): `occluder_count == 0`, быстрый путь, накладные расходы — один `clear` 170 KiB.

## Производительность

- Память: 43691 f32 ≈ 170 KiB на каллер, на стеке владельца. В куче не нуждается, кэшу дружелюбна (пирамида сворачивается линейными проходами).
- CPU на кадр: `clear` (33K записей) + растеризация (треугольники × тексели при 256×128 — бокс-окклюдер стоит десятки-сотни текселей) + `buildPyramid` (~44K max-операций) + запросы O(1) каждый (проекция 8 углов + несколько чтений мипа). Ориентир: 10 боксов-окклюдеров + 1000 запросов — доли миллисекунды, на порядок дешевле frustum-cull по тысяче сфер? нет — сопоставимо, выигрыш в спасённых draw calls, а не в CPU.
- Разрешение 256×128 фиксировано (константы `WIDTH`/`HEIGHT`, таблица мипов предвычислена): мелкие объекты (дальние столбы, провода) пирамидой не ловятся — это frustum-culling и LOD, не сюда.
- Статистика для тюнинга: `occluder_count`, `triangles_rasterized` (сколько реально записано), `culled_objects` у профайлера. Если `triangles_rasterized == 0` при `occluder_count > 0` — окклюдеры за камерой/вырождены, выбор окклюдеров сломан.

## Смотрите также

- `./frame-pipeline.md` — prepare-фаза: где вызывается каллинг относительно frustum-cull и сборки очередей.
- `./scene.md` — `BoundingBox`, `world_aabb`, хранение трансформов.
- `./render-pipeline.md` — что экономит отсечение (draw calls, pipeline switches).
- `./profiler.md` — `culled_objects`, `rendered_meshes`: проверка эффекта в цифрах.
- `./cameras.md` — `view_proj`, из которого строится пирамида.
- `./mesh.md` — CPU-геометрия (`positions`/`indices`) для `rasterizeOccluderMesh`.
