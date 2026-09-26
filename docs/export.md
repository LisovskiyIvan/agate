# Экспорт сцены

> Путь: src/agate/export/ · Импорт: agate.export_obj, agate.export_stl, agate.export_glb (root.zig) · Потоки: любой поток (чистый CPU, только `cpu_positions/cpu_indices` + матрицы)

## Что это

Три writer'а, сериализующие меши сцены в байты на аллокаторе вызывающего: Wavefront OBJ (+ MTL-компаньон), STL (ASCII/бинарный) и GLB (glTF 2.0 binary). Все читают только CPU-зеркала меша (`cpu_positions`, `cpu_indices`) и опционально мировые матрицы — GPU-буферы не затрагиваются. Форматы покрывают разные задачи: OBJ — обмен с DCC (материалы упрощённо), STL — 3D-печать/слайсеры (только триангуляция), GLB — round-trip обратно в движок через `SceneLoader`.

## Быстрый старт

```zig
const agate = @import("agate");

const meshes: []const *agate.Mesh = scene.meshes.items;

// OBJ + MTL (сохранить рядом: scene.obj + scene.mtl).
const obj_text = try agate.export_obj.writeObjAlloc(allocator, meshes, .{});
defer allocator.free(obj_text);
const mtl_text = try agate.export_obj.writeMtlAlloc(allocator, meshes, .{});
defer allocator.free(mtl_text);

// STL для печати, с мировыми трансформами.
const stl = try agate.export_stl.writeStlAlloc(allocator, meshes, .{
    .binary = true,
    .apply_world_transform = true,
    .solid_name = "part",
});
defer allocator.free(stl);

// GLB для round-trip.
const glb = try agate.export_glb.writeGlbAlloc(allocator, meshes, .{
    .emit_materials = true,
});
defer allocator.free(glb);
const one = try agate.export_glb.writeGlbMeshAlloc(allocator, mesh, .{});
defer allocator.free(one);
```

## API

### Общие правила

Все три writer'а пропускают меши без `cpu_positions` (нет CPU-зеркала — нечего писать). Опции везде включают `visible_only` (экспорт только видимых) и `apply_world_transform` (запечь `getWorldMatrix()` в вершины). Имена санитизируются: каждый байт вне `[A-Za-z0-9._-]` становится `_`, пустое имя — `"mesh"`; длина сохраняется, коллизии после санитизации молча мерджатся. Все возвращают владеющий `[]u8` (освобождает вызывающий).

### OBJ (`export/obj.zig`)

```zig
pub const mtl_filename: []const u8 = "scene.mtl";
pub const ObjExportOptions = struct {
    visible_only: bool = false,
    apply_world_transform: bool = false,
    emit_materials: bool = true,
};
pub fn sanitizeNameAlloc(allocator, name: []const u8) ![]u8;
pub fn writeObjAlloc(allocator, meshes: []const *Mesh, options: ObjExportOptions) ![]u8;
pub fn writeMtlAlloc(allocator, meshes: []const *Mesh, options: ObjExportOptions) ![]u8;
```

`writeObjAlloc` пишет `mtllib scene.mtl` (если есть хоть один материал и `emit_materials`), затем на меш: `o <имя>`, `usemtl <материал>`, вершины `v`, дедуплицированные нормали `vn` (квантованные, кэш по `[3]i32`), грани `f v//n` с глобальными 1-based базами. Треугольники: индексные тройки (out-of-range отбрасываются) либо суп из позиций. Нормали граней вычисляются из позиций (не из вершинных атрибутов). `writeMtlAlloc`: один `newmtl` на distinct санитизированное имя (первый побеждает при коллизии), `Kd` из diffuse/albedo, `d` из alpha; меши без материалов вклада не дают, пустой вход — пустой выход. MTL-файл сохраняется рядом с OBJ под именем `mtl_filename` (`"scene.mtl"`) — writer только возвращает байты, раскладку по файлам делает вызывающий.

```zig
// Сохранение пары OBJ+MTL рядом (псевдокод).
const obj = try agate.export_obj.writeObjAlloc(arena, meshes, .{ .emit_materials = true });
const mtl = try agate.export_obj.writeMtlAlloc(arena, meshes, .{});
try std.fs.cwd().writeFile(.{ .sub_path = "scene.obj", .data = obj });
try std.fs.cwd().writeFile(.{ .sub_path = agate.export_obj.mtl_filename, .data = mtl });
```

Дедуп нормалей согласован с OBJ-импортёром (ключ вершины — (позиция, uv, нормаль)): повтор той же нормали на смежных копланарных гранях не сплитит углы и не раздувает round-trip.

### STL (`export/stl.zig`)

```zig
pub const StlExportOptions = struct {
    visible_only: bool = false,
    apply_world_transform: bool = false,
    binary: bool = false,
    solid_name: []const u8 = "agate",
};
pub fn sanitizeNameAlloc(allocator, name: []const u8) ![]u8;
pub fn writeStlAsciiAlloc(allocator, meshes: []const *Mesh, options: StlExportOptions) ![]u8;
pub fn writeStlBinaryAlloc(allocator, meshes: []const *Mesh, options: StlExportOptions) ![]u8;
pub fn writeStlAlloc(allocator, meshes: []const *Mesh, options: StlExportOptions) ![]u8; // диспетчер по .binary
```

ASCII: `solid <имя>` … `facet normal`/`outer loop`/`vertex`×3 … `endsolid`. Пример фасетки:

```
solid part
  facet normal 0 0 1
    outer loop
      vertex 0 0 1
      vertex 1 0 1
      vertex 0 1 1
    endloop
  endfacet
endsolid part
```

Бинарный: 80-байтный заголовок (санитизированное имя, остаток — нули), u32 little-endian счётчик, на треугольник — 12 f32 (нормаль + 3 вершины) + u16 атрибутов (0). Размер всегда `84 + 50·tris`; пустой вход — 84-байтный заголовок с нулевым счётчиком (импортёр на нём сообщает `NoGeometry`). Счётчик u32: свыше 4G треугольников — `error.TooLarge`.

### GLB (`export/glb.zig`)

```zig
pub const GlbExportOptions = struct {
    visible_only: bool = false,
    apply_world_transform: bool = false,
    emit_materials: bool = true,
};
pub fn writeGlbAlloc(allocator, meshes: []const *Mesh, options: GlbExportOptions) ![]u8;
pub fn writeGlbMeshAlloc(allocator, mesh: *Mesh, options: GlbExportOptions) ![]u8;
```

Пишет валидный glTF 2.0 binary (magic `0x46546C67`, версия 2, JSON-чанк + BIN-чанк): позиции из `cpu_positions` с вычисленными min/max, индексы, материалы (упрощённо: base color, metallic/roughness). Round-trip покрыт тестом через cgltf-парсинг обратно. Внутренние константы чанков/типов (`CHUNK_TYPE_JSON/BIN`, `COMPONENT_UNSIGNED_SHORT/INT/FLOAT`, `TARGET_ARRAY_BUFFER/ELEMENT_ARRAY_BUFFER`) — приватные, в документации не фиксируются.

Выбор формата:

| Задача | Формат | Почему |
|---|---|---|
| Обмен с DCC (Blender, Maya) | OBJ + MTL | Читается везде, материалы упрощённо |
| 3D-печать, слайсеры | STL binary | Только триангуляция, минимальный размер |
| Отладка геометрии глазами | STL ASCII | Человекочитаемые фасетки |
| Round-trip в движок | GLB | Обратно через `SceneLoader.appendGlb` без потерь индексов |

## Потоки и владение

Writer'ы — чистые функции: входные меши только читаются (включая `getWorldMatrix()` при `apply_world_transform`), выходной буфер владеет вызывающий. Потокобезопасны при отсутствии конкурентной мутации мешей. GPU не используется — экспорт работает и без sg-контекста.

## Ошибки и краевые случаи

| Ситуация | Поведение |
|---|---|
| Меш без `cpu_positions` | Пропускается (все три формата) |
| Out-of-range индексы (OBJ/STL) | Треугольник отбрасывается |
| Пустое имя / не-ASCII | `"mesh"` / `_`-замены |
| Коллизия санитизированных имён | Тихий мердж (OBJ-объекты, MTL — первый побеждает) |
| Пустой вход STL binary | 84-байтный заголовок, count 0 |
| > u32 треугольников (STL binary) | `error.TooLarge` |
| Невидимые меши при `visible_only` | Пропускаются |

## Производительность

- Сложность O(V + T) на меш; форматирование через `print` на треугольник — для asset-масштабов приемлемо, для гигантских сцен предпочитать binary STL/GLB вместо ASCII OBJ.
- Память: выходной буфер + небольшие scratch-структуры (кэш нормалей OBJ — хешмап граней); стриминга в файл нет — очень большие сцены держат весь текст в памяти.
- `apply_world_transform` добавляет один проход матричного трансформа позиций.

## Смотрите также

- `./loader.md` — парсинг OBJ/STL/PLY и glTF, round-trip гарантии
- `./mesh.md` — `cpu_positions/cpu_indices`, мировые матрицы
- `./material.md` — что сохраняется в MTL/GLB-материалы (упрощённо)
- `./scene.md` — `scene.meshes` как источник экспорта
