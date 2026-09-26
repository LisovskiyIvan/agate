# Сериализация

> Путь: src/agate/serialization.zig + src/agate/serialization/ · Импорт: agate.serialization (root.zig) · Потоки: любой (чистые функции) + io_runner для async save/load.

## Что это

Модуль `serialization` — компактный бинарный снапшот сцены (формат AGSC, magic `"AGSC"`, текущая версия v3): capture сцены в `SceneState` → байты (в память или файл) → восстановление обратно. Задуман как «сейвы и префабы», а не как формат ассетов: сохраняется только то, что перечислено в `writer.zig`, всё остальное осознанно не сериализуется (геометрия, текстуры, анимации, физика — см. таблицу ниже).

Фасад `serialization.zig` — тонкий слой над листьями `serialization/`:

| Лист | Ответственность |
|---|---|
| `format.zig` | заголовок/версии (`MAGIC`, `VERSION`), лимиты, бинарные примитивы (`Writer`, `Reader`), кодек опциональных полей, контракт персистентности постпроцесса |
| `props.zig` | типы снапшота (`SceneState` и записи), кастомные свойства, entity ID/иерархия, material-kind хелперы |
| `writer.zig` | сцена → байты: `capture`, `serializeAlloc`, `saveFile`, async save |
| `reader.zig` | байты → сцена: `deserializeAlloc`, `restore`, `loadFile`, async load |

Все строки/слайсы в `SceneState` — owned (`allocator.dupe`), освобождаются через `SceneState.deinit`. Тесты — GPU-free: используются CPU-фикстуры `testing.zig` (`testScene`/`testMesh`), GPU-поля остаются `undefined` и никогда не дереференсятся в `capture`/`restore`.

## Быстрый старт

```zig
const agate = @import("agate");
const ser = agate.serialization;

// Сохранить сцену в файл (синхронно):
const state = try ser.capture(alloc, &scene);
defer state.deinit(alloc); // если serializeAlloc копирует — порядок см. ниже
try ser.saveFile(alloc, &state, "saves/slot1.agsc");

// Загрузить обратно:
var loaded = try ser.loadFile(alloc, "saves/slot1.agsc");
defer loaded.deinit(alloc);
ser.restore(&scene, &loaded);

// Через память (сеть, undo-буфер):
const bytes = try ser.serializeAlloc(alloc, &state);
defer alloc.free(bytes);
var back = try ser.deserializeAlloc(alloc, bytes);
defer back.deinit(alloc);

// Не блокировать кадр — через io_runner:
const save_task = try ser.saveFileAsync(alloc, &io_runner, state, "saves/slot1.agsc");
// ... позже: if (save_task.isDone()) { ok = save_task.isSuccess(); save_task.deinit(); }
const load_task = try ser.loadFileAsync(alloc, &io_runner, "saves/slot1.agsc");
```

Кастомные игровые свойства (уровень, счёт, флаги квестов) — часть снапшота:

```zig
// SceneState.properties: []GameProperty{ .key = "level_name", .value = "dungeon_02" }
```

## API

### Формат (`serialization/format.zig`)

```zig
pub const MAGIC: [4]u8 = .{ 'A', 'G', 'S', 'C' };
pub const VERSION: u32 = 3;
pub const MAX_ENTRIES: u32 = 1_000_000;
pub const MAX_STRING_BYTES: u32 = 8 * 1024 * 1024;
pub const MAX_FILE_BYTES: u64 = 256 * 1024 * 1024;
pub const DecodeError = error{ ... }; // BadMagic, UnsupportedVersion, Truncated, TooLarge, ...

pub const Writer = struct {
    pub fn bytes(self: *Writer, data: []const u8) !void
    pub fn byte(self: *Writer, v: u8) !void
    pub fn u32le / pub fn u64le / pub fn f32le / pub fn bool8 / pub fn vec3 / pub fn str(...) !void
};
pub const Reader = struct {
    pub fn readU8 / readU32 / readU64 / readF32 / readBool(...) DecodeError!T
    pub fn readRaw(self: *Reader, n: u32) DecodeError![]const u8
    pub fn readCount(self: *Reader) DecodeError!u32      // с проверкой MAX_ENTRIES
    pub fn readString(self: *Reader, allocator: std.mem.Allocator) (DecodeError || Allocator.Error)![]u8
    pub fn readVec3(self: *Reader) DecodeError![3]f32
};
// Кодек эволюции схемы: писать только перечисленные поля структуры:
pub fn writeField(w: *Writer, value: anytype) !void
pub fn readFieldAs(comptime T: type, r: *Reader) DecodeError!T
pub fn writeOptions(w: *Writer, options: anytype, comptime persisted: []const []const u8) !void
pub fn readOptions(comptime T: type, r: *Reader, comptime persisted: []const []const u8) DecodeError!T
pub const postprocess_persisted = [_][]const u8{ ... }; // какие поля PostProcessOptions хранятся
pub fn writePostProcess(w: *Writer, pp: *const PostProcessOptions) !void
pub fn readPostProcess(r: *Reader) DecodeError!PostProcessOptions
```

Формат — little-endian, строки — `u32 длина + байты`. `readCount`/`readString`/`readRaw` заранее отбивают злонамеренные длины лимитами (`MAX_ENTRIES`, `MAX_STRING_BYTES`), файл целиком — `MAX_FILE_BYTES = 256 MiB`. Версионность двоякая: `VERSION` заголовка (мажорная несовместимость) + options-codec (добавление полей без бампа версии: `persisted`-список фиксирует, какие поля структуры пишутся; неизвестные будущие поля читатель пропускает по длине).

### Типы снапшота (`serialization/props.zig`)

```zig
pub const StandardEntry = struct { ... };   // standard-материал меша
pub const PbrEntry = struct { ... };        // pbr-материал меша
pub const MaterialEntry = union(enum) { standard: StandardEntry, pbr: PbrEntry, ... };
pub const MeshEntry = struct {
    pub fn deinit(self: *MeshEntry, allocator: std.mem.Allocator) void
    ... // id/name, transform, material: MaterialEntry, custom props
};
pub const HemiEntry / DirectionalEntry / PointEntry / SpotEntry = struct {
    pub fn deinit(self: *HemiEntry, allocator: std.mem.Allocator) void ...
};
pub const ArcRotateEntry / FreeEntry / FollowEntry / TargetEntry / FlyEntry = struct { ... };
pub const CameraEntry = union(enum) { none, arc_rotate, free, follow, target, fly,
    pub fn deinit(self: *CameraEntry, allocator: std.mem.Allocator) void };
pub const RenderEntry = struct { skybox_enabled, skybox_exposure, shadows_enabled, shadow_softness, ibl_intensity };
pub const GameProperty = struct { key: []const u8 = "", value: []const u8 = "",
    pub fn deinit(self: *GameProperty, allocator: std.mem.Allocator) void };
pub const SceneState = struct {
    meshes: []MeshEntry = &.{},
    hemi: HemiEntry = .{},
    directional: ?DirectionalEntry = null,
    point_lights: []PointEntry = &.{},
    spot_lights: []SpotEntry = &.{},
    camera: CameraEntry = .none,
    render: RenderEntry = .{},
    postprocess: PostProcessOptions = .{},
    properties: []GameProperty = &.{},
    ... // entity IDs / hierarchy
    pub fn deinit(self: *SceneState, allocator: std.mem.Allocator) void
};
```

Что сохраняется, а что нет (явный контракт фасада):

| Сохраняется | НЕ сохраняется (by design) |
|---|---|
| меши: id/name, transform, значения материалов | геометрия (вершины/индексы) — только референс |
| свет: hemi/directional/point/spot параметры | текстуры и кубмапы (skybox/IBL содержимое) |
| камера: вариант + параметры (follow-ссылка `target_mesh` сброшена, `target_position` хранится) | скелеты/анимации, морф-таргеты |
| рендер: skybox вкл/экспозиция, shadows вкл/softness, ibl intensity | физтела, частицы, UI, инстансинг |
| постпроцесс: `postprocess_persisted`-поля | топология шаринга материалов (значения per-mesh) |
| кастомные `GameProperty` (строка→строка) | SSAO-конфиг, тонкий тюнинг теней сверх softness, GPU-хэндлы |
| entity ID + иерархия | bone attachments, follow-camera target link |

Меши при `restore` матчатся по id или имени; геометрия — ссылочная (сцена должна уже содержать меши с той же геометрией или догрузить их через `./loader.md`).

### Запись: сцена → байты (`serialization/writer.zig`)

```zig
pub fn capture(allocator: std.mem.Allocator, scene: *const Scene) !SceneState
pub fn serializeAlloc(allocator: std.mem.Allocator, state: *const SceneState) ![]u8
pub fn saveFile(allocator: std.mem.Allocator, state: *const SceneState, path: []const u8) !void
pub const AsyncSaveTask = struct {
    pub const State = enum(u8) { ... }; // pending/writing/completed/failed
    pub fn isDone(self: *const AsyncSaveTask) bool
    pub fn isSuccess(self: *const AsyncSaveTask) bool
    pub fn deinit(self: *AsyncSaveTask) void
};
pub fn saveFileAsync(allocator: std.mem.Allocator, runner: *jobs.TaskRunner, scene_state: SceneState, path: []const u8) !*AsyncSaveTask
```

Порядок владения: `capture` аллоцирует `SceneState` (все строки — dupes); `serializeAlloc`/`saveFile` только читают его — `state.deinit` после сериализации на вызывающем. `saveFileAsync` забирает `SceneState` внутрь задачи (передача по значению — вызывающий больше не владеет и не делает `deinit`). Сложность: `capture` O(меши + светила + свойства), `serializeAlloc` O(размер состояния); аллокации — пропорционально строкам и массивам.

### Чтение: байты → сцена (`serialization/reader.zig`)

```zig
pub fn restore(scene: *Scene, state: *const SceneState) void
pub fn deserializeAlloc(allocator: std.mem.Allocator, bytes: []const u8) !SceneState
pub fn loadFile(allocator: std.mem.Allocator, path: []const u8) !SceneState
pub const AsyncLoadTask = struct {
    pub const State = enum(u8) { ... };
    pub fn isDone(self: *const AsyncLoadTask) bool
    pub fn isSuccess(self: *const AsyncLoadTask) bool
    pub fn deinit(self: *AsyncLoadTask) void
};
pub fn loadFileAsync(allocator: std.mem.Allocator, runner: *jobs.TaskRunner, path: []const u8) !*AsyncLoadTask
```

`deserializeAlloc`/`loadFile` возвращают owned `SceneState` (вызывающий делает `deinit`). `restore` не аллоцирует под сцену заново, а применяет значения к существующим объектам (матчинг мешей по id/имени; отсутствующие в сцене записи пропускаются, лишние объекты сцены не трогаются). `restore` не возвращает ошибку — повреждённые данные должны быть отбиты раньше, на `deserializeAlloc` (`DecodeError`). Результат async-загрузки забирается из задачи после `isDone()` (успех → `isSuccess()`, затем забрать `SceneState` и `deinit` задачи).

## Потоки и владение

- `capture`/`serializeAlloc`/`deserializeAlloc`/`restore` — чистые функции, потокобезопасны при непересекающихся данных; `capture` требует стабильной сцены (вызывать из главного потока вне мутации сцены, либо под тем же exclusion, что и рендер-снапшот).
- `saveFile`/`loadFile` — синхронный файловый I/O, только из инструментов/загрузочных экранов; в игровом кадре — только `saveFileAsync`/`loadFileAsync` через `jobs.TaskRunner` (`Scene.io_runner`).
- Владение строками: всё в `SceneState` — owned dupes; двойной `deinit` — двойное освобождение, перемещение состояния — только передачей владения (как делает `saveFileAsync`). `bytes` для `serializeAlloc` — owned вызывающим (`alloc.free`).
- `GameProperty` — строки arbitrary bytes (UTF-8 по соглашению, не проверяется); ключи не уникализируются движком — дубликаты ключей легальны, интерпретация на игре.

## Ошибки и краевые случаи

- `DecodeError`: `BadMagic` (не AGSC), `UnsupportedVersion` (мажор v1/v2 читаются только своим кодом — см. «Совместимость»), `Truncated` (оборванный файл), `TooLarge`/`TooManyEntries`/`StringTooLong` (лимиты формата — защита от злонамеренных файлов). Ошибочный `SceneState` не возвращается частично — `deserializeAlloc` либо успех, либо ошибка (никаких полусостояний).
- Файл сверх `MAX_FILE_BYTES` отклоняется до парсинга (`loadFile`), не в середине.
- `restore` с несовпадающими мешами: записи без пары тихо пропускаются; это штатный путь «сейв от другой версии уровня», а не ошибка — проверяйте покрытие (число применённых записей) на стороне игры, если оно важно.
- Материалы восстанавливаются как значения (`MaterialEntry` per-mesh): шаринг двух мешей одним материалом после restore превращается в два равных значения (топология шаринга — в таблице «не сохраняется»).
- Постпроцесс при чтении проходит тот же `clamped()`-путь, что и в рантайме (см. `./postprocess.md`): out-of-range значения из старых файлов подтягиваются в валидные, а не роняют загрузку.
- Async-задачи: `deinit` до `isDone()` — use-after-free в worker-е; результат load-задачи забирать ровно один раз.
- Пустой `SceneState` (ноль мешей) — легален и сериализуется (заголовок + пустые секции).

### Совместимость AGSC v1–v3

- Текущий `VERSION = 3`. Читатель принимает только то, что понимает его `DecodeError`-ветка: v1/v2-файлы, использующие удалённые поля, отклоняются с `UnsupportedVersion`, а не читаются «как получится».
- Эволюция без бампа версии — через `persisted`-списки (`writeOptions`/`readOptions`, `postprocess_persisted`): новые поля дописываются в конец секции со своей длиной, старый читатель их пропускает. Правило для контрибьюторов: никогда не меняйте порядок и смысл уже persisted-полей — только добавляйте новые в конец и вносите в `persisted`-список.
- Строковые `GameProperty` — самый стабильный канал совместимости: игровой код, которому нужна вечная backward-совместимость, кладёт хрупкие данные туда (версию сейва игры, маппинги id), а не в структуру записей.

## Производительность

- `capture`: O(сцена), dup строк — единственные аллокации; держите имена короткими (они дюпятся и пишутся в файл).
- `serializeAlloc`: один проход, размер ≈ сумма строк + фиксированные структуры; `Writer` пишет напрямую в growable-буфер без промежуточных копий.
- `deserializeAlloc`: один проход + dup строк; `readString` проверяет длину ДО аллокации (защита от OOM на битых файлах).
- `restore`: O(записи × поиск пары) — матчинг по id через хеш, по имени линейный в худшем случае; для сцен с тысячами мешей предпочитайте стабильные id.
- Async-путь не копирует лишнего: `SceneState` переезжает в задачу, сериализация и запись — в worker-е, главный поток не блокируется вообще.

## Смотрите также

- `./scene.md` — структура `Scene`, матчинг мешей по id/имени.
- `./loader.md` — догрузка геометрии/текстур, на которые ссылается снапшот.
- `./postprocess.md` — `PostProcessOptions` и `clamped()`, `postprocess_persisted`-контракт.
- `./material.md` — значения `StandardEntry`/`PbrEntry`.
- `./cameras.md` — варианты камер и их параметры.
- `./lights.md` — параметры источников света.
- `./architecture.md` — `jobs.TaskRunner` / `io_runner`.
