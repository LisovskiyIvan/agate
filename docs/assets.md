# Ассеты и фоновые загрузки

> Путь: src/agate/assets.zig, src/agate/asset_manager.zig · Импорт: agate.assets, agate.AssetManager, agate.AssetTask, agate.AssetCache (root.zig) · Потоки: запросы с любого потока; декод на worker-пуле; GPU-финализация и drain только на context-потоке

## Что это

Два взаимодополняющих механизма. `assets.zig` — низкоуровневый `UploadQueue`: дедуплицированные async-загрузки текстур (файл/память) с декодом на пуле `jobs.TaskRunner` и дозированной GPU-выгрузкой (`drainCountedBudget`, бюджет `Scene.upload_budget_per_frame`). `asset_manager.zig` — высокоуровневый `AssetManager`: очередь задач (текст/бинарь/текстуры/меши в т.ч. async mesh-пул, кастомные задачи), прогресс-колбэки и кэш `AssetCache` (текст/бинарь/текстуры). Связка: сцена держит `uploads: ?UploadQueue` (плюс `io_runner` для файловых задач) и дренит её каждый кадр в prepare-фазе.

## Быстрый старт

```zig
const agate = @import("agate");

// Низкоуровнево: async-текстура с дедупликацией, слот патчится сам.
var queue = try agate.assets.UploadQueue.init(allocator, 2);
defer queue.deinit();
const slot = try queue.getOrRequestFile("assets/brick.png", .{ .srgb_to_linear = true }, .{});
slot.addTarget(&mat.albedo_texture); // патч при drain на context-потоке
// ... в кадре (context): _ = queue.drainCountedBudget(4, byte_budget);

// Высокоуровнево: менеджер с прогрессом.
var mgr = agate.AssetManager.init(allocator);
defer mgr.deinit();
_ = try mgr.addTextureTask("brick", "assets/brick.png", .{});
_ = try mgr.addMeshTaskAsync("hero", "assets/hero.glb", &scene);
try mgr.enableMeshAsync(2);
mgr.loadSync(); // или пошагово: while (!mgr.loadStep(4)) { /* прогресс */ }
std.debug.print("progress: {d}\n", .{mgr.progress()});
```

## API

### `UploadQueue` (`assets.zig`)

```zig
pub const TextureState = enum(u8) { ... }; // pending → ready → uploaded / failed / taken
pub const PendingTexture = struct {
    pub fn addTarget(self: *PendingTexture, slot: *?Texture) void; // слот-получатель патча
    pub fn take(self: *PendingTexture) ?Texture;                   // забрать владение
};
pub fn uploadBudgetExhausted(uploaded_count: usize, uploaded_bytes: u64, max_bytes: ?u64) bool;
pub const UploadQueue = struct {
    pub fn init(allocator: std.mem.Allocator, task_threads: usize) !UploadQueue; // 1–2 достаточно
    pub fn deinit(self: *UploadQueue) void; // сначала join воркеров, затем слоты
    pub fn requestFile(self, path: []const u8, options: Texture.Options, decode_opts: Texture.DecodeOptions) !*PendingTexture;
    pub fn findFile(self: *UploadQueue, path: []const u8) ?*PendingTexture; // живые слоты, кроме failed/taken
    pub fn getOrRequestFile(self, path, options, decode_opts) !*PendingTexture; // дедупликация
    pub fn requestMemory(self, bytes: []u8, options, decode_opts) !*PendingTexture; // владение байтами уходит очереди
    pub const DrainResult = struct { count: usize = 0, bytes: u64 = 0 };
    pub fn collectReady(self: *UploadQueue, out: []*PendingTexture) usize;
    pub fn drainCounted(self: *UploadQueue, max: ?usize) DrainResult;
    pub fn drainCountedBudget(self: *UploadQueue, max_count: ?usize, max_bytes: ?u64) DrainResult;
    pub fn drainBudget(self: *UploadQueue, max: usize) usize;
    pub fn drain(self: *UploadQueue) usize;
    pub fn release(self: *UploadQueue, p: *PendingTexture) void;
};
```

Контракт: слот валиден до `release`/`deinit`; таргеты регистрировать до drain. Состояния слота: ожидание декода → `ready` (CPU-буфер готов, ждёт GPU) → `uploaded` (хендл создан, таргеты пропатчены) либо `failed` (ошибка декода — слот мёртв, повторный запрос создаёт новый); `taken` — текстура забрана через `take` и слот больше не участвует в дедупликации. Спинлок покрывает только очередь/сканирование (батч ≤ 64 слотов — константа `drain_chunk`, чанк на стеке, без аллокаций в кадре); GPU-работа и патчи — после отпускания замка. `deinit` сначала гасит runner (join гарантирует завершение записей декодеров), затем освобождает слоты по состояниям (`uploaded` → `Texture.deinit`, `ready` → CPU-буферы). Дедупликация по пути: повторный запрос того же файла переиспользует in-flight или uploaded слот вместо второго декода и аплоада.

```zig
// Несколько материалов на одну текстуру — один декод и аплоад.
const shared = try queue.getOrRequestFile("assets/brick.png", .{ .srgb_to_linear = true }, .{});
shared.addTarget(&mat_a.albedo_texture);
shared.addTarget(&mat_b.albedo_texture);
```

Бюджет кадра: `Scene.upload_budget_per_frame = 4` текстуры за вызов плюс байтовый бюджет (`upload_byte_budget_per_frame`, см. `./scene.md`); остаток ждёт следующего кадра — отсюда pop-in при async-загрузках. Asset-масштабные неограниченные drain'ы идут чанками, не держа замок через GPU-работу.

### `AssetManager` (`asset_manager.zig`)

```zig
pub const TaskState = enum(u8) { ... }; // queued → running → done / failed
pub const TaskType = enum(u8) { ... };  // text / binary / texture / mesh / custom
pub const AssetTask = struct {
    pub fn takeMeshGeometry(self: *AssetTask) ?GeometryData; // забрать геометрию mesh-задачи
    pub fn isSuccess/isFailed(self: *const AssetTask) bool;
};
pub const AssetCache = struct {
    pub fn init/hasText/getText/putText(...) ...;
    pub fn hasBinary/getBinary/putBinary(...) ...;
    pub fn hasTexture/getTexture/putTexture(...) ...;
    pub fn clear/count(...) ...;
};
pub fn textureHandlesEqual(a: Texture, b: Texture) bool;
pub fn textureHandlesEqual(a: Texture, b: Texture) bool;
pub const TaskSuccessFn = *const fn (manager: *AssetManager, task: *AssetTask) void;
pub const TaskErrorFn = *const fn (manager: *AssetManager, task: *AssetTask, err: anyerror) void;
pub const ProgressFn = *const fn (manager: *AssetManager, remaining: usize, total: usize, task: *AssetTask) void;
pub const FinishFn = *const fn (manager: *AssetManager) void;
pub const AssetManager = struct {
    pub fn init(allocator: std.mem.Allocator) AssetManager;
    pub fn deinit/reset/resetAll(...) void;
    pub fn addTextFileTask(self, name, path: []const u8) !*AssetTask;
    pub fn addBinaryFileTask(self, name, path: []const u8) !*AssetTask;
    pub fn addTextureTask(self, name, path: []const u8, options: Texture.Options) !*AssetTask;
    pub fn addMeshTask(self, name, path: []const u8, scene: ?*Scene) !*AssetTask;
    pub fn addMeshTaskAsync(self, name, path: []const u8, scene: ?*Scene) !*AssetTask;
    pub fn enableMeshAsync(self: *AssetManager, thread_count: usize) !void;
    pub fn disableMeshAsync(self: *AssetManager) void;
    pub fn addCustomTask(self, ...) !*AssetTask;
    pub fn getTaskByName(self: *AssetManager, name: []const u8) ?*AssetTask;
    pub fn remainingCount/totalCount/progress/isDone/hasErrors(...) ...;
    pub fn loadStep(self: *AssetManager, max_tasks: usize) bool; // один шаг; true = ещё осталось
    pub fn loadSync(self: *AssetManager) void;                    // всё сразу (блокирует)
};
```

Задачи mesh без/с async: `addMeshTask` — синхронно в вызывающем потоке (только context), `addMeshTaskAsync` + `enableMeshAsync(n)` — парсинг геометрии на пуле, публикация через отложенный `uploadGeometry`-путь. `takeMeshGeometry` забирает результат для ручной публикации. `AssetCache.put*` возвращают канонические слайсы/регистрируют хендлы; ключи — обычно пути. `io_runner` сцены (`jobs.TaskRunner`, 1 поток, ленивая инициализация в `Scene.init`) обслуживает файловые чтения.

```zig
// Пошаговая загрузка с прогрессом (экран загрузки).
while (!mgr.loadStep(4)) {
    ui.setProgress(mgr.progress());
    ui.setStatus(mgr.remainingCount(), mgr.totalCount());
}
if (mgr.hasErrors()) ui.showError();
```

`progress()` возвращает долю 0..1, `isDone()` — ноль оставшихся, `getTaskByName` — доступ к задаче по имени для `takeMeshGeometry`/инспекции состояния.

## Потоки и владение

Запросы (`request*/add*Task`) — с любого потока (очередь под спинлоком). Декод — worker-пул (`task_threads` 1–2: один декод монолитен, параллелизм только по изображениям). Drain/финализация (`drain*`, `loadStep` для GPU-части, патч слотов) — только context-поток. Байты `requestMemory` уходят во владение очереди (освобождаются после декода). `PendingTexture` живёт до `release`; забранная через `take` текстура — владение вызывающего (`Texture.deinit` сам). `AssetManager.reset` чистит задачи, `resetAll` — плюс кэш.

## Ошибки и краевые случаи

| Ситуация | Поведение |
|---|---|
| Декод не удался | Слот `failed`; `findFile` его не возвращает, повторный запрос идёт заново |
| GPU-аплоад не удался | Слот остаётся `ready`, ретрай в следующем drain |
| Бюджет исчерпан | `uploadBudgetExhausted` → остаток на следующий кадр (pop-in, не ошибка) |
| Двойной запрос одного пути | Один слот на всех (`getOrRequestFile`), таргеты копятся |
| `deinit` при летящих декодах | Join воркеров первым — записи завершены до освобождения слотов |
| Async-mesh без `enableMeshAsync` | Выполняется синхронным путём |
| Ошибка задачи менеджера | `TaskState.failed`, `hasErrors()`, error-колбэк; остальные задачи продолжают |

## Производительность

- Параллелизм только по изображениям/файлам: 1–2 worker-потока достаточно; больше — только при десятках одновременных файлов.
- Дозированный drain (4 текстуры + байтовый кап на кадр) держит hitch кадра в бюджете ценой растянутого pop-in; для экранов загрузки — неограниченные чанковые drain'ы вне кадра.
- Дедупликация (`findFile`/`getOrRequestFile`, `image_cache` лоадера, `AssetCache`) убирает повторные декоды общих albedo/emissive и общих файлов между mesh-задачами.
- `DrainResult{ count, bytes }` (`scene.frame_uploads`) — observed-метрика для тюнинга бюджетов.

## Смотрите также

- `./texture.md` — форматы, декод, `Texture.Options`/`DecodeOptions`
- `./loader.md` — async-текстуры glTF поверх `UploadQueue`
- `./scene.md` — `scene.uploads`, `io_runner`, prepare-drain
- `./runtime.md` — кадровые фазы, где происходит drain
- `./profiler.md` — `frame_uploads`, учёт загрузок
