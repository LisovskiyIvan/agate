# Профайлер

> Путь: src/agate/profiler.zig + src/agate/profiler/ · Импорт: agate.profiler (root.zig) · Потоки: главный поток (recordFrame) + io_runner (асинхронная запись отчётов).

## Что это

Модуль `profiler` — бортовой flight recorder движка: каждый кадр складывает фазовые CPU-метрики и счётчики рендера в кольцевой буфер (`FrameRecord`), по запросу считает сводку сессии (`SessionSummary`), память (`MemorySnapshot`), диагностику узких мест (`DiagnosticFinding`) и генерирует отчёты в трёх форматах — HTML, Markdown и Chrome-trace JSON. Запись файлов на диск идёт только через очередь `ReportWriteTask → io_runner`, никогда синхронно из кадра.

Фасад `profiler.zig` реэкспортирует листья `profiler/`:

| Лист | Ответственность |
|---|---|
| `types.zig` | данные: `FrameRecord`, `MemorySnapshot`, `SessionSummary`, диагностика, `dominantPhase`; без sibling-импортов |
| `core.zig` | владелец `Profiler`: поля рекордера, жизненный цикл, glue отчётов/сохранения, форвардеры |
| `recording.zig` | `recordFrame` — вставка в кольцевой буфер |
| `snapshot.zig` | `captureMemorySnapshot` — перепись CPU/GPU-памяти |
| `summary.zig` | `summarize` — статистика сессии + pacing |
| `diagnostics.zig` | `analyze` — поиск бутылочных горлышек |
| `report.zig` | генераторы HTML/Markdown/Chrome-trace поверх плоских данных (импорт только `types`) |
| `report_queue.zig` | `ReportWriteTask` + `enqueueReportWrite` — enqueue-only половина lock-free окна |

CPU-фазы — время сабмита, GPU-поля — ПОСЛЕДНИЙ ЗАВЕРШЁННЫЙ timestamp sample,
с переменной async задержкой. Они не взаимозаменяемы. GPU availability определяется
submission id (0 = отсутствует), а не `ms > 0`: квантизация может дать валидный ноль.
Сводка усредняет уникальные available GPU samples, не CPU-кадры с повторным poll.
Metal command-buffer time, WebGPU native-pass span и GL pass sum имеют разные scopes:
см. [gpu-timing.md](./gpu-timing.md). Chrome-trace GPU events привязаны ко времени
CPU-наблюдения и помечены submission id; это не синхронизированная GPU timeline.

## Быстрый старт

```zig
const agate = @import("agate");

var prof = agate.profiler.Profiler.init(alloc);
defer prof.deinit();
prof.start();

// В конце каждого кадра (stats: *const SceneStats собирается сценой):
prof.recordFrame(frame_id, &stats);

// Разовый снимок памяти (владеет профайлер, latched mailbox — см. ниже):
const mem = try prof.captureMemorySnapshot(&scene);

// Диагностика узких мест (возвращает слайс — освободить вызывающему):
const findings = try prof.analyze(mem, alloc);
defer { for (findings) |*f| f.deinit(alloc); alloc.free(findings); }

// Синхронные отчёты (для отладки/тестов):
try prof.saveReports(&scene, "reports/session"); // .html + .md + .json

// Асинхронно из игры — только через очередь, без блокировки кадра:
const bundle = try prof.generateReportsAlloc(alloc); // ReportBundle: html/md/json байты
defer bundle.deinit(alloc);
const task = try prof.enqueueReportWrites(&io_runner, "reports/session", bundle, .all);
_ = task; // task.isDone()/isSuccess() — опрос; deinit после завершения
```

## API

### Запись кадров (`profiler/types.zig`, `recording.zig`, `core.zig`)

```zig
pub const FrameRecord = struct {
    frame_index: u64 = 0,
    timestamp_us: u64 = 0,
    dt_s: f32 = 0,
    fps: f32 = 0,               // = 1/dt_s
    frame_interval_ms: f32 = 0, // = dt_s * 1000, реальный pacing
    total_frame_ms: f32 = 0,    // СУММА CPU-фаз (время сабмита, не wall time)
    update_ms: f32 = 0, physics_ms: f32 = 0, prepare_ms: f32 = 0,
    shadow_ms: f32 = 0, main_ms: f32 = 0, post_ms: f32 = 0,
    gpu_frame_ms: f32 = 0,      // последний completed GPU sample, переменный лаг
    gpu_shadow_ms: f32 = 0, gpu_main_ms: f32 = 0, gpu_post_ms: f32 = 0, // supported Metal/WebGPU/GL
    gpu_frame_submit: u32 = 0, // 0 = unavailable; валидный ноль ms имеет ненулевой id
    gpu_shadow_submit: u32 = 0, gpu_main_submit: u32 = 0, gpu_post_submit: u32 = 0,
    gpu_frame_scope: agate.gpu_timing.FrameScope = .none,
    draw_calls: u32 = 0, triangles: u32 = 0, pipeline_switches: u32 = 0,
    rendered_meshes: u32 = 0, culled_objects: u32 = 0,
    uploaded_textures: u32 = 0, uploaded_bytes: usize = 0, // стриминг текстур, лимит 8 MiB
    updated_bytes: usize = 0,   // динамические GPU-буферы (sg.update/append), uncounted-budget
};
pub const PhaseCulprit = struct { ... };
pub fn dominantPhase(record: FrameRecord) PhaseCulprit
```

Жизненный цикл рекордера:

```zig
pub fn init(allocator: std.mem.Allocator) Profiler
pub fn deinit(self: *Profiler) void
pub fn setMaxFrames(self: *Profiler, max: usize) void
pub fn start(self: *Profiler) void
pub fn linearize(self: *Profiler) void
pub fn stop(self: *Profiler) void
pub fn reset(self: *Profiler) void
pub fn isRecording(self: *const Profiler) bool
pub fn recordFrame(self: *Profiler, frame_id: u64, stats: *const SceneStats) void
```

`recordFrame` — O(1) вставка в ring (перезаписывает старейшее при переполнении); пишет только пока `start`ed. `setMaxFrames` задаёт глубину окна (по умолчанию хватает на репрезентативный отрезок; увеличивайте для длинных сессий ценой памяти). `linearize` упорядочивает ring в хронологию для генераторов. `SceneStats` — структура сцены с теми же фазовыми полями (см. `./scene.md`).

### Сводка и диагностика (`summary.zig`, `diagnostics.zig`)

```zig
pub const SessionSummary = struct { ... }; // средние/p95/max по фазам, pacing-джиттер
pub fn summarize(self: *const Profiler) SessionSummary
pub const DiagnosticSeverity = enum { ... };
pub const DiagnosticFinding = struct {
    pub fn deinit(self: *DiagnosticFinding, allocator: std.mem.Allocator) void
};
pub fn analyze(self: *const Profiler, memory: ?*const MemorySnapshot, allocator: std.mem.Allocator) ![]DiagnosticFinding
```

`summarize` — чистая функция поверх записанных кадров (средние, хвосты, джиттер pacing по `frame_interval_ms`). `analyze` возвращает owned-слайс находок (память под строки внутри — вызывающий освобождает каждый `deinit` + весь слайс); `memory = null` — анализ только по фазам, без memory-части. Типичные находки: доминирующая фаза кадра, просадки pacing, переполнение лимита текстурного аплоада, рост памяти.

### Память (`snapshot.zig`, `types.zig`)

```zig
pub const TextureMemoryRecord = struct { name: []const u8, width: u32, height: u32, num_mips: u32, ... };
pub const MeshMemoryRecord = struct { ... };
pub const RenderTargetRecord = struct { ... };
pub const MemorySnapshot = struct {
    pub fn deinit(self: *MemorySnapshot, allocator: std.mem.Allocator) void
    ...
};
pub fn captureMemorySnapshot(self: *Profiler, scene: *const Scene) !*const MemorySnapshot
```

Снимок — перепись CPU/GPU-памяти: текстуры (имя, размеры, мипы, байты), меши, рендер-таргеты. Хранится ВНУТРИ профайлера (latched mailbox): указатель валиден до следующего `captureMemorySnapshot`/`reset`/`deinit` — не освобождайте и не храните долше этого окна. Ошибка только `OutOfMemory`.

### Отчёты (`report.zig`, `core.zig`)

```zig
// Чистые генераторы (report.zig) — только данные + аллокатор:
pub fn hasGpuData(frames: []const FrameRecord) bool
pub fn hasGpuPassData(frames: []const FrameRecord) bool
pub fn formatBytes(allocator: std.mem.Allocator, bytes: usize) ![]u8
pub fn generateReportHtml(...) ![]u8
pub fn generateReportMd(...) ![]u8
pub fn generateTraceJson(frames: []const FrameRecord, allocator: std.mem.Allocator) ![]u8

// Glue (core.zig) — методы Profiler:
pub fn generateReportHtml(self: *const Profiler, scene: ?*const Scene, allocator: std.mem.Allocator) ![]u8
pub fn generateReportMd(self: *const Profiler, scene: ?*const Scene, allocator: std.mem.Allocator) ![]u8
pub fn generateTraceJson(self: *const Profiler, allocator: std.mem.Allocator) ![]u8
pub fn saveReportHtml(self: *const Profiler, scene: ?*const Scene, path: []const u8) !void
pub fn saveReportMd(self: *const Profiler, scene: ?*const Scene, path: []const u8) !void
pub fn saveTraceJson(self: *const Profiler, path: []const u8) !void
pub fn saveReports(self: *const Profiler, scene: ?*const Scene, base_path: []const u8) !void
pub const ReportBundle = struct { html: []u8, md: []u8, json: []u8,
    pub fn deinit(self: *ReportBundle, allocator: std.mem.Allocator) void };
pub const ReportFiles = struct { html: bool = true, md: bool = true, json: bool = true,
    pub const all: ReportFiles = .{}; pub const html_only ...; pub const trace_only ...; };
pub fn generateReportsAlloc(self: *const Profiler, allocator: std.mem.Allocator) !ReportBundle
pub fn enqueueReportWrites(...) // см. ниже
```

Три формата: **HTML** — интерактивный отчёт (графики фаз, таблицы, память); **Markdown** — тот же контент текстом (для CI/ревью); **Chrome-trace JSON** — открывается в `chrome://tracing` / Perfetto (события по фазам кадра). `scene = null` — отчёт только по фазам без секции сцены/памяти. Синхронные `save*` — для тестов и инструментов; из игрового кадра — только асинхронный путь.

### Асинхронная запись (`report_queue.zig`)

```zig
pub const ReportWriteTask = struct {
    pub const State = enum(u8) { pending = 0, writing = 1, completed = 2, failed = 3 };
    allocator: std.mem.Allocator,
    path: []u8,   // owned dupe пути
    data: []u8,   // owned байты отчёта
    state: std.atomic.Value(State) = ...,
    bytes_written: usize = 0,
    err_name: ?[:0]const u8 = null,
    pub fn isDone(self: *const ReportWriteTask) bool
    pub fn isSuccess(self: *const ReportWriteTask) bool
    pub fn deinit(self: *ReportWriteTask) void
};
pub fn enqueueReportWrite(allocator: std.mem.Allocator, runner: *jobs.TaskRunner, path: []const u8, data: []u8) !*ReportWriteTask
```

Enqueue-only окно: профайлер только ставит задачу в `Scene.io_runner`, пишет worker-поток (`std.Io.Dir.cwd().writeFile`), главный поток опрашивает `isDone`/`isSuccess`. Владение: при успехе `enqueueReportWrite` забирает `data` (не освобождать!), `path` всегда дюпится; `deinit` задачи — после `isDone()`. Правило PendingTexture-инварианта: после терминального store в `state` полей не трогать. Метод `Profiler.enqueueReportWrites` — та же идея для всего бандла сразу (пути `base_path.html/.md/.json`, выбор через `ReportFiles`).

## Потоки и владение

- **Владение.** `Profiler` владеет ring-буфером кадров и последним `MemorySnapshot` (latched: один живой указатель; следующий `captureMemorySnapshot` перезаписывает). `ReportBundle` владеет тремя буферами (`deinit` с тем же аллокатором). `ReportWriteTask` владеет `path` + `data` и самой задачей (`allocator.destroy` в `deinit`).
- **Главный поток:** `recordFrame`, `summarize`, `analyze`, генерация (`generate*Alloc`), постановка (`enqueueReportWrites`). Генерация аллоцирует (строки отчёта) — не вызывать в realtime-колбэках.
- **io_runner:** только `runWriteTask` (запись файла + терминальный store). Никакого доступа к `Profiler` из worker-а — задача несёт всё своё (lock-free окно: данные пересекают границу один раз, дальше только атомик состояния).
- Опрос завершения — из главного потока (`isDone` с acquire). `err_name` читается только после `isDone() == true` и неуспеха.

## Ошибки и краевые случаи

- `captureMemorySnapshot` без сцены невозможен — требуется `*const Scene`; при `OutOfMemory` старый latched-снимок остаётся валиден.
- `analyze` с `memory = null` — легально, memory-секция пропускается (а не ошибка).
- `save*`/`generate*` на пустом рекордере (0 кадров) возвращают валидный пустой отчёт, а не ошибку — проверяйте `summarize().frame_count == 0` сами.
- `generateTraceJson` без GPU-данных пишет только CPU-события (`hasGpuData`/`hasGpuPassData` проверяют available submission ids; Metal per-pass требует поддержки timestamp counters).
- `enqueueReportWrite` при ошибке создания НЕ забирает `data` — остаётся вашей (освободите сами); при успехе — не трогать.
- `deinit` задачи до `isDone()` — use-after-free в worker-е. Всегда опрашивайте до освобождения.
- `setMaxFrames(0)` — вырожденное окно: `recordFrame` ничего не хранит, отчёты пустые.
- `reset` во время висящей async-записи безопасен: задача владеет своими байтами (копией из бандла), а не указателями в профайлер.

## Производительность

- `recordFrame`: O(1), одна структура копируется в ring; аллокаций ноль. Можно вызывать каждый кадр без опасений.
- `summarize`: O(frames) проход, без аллокаций (скаляры в `SessionSummary`).
- `analyze`: O(frames + memory_records) + аллокации строк находок; вызывайте по требованию (кнопка «диагностика», а не каждый кадр).
- `captureMemorySnapshot`: O(текстуры + меши + таргеты); таблицы имен дюпятся — не чаще раза в секунду в живой игре.
- Генераторы отчётов: O(frames) + форматирование; HTML самый тяжёлый (графики инлайнятся). Генерируйте в фоне паузы/по кнопке, пишите только через `io_runner`.
- Память рекордера: `max_frames × sizeof(FrameRecord)` (~сотня байт на кадр — тысячи кадров стоят копейки).

## Смотрите также

- `./scene.md` — `SceneStats`: источник фазовых данных для `recordFrame`.
- `./frame-pipeline.md` — что измеряют фазы update/physics/prepare/shadow/main/post.
- `./gpu-timing.md` — GPU-таймеры: scopes, completed submission ids и availability.
- `./architecture.md` — `io_runner` и потоковая модель движка.
- `./serialization.md` — снапшоты сцены vs снимки памяти профайлера (разные задачи).
- `./texture.md` — `uploaded_bytes` и лимит 8 MiB стриминга текстур.
