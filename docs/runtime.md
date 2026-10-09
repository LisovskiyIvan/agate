# Runtime и потоки кадра

> Путь: src/agate/runtime.zig, handoff.zig, gpu_thread.zig, jobs.zig, observable.zig, gpu_timing.zig, gpu_upload_meter.zig · Импорт: agate.runtime, agate.Runtime, agate.Handoff, agate.gpu_thread, agate.jobs, agate.gpu_timing (root.zig) · Потоки: game producer (update/produce) / context render (begin/finish/render) / audio / io_runner

## Что это

`runtime.zig` — тонкий фасад жизненного цикла кадра для threaded-приложений. Он владеет порядком вызовов (producer `tryClaimBuildSlot → build → stageUi → publish` на игровой стороне, `beginStagedPrepare` только по свежему полному билду на контекстной стороне, `finishStagedPrepare` + `render` без лока, fallback `renderReuse`), стартом/остановкой воркера и честными счётчиками. Хосты (демо `src/main.zig`, sandbox) компоновкой из него собирают цикл вместо ручной хореографии.

Staged-only: свежего producer-build нет — `beginStagedPrepare` возвращает `null`, никакого live-чтения. `Scene.buildPreparedFrame()` — `tryClaimBuild → stageUi → publish` полностью замороженного кадра. `Render` никогда сам не готовит свежий кадр: только подготовленный front или явный `renderReuse` валидного front. Latest-wins лизы сохраняются.

Порядок init: `agate.gpu_thread.markContextThread()` один раз до `Scene.initInto` — объект помечает GPU-владельца до сцены. Headless без маркера: только CPU-cleanup, это не живое GPU-владение.

## Быстрый старт

```zig
const agate = @import("agate");

var runtime: agate.Runtime = agate.Runtime.init();
defer runtime.deinit();

// Игровой поток: один продюсер.
fn gameLoop() void {
    runtime.gameLock();
    // ... simulate(scene, dt) ...
    _ = runtime.produceBuild(scene); // build BEFORE begin: claim → build → stageUi → publish
    scene.recordUpdateTime(dt_ms);
    runtime.gameUnlock();
}

// Простой путь (демо src/main.zig работает на этих двух вызовах):
// game: runtime.update(&scene, &tick_ctx, Tick.run);
// context: switch (runtime.renderFrame(&scene)) { .prepared, .reused, .skipped, .busy }
```

Контекстный поток должен один раз вызвать `agate.gpu_thread.markContextThread()` в init-колбэке до спавна игрового потока. `Runtime.spawnWorker(entry)` спавнит воркер с `fn () void`; `quiesce()`/`deinit()` джойнят его (мьютекс при этом НЕ держать).

## API

### runtime.zig — `Runtime`, `Metrics`, `BeginResult`, `FrameResult`

```zig
pub const BeginResult = struct {
    claim: ?Scene.PrepareClaim,
    busy: bool,
    wait_ns: u64,
    held_ns: u64,
};
pub const FrameResult = enum { prepared, reused, skipped, busy };

pub const Metrics = struct {
    producer_builds: u64 = 0,
    producer_skips: u64 = 0,
    begins: u64 = 0,
    begin_empty: u64 = 0,
    begin_busy: u64 = 0,
    finishes: u64 = 0,
    cancels: u64 = 0,
    reuses: u64 = 0,
    skipped_presents: u64 = 0,
};

pub const Runtime = struct {
    mutex: jobs.Mutex = .{},
    running: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    thread: ?std.Thread = null,
    metrics: Metrics = .{},
    lock_wait_ns: u64 = 0,
    producer_exclusion: bool = false,

    pub fn init() Runtime;
    pub fn setProducerExclusion(self: *Runtime, excluded: bool) void;
    pub fn setLockWaitNs(self: *Runtime, ns: u64) void;
    pub fn spawnWorker(self: *Runtime, comptime entry: fn () void) bool;
    pub fn quiesce(self: *Runtime) void;
    pub fn shouldRun(self: *const Runtime) bool;
    pub fn deinit(self: *Runtime) void;
    pub fn gameLock(self: *Runtime) void;
    pub fn gameUnlock(self: *Runtime) void;
    pub fn produceBuild(self: *Runtime, scene: *Scene) bool;
    pub fn produceBuildWithHostBytes(self: *Runtime, scene: *Scene, host_bytes: ?[]const u8) bool;
    pub fn update(self: *Runtime, scene: *Scene, ctx: anytype, comptime tick: fn (@TypeOf(ctx)) void) bool;
    pub fn renderFrame(self: *Runtime, scene: *Scene) FrameResult;
    pub fn tryRunLocked(self: *Runtime, comptime work: fn () void) bool;
    pub fn beginPrepare(self: *Runtime, scene: *Scene) BeginResult;
    pub fn beginPrepareWith(self: *Runtime, scene: *Scene, comptime work: fn () void) BeginResult;
    pub fn finishPrepare(self: *Runtime, scene: *Scene, claim: Scene.PrepareClaim) void;
    pub fn cancelPrepare(self: *Runtime, scene: *Scene, claim: Scene.PrepareClaim) void;
    pub fn reuseIfConsumable(self: *Runtime, scene: *Scene) bool;
};
```

Группировка:

| Группа | Методы | Смысл |
|---|---|---|
| Жизненный цикл воркера | `init`, `spawnWorker`, `quiesce`, `shouldRun`, `deinit` | Владеют порядком старт/стоп; `spawnWorker` возвращает `false` при ошибке спавна (деградация в single-threaded) |
| Простой путь | `update`, `renderFrame` | Весь кадр для обычных приложений; `update` держит мьютекс только при включённом exclusion |
| Game-side producer | `gameLock`/`gameUnlock`, `produceBuild`, `produceBuildWithHostBytes` | One-liner `tryClaimBuildSlot → build → stageUi → publish`; `false` — все не-front слоты заняты (counted skip) |
| Advanced context | `beginPrepare`, `beginPrepareWith`, `finishPrepare`, `cancelPrepare`, `reuseIfConsumable`, `tryRunLocked` | Инструментированные хосты; finish/cancel — one-shot, только на живом GPU-владельце |
| Тюнинг исключения | `setProducerExclusion`, `setLockWaitNs` | Переключение lock-free/exclusion и бюджет аквизиции (0 = чистый non-blocking try) |

Сложность/аллокации/ошибки: все методы O(1) поверх scene-операций, аллокаций нет (счётчики — обычные целые, не атомики: каждый метод выполняется на одном owner-потоке). `produceBuild` не возвращает ошибку — только `bool`. `renderFrame` вычитает `wait_ns` из `prepare_ms`, чтобы contention не двоился в prepare-тайминге.

### handoff.zig — `Handoff(T, slot_count)`

```zig
pub fn Handoff(comptime T: type, comptime slot_count: usize) type;
// методы инстанса:
pub fn claim(self: *Self) ?usize;
pub fn slot(self: *Self, i: usize) *T;
pub fn publish(self: *Self, i: usize) void;
pub fn releasePublished(self: *Self) void;
pub fn takeLatest(self: *Self, out: *T) bool;
```

Latest-wins mailbox: publisher заполняет слот и публикует, единственный consumer забирает newest complete и отпускает остальные. Старые незабранные payloads дропаются — медленный consumer не копит stale-кадры, быстрый publisher не блокируется (переиспользует свободные слоты или скипает). Слово слота — `seq << 2 | state` в одном atomic (`free / writing / published`), минимум 2 слота (comptime-параметр; стресс-тест гоняет 3). Копия — seqlock-валидация: слово читается (acquire) до и после `memcpy`, при расхождении скан рестартует; consumer помнит `floor` (newest delivered seq) и отбрасывает late-visible stale-кадры, так что доставки строго возрастают. `T` — plain data struct без указателей в себя. Глава файла честно фиксирует: ротация остаётся 2 для frame mailboxes; третий слот без consumer-side pin/lease протокола — retention без читателя.

### gpu_thread.zig — маркер контекстного потока

```zig
pub fn markContextThread() void;
pub fn resetContextThreadForTest() void; // test-хелпер
pub fn isOnContextThread() bool;
pub fn assertOnContextThread() void; // Debug + ReleaseSafe; в ReleaseFast/Small — no-op
```

Один раз из sokol init-колбэка до спавна потоков; дальше только чтения (spawn даёт happens-before). Без маркера (юнит-тесты, тулы) любой поток считается контекстным — сохраняется синхронное поведение. `assertOnContextThread` стоит на входах render-фазы.

### jobs.zig — пул fork-join, мьютекс, TaskRunner

```zig
pub const Pool = struct {
    pub const min_len_for_workers: usize = 4096;
    pub fn recommendedWorkerCount() usize; // min(max(cpus-1,1),8)
    pub fn init(allocator: std.mem.Allocator, worker_count: usize) !*Pool;
    pub fn deinit(self: *Pool) void;
    pub fn workerCount(self: *const Pool) usize;
    pub fn forkJoin(self: *Pool, comptime C: type, ctx: *C,
        comptime run: fn (ctx: *C, start: usize, end: usize) void, len: usize) void;
};
pub var global: ?*Pool = null;
pub fn parallelFor(pool: ?*Pool, comptime C: type, ctx: *C,
    comptime run: fn (ctx: *C, start: usize, end: usize) void, len: usize) void;
pub threadlocal var test_wait_observer: ?*const fn () void = null; // только тесты

pub const Mutex = struct { // SRWLock на Windows, pthread_mutex на POSIX
    pub fn lock(self: *Mutex) void;
    pub fn tryLock(self: *Mutex) bool;
    pub fn tryLockWithin(self: *Mutex, timeout_ns: u64) bool; // sleep-цикл шагом ≤50us
    pub fn unlock(self: *Mutex) void;
    pub fn deinit(self: *Mutex) void;
};
pub fn monoNs() u64;   // QPC / clock_gettime(CLOCK_MONOTONIC)
pub fn sleepNs(ns: u64) void; // Sleep / nanosleep

pub const TaskRunner = struct {
    pub fn init(allocator: std.mem.Allocator, thread_count: usize) !*TaskRunner;
    pub fn deinit(self: *TaskRunner) void; // дренирует очередь, затем join
    pub fn queuedCount(self: *TaskRunner) usize;
    pub fn post(self: *TaskRunner, ctx: *anyopaque, run: *const fn (ctx: *anyopaque) void) void;
};
pub fn SpscRing(comptime T: type, comptime capacity: usize) type; // push/pop/len
```

Контракт детерминизма: job, пишущий только в свои индексы, бит-идентичен при любом числе воркеров; шедулинг не наблюдаем через результаты. Правила: только CPU-работа, никакого `sg.*`, никаких аллокаций shared state, никакого вложенного `parallelFor` в воркере; `forkJoin` — single-producer (только владелец пула). `parallelFor` ниже `min_len_for_workers` или при `pool == null` идёт inline. `TaskRunner` — отдельная история (asset decode в фоне): `forkJoin` спинит вызывающего до конца, длинным задачам там не место.

### observable.zig — `Observable`, `ObservableValue`, `EventBus`, `Signal`

Реэкспорта в `root.zig` нет (internal-leaf; импортируется напрямую как `agate.observable` через модульный путь). Сигнатуры:

```zig
pub const EventState = struct { mask: u32, skip_next_observers: bool, ... };
pub fn stopPropagation(self: *EventState) void;
pub const ObserverOptions = struct { mask: u32, insert_first: bool, unregister_on_first_call: bool, order: i32 };
pub const ObserverId = u32; pub const INVALID_OBSERVER_ID: ObserverId = 0;
pub fn Observer(comptime T: type) type;
pub fn Observable(comptime T: type) type; // add/addTyped/addSimple/addFn/addFnSimple/addOnce/...,
//   remove/removeCallback/clear, notify/notifyMask/notifyVoid/notifyWithState/notifyObservers,
//   hasObservers/countObservers/hasObserver, init/deinit
pub fn ObservableValue(comptime T: type) type; // get/set/setSilent/subscribe*/unsubscribe
pub fn typeId(comptime T: type) u64; // zero-overhead статический адресный id
pub const EventBus = struct { // init/deinit/clear, subscribe*/publish*/hasSubscribers/countSubscribers/...
    pub fn subscribe(...); pub fn subscribeWithState(...); pub fn subscribeFn(...);
    pub fn subscribeTopic(...); pub fn subscribeTopicFn(...); pub fn subscribeOnce(...);
    pub fn unsubscribe(self: *EventBus, sub: Subscription) bool;
    pub fn publish(self: *EventBus, event: anytype) void;
    pub fn publishWithState(self: *EventBus, event: anytype, state: *EventState) void;
    pub fn publishTopic(self: *EventBus, topic: []const u8, event: anytype) void;
    ...
};
pub fn Signal(comptime T: type) type;
```

Гарантии: mutation-safe итерация (tombstones при re-entrant add/remove внутри notify), приоритеты `order` + `insert_first`, bitmask-фильтр `mask`, `stopPropagation`, zero-alloc fast path `hasObservers`, compile-time type ids. Аллокации: подписки аллоцируют из заданного аллокатора (`NoAllocatorProvided` без него); нотификация без подписчиков — без аллокаций.

### gpu_timing.zig — GPU-тайминги кадра

```zig
pub const Pass = enum(c_int) { shadow = 0, main = 1, post = 2 };
pub fn parseEnvFlag(value: ?[]const u8) bool; // "1"/"true"/"on"/"yes"
pub fn setEnabled(on: bool) void;   // любое место, C-флаг — лениво при валидном контексте
pub fn isEnabled() bool;
pub const Sample = struct { ms: f32, frame_index: u32 };
pub const FrameScope = enum { none, command_buffer, native_pass_span, pass_sum };
pub const Capabilities = struct { frame: bool, passes: bool, frame_scope: FrameScope };
pub fn capabilities() Capabilities; // поддержка device, независимо от intent
pub fn pollFrameSample() ?Sample;
pub fn pollPassSample(pass: Pass) ?Sample;
pub fn beginPass(pass: Pass) void;
pub fn endPass(pass: Pass) void;
```

Лист-модуль (std + sokol), без цикла импорта. `setEnabled` меняет только atomic intent;
begin/end/poll применяют его на **context thread**. Замеры асинхронные, задержка не
фиксирована в один кадр. `Sample.frame_index` — номер завершённой sokol submission,
не `Scene.frame_id`: повторный poll может вернуть тот же sample. `null` означает
off/headless/not-ready/unsupported, а `ms == 0` с ненулевым индексом —
настоящий квантизованный замер. `pollFrameMs/pollPassMs` удалены: только optional
`Sample`, наличие данных — по `null`, не по `ms > 0`.

Metal использует command-buffer frame time и, при поддержке timestamp counters,
phase spans; WebGPU — timestamp-query spans; desktop GL — `GL_TIME_ELAPSED` фаз и
их сумму **из одной submission**. Это разные scopes, не взаимозаменяемые frame
метрики. Включение — программно или `AGATE_GPU_TIMINGS=1`. Для WebGPU дополнительно
перед `sapp.run` задать `.wgpu_gpu_timing_enabled = gpu_timing.isEnabled()`:
features нельзя включить после создания device. Неподдерживаемый адаптер остаётся
рабочим, таймеры сообщают unavailable. `sg.shutdown()` сам освобождает timer
resources; вручную выключать их перед shutdown больше не требуется.
Подробности и GPU-гейт — [gpu-timing.md](./gpu-timing.md).

### gpu_upload_meter.zig — счётчик байт динамических GPU-апдейтов

```zig
pub fn record(bytes: usize) void;   // рядом с каждым фактическим sg.updateBuffer/appendBuffer
pub fn takeAndReset() u64;          // раз в кадр: prepare сбрасывает, render забирает в stats
pub fn peek() u64;                  // без сброса, тесты/отладка
```

Атомарный `u64` (стейджинг инстансов идёт с воркеров пула). Uncounted-budget метрика: на троттлинг текстурного стриминга (`upload_byte_budget_per_frame = 8 MiB`) не влияет, только честная диагностика в `SceneStats.updated_bytes_frame`.

## Потоки и владение

- Game producer (один, обязателен): `gameLock` поперёк всего тика (`simulate` + `produceBuild`); `update` берёт лок только при включённом exclusion. Пишет живые регистры, mailboxes, `pending_update_ms`; в `stats` напрямую — никогда.
- Context render: `beginPrepare*` (по умолчанию без мьютекса; с exclusion — bounded acquire `lock_wait_ns`, 0 = чистый try), затем `finishPrepare` + `render` разблокированно. Только контекст вызывает `begin/finish/cancel` (движок ассертит) и `markContextThread` до `Scene.initInto`.
- `finish`/`render` перекрываются со следующим update свободно. `update`/`renderFrame` одинаковы для worker и single-thread (`--no-threads` — тот же staged-алгоритм); `producer_exclusion` — только mutex-диагностика тех же замороженных данных, не откат render-пути.
- Lock-free контракт требует: все payloads заморожены в слот до `publish` (staging после publish — гонка с latch); missing/failure-пакеты — coherent-empty, живую canvas-геометрию никто не читает; host live-reads переехали в `stageHostBytes` или доказано context-owned; никакого registry add/remove поперёк in-flight latch (commit guards держат когерентность, но контракт приложения это запрещает).
- Jobs-воркеры: только CPU чанки через atomic cursor; `pending` считает участников (воркеры + вызывающий), вызывающий спинит до `pending == 0`. `TaskRunner` (io_runner, asset decode) — отдельные треды, `post` под parking-lot мьютексом, `deinit` дренирует.
- `quiesce` вызывать с НЕДЕРЖАЩИМСЯ мьютексом (воркер может быть припаркован на его аквизиции); после join game-owned plain-поля безопасно читать с этого потока.

## Ошибки и краевые случаи

- `beginPrepare` → `claim == null, busy == false`: свежего build нет — не live-fallback; caller делает reuse/skip. `busy == true` (только exclusion-mode): мьютекс не взялся в бюджет — тот же reuse/skip, но счёт как contention (`begin_busy`), не idle.
- `produceBuild → false`: все не-front слоты pinned/claimed (consumer lagging) — counted skip, контекст переиспользует front.
- Каждый успешный begin — ровно один `finish` или `cancel`. Нарушение пейринга (stale/double/потерянный токен) не клинит конвейер: лог err + защитный release активного claim (кадр теряется, begin-ы продолжаются). Finish/cancel — one-shot: контекст — живой зарегистрированный GPU-владелец.
- `render` между begin и finish видит `frame_prepared == false` и дропает present — держать пару смежно.
- Самозахваченный мьютекс (non-recursive): bounded begin сообщает `busy`, не блокируется.
- Первые кадры: `reuseIfConsumable` → false (нечего переиспользовать), `renderFrame` → `.skipped`, пока первый build не готов.
- `prepareSerial`/legacy-диагностика: удалены; single-thread идёт тем же staged-путём (`--no-threads` — тот же алгоритм).
- `Observable.add` без аллокатора → `error.NoAllocatorProvided`; `Handoff.claim` → null при насыщении (вместо блокировки — дроп/drain/skip).
- `gpu_timing` вне контекста или выключенный — `null` sample, без паник.

## Производительность

- `parallelFor` порог `min_len_for_workers = 4096`: короче — inline дешевле, чем будить воркеров. Чанки `max(len/((workers+1)*4), 1)` — баланс против contention на cursor. `recommendedWorkerCount = min(max(cpus-1,1),8)`.
- `Mutex.tryLockWithin` — sleep-цикл шагом ≤50us (не `pthread_mutex_timedlock` из-за darwin-деклараций); превышение бюджета — максимум один sleep-шаг + jitter шедулера. Контекстный acquire ограничен, игровой — всегда блокирующий (тик дропать нельзя).
- `renderFrame`/`beginPrepare` отчитываются `wait_ns`/`held_ns`: `prepare_ms` считается минус wait, contention не двоится.
- `claim`/`takeLatest` — lock-free CAS-сканы; `releasePublished` — монотонные фильтр-лоады (промах = неполный drain → skip кадра, не коррупция); torn-copy скрывается seqlock-ретраем вместо блокировки publisher.
- GPU-тайминги default OFF: без timer allocations/encoding, без синхронного ожидания
  результатов. Для WebGPU opt-in нужен до создания device; в браузере, где нет
  process environment, intent устанавливает приложение.

## Смотрите также

- `./frame-pipeline.md` — слоты, freeze→commit, host_bytes, staged prepare детально
- `./architecture.md` — карта модулей и слоёв движка
- `./scene.md` — `Scene.BuildClaim`/`PrepareClaim` со стороны сцены
- `./profiler.md` — `Profiler.recordFrame` в хвосте `render`
- `./assets.md` — `TaskRunner`/io_runner и async-загрузка
- `./visibility.md` — occlusion culler и queue builds поверх `parallelFor`
