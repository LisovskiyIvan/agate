# Потоки симуляции (threading)

> Путь: `src/threading.zig`, `src/scene/physics_worker.zig`, `src/scene/physics_layer.zig` (`kickWorker`/`joinWorker`/`workerCounters`), `src/scene/sim_api.zig` (`kickPhysics`/`joinPhysics`), `src/scene/frame_api.zig` (`Scene.update`), `src/physics/world.zig` (`parallel_window`, tripwire) · Импорт: `agate.ThreadingOptions` (`root.zig`) · Потоки: game (scene-update) + scene-owned physics worker

## Что это

Модель потоков на сегодня (см. `./architecture.md`, `./runtime.md`): game-поток мутирует живые регистры и строит кадр (`Scene.update` → producer build), context-поток готовит и рисует (`beginStagedPrepare` → `finishStagedPrepare` → `render`), пул `jobs.global` разбирает data-parallel чанки (fork-join), `TaskRunner`'ы (`uploads`, `io_runner`) крутят фоновые декоды и файловый I/O. У пользователя три независимые оси конфигурации:

| Ось | Ручка | Смысл |
|---|---|---|
| Кадровый конвейер | `Runtime` (host-структура) | serial staged-путь vs threaded producer/consumer (lock-free по умолчанию, `setProducerExclusion(true)` — только mutex-диагностика тех же данных) |
| Симуляция | `ThreadingOptions.physics` (этот документ) | `inline` (дефолт) vs `dedicated`: шаг физики на Scene-owned воркере, overlapped с физико-независимыми фазами внутри `Scene.update` |
| Внутри шага физики | `ThreadingOptions.physics_workers` | Box3D-воркеры: `1` (дефолт, serial) или `2..32` — Box3D сам распараллеливает step через встроенный C-scheduler. Ортогонально `physics`: `dedicated` уводит шаг с game-потока, `physics_workers` параллелит сам шаг |
| Воркер-пул | `jobs.global` | data-parallel чанки (анимационные скелеты от `min_skeletons_for_workers = 4`, инстанс-стейджинг, culling); детерминирован при disjoint-записях. **Opt-in**: движок не создаёт пул; хост ставит `jobs.global = try jobs.Pool.init(alloc, jobs.Pool.recommendedWorkerCount())` до первого кадра и уничтожает после `runtime.quiesce` (sandbox делает это в `main.zig`) |

```zig
var scene = agate.Scene.init(allocator);
scene.threading.physics = .dedicated;
scene.threading.physics_workers = 2; // Box3D: 1 step thread + 1 worker
```

(`inline` — Zig-ключевое слово, поэтому тег quoted: `.@"inline"` в местах использования, имя сохранено как в API.)

## Dedicated physics: семантика fork-join

Обычный порядок `Scene.update`: camera → lights → physics → animations → soft bodies → particles → decals → auto-exposure → snapshot. В `.dedicated`:

camera → lights → **[kick physics worker с dt]** → animations → soft bodies → particles → decals → **[join]** → auto-exposure → snapshot.

Kick и join выполняются на scene-update (game) потоке; воркер исполняет только `PhysicsWorld.step(dt)` в скобках `beginParallelWindow()`/`endParallelWindow()`. Хенд-офф — мьютекс + condvar с явным автоматом idle/work/done/stop (см. `physics_worker.zig`): без аллокаций на кадр, без busy-burn (воркер паркуется на condvar между киками). Zig 0.16 не отдаёт публичных мьютекса/condvar в `std.Thread` и не умеет `.name` в `SpawnConfig`, поэтому файл повторяет приём `jobs.zig:ParkingLot` (pthread на POSIX, SRWLock + CONDITION_VARIABLE на Windows), а имя потока (`agate-physics`) воркер ставит себе сам через `pthread_setname_np` best-effort.

Переключение режима работает в рантайме в обе стороны: dedicated→inline паркует воркер (работа не постится, поток живёт до `deinit`), inline→dedicated спавнит лениво на первом нужном апдейте. Воркер джойнится в `PhysicsIntegration.deinit` ДО `world.deinit()`. Если воркер не спавнится (`builtin.single_threaded`, wasm, ошибка спавна) — перманентная деградация в inline (стиль `Runtime.spawnWorker`: один принт, дальше тихо) + счётчик `fallbacks`.

Наблюдаемая семантика одинакова в обоих режимах: `pending_physics_ms` сторится ровно раз за апдейт (после join — из профиля отработавшего шага); мира нет — кика нет, сторится 0 как сегодня; `dt <= 0.0001` — инлайн early-return без thread roundtrip.

## Box3D внутри шага: `physics_workers`

Это не поток Agate, а встроенный scheduler Box3D. В `b3WorldDef.workerCount` передаётся `ThreadingOptions.physics_workers` при создании мира (`PhysicsWorld.initWithWorkerCount`); при `> 1` Box3D создаёт `N-1` native-потоков (`b3CreateScheduler`), а вызывающий поток участвует в step. Отдельный task-system не нужен: Box3D использует свой C-scheduler (`enqueueTask`/`finishTask` не задаются). Потоки живут вместе с миром и останавливаются в `b3DestroyWorld`.

- **Дефолт — `1`**: поведение и число потоков не меняются, wasm/`single_threaded` принудительно клампятся к `1` (scheduler требует native threads).
- **`2..32`**: ручка для тяжёлой физики; ставится до `scene.enablePhysics`/первого `createRigidBody` (более позднее значение применяется при следующем `enablePhysics`). Менять на ходу можно через `PhysicsWorld.setWorkerCount` только вне шага и вне parallel-window.
- **Детерминизм**: на бенче ниже финальные позиции/повороты **побитово совпали** для 1/2/4 воркеров и повторяются run-to-run. Это измеренный факт для текущего Box3D и сцены, а не формальная гарантия; воспроизводится командой ниже.
- **Цена**: `N-1` OS-потоков на каждый `PhysicsWorld`, создаются при создании мира. Для `dedicated` + `physics_workers = 2` физика занимает 2 потока (Scene-воркер + 1 Box3D-воркер), game-поток в это время делает анимации/частицы.

Замер (`examples/physics_workers_bench.zig`, ReleaseFast, 120 шагов, медиана, 512 одинаковых ящиков в стопке; ms/step):

| bodies | 1 worker | 2 workers | 4 workers |
|---|---|---|---|
| 64   | 0.023 | 0.022 | 0.023 |
| 256  | 0.135 | 0.101 | 0.083 |
| 512  | 0.361 | 0.241 | 0.191 |
| 1024 | 0.732 | 0.471 | 0.350 |
| 2048 | 1.560 | 0.953 | 0.675 |

На мелких сценах выигрыша нет (оверхед scheduler'а), с ~256 тел — 1.3–1.6× на двух воркерах; 4 воркера дают до ~2.3×. Проверка: `cd examples && zig build run-physics-workers -Doptimize=ReleaseFast -- --bodies 1024 --steps 120 --reps 5 --workers 1,2,4`.

## Изоляция от render-потока

Render/context-поток — единственный, кто вызывает `sg.*` и читает/пишет render-owned prepared payloads. Ни один наш или физический поток в него не заходит:

| Поток | Что делает | Чего не делает |
|---|---|---|
| context/render | `Scene.prepare`/`render`, `sg.*`, `latch*Debug`, upload | не вызывает `PhysicsWorld.step`, не освобождает стримы/декодеры, не мутирует живой мир |
| game/update | `Scene.update`, inline-физика, `updateStreams`/retire/deinit стримов, `buildPreparedFrame` | не вызывает `sg.*`; GPU-аплоады только стейджит в upload-пакеты |
| dedicated physics worker | только `PhysicsWorld.step(dt)` в окне `begin/endParallelWindow` | не трогает scene/GPU/стримы; `input` — только мир |
| Box3D workers (`physics_workers > 1`) | C-задачи solver'а внутри `b3World_Step` | не вызывают Zig-код: `enqueueTask`/`finishTask`/`createDebugShape`/friction/restitution callbacks не задаются; `sg.*` недоступен |
| audio | `renderFrames`: миксер, DSP, `processRetireRequests` (только перенос указателей в ack-ring) | не вызывает `deinit` (освобождение — на game-потоке через `drainRetireAcks`), не трогает GPU |
| jobs/io | data-parallel CPU / декоды | GPU-работы откладывают до context-flush; `sg.*` нет |

Инвариант «физика не на render-потоке» закреплён тестом `threading: dedicated physics with Box3D workers never steps on the render thread` (`src/scene/physics_thread_tests.zig`): тест помечает текущий поток как sg-context owner, гоняет `Scene.update` с `physics = .dedicated` и `physics_workers = 2`, затем проверяет `PhysicsWorld.getLastStepThreadId() != context_thread_id`. `getLastStepThreadId()` — диагностический геттер (одно relaxed-store за шаг).

Оговорка: в **serial**-хосте (без game-воркера) `Scene.update` и render легально живут на одном потоке — они выполняются последовательно, конкурентности нет. Запрет касается только одновременного выполнения; для этого и существуют `.dedicated`/staged-протокол.

## Аргумент детерминизма (проверен по коду)

Тот же dt + то же состояние мира на кике ⇒ тот же результат шага, что инлайн: шаг — чистая функция (dt, мир). Перекрытые фазы взаимно независимы с `PhysicsWorld`:

- **particles** (`scene/particle_layer.zig`): ноль референсов `PhysicsWorld`; коллизии — собственные статические сферы/плоскость (`particles/collisions.zig`), динамического coupling с телами нет (явный non-goal v1);
- **soft bodies** (`softbody.zig`): ноль референсов `PhysicsWorld`, свой PBD-солвер поверх своих cloth-мешей;
- **animations** (`scene/animation_runtime.zig`): вход — только группы/скелеты/меши. Скелетный eval пишет позы скелетов, не меши; проход по мешам читает только `morph_targets.len` (`hasMorphTargets`) и при dirty трогает только morph-хранилище (`applyMorphs`) — поля, которые воркер никогда не пишет (он пишет `mesh.position/rotation` тел + поля `RigidBody`). Дизъюнктны по полям, не по структурам.
- **decals**: таймеры отдельного слоя, мира не касаются.

Границы честности (контракт окна, см. ниже): нодовые каналы анимаций пишут напрямую в транформы мешей — такой канал на меше с телом в dedicated-режиме гонка (как и любая ручная запись в мир/тела из game-потока внутри `Scene.update`). Комбинация «морф + тело на одном меше» тоже вне контракта: morph-цикл читает меш, пока воркер пишет его позицию.

Тест `scene/physics_thread_tests.zig` сначала доказывает run-to-run битовую идентичность инлайна (120 фиксированных шагов `dt=1/60` падающего стека), затем сравнивает inline vs dedicated побитово (`mem.eql` байтов позиций/вращений) — тот же финальный стейт.

## Контракт окна (что нельзя коду приложения)

Пока окно открыто (кик → джойн внутри `Scene.update`), game-поток не вызывает публичные мутаторы/запросы мира: `createBody*`, `removeBody`, `add*Shape`, создание/уничтожение/сеттеры/геттеры джойнтов, `raycast`/`query*`/`spherecast*`, `findBody*`, `clearEvents`, `syncWorldParams`, `applyExplosion`, `enableContinuous`, `getProfile`/`getCounters`, `appendDebugLines`/`debugLineCount`, `evaluateAudioOcclusion`. `gravity` — plain-поле, гард его не перехватывает: писать только вне `Scene.update` (воркер применяет то, что актуально на старте шага через `syncWorldParams`). `step()` exempt — он внутри окна на воркере. Внутренние пути шага (`clearEvents`, `syncWorldParamsInner`, `drainPendingStructural`) обходят гард напрямую, иначе воркер паниковал бы сам на себе.

Практика: структурный чейндж между апдейтами легален всегда (окно закрыто) — тесты гоняют create/remove тел между апдейтами в dedicated-режиме. Внутри `Scene.update` в dedicated-режиме мир и тела — read/write-владение воркера.

## Tripwires и счётчики

- `PhysicsWorld.parallel_window: std.atomic.Value(bool)` + публичные `beginParallelWindow()`/`endParallelWindow()` (воркер скобит `step`) + `assertNoParallelWindow()` на входе каждого публичного мутатора/запроса выше. Философия как в `gpu_thread.zig`: Debug + ReleaseSafe, no-op в ReleaseFast/Small. Паника называет контракт: game-thread вызов во время dedicated physics window.
- Счётчики (`PhysicsIntegration.workerCounters()`, plain u64, читает game-поток): `kicks`, `joins`, `fallbacks`. Пример `examples/runtime_worker.zig -- --physics=dedicated` печатает `physics_kicks` и требует `> 0` для PASS.

## Смотрите также

- `./physics.md` — владение миром, owner-thread с dedicated-режимом.
- `./runtime.md` — жизненный цикл кадра и `Runtime` целиком.
- `./frame-pipeline.md` — слоты, freeze→commit, staged prepare детально.
- `./architecture.md` — карта модулей и слоёв движка.
- `./chaos.md` — хаос-раннер: seeded-стресс threaded-конвейера, H5/H7 гейты, TSan-статус.

Замеры: MEASUREMENTS.md волна 10 (12.10.2026) — матрица serial/threaded ×
physics inline/dedicated на sandbox: INCONCLUSIVE (workload present-bound,
шаг физики ~0.02 мс на пресете шаров); оси опт-ин, off = бит-идентично.
Волна 11 (17.10.2026) — headless `examples/jobs_bench.zig`: на крупных
чанках пул даёт ~5.2× на 8 воркерах (65k×64 и 4k×256), а прямой
`forkJoin` на len=4 стоит 15–23 µs wake-all и с 8 воркерами хуже inline;
`parallelFor` ниже `min_len_for_workers = 4096` такие размеры не отправляет,
поэтому production-анимации с 4 скелетами не затронуты.
