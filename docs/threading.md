# Потоки симуляции (threading)

> Путь: `src/agate/threading.zig`, `src/agate/scene/physics_worker.zig`, `src/agate/scene/physics_layer.zig` (`kickWorker`/`joinWorker`/`workerCounters`), `src/agate/scene/sim_api.zig` (`kickPhysics`/`joinPhysics`), `src/agate/scene/frame_api.zig` (`Scene.update`), `src/agate/physics/world.zig` (`parallel_window`, tripwire) · Импорт: `agate.ThreadingOptions` (`root.zig`) · Потоки: game (scene-update) + scene-owned physics worker

## Что это

Модель потоков на сегодня (см. `./architecture.md`, `./runtime.md`): game-поток мутирует живые регистры и строит кадр (`Scene.update` → producer build), context-поток готовит и рисует (`beginStagedPrepare` → `finishStagedPrepare` → `render`), пул `jobs.global` разбирает data-parallel чанки (fork-join), `TaskRunner`'ы (`uploads`, `io_runner`) крутят фоновые декоды и файловый I/O. У пользователя три независимые оси конфигурации:

| Ось | Ручка | Смысл |
|---|---|---|
| Кадровый конвейер | `Runtime` (host-структура) | serial staged-путь vs threaded producer/consumer (lock-free по умолчанию, `setProducerExclusion(true)` — только mutex-диагностика тех же данных) |
| Симуляция | `ThreadingOptions.physics` (этот документ) | `inline` (дефолт) vs `dedicated`: шаг физики на Scene-owned воркере, overlapped с физико-независимыми фазами внутри `Scene.update` |
| Воркер-пул | `jobs.global` | data-parallel чанки (анимационные скелеты от `min_skeletons_for_workers = 4`, инстанс-стейджинг, culling); детерминирован при disjoint-записях |

```zig
var scene = agate.Scene.init(allocator);
scene.threading.physics = .dedicated;
```

(`inline` — Zig-ключевое слово, поэтому тег quoted: `.@"inline"` в местах использования, имя сохранено как в API.)

## Dedicated physics: семантика fork-join

Обычный порядок `Scene.update`: camera → lights → physics → animations → soft bodies → particles → decals → auto-exposure → snapshot. В `.dedicated`:

camera → lights → **[kick physics worker с dt]** → animations → soft bodies → particles → decals → **[join]** → auto-exposure → snapshot.

Kick и join выполняются на scene-update (game) потоке; воркер исполняет только `PhysicsWorld.step(dt)` в скобках `beginParallelWindow()`/`endParallelWindow()`. Хенд-офф — мьютекс + condvar с явным автоматом idle/work/done/stop (см. `physics_worker.zig`): без аллокаций на кадр, без busy-burn (воркер паркуется на condvar между киками). Zig 0.16 не отдаёт публичных мьютекса/condvar в `std.Thread` и не умеет `.name` в `SpawnConfig`, поэтому файл повторяет приём `jobs.zig:ParkingLot` (pthread на POSIX, SRWLock + CONDITION_VARIABLE на Windows), а имя потока (`agate-physics`) воркер ставит себе сам через `pthread_setname_np` best-effort.

Переключение режима работает в рантайме в обе стороны: dedicated→inline паркует воркер (работа не постится, поток живёт до `deinit`), inline→dedicated спавнит лениво на первом нужном апдейте. Воркер джойнится в `PhysicsIntegration.deinit` ДО `world.deinit()`. Если воркер не спавнится (`builtin.single_threaded`, wasm, ошибка спавна) — перманентная деградация в inline (стиль `Runtime.spawnWorker`: один принт, дальше тихо) + счётчик `fallbacks`.

Наблюдаемая семантика одинакова в обоих режимах: `pending_physics_ms` сторится ровно раз за апдейт (после join — из профиля отработавшего шага); мира нет — кика нет, сторится 0 как сегодня; `dt <= 0.0001` — инлайн early-return без thread roundtrip.

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

Замеры: MEASUREMENTS.md волна 10 (12.10.2026) — матрица serial/threaded ×
physics inline/dedicated на sandbox: INCONCLUSIVE (workload present-bound,
шаг физики ~0.02 мс на пресете шаров); оси опт-ин, off = бит-идентично.
