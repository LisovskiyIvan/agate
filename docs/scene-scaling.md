# Масштабирование сцены: CPU-бенч сборки кадра

Headless-бенч CPU-пути сборки кадра (`examples/scene_scaling_bench.zig`,
шаг `run-scene-scaling` в `examples/build.zig`): реальный движковый путь
`Scene.buildPreparedFrame` — пре-стейдж инстансов, `buildFrameQueues`
(frustum-culling, LOD, очереди) и обе сортировки из
`queue_builder.prepareViewQueues`. GPU-контекста нет (`sg.isvalid() == false`,
флаг GPU-init подменён как в headless-фикстурах, render не запускается).

## Запуск

```sh
cd examples
zig build run-scene-scaling -Doptimize=ReleaseFast -- --meshes 24000 --frames 50 --mode both
# Instance staging, compaction и transparent pair-sort:
zig build run-scene-scaling -Doptimize=ReleaseFast -- --meshes 48000 --frames 150 --mode instanced
```

Флаги: `--meshes N` (100..100000), `--frames F` (5..200), `--warmup W`,
`--mode identical|distinct|both|instanced|all`. `both` и дефолтный запуск сохраняют
два прежних кейса, `all` добавляет третий. Аллокатор — production `smp_allocator`,
часы — `sokol.time`, камера фиксирована (arc_rotate, radius 90, target origin),
сцена — детерминированная сетка. Контроль DCE: Wyhash-чексумма опубликованного
back-слота (модель, `texture_id`, дистанция, весь `draw_record`) плюс
`doNotOptimizeAway`; `sort_probe` отдельно замеряет production-сортировку на
перевёрнутой копии очереди (диагностика, вне замера кейса).

## Конфиги

| Кейс             | Смысл                                                                                                                                              |
| ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `many-identical` | N обычных мешей на ОДНОМ общем `PBRMaterial` (+1 общий transparent на каждый 16-й)                                                                 |
| `many-distinct`  | N мешей, у каждого СВОЙ `PBRMaterial` (каждый 16-й `.blend`, metallic/roughness варьируются)                                                       |
| `many-instanced` | Приблизительно N инстансов по 256 на opaque-родителя (минимум 8 родителей) + один transparent-родитель на 1024 инстанса; каждый 32-й инстанс скрыт |

Два обычных кейса делят геометрию и камеру: различается только материальная нагрузка.
Порядок фиксирован (identical → distinct): сравнивать только like-with-like
между версиями кода, замеры — медианы ≥3 повторов на одной машине, каждый кейс
целевым ≥100 мс суммарно.

Instanced-кейс идёт через тот же `Scene.buildPreparedFrame`: TRS/AABB staging,
скрытые экземпляры, compaction, transparent pair-sort и culling батчей. Время —
**весь CPU-билд**, не изолированная сортировка и не GPU/FPS. Staging выполняется
до culling, поэтому число реально staged-матриц отдельно от числа экземпляров
в видимых draw-батчах; per-staged-instance использует первое число. Чексумма
включает полную staged-матрицу и payload батчей, но не абсолютные uid, зависящие
от порядка кейсов в процессе. Это не отдельный гейт velocity-пар.

## Протокол A/B

1. Та же машина, idle, ReleaseFast, дефолтные `--meshes/--frames`.
2. По 3 прогона до и после изменения, брать медиану медиан.
3. Значимо, если дельта медиан превышает разброс повторов (здесь ±1%).
4. Чексуммы обязаны совпадать побитово между версиями (иначе это не тот же кадр).

## Результаты (Apple Silicon macOS, 2026-10-13, 24000 мешей / 50 фреймов)

| Кейс              | До (direct sort)           | После (indexed sort)      | Ускорение |
| ----------------- | -------------------------- | ------------------------- | --------- |
| many-identical    | 5.71 мс/билд (~238 нс/меш) | 2.15 мс/билд (~90 нс/меш) | ~2.7x     |
| many-distinct     | 5.73 мс/билд               | 2.10 мс/билд              | ~2.7x     |
| sort 4053 (probe) | 3.9 мс                     | 0.09 мс                   | ~40x      |

Чексуммы до/после совпадают побитово в обоих кейсах
(`67644a5b6f507ff9`, `6939ce7313196f00`); ReleaseSafe даёт те же чексуммы
(медианы 2.53/2.56 мс — цена safety-проверок ~20%).

### Перепроверка после интеграции

Тот же pristine `df2d2bf` before-бинарь против сборки основного worktree;
3 чередующихся after/before прогона, 24000 мешей / **80** кадров (каждый
кейс >160 мс, не менее 100 мс). Медиана медиан:

| Кейс           | Direct   | Indexed  | Ускорение CPU-билда |
| -------------- | -------- | -------- | ------------------- |
| many-identical | 5.602 мс | 1.998 мс | 2.80x               |
| many-distinct  | 5.633 мс | 2.015 мс | 2.80x               |

Видимость идентична (`opaque=4063`, `transparent=277`, `rendered=4340`,
`culled=19660`), обе чексуммы выше совпали во всех повторах. Это локальный
результат данного CPU-пути, не обещание 2.8x FPS на произвольной сцене.

## Instanced baseline после добавления кейса (13.10.2026)

ReleaseFast, та же машина, `--meshes 48000 --frames 150 --mode instanced`,
три последовательных прогона без параллельных компиляций/GPU-гейтов:
медианы **0.863 / 0.869 / 0.866 мс**, медиана медиан **0.866 мс/билд**,
размах **0.006 мс**. Каждый кейс занял 130–147 мс (≥100 мс).

Состав: 188 родителей, 48896 созданных инстансов, **47556 staged-матриц**
(скрытые пропущены, source entries включены), 35604 экземпляра в 140 видимых
батчах (139 opaque + 1 transparent). Амортизированная цена полного билда —
около **18 нс/staged-матрицу**, не изолированная цена TRS или сортировки.
Checksum **`4a8d67ef818637b0`** совпал во всех трёх прогонах.

Legacy-чексуммы сохранились в `both` и `all`; instanced `all` совпадает с
instanced-only внутри того же optimize-режима. ReleaseSafe сохраняет состав
и детерминированность, но raw-float instanced checksum отличается
(`2708fce2fd96abc6` при том же 48000): сравнивать instanced-дайджесты только
при одинаковом optimize-режиме. Межрежимовый битовый паритет не заявляется.

Это новый baseline и покрытие прежде отсутствовавшей нагрузки. Доказанного
A/B-кандидата оптимизации в этом слайсе нет, production-staging не менялся;
ускорение всего движка/FPS и доля стоимости transparent sort не заявляются.

## Что изменено (indexed opaque sort)

Горячая точка: `std.mem.sort` целиком двигал `RenderMeshItem` (~0.6 КиБ)
при перестановках — при 4k очереди десятки МБ move-трафика на билд.
Замена (`src/scene/render_queue/items.zig`):

- `lessRenderItems` — ref-ядро компаратора, один источник порядка для обоих путей;
- `sortOpaqueItemsIndexed` — сортировка u32-индексов тем же порядком + in-place
  cycle-permute (visited-биты в high-битах скратча, после — очистка для reuse);
- `RenderQueues.sort_idx` — retained u32-скратч (растёт до крупнейшей очереди,
  `reset`/`deinit` покрыты); OOM ведёт на legacy direct sort тем же порядком;
- `queue_builder.prepareViewQueues` вызывает indexed-версию; transparent-сортировка
  (мелкие `TransparentDrawEntry`) не менялась.

Эквивалентность: 2 новых теста в `items_tests.zig` (побитовое совпадение с direct
sort на 1500 ключах с ties + reuse + empty/singleton; OOM-fallback через
`FailingAllocator` остаётся отсортированным), serial/parallel equivalence-тесты
зелены, полный Debug-сьют в terminal-режиме: 1684 passed / 6 skipped / 0 failed.

## Лимиты (честно)

- Только CPU-throughput сборки очередей; никаких заявлений про графический FPS.
- Бенч идёт serial-путём (`jobs.global == null` headless): parallel-cull покрыт
  equivalence-тестами, но не перф-замером.
- Legacy-режим `both` не нагружает инстансинг; для CPU-staging используется
  `instanced`/`all`. GPU-submit и instance uploads в headless-бенч не входят.
- С var.: существующие 4-wide culling и jobs-пороги не трогались; шейдеры/пассы/
  постпроцессинг вне скоупа (другие агенты).
- `zig build test` (server-режим) в этой песочнице падает на fork-тесте
  `physics_thread_tests` ("parallel-window tripwire") — воспроизводится и на
  чистом HEAD (preexisting, окружение); валидация выше — прямым запуском
  тест-бинаря (terminal-режим).

После интеграции основной `zig build test -Doptimize=Debug` проходит и в
server-режиме; panic дочернего intentional tripwire-теста не является
падением suite. После исправления стабильного адреса MP3-декодера
ReleaseSafe full-suite тоже проходит: 1774 passed / 6 skipped в обоих режимах
после последующего hemisphere AO correctness-слайса.

Оба пути используют стабильную `std.mem.sort` (`std.sort.block` в Zig 0.16),
поэтому равные ключи сохраняют порядок исходной очереди. Retained scratch
учитывается в `FrameDraws.cpuBytes` и освобождается вместе со слотом.
