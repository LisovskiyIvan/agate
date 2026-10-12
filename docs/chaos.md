# Хаос-раннер и гейты надёжности (robustness wave)

> Путь: `examples/chaos_runner.zig` (шаг `run-chaos`), `src/scene/gpu_retire.zig` + `gpu_retire_tests.zig` (H5), `src/scene/upload_packets_reuse_tests.zig` (H7), `src/particles/compute_mode.zig`, `src/scene/clustered_lights.zig` · Импорт: `agate.Runtime`, `agate.Scene`, `agate.testing.testScene` (`root.zig`) · Потоки: game (сценарий) + context (кадр)

## Что это

Многопоточный кадр доказан для ИЗВЕСТНЫХ сценариев (юнит-сьют + `runtime-worker`). Эта волна закрывает НЕИЗВЕСТНЫЕ: детерминированный (`--seed`) стресс-хост поверх ТОЛЬКО публичного API гоняет создание/уничтожение, переключение потоков, паузы продюсера и чейндж регистров, пока контекст готовит кадры. Два реальных бага найдены и исправлены именно хаосом (см. «Находки»).

## Как запускать

```sh
cd examples
# Headless (CPU-only staged путь, без sg — быстрый CI-режим):
zig build run-chaos -- --headless --ops 20000 --seed 1
zig build run-chaos -- --headless --ops 20000 --seed 2
zig build run-chaos -- --headless --ops 20000 --seed 3
# Оконный (полный GPU-путь: flush/render/reuse под валидацией sokol):
zig build run-chaos -- --frames 600 --seed 1
# H7-молот (всплески созданий мешей на паузе продюсера + compute-частицы
# поперёк reuse-окон; работает в обоих режимах):
zig build run-chaos -- --headless --ops 20000 --seed 7 --heavy-create
```

Флаги: `--seed N` (дефолт 1), `--ops N` (дефолт 20000, только headless — авто-выход по счётчику операций), `--frames N` (дефолт 600, только окно — авто-выход по кадрам), `--headless`, `--heavy-create`. ESC — досрочный выход из окна (идёт честный quiesce + bounded drain + вердикт).

Смесь сценария на game-потоке (всё — разрешённые контрактом мутации до `scene.update`): MeshBuilder-фигуры + прямые `uploadGeometry`-треугольники, rigid bodies на хаос-мешах (уничтожение меша сносит тело), до 4 particle-систем (первая пробует `.compute`), декали по живым целям, churn камер и clustered-источников, bursts добавлений/удалений регистров на фоне готовящегося контекста (движковый эквивалент телепортов зон), переключения `threading.physics` inline↔dedicated, запросы `setProducerExclusion` (применяются context-side — контракт), случайные паузы продюсера (форс reuse/skip-кадров).

Наблюдаемость: раз в секунду одна строка — ops/destroys/creates, builds/skips, begins/finishes/cancels/reuses/busy, pendingRetires/retained, capped, live/high-water аллокатора. В конце — вердикт:

```
chaos: verdict=PASS seed=.. ops=.. destroys=.. reuses=.. kicks=.. live_allocs=..
```

## Что проверяет вердикт

FAIL при любом из: паника; `log_errors > 0` (std.log err + ошибки валидации sokol); `begins != finishes + cancels`; `pendingRetires != 0` после quiesce + серийного drain (до 300 produce+prepare циклов — эпохи обязаны закрыться без опоры на безусловный drain в `deinit`); live-аллокации счётного аллокатора не вернулись к baseline после teardown сцены; `retainedCount > pending_cap`. `capped_drops > 0` — НЕ провал, если посчитан (flood капа — штатный backpressure, в этом и смысл счётчика; неучтённые утечки всё равно триггерят `log_errors`).

Headless-сцена — фикстура `agate.testing.testScene` + паттерн `stage1Scene` (куллинг выкл, поддельный GPU-init флаг `default_white_texture.view.id = 1`, тени выкл): staged build/begin/finish/reuse и эпохи retire — настоящие, GPU-половины — документированные coherent-empty no-op'ы. Оконный режим — полный `Scene.init`. Teardown зеркалит `stage1Deinit` + выросшие списки (системы частиц, физика, меши, слоты, retire, камеры, свет, ui, профайлер); `Scene.deinit` на фикстуре НЕ вызывается (он предполагает GPU-ресурсы полного init).

## Гейт H5 (retire-cap)

Тест `H5 retire-cap gate` в `src/scene/gpu_retire_tests.zig`: гонит `GpuRetireQueue` (малый `pending_cap`) мимо капа с незавершёнными эпохами и утверждает: дропы посчитаны ПО КАЖДОМУ виду (mesh/buffer/view/probe/ui3d/compute — по одному each), `retainedCount` ограничен капом, flush сносит ТОЛЬКО завершённые эпохи, `deinit` дренирует даже открытую эпоху, утечек нет. Чтобы ветки реальных дропов исполнялись под сьютой, cap-дропы логируют на уровне warn (bounded counted backpressure — см. комментарий у `capped_drops`); unbounded-опасность (очереди исчерпаны под OOM) осталась err.

## Гейт H7 (reuse-углы)

Непокрытые углы из `frame-pipeline.md` («Transient-буферы в reuse-кадре»): pending-mesh создания (id съедены commit'ом) и compute-буферы состояния (render-time compute перепрогоняется на reuse). Покрытие двухуровневое:

- Headless-инварианты (`upload_packets_reuse_tests.zig`): два угловых теста + H7-streak `rewrite is a fixed point` — rewrite бит-идентичен на форсированной серии reuse (×5), затем commit re-arm, затем снова серия (×3): съеденные id не воскресают, state-clear не реплеится, флаги не перевооружаются.
- Живой молот: `--heavy-create` + compute-система поперёк reuse-окон под валидацией sokol; гейт — 0 `[sg][error]`/паник. Реальных багов здесь не всплыло (углы держат).

## Находки (исправлены хаосом, узким слоем)

1. **Утечка view пула при churn clustered-света** (`clustered_lights.zig`, `gpu_retire.zig`): `retireBuffers` на каждый add/remove сбрасывал 3 view без destroy — каждые churn-цикл протекал 3 view до `VIEW_POOL_EXHAUSTED`. Фикс: новый вид очереди `Kind::view` + `retireView` (зеркало `retireBuffer`, та же эпоха), оба turnover-сайта (`retireBuffers`, growth в `ensureBuffer`) ретire'ят вью вместе с буфером. Default-путь побитово тот же (нули не ретire'ятся, дедуп по id).
2. **Гонка хэндлов clustered-слотов game↔context** (`clustered_lights.zig`, `lights_api.zig`): game-side `retireBuffers` занулял хэндлы, которые контекст читал в `upload` — check-then-write рвался, transient-писал в id 0 (`WRITE_BUFFER_TRANSIENT_BUFFER_ALIVE`, поймано lldb-брейком на `_sg_log`, доказано трейсом `upload:469/479`). Фикс: game ставит только атомный флаг `stale` (`markStale`), turnover целиком context-side в `upload` до `ensureSized`; контракт `retireBuffers` — только контекст. Поведение то же (ноль stale-кадров: retire+пересоздание до draw'ов того же рендера).

## Статус TSan (R3)

Честно пропущен: TSan сломан в Zig 0.16.0 для aarch64-macos на уровне тулчейна — даже минимальный `zig test -fsanitize-thread` (один поток, одна гонка) падает с SEGV до старта тестов; инструментация сьюта через временный `mod_agate.sanitize_thread = true` даёт тот же SEGV серверного раннера. Отдельно: `zig build` в 0.16 не пробрасывает `-fsanitize-thread` (компиляторный флаг есть у `zig test`/`build-exe`, у build-системы — нет; wiring возможен только через `Build.Module.sanitize_thread` в build.zig). TSan заведомо ВНЕ дефолтных шагов; эксперимент задокументирован здесь вместо поддельного покрытия. Контракт на отсутствие гонок держится конструкцией: все кросс-поточные данные хоста — атомики/мьютексные очереди/эпохи (см. дизайн выше).

## Смотрите также

- `./frame-pipeline.md` — слоты, freeze→commit, staged prepare, transient-переподача в reuse.
- `./threading.md` — dedicated-степ физики, контракты окон.
- `./runtime.md` — жизненный цикл `Runtime`.
- `./architecture.md` — место хаоса в устройстве движка.
