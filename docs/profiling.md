# Tracy-профайлер (opt-in)

> Включение: `-Dtracy=true`, только native. Без флага движок собирается и работает как раньше: вместо клиента Tracy подставляется no-op заглушка.

## Что это

Помимо бортового flight recorder ([profiler.md](./profiler.md)) и GPU-таймингов ([gpu-timing.md](./gpu-timing.md)) движок умеет отдавать CPU-зоны во внешний [Tracy Frame Profiler](https://github.com/wolfpld/tracy) через биндинги [ztracy](https://github.com/zig-gamedev/ztracy) (MIT; C++ клиент Tracy 0.13.0 вендорится внутри репозитория ztracy, отдельно качать нечего).

По умолчанию (`zig build`, `zig build test`) зависимость ztracy **не скачивается и не линкуется**: опция выключена, в модуль `agate` подставляется заглушка `src/profiler/ztracy_stub.zig` (тот же импорт `"ztracy"`, те же имена, все вызовы инлайнятся в пустоту). Пиксели и поведение default-сборок не меняются.

## Быстрый старт

1. Поставить и запустить Tracy GUI (порт по умолчанию 8086):
   - macOS/Windows: скачать готовый бинарник со страницы [релизов Tracy](https://github.com/wolfpld/tracy/releases) и запустить;
   - либо собрать из исходников (`docs/` репозитория Tracy).
2. Собрать движок и примеры с зонами (первая сборка скачает ztracy и скомпилирует C++ клиент — это долго, дальше кэшируется):
   ```sh
   zig build -Dtracy=true
   cd examples && zig build -Dtracy=true
   ```
3. Запустить пример и нажать **Connect** в GUI (или наоборот: сначала Connect, потом запуск — сервер подхватит клиента):
   ```sh
   AGATE_Q0_FRAMES=120 ./zig-out/bin/q0-interior
   ```
   Без подключённого сервера приложение работает как обычно и проходит те же гейты (`verdict=PASS`) — см. про `on_demand` ниже.

## Что инструментировано (v1)

Имена зон стабильные, в нижнем регистре, без книпкинга и без аллокаций сверх собственных нужд ztracy. Комментариев на колл-сайтах нет — имена самодостаточны.

| Файл | Зоны |
|---|---|
| `src/app.zig` (`onFrame`) | `frame` вокруг тика; `FrameMark` раз в кадр |
| `src/runtime.zig` (`update`) | `update` — game-side тик + producer build |
| `src/runtime.zig` (`renderFrame`) | `prepare` — begin + finish staged prepare; `gpusubmit` — `scene.render()` / `scene.renderReuse()` |
| `src/runtime.zig` (`beginPrepare`, `beginPrepareWith`, только exclusion-режим) | `waitc` — ожидание phase mutex; `holdc` — окно удержания вокруг begin |
| `src/scene/frame_api.zig` | `scene.update`, `scene.render` (обе входные точки, включая reuse) |
| `src/jobs.zig` | `jobs.dispatch` — publish + wake + join форка; `job` — forage одного участника |
| `src/asset_manager/core.zig`, `src/asset_manager.zig` | `asset.load` — синхронный `executeTask` и воркерный `asyncMeshRun` |

## Семантика `on_demand`

Клиент собирается с `TRACY_ON_DEMAND`: пока GUI-сервер не подключён, накладные расходы близки к нулю, приложение можно гонять в обычном режиме (так прогнаны гейты `-Dtracy=true` без сервера). Подключение среди кадра начинает показывать зоны с этого момента; история «до подключения» отсутствует — это свойство Tracy, а не движка.

## Ограничения (честно)

- **Нет GPU-зон.** Tracy не умеет в Metal-тайминги; время GPU по-прежнему измеряется штатным гейтом `gpu-timing` (см. [gpu-timing.md](./gpu-timing.md). Зона `gpusubmit` — это CPU-сабмит кадра, а не время исполнения на GPU.
- **Только native.** Для wasm-таргета `-Dtracy=true` — явная ошибка сборки (`error.TracyUnsupportedOnWasm`), а не тихий force-off.
- **Нет** трекинга памяти (`TracyAlloc`), fibers, callstack-захвата, per-pass инструментации и покрытия wasm. Сознательные non-goals v1.
- Собственный профайлер движка и `gpu-timing` продолжают работать параллельно с Tracy и никак от флага не зависят.

## Потребители (sandbox и др.)

Паттерн тот же, что в `examples/build.zig`: объявить у себя опцию `tracy` и пробросить её зависимостью agate:

```zig
const opt_tracy = b.option(bool, "tracy", "Enable Tracy profiler zones") orelse false;
const dep_agate = b.dependency("agate", .{
    .target = target,
    .optimize = optimize,
    .tracy = opt_tracy,
});
```

Линковка статического клиента Tracy и импорт `"ztracy"` доезжают транзитивно через модуль `agate` — отдельных `linkLibrary` на downstream-бинарниках не нужно. Отдельный sandbox-флаг не требуется: включается тем же `-Dtracy=true`.
