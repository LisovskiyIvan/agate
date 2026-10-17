# GPU timestamps

`agate.gpu_timing` использует наши `sokol`/`sokol-zig` forks, без patch-скриптов.
Таймеры по умолчанию **выключены**, не подменяют GPU execution CPU-submit временем
и не ждут текущий GPU-кадр ради результата.

Workspace dependency остаётся `../sokol-zig`: `agate`,
`git@github.com:LisovskiyIvan/sokol.git` и
`git@github.com:LisovskiyIvan/sokol-zig.git` нужно клонировать соседними каталогами.
После правки C fork регенерировать bindings из `sokol/bindgen` через
`python3 gen_zig_only.py` (локальный `bindgen/sokol-zig` указывает на sibling clone).
Vendored sokol и автоматических patch-скриптов в Agate больше нет.

## Включение и поддержка

```zig
const timing = agate.gpu_timing;
timing.setEnabled(true); // только atomic intent, допустим с game thread
sapp.run(.{
    .wgpu_gpu_timing_enabled = timing.isEnabled(), // до создания WebGPU device
    // остальные callbacks / window options
});
```

На native можно использовать `AGATE_GPU_TIMINGS=1` (`true`/`on`/`yes`).
В браузере intent устанавливает приложение. WebGPU device без запрошенного
`timestamp-query` остаётся без таймеров даже после `setEnabled(true)`.
Unsupported optional feature не должна ломать создание device.

После `sg.setup` вызвать `capabilities()` на context thread:

| Backend                               | Frame scope                                                                       | Shadow/main/post                                               |
| ------------------------------------- | --------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| Metal                                 | `command_buffer`: `GPUEndTime - GPUStartTime`                                     | Stage-boundary timestamp counters, если device поддерживает их |
| WebGPU                                | `native_pass_span`: первый begin → последний end render/compute pass в submission | `timestamp-query`: первый begin → последний end проходов фазы  |
| Desktop GL (поддерживаемый fork path) | `pass_sum`: сумма фаз **одной submission**, lower bound                           | `GL_TIME_ELAPSED`                                              |
| Остальные / unsupported               | `none`, unavailable                                                               | unavailable                                                    |

Capabilities описывают device support, не текущий intent и не готовность sample.
Наличие Metal device не доказывает поддержку counters. WebGPU frame span включает
gaps между native passes, но не измеряет transfers вне первого/последнего pass;
GL pass sum gaps не включает. Эти числа нельзя выдавать за одинаковую метрику.
Stage/pass spans при GPU pipelining могут перекрываться: это не эксклюзивные
«стоимости фаз», и их сумма не обязана совпадать с frame sample.

## Samples и фазовые brackets

Sokol API **engine-agnostic**: `sg_set_gpu_timing_enabled`,
`sg_gpu_timing_scope_begin/end`, `sg_query_gpu_scope_ms`,
`sg_query_gpu_scope_frame_index` и соответствующие frame queries/capabilities.
`SG_MAX_GPU_TIMING_SCOPES = 16`, IDs определяет вызывающая программа. В sokol
нет Agate-prefixed timing symbols, environment flag или имён shadow/main/post;
`Pass` и `AGATE_GPU_TIMINGS` — только политика Agate поверх общего API.

Без Agate (только sokol-zig):

```zig
const sg = @import("sokol").gfx;
sg.setGpuTimingEnabled(true);
sg.gpuTimingScopeBegin(7); // произвольная группа приложения
// sg.beginPass / draws или dispatch / sg.endPass, один или несколько проходов
sg.gpuTimingScopeEnd(7);
sg.commit();
const ms = sg.queryGpuScopeMs(7); // -1 = unavailable; 0 может быть реальным замером
const submission = sg.queryGpuScopeFrameIndex(7); // snapshot того же *_ms poll
```

```zig
timing.beginPass(.main); // вне sg.beginPass/sg.endPass
// один или несколько native render/compute passes
timing.endPass(.main);
sg.commit();
if (timing.pollPassSample(.main)) |sample| {
    // sample.ms может быть 0: точность/квантизация backend
    // sample.frame_index != 0: завершённая sokol submission
}
```

Фазы — `shadow`, `main`, `post`, последовательные, не вложенные. `Sample` содержит
`ms: f32` и `frame_index: u32`; `null` означает disabled, unsupported, not-ready или
отсутствие валидного контекста. Индекс — **не** `Scene.frame_id`, сбрасывается при
новом `sg.setup`; по нему отличать новый sample от повторного poll (индекс 0 —
unavailable, длительность 0 мс при ненулевом индексе валидна). Задержка
асинхронного результата переменна. Не смешивать фазовые значения разных submission
для искусственного frame total. `pollFrameMs/pollPassMs` удалены: наличие данных —
только по optional `Sample`, не по `ms > 0`.
В низкоуровневом C API сначала опрашивается `*_ms`, затем его snapshot index:
индексный getter не делает второй reap и не перепривязывает старое время к новому
кадру. Empty bracket без native pass на Metal/WebGPU не даёт измерения; GL может
измерить собственный пустой query interval. Это не ошибка и не синтетический ноль.

`SceneStats`/`FrameRecord` сохраняют `gpu_frame/shadow/main/post_submit` (0 =
unavailable) и `gpu_frame_scope` рядом с совместимыми float-полями. Профайлер
усредняет уникальные completed GPU samples и хранит их количество в
`SessionSummary.gpu_frame/shadow/main/post_samples`; отсутствие данных и
повторный async poll не понижают среднее искусственными нулями.

`examples/gpu_phase_bench.zig` также учитывает только уникальные completed
submissions: повторный или более старый poll не попадает в выборку и не
расходует warmup. Fence сохраняется через смену фазы/плеча; результаты,
наблюдённые в settle/transition/скрытом окне, не импортируются позднее.
После пересоздания `sg` нужен новый экземпляр bench. Это устраняет повторное
взвешивание async-сэмплов, но не добавляет точного config-tag к ещё не
завершившимся submission: transition discard остаётся частью протокола.

`Scene.render` выставляет brackets на существующих границах фаз. Probe/3D-UI
capture, prepare-side compute и другие unbracketed passes не автоматически
приписываются main/shadow/post. Для draw-level атрибуции нужен внешний GPU capture.

## Sandbox finite-bench: GPU-разбивка

В `../sandbox` доступны `--gpu-timings` и `--bench-warmup=N` (или отдельное
значение после флага). `AGATE_GPU_TIMINGS=1` включает тот же intent без флага.
Обычный запуск таймеры не включает. Например:

```sh
zig build run -Doptimize=ReleaseSafe -- --graphics-demo --bench=600 --bench-warmup=60 --gpu-timings
zig build run -Doptimize=ReleaseSafe -- --threaded --graphics-demo --bench=600 --bench-warmup=60 --gpu-timings
```

`GPU_TIMINGS` сообщает enabled/capabilities/frame scope и статус REAL,
UNAVAILABLE или DISABLED. Четыре строки `GPU_PHASE` содержат независимые
frame/shadow/main/post выборки: число уникальных completed submissions,
median/p95/p99 в мс, stale/unavailable/invalid/overflow counters. REAL означает
хотя бы один настоящий sample, не успешный perf-гейт и не поддержку всех фаз.
Пустой канал не печатает синтетическую длительность. `measured_frames` — число
представленных кадров после warmup, не число GPU samples.

Сэмплы читаются context-side сразу после render/reuse и до следующего prepare;
warmup продвигает valid submission fences без записи в выборку. Буферы фиксированы
(4096 samples на канал), без per-frame allocations; summary печатается после
остановки worker. Складывать фазовые median в «полный GPU-кадр» нельзя.
Это разбивка трёх существующих фаз, не атрибуция отдельных bloom/AO/SSR draws.

Живой baseline 13.10.2026, Apple M4/Metal, ReleaseSafe, graphics-demo, serial:
три последовательных прогона по 600 кадров, warmup 60; по **532–534 уникальных**
samples на канал, 6–8 повторных polls исключены. Медианы отдельных прогонов:
frame **12.713 / 12.710 / 12.880 мс**, shadow **0.558 / 0.565 / 0.570 мс**,
main **6.210 / 6.209 / 6.402 мс**, post **12.114 / 12.100 / 12.212 мс**.
Это backend spans, не эксклюзивные стоимости: их сумма больше frame, что
допустимо при GPU pipelining. Из этих чисел нельзя выводить процент стоимости
постэффектов или ускорение FPS. Фикстура живая, не fixed-tick A/B.

Дополнительно threaded 600 кадров: 539 unique samples на каждом канале, один
stale poll; `GRAPHICS_DEMO verdict=PASS`. CLI opt-in и env gateway дали REAL,
disabled-прогон — DISABLED с n=0; invalid/overflow=0 во всех каналах. Прогоны
проверяют instrumentation/lifecycle, не сравнивают serial/threaded скорость.

## Владение и ограничения

- begin/end/poll/capabilities — **context thread**; `setEnabled` — intent only.
- default OFF не создаёт GPU timer resources и не кодирует timestamps.
- bounded rings не stall-ят CPU: при насыщении/overflow sample пропускается,
  неполный span не публикуется как полный.
- disable очищает samples и освобождает timer resources; `sg.shutdown` делает
  teardown сам, включая pending async WebGPU callbacks.
- повторный `sg.setup` не наследует старые samples; Zig intent применяется заново
  на следующем begin/end/poll, без stale cache предыдущего контекста.

## Проверка на живом GPU

```sh
# в examples/:
zig build run-gpu-timing -Doptimize=ReleaseSafe
AGATE_GPU_TIMING_SMOKE_OFF=1 zig build run-gpu-timing -Doptimize=ReleaseSafe
zig build install-emsdk -Dtarget=wasm32-emscripten # один раз, если SDK ещё не установлен (в корне)
zig build -Dtarget=wasm32-emscripten -Doptimize=ReleaseFast # в examples/: *_web.zig варианты
```

Готовый package cache можно переиспользовать через `--system /path/to/zig-pkg`,
не устанавливая второй SDK (например, `--system ../sandbox/zig-pkg`).

Web результат: `zig-out/web/gpu-timing.html`, открыть через localhost HTTP server.
URL `?timings-off` проверяет отключённые таймеры; `?no-device-timestamps` — WebGPU
device без optional feature. Гейт конечный (270 кадров): реальная геометрия,
multi-pass shadow/post, compute если поддерживается, два enable-интервала,
disable и shutdown/re-setup **сразу после commit**, до browser event-loop yield
(WebGPU map callback текущей submission ещё pending), при enabled intent перед
shutdown. В каждом supported интервале требуются
уникальные completed samples и положительный замер; unsupported обязан возвращать
unavailable. Custom allocator должен иметь 0 живых sokol allocations сразу после
shutdown. Итог — `verdict=PASS` и `log_errors=0`. Сам PASS unsupported-path не
доказывает измерения: отдельно проверить `frame_supported`/`pass_supported` и counts.

### Подтверждённые прогоны — 04.10.2026, единый HDR/staged renderer

Apple M4, macOS; эти прогоны проверяют корректность и совместимость, не сравнивают
производительность backend-ов.

| Гейт                                                                                                                                                                                                    | Результат                                                                                                                   |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| Browser hdr-showcase, 4 legs × 240 кадров (`node tools/hdr_browser_gate.mjs --out DIR`; `--out` опционален — default свежий `$TMPDIR/opencode/hdr-browser-*`; `--frames >= 240`, builtin Chrome runner) | checks 1170 / 730 / 1170 / 1170 после strict-GPU + staged-гейта                                                             |
| Metal hdr-showcase 1x / 4x / hardware-sRGB, по 240 кадров (`AGATE_HDR_FRAMES=240`, `AGATE_HDR_MSAA=4`, `AGATE_HDR_SRGB=1`)                                                                              | 1170 / 730 / 1170 checks; 0 failures, errors и live SG alloc                                                                |
| Native RTT + refraction 4x, 120 кадров                                                                                                                                                                  | PASS, fresh-frame proof                                                                                                     |
| Metal gpu-timing, 270 кадров (`frame_limit = 270` в examples/gpu_timing.zig)                                                                                                                                       | 268 real compute dispatches, PASS; 0 failures, errors и live SG alloc, shutdown/re-setup                                    |
| Agate full unit Debug / ReleaseSafe                                                                                                                                                                     | 1418/1418 в каждом; включая cancel/repeated-build cache regressions и one-shot compute accounting                           |
| Sandbox CPU / staged-GPU live gate                                                                                                                                                                      | 29/29 CPU, 9/9 GPU legs; default, exclusion и serial используют один staged-протокол с включёнными cullers                  |
| `test_hdr_targets` (real pool rollback + live ownership)                                                                                                                                                | 97 checks в ReleaseSafe и ReleaseFast; image/view partial-allocation rollback, retry, owner/unregistered/foreign policy     |
| `bench-threads -- --runs 1 --frames 90`                                                                                                                                                                 | staged / exclusion / serial: все три smoke runs прошли, новые metrics/CLI парсятся; это не controlled performance benchmark |

Визуальное подтверждение — только скриншоты: GPU radiance readback нет,
экраны — coarse-only. HDR-калибровка radiance 1/4/16 — CPU golden +
отдельный scene visual proof.

Текущий browser-гейт проверяет HDR output/lifecycle, не timestamp-readback.
Timing runtime этой волны проверен на Metal; прежние browser/C timing gates
остаются историческими результатами и не заменяют этот прогон.

Native WebGPU app намеренно не поддерживается: WebGPU используется только в
браузере. Windows/D3D11 и Linux/GL 4.3 требуют live-прогонов на целевых ОС;
codegen всех shader legs не заменяет эти проверки.
