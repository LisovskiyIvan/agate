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

| Backend | Frame scope | Shadow/main/post |
|---|---|---|
| Metal | `command_buffer`: `GPUEndTime - GPUStartTime` | Stage-boundary timestamp counters, если device поддерживает их |
| WebGPU | `native_pass_span`: первый begin → последний end render/compute pass в submission | `timestamp-query`: первый begin → последний end проходов фазы |
| Desktop GL (поддерживаемый fork path) | `pass_sum`: сумма фаз **одной submission**, lower bound | `GL_TIME_ELAPSED` |
| Остальные / unsupported | `none`, unavailable | unavailable |

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
новом `sg.setup`; по нему отличать новый sample от повторного poll. Задержка
асинхронного результата переменна. Не смешивать фазовые значения разных submission
для искусственного frame total. Legacy `poll*Ms()` возвращают 0 для unavailable;
для проверки наличия данных использовать optional samples, не `ms > 0`.
В низкоуровневом C API сначала опрашивается `*_ms`, затем его snapshot index:
индексный getter не делает второй reap и не перепривязывает старое время к новому
кадру. Empty bracket без native pass на Metal/WebGPU не даёт измерения; GL может
измерить собственный пустой query interval. Это не ошибка и не синтетический ноль.

`SceneStats`/`FrameRecord` сохраняют `gpu_frame/shadow/main/post_submit` (0 =
unavailable) и `gpu_frame_scope` рядом с совместимыми float-полями. Профайлер
усредняет уникальные completed GPU samples и хранит их количество в
`SessionSummary.gpu_frame/shadow/main/post_samples`; отсутствие данных и
повторный async poll не понижают среднее искусственными нулями.

`Scene.render` выставляет brackets на существующих границах фаз. Probe/3D-UI
capture, prepare-side compute и другие unbracketed passes не автоматически
приписываются main/shadow/post. Для draw-level атрибуции нужен внешний GPU capture.

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
zig build example-gpu-timing -Doptimize=ReleaseSafe
AGATE_GPU_TIMING_SMOKE_OFF=1 zig build example-gpu-timing -Doptimize=ReleaseSafe
zig build install-emsdk -Dtarget=wasm32-emscripten # один раз, если SDK ещё не установлен
zig build gpu-timing -Dtarget=wasm32-emscripten -Doptimize=ReleaseFast
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

### Подтверждённые прогоны — 04.10.2026

Apple M4, macOS; эти прогоны проверяют корректность и совместимость, не сравнивают
производительность backend-ов.

| Гейт | Результат |
|---|---|
| Metal Debug и ReleaseSafe, enabled/OFF | 4/4 PASS, frame и все три фазы поддерживаются; 79–80 уникальных положительных samples на timer в каждом enabled-интервале |
| Chrome 155.0.8059.27 WebGPU, enabled/OFF/no-device-feature | 3/3 PASS; enabled — 79–80 уникальных положительных samples на timer в каждом интервале; без feature — capabilities false и samples unavailable |
| Toggle + shutdown/re-setup immediately after commit | Metal/WebGPU: 270 кадров и compute dispatches, 0 validation errors, 0 живых sokol allocations после shutdown |
| Sokol native compile + GL/GLES/Metal/ARC/D3D11 functional suites | 955/955 testcase executions PASS |
| Headless WGPU callback lifetime regression | 8/8 PASS: stale generation/context, reused slot, valid zero, completion order, abort; metadata не зависит от sg allocator |
| Agate Debug / ReleaseSafe | 1394/1394 в каждом режиме |
| Sandbox headless / native GPU | 29/29 tests, 8/8 GPU legs |
| RTT capture/refraction/4x MSAA | 3/3 PASS, 0 validation errors |
| Sandbox и bench WebGPU wasm builds | PASS |
| Default-OFF WebGPU visual compatibility | 6/6 PASS: twosided/transparency/cutout/environment/morph/particles; MAE 0.00/0.01/0.00/0.90/0.00/0.14, tolerances неизменны; canonical parity artifacts не изменены |

Native WebGPU **app** не проверен: cached emdawn header имеет существующий drift
в имени `WGPUSurfaceSourceMetalLayer`. C gfx timing path компилируется с этим
header; реальные timestamp/readback/lifecycle прогоны выше сделаны в браузере.
Linux linker path optional callback test настроен, но на этом macOS host не запускался.
