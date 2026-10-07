# Аудит: единый современный движок

Дата: 04.10.2026. Первоначальный аудит основан на исходниках Agate, потребителях
sandbox и HDR-diff до объединения renderer. Разделы 1–12 сохраняют найденные
проблемы и критерии приёмки; текущий результат реализации — в ledger в конце.
Цитаты путей/строк в исходном аудите относятся к срезу **до рефактора**:
часть указанных файлов/API уже удалена, это не ссылки на действующие контракты.

## Новый контракт

- Не сохранять поведение и публичные API только ради обратной совместимости.
- Один линейный HDR-контракт рендера: radiance → effects → exposure → tonemap →
  display transfer. HDR — основа renderer, а не второй opt-in renderer.
- Выключение художественных постэффектов не выключает необходимый output pass.
- Babylon полезен как сравнительный reference, но совпадение его пикселей не
  должно определять цветовое пространство, BRDF или архитектуру Agate.
- Один staged-протокол кадра для многопоточного и однопоточного запуска.
- Обязательные GPU-возможности проверяются явно. Их отсутствие — понятная
  ошибка запуска renderer; OOM — явная ошибка/пропуск кадра с rollback, а не
  незаметная смена цветового контракта.
- Рекомендуемый ближайший проверяемый graphics floor Agate: native Metal на
  macOS и браузерный WebGPU. Другие платформы не объявлять поддержанными по
  одному успешному codegen. Универсальные sokol/sokol-zig сохраняют независимый
  backend-контракт для других движков.

## Масштаб

Поиск исторической терминологии в авторских исходниках, документах, build-конфиге
и инструментах, включая `.zig/.glsl/.md/.py/.sh/.js/.mjs/.c/.h`; tracked + untracked,
без `src/agate/c`, `zig-pkg` и кэшей:

| Репозиторий | Просмотрено текстовых файлов | Файлов с совпадениями | Строк |
|---|---:|---:|---:|
| Agate | 405 | 95 | 481 |
| sandbox | 55 | 13 | 66 |
| Всего | 460 | 108 | 547 |

Числа относятся к срезу **до кодовых удалений**; сам этот отчёт исключён.
Текущий HDR-diff сам добавил 54 такие строки и удалил одну: выбранная ранее
стратегия сохранения двух режимов противоречит новому направлению.
Совпадение текста — не доказательство ненужности алгоритма.

Пути ниже относительны к `agate/`, если явно не указано `sandbox/` или `sokol-zig/`.
Номера строк — исходный рабочий срез аудита, включая тогдашний HDR WIP.

## P0 — перед продолжением HDR

### 1. Нелинейный output ради чужого renderer

**Где:** `src/agate/shaders/common/output_gamma.glsl:25–54`,
`src/agate/scene/core.zig:242–269`, `src/agate/scene/view_render.zig:141`.

`Scene.output_gamma` и `babylonOutputColor` сохраняют `pow(1/2.2)` с ранним
clamp для PBR, тогда как Standard пишет другой результат. В WIP добавлена
ещё одна uniform-ветка для обхода этого поведения.

**Убрать:** переключатель копирования output-stage Babylon, per-view override
и две интерпретации результата материала. Все scene shaders выдают линейный
цвет; ограничение диапазона float16 не путать с обрезанием до `[0,1]`.
Пересмотреть default clear color и авторские color inputs, а не просто удалить
вызов gamma. Единственная display conversion — на выходе renderer.

**Проверка:** значения radiance `1/4/16`, прозрачность/resolve в линейном
пространстве, тёмные полутона; одинаковое отображение через UNORM и hardware-sRGB
без двойного encode. Старые Babylon pixel tolerances не являются acceptance.

### 2. WIP создаёт две почти полные postprocess-цепочки

**Где:** `src/agate/shaders/postprocess.glsl:320–636,637–1124`,
`src/agate/postprocess/options.zig:18–23`,
`src/agate/scene/postfx_stack.zig:194–240`.

Дублируются LDR/HDR sampling, FXAA, TAA, DoF и bloom. `hdr_enabled` в options —
запрос режима; фактический shader switch — `hdr_params.x`
(`src/agate/shaders/postprocess.glsl:940`), по реальному формату main target.
`prepareMainTargets` при нехватке возможностей/аллокации возвращает SDR или прямой
swapchain. Это не целевая архитектура.

**Убрать:** SDR renderer и `hdr_enabled` как переключатель архитектуры.
Оставить один linear-HDR resolve, историю до exposure, одну пару exposure/tonemap
и корректный display transfer. Feature flags эффектов сохранить по смыслу.
Проверки capabilities и освобождение FAILED handles **сохранить**; заменить
именно смену цветового режима на явную политику ошибки.

**Проверка:** HDR history при смене exposure, resize/camera cut, FXAA/TAA/DoF,
HDR MSAA 1x/4x, unsupported/OOM, shutdown/re-setup, ноль validation errors.
Подключение WIP к `scene/frame_render.zig:218–244,308–321,449–480` ещё не
закончено; текущий diff нельзя считать готовой фичей.

### 3. Два bloom вместо одного

**Где:** `src/agate/postprocess/options.zig:25–34`,
`src/agate/postprocess/bloom.zig:57`,
`src/agate/shaders/postprocess.glsl:898–937,1077–1110`.

`bloom_pyramid` выбирает пирамида vs inline gather. Новый HDR-код сохраняет оба.

**Убрать:** inline bloom и выбор реализации. `bloom_enabled`, порог,
интенсивность и mip/quality budget остаются; реализация одна — HDR pyramid.
Не добавлять другую реализацию только ради неудавшейся аллокации.

**Проверка:** яркий маленький источник, radiance выше 1, 1×1/нечётные размеры,
нулевая интенсивность, конечные значения и ограниченный ресурсный бюджет.

## P1 — архитектура и рабочие потребители

### 4. Раздвоенные API target shape

**Где:** `src/agate/scene/lifecycle.zig:79–113`,
`src/agate/scene/forward_pipelines.zig:220–257`,
`src/agate/passes/skybox_pass.zig:23–35`,
`src/agate/scene/postfx_stack.zig:250–254`.

`initSampled/initTarget`, `ensureForwardMsaa/ensureForwardTarget`,
`beginMainPass/beginMainPassTarget` и отдельный `forward_hdr` сохраняют старые
sample-only входы рядом с format-aware входами.

**Заменить:** один явный target-shape API и ключ пайплайна
`color format + depth format + sample count`. Перевести draw/capture callers,
затем убрать дублирующие wrappers и отдельную SDR/HDR классификацию storage.
Разные реальные пайплайны для MSAA, history без depth и display по-прежнему нужны;
не удалять их под видом дубликатов. `RenderTarget.renderPrimaryView`
(`src/agate/render_target.zig:686–688`) тоже должен перестать требовать именно
swapchain color format.

**Проверка:** основная сцена, refraction/RTT, sky/particles/outline/debug/GUI3D,
multi-camera clears, 1x/4x, shape changes и корректный lifetime shader handles.

### 5. Staged кадр и live-read кадр существуют параллельно

**Где:** `src/agate/scene/frame_prepare.zig:467–473`,
`src/agate/scene/frame_api.zig:378–435`, `src/agate/runtime.zig:454–464`,
`src/agate/scene/ui_capture.zig:116–170`,
`src/agate/scene/frame_draws.zig:842–874`.

`prepareFrame/prepareSerial` допускают отдельный inline/live-read путь.
UI capture умеет читать живой canvas вместо slot packet; `backIndex/backSlot/publish`
оставлены рядом с claim/pin протоколом.

**Заменить:** build → stage → claim → finish/cancel → render для всех запусков.
Single-threaded — тот же протокол без worker, не отдельный алгоритм. Убрать
обходные slot helpers и live-canvas fallback после перевода приложений/тестов.
Мьютекс для диагностического exclusion допустим, если он не меняет источник данных.

**Проверка:** latest-wins, contention, pin/lease, отмена claim, UI OOM/absence,
zero live reads при render, retire epochs, serial и concurrent тесты одной модели.
`renderReuse` само по себе не устарело: повторная презентация валидного front
при медленном producer — осознанная scheduling-политика, не повод удалять safety.

### 6. Незарегистрированный GPU owner считается любым потоком

**Где:** `src/agate/gpu_thread.zig:11–13,35–39`,
`src/agate/scene/lifecycle.zig:116–121`.

Без `markContextThread` возвращается `true` на любом потоке. Это удобство тестов
попадает в контракт live context и маскирует ошибку инициализации приложения.

**Заменить:** обязательная регистрация owner для live GPU; отсутствие маркера —
явная ошибка до GPU-работы. Headless-тесты остаются CPU-only и не нуждаются
в фиктивном разрешении `sg.*` всем потокам.

**Проверка:** запуск без регистрации, чужой поток, корректный owner,
headless teardown и ReleaseSafe assertions.

### 7. Две lighting-модели и две системы локального света

**Где:** `src/agate/material/union.zig:12–15`,
`src/agate/scene/core.zig:269`, `src/agate/shaders/standard.glsl:573–587`,
`src/agate/scene/light_rig.zig:46–52,291–292`,
`src/agate/lights/clustered.zig:9–21`.

Blinn-Phong Standard — default material рядом с PBR; четыре top-K point slots
и отдельный clustered pool имеют разные packing/API/шейдинг.

**Направление:** PBR — основной lit material, явный unlit/stylized/custom shader
сохраняется как продуктовая возможность. Убрать именно отдельную Standard
lighting-модель и её parity-specific математику после переноса material library,
hook templates, serialization и сцен. Не считать любой простой shader ненужным.
Для local lights — единый storage/clustered routing; shadow budget и shadow
selection отделить от числа освещающих источников. Текущий clustered pool только
point/unshadowed, поэтому удалить top-K spot/shadow support без замены нельзя.

**Проверка:** library/node/custom materials, glTF, skin/instances, lights beyond
4, point/spot shadows, light lifetime и save/load. Не обещать ускорение без замера.

### 8. App backend matrix шире проверяемого контракта

**Где:** `build.zig:103–108,284–289,320–337,664–665`,
`docs/shaders.md:55–62`, `sokol-zig/build.zig:120–134`.

Agate генерирует GL/HLSL5/Metal/WGSL; допускает web link через WebGL.
Автовыбор зависимости также означает D3D11/GL на части native платформ.
Успешная генерация шейдера не доказывает работу всего renderer: forward уже
требует fragment storage, часть backend возможностей различается.

**Убрать в Agate:** обещание старого graphics floor и непроверяемые режимы запуска.
Для ближайшего Metal/WebGPU floor сократить engine codegen/CLI/CI до фактических
hosts. Если нужен Windows/Linux — сначала законченный современный native host
и live gates, а не декларация поддержки. Общие backend реализации sokol не
удалять автоматически: это отдельные библиотеки с другими потребителями.
GLSL как исходный shader language и split texture/sampler bindings не являются
устаревшими. Native WGPU имеет ранее обнаруженный blocker cached emdawn headers;
браузерный WebGPU gate не закрывает этот blocker.

## P2 — простые удаления и формат ассетов

### 9. GPU timing API с неоднозначным нулём

**Где:** `src/agate/gpu_timing.zig:106–138`, `docs/runtime.md:210–215`.

`pollFrameMs/pollPassMs` смешивают отсутствие с валидным 0ms и теряют submission id.
Production render уже использует `?Sample`; поиск по workspace нашёл вызовы
старых helpers только в их собственных unit tests.

**Убрать:** float-only helpers и их docs/tests. Оставить `?Sample { ms, frame_index }`.
Статистический duration 0 допустим только вместе с отдельным availability/index.

### 10. No-op CLI и отдельный диагностический renderer кадра

**Где:** `sandbox/src/main.zig:2267–2280`, `sandbox/build.zig:155–156`,
`sandbox/tools/measure_threads.py:55`, `sandbox/src/stands/frame_pipeline.zig:102`.

`--concurrent-build` оставлен как deprecated no-op для скриптов;
`--no-concurrent-build` включает другую prepare-модель.

**Убрать:** оба compatibility режима после staged-переноса потребителей.
Неизвестный старый флаг должен явно отвергаться. `--no-threads` может остаться
полезной диагностикой, но обязан исполнять тот же staged-протокол.
Сначала перевести build GPU legs и измерительный скрипт: один явно вызывает
`--no-concurrent-build`, другой — `--no-threads`; текст диагностического стенда
тоже должен соответствовать текущим режимам. Удаление флагов без этого сломает
собственные гейты/замеры.

### 11. AGSC reader двух версий

**Где:** `src/agate/serialization/reader.zig:417,422,448–464`,
`src/agate/serialization/format.zig:18–19`.

Writer использует v3, reader сохраняет v2-развилки. Проверен заголовок
`sandbox/sandbox_scene.agsc`: `AGSC`, версия **3**.

**Убрать:** runtime reader v2. При смене схемы из-за нового material/output
контракта выпускать одну текущую версию и при необходимости отдельную offline
конверсию, не цепочку compatibility readers в каждом запуске.
Проверить round-trip, rejection старой версии и fuzz; файл пользователя не удалять.

### 12. DDS FourCC и перегруженный color-space boolean

**Где:** `src/agate/dds.zig:10–11,65,92`,
`src/agate/texture/core.zig:622–639,885–892`.

`srgb_to_linear` означает и преобразование RGBA-пикселей, и выбор GPU-тега DDS;
FourCC headers не задают явный color-space. Проверен
`sandbox/assets/fox_basecolor.dds`: заголовок **DX10**, не FourCC DXT.

**Направление:** явный color/data-slot contract + KTX2 или DX10 metadata.
FourCC reader — кандидат на удаление после инвентаризации внешних ассетов.
BC1/BC3/BC7 как GPU-компрессия не устарели от возраста контейнера и удаляться
автоматически не должны. Проверить albedo/emissive vs normal/MR, authored mips
и async decode/upload.

## Ограничения качества — не путать с compatibility debt

- Probes уже используют RGBA16F, но `src/agate/shaders/probe_mip.glsl`
  по-прежнему усредняет texels. Следующая quality-волна — GGX-prefilter и
  diffuse irradiance, не переименование box-filter в готовый IBL.
- `src/agate/shaders/postprocess.glsl:90–138,237–246`: normals из depth,
  camera-only reprojection. Velocity rigid/skinned/instanced + rejection нужны
  для качества движения, но отсутствие velocity — не основание удалять весь TAA.
- CPU particle simulation, CPU occlusion, PCF, MSAA resolve, TAA ping-pong,
  capability checks, headless tests, retire/late-callback guards — рабочие
  алгоритмы/инварианты. Сохранить, описывать по смыслу, без исторических ярлыков.
- SDF/bitmap fallback UI не равен плохому тексту сам по себе. Для обычного UI
  использовать установленный TTF и его реальные metrics; не путать статический
  monospace measurement с фактическими метриками шрифта
  (`src/agate/ui/font.zig:102–108`).

## Порядок реализации и стоп-условие

1. Переделать текущий HDR WIP в один renderer: пункты 1–4; обязательный output
   независимо от effect flags, существующие consumers перевести сразу.
2. Пройти unit/ReleaseSafe, Metal + browser WebGPU, RTT/refraction, resize,
   multi-view, unsupported/allocation failure и shutdown gates. Новые визуальные
   references проверяют собственный контракт Agate, не чужие особенности.
3. Объединить staged-frame и GPU owner contract: пункты 5–6, 10.
4. Материалы/свет и app backend matrix: пункты 7–8, с реальными заменами функций.
5. Удалить лишние API/format branches: пункты 9, 11–12. Затем закончить sweep
   комментариев, тестовых названий, CLI/UI-текстов и документов. Живые алгоритмы
   описать как `isotropic`, `single-sample`, `serial scheduling`, `base GGX` и т.п.,
   а не механически переименовать старую ветку в современную.

## Текущий статус реализации (04.10.2026)

| Пункт | Результат |
|---|---|
| 1–3 | Один linear-HDR renderer и bloom pyramid; SDR/gamma/implementation switches удалены. Effects OFF сохраняет output pass. |
| 4 | Дублирующие target constructors удалены; color format и samples явные. Depth пока задан окружением, полная shared shape остаётся отдельной задачей. Реальные MSAA/history/display варианты сохранены. |
| 5 | Live-read prepare, UI fallback и обходные slot helpers удалены. Один claim/build/stage/publish → begin/finish/cancel → render протокол; render не готовит кадр автоматически. Cancel/repeated build получают отдельный per-attempt cache key, без тестовых invalidation/culling обходов. |
| 6 | Незарегистрированный live owner запрещён; assert действует во всех build modes, Scene init проверяет owner до GPU-работы. Headless CPU cleanup сохранён; owner/unregistered/foreign policy проверена на живом Metal в ReleaseSafe и ReleaseFast. |
| 7 | Default и GPU draw paths переведены на PBR; Standard/instanced shaders удалены. Публичный CPU `StandardMaterial` adapter и `Material.standard` удалены (07.10.2026), `Material` union сведён к `pbr | shader_material`. Local lights консолидированы в единый clustered storage/routing pool (`ClusterLightGpu`, binding 12, 64 байта) во всех трёх forward PBR шейдерах (`pbr.glsl`, `instanced_pbr.glsl`, `skinned_pbr.glsl`). Caster selection отвязано от числа освещающих источников (до 2 point и 2 spot shadow casters в atlas slots, до 64 освещающих источников). Uniform loops сохранены как fallback для режимов без SSBO (probe captures). |
| 8 | Codegen восстановлен для всех целей: Metal/WGSL + HLSL5 (D3D11/Windows) + GLSL 4.3 (GL/Linux). Native WGPU убран из конфигурации Agate (`-Dwgpu` больше не существует); web всегда WebGPU. Явный GL 4.3 в `sapp.run` сайтах (требование cluster storage buffers). Cross-compile доказан кодгеном; live-прогоны Windows/Linux на целевых ОС пока не выполнены и поддержанными не заявлены. Универсальные sokol forks не урезаны. |
| 9 | Float-only timing helpers удалены; `?Sample` сохраняет availability и submission id, валидные 0ms не теряются. |
| 10 | Старые CLI flags отвергаются с exit 2. `--no-threads` и `--prepare-exclusion` меняют только scheduling/exclusion того же staged-протокола; GPU legs и measurement script переведены. |
| 11 | Только AGSC v3 runtime reader, v2 отвергается. Порядок байтов v3 и значение `bloom_radius` сохранены; пользовательские assets не изменены. |
| 12 | Инвентаризация Agate/sandbox/bench нашла один DDS: `sandbox/assets/fox_basecolor.dds`, DX10/BC7 sRGB (DXGI 99). FourCC reader и color/data-slot API пока не переделаны; внешний набор ассетов этим поиском не покрыт. |

Текущая staged-сборка: browser WebGPU — четыре 240-frame leg (1× UNORM,
4× UNORM, 1× hardware-sRGB, запрос 2× → 1×), без ошибок и живых sokol allocations
после cleanup. Metal showcase 1×/4×/hardware-sRGB и RTT/refraction 4× прошли; timing gate после
переноса compute dispatch accounting — 270 frames, 268 dispatches, PASS.
Screenshots подтверждают визуальный вывод и близость UNORM/sRGB, **не** численную
радиансную GPU-readback точность. Agate Debug и ReleaseSafe: **1418/1418** каждый;
sandbox: **29/29** CPU и **9/9** GPU legs. HDR allocation/ownership gate — 97 checks
в ReleaseSafe и ReleaseFast; measurement tool прошёл staged/exclusion/serial smoke.
Staged FAILED mesh creation и retry проверены на одном пути, не через старый drain.

Воспроизведение (из соответствующего репозитория):

```sh
# agate
zig build test -Doptimize=Debug --summary all
zig build test hdr-showcase gpu-timing -Doptimize=ReleaseSafe --summary all
AGATE_HDR_FRAMES=240 AGATE_HDR_MSAA=4 ./zig-out/bin/hdr-showcase
AGATE_GPU_TIMING_TEST_FRAMES=270 ./zig-out/bin/gpu-timing
zig build example-rtt -Doptimize=ReleaseSafe -- --refraction --frames 120 --rtt-msaa 4
zig build hdr-showcase gpu-timing -Dtarget=wasm32-emscripten -Doptimize=ReleaseFast --system ../sandbox/zig-pkg
node tools/hdr_browser_gate.mjs --out /path/to/evidence
# sandbox
zig build test test-gpu -Doptimize=ReleaseSafe --summary all
zig build bench-threads -Doptimize=ReleaseSafe -- --runs 1 --frames 90 --out /path/to/evidence
```

Следующие большие задачи — полная target shape и пункт 12;
auto-exposure, GGX IBL и velocity — quality-волны. Кодовые изменения этой волны
коммитятся по мере готовности гейтов.
