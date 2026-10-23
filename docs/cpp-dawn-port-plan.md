# План порта Agate на C++ / Dawn (WebGPU)

Дата: 14.10.2026. Статус: предложение к обсуждению (решения не зафиксированы).
Цель документа: дать исполнимую карту переноса движка Agate (Zig 0.16 + sokol)
на C++ с рендером через Dawn (эталонная реализация WebGPU), с оценками, порядком
работ, критериями приёмки и списком решений, которые нужно принять до старта.

Связанные документы: [architecture.md](./architecture.md) (слои и потоки),
[frame-pipeline.md](./frame-pipeline.md) (staged-протокол), [render-pipeline.md](./render-pipeline.md),
[shaders.md](./shaders.md), [gpu-timing.md](./gpu-timing.md), [threading.md](./threading.md).

---

## 0. Резюме

- Объём переноса: **~114 000 строк Zig** без тестов (плюс вендорный C ~210 тыс. строк:
  box3d, cgltf, stb, meshopt, basisu, zstd — переносится без изменений).
- Целевой стек: **C++23 + CMake/Ninja + Dawn (`webgpu_cpp.h`) + sokol-форк (app/time/audio)
  - WGSL-шейдеры**. Dawn даёт Metal/Vulkan/D3D12/WebGPU одним API; GL/GLES3 и D3D11
    из текущей матрицы выпадают (Dawn-GL экспериментальный — отдельное решение).
- Ключевой факт: **инфраструктура уже наполовину готова**.
  - В форке sokol (`github.com/LisovskiyIvan/sokol`) `sokol_app.h` умеет **нативный WebGPU**:
    `_sapp_macos_wgpu_init` ([sokol_app.h:5578](https://github.com/LisovskiyIvan/sokol/blob/main/sokol_app.h)),
    `WGPUSurfaceSourceMetalLayer` (строка 3975), создание instance/adapter/device
    (строки 4335/4315/4282), `sapp_wgpu_get_device()`,
    `sapp_desc.wgpu_gpu_timing_enabled` — «tested with webgpu_dawn».
  - Agate уже компилирует шейдеры в **WGSL** (`build.zig:109`: `.wgsl = true`) и уже
    собирает браузерную цель под **emdawnwebgpu** (`build.zig:454`). То есть
    WGSL-варианты ~30 шейдеров существуют и проверены на WebGPU-стеке.
  - Форк sokol_gfx уже содержит engine-agnostic GPU timing hooks
    (`sg_set_gpu_timing_enabled`, `sg_gpu_timing_scope_begin/end`, `SG_MAX_GPU_TIMING_SCOPES`,
    sokol_gfx.h:2412, 5962-5966) — эта функциональность должна быть воспроизведена на
    Dawn `TimestampQuery` (см. §5).
- Черновой скелет M0 уже проверен в песочнице: `/Users/ivan/dev/myself/cplus/dwn`
  (предсобранный Dawn + CMake + поверхность из `CAMetalLayer`, треугольник рисуется).
- Оценка: **55-100 человеко-недель** (12-24 недели одним разработчиком) до паритета
  по фичам. Рендер-ядро (M0-M4) — примерно половина объёма и почти весь риск, поэтому
  порядок работ предполагается renderer-first.

---

## 1. Зафиксированные решения

| Слой            | Сейчас (Zig)                         | Цель (C++)                                                            | Примечание                                                                              |
| --------------- | ------------------------------------ | --------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| Окно/ивенты/DPI | sokol_app (+ Metal)                  | **sokol_app форка, `SOKOL_WGPU`**                                     | sapp создаёт окно, ивенты, свапчейн-поверхность; device отдаёт `sapp_wgpu_get_device()` |
| Свапчейн        | sokol_gfx pass                       | `sapp_acquire_swapchain()` → `WGPUTextureView` (render/resolve/depth) | `wgpu::Surface` в своём коде не нужен: маршрут как у sokol_gfx                          |
| GPU API         | sokol_gfx                            | **Dawn** (`webgpu_cpp.h`)                                             | предсобранные релизы `google/dawn` (headers + `libwebgpu_dawn.a` + `tint`)              |
| Шейдеры         | GLSL + sokol-shdc (4 языка)          | **WGSL напрямую**                                                     | WGSL-выход shdc оставляем как эталон для порта, не как зависимость                      |
| Время           | sokol_time                           | sokol_time (тот же C-хедер)                                           | без правок                                                                              |
| Аудио           | sokol_audio                          | sokol_audio (тот же C-хедер)                                          | DSP/микшер — перенос CPU-кода                                                           |
| Физика          | box3d (C)                            | box3d (C)                                                             | вендорный C не меняется                                                                 |
| Ассеты          | cgltf/stb/meshopt/basisu + свой Zig  | те же C + C++-порты парсеров                                          | ktx2/dds/exr/ttf — свои парсеры                                                         |
| Сборка          | build.zig (47k, shdc, реестр тестов) | CMake + Ninja + CTest                                                 | цели-примеры = критерии приёмки этапов                                                  |
| Web             | emscripten + emdawnwebgpu            | то же (emdawnwebgpu pkg)                                              | один и тот же `wgpu_cpp` код нативно и в браузере                                       |
| Тайминги GPU    | форк-хуки sokol_gfx                  | Dawn `FeatureName::TimestampQuery` + `QuerySet`/`WriteTimestamp`      | сохраняем 3 фиксированных скоупа (shadow/main/post)                                     |

**Не-цели первого прохода:** GL/GLES3 и D3D11 бэкенды; Android; wire-режим Dawn
(`dawn::wire`) и multi-process; замена AGSC-формата; hot-reload шейдеров на всех платформах;
video/streaming сверх текущих возможностей.

---

## 2. Что переносится механически, что переписывается

Классификация по группам модулей (LOC — из текущего дерева, без тестов).

| Группа                                                                                                                                                                                          |         LOC | Судьба                            | Заметки                                                                                         |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------: | --------------------------------- | ----------------------------------------------------------------------------------------------- |
| math, camera                                                                                                                                                                                    |      ~3 900 | механический перенос              | `@Vector` → `std::array`/`glm`-подход (или свой SIMD-модуль, как сейчас), детерминизм сохраняем |
| animation (скелеты, клипы, easing, retarget)                                                                                                                                                    |      ~3 200 | механический                      | чистый CPU, 64 кости                                                                            |
| physics (обёртки box3d, character/rope/joints, ragdoll/vehicle, softbody)                                                                                                                       |      ~7 200 | механический                      | box3d C остаётся; меняется только связка «мир → сцена»                                          |
| audio (микшер, шины, DSP, decode, стриминг)                                                                                                                                                     |      ~7 000 | механический                      | 0 аллокаций в аудиоколбэке — сохраняем контракт                                                 |
| ui + ttf (canvas, flex/grid/docking, TTF/SDF)                                                                                                                                                   |      ~6 000 | механический                      | GPU-часть UI — через Dawn-слой                                                                  |
| assets: ktx2, dds, exr, texture/core*, upload queue, asset_manager                                                                                                                              |     ~11 500 | механический + GPU-часть под Dawn | декодеры — чистый CPU; `UploadQueue`/`drainCountedBudget` — как есть                            |
| loader/export (obj/stl/ply, glTF-обвязка, meshopt)                                                                                                                                              | входит выше | механический                      | cgltf C без правок                                                                              |
| core: app, runtime, jobs, handoff, gpu_thread, ai, visibility, profiler, serialization                                                                                                          |      ~9 000 | механический                      | шаблоны → C++ templates; `Handoff(T,N)`, `SpscRing(T,N)`, `forkJoin`                            |
| **scene** (core/registry/content/snapshot/queue_builder/render_queue/frame_draws/view_render/draw/uniforms/instance_staging/clustered_lights/light_rig/postfx_stack/probe_layer/gui3d_layer...) |     ~29 000 | перенос с правками                | 60 файлов; точки правок — только вызовы GPU-слоя                                                |
| **mesh** (mesh, builders, tangents, csg, decal, trail, vat, morph_gpu, simplify)                                                                                                                |     ~10 100 | перенос с правками                | GPU-часть: VB/IB, morph-текстуры, VAT                                                           |
| **material** (pbr, union, draw_record, shader_material, node_material, material_library)                                                                                                        |      ~4 500 | перенос с правками                | hook-система шейдер-материалов переписывается под WGSL-мерж                                     |
| **passes** (shadow, skybox, particle, postprocess, bloom, glow, volumetric, highlight, outline, ssao, ssgi, ssr, ddgi, debug, velocity, depth_pyramid)                                          |      ~9 700 | переписать под Dawn               | 18 файлов; все на `sg.beginPass`/bindings/draw                                                  |
| renderer extra: render_target, postprocess/ (options+types), ssao.zig, gi/, compute, gpu_timing, target_shape                                                                                   |      ~5 500 | переписать под Dawn               | ядро абстракций                                                                                 |
| shaders (~30 базовых + snippets + hook-мерж + runtime user shaders)                                                                                                                             |           — | **WGSL-порт**                     | см. §5                                                                                          |

Инварианты, которые переносим без изменений (иначе потеряем свойства движка):

- staged-протокол кадра: `claim → build → stageUi → publish`, `pin/unpin`, lease,
  lock-free контракт (`frame-pipeline.md`);
- владение GPU: маркер context-потока, «обязательная регистрация GPU owner»,
  fail-closed в headless;
- `TargetShape` (color+depth+stencil+sample_count) как единственный контракт совместимости;
- фиксированные slot-константы биндингов (`UB_*`, `VIEW_*`, `SMP_*` из shdc) — они
  превращаются в явные bind group layout'ы Dawn (см. §3);
- единый linear-HDR пайплайн `RGBA16F` + пирамидальный bloom + composite-стек;
- 3 фиксированных GPU-скоупа таймингов.

---

## 3. Ключевое решение: shim «sg-over-Dawn» или прямой `wgpu::`

### Вариант A — тонкий C++-слой sg-семантики поверх Dawn (рекомендуется)

Порядка 2 500-4 000 строк C++: buffer/image/view/sampler/shader/pipeline/pass/
bindings/uniforms/transient-ring/registry. Перенос `scene/` (~29k) и `passes/` (~10k)
становится построчным (`sg.makeBuffer` → `sgl::makeBuffer`), а весь риск
сосредотачивается в одном небольшом модуле, который тестируется отдельно
(golden-картинки на каждый пасс).

Маппинг (проверено по текущему коду и инвентарю использований):

| sokol_gfx (использований)                                             | Dawn                                                     | Подводные камни                                                                                                      |
| --------------------------------------------------------------------- | -------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `makeBuffer` (85)                                                     | `CreateBuffer` + ring-буфер для transient                | `WriteBuffer` требует кратности 4, offset; выравнивание 256B для dynamic offsets                                     |
| `makeImage` (49)                                                      | `CreateTexture`                                          | storage-образов в движке нет; mips/readback'ов нет — только рендер                                                   |
| `makeView` (83)                                                       | `CreateTextureView` (att/tex/resolve/depth)              | view-алиасинг 1 image → 4 view': кэшировать, не пересоздавать каждый кадр                                            |
| `makeSampler` (35)                                                    | `CreateSampler`                                          | ключ кэша: адресные режимы + фильтры + aniso + LOD-клампы                                                            |
| `makeShader` (48)                                                     | `CreateShaderModule` (WGSL)                              | per-backend `*_Desc` из shdc исчезает: одна WGSL-программа на пайплайн                                               |
| `makePipeline` (74)                                                   | `CreateRenderPipeline`/`CreateComputePipeline`           | нужен явный `PipelineLayout` из slot-констант (авто-layout Dawn не даст стабильных групп)                            |
| `beginPass` (32, 2 compute)                                           | `BeginRenderPass`/`BeginComputePass`                     | `resolves[]` → `RenderPassColorAttachment.resolveTarget` (1:1); `DONTCARE` → `StoreOp::Discard`; MSAA-тайлы как есть |
| `applyPipeline` (44)                                                  | `pass.SetPipeline`                                       | —                                                                                                                    |
| `applyBindings` (44)                                                  | `pass.SetBindGroup`                                      | Dawn требует BG, соответствующие layout; **нужен кэш BG по (layout, ресурсы)**: 200+ draw/кадр                       |
| `applyUniforms` (73)                                                  | `queue.WriteBuffer` в uniform-ring + dynamic offsets     | сейчас полные блоки per draw (2560B legacy, bones 64×mat4) — ring + батчинг обязательны                              |
| `applyViewport/applyScissorRect` (19/18)                              | `SetViewport/SetScissorRect`                             | mid-pass переключения (CSM-каскады, PIP-камеры) — 1:1                                                                |
| `writeBufferTransient` (40)                                           | ring + `CopyBufferToBuffer`/offset-запись                | ломается правило «один апдейт на буфер за кадр» — надо явно зафиксировать новое правило в shim                       |
| `draw` (39) / `dispatch` (3)                                          | `Draw*`/`Dispatch*`                                      | indirect — при необходимости `DrawIndirect`                                                                          |
| `updateImage` (0) / readback (0)                                      | —                                                        | удобно: нет обратных копий, кроме пары debug-путей                                                                   |
| `query*State` (48/41/80/16/87/4)                                      | собственная таблица ресурсов                             | Dawn не отдаёт «валидность»: shim ведёт registry сам (это уже принятая в движке дисциплина)                          |
| `queryBackend/queryLimits/queryFeatures/queryPixelformat` (49/3/1/22) | adapter/device limits+features, `DawnFormatCapabilities` | `queryBackend()`-ветки `*ShaderDesc` схлопываются в единственную WGSL-ветку                                          |
| форк-тайминги                                                         | `QuerySet(TimestampQuery)` + resolve                     | API сохраняем как есть (`gpu_timing.zig` → `gpu_timing.cpp`)                                                         |
| `sg.commit` (16) / `sg.isvalid` (148)                                 | `Queue.Submit` (1 submit/кадр) / registry                | multi-flush собрать в один `CommandBuffer` на кадр либо N submit'ов — решается на M1                                 |

### Вариант B — писать рендер сразу на `wgpu::` напрямую

Идиоматичнее, меньше слоёв, торчит настоящая модель bind groups. Цена: +30-50% времени
на M1-M5, выше риск ошибок при переносе 74 pipeline-сайтов, 44 applyBindings и 73
applyUniforms, и каждое место в `scene/` требует ручной правки вместо замены имени.
Итог: как вариант «A, потом утоньшить shim в горячих местах» — компромисс по умолчанию.

### Вариант C — мост без переписывания (отдельный трек)

Добавить в форк sokol_gfx **нативный WGPU-бэкенд** (по образцу emscripten-пути, который
уже ходит в Dawn): движок остаётся на Zig, но получает Dawn на macOS/Windows/Linux
вместо Metal/D3D11/GL. Это дешёвый способ получить «Dawn везде» и одновременно
**эталон паритета** для варианта A (одни и те же сцены двумя реализациями).
Совместим с A, но требует поддержки двух рендер-путей, пока A не дойдёт до M4.

### Рекомендация

Порядок: **C (как мост и эталон, опционально) → A (основной трек) → точечный переход
отдельных мест на прямой Dawn**. Вариант B целиком — только если приоритет «чистота
кода» выше сроков.

---

## 4. Шейдеры: GLSL(+shdc) → WGSL

Что есть сейчас: ~30 базовых GLSL-программ (`src/shaders/*.glsl`), чанки
`shaders/common/` через `// @include` (свой препроцессор `expand_shader_includes`,
build.zig:397), hook-система шейдер-материалов (`// @hook(albedo)` + `sm_user_params`,
мерж-тул build.zig:758, `src/shader_material/merge.zig`), runtime user shaders
(`compileUserShader`, build.zig:130), registry (`shader_material_registry`).
Один прогон shdc генерирует metal_macos + hlsl5 + glsl430 + **wgsl**.

План:

1. **Базовые шейдеры**: переносить пофайлово в WGSL, используя текущий wgsl-выход shdc
   как эталон поведения, а не как источник. Проверка синтаксиса/валидности —
   `tint --validate` (бинарь `tint` уже входит в релиз Dawn) в CI на каждый шейдер.
2. **Инклюды**: WGSL не имеет `#include`. Логику `expand_shader_includes` переносим
   в CMake-шаг препроцессинга (конкатенация чанков, как сейчас) — генератор никак
   не связан с sokol.
3. **Hook-мерж шейдер-материалов**: `merge_shader_material` переносится в C++/CMake-шаг
   и генерирует WGSL-варианты + структуры параметров (`sm_user_params`, 128B, 8×vec4);
   union-подход движка (сниппет + базовая программа) сохраняется.
4. **Сэмплеры**: sokol комбинирует texture+sampler в одном биндинге, WGSL разделяет.
   У движка уже фиксированные слоты (`SMP_brdf=7`, `morph_smp=4` и т.д.) —
   маппинг механический: `binding=N` (текстура) + `binding=M` (сэмплер) в одном layout'е.
5. **Лейауты uniform'ов**: WGSL строже GLSL/std140 (vec3 → 16B, выравнивание структур).
   Пересчитать `VsParams/VsSkin/FsView/FsMat/sm_user_params`; существующие comptime-проверки
   байтового паритета (`scene/draw_tests.zig:87`, `uniform_split_tests.zig:182`) превратить
   в `static_assert` C++ и сохранить как гейт.
6. **Компиляция на рантайме**: Dawn компилирует WGSL в MSL/SPIR-V/DXIL сам (Tint внутри),
   поэтому кодогенерации в сборке становится меньше, а не больше. Уникальная ветка
   `queryBackend()`-шейдер-дескрипторов исчезает.

---

## 5. Этапы (milestones), критерии приёмки

Критерий каждого этапа — соответствующий пример из `examples/` и существующие гейты
(см. §7). Примеры — это готовые «приёмочные сцены», их переносить не нужно, нужно
сделать так, чтобы C++-движок их обслуживал.

| Этап                                  | Содержание                                                                                                                                                                                                                                                                                            |    Оценка | Критерий приёмки                                                                                                                |
| ------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------: | ------------------------------------------------------------------------------------------------------------------------------- |
| **M0. Фундамент**                     | репо `agate-cpp`; CMake/Ninja; предсобранный Dawn (fetch-скрипт); sokol-форк с `SOKOL_WGPU` (окно, ивенты, DPI, свапчейн); clear-кадр; `tint --validate` в CI; заглушка `gpu_timing`                                                                                                                  |  1-2 нед. | окно с clear-цветом, ресайз/DPI, `examples/render_target_basic`-подобная пустышка, 0 validation-ошибок Dawn                     |
| **M1. GPU-слой (shim)**               | buffer/texture/view/sampler/shader/pipeline/pass/bindings/uniform-ring/transient-ring/registry/BG-кэш/pipeline-кэш                                                                                                                                                                                    |  2-3 нед. | куб с PBR-заглушкой, MSAA4 + resolve, depth, shadow-пасс с viewport-трюком, скриншот-гейт                                       |
| **M2. RenderTarget + PostFX-каркас**  | `RenderTarget` (RGBA16F/depth/MSAA/borrowed), `TargetShape`, полноэкранный пост-пасс, bloom-пирамида, tonemap/exposure, composite                                                                                                                                                                     |  2-4 нед. | `examples/hdr_showcase`-эквивалент: детерминированные кадры, golden-PNG, HDR-лестница MSAA/sRGB как в `hdr_browser_gate.mjs`    |
| **M3. Сцена и forward-PBR**           | scene core/registry/content/snapshot/queue_builder/render_queue/frame_draws, staged-протокол с pin/lease, mesh/instanced/LOD, PBR+shader-материалы (WGSL hook-мерж), камеры, lights+clustered SSBO, `draw.zig`/`uniforms.zig`/`instance_staging.zig`, фабрика пайплайнов (24 + 12 DS + hook-варианты) |  4-8 нед. | `examples/q0_interior`-эквивалент: интерьер с материалами/текстурами/светом, паритет картинки с Zig-версией (метрика), перф-чек |
| **M4. Тени/скайбокс/velocity/TAA/UI** | CSM+spot+point (атлас+viewport), skybox, velocity-пасс + TAA (та же математика rejection/variance), UI-пасс + ttf-атлас, debug/outline/highlight                                                                                                                                                      |  4-8 нед. | `examples/taa_quality`-эквивалент (те же метрики ghosting/контраста), `taa_browser_gate`-леги                                   |
| **M5. Пост-GI и экстра**              | SSAO/SSGI/SSR (half-res + temporal), volumetric, probe/IBL-prefilter, depth pyramid, DDGI, compute-частицы, compute-скины/morph                                                                                                                                                                       | 4-10 нед. | соответствующие пассы с golden-гейтами, скриншот-паритет с Zig                                                                  |
| **M6. Сервисы и паритет**             | physics-связка, animation/ragdoll/vehicle/softbody, audio, async-загрузка ассетов (jobs/handoff), AGSC, profiler/HTML/trace, порт `sandbox` (18.6k LOC) или новый C++-набор демок                                                                                                                     | 6-12 нед. | `examples/chaos_runner`, `runtime_worker`, `scene_scaling_bench`-эквиваленты; перф-паритет по bench-harness                     |
| **M7. Web**                           | emdawnwebgpu сборка, браузерные гейты (`hdr/taa_browser_gate.mjs`) на Chromium/WebGPU                                                                                                                                                                                                                 |  2-4 нед. | зелёные браузерные леги для hdr/taa                                                                                             |

Итого: **55-100 человеко-недель** (12-24 недели одиночной работы при 4-5 днях в неделю),
с буфером 25% на неожиданности Dawn/WGSL.

---

## 6. Порядок внутри этапов и точки синхронизации

1. Ничего не переносим до того, как M1 закрыт golden-гейтом: shim — фундамент.
2. `scene/` (29k) не трогаем до M3: правки в Zig-версии должны заканчиваться
   (см. §8 про заморозку), иначе придётся дважды синхронизировать протокол кадра.
3. CPU-сервисы (`math`, `physics`, `audio`, `ui`, `animation`) можно тянуть
   параллельно с M2-M4: они не зависят от GPU-слоя и дают быстрый видимый прогресс.
4. `sandbox` (18.6k) портируется после M3 — до этого движок без потребителя, и это
   главный риск «написать красиво и никому не нужно».
5. Раз в этап: прогон `agate-bench-harness` на одних и тех же сценах для Zig и C++
   (median frame time, memory, checksums) — расхождение >10% считать регрессом.

---

## 7. Валидация и паритет

Ресурсы, которые уже есть и переносятся «как есть» в виде протоколов:

- `tools/test_runner.zig` + `tools/check_test_registry.zig` (24k + 2k) — свой раннер
  с реестром тестов. В C++: CTest + аналогичный реестр (или Catch2/GoogleTest —
  см. §8).
- `tools/hdr_browser_gate.mjs` (32k, 4 лега: msaa1/msaa4/sRGB/2x-policy) и
  `tools/taa_browser_gate.mjs` (44k) — headless Chromium + WebGPU, вердикты по кадрам.
  Для C++: те же гейты под web-цель M7 **плюс** их нативный аналог (рендер в
  offscreen-таргет → PNG → метрика) для M2/M4.
- `../agate-bench-harness/` (perf_run.sh, perf_median.py, perf_counts.py, perf_mem.sh,
  checksums) — перф-паритет до/после.
- Golden-картинки: каждый пасс приёмки даёт детерминированные кадры (фиксированный
  dt, фиксированные сиды анимации) — как в текущих web-гейтах.
- Отладочные ручки Dawn: `DawnTogglesDescriptor` (`dump_shaders`, `disable_symbol_renaming`),
  uncaptured error callback + device lost callback обязательны с M0.

---

## 8. Риски и решения

| #   | Риск                                                                                     | Митигация                                                                                                                   |
| --- | ---------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| 1   | Bind group per draw (200+ draw/кадр, 44 applyBindings) — CPU-цена                        | кэш BG по (layout, набор ресурсов), группировка по материалу; метрика `applyBindings` в profiler с M1                       |
| 2   | Per-draw uniform updates (73 сайта, полные блоки до 2560B, bones 64×mat4)                | uniform-ring + dynamic offsets (256B), батчинг `WriteBuffer` на кадр — модель уже фактически такая в `instance_staging.zig` |
| 3   | Правило «один апдейт на буфер за кадр» ломается на 40 сайтах `writeBufferTransient`      | явный контракт в shim: один ring-буфер на кадр, оффсеты выдаёт аллокатор; правило фиксируется в docs/threading              |
| 4   | WGSL-порт 30 шейдеров + hook-мерж + чанки                                                | `tint --validate` в CI, порт по пассам, golden-гейт на каждый; существующий wgsl-выход shdc как эталон                      |
| 5   | Расхождение потоковой модели (Dawn device thread-safe, но submit должен быть упорядочен) | сохранить `gpu_thread`: весь encoding/submit на context-потоке, `Queue.WriteBuffer` с других потоков запрещён               |
| 6   | Двойное сопровождение Zig и C++                                                          | заморозка фич Zig-версии после M3 (только багфиксы), паритетные гейты по картинкам; вариант C как эталон                    |
| 7   | Web-цель                                                                                 | emdawnwebgpu даёт тот же `wgpu_cpp` API; сама web-сборка эмскриптен **и так** идёт через Dawn                               |
| 8   | `sandbox` (18.6k) без пары                                                               | порт sandbox после M3 обязателен как критерий M6; иначе движок мёртвый                                                      |
| 9   | Тайминги: Dawn `TimestampQuery` требует фичи + буфер resolve, точность на Metal          | `sapp_desc.wgpu_gpu_timing_enabled` уже есть в форке; реализовать 3 скоупа, гейт как `gpu-timing` пример                    |
| 10  | Потеря «дешёвых» платформ: D3D11 и GL 4.3 выпадают                                       | если Windows/Linux критичны — Dawn покрывает D3D12/Vulkan; GL — только Android-путь (решение §9)                            |

---

## 9. Открытые решения (нужны до старта M1)

1. **A или B**: shim поверх Dawn (рекомендация) против прямого `wgpu::` в рендере.
2. **Трек C**: делать ли нативный WGPU-бэкенд в форке sokol_gfx как мост/эталон
   (и сколько на него готовы потратить).
3. **Судьба Zig-ветки**: заморозка после M3, параллельная разработка, или Zig —
   «референс» без новых фич.
4. **Владелец device**: берём device у `sapp_wgpu_get_device()` (быстро, но фичи/лимиты
   задаёт sapp) или создаём свой instance/adapter/device с нужными фичами
   (TimestampQuery, limits) и патчим форк на «не создавать device».
5. **Тестовый стек**: CTest + минимальный раннер (ближе к текущему) или Catch2/GoogleTest.
6. **AGSC**: сохраняем байтовую совместимость v1-v3 (тогда C++ писатель/ридер обязан
   повторить формат) или вводим v4 с новым C++-сериализатором.
7. **Порядок веток работ**: renderer-first (рекомендация) или сначала CPU-сервисы.
8. **Имя и расположение репо**: предлагается `engine/agate-cpp` (движок) и
   `engine/sandbox-cpp` (потребитель), чтобы Zig-дерево осталось нетронутым.
9. **Нужны ли GL/Android**: если да — закладываем `BackendKind` в shim сейчас,
   иначе не закладываем вовсе.

---

## 10. Приложения

### 10.1 Инвентарь GPU-вызовов (по текущему дереву)

| API                                                                   |       ~кол-во сайтов | Комментарий                                                                                           |
| --------------------------------------------------------------------- | -------------------: | ----------------------------------------------------------------------------------------------------- |
| `makeBuffer` / `makeImage` / `makeView` / `makeSampler`               |    85 / 49 / 83 / 35 | view-алиасинг 1:N на текстуру — кэшировать                                                            |
| `makeShader` / `makePipeline`                                         |              48 / 74 | пайплайны: 3 семейства × 8 (opaque/blend × u16/u32 × cull/cull-off) + 12 DS-двойников + hook-варианты |
| `destroy*`                                                            | 164/75/43/57/126/137 | нужен registry «валидность» в shim                                                                    |
| `beginPass` / `endPass`                                               |  32 (2 compute) / 33 | compute-пассы: частицы + staging                                                                      |
| `applyPipeline` / `applyBindings` / `applyUniforms`                   |         44 / 44 / 73 | все — потенциальные CPU-горячие точки                                                                 |
| `applyViewport` / `applyScissorRect`                                  |              19 / 18 | CSM-каскады, PIP-камеры, probe-mip                                                                    |
| `writeBufferTransient` / `updateBuffer` / `appendBuffer`              |          40 / ~5 / 2 | ring-модель обязательна                                                                               |
| `updateImage` / readback                                              |                0 / 0 | обратных копий нет                                                                                    |
| `draw` / `dispatch`                                                   |               39 / 3 | —                                                                                                     |
| `query*State`                                                         |           ~276 всего | Dawn-эквивалента нет → свой registry                                                                  |
| `queryBackend` / `queryPixelformat` / `queryLimits` / `queryFeatures` |      49 / 22 / 3 / 1 | ветки `*ShaderDesc` по бэкенду схлопываются в WGSL                                                    |
| форк-тайминги                                                         |        1 сайт на API | воспроизводим на Dawn `QuerySet`                                                                      |

### 10.2 Слоты биндингов PBR (заготовка bind group layouts)

- UB: `vs_params=0`, `UB_fs_view=1` (skinned: 2), `sm_user_params fs=3` / `vs=4`,
  `UB_fs_mat=5`, `vs_morph=2`, `vs_user_ub`.
- VIEW (текстуры): `9` morph_tex, `12/13/14` clustered SSBO triple, `15/16/17`
  clearcoat/sheen/brdf, `19` probe2.
- SMP (сэмплеры): `4` morph_smp, `7` brdf_smp.
- Плюс ~20 текстур PBR (albedo/normal/mr/emissive/occlusion/clearcoat/sheen/
  refraction/env/shadow×6/probe×2/ddgi) — в Dawn это одна или несколько
  bind group'ов на пайплайн, layout генерируется из slot-таблицы автоматически.

### 10.3 Что уже проверено (песочница)

`/Users/ivan/dev/myself/cplus/dwn`: предсобранный Dawn (релиз `v20261013.193955`,
`dawn::webgpu_dawn` + `tint`), CMake, `CAMetalLayer` → `SurfaceSourceMetalLayer`,
`RequestAdapter(backendType=Metal)` → `RequestDevice` → пайплайн с `Draw(3)`,
`instance.WaitAny` + `ProcessEvents`, ресайз через повторный `surface.Configure`.
Этот код — заготовка M0 (за вычетом того, что в целевой сборке поверхность отдаёт
`sokol_app`).

### 10.4 Команды валидации (после M0-M2)

```sh
# Сборка и тесты C++-движка
cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo && cmake --build build -j
ctest --test-dir build --output-on-failure

# Валидация шейдеров (бинарь tint из релиза Dawn)
./third_party/dawn/bin/tint --validate --input=src/shaders/wgsl/pbr.wgsl

# Нативные golden-гейты (аналог browser-гейтов)
./build/examples/hdr_gate --frames 240 --msaa 1 --dump out/hdr_msaa1.png
./build/examples/taa_gate --frames 240 --metric ghosting

# Перф-паритет с Zig-версией
../agate-bench-harness/perf_run.sh --engine zig   --scene q0 --frames 300
../agate-bench-harness/perf_run.sh --engine cpp   --scene q0 --frames 300
python3 ../agate-bench-harness/perf_median.py --before zig --after cpp
```
