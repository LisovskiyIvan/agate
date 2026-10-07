# Графика Agate: качество, масштабирование, удобный API

Дата: 03.10.2026. Основание: чтение текущего кода и два независимых исследования.
Это **план**, а не список реализованных фич или обещание ускорения.

Обновление направления 04.10.2026: только современная архитектура, без сохранения
второго renderer и API ради обратной совместимости. Приоритеты удаления и
переходов — [modernization-audit.md](./modernization-audit.md). HDR WIP
закрыт 04.10.2026 единой linear-HDR цепочкой (см. Q1 ниже); двух режимов
нет, staged-frame — единственная основа.

Цель: выразительный PBR-свет и стабильная картинка в заданном бюджете GPU,
при этом создание сцены остаётся простым, как в Babylon. Не копировать Unreal
целиком и не добавлять архитектуру без проверяемого потребителя.

## TODO-ledger (05.10.2026, волна 3)

Приоритетный порядок следующих волн. Каждая требует гейтов (CPU golden +
Metal native + browser WebGPU legs) перед закрытием.

### A. Материалы/свет — консолидация (аудит п.7) — ВЫПОЛНЕНО (07.10.2026)

1. PBR-эквивалент matte: правило конверсии `specular_color/power → roughness` (`roughnessFromSpecularPower`).
2. Миграция `createStandardMaterial` → `createPBRMaterial`: все демо, showcases, stands, bench.
3. `default_material` → PBR-backed.
4. **ВЫПОЛНЕНО (07.10.2026):** Clustered spot storage + atlas pages + caster selection.
6. **ВЫПОЛНЕНО (07.10.2026):** удалён публичный CPU `StandardMaterial` adapter (`src/agate/material/standard.zig`), `Material` union сведён к `pbr | shader_material`, `createStandardMaterial`/`setStandardMaterial` устранены, движок и тесты консолидированы на `PBRMaterial`. В AGSC v3 reader сохранена обратная совместимость десериализации старых standard-записей через эквивалентный dielectric matte `PBRMaterial`.
7. **ВЫПОЛНЕНО (07.10.2026):** сведено хранение и шейдинг local lights к единому clustered storage/routing (`ClusterLightGpu`, binding 12, 64 байта) во всех трёх forward PBR шейдерах (`pbr.glsl`, `instanced_pbr.glsl`, `skinned_pbr.glsl`). Caster selection (до 2 point и 2 spot shadow casters) отвязано от числа освещающих источников (до 64 в clustered пуле) с динамической адресацией atlas slots и параметров смещения. Старые uniform loops сохранены как fallback для режимов без SSBO (probe captures, `clustered_params.w < 0.5`).

### B. Чистка комментариев и разбиение тестов — ВЫПОЛНЕНО (04.10.2026)

1. Чистка комментариев: `scene/core.zig`, `scene/postfx_stack.zig`, `passes/postprocess_pass.zig`, `profiler/snapshot.zig`, `scene/frame_draws.zig`, `scene/upload_packets.zig` (исторические сравнения и нумерация волн убраны, сохранены контракты threading/ownership).
2. Разбиение `scene/tests.zig` на доменные файлы:
   - `scene/softbody_tests.zig`
   - `scene/gui3d_tests.zig`
   - `scene/probe_tests.zig`
   - `scene/allocator_tests.zig`
   - `scene/mesh_lifecycle_tests.zig`
   - `scene/camera_tests.zig`
   - `scene/clustered_lights_tests.zig`
   - `scene/tests.zig` сфокусирован на staged-frame/concurrency ядре.


### C. Качество графики — волны к Unreal-уровню (порядок по стоимости/эффекту)

1. **Auto-exposure** (S) — ВЫПОЛНЕНО (07.10.2026): гистограмма/luminance math и temporal adaptation (`postprocess/auto_exposure.zig`), аналитический scene metering source (`Scene.estimateSceneLuminance`), автоматический temporal lifecycle в `Scene.update` с очисткой camera-cut и сбросом при отключении. Unit tests и live gates пройдены.
2. **GGX IBL** (M) — ВЫПОЛНЕНО (07.10.2026): GGX importance-sampled probe prefiltering, cosine-convolved diffuse irradiance, continuous multi-probe blending (`texture/ibl_prefilter.zig`, `shaders/probe_mip.glsl`, `scene/probe_render.zig`, `scene/probe_layer.zig`, `scene/draw.zig`, PBR shaders). Чтение незаполненных source mips исключено отдельным mip-0 bake-view (`probe.gpu.bake_view`). Instanced batches подключены к пространственным probes через `RenderInstancedBatch.world_center` и continuous blending в `instanced_pbr.glsl`. Fresh on-demand capture и GPU-гейты пройдены (`hdr-showcase`, 240 кадров, 3737 checks, 0 failures).
3. **Velocity/TAA** (M) — реализовано, live-гейты пройдены (05.10.2026): rigid/skinned/instanced velocity target; история привязана к последнему ОТРИСОВАННОМУ staged-кадру (cancel/reuse/multi-view не сдвигают prev, несоответствие поколения = zero motion); instance pairing по identity layout (тот же buffer id в пределах capacity, prev-матрицы из удерживаемого предыдущего слота, рост/перестановка = zero motion); morph-active draws идут в depth-fallback; alpha-cutout повторяет discard главного прохода; velocity-проход заимствует главный EQUAL-depth (writes off), а не приватный; cut/reset подавляют velocity+blur на один кадр; под MSAA velocity отключён (blur через camera reprojection). Live-доказательство: hdr-showcase Metal 1x/4x/sRGB (240 кадров, 3737 checks, 0 failures) и WebGPU browser gate, sandbox test-gpu P5/P7. Осталось отдельным гейтом: pixel-level temporal quality (ghosting/blur-метрики), а не только pipeline/lifetime-проверки.
4. **Clustered shadows** (M) — ВЫПОЛНЕНО (07.10.2026): spot/point shadow pages в atlas + routing (Track A.4/A.7). До 2 point и 2 spot shadow casters динамически выбираются по значимости и адресуются в atlas slots с PCF фильтрацией внутри единого clustered tile loop.
5. **Screen-space: GPU depth pyramid** (M) — ВЫПОЛНЕНО (07.10.2026): иерархический Z-пирамидный даунсэмплинг (`depth_pyramid.glsl`, `passes/depth_pyramid_pass.zig`, `postprocess/depth_pyramid.zig`) с консервативной редукцией $2\times 2$ (с корректной обработкой нечётных размеров) до 8 уровней мипов в `RGBA16F`. Интегрирован в `PostFXStack.renderChain`, поддержан в Scene (`depthPyramidView`, `depthPyramidMip`), проверен на Metal (120 кадров, 1805 checks, 0 failures). Оснащает SSR, contact shadows и Hi-Z culling.
6. **Contact shadows / локальный AO** (S) — ВЫПОЛНЕНО (07.10.2026): экранно-пространственный рейтрейсинг контактных теней и локальной окклюзии вдоль вектора солнечного света (`postprocess/contact_shadows.zig`, `shaders/postprocess.glsl`, `passes/postprocess_pass.zig`). Быстрый перспективно-линейный клип-маршинг с Interleaved Gradient Noise джиттером, мягким спадом толщины, краевым затуханием экрана и N·L модуляцией. Полная интеграция в `PostProcessOptions`, юнит-тесты и live-гейты пройдены (`hdr-showcase`, 240 кадров, 3737 checks, 0 failures).
7. **Local tonemapping + film LUT** (S): grading уже частично есть.
8. **GI** (L): DDGI или baked probes — только после IBL; не раньше.
   Не делать: deferred switch, mesh shaders, RTX — D3D11/sokol floor.

### D. Перф — замерить перед оптимизацией

- `bench-threads`/`phase_metrics`: добавить GPU-pass breakdown (timing API
  уже `?Sample`), per-frame alloc tally в staged слотах.
- Кандидаты по коду (проверить замером): per-draw uniform uploads в
  `scene/draw.zig`, instance CPU sort в `instance_staging.zig`, bloom mip
  count 3..7 vs fullscreen passes, occlusion CPU cost, jobs pool sizing.
- Холодные пути (init, load) не трогать. После A.1-3 перезапустить
  agate-vs-Babylon bench на новой цепочке (не acceptance, метрика регресса).

### E. Прочее

- `render_target.zig`: 1 тест остался inline (private coupling) — ВЫПОЛНЕНО (04.10.2026, вынесен в `render_target_tests.zig`).
- App-обёртка (секция ниже) — ВЫПОЛНЕНО (04.10.2026: `agate.App`, `sandbox` и `rtt-smoke` переведены, headless тесты пройдены).

## Обёртка App над окном sokol (двухуровневый API) — ВЫПОЛНЕНО (04.10.2026)

Движок должен закрывать весь жизненный цикл приложения, а не только кадр:

- **Высокоуровневый API**: `agate.App` владеет окном sokol_app (создание,
  lifecycle `init/frame/cleanup/event`, resize, DPI, ввод, тайминги, потоковая
  модель из `runtime.zig` — staged-протокол остаётся единственным). Конфиг —
  явные backend/window-параметры (Metal/D3D11/GL 4.3/WebGPU, MSAA, sRGB
  backbuffer, GL 4.3 floor). Типовой запуск — одна структура колбэков, без
  ручного `sapp.run` в каждом приложении.
- **Низкоуровневый выход**: полный доступ к sokol не прячется и не оборачивается
  «на всякий случай» — raw `sokol` import остаётся публичным, существующие
  examples с прямыми `sg.*`/`sapp.*` продолжают собираться. Обёртка — фасад над
  тем же контрактом (GPU owner registration, context thread), не замена.
- Приёмка: demo/sandbox переведены на `App` без потери возможностей; пример с
  raw-sokol рисованием рядом с scene-renderer работает; headless-тесты App
  (без окна) не требуют GPU; browser WebGPU leg собирается через ту же
  конфигурацию. Не добавлять второй windowing backend и не прятать sokol
  behind plugin-абстракциями.


## Что исправлено в первоначальном плане

| Прежняя предпосылка | Уточнение |
|---|---|
| Начать с auto-exposure | Сначала единая линейная HDR-цепочка. Исходный main/history следовал backbuffer (обычно BGRA8/RGBA8); ранний PBR output-stage также обрезал цвет. Незавершённый HDR WIP не считать закрытием этой проблемы. |
| Для SSGI уже есть Hi-Z | `visibility/` — CPU-пирамида из упрощённых окклюдеров, не GPU-глубина сцены. Для экранных лучей нужна отдельная GPU depth pyramid. |
| DDGI даст свет вне экрана через Hi-Z | Экранные данные не содержат скрытую геометрию. Нужны baked probes либо трассировка по геометрии/BVH/SDF, материалы и проверка видимости. |
| Indirect уменьшит draw calls | Один indirect на батч заменяет CPU-аргументы на GPU-аргументы. Число submissions/bindings само не уменьшается. |
| Frame graph даст async compute | Граф описывает зависимости/время жизни ресурсов. Текущий sokol не предоставляет API нескольких очередей; D3D11 не превращается в D3D12. |
| TAA автоматически даст скорость | TAA камерный, history уже HDR. Сам по себе это дополнительная работа; экономия возможна через меньшее render resolution и проверенный reconstruction. |
| IBL уже завершён | Исходный box-prefilter заменён GGX/irradiance, nearest-only выбор — top-2 blending. CPU-goldens не заменяют GPU-проверку source-mip lifetime, seams и переходов; instanced spatial probes ещё не реализованы. |
| «80% Babylon API», «150–400 строк на backend» | Эти оценки не измерены. Использовать конкретные сценарии API и backend acceptance matrix. |

Пути к доказательствам:

- Main/history format: `src/agate/passes/postprocess_pass.zig`, методы
  `resize` и `ensureTaaHistory`.
- Ранний gamma/clamp для совпадения с Babylon удалён из scene shading;
  линейный выход — `src/agate/shaders/common/linear_output.glsl`, единый
  output — POST-проход `postprocess.glsl`.
- TAA: `src/agate/shaders/postprocess.glsl`, `applyTAA`;
  jitter: `src/agate/postprocess/taa.zig`.
- Probe convolution и source LOD: `src/agate/shaders/probe_mip.glsl`, `scene/probe_render.zig`.
- CPU occlusion: `src/agate/visibility/hiz_buffer.zig`, [visibility.md](./visibility.md).
- Sorting/compaction уже реализованы: `scene/render_queue/items.zig`,
  `scene/instance_staging.zig`. Не планировать их заново.
- Sokol draw/compute/usage: `sokol/sokol_gfx.h` относительно workspace `engine/`;
  indirect-consumer API и публичного buffer readback сейчас нет.

## Проверка собственного контракта

**Корректность:** radiance, blending, resolve, transfer, владение и жизненный цикл
проверяются по контракту Agate. Babylon — сравнительный reference, а не требование
сохранять его output-stage или pixel tolerances при смене renderer.

**Качество:** отдельные reference-сцены, экспозиция/свет зафиксированы,
проверяются клиппинг, light leaks, temporal artifacts и стоимость каждого эффекта.
Совпадение с Babylon не доказывает качество GI или temporal reconstruction.

Native и WebGPU мерить отдельно: одинаковые ассеты, разрешение, качество,
warmup и повторяемый путь камеры. GPU frame, CPU update/prepare/submit,
wall interval, P95/P99, VRAM, texture uploads и dynamic updates — разные метрики.
60 FPS при vsync — не доказательство запаса GPU.

## Волны качества

### Q0. Калибровочная сцена и бюджет

Одна аккуратная интерьерная сцена + одна outdoor-сцена вместо галереи эффектов:
хорошие normals/tangents, real-world scale, roughness, тени, sun/sky rig.
Зафиксировать целевые устройства, разрешение и 60/120 FPS; не выбирать универсальный
бюджет за пользователя. Для 60 FPS весь GPU-кадр должен укладываться в 16.67 мс
с резервом на игру, а не отдавать весь бюджет постэффектам.

**Гейт:** воспроизводимый baseline, лицензированные ассеты, стабильная камера;
скриншоты и короткие траектории движения, не только замороженный кадр.

### Q1. Linear HDR → exposure → final output — инфраструктура выполнена 04.10.2026

Единый HDR main target (обязательный RGBA16F), согласованные пайплайны/resolve/MSAA (`prepareMainTargets` канонический, fail-closed, без SDR-fallback; обязательные sample/filter/render/blend-caps — явная ошибка старта). Ранний display gamma/clamp удалён; тонемаппинг и display transfer — один раз в конце (один IEC transfer, ручной UNORM / hw-sRGB). Отключение эффектов не отключает output pass (exposure+tonemap всегда). Exposure: ручной `exposure` и автоматический auto-exposure (аналитический scene metering source с калибровкой middle-gray 18%, temporal eye adaptation с настраиваемыми скоростями speed_up/down, EV/exposure limits, автоматический camera-cut reset при `Scene.update`).

**Гейт:** свет >1 сохраняется до tonemap; тёмная комната → яркое окно без
клиппинга промежуточных targets, pumping, NaN или двойной gamma. Resize,
MSAA и неподдерживаемый формат проверяются отдельно.

### Q2. Корректный IBL и пространственные пробы

Сначала offline GGX-prefilter для environment + diffuse SH/irradiance.
Проверить BRDF-LUT/roughness/LOD conventions и ориентацию cube faces.
On-demand probes: HDR capture (RGBA16F), GGX-prefilter, diffuse irradiance и top-2 spatial blending реализованы; source-mip correctness и GPU-quality гейты ещё требуется закрыть. Instanced spatial probes и при необходимости box projection — дальше.
Это не автоматический GI и не «маленький дифф» без проверки всех PBR-контуров.

**Гейт:** металлические/диэлектрические шары roughness 0→1, яркие маленькие
источники в env, переход между пробами без скачка. Проверить seams и баланс
энергии; отделять runtime capture cost от offline bake cost.

### Q3. Выбрать GI для целевой сцены

- **Статичный интерьер / максимальная скорость:** baked lightmaps для статики
  + irradiance probes для динамических объектов. Нужен отдельный bake/import
  workflow, корректные UV/charts/padding и visibility против утечек через стены.
  Наличие UV1 в glTF не означает, что lightmap-пайплайн уже готов.
- **Меняющийся свет / экранные детали:** SSGI как дополнение, а не единственный GI.
  Предусловия: HDR radiance, стабильные normals, GPU depth pyramid, temporal
  rejection/denoise; half-resolution — кандидат, не обещание выигрыша.
- **Динамический offscreen GI:** отдельный DDGI spike после выбора источника лучей
  (geometry/BVH, SDF, platform RT или ограниченная proxy-сцена). Probe visibility,
  relocation, update budget и light leaks входят в стоимость проекта.

Не внедрять lightmaps, SSGI и DDGI одновременно. Выбрать один продуктовый сценарий.
Камерные depth-derived normals допустимы для дешёвого прототипа, но их ограничения
на силуэтах/тонких объектах должны быть явно проверены, а не спрятаны денойзером.

**Гейт:** Cornell-room, окно/один источник bounce, тонкая стена, источник за камерой,
moving object. Для SSGI свет за экраном обязан иметь известный fallback, а не
объявляться полноценной offscreen-трассировкой.

### Q4. Стабильная temporal-основа → render scale/upscale

Добавить velocity для opaque rigid/skinned/instanced, предыдущие трансформы
в prepared snapshots, depth/history rejection и reset при resize/camera cut.
Камера TAA — без velocity-формирования; history уже HDR (не LDR). Потом variance clipping/reactive treatment для стекла, частиц и disocclusion.
Выбрать HDR/pre-exposed историю и согласовать её с exposure, не просто менять формат.

Затем сравнить native resolution с render-scale + spatial/temporal reconstruction.
Spatial upscale можно оценить отдельно раньше, не выдавая его за TSR.

**Гейт:** статичная мелкая геометрия, движущийся вентилятор, персонаж с анимацией,
телепорт, прозрачность, jitter+refraction. Сравнивать и GPU cost, и устойчивость
деталей в движении; не принимать «выше FPS» ценой заметных шлейфов.

### Q5. Атмосфера/туман и точечные polish-фичи

После HDR: согласованные sky/sun radiance, aerial perspective, height fog,
затем budgeted volumetric scattering. Shafts v1 уже есть — не реализовывать снова.
SSR-generalize/contact shadows/GTAO — отдельные кандидаты после normals/depth
контракта. OpenPBR, physical SSS и displacement — только если reference-сцена
доказывает, что именно они ограничивают качество.

**Гейт:** рассвет/закат, вид на горизонт, контровой свет, помещение рядом с наружной
сценой; нет несовпадения fog/sky или light leaks. Источники bloom/volumetrics
не должны маскировать ошибки освещения.

Q5 может идти после Q2 параллельно выбранному GI/temporal-треку; это не обязательная
зависимость от завершения DDGI.

## Волны масштабирования GPU

### P0. Найти ограничение, не переписывать CPU-cull заранее

Sorting, CPU Hi-Z, SIMD, инстансинг и компактные prepared snapshots уже есть.
Проверить реальные batch breaks, bind/uniform churn, shadow geometry, overdraw,
postprocess bandwidth и shader variants. Сцены: много одинаковых копий;
много уникальных материалов; почти всё видно / почти всё скрыто; тяжёлый pixel load.
Оптимизировать только измеренный ограничитель.

В наших forks есть opt-in frame/phase timestamps для Metal и WebGPU с runtime
capability gates, availability и submission ids — [gpu-timing.md](./gpu-timing.md).
Metal frame — command-buffer duration, WebGPU frame — native-pass span, GL —
сумма фаз одной submission: scope учитывать при сравнении. Shadow/main/post —
группы проходов, не отдельные draw calls; unbracketed compute/probe/UI-capture
не автоматически входят в фазовые метрики. Для более тонкой атрибуции нужен GPU
capture, не CPU-submit таймер с подписью GPU. Параллельные сборки/рендеры не
допускаются во время сравнительного perf-прогона.

### P1. Минимальный indirect контракт в наших форках

CPU-authored indexed arguments → compute-authored arguments для **фиксированных
батчей**, без bindless/meshlets/full frame graph в первом эксперименте.

Нужны: indirect+storage usage, точный ABI/count/offset/alignment, index type,
base vertex/instance capability gates, overflow/bounds validation, trace hooks
и backend-owned compute-write→indirect-read synchronization. В Agate проверить
Metal и WebGPU; общий fork сокола отдельно проверяет свои backend-контракты
для других потребителей. Unsupported-политика явная; CPU-authored arguments
остаются полезным reference и рабочим вариантом, а не compatibility renderer.
Сокращение числа вызовов — отдельная multi-draw/material-routing задача.

**Гейт:** CPU- и GPU-authored args дают одинаковые pixels/counts; zero instances,
buffer growth, partial failure, reuse frames, shadows и несколько views.
Не добавлять синхронный readback. Пока API readback нет, проверять диагностическим
draw из SSBO/внешним capture и сравнивать финальные изображения.

**Стоп:** усложнение fork без выигрыша на целевой нагрузке; capability holes;
stall ради подсчёта видимости; нарушение context-thread/epoch lifetime.

### P2. GPU visibility только после P1 и измерения CPU-предела

Начать с frustum/LOD и компактных instance IDs для фиксированных material/pipeline
батчей. Затем GPU scene-depth pyramid, temporal conservativeness и shadow-view
culling. CPU software Hi-Z остаётся отдельным корректным fallback.
GPU batching потребует маршрутизации mesh/material/probe IDs; один args buffer
не заменяет существующие ~10 texture bindings каждого PBR-материала.

### P3. Frame graph по реальным ресурсным конфликтам

Когда HDR/normals/history/depth pyramid создадут измеримую проблему управления
targets, ввести граф зависимостей, validation read/write, resize/history lifetimes
и reuse scratch targets. Aliasing/async compute — не автоматические свойства;
их добавлять только с поддерживаемым backend-контрактом и измерениями.

### P4. Геометрическая детализация — только по bottleneck

Сначала asset LOD/QEM (уже есть), spatial chunking, streaming/texture mips,
impostors для нужного контента. Meshlets/visibility buffer/virtual geometry —
отдельный проект, если геометрия доказанно доминирует. Не обещать Nanite 1:1;
сегодня у sokol нет mesh-shader/RT/bindless API. Texture arrays/atlases — варианты
батчинга с ограничениями размеров, форматов, samplers и mip-bleeding, не бесплатный bindless.

## Babylon-like API: самостоятельный, небольшой трек

Оставить `Scene`, `MeshBuilder`, `Material`, `RenderTarget`, asset-loading facade.
Пользователь управляет сценой, renderer — snapshots, passes, GPU handles и budgets.
Полезнее 3–5 законченных examples с ясным ownership, чем aliases ради совпадения имён.
Quality presets/feature capabilities должны быть явными; unsupported behavior
документируется. Не добавлять JS-фасад, fluent API или Behaviors/Actions без
отдельного сценария пользователя: текущая product-web платформа не согласована.

## Ближайший выбор и критерий остановки

После закрытия WIP — **Q0 → Q1 (инфраструктура и auto-exposure выполнены) → Q2**. Далее выбрать **один GI-трек Q3**,
а temporal Q4 / atmosphere Q5 ранжировать по reference-сценам. P0 идёт рядом,
P1 запускается только при доказанной задаче GPU visibility или CPU submission.
Художественные эффекты включаются явно; HDR/output и staged-frame — единая основа,
не альтернативные режимы. Смену основы завершать переносом consumers и новыми
визуальными/ресурсными гейтами, без постоянного второго renderer. Достигли целевой
картинки и frame budget — остановиться, а не автоматически реализовывать весь backlog.

## Первичные источники

- [Filament: physically based rendering, IBL и lighting](https://google.github.io/filament/Filament.html).
- [Babylon: HDR environment](https://doc.babylonjs.com/features/featuresDeepDive/materials/using/HDREnvironment).
- [Epic: Lumen Technical Details](https://dev.epicgames.com/documentation/unreal-engine/lumen-technical-details-in-unreal-engine).
- [DDGI: Dynamic Diffuse Global Illumination with Ray-Traced Irradiance Fields](https://jcgt.org/published/0008/02/01/).
- [WebGPU: indexed indirect draw](https://www.w3.org/TR/webgpu/#dom-gpurendercommandsmixin-drawindexedindirect).

Поддержка низкоуровневой возможности платформой не доказывает, что её предоставляет
текущая версия sokol или что она ускорит конкретную сцену Agate.
