# Графика Agate: качество, масштабирование, удобный API

Дата: 03.10.2026. Основание: чтение текущего кода и два независимых исследования.
Это **план**, а не список реализованных фич или обещание ускорения.

Цель: выразительный PBR-свет и стабильная картинка в заданном бюджете GPU,
при этом создание сцены остаётся простым, как в Babylon. Не копировать Unreal
целиком и не добавлять архитектуру без проверяемого потребителя.

## Что исправлено в первоначальном плане

| Прежняя предпосылка | Уточнение |
|---|---|
| Начать с auto-exposure | Сначала линейная HDR-цепочка. Main/history сейчас следуют формату backbuffer, обычно BGRA8/RGBA8; PBR compatibility-output может также обрезать цвет до записи. |
| Для SSGI уже есть Hi-Z | `visibility/` — CPU-пирамида из упрощённых окклюдеров, не GPU-глубина сцены. Для экранных лучей нужна отдельная GPU depth pyramid. |
| DDGI даст свет вне экрана через Hi-Z | Экранные данные не содержат скрытую геометрию. Нужны baked probes либо трассировка по геометрии/BVH/SDF, материалы и проверка видимости. |
| Indirect уменьшит draw calls | Один indirect на батч заменяет CPU-аргументы на GPU-аргументы. Число submissions/bindings само не уменьшается. |
| Frame graph даст async compute | Граф описывает зависимости/время жизни ресурсов. Текущий sokol не предоставляет API нескольких очередей; D3D11 не превращается в D3D12. |
| TAA автоматически даст скорость | TAA сейчас камерный, с LDR-history. Сам по себе это дополнительная работа; экономия возможна через меньшее render resolution и проверенный reconstruction. |
| IBL уже завершён | Probe mips — box-аппроксимация, не GGX-prefilter. Выбирается ближайшая проба без блендинга. |
| «80% Babylon API», «150–400 строк на backend» | Эти оценки не измерены. Использовать конкретные сценарии API и backend acceptance matrix. |

Пути к доказательствам:

- Main/history format: `src/agate/passes/postprocess_pass.zig:208–237,252–285`.
- Gamma/clamp compatibility-path: `src/agate/shaders/common/output_gamma.glsl:25–42`.
- TAA: `src/agate/shaders/postprocess.glsl:367–417`; jitter: `postprocess/taa.zig`.
- Probe approximation: `src/agate/shaders/probe_mip.glsl:1–7`.
- CPU occlusion: `src/agate/visibility/hiz_buffer.zig`, [visibility.md](./visibility.md).
- Sorting/compaction уже реализованы: `scene/render_queue/items.zig`,
  `scene/instance_staging.zig`. Не планировать их заново.
- Sokol draw/compute/usage: `sokol/sokol_gfx.h` относительно workspace `engine/`;
  indirect-consumer API и публичного buffer readback сейчас нет.

## Два независимых режима проверки

**Совместимость:** оставить существующий Babylon parity-path и неизменные
пороги `bench/visual_gate.py`. Выключенные новые эффекты не меняют этот режим.

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

### Q1. Linear HDR → exposure → final output

Opt-in HDR main target (кандидат RGBA16F), совместимые пайплайны/resolve/MSAA.
Проверить не только renderability, но sampling/filtering/blending и sample counts.
Не менять один enum: scene shading, transparency, sky/probes, bloom и output
должны согласовать цветовое пространство. Убрать ранний gamma/clamp только
в новом HDR-path; тонемаппинг и display transfer выполняются один раз в конце.
Compatibility-path остаётся прежним.

Затем добавить экспонометр: сначала downsample log-luminance, при необходимости
histogram compute; metering mask, EV/min-max limits, asymmetric adaptation,
camera-cut reset, детерминированный manual exposure для тестов.
Не читать среднюю яркость обратно на CPU каждый кадр.

**Гейт:** свет >1 сохраняется до tonemap; тёмная комната → яркое окно без
клиппинга промежуточных targets, pumping, NaN или двойной gamma. Resize,
MSAA и неподдерживаемый формат проверяются отдельно.

### Q2. Корректный IBL и пространственные пробы

Сначала offline GGX-prefilter для environment + diffuse SH/irradiance.
Проверить BRDF-LUT/roughness/LOD conventions и ориентацию cube faces.
Потом улучшать on-demand probes: HDR capture, корректный prefilter,
обоснованные spatial weights/blending и при необходимости box projection.
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
Потом variance clipping/reactive treatment для стекла, частиц и disocclusion.
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

Metal сейчас даёт frame GPU time; per-pass Metal/WGPU не реализованы в hooks.
Для атрибуции — GPU capture или отдельно спроектированные timestamps, не
CPU-submit таймер с подписью GPU. Параллельные сборки/рендеры не допускаются
во время сравнительного perf-прогона.

### P1. Минимальный indirect контракт в наших форках

CPU-authored indexed arguments → compute-authored arguments для **фиксированных
батчей**, без bindless/meshlets/full frame graph в первом эксперименте.

Нужны: indirect+storage usage, точный ABI/count/offset/alignment, index type,
base vertex/instance capability gates, overflow/bounds validation, trace hooks
и backend-owned compute-write→indirect-read synchronization. Проверить Metal,
D3D11, GL и WebGPU отдельно; unsupported-политика явная, CPU path сохраняется.
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

После закрытия WIP — **Q0 → Q1 → Q2**. Далее выбрать **один GI-трек Q3**,
а temporal Q4 / atmosphere Q5 ранжировать по reference-сценам. P0 идёт рядом,
P1 запускается только при доказанной задаче GPU visibility или CPU submission.
Всё новое opt-in до визуальных и ресурсных гейтов. Достигли целевой картинки
и frame budget — остановиться, а не автоматически реализовывать весь backlog.

## Первичные источники

- [Filament: physically based rendering, IBL и lighting](https://google.github.io/filament/Filament.html).
- [Babylon: HDR environment](https://doc.babylonjs.com/features/featuresDeepDive/materials/using/HDREnvironment).
- [Epic: Lumen Technical Details](https://dev.epicgames.com/documentation/unreal-engine/lumen-technical-details-in-unreal-engine).
- [DDGI: Dynamic Diffuse Global Illumination with Ray-Traced Irradiance Fields](https://jcgt.org/published/0008/02/01/).
- [WebGPU: indexed indirect draw](https://www.w3.org/TR/webgpu/#dom-gpurendercommandsmixin-drawindexedindirect).

Поддержка низкоуровневой возможности платформой не доказывает, что её предоставляет
текущая версия sokol или что она ускорит конкретную сцену Agate.
