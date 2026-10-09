# Аудит модернизации: открытый бэклог архитектуры

Дата: 08.10.2026.
Все первичные задачи модернизации (единый linear-HDR пайплайн, удаление SDR-ветки и Babylon gamma, пирамидальный bloom, консолидация PBR-материалов и clustered lights, staged-кадр, обязательная регистрация GPU owner, AGSC v3, Sample timing API) **завершены и закрыты**.

Ниже зафиксированы оставшиеся открытые архитектурные задачи и ограничения.

---

## Завершённые задачи модернизации

### 1. Полная унификация Target Shape (из п. 4 аудита) — ЗАВЕРШЕНО
- Введён единый контракт `TargetShape` (`color_format + depth_format + stencil_format + sample_count`) в [`src/target_shape.zig`](file:///Users/ivan/dev/myself/engine/agate/src/target_shape.zig).
- Устранена неявная зависимость от depth-формата внешнего окружения во всех пайплайнах и проходах (`scene/pipelines.zig`, `scene/forward_pipelines.zig`, `passes/skybox_pass.zig`, `passes/outline_pass.zig`, `passes/particle_pass.zig`, `passes/debug_pass.zig`, `passes/postprocess_pass.zig`).
- Кэш материалов `ShaderMaterialCache` и шейдеров обновлён на 64-битное хеширование полной формы таргета (Wyhash).

### 2. Инвентаризация внешних ассетов и удаление legacy FourCC DDS (из п. 12 аудита) — ЗАВЕРШЕНО
- Удалена папка кастомных нестандартных BC7 KTX2 файлов (`sandbox/assets/ktx2/`).
- Модели glTF переведены на официальные стандарты Khronos glTF 2.0 с встроенными текстурами.
- Добавлен официальный Basis Universal KTX2 образец (`sandbox/assets/standard_demo.ktx2`, UASTC с транскодингом в BC7/RGBA8).
- Из `src/dds.zig` удалён парсер устаревших FourCC DXT1..5 заголовков; строго обязателен расширенный заголовок DX10 с явным DXGI-форматом.

### 3. TAA и Motion Velocity: Rejection и Temporal Quality Gate — ЗАВЕРШЕНО
- Внедрён поиск ближайшей глубины в окрестности 3x3 cross (dilated velocity) для устранения смазывания силуэтных рёбер.
- Внедрено статистическое variance bounding (mean $\pm 1.25\sigma$) поверх min/max окрестности.
- Реализована плавная отбраковка истории (rejection weight $1 / (1 + 4d^2)$) при дизокклюзии и выходе истории за пределы окрестности.
- Добавлены количественные тесты в `src/postprocess/taa.zig`: падение ошибки шлейфа (ghosting) $>85\%$ в первом же кадре при резком смещении, сохранение высокочастотного контраста при unsharp.

### 4. UI: TTF-метрики и точные измерения символов — ЗАВЕРШЕНО
- Заменены приближённые вычисления ширины (`text.len * font_size * 0.5`) на вызовы canvas-aware измерений (`measureForCanvas`, `measureTextCurrent`) во всех виджетах (`widgets.zig`: `drawButton`, `drawBadge`; `style.zig`: `drawStyledButton`; `stack.zig`: `label`, `button`, `checkbox`, `badge`).
- Добавлено декодирование UTF-8 codepoints в базовых методах `measureText` и `drawTextInternal`.

### 5. Box Projection (Parallax Correction) для пространственных отражений — ЗАВЕРШЕНО
- Реализован алгоритм коррекции параллакса Lagarde 2012 для интерьерных зондов отражения (`src/shaders/common/box_project.glsl`).
- Обеспечен строгий байтовый паритет и выравнивание uniform blocks: поле `vec4 probe_box[2]` синхронизировано во всех трех PBR-шейдерах (`pbr.glsl`, `skinned_pbr.glsl`, `instanced_pbr.glsl`) и проверено comptime-тестами в `scene/draw.zig`.
- Расширен слой зондов (`probe_layer.zig`) с поддержкой `box_extents`, многозондовым плавным блендингом и передачей в `ProbeDrawState`.
- Реализован CPU reference и покрыт тестами в `src/texture/ibl_prefilter.zig`.

### 6. Физический атмосферный высотный туман (Beer-Lambert Transmittance) — ЗАВЕРШЕНО
- Заменено эмпирическое умножение множителей на физическую интеграцию оптической толщины $\tau = \text{eff\_dist} \cdot \rho_0 \cdot \text{height\_density}$ с экспоненциальным затуханием $1 - e^{-\tau}$ в `postprocess.glsl`.
- Создан модуль CPU-верификации `src/postprocess/fog.zig` с аналитическим интегрированием высоты, Mie forward inscattering и горизонтным дымчатым переходом.

---

## Открытые архитектурные задачи

### 1. Live-верификация нативных платформ Windows и Linux (из п. 8 аудита)

- **Текущее состояние:** Генерация шейдеров восстановлена для всех целевых бэкендов: Metal (macOS), WGSL (WebGPU/браузер), HLSL5 (D3D11/Windows), GLSL 4.3 (GL/Linux). Кросс-компиляция подтверждена.
- **Осталось сделать:**
  - Нативные live-прогоны и запуск тестов на целевых ОС Windows (D3D11) и Linux (GL 4.3).
  - Проверка работы SSBO-путей (clustered lights, compute particles) на нативном OpenGL 4.3 и D3D11. До этого момента платформы не объявляются официально поддержанными.

### 2. Unicode-шейпинг сложных скриптов поверх TTF-атласа

- **Текущее состояние:** Полный TrueType парсер, растеризатор и атлас (cmap 4/12, hmtx, kern) с UTF-8 декодированием работают для простых письменностей (латиница, кириллица, CJK).
- **Осталось сделать:**
  - Поддержка сложных контекстных скриптов (GSUB/GPOS ligatures, bidirectional Arabic/Hebrew, Indic shaping).

---

## Команды валидации открытых гейтов

```sh
# Запуск тестов Agate (Debug и ReleaseSafe)
zig build test -Doptimize=Debug --summary all
zig build test -Doptimize=ReleaseSafe --summary all

# Запуск графических тестов и гейтов тайминга
zig build test hdr-showcase gpu-timing -Doptimize=ReleaseSafe --summary all
AGATE_HDR_FRAMES=240 AGATE_HDR_MSAA=4 ./zig-out/bin/hdr-showcase
AGATE_GPU_TIMING_TEST_FRAMES=270 ./zig-out/bin/gpu-timing

# Запуск тестов Sandbox и многопоточного профилирования
zig build test test-gpu -Doptimize=ReleaseSafe --summary all
zig build bench-threads -Doptimize=ReleaseSafe -- --runs 1 --frames 90 --out /path/to/evidence
```
