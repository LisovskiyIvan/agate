# Графика Agate: качество, масштабирование, открытый бэклог

Дата: 08.10.2026.
Это **план и открытый бэклог**, а не список завершённых фич. Все ранее выполненные этапы (консолидация PBR/материалов, clustered shadows/lights, auto-exposure, GGX IBL core, depth pyramid, contact shadows, локальный тонемаппинг/AgX/Filmic, App wrapper) закрыты и исключены из этого документа.

Цель: выразительный PBR-свет и стабильная картинка в заданном бюджете GPU, при этом создание сцены остаётся простым, как в Babylon. Не копировать Unreal целиком и не добавлять архитектуру без проверяемого потребителя.

---

## Открытый TODO-ledger

Каждая задача требует прохождения гейтов (CPU golden + Metal native + browser WebGPU legs) перед закрытием.

### 1. Качество графики (Track C — открытые задачи)

- **C.3. Pixel-level temporal quality gate для TAA / Velocity** (M):
  Пайплайн rigid/skinned/instanced velocity и temporal lifecycle реализованы. Открыт гейт:
  - Количественные метрики качества на уровне пикселей: измерение ghosting/blur при движении.
  - Variance clipping — сделано (variance box + rejection weight в TAA-резолве); reactive treatment v1 — сделано (см. ниже).
  - Сравнение нативного разрешения с render scale + reconstruction.

  - *Статус reactive v1 (09.10.2026):* расследование показало, что частицы (`particle_pass.zig`: depth write OFF) и стекло/прозрачность (прозрачный twin пайплайна: depth write OFF; `itemContributesDepth` исключает transparent/decal/hook из depth prepass и velocity pass) НЕ пишут глубину и НЕ попадают в velocity-буфер — пиксель под частицей несёт глубину и скорость фона, и resolve-side маска не может отличить его по depth/velocity. Поэтому v1 — честный консервативный механизм на сигналах, которые реально есть: fallback-путь (маска velocity < 0.5) + неконсистентная глубина 3×3-креста (колени 0.0001/0.002 raw depth) гасят вес истории (`taa_reactive_params`, `taa_reactive_enabled`/`taa_reactive_strength`, default OFF — бит-идентично; CPU-зеркало `taaReactiveBoost`/`taaResolvePixelReactive`, калибровочная сцена q0 включает через `AGATE_Q0_NO_REACTIVE=1` для baseline). Статичные плоские пиксели и пиксели с per-object velocity получают ровно нулевой буст. Открыто для v2 (нужна MRT/attachment-разводка — территория frame_draws/refactor arc, НЕ делать здесь): запись per-object velocity для частиц/стекла (velocity pass должен перерисовывать их билиборды/прозрачные меши; цена — один дополнительный draw-поток в velocity-таргет) либо отдельный reactive/stencil-таргет из главного прохода.
  - *Статус reactive v2 — true marker для частиц (09.10.2026):* v2 закрывает частичную velocity-перерисовку БЕЗ MRT: velocity pass после opaque-прохода (EQUAL) штампует покрытие частиц константой 0.25 в альфу velocity-буфера (`velocity.glsl`: программы `reactive_marker`/`reactive_marker_gpu`, пайплайны `pip_marker_cpu`/`pip_marker_gpu` с depth LESS_EQUAL — частицы не пишут глубину, их фрагменты лежат ПЕРЕД фоновой глубиной и EQUAL бы их убил; окклюзия ближним opaque отсекается корректно). TAA-резолв классифицирует полосу 0.15–0.35 как reactive (`taaVelocityKind`) и гасит историю по выделенному слоту `taa_reactive_params.z` (`taa_reactive_marker_strength`, default 1.0 = полный reject, без edge ramp — покрытие И ЕСТЬ сигнал), ЗАМЕНЯЯ v1-эвристику на маркированных пикселях; немаркированные пиксели держат v1 без изменений. Опция одна на семейство (`taa_reactive_enabled`, default OFF — бит-идентично; CPU-зеркала `taaReactiveMarkerBoost`/`taaResolvePixelMarker`; q0: пылевой эмиттер `q0_dust` + `AGATE_Q0_NO_PARTICLES=1` для baseline, маркерные draws доказаны счётчиками: +1 main / +1 velocity против particle-free). Честные остатки: (а) СТЕКЛО НЕ маркировано — transparent-mesh redraw (rigid/skinned/instanced LESS_EQUAL marker-пайплайны + transparent-очереди в velocity pass) отложен, там v1-эвристика остаётся единственным лечением; (б) маркер полно-квадный без alpha-test текстуры — прозрачные углы спрайта over-mark'ят фон (ограниченный лишний shed истории, никогда пропущенный ghost); (в) MSAA: маркеры только 1x (velocity pass подавлен под MSAA), а TAA под MSAA выключен целиком — деградация до spatial AA, не до v1.
  - *Статус reactive v2 — маркер для стекла, вторая половина (11.10.2026):* пункт (а) выше закрыт тем же механизмом БЕЗ MRT и БЕЗ второго бэнда/флага: velocity pass после particle-маркеров перерисовывает transparent-очередь (`queues.transparent` + `queues.transparent_instanced` в том же velocity action) константой 0.25 через новые программы `glass_marker`/`glass_marker_inst`/`glass_marker_skin` (`velocity.glsl`: минимальные VS только с current-трансформом — покрытию история не нужна, поэтому instanced-путь биндит один current-буфер вместо пары cur/prev; общий `fs_marker`, то есть значение и контракт записи/чтения те же) и шесть пайплайнов `pip_marker_glass_{rigid,inst,skin}_{u16,u32}` (depth LESS_EQUAL, write OFF, BACK cull как в opaque-пути — односторонняя панель спиной к камере не должна маркировать фон). Покрыты все три формы, которые transparent-очередь вообще может произвести: rigid, skinned (текущие кости), instanced (текущие матрицы). Осознанно НЕ маркируются и остаются на v1-эвристике: декали (несут velocity своего ресивера — его уже записал opaque-проход, маркировка сбрасывала бы валидную историю), hook-материалы (кастомный вершинный стейдж невоспроизводим дешёвым маркером — неверное покрытие хуже отсутствия), morph-смещённые draws (morph-пути нет и в opaque-velocity, пропуск симметричен). Over-mark класс тот же, что у частиц: без alpha-test текстуры маркируется полный треугольник, включая почти-прозрачные тексели. MSAA-статус без изменений: маркеры только 1x, TAA под MSAA выключен. Цена: один дополнительный проход по геометрии transparent-очереди за TAA-кадр (только вершинная работа — константный FS без сэмплов текстур и циклов; перекрывающиеся слои стекла штампуют то же значение, так что overdraw стоит лишь fill, не корректность). q0: тонированная панель `q0_glass_pane` + `AGATE_Q0_NO_GLASS=1` для baseline; маркер доказан счётчиками draws (см. коммит).

- **C.8. Global Illumination (GI)** (L):
  - Выбор и реализация одного продуктового сценария: DDGI (Dynamic Diffuse Global Illumination) либо baked lightmaps / irradiance probes (только после полноценного IBL).
  - *Non-goals:* deferred switch, mesh shaders, аппаратный RTX (сохраняется D3D11 / sokol floor).

### 2. Производительность (Track D — замерить перед оптимизацией)

- **Инструментарий `bench-threads` / `phase_metrics`:**
  - Добавить разбивку по GPU-проходам (GPU-pass breakdown через `?Sample` timing API).
  - Подсчёт per-frame alloc tally в staged слотах.
- **Профилирование кандидатов по коду:**
  - Измерение накладных расходов per-draw uniform uploads в `scene/draw.zig`.
  - Профилирование instance CPU sort в `instance_staging.zig`.
  - Анализ числа мипов bloom (3..7) против fullscreen-проходов.
  - Оценка стоимости CPU occlusion.
  - Калибровка размера пула воркеров jobs.

---

## Волны качества (Q — открытый бэклог)

### Q0. Калибровочные сцены и бюджет

- Создание двух эталонных сцен: 1 аккуратная интерьерная сцена + 1 outdoor-сцена (вместо галереи эффектов): корректные normals/tangents, реальный масштаб, шероховатость, тени, связка sun/sky.
- Фиксация целевых метрик: 60 / 120 FPS, весь GPU-кадр $\le 16.67$ мс с резервом на игровую логику, а не с расходом всего бюджета на постэффекты.
- **PCSS (percentage-closer soft shadows):** солнце — сделано (поиск блокеров 12 сэмплов → средний блокер → радиус полутени → PCF 16/8 тапов); споты — сделано v1 (`SpotLight.source_radius`: 0 = legacy фиксированный 4-тап PCF бит-в-бит, > эпсилон = поиск блокеров 8 сэмплов в своей странице атласа → полутень по той же формуле → переменный PCF 8 тапов; CPU-зеркало в `src/scene/shadow_pcss.zig`, калибровка на q0-интерьере). Поинты — сделано v2 border-correct (`PointLight.source_radius`: 0 = legacy фиксированный 4-тап PCF бит-в-бит, > эпсилон = дисковые оффсеты в world space вокруг ресивера (8 тапов поиска блокеров + 8 тапов PCF) → выбор грани каждого тапа по доминантной оси + проекция через её матрицу → сравнение в глубинном пространстве грани тапа (кросс-грановая нормализация) → средний относительный гэп → полутень по той же формуле → переменный world-space PCF; CPU-зеркало в `src/scene/shadow_pcss.zig`, калибровка на q0-интерьере). Артефакт v1 (кламп тапов к тайлу своей грани, полосы вдоль 12 куб-рёбер) устранён: тапы за границей грани семплят соседнюю грань, вне карты/диапазона тапы скипаются без вклада. Маршрут — derived, без новых юниформов: все 12 матриц граней и позиция света уже были во фрагментном шейдере, тайловые UV выводятся арифметически из известного лэйаута атласа (6 граней слева направо, строка на слот; world-радиус = atlasUV-радиус × 12 × дистанция ресивера). Честное остаточное приближение v2: средний гэп безразмерен, поэтому смесь граней усредняется точно только в одно-граневом случае (там бьёт в среднее v1); у самого ребра same-ray геометрия даёт одинаковое отношение view-глубин во всех гранях, наружу просачивается только нелинейность перспективного мэппинга (второй порядок, обычно съедается клампами полутени); w-лейн `point_shadow_params` по-прежнему занят под `source_radius`.
- **Гейт:** воспроизводимый baseline, лицензированные ассеты, стабильная траектория камеры; скриншоты и видео/прогоны движения, а не только статичный кадр.

### Q2. Пространственные пробы и валидация качества IBL — ЗАВЕРШЕНО
- Валидация качества пространственных проб на GPU: source-mip correctness, баланс энергии (split-sum GGX + cosine irradiance E/pi).
- Box projection для интерьерных зондов: внедрён `src/shaders/common/box_project.glsl` (алгоритм Lagarde 2012), единый лэйаут `vec4 probe_box[2]` синхронизирован во всех 3 PBR-шейдерах (`pbr.glsl`, `skinned_pbr.glsl`, `instanced_pbr.glsl`), CPU reference и тесты в `src/texture/ibl_prefilter.zig`.
- Непрерывный C1 smoothstep переход между перекрывающимися зондами (`blendProbeWeights`) без скачков освещённости.

### Q3. Выбор и прототипирование GI для целевой сцены

Выбрать ровно **один** продуктовый сценарий (не внедрять всё одновременно):
- **Сценарий 1 (статичный интерьер / максимальная скорость):** Baked lightmaps для статики + irradiance probes для динамических объектов. Требует отдельного workflow запекания/импорта, корректных UV1/padding и visibility против утечек света через стены.
- **Сценарий 2 (динамический свет / экранные детали):** SSGI как дополнение на базе HDR radiance и GPU depth pyramid с temporal rejection/denoise. Свет за экраном обязан иметь известный fallback.
  - *Статус v1 (09.10.2026):* экранное color-bleed в композите (`postprocess/ssgi.zig` + `applySSGI`): джиттерный диск-гатер, косинус×фаллофф² веса, firefly-кэп, TAA-денойз; дифф. стоимость +1.3 мс median (16 шагов, 720×540, q0-интерьер). Открыто для v2: отдельный проход в half-res + билатеральный денойз + temporal аккумуляция (нужен отдельный target — P3-территория).
- **Сценарий 3 (динамический offscreen GI):** DDGI spike после выбора источника лучей (geometry/BVH, SDF или proxy-сцена). Probe relocation, бюджет обновления и подавление light leaks.
- **Гейт:** Cornell box, окно / один источник bounce, тонкая перегородка, источник за камерой, движущийся объект.

### Q4. Стабильная temporal-основа → Render Scale / Upscale

- Variance clipping / reactive mask для полупрозрачности, частиц и зон disocclusion (variance box + rejection weight в TAA-резолве — сделано; reactive v1 для velocity-less контента — сделано, см. C.3; reactive v2 — true velocity-alpha marker для ЧАСТИЦ — сделано, см. C.3; маркер для СТЕКЛА (rigid/skinned/instanced transparent redraw) — сделано, см. C.3; на v1-эвристике остаются только декали/hook/morph — см. честные остатки в C.3).
- **Render scale — сделано (Q4-1):** `PostProcessOptions.render_scale` (1.0 = native bit-identical, кламп `[0.25, 1]`). Main HDR-таргет, TAA-история, velocity, SSAO/bloom/glow/shafts/highlight считаются в масштабе; джиттер TAA — в пикселях таргета; композит/UI презентятся в натуральном разрешении (UI остаётся резким). Механика проверена на hdr-showcase (`AGATE_HDR_RENDER_SCALE`, веб: `?rscale=`): таргет 960×600 → 720×450 (0.75) → 634×396 (0.66), кадр презентится, новых валидационных ошибок нет. CPU-гейт: `postprocess/options_tests.zig`.
- **Гейт производительности — отложен (Q4-2):** 60-кадровый прогон в окне даёт mean 10.1 / 8.6 / 9.5 мс (1.0 / 0.75 / 0.66) — шум харнесса (их же раздел «Harness» в `../MEASUREMENTS.md`), не сигнал. Настоящий A/B — на sandbox bench (bench-threads + gpu-timing), и он упирается в известный баг reuse-пути с transient-буферами (`./frame-pipeline.md` → «Известная проблема: transient-буферы в reuse-кадре»). Первый блокер снят: transient-rewrite для reuse-кадров (см. `./frame-pipeline.md`), длинные прогоны hdr-showcase теперь чистые — замер переносится на sandbox bench.
- Сравнение качества и производительности: native resolution vs render scale (0.5x–0.75x) + пространственный/временной апскейл (сейчас — билинейный upscale композита; TAA-реконструкция — следующий слой).
- **Гейт:** статичная мелкая геометрия, быстрое вращение/движение, анимированный персонаж, телепортация камеры, джиттер с преломлением. Оценка GPU-времени и устойчивости к шлейфам (ghosting).

### Q5. Атмосфера, туман и polish-фичи

- Согласованный sky/sun radiance, aerial perspective, height fog: внедрён физический закон Бугера — Ламберта — Бера с интеграцией оптической толщины $\tau = \text{eff\_dist} \cdot \rho_0 \cdot \text{height\_density}$, аналитический интеграл высоты, Mie forward inscattering и horizon haze (CPU golden в `src/postprocess/fog.zig`, GLSL в `postprocess.glsl`).
- Бюджетированный volumetric scattering (развитие после shafts v1).
- Генерализация SSR, contact shadows и GTAO по контракту нормалей и глубин.
- **Гейт:** рассвет/закат, горизонт, контровой свет, стык интерьера и экстерьера без утечек света.

---

## Волны масштабирования GPU (P — открытый бэклог)

### P0. Поиск узкого места через GPU/CPU профилирование

- Изоляция реальных ограничений: batch breaks, bind/uniform churn, геометрия теней, overdraw, bandwidth постпроцесса, переключение вариантов шейдеров.
- Тестовые профили: множество одинаковых копий vs множество уникальных материалов; предельная видимость vs предельное отсечение; тяжёлая пиксельная нагрузка.
- Оптимизировать только доказанные заsnapshotенные узкие места.

### P1. Минимальный Indirect-контракт в форках Sokol

- Переход от CPU-authored indexed arguments к compute-authored arguments для **фиксированных батчей** (без bindless / meshlets в первой фазе).
- Требования: indirect + storage usage, выравнивание ABI, index type, capability gates базовых вершин/инстансов, валидация границ, синхронизация compute-write $\to$ indirect-read.
- **Гейт:** идентичность пикселей и счётчиков между CPU- и GPU-authored args; поведение при 0 инстансов, рост буферов, тени и несколько viewports.

### P2. GPU Visibility

- Реализация GPU culling для фиксированных material/pipeline батчей: frustum culling, LOD selection, компактные instance ID.
- Иерархический GPU depth pyramid culling (Hi-Z на GPU), temporal conservativeness и отсечение теневых каскадов.
- Маршрутизация mesh/material/probe ID без разрушения батчинга.

### P3. Frame Graph по ресурсным конфликтам

- Введение графа зависимостей проходов при росте числа targets (HDR, normals, velocity, depth pyramid, history):
  - Валидация чтения/записи (read/write hazards).
  - Автоматическое управление временем жизни targets при ресайзе и смене истории.
  - Повторное использование (aliasing/reuse) промежуточных scratch-текстур.

### P4. Геометрическая детализация

- Spatial chunking, потоковая загрузка текстурных мипов (texture streaming), impostors.
- Meshlets / visibility buffer / virtual geometry — только если доказано доминирование геометрической плотности сцены.

---

## Ближайший фокус и критерий остановки

1. **Фокус:** Q0 (калибровочная сцена) $\to$ Q2 (валидация IBL/проб) $\to$ Q3 (выбор единственного GI-трека).
2. **Параллельно:** P0 (профилирование узких мест) $\to$ P1 (indirect прототип при наличии выигрыша).
3. **Критерий остановки:** достижение целевой визуальной планки и соблюдение бюджета 16.67 мс / 8.33 мс на целевом оборудовании. Не внедрять эффекты и абстракции без измеримой пользы.

---

## Первичные источники

- [Filament: physically based rendering, IBL и lighting](https://google.github.io/filament/Filament.html)
- [Babylon: HDR environment](https://doc.babylonjs.com/features/featuresDeepDive/materials/using/HDREnvironment)
- [Epic: Lumen Technical Details](https://dev.epicgames.com/documentation/unreal-engine/lumen-technical-details-in-unreal-engine)
- [DDGI: Dynamic Diffuse Global Illumination with Ray-Traced Irradiance Fields](https://jcgt.org/published/0008/02/01/)
- [WebGPU: indexed indirect draw](https://www.w3.org/TR/webgpu/#dom-gpurendercommandsmixin-drawindexedindirect)
