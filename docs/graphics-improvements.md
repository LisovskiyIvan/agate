# Графическая волна: SSR, AO и масштабирование сцены

Три независимых изменения, сведённые вместе. Исторические режимы остаются
дефолтными; новые графические пути включаются явно.

## Включение

```zig
scene.post_process.ssr.mode = .half_res_pass;
scene.ssao.mode = .horizon;
scene.ssao.temporal.enabled = true;
```

- **SSR:** half-res trace → confidence-normalized bilateral denoise →
  depth-aware 2×2 reconstruction в полноразмерном композите. Два дополнительных
  draw вместо raymarch внутри композита. Общая реконструкция позиции/нормали
  вынесена в `shaders/common/ssr_trace.glsl`.
- **Horizon AO:** приближённый horizon-поиск по четырём слайсам; это не полная
  аналитическая реализация GTAO. Использует существующие radius/bias/power и
  sample_count (минимум 16 horizon-тапов).
- **Hemisphere AO:** нормализация по сумме cosine-весов выбранных тапов и
  midpoint-стратификация по всему ядру при бюджете 4..32. Убран прежний усиленный
  делитель, превращавший обычные углы в чёрные полосы; включённый hemisphere
  намеренно даёт другую картинку. Вырожденная нормаль даёт AO=1 по относительному
  критерию. Horizon-оценка и temporal lifecycle не менялись.
- **AO temporal:** история AO + oct-нормаль + линейная глубина; velocity с
  camera-reprojection fallback, variance clipping и depth/normal/marker rejection.
  Сброс при смене режима, проекции, размера, camera cut и `resetTaa`.
  MSAA и multi-camera не используют AO-историю.
  Gather вращает направления в восьмикадровом цикле только при реально
  доступном temporal resolve: horizon смещается на π/32 между кадрами
  (64 направления за цикл, не перестановка прежних восьми), hemisphere —
  на 2π/8. При сбросе или отсутствии истории используется нулевая фаза.
  Horizon-поиск прекращает цикл до лишних depth-fetches: 24 вместо 32
  поисковых тапов при стандартном sample_count=24.
- **Opaque sort:** сортируются компактные индексы, затем записи переставляются
  по циклам. Порядок и стабильность прежние, scratch переиспользуется;
  нехватка памяти возвращает прямую сортировку. Прозрачная очередь не менялась.
- **Двусторонний velocity:** opaque velocity и glass reactive markers
  используют frozen `double_sided` материала для rigid/skinned/instanced,
  u16/u32. Cull-off twins покрывают задние грани; односторонние материалы
  сохраняют BACK-cull. Общие layout/depth планы вынесены в фабрику.
- **WebGPU display transfer:** исправлено прежнее wasm-исключение, всегда
  выключавшее manual sRGB encode. Теперь UNORM кодируется в композите,
  sRGB framebuffer — аппаратно, ровно один раз, как на native. Поэтому
  яркость WebGPU/UNORM намеренно меняется относительно ошибочного baseline.
- **Local tonemap:** перенесён фикс из `lighting-local-tonemap`: центральная
  яркость и непороговое пяти-таповое среднее берутся из `scene_tex` до экспозиции,
  без зависимости от bloom. Равномерное поле остаётся нейтральным; выключенная
  ветвь не меняется. CPU и GLSL одинаково ограничивают radiance-тапы.

## Демонстрация и проверки

Из `examples/`:

```sh
# Совместный путь, проверка активности SSR и AO history на кадре 30.
AGATE_Q0_FRAMES=180 AGATE_Q0_STILL=1 AGATE_Q0_SSR_PASS=1 \
  AGATE_Q0_AO_MODE=horizon AGATE_Q0_AO_TEMPORAL=1 \
  zig build run-q0 -Doptimize=ReleaseSafe

# AO history без velocity; дробный render scale проверяет нечётные размеры.
AGATE_Q0_FRAMES=180 AGATE_Q0_STILL=1 AGATE_Q0_SSR_PASS=1 \
  AGATE_Q0_AO_MODE=horizon AGATE_Q0_AO_TEMPORAL=1 \
  AGATE_Q0_NO_TAA=1 AGATE_Q0_RENDER_SCALE=0.67 \
  zig build run-q0 -Doptimize=ReleaseSafe

# GPU-cost A/B в одном окне: inline SSR / half-res SSR.
AGATE_Q0_STILL=1 AGATE_Q0_TIMINGS=1 AGATE_Q0_AB=ssr \
  zig build run-q0 -Doptimize=ReleaseSafe

# CPU-throughput сборки очередей, не графический FPS.
zig build run-scene-scaling -Doptimize=ReleaseFast -- \
  --meshes 24000 --frames 50 --mode both
```

GPU A/B измеряет **стоимость**, а не пиксельный паритет: half-res + denoise
намеренно дают другой результат. Скорость и качество оценивать отдельно;
INCONCLUSIVE не трактовать как ускорение. CPU-протокол и результаты —
[scene-scaling.md](./scene-scaling.md).

### Browser WebGPU и пиксельный smoke

Q0 использует тот же staged-путь и сцену на Metal и WebGPU. Браузерные
query-параметры передаются в перечисленные Q0 env-knobs перед запуском;
новой копии рендера или настроек нет. Из `examples/`:

```sh
# Один раз: SDK должен быть установлен до компиляции WebGPU ports.
# Если bundled Python/Node отсутствуют, используются локальные исполняемые файлы.
EMSDK_PYTHON="$(command -v python3)" EMSDK_NODE="$(command -v node)" \
  zig build install-emsdk -Dtarget=wasm32-emscripten
EMSDK_PYTHON="$(command -v python3)" EMSDK_NODE="$(command -v node)" \
  zig build web-q0-interior -Dtarget=wasm32-emscripten -Doptimize=ReleaseFast
```

Из корня движка:

```sh
node --test ../agate-tools/hdr_browser_gate.test.mjs
node ../agate-tools/hdr_browser_gate.mjs --scene q0
```

### Sandbox: интерактивная лаборатория

В `../sandbox`: TAB открывает HUD, `Gfx` → `PostFX` содержит переключатели
SSR INLINE/HALF, AO HEMI/HORIZON, AO temporal, SSGI enable, SSGI INLINE/HALF
и SSGI temporal. `AO/SSR Demo: Zone 6 Preset` включает horizon/temporal AO,
half-res SSR и half-res/temporal SSGI и переносит камеру в зону 6.
Экспозиция, bloom, fog и прочие независимые настройки сохраняются.

Флаги показывают запрос, не доказательство GPU-активности. MSAA > 1 подавляет
depth-эффекты; multi-camera обходит AO-историю. SSGI temporal в INLINE
показывается как INERT, в HALF применяется через существующий движковый гейт.

Из `../sandbox`:

```sh
zig build run -Doptimize=ReleaseSafe -- --graphics-demo --bench=120
zig build run -Doptimize=ReleaseSafe -- --threaded --graphics-demo --bench=120
```

`--graphics-demo` применяет тот же пресет до запуска worker и имеет приоритет
над `--zone`. Обычный старт и выбор зон 1–8 не меняются. Конечный прогон печатает
`GRAPHICS_DEMO verdict=PASS` при ненулевых AO/SSGI temporal draws, живых history
и привязанном SSR-выходе; это activity/lifecycle smoke, не temporal-quality гейт.

Runner требует реальный WebGPU adapter (не SwiftShader), ненулевой кадр,
SSR inline/pass/off, AO-историю с velocity и camera-only fallback,
нулевые sokol allocations после teardown и отсутствие ошибок валидатора.
Screenshot-сравнение доказывает видимый вклад SSR, +2 post draws для pass,
точное RGBA-совпадение отключённых новых режимов; RMSE inline/half-res
публикуется отдельно, без заявления о равном качестве. Это **не** численный
гейт ghosting/radiance: C.3 в graphics-roadmap остаётся открытым. Артефакты
runner хранит вне репозитория (`--out DIR`). Пороги HDR-сравнения сохранены;
известные строки render-target/phase progress на stderr больше не считаются
ошибками, но validation/panic/FAIL по-прежнему отклоняют прогон.

### Результаты интеграционных гейтов

- Debug и ReleaseSafe: **1774 passed, 6 skipped** в каждом
  (`zig build test -Doptimize=Debug`, `zig build test -Doptimize=ReleaseSafe`).
- Native Metal: два Q0-прогона по 120 кадров — SSR + horizon/temporal AO,
  с TAA и без velocity при scale=0.67; PASS, log_errors=0,
  sokol_live_allocations=0 после teardown.
- WebGPU Q0: **7×120 кадров PASS** на реальном adapter; отключённые режимы
  имеют 0 различающихся RGBA-пикселей, SSR pass даёт +2 post draws и видимый
  вклад. SSR half/inline RMSE RGB ≈0.00195 (screenshot, не radiance).
- WebGPU HDR: **4×240 кадров PASS** (UNORM, MSAA4, sRGB, MSAA2→1 policy).
  После display-transfer фикса meanAbs UNORM/sRGB ≈0.0002, p99≈0.0028
  без ослабления исходных порогов 0.03/0.15.
- Shader compile: Metal/WGSL/HLSL5/GLSL430; browser сборки Q0/HDR/GPU timing
  успешно линкуются. Live Windows/Linux в эту проверку не входят.
- Node-тесты screenshot/diagnostics helper: 5/5 PASS; TAA quality gate tests: 16/16 PASS.
- C.3 WebGPU TAA temporal-гейт: **16/16 ног PASS** на реальном WebGPU adapter
  (`node ../agate-tools/taa_browser_gate.mjs --classes`). Покрыты rigid hero, glass
  pane, skinned bones, 32-instance batch и частицы (reactive-маркер в velocity pass,
  delta draws доказана).
- ReleaseSafe-блокер compressed MP3 memory stream снят: `dr_mp3` удерживал
  self-pointer на скопированное стековое состояние. Декодер теперь владеет
  C-объектом со стабильным адресом; добавлены проверки PCM-паритета memory/file
  через refill/seek/rewind и rollback всех аллокаций. Example harness tests
  ReleaseSafe также проходят.
- После local-tonemap фикса повторены Metal Q0 (120 кадров), WebGPU Q0
  (7×120) и `--scene q0-lighting` (10×120): PASS. Lighting-прогон — изоляционные
  скриншоты, не численный гейт radiance/ghosting и не оценка эстетики.
- После hemisphere-фикса: Metal Q0 — hemisphere/temporal и horizon/temporal
  camera-only при scale=0.67, по 120 кадров; WebGPU Q0 7×120 и lighting 10×120 —
  PASS. A/B с предыдущими lighting-скриншотами: hemisphere изменил 51611
  RGBA-пикселей; horizon, no-AO и direct — точное совпадение (0 changed).
  Это доказательство локальности изменения, не гейт ghosting/blur.
- Sandbox Debug/ReleaseSafe: **14/14** в каждом. Metal graphics-demo serial и
  threaded: по 120 кадров PASS, SSR pass активен, AO/SSGI temporal draws > 0;
  `post_draws=27`. Опции мутируют только game-side, proof читает context-owned
  проходы после остановки worker.

Ранее зафиксированный CPU A/B после интеграции — 2.80x queue-build throughput
на описанной сцене; GPU SSR A/B — INCONCLUSIVE (дельта −0.407 мс ниже
noise_floor 3.180 мс). Ни прирост общего FPS, ни равное качество SSR по этим
данным не заявляются; эти opaque-sort/SSR A/B после local-tonemap и hemisphere
фиксов не повторялись. Новый instanced CPU-baseline приведён отдельно в
[scene-scaling.md](./scene-scaling.md).

### Завершение волны и интеграция

Все ветки и ворктри сведены в `main`:
- Пять `graphics-*` направлений (SSR half-res pass, Horizon/Hemisphere AO, AO temporal history, double-sided velocity pass, indexed opaque sort).
- `lighting-local-tonemap`: синхронизация finite-radiance guards до экспозиции.
- `lighting-ao-corners`: взвешенная hemisphere-нормализация, стратификация ядра и guard вырожденной нормали.
- `C.3 temporal gate`: детерминированная фикстура и 16/16 ног PASS на реальном WebGPU (rigid, glass, skinned, instanced, particles).
Временные ворктри удалены, рабочее дерево чистое и верифицированное.

## Владение и ограничения

- SSR-таргеты создаются лениво и сохраняются после возврата в inline/off,
  чтобы переключение качества не вызывало повторных аллокаций; освобождение —
  resize/deinit. Профайлер считает именно удерживаемую VRAM, не enable-флаги.
- AO temporal-пара освобождается при отключении или пропуске прохода.
  Старый `SSAOPass.render(...)` остаётся spatial-only; temporal требует
  `renderWithContext` с реальными кадровыми матрицами (Scene передаёт их сама).
- SSR сохраняет floor-only ограничение прежнего алгоритма и не отражает
  источники вне экрана. Half-res может терять тонкую геометрию.
- Velocity не хранит предыдущую глубину объекта: AO при движении объекта
  по глубине консервативно отбрасывает историю.
- Новые mode/temporal настройки не добавлены в AGSC: после загрузки состояния
  приложение повторно применяет свой графический пресет.
- Генерация шейдеров проверяется для Metal/WGSL/HLSL5/GLSL430; нативные
  Windows/Linux live-прогоны остаются отдельными платформенными гейтами.
