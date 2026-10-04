# Perf baseline (worktree `perf`)

Дата: 05.10.2026. Хост: Apple M4, macOS, ReleaseFast (bench) / ReleaseSafe (timing).
Сцена: sandbox showcase (windowed, vsync off; wall сошёлся к 16.667 ms — 60 Hz
композиции окна).

## CPU (bench-threads, 3 runs × 300 frames, means, ms)

| mode | waitC | holdC | prepare | update | gpuSubmit | cbWall |
|---|---|---|---|---|---|---|
| staged | 0.000 | 0.152 | 0.177 | 0.672 | 1.056 | 1.218 |
| exclusion | 0.131 | 0.136 | 0.158 | 0.619 | 0.957 | 1.228 |
| serial | 0.000 | 0.134 | 0.156 | 1.951 | 0.878 | 2.876 |

`gpuSubmit` — CPU wall подачи scene draws (не GPU время). Context-кадр
(prepare + submit) ≈ 1.2 ms при бюджете 16.7 ms: CPU headroom ~13×.

## GPU (gpu-timing gate, Metal, ReleaseSafe, 270 frames, PASS)

Полный GPU кадр (timer=0, scene): **5.1–6.4 ms**. Все 4 таймера дают
уникальные положительные семплы (80/79/79/79 на волну), compute_dispatches=268,
0 ошибок, 0 живых аллокаций.

## Выводы

- Движок НЕ CPU-bound (1.2/16.7 ms) и не упирается в GPU (≈6/16.7 ms) на
  этой сцене; до 60 fps запас ~2.7× по GPU.
- Доминирующая статья context-потока — submit draws (~1.06 ms): per-draw
  `sg.applyUniforms` в `scene/draw.zig` (vs_params на меш + fs_params на
  материал). Следующий реальный шаг — instance/storage-buffer packing
  (gpu-driven батчи) — это архитектурная волна, не точечный фикс.
- serial update (1.95 ms vs 0.67 staged) — ожидаемо: один поток делает
  update+render последовательно; cbWall serial 2.88 ms всё ещё < 18% бюджета.
- exclusion-режим стабильно дороже на waitC (0.131 ms) — цена диагностического
  мьютекса, данных не меняет.

## Правило волны

Не оптимизировать без замера: bench-threads (CPU) + gpu-timing (GPU) до и
после каждого изменения; сравнивать только same-machine back-to-back.

## Волна 1 (05.10.2026): uniform-upload dedup — отрицательный результат

Кандидат (откачен, в шип не вошёл): `scene/draw.zig` — пропуск
повторных `sg.applyUniforms` при побайтово идентичном блоке на том же
ключе (pipeline id + UB slot), статичный threadlocal-кэш 16 слотов,
без аллокаций на кадр. Дедуплицировались только fs_params / vs_morph /
instanced-vs (regular vs_params всегда уникален — mvp/model).
Очередь не менялась (opaque уже группируется по texture_id).

Замер staged gpuSubmit (CPU submit wall, ms), same-machine back-to-back:

| замер | run1 | run2 | run3 | mean |
|---|---|---|---|---|
| baseline | — | — | — | 1.056 |
| after (A) | 1.091 | 1.013 | 0.251* | 0.785 |
| after (B) | 1.184 | 1.050 | 0.770 | 1.001 |

\* run3(A) — артефакт окружения: одновременно рухнули prepare/holdC
при живых 16.667 ms wall (окклюзия окна, см. предупреждение harness);
к сравнению непригоден.

Вердикт: на прогретых ранах 1.01–1.18 против baseline 1.056 —
движения нет (разброс ±20% — шум оконного harness). Вывод:
многокилобайтный memcpy fs_params — не доминанта submit-стены;
её определяют фиксированные per-draw издержки (applyBindings +
Metal-диспетч кодера), дедуп их не снимает. Сортировка opaque под
дедуп (план Б) при таком шуме измеримые ≥15% дать не может —
не делаем. Кандидат откачен (`git status` чист от кода волны),
сложность без выигрыша не шипаем. Gates кандидата: `zig build test`
зелёный (1408/1408, exit 0; строка "failed command" в логе —
предсуществующий шум harness, есть и на baseline); gpu-timing на
откаченном дереве = baseline PASS из таблицы выше.

## Harness (04.10.2026): occlusion-gate + median/p95 для gpuSubmit

Проблема: gpuSubmit-шум ±20% (1.091/1.013/0.251 — run3 с рухнувшими
prepare/holdC при живом wall 16.667 ms, окклюзия окна) делал gates
неразрешимыми. Фикс (только harness, поведения движка не меняет):

- Occluded/minimized-кадры (ICONIFIED/SUSPENDED, окно hidden, нулевой
  framebuffer) по-прежнему рендерятся, но НЕ семплируют: ни phase-rings,
  ни frame_stats-суммы/серии. Считаются отдельно: `occluded_frames`
  (phase, post-warmup 30) + `Occluded/minimized frames` (frame_stats,
  post-warmup 5) — никогда молча не дропаются и не двоятся с busy_skips
  (ровно один tally на кадр). Знаменатели средних уменьшаются на occluded,
  средние остаются точными по семплированным кадрам.
- `Render GPU Scene submit series`: per-frame серия submit-стен
  (та же популяция, что и у среднего, n=295 при --frames 300) —
  median/p95; среднее оставлено для непрерывности.
- Warmup: phase-сторона — общий 30-кадровый gate (без изменений);
  frame_stats-серия — 5-кадровый gate (как было у среднего).
- Парсер `measure_threads.py`: колонки `sub_med`/`sub_p95`/`occl` +
  spread медиан/p95 и список occluded; отсутствие любого нового ключа —
  громкий FAIL (проверено негативными пробами на логе).
- Ограничение: полная окклюзия чужим окном на macOS не даёт sokol-события
  и НЕ детектируется — окно бенча держать frontmost; остаток ловится
  spread-проверкой между ранами.

Замер staged gpuSubmit (3×300, same-machine back-to-back, wall 16.667
стабилен, occluded=[0,0,0] во всех модах):

| метрика | run1 | run2 | run3 | spread |
|---|---|---|---|---|
| submit mean | 0.953 | 1.007 | 0.971 | 5.7% |
| submit median | 1.045 | 1.078 | 1.045 | 3.2% |
| submit p95 | 1.642 | 1.603 | 1.621 | 2.4% |

Было: 1.091/1.013/0.251 (артефакт) → прогретая полоса 1.01–1.18
(±20%). Стало: медианы 1.045–1.078 (spread <5%). Gate ≥15% из волны 1
снова разрешим: медиана — канонический сигнал для сравнений до/после.

## Волна 2 (04.10.2026): vertex-pulled batching — отрицательный результат, прототип не строился

API-фиasibility: ДА, блокера в sokol/shdc НЕТ. `sokol_gfx.h` (§storage
buffers) документирует vertex pulling без `.layout` (vertexpull /
instancing-pull samples) и shdc-синтаксис
`layout(binding=N) readonly buffer` + `gl_VertexIndex` в `@vs`; тот же
синтаксис уже живёт в дереве (`common/cluster.glsl` bindings 12–14,
`particle_compute.glsl`), storage views биндятся через `views[]`
(`draw.zig:652` `bindClusteredViews`). Пустой vertex layout валидацией
не запрещён.

Потолок сцены: батчить НЕЧЕГО. Gate-замер ниже (код не менялся):
primary view рисует opaque=34/58/34 (по прогонам), opaque_inst=0,
trans=6 (`mesh-vanish-probe` frame 0); в showcase ~61 distinct material
(`createPBRMaterial|createStandardMaterial` в `sandbox_showcase.zig`).
Единственная same-material группа (32 hidden gems, один `gem_mat`)
за стеной и окклюдится — иначе заняла бы почти всю очередь из 34.
Same-geometry группы (6 hysteresis orbs) — 6 РАЗНЫХ материалов
(per-color), им нужен per-instance material indexing (вне bounded
scope). Прототип (новое семейство шейдеров + пайплайны + merge staging
поверх P4-снапшотов очередей) снял бы ~0 draws → gate ≥15% недостижим
по построению. Не строим (прецедент волны 1, план Б); сложности без
выигрыша не шипаем, откатывать нечего (`git status` чист в обоих
worktree).

Замер staged gpuSubmit 3×300 (same-machine, wall 16.667–16.669,
occluded=[0,0,0], без изменений кода — сертификация harness):

| метрика | run1 | run2 | run3 | spread |
|---|---|---|---|---|
| submit mean | 1.556 | 1.559 | 1.563 | 0.4% |
| submit median | 1.541 | 1.546 | 1.551 | 0.6% |
| submit p95 | 1.625 | 1.614 | 1.632 | 1.1% |

Gates: `zig build test` exit 0 (зелёный; строка "failed command" —
предсуществующий шум harness, как в волне 1).

Дрейф: медиана 1.541–1.551 против baseline 1.045–1.078 (+47%) при
чистых деревьях — machine-state, не код. Правило: абсолютные медианы
между сессиями НЕ переносить; все сравнения до/после — только
back-to-back внутри одной сессии (spread внутри сессии 0.6%).
