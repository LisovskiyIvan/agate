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
