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
