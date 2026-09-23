//! Profiler bottleneck diagnostics. Split out of `profiler.zig` (facade).
//!
//! `analyze` takes the profiler as `anytype` (a `*const Profiler` from
//! `core.zig` in practice) so this module never imports `core.zig` or the
//! facade back — same discipline as `ui/*` taking a generic canvas.
//! `core.zig` owns the `Profiler` type and forwards `analyze` here.
//! Leaf: imports `types.zig` only (`self.summarize()` resolves on the
//! caller's concrete profiler, no sibling import needed).

const std = @import("std");

const types = @import("types.zig");

const MemorySnapshot = types.MemorySnapshot;
const DiagnosticSeverity = types.DiagnosticSeverity;
const DiagnosticFinding = types.DiagnosticFinding;
const dominantPhase = types.dominantPhase;

/// Runs automated bottleneck diagnostics on recorded metrics and memory.
pub fn analyze(self: anytype, memory: ?*const MemorySnapshot, allocator: std.mem.Allocator) ![]DiagnosticFinding {
    var findings = std.ArrayListUnmanaged(DiagnosticFinding).empty;
    errdefer {
        for (findings.items) |*f| f.deinit(allocator);
        findings.deinit(allocator);
    }

    const summary = self.summarize();

    // 1. Frame Pacing / Target FPS Check (observed wall-clock intervals).
    // Phase attribution in section 2 stays CPU-submit based.
    if (summary.frame_count >= 5 and summary.max_interval_ms > 0) {
        if (summary.observed_avg_fps >= 58.0 and summary.p99_interval_ms <= 18.0 and summary.interval_hitches_over_33ms == 0) {
            try findings.append(allocator, .{
                .severity = .good,
                .title = try allocator.dupe(u8, "Стабильный кадровый темп 60+ FPS"),
                .details = try std.fmt.allocPrint(allocator, "Наблюдаемый средний FPS (wall-интервал): {d:.1}, P99 интервала: {d:.1} мс, средний CPU-submit: {d:.2} мс. Просадок ниже 30 FPS не зафиксировано.", .{ summary.observed_avg_fps, summary.p99_interval_ms, summary.avg_frame_ms }),
                .recommendation = try allocator.dupe(u8, "Производительность соответствует целевому бюджету времени кадра (16.6 мс)."),
            });
        } else if (summary.interval_hitches_over_33ms > 0) {
            const sev: DiagnosticSeverity = if (summary.interval_hitches_over_33ms > summary.frame_count / 10 or summary.max_interval_ms > 50.0) .critical else .warning;
            try findings.append(allocator, .{
                .severity = sev,
                .title = try allocator.dupe(u8, "Просадки кадровой частоты ниже 30 FPS"),
                .details = try std.fmt.allocPrint(allocator, "Зафиксировано {d} wall-интервалов > 33.3 мс (худший интервал: {d:.1} мс, 1% Low (wall): {d:.1} FPS). Пиковый CPU-submit кадра: {d:.1} мс.", .{ summary.interval_hitches_over_33ms, summary.max_interval_ms, summary.observed_fps_1pct_low, summary.max_frame_ms }),
                .recommendation = try allocator.dupe(u8, "Изучите таблицу Spike Frames (время CPU-submit) для определения виновной фазы и исключите блокирующие операции на главном потоке."),
            });
        } else if (summary.interval_hitches_over_16ms > 0) {
            try findings.append(allocator, .{
                .severity = .info,
                .title = try allocator.dupe(u8, "Периодические просадки ниже 60 FPS"),
                .details = try std.fmt.allocPrint(allocator, "Зафиксировано {d} wall-интервалов длительностью от 16.7 до 33.3 мс.", .{summary.interval_hitches_over_16ms}),
                .recommendation = try allocator.dupe(u8, "Оптимизируйте наиболее тяжелые CPU-фазы (Main pass, Shadows) для достижения чистых 60 FPS."),
            });
        }
    }

    // 2. Worst Phase Attribution on Spikes (CPU-submit times, not GPU time)
    if (self.frames.items.len > 0) {
        var worst_frame = self.frames.items[0];
        for (self.frames.items) |f| {
            if (f.total_frame_ms > worst_frame.total_frame_ms) worst_frame = f;
        }

        if (worst_frame.total_frame_ms > 20.0) {
            const culprit = dominantPhase(worst_frame);
            if (std.mem.eql(u8, culprit.name, "Main Pass") and culprit.percent > 45.0) {
                try findings.append(allocator, .{
                    .severity = .warning,
                    .title = try allocator.dupe(u8, "Узкое горлышко: Main Render Pass"),
                    .details = try std.fmt.allocPrint(allocator, "В пиковом кадре #{d} (CPU-submit {d:.1} мс) фаза Main Pass (CPU submit) заняла {d:.1} мс ({d:.1}% всего кадра).", .{ worst_frame.frame_index, worst_frame.total_frame_ms, culprit.ms, culprit.percent }),
                    .recommendation = try allocator.dupe(u8, "Сократите количество вызовов отрисовки через InstancedMesh, объедините меши с одинаковыми материалами и включите Occlusion Culling."),
                });
            } else if (std.mem.eql(u8, culprit.name, "Shadow Pass") and culprit.percent > 35.0) {
                try findings.append(allocator, .{
                    .severity = .warning,
                    .title = try allocator.dupe(u8, "Узкое горлышко: CSM Shadow Pass"),
                    .details = try std.fmt.allocPrint(allocator, "В пиковом кадре #{d} рендеринг теней (CPU submit) занял {d:.1} мс ({d:.1}% кадра).", .{ worst_frame.frame_index, culprit.ms, culprit.percent }),
                    .recommendation = try allocator.dupe(u8, "Отключите cast_shadows для мелких мешей, уменьшите дистанцию теневых каскадов или отключите тени для точечных источников."),
                });
            } else if (std.mem.eql(u8, culprit.name, "Update") and culprit.percent > 40.0) {
                try findings.append(allocator, .{
                    .severity = .warning,
                    .title = try allocator.dupe(u8, "Узкое горлышко: CPU Update / Скрипты"),
                    .details = try std.fmt.allocPrint(allocator, "В пиковом кадре #{d} обновление логики заняло {d:.1} мс ({d:.1}% CPU-submit кадра).", .{ worst_frame.frame_index, culprit.ms, culprit.percent }),
                    .recommendation = try allocator.dupe(u8, "Оптимизируйте анимации, физическую симуляцию или перенесите тяжелые расчеты в фоновый пул jobs.TaskRunner."),
                });
            } else if (std.mem.eql(u8, culprit.name, "Physics / Box3D") and culprit.percent > 30.0) {
                try findings.append(allocator, .{
                    .severity = .warning,
                    .title = try allocator.dupe(u8, "Узкое горлышко: Физика (Box3D)"),
                    .details = try std.fmt.allocPrint(allocator, "В пиковом кадре #{d} физическая симуляция заняла {d:.1} мс ({d:.1}% CPU-submit кадра).", .{ worst_frame.frame_index, culprit.ms, culprit.percent }),
                    .recommendation = try allocator.dupe(u8, "Уменьшите количество сабстепов PhysicsWorld, включите непрерывное детектирование (CCD) только для скоростных пуль, используйте примитивные коллайдеры вместо сложных мешей."),
                });
            } else if (std.mem.eql(u8, culprit.name, "PostFX") and culprit.percent > 40.0) {
                try findings.append(allocator, .{
                    .severity = .warning,
                    .title = try allocator.dupe(u8, "Узкое горлышко: PostFX Stack"),
                    .details = try std.fmt.allocPrint(allocator, "В пиковом кадре #{d} пост-обработка (CPU submit) заняла {d:.1} мс ({d:.1}% кадра).", .{ worst_frame.frame_index, culprit.ms, culprit.percent }),
                    .recommendation = try allocator.dupe(u8, "Проверьте настройки SSAO (уменьшите sample count), снизьте bloom pyramid mips или уменьшите разрешение буфера."),
                });
            }

            if (worst_frame.uploaded_bytes > 4 * 1024 * 1024) {
                try findings.append(allocator, .{
                    .severity = .warning,
                    .title = try allocator.dupe(u8, "Задержка из-за стриминга текстур на GPU"),
                    .details = try std.fmt.allocPrint(allocator, "В кадре #{d} стрим текстур (UploadQueue) записал {d:.2} МБ на видеокарту.", .{ worst_frame.frame_index, @as(f32, @floatFromInt(worst_frame.uploaded_bytes)) / (1024.0 * 1024.0) }),
                    .recommendation = try allocator.dupe(u8, "Используйте асинхронную очередь Scene.uploads (UploadQueue) с лимитом байт на кадр (max_bytes_per_frame)."),
                });
            }

            // Порог 16 MiB = 2x текстурного бюджета: легитимные массовые
            // изменения (полная перезаливка инстансов/морфов) дают единицы
            // МБ (100k инстансов x 64 Б = 6.4 МБ), а UI/debug/трейлы —
            // десятки-сотни КБ. Выше — патологическая перезапись динамики.
            if (worst_frame.updated_bytes > 16 * 1024 * 1024) {
                try findings.append(allocator, .{
                    .severity = .warning,
                    .title = try allocator.dupe(u8, "Массовые динамические обновления GPU-буферов"),
                    .details = try std.fmt.allocPrint(allocator, "В кадре #{d} динамические буферы (sg.updateBuffer: инстансы, морфы, частицы) записали {d:.2} МБ.", .{ worst_frame.frame_index, @as(f32, @floatFromInt(worst_frame.updated_bytes)) / (1024.0 * 1024.0) }),
                    .recommendation = try allocator.dupe(u8, "Проверьте частоту полных перезаливок instance-буферов (dedup по hash+count уже пропускает неизменные), вес CPU-морфов и число активных частиц."),
                });
            }
        }
    }

    // 3. Draw Calls & Pipeline Switches
    if (summary.frame_count > 0) {
        if (summary.avg_draw_calls > 1000) {
            try findings.append(allocator, .{
                .severity = .critical,
                .title = try allocator.dupe(u8, "Критическое количество Draw Calls (> 1000)"),
                .details = try std.fmt.allocPrint(allocator, "В среднем {d} draw calls за кадр (максимум {d}).", .{ summary.avg_draw_calls, summary.max_draw_calls }),
                .recommendation = try allocator.dupe(u8, "Используйте InstancedMesh для повторяющихся объектов, объединяйте статические меши и проверьте фрустум-куллинг."),
            });
        } else if (summary.avg_draw_calls > 400) {
            try findings.append(allocator, .{
                .severity = .warning,
                .title = try allocator.dupe(u8, "Повышенное количество Draw Calls (> 400)"),
                .details = try std.fmt.allocPrint(allocator, "В среднем {d} draw calls за кадр (максимум {d}).", .{ summary.avg_draw_calls, summary.max_draw_calls }),
                .recommendation = try allocator.dupe(u8, "Рекомендуется батчинг и инстансинг для снижения нагрузки на CPU драйвера."),
            });
        }

        if (summary.avg_pipeline_switches > 120) {
            try findings.append(allocator, .{
                .severity = .warning,
                .title = try allocator.dupe(u8, "Частые переключения шейдеров и пайплайнов"),
                .details = try std.fmt.allocPrint(allocator, "В среднем {d} переключений пайплайнов за кадр.", .{summary.avg_pipeline_switches}),
                .recommendation = try allocator.dupe(u8, "Сортируйте меши по материалам и шейдерам перед отрисовкой для минимизации смены состояний GPU."),
            });
        }
    }

    // 4. Memory & Assets Checks (if snapshot available)
    if (memory) |mem| {
        // High total VRAM check
        if (mem.total_gpu_vram_bytes > 512 * 1024 * 1024) {
            try findings.append(allocator, .{
                .severity = .warning,
                .title = try allocator.dupe(u8, "Высокое потребление видеопамяти (> 512 МБ)"),
                .details = try std.fmt.allocPrint(allocator, "Суммарно VRAM: {d:.1} МБ (текстуры: {d:.1} МБ, меши: {d:.1} МБ, буферы кадров: {d:.1} МБ).", .{
                    @as(f32, @floatFromInt(mem.total_gpu_vram_bytes)) / (1024.0 * 1024.0),
                    @as(f32, @floatFromInt(mem.textures_vram_bytes)) / (1024.0 * 1024.0),
                    @as(f32, @floatFromInt(mem.meshes_vram_bytes)) / (1024.0 * 1024.0),
                    @as(f32, @floatFromInt(mem.render_targets_vram_bytes)) / (1024.0 * 1024.0),
                }),
                .recommendation = try allocator.dupe(u8, "Примените KTX2 GPU-сжатие текстур (Basis Universal / BCn / ETC2) для снижения объема памяти на 70-80%."),
            });
        }

        // Heavy uncompressed textures
        for (mem.textures) |tex| {
            if (tex.width >= 2048 and tex.height >= 2048 and tex.gpu_bytes >= 16 * 1024 * 1024 and tex.format == .RGBA8) {
                try findings.append(allocator, .{
                    .severity = .warning,
                    .title = try allocator.dupe(u8, "Тяжелая несжатая 2K+ текстура"),
                    .details = try std.fmt.allocPrint(allocator, "Текстура '{s}' ({d}x{d}, RGBA8) занимает {d:.1} МБ видеопамяти.", .{ tex.name, tex.width, tex.height, @as(f32, @floatFromInt(tex.gpu_bytes)) / (1024.0 * 1024.0) }),
                    .recommendation = try allocator.dupe(u8, "Сконвертируйте текстуру в формат KTX2 с блочным сжатием (BC7 / BC1 / ETC1S) или уменьшите разрешение до 1024x1024."),
                });
                break;
            }
        }

        // Missing mipmaps
        for (mem.textures) |tex| {
            if (tex.width >= 512 and tex.height >= 512 and tex.num_mips <= 1 and !tex.is_cube) {
                try findings.append(allocator, .{
                    .severity = .warning,
                    .title = try allocator.dupe(u8, "Отсутствуют mip-уровни у большой текстуры"),
                    .details = try std.fmt.allocPrint(allocator, "Текстура '{s}' ({d}x{d}) загружена без цепочки mipmaps.", .{ tex.name, tex.width, tex.height }),
                    .recommendation = try allocator.dupe(u8, "Включите генерацию mipmaps (options.mipmaps = true) для устранения мерцания при удалении и повышения попаданий в кэш GPU."),
                });
                break;
            }
        }

        // Inefficient 32-bit indices
        var wasteful_u32_count: usize = 0;
        for (mem.meshes) |m| {
            if (m.index_type == .UINT32 and m.vertex_count > 0 and m.vertex_count <= 65535) {
                wasteful_u32_count += 1;
            }
        }
        if (wasteful_u32_count > 0) {
            try findings.append(allocator, .{
                .severity = .info,
                .title = try allocator.dupe(u8, "Неоптимальные 32-битные индексные буферы"),
                .details = try std.fmt.allocPrint(allocator, "{d} мешей используют UINT32 индексы при менее чем 65536 вершинах.", .{wasteful_u32_count}),
                .recommendation = try allocator.dupe(u8, "UINT16 индексы экономят 50% памяти индексов и снижают нагрузку на шину памяти GPU."),
            });
        }

        // Heavy dense meshes
        for (mem.meshes) |m| {
            if (m.vertex_count > 60000) {
                try findings.append(allocator, .{
                    .severity = .warning,
                    .title = try allocator.dupe(u8, "Высокая плотность вершин в меше (> 60k)"),
                    .details = try std.fmt.allocPrint(allocator, "Меш '{s}' содержит {d} вершин и {d} индексов ({d:.1} МБ VRAM).", .{ m.name, m.vertex_count, m.index_count, @as(f32, @floatFromInt(m.gpu_bytes)) / (1024.0 * 1024.0) }),
                    .recommendation = try allocator.dupe(u8, "Используйте meshopt simplification для генерации уровней детализации (LOD)."),
                });
                break;
            }
        }
    }

    // Default good condition if no warnings or criticals were produced
    var has_issues = false;
    for (findings.items) |f| {
        if (f.severity == .warning or f.severity == .critical) {
            has_issues = true;
            break;
        }
    }
    if (!has_issues and findings.items.len == 0) {
        try findings.append(allocator, .{
            .severity = .good,
            .title = try allocator.dupe(u8, "Все основные параметры в норме"),
            .details = try allocator.dupe(u8, "Критических задержек, перерасхода памяти или чрезмерного количества вызовов отрисовки не обнаружено."),
            .recommendation = try allocator.dupe(u8, "Текущая конфигурация сцены работает оптимально."),
        });
    }

    return findings.toOwnedSlice(allocator);
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

/// Sleep helper for pacing-sensitive tests (Zig 0.16 has no Thread.sleep).
fn testSleepMs(ms: u64) void {
    const ts = std.c.timespec{ .sec = @intCast(ms / 1000), .nsec = @intCast((ms % 1000) * 1_000_000) };
    var rem: std.c.timespec = undefined;
    _ = std.c.nanosleep(&ts, &rem);
}

test "Profiler analyzer detects bottlenecks" {
    const Profiler = @import("core.zig").Profiler;
    const sokol = @import("sokol");
    const SceneStats = @import("../scene/stats.zig").SceneStats;
    sokol.time.setup();
    const ally = std.testing.allocator;
    var prof = Profiler.init(ally);
    defer prof.deinit();

    prof.start();
    // Simulate high draw calls and severe hitch. Pacing findings now come from
    // real wall-clock intervals, so sleep ~40ms between records to register
    // genuine interval hitches (~0.4s total for 10 frames).
    var stats: SceneStats = .{
        .update_ms = 1.0,
        .prepare_ms = 0.5,
        .shadow_ms = 1.0,
        .main_ms = 45.0, // huge hitch in main pass
        .post_ms = 1.0,
        .draw_calls = 1200, // critical draw call count
        .triangles = 200000,
        .pipeline_switches = 150,
    };
    for (0..10) |i| {
        if (i > 0) testSleepMs(40);
        prof.recordFrame(i, &stats);
    }
    prof.stop();

    const summary = prof.summarize();
    // Pacing hitches must be observed on the wall clock, not just CPU sums.
    try std.testing.expect(summary.interval_hitches_over_33ms > 0);

    const findings = try prof.analyze(null, ally);
    defer {
        for (findings) |*f| @constCast(f).deinit(ally);
        ally.free(findings);
    }

    var found_draw_calls = false;
    var found_main_pass = false;
    var found_hitch = false;

    for (findings) |f| {
        if (std.mem.indexOf(u8, f.title, "Draw Calls") != null) found_draw_calls = true;
        if (std.mem.indexOf(u8, f.title, "Main Render Pass") != null) found_main_pass = true;
        if (std.mem.indexOf(u8, f.title, "Просадки") != null) found_hitch = true;
    }

    try std.testing.expect(found_draw_calls);
    try std.testing.expect(found_main_pass);
    try std.testing.expect(found_hitch);
}
