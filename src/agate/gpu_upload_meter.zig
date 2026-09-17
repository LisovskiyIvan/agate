const std = @import("std");

// Счётчик байтов динамических обновлений GPU-буферов за текущий кадр.
//
// Каждый фактический вызов sg.updateBuffer / sg.appendBuffer в движке
// сопровождается вызовом record() с числом байт переданного диапазона.
// Метрика uncounted-budget: на троттлинг текстурного стриминга
// (upload_byte_budget_per_frame = 8 MiB) не влияет, служит только честной
// диагностике в SceneStats.updated_bytes_frame и профайлере.
//
// Потокобезопасность: стейджинг инстансов может выполняться с воркеров
// пула (scene/instance_staging.zig), поэтому счётчик атомарный.
// Сброс раз в кадр: Scene.prepareFrame обнуляет счётчик в начале кадра,
// Scene.render забирает значение в stats.updated_bytes_frame перед
// Profiler.recordFrame (включая UI/debug-апдейты, идущие уже внутри render).
var pending_bytes: std.atomic.Value(u64) = std.atomic.Value(u64).init(0);

/// Учитывает очередные `bytes` байт, записанные в GPU-буфер.
/// Вызывать строго рядом с фактическим sg.updateBuffer/appendBuffer,
/// внутри того же guard'а (без sokol-контекста эти ветки недостижимы,
/// счётчик в тестах остаётся нулевым).
pub fn record(bytes: usize) void {
    _ = pending_bytes.fetchAdd(@as(u64, @intCast(bytes)), .monotonic);
}

/// Забирает накопленное значение и обнуляет счётчик (раз в кадр).
pub fn takeAndReset() u64 {
    return pending_bytes.swap(0, .acq_rel);
}

/// Текущее значение без сброса (для тестов и отладки).
pub fn peek() u64 {
    return pending_bytes.load(.monotonic);
}

test "record накапливает, takeAndReset возвращает сумму и обнуляет" {
    _ = takeAndReset();
    record(100);
    record(56);
    try std.testing.expectEqual(@as(u64, 156), peek());
    try std.testing.expectEqual(@as(u64, 156), takeAndReset());
    try std.testing.expectEqual(@as(u64, 0), peek());
    try std.testing.expectEqual(@as(u64, 0), takeAndReset());
}

test "параллельные record с воркеров не теряют байты" {
    _ = takeAndReset();
    const Worker = struct {
        fn run(n: usize) void {
            var i: usize = 0;
            while (i < n) : (i += 1) record(64);
        }
    };
    const thread_count = 4;
    const per_thread = 1000;
    var threads: [thread_count]std.Thread = undefined;
    for (&threads) |*t| t.* = try std.Thread.spawn(.{}, Worker.run, .{per_thread});
    for (&threads) |*t| t.join();
    try std.testing.expectEqual(@as(u64, thread_count * per_thread * 64), takeAndReset());
}
