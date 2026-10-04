//! Tests for `gpu_retire.zig` (moved verbatim from inline blocks; production code unchanged).
const std = @import("std");
const Mesh = @import("../mesh.zig").Mesh;
const gpu_thread = @import("../gpu_thread.zig");
const prod = @import("gpu_retire.zig");
const GpuRetireQueue = prod.GpuRetireQueue;
const Kind = prod.Kind;
const ComputeBundle = prod.ComputeBundle;

fn makeMesh(allocator: std.mem.Allocator, name: []const u8) !*Mesh {
    const m = try allocator.create(Mesh);
    m.* = @import("../testing.zig").testMesh(name);
    return m;
}

test "retire ждёт complete своего epoch; flush идемпотентен" {
    const alloc = std.testing.allocator;
    var q: GpuRetireQueue = .{};
    defer q.deinit(alloc);

    const e = q.begin();
    const m = try makeMesh(alloc, "epoch_probe");
    q.retireMesh(alloc, m);
    try std.testing.expectEqual(@as(usize, 1), q.retainedCount());
    // Flush до complete(e): запись текущего незавершённого epoch ждёт.
    q.flush(alloc);
    try std.testing.expectEqual(@as(usize, 1), q.retainedCount());
    // После complete(e): ровно одно уничтожение; повторный flush — no-op.
    q.complete(e);
    q.flush(alloc);
    try std.testing.expectEqual(@as(usize, 0), q.retainedCount());
    q.flush(alloc);
    try std.testing.expectEqual(@as(usize, 0), q.retainedCount());
}

test "несколько epoch: flush уничтожает только завершённые" {
    const alloc = std.testing.allocator;
    var q: GpuRetireQueue = .{};
    defer q.deinit(alloc);

    const e1 = q.begin();
    const m1 = try makeMesh(alloc, "epoch_first");
    q.retireMesh(alloc, m1);
    // begin(e2) закрывает e1 неявно, но complete(e1) здесь зовём явно —
    // как это делает Scene.render в конце кадра.
    const e2 = q.begin();
    const m2 = try makeMesh(alloc, "epoch_second");
    q.retireMesh(alloc, m2);
    try std.testing.expect(e2 == e1 + 1);

    q.complete(e1);
    q.flush(alloc);
    // m1 (epoch e1) уничтожен, m2 (текущий e2) ждёт.
    try std.testing.expectEqual(@as(usize, 1), q.retainedCount());

    q.complete(e2);
    q.flush(alloc);
    try std.testing.expectEqual(@as(usize, 0), q.retainedCount());
}

test "retire из потоков: без гонок и потерь" {
    const alloc = std.testing.allocator;
    var q: GpuRetireQueue = .{};
    defer q.deinit(alloc);
    const e = q.begin();

    const thread_count = 4;
    const per_thread = 16;
    const total = thread_count * per_thread;
    var meshes: [total]*Mesh = undefined;
    for (&meshes) |*slot| {
        slot.* = try makeMesh(alloc, "thread_retire");
    }
    // Append в воркерах не должен аллоцировать: резервируем заранее на
    // основном потоке, чтобы конкурентный путь был чистым lock+store.
    try q.pending.ensureTotalCapacity(alloc, total);

    const Worker = struct {
        fn run(queue: *GpuRetireQueue, allocator: std.mem.Allocator, batch: []*Mesh) void {
            for (batch) |m| queue.retireMesh(allocator, m);
        }
    };
    var threads: [thread_count]std.Thread = undefined;
    var offset: usize = 0;
    for (&threads) |*t| {
        t.* = try std.Thread.spawn(.{}, Worker.run, .{ &q, alloc, meshes[offset .. offset + per_thread] });
        offset += per_thread;
    }
    for (&threads) |*t| t.join();

    // Ни одна запись не потеряна и не задвоена: все total на месте.
    try std.testing.expectEqual(@as(usize, total), q.retainedCount());
    q.complete(e);
    q.flush(alloc);
    try std.testing.expectEqual(@as(usize, 0), q.retainedCount());
    // Отсутствие утечек/двойных free проверяет сам testing.allocator.
}

test "deinit уничтожает хвосты, включая незавершённый epoch" {
    const alloc = std.testing.allocator;
    var q: GpuRetireQueue = .{};

    _ = q.begin();
    const m1 = try makeMesh(alloc, "tail_done");
    q.retireMesh(alloc, m1);
    // Второй epoch намеренно не завершаем: deinit обязан забрать и его.
    _ = q.begin();
    const m2 = try makeMesh(alloc, "tail_open");
    q.retireMesh(alloc, m2);
    // Overflow-хвост тоже: буфер-free меш, ни одного sg.* не будет.
    const m3 = try makeMesh(alloc, "tail_overflow");
    q.overflow[0] = .{ .kind = .mesh, .mesh = m3, .epoch = q.current_epoch };
    q.overflow_len = 1;
    try std.testing.expectEqual(@as(usize, 3), q.retainedCount());

    q.deinit(alloc);
    // После deinit поля очереди читать нельзя (ArrayListUnmanaged.deinit
    // помечает self как undefined); что все три меша уничтожены ровно по
    // разу и очередей не осталось, проверяет сам testing.allocator
    // (утечка/двойной free уронили бы тест).
}

test "retireBuffer штампует текущий epoch; пустой handle — no-op" {
    const alloc = std.testing.allocator;
    gpu_thread.markContextThread();
    var q: GpuRetireQueue = .{};
    // Ручная чистка вместо q.deinit: fake-хендлы буферов (id без живого
    // GPU-контекста) нельзя прогонять через sg.destroyBuffer; за ними нет
    // ни GPU-ресурса, ни CPU-памяти — достаточно освободить список.
    defer q.pending.deinit(alloc);

    const e = q.begin();
    q.retireBuffer(alloc, .{});
    try std.testing.expectEqual(@as(usize, 0), q.retainedCount());

    const m = try makeMesh(alloc, "mixed_probe");
    q.retireMesh(alloc, m);
    q.retireBuffer(alloc, .{ .id = 41 });
    try std.testing.expectEqual(@as(usize, 2), q.retainedCount());
    try std.testing.expectEqual(Kind.mesh, q.pending.items[0].kind);
    try std.testing.expectEqual(Kind.buffer, q.pending.items[1].kind);
    try std.testing.expectEqual(e, q.pending.items[1].epoch);
    try std.testing.expectEqual(@as(u32, 41), q.pending.items[1].buffer.id);

    // Flush до complete(e): записи текущего незавершённого epoch ждут,
    // sg.* не вызывается ни по одной из них.
    q.flush(alloc);
    try std.testing.expectEqual(@as(usize, 2), q.retainedCount());

    // Ручная чистка в том же порядке, что drainLocked: меш — штатно
    // (буферов у testMesh нет, sg.* не вызывается), fake-буфер — дроп
    // записи без destroy (ресурса за id 41 не существует).
    m.deinit(alloc);
    alloc.destroy(m);
    q.pending.clearRetainingCapacity();
    try std.testing.expectEqual(@as(usize, 0), q.retainedCount());
}

test "retireBuffer OOM уходит в overflow[8]" {
    const alloc = std.testing.allocator;
    var q: GpuRetireQueue = .{};
    defer q.pending.deinit(alloc);
    defer {
        @memset(&q.overflow, null);
        q.overflow_len = 0;
    }
    _ = q.begin();

    // Каждый append падает: все записи паркуются в безаллокационный
    // overflow, thread-affinity не нарушается. (Полное переполнение
    // overflow — лог+утечка — тестом не дёргается: кастомный test_runner
    // считает любой std.log.err падением сборки; ветка — трёхстрочное
    // зеркало давно существующего mesh-пути.)
    var failing = std.testing.FailingAllocator.init(alloc, .{ .fail_index = 0 });
    var i: u32 = 0;
    while (i < 8) : (i += 1) {
        q.retireBuffer(failing.allocator(), .{ .id = 100 + i });
    }
    try std.testing.expectEqual(@as(usize, 8), q.retainedCount());
    try std.testing.expectEqual(@as(u32, 100), q.overflow[0].?.buffer.id);
    try std.testing.expectEqual(@as(u32, 107), q.overflow[7].?.buffer.id);
    try std.testing.expectEqual(Kind.buffer, q.overflow[7].?.kind);
}

test "retire dedups the same handle, counting duplicates" {
    const alloc = std.testing.allocator;
    gpu_thread.markContextThread();
    var q: GpuRetireQueue = .{};
    defer q.pending.deinit(alloc);
    _ = q.begin();

    // Same mesh twice: one entry, one counted duplicate — a second destroy
    // of the same object would corrupt, so the queue keeps exactly one.
    const m = try makeMesh(alloc, "dup_mesh");
    q.retireMesh(alloc, m);
    q.retireMesh(alloc, m);
    try std.testing.expectEqual(@as(usize, 1), q.retainedCount());
    try std.testing.expectEqual(@as(u64, 1), q.duplicateDropCount());
    try std.testing.expectEqual(@as(u64, 0), q.cappedDropCount());

    // Same buffer id twice (distinct kinds never collide: mesh pointer vs
    // buffer id are compared only within their kind).
    q.retireBuffer(alloc, .{ .id = 77 });
    q.retireBuffer(alloc, .{ .id = 77 });
    q.retireBuffer(alloc, .{ .id = 78 });
    try std.testing.expectEqual(@as(usize, 3), q.retainedCount());
    try std.testing.expectEqual(@as(u64, 2), q.duplicateDropCount());

    // Manual cleanup (fake buffer ids have no GPU resource behind them):
    // the mesh exactly once — dedup prevented the double destroy.
    m.deinit(alloc);
    alloc.destroy(m);
    q.pending.clearRetainingCapacity();
    try std.testing.expectEqual(@as(usize, 0), q.retainedCount());
}

test "retire dedups across the overflow spillover" {
    const alloc = std.testing.allocator;
    var q: GpuRetireQueue = .{};
    defer q.pending.deinit(alloc);
    defer {
        @memset(&q.overflow, null);
        q.overflow_len = 0;
    }
    _ = q.begin();

    // Every append fails: entries park in overflow; the repeat of id 200
    // must hit the overflow half of the dedup scan, not append twice.
    var failing = std.testing.FailingAllocator.init(alloc, .{ .fail_index = 0 });
    q.retireBuffer(failing.allocator(), .{ .id = 200 });
    q.retireBuffer(failing.allocator(), .{ .id = 200 });
    q.retireBuffer(failing.allocator(), .{ .id = 201 });
    try std.testing.expectEqual(@as(usize, 2), q.retainedCount());
    try std.testing.expectEqual(@as(usize, 2), q.overflow_len);
    try std.testing.expectEqual(@as(u64, 1), q.duplicateDropCount());
    try std.testing.expectEqual(@as(u64, 0), q.cappedDropCount());
}

test "cap admission probe stops at pending_cap without dropping" {
    const alloc = std.testing.allocator;
    gpu_thread.markContextThread();
    var q: GpuRetireQueue = .{};
    defer q.deinit(alloc);
    q.pending_cap = 2;
    const e = q.begin();

    try std.testing.expect(q.admitsOneMore());
    const m1 = try makeMesh(alloc, "cap_first");
    q.retireMesh(alloc, m1);
    try std.testing.expect(q.admitsOneMore());
    const m2 = try makeMesh(alloc, "cap_second");
    q.retireMesh(alloc, m2);
    // At the cap: the probe reports full, nothing was dropped or logged
    // (the drop+log path itself cannot run under the test runner's
    // log.err policy — same precedent as the overflow-exhausted branch —
    // so the test pins the boundary from the admission side).
    try std.testing.expect(!q.admitsOneMore());
    try std.testing.expectEqual(@as(usize, 2), q.retainedCount());
    try std.testing.expectEqual(@as(u64, 0), q.cappedDropCount());

    q.complete(e);
    q.flush(alloc);
    try std.testing.expectEqual(@as(usize, 0), q.retainedCount());
    try std.testing.expect(q.admitsOneMore());
}

test "retireProbeTarget waits for its epoch and dedups by cube image" {
    const alloc = std.testing.allocator;
    gpu_thread.markContextThread();
    var q: GpuRetireQueue = .{};
    defer q.deinit(alloc);
    const e = q.begin();

    // Empty (pre-capture) targets retire as uniform no-op-destroy entries:
    // bookkeeping is identical, sg.destroy* on empty handles is headless-safe.
    q.retireProbeTarget(alloc, .{});
    q.retireProbeTarget(alloc, .{});
    try std.testing.expectEqual(@as(usize, 1), q.retainedCount());
    try std.testing.expectEqual(@as(u64, 1), q.duplicateDropCount());
    try std.testing.expectEqual(Kind.probe, q.pending.items[0].kind);
    try std.testing.expectEqual(e, q.pending.items[0].epoch);

    // A target with a (fake-id) cube image is a distinct entry; a repeat of
    // the same image dedups. Manual id assignment: no sg.* runs in this
    // test until flush, and flush only destroys id-0 handles here... the
    // nonzero entry is dropped from the list by hand instead (fake ids have
    // no GPU resource behind them, same precedent as the buffer test).
    q.retireProbeTarget(alloc, .{ .image = .{ .id = 501 } });
    q.retireProbeTarget(alloc, .{ .image = .{ .id = 501 } });
    try std.testing.expectEqual(@as(usize, 2), q.retainedCount());
    try std.testing.expectEqual(@as(u64, 2), q.duplicateDropCount());

    // Flush before complete: both entries wait (current epoch open).
    q.flush(alloc);
    try std.testing.expectEqual(@as(usize, 2), q.retainedCount());

    // Drop the fake-id entry by hand, then complete + flush drains the
    // empty one through ProbeGpu.deinit headlessly.
    _ = q.pending.orderedRemove(1);
    q.complete(e);
    q.flush(alloc);
    try std.testing.expectEqual(@as(usize, 0), q.retainedCount());
}

test "retireUi3dTarget waits for its epoch and dedups by rt image" {
    const alloc = std.testing.allocator;
    gpu_thread.markContextThread();
    var q: GpuRetireQueue = .{};
    defer q.deinit(alloc);
    const e = q.begin();

    // Empty (pre-capture) targets retire as uniform no-op-destroy entries:
    // bookkeeping is identical, sg.destroy* on empty handles is headless-safe.
    q.retireUi3dTarget(alloc, .{});
    q.retireUi3dTarget(alloc, .{});
    try std.testing.expectEqual(@as(usize, 1), q.retainedCount());
    try std.testing.expectEqual(@as(u64, 1), q.duplicateDropCount());
    try std.testing.expectEqual(Kind.ui3d, q.pending.items[0].kind);
    try std.testing.expectEqual(e, q.pending.items[0].epoch);

    // A target with a (fake-id) RT image is a distinct entry; a repeat of
    // the same image dedups. Manual id assignment: no sg.* runs in this
    // test until flush, and flush only destroys id-0 handles here... the
    // nonzero entry is dropped from the list by hand instead (fake ids have
    // no GPU resource behind them, same precedent as the buffer test).
    q.retireUi3dTarget(alloc, .{ .image = .{ .id = 601 } });
    q.retireUi3dTarget(alloc, .{ .image = .{ .id = 601 } });
    try std.testing.expectEqual(@as(usize, 2), q.retainedCount());
    try std.testing.expectEqual(@as(u64, 2), q.duplicateDropCount());

    // Flush before complete: both entries wait (current epoch open).
    q.flush(alloc);
    try std.testing.expectEqual(@as(usize, 2), q.retainedCount());

    // Drop the fake-id entry by hand, then complete + flush drains the
    // empty one through Ui3dTarget.deinit headlessly.
    _ = q.pending.orderedRemove(1);
    q.complete(e);
    q.flush(alloc);
    try std.testing.expectEqual(@as(usize, 0), q.retainedCount());
}

test "retireComputeBundle zero is a no-op; full bundle waits for its epoch" {
    const alloc = std.testing.allocator;
    gpu_thread.markContextThread();
    var q: GpuRetireQueue = .{};
    // Manual teardown throughout (fake ids have no GPU resource behind
    // them): entries are dropped by hand, never flushed live — the live
    // destroy proof is the P6 gate (runComputeBundleRetire), which drives
    // a real bundle through complete + flush to INVALID on-context.
    defer {
        q.pending.clearRetainingCapacity();
        @memset(&q.overflow, null);
        q.overflow_len = 0;
        q.pending.deinit(alloc);
    }
    const e = q.begin();

    // Empty bundle: nothing to retire, no entry, no counts.
    q.retireComputeBundle(alloc, .{});
    try std.testing.expectEqual(@as(usize, 0), q.retainedCount());
    try std.testing.expectEqual(@as(u64, 0), q.duplicateDropCount());
    try std.testing.expectEqual(@as(u64, 0), q.cappedDropCount());

    // Full losing outcome (fake ids): exactly one stamped entry.
    q.retireComputeBundle(alloc, .{
        .state_buffer = .{ .id = 101 },
        .spawn_buffer = .{ .id = 102 },
        .draw_buffer = .{ .id = 103 },
        .state_view = .{ .id = 104 },
        .spawn_view = .{ .id = 105 },
        .draw_view = .{ .id = 106 },
        .shader = .{ .id = 107 },
        .pipeline = .{ .id = 108 },
    });
    try std.testing.expectEqual(@as(usize, 1), q.retainedCount());
    try std.testing.expectEqual(Kind.compute, q.pending.items[0].kind);
    try std.testing.expectEqual(e, q.pending.items[0].epoch);
    try std.testing.expectEqual(@as(u32, 104), q.pending.items[0].compute.state_view.id);
    try std.testing.expectEqual(@as(u32, 108), q.pending.items[0].compute.pipeline.id);

    // Flush before complete: the open-epoch entry waits, sg.* untouched.
    q.flush(alloc);
    try std.testing.expectEqual(@as(usize, 1), q.retainedCount());

    // Cap probe still admits around it (no drop+log path is exercised:
    // the test runner fails on any std.log.err, same precedent as the
    // mesh/buffer cap tests — the boundary is pinned from the admission
    // side).
    q.pending_cap = 1;
    try std.testing.expect(!q.admitsOneMore());
    try std.testing.expectEqual(@as(u64, 0), q.cappedDropCount());
    q.pending_cap = 8192;

    // Hand-drop the fake-id entry (no GPU resource behind it).
    q.pending.clearRetainingCapacity();
    try std.testing.expectEqual(@as(usize, 0), q.retainedCount());
}

test "retireComputeBundle dedups the exact tuple, isolates handle types" {
    const alloc = std.testing.allocator;
    gpu_thread.markContextThread();
    var q: GpuRetireQueue = .{};
    defer {
        q.pending.clearRetainingCapacity();
        @memset(&q.overflow, null);
        q.overflow_len = 0;
        q.pending.deinit(alloc);
    }
    _ = q.begin();

    // Same losing outcome retired twice: one entry, one counted duplicate
    // (a second destroy of the same eight handles would corrupt).
    const loser = ComputeBundle{
        .state_buffer = .{ .id = 201 },
        .spawn_buffer = .{ .id = 202 },
        .draw_buffer = .{ .id = 203 },
        .state_view = .{ .id = 204 },
        .spawn_view = .{ .id = 205 },
        .draw_view = .{ .id = 206 },
        .shader = .{ .id = 207 },
        .pipeline = .{ .id = 208 },
    };
    q.retireComputeBundle(alloc, loser);
    q.retireComputeBundle(alloc, loser);
    try std.testing.expectEqual(@as(usize, 1), q.retainedCount());
    try std.testing.expectEqual(@as(u64, 1), q.duplicateDropCount());

    // Same NUMERIC id in different handle fields is NOT a duplicate: a
    // buffer id and a view id live in different sokol pools.
    q.retireComputeBundle(alloc, .{ .state_buffer = .{ .id = 300 } });
    q.retireComputeBundle(alloc, .{ .state_view = .{ .id = 300 } });
    try std.testing.expectEqual(@as(usize, 3), q.retainedCount());
    try std.testing.expectEqual(@as(u64, 1), q.duplicateDropCount());

    // Different losing outcomes sharing one handle id are different
    // outcomes (each created handle belongs to exactly one outcome, so a
    // shared id here means distinct packets, never a double-retire).
    q.retireComputeBundle(alloc, .{
        .state_buffer = .{ .id = 300 },
        .state_view = .{ .id = 301 },
    });
    try std.testing.expectEqual(@as(usize, 4), q.retainedCount());
    try std.testing.expectEqual(@as(u64, 1), q.duplicateDropCount());
    try std.testing.expectEqual(@as(u64, 0), q.cappedDropCount());
}

test "retireComputeBundle OOM parks in overflow[8], dedups across it" {
    const alloc = std.testing.allocator;
    var q: GpuRetireQueue = .{};
    defer {
        @memset(&q.overflow, null);
        q.overflow_len = 0;
        q.pending.deinit(alloc);
    }
    _ = q.begin();

    // Every append fails: entries park in the allocation-free overflow
    // without touching thread-affinity (the overflow-exhausted log+leak
    // branch is not exercised — same precedent as the buffer OOM test).
    var failing = std.testing.FailingAllocator.init(alloc, .{ .fail_index = 0 });
    var i: u32 = 0;
    while (i < 8) : (i += 1) {
        q.retireComputeBundle(failing.allocator(), .{ .state_buffer = .{ .id = 400 + i } });
    }
    try std.testing.expectEqual(@as(usize, 8), q.retainedCount());
    try std.testing.expectEqual(@as(usize, 8), q.overflow_len);
    try std.testing.expectEqual(@as(u32, 400), q.overflow[0].?.compute.state_buffer.id);
    try std.testing.expectEqual(Kind.compute, q.overflow[7].?.kind);

    // The repeat of an overflow-parked bundle hits the overflow half of
    // the dedup scan instead of appending twice.
    q.retireComputeBundle(failing.allocator(), .{ .state_buffer = .{ .id = 400 } });
    try std.testing.expectEqual(@as(usize, 8), q.retainedCount());
    try std.testing.expectEqual(@as(u64, 1), q.duplicateDropCount());
    try std.testing.expectEqual(@as(u64, 0), q.cappedDropCount());
}
