//! GPU-handle lifetime epochs (P3 «parallel update/render»): единый проверяемый
//! механизм отложенного уничтожения GPU-ресурсов.
//!
//! Ownership-блок:
//! - ПИШЕТ любой поток через `retireMesh` (сегодня — только Scene.destroyMesh
//!   вне context-потока; в будущем — update-поток, выводящий объекты из
//!   эксплуатации). Под мьютексом только штамп epoch + append указателя,
//!   никаких sg.* и никакого освобождения памяти.
//! - ЧИТАЕТ/УНИЧТОЖАЕТ только context-поток: `flush` (начало render-кадра) и
//!   `deinit` (конец жизни сцены). Только здесь вызываются `Mesh.deinit`
//!   (sg.destroy*) и `allocator.destroy`.
//! Правила epoch:
//! - `begin` открывает новый epoch кадра; `complete(e)` закрывает epoch e;
//!   кадры строго последовательны, поэтому `begin` заодно закрывает
//!   предыдущий незакрытый epoch (защита от prepareFrame без парного render).
//! - Запись, ушедшая в ретенцию в epoch E, уничтожается первым `flush` после
//!   `complete(E)` (условие `entry.epoch <= lastCompleted()`). Запись текущего
//!   (ещё не завершённого) epoch ждёт следующего завершения — render, который
//!   мог её использовать, гарантированно закончился.
//! - Однопоточное поведение не меняется: отсрочка максимум на кадр, как раньше
//!   с `pending_gpu_destroys`.
//! Где flush: `Scene.flushPendingGpuUploads` (начало кадра, внутри prepareFrame)
//! и `Scene.deinit` (через `deinit`, уничтожающий и незавершённые эпохи).
//! OOM-контракт как раньше: очередь не растёт бесконечно — при OOM append
//! запись паркуется в безаллокационный overflow[8], при переполнении и его —
//! лог + утечка меша, но никогда sg.* вне context-потока.
//! Tripwire P5/P6: новые kind'ы записей (не меши) добавлять сюда же — расширять
//! запись/очередь, а не заводить новые очереди в Scene.

const std = @import("std");
const Mesh = @import("../mesh.zig").Mesh;
const gpu_thread = @import("../gpu_thread.zig");

/// Номер render-кадра. 0 — «кадра ещё не было»; счётчик стартует с 1.
pub const Epoch = u64;

/// Ёмкость безаллокационного spillover на случай OOM в retireMesh — как раньше
/// в Scene (`pending_gpu_destroys_overflow`).
const overflow_cap: usize = 8;

/// Одна отложенная запись: уже отвязанный от сцены меш + кадр ухода в ретенцию.
const Entry = struct {
    mesh: *Mesh,
    epoch: Epoch,
};

/// Спин по образцу assets.UploadQueue: критические секции — bump счётчика или
/// append одного указателя, вызовы retire редкие.
fn lockSpin(m: *std.atomic.Mutex) void {
    while (!m.tryLock()) std.atomic.spinLoopHint();
}

pub const GpuRetireQueue = struct {
    const Self = @This();

    mutex: std.atomic.Mutex = .unlocked,
    current_epoch: Epoch = 0,
    completed_epoch: Epoch = 0,
    pending: std.ArrayListUnmanaged(Entry) = .empty,
    overflow: [overflow_cap]?Entry = [_]?Entry{null} ** overflow_cap,
    overflow_len: usize = 0,

    /// Начало render-кадра: открывает новый epoch и возвращает его. Заодно
    /// закрывает предыдущий незакрытый epoch — кадры строго последовательны
    /// (владение фазой P1), поэтому старт кадра N+1 означает конец кадра N.
    /// Только context-поток (зовёт Scene.prepareFrame).
    pub fn begin(self: *Self) Epoch {
        lockSpin(&self.mutex);
        defer self.mutex.unlock();
        if (self.current_epoch > self.completed_epoch) {
            self.completed_epoch = self.current_epoch;
        }
        self.current_epoch +%= 1;
        return self.current_epoch;
    }

    /// Конец render-кадра: закрывает epoch e (и все более ранние).
    /// Идемпотентен (повторный вызов с тем же e — no-op). Только context-поток
    /// (зовёт Scene.render на всех выходах, включая возврат без камеры).
    pub fn complete(self: *Self, e: Epoch) void {
        lockSpin(&self.mutex);
        defer self.mutex.unlock();
        std.debug.assert(e <= self.current_epoch);
        if (e > self.completed_epoch) {
            self.completed_epoch = e;
        }
    }

    /// Текущий (открытый) epoch. Блокирующий: читает под мьютексом.
    pub fn current(self: *Self) Epoch {
        lockSpin(&self.mutex);
        defer self.mutex.unlock();
        return self.current_epoch;
    }

    /// Последний завершённый epoch. Блокирующий: читает под мьютексом.
    pub fn lastCompleted(self: *Self) Epoch {
        lockSpin(&self.mutex);
        defer self.mutex.unlock();
        return self.completed_epoch;
    }

    /// Уход меша в ретенцию: штамп текущего epoch + постановка в очередь.
    /// Можно звать с любого потока; вызовы редкие (только destroy вне
    /// контекста). Аллокатор — параметром: очередь принципиально не хранит
    /// аллокатор (им владеет Scene), append под тем же спинлоком.
    /// Никогда не вызывает sg.* и не освобождает память: при OOM запись
    /// паркуется в безаллокационный overflow, при переполнении и его — лог +
    /// утечка меша.
    pub fn retireMesh(self: *Self, allocator: std.mem.Allocator, mesh: *Mesh) void {
        lockSpin(&self.mutex);
        defer self.mutex.unlock();
        const entry = Entry{ .mesh = mesh, .epoch = self.current_epoch };
        self.pending.append(allocator, entry) catch {
            // OOM в append: слот overflow не требует аллокации, поэтому
            // thread-affinity не нарушается и здесь.
            if (self.overflow_len < self.overflow.len) {
                self.overflow[self.overflow_len] = entry;
                self.overflow_len += 1;
            } else {
                // Обе очереди исчерпаны при патологическом OOM: утечка меша
                // (с логом), но не sg.* вне context-потока.
                std.log.err("scene: destroy queues exhausted, leaking mesh '{s}'", .{mesh.name});
            }
        };
    }

    /// Уничтожение due-записей (`epoch <= lastCompleted()`): Mesh.deinit
    /// (sg.*) + free. Только context-поток. Идемпотентен на пустой очереди.
    /// Держит спинлок и во время sg-teardown: retire редкий и короткий,
    /// вложенных захватов нет — дедлока быть не может.
    pub fn flush(self: *Self, allocator: std.mem.Allocator) void {
        gpu_thread.assertOnContextThread();
        lockSpin(&self.mutex);
        defer self.mutex.unlock();
        self.drainLocked(allocator, self.completed_epoch);
    }

    /// Финал: уничтожает ВСЕ оставшиеся записи, включая незавершённые эпохи,
    /// и освобождает очереди. Только context-поток (Scene.deinit, sg ещё жив).
    /// После вызова очередь пуста; повторное использование — только через
    /// новые retire (ёмкость списков сброшена).
    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        gpu_thread.assertOnContextThread();
        lockSpin(&self.mutex);
        defer self.mutex.unlock();
        self.drainLocked(allocator, std.math.maxInt(Epoch));
        self.pending.deinit(allocator);
        @memset(&self.overflow, null);
        self.overflow_len = 0;
    }

    /// Сколько записей ждёт уничтожения (основная очередь + overflow).
    /// Для тестов и инвариантов; читает под мьютексом.
    pub fn retainedCount(self: *Self) usize {
        lockSpin(&self.mutex);
        defer self.mutex.unlock();
        return self.pending.items.len + self.overflow_len;
    }

    /// Вызывается только под захваченным спинлоком: уничтожает записи с
    /// `epoch <= done` (основная очередь — компактификацией на месте с
    /// сохранением ёмкости, overflow — со сдвигом хвоста).
    fn drainLocked(self: *Self, allocator: std.mem.Allocator, done: Epoch) void {
        var kept: usize = 0;
        for (self.pending.items) |entry| {
            if (entry.epoch <= done) {
                entry.mesh.deinit(allocator);
                allocator.destroy(entry.mesh);
            } else {
                self.pending.items[kept] = entry;
                kept += 1;
            }
        }
        self.pending.items.len = kept;
        var okept: usize = 0;
        for (self.overflow[0..self.overflow_len]) |slot| {
            if (slot) |entry| {
                if (entry.epoch <= done) {
                    entry.mesh.deinit(allocator);
                    allocator.destroy(entry.mesh);
                    continue;
                }
                self.overflow[okept] = entry;
                okept += 1;
            }
        }
        @memset(self.overflow[okept..], null);
        self.overflow_len = okept;
    }
};

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
    q.overflow[0] = .{ .mesh = m3, .epoch = q.current_epoch };
    q.overflow_len = 1;
    try std.testing.expectEqual(@as(usize, 3), q.retainedCount());

    q.deinit(alloc);
    // После deinit поля очереди читать нельзя (ArrayListUnmanaged.deinit
    // помечает self как undefined); что все три меша уничтожены ровно по
    // разу и очередей не осталось, проверяет сам testing.allocator
    // (утечка/двойной free уронили бы тест).
}
