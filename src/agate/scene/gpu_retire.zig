//! GPU-handle lifetime epochs (P3 «parallel update/render»): единый проверяемый
//! механизм отложенного уничтожения GPU-ресурсов.
//!
//! Ownership-блок:
//! - ПИШЕТ любой поток через `retireMesh`/`retireBuffer` (сегодня — только
//!   Scene.destroyMesh вне context-потока и рост instance-буферов в стейджинге;
//!   в будущем — update-поток, выводящий объекты из эксплуатации). Под
//!   мьютексом только штамп epoch + append записи, никаких sg.* и никакого
//!   освобождения памяти.
//! - ЧИТАЕТ/УНИЧТОЖАЕТ только context-поток: `flush` (начало render-кадра) и
//!   `deinit` (конец жизни сцены). Только здесь вызываются `Mesh.deinit`
//!   (sg.destroy*) / `sg.destroyBuffer` и `allocator.destroy`.
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
//! Каденция flush — context-поток при УСПЕШНОМ prepare: пока prepare не
//! проходит (длинная серия `Scene.renderReuse` без владения фазой), flush не
//! наступает и ретенция растёт как skip-streak × destroy-rate. Рост НЕ
//! безграничен: `pending_cap` ограничивает суммарно удерживаемые записи
//! (основная очередь + overflow); сверх лимита запись считается капнутой —
//! счётчик `capped_drops` + лог, без молчаливых потерь (та же видимость, что
//! у исчерпания overflow[8]; капнутая запись утекает, как и там —
//! разрушать её негде: due-записей после ведущего flush не осталось, а
//! недозревшие может ещё читать отрисованный фронт). Дефолт 8192: устойчивое
//! состояние держит единицы записей (ретайр живёт ≤ 1 кадр), так что лимит
//! срабатывает только при патологическом забросе destroy при длинном reuse-
//! стрике — т.е. при нарушении контракта вызывающей стороны (не стрикать
//! reuse бесконечно). Текущая глубина видна через `retainedCount`, капли —
//! через `cappedDropCount`, длина стрика — через `Scene.reuseStreak`:
//! вместе они делают staleness наблюдаемой вместо молчаливой.
//! Дубликаты (тот же меш-указатель / тот же buffer id уже в очереди)
//! отбрасываются счётно (`duplicateDropCount`): повторный ретайр одного
//! хендла был бы двойным destroy, так что dedup — fail-safe, а не
//! оптимизация; у корректных вызывающих дубликатов нет и поведение не
//! меняется.
//! OOM-контракт как раньше: очередь не растёт бесконечно — при OOM append
//! запись паркуется в безаллокационный overflow[8], при переполнении и его —
//! лог + утечка записи, но никогда sg.* вне context-потока.
//! P5 (instance staging ownership): выросший instance-буфер уходит сюда же
//! записью kind=.buffer — СНАЧАЛА новый буфер создан+залит успешно, затем
//! старый ретайрится (никакого немедленного destroy старого). Mesh.deinit
//! уничтожает только текущий instance_render.buffer — ретайренные старые
//! буферы принадлежат очереди, двойного free нет.
//! Tripwire P6: новые kind'ы записей (не меши/буферы) добавлять сюда же —
//! расширять запись/очередь, а не заводить новые очереди в Scene.

const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const Mesh = @import("../mesh.zig").Mesh;
const gpu_thread = @import("../gpu_thread.zig");

/// Номер render-кадра. 0 — «кадра ещё не было»; счётчик стартует с 1.
pub const Epoch = u64;

/// Ёмкость безаллокационного spillover на случай OOM в retireMesh — как раньше
/// в Scene (`pending_gpu_destroys_overflow`).
const overflow_cap: usize = 8;

/// Одна отложенная запись: уже отвязанный от сцены меш либо вытесненный
/// старый instance-буфер (P5) + кадр ухода в ретенцию.
const Entry = struct {
    kind: Kind,
    mesh: ?*Mesh = null,
    buffer: sg.Buffer = .{},
    epoch: Epoch,
};

pub const Kind = enum { mesh, buffer };

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
    /// Bound on jointly retained entries (`pending.items.len + overflow_len`):
    /// a skip-streak × destroy-rate burst past this is dropped counted, not
    /// grown silently (see header for why a drop leaks by necessity). Steady
    /// state holds a handful of entries (a retire lives <= 1 frame), so the
    /// default only trips on a pathological streak. Field, not const, so apps
    /// and tests can tighten it; read under the spinlock, written rarely.
    pending_cap: usize = 8192,
    /// Entries dropped by the cap (observable; each also logs — never silent).
    capped_drops: u64 = 0,
    /// Duplicate retires skipped (same mesh pointer / buffer id already
    /// queued): a second destroy of one handle would corrupt, so dedup is a
    /// fail-safe. Correct callers never produce duplicates.
    duplicate_drops: u64 = 0,

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
    /// Никогда не вызывает sg.* и не освобождает память: дубликат уже
    /// стоящего в очереди меша отбрасывается счётно (fail-safe против
    /// двойного destroy), сверх `pending_cap` запись отбрасывается счётно
    /// (счётчик + лог, без молчаливого роста); при OOM запись паркуется в
    /// безаллокационный overflow, при переполнении и его — лог + утечка меша.
    pub fn retireMesh(self: *Self, allocator: std.mem.Allocator, mesh: *Mesh) void {
        lockSpin(&self.mutex);
        defer self.mutex.unlock();
        const entry = Entry{ .kind = .mesh, .mesh = mesh, .epoch = self.current_epoch };
        if (self.containsLocked(entry)) {
            self.duplicate_drops += 1;
            return;
        }
        if (!self.admitsLocked()) {
            self.capped_drops += 1;
            std.log.err("scene: retire queue cap {d} reached, leaking mesh '{s}' (bound skip-streak destroys; see pending_cap)", .{ self.pending_cap, mesh.name });
            return;
        }
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

    /// Уход старого instance-буфера в ретенцию (P5: рост буфера в стейджинге —
    /// новый создан+залит, затем старый сюда). Можно звать с любого потока;
    /// тот же epoch/overflow[8]/log+leak контракт, что у retireMesh, плюс
    /// dedup/cap выше: под мьютексом только штамп epoch + проверки + append,
    /// никаких sg.*. Пустой handle — no-op (нечего ретайрить). Уничтожение
    /// (`sg.destroyBuffer`) — только context-поток во flush/deinit.
    pub fn retireBuffer(self: *Self, allocator: std.mem.Allocator, buf: sg.Buffer) void {
        if (buf.id == 0) return;
        lockSpin(&self.mutex);
        defer self.mutex.unlock();
        const entry = Entry{ .kind = .buffer, .buffer = buf, .epoch = self.current_epoch };
        if (self.containsLocked(entry)) {
            self.duplicate_drops += 1;
            return;
        }
        if (!self.admitsLocked()) {
            self.capped_drops += 1;
            std.log.err("scene: retire queue cap {d} reached, leaking instance buffer (id {})", .{ self.pending_cap, buf.id });
            return;
        }
        self.pending.append(allocator, entry) catch {
            if (self.overflow_len < self.overflow.len) {
                self.overflow[self.overflow_len] = entry;
                self.overflow_len += 1;
            } else {
                std.log.err("scene: destroy queues exhausted, leaking instance buffer (id {})", .{buf.id});
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

    /// Сколько записей отброшено капом `pending_cap` (каждая с логом).
    /// Ненулевое значение = нарушен streak-контракт вызывающей стороны.
    pub fn cappedDropCount(self: *Self) u64 {
        lockSpin(&self.mutex);
        defer self.mutex.unlock();
        return self.capped_drops;
    }

    /// Сколько повторных ретайров одного хендла отброшено dedup'ом.
    /// Ненулевое значение = вызывающий ретайрит дважды (было бы двойным
    /// destroy без dedup).
    pub fn duplicateDropCount(self: *Self) u64 {
        lockSpin(&self.mutex);
        defer self.mutex.unlock();
        return self.duplicate_drops;
    }

    /// Проба приёма для тестов/конфига: влезет ли ещё одна запись под
    /// `pending_cap`. Читает под мьютексом; сам приём — через retire*.
    pub fn admitsOneMore(self: *Self) bool {
        lockSpin(&self.mutex);
        defer self.mutex.unlock();
        return self.admitsLocked();
    }

    /// Только под захваченным спинлоком: влезет ли ещё одна запись
    /// (основная очередь + overflow против `pending_cap`).
    fn admitsLocked(self: *const Self) bool {
        return self.pending.items.len + self.overflow_len < self.pending_cap;
    }

    /// Только под захваченным спинлоком: стоит ли идентичная запись уже в
    /// очереди (тот же меш-указатель / тот же buffer id того же kind'а).
    fn containsLocked(self: *const Self, entry: Entry) bool {
        for (self.pending.items) |e| {
            if (sameHandle(e, entry)) return true;
        }
        for (self.overflow[0..self.overflow_len]) |slot| {
            if (slot) |e| {
                if (sameHandle(e, entry)) return true;
            }
        }
        return false;
    }

    /// Идентичность ретайр-записей: kind совпадает и хендл тот же. Меши —
    /// по указателю (повторный ретайр одного объекта), буферы — по id
    /// (повторный ретайр одного GPU-хендла).
    fn sameHandle(a: Entry, b: Entry) bool {
        if (a.kind != b.kind) return false;
        return switch (a.kind) {
            .mesh => a.mesh == b.mesh,
            .buffer => a.buffer.id == b.buffer.id,
        };
    }

    /// Вызывается только под захваченным спинлоком: уничтожает записи с
    /// `epoch <= done` (основная очередь — компактификацией на месте с
    /// сохранением ёмкости, overflow — со сдвигом хвоста).
    fn drainLocked(self: *Self, allocator: std.mem.Allocator, done: Epoch) void {
        var kept: usize = 0;
        for (self.pending.items) |entry| {
            if (entry.epoch <= done) {
                switch (entry.kind) {
                    .mesh => {
                        entry.mesh.?.deinit(allocator);
                        allocator.destroy(entry.mesh.?);
                    },
                    // P5: вытесненный старый instance-буфер — только GPU-хендл,
                    // CPU-памяти за ним нет. Контекстный поток гарантирован
                    // вызывающими flush/deinit.
                    .buffer => sg.destroyBuffer(entry.buffer),
                }
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
                    switch (entry.kind) {
                        .mesh => {
                            entry.mesh.?.deinit(allocator);
                            allocator.destroy(entry.mesh.?);
                        },
                        .buffer => sg.destroyBuffer(entry.buffer),
                    }
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
