//! GPU-handle lifetime epochs (P3 «parallel update/render»): единый проверяемый
//! механизм отложенного уничтожения GPU-ресурсов.
//!
//! Ownership-блок:
//! - ПИШЕТ любой поток через `retireMesh`/`retireBuffer`/`retireProbeTarget`/
//!   `retireUi3dTarget`/`retireComputeBundle` (сегодня — destroy вне
//!   context-потока, рост instance-буферов в стейджинге, снятие проб/панелей
//!   и проигравшие compute-наборы game-side коммита; в будущем —
//!   update-поток, выводящий объекты из эксплуатации). Под
//!   мьютексом только штамп epoch + append записи, никаких sg.* и никакого
//!   освобождения памяти.
//! - ЧИТАЕТ/УНИЧТОЖАЕТ только context-поток: `flush` (начало render-кадра) и
//!   `deinit` (конец жизни сцены). Только здесь вызываются `Mesh.deinit`
//!   (sg.destroy*) / `sg.destroyBuffer` и `allocator.destroy`.
//! Правила epoch:
//! - `begin` открывает новый epoch кадра; `complete(e)` закрывает epoch e;
//!   кадры строго последовательны, поэтому `begin` заодно закрывает
//!   предыдущий незакрытый epoch (защита от staged begin без парного render/finish).
//! - Запись, ушедшая в ретенцию в epoch E, уничтожается первым `flush` после
//!   `complete(E)` (условие `entry.epoch <= lastCompleted()`). Запись текущего
//!   (ещё не завершённого) epoch ждёт следующего завершения — render, который
//!   мог её использовать, гарантированно закончился.
//! - Однопоточное поведение не меняется: отсрочка максимум на кадр, как раньше
//!   с `pending_gpu_destroys`.
//! Где flush: staged `flushSlotUploads` (retire.flush + создание из slot
//! byte-пакетов; продьюсер коммитит созданные хендлы game-side следующим
//! билдом) и quiesced drain через `Scene.flushPendingGpuUploads`
//! (тесты/teardown/creation drain + P5 seam — НЕ нормальный кадровый путь),
//! плюс `Scene.deinit` (через `deinit`, уничтожающий и незавершённые эпохи).
//! Каденция flush — context-поток при УСПЕШНОМ staged begin: пока begin не
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
const probe_layer = @import("probe_layer.zig");
const gui3d_layer = @import("gui3d_layer.zig");

/// Номер render-кадра. 0 — «кадра ещё не было»; счётчик стартует с 1.
pub const Epoch = u64;

/// Ёмкость безаллокационного spillover на случай OOM в retireMesh — как раньше
/// в Scene (`pending_gpu_destroys_overflow`).
const overflow_cap: usize = 8;

/// Одна отложенная запись: уже отвязанный от сцены меш, вытесненный
/// старый instance-буфер (P5), либо снятый с учёта reflection-проб таргет
/// (wave 25: куб + его вьюхи/сэмплер + глубина одним значением), либо снятый
/// 3D-GUI-панель таргет (wave 28: RT + его вьюхи/сэмплер + панельные
/// UI-буферы одним значением), либо проигравший compute-набор (staged
/// compute-создание: созданные флешем буферы + вьюхи + шейдер + пайплайн,
/// которые game-side коммит не смог установить — дубликат после гонки
/// созданий или владелец исчез; уничтожаются одним значением в
/// dependency-порядке) + кадр ухода в ретенцию.
const Entry = struct {
    kind: Kind,
    mesh: ?*Mesh = null,
    buffer: sg.Buffer = .{},
    probe: probe_layer.ProbeGpu = .{},
    ui3d: gui3d_layer.Ui3dTarget = .{},
    compute: ComputeBundle = .{},
    epoch: Epoch,
};

pub const Kind = enum { mesh, buffer, probe, ui3d, compute };

/// Проигравший исход staged compute-создания (см. installComputeCreated /
/// retireComputeCreated в upload_packets.zig): все созданные флешем хендлы,
/// которые коммит не установил поверх живых id. Едут одной записью, чтобы
/// порядок уничтожения всегда был dependency-безопасным (вьюхи ссылаются
/// на буферы, пайплайн — на шейдер), независимо от порядка записей в
/// очереди. Частичные исходы — норма (упавший make* останавливает флеш
/// раньше): нулевые поля при уничтожении пропускаются.
pub const ComputeBundle = struct {
    state_buffer: sg.Buffer = .{},
    spawn_buffer: sg.Buffer = .{},
    draw_buffer: sg.Buffer = .{},
    state_view: sg.View = .{},
    spawn_view: sg.View = .{},
    draw_view: sg.View = .{},
    shader: sg.Shader = .{},
    pipeline: sg.Pipeline = .{},

    /// Пустой набор — нечего ретайрить (победа по всем фронтам или
    /// outcome без созданий): retireComputeBundle — no-op.
    pub fn isEmpty(self: ComputeBundle) bool {
        return self.state_buffer.id == 0 and self.spawn_buffer.id == 0 and
            self.draw_buffer.id == 0 and self.state_view.id == 0 and
            self.spawn_view.id == 0 and self.draw_view.id == 0 and
            self.shader.id == 0 and self.pipeline.id == 0;
    }

    /// Уничтожение одного набора в dependency-порядке: сначала вьюхи (они
    /// ссылаются на буферы), затем пайплайн (ссылается на шейдер), затем
    /// шейдер и только потом буферы. Нулевые поля (частичный исход) —
    /// пропуск. Только context-поток, под локом очереди из drainLocked.
    pub fn deinit(self: *ComputeBundle) void {
        if (self.state_view.id != 0) sg.destroyView(self.state_view);
        if (self.spawn_view.id != 0) sg.destroyView(self.spawn_view);
        if (self.draw_view.id != 0) sg.destroyView(self.draw_view);
        if (self.pipeline.id != 0) sg.destroyPipeline(self.pipeline);
        if (self.shader.id != 0) sg.destroyShader(self.shader);
        if (self.state_buffer.id != 0) sg.destroyBuffer(self.state_buffer);
        if (self.spawn_buffer.id != 0) sg.destroyBuffer(self.spawn_buffer);
        if (self.draw_buffer.id != 0) sg.destroyBuffer(self.draw_buffer);
    }
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
    /// Только context-поток (зовёт staged begin).
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

    /// Уход снятого reflection-проб таргета в ретенцию (wave 25: удаление
    /// или пересоздание пробы со стороны игры). Можно звать с любого потока;
    /// тот же epoch/overflow[8]/log+leak контракт, что у retireMesh, плюс
    /// dedup/cap выше: под мьютексом только штамп epoch + проверки + append,
    /// никаких sg.*. Пустой (несозданный) таргет — тоже запись: уничтожать
    /// нечего, но дисциплина остаётся uniform (все sg.destroy* — только
    /// context-поток во flush/deinit). Уничтожение (`ProbeGpu.deinit`) —
    /// только context-поток во flush/deinit.
    pub fn retireProbeTarget(self: *Self, allocator: std.mem.Allocator, target: probe_layer.ProbeGpu) void {
        lockSpin(&self.mutex);
        defer self.mutex.unlock();
        const entry = Entry{ .kind = .probe, .probe = target, .epoch = self.current_epoch };
        if (self.containsLocked(entry)) {
            self.duplicate_drops += 1;
            return;
        }
        if (!self.admitsLocked()) {
            self.capped_drops += 1;
            std.log.err("scene: retire queue cap {d} reached, leaking probe target (image id {})", .{ self.pending_cap, target.image.id });
            return;
        }
        self.pending.append(allocator, entry) catch {
            if (self.overflow_len < self.overflow.len) {
                self.overflow[self.overflow_len] = entry;
                self.overflow_len += 1;
            } else {
                std.log.err("scene: destroy queues exhausted, leaking probe target (image id {})", .{target.image.id});
            }
        };
    }

    /// Уход снятого 3D-GUI-панель таргета в ретенцию (wave 28: удаление
    /// панели со стороны игры). Можно звать с любого потока; тот же
    /// epoch/overflow[8]/log+leak контракт, что у retireMesh, плюс dedup/cap
    /// выше: под мьютексом только штамп epoch + проверки + append, никаких
    /// sg.*. Пустой (несозданный) таргет — тоже запись: уничтожать нечего,
    /// но дисциплина остаётся uniform. Уничтожение (`Ui3dTarget.deinit`) —
    /// только context-поток во flush/deinit.
    pub fn retireUi3dTarget(self: *Self, allocator: std.mem.Allocator, target: gui3d_layer.Ui3dTarget) void {
        lockSpin(&self.mutex);
        defer self.mutex.unlock();
        const entry = Entry{ .kind = .ui3d, .ui3d = target, .epoch = self.current_epoch };
        if (self.containsLocked(entry)) {
            self.duplicate_drops += 1;
            return;
        }
        if (!self.admitsLocked()) {
            self.capped_drops += 1;
            std.log.err("scene: retire queue cap {d} reached, leaking ui3d target (image id {})", .{ self.pending_cap, target.image.id });
            return;
        }
        self.pending.append(allocator, entry) catch {
            if (self.overflow_len < self.overflow.len) {
                self.overflow[self.overflow_len] = entry;
                self.overflow_len += 1;
            } else {
                std.log.err("scene: destroy queues exhausted, leaking ui3d target (image id {})", .{target.image.id});
            }
        };
    }

    /// Уход проигравшего compute-набора в ретенцию (staged compute-путь:
    /// дубликат после гонки созданий или владелец исчез — коммит в
    /// upload_packets.zig сам решает, что победило, и сдаёт сюда
    /// остальное одним значением). Можно звать с любого потока; тот же
    /// epoch/overflow[8]/log+leak контракт, что у retireMesh, плюс
    /// dedup/cap выше: под мьютексом только штамп epoch + проверки +
    /// append, никаких sg.*. Пустой набор — no-op (нечего ретайрить).
    /// Уничтожение (`ComputeBundle.deinit`: вьюхи, затем пайплайн, шейдер,
    /// затем буферы) — только context-поток во flush/deinit.
    pub fn retireComputeBundle(self: *Self, allocator: std.mem.Allocator, bundle: ComputeBundle) void {
        if (bundle.isEmpty()) return;
        lockSpin(&self.mutex);
        defer self.mutex.unlock();
        const entry = Entry{ .kind = .compute, .compute = bundle, .epoch = self.current_epoch };
        if (self.containsLocked(entry)) {
            self.duplicate_drops += 1;
            return;
        }
        if (!self.admitsLocked()) {
            self.capped_drops += 1;
            std.log.err("scene: retire queue cap {d} reached, leaking compute bundle (state buffer id {})", .{ self.pending_cap, bundle.state_buffer.id });
            return;
        }
        self.pending.append(allocator, entry) catch {
            if (self.overflow_len < self.overflow.len) {
                self.overflow[self.overflow_len] = entry;
                self.overflow_len += 1;
            } else {
                std.log.err("scene: destroy queues exhausted, leaking compute bundle (state buffer id {})", .{bundle.state_buffer.id});
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
    /// (повторный ретайр одного GPU-хендла), проб-таргеты — по id куб-имиджа,
    /// ui3d-таргеты — по id RT-имиджа (повторный ретайр одного таргета),
    /// compute-наборы — по полному кортежу всех восьми id (один и тот же
    /// проигравший исход; один и тот же числовой id в РАЗНЫХ полях —
    /// например буфер 5 против вьюхи 5 — совпадением не считается).
    fn sameHandle(a: Entry, b: Entry) bool {
        if (a.kind != b.kind) return false;
        return switch (a.kind) {
            .mesh => a.mesh == b.mesh,
            .buffer => a.buffer.id == b.buffer.id,
            .probe => a.probe.image.id == b.probe.image.id,
            .ui3d => a.ui3d.image.id == b.ui3d.image.id,
            .compute => std.meta.eql(a.compute, b.compute),
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
                    // Wave 25: снятый проб-тагрет — куб, его вьюхи/сэмплер
                    // и глубина одним значением (пустой pre-capture таргет —
                    // no-op destroy'ы, но дисциплина uniform).
                    .probe => {
                        var target = entry.probe;
                        target.deinit();
                    },
                    // Wave 28: снятый ui3d-таргет — RT, его вьюхи/сэмплер
                    // и панельные UI-буферы одним значением (пустой
                    // pre-capture таргет — no-op destroy'ы, но дисциплина
                    // uniform).
                    .ui3d => {
                        var target = entry.ui3d;
                        target.deinit();
                    },
                    // Staged compute-путь: проигравший набор — вьюхи,
                    // затем пайплайн, шейдер, затем буферы (см.
                    // ComputeBundle.deinit); частичные исходы несут нули
                    // в несозданных полях.
                    .compute => {
                        var bundle = entry.compute;
                        bundle.deinit();
                    },
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
                        .probe => {
                            var target = entry.probe;
                            target.deinit();
                        },
                        .ui3d => {
                            var target = entry.ui3d;
                            target.deinit();
                        },
                        .compute => {
                            var bundle = entry.compute;
                            bundle.deinit();
                        },
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

// GPU-retire regression tests live in `gpu_retire_tests.zig` (same directory,
// imported below so the test registry picks them up exactly once).

test {
    _ = @import("gpu_retire_tests.zig");
}
