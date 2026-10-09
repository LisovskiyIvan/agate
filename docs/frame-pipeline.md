# Кадровый конвейер (многопоточный кадр)

> Путь: `src/agate/scene/frame_api.zig`, `frame_prepare.zig`, `frame_build.zig`, `frame_draws.zig`, `frame_render.zig`, `upload_packets.zig` (фасад: `upload_packets_stage/flush/commit/transient.zig`), `gpu_retire.zig`, `patch_instance_refs.zig`, `ui_capture.zig`, `ui_frame.zig` (+ фасад `src/agate/runtime.zig`) · Импорт: `agate.Scene`, `agate.Runtime` (`root.zig`) · Потоки: game (producer) и context (consumer); finish/render — без мьютекса, begin — по режиму.

## Что это

Многопоточный кадровый конвейер отделяет построение кадра (game-поток) от его подготовки и отрисовки (context-поток). Game-сторона замораживает весь изменяемый полезный груз в слот тройного буфера (`FrameDraws`, три `FrameDrawSlot`), context-сторона потребляет только замороженное. Инварианты: один producer; context владеет переворотом `front`; GPU-эпохи принадлежат только context; отмена claim ничего не потребляет (флаги перевооружаются через `restageDroppedSlot`).

## Быстрый старт

Простой путь — фасад `Runtime` (пример — `agate/examples/shader_materials`, демо `main` на `update` + `renderFrame`):

```zig
var rt = agate.Runtime.init();
defer rt.deinit();
// Игровой поток:
rt.gameLock();
scene.update(dt) catch {};
const built = rt.produceBuild(&scene); // claim -> build -> stageUi -> publish
rt.gameUnlock();
_ = built;
// Context-поток (каждый кадр):
switch (rt.renderFrame(&scene)) {
    .prepared => {},
    .reused => {}, // свежего билда нет, показан предыдущий front
    .skipped => {},
    .busy => {},
}
```

Продвинутый путь (ручная хореография, как в Sandbox):

```zig
// game:
if (scene.tryClaimBuildSlot()) |*claim| {
    claim.build();
    claim.stageUi();
    claim.stageHostBytes(host_bytes);
    claim.publish();
}
// или one-liner: _ = scene.buildPreparedFrame(); // tryClaimBuild + stageUi + publish, полный freeze
// context (begin — без мьютекса по умолчанию, finish — без):
if (scene.beginStagedPrepare()) |claim| {
    scene.finishStagedPrepare(claim);
} 
...
scene.render(); // никогда не готовит свежий кадр сам; без prepared — только renderReuse валидного front
```

Серийный путь (один поток, `--no-threads`): тот же staged-алгоритм — `rt.update` + `rt.renderFrame`.

> Breaking API: `Scene.prepareFrame`, `Runtime.prepareSerial`, `FrameDraws.backIndex/backSlot/publish`, `Scene.lock_free_prepare` удалены. `PrepareClaim` — `{ token_id, back_idx, build_seq, host_bytes }`.

## API

### Claim'ы producer'а (`scene/core.zig`, `scene/frame_build.zig`)

```zig
pub fn tryClaimBuildSlot(self: *Scene) ?BuildClaim
pub const BuildClaim = struct {
    pub fn build(self: *BuildClaim) void
    pub fn stageUi(self: *BuildClaim) void
    pub fn stageHostBytes(self: *BuildClaim, bytes: []const u8) void
    pub fn publish(self: *BuildClaim) void
    pub fn cancel(self: *BuildClaim) void
};
pub fn buildIntoClaimedSlot(scene: anytype, slot: usize, seq: u64) void
```

`tryClaimBuildSlot` резервирует свободный слот (`draws.claimBack()`, `null` = насыщение, кадр пропускается — latest-wins, никогда блокировка) и следующее поколение `seq`. `build` выполняет общее ядро сборки: сначала game-side commit результатов прошлого latch'а, затем сброс слота, снимок, CPU-стейджинг инстансов, заморозка записей, capture частиц/физики, `stageUploads` и построение очередей с `instances_prepared=true`. `stageUi`/`stageHostBytes` вызываются строго после `build` (сборка сбрасывает слот). `publish` фиксирует `(build_slot, build_seq)` и снимает `WRITING` без переворота `front`; `cancel` перевооружает замороженные флаги (`restageDroppedSlot`) и снимает claim без handoff. Ядро сборки sg-free и никогда не трогает эпохи `GpuRetire` (проверено тестом).

Порядок в `buildIntoClaimedSlot`: commit прошлого front под read-lease (`pinFrontReader`/`unpinReader`) → `back.reset()` → снимок (`takeLatest` из `frame_handoff`, иначе свежий `packFrameSnapshot`) → `stageInstancesCpu` → заморозка `instance_build_view` → `freezeStagedRecords` → `particles.buildCapture` / `physics.buildDebug` → `stageIntoSlot` в слот (срезы 4/5) → `stageUploads` (срез 6) → `buildQueuesInto` с подменой scratch (восстанавливается после).

### Staged prepare (`scene/frame_prepare.zig`, `scene/frame_api.zig`)

```zig
pub const PrepareClaim = struct {
    token_id: u64,
    back_idx: usize,
    build_seq: u64,
    host_bytes: []const u8 = &.{},
};
pub fn beginStagedPrepare(self: anytype) ?PrepareClaim
pub fn finishStagedPrepare(self: anytype, claim: PrepareClaim) void
pub fn cancelStagedPrepare(self: anytype, claim: PrepareClaim) void
```

`beginStagedPrepare`: null при отсутствии свежего полного билда — никогда live-чтений. Выполняет: lease-claim слота (удерживается весь prepare), сброс `frame_prepared`, `retire_epoch = gpu_retire.begin()`, сброс статистики с переносом staged скаляров (`pending_update_ms`/`pending_physics_ms` через acquire), закрытие предыдущего незакрытого epoch. `finishStagedPrepare`: потребляет токен один раз (one-shot, живой GPU-владелец) — только latch (GPU-половины по слотовым записям, `patchInstanceRefs`, merge `build_stats`), без перестройки очередей; затем `tryPublish` и штамп `last_latched_seq`. `cancelStagedPrepare`: снимает lease, `frame_prepared=false`, `gpu_retire.complete(epoch)`.

### Тройной буфер (`scene/frame_draws.zig`)

```zig
pub const SLOT_COUNT: usize = 3;
pub const LeaseError = error{ SlotBusy, PinnedSlot, NotClaimed, NotPinned, ... };
pub const HandoffClaim = struct { slot: usize, seq: u64, has_scene_build: bool };
pub const FrameDraws = struct {
    pub fn claimBack(self: *FrameDraws) ?usize
    pub fn claimSlot(self: *FrameDraws, idx: usize) LeaseError!void
    pub fn claimLatestHandoff(self: *FrameDraws, build_slot: ..., build_seq: ..., latched: u64, staged_only: bool) ...! ?HandoffClaim
    pub fn tryPublish(self: *FrameDraws, back_idx: usize) LeaseError!void
    pub fn releaseHandoff(self: *FrameDraws, back_idx: usize) LeaseError!void
    pub fn releaseHandoffWithSeq(self: *FrameDraws, back_idx: usize, seq: u64, ...) LeaseError!void
    pub fn cancelClaim(self: *FrameDraws, back_idx: usize) LeaseError!void
    pub fn cancelHandoffClaim(self: *FrameDraws, back_idx: usize, seq: u64, ...) LeaseError!void
    pub fn pin(self: *FrameDraws, idx: usize) LeaseError!void
    pub fn pinFront(self: *FrameDraws) usize
    pub fn pinFrontReader(self: *FrameDraws) usize
    pub fn unpin(self: *FrameDraws, idx: usize) LeaseError!void
    pub fn unpinReader(self: *FrameDraws, idx: usize) LeaseError!void
    pub fn frontIndex(self: *FrameDraws) usize
    pub fn isPinned(self: *FrameDraws, idx: usize) bool
    pub fn isReadPinned(self: *FrameDraws, idx: usize) bool
    pub fn pinsHeld(self: *FrameDraws) usize
    pub fn cpuBytes(self: *const FrameDraws) usize
    pub fn deinit(self: *FrameDraws, allocator: std.mem.Allocator) void
    // Claim/lease/pin только через claimBack/claimSlot/claimLatestHandoff/tryPublish/releaseHandoff/pin/unpin.
};
```

Протокол: `claimBack` → заполнить → `tryPublish` (producer) или `releaseHandoff` (передача prepare без переворота `front`); consumer — `pin`/`pinFront` → читать → `unpin`. `pin` отказывает пишущему слоту (`SlotBusy`), publish в pinned слот отказывает (`PinnedSlot`) — всё со счётчиками (`saturation_skips`, `publish_refusals`, `pin_denials`, `unpin_denials`). Потерянный `unpin` — никогда клин: только сжимает свободное множество; `deinit` ассертит отсутствие пинов.

### Содержимое слота (`FrameDrawSlot`)

| Группа | Поля |
|---|---|
| Очереди отрисовки | `primary: RenderQueues`, `views: [MAX_CAMERAS]RenderQueues`, `outline_items`, `outline_skins`, `highlight_items`, `shadow: PreparedShadowDraws` |
| Срез 1 — инстансы | `staged_instances: []StagedInstanceRecord` |
| Срез 2 — UI | `ui_vertices`, `ui_indices`, `ui_packet: UiPacketState` |
| Волна 27 — снимок | `snapshot: SceneFrameSnapshot` (копия по значению) |
| Счётчики | `build_stats: SceneStats` (merge в prepare, затем очистка) |
| Срезы 4/5 — capture | `particle_draws`, `physics_lines`, `physics_visible` |
| Срез 6 — загрузки | `morph_uploads/morph_data`, `p_cpu_uploads/p_cpu_data`, `p_gpu_uploads/p_gpu_data`, `p_compute_uploads/p_compute_data`, `trail_uploads/...`, `soft_uploads/...`, `greased_uploads/...`, `pending_uploads/pending_verts/pending_indices/pending_delta_data` |
| Host-канал | `host_bytes: []u8` (см. ниже) |
| Идентификация | `frame_id: u64`, `retire_epoch: Epoch`, `build_seq: u64`, `has_scene_build: bool` |

Методы: `reset()` (чистит длины всех списков, сохраняя capacity — stale невозможен), `deinit(allocator)`, `cpuBytes()` (сумма retained capacity; фиксированный снимок не считается).

Прочие типы пакетов (`frame_draws.zig`): `MorphUpload`, `ParticleCpuUpload`, `ParticleGpuUpload`, `ParticleComputeUpload`, `TrailUpload`, `SoftUpload`, `GreasedUpload`, `PendingMeshUpload` — у каждого `token`/`uid`/индексы для валидации commit'а и флаг `delivered` (плюс `created_*_id` для установок хэндлов).

### Заморозка загрузок (`scene/upload_packets.zig` — фасад; код разложен по типам ресурсов и фазам: `upload_packets_stage.zig` — producer-freeze, `upload_packets_flush.zig` — context-flush, `upload_packets_commit.zig` — game-side commit + re-arm, `upload_packets_transient.zig` — reuse-переподача; публичный API фасада неизменен)

```zig
pub fn stageUploads(scene: anytype, slot: anytype) void
pub fn flushSlotUploads(scene: anytype, slot: anytype) void
pub fn commitSlotResults(scene: anytype, front: anytype) void
pub fn restageDroppedSlot(scene: anytype, slot: anytype) void
```

Контракт lock-free публикации: producer гасит live-флаги в момент заморозки (`m.morph_upload_needed = false`, `ps.instance_dirty`, `gpu_dirty`, `compute_flush_pending`, `tm.gpu_dirty`, …) — context их никогда не пишет. Context (`flushSlotUploads`, только при `sg.isvalid()`) заливает слотовые копии и ставит исходы (`delivered`, `created_*_id`). Game-side `commitSlotResults` (при следующем билде, один раз на `frame_id` — guard `last_upload_commit_frame`) применяет исходы по валидации `token/index/uid`: ставит хэндлы, публикует скаляры, освобождает pending-массивы, продвигает compute-кольцо; недоставленное перевооружает для повторной заморозки. OOM при stage — пропуск пакета с сохранённым флагом (следующий билд попробует снова); staged-wins для capture'ов (fail-closed в coherent-empty).

Два узких исключения остаются под исключением (см. заголовок модуля): прямое создание mesh-буферов и часть compute-стейджинга.

### Retire-очередь (`scene/gpu_retire.zig`)

```zig
pub const Epoch: type = u64;
pub const Kind = enum { mesh, buffer, probe, ui3d };
pub const GpuRetireQueue = struct {
    pub fn begin(self: *Self) Epoch
    pub fn complete(self: *Self, e: Epoch) void
    pub fn current(self: *Self) Epoch
    pub fn lastCompleted(self: *Self) Epoch
    pub fn retireMesh(self: *Self, allocator: std.mem.Allocator, mesh: *Mesh) void
    pub fn retireBuffer(self: *Self, allocator: std.mem.Allocator, buf: sg.Buffer) void
    pub fn retireProbeTarget(self: *Self, allocator: std.mem.Allocator, target: probe_layer.ProbeGpu) void
    pub fn retireUi3dTarget(self: *Self, allocator: std.mem.Allocator, target: gui3d_layer.Ui3dTarget) void
    pub fn flush(self: *Self, allocator: std.mem.Allocator) void
    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void
    pub fn retainedCount(self: *Self) usize
    pub fn cappedDropCount(self: *Self) u64
    pub fn duplicateDropCount(self: *Self) u64
    pub fn admitsOneMore(self: *Self) bool
};
```

Эпохи принадлежат context: `begin` в `beginPrepare` (заодно закрывает предыдущий незакрытый), `complete` в render/reuse/cancel. `flush` уничтожает только завершённые эпохи. Очередь ограничена (`pending_cap`): переполнение — счётный дроп (`cappedDropCount`), дубли — `duplicateDropCount`. Слот `FrameDrawSlot` никогда не уничтожает GPU-хэндлы (заимствованные значения под P3-дисциплиной).

### Прочее

```zig
// patch_instance_refs.zig — финализация provisional build_view хэндлов
pub fn patchInstanceRefs(back: *FrameDrawSlot) void
// ui_capture.zig
// ui_capture.zig (внутреннее, только через BuildClaim.stageUi / buildPreparedFrame):
pub fn stageUiPacketInto(scene: anytype, slot: usize) void
pub fn captureUiFrame(scene: anytype, snap: *const SceneFrameSnapshot, back: *FrameDrawSlot) void
// ui_frame.zig
pub const UploadResult = enum { ... };
pub const UiPacketHandles = struct { ... };
pub const UiPacketState = struct { ... };
pub const UiUploadContext = struct { ... };
pub const UiFrame = struct {
    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void
    pub fn clearEmpty(self: *Self) void
    pub fn capture(self: *Self, allocator: std.mem.Allocator, canvas: *const UICanvas, screen_w: f32, screen_h: f32) void
    pub fn capturePacket(self: *Self, ...) void
    pub fn upload(self: *Self, canvas: *UICanvas, ctx: UiUploadContext) UploadResult
    pub fn drawPrepared(self: *const Self) void
};
// frame_render.zig
pub fn render(scene: anytype) void
pub fn renderReuse(scene: *anytype) void
```

`patchInstanceRefs` сверяет `source_uid` записей с live-мешами и финализирует `instance_buffer`/`visible_instance_count` из post-latch `instance_render`; несовпадение — fail-closed в ноль (невидимо). UI staged-пакет — latch из слота, не live-канвас (first-wins lifecycle; uploads/retires валидны); missing/failure — coherent-empty. `UiFrame` — одиночный committed кадр вне слотов: `capture`/`capturePacket` — CPU, `upload` — GPU, `drawPrepared` — отрисовка без загрузок. `render` читает только pinned front и никогда не готовит свежий кадр сам; `renderReuse` перепрезентует последний валидный front (счётчик `reuse_streak`).

Наблюдаемость (`frame_api.zig`): `hasConsumableFrame() bool`, `reuseStreak() u64`, `uiPacketLatchedCount() u64`, `pendingRetires() usize`, `preparedDraws() *const FrameDrawSlot` (валиден только пока `frame_prepared` или под pin'ом).

### Фасад Runtime (`src/agate/runtime.zig`)

```zig
pub const BeginResult = struct { claim: ?Scene.PrepareClaim, busy: bool, wait_ns: u64, held_ns: u64 };
pub const FrameResult = enum { prepared, reused, skipped, busy };
pub const Metrics = struct { producer_builds, producer_skips, begins, begin_empty, begin_busy, finishes, cancels, reuses, skipped_presents: u64 };
pub const Runtime = struct {
    pub fn init() Runtime
    pub fn setProducerExclusion(self: *Runtime, excluded: bool) void
    pub fn setLockWaitNs(self: *Runtime, ns: u64) void
    pub fn spawnWorker(self: *Runtime, comptime entry: fn () void) bool
    pub fn quiesce(self: *Runtime) void
    pub fn shouldRun(self: *const Runtime) bool
    pub fn deinit(self: *Runtime) void
    pub fn gameLock(self: *Runtime) void
    pub fn gameUnlock(self: *Runtime) void
    pub fn produceBuild(self: *Runtime, scene: *Scene) bool
    pub fn produceBuildWithHostBytes(self: *Runtime, scene: *Scene, host_bytes: ?[]const u8) bool
    pub fn update(self: *Runtime, scene: *Scene, ctx: anytype, comptime tick: fn (@TypeOf(ctx)) void) bool
    pub fn renderFrame(self: *Runtime, scene: *Scene) FrameResult
    pub fn tryRunLocked(self: *Runtime, comptime work: fn () void) bool
    pub fn beginPrepare(self: *Runtime, scene: *Scene) BeginResult
    pub fn beginPrepareWith(self: *Runtime, scene: *Scene, ...) BeginResult
    pub fn finishPrepare(self: *Runtime, scene: *Scene, claim: Scene.PrepareClaim) void
    pub fn cancelPrepare(self: *Runtime, scene: *Scene, claim: Scene.PrepareClaim) void
    pub fn reuseIfConsumable(self: *Runtime, scene: *Scene) bool
};
```

| Режим | Begin | Finish/render | Когда |
|---|---|---|---|
| Staged lock-free (default) | без мьютекса | без мьютекса | свежий билд заморозил всё; commit — game-side |
| Staged с исключением (`--prepare-exclusion`) | под `mutex` (`lock_wait_ns`, 0 = try) | без | диагностика тех же данных/алгоритма, не откат render-пути |
| Single-thread (`--no-threads`) | тот же staged-путь | render перекрывается | один поток, `rt.update` + `rt.renderFrame` |

Сложность/аллокации: claim/pin/publish — O(1) под внутренним мьютексом слотов; `reset` — O(списки) без освобождения (retained capacity); slot payload растёт до high-water-mark и переиспользуется; `cpuBytes` — O(число списков).

## Потоки и владение

- Game-поток: `update` (только update-owned состояние + mailbox'ы + `pending_update_ms`), `tryClaimBuildSlot`/`build`/`stageUi`/`stageHostBytes`/`publish`/`cancel`, `commitSlotResults` + `commitPublishedRecords` в начале каждого билда. Никогда `sg.*`, никогда эпохи retire.
- Context-поток: `beginPrepare*` (по режиму — с исключением producer'а или без), `finishPrepare`/`cancelPrepare` (всегда без фазового мьютекса — только слот + context-owned), `render`/`renderReuse`, `flushSlotUploads`, `GpuRetire.begin/complete/flush`.
- Lock-free контракт требует одновременно: билд заморозил загрузки (`stageUploads`) + UI (`stageUi`) + host-байты; host читает live только через `stageHostBytes` или доказано context-owned; нет гонок registry add/remove с in-flight latch (commit-guard'ы держат когерентность, но контракт приложения это запрещает).
- Skip-if-newer скаляры: `recordUpdateTime(ms)`/`recordPhysicsTime(ms)` пишут в атомики (`pending_update_ms`, release); begin читает (acquire) в `stats` — монотонная передача без мьютекса, последнее значение побеждает.
- `host_bytes`/`stageHostBytes`: game копирует мелкие host-пейлоады (имя пика, tally памяти, ≤128B на практике) в claimed слот; context читает `PrepareClaim.host_bytes` вместо live host-состояния. OOM — fail-closed (область чистится, счётчик `host_bytes_oob_drops`).
- UI: staged пакет (`ui_vertices`/`ui_indices` + `ui_packet` с замороженными хэндлами) — latch из слота без чтения live-канваса (GPU canvas metadata + first-wins/uploads/retires валидны); мутации `ui_canvas` должны затихнуть до prepare.
- Эпохи retire открываются/закрываются только на context; повторный prepare сбрасывает `frame_prepared` до flush (старый front теряет GPU-потребляемость, CPU-память остаётся как scratch).

## Ошибки и краевые случаи

- Насыщение (`claimBack → null`): пропуск кадра со счётчиком, latest-wins — никогда блокировка. Contention-yield (`concurrent_yield_ns != 0`): park перед reserve, если прошлый билд не потреблён.
- `LeaseError`: `SlotBusy` (pin на пишущий слот — retry на новый front), `PinnedSlot` (publish в pinned — counted skip), `NotClaimed`/`NotPinned` — баги вызывающей стороны.
- Потерянный begin-токен НЕ клинит конвейер: несовпадение/двойной `finish`/`cancel` — лог err + защитный release активного claim (`prepare_claim_active` сбрасывается, кадр теряется). Контракт остаётся: каждый успех — ровно один `finish`/`cancel`, на всех путях включая ошибки.
- Двойной `finish`/`publish`/`cancel` — идемпотентный лог (нет активного claim — no-op); несовпадение токена — release + возврат.
- OOM в stage: пакет пропускается, флаг остаётся (следующий билд повторит); OOM в capture — staged-wins в coherent-empty.
- Мутация mesh-листа между build и latch: latch не читает live-список; commit-guard следующего билда ловит по `token/index/uid`.
- Повторный билд без промежуточного latch — commit пропускается (`last_upload_commit_frame`), исходы не двоятся.
- `render` между `begin` и `finish` видит `frame_prepared == false` и роняет present — держать пару смежно.
- Отмена prepare закрывает epoch паринга (`complete`), иначе эпоха повиснет до следующего begin (авто-закрытие там же).

## Transient-буферы в reuse-кадре (решено)

**Симптом (был).** `renderReuse` перепрезентирует front-слот БЕЗ staged prepare: `flushSlotUploads` не выполнялся, а draw-records биндили буферы с `usage.write_transient` (instance-матрицы, morph-дельты, particle cpu/gpu payloads, трейлы/softbody/greased). sokol валидатор ловил `VALIDATE_DRAW_WRITE_BUFFER_TRANSIENT_MISSING`, а в non-`SOKOL_VALIDATE_NON_FATAL` сборках — паника (`VALIDATION_FAILED`). Гейт-воспроизведение: `hdr-showcase --frames >= 400` (reuse-проба на кадре 200).

**Фикс.** Reuse-кадр переподаёт staged payload'ы фрот-лота в те же transient-буферы ДО любых draw'ов: `upload_packets.rewriteTransientWrites` (`upload_packets_transient.zig`: morph / particle cpu+gpu / trails / softbodies / greased) и `instance_staging.rewriteSlotInstanceBuffers` (cur в bound-буфер; парный prev — тем же payload'ом, повтор кадра = нулевая instance-моушн-семантика). Вызов из `frame_render.render()` под `if (scene.rendering_reuse)`. Write-only по контракту: без upload-meter, без delivered/outcome-мутаций, без создания буферов и без particle-compute state clear (это семантика — состояние симуляции переживает повтор).

**Гейт.** `hdr-showcase --frames 400`: 0 ошибок валидации, 0 паник (было 1 + panic на baseline). Известные непокрытые corner'ы (осознанно, не в шипе): pending-mesh создания (their ids съедены commit'ом) и compute-particle state буферы (пишутся на render-time compute-пути, который reuse-кадр перепрогоняет).

## Производительность

- Горячий путь аллокаций не делает: слоты reuse capacity, latch/patch поверх записей без live-чтений; `build_stats` мержится из immutable копии без общего аккумулятора.
- Параллелизм: CPU-стейджинг инстансов через `jobs.global`; game-билд перекрывается с context-презентом предыдущего front (3-й слот — именно для этого).
- `reuseIfConsumable`/`renderReuse`: без свежего билда — повтор front без prepare-стоимости; `reuse_streak × destroy-rate` ограничен `pending_cap` retire-очереди.
- Наблюдаемость: `Metrics` фасада (builds/skips/begins/begin_empty/begin_busy/finishes/cancels/reuses), `SceneStats`, `cpuBytes()`, `pendingRetires()`, `reuseStreak()`, `host_bytes_oob_drops`, `build_oom_drops` (дропы очередей/пакетов аплоадов при OOM).

## Смотрите также

- `./architecture.md` — место кадра в устройстве движка.
- `./runtime.md` — жизненный цикл приложения и `Runtime` целиком.
- `./render-pipeline.md` — что происходит внутри построенных очередей.
- `./scene.md` — `Scene.update`, снимки, mailbox'ы.
- `./ui.md` — канвас, `UiFrame`, staged UI-пакеты.
- `./profiler.md` — `update_ms`/`prepare_ms`, счётчики кадров.
