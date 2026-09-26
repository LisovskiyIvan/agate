# Анимация

> Путь: src/agate/animation/*.zig · Импорт: agate.Skeleton, agate.AnimationGroup, agate.evaluateSkeleton, agate.retargetAnimationGroup (root.zig) · Потоки: update-поток пишет позы, render-поток читает skin-матрицы через double-buffer

## Что это

Скелетная и нодовая анимация в стиле glTF: `skeleton.zig` (кости, skin-матрицы, double-buffer), `sampler.zig` (интерполяция LINEAR/STEP/CUBICSPLINE), `channels.zig` (привязки кость/нода/морф), `group.zig` (клип с весом, скоростью, фейдами, событиями), `eval.zig` (блендинг клипов в позу), `retarget.zig` (перенос клипов между ригами), `easing.zig` (кривые для нодовых треков), `animation.zig` (фасад). Лимит — `MAX_BONES = 64` кости на скелет; skinning считается на CPU, в шейдер едут готовые матрицы.

Две семьи треков: skeleton-каналы (`AnimationChannel`: `bone_index + target_path + sampler`) двигают кости; node-каналы (`NodeChannel`: `target + target_path + easing`) двигают трансформы мешей напрямую, weights-каналы — морфы. Скелетный блендинг множит клипы, нодовые треки НЕ кросс-блендятся между группами («последняя обновлённая группа побеждает»).

## Быстрый старт

```zig
// Скелет из загрузчика (кости с bind-позой и inverse-bind матрицами),
// клип из glTF-треков:
clip.skeleton = skel;
clip.play(true);

// Кадр:
clip.update(dt); // двигает current_time, применяет нодовые треки
agate.evaluateSkeleton(skel, &.{clip}, &.{}); // бленд в кости + skel.update()
const skin = skel.getRenderSkinMatrices(); // *const [64]Mat4 для рендера

// Кроссфейд ходьба -> бег:
walk.crossFadeTo(run, 0.25);

// Событие на таймлайне (шаг, звук):
try clip.setEvents(allocator, &.{ .{ .time = 0.5, .name = "footstep" } });
clip.on_event = myCallback;
const fired = clip.drainFiredEvents(); // имена за последний update
```

Ретаргет между ригами с разными именами:

```zig
const map = [_]agate.BoneMap{
    .{ .source = "b_Hip_01", .target = "Skeleton_torso_joint_1" },
};
const run2 = try agate.retargetAnimationGroup(allocator, run, src_skel, dst_skel,
    .{ .bone_map = &map, .translation_mode = .scale_by_bone_length });
defer run2.deinit();
```

## API

### Скелет

```zig
pub const MAX_BONES: usize = 64;
pub const Bone = struct { name: []const u8 = "", parent_index: ?usize = null,
    local_position: Vec3 = ..., local_rotation: Quat = ..., local_scale: Vec3 = ...,
    bind_position: Vec3 = ..., bind_rotation: Quat = ..., bind_scale: Vec3 = ...,
    inverse_bind_matrix: Mat4 = Mat4.identity, model_matrix: Mat4 = Mat4.identity };
pub fn init(allocator: std.mem.Allocator, bone_count: usize) !*Skeleton
pub fn deinit(self: *Skeleton) void
pub fn resetToBindPose(self: *Skeleton) void
pub fn update(self: *Skeleton) void
pub fn getRenderSkinMatrices(self: *const Skeleton) *const [MAX_BONES]Mat4
pub fn findBoneIndex(self: *const Skeleton, name: []const u8) ?usize
pub fn getBoneModelMatrix(self: *const Skeleton, bone_index: usize) Mat4
pub fn getBoneWorldMatrix(self: *const Skeleton, bone_index: usize, host_world_matrix: Mat4) Mat4
pub fn getBoneWorldPosition(self: *const Skeleton, bone_index: usize, host_world_matrix: Mat4) Vec3
```

`update` рекурсивно считает `model_matrix` (родитель × локальный TRS через `Mat4.fromQuatTranslationScale`, сверх лимита — обрезка до 64) и `skin = model × inverse_bind`, публикуя в double-buffer: писатели льют в `(1 − render_slot)`, читатели берут `render_slot` с acquire/release семантикой (`skin_matrices` — sim-представление для совместимости). `getBoneWorldPosition` — сокет для аттачментов (оружие, эффекты). O(кости), аллокаций нет. Вне диапазона — `identity`/host-матрица, не краш.

### Клип (AnimationGroup)

```zig
pub fn init(allocator: std.mem.Allocator, name: []const u8, channels: []AnimationChannel, duration: f32) !*AnimationGroup
pub fn deinit(self: *AnimationGroup) void
pub fn play(self: *AnimationGroup, loop: bool) void
pub fn playRange(self: *AnimationGroup, from: f32, to: f32, loop: bool, speed: ?f32) void
pub fn setSpeed(self: *AnimationGroup, speed: f32) void
pub fn setWeight(self: *AnimationGroup, w: f32) void
pub fn setAdditive(self: *AnimationGroup, additive: bool) void
pub fn fadeTo(self: *AnimationGroup, target_weight: f32, duration: f32, stop_if_zero: bool) void
pub fn fadeIn(self: *AnimationGroup, duration: f32) void
pub fn fadeOut(self: *AnimationGroup, duration: f32) void
pub fn crossFadeTo(self: *AnimationGroup, target: *AnimationGroup, duration: f32) void
pub fn pause(self: *AnimationGroup) void
pub fn stop(self: *AnimationGroup) void
pub fn goToFrame(self: *AnimationGroup, time: f32) void
pub fn update(self: *AnimationGroup, dt: f32) void
pub fn applyAtTime(self: *AnimationGroup, time: f32) void
pub fn applySkeletonAtTime(self: *AnimationGroup, time: f32) void
pub fn applyNodesAtTime(self: *AnimationGroup, time: f32) void
pub fn sampleBoneAtTime(self: *const AnimationGroup, bone_idx: usize, time: f32, out_pos: *?Vec3, out_rot: *?Quat, out_scale: *?Vec3) void
pub fn sampleNodeAtTime(self: *const AnimationGroup, target_idx: usize, time: f32, out_pos: *?Vec3, out_rot: *?Quat, out_scale: *?Vec3) void
```

Поля: `duration/from/to/current_time`, `speed_ratio` (отрицательный — реверс), `weight` (clamp [0,1]), `is_additive`, `is_playing`, `loop`. `play` сбрасывает диапазон на `[0, duration]`; `playRange` клампит и ставит курсор внутрь; `goToFrame` — сик без событий. `update` двигает время (`dt * speed_ratio`, wrap по модулю при loop), применяет нодовые треки, собирает события; нулевая длина клипа всё равно драйвит ноды. `stop` возвращает bind-позу скелета и rest-позы нод/морфов. Фейды линейны по весу; `fadeOut` останавливает в конце; `crossFadeTo(self == target)` — no-op.

События:

```zig
pub const AnimationEvent = struct { time: f32, name: []const u8 };
pub fn setEvents(self: *AnimationGroup, allocator: std.mem.Allocator, events: []const AnimationEvent) !void // имена дублируются, вход — borrowed
pub fn drainFiredEvents(self: *AnimationGroup) []const []const u8 // валиден до следующего update/setEvents/deinit
on_event: ?*const fn (ctx: ?*anyopaque, name: []const u8) void = null;
```

Стреляют только в `update` при продвижении времени: вперёд `(prev, curr]` с разбивкой wrap'а, назад зеркально; сики, `stop`, завершение fade-out — никогда. Один проход через весь диапазон стреляет каждым событием ровно раз. Ёмкость `fired_names == events.len`, сам `update` не аллоцирует.

Привязки нод и морфов:

```zig
pub fn bindNodeTarget(self: *AnimationGroup, position: *Vec3, rotation_euler: *Vec3, scaling: *Vec3, skip: bool) !usize
pub fn bindMorphTarget(self: *AnimationGroup, weights: []f32, dirty: *bool) !usize
pub fn addNodeChannel(self: *AnimationGroup, channel: NodeChannel) !void
pub fn captureNodeRestPoses(self: *AnimationGroup) void
pub fn restoreNodeRestPose(self: *AnimationGroup) void
pub fn restoreMorphRestWeights(self: *AnimationGroup) void
```

Указатели на трансформы сцены должны жить дольше группы; `rest_*` снапшотятся при bind'е. `skip = true` — трек распарсен, но не применяется (сустав скинированного меша, чтобы не дабл-применять). Веса морфов блендятся с rest и клампятся в [0,1], меш помечается dirty.

### Сэмплер, каналы, easing

```zig
pub const AnimationPath = enum { translation, rotation, scale, weights };
pub const AnimationInterpolation = enum { linear, step, cubic_spline };
pub const AnimationSampler = struct { timestamps: []const f32, outputs: []const f32, interpolation: AnimationInterpolation = .linear };
pub fn sampleVec3(self: AnimationSampler, time: f32) Vec3
pub fn sampleVec3Eased(self: AnimationSampler, time: f32, easing: EasingType) Vec3
pub fn sampleQuat(self: AnimationSampler, time: f32) Quat
pub fn sampleQuatEased(self: AnimationSampler, time: f32, easing: EasingType) Quat
pub fn sampleWeightsInto(self: AnimationSampler, time: f32, out: []f32, easing: EasingType) void
pub const AnimationChannel = struct { bone_index: usize, target_path: AnimationPath, sampler: AnimationSampler };
pub const NodeChannel = struct { target: usize, target_path: AnimationPath, sampler: AnimationSampler, easing: EasingType = .linear };
pub const NodeTarget = struct { position: *Vec3, rotation_euler: *Vec3, scaling: *Vec3, rest_position, rest_rotation: Quat, rest_scale, skip: bool };
pub const MorphWeightsTarget = struct { weights: []f32, rest_weights: []f32, dirty: ?*bool };
pub fn samplerHasFrames(sampler: AnimationSampler, path: AnimationPath, weight_count: usize) bool
pub const EasingType = enum { linear, ease_in_quad, ease_out_quad, ease_in_out_quad, ease_in_cubic, ease_out_cubic, ease_in_out_cubic, ease_in_sine, ease_out_sine, ease_in_out_sine };
pub fn evaluate(t: EasingType, x: f32) f32 // f(0)=0, f(1)=1, вход клампится
pub fn easingName(t: EasingType) []const u8
pub fn easingFromName(name: []const u8) ?EasingType
```

glTF CUBICSPLINE — настоящий Hermite: выходы хранят `(in-tangent, value, out-tangent)` на ключ (stride×3 floats), тангенты — per-second склоны, умножаются на `dt` интервала; кватернионы — покомпонентный Hermite + normalize, антиподальные ключи идут коротким путём. Easing варпит только LINEAR/STEP-фактор, Hermite-форму не трогает; STEP easing игнорирует. Поиск кейфрейма — бинарный (`findKeyframeIndex`), O(log n). Битые/урезанные LINEAR/STEP-треки пропускаются с сохранением позы (`samplerHasFrames` считает stride×mult, для weights — по числу морфов меша); кубические деградируют до ближайшего usable-кадра (или нуля/identity). `sampleWeightsInto` с пустым выходом или без ключей — no-op.

### Оценка и блендинг

```zig
pub fn evaluateSkeleton(skel: *Skeleton, active_base: []const *AnimationGroup, active_additive: []const *AnimationGroup) void
pub fn evaluateSkeletonPoseOnly(skel: *Skeleton, active_base: []const *AnimationGroup, active_additive: []const *AnimationGroup) void
```

Пустые списки — no-op. Фастпас: один базовый клип с весом ≥ 0.999 применяется напрямую. Два клипа — lerp/slerp по нормированной альфе; N клипов — взвешенное среднее позиций/скейлов + усреднение кватернионов с выровненным знаком и normalize (буфер нормированных весов на 16 клипов на стеке). Непокрытые кости падают в bind-позу; клипы с весом ≤ 0.0001 пропускаются. Аддитивные слои: `pos += (sample − bind) * w`, скейл так же, ротация через `bind⁻¹ * sample`, взвешенный slerp от identity. В конце всегда `skel.update()`.

`evaluateSkeletonPoseOnly` — бит-идентична по костям, но не трогает ноды/морфы: убирает единственный кросс-скелетонный shared write (две группы на одном меше, «last wins» стал бы schedule-зависимым). Предусловие: никакого конкурентного `update()` тех же групп. Путь для параллельного прогона (см. ниже); скелеты с нодовыми каналами идут серийным `evaluateSkeleton`.

### Ретаргет

```zig
pub const TranslationMode = enum { keep, scale_by_bone_length, drop };
pub const BoneMap = struct { source: []const u8, target: []const u8 };
pub const RetargetOptions = struct { rotation_only: bool = false,
    translation_mode: TranslationMode = .scale_by_bone_length, bone_map: []const BoneMap = &.{} };
pub fn boneRestLength(bone: Bone) f32 // длина bind-смещения от родителя; 0 у корней
pub fn translationScaleFactor(source_bone: Bone, target_bone: Bone) f32 // dst/src, 1.0 при нулевой длине
pub fn retargetAnimationGroup(allocator: std.mem.Allocator, source_group: *const AnimationGroup,
    source_skeleton: *const Skeleton, target_skeleton: *const Skeleton, options: RetargetOptions) !*AnimationGroup
```

Мэппинг на кость источника по порядку: (1) явный `bone_map` по имени → цель по имени; (2) одноимённая кость (`findBoneIndex`); (3) тот же индекс для безымянных. Без пары — пропуск + один `std.log.warn` с числом за вызов. Ротации копируются как есть; трансляции — по `translation_mode` (`scale_by_bone_length` умножает ВСЕ выходы, включая тангенты, на `translationScaleFactor`); скейл/веса — как есть; `rotation_only` режет всё кроме ротаций. Возвращённая группа привязана к целевому скелету (`duration/from/to` скопированы), владеет каналами — `deinit` за вызывающим. Ограничения MVP: только скелетные каналы; ноды/морфы/события/биндинги не копируются.

## Потоки и владение

`AnimationGroup` владеет каналами, сэмплерными буферами, именами событий и morph rest-весами (`deinit` всё освобождает); скелет владеет костями и именами (`Skeleton.init` аллоцирует, `deinit` освобождает). Сиквенс кадра: `group.update(dt)` (серийно, двигает время + ноды) → `evaluateSkeleton*` → рендер читает `getRenderSkinMatrices()`. Double-buffer skin-матриц — единственная thread-safe граница: sim пишет в скрытый слот, render читает опубликованный. Параллельная оценка (`scene/animation_runtime.zig`, `min_skeletons_for_workers = 4`): ниже 4 скелетов или без пула — серийно; выше — сначала серийно скелеты с нодовыми записями (точный legacy-порядок), потом `jobs.Pool.forkJoin` по дизъюнктным скелетам через `evaluateSkeletonPoseOnly`. Плейбек (`applyAtTime`, `evaluateSkeleton`) никогда не логирует и не аллоцирует.

## Ошибки и краевые случаи

- OOM только в конструкторах (`init`, `bind*`, `addNodeChannel`, `setEvents`, `retargetAnimationGroup` через `!*AnimationGroup`); кадр (`update`, `evaluate*`, `apply*`) не аллоцирует.
- Вне диапазона: семплинг клампится к крайним ключам; пустые timestamps — `Vec3.zero`/`Quat.identity`/no-op для весов.
- `findBoneIndex` miss, `bone_index >= len`, `target >= len`, `skip` — пропуск, не краш.
- Ретаргет без пар даёт пустую группу (валидна, w=поза bind) + warn; `bone_map` на несуществующую цель — пропуск.
- Веса клипов вне [0,1] клампятся на входе (`setWeight`, `fadeTo`, apply); суммарный базовый вес ≤ 0.0001 — остаётся bind.
- `stop()` во время fade сбрасывает таймеры; `fadeTo` на неиграющей группе с целью > 0 стартует её.
- Лимит 16 нормированных весов на стеке: клипы 17+ нормируются делением на лету — результат тот же.

## Производительность

- Оценка скелета O(кости × клипы); фастпас одного клипа — прямое применение без аллокаций и бленд-веток.
- Нормировка весов вынесена из покостного цикла; кватернионное усреднение N>2 — один проход с выравниванием знака.
- Сэмплинг O(log ключи) на канал; Hermite — ~4 скалярных базиса на компоненту, без аллокаций.
- Порог параллелизма — 4 скелета (`min_skeletons_for_workers`): один скелет ≤ 64 костей, бенчмарк ~130 мкс на 64 кости/3 клипа; половина скелетов при N=4 ещё окупает forkJoin. Бит-идентичность serial vs parallel закреплена дизайном (pose-only без shared writes).
- GPU skinning: матрицы 64×Mat4 в double-buffer; рендер берёт указатель на опубликованный слот без копирования.

## Смотрите также

- `./mesh.md` — скининг меша, morph-веса, `BoneAttachment`.
- `./loader.md` — разбор glTF-анимаций в каналы, `skip` для суставов.
- `./scene.md` — `Scene.updateAnimations`, runtime-порядок оценки.
- `./math.md` — `Quat.slerp/nlerp`, `Mat4.fromQuatTranslationScale`.
- `./profiler.md` — бенчмарки оценки (benchmark 9).
