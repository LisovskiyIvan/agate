# Конвейер отрисовки (render pipeline)

> Путь: `src/agate/scene/render_queue.zig`, `scene/render_queue/`, `scene/instance_staging.zig`, `scene/pipelines.zig`, `scene/forward_pipelines.zig`, `scene/uniforms.zig`, `scene/draw.zig`, `scene/view_render.zig`, `scene/viewport_clear.zig`, `scene/projection.zig`, `scene/project_cache.zig`, `scene/postfx_stack.zig`, `scene/msaa.zig`, `scene/shadow_system.zig`, `scene/cascades.zig`, `scene/shadow_pcss.zig`, `scene/queue_builder.zig`, `scene/probe_render.zig` · Импорт: `agate.Scene`, `agate.RenderMeshItem` (`root.zig`); остальное — internal через `Scene` · Потоки: build — game или context (sg-free при `instances_prepared=true`); latch/upload/draw — только context.

## Что это

Render pipeline превращает live-меши сцены в отсортированные очереди отрисовки и исполняет их по видам (primary + PIP), со снепшотными материалами, инстансингом, тенями и пост-эффектами. Ключевые инварианты: draw-элементы zero-dereference (все GPU-хэндлы и материальные записи — копии значений, никаких `*Mesh` на draw-пути); построение очередей sg-free при staged-пути; GPU-стейджинг инстансов отделён от CPU-cull'а (freeze → latch → patch → commit).

## Быстрый старт

Напрямую эти модули обычно не вызываются — их оркеструет кадр (см. `./frame-pipeline.md`):

```zig
// Кадр: producer build (game) + staged prepare/finish (context) + render.
// render сам ничего не готовит: без consumable frame — пропуск/reuse.
const built = scene.buildPreparedFrame(); // tryClaimBuildSlot+build+stageUi+publish; false = saturation
_ = built;
if (scene.beginStagedPrepare()) |claim| {
    scene.finishStagedPrepare(claim); // queue_builder -> render_queue -> shadow -> latch
}
scene.render();       // view_render -> draw -> postfx_stack.renderChain
```

Точечное использование (диагностика, инструменты):

```zig
// Cull/sort одного вида поверх готового контекста:
scene.buildQueuesInto(&slot, .{
    .snap = &snap,
    .cache_key = scene.frame_id,
    .stats = &scene.stats,
    .eye = eye,
    .sky_texture = sky_tex,
    .ibl_intensity = 1.0,
    .instances_prepared = true,
    .instance_source = .published,
});
// MSAA-политика:
const n = agate_msaa.effectiveSampleCount(requested, .{
    .post_enabled = true,
    .formats_msaa_capable = true,
    .backend = sg.queryBackend(),
});
```

## API

### Очереди: типы и сборка (`render_queue.zig`, `render_queue/`)

Фасад `render_queue.zig` — только реэкспорт (`items`, `cull`, `build`); правило против циклов: листья никогда не импортируют фасад.

```zig
// render_queue.zig (все — реэкспорт из items/cull/build):
pub const RenderMeshItem = items.RenderMeshItem;
pub const RenderInstancedBatch = items.RenderInstancedBatch;
pub const SkinStorage = items.SkinStorage;
pub const ShaderStorage = items.ShaderStorage;
pub const CoatStorage = items.CoatStorage;
pub const RenderQueues = items.RenderQueues;
pub const CulledMesh = items.CulledMesh;
pub const TransparentKind = items.TransparentKind;
pub const TransparentDrawEntry = items.TransparentDrawEntry;
pub const ParallelCullScratch = items.ParallelCullScratch;
pub const FrameCullContext = cull.FrameCullContext;
pub fn buildFrameQueues(ctx: FrameCullContext) void // build/frame.zig
pub fn sortRenderItems(...) void
pub fn sortTransparentDrawOrder(...) void
pub fn materialIsTransparent(m: ...) bool
pub fn materialIsCutout(m: ...) bool
pub fn materialIsDoubleSided(m: ...) bool
pub fn blendDescFor(m: ...) sg.BlendState
pub fn worldMatrixCached(cache_key: u64, mesh: *Mesh) Mat4
pub fn worldAABBCached(cache_key: u64, mesh: *Mesh) BoundingBox
```

`RenderMeshItem` (zero-dereference): `model: Mat4`, `distance_sq: f32`, `is_pbr`, `texture_id`, `mesh_index`, `draw_record: MaterialDrawRecord` (~120–240 B, замороженная копия факторов/UV/текстур), флаги `transparent`/`double_sided`/`is_decal`/`receive_shadows`, индексы снепшот-хранилищ `skin_index`/`shader_index`/`coat_index`, `morph_uniforms` + `morph_view`, самодостаточная геометрия (`vertex_buffer`, `index_buffer`, `index_count`, `index_type`, `base_vertex`, `is_u32`, `is_skinned`). `RenderInstancedBatch`: те же геометрия/запись + `instance_buffer`, `visible_instance_count`, `source_uid`/`source_mesh` для валидации patch'а. `RenderQueues`: обычные/прозрачные списки, `transparent_order`, инстанс-группы, `instance_matrices` scratch, `skin_storage`/`shader_storage`/`coat_storage` (по одному слоту на draw, резолв по индексу после всех реаллокаций).

`buildFrameQueues(ctx: FrameCullContext)`: растеризация окклюдеров → параллельный chunked cull (`build/parallel.zig`, детерминированный merge, OOM-fallback в сериальный) → сериальный инстанс-хвост. `FrameCullContext` — явные параметры (аллокатор, меши, `cache_key`, `view_proj`, `eye`, frustum/occlusion-флаги, маска, дефолтные материалы/текстуры, `gpu_retire`, `instances_prepared`).

Биннинг прозрачности: `TransparentDrawEntry` на группу (обычная + инстанс-группы) сортируется глобально back-to-front (`sortTransparentDrawOrder`); точные ничьи рвутся по `mesh_index` — сериальный и параллельный пути идентичны. Декали (`is_decal`) идут в прозрачную очередь после непрозрачной геометрии без записи глубины — без z-fighting.

### Построитель видов (`queue_builder.zig`)

```zig
pub const QueueBuildParams = struct {
    snap: *const SceneFrameSnapshot,
    cache_key: u64,
    stats: *SceneStats,
    eye: Vec3,
    sky_texture: ?CubeTexture,
    ibl_intensity: f32,
    instances_prepared: bool,
    instance_source: InstanceSource,
};
pub fn prepareViewQueues(scene: anytype, queues: *RenderQueues, cam_snap: CameraSnapshot, sky_texture: ?CubeTexture, ibl_intensity: f32, cache_key: u64, stats: *SceneStats, instances_prepared: bool, instance_source: InstanceSource) void
pub fn buildQueuesInto(scene: anytype, back: *FrameDrawSlot, params: QueueBuildParams) void
```

`buildQueuesInto` — общий строитель для game-билда (`.snap=&build_snapshot`, `.build_view`, ключ `seq|2^63`, `&build_stats`) и fallback-latch'а (слотовый снимок, `.published`, `frame_id`, `&stats`). Читает камеры только из `params.snap`. `prepareViewQueues`: `reset` → `buildFrameQueues` → сортировка opaque + `transparent_order`. Сложность: cull O(меши × глубина иерархии) через `worldMatrixCached` (каждый world-матрикс — не более одного раза на `cache_key`), сортировки O(n log n).

### Инстансинг (`instance_staging.zig`)

```zig
pub const InstanceStageContext = struct { ... };
pub fn stageInstancedMesh(sc: InstanceStageContext, mesh: *Mesh) void
pub const CpuStageResult = struct { ... };
pub fn stageSegmentCpu(...) CpuStageResult
pub const GpuStageContext = struct { ... };
pub fn stageInstancesGpu(gctx: GpuStageContext, mesh: *Mesh, matrices: []const Mat4, cpu: CpuStageResult) void
pub fn stageInstancesGpuState(gctx: GpuStageContext, st: *InstanceRenderState, matrices: []const Mat4, cpu: CpuStageResult) void
pub const CpuStageContext = struct { ... };
pub fn stageInstancesCpu(ctx: CpuStageContext, meshes: []const *Mesh, build_seq: u64) void
pub fn freezeStagedRecords(allocator: std.mem.Allocator, records: *..., meshes: []const *Mesh, seq: u64) void
pub fn stageInstancesLatch(ctx: ..., records: []const StagedInstanceRecord, scratch: ...) void
pub fn commitPublishedRecords(records: []const StagedInstanceRecord, meshes: []const *Mesh, frame_id: u64) void
pub fn stageInstances(sc: InstanceStageContext, meshes: []const *Mesh) void
// Тестовый шов P5 live-GPU gate:
pub fn testArmGrowthFailOnce() void
pub fn testLastInjectedFailId() u32
```

Четыре фазы: CPU-stage (сортировка сегментов по глубине, превью в `instance_preview` + штамп `build_seq`), freeze (слотовые `StagedInstanceRecord` в порядке mesh-листа), latch на context (GPU-заливка из слотовых записей + scratch, исходы зеркалятся в записи, live не читается/не пишется), commit на game при следующем билде (зеркала → `instance_render`, guard по `frame_id`, stale-поколения не коммитятся). Старый выросший буфер уходит в `GpuRetire`, не уничтожается инлайн. `stageInstances` — legacy инлайн-путь (CPU+GPU сразу, только context). P4-лимит: item держит 8-байтовые индексы вместо факторов (`coat_index` + нейтральный `CoatParams` для остальных).

### Пайплайны (`pipelines.zig`, `forward_pipelines.zig`)

```zig
pub const PipelineFamily = enum { standard, pbr, instanced, skinned_pbr, instanced_pbr };
pub fn forwardBaseDesc(shader: sg.Shader, sample_count: i32) sg.PipelineDesc
pub fn pipelineLayoutFor(family: PipelineFamily, desc: *sg.PipelineDesc) void
pub fn makePipelinePair(base: sg.PipelineDesc, opaque_u16: *sg.Pipeline, opaque_u32: *sg.Pipeline, blend_u16: *sg.Pipeline, blend_u32: *sg.Pipeline) void
pub fn cullOffDescFor(base: sg.PipelineDesc) sg.PipelineDesc
pub fn makeCullOffPair(base: sg.PipelineDesc, opaque_u16: *sg.Pipeline, opaque_u32: *sg.Pipeline, blend_u16: *sg.Pipeline, blend_u32: *sg.Pipeline) void
pub const DoubleSidedSourceShaders = struct { ... };
pub const DoubleSidedPipelines = struct {
    pub fn initFromShaders(self: *DoubleSidedPipelines, shaders: DoubleSidedSourceShaders, sample_count: i32) void
    pub fn deinit(self: *DoubleSidedPipelines) void
};
pub fn pipelineForRegularItem(scene: anytype, item: anytype) u32
pub fn pipelineForInstancedMesh(scene: anytype, is_pbr: bool, transparent: bool, is_u32: bool, double_sided: bool) u32
// forward_pipelines.zig:
pub fn shaderMaterialSlotKey(key: u64, sample_count: i32) u64
pub const ShaderMaterialSet = struct {
    pub fn pipelineFor(self: *const ShaderMaterialSet, transparent: bool, is_u32: bool, double_sided: bool) u32
};
pub const ShaderMaterialCache = struct {
    pub const max_entries = 32;
    pub fn lookup(self: *const ShaderMaterialCache, key: u64) ?*const ShaderMaterialSet
    pub fn getOrCreate(self: *ShaderMaterialCache, key: u64) ?*const ShaderMaterialSet
    pub fn deinit(self: *ShaderMaterialCache) void
};
pub const ForwardPipelines = struct {
    pub fn init() ForwardPipelines
    pub fn initSampled(sample_count: i32) ForwardPipelines
    pub fn initSampledWithShaders(sample_count: i32, family_shaders: DoubleSidedSourceShaders) ForwardPipelines
    pub fn deinit(self: *ForwardPipelines) void
    pub fn forRegularItem(self: *const ForwardPipelines, item: RenderMeshItem) u32
    pub fn forInstancedMesh(self: *const ForwardPipelines, is_pbr: bool, transparent: bool, is_u32: bool, double_sided: bool) u32
};
```

Фабрика: базовый дескриптор → раскладка по семейству → пары opaque/blend × u16/u32 (+ cull-off twin'ы для двусторонних). Выбор пайплайна только по снепшоту item'а (прозрачность, u32, двусторонность, PBR/skin), не по live-материалу. Draw минимизирует переключения (`current_pipeline_id`, счётчик `pipeline_switches`). Shader-материалы — ленивый кэш (≤32 записей) по `shaderMaterialSlotKey`.

### Юниформы (`uniforms.zig`)

```zig
pub const FrameContext = struct {
    view_proj: Mat4, eye: Vec3, sun_dir: Vec3, sun_color: Color3, sun_intensity: f32,
    directional_dir/color_int: [4][4]f32, cascades: [4]Mat4, light_counts: [4]f32,
    point_pos_range/point_color_int: [4][4]f32,
    spot_pos_range/spot_dir_inner/spot_color_outer/spot_intensity: [2][4]f32,
    spot_view_proj: [2]Mat4, spot_shadow_params: [2][4]f32,
    point_view_proj: [12]Mat4, point_shadow_params: [4][4]f32,
    area_center_int/area_right/area_up/area_color: [2][4]f32,
    clustered_params/clustered_viewport: [4]f32,
    uniforms_with_shadows/without_shadows: ?*const FrameUniforms,
};
pub const FrameUniforms = struct { eye_pos, light_dir, light_color: [4]f32, ... };
pub const ShadowState = struct { ... };
pub fn buildFrameUniforms(shadow: ShadowState, ctx: *const FrameContext) FrameUniforms
pub fn alphaCutoffFor(mat: ?Material) f32
```

`FrameContext` собирается раз на вид из staged снимка (см. `LightRig.FramePack` + `ShadowSystem.uniformState`); шейдеры standard/PBR/instanced делят имена/типы фрагментных юниформов.

### Исполнение (`draw.zig`, `view_render.zig`, `viewport_clear.zig`)

```zig
pub const FrameContext = uniforms.FrameContext; // реэкспорт
pub const Environment = struct { pipelines, stats, clustered_slot, ... };
pub fn drawRegularItem(env: *const Environment, item: RenderMeshItem, ctx: *const FrameContext, current_pipeline_id: *u32, skins: []const [MAX_BONES]Mat4, shaders: []const ShaderDrawSnapshot, coats: []const CoatParams) void
pub fn instancedDrawFlags(material: ?Material, is_decal: bool) struct { transparent: bool, double_sided: bool }
pub fn drawInstancedBatch(env: *const Environment, batch: RenderInstancedBatch, ctx: *const FrameContext, current_pipeline_id: *u32, coats: []const CoatParams) void
// view_render.zig:
pub fn renderSceneView(scene: anytype, cam_snap: CameraSnapshot, queues: *const RenderQueues, outline_items: []const OutlineDrawItem, outline_skins: []const [MAX_BONES]Mat4, samples: i32, snap: *const SceneFrameSnapshot, env: Environment, view_slot: usize) void
// viewport_clear.zig:
pub const ViewportClearPass = struct {
    pub fn deinit(self: *ViewportClearPass) void
    pub fn ensureResources(self: *ViewportClearPass, samples: i32) void
    pub fn clear(self: *ViewportClearPass, color: Color4, samples: i32) void
};
```

`drawRegularItem`: ранний выход при `index_count == 0` или невалидных буферах; hook-материалы (`shader_index`) — отдельный путь через `ShaderMaterialCache`; пайплайн — `forRegularItem(item)`, биндинги из item'а, скин — по `skin_index`, слои PBR — по `coat_index`. Каждый вид (`view_slot`: 0 = primary, 1+ = вторичные) перестраивает кластерные тайлы под свой слот (sokol one-update rule) и биндит свой слот (или общий dummy).

После shadow/probe/UI3D capture, до depth-prepass/main, opt-in refraction
снимает opaque-фон в half-resolution RT. Его clustered-слот (8) не пересекается
с камерами (0…7) или ручным RTT (9). Capture читает prepared queues под
consumer pin, исключает transparent/UI/outline/particles и использует linear
forward output. `renderReuse` сохраняет захват того же `frame_id`, без новых
uploads; multi-camera использует legacy transmission fallback. Ручной
`RenderTarget.renderPrimaryView` рисует весь primary view, не вызывает
рекурсивный refraction capture; см. [render-target.md](./render-target.md).

### Проекция (`projection.zig`, `project_cache.zig`)

```zig
pub fn camerasEqualForProjection(a: Camera, b: Camera) bool
pub const ProjectCache = struct {
    pub fn viewProjection(self: *ProjectCache, cam: Camera, w: f32, h: f32) Mat4
};
```

Кэш view-projection с инвалидацией по сравнению камер; используется очередями и cull'ом, чтобы не пересчитывать матрицы на каждый вид.

### Пост-эффекты (`postfx_stack.zig`, `msaa.zig`)

```zig
pub const PostFXStack = struct {
    pub fn init() PostFXStack
    pub fn deinit(self: *PostFXStack) void
    pub fn resizeAll(self: *PostFXStack, width: i32, height: i32) void
    pub fn destroyMsaaDepth(self: *PostFXStack) void
    pub fn renderMsaaDepthPrepass(...) void
    pub fn depthSampleView(self: *const PostFXStack, prepass_active: bool) sg.View
    pub fn taaReset(self: *PostFXStack) void
    pub fn beginMainPass(self: *PostFXStack, main_pass_action: sg.PassAction, post_enabled: bool, samples: i32, width: i32, height: i32) void
    pub fn renderOutline(...) void
    pub fn renderOutlineExplicit(...) void
    pub fn renderOutlineItems(...) void
    pub const ChainParams = struct { post, ssao, camera, aspect, view_proj, eye, sun_dir, sun_color, default_white_view, stats, main_samples: i32 = 1, msaa_depth_prepass: bool = false, highlight_items, highlight_viewport, shaft_shadow_view, shaft_cascades, shaft_splits, shaft_shadow_bias, shadows_enabled, ui: ?*const UiFrame };
    pub fn renderChain(self: *PostFXStack, params: ChainParams, cur_w: i32, cur_h: i32) void
};
// msaa.zig:
pub const valid_sample_counts = [_]i32{ 1, 2, 4, 8 };
pub fn maxSamplesForBackend(backend: sg.Backend) i32
pub fn clampSampleCount(backend: sg.Backend, requested: i32) i32
pub const Inputs = struct { post_enabled, formats_msaa_capable, backend };
pub fn effectiveSampleCount(requested: i32, in: Inputs) i32
pub fn depthEffectsActive(post_enabled: bool, ssao_enabled: bool, ssao_debug: bool, ssr_enabled: bool, dof_enabled: bool, fog_enabled: bool) bool
pub fn depthPrepassActive(post_enabled: bool, gate: bool, main_samples: i32) bool
pub fn suppressDepthEffects(main_samples: i32, gate: bool) bool
pub fn needsResolveAttachment(sample_count: i32) bool
pub const WarnOnce = struct { pub fn warn(self: *WarnOnce, comptime fmt: []const u8, args: anytype) bool };
pub fn mainTargetFormatsMsaaCapable() bool
```

Проходы: 1.7 MSAA depth-prepass → main → 2.5 SSAO → 2.75 bloom → 2.8 glow → 2.85 highlight → 2.9 volumetric shafts → 3 composite + UI overlay. Depth-эффекты несовместимы с MSAA-main и подавляются (one-shot warning), если не активен prepass-gate (`Scene.msaa_depth_prepass`). MSAA-twin outline-пасса создаётся лениво; TAA хранит `prev_view_proj` для репроекции. Композит работает только при включённом посте; остальные проходы тогда лишь обновляют входы.

### Тени (`shadow_system.zig`, `cascades.zig`, `shadow_pcss.zig`)

```zig
// shadow_system.zig:
pub const ShadowSystem = struct {
    pub fn init(allocator: std.mem.Allocator) ShadowSystem
    pub fn deinit(self: *ShadowSystem) void
    pub fn computeCascades(self: *ShadowSystem, camera: Camera, aspect: f32, norm_light_dir: Vec3) [4]Mat4
    pub fn uniformState(self: *const ShadowSystem, ground_color: Color3) ShadowState
};
// cascades.zig:
pub fn computeCascades(camera: Camera, aspect: f32, norm_light_dir: Vec3, splits: [4]f32) [4]Mat4
// shadow_pcss.zig:
pub const blocker_sample_count: u32 = 12;
pub const pcf_taps_near: u32 = 16;
pub const pcf_taps_far: u32 = 8;
pub const default_light_size: f32 = 0.02;
pub const default_blocker_radius: f32 = 0.01;
pub const default_min_penumbra: f32 = 0.0005;
pub const default_max_penumbra: f32 = 0.01;
pub const no_blockers: f32 = -1.0;
pub const min_blocker_depth: f32 = 0.0001;
pub fn isEnabled(flag: f32) bool
pub fn pcfTapCount(cascade_idx: u32) u32
pub fn tapWeight(taps: u32) f32
pub fn averageBlocker(sum: f32, count: u32) ?f32
pub fn penumbraRadius(receiver_depth: f32, blocker_avg: f32, light_size: f32, min_penumbra: f32, max_penumbra: f32) f32
pub fn resolveLit(blocker_avg: ?f32, pcf_lit: f32) f32
```

Солнце: CSM-атлас из 4 каскадов (`computeCascades` с фиксированными сплитами внутри `ShadowSystem`); споты — до 2 теневых (view-proj + bias в `FramePack`); поинты — до 2 теневых слотов по 6 граней (`point_view_proj[12]`). PCSS: поиск блокеров (12 сэмплов) → средний блокер → радиус полутени → PCF (16 тапов вблизи, 8 вдали); нет блокеров — чистый PCF. Важно для порядка: стейджинг инстансов идёт до `ShadowPass.prepare`, иначе тени отстают на кадр.

### Зонды в кадре (`probe_render.zig`)

```zig
pub fn captureDirtyProbes(scene: anytype, snap: *const SceneFrameSnapshot) void
pub fn renderProbeFace(scene: anytype, ...) void
pub fn runProbePrefilter(scene: anytype, index: usize) void
```

Не более одного dirty-зонда за кадр (`max_captures_per_frame` дисциплина): рендер 6 граней → prefilter mip-цепочки; детали — в `./scene-layers.md`.

## Потоки и владение

- Build очередей — sg-free при `instances_prepared=true`: может идти на game (staged-путь, источник `.build_view`) или на context (fallback, источник `.published`). Live-чтения (TRS, материалы, флаги cull'а) — только под исключением update-vs-prepare.
- Latch/upload/draw — строго context: `stageInstancesLatch`, `flushSlotUploads`, `patchInstanceRefs`, `renderSceneView`, `draw*`, `renderChain`. Draw читает только front-слот через const-пейлоады либо под pin'ом.
- Снепшот-правило P4: draw-путь никогда не разыменовывает live-указатели; `skin_index`/`shader_index`/`coat_index` резолвятся в хранилищах той же очереди после всех реаллокаций.
- GPU-хэндлы в очередях — заимствованные (P3/эпохи retire); очереди их не уничтожают и не дублируют.
- Кластерные буферы: каждый вид обновляет только свой слот; данные света — в storage-буферах, в юниформах — только дескриптор сетки.

## Ошибки и краевые случаи

- OOM в построении очередей: параллельный путь падает в сериальный; записи без staged-записей (OOM-пропуск, пост-билд меши) патчатся в невидимые — без частичной публикации.
- `forRegularItem` → 0: item пропускается (нет пайплайна); невалидные буферы — пропуск без sg-ошибок.
- Пустые очереди/нет камеры: когерентный пустой кадр; `eye` — ноль при `has_camera=false`.
- MSAA на бэкенде без поддержки — clamp через `clampSampleCount` (GL — 1x/ограниченно); depth-эффекты подавляются с предупреждением once.
- Тени без света-кастера: слоты занулены, шейдеры выходят досрочно с нулевой стоимостью.
- Несовпадение `source_uid` при patch — fail-closed в ноль инстансов (невидимо), ловится commit-guard'ом.

## Производительность

- Cull O(n) с кэшем матриц O(depth) суммарно; сортировки O(n log n) — opaque по материалу/глубине (минимум `pipeline_switches`), прозрачные глобально back-to-front.
- Параллельный cull через `jobs.global` (чанки + детерминированный merge); порог воркеров для скелетов — см. `./scene-layers.md`.
- Пайплайны/шейдеры/таргеты пост-стека — retained между кадрами (`clearRetainingCapacity`/`resizeAll` только при смене размера); per-frame churn нет.
- Счётчики `SceneStats` (draw calls, pipeline switches, culled counts) — см. `./profiler.md`.
- Кластерный тайлинг: сетка 64px (`tile_size_px`), per-tile списки света — отсечение внеэкранного света до шейдера.

## Смотрите также

- `./frame-pipeline.md` — слоты, stage/latch/commit, режимы prepare.
- `./scene.md` — снимки (`SceneFrameSnapshot`), камеры, виды.
- `./lights.md` — источники света, `LightRig`, тени света.
- `./material.md` — `MaterialDrawRecord`, слои PBR, shader-материалы.
- `./shadows.md` — проходы теней (если есть; иначе `./passes.md`).
- `./postprocess.md` — проходы SSAO/bloom/glow/highlight/volumetric.
- `./scene-layers.md` — зонды, небо, 3D-GUI в кадре.
