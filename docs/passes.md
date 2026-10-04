# Рендер-пассы

> Путь: src/agate/passes/ · Импорт: agate.passes (через mod.zig) · Потоки: render-поток (GPU-команды внутри sg-пассов).

## Что это

Каталог `passes/` — отдельные GPU-пассы движка: каждый пасс владеет своими таргетами/пайплайнами и умеет `init` / `resize` / `render` / `deinit`. Оркестрацией занимается сцена/`PostFXStack` (см. `./frame-pipeline.md`): тени → сцена → SSAO → bloom/glow/highlight/volumetric → composite-постпроцесс → outline → частицы → skybox/debug. Здесь — что делает каждый пасс, какие ресурсы держит и как включается.

Фасад `passes/mod.zig` реэкспортирует все пассы и хелперы (`ShadowPass`, `BloomPass`, `GlowPass`, `HighlightPass` + `HighlightDrawItem/Result`, `OutlinePass` + `OutlineDrawItem`, `SSAOPass`, `PostProcessPass`, `ParticlePass`, `SkyboxPass`, `VolumetricPass`, `DebugPass`, `MsaaDepthPass`, point/spot-инфо теней). Теневой пасс сам — фасад над `passes/shadow/` (types/core/binning/prepare/buckets/csm/spot/point).

Общий контракт пасса: `init()` (дефолтные таргеты нулевые, пайплайны создаются лениво или сразу), `resize(w, h)` перед первым `render` нового разрешения, `render(...)` открывает свои sg-пассы и не трогает чужие, `deinit()` освобождает GPU-ресурсы. Все `render` — no-op с нулевым результатом при неинициализированных пайплайнах/нулевых вьюхах (защиты `if (...id == 0) return`), а не assert'ы.

## Быстрый старт

```zig
const passes = @import("passes"); // agate.passes

var bloom = passes.BloomPass.init();
var glow = passes.GlowPass.init();
var highlight = passes.HighlightPass.init();
var ssao = passes.SSAOPass.init();
var postfx = passes.PostProcessPass.init();
defer bloom.deinit(); defer glow.deinit(); // + остальные

// При смене разрешения (или перед первым кадром):
bloom.resize(w, h); glow.resize(w, h); highlight.resize(w, h);
ssao.resize(w, h); postfx.resize(w, h, sample_count);

// В кадре (упрощённо; реальный порядок — в PostFXStack.renderChain):
shadow.render(...);          // CSM + spot + point атласы
main.render(...);            // сцена (не пасс этого каталога)
const ao = ssao.render(...); // опционально
const bloom_tex = bloom.render(hdr_view, threshold, mips, w, h);
const glow_tex = glow.render(hdr_view, threshold, intensity, ...);
postfx.setBloomTexture(bloom_tex);
postfx.setGlowTexture(glow_tex);
postfx.render(...);          // composite: postprocess.glsl
```

Включение — флагами сцены/постпроцесса (`shadows_enabled`, `PostProcessOptions.*_enabled`), а не вызовами пассов напрямую: незадействованный пасс просто не вызывается и не тратит GPU.

## API

### ShadowPass (`passes/shadow_pass.zig` + `passes/shadow/`)

```zig
pub const ShadowPass = core.ShadowPass; // owns ShadowDrawItem + PreparedShadowDraws
pub const SHADOW_ATLAS_SIZE / SPOT_SHADOW_MAP_WIDTH / SPOT_SHADOW_MAP_HEIGHT / SPOT_SHADOW_RES = ...;
pub const POINT_SHADOW_SLOTS / POINT_SHADOW_FACES / POINT_SHADOW_RES / POINT_SHADOW_MAP_WIDTH / POINT_SHADOW_MAP_HEIGHT = ...;
pub const SpotShadowRenderInfo / PointShadowRenderInfo = ...;
pub fn pointFaceForDir(dir: Vec3) u32
pub fn pointTileOrigin(slot: u32, face: u32) [2]u32
// core.ShadowPass: init/deinit, prepare/renderPrepared/render, binMeshes,
// renderBuckets, renderCsm/renderSpot/renderPoint
```

Что рендерит: три атласа глубины — CSM (4 каскадные плитки направленного/солнечного света), spot-атлас (тайлы прожекторов), point-атлас (6 граней × слоты точечных). `binMeshes` раскладывает меши по бакетам пайплайнов (serial + parallel), `prepareInto` строит render-owned снапшот отрисовок, `renderBuckets` — per-item каллинг, группировка по пайплайнам, упаковка юниформ. Ресурсы: атласные `sg.Image` + depth-пайплайны; таргеты — переиспользуемые, ресайз по `SHADOW_ATLAS_SIZE`. Включение: `shadows_enabled` сцены; шафты (`VolumetricPass`) требуют живой CSM (см. `./postprocess.md`).

### BloomPass (`passes/bloom_pass.zig`)

```zig
pub const BloomPass = struct {
    pub fn bloomPixelFormat() sg.PixelFormat // RGBA16F, обязательный
    pub fn init() BloomPass
    pub fn resize(self: *BloomPass, width: i32, height: i32) void
    pub fn render(self: *BloomPass, src_tex: sg.View, threshold: f32, radius: f32, mip_count: u32, base_w: i32, base_h: i32) sg.View
    pub fn deinit(self: *BloomPass) void
};
```

Что рендерит: единую HDR-пирамиду: down-цепочку (`bloom_down.glsl`: первый уровень — bright-pass с `threshold`, глубже — Karis-average) + up-цепочку (`bloom_up.glsl`, tent-апсемпл с `radius` в текселях coarse-мипа [0,16]) по `mip_count` (`clampBloomMips` [3,7]). Возвращает view верхнего мипа для composite (`PostProcessPass.setBloomTexture`). Таргеты: `down_images[]`/`up` половинного разрешения лесенкой; формат — `bloomPixelFormat()` (обязательный RGBA16F). `render` при нулевых пайплайнах/вьюхах возвращает пустой view. Включение: `bloom_enabled` (inline-пути нет).

### GlowPass (`passes/glow_pass.zig`)

```zig
pub const GlowPass = struct {
    pub fn glowPixelFormat() sg.PixelFormat
    pub fn glowBytesPerPixel() usize
    pub fn targetBytes(width: i32, height: i32, bytes_per_pixel: usize) usize
    pub fn init() GlowPass
    pub fn resize(self: *GlowPass, width: i32, height: i32) void
    pub fn render(self: *GlowPass, src: sg.View, threshold: f32, intensity: f32, ...) sg.View
    pub fn deinit(self: *GlowPass) void
};
```

Что рендерит: luminance-extract (`glow_extract.glsl`, порог `threshold`) + сепарабельный blur (`glow_blur.glsl`, `GLOW_BLUR_TAPS`, `GLOW_PASS_DRAWS` проходов) + возврат view для аддитивного composite после bloom. `targetBytes` — оценка видеопамяти таргетов (для профайлера/бюджетов). Включение: `glow_enabled` (default OFF — composite bit-identical без пасса).

### HighlightPass (`passes/highlight_pass.zig`)

```zig
pub const HighlightDrawItem = struct { ... }; // mesh + options + source id
pub fn makeHighlightDrawItem(mesh: *Mesh, options: HighlightOptions, source_mesh: u32) ?HighlightDrawItem
pub fn highlightMaskColor(item: HighlightDrawItem) [4]f32
pub fn highlightFrameSigma(items: []const HighlightDrawItem) f32
pub fn highlightMaskViewport(rect: Viewport.PixelRect, base_w: i32, base_h: i32) Viewport.PixelRect
pub fn configureHighlightMaskDesc(desc: *sg.PipelineDesc) void
pub const HighlightResult = struct { ... }; // mask view + sigma
pub const HighlightPass = struct {
    pub fn init() HighlightPass
    pub fn resize(self: *HighlightPass, width: i32, height: i32) void
    pub fn render(...) HighlightResult
    pub fn deinit(self: *HighlightPass) void
};
```

Что рендерит: маску выбранных объектов (id-цвета через `highlightMaskColor`, `configureHighlightMaskDesc` задаёт desc масочного пайплайна) + inner glow с `highlightFrameSigma`. `makeHighlightDrawItem` возвращает `null` для неподходящих мешей (невалидный/невидимый — пропуск, не ошибка). Результат скармливается composite (`setHighlightTexture`/`setHighlightMaskTexture`). Включение: наличие highlight-айтемы + `highlightActive`.

### OutlinePass (`passes/outline_pass.zig`)

```zig
pub fn clampWidthPx(width_px: f32) f32
pub fn ndcExpandForViewport(width_px: f32, viewport_w: f32, viewport_h: f32) [2]f32
pub fn expandVertex(pos: [3]f32, normal: [3]f32, scale: f32) [3]f32
pub fn outlineParamsFor(width_px: f32) [4]f32
pub fn shouldOutlineMesh(mesh: *const Mesh) bool
pub const CutoutInfo = struct { ... };
pub fn cutoutInfoFor(mesh: *const Mesh) ?CutoutInfo
pub fn configureOutlineDesc / configureOutlineInstDesc / configureOutlineCutoutDesc / configureOutlineSkinnedDesc(desc: *sg.PipelineDesc) void
pub const OutlineDrawItem = struct { ... }; // render-owned индексы, НЕ живые указатели
pub fn makeOutlineDrawItem(...) ?OutlineDrawItem
pub const OutlinePass = struct {
    pub fn init() OutlinePass
    pub fn initSampled(sample_count: i32) OutlinePass
    pub fn resize(w: i32, h: i32) void
    pub fn renderItems(...) void
    pub fn render(self: *OutlinePass, view_proj: Mat4, camera_pos: Vec3, meshes: []const *Mesh, color: Color4, width_px: f32) void
    pub fn deinit(self: *OutlinePass) void
};
```

Что рендерит: контур выбранного (inflated backface-shell: `expandVertex` по нормали, `outlineParamsFor` пакует ширину/NDC-раскрытие). Четыре desc-конфигуратора — обычный/инстансинг/cutout/скиннинг пайплайны. Low-level API несёт render-owned индексы вместо живых указателей (осознанное изменение, зафиксировано в `mod.zig`; стабильные точки — `Scene.buildPreparedFrame` / `beginStagedPrepare`+`finishStagedPrepare` / `render` и immediate `OutlinePass.render`). P4-контракт: `renderItems` для подготовленных айтемов, `render` — immediate по мешам. Включение: наличие outline-цели + ширина > 0 после `clampWidthPx`.

### ParticlePass (`passes/particle_pass.zig`)

```zig
pub const ParticlePass = struct {
    pub fn init() ParticlePass
    pub fn initSampled(sample_count: i32) ParticlePass
    pub fn render(...) void
    pub fn renderDraws(...) void
    pub fn deinit(self: *ParticlePass) void
};
```

Что рендерит: билборды/инстансы GPU-частиц (`particle.glsl`; compute-состояние — `particle_compute.glsl`, см. `./particles.md`). `render` — полный проход по системе, `renderDraws` — по предподготовленным draws (render-owned путь кадра). Таргетов своих нет (рисует в текущий framebuffer/таргет сцены). Включение: живая particle-система в сцене.

### PostProcessPass (`passes/postprocess_pass.zig`)

```zig
pub const PostProcessPass = struct {
    pub fn init() PostProcessPass
    pub fn taaReset(self: *PostProcessPass) void
    pub fn ensureTaaHistory(self: *PostProcessPass, width: i32, height: i32) bool
    pub fn taaReadView(self: *const PostProcessPass) sg.View
    pub fn taaWriteAttView(self: *const PostProcessPass) sg.View
    pub fn resize(self: *PostProcessPass, width: i32, height: i32, sample_count: i32) void
    pub fn depthSampleView(self: *const PostProcessPass) sg.View
    pub fn render(...) void  // composite: postprocess.glsl, ChainParams.post in
    pub fn setBloomTexture(self: *PostProcessPass, view: sg.View) void
    pub fn setGlowTexture / setHighlightTexture / setHighlightMaskTexture / setShaftTexture(...) void
    pub fn deinit(self: *PostProcessPass) void
};
```

Что рендерит: финальный fullscreen composite (`postprocess.glsl`): обязательный output pass (exposure+tonemap+display transfer выполняется даже при `enabled = false`) плюс опциональные bloom-pyramid/glow/highlight/shafts подмесы → grading/LUT → vignette → FXAA/TAA → sharpen/grain (порядок — в `./postprocess.md`). Держит fullscreen-квад, composite-пайплайн, depth-вью для DOF/SSR (`depthSampleView`) и два TAA history-таргета (`ensureTaaHistory` создаёт/пересоздаёт при ресайзе, `taaReset` сбрасывает после `taa_camera_cut`). `set*Texture` привязывают выходы других пассов (пустой view = ветвь скипается). Сложность — O(пиксели × включённые эффекты).

### SSAOPass (`passes/ssao_pass.zig`)

```zig
pub const SSAOPass = struct {
    pub fn init() SSAOPass
    pub fn resize(self: *SSAOPass, width: i32, height: i32) void
    pub fn render(...) sg.View // ssao.glsl + ssao_blur.glsl
    pub fn deinit(self: *SSAOPass) void
};
```

Что рендерит: raw AO по глубине/нормалям (`ssao.glsl`) + blur (`ssao_blur.glsl`), возврат view AO для forward/composite. Таргеты — half-res внутри; ресайз пересоздаёт. Включение: флаг SSAO сцены (конфиг — см. `./render-pipeline.md`; в `PostProcessOptions` не входит осознанно).

### MsaaDepthPass (`passes/msaa_depth_pass.zig`)

```zig
pub const MsaaDepthPass = struct {
    pub fn init() MsaaDepthPass
    pub fn render(...) void
    pub fn deinit(self: *MsaaDepthPass) void
};
pub const msaa_depth_pass = @import("msaa_depth_pass.zig"); // доп. хелперы
```

Что рендерит: depth-prepass MSAA-таргета (`msaa_depth.glsl` — depth-only близнец `shadow.glsl`: идентичная растеризация, камерные матрицы, свой bias). Нужен для корректного resolve/семплирования глубины при включённом MSAA (DOF/SSR/TAA-depth читают отсюда через `PostProcessPass.depthSampleView`). Включение: `sample_count > 1`.

### DebugPass (`passes/debug_pass.zig`)

```zig
pub const DebugPass = struct {
    pub fn init(allocator: std.mem.Allocator) !DebugPass
    pub fn initSampled(allocator: std.mem.Allocator, sample_count: i32) !DebugPass
    pub fn render(self: *DebugPass, view_proj: Mat4, lines: []const DebugLine) void
    pub fn deinit(self: *DebugPass) void
};
```

Что рендерит: debug-линии/боксы поверх кадра (`debug.glsl`, `DebugLine` — пара точек + цвет). Единственный пасс с аллокатором в `init` (CPU-буфер линий). Включение: непустой `lines` (дебаг сборки/флаги сцены). Не влияет на продуктовый рендер при пустом входе.

### SkyboxPass (`passes/skybox_pass.zig`)

```zig
pub const SkyboxPass = struct {
    pub fn init() SkyboxPass
    pub fn initSampled(sample_count: i32) SkyboxPass
    pub fn render(self: *SkyboxPass, camera: Camera, aspect: f32, cube_tex: CubeTexture, exposure: f32) void
    pub fn renderMatrices(self: *SkyboxPass, rot_view: Mat4, proj: Mat4, cube_tex: CubeTexture, exposure: f32) void
    pub fn deinit(self: *SkyboxPass) void
};
```

Что рендерит: кубмапу фона (`skybox.glsl`) с экспозицией; `renderMatrices` — вариант без камеры (только rot-view + proj, для cubemap-проб/рефлексий). Рисует первым (дальняя плоскость, depth-write off) или как fallback при отсутствии skybox-геометрии. Включение: `skybox_enabled` + валидная кубмапа.

### VolumetricPass (`passes/volumetric_pass.zig`)

```zig
pub const VolumetricPass = struct {
    pub fn shaftPixelFormat() sg.PixelFormat
    pub fn shaftBytesPerPixel() usize
    pub fn targetBytes(base_w: i32, base_h: i32, bytes_per_pixel: usize, res: pp.ShaftResolution) usize
    pub fn init() VolumetricPass
    pub fn resize(self: *VolumetricPass, base_w: i32, base_h: i32, res: pp.ShaftResolution) void
    pub const RenderArgs = struct { ... }; // hdr/depth/csm views, матрицы, sun, params
    pub fn render(self: *VolumetricPass, args: RenderArgs) sg.View
    pub fn deinit(self: *VolumetricPass) void
};
```

Что рендерит: low-res raymarch (`volumetric_raymarch.glsl`) + bilateral blur (`volumetric_blur.glsl`) по `RenderArgs` (требует живой CSM — иначе пустой view). `targetBytes` — бюджет видеопамяти под `half`/`quarter`. Результат — в `PostProcessPass.setShaftTexture`. Включение: `shaftActive` (shafts enabled + shadows enabled).

## Потоки и владение

- Все пассы принадлежат render-потоку (или главному в single-thread конфигурации); таргеты/пайплайны создаются в `init`/`resize`, уничтожаются в `deinit`. Двойной `deinit` небезопасен; `resize` до `init` — no-op по нулевым хэндлам.
- `render*` не владеют входными view (заимствуют на вызов); возвращаемые view валидны до следующего `resize`/`deinit` пасса-владельца.
- `OutlineDrawItem`/`HighlightDrawItem`/`PreparedShadowDraws` — render-owned индексы/снапшоты, не указатели в сцену: безопасно пересекают границу update→render.
- `RenderArgs` (volumetric) и `ChainParams.post` (composite) — plain structs, копируются в кадр; LUT-хэндл внутри — by-value (см. `./postprocess.md`).

## Ошибки и краевые случаи

- Нулевые хэндлы (`pipeline.id == 0`, `view.id == 0`, `w/h <= 0`) — тихий no-op/пустой view, не assert. Это позволяет вызывать пассы до ленивой инициализации пайплайнов.
- `resize` с тем же разрешением — дешёвый (пересоздания нет); с нулевым — таргеты инвалидируются до следующего валидного ресайза.
- Потерянные входы composite (`setBloomTexture` не вызван, а `bloom_enabled` включён) — ветвь composite работает с пустым view как с чёрным полем, а не падает; визуальный баг, не краш — проверяйте wiring при добавлении пассов.
- `ensureTaaHistory` возвращает `false` при неудаче аллокации таргетов — вызывающий должен идти no-TAA путём в этом кадре.
- Point-атлас переполнен (слотов `POINT_SHADOW_SLOTS` не хватило) — дальние точечные без теней в этом кадре (деградация, не ошибка).
- `makeOutlineDrawItem`/`makeHighlightDrawItem` → `null` — штатный пропуск (скрытый/невалидный меш), не ошибка вызывающего.

## Производительность

- Порядок цены пассов (типично): volumetric raymarch > bloom-пирамида > SSAO > glow-blur > composite > shadow-атласы (зависят от числа кастеров) > outline/highlight-маски > skybox/debug (пренебрежимо).
- Таргеты bloom/glow/shaft/SSAO — пониженного разрешения (лесенка мипов, half/quarter) — главный рычаг цены; composite и TAA-history — полного (их удешевить нельзя без потери качества).
- `targetBytes` (glow/volumetric) + `bloomMipSize`/`shaftTargetSize` (см. `./postprocess.md`) — считайте бюджет видеопамяти до включения всего сразу.
- Shadow-бининг параллельный (`binMeshes`) — масштабируется с числом мешей; `renderBuckets` группирует по пайплайнам, минимизируя switches (сверяйтесь с `pipeline_switches` в `./profiler.md`).

## Смотрите также

- `./frame-pipeline.md` — порядок пассов в кадре, `PostFXStack.renderChain`, `ChainParams`.
- `./postprocess.md` — параметры эффектов, клампы, порядок стека composite.
- `./shaders.md` — исходники `.glsl`, slang-леги, include-чанки.
- `./lights.md` — CSM/spot/point данные для ShadowPass.
- `./particles.md` — CPU-сторона частиц, `renderDraws`-путь.
- `./render-pipeline.md` — таргеты, MSAA, семплирование глубины.
- `./profiler.md` — `shadow_ms`/`main_ms`/`post_ms`, `draw_calls`, `pipeline_switches`.
