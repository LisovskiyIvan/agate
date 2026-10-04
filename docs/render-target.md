# Render Target (RTT v1)

> Путь: src/agate/render_target.zig · Импорт: agate.RenderTarget, agate.RenderTargetDesc, agate.render_target (root.zig) · Потоки: создание/ресайз/begin/end/render — только на context-потоке; валидация/оценки — на любом

## Что это

Настоящий render-to-texture примитив: 2D color attachment (+ опциональный depth) с sampling views, в который можно отрендерить сцену или clear-цвет и потом сэмплить результат в последующих проходах. Фундамент для refraction (преломление через transmission-сэмпл), зеркал, динамических превью и любых эффектов «отрендерить → использовать как текстуру».

Это реальные sokol-ресурсы (`sg.Image` color/depth + color-attachment/texture views + sampler), а не CPU-буфер и не метаданные. Создание — по образцу проб/постпроцесса (`makeImage` + `makeView` с `errdefer`-откатом), уничтожение — немедленное на context-потоке, как у всех pass-owned таргетов (`PostProcessPass`, кубы проб).

## Быстрый старт

```zig
const agate = @import("agate");

// Создание (context-поток, после sg.setup).
var rt = try agate.RenderTarget.create(.{ .width = 512, .height = 512 });
defer rt.deinit();

// Вариант: без глубины (только цвет) или с MSAA.
var rt_nodepth = try agate.RenderTarget.create(.{
    .width = 256, .height = 256, .depth_enabled = false,
});

// 1. Отрендерить подготовленный primary view в таргет (между
//    staged build + begin/finish и Scene.render, пока prepared draws
//    консьюмабельны). Неподдержанный захват — это ошибка
//    (CaptureError), а не тихий пропуск: fallback — пропустить
//    transmission-кадр и сэмплить прошлый захват.
rt.renderPrimaryView(&scene, .{ .clear_color = agate.Color4.new(0, 0, 0, 1) }) catch |err| {
    std.log.warn("rtt capture skipped: {s}", .{@errorName(err)});
};

// 2. Использовать в своём проходе как обычную текстуру:
pass_bind.views[SLOT_scene_tex] = rt.sampleView();
pass_bind.samplers[SLOT_smp] = rt.sampleSampler();
// глубина (1x таргеты с depth): rt.depthSampleView()

// 3. Ресайз при смене разрешения (старый контент переживает ошибку):
_ = rt.resize(new_w, new_h);

// 4. Заимствованный Texture для слотов, говорящих на Texture:
const borrowed = rt.asTexture(); // owns_handles=false; deinit borrow — no-op
```

## API

### `RenderTargetDesc` / `create` / `deinit`

```zig
pub const RenderTargetDesc = struct {
    width: u32 = 256,
    height: u32 = 256,
    sample_count: i32 = 1,             // MSAA-запрос; см. ниже
    color_format: sg.PixelFormat = .DEFAULT, // DEFAULT = swapchain color (fallback BGRA8)
    depth_format: sg.PixelFormat = .DEFAULT, // DEFAULT = DEPTH; .NONE = без глубины
    depth_enabled: bool = true,
    min_filter: sg.Filter = .LINEAR,  // семплирование захваченного цвета
    mag_filter: sg.Filter = .LINEAR,
};
pub fn create(desc: RenderTargetDesc) CreateError!RenderTarget;
pub fn deinit(self: *RenderTarget) void; // немедленное, context-поток; headless-safe
```

Ошибки: `InvalidDimensions` (нулевой размер), `ImageTooLarge` (переполнение `w*h` или превышение живого `max_image_size_2d`), `NoContext` (нет `sg`), `TargetCreationFailed` (откат уже выполнен; проверяется состояние ресурса, не только ненулевой id), `UnsupportedColorFormat` (формат не рендерится/не сэмплируется или не поддерживает запрошенную фильтрацию), `UnsupportedDepthFormat` (не depth-формат).

MSAA: запрос прогоняется через ту же политику, что у main-таргета (`msaa.effectiveSampleCount`): неподдерживаемый уровень — явная ошибка запуска/panic на старте, а не тихая смена контракта. Обязательные sample/filter/render/blend-caps проверяются на старте. Фактический уровень — в `sample_count`. При `samples > 1` таргет несёт resolve-пару (цвет резолвится в конце прохода, сэмплинг читает 1x-копию), а depth texture view отсутствует (в sokol нет depth resolve — то же правило, что у main-таргета, см. `./render-pipeline.md` и `scene/msaa.zig`).

### Проходы: `begin` / `end` / `clear`

```zig
pub fn begin(self: *RenderTarget, clear_color: Color4, depth_clear: f32) bool;
pub fn end(self: *RenderTarget) void; // ровно один на успешный begin; no-op без открытого pass
pub fn clear(self: *RenderTarget, clear_color: Color4, depth_clear: f32) bool;
pub fn isCapturing(self) bool; // true между begin и end
```

`begin` открывает pass с CLEAR/STORE и ставит viewport/scissor под размер таргета. Возвращает `false` fail-closed (таргет невалиден, нет `sg`, pass уже открыт) — тогда НЕ вызывать `end` и не рисовать. `end` без открытого pass — безопасный no-op. После `end` viewport восстанавливает вызывающий код (в `renderPrimaryView` восстановление встроено: возврат к `snap.screen_w/h`). `resize` при открытом pass возвращает `false`; `deinit` при открытом pass — assert в Debug/ReleaseSafe.

### Сэмплинг и заимствование

```zig
pub fn sampleView(self) sg.View;      // цвет для сэмплирования (resolve при MSAA)
pub fn sampleSampler(self) sg.Sampler;
pub fn depthSampleView(self) sg.View; // пусто без depth, при MSAA или при открытом pass
pub fn colorAttachmentView(self) sg.View; // для сборки кастомных проходов
pub fn depthAttachmentView(self) sg.View;
pub fn asTexture(self) Texture;       // borrowed; owns_handles=false
```

Пока pass открыт (`isCapturing()`), sampling views и `asTexture` отдают пустые хендлы. `renderPrimaryView` до открытия прохода проверяет staged material/shader/morph/particle views, включая альтернативные views того же image: feedback возвращает `FeedbackLoop`. Для ручного `begin` отсутствие feedback обеспечивает вызывающий код. Сохранённый borrow предыдущего кадра **не** делает чтение того же attachment безопасным: используйте отдельную display-сцену/проход либо два разных ping-pong таргета.

`asTexture` — для обычных слотов материалов. При MSAA содержит resolve image, не multisampled attachment. Владение не передаётся: `Texture.deinit` на borrow — no-op. `resize`/`deinit` таргета инвалидируют сохранённые хендлы: ресайз, обновление borrow и новый prepare должны произойти **до** render; нельзя replay'ить старый prepared draw с уничтоженным view.

### Сцена в таргет: `renderPrimaryView`

```zig
pub const SceneRenderOptions = struct {
    clear_color: Color4 = ...,
    depth_clear: f32 = 1.0,
    view_slot: usize = clustered_lights.RTT_VIEW_SLOT, // сейчас 9
};
pub const CaptureError = error{
    InvalidTarget, NoContext, PassAlreadyOpen, NoConsumableFrame,
    InvalidViewSlot, NoCamera, IncompatibleColorFormat, DepthRequired,
    IncompatibleDepthFormat, PassBeginFailed, FeedbackLoop,
};
pub fn renderPrimaryView(self: *RenderTarget, scene: anytype, opts: SceneRenderOptions) CaptureError!void;
```

Использует `Scene.renderSceneView`: те же opaque/transparent очереди, небо, частицы и дебаг-линии из staged-снапшота последнего prepare. Захват всегда primary view, unjittered; viewport на весь RT, projection/aspect сохранены из снимка. Consumer pin (`pinFront`/`unpin`) защищает кадр от параллельного producer; без `hasConsumableFrame()` возвращается ошибка.

Неподдержанный захват — ошибки, а не проза:

| Ошибка | Когда |
|---|---|
| `InvalidTarget` | таргет не создан/уничтожен (чистая проверка, без контекста) |
| `InvalidViewSlot` | `view_slot >= MAX_VIEW_SLOTS` (сейчас 10; молчаливого клампа нет) |
| `NoContext` | нет живого `sg` |
| `NoConsumableFrame` | `!scene.hasConsumableFrame()` — prepare ещё не публиковал кадр |
| `NoCamera` | в staged-снапшоте нет камеры |
| `IncompatibleColorFormat` | color таргета ≠ forward-формату сцены (обязательный RGBA16F; явный RGBA8/sRGB — молча менять каналы/кодирование нельзя) |
| `DepthRequired` | у таргета нет depth (forward-пайплайны depth-tested, pass без depth невалиден для них) |
| `IncompatibleDepthFormat` | depth таргета ≠ `defaultDepthFormat()` |
| `PassBeginFailed` | защитная (все гейты пройдены, а `begin` отказал — на одном потоке недостижимо) |
| `FeedbackLoop` | staged draw сэмплирует image этого таргета |

Контракт порядка кадров (для владельца пайплайна): вызывать между prepare (или staged begin/finish) и `Scene.render`, пока prepared draws консьюмабельны. Тени и пробы в захвате — с прошлого кадра, если захват идёт до теневого/проб-проходов текущего кадра (тот же документированный лаг в один кадр, что у проб).

Clustered-слоты: камеры занимают 0…7, refraction — отдельный 8, один ручной RTT-захват — 9 по умолчанию; дополнительные GPU-буферы создаются лениво. Несколько ручных захватов одной сцены в одном sokol-кадре требуют разных свободных слотов: один буфер нельзя обновлять дважды за кадр. Не используйте camera/refraction-слот, если он тоже рендерится в этом кадре. Без clustered-света аплоада нет.

Захват сцены требует совпадения форматов таргета с forward-пайплайнами. Кастомные форматы — clear/sample либо рисование собственными совместимыми пайплайнами. Внутренние `Environment.capture_opaque_only`/`gamma_override` используются автоматическим refraction-проходом; общий RTT-захват не меняет их.

### Возможности и лимиты

```zig
pub const Capabilities = struct { backend, color_sample/filter/render/msaa, depth_render/msaa, max_image_2d, max_samples, ... };
pub fn queryCapabilities(color, depth) CreateError!Capabilities; // нужен контекст
pub fn snappedSamples(requested, backend) i32;  // чистая политика MSAA
pub fn needsResolve(sample_count) bool;
pub fn isSrgbFormat(fmt) bool;                  // хардварные sRGB-варианты
pub fn estimatedBytesFor(w, h, color, depth, samples) usize; // чистый бюджет VRAM
pub fn validateDimensions(w, h) CreateError!void;            // чистая валидация
```

## Linear/sRGB семантика (HDR)

Сцена/history/probe/refraction — обязательный RGBA16F linear. Main-таргеты создаются каноническим `prepareMainTargets` (required caps + alloc errors, fail-closed, без SDR-fallback); `resize(w, h, samples) bool`, `scene.forwardFor(samples, color_format)`. Таргет хранит то, что записал pipeline. UNORM — только display-выход: один IEC display transfer, ручной UNORM / hw-sRGB путь; output gamma-флаги удалены. Общий scene capture sRGB-варианты не принимает из-за несовместимости pipeline-формата. Скриншоты — визуальное подтверждение, не числовая radiance-метрика (GPU readback нет).

## Потоки и владение

| Операция | Поток | Контекст |
|---|---|---|
| `create` / `resize` / `deinit` | только context | нужен `sg` (иначе `NoContext`/`false`); `resize`/`deinit` при открытом pass — `false`/assert |
| `begin` / `end` / `clear` | только context | fail-closed `false` без `sg`; `begin` при открытом pass — `false`, `end` без pass — no-op |
| `renderPrimaryView` | только context | `CaptureError` (таблица выше) вместо bool |
| `validateDimensions`, `snappedSamples`, `needsResolve`, `isSrgbFormat`, `isDepthFormat`, `estimatedBytesFor`, предикаты `Capabilities` | любой | не нужен |
| `queryCapabilities`, `defaultColorFormat`, `defaultDepthFormat` | любой, но | нужен `sg` |

Владелец хендлов — всегда таргет; `asTexture` — borrow без передачи владения (см. выше). Очередь ретайра (`GpuRetireQueue`) в v1 не задействована: её kind'ы живут под `scene/` (владение main-потока), а таргет — context-thread owned с немедленным destroy, как таргеты `PostProcessPass`.

## Ошибки и краевые случаи

| Ситуация | Поведение |
|---|---|
| Нет `sg`-контекста (headless-тесты, воркеры) | `create` → `NoContext`, pass-входы → `false`, `renderPrimaryView` → `NoContext`; таргет остаётся невалидным |
| Частичный фейл создания | `errdefer`-цепочка сносит созданное, ошибки наружу, полуживого таргета нет |
| `resize` мимо | `false`, старый таргет и его контент целы (rollback; зафиксировано тестом) |
| `resize` в тот же размер | no-op `true` (при валидном таргете) |
| `resize`/`begin` при открытом pass | `false` (нельзя трогать bound-хендлы / вкладывать проходы) |
| `end` без открытого pass | безопасный no-op (голого `sg.endPass` нет) |
| Сэмплинг при открытом pass | пустые хендлы (`sampleView`, `depthSampleView`, `sampleSampler`, хендлы `asTexture`) |
| Нет prepared-кадра / камеры | `NoConsumableFrame` / `NoCamera` (pin берётся только после гейтов; частичных проходов нет) |
| Слот вне таблицы / чужой формат / нет depth | `InvalidViewSlot` / `Incompatible*` / `DepthRequired` (см. таблицу захвата) |
| MSAA + `depthSampleView` | Пусто by design (нет depth resolve в sokol) |
| `asTexture` + `deinit` заимствованного | Безопасный no-op; сам таргет инвалидирует borrow при `resize`/`deinit` |

## GPU smoke: `examples/render_target_basic.zig`

Sokol-app программа с двумя сценами: capture-сцена (камера+свет+куб) рисуется в color+depth RT через `renderPrimaryView`; display-сцена рисует unlit-панель с `diffuse_texture = rtt.asTexture()`. Borrow обновляется после resize, **до prepare**. `--rtt-msaa 4` проверяет capture→resolve→sample; дополнительный MSAA RT проверяет clear/resize и сохранение NEAREST-фильтрации. Проверяются `InvalidDimensions`, сохранение старого RT при failed resize, безопасный `borrow.deinit`, preflight `FeedbackLoop` для альтернативного view того же image (без открытого pass и без GPU errors). Обычные кадры используют раздельные capture/display сцены без feedback. Итог — `verdict=PASS/FAIL`, ненулевые draws, ноль sokol errors; сбой даёт exit(1). Borrow'ы перед teardown отцепляются.

`--refraction` добавляет стеклянную сферу (`factor=1, refract=true`, `alpha=1`) перед панелью; `--ior F` и `--thickness F` задают оптические параметры, `--freeze` фиксирует геометрию. Проверяется наличие refraction RT при opt-in и отсутствие без него. На кадре 10 выполняется `renderReuse`: streak растёт, stats восстанавливаются, захват сохраняется.

Две процедурные полоски с UV0 ≠ UV1 (mirrored U) используют checker-текстуру: Standard выбирает UV0, PBR — UV1. Import/accessor/override/тангенты дополнительно покрыты GPU-free тестами.

Smoke проверяет реальные ненулевые draws и GPU-валидацию, но не заменяет проверку пикселей. Универсального readback API в примере нет: окно можно захватить средствами ОС. Проверено 04.10.2026 на Metal/macOS: RT capture 120 кадров + refraction 4x — PASS, 0 ошибок. Полная shared-shape (depth+color+samples) — не завершена, не заявляется.

Попиксельная проверка собственного окна, фиксированные камера/IOR=1.5: `--thickness 0` против `0.7` меняет 32 380 пикселей (31 288 с |Δ|>8), **все внутри сферы**, вне неё — ноль отличий. При толщине 0.7 изменение IOR 1→1.5 меняет 115 788 пикселей (30 956 с |Δ|>8), также только в сфере. Артефакты сохранены в `bench/screenshots/rtt_*`, метрики — `bench/rtt_pixel_proof.json`, логи — `bench/native_reports/rtt_*`. Это подтверждает работу выборки фона/оптических параметров в rigid PBR на Metal, не физическую точность или cross-backend parity.

Запуск: `zig build example-rtt -- --frames 120 --refraction --ior 1.5 --thickness 0.7`. Флаги: `--frames N`, `--rtt-size N`, `--rtt-msaa 1|2|4`, `--refraction`, `--ior F`, `--thickness F`, `--freeze`.

Шаг `example-rtt` уже подключён в `build.zig`:

```zig
const rtt_smoke = b.addExecutable(.{
    .name = "rtt-smoke",
    .root_module = b.createModule(.{
        .root_source_file = b.path("examples/render_target_basic.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "sokol", .module = mod_sokol },
            .{ .name = "agate", .module = mod_agate },
        },
    }),
});
b.installArtifact(rtt_smoke);
const run_rtt = b.addRunArtifact(rtt_smoke);
if (b.args) |args| run_rtt.addArgs(args);
b.step("example-rtt", "Run the RTT capture/sample smoke (needs GPU/display)").dependOn(&run_rtt.step);
```

Шаг — только run (GPU), никогда часть `test`: smoke требует дисплея/контекста и конечного verdict-лога, а не pass/fail юнит-раннера.

## Производительность

- Полноэкранный захват 1080p RGBA8+DEPTH ≈ 8 МБ (`estimatedBytes`), 4x MSAA ≈ ×4 + resolve; полуразрешение для refraction — четверть.
- Один захват = один полный проход primary-очередей (opaque + transparent + небо + частицы): бюджетировать как второй main-проход в уменьшенном разрешении; clustered-перестройка идёт на свой слот.
- `resize` — destroy+create (потеря контента); не вызывать каждый кадр — только на смену разрешения.

## Автоматическое преломление

`Scene.refraction` лениво создаёт собственный RT для `PBRMaterial.transmission.refract = true`, снимает opaque-фон после теней/проб перед main-pass в half-resolution, исключает glass/transparent draws и отказывает при feedback. Uniforms/view/sampler передаются всем трём PBR-семействам. Capture хранит `frame_id`: reuse не загружает буферы/не переснимает фон и не использует захват другого кадра. Ограничения и пример — [material.md](./material.md#преломление-v1-отключено-по-умолчанию).

## Смотрите также

- `./texture.md` — `Texture`, сэмплеры, sRGB-контракты, бюджеты VRAM
- `./render-pipeline.md` — порядок проходов, куда встанет захват
- `./frame-pipeline.md` — staged prepare, консьюмабельность prepared draws
- `./material.md` — слоты текстур для проброса transmission-сэмпла
