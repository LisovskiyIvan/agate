# Постпроцессинг

> Путь: src/postprocess.zig + src/postprocess/ · Импорт: agate.postprocess (root.zig: PostProcessOptions, TonemappingType, ShaftResolution + per-эффект листья: BloomOptions, GlowOptions, VignetteOptions, GradeOptions, DofOptions, LutOptions, FxaaOptions, FogOptions, SsrOptions, SsgiOptions (+SsgiTemporalOptions), ContactShadowsOptions, LocalTonemapOptions, SharpenOptions, GrainOptions, WhiteBalanceOptions, MotionBlurOptions, TaaOptions (+TaaReactiveOptions), ShaftsOptions, AutoExposureOptions — через postprocess.options) · Потоки: главный поток (конфиг) + GPU (composite в postprocess.glsl через PostProcessPass).

## Что это

Модуль `postprocess` — CPU-сторона единой HDR-цепочки постэффектов: конфиг `PostProcessOptions`, чистая математика эффектов и клампы (`clamped()`), плюс per-эффект хелперы, которыми `PostProcessPass` (см. `./passes.md`) пакует юниформы для composite-шейдера `postprocess.glsl`. Цепочка одна (HDR linear → effects → exposure → tonemap → display transfer); `enabled = false` не выключает обязательный output pass (exposure+tonemap всё равно выполняются).

Структура (`postprocess.zig` — фасад-реэкспорт, `postprocess/` — листья):

| Лист               | Ответственность                                                                                                     |
| ------------------ | ------------------------------------------------------------------------------------------------------------------- |
| `types.zig`        | `TonemappingType`, `ShaftResolution`, `BloomMipSize`, `ShaftTargetSize`, `TaaReset/Bounds`, `LutStripLayout/Sample` |
| `options.zig`      | `PostProcessOptions` + per-эффект листья + `clamped()` + `LutOptions.setColorGradingLut`/`clearColorGradingLut`     |
| `bloom.zig`        | пирамида, Karis, tent-веса                                                                                          |
| `glow.zig`         | глобальный halo-слой (порог, ядро, тинт)                                                                            |
| `highlight.zig`    | подсветка выбранного (маска, inner glow, composite)                                                                 |
| `dof.zig`          | глубина резкости (линеаризация, CoC, golden-angle тапы)                                                             |
| `color_curves.zig` | параметрический грейдинг (lifts по зонам)                                                                           |
| `lut.zig`          | текстурный LUT (Babylon-паритет, 2D-strip)                                                                          |
| `taa.zig`          | временное сглаживание (Halton-джиттер, neighborhood clamp, resolve)                                                 |
| `shafts.zig`       | волюметрические shafts (Хеньи–Гринштейн, bilateral-веса)                                                            |
| `depth_normal.zig` | нормаль из depth-градиентов для SSR/contact shadows/SSGI, независимая от размера пикселя                            |

Дизайн-принцип: каждый эффект по умолчанию либо выключен, либо bit-identical (выключенная ветвь в шейдере скипается до сэмплов — рендер не меняется ни на бит). Новые эффекты добавляются как «default OFF + скип в шейдере + `*Active` гейт на CPU».

## Быстрый старт

```zig
const agate = @import("agate");

var pp = agate.postprocess.PostProcessOptions{
    .enabled = true,
    .exposure = 1.1,
    .tonemapping = .aces,
    .bloom = .{ .enabled = true, .threshold = 0.8, .intensity = 0.5 },
    .vignette = .{ .enabled = true },
};
// Санитизация перед заливкой в юниформы / после загрузки из файла:
pp = pp.clamped();
scene.post_process = pp;

// TAA с одноразовым сбросом истории (телепорт камеры):
scene.post_process.taa.enabled = true;
scene.post_process.taa.camera_cut = true; // сбросится после одного кадра

// LUT-грейдинг из 2D-strip текстуры (валидация внутри):
scene.post_process.lut.setColorGradingLut(lut_tex, 16);

// Джиттер TAA для jittered view-proj (каждый кадр):
const j = agate.postprocess.taaJitter(frame_index, 1.0);
view_proj = agate.postprocess.applyTaaJitterToViewProj(view_proj, j, screen_w, screen_h);
```

## API

### Конфиг (`postprocess/options.zig`)

```zig
pub const PostProcessOptions = struct {
    enabled: bool = false,
    exposure: f32 = 1.0, tonemapping: TonemappingType = .aces,
    render_scale: f32 = 1.0, depth_pyramid_enabled: bool = false,
    auto_exposure: AutoExposureOptions = .{},
    bloom: BloomOptions = .{}, // enabled/threshold/intensity/radius/mips
    glow: GlowOptions = .{}, // enabled/threshold/intensity/radius/tint
    vignette: VignetteOptions = .{}, // enabled/intensity/radius
    grade: GradeOptions = .{}, // saturation/contrast/chromatic_aberration/shadows/midtones/highlights
    dof: DofOptions = .{}, // enabled/focus_distance/focus_range/max_blur
    lut: LutOptions = .{}, // enabled/intensity/size/texture
    fxaa: FxaaOptions = .{}, // enabled
    fog: FogOptions = .{}, // enabled/density/height_falloff/start_distance/color/sun_scattering
    ssr: SsrOptions = .{}, // enabled/intensity/steps
    ssgi: SsgiOptions = .{}, // enabled/mode/intensity/radius/steps/temporal.*
    contact_shadows: ContactShadowsOptions = .{}, // enabled/intensity/distance/thickness/steps
    local_tonemapping: LocalTonemapOptions = .{}, // enabled/intensity
    sharpen: SharpenOptions = .{}, // enabled/intensity
    grain: GrainOptions = .{}, // enabled/intensity
    white_balance: WhiteBalanceOptions = .{}, // temperature/tint
    motion_blur: MotionBlurOptions = .{}, // enabled/max_blur_px/steps
    taa: TaaOptions = .{}, // enabled/blend/jitter_scale/sharpness/clamp_intensity/reactive.*/camera_cut
    shafts: ShaftsOptions = .{}, // enabled/intensity/steps/density/anisotropy/max_distance/resolution/blur_sigma/edge_sigma
    pub fn clamped(self: PostProcessOptions) PostProcessOptions
    pub fn forFrame(self: PostProcessOptions) PostProcessOptions
    pub fn autoExposureOptions(self: PostProcessOptions) auto_exposure.AutoExposureOptions
};
pub const LutOptions = struct {
    pub fn setColorGradingLut(self: *LutOptions, tex: ?texture_mod.Texture, size: u8) void
    pub fn clearColorGradingLut(self: *LutOptions) void
};
pub const TonemappingType = enum { ... }; // none/aces/filmic/reinhard/... (см. types.zig)
pub const ShaftResolution = enum { half, quarter };
```

Кламп-машинерия — одна центральная таблица (`clamp_table`: dotted-путь → sanitize-op) + exemption-список + `frame_disable_flags` + comptime-проверка полноты: новый лист без строки таблицы/обработчика/exemption — ошибка компиляции. Depth-2 листья (`ssgi.temporal.*`, `taa.reactive.*`) клампятся прямыми строками (generic `@field` второго уровня не вычисляется в comptime).

`clamped()` никогда не падает: тянет out-of-range в валидные (экспозиция/интенсивности — `max(0)`, шаги/сэмплы — в `[min, max]`, анизотропия — в `[-0.9, 0.9]`, грейды — в `[-1, 1]`, `lut.intensity` — в `[0, 1]`), а битую LUT-привязку (нет live view, неверный размер strip) сбрасывает в `null`. Безопасно применять при загрузке и перед заливкой юниформ. `lut.setColorGradingLut` с невалидным входом (включая `null`) — тоже сброс (шейдер никогда не сэмплит мусор).

Полный список эффектов и параметров (дефолты — verbatim из старой плоской структуры):

| Эффект                               | Поля                                                                                                                                                                              | Дефолт                                            | Клампы                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| ------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Master                               | `enabled`, `exposure`, `tonemapping`                                                                                                                                              | off, 1.0, aces                                    | exposure ≥ 0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| Bloom (HDR pyramid, одна реализация) | `bloom.enabled`, `bloom.threshold`, `bloom.intensity`, `bloom.radius`, `bloom.mips`                                                                                               | on, 0.8, 0.5, 2.0, 5                              | threshold/intensity ≥ 0; `bloom.radius` — quality upsample-tent по coarse-мипам в текселях [0,16]; mips → `clampBloomMips` [3,7]                                                                                                                                                                                                                                                                                                                                                                  |
| Glow v1 (global halo)                | `glow.enabled`, `glow.threshold`, `glow.intensity`, `glow.radius`, `glow.tint`                                                                                                    | off, const-дефолты, тинт {1,1,1}                  | ≥ 0; тинт `clampTint` в [0,1]                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| Vignette                             | `vignette.enabled`, `vignette.intensity`, `vignette.radius`                                                                                                                       | on, 0.35, 0.8                                     | —                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| Grading (параметрический)            | `grade.saturation`, `grade.contrast`, `grade.chromatic_aberration`, `grade.shadows/midtones/highlights`                                                                           | 1.05, 1.05, 0.0, нули                             | грейды `clampGrade` в [-1,1]                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| LUT текстурный                       | `lut.enabled`, `lut.intensity`, `lut.size`, `lut.texture`                                                                                                                         | off, 1.0, 16, null                                | intensity [0,1]; размер `validLutSize`                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| FXAA 3.11                            | `fxaa.enabled`                                                                                                                                                                    | on                                                | —                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| Fog (density+height)                 | `fog.enabled`, `fog.density`, `fog.height_falloff`, `fog.start_distance`, `fog.color`, `fog.sun_scattering`                                                                       | on, 0.015, 0.08, 5.0, светло-голубой, 0.8         | —                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| SSR                                  | `ssr.enabled`, `ssr.intensity`, `ssr.steps`, `ssr.mode`                                                                                                                           | on, 0.55, 16, `inline_`                           | steps [4,64]; `ssr.mode`: `inline_` (v1 raymarch в композите) / `half_res_pass` (v2: отдельный half-res trace + билатеральный depth-aware денойз/апскейл, `SsrPass`); интенсивность применяется в композите после денойза                                                                                                                                                                                                                                                                         |
| Local tonemap                        | `local_tonemapping.enabled`, `local_tonemapping.intensity`                                                                                                                        | off, 0.5                                          | intensity [0,1]; gain [0.25,4.0], фиксированный contrast 0.3                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| Sharpen (unsharp)                    | `sharpen.enabled`, `sharpen.intensity`                                                                                                                                            | off, 0.3                                          | —                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| Grain                                | `grain.enabled`, `grain.intensity`                                                                                                                                                | off, 0.05                                         | —                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| White balance                        | `white_balance.temperature`, `white_balance.tint`                                                                                                                                 | 0, 0 (нейтрально)                                 | —                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| Motion blur                          | `motion_blur.enabled`, `motion_blur.max_blur_px`, `motion_blur.steps`                                                                                                             | off, 32.0, 8                                      | max [1,128], steps [2,32]                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| TAA (+reactive)                      | `taa.enabled`, `taa.blend`, `taa.jitter_scale`, `taa.sharpness`, `taa.clamp_intensity`, `taa.reactive.enabled/intensity/marker_intensity`, `taa.camera_cut`                       | off, 0.9, 1.0, 0.0, 1.0, off/0.5/1.0, false       | blend/sharp/clamp/reactive [0,1], jitter [0,4]                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| Render scale                         | `render_scale`                                                                                                                                                                    | 1.0 (native)                                      | [0.25, 1]; main HDR-таргет/TAA-история/velocity/эффекты считаются в масштабе, композит+UI — в натуральном разрешении                                                                                                                                                                                                                                                                                                                                                                              |
| SSGI v1 (color bleed)                | `ssgi.enabled`, `ssgi.intensity`, `ssgi.radius`, `ssgi.steps`, `ssgi.mode`, `ssgi.temporal.enabled`, `ssgi.temporal.blend`                                                        | off, 0.5, 1.5, 12, `inline_gather`, off, 0.9      | intensity [0,1], radius [0.1, 8] м, steps [4,32]; `ssgi.mode`: `inline_gather` (v1 gather в композите) / `half_res_pass` (v2: отдельный half-res gather + билатеральный денойз, `SsgiPass`); `ssgi.temporal.enabled` — temporal аккумуляция gather-выхода (v2.1, только `half_res_pass`, default OFF — в inline-режиме предупреждает и игнорируется); `ssgi.temporal.blend` — вес истории [0,1], свит-спот [0.85, 0.95]; one-bounce экранное цветовое кровотечение (radiance+depth), TAA денойзит |
| Shafts v1                            | `shafts.enabled`, `shafts.intensity`, `shafts.steps`, `shafts.density`, `shafts.anisotropy`, `shafts.max_distance`, `shafts.resolution`, `shafts.blur_sigma`, `shafts.edge_sigma` | off, 1.0, 12, 0.05, 0.4, 60.0, quarter, 2.0, 0.02 | steps [4,32], anisotropy ±0.9, остальное ≥ 0                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| DOF                                  | `dof.enabled`, `dof.focus_distance`, `dof.focus_range`, `dof.max_blur`                                                                                                            | off, 10.0, 5.0, 8.0                               | все ≥ 0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |

Порядок стека в composite-шейдере (`postprocess.glsl`, реализован `PostProcessPass`): bloom-pyramid → glow → highlight → shafts → grading/LUT → vignette → exposure/tonemap → display transfer → FXAA/TAA-resolve → sharpen/grain. Exposure ручной (`exposure`); auto-exposure — адаптивная выдержка (`auto_exposure.*`, `Scene.updateAutoExposure*` + гистограммный метринг). Output gamma-флаги удалены: один IEC display transfer, ручной UNORM / hw-sRGB путь.

### Bloom (`postprocess/bloom.zig`)

```zig
pub const BLOOM_PYRAMID_MIPS_MIN: u32 = 3; // quality-бюджет мипов
pub const BLOOM_PYRAMID_MIPS_MAX: u32 = 7;
pub fn clampBloomMips(mips: u32) u32
pub fn bloomMipSize(base_w: i32, base_h: i32, mip: u32) BloomMipSize
pub fn karisWeight(color: [3]f32) f32
pub fn tentWeight1D(x: f32) f32
pub fn bloomTentWeight(x: f32, y: f32) f32
```

Одна реализация — HDR bloom-pyramid (`BloomPass`: Karis-down + tent-up); `bloom_pyramid` удалён, inline-gather пути нет. `bloomMipSize` — геометрия мипа для аллокации таргетов; `karisWeight`/`tentWeight` — CPU-зеркала шейдерных весов (для тестов паритета и тулзов). Сложность O(пиксели × mips). `bloom.radius` — quality радиуса upsample-tent по coarse-мипам в текселях [0,16]; `bloom.mips` — quality-бюджет [3,7].

### Glow (`postprocess/glow.zig`)

```zig
pub const GLOW_BLUR_TAPS / GLOW_HALF_TAPS / GLOW_PASS_DRAWS = ...;
pub const GLOW_THRESHOLD_DEFAULT / GLOW_INTENSITY_DEFAULT / GLOW_RADIUS_DEFAULT = ...;
pub fn clampTint(tint: [3]f32) [3]f32
pub fn validateGlow(...) ...
pub fn glowActive(options: PostProcessOptions) bool
pub fn glowExtract(...) ...       // CPU-зеркало extract-стадии
pub fn glowGaussianWeight(x: f32, sigma: f32) f32
pub fn glowKernelSum(sigma: f32) f32
pub fn glowParams(...) ... / pub fn glowTintParams(...) ...
```

Glow v1 — глобальный halo, отличный от bloom: threshold-extract по яркости + сепарабельный blur (`GlowPass`) + аддитивный composite ПОСЛЕ bloom. Per-mesh веса — вне скоупа v1 (нужны изменения пайплайна). `glowParams`/`glowTintParams` пакуют юниформы; `glowKernelSum` — нормализация ядра.

### Highlight (`postprocess/highlight.zig`)

```zig
pub fn highlightActive(...) bool
pub fn highlightParams(...) ...
pub fn highlightInnerGlow(...) ...
pub fn highlightComposite(...) ...
```

Подсветка выбранного объекта: маска (`HighlightPass` рисует id-цвета) → inner glow → composite поверх bloom/glow. Параметры — цвет, ширина glow, интенсивность (см. `./passes.md` про `HighlightDrawItem`).

### DOF (`postprocess/dof.zig`)

```zig
pub const DOF_GOLDEN_ANGLE: f32 = ...;
pub const DOF_TAPS: u32 = ...;
pub fn linearizeDepth(raw_depth: f32, near: f32, far: f32) f32
pub fn circleOfConfusion(depth: f32, focus_distance: f32, focus_range: f32, max_blur: f32) f32
pub fn dofTapOffset(tap: u32, radius: f32) [2]f32
```

Gather-blur по линеаризованной глубине; тапы — golden-angle спираль (`dofTapOffset`), CoC — гладкая рампа от фокусной плоскости. CPU-функции — зеркала шейдера для тестов и тулзов превью.

### Grading + LUT (`color_curves.zig`, `lut.zig`)

```zig
pub fn rgbLuma(rgb: [3]f32) f32
pub fn clampGrade(lift: [3]f32) [3]f32
pub fn applyGrade(color: [3]f32, shadows: [3]f32, midtones: [3]f32, highlights: [3]f32) [3]f32
pub const LUT_SIZE_MIN / LUT_SIZE_MAX = ...;
pub fn validLutSize(size: u8) bool
pub fn lutTextureValid(tex: Texture, size: u8) bool
pub fn lutStripLayout(size: u8) LutStripLayout
pub fn lutStripUv(...) ... / pub fn lutSampleUv(...) ...
pub fn applyLutStrip(...) ...
pub fn writeIdentityLutStrip(...) ... / pub fn buildIdentityLutStrip(...) ...
pub fn lutParams(...) ...
```

Два независимых пути грейдинга, порядок: сначала параметрические кривые (`applyGrade`: аддитивные lifts в [-1,1], взвешенные по зонам shadows/midtones/highlights через `rgbLuma`), затем текстурный LUT. LUT — 2D-strip (Babylon-паритет): ширина == size×size, высота == size; `lutTextureValid` проверяет live view + геометрию strip. `buildIdentityLutStrip` — нейтральный LUT для тестов/стартовой точки цветокоррекции. С `lut_texture == null` шейдер скипает ветвь целиком (bit-identical).

### Локальный тонемаппинг (`postprocess/hdr.zig`)

Детальный gain вычисляется из яркости центрального HDR-тапа и среднего пяти
непороговых тапов `scene_tex` **до экспозиции**, в одном radiance-домене.
Bloom не используется как оценка локальной яркости: его порог/интенсивность
не меняют адаптацию. Равномерное поле остаётся нейтральным при любой экспозиции
и bloom-добавке; полученный gain применяется к собранному цвету после экспозиции,
перед общей tonemap-кривой. CPU-зеркала: `localLuma`, `localTonemapGain`,
`applyLocalTonemap`; CPU и GLSL ограничивают тапы конечным half-диапазоном.
Выключенный эффект и intensity ≤ 0.001 сохраняют точный identity-путь.

### TAA (`postprocess/taa.zig`)

```zig
pub const TAA_JITTER_PERIOD: u32 = ...; // период последовательности Halton
pub fn halton(index: u32, base: u32) f32
pub fn taaJitter(frame_index: u64, scale: f32) [2]f32
pub fn applyTaaJitterToViewProj(view_proj: Mat4, jitter: [2]f32, width: i32, height: i32) Mat4
pub fn taaReadIndex / taaWriteIndex(...) ...  // ping-pong истории в PostProcessPass
pub fn taaShouldReset(camera_cut: bool, ...) bool
pub fn taaNeighborhoodBounds(...) TaaBounds
pub fn taaNeighborhoodAvg(...) ...
pub fn taaClampHistory(...) ...   // кламп истории в 3×3-бокс текущего кадра
pub fn taaResolve(...) ... / pub fn taaResolvePixel(...) ...
pub fn taaApplySharpen(...) ...
pub fn taaParams(...) ... / pub fn taaState(...) ...
```

Субпиксельный Halton-джиттер (±0.5px × `taa.jitter_scale`) с jittered view-proj от сцены; history ping-pong в `PostProcessPass` (`ensureTaaHistory`, `taaReadView`/`taaWriteAttView`); neighborhood-clamp против гостинга (`taa.clamp_intensity = 1` — полный кламп); пост-unsharp с ре-клампом (`taa.sharpness`). `taa.camera_cut` — одноразовый сброс истории (телепорт/монтаж; флаг несётся снапшотом кадра, потокобезопасно между update и render). Выключенный TAA — early-out до сэмплов истории/глубины (bit-identical).

### Shafts (`postprocess/shafts.zig`)

```zig
pub const SHAFT_STEPS_MIN / SHAFT_STEPS_MAX / SHAFT_MARCH_MAX = ...; // шаги [4,32]
pub const SHAFT_PASS_DRAWS / SHAFT_ANISOTROPY_MAX / SHAFT_BLUR_TAPS / SHAFT_BLUR_HALF_TAPS = ...;
pub fn shaftTargetSize(base_w: i32, base_h: i32, res: ShaftResolution) ShaftTargetSize
pub fn shaftActive(options: PostProcessOptions, shadows_enabled: bool) bool
pub fn shaftParams(...) ... / pub fn validateShaft(...) ...
pub fn hgPhase(cos_theta: f32, g: f32) f32
pub fn shaftCascadeIndex(...) ...
pub fn shaftBilateralWeight(depth_diff: f32, edge_sigma: f32) f32
pub fn shaftKernelSum(...) f32
```

Шафты v1: скринспейс-реймарш от камеры до поверхности глубины в low-res (`half`/`quarter`), на каждом шаге — bilinear 2×2 PCF в CSM-атлас солнца (интерполируются результаты сравнения, не raw-depth; фильтр не выходит в соседний тайл), single-scatter с фазой Хеньи–Гринштейна (`hgPhase`) и точным интегралом Бера–Ламберта по сегменту; bilateral (depth-aware) H/V-blur; аддитивный composite после highlight-блока. Энергия полностью освещённого луча не зависит от числа шагов. Требует живой CSM (`shaftActive` гейтится на `shadows_enabled` — без отрендеренного атласа маршировать не против чего, v1 отказывается «фейкать»). Небесные пиксели маршируют полный `shaft_max_distance`.

```zig
pub const SsrMode = enum { inline_, half_res_pass };
pub fn ssrActive / ssrInlineActive / ssrPassActive(config: PostProcessOptions) bool
pub fn ssrParams(config: PostProcessOptions) [4]f32            // inline-лейн
pub fn ssrStepsParams(config: PostProcessOptions) [4]f32       // inline steps-лейн
pub fn ssrPassParams(config: PostProcessOptions, tex_valid: bool) [4]f32 // pass-лейн
pub fn ssrFresnel / ssrEdgeFade / ssrDistFade / ssrBlendWeight / ssrMix(...) ... // CPU-зеркала весов
pub fn ssrBilateralWeight(...) f32                             // CPU-зеркало blur-гейта
pub fn ssrUpscaleResolve(...) ...                              // CPU-зеркало composite-реконструкции 2x2
pub fn ssrTargetSize / ssrTargetBytes / ssrHalfResSize(...) ... // half-res sizing (пасс и композит делят ceil-half)
```

Режимы: `inline_` (default, бит-в-бит v1 — raymarch внутри композита) / `half_res_pass` (v2: `SsrPass` — half-res RGBA16F trace-порт v1-математики с тем же floor-гейтом, clip-space маршем и edge/dist/fresnel-весами + билатеральный depth-aware денойз 3×3 с гейтом 0.05 м; финальный depth-aware апскейл — 2×2 билатеральный gather в самом композите (`sampleSSRPass`, зеркало `ssrUpscaleResolve`), а не слепой билинейный сэмпл, поэтому граница силуэта держится и в финальном пикселе; intensity применяется в композите после реконструкции; sizing — половина EFFECTIVE-размера, пасс и композит делят один ceil-half (`ssrTargetSize`/`ssrHalfResSize`); MSAA-подавление — через общий `ssr.enabled`). Выключенный режим не аллоцирует таргетов и не добавляет draws (ленивая аллокация на первом активном кадре). q0: `AGATE_Q0_SSR_PASS=1` (фиксированный half-res), `AGATE_Q0_AB=ssr` (чередование inline/half-res по блокам с теми же intensity/steps + `ssr-proof` на 30-м кадре: размер/живость таргета в pass-режиме, отсутствие аллокаций в inline/off).

### SSGI (`postprocess/ssgi.zig`, `passes/ssgi_pass.zig`, `shaders/ssgi*.glsl`)

```zig
pub const SsgiMode = enum { inline_gather, half_res_pass };
pub fn ssgiActive / ssgiInlineActive / ssgiPassActive(config: PostProcessOptions) bool
pub fn ssgiParams(config: PostProcessOptions) [4]f32        // inline-лейн
pub fn ssgiPassParams(config: PostProcessOptions, tex_valid: bool) [4]f32 // pass-лейн
pub fn ssgiTemporalActive(config: PostProcessOptions) bool  // pass + temporal-флаг
pub fn ssgiTemporalBlend(config: PostProcessOptions) f32
pub fn ssgiTemporalShouldReset(TaaReset) bool
```

Режимы: `inline_gather` (default, бит-в-бит v1 — gather внутри композита) / `half_res_pass` (v2: `SsgiPass` — half-res RGBA16F gather-порт v1-математики с тем же мировым диском + билатеральный depth-aware денойз 3×3; intensity применяется в композите после денойза; sizing — половина EFFECTIVE-размера; MSAA-подавление — через общий `ssgi_enabled`, temporal присоединяется бесплатно). Temporal аккумуляция (v2.1, `ssgi_temporal_enabled`, default OFF — выключенная ветвь пропускает стадию, денойз сэмплит gather напрямую, бит-в-бит): порядок gather → temporal → denoise → composite (история хранит сырой накопленный сигнал, денойз — последний чистильщик перед композитом). Репроекция — по velocity-буферу (uv_curr − uv_prev; живёт в render-scale разрешении, сэмплится в том же UV), без velocity — камерная depth-репроекция через `prev_view_proj`. Реджект: variance-расхождение истории от текущего 3×3 gather-окна (mean ± `SSGI_TEMPORAL_GAMMA=1.0`·σ, фоллофф `1/(1+4d²)` как в TAA), консистентность глубины против хранимой в альфе линеаризованной глубины (билатеральный гейт `max(0.001, radius·0.2)`), консистентность нормалей (house depth-normal реконструкция с обоих концов, косинус ≥ 0.9), reactive-маркеры (альфа velocity 0.15–0.35 = партикли/стекло → ноль истории без рампы). Сброс истории: первый кадр / toggle флага / ресайз (пересоздание таргетов) / camera cut (общий снапшот-флаг `taa_camera_cut`) — зеркало `taaShouldReset`; константы гейтов документированы в `ssgi.zig`. В inline-режиме флаг предупреждает один раз (WarnOnce) и игнорируется. q0: `AGATE_Q0_SSGI_PASS=1`, `AGATE_Q0_SSGI_TEMPORAL=1` (требует PASS=1, иначе варнинг).

Для визуального A/B света в `examples/q0_interior.zig`: `AGATE_Q0_STILL=1` фиксирует камеру, `AGATE_Q0_NO_DOF=1` убирает размытие, `AGATE_Q0_NO_SHAFTS=1` отключает только объёмные лучи, `AGATE_Q0_NO_SSR=1` изолирует отражения. Прямоугольное солнечное пятно от открытого фасада комнаты при этом остаётся — это геометрия проёма, а не граница атласа теней.

Нормаль из depth проверяется относительным критерием вырожденности (`|dx×dy|² / (|dx|²·|dy|²)`), а не абсолютным порогом в мировых единицах. Иначе малый пиксель ошибочно превращал вертикальную стену в «пол», и SSR рисовал на ней прямоугольные полосы отражений. Сосед для градиента выбирается по разнице линейной view-depth (ось из четвёртой строки perspective VP), не мирового Z: поворот камеры не переносит нормаль на фон через силуэт. Вырожденная нормаль возвращает ноль: SSR/contact shadows/SSGI не добавляют фиктивный эффект.

## Потоки и владение

- `PostProcessOptions` — plain value-struct, копируется присвоением (включая LUT: хэндл `Texture` — plain GPU-хэндлы без CPU-рефов, поэтому структура целиком копируется в `SceneFrameSnapshot.post_process`). Снапшот несёт конфиг из update-потока в render без синхронизации.
- Все функции листьев — чистые (конфиг/геометрия in → числа out), без аллокаций, кроме LUT-билдеров (`buildIdentityLutStrip` аллоцирует через переданный аллокатор) и `lutParams` (возвращает структуры, не память).
- `taa.camera_cut`: ставит update-поток, читает/сбрасывает render — через снапшот, гонок нет.
- `LutOptions.setColorGradingLut`/`clearColorGradingLut` — методы конфига (не пасса): вызываются из главного потока до упаковки снапшота. `lut.enabled` — обычный bool (не derived от `texture != null`): `forFrame()` гасит флаг эффекта, а привязка переживает master-выключатель, как раньше.
- `PostProcessPass` (см. `./passes.md`) держит все GPU-хэндлы сам; операции разделены по листьям в `src/passes/postprocess/`: `composite.zig` — чистый (binding/uniform-упаковка без sg-вызовов), `targets.zig` — GPU-жизненный цикл main-таргета и TAA-истории (`resize`/`ensureTaaHistory`/`taaReset` + `*Valid`/`*View`-запросы, с sg create/destroy/query внутри). Методы пасса — тонкие форвардеры, порядок create/destroy и fail-closed поведение без изменений.

## Ошибки и краевые случаи

- Ошибок как значений почти нет: все клампы total (`clamped()` не падает). Единственный fallible путь — LUT-билдеры (`OutOfMemory`) и `setColorGradingLut` с невалидной текстурой (не ошибка — тихий сброс в `null` + `lut.enabled = false`).
- `bloom.mips` вне `[3, 7]` — кламп, а не игнор: эффект остаётся включённым с ближайшим валидным числом мипов.
- `shafts.steps` клампится к [4,32] (граница шейдерного цикла); `shafts.anisotropy` — к ±0.9 (сингулярность HG-фазы при |g|→1).
- `motion_blur.steps`/`ssr.steps` клампятся к своим диапазонам; нулевые значения невозможны после `clamped()`.
- Motion blur собирает центрированное box-ядро по полному отрезку `[-0.5, +0.5]` (N тапов, `i = 0..N-1`, формула `t = i/(N-1) - 0.5`): нулевое среднее при любом `samples` 2..32, при минимуме 2 тапы `{-0.5, +0.5}`. Центральная поправка сохраняет CA-дельту `sampleScene` (`delta/N`). Стоимость: N gather-чтений сцены в цикле плюс 1 чтение сырого центра для поправки (раньше: N-1 чтений в цикле плюс готовый `color`); без предположений о CSE компилятора. Паритет ядра пинится `src/postprocess/motion_blur_tests.zig`.
- LUT с `lut.size`, не прошедшим `validLutSize`, или с геометрией strip, не совпадающей с размером, — сброс привязки (шейдер идёт по no-LUT пути, а не сэмплит мусор).
- `taa.camera_cut = true` при выключенном TAA — безвреден (истории нет, сбрасывать нечего).
- Все `*Active` гейты (`glowActive`, `highlightActive`, `shaftActive`) — единственный источник правды для `PostProcessPass` о том, запускать ли пассы; bloom идёт единой пирамидой при `bloom.enabled`.

## Производительность

- Выключенные эффекты стоят ноль GPU (скип ветви + незапущенные пассы) и ноль CPU (гейт до упаковки юниформ).
- Дорогие по GPU (порядок): shafts (реймарш × шаги × low-res + bilateral blur) > bloom-пирамида (mips даун/ап) > SSR (шаги луча 4–64) > motion blur (сэмплы 2–32) > DOF-gather > glow-blur > single-shader bloom. Ориентир: начинайте с quarter-шафтов и 5 мипов пирамиды.
- CPU-математика модуля — O(1) на вызов (веса, UV, джиттер); исключение — `buildIdentityLutStrip` O(size³) однократно при создании LUT.
- TAA держит два history-таргета полного разрешения (`ensureTaaHistory`) — самая заметная видеопамять модуля; `taaReset` при ресайзе обязателен (иначе репроекция из чужого разрешения).
- `clamped()` — дешёвый (скаляры), вызывайте без страха каждый кадр после твиков UI; кэшировать не нужно.
- HDR-showcase: `zig build run-hdr` (в `examples/`), тёмная студия + клавиши B/E/Space (bloom toggle, exposure cycle, effects master); конечный прогон `AGATE_HDR_FRAMES=240`, `MSAA4`, `SRGB1`; browser-кадры — query-параметром. Скриншоты — визуальные, не числовая radiance-метрика.

## Смотрите также

- `./passes.md` — `PostProcessPass`, `BloomPass`, `GlowPass`, `VolumetricPass`: GPU-сторона стека.
- `./shaders.md` — `postprocess.glsl`, include-чанки, slang-леги.
- `./frame-pipeline.md` — `Scene.postfx` / `PostFXStack.renderChain`, `ChainParams.post`.
- `./texture.md` — `Texture`, LUT-текстуры, live view.
- `./cameras.md` — jittered view-proj для TAA.
- `./lights.md` — CSM солнца для шафтов.
- `./serialization.md` — `postprocess_persisted`: какие поля переживают сейв.
