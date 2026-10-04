# Постпроцессинг

> Путь: src/agate/postprocess.zig + src/agate/postprocess/ · Импорт: agate.postprocess (root.zig: PostProcessOptions, TonemappingType, LutFormat, ShaftResolution) · Потоки: главный поток (конфиг) + GPU (composite в postprocess.glsl через PostProcessPass).

## Что это

Модуль `postprocess` — CPU-сторона единой HDR-цепочки постэффектов: конфиг `PostProcessOptions`, чистая математика эффектов и клампы (`clamped()`), плюс per-эффект хелперы, которыми `PostProcessPass` (см. `./passes.md`) пакует юниформы для composite-шейдера `postprocess.glsl`. Цепочка одна (HDR linear → effects → exposure → tonemap → display transfer); `enabled = false` не выключает обязательный output pass (exposure+tonemap всё равно выполняются).

Структура (`postprocess.zig` — фасад-реэкспорт, `postprocess/` — листья):

| Лист | Ответственность |
|---|---|
| `types.zig` | `TonemappingType`, `LutFormat`, `ShaftResolution`, `BloomMipSize`, `ShaftTargetSize`, `TaaReset/Bounds`, `LutStripLayout/Sample` |
| `options.zig` | `PostProcessOptions` + `clamped()` + `setColorGradingLut`/`clearColorGradingLut` |
| `bloom.zig` | пирамида, Karis, tent-веса |
| `glow.zig` | глобальный halo-слой (порог, ядро, тинт) |
| `highlight.zig` | подсветка выбранного (маска, inner glow, composite) |
| `dof.zig` | глубина резкости (линеаризация, CoC, golden-angle тапы) |
| `color_curves.zig` | параметрический грейдинг (lifts по зонам) |
| `lut.zig` | текстурный LUT (Babylon-паритет, 2D-strip) |
| `taa.zig` | временное сглаживание (Halton-джиттер, neighborhood clamp, resolve) |
| `shafts.zig` | волюметрические shafts (Хеньи–Гринштейн, bilateral-веса) |

Дизайн-принцип: каждый эффект по умолчанию либо выключен, либо bit-identical (выключенная ветвь в шейдере скипается до сэмплов — рендер не меняется ни на бит). Новые эффекты добавляются как «default OFF + скип в шейдере + `*Active` гейт на CPU».

## Быстрый старт

```zig
const agate = @import("agate");

var pp = agate.postprocess.PostProcessOptions{
    .enabled = true,
    .exposure = 1.1,
    .tonemapping = .aces,
    .bloom_enabled = true,
    .bloom_threshold = 0.8,
    .bloom_intensity = 0.5,
    .vignette_enabled = true,
};
// Санитизация перед заливкой в юниформы / после загрузки из файла:
pp = pp.clamped();
scene.post_process = pp;

// TAA с одноразовым сбросом истории (телепорт камеры):
scene.post_process.taa_enabled = true;
scene.post_process.taa_camera_cut = true; // сбросится после одного кадра

// LUT-грейдинг из 2D-strip текстуры (валидация внутри):
scene.post_process.setColorGradingLut(lut_tex, 16);

// Джиттер TAA для jittered view-proj (каждый кадр):
const j = agate.postprocess.taaJitter(frame_index, screen_w, screen_h, 1.0);
view_proj = agate.postprocess.applyTaaJitterToViewProj(view_proj, j);
```

## API

### Конфиг (`postprocess/options.zig`)

```zig
pub const PostProcessOptions = struct {
    enabled: bool = false,
    exposure: f32 = 1.0, tonemapping: TonemappingType = .aces,
    // Bloom / bloom-пирамида / glow / vignette / grading / LUT / FXAA /
    // fog / SSR / sharpen / grain / white balance / motion blur /
    // TAA / shafts — см. таблицу эффектов ниже
    pub fn clamped(self: PostProcessOptions) PostProcessOptions
    pub fn setColorGradingLut(self: *PostProcessOptions, tex: ?texture_mod.Texture, size: u8) void
    pub fn clearColorGradingLut(self: *PostProcessOptions) void
};
pub const TonemappingType = enum { ... }; // none/aces/filmic/reinhard/... (см. types.zig)
pub const LutFormat = enum { ... };       // strip_2d (+ будущие)
pub const ShaftResolution = enum { half, quarter };
```

`clamped()` никогда не падает: тянет out-of-range в валидные (экспозиция/интенсивности — `max(0)`, шаги/сэмплы — в `[min, max]`, анизотропия — в `[-0.9, 0.9]`, грейды — в `[-1, 1]`, `lut_strength` — в `[0, 1]`), а битую LUT-привязку (нет live view, неверный размер strip) сбрасывает в `null`. Безопасно применять при загрузке и перед заливкой юниформ. `setColorGradingLut` с невалидным входом (включая `null`) — тоже сброс (шейдер никогда не сэмплит мусор).

Полный список эффектов и параметров:

| Эффект | Поля | Дефолт | Клампы |
|---|---|---|---|
| Master | `enabled`, `exposure`, `tonemapping` | off, 1.0, aces | exposure ≥ 0 |
| Bloom (HDR pyramid, одна реализация) | `bloom_enabled`, `bloom_threshold`, `bloom_intensity`, `bloom_radius`, `bloom_pyramid_mips` | on, 0.8, 0.5, 2.0, 5 | threshold/intensity ≥ 0; `bloom_radius` — quality upsample-tent по coarse-мипам в текселях [0,16]; mips → `clampBloomMips` [3,7] |
| Glow v1 (global halo) | `glow_enabled`, `glow_threshold`, `glow_intensity`, `glow_radius`, `glow_tint` | off, const-дефолты, тинт {1,1,1} | ≥ 0; тинт `clampTint` в [0,1] |
| Vignette | `vignette_enabled`, `vignette_intensity`, `vignette_radius` | on, 0.35, 0.8 | — |
| Grading (параметрический) | `saturation`, `contrast`, `grade_shadows/midtones/highlights` | 1.05, 1.05, нули | грейды `clampGrade` в [-1,1] |
| LUT текстурный | `lut_enabled`, `lut_strength`, `lut_size`, `lut_format`, `lut_texture` | off, 1.0, 16, strip_2d, null | strength [0,1]; размер `validLutSize` |
| FXAA 3.11 | `fxaa_enabled` | on | — |
| Fog (density+height) | `fog_enabled`, `fog_density`, `fog_height_falloff`, `fog_start_distance`, `fog_color`, `fog_sun_scattering` | on, 0.015, 0.08, 5.0, светло-голубой, 0.8 | — |
| SSR | `ssr_enabled`, `ssr_intensity`, `ssr_max_distance`, `ssr_thickness`, `ssr_steps` | on, 0.55, 25.0, 0.4, 16 | steps [4,64] |
| Sharpen (unsharp) | `sharpen_enabled`, `sharpen_amount` | off, 0.3 | — |
| Grain | `grain_enabled`, `grain_intensity` | off, 0.05 | — |
| White balance | `temperature`, `tint` | 0, 0 (нейтрально) | — |
| Motion blur | `motion_blur_enabled`, `motion_blur_intensity`, `motion_blur_max_blur_px`, `motion_blur_samples` | off, 0.5, 32.0, 8 | intensity [0,3], max [1,128], samples [2,32] |
| TAA | `taa_enabled`, `taa_blend`, `taa_jitter_scale`, `taa_sharpness`, `taa_clamp_strength`, `taa_camera_cut` | off, 0.9, 1.0, 0.0, 1.0, false | blend/sharp/clamp [0,1], jitter [0,4] |
| Shafts v1 | `shaft_enabled`, `shaft_intensity`, `shaft_steps`, `shaft_density`, `shaft_anisotropy`, `shaft_max_distance`, `shaft_resolution`, `shaft_blur_sigma`, `shaft_edge_sigma` | off, 1.0, 12, 0.05, 0.4, 60.0, quarter, 2.0, 0.02 | steps [4,32], anisotropy ±0.9, остальное ≥ 0 |
| DOF | `dof_enabled`, `dof_focus_distance`, `dof_focus_range`, `dof_max_blur` | off, 10.0, 5.0, 8.0 | все ≥ 0 |

Порядок стека в composite-шейдере (`postprocess.glsl`, реализован `PostProcessPass`): bloom-pyramid → glow → highlight → shafts → grading/LUT → vignette → exposure/tonemap → display transfer → FXAA/TAA-resolve → sharpen/grain. Exposure ручной (`exposure`); auto-exposure — будущее. Output gamma-флаги удалены: один IEC display transfer, ручной UNORM / hw-sRGB путь.

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

Одна реализация — HDR bloom-pyramid (`BloomPass`: Karis-down + tent-up); `bloom_pyramid` удалён, inline-gather пути нет. `bloomMipSize` — геометрия мипа для аллокации таргетов; `karisWeight`/`tentWeight` — CPU-зеркала шейдерных весов (для тестов паритета и тулзов). Сложность O(пиксели × mips). `bloom_radius` — quality радиуса upsample-tent по coarse-мипам в текселях [0,16]; `bloom_pyramid_mips` — quality-бюджет [3,7].

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

### TAA (`postprocess/taa.zig`)

```zig
pub const TAA_JITTER_PERIOD: u32 = ...; // период последовательности Halton
pub fn halton(index: u32, base: u32) f32
pub fn taaJitter(frame_index: u64, width: i32, height: i32, scale: f32) [2]f32
pub fn applyTaaJitterToViewProj(view_proj: Mat4, jitter: [2]f32) Mat4
pub fn taaReadIndex / taaWriteIndex(...) ...  // ping-pong истории в PostProcessPass
pub fn taaShouldReset(camera_cut: bool, ...) bool
pub fn taaNeighborhoodBounds(...) TaaBounds
pub fn taaNeighborhoodAvg(...) ...
pub fn taaClampHistory(...) ...   // кламп истории в 3×3-бокс текущего кадра
pub fn taaResolve(...) ... / pub fn taaResolvePixel(...) ...
pub fn taaApplySharpen(...) ...
pub fn taaParams(...) ... / pub fn taaState(...) ...
```

Субпиксельный Halton-джиттер (±0.5px × `taa_jitter_scale`) с jittered view-proj от сцены; history ping-pong в `PostProcessPass` (`ensureTaaHistory`, `taaReadView`/`taaWriteAttView`); neighborhood-clamp против гостинга (`taa_clamp_strength = 1` — полный кламп); пост-unsharp с ре-клампом (`taa_sharpness`). `taa_camera_cut` — одноразовый сброс истории (телепорт/монтаж; флаг несётся снапшотом кадра, потокобезопасно между update и render). Выключенный TAA — early-out до сэмплов истории/глубины (bit-identical).

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

Шафты v1: скринспейс-реймарш от камеры до поверхности глубины в low-res (`half`/`quarter`), на каждом шаге — один raw-depth тап в CSM-атлас солнца (без PCF), single-scatter с фазой Хеньи–Гринштейна (`hgPhase`) и затуханием Бера–Ламберта; bilateral (depth-aware) H/V-blur; аддитивный composite после highlight-блока. Требует живой CSM (`shaftActive` гейтится на `shadows_enabled` — без отрендеренного атласа маршировать не против чего, v1 отказывается «фейкать»). Небесные пиксели маршируют полный `shaft_max_distance`.

## Потоки и владение

- `PostProcessOptions` — plain value-struct, копируется присвоением (включая LUT: хэндл `Texture` — plain GPU-хэндлы без CPU-рефов, поэтому структура целиком копируется в `SceneFrameSnapshot.post_process`). Снапшот несёт конфиг из update-потока в render без синхронизации.
- Все функции листьев — чистые (конфиг/геометрия in → числа out), без аллокаций, кроме LUT-билдеров (`buildIdentityLutStrip` аллоцирует через переданный аллокатор) и `lutParams` (возвращает структуры, не память).
- `taa_camera_cut`: ставит update-поток, читает/сбрасывает render — через снапшот, гонок нет.
- `setColorGradingLut`/`clearColorGradingLut` — методы конфига (не пасса): вызываются из главного потока до упаковки снапшота.

## Ошибки и краевые случаи

- Ошибок как значений почти нет: все клампы total (`clamped()` не падает). Единственный fallible путь — LUT-билдеры (`OutOfMemory`) и `setColorGradingLut` с невалидной текстурой (не ошибка — тихий сброс в `null` + `lut_enabled = false`).
- `bloom_pyramid_mips` вне `[3, 7]` — кламп, а не игнор: эффект остаётся включённым с ближайшим валидным числом мипов.
- `shaft_steps` клампится к [4,32] (граница шейдерного цикла); `shaft_anisotropy` — к ±0.9 (сингулярность HG-фазы при |g|→1).
- `motion_blur_samples`/`ssr_steps` клампятся к своим диапазонам; нулевые значения невозможны после `clamped()`.
- LUT с `lut_size`, не прошедшим `validLutSize`, или с геометрией strip, не совпадающей с размером, — сброс привязки (шейдер идёт по no-LUT пути, а не сэмплит мусор).
- `taa_camera_cut = true` при выключенном TAA — безвреден (истории нет, сбрасывать нечего).
- Все `*Active` гейты (`glowActive`, `highlightActive`, `shaftActive`) — единственный источник правды для `PostProcessPass` о том, запускать ли пассы; bloom идёт единой пирамидой при `bloom_enabled`.

## Производительность

- Выключенные эффекты стоят ноль GPU (скип ветви + незапущенные пассы) и ноль CPU (гейт до упаковки юниформ).
- Дорогие по GPU (порядок): shafts (реймарш × шаги × low-res + bilateral blur) > bloom-пирамида (mips даун/ап) > SSR (шаги луча 4–64) > motion blur (сэмплы 2–32) > DOF-gather > glow-blur > single-shader bloom. Ориентир: начинайте с quarter-шафтов и 5 мипов пирамиды.
- CPU-математика модуля — O(1) на вызов (веса, UV, джиттер); исключение — `buildIdentityLutStrip` O(size³) однократно при создании LUT.
- TAA держит два history-таргета полного разрешения (`ensureTaaHistory`) — самая заметная видеопамять модуля; `taaReset` при ресайзе обязателен (иначе репроекция из чужого разрешения).
- `clamped()` — дешёвый (скаляры), вызывайте без страха каждый кадр после твиков UI; кэшировать не нужно.
- HDR-showcase: `zig build hdr-showcase` / `run-hdr-showcase`, сцены Studio B/E/Space; конечный прогон `AGATE_HDR_FRAMES=240`, `MSAA4`, `SRGB1`; browser-кадры — query-параметром. Скриншоты — визуальные, не числовая radiance-метрика.

## Смотрите также

- `./passes.md` — `PostProcessPass`, `BloomPass`, `GlowPass`, `VolumetricPass`: GPU-сторона стека.
- `./shaders.md` — `postprocess.glsl`, include-чанки, slang-леги.
- `./frame-pipeline.md` — `Scene.postfx` / `PostFXStack.renderChain`, `ChainParams.post`.
- `./texture.md` — `Texture`, LUT-текстуры, live view.
- `./cameras.md` — jittered view-proj для TAA.
- `./lights.md` — CSM солнца для шафтов.
- `./serialization.md` — `postprocess_persisted`: какие поля переживают сейв.
