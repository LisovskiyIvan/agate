# Текстуры

> Путь: src/agate/texture.zig, src/agate/texture/, src/agate/dds.zig, src/agate/exr.zig, src/agate/ktx2.zig · Импорт: agate.Texture, agate.CubeTexture, agate.SkyboxOptions (root.zig) · Потоки: декод на любом потоке/worker, создание sg-image/view/sampler только на context-потоке

## Что это

Текстуры 2D (`Texture`) и кубические (`CubeTexture`): CPU-декодирование популярных форматов, генерация мип-цепочек box-фильтром, sRGB→linear конвертация до фильтрации, блочные форматы без декодирования (DDS BC1–BC7, KTX2 + Basis-транскод), HDR-пути (Radiance HDR, OpenEXR), процедурные скайбоксы и equirect→cube конвертеры. Листья: `color.zig` (sRGB, half-float), `mip.zig` (даунсемпл, размеры), `core.zig` (тип `Texture` + все пути загрузки), `cube.zig` (`CubeTexture`); читатели контейнеров `dds.zig`/`exr.zig`/`ktx2.zig` — топ-модули.

## Быстрый старт

```zig
const agate = @import("agate");

// Из файла: формат определяется по сигнатуре (PNG/JPEG/HDR/EXR/DDS/KTX2).
var tex = try agate.Texture.fromFile(allocator, "assets/brick_albedo.png", .{
    .srgb_to_linear = true, // цветовая карта для освещения
    .mipmaps = true,
});
defer tex.deinit();

// Дата-карта (нормалмап): без sRGB-конвертации!
var nrm = try agate.Texture.fromFile(allocator, "assets/brick_nrm.png", .{});

// Сырые пиксели RGBA8.
var raw = try agate.Texture.fromMemory(allocator, png_bytes, .{});

// Блочный DDS/KTX2 напрямую в GPU (без CPU-декода).
var bc = try agate.Texture.fromDdsFile(allocator, "assets/atlas_bc7.dds", .{});

// Кубический скайбокс из панорамы.
var sky = try agate.CubeTexture.fromEquirectangularFile(allocator, "assets/sky.hdr", 512);
defer sky.deinit();

// Заглушки без GPU-затрат на файл.
const white = agate.Texture.createWhite1x1();
```

## API

### `Texture.Options` и `DecodeOptions` (`texture/core.zig`)

```zig
pub const Options = struct {
    min_filter: sg.Filter = .LINEAR,
    mag_filter: sg.Filter = .LINEAR,
    mip_filter: sg.Filter = .LINEAR, // только при наличии цепочки
    wrap_u: sg.Wrap = .REPEAT,
    wrap_v: sg.Wrap = .REPEAT,
    max_anisotropy: u32 = 4,         // 1..16; дефолт = Babylon DEFAULT_ANISOTROPIC_FILTERING_LEVEL
    mipmaps: bool = true,            // false для SDF/шрифтов/LUT
    srgb_to_linear: bool = false,    // true для albedo/emissive; false для data-карт
};
pub const DecodeOptions = struct {
    gen_mipmaps: bool = true,
    srgb_to_linear: bool = false,
    basis_target: ?ktx2.BasisTarget = null, // null = десктопный .bc7 дефолт
};
```

`max_anisotropy = 4` повторяет Babylon (`Texture.DEFAULT_ANISOTROPIC_FILTERING_LEVEL = 4`; glTF-загрузчик Babylon его не переопределяет, так что все текстуры glTF сэмплируются с 4). Ручка на время загрузки — `SceneLoader.LoadOptions.max_anisotropy: ?u32` (null = дефолт движка, 1 = выключить анизотропию).

Анизотропия > 1 с не-LINEAR min/mag/mip фильтрами молча клампится к 1 (`Texture.effectiveAnisotropy` — требование валидации sokol, Babylon клампит так же по sampling mode); одноуровневые текстуры (без мипов) тоже получают 1. `srgb_to_linear` применяется до генерации мипов — усреднение идёт в линейном пространстве.

### Форматы и точки входа

| Формат | Детект | Декод | Точка входа |
|---|---|---|---|
| PNG / JPEG (stb) | сигнатура | CPU RGBA8 | `fromFile/fromMemory`, `decodeMemory/decodeFile`, `decodeImageMemory/decodeImageFile` |
| Radiance HDR (.hdr) | сигнатура | CPU RGBE→f16 | `decodeHDRMemory/decodeHDRFile`, `fromHDRMemory/loadHDRFile`, `initRawHdr/fromRawHdr` |
| OpenEXR (none/RLE/ZIP/ZIPS) | `exr.sniff` (magic `76 2f 31 01`) | CPU half/float | `fromExrMemory/fromExrFile` (`decodeHDRMemory` роутит по sniff) |
| DDS BC1/BC2/BC3/BC7 | `dds.sniff` | Нет (прямая загрузка блоков) | `fromDdsMemory/fromDdsFile` (обёртки над `dds.decodeBlock2D`) |
| KTX2 блочные (BC1–BC3/BC7, ETC2 RGBA8, ASTC 4x4) | `ktx2.sniff`/`isBlockKtx2` | Нет | `decodeImageMemory` роутит автоматически |
| KTX2 Basis (ETC1S/UASTC) | `isBasisKtx2` + `basisInfo` | Транскод в `basis_target` | `decodeBasis2D`, `preferredBasisTarget(support)`, `basisTargetForCurrentThread()` |
| Сырые RGBA8 / RGBA f16 | — | — | `initRaw/initRawMipped/fromRaw/buildRaw`, `initRawHdr/fromRawHdr` |

Ошибки декода: `DdsDecodeError`/`ExrDecodeError`/`Ktx2DecodeError` (реэкспортированы в root как `DdsDecodeError`, `Ktx2DecodeError`, `ExrDecodeError`), плюс `error{ BlockFormatNotSupportedByBackend }` когда бэкенд не умеет сэмплить точный вариант формата (UNORM vs SRGB различаются — это разные GPU-форматы). CPU-фолбэка для блоков нет by design: без транскодера — громкая ошибка, не тихая порча.

Ключевые функции `Texture`:

```zig
pub fn mipLevelCount(width: u32, height: u32) u32;
pub fn getGpuMemoryBytes(self: *const Texture) usize; // все уровни, через pixelFormatBytes
pub fn deinit(self: *Texture) void;                   // image + view + sampler
pub fn createWhite1x1/createBlack1x1/createFlatNormal1x1() Texture;
pub fn createCheckerboard(...) Texture;
pub fn createDefaultParticleDot32() Texture;
pub fn createParticleDot(allocator, size: u32) !Texture;
pub fn floatToHalfBits(value: f32) u16; // + halfBitsToFloat
pub fn queryBlockSupport() BlockSupport; // снапшот возможностей бэкенда
pub fn sgPixelFormatForBlock(format: ktx2.BlockFormat) sg.PixelFormat;
pub fn fromRawBlock(raw: *const ktx2.RawBlockTexture, options: Options) error{BlockFormatNotSupportedByBackend}!Texture;
```

`BlockSupport` (`bc1_sample/...`, `supportsFormat(fmt)`, `preferred()`) — чистые хелперы выбора цели транскода; снапшот снимается через `sg.queryPixelformat`.

### Мипмапы и цвет (`texture/mip.zig`, `texture/color.zig`)

```zig
pub fn boxDownsampleU8(src, src_w/h, dst, dst_w/h: ...) void; // LDR-цепочки
pub fn boxDownsampleF16(...) void;                            // HDR-цепочки
pub fn checkedFaceBytes(size: u32) !usize;                    // защита от переполнения граней куба
pub fn pixelFormatBytes(format: sg.PixelFormat) usize;        // аппроксимация байт/пиксель (capacity-метрики)
pub fn srgbToLinearU8(value: u8) u8;                          // точная побайтовая таблица
pub fn convertSrgbToLinearInPlace(pixels: []u8) void;
pub fn particleDotAlpha(x, y, center, radius: ...) u8;
```

Полные цепочки до 1×1; блочные текстуры грузятся как authored (без CPU-досинтеза — нужен кодер; GPU клампит LOD к младшему присутствующему уровню). SDF/шрифты: `mipmaps = false`, иначе тонкие штрихи расплываются и край шейдера дрейфует.

### `CubeTexture` (`texture/cube.zig`)

```zig
pub const SkyboxOptions = struct { ... }; // градиент/параметры процедурного неба
pub const CubeTexture = struct {
    pub fn deinit(self: *CubeTexture) void;
    pub fn getGpuMemoryBytes(self: *const CubeTexture) usize;
    pub fn createDefault1x1(color: [4]u8) CubeTexture;
    pub fn initRawFaces(allocator, size: u32, faces: [6][]const u8, generate_mips: bool) !CubeTexture;
    pub fn initRawFacesHdr/initRawFacesMips(...) ...;
    pub fn createProceduralSkybox(allocator, config: SkyboxOptions) !CubeTexture;
    pub fn fromFiles(allocator, face_paths: [6][]const u8) !CubeTexture;
    pub fn fromEquirectangular(allocator, panorama_bytes, face_size: u32) !CubeTexture;
    pub fn fromEquirectangularFile(allocator, file_path, face_size: u32) !CubeTexture;
    pub fn fromEquirectangularHDR(...) ...;
    pub fn convertEquirectangularHDR(...) RawHdrCube;
    pub fn buildRawFacesHdr(...) RawHdrCubeMips;
    pub const RawCubeMips/RawHdrCube/RawHdrCubeMips = ...; // сырые контейнеры с deinit
};
```

Порядок граней — фиксированный engine-порядок (см. источник `fromFiles`); HDR-конвертеры идут через f16-буферы с `boxDownsampleF16`-мипами.

### Ёмкости (capacities)

`getGpuMemoryBytes` (2D и куб) считает все уровни через `pixelFormatBytes` — использовать для бюджетов видеопамяти. Ориентиры: RGBA8 1024² с полной цепочкой ≈ 5.6 МБ; BC7 1024² ≈ 1.4 МБ; RGBA32F дельта-текстуры морфов — отдельно в `./mesh.md`.

## Потоки и владение

`Texture`/`CubeTexture` владеют sg-хендлами (image, view, sampler): создание и `deinit` — только на context-потоке. Декодирование (`decode*`, `buildRaw`, контейнерные `decode`) — чистые CPU-функции, безопасны на worker-потоках; именно их выполняет `UploadQueue`/`io_runner` (см. `./assets.md`). Сырые контейнеры (`RawTexture`, `RawHdrTexture`, `RawBlockTexture`, `DecodedImage`, `RawCubeMips`) владеют CPU-буферами и освобождаются через `deinit(allocator)` на любом потоке. Материальные слоты изначально null (рендерится `default_white`), патч — через `PendingTexture.addTarget`.

## Ошибки и краевые случаи

| Ситуация | Поведение |
|---|---|
| Неизвестная сигнатура | Ошибка декода соответствующего читателя |
| EXR со сжатием вне {none, RLE, ZIPS, ZIP} | `DecodeError` (поддержаны только 4; PIZ/PIZ-варианты — нет) |
| Блочный формат без sample-поддержки бэкенда | `BlockFormatNotSupportedByBackend`, без фолбэка |
| Частичная mip-цепочка в KTX2/DDS | Грузится как есть, LOD клампится GPU |
| `checkedFaceBytes` переполнение | Ошибка вместо обёртки размера |
| Анизотропия с NEAREST-фильтрами | Кламп к 1 |
| Текстура 0×0 | Не создаётся (валидация размеров) |

## Производительность

- Блочные форматы (BC7/ASTC) в ~4 раза легче RGBA8 в памяти и пропускной способности; предпочитать для albedo/ORM-паков на диске и в VRAM.
- Мип-цепочки обязательны для 3D-сцен (без них — алиасинг и промахи кэша текстур); отключать только для SDF/LUT/спрайтов без освещения.
- `srgb_to_linear` — один проход по байтам до даунсемпла; дешевле корректного освещения, чем гамма-артефакты.
- Basis-транскод — офлайн-затрата загрузки (потоковая через UploadQueue, см. `./assets.md`); в кадре не вызывается.

## Смотрите также

- `./material.md` — слоты текстур, sRGB-контракты по слотам
- `./assets.md` — `UploadQueue`, дедупликация, бюджет загрузок на кадр
- `./loader.md` — декод изображений glTF, семплеры, async-текстуры
- `./shaders.md` — семплирование в шейдерах, лимиты слотов
