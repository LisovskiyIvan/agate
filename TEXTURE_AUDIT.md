# Аудит текстурной подсистемы agate (wave/textures)

Дата: 2026-09-12. Ворктри: `agate-wt-textures`, база `7c168b2`.
Метод: код-трассировка (texture.zig → draw.zig → *.glsl), sokol_gfx.h / stb_image.h /
cgltf.h (vendored), базлайн `zig build test` — зелёный (exit 0) до изменений.

## Итог (финализировано)

- Фиксы + тесты: `d09991e` feat(textures): real mip/sRGB/filter/slot behavior per the audit.
- Верификация: `zig build test --summary all` → **465/465 passed, 0 failed, exit 0**
  (базлайн 451 + 14 новых golden-тестов); `zig build` → exit 0 (pbr/instanced_pbr/
  skinned_pbr перекомпилированы sokol-shdc для glsl410/metal_macos/hlsl5);
  `zig build fmt` → exit 0; рабочее дерево чистое.
- Примечание: строка `failed command: .../test` в выводе build-раннера —
  pre-existing артефакт (воспроизводится на базовом `f7f7d11`, 451/451, exit 0).

## Сводная таблица

| # | Позиция | Было | Статус | Доказательство |
|---|---------|------|--------|----------------|
| 1 | Мипмапы 2D end-to-end | работают | ЧЕСТНО | `buildRaw` (texture.zig:167) генерирует цепочку; `fromRaw` (texture.zig:196-228) грузит ВСЕ уровни: `img_desc.num_mipmaps = raw.num_levels` + `img_desc.data.mip_levels[m]` в цикле — классического бага «в GPU уходит только уровень 0» НЕТ. Сэмплер: `mipmap_filter = LINEAR` при >1 уровне; sokol lod-диапазон по умолчанию 0..FLT_MAX. Тест `decodeMemory returns RGBA levels...` проверяет num_levels/размеры |
| 2 | Мипмапы LDR cube | работают | ЧЕСТНО | `CubeTexture.initRawFaces` (texture.zig:689-777): цепочка по 6 граней, `img_desc.data.mip_levels[m] = ...` ДО `makeImage`, `mipmap_filter = LINEAR` |
| 3 | Мипмапы HDR cube (RGBA16F) | только 1 уровень, а IBL-шейдер делает `textureLod(lod≤7)` | ОБМАНКА → исправлено | `initRawFacesHdr` (texture.zig:1090) — `num_mipmaps = 1`; pbr.glsl:561-564 (`max_lod = 7.0`) — roughness-префильтр и irradiance молча вырождаются в уровень 0. Исправлено: f16 box-мипы генерируются и загружаются |
| 4 | sRGB/гамма LDR color-текстур | PNG/JPEG сэмплируются «как есть», шейдеры считают их линейными | ОБМАНКА → исправлено (по флагу) | Ни одной sRGB-конверсии в texture.zig; postprocess.glsl тонмапит (ACES/Reinhard), но НЕ гамма-кодирует. HDR (RGBA16F) линейный — ок. Исправлено: CPU-конверсия srgb→linear при загрузке color-слотов (флаг `srgb_to_linear`), НЕ для data-текстур |
| 5 | wrap_u/v, min/mag filter | применяются | ЧЕСТНО | `Texture.Options` → `sg.makeSampler` в `initRaw`/`fromRaw` (texture.zig:105, 213) |
| 6 | mip_filter (настройка) | не было; glTF mip-половина min_filter игнорировалась | ОБМАНКА → исправлено | `Options.mip_filter` добавлен; loader/materials.zig:152-156 мапил {9984,9986}→NEAREST / {9985,9987}→LINEAR только min-половину; mip-часть (9984/9985→NEAREST) терялась |
| 7 | Анизотропия | «в sokol недоступна» — НЕВЕРНО | исправлено (экспонировано) | sokol master (`sokol_gfx.h:3786`) имеет `max_anisotropy` 1..16, валидация требует LINEAR-фильтры. Добавлено в `Options` с guard-ом |
| 8 | lod bias | отсутствует и в sokol | ЧЕСТНО N/A | в `sg_sampler_desc` нет lod bias |
| 9 | lod_min/lod_max | есть в sokol (дефолт 0..FLT_MAX), не экспонированы в Options | ЧЕСТНО (не экспонировано, задокументировано) | `sokol_gfx.h` sg_sampler_desc |
| 10 | Слоты PBR end-to-end: albedo/normal/MR/emissive/AO | сэмплируются во всех трёх PBR-шейдерах | ЧЕСТНО | pbr.glsl:423/443/438/580/553; instanced_pbr.glsl:348/361/356/498/471; skinned_pbr.glsl:413/426/421/558/533. Биндинги с фолбэками: draw.zig:93-103, 463-467 (white / flat-normal). standard/instanced — только diffuse (по дизайну) |
| 11 | AO: пакованная ORM (R-канал) | поддержана | ЧЕСТНО | `texture(occlusion_tex).r` + `pbr_factors.z` (occlusion_strength): pbr.glsl:553-554. MR: G=roughness, B=metallic (glTF-раскладка): pbr.glsl:439-440 |
| 12 | Normal scale | отсутствует полностью (ни поля, ни юниформа; glTF `normalTexture.scale` терялся) | ОБМАНКА → исправлено | cgltf парсит scale (cgltf.h:3861 дефолт 1.0), loader не читал; в шейдере `map_n = tex*2-1` без масштаба. Исправлено: `PBRMaterial.normal_scale` + юниформ `normal_scale` (append-last после alpha_cutoff) + loader |
| 13 | Пер-слотовые сэмплеры PBR | 4 data-текстуры сэмплируются САМПЛЕРОМ ALBEDO | ОБМАНКА → исправлено | draw.zig:104/474: `bind.samplers[SMP_smp] = albedo_tex.sampler` — один самплер на все 5 слотов; собственные samplers normal/MR/AO/emissive создавались и игнорировались. Исправлено: `data_smp` (binding 5) в 3 PBR-шейдерах, приоритет normal→MR→AO→emissive |
| 14 | PNG 8-bit (gray/palette/interlaced/RGBA) | корректно (stb, req_comp=4) | ЧЕСТНО | stb конвертит в RGBA8; gray реплицируется в RGB; interlace поддержан stb. Тест: font_sdf.png + новые синтез-фикстуры |
| 15 | PNG 16-bit | без порчи: stb берёт СТАРШИЙ байт | ЧЕСТНО (задокументировано) | stb_image.h:1200 `(orig[i] >> 8) & 0xFF` — точность 8 бит, байты не перепутаны. Golden-тест 0xABCD→0xAB |
| 16 | JPEG (YCbCr→RGB) | корректно (stb) | ЧЕСТНО | стандартный путь stb; тест декодирования JPEG-фикстуры |
| 17 | HDR (.hdr) | линейный RGBA16F | ЧЕСТНО | stbi_loadf → f32→f16; точные golden-тесты уже были (0x3800/0x3C00) |
| 18 | KTX2 / BasisU | НЕТ поддержки | ЧЕСТНО НЕ СДЕЛАНО | нужен транскодер (basis_universal/KTX-Software) — read-only zig-pkg, новая C-зависимость вне минимального скоупа. Документировано |
| 19 | glTF-сэмплеры: wrap_s/wrap_t, mag/min | применяются | ЧЕСТНО | loader/materials.zig:134-157, cgltf-энумы 33071/33648/10497, 9728/9729 |
| 20 | glTF KHR_texture_transform (uv offset/rotate/scale) | игнорируется | ЧЕСТНО НЕ СДЕЛАНО | требует uv-матрицу в 5 шейдерах + v_uv2 — вне минимального скоупа, задокументировано |
| 21 | emissive_factor / emissive_color | работает (loader поднимает white при нулевом факторе с текстурой) | ЧЕСТНО | loader/materials.zig:248-260 |

## Ключевые решения по фиксам (Фаза 2)

1. **sRGB — CPU-конверсия при загрузке, не sRGB-пиксель-формат.** Выбор: (а) работает одинаково на всех бэкендах sokol; (б) конверсия ДО билда мипов ⇒ боксовый фильтр усредняет в линейном пространстве (корректнее); (в) флаг опционален (`Options.srgb_to_linear`, дефолт false) — существующие вызовы (UI/SDF-шрифты, частицы, процедурные) не меняют вид. glTF-лоадер включает флаг для albedo и emissive, НЕ для normal/MR/AO (data-текстуры уже линейные). Полный «линейный рендер» (гамма-энкод на выходе + линейные light colors) — политика рендерера, сознательно оставлена как есть (тонмаппинг без энкода), задокументировано.
2. **data_smp (per-slot sampler)**: приоритет источника normal→MR→occlusion→emissive→albedo (у normal-мапы wrap критичнее всего). Binding 5, чтобы не пересекаться с vs `morph_smp` (binding 4, sokol требует общий пул слотов между стадиями — см. комментарий в pbr.glsl).
3. **HDR-cube мипы**: box-фильтр в f32 над f16-текселями (конверсия через floatToHalfBits/halfBitsToFloat), цепочка до 16 уровней, тот же контрак что у LDR-куба.
4. **16-бит PNG** — порчи не было, поведение задокументировано и закреплено тестом (старший байт).
5. **Анизотропия**: `Options.max_anisotropy` (дефолт 1 = без изменений); при не-LINEAR фильтрах молча зажимается до 1 (sokol-валидация иначе фейлит makeSampler).

## Честно НЕ сделано (и почему)

- **KTX2/BasisU**: требует C-транскодер; zig-pkg read-only, новых зависимостей не добавляем.
- **Полный linear-рендер** (sRGB render target / гамма-энкод в postprocess): меняет картинку ВСЕХ примеров и пользовательских шейдеров; это отдельная веха рендерера, а не текстурный фикс. Загрузочная конверсия srgb→linear готовит почву.
- **KHR_texture_transform**: uv-матрица в шейдерах (5 штук × 3 слэнга) + расширение vertex-контракта.
- **lod_min/lod_max в Options**: не было запросов, sokol-дефолты (0..FLT_MAX) корректны; экспонирование тривиально при необходимости.
- **Анизотропия для HDR-cube/дефолтных текстур**: применена только там, где есть Options.

---

# Волна wave/ktx2 (2026-09-12): KTX2-контейнер, KHR_texture_transform, каналы

Ворктри: `agate-wt-ktx2`, база `8506814` (финал wave/textures).
Коммиты: `260c2c7` feat(ktx2), `0ed47e0` feat(material) (transform + каналы).

## Сводная таблица

| # | Позиция | Было | Стало | Доказательство |
|---|---------|------|-------|----------------|
| K1 | KTX2: контейнер + несжатые LDR-форматы | НЕТ | ДЕЛАЕТ | `src/agate/ktx2.zig`: sniff/decode2D/decodeCube; vkFormat 9/15 (R8), 16/22 (R8G8), 37/43 (R8G8B8A8), 44/50 (B8G8R8A8), 51/57 (A8B8G8R8_PACK32, тот же LE-байтовый порядок) — UNORM и SRGB; только supercompressionScheme 0 (NONE); заголовок 80Б + LevelIndex (24Б/уровень, от крупного к мелкому); жёсткая валидация `byteLength == uncompressedByteLength == w*h*texelBlockSize*faceCount`, границ файла, levelCount<=16, typeSize==1 |
| K2 | KTX2 mip-цепочки | НЕТ | ДЕЛАЕТ | цепочки из файла грузятся как авторские (генерация поверх авторской молча теряла бы данные); single-level файл с `gen_mipmaps` → движковая box-цепочка (`Texture.buildRaw`, теперь pub) |
| K3 | KTX2 cube faces | НЕТ | ДЕЛАЕТ | faceCount 6, порядок граней KTX2 (+X,-X,+Y,-Y,+Z,-Z) совпадает с движковым; GPU-free `decodeCube` -> `CubeTexture.RawCubeMips` -> `CubeTexture.initRawFacesMips` (полная авторская цепочка, в отличие от `initRawFaces`, который регенерирует) |
| K4 | KTX2 в glTF-пайплайне | НЕТ | ДЕЛАЕТ (2D) | `Texture.decodeMemory`/`decodeFile` снифят 12-байтовую магию KTX2 -> .ktx2-изображения (embedded buffer view или URI) грузятся прозрачно; `decodeFile` читает файл в память; cube-KTX2 в 2D-пути — error.UnsupportedFaceCount (не молча) |
| K5 | KHR_texture_transform (uv offset/rotate/scale) | игнорировался | ДЕЛАЕТ (texcoord0) | `material.UvTransform` (R(rot)*diag(scale)+offset, packing matrixRows/offsetPacked); 5 полей в PBRMaterial (albedo/normal/MR/emissive/occlusion) + diffuse в StandardMaterial; `uv_matrix`/`uv_offset` appended-last в fs_params всех 5 forward-шейдеров (pbr, skinned_pbr, instanced_pbr, standard, instanced), `uvApply()` на каждой выборке; identity-юниформы = no-op, картинка без расширения не меняется; loader/materials.zig читает cgltf `has_transform/offset/rotation/scale` |
| K6 | Канальное чтение (AO/MR) | жёстко R/G/B | API ДЕЛАЕТ | `material.Channel {r,g,b,a}` + `occlusion_channel`/`roughness_channel`/`metallic_channel` (дефолты = glTF-конвенции); юниформ `channel_selectors` (append-last); шейдерный `channelSelect` через константные ветки (SPIRV-Cross не умеет динамический индекс компонент — тот же приём, что morphWeight); glTF выбора канала не даёт — это API для ручных материалов |
| K7 | DDS | НЕТ | НЕ В СКОУПЕ (зафиксировано) | см. «Честно НЕ сделано» этой волны |

## Ключевые решения

1. **Нормализация в RGBA8 на CPU.** Поддержанные vkFormat-ы раскладываются в RGBA8-байты движка (BGRA свизлится, R8 реплицируется в RGB как grayscale-PNG, RG8 -> RG01+A=255) — KTX2 переиспользует существующий upload-путь (`Texture.RawTexture`/`fromRaw`) без новых пиксельных форматов. Расширение на 16F/упакованные форматы потребовало бы mipped-f16 2D-uploader'а — отдельная работа.
2. **DFD/KVD не интерпретируются** — числовой vkFormat один-в-один задаёт раскладку текселей для поддержанного подмножества; блоки пропускаются по смещениям. Чтение DFD (colorModel/primaries) имеет смысл только вместе с широким форматным покрытием.
3. **sRGB-политика.** У собственного API ktx2 `DecodeOptions.srgb_to_linear: ?bool = null` = auto (конвертируются ровно _SRGB-форматы по тегу формата). Сниф-маршрут из `Texture.decodeMemory` передаёт булев флаг слота как есть (glTF: color-слоты true, data-слоты false) — поведение консистентно с PNG-путём волны wave/textures, data-слоты никогда не конвертируются «сюрпризом».
4. **Transform: юниформ-матрицы per-slot** (вариант (а), 2x vec4 на слот = 160Б на PBR-draw) — против (б) CPU-предеформации UV (невозможна для shared-атрибутов меша) и (в) vertex-паковки (доп. varyings на 5 слотов дороже в FS-интерполяции и всё равно требует FS-ветки для cutout-альфы). 160Б на фоне существующего fs_params (4 cascade mat4 + массивы света) — шум. Лимитация: glTF `texCoord > 0` фолбэчится на texcoord0 (второго UV-сета в движке нет), задокументировано в UvTransform и лоадере.
5. **Каналы: юниформ-маски, не comptime-ветки.** Компиляция вариантов шейдера per-channel комбинацию умножила бы пайплайны; юниформ + 3 константные ветки дешевле порога и переносимо (GL410/Metal/HLSL5).

## Честно НЕ сделано (и почему)

- **BasisLZ / Zstandard / Zlib суперкомпрессия и BC/ETC/ASTC в KTX2**: нужен транскодер (basis_universal / KTX-Software); zig-pkg read-only, добавление новой C-зависимости — отдельная веха (future work). Файлы с scheme != 0 и блочными форматами отбрасываются с точными ошибками (`UnsupportedSupercompression` / `UnsupportedVkFormat`), мусорного декода нет.
- **HDR KTX2 (R16G16B16A16_SFLOAT и другие 16/32-битные форматы)**: движковый HDR-2D путь одноуровневый (`initRawHdr`, num_mipmaps=1); честной загрузки авторских 16F-цепочек нет. Cube-IBL env-map в KTX2 — естественный первый клиент, когда появится.
- **3D-текстуры (pixelDepth>1) и массивы (layerCount>0)**: движок их не грузит вовсе; KTX2-ридер отклоняет их явно.
- **DDS — вне скоупа сознательно (зафиксировано)**: отдельный контейнер с собственным реестром FourCC/DXGI-форматов; без транскодера (см. Basis) DDS дал бы лишь второй несжатый контейнер рядом с KTX2. Кандидат в бэклог вместе с KTX2-транскодингом, не этой волной.
- **KHR_texture_transform: texCoord > 0 (второй UV-сет)**: требует v_uv2 varying + расширение vertex-контракта в 5 шейдерах x 3 слэнга; roadmap.
- **Каналы для color-слотов (albedo/emissive)**: цветовые слоты по смыслу RGBA; выбор одиночной полосы сценария не имеет — API ограничен data-слотами.

## Верификация

- `zig build test --summary all` -> **481/481 passed, 0 failed** (базлайн 465 + 16 новых: 10 ktx2, 2 material, 2 loader, 2 draw-контракт).
- `zig build` -> exit 0; все 5 forward-шейдеров перекомпилированы sokol-shdc под glsl410/metal_macos/hlsl5 с новыми юниформ-блоками.
- `zig build fmt` -> exit 0.
- KTX2-фикстуры синтезируются в тестах in-memory (билдер TestKtx2: заголовок + LevelIndex + dummy-DFD + данные с mipPadding по spec); внешних .ktx2 файлов нет. Прочие проверки: swizzle BGRA, расширение R8/RG8, auto/force/forbid sRGB, отклонение всех неподдерживаемых форм контейнера (BasisLZ/Zstd/Zlib, BC3, RGBA16F, 3D, массивы, faceCount 3, 16-битный typeSize, 17 уровней, усечённые данные).
