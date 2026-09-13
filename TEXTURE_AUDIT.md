# Аудит текстурной подсистемы agate (wave/textures)

Дата: 2026-09-12. Ворктри: `agate-wt-textures`, база `7c168b2`.
Метод: код-трассировка (texture.zig → draw.zig → *.glsl), sokol_gfx.h / stb_image.h /
cgltf.h (vendored), базлайн `zig build test` — зелёный (exit 0) до изменений.

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
