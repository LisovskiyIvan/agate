# Agate Roadmap vs Babylon.js: открытый бэклог

> Это карта возможностей и открытый бэклог нативного движка **Agate** (Zig 0.16, sokol, Metal/WebGPU/D3D11/GL).
> Исторический журнал завершённых волн реализации очищен. Ниже зафиксировано **только то, что ещё не сделано** или реализовано частично.
> Дополнительный графический план: [docs/graphics-roadmap.md](./docs/graphics-roadmap.md). Архитектурный аудит: [docs/modernization-audit.md](./docs/modernization-audit.md).

**Легенда**

| Знак | Значение |
|:---:|---|
| 🟡 | Реализовано частично / упрощённо (указано, чего не хватает) |
| ❌ | Не реализовано (потенциальный бэклог, не веб-специфика) |
| 🚫 | Не планируется (вне области нативного движка: браузер/JS/веб) |

---

## 🟡 Что сделано частично

| Направление | Что есть в Agate | Чего не хватает (открытые задачи) |
|---|---|---|
| **Камеры и ввод** | ArcRotate, Free, Fly, Follow, Target, union `Camera`, мультикамера/PIP, инерция/сглаживание, CameraRig (dual, quad, CAD, stereoscopic 3D VR) | Touch/pinch жесты, геймпад, наэкранные стики |
| **Свет и тени** | Hemispheric, Directional (CSM до 4 каскадов), Point (до 2 с тенями в атласе), Spot (до 2 с тенями), RectArea (до 2, closest-point без теней), clustered storage до 64 источников | Кластерное освещение сверх 64 источников; тени от всех источников в кластере; ESM; каскадные настройки per-light |
| **PBR и материалы** | Metallic-roughness, LUT IBL, clearcoat/sheen с независимыми UV, anisotropy v1, screen-space refraction (IOR/thickness), SSS v1 | OpenPBR, объёмное/raymarched refraction, transparent recursion/offscreen recovery, физическая SSS/BSSRDF, анизотропные roughness-карты |
| **Прозрачность** | Opaque/cutout/blend, double-sided, `two_sided_lighting`, единый depth-sort обычных и инстансированных мешей | Пиксельный WBOIT (Order-Independent Transparency) |
| **Текстуры** | PNG/JPEG, HDR Radiance, EXR (HALF/FLOAT scanline), DDS (BC1/BC2/BC3/BC7), KTX2 LDR, Basis транскодинг (ETC1S/UASTC $\to$ BC7/ASTC/RGBA32), мипмапы | BC4/BC5/BC6, HDR-16F в KTX2, видеотекстуры, refraction probes |
| **Постобработка** | Linear-HDR pipeline, auto-exposure, tonemapping (ACES, Reinhard, Filmic, AgX, Neutral), film LUTs, bloom pyramid, DoF, motion blur, contact shadows, depth pyramid, SSAO, SSR, TAA (jitter+reprojection+clamp) | TAA под MSAA (сейчас TAA при MSAA отключается); расширенные reactive masks / variance clipping для TAA |
| **Анимация** | Скелетная (до 64 костей, GPU skinning), glTF node TRS-анимации, morph targets (CPU + GPU delta-texture), cubic-spline (Hermite), easing, ретаргетинг | Редактор анимаций |
| **Частицы** | CPU-симуляция + GPU-инстансы, спрайт-листы, sub-emitters, flow maps, CPU-коллизии (сферы cap 8 + ground plane), stateful GPU compute-симуляция | Коллизии частиц с мешами / rigid-body coupling, CCD, нодовый визуальный редактор |
| **Геометрия** | 16 примитивов, terrain, LOD, декали, Polygon, TrailMesh, CSG (BSP union/subtract/intersect), GreasedLine, QEM mesh simplification | CSG2 |
| **glTF** | GLB/glTF 2.0, EXT_meshopt_compression, KHR_mesh_quantization, авто-нормали, KHR_lights_punctual, KHR_texture_transform, clearcoat/sheen, экспорт GLB | Draco-декомпрессия, UV-наборы > 1 в экспорте |
| **Физика** | Box3D v0.1.0 (коллайдеры, compound, суставы, character, rope, ragdoll/vehicle), debug-линии, PBD cloth v1 (ткань cap 4) | Импорт коллайдеров из файлов сцен; soft body за пределами PBD cloth v1 |
| **UI/GUI** | Canvas, SDF-текст + TTF-растеризатор, базовые контролы, 3D world-space панели (до 4), LayoutStack (HStack/VStack/Flex/Grid/Anchors/Docking) | Фокус/состояния клавиатуры, точные TTF advance-метрики, визуальный GUI-редактор |
| **Аудио** | Процедурный синтез, WAV/OGG/MP3 стриминг, SPSC lock-free буферы, кроссфейд, 24 голоса, динамический DAG шин, Doppler, biquad IIR, Freeverb, окклюзия | Микро-чанковый асинхронный I/O менеджер фонового дискового кэширования для сотен одновременных дорожек |
| **Материалы** | PBR + ShaderMaterial (engine-hook + внешний shdc) + Material Library + NodeMaterial v1 (типизированный граф 10 нод $\to$ GLSL codegen) | NodeMaterial v2 (PBR-output, вершинные хуки, vec4-порты, больше текстурных слотов, сериализация графа, runtime-компиляция); визуальный редактор графа |
| **Инструменты** | SceneStats, профилировщик фаз кадра (HTML/MD/Chrome Trace), opt-in GPU frame/phase тайминги, MemorySnapshot, инспектор сущностей в Sandbox | Полноценный in-game визуальный редактор сцены с редактированием на лету |
| **Веб-платформа** | Экспериментальный сборочный таргет wasm32-emscripten + WebGPU для бенча и демо | Полноценная веб-интеграция вне тестового бенча |

---

## ❌ Чего нет (открытый бэклог, не веб-специфика)

### Камеры и ввод
- Нативное тач-управление, геймпад, виртуальные джойстики на экране.
- Встроенное высокоуровневое управление персонажем от первого/третьего лица (помимо физического `CharacterController`).

### Свет и тени
- Тени от point-светов сверх лимита (сейчас до 2 в атласе).
- PCSS / contact hardening для точечных источников, ESM, blur-exponential тени.
- Тени от прямоугольных источников света (RectArea lights).
- Динамический IBL в реальном времени, полноценная физическая атмосфера / volumetric fog (помимо shafts v1).

### Материалы и текстуры
- OpenPBR спецификация.
- Объёмное / raymarched преломление (refraction), прозрачная рекурсия, offscreen recovery.
- Физический подповерхностный рассеиватель (SSS / BSSRDF).
- Анизотропные карты шероховатости (roughness maps).
- Форматы сжатия BC4, BC5, BC6.
- 16-битный float HDR в KTX2 контейнерах.
- Видеотекстуры.
- Refraction probes.
- Пиксельный WBOIT (Order-Independent Transparency).

### Постобработка и эффекты
- SSAA (суперсэмплинг всей сцены).
- Lens flares (блики объектива).
- Snapshot / offscreen high-res рендеринг.
- SSR / SSAO повышенного качества (с temporal filter).

### Геометрия
- Инстансинг с уникальными per-instance материалами (сейчас поддерживается PBR-инстансинг одного материала на батч).
- Морф-таргеты сверх 8 целей.

### Анимация
- Встроенный интерактивный редактор анимаций и анимационных графов.

### Частицы
- Нодовый визуальный редактор систем частиц.
- Непрерывное обнаружение столкновений (CCD) и физический rigid-body coupling частиц.

### Физика
- Импорт готовых физических коллайдеров из glTF/файлов сцены.
- Продвинутые деформируемые тела (soft body) за пределами PBD cloth v1.

### UI/GUI
- Unicode-шейпинг сложных шрифтов (HarfBuzz-подобный) поверх TTF-растеризатора.

### Аудио
- Интерактивный FMOD/Wwise-подобный нодовый секвенсер и звуковой граф.

### Ассеты и форматы
- Декомпрессия Draco для glTF геометрии.
- Поддержка 3D Tiles для стриминга геометрии.

### Архитектура рендера
- Frame graph / render graph для автоматического управления ресурсами, барьерами и временем жизни проходов.
- GPU compute culling геометрии на уровне кластеров/мешлетов.
- Large world rendering (floating origin / перебазирование координат для больших миров).
- Realtime ray tracing / Gaussian splatting.

### Сеть и игровая логика
- Сетевой стек / мультиплеер, репликация состояний, WebSocket/WebRTC.
- Высокоуровневая система Behaviors/Actions, визуальный flow graph логики.
- Локализация текстов и ресурсов.

---

## 🚫 Что не планируется (вне области нативного движка)

Agate — нативный движок на Zig без JavaScript/TypeScript слоя:

- **Веб-платформа и JS:** WebGL-рендер, HTML-канвас, DOM, CSS-интеграция, npm-пакеты, ESM/tree-shaking. (wasm32-emscripten + WebGPU таргет служит только для бенчмарков и веб-демо самого движка).
- **Браузерные механизмы:** Web Workers, браузерный LocalStorage/IndexedDB, CDN-загрузка.
- **Инструменты экосистемы Babylon.js:** Браузерный Playground, Spector.js, веб-редакторы (NME, GUI Editor, Node Particle Editor).
- **XR и веб-медиа:** WebXR (VR/AR в браузере), WebAudio API.
- **JS-рантаймы:** Node.js NullEngine, Babylon Native / React Native мосты.
- **Веб-картография:** Geospatial/Cesium интеграции, завязанные на браузер.

---

## Первичные источники

- Agate: `agate/src/` (`root.zig`, `scene/`, `mesh/`, `material/`, `texture/`, `lights/`, `camera/`, `particles/`, `postprocess/`, `ui/`, `audio/`, `physics/`, `animation/`, `loader/`, `passes/`, `shaders/`).
- Babylon.js: документация и фичи 9.x (2026) — WebGPU/WebGL2, PBR/OpenPBR, Node Material, GUI, Havok, frame graph, clustered lighting, volumetric lighting, Gaussian splatting, geospatial.
