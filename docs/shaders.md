# Шейдеры и шейдерные материалы

> Путь: src/agate/shaders/ + src/agate/shader_material/ + build.zig · Импорт: сгенерированные модули (standard_shader, pbr_shader, …) + agate.shader_material · Потоки: build-time (компиляция shdc); runtime — context thread (registerRuntime).

## Что это

Каталог `shaders/` — все GLSL-исходники движка в формате sokol-shdc (`@vs`/`@fs`/`@cs`/`@program`-блоки), каталог `shader_material/` — инструментарий кастомизации (merge hook-сниппеты в шаблоны, `// @include`-препасс), а `build.zig` — таблица сборки «имя модуля → вход/выход/slang». Engine codegen floor: Metal + WGSL. Старые GL/HLSL-леги и WebGL-линк удалены из engine-контракта; native WGPU не заявлен (blocked: cached emdawn headers, нет verified native-прогона — Windows/Linux поддержанными не объявлять). Универсальные Sokol-форки без изменений, их backend-контракт независим. Пользовательские шейдеры подключаются двумя путями: hook-сниппеты в базовые шаблоны (build-time, `user_shader_materials`) и полностью свои `.glsl` через `compileUserShader` из собственного `build.zig` (без правок исходников agate).

Ключевое ограничение, породившее всю механику: sokol-shdc не поддерживает `#include` (проверено: `#include` падает в glslang, флага `-I` у вендорного бинарника нет). Поэтому sharing — текстовая подстановка `// @include "common/<file>"` хост-утилитой `expand_shader_includes` (см. `shader_material/include.zig`) до запуска shdc.

## Быстрый старт

```zig
// 1. Hook-материал (свой эффект на базовом шаблоне). В build.zig проекта:
// (в agate из коробки: ramp_wave, matlib_sky/gradient/grid/triplanar)
pub const user_shader_materials = [_]UserShaderMaterial{
    .{ .name = "my_wave", .snippet = "shaders/my_wave.glsl", .base = .standard },
};
// Сниппет — statement-level GLSL, валидный для Metal + WGSL-легов.
// Хуки шаблона: decls / pre_lighting / post_lighting (имена фиксированы,
// неизвестный хук = Error.UnknownHook на сборке).

// 2. Полностью свой шейдер из downstream build.zig (без правок agate):
const agate_build = @import("agate");
const my_shader = try agate_build.compileUserShader(b, dep_agate, .{
    .name = "my_shader",
    .input = "shaders/my.glsl", // sokol-shdc формат, как standard.glsl
    .target = target, .optimize = optimize, // те же, что у dep agate
});
exe.root_module.addImport("my_shader", my_shader);

// 3. Регистрация в рантайме (context thread!):
const my_mod = @import("my_shader");
_ = try agate.shader_material.registerRuntime(.{
    .name = "my_effect",
    .make_shader = Entry.makeShader, // fn (sg.Backend) sg.Shader over my_mod.myShaderDesc
    .engine_template = false,
    .user_ub = my_mod.UB_my_user_block,
    .params = &my_params, // f32-смещения в user uniform storage
});
const mat = scene.createShaderMaterial("fx", "my_effect") orelse unreachable;
mesh.material = .{ .shader_material = mat };
```

`// @include` в авторском `.glsl`:

```glsl
// @include "common/fullscreen_vs.glsl"
```

## API

### Таблица шейдеров (`build.zig`)

Наборы slang (engine floor: Metal + WGSL):

| Набор | Леги | Кто |
|---|---|---|
| `default_slang` | metal_macos + wgsl | большинство шейдеров |
| `forward_slang` | metal_macos + wgsl | forward-шейдеры (SSBO-кластеры) |
| compute-набор | metal_macos + wgsl | `particle_compute` |
| `engine_shader_slang` | metal_macos + wgsl | пользовательские шейдеры через `compileUserShader`, hook-merge |

Таблица (`ShaderSpec`: `name`, `input`, `output`, `slang?`, `includes`):

| Модуль | Вход | slang | includes |
|---|---|---|---|
| `shader` | `standard.glsl` | forward | да |
| `pbr_shader` | `pbr.glsl` | forward | да |
| `skinned_pbr_shader` | `skinned_pbr.glsl` | forward | да |
| `instanced_shader` | `instanced.glsl` | forward | да |
| `instanced_pbr_shader` | `instanced_pbr.glsl` | forward | да |
| `shadow_shader` | `shadow.glsl` | default | нет |
| `msaa_depth_shader` | `msaa_depth.glsl` | default | нет |
| `skybox_shader` | `skybox.glsl` | default | нет |
| `postprocess_shader` | `postprocess.glsl` | default | да |
| `particle_shader` | `particle.glsl` | default | нет |
| `particle_compute_shader` | `particle_compute.glsl` | compute | нет |
| `ui_shader` | `ui.glsl` | default | нет |
| `ssao_shader` / `ssao_blur_shader` | `ssao.glsl` / `ssao_blur.glsl` | default | да |
| `debug_shader` | `debug.glsl` | default | нет |
| `bloom_down_shader` / `bloom_up_shader` | `bloom_down.glsl` / `bloom_up.glsl` | default | да |
| `glow_extract_shader` / `glow_blur_shader` | `glow_extract.glsl` / `glow_blur.glsl` | default | да |
| `volumetric_raymarch_shader` / `volumetric_blur_shader` | `volumetric_*.glsl` | default | да |
| `outline_shader` | `outline.glsl` | default | нет |
| `probe_mip_shader` | `probe_mip.glsl` | default | нет (осознанно!) |
| `ui3d_panel_shader` | `ui3d_panel.glsl` | default | нет |

Каждый сгенерированный модуль безусловно `@import("math")` (подключается в `build.zig`). shdc компилирует все леги одним вызовом — сломанный Metal/WGSL-лег падает на сборке, никогда молча в рантайме на другой ОС.

Назначение шейдеров (кратко; детали пассов — в `./passes.md`, эффектов — в `./postprocess.md`):

- `standard` / `pbr` / `skinned_pbr` / `instanced` / `instanced_pbr` — forward-материалы (кластерный свет, PCF-тени, зонды; PBR-варианты + BRDF).
- `shadow` / `msaa_depth` — depth-only близнецы (световые vs камерные матрицы, разный bias).
- `postprocess` — весь composite-стек постэффектов одним шейдером.
- `bloom_down/up`, `glow_extract/blur`, `ssao`/`ssao_blur`, `volumetric_raymarch/blur` — многостадийные пассы.
- `particle` + `particle_compute` — GPU-частицы (рендер + stateful compute).
- `skybox`, `probe_mip` (миппинг зондов, fullscreen vs БЕЗ Y-flip), `debug` (визуализация), `ui` (2D-канвас), `ui3d_panel` (панели в мире).

### Include-блоки (`shaders/common/`, README)

| Чанк | Строк | Потребители |
|---|---|---|
| `fullscreen_vs.glsl` | 13 | `bloom_down/up`, `glow_blur/extract`, `ssao`, `ssao_blur`, `volumetric_blur/raymarch`, `postprocess` (9) |
| `cluster.glsl` | 19 | 5 forward-шейдеров |
| `shadow_pcf.glsl` | 248 | те же 5 (`hash01` → `areaLightFactor`) |
| `uv_apply.glsl` | 3 | те же 5 |
| `pbr_brdf.glsl` | 103 | `pbr`, `instanced_pbr`, `skinned_pbr` (3) |
| `channel_select.glsl` | 6 | те же 3 |

Правила: чанки — чистые кодовые фрагменты БЕЗ заголовка-провенанса (иначе ломается byte-identical гарантия: раскрытие сконвертированного шейдера воспроизводит дорефакторный исходник байт-в-байт, выход shdc не меняется). Шеринг документируется в README, не в чанках. Без `#line`: ошибка shdc после директивы указывает РАСКРЫТУЮ строку (`expanded = authored + Σ(include_lines − 1)`; измерено: 13-строчный `fullscreen_vs` сдвигает на +12) — маппинг назад инспекцией.

Осознанно НЕ shared (не «чинить» без ревью): `fs_params`-блоки пяти forward-шейдеров (порядок полей + probe-семантика различаются, переупорядочивание сдвигает все uniform-офсеты); morph-хелперы (`tan_xyz` только в PBR); `shadow` vs `msaa_depth` (идентичная растеризация, разные матрицы/bias); `probe_mip` fullscreen vs (без Y-flip); `particle`/`outline`/`skybox`/`debug`/`ui`/`ui3d_panel`/`particle_compute` (уникальные вершинные стадии); сниппеты `examples/shader_materials/` (тела хуков — мерджатся, не инклудятся).

API препасса (`shader_material/include.zig`):

```zig
pub const Error = error{ ... }; // NotFound, Cycle, EscapeRoot, ...
pub const FileSystem = struct {
    pub const ReadError = error{ NotFound, OutOfMemory, Io };
    pub fn read(self: FileSystem, allocator: std.mem.Allocator, path: []const u8) FileSystem.ReadError![]u8
};
pub fn parseIncludeDirective(line: []const u8) Error!?[]const u8 // только точная форма, иначе null
pub fn expand(allocator: std.mem.Allocator, entry_text: []const u8, entry_name: []const u8, root: []const u8, fs: FileSystem) Error![]u8
```

Директива признаётся только в точной форме (ведущий пробел + `// @include "path"`); вложенные инклуды раскрываются рекурсивно, циклы и выход за root — ошибки сборки. Тулза `expand_shader_includes` (`expand_main.zig`, `pub fn main(init: std.process.Init) !void`) — host-executable, собирается под host с `ReleaseSafe`.

### Hook-материалы (`shader_material/merge.zig`, `shader_material.zig`)

```zig
// merge.zig:
pub const user_slot_count: usize = 8;
pub const user_word_count: usize = user_slot_count * 4;
pub const fs_params_binding: u32 = 3;
pub const vs_params_binding: u32 = 4;
pub const Param = struct { ... }; // имя + offset/comps в user storage
pub const Error = error{ ... };   // UnknownHook, ...
pub const Options = struct { template: []const u8, snippet: ?[]const u8, ... };
pub const Result = struct { ... }; // merged GLSL + params
pub const Stage = enum { vs, fs };
pub const reserved_param_names = [_][]const u8{ ... };
pub fn generateUserParamsGlsl(allocator: std.mem.Allocator, stage: Stage, material_name: []const u8, params: []const Param) ![]u8
pub fn merge(allocator: std.mem.Allocator, opts: Options) Error!Result

// shader_material.zig (runtime):
pub const Base = enum { standard, pbr };
pub const Param = struct { ... };
pub const vs_params_ub: u32 = 0; // {mat4 mvp, mat4 model}
pub const fs_params_ub: u32 = 1; // engine frame uniforms
pub const vs_morph_ub: u32 = 2;  // GPU morphs (только engine-шаблоны)
pub const Entry = struct { ... };
pub const invalid_index: u32 = std.math.maxInt(u32);
pub const max_runtime_entries = 16;
pub const max_external_uniform_vec4: u32 = 2;
pub const max_external_uniform_words: u8 = max_external_uniform_vec4 * 4;
pub const RuntimeDesc = struct { name, make_shader, engine_template, user_ub, user_bytes, params };
pub fn registerRuntime(desc: RuntimeDesc) error{ RegistryFull, DuplicateName, UniformLimitExceeded }!u32
pub fn entryCount() usize
pub fn entry(index: u32) ?*const Entry
pub fn keyForName(name: []const u8) u64  // Wyhash
pub fn indexForName(name: []const u8) ?u32
pub fn entryForKey(key: u64) ?*const Entry
pub const UniformStorage = [merge.user_slot_count][4]f32; // 8 vec4
pub const UniformValue = union(enum) { ... };
pub const SetUniformError = error{ ... };
pub fn findParam(params: []const Param, name: []const u8) ?Param
pub fn defaultUniformStorage(params: []const Param) UniformStorage
pub fn setUniform(storage: *UniformStorage, params: []const Param, name: []const u8, value: UniformValue) SetUniformError!void
pub fn uniformBytes(storage: *const UniformStorage) []const u8
```

Конвейер hook-материала: `merge` (сниппет → хуки `decls/pre_lighting/post_lighting` базового шаблона `standard`/`pbr`; неизвестный хук сниппета — `UnknownHook`) → `expand` инклудов → shdc теми же slang, что движок → модуль `shader_material_registry`, из которого рантайм резолвит регистрации. Параметры сниппета (`params` с f32-офсетами) упаковываются в `UniformStorage` (8 vec4 = 32 слова); wire-лимит внешнего uniform-блока строже: `user_bytes` кратно 16, ≤ 2 vec4 — иначе `UniformLimitExceeded` уже в `registerRuntime`, а не валидацией sokol на draw. Юниформы ставятся через `setUniform` по имени (`findParam` для ручного пути), на draw заливается `uniformBytes`.

Из коробки (`user_shader_materials`): `ramp_wave` (демо), `matlib_sky`, `matlib_gradient`, `matlib_grid`, `matlib_triplanar` — процедурные пресеты матбиблиотеки (см. `./material.md`).

### compileUserShader (`build.zig`)

```zig
pub const engine_shader_slang = sokol.shdc.Slang{ .metal_macos = true, .wgsl = true };
pub const UserShaderSpec = struct {
    name: []const u8, input: []const u8, output: ?[]const u8 = null,
    slang: ?sokol.shdc.Slang = null, // default engine_shader_slang
    target: Build.ResolvedTarget, optimize: std.builtin.OptimizeMode,
};
pub fn compileUserShader(b: *Build, dep_agate: *Build.Dependency, spec: UserShaderSpec) !*Build.Module
```

Внешний путь НЕ идёт через merge/expand (свои инклуды проект организует сам). Резолв sokol/shdc — через `dep_agate.builder`, поэтому downstream делит один инстанс sokol-модуля (без второй зависимости sokol и дублирования sg-состояния). Пустой `name`/`input` → `error.UserShaderBadSpec`. Ручной argv-эквивалент `createModule` задокументирован рядом (`-l <slang> -f sokol_zig`, genver/ifdef/tmpdir выключены как у враппера).

### Drift-тесты (`shader_material/include.zig`)

Осознанные дубли, которые должны эволюционировать синхронно (тесты падают при рассинхроне — это их работа):

- `drift: fs_params probe lane parity` — пять forward-шейдеров держат одинаковый layout probe-полей;
- `drift: depth-only twin programs (shadow vs msaa_depth)` — одинаковые `@program`-наборы;
- `drift: morph helper arity (standard vs PBR variants)` — арность morph-хелперов;
- `drift: probe_mip stays excluded from the fullscreen include` — `probe_mip.glsl` не должен подхватить Y-flip;
- `converted shaders expand fully` — раскрытие любого конвертированного шейдера не оставляет директив и содержит shared-чанки.

## Потоки и владение

- Build-time: всё выше — чистые функции сборки; `expand`/`merge` аллоцируют через переданный аллокатор (владелец — вызывающий таргет сборки).
- Runtime: `registerRuntime` — только context thread (`gpu_thread.assertOnContextThread`), реестр — фиксированный массив на 16 записей (`RegistryFull` сверх), имена уникальны (`DuplicateName`, ключ — Wyhash от имени). `UniformStorage` — value-массив 8×vec4 на материал (копируется, не алиасится).
- Сгенерированные `*_shader.zig`-модули — comptime-импорты (`@import("pbr_shader")` и т.д.); `xxxDesc(backend)` выбирает скомпилированный лег под текущий бэкенд в рантайме.

## Ошибки и краевые случаи

- `parseIncludeDirective` строга к форме: `/* @include */`, `#include`, одинарные кавычки — не директивы, строка уходит в shdc как есть (и падает там, если это был настоящий `#include`).
- Цикл инклудов (прямой или транзитивный, включая само-включение) — ошибка, а не зависание; выход за root — ошибка.
- `merge`: неизвестный хук сниппета — `UnknownHook` (почти наверняка опечатка в имени хука); params без `decls`-хука в шаблоне — ошибка (параметрам некуда приземлиться; `fs`-decls обязателен, `vs`-блок генерится только если вершинный хук их использует).
- Зарезервированные имена параметров (`reserved_param_names`) в сниппете запрещены — коллизия с engine-юниформами.
- `registerRuntime`: `user_bytes == 0`, не кратно 16 или > 32 байт → `UniformLimitExceeded`; `params` с `comps != 1/4` или выходящие за `user_bytes` → та же ошибка.
- Сломанный Metal/WGSL-лег при валидном GLSL — падение сборки (shdc один вызов на все леги). Проверяйте сниппеты на обоих диалектах, а не только в GLSL.
- `probe_mip` без Y-flip: использование общего `fullscreen_vs` там — баг инвертированного зонда, отловленный drift-тестом.

## Производительность

- Цена sharing — ноль в рантайме: подстановка текстовая на сборке, шейдеры компилируются как раньше (byte-identical выход shdc для конвертированных шейдеров).
- Forward-пассы несут кластерные SSBO только в glsl430-леге; на macOS GL 4.1 storage buffers недоступны (см. `./render-pipeline.md` — гейт через `sg.queryFeatures().compute` и `compute.zig`).
- Hook-юниформы: на draw заливается ровно `user_bytes` (≤ 32 байт) — дешевле полного перезалива `fs_params`.
- Время сборки растёт линейно с числом hook-материалов (merge → expand → shdc на каждый); держите count сниппетов низким или выносите вариации в юниформы, а не в материалы.

## Смотрите также

- `./passes.md` — какие пассы используют какие шейдеры.
- `./postprocess.md` — `postprocess.glsl` и мультипассовые шейдеры эффектов.
- `./material.md` — `StandardMaterial`/PBR, `createShaderMaterial`, матбиблиотека.
- `./render-pipeline.md` — бэкенды, slang-леги, `compute.zig`-гейты.
- `./ui.md` — `ui.glsl`, `ui3d_panel.glsl`.
- `./architecture.md` — `gpu_thread` и single-context-thread контракт.
