# UI

> Путь: src/agate/ui.zig + src/agate/ui/ + src/agate/ttf/ · Импорт: agate.ui (root.zig: UICanvas, UIVertex, UIStyle*, UITheme, LayoutStack, …) · Потоки: главный/рендер (CPU-кадр → GPU-буферы, same-frame guard).

## Что это

Модуль `ui` — immediate-mode 2D-интерфейс движка: `UICanvas` принимает команды рисования в течение кадра (прямоугольники, текст, кнопки, слайдеры, скроллы, ввод текста), складывает их в CPU-батч и одним проходом заливает в GPU-буферы (`sg`) и рисует поверх 3D-сцены. Стиль — CSS-подобный каскад без парсинга строк в рантайме; раскладка — контейнеры `LayoutStack` (стеки, flex, grid, anchors, docking); текст — два пути: встроенный растровый SDF-атлас и TrueType через модуль `ttf`. Трёхмерные панели (плавающие экраны в мире) живут отдельно в `scene/gui3d_layer.zig` (`Ui3dPanel`, см. конец файла) и рисуются шейдером `ui3d_panel.glsl`.

Фасад `ui.zig` реэкспортирует листья `ui/` без изменения публичного API:

| Лист | Ответственность |
|---|---|
| `canvas.zig` | владелец `UICanvas`: состояние, ресурсы, жизненный цикл, форвардеры |
| `draw.zig` | примитивы и чистая батч-математика; `UIVertex` |
| `text.zig` | SDF-атлас, TrueType-путь, `GlyphUV`, `getGlyphUV`, измерение |
| `widgets.zig` | stateless immediate-контролы (кнопки, чекбоксы, слайдеры, дропдауны, скроллбары, ввод текста) |
| `input_state.zig` | хит-тест, `ScrollState`, `TextInputState` |
| `style.zig` | каскад стилей, retained-переходы, стилизованное рисование |
| `font.zig` | создание атласа шрифта, TrueType-оверрайд (`TtfFont`) |
| `gpu.zig` | пара буферов, загрузка/отрисовка, same-frame guard, `render` |
| `stack.zig` | `LayoutStack` — контейнеры раскладки + grid-хелперы |
| `layout.zig` | солверы sizing/dock/flex/grid (`UISize`, `UIAnchor`, `UIDock`, `solveFlex`, `solveGridTracks`, …) |
| `types.zig` / `theme.zig` / `transition.zig` | значения стилей, `UITheme`, математика переходов (`lerpStyle`, `styleEql`) |
| `css_parser.zig` | CSS-подмножество: `parseCss`, `loadThemeFile`, `CssTheme` |

Модуль `ttf/` — автономный TrueType-парсер без внешних зависимостей (см. раздел «Текст»).

## Быстрый старт

```zig
const agate = @import("agate");

var canvas = try agate.ui.UICanvas.init(alloc);
defer canvas.deinit();

// Ввод: мышь каждый кадр (координаты, кнопка, клик).
canvas.setInput(mouse_x, mouse_y, mouse_down, clicked);
canvas.begin(); // сброс батча, начало кадра

// Низкоуровневые примитивы:
canvas.drawPanel(10, 10, 300, 200, bg, border, 1.0);
canvas.drawText("Здоровье: 100", 20, 30, 16.0, white);

// Контейнерная раскладка: вертикальный стек + кнопка возвращает клик.
var stack = agate.ui.LayoutStack.init(&canvas);
if (stack.beginVStack(.{ 10, 10, 300, 400 }, .{})) {
    if (stack.button("Играть", .{})) startGame();
    if (stack.button("Выйти", .{})) quit();
    stack.end();
}

// Заливка в GPU и отрисовка поверх кадра (screen_w/h — размер экрана).
canvas.render(screen_w, screen_h);
```

Стилизованная кнопка с классом и темой из CSS:

```zig
canvas.setStyleClass("danger", .{ .normal = .{ .bg = .{ 0.6, 0.1, 0.1, 1.0 } }, ... });
const css = try agate.ui.parseCss(alloc, ".btn { background: #336699; }");
canvas.applyCssTheme(css);
canvas.drawStyledButton("Удалить", .{ 10, 60, 160, 36 }, 15.0, .{ .class_name = "danger" });
```

TrueType-шрифт вместо встроенного атласа:

```zig
var font = try agate.ttf.TtfFont.init(alloc, ttf_bytes, pixel_size);
defer font.deinit();
canvas.setFontTtf(&font);
canvas.drawText("Привет, мир!", x, y, 18.0, white); // та же строка, другой атлас
canvas.clearFontTtf();
```

## API

### Canvas: жизненный цикл и ввод (`ui/canvas.zig`)

```zig
pub fn init(allocator: std.mem.Allocator) !UICanvas
pub fn initCpuOnly(allocator: std.mem.Allocator) UICanvas
pub fn deinit(self: *UICanvas) void
pub fn begin(self: *UICanvas) void
pub fn setInput(self: *UICanvas, mx: f32, my: f32, is_down: bool, is_clicked: bool) void
pub fn render(self: *UICanvas, screen_w: f32, screen_h: f32) void
pub fn makeFontTexture(allocator: std.mem.Allocator) !Texture
```

`initCpuOnly` — канвас без GPU-ресурсов (тесты, headless). `begin` обязателен каждый кадр до любых команд. `setInput` — единственный канал ввода; виджеты от кадра к кадру stateless, состояние (фокус, драг слайдера, текст) хранится в `input_state`.

Хит-тест и геометрия-утилиты:

```zig
pub fn isPointInRect(px: f32, py: f32, x: f32, y: f32, w: f32, h: f32) bool
pub fn sliderValueAt(x: f32, w: f32, mouse_x: f32) f32
pub fn checkboxHitRect(x: f32, y: f32, size: f32) [4]f32
pub fn measureText(text: []const u8, font_size: f32) Vec2
pub fn measureTextCurrent(self: *const UICanvas, text: []const u8, font_size: f32) Vec2
```

### Примитивы (`ui/draw.zig`, через методы канваса)

```zig
pub const UIVertex = ui_draw.UIVertex; // позиция + uv + цвет вершины батча
pub fn addQuad(...) void
pub fn drawRect(self: *UICanvas, x: f32, y: f32, w: f32, h: f32, color: Color4) void
pub fn drawRectOutline(self: *UICanvas, x: f32, y: f32, w: f32, h: f32, thickness: f32, color: Color4) void
pub fn drawRectRoundedFill(self: *UICanvas, x: f32, y: f32, w: f32, h: f32, radius: f32, color: Color4) void
pub fn drawRectRoundedOutline(self: *UICanvas, x: f32, y: f32, w: f32, h: f32, radius: f32, thickness: f32, color: Color4) void
pub fn drawPanel(self: *UICanvas, x: f32, y: f32, w: f32, h: f32, bg_col: Color4, border_col: Color4, border_width: f32) void
pub fn drawLine(self: *UICanvas, x0: f32, y0: f32, x1: f32, y1: f32, thickness: f32, color: Color4) void
pub fn lineCorners(x0: f32, y0: f32, x1: f32, y1: f32, thickness: f32) [4][2]f32
pub fn drawProgressBar(self: *UICanvas, x: f32, y: f32, w: f32, h: f32, progress: f32, bg_col: Color4, fill_col: Color4) void
```

Прямоугольники — ось--aligned квады в пикселях экранного пространства, начало в левом верхнем углу; скругление — отдельный путь с радиусом (клампится к половине минимальной стороны).

### Виджеты (`ui/widgets.zig`, методы канваса — те же сигнатуры с `self`)

```zig
pub fn drawButton(canvas: anytype, text: []const u8, x: f32, y: f32, w: f32, h: f32, font_size: f32, is_hovered: bool, is_pressed: bool) void
pub fn drawBadge(...) void
pub fn drawCheckbox(canvas: anytype, x: f32, y: f32, size: f32, checked: bool, is_hovered: bool, label: ?[]const u8, label_size: f32) void
pub fn drawSlider(canvas: anytype, x: f32, y: f32, w: f32, h: f32, value: f32, is_hovered: bool, is_dragging: bool) f32
pub fn drawDivider(canvas: anytype, x: f32, y: f32, w: f32, thickness: f32, color: Color4) void
pub fn drawArrowDown(canvas: anytype, x: f32, y: f32, size: f32, color: Color4) void
pub fn drawDropdown(...) // + хелперы: dropdownItemHeight, dropdownItemRect, dropdownHit
pub fn drawScrollbar(canvas: anytype, track: [4]f32, content_h: f32, view_h: f32, offset: f32) void
pub fn drawTextInput(canvas: anytype, rect: [4]f32, state: *const TextInputState, focused: bool, font_size: f32) void
```

Виджеты рисуют, но не обрабатывают ввод сами: вызывающий код делает хит-тест (`isPointInRect`) и передаёт `is_hovered`/`is_pressed`. Исключение — `LayoutStack`-обёртки (`button`, `checkbox`, `slider` возвращают события/значения, см. ниже). `drawSlider` возвращает значение после драга. Состояние скролла и текстового ввода:

```zig
pub const ScrollState = ...; // + scrollClamp(state: *ScrollState, delta: f32) void
pub const TextInputState = ...; // буфер, курсор, выделение
pub fn scrollOffsetForItem(offset: f32, item_y: f32, item_h: f32, view_h: f32, content_h: f32) f32
pub fn scrollbarThumbRect(track: [4]f32, content_h: f32, view_h: f32, offset: f32) [4]f32
```

### Раскладка: LayoutStack, flex/grid, anchors/docking (`ui/stack.zig`, `ui/layout.zig`)

```zig
// stack.zig — дженерик по типу канваса; фасад пинит инстанциацию UICanvas.
pub const LayoutStack = ui_stack.LayoutStack(ui_canvas.UICanvas);
pub const LayoutAlign = enum { start, center, end };
pub const LayoutAlignCross = enum { start, center, end, stretch };
pub fn layoutAlignOffset(extent: f32, size: f32, alignment: LayoutAlign) f32
pub const LayoutGridSpec = struct {
    pub fn cellRect(self: LayoutGridSpec, col_in: usize, row_in: usize) [4]f32
    pub fn placedRect(self: LayoutGridSpec, index: usize, w: f32, h: f32) [4]f32
};
pub const LayoutFlowOptions = struct { ... };  // padding/spacing/align для стеков
pub const LayoutGridOptions = struct { ... };  // колонки/строки/отступы
pub const LayoutFlexOptions = struct { ... };  // веса
// Опции виджетов в стеке:
pub const LayoutLabelOptions / LayoutButtonOptions / LayoutCheckboxOptions
pub const LayoutSliderOptions / LayoutProgressOptions / LayoutDividerOptions / LayoutBadgeOptions

// Методы стека:
pub fn init(canvas: *Canvas) Self
pub fn reset(self: *Self) void
pub fn beginHStack(self: *Self, rect: [4]f32, opts: LayoutFlowOptions) bool
pub fn beginVStack(self: *Self, rect: [4]f32, opts: LayoutFlowOptions) bool
pub fn beginFlex(self: *Self, rect: [4]f32, opts: LayoutFlexOptions) bool
pub fn beginGrid(self: *Self, rect: [4]f32, opts: LayoutGridOptions) bool
pub fn end(self: *Self) void
pub fn innerRect(self: *const Self) [4]f32
pub fn place(self: *Self, w: f32, h: f32) [4]f32
pub fn placeAligned(self: *Self, w: f32, h: f32, align_cross: ?LayoutAlignCross) [4]f32
pub fn placeSize(self: *Self, w: UISize, h: UISize) [4]f32
pub fn placeFlex(self: *Self, weight: f32) [4]f32
pub fn spacer / pub fn spacerWeight(self: *Self, weight: f32) [4]f32
pub fn anchor(self: *Self, w: f32, h: f32, anchor_pt: UIAnchor, margin: UIEdges) [4]f32
pub fn dock(self: *Self, dock_side: UIDock, size: f32, margin: UIEdges) [4]f32
pub fn label(self: *Self, text: []const u8, opts: LayoutLabelOptions) void
pub fn button(self: *Self, text: []const u8, opts: LayoutButtonOptions) bool // true = клик
pub fn checkbox(self: *Self, label_text: ?[]const u8, checked: *bool, opts: LayoutCheckboxOptions) bool
pub fn slider(self: *Self, value: f32, min_val: f32, max_val: f32, opts: LayoutSliderOptions) f32
pub fn progressBar(self: *Self, fraction: f32, opts: LayoutProgressOptions) void
pub fn divider / pub fn badge / pub fn placeBox / pub fn placeGridSpan(...)
```

```zig
// layout.zig — чистые солверы (без канваса, тестируемы отдельно):
pub const UISize = union(enum) { px: f32, percent: f32, auto, flex: f32, ... };
pub const UIEdges = struct { ... };
pub const UIAnchor = enum { ... }; // углы + центры + stretch
pub const UIDock = enum { ... };   // left/right/top/bottom/fill
pub fn anchorRect(...) [4]f32
pub fn dockRect(...) [4]f32
pub const FlexDirection / JustifyContent / AlignItems = enum {...};
pub const LayoutItem = struct { ... };
pub fn solveFlex(...) ...
pub const GridTrack = ...;
pub fn solveGridTracks(...) ...
pub const AdvancedGridSpec = ...;
```

Стек — это курсор раскладки внутри `rect`: `beginVStack`/`beginHStack` открывают контейнер (возвращают `false`, если места нет — содержимое пропускается), `place*` выдают следующий rect, `end` закрывает. Стеки вкладываются. `anchor`/`dock` позиционируют относительно текущего контейнера — удобно для тулбаров и прибитых к краям панелей.

### Темы, стили, переходы, CSS (`ui/types.zig`, `theme.zig`, `transition.zig`, `css_parser.zig`)

Каскад разрешения одного конкретного `UIStyle` на вызов:

`fallback движка < default темы (per UIStyleKind) < именованный класс < per-call override < state-дельты (:hover/:active/...)`

```zig
pub const UIState = ...;          // hover/active/focus/disabled биты
pub const UIStyle = ...;          // полностью resolved (все поля конкретные)
pub const UIStyleOverride = ...;  // частичный (null = наследовать)
pub const UIStyleSet = ...;       // normal + per-state дельты
pub const UIStyleKind = enum {...};// button/panel/badge/... — ключ в теме
pub const UIStyleRequest = ...;   // kind + class + override + state
pub const UIStyledOptions = ...;  // то же для drawStyled*
pub const UIStyleClass = ...;
pub const UIBoxStyle = ...;
pub const max_style_classes = ...;
pub const UITheme = struct {
    pub fn defaults() UITheme
    pub fn parseCss(allocator: std.mem.Allocator, text: []const u8) css.ParseError!css.CssTheme
    pub fn setFor(self: *const UITheme, kind: UIStyleKind) *const UIStyleSet
    pub fn setForMut(self: *UITheme, kind: UIStyleKind) *UIStyleSet
};
pub const UIStyleTransition = ...;
pub const max_style_transitions = ...;
pub fn lerpStyle(a: UIStyle, b: UIStyle, t: f32) UIStyle
pub fn styleEql(a: UIStyle, b: UIStyle) bool

// Canvas-методы:
pub fn setStyleClass(self: *UICanvas, name: []const u8, set: UIStyleSet) void
pub fn styleClass(self: *const UICanvas, name: []const u8) ?UIStyleSet
pub fn resolveStyle(self: *const UICanvas, request: UIStyleRequest) UIStyle
pub fn resolveAnimatedStyle(self: *UICanvas, kind: UIStyleKind, opts: UIStyledOptions) UIStyle
pub fn applyCssTheme(self: *UICanvas, parsed: css_parser.CssTheme) void
pub fn drawStyleRect / drawStyledPanel / drawStyledButton / drawStyledCheckbox / drawStyledSlider / drawStyledBadge

// CSS-подмножество:
pub const CssTheme / CssClass / CssDiag / CssDiagKind = ...;
pub fn parseCss(...) ... // свободная функция (то же, что UITheme.parseCss)
pub fn loadThemeFile(allocator: std.mem.Allocator, path: []const u8) ... // чтение .css с диска
```

Переходы — retained: `resolveAnimatedStyle` интерполирует текущий стиль к целевому через `lerpStyle` (`TransitionOptions` задаёт длительность/кривую). Никаких строк и аллокаций в рантайме стиля — только структуры.

### Текст: SDF + TrueType (`ui/text.zig`, `ui/font.zig`, `ttf/`)

```zig
// Встроенный битмап/SDF-путь:
pub const GlyphUV = ...;
pub fn getGlyphUV(char_code: u8) GlyphUV
pub fn drawText(canvas: anytype, text: []const u8, x: f32, y: f32, font_size: f32, color: Color4) void
pub fn drawTextBold(... extra_boldness: f32) void
pub fn drawTextWithOutline(... outline_width: f32) void
pub fn measureText(text: []const u8, font_size: f32) Vec2

// TrueType-путь:
pub fn drawTextTtf(...) void
pub fn measureTextTtf(font: *const ttf.TtfFont, text: []const u8, font_size: f32) Vec2
pub fn measureForCanvas(canvas: anytype, text: []const u8, font_size: f32) Vec2

// Canvas-методы TTF:
pub fn setFontTtf(self: *UICanvas, font: ?*const TtfFont) void
pub fn clearFontTtf(self: *UICanvas) void
pub fn hasTtfFont(self: *const UICanvas) bool
pub fn activeFontView(self: *const UICanvas) sg.View
pub fn activeFontSampler(self: *const UICanvas) sg.Sampler
pub fn measureTextCurrent(self: *const UICanvas, text: []const u8, font_size: f32) Vec2
pub fn makeFontTexture(allocator: std.mem.Allocator) !Texture
```

Оба пути рисуют одни и те же `UIVertex`-квады (`ui/text.zig` + `ui3d` используют ту же вершинную схему; TTF отличается только атласом и UV/advance). Когда TTF-шрифт установлен, `drawText` идёт через coverage-сэмплирование (mode 3); без него — через встроенный атлас.

Модуль `ttf/` (импорт `agate.ttf`):

```zig
pub const TtfError = ...; // UnsupportedCff, UnsupportedVariableFont, UnsupportedCmap, Truncated, BadTable/...
pub const atlas_size: u32 = ...;
pub fn sniff(...) ...            // проверка sfnt-сигнатуры (0x00010000, 'true', 'typ1')
pub const Font = ...;            // borrowing-хэндл: парсинг, cmap 4/12, advances, kern
pub const OutlinePoint / Contour = ...;
pub fn extractOutline(...) ...   // простые + составные глифы (вложенность ≤ 8)
pub const Segment = ...;
pub fn flattenContours(...) ...  // адаптивное сплющивание квадратичных кривых
pub fn rasterizeSegments(...) ...// суперсэмплированный scanline fill, non-zero winding
pub const GlyphInfo = ...;
pub const TtfFont = struct {     // UI-хэндл: атлас + метрики + кернинг + UV
    pub fn init(allocator: std.mem.Allocator, data: []const u8, pixel_size: u32) ...
    pub fn deinit(self: *TtfFont) void
    pub fn lookup(self: *const TtfFont, codepoint: u21) ?GlyphInfo
    pub fn kernPx(self: *const TtfFont, left: GlyphInfo, right: GlyphInfo) f32
    pub fn glyphUv(self: *const TtfFont, g: GlyphInfo) [4]f32
};
```

Поддержано: sfnt `0x00010000`/`true`/`typ1` с glyf-контурами, `loca` обоих форматов, `maxp` короткий/полный, `cmap` 4 (BMP) и 12 (полный Unicode, приоритет), `kern` format 0, составные глифы (1/2-байтные аргументы, uniform/x-and-y/2x2-трансформы). Явно отвергается с ошибкой (не молчаливым фолбэком): CFF/OTF (`OTTO`), variable fonts (`fvar`), cmap без 4/12, усечённые таблицы, композитные циклы/глубина > 8. Вне скоупа: хинтинг, лигатуры/шейпинг, RTL/bidi, субпиксельное позиционирование (advance снаппится к целым пикселям), цветные шрифты, вертикальные метрики, проверка чексумм. `Font` заимствует слайс `data` (без копии — вызывающий держит alive); `TtfFont` владеет пикселями атласа и списком глифов. Немапленный кодпоинт → `.notdef` (gid 0): рисует пустоту, но сохраняет advance, раскладка не ломается.

### GPU-загрузка (`ui/gpu.zig`, методы канваса)

```zig
pub fn batchUploadBytes(vert_count: usize, index_count: usize) usize
pub fn isUploadOpen(self: *const UICanvas) bool
pub fn markUiUploaded(self: *UICanvas) void
pub fn clampedVertCount(len: usize) usize
pub fn grownCapacity(current: usize, need: usize) usize
pub fn ensureUiBufferPair(...) void
pub fn uploadUiBuffers(...) void
pub fn drawUiBuffers(...) void
```

Двойной буфер вершин/индексов с ростом ёмкости (`grownCapacity`), same-frame guard (`isUploadOpen`/`markUiUploaded`) защищает от двойной загрузки одного кадра. Сложность загрузки O(вершины + индексы).

### 3D-панели (ui3d)

Плавающие UI-панели в мире — не часть `UICanvas`, а слой сцены:

- `Scene.Ui3dPanel` / `Ui3dPanelOptions` / `Ui3dFaceMode` / `Ui3dPickHit` (`src/agate/scene/gui3d_layer.zig`, реэкспорт в `root.zig`, см. `./scene-layers.md`);
- шейдер `src/agate/shaders/ui3d_panel.glsl` (в отличие от `ui.glsl` — проекция в мире, а не в экранных пикселях).

Пикинг панелей возвращает `Ui3dPickHit` с UV-точкой — её можно скормить хит-тесту канваса для кликабельных 3D-экранов.

## Потоки и владение

- `UICanvas` принадлежит главному потоку; весь кадр (`begin` → команды → `render`) выполняется там же. Многопоточного доступа нет — guard только от повторной загрузки в том же кадре, а не от гонок.
- Память: батч вершин/индексов и таблицы стилей владеет канвас (`deinit` освобождает; `init`/`makeFontTexture` требуют аллокатор). `TtfFont` владеет атласом; сырые `ttf_bytes` для `Font`/`TtfFont.init` — заимствованы, должны жить дольше шрифта. `CssTheme` после `applyCssTheme` копируется в тему канваса (парсинговая арена освобождается вызывающим).
- `LayoutStack` — тонкая оболочка над `*UICanvas` (без владения): `init(&canvas)` каждый кадр или переиспользование через `reset`.
- GPU-ресурсы (пара буферов, атлас шрифта, view/sampler) создаются при `init`/`ensureUiBufferPair` и уничтожаются в `deinit`; `initCpuOnly` их не создаёт.

## Ошибки и краевые случаи

- Пропущенный `begin` — батч прошлого кадра перерисуется повторно; пропущенный `render` — кадр без UI, без утечек.
- `beginHStack`/`beginVStack`/`beginGrid` возвращают `false` при нулевой площади — содержимое между begin/end должно быть условно пропущено, иначе виджеты получат вырожденные rect.
- `setStyleClass` сверх `max_style_classes` — старые классы не вытесняются молча (поведение фиксировано лимитом; проверяйте количество классов темы).
- `loadThemeFile`/`parseCss` возвращают `ParseError` с диагностикой `CssDiag` (`CssDiagKind` указывает на правило/селектор); частично распарсенная тема не применяется — сначала чините CSS.
- TTF: отсутствующий глиф → `.notdef` с advance (невидим, но место занимает); файл с `CFF`/`fvar`/без cmap 4/12 → явная `TtfError`, канвас остаётся на встроенном атласе.
- `measureText` без TTF и `measureTextTtf`/`measureTextCurrent` с TTF дают разные ширины (разные advance) — не смешивайте измерение и рисование разными путями, иначе поплывёт выравнивание.
- `drawSlider` при `w <= 0` — деление защищено клампом, но значение будет крайним; нулевые размеры виджетов всегда ошибка вызывающего.

## Производительность

- Батчинг: весь UI кадра — одна загрузка (`uploadUiBuffers`, O(n)) и минимум draw calls (`drawUiBuffers` режет по смене текстуры атласа: встроенный vs TTF). Держите один атлас на кадр ради одного батча.
- Текст: измерение O(длина); SDF-путь дешевле TTF-coverage при мелких кеглях. Кернинг TTF применяется только при установленном шрифте.
- Стили: `resolveStyle` — чистая структура без аллокаций; `resolveAnimatedStyle` добавляет `lerpStyle` на анимирующиеся виджеты (O(1) на виджет, ограничено `max_style_transitions`).
- Раскладка: солверы flex/grid O(n) по детям; `LayoutStack` не аллоцирует (фиксированный стек контейнеров).
- GPU: рост буферов — геометрический (`grownCapacity`), shrinkage нет — пик UI-кадра удерживает память до `deinit`.

## Смотрите также

- `./architecture.md` — место UI в кадре движка.
- `./frame-pipeline.md` — когда вызывается `canvas.render` относительно 3D-пассов.
- `./texture.md` — `Texture`, атласы, view/sampler шрифта.
- `./scene-layers.md` — `Ui3dPanel` (gui3d_layer) для 3D-панелей.
- `./shaders.md` — `ui.glsl`, `ui3d_panel.glsl`, `// @include`-механика.
- `./serialization.md` — UI-состояние не входит в снапшот сцены.
