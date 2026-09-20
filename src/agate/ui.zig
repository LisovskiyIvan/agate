//! Facade for the UI modules. The `ui.zig` canvas monolith was split into
//! focused leaves under `ui/` following the repo pattern (free functions +
//! thin forwarders; Zig 0.16 has no usingnamespace; see `scene/render_queue.zig`,
//! `serialization.zig`, `profiler.zig`):
//!
//! - `ui/canvas.zig` — owns the `UICanvas` type: draw state, GPU resources,
//!   style storage, input state, the trivial lifecycle plus thin forwarders
//!   into the siblings below, so every call site keeps working unchanged.
//! - `ui/draw.zig` — draw primitives and the pure batch math.
//! - `ui/text.zig` — SDF glyph atlas plus the TrueType coverage text path.
//! - `ui/widgets.zig` — stateless immediate-mode controls.
//! - `ui/input_state.zig` — hit-testing, scroll and text-input state.
//! - `ui/style.zig` — style cascade, retained transitions, styled drawing.
//! - `ui/font.zig` — font atlas creation and the TrueType override.
//! - `ui/gpu.zig` — buffer-pair ensure, upload/draw, same-frame guard, render.
//! - `ui/stack.zig` — layout containers (`LayoutStack`) and grid helpers.
//! - `ui/layout.zig` — sizing/dock/flex/grid solvers (re-exported unchanged).
//! - `ui/types.zig`, `ui/theme.zig`, `ui/transition.zig`,
//!   `ui/css_parser.zig` — style value types, theme, transition math, CSS
//!   parsing (re-exported unchanged).
//!
//! Everything that was public before the split is re-exported here
//! unchanged; consumers (`scene.zig`, `scene/*`, `root.zig`, the sandbox)
//! see the same API as when everything lived in this file.
//!
//! Documented anti-cycle rule: leaves must never import this facade —
//! importing it back would make the re-exports depend on their own
//! consumers. The method bodies take the canvas as `anytype` (same
//! discipline as `scene/`), so library code has no leaf-to-owner edge at
//! all; moved tests reach `canvas.UICanvas` through a block-scoped import
//! that exists only in test builds.

const ui_canvas = @import("ui/canvas.zig");
const ui_draw = @import("ui/draw.zig");
const ui_text = @import("ui/text.zig");
const ui_input = @import("ui/input_state.zig");
const ui_font = @import("ui/font.zig");
const ui_style = @import("ui/style.zig");
const ui_gpu = @import("ui/gpu.zig");
const ui_stack = @import("ui/stack.zig");
const ui_layout = @import("ui/layout.zig");
const ui_types = @import("ui/types.zig");
const ui_theme_mod = @import("ui/theme.zig");
const ui_transition = @import("ui/transition.zig");
const css_parser = @import("ui/css_parser.zig");

// Canvas type (lives in ui/canvas.zig).
pub const UICanvas = ui_canvas.UICanvas;

// Re-exports from the split leaf modules (public API unchanged).
pub const UIVertex = ui_draw.UIVertex;
pub const GlyphUV = ui_text.GlyphUV;
pub const getGlyphUV = ui_text.getGlyphUV;
pub const ScrollState = ui_input.ScrollState;
pub const TextInputState = ui_input.TextInputState;
pub const TtfFont = ui_font.TtfFont;

// Re-exports from layout module
pub const UISize = ui_layout.UISize;
pub const UIEdges = ui_layout.UIEdges;
pub const UIAnchor = ui_layout.UIAnchor;
pub const UIDock = ui_layout.UIDock;
pub const anchorRect = ui_layout.anchorRect;
pub const dockRect = ui_layout.dockRect;
pub const FlexDirection = ui_layout.FlexDirection;
pub const JustifyContent = ui_layout.JustifyContent;
pub const AlignItems = ui_layout.AlignItems;
pub const LayoutItem = ui_layout.LayoutItem;
pub const solveFlex = ui_layout.solveFlex;
pub const GridTrack = ui_layout.GridTrack;
pub const solveGridTracks = ui_layout.solveGridTracks;
pub const AdvancedGridSpec = ui_layout.AdvancedGridSpec;

// Re-exports from the layout container module (ui/stack.zig)
pub const LayoutAlign = ui_stack.LayoutAlign;
pub const LayoutAlignCross = ui_stack.LayoutAlignCross;
pub const layoutAlignOffset = ui_stack.layoutAlignOffset;
pub const gridExtentSize = ui_stack.gridExtentSize;
pub const gridExtentOffset = ui_stack.gridExtentOffset;
pub const LayoutGridSpec = ui_stack.LayoutGridSpec;
pub const LayoutFlowOptions = ui_stack.LayoutFlowOptions;
pub const LayoutGridOptions = ui_stack.LayoutGridOptions;
pub const LayoutFlexOptions = ui_stack.LayoutFlexOptions;
pub const LayoutLabelOptions = ui_stack.LayoutLabelOptions;
pub const LayoutButtonOptions = ui_stack.LayoutButtonOptions;
pub const LayoutCheckboxOptions = ui_stack.LayoutCheckboxOptions;
pub const LayoutSliderOptions = ui_stack.LayoutSliderOptions;
pub const LayoutProgressOptions = ui_stack.LayoutProgressOptions;
pub const LayoutDividerOptions = ui_stack.LayoutDividerOptions;
pub const LayoutBadgeOptions = ui_stack.LayoutBadgeOptions;
// LayoutStack is generic over the canvas pointer; the facade pins the
// production instantiation so `LayoutStack.init(&canvas)` keeps working.
pub const LayoutStack = ui_stack.LayoutStack(ui_canvas.UICanvas);

// CSS-subset theme parsing (see ui/css_parser.zig). `UITheme.parseCss` is
// the same parser as a method on the theme type; install a parsed theme on
// a canvas with UICanvas.applyCssTheme.
pub const CssTheme = css_parser.CssTheme;
pub const CssClass = css_parser.CssClass;
pub const CssDiag = css_parser.CssDiag;
pub const CssDiagKind = css_parser.CssDiagKind;
pub const parseCss = css_parser.parseCss;
pub const loadThemeFile = css_parser.loadThemeFile;

// ---------------------------------------------------------------------
// Styling (CSS-like, immediate-mode; no string parsing, no allocation)
// ---------------------------------------------------------------------
//
// The model follows the spirit of clay.h / Dear ImGui styles: concrete
// style values live in plain structs, and widgets resolve one concrete
// `UIStyle` per call through a cascade:
//
//   engine fallback < theme default (per widget kind) < named class
//                    < per-call override < state deltas applied per layer
//
// - `UIStyle` is fully resolved (all fields concrete).
// - `UIStyleOverride` is a partial (all fields optional; null = inherit).
// - `UIStyleSet` is a partial "normal" look plus optional per-state deltas
//   (the pseudo-class analog: :hover / :active / :focus / :disabled).
// - `UICanvas.theme` holds the defaults per widget kind; named classes are
//   registered on the canvas with `setStyleClass` and referenced by name.

// The style value types live in ui/types.zig, the theme in ui/theme.zig and
// the transition math in ui/transition.zig; the facade re-exports them under
// their historical names so callers keep one import path.
pub const UIState = ui_types.UIState;
pub const UIGradient = ui_types.UIGradient;
pub const UIShadow = ui_types.UIShadow;
pub const UIStyle = ui_types.UIStyle;
pub const UIStyleOverride = ui_types.UIStyleOverride;
pub const UIStyleSet = ui_types.UIStyleSet;
pub const TransitionOptions = ui_types.TransitionOptions;
pub const UIStyleKind = ui_types.UIStyleKind;
pub const UIStyledOptions = ui_types.UIStyledOptions;
pub const UIStyleRequest = ui_types.UIStyleRequest;
pub const UIStyleClass = ui_types.UIStyleClass;
pub const UIBoxStyle = ui_types.UIBoxStyle;
pub const max_style_classes = ui_types.max_style_classes;
pub const UITheme = ui_theme_mod.UITheme;
pub const UIStyleTransition = ui_transition.UIStyleTransition;
pub const max_style_transitions = ui_transition.max_style_transitions;
pub const lerpStyle = ui_transition.lerpStyle;
pub const styleEql = ui_transition.styleEql;
