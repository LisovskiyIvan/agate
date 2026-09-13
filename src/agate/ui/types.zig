//! Style-system value types shared by the UI canvas, the theme parser and
//! the transition machinery. Pure data + pure helpers only: nothing here
//! touches sokol or allocators, so the parser and transition math can be
//! unit-tested without a GPU context.

const std = @import("std");

const math = @import("math");
const Color4 = math.Color4;

const easing = @import("../animation/easing.zig");

/// Widget state used for style resolution (pseudo-class analog).
pub const UIState = enum(u8) {
    normal,
    hover,
    active,
    focus,
    disabled,

    /// Derives the state from hit-test flags in priority order: disabled
    /// wins, then pressed (active), then focus, then hover.
    pub fn fromFlags(hovered: bool, pressed: bool, focused: bool, disabled: bool) UIState {
        if (disabled) return .disabled;
        if (pressed) return .active;
        if (focused) return .focus;
        if (hovered) return .hover;
        return .normal;
    }
};

/// Two-stop vertical gradient. The UI pipeline draws plain quads, so a
/// gradient is rasterized as a fixed number of horizontal color bands.
pub const UIGradient = struct {
    top: Color4,
    bottom: Color4,
};

/// Drop shadow. There is no blur pass in the UI pipeline, so blur is
/// approximated by stacking a few expanding translucent rounded rects.
pub const UIShadow = struct {
    color: Color4 = Color4.new(0.0, 0.0, 0.0, 0.35),
    offset_x: f32 = 0.0,
    offset_y: f32 = 2.0,
    blur: f32 = 4.0,
};

/// Style transition (animation) configuration. `duration_ms == 0` means
/// "instant": styles snap and nothing is stored per widget. The easing
/// reuses the shared animation curves so files can name them directly.
pub const TransitionOptions = struct {
    duration_ms: f32 = 0.0,
    easing: easing.EasingType = .ease_out_quad,
};

/// Fully resolved concrete style. The defaults draw nothing (transparent
/// background, no border): a style only shows what it actually sets.
pub const UIStyle = struct {
    background: Color4 = Color4.transparent,
    gradient: ?UIGradient = null,
    border_color: Color4 = Color4.transparent,
    border_width: f32 = 0.0,
    corner_radius: f32 = 0.0,
    /// Content inset for layout containers (see LayoutStack); widgets may
    /// also read it to inset their own content.
    padding: f32 = 0.0,
    /// Outer spacing applied by LayoutStack.placeBox.
    margin: f32 = 0.0,
    shadow: ?UIShadow = null,
    text_color: Color4 = Color4.white,
    /// Accent used by progress-like fills (slider/progress fill); null
    /// falls back to `UITheme.accent`.
    accent: ?Color4 = null,
    /// Multiplies background/border/text alpha at draw time (CSS opacity).
    opacity: f32 = 1.0,
};

/// Partial style: `null` fields inherit from the previous cascade layer.
pub const UIStyleOverride = struct {
    background: ?Color4 = null,
    gradient: ?UIGradient = null,
    border_color: ?Color4 = null,
    border_width: ?f32 = null,
    corner_radius: ?f32 = null,
    padding: ?f32 = null,
    margin: ?f32 = null,
    shadow: ?UIShadow = null,
    text_color: ?Color4 = null,
    accent: ?Color4 = null,
    opacity: ?f32 = null,

    /// Returns `base` with every non-null field of `self` applied over it.
    pub fn apply(self: UIStyleOverride, base: UIStyle) UIStyle {
        var s = base;
        if (self.background) |v| s.background = v;
        if (self.gradient) |v| s.gradient = v;
        if (self.border_color) |v| s.border_color = v;
        if (self.border_width) |v| s.border_width = v;
        if (self.corner_radius) |v| s.corner_radius = v;
        if (self.padding) |v| s.padding = v;
        if (self.margin) |v| s.margin = v;
        if (self.shadow) |v| s.shadow = v;
        if (self.text_color) |v| s.text_color = v;
        if (self.accent) |v| s.accent = v;
        if (self.opacity) |v| s.opacity = v;
        return s;
    }

    /// Overwrites every non-null field of `dst` with `self`'s. Used by the
    /// theme parser to merge several selector blocks into one slot.
    pub fn mergeInto(self: UIStyleOverride, dst: *UIStyleOverride) void {
        if (self.background) |v| dst.background = v;
        if (self.gradient) |v| dst.gradient = v;
        if (self.border_color) |v| dst.border_color = v;
        if (self.border_width) |v| dst.border_width = v;
        if (self.corner_radius) |v| dst.corner_radius = v;
        if (self.padding) |v| dst.padding = v;
        if (self.margin) |v| dst.margin = v;
        if (self.shadow) |v| dst.shadow = v;
        if (self.text_color) |v| dst.text_color = v;
        if (self.accent) |v| dst.accent = v;
        if (self.opacity) |v| dst.opacity = v;
    }
};

/// One cascade layer: a partial "normal" look plus optional per-state
/// deltas. `resolve` applies the normal look over `base`, then the delta
/// for `state` (if any), so each layer contributes only what it defines.
/// `transition` is set-level: it animates every state/target change of
/// widgets resolving through this set (the theme parser maps the
/// `transition_*` properties of any block of a selector onto it).
pub const UIStyleSet = struct {
    normal: UIStyleOverride = .{},
    hover: ?UIStyleOverride = null,
    active: ?UIStyleOverride = null,
    focus: ?UIStyleOverride = null,
    disabled: ?UIStyleOverride = null,
    transition: TransitionOptions = .{},

    pub fn resolve(self: UIStyleSet, base: UIStyle, state: UIState) UIStyle {
        var s = self.normal.apply(base);
        const delta: ?UIStyleOverride = switch (state) {
            .normal => null,
            .hover => self.hover,
            .active => self.active,
            .focus => self.focus,
            .disabled => self.disabled,
        };
        if (delta) |d| s = d.apply(s);
        return s;
    }
};

/// Widget kinds a style can target (theme slots). The three additions are
/// used by the styled dropdown / text input / progress widgets.
pub const UIStyleKind = enum { panel, button, checkbox, slider, badge, container, dropdown, text_input, progress };

/// Per-call options shared by the styled widget variants. `class` names a
/// style class registered with `UICanvas.setStyleClass`; `style` is the
/// per-call override (wins over class, which wins over the theme).
/// `anim_key` opts a widget into style transitions: pass a string that is
/// stable across frames (a widget id), not something derived from a moving
/// rect. `transition` overrides the cascade-resolved transition config.
pub const UIStyledOptions = struct {
    class: ?[]const u8 = null,
    style: ?UIStyleOverride = null,
    state: UIState = .normal,
    anim_key: ?[]const u8 = null,
    transition: ?TransitionOptions = null,
};

/// Style lookup request for `UICanvas.resolveStyle`.
pub const UIStyleRequest = struct {
    kind: UIStyleKind,
    class: ?[]const u8 = null,
    override: ?UIStyleOverride = null,
    state: UIState = .normal,
};

/// A named style class registered on the canvas ("panel.dark" etc.).
pub const UIStyleClass = struct {
    name: []const u8,
    set: UIStyleSet,
};

/// Fixed class registry size; UI themes use a handful of classes, and the
/// registry silently ignores registration past the cap (no allocation).
pub const max_style_classes = 32;

/// Per-call style info for layout boxes placed with `LayoutStack.placeBox`.
pub const UIBoxStyle = struct {
    class: ?[]const u8 = null,
    style: ?UIStyleOverride = null,
};

/// Hashes a stable widget id into a transition-storage key. A fixed
/// non-zero seed keeps unrelated hashes from collapsing onto 0 (the
/// "empty slot" sentinel would not care, but distinct keys read better).
pub fn animKeyHash(name: []const u8) u64 {
    return std.hash.Wyhash.hash(0x9e3779b9, name);
}

/// Multiplies a color's alpha by `k`, clamped (draw-time opacity helper).
pub fn mulAlpha(c: Color4, k: f32) Color4 {
    return Color4.new(c.r, c.g, c.b, std.math.clamp(c.a * k, 0.0, 1.0));
}
