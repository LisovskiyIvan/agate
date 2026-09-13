//! Theme: default style sets per widget kind. Parsed from CSS-subset text
//! by `ui/css_parser.zig` (see `UITheme.parseCss`).

const std = @import("std");

const types = @import("types.zig");
const css = @import("css_parser.zig");

const UIStyleSet = types.UIStyleSet;
const UIStyleKind = types.UIStyleKind;
const Color4 = @import("math").Color4;

/// Theme: default style sets per widget kind. `defaults()` mirrors the
/// hardcoded palettes of the legacy draw* widgets so styled variants look
/// the same unless a class/override changes them.
pub const UITheme = struct {
    panel: UIStyleSet = .{},
    button: UIStyleSet = .{},
    checkbox: UIStyleSet = .{},
    slider: UIStyleSet = .{},
    badge: UIStyleSet = .{},
    /// Layout containers default to fully invisible so adding a layout
    /// container never changes what is on screen.
    container: UIStyleSet = .{},
    /// Styled dropdown control (closed button + open list rows).
    dropdown: UIStyleSet = .{},
    /// Single-line text field (panel, text and cursor color).
    text_input: UIStyleSet = .{},
    /// Progress bar track; the fill comes from `accent`/style accent.
    progress: UIStyleSet = .{},
    /// Fill color for progress-like elements (slider/progress fill).
    accent: Color4 = Color4.new(0.20, 0.46, 0.82, 0.95),

    pub fn defaults() UITheme {
        return .{
            .panel = .{
                .normal = .{
                    .background = Color4.new(0.10, 0.12, 0.18, 0.9),
                    .border_color = Color4.new(0.3, 0.4, 0.52, 0.75),
                    .border_width = 1.0,
                    .corner_radius = 6.0,
                    .padding = 8.0,
                },
            },
            .button = .{
                .normal = .{
                    .background = Color4.new(0.14, 0.18, 0.25, 0.85),
                    .border_color = Color4.new(0.3, 0.4, 0.52, 0.75),
                    .border_width = 1.5,
                    .corner_radius = 4.0,
                },
                .hover = .{
                    .background = Color4.new(0.24, 0.32, 0.44, 0.92),
                    .border_color = Color4.new(0.65, 0.85, 1.0, 0.95),
                },
                .active = .{
                    .background = Color4.new(0.18, 0.42, 0.78, 0.95),
                    .border_color = Color4.new(0.4, 0.75, 1.0, 1.0),
                },
                .disabled = .{ .opacity = 0.45 },
            },
            .checkbox = .{
                .normal = .{
                    .background = Color4.new(0.14, 0.18, 0.25, 0.85),
                    .border_color = Color4.new(0.3, 0.4, 0.52, 0.75),
                    .border_width = 1.5,
                    .corner_radius = 3.0,
                },
                .hover = .{
                    .background = Color4.new(0.24, 0.32, 0.44, 0.92),
                    .border_color = Color4.new(0.65, 0.85, 1.0, 0.95),
                },
                .active = .{ .background = Color4.new(0.18, 0.42, 0.78, 0.95) },
            },
            .slider = .{
                .normal = .{
                    .background = Color4.new(0.10, 0.12, 0.18, 0.9),
                    .border_color = Color4.new(0.45, 0.5, 0.6, 0.7),
                    .border_width = 1.0,
                    .corner_radius = 2.0,
                },
            },
            .badge = .{
                .normal = .{
                    .background = Color4.new(0.14, 0.18, 0.25, 0.85),
                    .corner_radius = 8.0,
                },
            },
            // Same look as the legacy dropdown (which reuses drawButton).
            .dropdown = .{
                .normal = .{
                    .background = Color4.new(0.14, 0.18, 0.25, 0.85),
                    .border_color = Color4.new(0.3, 0.4, 0.52, 0.75),
                    .border_width = 1.5,
                    .corner_radius = 4.0,
                },
                .hover = .{
                    .background = Color4.new(0.24, 0.32, 0.44, 0.92),
                    .border_color = Color4.new(0.65, 0.85, 1.0, 0.95),
                },
                .active = .{
                    .background = Color4.new(0.18, 0.42, 0.78, 0.95),
                    .border_color = Color4.new(0.4, 0.75, 1.0, 1.0),
                },
            },
            .text_input = .{
                .normal = .{
                    .background = Color4.new(0.10, 0.12, 0.18, 0.9),
                    .border_color = Color4.new(0.3, 0.4, 0.52, 0.75),
                    .border_width = 1.5,
                },
                .focus = .{
                    .background = Color4.new(0.09, 0.11, 0.16, 0.95),
                    .border_color = Color4.new(0.4, 0.75, 1.0, 1.0),
                },
            },
            .progress = .{
                .normal = .{
                    .background = Color4.new(0.10, 0.12, 0.18, 0.9),
                    .border_color = Color4.new(0.45, 0.5, 0.6, 0.7),
                    .border_width = 1.0,
                    .corner_radius = 2.0,
                },
            },
        };
    }

    /// Parses CSS-subset `text` into a theme + class list + diagnostics.
    /// The result references `allocator`-owned memory (pass an arena to
    /// free everything at once); syntax errors are reported as diagnostics,
    /// not error returns (only allocation failures are fatal).
    pub fn parseCss(allocator: std.mem.Allocator, text: []const u8) css.ParseError!css.CssTheme {
        return css.parseCss(allocator, text);
    }

    /// Style-set slot for a widget kind (cascade reads resolve through it).
    pub fn setFor(self: *const UITheme, kind: UIStyleKind) *const UIStyleSet {
        return switch (kind) {
            .panel => &self.panel,
            .button => &self.button,
            .checkbox => &self.checkbox,
            .slider => &self.slider,
            .badge => &self.badge,
            .container => &self.container,
            .dropdown => &self.dropdown,
            .text_input => &self.text_input,
            .progress => &self.progress,
        };
    }

    /// Mutable variant for writers (the parser applies selector blocks).
    pub fn setForMut(self: *UITheme, kind: UIStyleKind) *UIStyleSet {
        return switch (kind) {
            .panel => &self.panel,
            .button => &self.button,
            .checkbox => &self.checkbox,
            .slider => &self.slider,
            .badge => &self.badge,
            .container => &self.container,
            .dropdown => &self.dropdown,
            .text_input => &self.text_input,
            .progress => &self.progress,
        };
    }
};
