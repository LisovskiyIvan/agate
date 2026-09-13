//! Minimal CSS-subset theme parser. Hand-written scanner, no regex, no
//! general-purpose allocator traffic: all result memory (class names,
//! diagnostics) comes from the caller-supplied allocator, which callers
//! should wrap in an arena (see `loadThemeFile` for the canonical flow).
//!
//! Accepted syntax:
//!
//!   /* comment */
//!   theme   { accent: #7cb3ff; opacity: 1; transition_duration: 120ms; }
//!   panel   { background: #101828e6; corner_radius: 6px; padding: 8px;
//!             gradient: #202a3c #101828;
//!             shadow_color: #00000066; shadow_offset_y: 3px; shadow_blur: 6px; }
//!   button:hover { background: rgb(36, 48, 66); border_color: #a6d4ff; }
//!   .danger { background: #b3261e; }
//!   .danger:disabled { opacity: 0.4; }
//!
//! - A bare selector naming a widget kind (`panel`, `button`, `checkbox`,
//!   `slider`, `badge`, `container`, `dropdown`, `text_input`, `progress`)
//!   targets that theme slot; kind names are therefore reserved for kinds —
//!   use `.panel` for a class literally called "panel".
//! - `theme` targets every kind's normal set at once plus the theme accent
//!   (the "defaults for all widgets" layer).
//! - `.name` registers a style class; `:hover` / `:active` / `:focus` /
//!   `:disabled` suffixes attach per-state deltas.
//! - Values: colors `#rrggbb[aa]`, `rgb(r,g,b)`, `rgba(r,g,b,a)`; lengths
//!   `N` or `Npx`; times `Nms`/`Ns`; plain numbers for opacity; easing
//!   names as in `animation/easing.zig` (`ease-out-quad` normalization of
//!   dashes allowed).
//! - `transition_duration` / `transition_easing` set the enclosing style
//!   set's `TransitionOptions` (one config per set, whichever block set it
//!   last wins).
//!
//! Error model: malformed input never aborts the parse. Problems are
//! reported as diagnostics with line/column positions; only allocation
//! failure and oversized files are error returns.

const std = @import("std");

const math = @import("math");
const Color4 = math.Color4;

const easing = @import("../animation/easing.zig");
const types = @import("types.zig");
const theme_mod = @import("theme.zig");

const UIStyleOverride = types.UIStyleOverride;
const UIStyleSet = types.UIStyleSet;
const UIState = types.UIState;
const UIStyleKind = types.UIStyleKind;
const TransitionOptions = types.TransitionOptions;
const UITheme = theme_mod.UITheme;

/// Hard cap on theme file size; protects against pathological inputs.
pub const max_theme_css_bytes: usize = 1 << 20;
/// Diagnostics cap: after this many problems the parser keeps going but
/// stops recording (a broken file never needs more than a screenful).
pub const max_diags: usize = 128;

pub const ParseError = error{ OutOfMemory, ThemeFileTooLarge };

pub const CssDiagKind = enum {
    syntax,
    unknown_property,
    unknown_selector,
    invalid_value,
};

pub const CssDiag = struct {
    line: usize,
    col: usize,
    kind: CssDiagKind,
    /// Static message or an allocator-owned copy of the offending token.
    message: []const u8,
};

/// A parsed style class. The name is allocator-owned (copied out of the
/// source, which the caller may free after parsing).
pub const CssClass = struct {
    name: []const u8,
    set: UIStyleSet,
};

pub const CssTheme = struct {
    theme: UITheme,
    classes: []const CssClass,
    diags: []const CssDiag,
    /// True when the `theme` selector set the theme accent (distinguishable
    /// from "not set", which keeps the existing canvas accent on merge).
    accent_set: bool,
};

// ----------------------------------------------------------------------
// Value parsers (public for tests / reuse)
// ----------------------------------------------------------------------

/// Parses `#rgb`, `#rgba`, `#rrggbb`, `#rrggbbaa` hex colors.
pub fn parseHexColor(text_in: []const u8) ?Color4 {
    const text = std.mem.trim(u8, text_in, " \t");
    if (text.len == 0 or text[0] != '#') return null;
    const digits = text[1..];
    const n = digits.len;
    if (n != 3 and n != 4 and n != 6 and n != 8) return null;
    // Digits per channel (1 = shorthand); a missing alpha channel stays 255. charToDigit
    // yields nibbles, so channel bytes are composed pairwise here; a
    // shorthand digit doubles itself (CSS rule: #abc == #aabbcc).
    const nib: usize = if (n <= 4) 1 else 2;
    const count: usize = n / nib;
    var chan: [4]u8 = .{ 0, 0, 0, 255 };
    for (0..count) |ci| {
        const c0 = std.fmt.charToDigit(digits[ci * nib], 16) catch return null;
        const c1 = if (nib == 2)
            std.fmt.charToDigit(digits[ci * nib + 1], 16) catch return null
        else
            c0;
        chan[ci] = c0 * 16 + c1;
    }
    const f = struct {
        fn byte(b: u8) f32 {
            return @as(f32, @floatFromInt(b)) / 255.0;
        }
    };
    return Color4.new(f.byte(chan[0]), f.byte(chan[1]), f.byte(chan[2]), f.byte(chan[3]));
}

/// Parses `rgb(r,g,b)` / `rgba(r,g,b,a)` with 0..255 byte channels and a
/// 0..1 float alpha. Channel values are clamped, matching lenient CSS.
pub fn parseRgbColor(text_in: []const u8) ?Color4 {
    const text = std.mem.trim(u8, text_in, " \t");
    const rgba = std.mem.startsWith(u8, text, "rgba(");
    if (!rgba and !std.mem.startsWith(u8, text, "rgb(")) return null;
    if (!std.mem.endsWith(u8, text, ")")) return null;
    const open_len: usize = if (rgba) 5 else 4;
    const inner = text[open_len .. text.len - 1];
    var it = std.mem.splitScalar(u8, inner, ',');
    const r = parseChannel(it.next() orelse return null) orelse return null;
    const g = parseChannel(it.next() orelse return null) orelse return null;
    const b = parseChannel(it.next() orelse return null) orelse return null;
    var a: f32 = 1.0;
    if (rgba) {
        const a_txt = std.mem.trim(u8, it.next() orelse return null, " \t");
        a = std.math.clamp(std.fmt.parseFloat(f32, a_txt) catch return null, 0.0, 1.0);
    }
    if (it.next() != null) return null;
    return Color4.new(r, g, b, a);
}

fn parseChannel(text_in: []const u8) ?f32 {
    const text = std.mem.trim(u8, text_in, " \t");
    if (text.len == 0) return null;
    const v = std.fmt.parseFloat(f32, text) catch return null;
    return std.math.clamp(v, 0.0, 255.0) / 255.0;
}

/// Parses any color the subset supports (hex or functional).
pub fn parseColor(text: []const u8) ?Color4 {
    if (parseHexColor(text)) |c| return c;
    return parseRgbColor(text);
}

/// Parses a length: `N` or `Npx` (floats allowed). No unit means pixels.
pub fn parseLength(text_in: []const u8) ?f32 {
    const text = std.mem.trim(u8, text_in, " \t");
    const body = if (std.mem.endsWith(u8, text, "px")) text[0 .. text.len - 2] else text;
    if (body.len == 0) return null;
    return std.fmt.parseFloat(f32, body) catch null;
}

/// Parses a duration in `Nms` or `Ns`, returned as milliseconds.
pub fn parseDurationMs(text_in: []const u8) ?f32 {
    const text = std.mem.trim(u8, text_in, " \t");
    if (std.mem.endsWith(u8, text, "ms")) {
        return std.fmt.parseFloat(f32, text[0 .. text.len - 2]) catch null;
    }
    if (std.mem.endsWith(u8, text, "s")) {
        const s = std.fmt.parseFloat(f32, text[0 .. text.len - 1]) catch return null;
        return s * 1000.0;
    }
    // Bare number: milliseconds (the property name carries the unit).
    return std.fmt.parseFloat(f32, text) catch null;
}

/// Parses an easing name; `-` normalizes to `_` so CSS-style names
/// (`ease-out-quad`) map onto the shared `EasingType` enum.
pub fn parseEasing(text_in: []const u8) ?easing.EasingType {
    const text = std.mem.trim(u8, text_in, " \t");
    if (text.len == 0 or text.len > 32) return null;
    var buf: [32]u8 = undefined;
    for (text, 0..) |c, i| {
        buf[i] = if (c == '-') '_' else c;
    }
    return std.meta.stringToEnum(easing.EasingType, buf[0..text.len]);
}

// ----------------------------------------------------------------------
// Property table
// ----------------------------------------------------------------------

const PropKind = enum {
    background,
    gradient,
    border_color,
    border_width,
    corner_radius,
    padding,
    margin,
    shadow_color,
    shadow_offset_x,
    shadow_offset_y,
    shadow_blur,
    text_color,
    accent,
    opacity,
    transition_duration,
    transition_easing,
};

const prop_map = std.StaticStringMap(PropKind).initComptime(.{
    .{ "background", .background },
    .{ "gradient", .gradient },
    .{ "border_color", .border_color },
    .{ "border-color", .border_color },
    .{ "border_width", .border_width },
    .{ "border-width", .border_width },
    .{ "corner_radius", .corner_radius },
    .{ "corner-radius", .corner_radius },
    .{ "padding", .padding },
    .{ "margin", .margin },
    .{ "shadow_color", .shadow_color },
    .{ "shadow-color", .shadow_color },
    .{ "shadow_offset_x", .shadow_offset_x },
    .{ "shadow-offset-x", .shadow_offset_x },
    .{ "shadow_offset_y", .shadow_offset_y },
    .{ "shadow-offset-y", .shadow_offset_y },
    .{ "shadow_blur", .shadow_blur },
    .{ "shadow-blur", .shadow_blur },
    .{ "text_color", .text_color },
    .{ "text-color", .text_color },
    .{ "accent", .accent },
    .{ "opacity", .opacity },
    .{ "transition_duration", .transition_duration },
    .{ "transition-duration", .transition_duration },
    .{ "transition_easing", .transition_easing },
    .{ "transition-easing", .transition_easing },
});

// ----------------------------------------------------------------------
// Parser
// ----------------------------------------------------------------------

const Target = union(enum) {
    theme_all,
    kind: UIStyleKind,
    class_idx: usize,
};

const Block = struct {
    target: Target,
    state: UIState = .normal,
    override: UIStyleOverride = .{},
    transition: ?TransitionOptions = null,
};

const Parser = struct {
    allocator: std.mem.Allocator,
    text: []const u8,
    i: usize = 0,
    line: usize = 1,
    col: usize = 1,
    theme: UITheme,
    classes: std.ArrayListUnmanaged(CssClass) = .empty,
    diags: std.ArrayListUnmanaged(CssDiag) = .empty,
    accent_set: bool = false,
    /// Set once the diag cap is hit; further problems are skipped silently.
    diags_full: bool = false,

    fn addDiag(self: *Parser, kind: CssDiagKind, message: []const u8) void {
        self.addDiagAt(self.line, self.col, kind, message, false);
    }

    fn addDiagToken(self: *Parser, line: usize, col: usize, kind: CssDiagKind, token: []const u8) void {
        self.addDiagAt(line, col, kind, token, true);
    }

    fn addDiagAt(self: *Parser, line: usize, col: usize, kind: CssDiagKind, message: []const u8, copy: bool) void {
        if (self.diags_full) return;
        if (self.diags.items.len >= max_diags) {
            self.diags_full = true;
            return;
        }
        const msg = if (copy) (self.allocator.dupe(u8, message) catch return) else message;
        self.diags.append(self.allocator, .{ .line = line, .col = col, .kind = kind, .message = msg }) catch {
            self.diags_full = true;
        };
    }

    fn peek(self: *const Parser) ?u8 {
        if (self.i >= self.text.len) return null;
        return self.text[self.i];
    }

    fn advance(self: *Parser) void {
        if (self.i >= self.text.len) return;
        if (self.text[self.i] == '\n') {
            self.line += 1;
            self.col = 1;
        } else {
            self.col += 1;
        }
        self.i += 1;
    }

    fn skipWs(self: *Parser) void {
        while (self.peek()) |c| {
            switch (c) {
                ' ', '\t', '\r', '\n' => self.advance(),
                else => return,
            }
        }
    }

    /// Skips whitespace and `/* ... */` comments (anywhere between tokens).
    fn skipWsAndComments(self: *Parser) void {
        while (self.peek()) |c| {
            if (c == '/' and self.i + 1 < self.text.len and self.text[self.i + 1] == '*') {
                const start_line = self.line;
                const start_col = self.col;
                self.advance();
                self.advance();
                var closed = false;
                while (self.peek()) |cc| {
                    if (cc == '*' and self.i + 1 < self.text.len and self.text[self.i + 1] == '/') {
                        self.advance();
                        self.advance();
                        closed = true;
                        break;
                    }
                    self.advance();
                }
                if (!closed) {
                    self.addDiagAt(start_line, start_col, .syntax, "unterminated comment", false);
                    return;
                }
            } else if (c == ' ' or c == '\t' or c == '\r' or c == '\n') {
                self.advance();
            } else {
                return;
            }
        }
    }

    /// Scans until one of `stops` (not consumed). Returns the trimmed slice.
    fn scanUntil(self: *Parser, stops: []const u8) struct { slice: []const u8, start_line: usize, start_col: usize } {
        const start_line = self.line;
        const start_col = self.col;
        const start = self.i;
        while (self.peek()) |c| {
            var stopped = false;
            for (stops) |s| {
                if (c == s) {
                    stopped = true;
                    break;
                }
            }
            if (stopped) break;
            self.advance();
        }
        const raw = self.text[start..self.i];
        return .{ .slice = std.mem.trim(u8, raw, " \t\r\n"), .start_line = start_line, .start_col = start_col };
    }

    fn parse(self: *Parser) ParseError!CssTheme {
        while (true) {
            self.skipWsAndComments();
            if (self.peek() == null) break;

            const sel = (try self.scanSelector()) orelse continue;
            try self.parseBlock(sel);
        }
        return .{
            .theme = self.theme,
            .classes = try self.classes.toOwnedSlice(self.allocator),
            .diags = try self.diags.toOwnedSlice(self.allocator),
            .accent_set = self.accent_set,
        };
    }

    const Selector = struct {
        target: Target,
        state: UIState = .normal,
    };

    /// Parses and validates a selector; expects the caller to be positioned
    /// right before it. Returns null after reporting a bad selector (the
    /// block is skipped up to and including its '}').
    fn scanSelector(self: *Parser) ParseError!?Selector {
        const sel = self.scanUntil("{");
        if (self.peek() == null) {
            // Stray tokens with no block: report and drop them.
            if (sel.slice.len > 0) {
                self.addDiagAt(sel.start_line, sel.start_col, .syntax, "expected '{'", false);
            }
            return null;
        }
        const raw = sel.slice;
        if (raw.len == 0) {
            self.addDiagAt(sel.start_line, sel.start_col, .syntax, "empty selector", false);
            return null;
        }

        // Split off a ":state" suffix.
        var name = raw;
        var state: UIState = .normal;
        var has_state = false;
        if (std.mem.indexOfScalar(u8, raw, ':')) |colon| {
            name = std.mem.trim(u8, raw[0..colon], " \t");
            const state_name = raw[colon + 1 ..];
            if (std.meta.stringToEnum(UIState, state_name)) |st| {
                if (st == .normal) {
                    self.addDiagAt(sel.start_line, sel.start_col, .unknown_selector, raw, true);
                    self.skipBlock();
                    return null;
                }
                state = st;
                has_state = true;
            } else {
                self.addDiagAt(sel.start_line, sel.start_col, .unknown_selector, raw, true);
                self.skipBlock();
                return null;
            }
        }

        if (name.len > 0 and name[0] == '.') {
            const class_name = name[1..];
            if (class_name.len == 0) {
                self.addDiagAt(sel.start_line, sel.start_col, .syntax, "empty class name", false);
                self.skipBlock();
                return null;
            }
            const idx = try self.classIndex(class_name);
            return Selector{ .target = .{ .class_idx = idx }, .state = state };
        }

        if (std.mem.eql(u8, name, "theme")) {
            // State suffixes on `theme` are meaningless (it fans out to all
            // widget kinds); report and skip rather than silently misapply.
            if (has_state) {
                self.addDiagAt(sel.start_line, sel.start_col, .unknown_selector, raw, true);
                self.skipBlock();
                return null;
            }
            return Selector{ .target = .theme_all };
        }

        if (kindFromName(name)) |kind| {
            return Selector{ .target = .{ .kind = kind }, .state = state };
        }

        self.addDiagAt(sel.start_line, sel.start_col, .unknown_selector, raw, true);
        self.skipBlock();
        return null;
    }

    /// Advances past the next block body (assumes the cursor is inside a
    /// selector). No nesting in this grammar: the first '}' closes.
    fn skipBlock(self: *Parser) void {
        while (self.peek()) |c| {
            self.advance();
            if (c == '}') return;
        }
    }

    fn classIndex(self: *Parser, name: []const u8) ParseError!usize {
        for (self.classes.items, 0..) |c, i| {
            if (std.mem.eql(u8, c.name, name)) return i;
        }
        const owned = try self.allocator.dupe(u8, name);
        try self.classes.append(self.allocator, .{ .name = owned, .set = .{} });
        return self.classes.items.len - 1;
    }

    fn parseBlock(self: *Parser, sel: Selector) ParseError!void {
        self.advance(); // consume '{'
        var block = Block{ .target = sel.target, .state = sel.state };
        while (true) {
            self.skipWsAndComments();
            const c = self.peek() orelse {
                self.addDiag(.syntax, "unterminated block");
                break;
            };
            if (c == '}') {
                self.advance();
                break;
            }
            try self.parseProperty(&block);
        }
        self.flushBlock(&block);
    }

    fn parseProperty(self: *Parser, block: *Block) ParseError!void {
        const prop_scan = self.scanUntil(":;}");

        if (prop_scan.slice.len == 0) {
            // Nothing before the terminator: ';' is junk (consume it so the
            // block loop makes progress), '}' / EOF just close the block.
            if (atStop(self, ";")) self.advance();
            return;
        }
        // A ';' or '}' before the ':' means a malformed property; report it
        // and let the block loop consume the terminator.
        if (self.peek() == null or atStop(self, ";}")) {
            self.addDiagAt(prop_scan.start_line, prop_scan.start_col, .syntax, "missing ':'", false);
            return;
        }
        self.advance(); // consume ':'
        const value_scan = self.scanUntil(";}");
        if (atStop(self, ";")) self.advance();

        const prop = prop_map.get(prop_scan.slice) orelse {
            // Unknown properties are a warning, not an error: theme files
            // from newer engine versions keep loading.
            self.addDiagAt(prop_scan.start_line, prop_scan.start_col, .unknown_property, prop_scan.slice, true);
            return;
        };
        if (value_scan.slice.len == 0) {
            self.addDiagAt(value_scan.start_line, value_scan.start_col, .invalid_value, prop_scan.slice, true);
            return;
        }
        self.applyProperty(block, prop, value_scan.slice, value_scan.start_line, value_scan.start_col);
    }

    fn applyProperty(self: *Parser, block: *Block, prop: PropKind, value: []const u8, line: usize, col: usize) void {
        const bad = struct {
            fn f(p: *Parser, l: usize, c: usize, v: []const u8) void {
                p.addDiagAt(l, c, .invalid_value, v, true);
            }
        }.f;

        switch (prop) {
            .background, .border_color, .text_color, .accent, .shadow_color => {
                const col_v = parseColor(value) orelse {
                    bad(self, line, col, value);
                    return;
                };
                switch (prop) {
                    .background => block.override.background = col_v,
                    .border_color => block.override.border_color = col_v,
                    .text_color => block.override.text_color = col_v,
                    .accent => block.override.accent = col_v,
                    .shadow_color => block.override.shadow = withShadowColor(block.override.shadow, col_v),
                    else => unreachable,
                }
            },
            .gradient => {
                // Two colors separated by whitespace: top then bottom.
                var it = std.mem.tokenizeAny(u8, value, " \t");
                const top_txt = it.next() orelse {
                    bad(self, line, col, value);
                    return;
                };
                const bottom_txt = it.next() orelse {
                    bad(self, line, col, value);
                    return;
                };
                if (it.next() != null) {
                    bad(self, line, col, value);
                    return;
                }
                const top = parseColor(top_txt) orelse {
                    bad(self, line, col, value);
                    return;
                };
                const bottom = parseColor(bottom_txt) orelse {
                    bad(self, line, col, value);
                    return;
                };
                block.override.gradient = .{ .top = top, .bottom = bottom };
            },
            .border_width => {
                block.override.border_width = parseLength(value) orelse {
                    bad(self, line, col, value);
                    return;
                };
            },
            .corner_radius => {
                block.override.corner_radius = parseLength(value) orelse {
                    bad(self, line, col, value);
                    return;
                };
            },
            .padding => {
                block.override.padding = parseLength(value) orelse {
                    bad(self, line, col, value);
                    return;
                };
            },
            .margin => {
                block.override.margin = parseLength(value) orelse {
                    bad(self, line, col, value);
                    return;
                };
            },
            .shadow_offset_x => {
                block.override.shadow = withShadowOffsetX(block.override.shadow, parseLength(value) orelse {
                    bad(self, line, col, value);
                    return;
                });
            },
            .shadow_offset_y => {
                block.override.shadow = withShadowOffsetY(block.override.shadow, parseLength(value) orelse {
                    bad(self, line, col, value);
                    return;
                });
            },
            .shadow_blur => {
                block.override.shadow = withShadowBlur(block.override.shadow, parseLength(value) orelse {
                    bad(self, line, col, value);
                    return;
                });
            },
            .opacity => {
                const v = std.fmt.parseFloat(f32, std.mem.trim(u8, value, " \t")) catch {
                    bad(self, line, col, value);
                    return;
                };
                block.override.opacity = v;
            },
            .transition_duration => {
                const ms = parseDurationMs(value) orelse {
                    bad(self, line, col, value);
                    return;
                };
                if (block.transition == null) block.transition = TransitionOptions{};
                block.transition.?.duration_ms = @max(ms, 0.0);
            },
            .transition_easing => {
                const e = parseEasing(value) orelse {
                    bad(self, line, col, value);
                    return;
                };
                if (block.transition == null) block.transition = TransitionOptions{};
                block.transition.?.easing = e;
            },
        }
    }

    /// Applies the accumulated block onto its destination: theme fan-out,
    /// one kind slot or one class slot. Later blocks merge field-wise.
    fn flushBlock(self: *Parser, block: *Block) void {
        switch (block.target) {
            .theme_all => {
                const all_sets = .{
                    &self.theme.panel,    &self.theme.button,     &self.theme.checkbox,
                    &self.theme.slider,   &self.theme.badge,      &self.theme.container,
                    &self.theme.dropdown, &self.theme.text_input, &self.theme.progress,
                };
                inline for (all_sets) |set| applyToSet(set, block);
                if (block.override.accent) |acc| {
                    self.theme.accent = acc;
                    self.accent_set = true;
                }
            },
            .kind => |kind| {
                applyToSet(kindSet(&self.theme, kind), block);
            },
            .class_idx => |idx| {
                applyToSet(&self.classes.items[idx].set, block);
            },
        }
    }

    fn applyToSet(set: *UIStyleSet, block: *Block) void {
        switch (block.state) {
            .normal => block.override.mergeInto(&set.normal),
            .hover => mergeDelta(&set.hover, block.override),
            .active => mergeDelta(&set.active, block.override),
            .focus => mergeDelta(&set.focus, block.override),
            .disabled => mergeDelta(&set.disabled, block.override),
        }
        if (block.transition) |t| set.transition = t;
    }

    fn mergeDelta(slot: *?UIStyleOverride, o: UIStyleOverride) void {
        if (slot.*) |*existing| {
            o.mergeInto(existing);
        } else {
            slot.* = o;
        }
    }
};

/// True when the cursor sits on one of `stops` (or at EOF when `stops`
/// contains nothing meaningful — EOF never matches).
fn atStop(self: *const Parser, stops: []const u8) bool {
    const c = self.peek() orelse return false;
    for (stops) |s| {
        if (c == s) return true;
    }
    return false;
}

fn withShadowColor(sh: ?types.UIShadow, color: Color4) types.UIShadow {
    var s = sh orelse types.UIShadow{};
    s.color = color;
    return s;
}

fn withShadowOffsetX(sh: ?types.UIShadow, v: f32) types.UIShadow {
    var s = sh orelse types.UIShadow{};
    s.offset_x = v;
    return s;
}

fn withShadowOffsetY(sh: ?types.UIShadow, v: f32) types.UIShadow {
    var s = sh orelse types.UIShadow{};
    s.offset_y = v;
    return s;
}

fn withShadowBlur(sh: ?types.UIShadow, v: f32) types.UIShadow {
    var s = sh orelse types.UIShadow{};
    s.blur = v;
    return s;
}

fn kindSet(theme: *UITheme, kind: UIStyleKind) *UIStyleSet {
    return theme.setForMut(kind);
}

/// Bare selector names reserved for theme kinds.
fn kindFromName(name: []const u8) ?UIStyleKind {
    const map = std.StaticStringMap(UIStyleKind).initComptime(.{
        .{ "panel", .panel },
        .{ "button", .button },
        .{ "checkbox", .checkbox },
        .{ "slider", .slider },
        .{ "badge", .badge },
        .{ "container", .container },
        .{ "dropdown", .dropdown },
        .{ "text_input", .text_input },
        .{ "text-input", .text_input },
        .{ "progress", .progress },
    });
    return map.get(name);
}

/// Parses theme CSS `text`. Result memory (class names, diag messages,
/// slices) is allocated from `allocator` — pass an arena to free in one
/// call. Malformed input produces diagnostics and keeps whatever parsed.
pub fn parseCss(allocator: std.mem.Allocator, text: []const u8) ParseError!CssTheme {
    var p = Parser{ .allocator = allocator, .text = text, .theme = UITheme.defaults() };
    return p.parse();
}

/// Filesystem errors `loadThemeFile` can surface beyond the parse errors.
pub const ThemeFileError = std.Io.File.OpenError || std.Io.File.StatError || std.Io.File.ReadPositionalError;

/// Reads a theme file (capped at `max_theme_css_bytes`) and parses it.
/// The returned class names and diagnostics are `gpa`-owned copies.
pub fn loadThemeFile(gpa: std.mem.Allocator, path: []const u8) (ParseError || ThemeFileError)!CssTheme {
    // Zig 0.16 removed std.fs.cwd(); read through the global single-threaded
    // Io (the engine-wide idiom, see texture.zig) so the public
    // (allocator, path) signature stays usable from app code.
    const io = std.Io.Threaded.global_single_threaded.io();
    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const file_size = try file.length(io);
    if (file_size > max_theme_css_bytes) return error.ThemeFileTooLarge;

    // Parse through a scratch arena, then copy the result out so the caller
    // gets a compact, independently freeable result.
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const buf = try arena.allocator().alloc(u8, @intCast(file_size));
    const read = try file.readPositionalAll(io, buf, 0);
    // A file truncated between stat and read parses as much as it still has
    // (the parser is lenient by design); nothing worth an error return.
    const parsed = try parseCss(arena.allocator(), buf[0..read]);

    var classes = try gpa.alloc(CssClass, parsed.classes.len);
    for (parsed.classes, 0..) |c, i| {
        classes[i] = .{ .name = try gpa.dupe(u8, c.name), .set = c.set };
    }
    var diags = try gpa.alloc(CssDiag, parsed.diags.len);
    for (parsed.diags, 0..) |d, i| {
        diags[i] = .{ .line = d.line, .col = d.col, .kind = d.kind, .message = try gpa.dupe(u8, d.message) };
    }
    return .{
        .theme = parsed.theme,
        .classes = classes,
        .diags = diags,
        .accent_set = parsed.accent_set,
    };
}

// ----------------------------------------------------------------------
// Tests
// ----------------------------------------------------------------------

const testing = std.testing;

/// Shared fixture: one of every selector shape the subset supports.
const fixture_css =
    \\/* engine theme */
    \\theme { accent: #7cb3ff; transition_duration: 120ms; }
    \\button { background: #141925d9; corner_radius: 4px; }
    \\button:hover { background: rgb(36, 48, 66); border_color: #a6d4ff; }
    \\.danger { background: #b3261e; transition_easing: ease-in-cubic; }
    \\.danger:disabled { opacity: 0.4; }
;

fn expectColorEql(a: Color4, b: Color4) !void {
    try testing.expectApproxEqAbs(a.r, b.r, 1e-4);
    try testing.expectApproxEqAbs(a.g, b.g, 1e-4);
    try testing.expectApproxEqAbs(a.b, b.b, 1e-4);
    try testing.expectApproxEqAbs(a.a, b.a, 1e-4);
}

test "parseCss: valid stylesheet yields theme slots, classes and state deltas" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const parsed = try parseCss(arena.allocator(), fixture_css);

    // A clean fixture produces no diagnostics.
    try testing.expectEqual(@as(usize, 0), parsed.diags.len);
    try testing.expect(parsed.accent_set);
    try expectColorEql(Color4.new(@as(f32, 0x7c) / 255.0, @as(f32, 0xb3) / 255.0, @as(f32, 0xff) / 255.0, 1.0), parsed.theme.accent);

    // `theme` fans the transition out to every kind's set.
    try testing.expectApproxEqAbs(@as(f32, 120.0), parsed.theme.button.transition.duration_ms, 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 120.0), parsed.theme.panel.transition.duration_ms, 1e-5);

    // Kind selector with hex color + alpha and a px length.
    const btn = parsed.theme.button;
    try expectColorEql(Color4.new(@as(f32, 0x14) / 255.0, @as(f32, 0x19) / 255.0, @as(f32, 0x25) / 255.0, @as(f32, 0xd9) / 255.0), btn.normal.background.?);
    try testing.expectApproxEqAbs(@as(f32, 4.0), btn.normal.corner_radius.?, 1e-5);

    // :hover delta with an rgb() color.
    try expectColorEql(Color4.new(36.0 / 255.0, 48.0 / 255.0, 66.0 / 255.0, 1.0), btn.hover.?.background.?);
    try expectColorEql(Color4.new(@as(f32, 0xa6) / 255.0, @as(f32, 0xd4) / 255.0, @as(f32, 0xff) / 255.0, 1.0), btn.hover.?.border_color.?);
    // Untouched states are not cleared: parseCss layers onto the default
    // palette, so :active still holds its built-in delta.
    try testing.expectApproxEqAbs(@as(f32, 0.18), btn.active.?.background.?.r, 1e-4);

    // Class with a per-state :disabled delta and its own transition easing.
    try testing.expectEqual(@as(usize, 1), parsed.classes.len);
    try testing.expectEqualStrings("danger", parsed.classes[0].name);
    const danger = parsed.classes[0].set;
    try expectColorEql(Color4.new(@as(f32, 0xb3) / 255.0, @as(f32, 0x26) / 255.0, @as(f32, 0x1e) / 255.0, 1.0), danger.normal.background.?);
    try testing.expectApproxEqAbs(@as(f32, 0.4), danger.disabled.?.opacity.?, 1e-5);
    try testing.expectEqual(easing.EasingType.ease_in_cubic, danger.transition.easing);

    // End-to-end: the parsed set resolves states like a hand-built one.
    const resolved = danger.resolve(.{}, .disabled);
    try testing.expectApproxEqAbs(@as(f32, 0.4), resolved.opacity, 1e-5);
    try expectColorEql(Color4.new(@as(f32, 0xb3) / 255.0, @as(f32, 0x26) / 255.0, @as(f32, 0x1e) / 255.0, 1.0), resolved.background);
}

test "parseCss: syntax error reports a diagnostic with line/column" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    // Line 2 has a value with no ':' before it (the ';' hits first).
    const src = "button { background: #101828; }\npanel { corner_radius 8px; }";
    const parsed = try parseCss(arena.allocator(), src);
    try testing.expect(parsed.diags.len >= 1);
    const d = parsed.diags[0];
    try testing.expectEqual(CssDiagKind.syntax, d.kind);
    try testing.expectEqual(@as(usize, 2), d.line);
    try testing.expect(d.col >= 8 and d.col <= 30);

    // The parser recovers: the valid first block was still applied.
    try testing.expect(parsed.theme.button.normal.background != null);
}

test "parseCss: unknown selector and unknown property warn but do not abort" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const src =
        \\frobnicator { background: #fff; }
        \\panel { background: #101828; glow_radius: 7px; padding: 4px; }
    ;
    const parsed = try parseCss(arena.allocator(), src);
    try testing.expectEqual(@as(usize, 2), parsed.diags.len);
    try testing.expectEqual(CssDiagKind.unknown_selector, parsed.diags[0].kind);
    try testing.expectEqualStrings("frobnicator", parsed.diags[0].message);
    try testing.expectEqual(CssDiagKind.unknown_property, parsed.diags[1].kind);
    try testing.expectEqualStrings("glow_radius", parsed.diags[1].message);

    // Both valid properties of the surviving block applied.
    try testing.expectApproxEqAbs(@as(f32, 4.0), parsed.theme.panel.normal.padding.?, 1e-5);
    try testing.expect(parsed.theme.panel.normal.background != null);
    // The skipped selector's white never leaked into the (default) button slot.
    try testing.expect(parsed.theme.button.normal.background.?.r < 0.5);
}

test "value parsers: hex shorthand, rgb/rgba, lengths, durations, easing" {
    // Shorthand expands by digit doubling (CSS #abc == #aabbcc).
    try expectColorEql(Color4.new(@as(f32, 0xaa) / 255.0, @as(f32, 0xbb) / 255.0, @as(f32, 0xcc) / 255.0, 1.0), parseHexColor("#abc").?);
    try expectColorEql(Color4.new(@as(f32, 0xaa) / 255.0, @as(f32, 0xbb) / 255.0, @as(f32, 0xcc) / 255.0, @as(f32, 0xdd) / 255.0), parseHexColor("#abcd").?);
    try expectColorEql(Color4.new(@as(f32, 0x10) / 255.0, @as(f32, 0x18) / 255.0, @as(f32, 0x28) / 255.0, 1.0), parseHexColor("#101828").?);
    try testing.expect(parseHexColor("#12345") == null);
    try testing.expect(parseHexColor("101828") == null);

    try expectColorEql(Color4.new(1.0 / 255.0, 2.0 / 255.0, 3.0 / 255.0, 1.0), parseRgbColor("rgb(1,2,3)").?);
    try expectColorEql(Color4.new(0, 0, 0, 0.5), parseRgbColor("rgba(0, 0, 0, 0.5)").?);
    // Channels clamp, matching lenient CSS.
    try expectColorEql(Color4.new(1.0, 0.0, 1.0, 1.0), parseRgbColor("rgb(300,-5,255)").?);

    // parseColor covers both forms; garbage fails.
    try expectColorEql(Color4.new(1, 0, 0, 1), parseColor("#f00").?);
    try expectColorEql(Color4.new(0, 1, 0, 1), parseColor("rgb(0,255,0)").?);
    try testing.expect(parseColor("null") == null);

    try testing.expectApproxEqAbs(@as(f32, 8.0), parseLength("8px").?, 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 1.5), parseLength("1.5").?, 1e-6);
    try testing.expect(parseLength("px") == null);

    try testing.expectApproxEqAbs(@as(f32, 150.0), parseDurationMs("150ms").?, 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 2000.0), parseDurationMs("2s").?, 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 90.0), parseDurationMs("90").?, 1e-6);

    try testing.expectEqual(easing.EasingType.ease_out_quad, parseEasing("ease-out-quad").?);
    try testing.expectEqual(easing.EasingType.linear, parseEasing("linear").?);
    try testing.expect(parseEasing("no-such-easing") == null);
}

test "loadThemeFile: reads, parses and copies a theme file" {
    const io = std.Io.Threaded.global_single_threaded.io();
    const path = "agate_ui_css_parser_test.tmp.css";
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = fixture_css });
    defer std.Io.Dir.cwd().deleteFile(io, path) catch {};

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const parsed = try loadThemeFile(arena.allocator(), path);
    try testing.expectEqual(@as(usize, 0), parsed.diags.len);
    try testing.expectEqual(@as(usize, 1), parsed.classes.len);
    try testing.expectEqualStrings("danger", parsed.classes[0].name);
    try testing.expect(parsed.theme.button.hover != null);
    try testing.expect(parsed.accent_set);
}

test "loadThemeFile: missing file is a file error, not a parse error" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const result = loadThemeFile(arena.allocator(), "agate_ui_css_parser_no_such_file.css");
    try testing.expectError(error.FileNotFound, result);
}
