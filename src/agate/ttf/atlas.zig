//! CPU atlas + UI-facing font handle (split out of `ttf.zig`, facade).
//!
//! Owns `GlyphInfo` and `TtfFont` (bake, shelf packer, lookup, kern, UVs)
//! with all of its methods. Cross-leaf calls reach the `outline`/`raster`
//! siblings directly (`outline.extractOutline`, `raster.flattenContours`,
//! ...) — same discipline as `audio/*`; only the module prefix changed,
//! behavior is identical. `Font` resolves through the `font` sibling.

const std = @import("std");
const types = @import("types.zig");
const font_mod = @import("font.zig");
const outline = @import("outline.zig");
const raster = @import("raster.zig");

const TtfError = types.TtfError;
const max_glyph_bitmap: u32 = types.max_glyph_bitmap;
const atlas_size: u32 = types.atlas_size;
const atlas_pad: u32 = types.atlas_pad;
const Font = font_mod.Font;

/// One baked glyph: atlas rect + font-pixel metrics (bake scale applied).
pub const GlyphInfo = struct {
    codepoint: u21,
    gid: u16,
    ax: u32, // atlas origin (content, without gutter)
    ay: u32,
    w: u32,
    h: u32,
    bearing_x: f32, // bitmap left relative to pen, bake px
    bearing_y: f32, // bitmap top BELOW... (see below), bake px
    advance: f32, // horizontal advance, bake px
};

/// UI-facing TrueType font: parsed `Font` (borrowing caller `data`) plus
/// an RGBA8 shelf atlas (white + alpha) baked at `pixel_size`. Obtain via
/// `init`; free with `deinit`. The atlas pixels (`atlas_pixels`,
/// `atlas_size` x `atlas_size`) upload 1:1 through Texture.initRaw.
pub const TtfFont = struct {
    allocator: std.mem.Allocator,
    font: Font,
    pixel_size: f32,
    scale: f32, // pixel_size / unitsPerEm
    ascent: f32, // bake px
    descent: f32, // bake px (negative or zero)
    line_gap: f32, // bake px
    line_height: f32, // bake px
    glyphs: []GlyphInfo,
    notdef: ?GlyphInfo,
    atlas_pixels: []u8, // RGBA8, atlas_size^2 * 4
    shelf_x: u32 = 0,
    shelf_y: u32 = 0,
    shelf_h: u32 = 0,

    /// Bakes `codepoints` (plus .notdef) at `pixel_size` px em. Duplicate
    /// codepoints mapping to one gid share the atlas cell. Errors:
    /// parse failures, error.AtlasFull (glyphs do not fit 512x512 —
    /// bake fewer/smaller), error.InvalidPixelSize for px outside 4..256.
    pub fn init(
        allocator: std.mem.Allocator,
        data: []const u8,
        pixel_size: f32,
        codepoints: []const u21,
    ) TtfError!TtfFont {
        if (!(pixel_size >= 4.0 and pixel_size <= 256.0)) return error.InvalidPixelSize;
        const font = try Font.parse(data);
        const scale = pixel_size / @as(f32, @floatFromInt(font.units_per_em));
        var self = TtfFont{
            .allocator = allocator,
            .font = font,
            .pixel_size = pixel_size,
            .scale = scale,
            .ascent = @as(f32, @floatFromInt(font.ascender)) * scale,
            .descent = @as(f32, @floatFromInt(font.descender)) * scale,
            .line_gap = @as(f32, @floatFromInt(font.line_gap)) * scale,
            .line_height = (@as(f32, @floatFromInt(font.ascender)) -
                @as(f32, @floatFromInt(font.descender)) +
                @as(f32, @floatFromInt(font.line_gap))) * scale,
            .glyphs = &.{},
            .notdef = null,
            .atlas_pixels = allocator.alloc(u8, @as(usize, atlas_size) * atlas_size * 4) catch return error.OutOfMemory,
        };
        errdefer self.deinit();
        @memset(self.atlas_pixels, 0);

        // Unique gid set: requested codepoints + gid 0 (.notdef).
        var ugids: [512]u16 = undefined;
        var n_ugids: usize = 0;
        ugids[0] = 0;
        n_ugids = 1;
        for (codepoints) |cp| {
            const g = try font.glyphIndex(cp);
            var found = false;
            for (ugids[0..n_ugids]) |u| {
                if (u == g) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                if (n_ugids >= ugids.len) return error.AtlasFull;
                ugids[n_ugids] = g;
                n_ugids += 1;
            }
        }

        // Bake each unique gid once; remember its cell + metrics.
        var baked = allocator.alloc(BakedGlyph, n_ugids) catch return error.OutOfMemory;
        defer allocator.free(baked);
        for (ugids[0..n_ugids], 0..) |gid, k| {
            baked[k] = try self.bakeGlyph(gid);
        }

        // Per-codepoint entries (shared cells for shared gids).
        var list: std.ArrayListUnmanaged(GlyphInfo) = .empty;
        errdefer list.deinit(allocator);
        for (codepoints) |cp| {
            const g = try font.glyphIndex(cp);
            for (baked) |b| {
                if (b.gid == g) {
                    try list.append(allocator, .{
                        .codepoint = cp,
                        .gid = g,
                        .ax = b.ax,
                        .ay = b.ay,
                        .w = b.w,
                        .h = b.h,
                        .bearing_x = b.bearing_x,
                        .bearing_y = b.bearing_y,
                        .advance = b.advance,
                    });
                    break;
                }
            }
        }
        self.glyphs = try list.toOwnedSlice(allocator);
        for (baked) |b| {
            if (b.gid == 0) {
                self.notdef = GlyphInfo{
                    .codepoint = 0xFFFD,
                    .gid = 0,
                    .ax = b.ax,
                    .ay = b.ay,
                    .w = b.w,
                    .h = b.h,
                    .bearing_x = b.bearing_x,
                    .bearing_y = b.bearing_y,
                    .advance = b.advance,
                };
                break;
            }
        }
        return self;
    }

    pub fn deinit(self: *TtfFont) void {
        if (self.atlas_pixels.len > 0) self.allocator.free(self.atlas_pixels);
        if (self.glyphs.len > 0) self.allocator.free(self.glyphs);
        self.* = undefined;
    }

    /// Codepoint lookup within the baked set; unmapped codepoints fall
    /// back to .notdef, unknown-to-the-set returns null (caller skips).
    pub fn lookup(self: *const TtfFont, codepoint: u21) ?GlyphInfo {
        for (self.glyphs) |g| {
            if (g.codepoint == codepoint) return g;
        }
        return self.notdef;
    }

    /// Kerning between two baked glyphs in bake px (0 without kern data).
    pub fn kernPx(self: *const TtfFont, left: GlyphInfo, right: GlyphInfo) f32 {
        const k = self.font.kernUnits(left.gid, right.gid) catch 0;
        return @as(f32, @floatFromInt(k)) * self.scale;
    }

    /// Normalized atlas UVs for a baked glyph (content rect, no gutter).
    pub fn glyphUv(self: *const TtfFont, g: GlyphInfo) [4]f32 {
        _ = self;
        const s: f32 = @floatFromInt(atlas_size);
        return .{
            @as(f32, @floatFromInt(g.ax)) / s,
            @as(f32, @floatFromInt(g.ay)) / s,
            @as(f32, @floatFromInt(g.ax + g.w)) / s,
            @as(f32, @floatFromInt(g.ay + g.h)) / s,
        };
    }

    const BakedGlyph = struct {
        gid: u16,
        ax: u32,
        ay: u32,
        w: u32,
        h: u32,
        bearing_x: f32,
        bearing_y: f32,
        advance: f32,
    };

    fn bakeGlyph(self: *TtfFont, gid: u16) TtfError!BakedGlyph {
        const advance = @as(f32, @floatFromInt(self.font.advanceUnits(gid))) * self.scale;
        const contours = try outline.extractOutline(self.allocator, &self.font, gid);
        defer outline.freeContours(self.allocator, contours);
        if (contours.len == 0) {
            // Empty glyph (space, empty .notdef): valid entry, no cell.
            return .{ .gid = gid, .ax = 0, .ay = 0, .w = 0, .h = 0, .bearing_x = 0, .bearing_y = 0, .advance = advance };
        }
        const segs = try raster.flattenContours(self.allocator, contours, self.scale, 0, 0, 0.25);
        defer self.allocator.free(segs);
        if (segs.len == 0) {
            return .{ .gid = gid, .ax = 0, .ay = 0, .w = 0, .h = 0, .bearing_x = 0, .bearing_y = 0, .advance = advance };
        }
        var min_x: f32 = segs[0].x0;
        var max_x: f32 = segs[0].x0;
        var min_y: f32 = segs[0].y0;
        var max_y: f32 = segs[0].y0;
        for (segs) |s| {
            min_x = @min(min_x, @min(s.x0, s.x1));
            max_x = @max(max_x, @max(s.x0, s.x1));
            min_y = @min(min_y, @min(s.y0, s.y1));
            max_y = @max(max_y, @max(s.y0, s.y1));
        }
        // y-down device space with baseline at 0: bearing_y is the bitmap
        // top's distance BELOW the baseline's upward extent... concretely:
        // bitmap rows cover [floor(min_y), ceil(max_y)); drawing places
        // the bitmap top at baseline - bearing_y where
        // bearing_y = -floor(min_y) (positive when the glyph rises above
        // the baseline, as usual for Latin ascenders).
        const ox = @as(i32, @intFromFloat(@floor(min_x)));
        const oy = @as(i32, @intFromFloat(@floor(min_y)));
        const w: u32 = @intCast(@as(i32, @intFromFloat(@ceil(max_x))) - ox);
        const h: u32 = @intCast(@as(i32, @intFromFloat(@ceil(max_y))) - oy);
        if (w == 0 or h == 0) {
            return .{ .gid = gid, .ax = 0, .ay = 0, .w = 0, .h = 0, .bearing_x = 0, .bearing_y = 0, .advance = advance };
        }
        if (w > max_glyph_bitmap or h > max_glyph_bitmap) return error.GlyphTooLarge;
        const alpha = try raster.rasterizeSegments(self.allocator, segs, ox, oy, w, h);
        defer self.allocator.free(alpha);
        const cell = try self.placeCell(w, h);
        // Blit white + alpha (mode 3 samples .a for coverage).
        var row: u32 = 0;
        while (row < h) : (row += 1) {
            var col: u32 = 0;
            while (col < w) : (col += 1) {
                const dst = (@as(usize, cell.y + row) * atlas_size + cell.x + col) * 4;
                self.atlas_pixels[dst + 0] = 255;
                self.atlas_pixels[dst + 1] = 255;
                self.atlas_pixels[dst + 2] = 255;
                self.atlas_pixels[dst + 3] = alpha[@as(usize, row) * w + col];
            }
        }
        return .{
            .gid = gid,
            .ax = cell.x,
            .ay = cell.y,
            .w = w,
            .h = h,
            .bearing_x = @as(f32, @floatFromInt(ox)),
            .bearing_y = -@as(f32, @floatFromInt(oy)),
            .advance = advance,
        };
    }

    /// Shelf packer: rows left to right, 1px gutters. error.AtlasFull
    /// when the cell fits neither the current row nor a new row.
    fn placeCell(self: *TtfFont, w: u32, h: u32) TtfError!struct { x: u32, y: u32 } {
        const cw = w + atlas_pad * 2;
        const ch = h + atlas_pad * 2;
        if (cw > atlas_size or ch > atlas_size) return error.AtlasFull;
        if (self.shelf_x + cw > atlas_size) {
            self.shelf_x = 0;
            self.shelf_y += self.shelf_h;
            self.shelf_h = 0;
        }
        if (self.shelf_y + ch > atlas_size) return error.AtlasFull;
        const x = self.shelf_x + atlas_pad;
        const y = self.shelf_y + atlas_pad;
        self.shelf_x += cw;
        self.shelf_h = @max(self.shelf_h, ch);
        return .{ .x = x, .y = y };
    }
};
