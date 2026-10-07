//! Shared vocabulary for the TTF leaves (split out of `ttf.zig`, facade).
//!
//! Error set, bounds-checked big-endian readers, sfnt table tags, parse
//! limits and atlas sizing. Imports `std` only — every other leaf may
//! import this module; it imports nothing back (anti-cycle root).

const std = @import("std");

pub const TtfError = error{
    NotTtf,
    Truncated,
    MissingTable,
    BadTable,
    BadHead,
    BadCmap,
    BadLoca,
    BadGlyf,
    UnsupportedCff,
    UnsupportedVariableFont,
    UnsupportedCmap,
    CompositeTooDeep,
    CyclicComposite,
    AtlasFull,
    GlyphTooLarge,
    InvalidPixelSize,
    OutOfMemory,
};

pub fn readU16At(bytes: []const u8, off: usize) TtfError!u16 {
    if (off > bytes.len or 2 > bytes.len - off) return error.Truncated;
    return std.mem.readInt(u16, bytes[off..][0..2], .big);
}

pub fn readI16At(bytes: []const u8, off: usize) TtfError!i16 {
    return @bitCast(try readU16At(bytes, off));
}

pub fn readU32At(bytes: []const u8, off: usize) TtfError!u32 {
    if (off > bytes.len or 4 > bytes.len - off) return error.Truncated;
    return std.mem.readInt(u32, bytes[off..][0..4], .big);
}

pub const tag_head: u32 = 0x68656164; // 'head'
pub const tag_maxp: u32 = 0x6D617870; // 'maxp'
pub const tag_cmap: u32 = 0x636D6170; // 'cmap'
pub const tag_loca: u32 = 0x6C6F6361; // 'loca'
pub const tag_glyf: u32 = 0x676C7966; // 'glyf'
pub const tag_hhea: u32 = 0x68686561; // 'hhea'
pub const tag_hmtx: u32 = 0x686D7478; // 'hmtx'
pub const tag_kern: u32 = 0x6B65726E; // 'kern'
pub const tag_cff: u32 = 0x43464620; // 'CFF '
pub const tag_cff2: u32 = 0x43464632; // 'CFF2'
pub const tag_fvar: u32 = 0x66766172; // 'fvar'

pub const max_tables: usize = 64;
pub const max_cmap_records: usize = 64;
pub const max_contours: usize = 256;
pub const max_points: usize = 8192;
pub const max_components: usize = 64;
pub const max_composite_depth: u8 = 8;
pub const max_glyph_bitmap: u32 = 512;

pub const atlas_size: u32 = 512;
pub const atlas_pad: u32 = 1; // 1px gutter stops LINEAR bleed between cells
