//! Asset cache. Split out of `asset_manager.zig` (facade).
//!
//! - `AssetCache`: in-memory maps (text / binary / texture) keyed by path,
//!   preventing duplicate disk reads and duplicate allocations.
//! - `textureHandlesEqual`: pure GPU-handle identity keeping `putTexture`
//!   self-put safe. Leaf: imports `std` and `texture.zig` only.

const std = @import("std");

const texture_mod = @import("../texture.zig");

const Texture = texture_mod.Texture;

/// In-memory asset cache indexing loaded resources by file path or identifier.
pub const AssetCache = struct {
    allocator: std.mem.Allocator,
    text_map: std.StringHashMapUnmanaged([]const u8) = .empty,
    binary_map: std.StringHashMapUnmanaged([]const u8) = .empty,
    texture_map: std.StringHashMapUnmanaged(Texture) = .empty,

    pub fn init(allocator: std.mem.Allocator) AssetCache {
        return .{
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *AssetCache) void {
        self.clear();
    }

    pub fn clear(self: *AssetCache) void {
        var text_it = self.text_map.iterator();
        while (text_it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*);
        }
        self.text_map.deinit(self.allocator);
        self.text_map = .empty;

        var bin_it = self.binary_map.iterator();
        while (bin_it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*);
        }
        self.binary_map.deinit(self.allocator);
        self.binary_map = .empty;

        var tex_it = self.texture_map.iterator();
        while (tex_it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            entry.value_ptr.deinit();
        }
        self.texture_map.deinit(self.allocator);
        self.texture_map = .empty;
    }

    pub fn hasText(self: *const AssetCache, key: []const u8) bool {
        return self.text_map.contains(key);
    }

    pub fn getText(self: *const AssetCache, key: []const u8) ?[]const u8 {
        return self.text_map.get(key);
    }

    pub fn putText(self: *AssetCache, key: []const u8, text: []const u8) ![]const u8 {
        const key_dupe = try self.allocator.dupe(u8, key);
        errdefer self.allocator.free(key_dupe);
        const text_dupe = try self.allocator.dupe(u8, text);
        errdefer self.allocator.free(text_dupe);
        const gop = try self.text_map.getOrPut(self.allocator, key);
        if (gop.found_existing) {
            // Overwrite: keep the original key, free the dupe + old value.
            self.allocator.free(key_dupe);
            self.allocator.free(gop.value_ptr.*);
            gop.value_ptr.* = text_dupe;
            return gop.value_ptr.*;
        }
        gop.key_ptr.* = key_dupe;
        gop.value_ptr.* = text_dupe;
        return text_dupe;
    }

    pub fn hasBinary(self: *const AssetCache, key: []const u8) bool {
        return self.binary_map.contains(key);
    }

    pub fn getBinary(self: *const AssetCache, key: []const u8) ?[]const u8 {
        return self.binary_map.get(key);
    }

    pub fn putBinary(self: *AssetCache, key: []const u8, data: []const u8) ![]const u8 {
        const key_dupe = try self.allocator.dupe(u8, key);
        errdefer self.allocator.free(key_dupe);
        const data_dupe = try self.allocator.dupe(u8, data);
        errdefer self.allocator.free(data_dupe);
        const gop = try self.binary_map.getOrPut(self.allocator, key);
        if (gop.found_existing) {
            self.allocator.free(key_dupe);
            self.allocator.free(gop.value_ptr.*);
            gop.value_ptr.* = data_dupe;
            return gop.value_ptr.*;
        }
        gop.key_ptr.* = key_dupe;
        gop.value_ptr.* = data_dupe;
        return data_dupe;
    }

    pub fn hasTexture(self: *const AssetCache, key: []const u8) bool {
        return self.texture_map.contains(key);
    }

    pub fn getTexture(self: *const AssetCache, key: []const u8) ?Texture {
        return self.texture_map.get(key);
    }

    pub fn putTexture(self: *AssetCache, key: []const u8, texture: Texture) !void {
        const key_dupe = try self.allocator.dupe(u8, key);
        errdefer self.allocator.free(key_dupe);
        const gop = try self.texture_map.getOrPut(self.allocator, key);
        if (gop.found_existing) {
            // Self-put of the identical GPU handles (e.g. re-caching the
            // borrowed alias) is a no-op: deinit would destroy the live
            // texture out from under the inserted alias.
            if (textureHandlesEqual(gop.value_ptr.*, texture)) {
                self.allocator.free(key_dupe);
                return;
            }
            // Overwrite: keep the original key, destroy the old GPU texture.
            // Transfer semantics: `texture` ownership moves to the cache;
            // never pass a borrowed cache alias with different handles.
            self.allocator.free(key_dupe);
            gop.value_ptr.deinit();
            gop.value_ptr.* = texture;
            return;
        }
        gop.key_ptr.* = key_dupe;
        gop.value_ptr.* = texture;
    }

    pub fn count(self: *const AssetCache) usize {
        return self.text_map.count() + self.binary_map.count() + self.texture_map.count();
    }
};

/// Pure GPU-handle identity for textures (no sg calls): image/view/sampler
/// ids plus dimensions. Used to make `putTexture` self-put safe.
pub fn textureHandlesEqual(a: Texture, b: Texture) bool {
    return a.image.id == b.image.id and
        a.view.id == b.view.id and
        a.sampler.id == b.sampler.id and
        a.width == b.width and
        a.height == b.height;
}
