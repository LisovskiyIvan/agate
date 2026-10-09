//! Fast, expressive object tagging and boolean query system (TagSet, TagQuery).
//! Supports boolean logic: AND (&, &&, and), OR (|, ||, or), NOT (!, not),
//! parentheses, implicit AND, and case-insensitive matching.

const std = @import("std");

pub const TagSet = struct {
    tags: std.ArrayListUnmanaged([]const u8) = .empty,

    /// Adds a tag to the set if not already present (case-insensitive).
    /// Returns true if added, false if already present.
    pub fn add(self: *TagSet, allocator: std.mem.Allocator, tag_str: []const u8) !bool {
        const trimmed = std.mem.trim(u8, tag_str, " \t\r\n");
        if (trimmed.len == 0) return false;
        if (self.has(trimmed)) return false;

        const duped = try allocator.dupe(u8, trimmed);
        try self.tags.append(allocator, duped);
        return true;
    }

    /// Adds multiple tags separated by whitespace or commas (e.g. "enemy, boss, flying").
    /// Returns the number of tags newly added.
    pub fn addMultiple(self: *TagSet, allocator: std.mem.Allocator, text: []const u8) !usize {
        var added: usize = 0;
        var i: usize = 0;
        while (i < text.len) {
            // Skip delimiters
            while (i < text.len and (text[i] == ' ' or text[i] == '\t' or text[i] == '\r' or text[i] == '\n' or text[i] == ',')) : (i += 1) {}
            if (i >= text.len) break;
            const start = i;
            while (i < text.len and text[i] != ' ' and text[i] != '\t' and text[i] != '\r' and text[i] != '\n' and text[i] != ',') : (i += 1) {}
            const chunk = text[start..i];
            if (try self.add(allocator, chunk)) {
                added += 1;
            }
        }
        return added;
    }

    /// Removes a tag from the set (case-insensitive).
    /// Returns true if removed, false if it was not in the set.
    pub fn remove(self: *TagSet, allocator: std.mem.Allocator, tag_str: []const u8) bool {
        const trimmed = std.mem.trim(u8, tag_str, " \t\r\n");
        if (trimmed.len == 0) return false;

        for (self.tags.items, 0..) |t, idx| {
            if (std.ascii.eqlIgnoreCase(t, trimmed)) {
                allocator.free(t);
                _ = self.tags.orderedRemove(idx);
                return true;
            }
        }
        return false;
    }

    /// Checks if a tag is in the set (case-insensitive).
    pub fn has(self: TagSet, tag_str: []const u8) bool {
        const trimmed = std.mem.trim(u8, tag_str, " \t\r\n");
        if (trimmed.len == 0) return false;

        for (self.tags.items) |t| {
            if (std.ascii.eqlIgnoreCase(t, trimmed)) return true;
        }
        return false;
    }

    /// Returns true if all given tags exist in this set.
    pub fn hasAll(self: TagSet, required_tags: []const []const u8) bool {
        for (required_tags) |t| {
            if (!self.has(t)) return false;
        }
        return true;
    }

    /// Returns true if at least one of the given tags exists in this set.
    pub fn hasAny(self: TagSet, candidates: []const []const u8) bool {
        for (candidates) |t| {
            if (self.has(t)) return true;
        }
        return false;
    }

    pub fn count(self: TagSet) usize {
        return self.tags.items.len;
    }

    pub fn get(self: TagSet, index: usize) ?[]const u8 {
        if (index < self.tags.items.len) return self.tags.items[index];
        return null;
    }

    pub fn slice(self: TagSet) []const []const u8 {
        return self.tags.items;
    }

    pub fn clear(self: *TagSet, allocator: std.mem.Allocator) void {
        for (self.tags.items) |t| {
            allocator.free(t);
        }
        self.tags.clearRetainingCapacity();
    }

    pub fn deinit(self: *TagSet, allocator: std.mem.Allocator) void {
        for (self.tags.items) |t| {
            allocator.free(t);
        }
        self.tags.deinit(allocator);
    }

    pub fn clone(self: TagSet, allocator: std.mem.Allocator) !TagSet {
        var copy = TagSet{};
        errdefer copy.deinit(allocator);
        for (self.tags.items) |t| {
            _ = try copy.add(allocator, t);
        }
        return copy;
    }

    /// Evaluates a pre-compiled boolean TagQuery against this set.
    pub fn matches(self: TagSet, query: *const TagQuery) bool {
        return query.matches(self);
    }

    /// Evaluates a boolean query string against this set with stack buffer optimization.
    pub fn matchesQuery(self: TagSet, query_str: []const u8) bool {
        var buf: [1024]u8 = undefined;
        var fba = std.heap.FixedBufferAllocator.init(&buf);
        var q = TagQuery.parse(fba.allocator(), query_str) catch {
            var fallback = TagQuery.parse(std.heap.page_allocator, query_str) catch return false;
            defer fallback.deinit();
            return fallback.matches(self);
        };
        defer q.deinit();
        return q.matches(self);
    }
};

pub const TagQuery = struct {
    arena: std.heap.ArenaAllocator,
    root: ?*TagQueryNode = null,

    pub const TagQueryNode = union(enum) {
        tag: []const u8,
        not_op: *TagQueryNode,
        and_op: struct { left: *TagQueryNode, right: *TagQueryNode },
        or_op: struct { left: *TagQueryNode, right: *TagQueryNode },

        pub fn evaluate(self: *const TagQueryNode, tag_set: anytype) bool {
            return switch (self.*) {
                .tag => |t| tag_set.has(t),
                .not_op => |child| !child.evaluate(tag_set),
                .and_op => |pair| pair.left.evaluate(tag_set) and pair.right.evaluate(tag_set),
                .or_op => |pair| pair.left.evaluate(tag_set) or pair.right.evaluate(tag_set),
            };
        }
    };

    pub fn parse(child_allocator: std.mem.Allocator, query_str: []const u8) !TagQuery {
        var arena = std.heap.ArenaAllocator.init(child_allocator);
        errdefer arena.deinit();
        const alloc = arena.allocator();

        var parser = Parser{
            .allocator = alloc,
            .src = query_str,
            .cursor = 0,
        };
        const root = try parser.parseExpression();
        return .{
            .arena = arena,
            .root = root,
        };
    }

    pub fn matches(self: TagQuery, tag_set: anytype) bool {
        if (self.root) |r| {
            return r.evaluate(tag_set);
        }
        return true; // Empty query matches all
    }

    pub fn deinit(self: *TagQuery) void {
        self.arena.deinit();
    }
};

const TokenKind = enum {
    ident,
    not_op,
    and_op,
    or_op,
    lparen,
    rparen,
    eof,
};

const Token = struct {
    kind: TokenKind,
    text: []const u8,
};

const Parser = struct {
    allocator: std.mem.Allocator,
    src: []const u8,
    cursor: usize = 0,

    fn isDelimiter(c: u8) bool {
        return c == ' ' or c == '\t' or c == '\r' or c == '\n' or c == ',';
    }

    fn skipWhitespace(self: *Parser) void {
        while (self.cursor < self.src.len and isDelimiter(self.src[self.cursor])) : (self.cursor += 1) {}
    }

    fn peek(self: *Parser) Token {
        const saved = self.cursor;
        const tok = self.next();
        self.cursor = saved;
        return tok;
    }

    fn next(self: *Parser) Token {
        self.skipWhitespace();
        if (self.cursor >= self.src.len) {
            return .{ .kind = .eof, .text = "" };
        }

        const c = self.src[self.cursor];
        if (c == '(') {
            self.cursor += 1;
            return .{ .kind = .lparen, .text = "(" };
        }
        if (c == ')') {
            self.cursor += 1;
            return .{ .kind = .rparen, .text = ")" };
        }
        if (c == '!') {
            self.cursor += 1;
            return .{ .kind = .not_op, .text = "!" };
        }
        if (c == '&') {
            self.cursor += 1;
            if (self.cursor < self.src.len and self.src[self.cursor] == '&') {
                self.cursor += 1;
            }
            return .{ .kind = .and_op, .text = "&" };
        }
        if (c == '|') {
            self.cursor += 1;
            if (self.cursor < self.src.len and self.src[self.cursor] == '|') {
                self.cursor += 1;
            }
            return .{ .kind = .or_op, .text = "|" };
        }

        // Identifier / Tag name
        const start = self.cursor;
        while (self.cursor < self.src.len) : (self.cursor += 1) {
            const ch = self.src[self.cursor];
            if (isDelimiter(ch) or ch == '(' or ch == ')' or ch == '&' or ch == '|' or ch == '!') {
                break;
            }
        }
        const text = self.src[start..self.cursor];

        if (std.ascii.eqlIgnoreCase(text, "and")) return .{ .kind = .and_op, .text = text };
        if (std.ascii.eqlIgnoreCase(text, "or")) return .{ .kind = .or_op, .text = text };
        if (std.ascii.eqlIgnoreCase(text, "not")) return .{ .kind = .not_op, .text = text };

        return .{ .kind = .ident, .text = text };
    }

    pub fn parseExpression(self: *Parser) !?*TagQuery.TagQueryNode {
        return self.parseOr();
    }

    fn parseOr(self: *Parser) anyerror!?*TagQuery.TagQueryNode {
        var left = (try self.parseAnd()) orelse return null;
        while (true) {
            const tok = self.peek();
            if (tok.kind == .or_op) {
                _ = self.next();
                const right = (try self.parseAnd()) orelse return error.TrailingOperator;
                const node = try self.allocator.create(TagQuery.TagQueryNode);
                node.* = .{ .or_op = .{ .left = left, .right = right } };
                left = node;
            } else {
                break;
            }
        }
        return left;
    }

    fn parseAnd(self: *Parser) anyerror!?*TagQuery.TagQueryNode {
        var left = (try self.parseUnary()) orelse return null;
        while (true) {
            const tok = self.peek();
            if (tok.kind == .and_op) {
                _ = self.next();
                const right = (try self.parseUnary()) orelse return error.TrailingOperator;
                const node = try self.allocator.create(TagQuery.TagQueryNode);
                node.* = .{ .and_op = .{ .left = left, .right = right } };
                left = node;
            } else if (tok.kind == .ident or tok.kind == .lparen or tok.kind == .not_op) {
                // Implicit AND between adjacent terms (e.g. "enemy boss" or "enemy !passive")
                const right = (try self.parseUnary()) orelse return error.TrailingOperator;
                const node = try self.allocator.create(TagQuery.TagQueryNode);
                node.* = .{ .and_op = .{ .left = left, .right = right } };
                left = node;
            } else {
                break;
            }
        }
        return left;
    }

    fn parseUnary(self: *Parser) anyerror!?*TagQuery.TagQueryNode {
        const tok = self.peek();
        if (tok.kind == .not_op) {
            _ = self.next();
            const child = (try self.parseUnary()) orelse return error.ExpectedOperand;
            const node = try self.allocator.create(TagQuery.TagQueryNode);
            node.* = .{ .not_op = child };
            return node;
        }
        return self.parsePrimary();
    }

    fn parsePrimary(self: *Parser) anyerror!?*TagQuery.TagQueryNode {
        const tok = self.peek();
        if (tok.kind == .ident) {
            _ = self.next();
            const node = try self.allocator.create(TagQuery.TagQueryNode);
            node.* = .{ .tag = tok.text };
            return node;
        } else if (tok.kind == .lparen) {
            _ = self.next();
            const inner = (try self.parseOr()) orelse return error.EmptyParentheses;
            const closing = self.next();
            if (closing.kind != .rparen) return error.MismatchedParentheses;
            return inner;
        } else if (tok.kind == .eof) {
            return null;
        } else {
            return error.UnexpectedToken;
        }
    }
};
