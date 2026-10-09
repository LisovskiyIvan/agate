const std = @import("std");
const tags = @import("tags.zig");
const TagSet = tags.TagSet;
const TagQuery = tags.TagQuery;

test "TagSet basic operations" {
    const alloc = std.testing.allocator;
    var set = TagSet{};
    defer set.deinit(alloc);

    try std.testing.expect(try set.add(alloc, "Enemy"));
    try std.testing.expect(!try set.add(alloc, "enemy")); // Deduplication
    try std.testing.expectEqual(@as(usize, 1), set.count());
    try std.testing.expect(set.has("enemy"));
    try std.testing.expect(set.has("ENEMY"));
    try std.testing.expect(!set.has("boss"));

    _ = try set.addMultiple(alloc, "boss, flying, elite");
    try std.testing.expectEqual(@as(usize, 4), set.count());
    try std.testing.expect(set.has("boss"));
    try std.testing.expect(set.has("flying"));
    try std.testing.expect(set.has("elite"));

    try std.testing.expect(set.hasAll(&[_][]const u8{ "enemy", "boss" }));
    try std.testing.expect(!set.hasAll(&[_][]const u8{ "enemy", "passive" }));
    try std.testing.expect(set.hasAny(&[_][]const u8{ "passive", "elite" }));
    try std.testing.expect(!set.hasAny(&[_][]const u8{ "passive", "friendly" }));

    try std.testing.expect(set.remove(alloc, "BOSS"));
    try std.testing.expect(!set.has("boss"));
    try std.testing.expectEqual(@as(usize, 3), set.count());
}

test "TagQuery boolean expression parser and evaluation" {
    const alloc = std.testing.allocator;
    var set = TagSet{};
    defer set.deinit(alloc);
    _ = try set.addMultiple(alloc, "enemy, flying, undead");

    // Simple tag
    try std.testing.expect(set.matchesQuery("enemy"));
    try std.testing.expect(!set.matchesQuery("boss"));

    // Negation
    try std.testing.expect(set.matchesQuery("!boss"));
    try std.testing.expect(!set.matchesQuery("!enemy"));
    try std.testing.expect(set.matchesQuery("not boss"));

    // Conjunction (AND)
    try std.testing.expect(set.matchesQuery("enemy & flying"));
    try std.testing.expect(set.matchesQuery("enemy && flying"));
    try std.testing.expect(set.matchesQuery("enemy and flying"));
    try std.testing.expect(!set.matchesQuery("enemy & boss"));

    // Disjunction (OR)
    try std.testing.expect(set.matchesQuery("boss | undead"));
    try std.testing.expect(set.matchesQuery("boss || undead"));
    try std.testing.expect(set.matchesQuery("boss or undead"));
    try std.testing.expect(!set.matchesQuery("boss | humanoid"));

    // Complex expressions with parentheses
    try std.testing.expect(set.matchesQuery("enemy & (boss | flying)"));
    try std.testing.expect(set.matchesQuery("(enemy | boss) & (flying | aquatic)"));
    try std.testing.expect(!set.matchesQuery("enemy & (boss | aquatic)"));
    try std.testing.expect(set.matchesQuery("enemy && !boss && (flying || mechanical)"));

    // Implicit AND
    try std.testing.expect(set.matchesQuery("enemy flying"));
    try std.testing.expect(!set.matchesQuery("enemy boss"));
    try std.testing.expect(set.matchesQuery("enemy !boss"));
}

test "TagQuery pre-compiled reuse" {
    const alloc = std.testing.allocator;
    var set_enemy = TagSet{};
    defer set_enemy.deinit(alloc);
    _ = try set_enemy.addMultiple(alloc, "enemy, boss");

    var set_hero = TagSet{};
    defer set_hero.deinit(alloc);
    _ = try set_hero.addMultiple(alloc, "player, hero");

    var query = try TagQuery.parse(alloc, "enemy && (boss || elite)");
    defer query.deinit();

    try std.testing.expect(query.matches(set_enemy));
    try std.testing.expect(!query.matches(set_hero));
}
