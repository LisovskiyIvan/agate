const std = @import("std");
const input_state = @import("input_state.zig");

const isPointInRect = input_state.isPointInRect;
const sliderValueAt = input_state.sliderValueAt;
const checkboxHitRect = input_state.checkboxHitRect;
const dropdownItemRect = input_state.dropdownItemRect;
const dropdownItemHeight = input_state.dropdownItemHeight;
const dropdownHit = input_state.dropdownHit;
const ScrollState = input_state.ScrollState;
const scrollClamp = input_state.scrollClamp;
const scrollOffsetForItem = input_state.scrollOffsetForItem;
const scrollbarThumbRect = input_state.scrollbarThumbRect;
const TextInputState = input_state.TextInputState;

test "UICanvas isPointInRect" {
    try std.testing.expect(isPointInRect(50, 50, 0, 0, 100, 100));
    try std.testing.expect(!isPointInRect(150, 50, 0, 0, 100, 100));
}

test "UICanvas sliderValueAt" {
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), sliderValueAt(10, 100, 60), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), sliderValueAt(10, 100, 10), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sliderValueAt(10, 100, 110), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), sliderValueAt(10, 100, -50), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sliderValueAt(10, 100, 500), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), sliderValueAt(10, 0, 60), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), sliderValueAt(10, -20, 60), 1e-5);
}

test "UICanvas checkboxHitRect" {
    const r = checkboxHitRect(10, 20, 24);
    try std.testing.expectApproxEqAbs(@as(f32, 10), r[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 20), r[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 24), r[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 24), r[3], 1e-5);
    try std.testing.expect(isPointInRect(15, 25, r[0], r[1], r[2], r[3]));
    try std.testing.expect(!isPointInRect(100, 100, r[0], r[1], r[2], r[3]));
}

test "UICanvas dropdownItemRect" {
    const btn: [4]f32 = .{ 10, 20, 120, 28 };
    const r0 = dropdownItemRect(btn, 24, 0);
    try std.testing.expectApproxEqAbs(@as(f32, 10), r0[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 48), r0[1], 1e-5); // stacked below: 20 + 28
    try std.testing.expectApproxEqAbs(@as(f32, 120), r0[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 24), r0[3], 1e-5);

    const r2 = dropdownItemRect(btn, 24, 2);
    try std.testing.expectApproxEqAbs(@as(f32, 48 + 2 * 24), r2[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 120), r2[2], 1e-5);

    // Shared item height: draw + hit-test geometry always agree.
    try std.testing.expectApproxEqAbs(@as(f32, 24.0), dropdownItemHeight(16.0), 1e-5);
}

test "UICanvas dropdownHit" {
    const btn: [4]f32 = .{ 10, 20, 120, 28 };
    const item_h: f32 = 24; // open list spans y = 48..120 for count 3
    try std.testing.expectEqual(@as(?usize, 0), dropdownHit(btn, item_h, 3, 50, 48)); // top edge
    try std.testing.expectEqual(@as(?usize, 0), dropdownHit(btn, item_h, 3, 50, 60));
    try std.testing.expectEqual(@as(?usize, 1), dropdownHit(btn, item_h, 3, 50, 72)); // row boundary -> next row
    try std.testing.expectEqual(@as(?usize, 2), dropdownHit(btn, item_h, 3, 50, 119));
    try std.testing.expectEqual(@as(?usize, 2), dropdownHit(btn, item_h, 3, 50, 120)); // bottom edge inclusive
    try std.testing.expectEqual(@as(?usize, 0), dropdownHit(btn, item_h, 3, 10, 60)); // x edges inclusive
    try std.testing.expectEqual(@as(?usize, 2), dropdownHit(btn, item_h, 3, 130, 100));

    // The closed button is NOT part of the list hit area.
    try std.testing.expectEqual(@as(?usize, null), dropdownHit(btn, item_h, 3, 50, 30));
    try std.testing.expectEqual(@as(?usize, null), dropdownHit(btn, item_h, 3, 50, 47.9));
    // Outside the list: x miss, below the list, empty list, degenerate height.
    try std.testing.expectEqual(@as(?usize, null), dropdownHit(btn, item_h, 3, 9, 60));
    try std.testing.expectEqual(@as(?usize, null), dropdownHit(btn, item_h, 3, 131, 60));
    try std.testing.expectEqual(@as(?usize, null), dropdownHit(btn, item_h, 3, 50, 121));
    try std.testing.expectEqual(@as(?usize, null), dropdownHit(btn, item_h, 0, 50, 60));
    try std.testing.expectEqual(@as(?usize, null), dropdownHit(btn, 0, 3, 50, 60));
}

test "UICanvas scrollClamp" {
    var s = ScrollState{ .offset = 0, .content_h = 500, .view_h = 200 }; // max 300
    scrollClamp(&s, 100);
    try std.testing.expectApproxEqAbs(@as(f32, 100), s.offset, 1e-5);
    scrollClamp(&s, 500); // clamp at the bottom
    try std.testing.expectApproxEqAbs(@as(f32, 300), s.offset, 1e-5);
    scrollClamp(&s, -1000); // clamp at the top
    try std.testing.expectApproxEqAbs(@as(f32, 0), s.offset, 1e-5);

    // Content smaller than the view: no scrolling, offset resets to 0.
    var small = ScrollState{ .offset = 50, .content_h = 100, .view_h = 200 };
    scrollClamp(&small, 10);
    try std.testing.expectApproxEqAbs(@as(f32, 0), small.offset, 1e-5);

    // scrollOffsetForItem: visible item keeps the offset, hidden item scrolls minimally.
    try std.testing.expectApproxEqAbs(@as(f32, 100), scrollOffsetForItem(100, 150, 20, 200, 500), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 10), scrollOffsetForItem(100, 10, 20, 200, 500), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 220), scrollOffsetForItem(100, 400, 20, 200, 500), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0), scrollOffsetForItem(100, 400, 20, 200, 100), 1e-5);
}

test "UICanvas scrollbarThumbRect" {
    const track: [4]f32 = .{ 0, 0, 12, 200 };
    // Content fits: thumb covers the full track.
    const full = scrollbarThumbRect(track, 100, 200, 0);
    try std.testing.expectApproxEqAbs(track[0], full[0], 1e-5);
    try std.testing.expectApproxEqAbs(track[1], full[1], 1e-5);
    try std.testing.expectApproxEqAbs(track[2], full[2], 1e-5);
    try std.testing.expectApproxEqAbs(track[3], full[3], 1e-5);

    // Half visible: half-height thumb, top at offset 0...
    const top = scrollbarThumbRect(track, 400, 200, 0);
    try std.testing.expectApproxEqAbs(@as(f32, 100), top[3], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0), top[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 12), top[2], 1e-5);
    // ...pinned to the track bottom at max offset.
    const bottom = scrollbarThumbRect(track, 400, 200, 200);
    try std.testing.expectApproxEqAbs(@as(f32, 100), bottom[3], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 100), bottom[1], 1e-5); // 200 - 100

    // Tiny view ratio: thumb clamped to the 16px minimum.
    const tiny = scrollbarThumbRect(track, 4000, 200, 0);
    try std.testing.expectApproxEqAbs(@as(f32, 16), tiny[3], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0), tiny[1], 1e-5);
}

test "UICanvas TextInput editing" {
    var st = TextInputState{};
    try std.testing.expect(st.insertChar('h'));
    try std.testing.expect(st.insertChar('i'));
    try std.testing.expectEqualStrings("hi", st.text());
    try std.testing.expectEqual(@as(usize, 2), st.cursor);

    _ = st.moveLeft();
    try std.testing.expect(st.insertChar('e'));
    try std.testing.expectEqualStrings("hei", st.text());
    _ = st.moveRight();
    try std.testing.expectEqual(@as(usize, 3), st.cursor);

    try std.testing.expect(st.backspace());
    try std.testing.expectEqualStrings("he", st.text());
    st.home();
    try std.testing.expect(st.deleteForward());
    try std.testing.expectEqualStrings("e", st.text());
    st.end();
    try std.testing.expectEqual(@as(usize, 1), st.cursor);

    // Edges are no-ops reporting false.
    st.home();
    try std.testing.expect(!st.backspace());
    try std.testing.expect(!st.moveLeft());
    st.end();
    try std.testing.expect(!st.deleteForward());
    try std.testing.expect(!st.moveRight());

    // setText replaces the buffer and moves the cursor to the end.
    st.setText("hello");
    try std.testing.expectEqualStrings("hello", st.text());
    try std.testing.expectEqual(@as(usize, 5), st.cursor);
}

test "UICanvas TextInput UTF-8 boundaries" {
    var st = TextInputState{};
    try std.testing.expect(st.insertChar('ж')); // U+0436, 2 bytes in UTF-8
    try std.testing.expectEqual(@as(usize, 2), st.len);
    try std.testing.expectEqual(@as(usize, 2), st.cursor);

    // Cursor moves step over the whole codepoint, never splitting it.
    try std.testing.expect(st.moveLeft());
    try std.testing.expectEqual(@as(usize, 0), st.cursor);
    try std.testing.expect(st.moveRight());
    try std.testing.expectEqual(@as(usize, 2), st.cursor);

    // Backspace removes both bytes at once.
    try std.testing.expect(st.backspace());
    try std.testing.expectEqual(@as(usize, 0), st.len);
    try std.testing.expect(!st.backspace());

    // deleteForward removes a whole multibyte codepoint (e.g. 2-byte UTF-8 char between ASCII).
    st.setText("aжb");
    try std.testing.expectEqual(@as(usize, 4), st.len);
    st.home();
    _ = st.moveRight(); // cursor 1, right before multibyte char
    try std.testing.expect(st.deleteForward());
    try std.testing.expectEqualStrings("ab", st.text());

    // Truncation never splits a codepoint: 127 x 'a' + 2 x 2-byte char (131 bytes)
    // keeps exactly the 127 ASCII bytes.
    var big: [140]u8 = undefined;
    @memset(big[0..127], 'a');
    big[127] = 0xD0;
    big[128] = 0xB6; // U+0436 (cyrillic zhe) byte 2
    big[129] = 0xD0;
    big[130] = 0xB6; // U+0436 (cyrillic zhe) byte 2
    st.setText(big[0..131]);
    try std.testing.expectEqual(@as(usize, 127), st.len);
    try std.testing.expectEqual(@as(usize, 127), st.cursor);
    try std.testing.expectEqual(@as(u8, 'a'), st.text()[126]);

    // Full buffer rejects further input without modification.
    st.setText(big[0..128]);
    try std.testing.expectEqual(@as(usize, 128), st.len);
    try std.testing.expect(!st.insertChar('b'));
    try std.testing.expect(!st.insertChar('ж'));
    try std.testing.expectEqual(@as(usize, 128), st.len);
}
