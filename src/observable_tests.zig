const std = @import("std");
const obs_mod = @import("observable.zig");

const EventState = obs_mod.EventState;
const ObserverOptions = obs_mod.ObserverOptions;
const ObserverId = obs_mod.ObserverId;
const INVALID_OBSERVER_ID = obs_mod.INVALID_OBSERVER_ID;
const Observer = obs_mod.Observer;
const Observable = obs_mod.Observable;
const ObservableValue = obs_mod.ObservableValue;
const Subscription = obs_mod.Subscription;
const typeId = obs_mod.typeId;
const EventBus = obs_mod.EventBus;
const Signal = obs_mod.Signal;

// -------------------------------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------------------------------

test "Observable: basic subscribe and notify" {
    const alloc = std.testing.allocator;
    var obs = Observable(i32).init(alloc);
    defer obs.deinit();

    try std.testing.expect(!obs.hasObservers());
    try std.testing.expectEqual(@as(usize, 0), obs.countObservers());

    var sum: i32 = 0;
    const Handler = struct {
        fn onVal(ctx: *i32, val: i32) void {
            ctx.* += val;
        }
    };

    const id = try obs.addSimple(i32, &sum, Handler.onVal);
    try std.testing.expect(obs.hasObservers());
    try std.testing.expect(obs.hasObserver(id));
    try std.testing.expectEqual(@as(usize, 1), obs.countObservers());

    obs.notify(10);
    try std.testing.expectEqual(@as(i32, 10), sum);

    obs.notify(25);
    try std.testing.expectEqual(@as(i32, 35), sum);

    try std.testing.expect(obs.remove(id));
    try std.testing.expect(!obs.hasObservers());

    obs.notify(100);
    try std.testing.expectEqual(@as(i32, 35), sum);
}

test "Observable: mask filtering" {
    const alloc = std.testing.allocator;
    var obs = Observable(u32).init(alloc);
    defer obs.deinit();

    var keyboard_count: u32 = 0;
    var mouse_count: u32 = 0;

    const MASK_KEY: u32 = 0b01;
    const MASK_MOUSE: u32 = 0b10;

    const Handler = struct {
        fn onKey(ctx: *u32, _: u32) void {
            ctx.* += 1;
        }
        fn onMouse(ctx: *u32, _: u32) void {
            ctx.* += 1;
        }
    };

    _ = try obs.addTyped(u32, &keyboard_count, struct {
        fn run(ctx: *u32, _: u32, _: *EventState) void {
            Handler.onKey(ctx, 0);
        }
    }.run, .{ .mask = MASK_KEY });

    _ = try obs.addTyped(u32, &mouse_count, struct {
        fn run(ctx: *u32, _: u32, _: *EventState) void {
            Handler.onMouse(ctx, 0);
        }
    }.run, .{ .mask = MASK_MOUSE });

    obs.notifyMask(1, MASK_KEY);
    try std.testing.expectEqual(@as(u32, 1), keyboard_count);
    try std.testing.expectEqual(@as(u32, 0), mouse_count);

    obs.notifyMask(2, MASK_MOUSE);
    try std.testing.expectEqual(@as(u32, 1), keyboard_count);
    try std.testing.expectEqual(@as(u32, 1), mouse_count);

    obs.notifyMask(3, MASK_KEY | MASK_MOUSE);
    try std.testing.expectEqual(@as(u32, 2), keyboard_count);
    try std.testing.expectEqual(@as(u32, 2), mouse_count);
}

test "Observable: priority order and insert_first" {
    const alloc = std.testing.allocator;
    var obs = Observable(i32).init(alloc);
    defer obs.deinit();

    var order_log: [4]i32 = undefined;
    var log_len: usize = 0;

    const RecCtx = struct { buf: *[4]i32, len: *usize };
    const Recorder = struct {
        fn record(ctx: *RecCtx, val: i32) void {
            ctx.buf[ctx.len.*] = val;
            ctx.len.* += 1;
        }
    };

    var rec_ctx = RecCtx{ .buf = &order_log, .len = &log_len };

    // Register with order: 10, -5, 0, and insert_first
    _ = try obs.addSimple(RecCtx, &rec_ctx, struct {
        fn run(ctx: *RecCtx, _: i32) void {
            Recorder.record(ctx, 10);
        }
    }.run); // order 0 by default

    _ = try obs.addTyped(RecCtx, &rec_ctx, struct {
        fn run(ctx: *RecCtx, _: i32, _: *EventState) void {
            Recorder.record(ctx, 100);
        }
    }.run, .{ .order = 100 });

    _ = try obs.addTyped(RecCtx, &rec_ctx, struct {
        fn run(ctx: *RecCtx, _: i32, _: *EventState) void {
            Recorder.record(ctx, -50);
        }
    }.run, .{ .order = -50 });

    _ = try obs.addTyped(RecCtx, &rec_ctx, struct {
        fn run(ctx: *RecCtx, _: i32, _: *EventState) void {
            Recorder.record(ctx, -999);
        }
    }.run, .{ .insert_first = true });

    obs.notify(1);

    try std.testing.expectEqual(@as(usize, 4), log_len);
    try std.testing.expectEqual(@as(i32, -999), order_log[0]);
    try std.testing.expectEqual(@as(i32, -50), order_log[1]);
    try std.testing.expectEqual(@as(i32, 10), order_log[2]);
    try std.testing.expectEqual(@as(i32, 100), order_log[3]);
}

test "Observable: addOnce auto-unregisters" {
    const alloc = std.testing.allocator;
    var obs = Observable(void).init(alloc);
    defer obs.deinit();

    var fired: u32 = 0;
    const Handler = struct {
        fn onFire(ctx: *u32, _: void) void {
            ctx.* += 1;
        }
    };

    _ = try obs.addOnceTyped(u32, &fired, Handler.onFire);
    try std.testing.expectEqual(@as(usize, 1), obs.countObservers());

    obs.notifyVoid();
    try std.testing.expectEqual(@as(u32, 1), fired);
    try std.testing.expectEqual(@as(usize, 0), obs.countObservers());

    obs.notifyVoid();
    try std.testing.expectEqual(@as(u32, 1), fired);
}

test "Observable: stopPropagation stops further observers" {
    const alloc = std.testing.allocator;
    var obs = Observable(i32).init(alloc);
    defer obs.deinit();

    var first_ran = false;
    var second_ran = false;

    _ = try obs.addTyped(bool, &first_ran, struct {
        fn run(ctx: *bool, _: i32, state: *EventState) void {
            ctx.* = true;
            state.stopPropagation();
        }
    }.run, .{ .order = 1 });

    _ = try obs.addTyped(bool, &second_ran, struct {
        fn run(ctx: *bool, _: i32, _: *EventState) void {
            ctx.* = true;
        }
    }.run, .{ .order = 2 });

    obs.notify(42);
    try std.testing.expect(first_ran);
    try std.testing.expect(!second_ran);
}

test "Observable: mutation safety - removal from callback" {
    const alloc = std.testing.allocator;
    var obs = Observable(i32).init(alloc);
    defer obs.deinit();

    var victim_ran = false;
    var killer_id: ObserverId = 0;
    var victim_id: ObserverId = 0;

    const KillerContext = struct {
        obs: *Observable(i32),
        target_id: *ObserverId,
    };

    var kctx = KillerContext{
        .obs = &obs,
        .target_id = &victim_id,
    };

    killer_id = try obs.addTyped(KillerContext, &kctx, struct {
        fn run(ctx: *KillerContext, _: i32, _: *EventState) void {
            _ = ctx.obs.remove(ctx.target_id.*);
        }
    }.run, .{ .order = 1 });

    victim_id = try obs.addTyped(bool, &victim_ran, struct {
        fn run(ctx: *bool, _: i32, _: *EventState) void {
            ctx.* = true;
        }
    }.run, .{ .order = 2 });

    try std.testing.expectEqual(@as(usize, 2), obs.countObservers());

    obs.notify(1);

    // Victim was removed before its turn in the loop!
    try std.testing.expect(!victim_ran);
    // After notify completes, tombstones are purged
    try std.testing.expectEqual(@as(usize, 1), obs.countObservers());
    try std.testing.expect(obs.hasObserver(killer_id));
    try std.testing.expect(!obs.hasObserver(victim_id));
}

test "ObservableValue: reactive get, set, setSilent, subscribe" {
    const alloc = std.testing.allocator;
    var val = ObservableValue(f32).init(alloc, 100.0);
    defer val.deinit();

    try std.testing.expectEqual(@as(f32, 100.0), val.get());

    var observed: f32 = 0.0;
    var old_observed: f32 = 0.0;

    const ObserverCtx = struct {
        cur: *f32,
        old: *f32,
    };
    var octx = ObserverCtx{ .cur = &observed, .old = &old_observed };

    const sub_id = try val.subscribeChange(ObserverCtx, &octx, struct {
        fn run(ctx: *ObserverCtx, ev: ObservableValue(f32).ChangeEvent) void {
            ctx.old.* = ev.old_value;
            ctx.cur.* = ev.new_value;
        }
    }.run);

    val.set(150.0);
    try std.testing.expectEqual(@as(f32, 150.0), val.get());
    try std.testing.expectEqual(@as(f32, 100.0), old_observed);
    try std.testing.expectEqual(@as(f32, 150.0), observed);

    val.setSilent(200.0);
    try std.testing.expectEqual(@as(f32, 200.0), val.get());
    // Observers were NOT notified
    try std.testing.expectEqual(@as(f32, 150.0), observed);

    try std.testing.expect(val.unsubscribe(sub_id));
    val.set(300.0);
    try std.testing.expectEqual(@as(f32, 150.0), observed);
}

test "EventBus: type-safe pub-sub and topic isolation" {
    const alloc = std.testing.allocator;
    var bus = EventBus.init(alloc);
    defer bus.deinit();

    const PlayerDamageEvent = struct {
        player_id: u32,
        damage: f32,
    };
    const ScoreEvent = struct {
        points: i32,
    };

    var total_damage: f32 = 0.0;
    var total_score: i32 = 0;
    var topic_damage: f32 = 0.0;

    const sub1 = try bus.subscribe(PlayerDamageEvent, f32, &total_damage, struct {
        fn onDmg(ctx: *f32, ev: PlayerDamageEvent) void {
            ctx.* += ev.damage;
        }
    }.onDmg);

    const sub2 = try bus.subscribe(ScoreEvent, i32, &total_score, struct {
        fn onScore(ctx: *i32, ev: ScoreEvent) void {
            ctx.* += ev.points;
        }
    }.onScore);

    const sub_topic = try bus.subscribeTopic("pvp_arena", PlayerDamageEvent, f32, &topic_damage, struct {
        fn onDmg(ctx: *f32, ev: PlayerDamageEvent) void {
            ctx.* += ev.damage;
        }
    }.onDmg);

    try std.testing.expect(bus.hasSubscribers(PlayerDamageEvent));
    try std.testing.expect(bus.hasSubscribers(ScoreEvent));
    try std.testing.expectEqual(@as(usize, 1), bus.countSubscribers(PlayerDamageEvent));
    try std.testing.expectEqual(@as(usize, 1), bus.countTopicSubscribers("pvp_arena", PlayerDamageEvent));

    // Normal typed publish
    bus.publish(PlayerDamageEvent{ .player_id = 1, .damage = 25.5 });
    try std.testing.expectEqual(@as(f32, 25.5), total_damage);
    try std.testing.expectEqual(@as(f32, 0.0), topic_damage);
    try std.testing.expectEqual(@as(i32, 0), total_score);

    // Topic publish
    bus.publishTopic("pvp_arena", PlayerDamageEvent{ .player_id = 2, .damage = 50.0 });
    try std.testing.expectEqual(@as(f32, 25.5), total_damage);
    try std.testing.expectEqual(@as(f32, 50.0), topic_damage);

    // Score publish
    bus.publish(ScoreEvent{ .points = 100 });
    try std.testing.expectEqual(@as(i32, 100), total_score);

    // Unsubscribe
    try std.testing.expect(bus.unsubscribe(sub1));
    bus.publish(PlayerDamageEvent{ .player_id = 1, .damage = 10.0 });
    try std.testing.expectEqual(@as(f32, 25.5), total_damage);

    _ = bus.unsubscribe(sub2);
    _ = bus.unsubscribe(sub_topic);
}

test "EventBus: subscribeOnce and re-entrancy" {
    const alloc = std.testing.allocator;
    var bus = EventBus.init(alloc);
    defer bus.deinit();

    const PingEvent = struct { count: u32 };
    const PongEvent = struct { count: u32 };

    var ping_calls: u32 = 0;
    var pong_calls: u32 = 0;

    const ReentrantCtx = struct {
        bus: *EventBus,
        pings: *u32,
        pongs: *u32,
    };

    var rctx = ReentrantCtx{
        .bus = &bus,
        .pings = &ping_calls,
        .pongs = &pong_calls,
    };

    _ = try bus.subscribe(PingEvent, ReentrantCtx, &rctx, struct {
        fn onPing(ctx: *ReentrantCtx, ev: PingEvent) void {
            ctx.pings.* += 1;
            if (ev.count > 0) {
                ctx.bus.publish(PongEvent{ .count = ev.count });
            }
        }
    }.onPing);

    _ = try bus.subscribeOnce(PongEvent, ReentrantCtx, &rctx, struct {
        fn onPong(ctx: *ReentrantCtx, ev: PongEvent) void {
            ctx.pongs.* += ev.count;
        }
    }.onPong);

    bus.publish(PingEvent{ .count = 10 });

    try std.testing.expectEqual(@as(u32, 1), ping_calls);
    try std.testing.expectEqual(@as(u32, 10), pong_calls);

    // Pong was subscribeOnce, second ping shouldn't trigger pong
    bus.publish(PingEvent{ .count = 5 });
    try std.testing.expectEqual(@as(u32, 2), ping_calls);
    try std.testing.expectEqual(@as(u32, 10), pong_calls);
}

test "Observable: insert_first during notify is deferred, not re-delivered" {
    const alloc = std.testing.allocator;
    var obs = Observable(i32).init(alloc);
    defer obs.deinit();

    const Ctx = struct {
        const Self = @This();
        obs: *Observable(i32),
        log: *[4]i32,
        len: *usize,

        fn first(_: i32, _: *EventState, raw: ?*anyopaque) void {
            const c: *Self = @ptrCast(@alignCast(raw.?));
            c.log[c.len.*] = 1;
            c.len.* += 1;
            if (c.len.* == 1) {
                _ = c.obs.add(second, c, .{ .insert_first = true }) catch {};
            }
        }

        fn second(_: i32, _: *EventState, raw: ?*anyopaque) void {
            const c: *Self = @ptrCast(@alignCast(raw.?));
            c.log[c.len.*] = 2;
            c.len.* += 1;
        }
    };

    var log: [4]i32 = undefined;
    var len: usize = 0;
    var ctx = Ctx{ .obs = &obs, .log = &log, .len = &len };

    _ = try obs.add(Ctx.first, &ctx, .{});
    obs.notify(1);

    // The inserted observer must not be delivered in the same pass, and the
    // first observer must run exactly once (no shifted duplicate).
    try std.testing.expectEqual(@as(usize, 1), len);
    try std.testing.expectEqual(@as(i32, 1), log[0]);
    try std.testing.expectEqual(@as(usize, 2), obs.countObservers());

    // Deferred insert already applied; insert_first puts it ahead next time.
    len = 0;
    obs.notify(2);
    try std.testing.expectEqual(@as(usize, 2), len);
    try std.testing.expectEqual(@as(i32, 2), log[0]);
    try std.testing.expectEqual(@as(i32, 1), log[1]);
}

test "once observer is consumed before a recursive notify" {
    const Context = struct {
        source: *Observable(u32),
        calls: usize = 0,
        fn callback(self: *@This(), value: u32) void {
            self.calls += 1;
            if (self.calls == 1) self.source.notify(value + 1);
        }
    };
    var source = Observable(u32).init(std.testing.allocator);
    defer source.deinit();
    var context = Context{ .source = &source };
    _ = try source.addOnceTyped(Context, &context, Context.callback);
    source.notify(1);
    try std.testing.expectEqual(@as(usize, 1), context.calls);
    source.notify(2);
    try std.testing.expectEqual(@as(usize, 1), context.calls);
}

test "once handler is consumed before a recursive publish" {
    const Event = struct { value: u32 };
    const Context = struct {
        bus: *obs_mod.EventBus,
        calls: usize = 0,
        fn callback(self: *@This(), event: Event) void {
            self.calls += 1;
            if (self.calls == 1) self.bus.publish(Event{ .value = event.value + 1 });
        }
    };
    var bus = obs_mod.EventBus.init(std.testing.allocator);
    defer bus.deinit();
    var context = Context{ .bus = &bus };
    _ = try bus.subscribeOnce(Event, Context, &context, Context.callback);
    bus.publish(Event{ .value = 1 });
    try std.testing.expectEqual(@as(usize, 1), context.calls);
    bus.publish(Event{ .value = 2 });
    try std.testing.expectEqual(@as(usize, 1), context.calls);
}

test "removeCallback removes an ordered subscription added in a callback" {
    const Context = struct {
        source: *Observable(u32),
        added: bool = false,
        removed: bool = false,
        unexpected_calls: usize = 0,
        fn target(_: u32, _: *EventState, raw: ?*anyopaque) void {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            self.unexpected_calls += 1;
        }
        fn callback(self: *@This(), _: u32) void {
            if (self.added) return;
            self.added = true;
            _ = self.source.add(target, self, .{ .insert_first = true }) catch @panic("unexpected OOM");
            self.removed = self.source.removeCallback(target, self);
        }
    };
    var source = Observable(u32).init(std.testing.allocator);
    defer source.deinit();
    var context = Context{ .source = &source };
    _ = try source.addSimple(Context, &context, Context.callback);
    source.notify(1);
    source.notify(2);
    try std.testing.expect(context.removed);
    try std.testing.expectEqual(@as(usize, 0), context.unexpected_calls);
}

test "once callback may add and remove subscriptions" {
    const Context = struct {
        source: *Observable(u32),
        added_calls: usize = 0,
        removed_calls: usize = 0,
        fn added(_: u32, _: *EventState, raw: ?*anyopaque) void {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            self.added_calls += 1;
        }
        fn removedCb(_: u32, _: *EventState, raw: ?*anyopaque) void {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            self.removed_calls += 1;
        }
        fn once(self: *@This(), _: u32) void {
            _ = self.source.add(added, self, .{}) catch @panic("unexpected OOM");
            _ = self.source.add(removedCb, self, .{}) catch @panic("unexpected OOM");
            if (!self.source.removeCallback(removedCb, self)) @panic("remove failed");
        }
    };
    var source = Observable(u32).init(std.testing.allocator);
    defer source.deinit();
    var context = Context{ .source = &source };
    _ = try source.addOnceTyped(Context, &context, Context.once);
    source.notify(1);
    try std.testing.expectEqual(@as(usize, 0), context.added_calls);
    try std.testing.expectEqual(@as(usize, 0), context.removed_calls);
    source.notify(2);
    try std.testing.expectEqual(@as(usize, 1), context.added_calls);
    try std.testing.expectEqual(@as(usize, 0), context.removed_calls);
    source.notify(3);
    try std.testing.expectEqual(@as(usize, 2), context.added_calls);
    try std.testing.expectEqual(@as(usize, 0), context.removed_calls);
}
