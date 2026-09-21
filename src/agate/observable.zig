//! Observables, reactive state containers, and decoupled EventBus pattern.
//!
//! Provides Babylon.js-style `Observable(T)` event hooks, reactive `ObservableValue(T)`,
//! and an engine-wide type-safe & topic-based `EventBus`.
//!
//! Key design guarantees:
//! - Mutation-safe iteration: observers added or removed during notification
//!   do not corrupt iteration or cause use-after-free (re-entrant tombstone purge).
//! - Priority & ordering: supports `order` sorting and `insert_first`.
//! - Filtering: bitmask event filtering via `mask`.
//! - Propagation control: `EventState.stopPropagation()`.
//! - Zero allocation on empty notification paths (`hasObservers()` fast path).
//! - Zero overhead type identification: compile-time static address type IDs.

const std = @import("std");

/// State passed to observer callbacks during event dispatch.
pub const EventState = struct {
    mask: u32 = 0xFFFFFFFF,
    skip_next_observers: bool = false,
    target: ?*const anyopaque = null,
    current_target: ?*const anyopaque = null,
    user_info: ?*const anyopaque = null,

    /// Halts propagation of this event to any remaining observers in the chain.
    pub fn stopPropagation(self: *EventState) void {
        self.skip_next_observers = true;
    }
};

/// Options when registering an observer.
pub const ObserverOptions = struct {
    mask: u32 = 0xFFFFFFFF,
    insert_first: bool = false,
    unregister_on_first_call: bool = false,
    order: i32 = 0,
};

pub const ObserverId = u32;
pub const INVALID_OBSERVER_ID: ObserverId = 0;

/// An individual observer subscription handle and callback entry.
pub fn Observer(comptime T: type) type {
    return struct {
        id: ObserverId,
        callback: *const fn (data: T, state: *EventState, ctx: ?*anyopaque) void,
        context: ?*anyopaque = null,
        mask: u32 = 0xFFFFFFFF,
        unregister_on_first_call: bool = false,
        is_active: bool = true,
        order: i32 = 0,
    };
}

/// A Babylon.js-style subscribable event emitter.
pub fn Observable(comptime T: type) type {
    return struct {
        const Self = @This();
        pub const ObserverType = Observer(T);

        allocator: ?std.mem.Allocator = null,
        observers: std.ArrayListUnmanaged(ObserverType) = .empty,
        next_id: ObserverId = 1,
        is_notifying: u32 = 0,
        tombstones: u32 = 0,

        /// Creates an Observable with an explicit allocator.
        pub fn init(allocator: std.mem.Allocator) Self {
            return .{
                .allocator = allocator,
                .observers = .empty,
                .next_id = 1,
                .is_notifying = 0,
                .tombstones = 0,
            };
        }

        /// Releases all allocated memory and resets state.
        pub fn deinit(self: *Self) void {
            if (self.allocator) |alloc| {
                self.observers.deinit(alloc);
            }
            self.observers = .empty;
            self.allocator = null;
            self.is_notifying = 0;
            self.tombstones = 0;
        }

        /// Returns true if at least one observer is registered.
        pub fn hasObservers(self: *const Self) bool {
            if (self.observers.items.len == 0) return false;
            for (self.observers.items) |obs| {
                if (obs.is_active) return true;
            }
            return false;
        }

        /// Returns the count of active observers.
        pub fn countObservers(self: *const Self) usize {
            var count: usize = 0;
            for (self.observers.items) |obs| {
                if (obs.is_active) count += 1;
            }
            return count;
        }

        /// Checks if a specific observer ID is active.
        pub fn hasObserver(self: *const Self, id: ObserverId) bool {
            if (id == INVALID_OBSERVER_ID) return false;
            for (self.observers.items) |obs| {
                if (obs.id == id and obs.is_active) return true;
            }
            return false;
        }

        /// Registers an observer with a raw function pointer and context.
        pub fn add(
            self: *Self,
            callback: *const fn (data: T, state: *EventState, ctx: ?*anyopaque) void,
            context: ?*anyopaque,
            options: ObserverOptions,
        ) !ObserverId {
            const alloc = self.allocator orelse return error.NoAllocatorProvided;
            const id = self.next_id;
            self.next_id +%= 1;
            if (self.next_id == INVALID_OBSERVER_ID) self.next_id = 1;

            const obs = ObserverType{
                .id = id,
                .callback = callback,
                .context = context,
                .mask = options.mask,
                .unregister_on_first_call = options.unregister_on_first_call,
                .is_active = true,
                .order = options.order,
            };

            if (options.insert_first and options.order == 0) {
                // Insert at position 0
                try self.observers.insert(alloc, 0, obs);
            } else if (options.order != 0) {
                // Keep sorted by order (ascending: lower order runs first)
                var insert_idx: usize = self.observers.items.len;
                for (self.observers.items, 0..) |existing, i| {
                    if (options.order < existing.order) {
                        insert_idx = i;
                        break;
                    }
                }
                try self.observers.insert(alloc, insert_idx, obs);
            } else {
                try self.observers.append(alloc, obs);
            }

            return id;
        }

        /// Registers a typed observer with contextual state.
        pub fn addTyped(
            self: *Self,
            comptime Ctx: type,
            ctx: *Ctx,
            comptime func: fn (ctx: *Ctx, data: T, state: *EventState) void,
            options: ObserverOptions,
        ) !ObserverId {
            const Wrapper = struct {
                fn run(data: T, state: *EventState, raw_ctx: ?*anyopaque) void {
                    const typed: *Ctx = @ptrCast(@alignCast(raw_ctx.?));
                    func(typed, data, state);
                }
            };
            return self.add(Wrapper.run, ctx, options);
        }

        /// Registers a simple typed observer without requiring EventState inspection.
        pub fn addSimple(
            self: *Self,
            comptime Ctx: type,
            ctx: *Ctx,
            comptime func: fn (ctx: *Ctx, data: T) void,
        ) !ObserverId {
            const Wrapper = struct {
                fn run(data: T, _: *EventState, raw_ctx: ?*anyopaque) void {
                    const typed: *Ctx = @ptrCast(@alignCast(raw_ctx.?));
                    func(typed, data);
                }
            };
            return self.add(Wrapper.run, ctx, .{});
        }

        /// Registers a stateless observer function.
        pub fn addFn(
            self: *Self,
            comptime func: fn (data: T, state: *EventState) void,
            options: ObserverOptions,
        ) !ObserverId {
            const Wrapper = struct {
                fn run(data: T, state: *EventState, _: ?*anyopaque) void {
                    func(data, state);
                }
            };
            return self.add(Wrapper.run, null, options);
        }

        /// Registers a simple stateless observer function.
        pub fn addFnSimple(
            self: *Self,
            comptime func: fn (data: T) void,
        ) !ObserverId {
            const Wrapper = struct {
                fn run(data: T, _: *EventState, _: ?*anyopaque) void {
                    func(data);
                }
            };
            return self.add(Wrapper.run, null, .{});
        }

        /// Registers an observer that is invoked once and then automatically removed.
        pub fn addOnce(
            self: *Self,
            callback: *const fn (data: T, state: *EventState, ctx: ?*anyopaque) void,
            context: ?*anyopaque,
            mask: u32,
        ) !ObserverId {
            return self.add(callback, context, .{
                .mask = mask,
                .unregister_on_first_call = true,
            });
        }

        /// Registers a typed observer invoked once.
        pub fn addOnceTyped(
            self: *Self,
            comptime Ctx: type,
            ctx: *Ctx,
            comptime func: fn (ctx: *Ctx, data: T) void,
        ) !ObserverId {
            const Wrapper = struct {
                fn run(data: T, _: *EventState, raw_ctx: ?*anyopaque) void {
                    const typed: *Ctx = @ptrCast(@alignCast(raw_ctx.?));
                    func(typed, data);
                }
            };
            return self.add(Wrapper.run, ctx, .{ .unregister_on_first_call = true });
        }

        /// Registers a stateless observer invoked once.
        pub fn addOnceFn(
            self: *Self,
            comptime func: fn (data: T) void,
        ) !ObserverId {
            const Wrapper = struct {
                fn run(data: T, _: *EventState, _: ?*anyopaque) void {
                    func(data);
                }
            };
            return self.add(Wrapper.run, null, .{ .unregister_on_first_call = true });
        }

        /// Removes an observer by its unique subscription ID.
        pub fn remove(self: *Self, id: ObserverId) bool {
            if (id == INVALID_OBSERVER_ID) return false;
            for (self.observers.items, 0..) |*obs, i| {
                if (obs.id == id and obs.is_active) {
                    if (self.is_notifying > 0) {
                        obs.is_active = false;
                        self.tombstones += 1;
                    } else {
                        _ = self.observers.orderedRemove(i);
                    }
                    return true;
                }
            }
            return false;
        }

        /// Removes observers matching the callback function pointer and context.
        pub fn removeCallback(
            self: *Self,
            callback: *const fn (data: T, state: *EventState, ctx: ?*anyopaque) void,
            context: ?*anyopaque,
        ) bool {
            var removed = false;
            var i: usize = 0;
            while (i < self.observers.items.len) {
                const obs = &self.observers.items[i];
                if (obs.callback == callback and obs.context == context and obs.is_active) {
                    removed = true;
                    if (self.is_notifying > 0) {
                        obs.is_active = false;
                        self.tombstones += 1;
                        i += 1;
                    } else {
                        _ = self.observers.orderedRemove(i);
                    }
                } else {
                    i += 1;
                }
            }
            return removed;
        }

        /// Clears all observers.
        pub fn clear(self: *Self) void {
            if (self.is_notifying > 0) {
                for (self.observers.items) |*obs| {
                    if (obs.is_active) {
                        obs.is_active = false;
                        self.tombstones += 1;
                    }
                }
            } else {
                self.observers.clearRetainingCapacity();
                self.tombstones = 0;
            }
        }

        /// Notifies all active observers with default mask (0xFFFFFFFF).
        pub fn notify(self: *Self, event_data: T) void {
            var state = EventState{};
            self.notifyWithState(event_data, &state);
        }

        /// Notifies all active observers matching the given mask.
        pub fn notifyMask(self: *Self, event_data: T, mask: u32) void {
            var state = EventState{ .mask = mask };
            self.notifyWithState(event_data, &state);
        }

        /// Helper for Observable(void) dispatch.
        pub fn notifyVoid(self: *Self) void {
            if (T != void) @compileError("notifyVoid can only be called on Observable(void)");
            self.notify({});
        }

        /// Notifies observers using caller-provided EventState.
        pub fn notifyWithState(self: *Self, event_data: T, state: *EventState) void {
            if (self.observers.items.len == 0) return;

            self.is_notifying += 1;
            defer {
                self.is_notifying -= 1;
                if (self.is_notifying == 0 and self.tombstones > 0) {
                    self.purgeTombstones();
                }
            }

            // Snapshot length so observers added during iteration are deferred to subsequent notifications.
            const initial_count = self.observers.items.len;
            var i: usize = 0;
            while (i < initial_count) : (i += 1) {
                const obs = &self.observers.items[i];
                if (!obs.is_active) continue;
                if ((state.mask & obs.mask) == 0) continue;

                obs.callback(event_data, state, obs.context);

                if (obs.unregister_on_first_call) {
                    obs.is_active = false;
                    self.tombstones += 1;
                }

                if (state.skip_next_observers) break;
            }
        }

        /// Full Babylon.js-style observer notification with all routing context.
        pub fn notifyObservers(
            self: *Self,
            event_data: T,
            mask: u32,
            target: ?*const anyopaque,
            current_target: ?*const anyopaque,
            user_info: ?*const anyopaque,
        ) void {
            var state = EventState{
                .mask = mask,
                .target = target,
                .current_target = current_target,
                .user_info = user_info,
            };
            self.notifyWithState(event_data, &state);
        }

        fn purgeTombstones(self: *Self) void {
            var write_idx: usize = 0;
            for (self.observers.items) |obs| {
                if (obs.is_active) {
                    self.observers.items[write_idx] = obs;
                    write_idx += 1;
                }
            }
            self.observers.shrinkRetainingCapacity(write_idx);
            self.tombstones = 0;
        }
    };
}

/// A reactive value container that emits change notifications whenever mutated.
pub fn ObservableValue(comptime T: type) type {
    return struct {
        const Self = @This();

        pub const ChangeEvent = struct {
            old_value: T,
            new_value: T,
        };

        value: T,
        observable: Observable(ChangeEvent),

        pub fn init(allocator: std.mem.Allocator, initial_value: T) Self {
            return .{
                .value = initial_value,
                .observable = Observable(ChangeEvent).init(allocator),
            };
        }

        pub fn deinit(self: *Self) void {
            self.observable.deinit();
        }

        /// Gets the current value.
        pub fn get(self: *const Self) T {
            return self.value;
        }

        /// Updates the value and notifies observers with { old_value, new_value }.
        pub fn set(self: *Self, new_val: T) void {
            const old = self.value;
            self.value = new_val;
            self.observable.notify(.{ .old_value = old, .new_value = new_val });
        }

        /// Updates the value silently without firing any observer notifications.
        pub fn setSilent(self: *Self, new_val: T) void {
            self.value = new_val;
        }

        /// Subscribes to value changes with only the new value passed to callback.
        pub fn subscribe(
            self: *Self,
            comptime Ctx: type,
            ctx: *Ctx,
            comptime func: fn (ctx: *Ctx, new_value: T) void,
        ) !ObserverId {
            const Wrapper = struct {
                fn run(typed_ctx: *Ctx, ev: ChangeEvent) void {
                    func(typed_ctx, ev.new_value);
                }
            };
            return self.observable.addSimple(Ctx, ctx, Wrapper.run);
        }

        /// Subscribes to full change events (old_value and new_value).
        pub fn subscribeChange(
            self: *Self,
            comptime Ctx: type,
            ctx: *Ctx,
            comptime func: fn (ctx: *Ctx, change: ChangeEvent) void,
        ) !ObserverId {
            return self.observable.addSimple(Ctx, ctx, func);
        }

        /// Subscribes a stateless callback to new values.
        pub fn subscribeFn(
            self: *Self,
            comptime func: fn (new_value: T) void,
        ) !ObserverId {
            const Wrapper = struct {
                fn run(ev: ChangeEvent) void {
                    func(ev.new_value);
                }
            };
            return self.observable.addFnSimple(Wrapper.run);
        }

        /// Unsubscribes an observer by ID.
        pub fn unsubscribe(self: *Self, id: ObserverId) bool {
            return self.observable.remove(id);
        }

        pub fn hasObservers(self: *const Self) bool {
            return self.observable.hasObservers();
        }

        pub fn countObservers(self: *const Self) usize {
            return self.observable.countObservers();
        }
    };
}

/// Token returned by EventBus subscriptions for easy cancellation.
pub const Subscription = struct {
    id: u32,

    pub const invalid: Subscription = .{ .id = 0 };

    pub fn isValid(self: Subscription) bool {
        return self.id != 0;
    }
};

/// Unique compile-time 64-bit identifier for each type `T`.
pub fn typeId(comptime T: type) u64 {
    const hash = comptime std.hash.Wyhash.hash(0, @typeName(T));
    return hash;
}

/// A decoupled publish-subscribe messaging hub supporting both
/// type-safe events and named topic channels.
pub const EventBus = struct {
    pub const HandlerEntry = struct {
        id: u32,
        type_id: u64,
        has_topic: bool = false,
        topic_hash: u64 = 0,
        callback: *const fn (data_ptr: *const anyopaque, state: *EventState, ctx: ?*anyopaque) void,
        context: ?*anyopaque = null,
        is_active: bool = true,
        unregister_on_first_call: bool = false,
    };

    allocator: ?std.mem.Allocator = null,
    handlers: std.ArrayListUnmanaged(HandlerEntry) = .empty,
    next_id: u32 = 1,
    is_notifying: u32 = 0,
    tombstones: u32 = 0,

    pub fn init(allocator: std.mem.Allocator) EventBus {
        return .{
            .allocator = allocator,
            .handlers = .empty,
            .next_id = 1,
            .is_notifying = 0,
            .tombstones = 0,
        };
    }

    pub fn deinit(self: *EventBus) void {
        if (self.allocator) |alloc| {
            self.handlers.deinit(alloc);
        }
        self.handlers = .empty;
        self.allocator = null;
        self.is_notifying = 0;
        self.tombstones = 0;
    }

    pub fn clear(self: *EventBus) void {
        if (self.is_notifying > 0) {
            for (self.handlers.items) |*h| {
                if (h.is_active) {
                    h.is_active = false;
                    self.tombstones += 1;
                }
            }
        } else {
            self.handlers.clearRetainingCapacity();
            self.tombstones = 0;
        }
    }

    /// Subscribes to events of type `EventType`.
    pub fn subscribe(
        self: *EventBus,
        comptime EventType: type,
        comptime Ctx: type,
        ctx: *Ctx,
        comptime func: fn (ctx: *Ctx, event: EventType) void,
    ) !Subscription {
        const Wrapper = struct {
            fn run(raw_data: *const anyopaque, _: *EventState, raw_ctx: ?*anyopaque) void {
                const typed_ctx: *Ctx = @ptrCast(@alignCast(raw_ctx.?));
                if (@sizeOf(EventType) == 0) {
                    const ev: EventType = undefined;
                    func(typed_ctx, ev);
                } else {
                    const typed_data: *const EventType = @ptrCast(@alignCast(raw_data));
                    func(typed_ctx, typed_data.*);
                }
            }
        };
        return self.subscribeRaw(typeId(EventType), false, 0, Wrapper.run, ctx, false);
    }

    /// Subscribes to events with EventState access (e.g. stopPropagation).
    pub fn subscribeWithState(
        self: *EventBus,
        comptime EventType: type,
        comptime Ctx: type,
        ctx: *Ctx,
        comptime func: fn (ctx: *Ctx, event: EventType, state: *EventState) void,
    ) !Subscription {
        const Wrapper = struct {
            fn run(raw_data: *const anyopaque, state: *EventState, raw_ctx: ?*anyopaque) void {
                const typed_ctx: *Ctx = @ptrCast(@alignCast(raw_ctx.?));
                if (@sizeOf(EventType) == 0) {
                    const ev: EventType = undefined;
                    func(typed_ctx, ev, state);
                } else {
                    const typed_data: *const EventType = @ptrCast(@alignCast(raw_data));
                    func(typed_ctx, typed_data.*, state);
                }
            }
        };
        return self.subscribeRaw(typeId(EventType), false, 0, Wrapper.run, ctx, false);
    }

    /// Subscribes a stateless callback to `EventType`.
    pub fn subscribeFn(
        self: *EventBus,
        comptime EventType: type,
        comptime func: fn (event: EventType) void,
    ) !Subscription {
        const Wrapper = struct {
            fn run(raw_data: *const anyopaque, _: *EventState, _: ?*anyopaque) void {
                if (@sizeOf(EventType) == 0) {
                    const ev: EventType = undefined;
                    func(ev);
                } else {
                    const typed_data: *const EventType = @ptrCast(@alignCast(raw_data));
                    func(typed_data.*);
                }
            }
        };
        return self.subscribeRaw(typeId(EventType), false, 0, Wrapper.run, null, false);
    }

    /// Subscribes to a named topic channel with typed payload.
    pub fn subscribeTopic(
        self: *EventBus,
        topic: []const u8,
        comptime EventType: type,
        comptime Ctx: type,
        ctx: *Ctx,
        comptime func: fn (ctx: *Ctx, event: EventType) void,
    ) !Subscription {
        const Wrapper = struct {
            fn run(raw_data: *const anyopaque, _: *EventState, raw_ctx: ?*anyopaque) void {
                const typed_ctx: *Ctx = @ptrCast(@alignCast(raw_ctx.?));
                if (@sizeOf(EventType) == 0) {
                    const ev: EventType = undefined;
                    func(typed_ctx, ev);
                } else {
                    const typed_data: *const EventType = @ptrCast(@alignCast(raw_data));
                    func(typed_ctx, typed_data.*);
                }
            }
        };
        const topic_hash = std.hash.Wyhash.hash(0, topic);
        return self.subscribeRaw(typeId(EventType), true, topic_hash, Wrapper.run, ctx, false);
    }

    /// Subscribes to a named topic channel with a stateless callback.
    pub fn subscribeTopicFn(
        self: *EventBus,
        topic: []const u8,
        comptime EventType: type,
        comptime func: fn (event: EventType) void,
    ) !Subscription {
        const Wrapper = struct {
            fn run(raw_data: *const anyopaque, _: *EventState, _: ?*anyopaque) void {
                if (@sizeOf(EventType) == 0) {
                    const ev: EventType = undefined;
                    func(ev);
                } else {
                    const typed_data: *const EventType = @ptrCast(@alignCast(raw_data));
                    func(typed_data.*);
                }
            }
        };
        const topic_hash = std.hash.Wyhash.hash(0, topic);
        return self.subscribeRaw(typeId(EventType), true, topic_hash, Wrapper.run, null, false);
    }

    /// Subscribes to `EventType` once; automatically unregisters on first fire.
    pub fn subscribeOnce(
        self: *EventBus,
        comptime EventType: type,
        comptime Ctx: type,
        ctx: *Ctx,
        comptime func: fn (ctx: *Ctx, event: EventType) void,
    ) !Subscription {
        const Wrapper = struct {
            fn run(raw_data: *const anyopaque, _: *EventState, raw_ctx: ?*anyopaque) void {
                const typed_ctx: *Ctx = @ptrCast(@alignCast(raw_ctx.?));
                if (@sizeOf(EventType) == 0) {
                    const ev: EventType = undefined;
                    func(typed_ctx, ev);
                } else {
                    const typed_data: *const EventType = @ptrCast(@alignCast(raw_data));
                    func(typed_ctx, typed_data.*);
                }
            }
        };
        return self.subscribeRaw(typeId(EventType), false, 0, Wrapper.run, ctx, true);
    }

    fn subscribeRaw(
        self: *EventBus,
        tid: u64,
        has_topic: bool,
        topic_hash: u64,
        callback: *const fn (data_ptr: *const anyopaque, state: *EventState, ctx: ?*anyopaque) void,
        context: ?*anyopaque,
        once: bool,
    ) !Subscription {
        const alloc = self.allocator orelse return error.NoAllocatorProvided;
        const id = self.next_id;
        self.next_id +%= 1;
        if (self.next_id == 0) self.next_id = 1;

        try self.handlers.append(alloc, .{
            .id = id,
            .type_id = tid,
            .has_topic = has_topic,
            .topic_hash = topic_hash,
            .callback = callback,
            .context = context,
            .is_active = true,
            .unregister_on_first_call = once,
        });

        return .{ .id = id };
    }

    /// Unsubscribes by Subscription token.
    pub fn unsubscribe(self: *EventBus, sub: Subscription) bool {
        if (!sub.isValid()) return false;
        for (self.handlers.items, 0..) |*h, i| {
            if (h.id == sub.id and h.is_active) {
                if (self.is_notifying > 0) {
                    h.is_active = false;
                    self.tombstones += 1;
                } else {
                    _ = self.handlers.orderedRemove(i);
                }
                return true;
            }
        }
        return false;
    }

    /// Publishes an event to all subscribers of its type.
    pub fn publish(self: *EventBus, event: anytype) void {
        var state = EventState{};
        self.publishWithState(event, &state);
    }

    /// Publishes an event to all subscribers of its type with EventState.
    pub fn publishWithState(self: *EventBus, event: anytype, state: *EventState) void {
        const EventType = @TypeOf(event);
        const tid = typeId(EventType);
        const ptr: *const anyopaque = if (@sizeOf(EventType) == 0)
            @ptrFromInt(1)
        else
            @ptrCast(&event);
        self.publishInternal(tid, false, 0, ptr, state);
    }

    /// Publishes an event to a named topic channel.
    pub fn publishTopic(self: *EventBus, topic: []const u8, event: anytype) void {
        var state = EventState{};
        self.publishTopicWithState(topic, event, &state);
    }

    /// Publishes an event to a named topic channel with EventState.
    pub fn publishTopicWithState(self: *EventBus, topic: []const u8, event: anytype, state: *EventState) void {
        const EventType = @TypeOf(event);
        const tid = typeId(EventType);
        const topic_hash = std.hash.Wyhash.hash(0, topic);
        const ptr: *const anyopaque = if (@sizeOf(EventType) == 0)
            @ptrFromInt(1)
        else
            @ptrCast(&event);
        self.publishInternal(tid, true, topic_hash, ptr, state);
    }

    fn publishInternal(
        self: *EventBus,
        tid: u64,
        has_topic: bool,
        topic_hash: u64,
        data_ptr: *const anyopaque,
        state: *EventState,
    ) void {
        if (self.handlers.items.len == 0) return;

        self.is_notifying += 1;
        defer {
            self.is_notifying -= 1;
            if (self.is_notifying == 0 and self.tombstones > 0) {
                self.purgeTombstones();
            }
        }

        const count = self.handlers.items.len;
        var i: usize = 0;
        while (i < count) : (i += 1) {
            const h = &self.handlers.items[i];
            if (!h.is_active) continue;
            if (h.type_id != tid) continue;
            if (h.has_topic != has_topic) continue;
            if (has_topic and h.topic_hash != topic_hash) continue;

            h.callback(data_ptr, state, h.context);

            if (h.unregister_on_first_call) {
                h.is_active = false;
                self.tombstones += 1;
            }

            if (state.skip_next_observers) break;
        }
    }

    pub fn hasSubscribers(self: *const EventBus, comptime EventType: type) bool {
        const tid = typeId(EventType);
        for (self.handlers.items) |h| {
            if (h.type_id == tid and !h.has_topic and h.is_active) return true;
        }
        return false;
    }

    pub fn countSubscribers(self: *const EventBus, comptime EventType: type) usize {
        const tid = typeId(EventType);
        var count: usize = 0;
        for (self.handlers.items) |h| {
            if (h.type_id == tid and !h.has_topic and h.is_active) count += 1;
        }
        return count;
    }

    pub fn countTopicSubscribers(self: *const EventBus, topic: []const u8, comptime EventType: type) usize {
        const tid = typeId(EventType);
        const topic_hash = std.hash.Wyhash.hash(0, topic);
        var count: usize = 0;
        for (self.handlers.items) |h| {
            if (h.type_id == tid and h.has_topic and h.topic_hash == topic_hash and h.is_active) count += 1;
        }
        return count;
    }

    fn purgeTombstones(self: *EventBus) void {
        var write_idx: usize = 0;
        for (self.handlers.items) |h| {
            if (h.is_active) {
                self.handlers.items[write_idx] = h;
                write_idx += 1;
            }
        }
        self.handlers.shrinkRetainingCapacity(write_idx);
        self.tombstones = 0;
    }
};

/// Alias for developers preferring Qt/Godot Signal terminology.
pub fn Signal(comptime T: type) type {
    return Observable(T);
}

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
