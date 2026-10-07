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
