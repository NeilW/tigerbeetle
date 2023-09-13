const std = @import("std");
const constants = @import("../constants.zig");

const stdx = @import("../stdx.zig");
const assert = std.debug.assert;
const maybe = stdx.maybe;

const SetAssociativeCacheType = @import("set_associative_cache.zig").SetAssociativeCacheType;
const ScopeCloseMode = @import("tree.zig").ScopeCloseMode;

/// A CacheMap is a hybrid between our SetAssociativeCache and a HashMap. The SetAssociativeCache
/// sits on top and absorbs the majority of get / put requests. Below that, lives a HashMap.
/// Should an insert() cause an eviction (which can happen either because the Key is the same,
/// or because our Way is full), the evicted value is caught and put in the HashMap.
///
/// Within our LSM, it's the backing for the combined Groove prefetch + cache. The cache part
/// fills the use case of an object cache, while the HashMap ensures that values put in for
/// prefetch will exist as long as required.
///
/// Cache invalidation for the HashMap is handled by `compact`.
pub fn CacheMapType(
    comptime Key: type,
    comptime Value: type,
    comptime key_from_value: fn (*const Value) callconv(.Inline) Key,
    comptime hash_from_key: fn (Key) callconv(.Inline) u64,
    comptime equal: fn (Key, Key) callconv(.Inline) bool,
    comptime tombstone_from_key: fn (Key) callconv(.Inline) Value,
    comptime tombstone: fn (*const Value) callconv(.Inline) bool,
) type {
    const _Cache = SetAssociativeCacheType(
        Key,
        Value,
        key_from_value,
        hash_from_key,
        equal,
        .{},
    );

    const HashMapContextValue = struct {
        const Self = @This();

        pub inline fn eql(_: Self, a: Value, b: Value) bool {
            return equal(key_from_value(&a), key_from_value(&b));
        }

        pub inline fn hash(_: Self, value: Value) u64 {
            return stdx.hash_inline(key_from_value(&value));
        }
    };

    const map_load_percentage_max = 50;
    const _Map = std.HashMapUnmanaged(
        Value,
        void,
        HashMapContextValue,
        map_load_percentage_max,
    );

    return struct {
        const Self = @This();

        pub const Cache = _Cache;
        pub const Map = _Map;

        pub const Options = struct {
            cache_value_count_max: u32,
            map_value_count_max: u32,
            scope_value_count_max: u32,
            name: []const u8,
        };

        // The hierarchy for lookups is cache -> map_1 -> map_2. Lower levels _may_ have stale
        // values, provided the correct value exists in one of the levels above. We have two
        // maps to implement our compact() support. Evictions from the cache first flow into
        // map_1, with .compact() clearing map_2 and swapping it.
        cache: Cache,
        map_1: Map,
        map_2: Map,

        // Scopes allow you to perform operations on the CacheMap before either persisting or
        // discarding them. There are a few cases that need to be considered, given the interaction
        // of our cache, and our stash:
        // 1. After an upsert, we have evicted an item that was an exact match. This means we're
        //    doing an update of an item that's in the cache. Store the original item without
        //    clobbering in our scope_map.
        // 2. After an upsert, we have evicted an item that was not an exact match. This means we're
        //    doing an insert of a new value, but two keys have the same tags. Store the evicted
        //    item without clobbering in our scope_map.
        // 3. After an upsert, we haven't evicted anything, check our stash:
        //    a. If a matching item exists there, it means we're doing an update of an item that's
        //       in the stash. Store the original item without clobbering in our scope_map.
        //    b. If no matching item exists there, it means it's an insert. Store a tombstone
        //       without clobbering in our scope_map.
        scope_is_active: bool = false,
        scope_map: Map,

        last_upsert_was_update_with_eviction: bool = undefined,
        options: Options,

        pub fn init(allocator: std.mem.Allocator, options: Options) !Self {
            assert(options.cache_value_count_max > 0);
            assert(options.map_value_count_max > 0);
            maybe(options.scope_value_count_max == 0);

            var cache: Cache = try Cache.init(
                allocator,
                options.cache_value_count_max,
                .{ .name = options.name },
            );
            errdefer cache.deinit(allocator);

            var map_1: Map = .{};
            try map_1.ensureTotalCapacity(allocator, options.map_value_count_max);
            errdefer map_1.deinit(allocator);

            var map_2: Map = .{};
            try map_2.ensureTotalCapacity(allocator, options.map_value_count_max);
            errdefer map_2.deinit(allocator);

            var scope_map: Map = .{};
            try scope_map.ensureTotalCapacity(allocator, options.scope_value_count_max);
            errdefer scope_map.deinit(allocator);

            return Self{
                .cache = cache,
                .map_1 = map_1,
                .map_2 = map_2,
                .scope_map = scope_map,
                .options = options,
            };
        }

        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            self.cache.deinit(allocator);
            self.map_1.deinit(allocator);
            self.map_2.deinit(allocator);
            self.scope_map.deinit(allocator);
        }

        pub fn reset(self: *Self) void {
            self.cache.reset();
            self.map_1.clearRetainingCapacity();
            self.map_2.clearRetainingCapacity();
            self.scope_map.clearRetainingCapacity();
            self.scope_is_active = false;
            self.last_upsert_was_update_with_eviction = undefined;
        }

        pub fn has(self: *Self, key: Key) bool {
            return self.cache.get_index(key) != null or
                self.map_1.getKeyPtr(tombstone_from_key(key)) != null or
                self.map_2.getKeyPtr(tombstone_from_key(key)) != null;
        }

        pub fn get(self: *Self, key: Key) ?*Value {
            return self.cache.get(key) orelse
                self.map_1.getKeyPtr(tombstone_from_key(key)) orelse
                self.map_2.getKeyPtr(tombstone_from_key(key));
        }

        pub fn upsert(self: *Self, value: *const Value) void {
            self.last_upsert_was_update_with_eviction = false;
            _ = self.cache.upsert_index(value, upsert_on_eviction);

            if (self.scope_is_active and !self.last_upsert_was_update_with_eviction) {
                if (self.map_1.getKey(value.*)) |stash_value| {
                    // Scope Map: Case 2a.
                    self.scope_map.putAssumeCapacity(stash_value, {});
                } else {
                    // Scope Map: Case 2b.
                    self.scope_map.putAssumeCapacity(
                        tombstone_from_key(key_from_value(value)),
                        {},
                    );
                }
            }
        }

        fn upsert_on_eviction(cache: *Cache, value: *const Value, updated: bool) void {
            var self = @fieldParentPtr(Self, "cache", cache);
            if (updated) {
                // Scope Map: Case 1.
                self.last_upsert_was_update_with_eviction = true;
                if (self.scope_is_active) {
                    self.scope_map.putAssumeCapacity(value.*, {});
                }
            } else {
                if (self.scope_is_active) {
                    // Scope Map: Case 3.
                    self.scope_map.putAssumeCapacity(value.*, {});

                    const gop = self.map_1.getOrPutAssumeCapacity(value.*);
                    gop.key_ptr.* = value.*;
                } else {
                    const gop = self.map_1.getOrPutAssumeCapacity(value.*);
                    gop.key_ptr.* = value.*;
                }
            }
        }

        pub fn remove(self: *Self, key: Key) void {
            // The only thing that tests this in any depth is the cache_map fuzz itself.
            // Make sure we aren't being called in regular code without another once over.
            assert(constants.verify);

            const maybe_removed = self.cache.remove(key);

            if (maybe_removed) |removed| {
                if (self.scope_is_active) {
                    self.scope_map.putAssumeCapacity(removed, {});
                }
            } else {
                if (self.scope_is_active) {
                    // TODO: Actually, does the fuzz catch this...
                    // TODO: So if we delete from map_2 and put to map_1, there's a problem;
                    //       because when we undo our scope we insert back to map_1 :/
                    const maybe_map_removed = self.map_1.getKey(tombstone_from_key(key)) orelse
                        self.map_2.getKey(tombstone_from_key(key));
                    if (maybe_map_removed) |map_removed| {
                        self.scope_map.putAssumeCapacity(map_removed, {});
                    }
                }
            }

            // We always need to try remove from the stash; since it could have a stale value.
            _ = self.map_1.remove(tombstone_from_key(key));
            _ = self.map_2.remove(tombstone_from_key(key));
        }

        /// Start a new scope. Within a scope, changes can be persisted
        /// or discarded. At most one scope can be active at a time.
        pub fn scope_open(self: *Self) void {
            assert(!self.scope_is_active);
            assert(self.scope_map.count() == 0);
            self.scope_is_active = true;
        }

        pub fn scope_close(self: *Self, mode: ScopeCloseMode) void {
            assert(self.scope_is_active);
            self.scope_is_active = false;

            // We don't need to do anything to persist a scope.
            if (mode == .persist) {
                self.scope_map.clearRetainingCapacity();
                return;
            }

            // The scope_map stores the operations we need to reverse the changes a scope made.
            // Replay them back.
            var scope_values = self.scope_map.keyIterator();
            while (scope_values.next()) |scope_value| {
                if (tombstone(scope_value)) {
                    // Reverting an insert consists of a .remove call. The value in here will be a
                    // tombstone indicating the original value didn't exist. We don't touch map_2;
                    // since we can never insert into it directly (only a .compact() can).
                    const key = key_from_value(scope_value);
                    _ = self.cache.remove(key);
                    _ = self.map_1.remove(tombstone_from_key(key));
                } else {
                    // Reverting an update or delete consists of an insert of the original value.
                    self.upsert(scope_value);
                }
            }
            self.scope_map.clearRetainingCapacity();
        }

        pub fn compact(self: *Self) void {
            assert(!self.scope_is_active);
            assert(self.scope_map.count() == 0);

            self.map_2.clearRetainingCapacity();
            std.mem.swap(Map, &self.map_1, &self.map_2);
        }
    };
}

pub const TestTable = struct {
    pub const Key = u32;
    pub const Value = struct {
        key: Key,
        value: u32,
        tombstone: bool = false,
        padding: [7]u8 = undefined,
    };

    pub inline fn key_from_value(v: *const Value) u32 {
        return v.key;
    }

    pub inline fn compare_keys(a: Key, b: Key) std.math.Order {
        return std.math.order(a, b);
    }

    pub inline fn tombstone_from_key(a: Key) Value {
        return Value{ .key = a, .value = 0, .tombstone = true };
    }

    pub inline fn tombstone(a: *const TestTable.Value) bool {
        return a.tombstone;
    }

    pub inline fn hash(key: TestTable.Key) u64 {
        return stdx.hash_inline(key);
    }

    pub inline fn equal(a: TestTable.Key, b: TestTable.Key) bool {
        return a == b;
    }
};

pub const TestCacheMap = CacheMapType(
    TestTable.Key,
    TestTable.Value,
    TestTable.key_from_value,
    TestTable.hash,
    TestTable.equal,
    TestTable.tombstone_from_key,
    TestTable.tombstone,
);

test "cache_map: unit" {
    const testing = std.testing;

    const allocator = testing.allocator;

    var cache_map = try TestCacheMap.init(allocator, .{
        .cache_value_count_max = 2048,
        .scope_value_count_max = 32,
        .map_value_count_max = 32,
        .name = "test map",
    });
    defer cache_map.deinit(allocator);

    cache_map.upsert(&.{ .key = 1, .value = 1, .tombstone = false });
    try testing.expectEqual(.{ .key = 1, .value = 1, .tombstone = false }, cache_map.get(1).?.*);

    // Test scope persisting
    cache_map.scope_open();
    cache_map.upsert(&.{ .key = 2, .value = 2, .tombstone = false });
    try testing.expectEqual(.{ .key = 2, .value = 2, .tombstone = false }, cache_map.get(2).?.*);
    cache_map.scope_close(.persist);
    try testing.expectEqual(.{ .key = 2, .value = 2, .tombstone = false }, cache_map.get(2).?.*);

    // Test scope discard on updates
    cache_map.scope_open();
    cache_map.upsert(&.{ .key = 2, .value = 22, .tombstone = false });
    cache_map.upsert(&.{ .key = 2, .value = 222, .tombstone = false });
    cache_map.upsert(&.{ .key = 2, .value = 2222, .tombstone = false });
    try testing.expectEqual(
        .{ .key = 2, .value = 2222, .tombstone = false },
        cache_map.get(2).?.*,
    );
    cache_map.scope_close(.discard);
    try testing.expectEqual(.{ .key = 2, .value = 2, .tombstone = false }, cache_map.get(2).?.*);

    // Test scope discard on inserts
    cache_map.scope_open();
    cache_map.upsert(&.{ .key = 3, .value = 3, .tombstone = false });
    try testing.expectEqual(.{ .key = 3, .value = 3, .tombstone = false }, cache_map.get(3).?.*);
    cache_map.upsert(&.{ .key = 3, .value = 33, .tombstone = false });
    try testing.expectEqual(.{ .key = 3, .value = 33, .tombstone = false }, cache_map.get(3).?.*);
    cache_map.scope_close(.discard);
    assert(!cache_map.has(3));
    assert(cache_map.get(3) == null);

    // Test scope discard on removes
    cache_map.scope_open();
    cache_map.remove(2);
    assert(!cache_map.has(2));
    assert(cache_map.get(2) == null);
    cache_map.scope_close(.discard);
    try testing.expectEqual(.{ .key = 2, .value = 2, .tombstone = false }, cache_map.get(2).?.*);
}
