const std = @import("std");
const testing = std.testing;
const allocator = testing.allocator;
const assert = std.debug.assert;

const config = @import("../config.zig");
const fuzz = @import("../test/fuzz.zig");
const vsr = @import("../vsr.zig");
const log = std.log.scoped(.lsm_forest_fuzz);

const MessagePool = @import("../message_pool.zig").MessagePool;
const Transfer = @import("../tigerbeetle.zig").Transfer;
const Account = @import("../tigerbeetle.zig").Account;
const Storage = @import("../test/storage.zig").Storage;
const StateMachine = @import("../state_machine.zig").StateMachineType(Storage, .{
    .message_body_size_max = config.message_body_size_max,
});

const GridType = @import("grid.zig").GridType;
const NodePool = @import("node_pool.zig").NodePool(config.lsm_manifest_node_size, 16);
const Table = @import("table.zig").TableType(
    Key,
    Key.Value,
    Key.compare_keys,
    Key.key_from_value,
    Key.sentinel_key,
    Key.tombstone,
    Key.tombstone_from_key,
);
const Tree = @import("tree.zig").TreeType(Table, Storage, @typeName(Table) ++ "_test");

const Grid = GridType(Storage);
const SuperBlock = vsr.SuperBlockType(Storage);

const Key = packed struct {
    id: u64 align(@alignOf(u64)),

    const Value = packed struct {
        id: u64,
        value: u63,
        tombstone: u1 = 0,
    };

    inline fn compare_keys(a: Key, b: Key) std.math.Order {
        return std.math.order(a.id, b.id);
    }

    inline fn key_from_value(value: *const Key.Value) Key {
        return Key{ .id = value.id };
    }

    const sentinel_key = Key{
        .id = std.math.maxInt(u64),
    };

    inline fn tombstone(value: *const Key.Value) bool {
        return value.tombstone != 0;
    }

    inline fn tombstone_from_key(key: Key) Key.Value {
        return Key.Value{
            .id = key.id,
            .value = 0,
            .tombstone = 1,
        };
    }
};

const FuzzOp = union(enum) {
    // TODO Test checkpointing, range queries.
    compact,
    put: Key.Value,
    remove: Key.Value,
    get: Key,
};

const Environment = struct {
    const cluster = 32;
    const replica = 4;
    // TODO Is this appropriate for the number of fuzz_ops we want to run?
    const size_max = vsr.Zone.superblock.size().? + vsr.Zone.wal.size().? + 1024 * 1024 * 1024;

    const node_count = 1024;
    // This is the smallest size that set_associative_cache will allow us.
    const cache_entries_max = 2048;
    const batch_size_max = config.message_size_max - @sizeOf(vsr.Header);
    const commit_entries_max = @divFloor(batch_size_max, @sizeOf(Key.Value));
    const tree_options = Tree.Options{
        .commit_entries_max = commit_entries_max,
    };

    const State = enum {
        uninit,
        init,
        formatted,
        superblock_open,
        tree_open,
        tree_compacting,
        tree_lookup,
    };

    state: State,
    storage: *Storage,
    message_pool: MessagePool,
    superblock: SuperBlock,
    superblock_context: SuperBlock.Context = undefined,
    grid: Grid,
    node_pool: NodePool,
    cache: Tree.TableMutable.ValuesCache,
    tree: Tree,
    // We need @fieldParentPtr() of tree, so we can't use an optional Tree.
    tree_exists: bool,
    lookup_context: Tree.LookupContext = undefined,
    lookup_value: ?*const Key.Value = null,

    fn init(env: *Environment, storage: *Storage) !void {
        env.state = .uninit;

        env.storage = storage;
        errdefer env.storage.deinit(allocator);

        env.message_pool = try MessagePool.init(allocator, .replica);
        errdefer env.message_pool.deinit(allocator);

        env.superblock = try SuperBlock.init(allocator, env.storage, &env.message_pool);
        errdefer env.superblock.deinit(allocator);

        env.grid = try Grid.init(allocator, &env.superblock);
        errdefer env.grid.deinit(allocator);

        env.node_pool = try NodePool.init(allocator, node_count);
        errdefer env.node_pool.deinit(allocator);

        env.cache = try Tree.TableMutable.ValuesCache.init(allocator, cache_entries_max);
        errdefer env.cache.deinit(allocator);

        // Tree must be initialized with an open superblock.
        env.tree = undefined;
        env.tree_exists = false;

        env.state = .init;
    }

    fn deinit(env: *Environment) void {
        assert(env.state != .uninit);

        if (env.tree_exists) {
            env.tree.deinit(allocator);
            env.tree_exists = false;
        }
        env.cache.deinit(allocator);
        env.node_pool.deinit(allocator);
        env.grid.deinit(allocator);
        env.superblock.deinit(allocator);
        env.message_pool.deinit(allocator);

        env.state = .uninit;
    }

    fn tick(env: *Environment) void {
        env.grid.tick();
        env.storage.tick();
    }

    fn tick_until_state_change(env: *Environment, current_state: State, next_state: State) void {
        // Sometimes IO completes synchronously (eg if cached), so we might already be in next_state before ticking.
        assert(env.state == current_state or env.state == next_state);
        while (env.state == current_state) env.tick();
        assert(env.state == next_state);
    }

    fn change_state(env: *Environment, current_state: State, next_state: State) void {
        assert(env.state == current_state);
        env.state = next_state;
    }

    pub fn format(storage: *Storage) !void {
        var env: Environment = undefined;

        try env.init(storage);
        defer env.deinit();

        assert(env.state == .init);
        env.superblock.format(superblock_format_callback, &env.superblock_context, .{
            .cluster = cluster,
            .replica = replica,
            .size_max = size_max,
        });
        env.tick_until_state_change(.init, .formatted);
    }

    fn superblock_format_callback(superblock_context: *SuperBlock.Context) void {
        const env = @fieldParentPtr(@This(), "superblock_context", superblock_context);
        env.change_state(.init, .formatted);
    }

    pub fn open(env: *Environment) void {
        assert(env.state == .init);
        env.superblock.open(superblock_open_callback, &env.superblock_context);
        env.tick_until_state_change(.init, .tree_open);
    }

    fn superblock_open_callback(superblock_context: *SuperBlock.Context) void {
        const env = @fieldParentPtr(@This(), "superblock_context", superblock_context);
        env.change_state(.init, .superblock_open);
        env.tree = Tree.init(allocator, &env.node_pool, &env.grid, &env.cache, tree_options) catch unreachable;
        env.tree_exists = true;
        env.tree.open(tree_open_callback);
    }

    fn tree_open_callback(tree: *Tree) void {
        const env = @fieldParentPtr(@This(), "tree", tree);
        env.change_state(.superblock_open, .tree_open);
    }

    pub fn compact(env: *Environment, op: u64) void {
        env.change_state(.tree_open, .tree_compacting);
        env.tree.compact(tree_compact_callback, op);
        env.tick_until_state_change(.tree_compacting, .tree_open);
    }

    fn tree_compact_callback(tree: *Tree) void {
        const env = @fieldParentPtr(@This(), "tree", tree);
        env.change_state(.tree_compacting, .tree_open);
    }

    fn get(env: *Environment, key: Key) ?*const Key.Value {
        if (env.tree.lookup_from_memory(env.tree.lookup_snapshot_max, key)) |value| {
            return Tree.unwrap_tombstone(value);
        } else {
            env.change_state(.tree_open, .tree_lookup);
            env.lookup_context = undefined;
            env.lookup_value = null;
            env.tree.lookup_from_levels(get_callback, &env.lookup_context, env.tree.lookup_snapshot_max, key);
            env.tick_until_state_change(.tree_lookup, .tree_open);
            return env.lookup_value;
        }
    }

    fn get_callback(lookup_context: *Tree.LookupContext, value: ?*const Key.Value) void {
        const env = @fieldParentPtr(Environment, "lookup_context", lookup_context);
        assert(env.lookup_value == null);
        env.lookup_value = value;
        env.change_state(.tree_lookup, .tree_open);
    }

    fn run(storage: *Storage, fuzz_ops: []const FuzzOp) !void {
        var env: Environment = undefined;

        try env.init(storage);
        defer env.deinit();

        // Open the superblock then forest.
        env.open();

        // The tree should behave like a simple key-value data-structure.
        // We'll compare it to a hash map.
        var model = std.hash_map.AutoHashMap(Key, Key.Value).init(allocator);
        defer model.deinit();

        // We need an op number for compact.
        var op: u64 = 1;

        for (fuzz_ops) |fuzz_op, fuzz_op_index| {
            log.debug("Running fuzz_ops[{}] == {}", .{ fuzz_op_index, fuzz_op });
            // Apply fuzz_op to the forest and the model.
            switch (fuzz_op) {
                .compact => {
                    env.compact(op);
                    op += 1;
                },
                .put => |value| {
                    env.tree.put(&value);
                    try model.put(Key.key_from_value(&value), value);
                },
                .remove => |value| {
                    env.tree.remove(&value);
                    _ = model.remove(Key.key_from_value(&value));
                },
                .get => |key| {
                    // Get account from lsm.
                    const tree_value = env.get(key);

                    // Compare result to model.
                    const model_value = model.get(key);
                    if (model_value == null) {
                        assert(tree_value == null);
                    } else {
                        assert(std.mem.eql(
                            u8,
                            std.mem.asBytes(&model_value.?),
                            std.mem.asBytes(tree_value.?),
                        ));
                    }
                },
            }
        }
    }
};

pub fn run_fuzz_ops(fuzz_ops: []const FuzzOp) !void {
    // Init mocked storage.
    var storage = try Storage.init(
        allocator,
        Environment.size_max,
        Storage.Options{
            // We don't apply storage faults yet, so this seed doesn't matter.
            .seed = 0xdeadbeef,
            .read_latency_min = 0,
            .read_latency_mean = 0,
            .write_latency_min = 0,
            .write_latency_mean = 0,
            .read_fault_probability = 0,
            .write_fault_probability = 0,
        },
        0,
        .{
            .first_offset = 0,
            .period = 0,
        },
    );
    defer storage.deinit(allocator);

    try Environment.format(&storage);
    try Environment.run(&storage, fuzz_ops);
}

fn random_id(random: std.rand.Random, comptime Int: type) Int {
    // We have two opposing desires for random ids:
    const avg_int: Int = if (random.boolean())
        // 1. We want to cause many collisions.
        8
    else
        // 2. We want to generate enough ids that the cache can't hold them all.
        Environment.cache_entries_max;
    return fuzz.random_int_exponential(random, Int, avg_int);
}

pub fn generate_fuzz_ops(random: std.rand.Random) ![]const FuzzOp {
    const fuzz_op_count = @minimum(
        @as(usize, 1E7),
        fuzz.random_int_exponential(random, usize, 1E6),
    );
    const fuzz_ops = try allocator.alloc(FuzzOp, fuzz_op_count);
    errdefer allocator.free(fuzz_ops);

    const fuzz_op_distribution = fuzz.random_enum_distribution(random, std.meta.Tag(FuzzOp));
    log.info("fuzz_op_distribution = {d:.2}", .{fuzz_op_distribution});

    // We're not allowed to go more than Environment.cache_entries_max puts without compacting.
    var puts_since_compact: usize = 0;

    for (fuzz_ops) |*fuzz_op| {
        fuzz_op.* = if (puts_since_compact >= Environment.cache_entries_max)
            // We have to compact before doing any other operations.
            FuzzOp{ .compact = {} }
        else
        // Otherwise pick a random FuzzOp.
        switch (fuzz.random_enum(random, std.meta.Tag(FuzzOp), fuzz_op_distribution)) {
            .compact => FuzzOp{
                .compact = {},
            },
            .put => FuzzOp{ .put = .{
                .id = random_id(random, u64),
                .value = random.int(u63),
            } },
            .remove => FuzzOp{ .remove = .{
                .id = random_id(random, u64),
                .value = random.int(u63),
            } },
            .get => FuzzOp{ .get = .{
                .id = random_id(random, u64),
            } },
        };
        switch (fuzz_op.*) {
            .compact => puts_since_compact = 0,
            // Tree.remove() works by inserting a tombstone, so it counts as a put.
            .put, .remove => puts_since_compact += 1,
            .get => {},
        }
    }

    return fuzz_ops;
}

pub fn main() !void {
    const fuzz_args = try fuzz.parse_fuzz_args(allocator);
    var rng = std.rand.DefaultPrng.init(fuzz_args.seed);

    const fuzz_ops = try generate_fuzz_ops(rng.random());
    defer allocator.free(fuzz_ops);

    try run_fuzz_ops(fuzz_ops);
}
