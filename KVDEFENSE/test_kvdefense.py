import os
assert os.environ.get("KVDEFENSE_ENABLE") == "1", "run with KVDEFENSE_ENABLE=1"

from vllm.v1.core.block_pool import BlockPool

pool = BlockPool(num_gpu_blocks=6, enable_caching=True, hash_block_size=16)
print("null_block id:", pool.null_block.block_id)
assert pool._kvdefense_reuse_count is not None, "patch did not apply (no _kvdefense_reuse_count attr)"

# 1. Dispense all 5 real blocks (fresh/hashless -> tier 0 path).
blocks = pool.get_new_blocks(5)
ids = [b.block_id for b in blocks]
print("dispensed (fresh):", ids)
assert set(ids) == {1, 2, 3, 4, 5}

# 2. Give each a fake hash (simulating cache_full_blocks having cached them).
for i, b in enumerate(blocks):
    b.set_block_hash(f"hash{i}", num_tokens=16)

# 3. Free all 5 -- all become probation (reuse_count == 0).
pool.free_blocks(blocks)
print("probation_ids after freeing all 5:", list(pool._kvdefense_probation_ids.keys()))
assert set(pool._kvdefense_probation_ids.keys()) == set(ids)

# 4. Touch block id=3 (a cache hit) -- reuse_count -> 1, pulled out of the
#    free queue and out of probation, ref_cnt -> 1.
target = next(b for b in blocks if b.block_id == 3)
pool.touch([target])
print("reuse_count[3] after touch:", pool._kvdefense_reuse_count[3])
print("3 still in probation?", 3 in pool._kvdefense_probation_ids)
assert pool._kvdefense_reuse_count[3] == 1
assert 3 not in pool._kvdefense_probation_ids
assert target.ref_cnt == 1

# 5. Free it again -- now reuse_count=1, so it must NOT re-enter probation
#    (stays purely in free_block_queue's native tail position = protected).
pool.free_blocks([target])
print("3 in probation after re-free?", 3 in pool._kvdefense_probation_ids)
assert 3 not in pool._kvdefense_probation_ids

# 6. Drain everything and check eviction ORDER: the 4 never-reused blocks
#    must come out before block 3 (the once-reused one), even though block
#    3 was freed in between (i.e. reuse count beats raw recency).
drained = pool.get_new_blocks(5)
drained_ids = [b.block_id for b in drained]
print("eviction order:", drained_ids)
assert drained_ids[-1] == 3, f"expected block 3 (reused) evicted LAST, got order {drained_ids}"
assert set(drained_ids[:4]) == {1, 2, 4, 5}, "expected the 4 never-reused blocks evicted first"

print("\nALL ASSERTIONS PASSED")
