"""Segmented (reuse-aware) eviction defense for vLLM 0.25.1's v1 BlockPool.

Threat model this defends against: an unprivileged flooding attacker who
evicts a victim's cached prefix by exhausting shared KV cache capacity with
unique, single-use filler prompts (the "eviction channel" in this project).
Plain LRU cannot distinguish a victim's write-once-read-many prefix from an
attacker's read-once filler -- both are just blocks with a recency
timestamp. This patch adds the one signal that does distinguish them: how
many times a block's cached content has actually been reused.

Design: Segmented LRU (SLRU), two segments -- PROBATION (reuse_count == 0,
i.e. cached but never yet reused) and PROTECTED (reuse_count >= 1, reused at
least once). This is the same shape as MySQL InnoDB's buffer-pool young/old
sublist split and the classic SLRU cache-admission policy: new content always
enters probation; a block only earns a place in the protected segment (and
therefore later eviction priority) by actually being reused. On eviction,
probation is drained before protected is ever touched, so an attacker's
flood -- which by construction reuses nothing -- evicts itself before it can
evict a victim's genuinely-revisited prefix. Ties within a segment are
broken by recency (FIFO within probation; vLLM's native LRU order within
protected), matching "two segments ... breaking ties by LRU recency" from
the design brief.

Implementation choice: rather than reimplementing FreeKVCacheBlockQueue's
doubly-linked list (fragile, and this file must never edit anything in
site-packages/vllm/), vLLM's own free_block_queue remains the SOLE physical
store of free blocks, so get_num_free_blocks()/get_usage() (unpatched) stay
exactly correct with zero shadow bookkeeping to keep in sync. A lightweight
OrderedDict of block_ids (`_kvdefense_probation_ids`) is maintained
alongside it purely as a priority index: at eviction time, probation-listed
block_ids are pulled out of the middle of free_block_queue via its existing
O(1) `.remove()` primitive (already used by touch() for the same purpose),
falling back to the queue's native `.popleft()` only once probation is
empty. No new data structure duplicates what free_block_queue already does;
this only changes SELECTION ORDER within it.

Patched methods on vllm.v1.core.block_pool.BlockPool:
  __init__        -- adds _kvdefense_reuse_count / _kvdefense_probation_ids
  touch            -- +1 reuse_count on every cache hit; drops from probation
                       (a block that's been reused can never re-enter
                       probation until it's redispensed for new content)
  free_blocks      -- after the original runs, any block that just became
                       free (ref_cnt hit 0) AND is still cached (has a hash)
                       AND has reuse_count == 0 is added to probation
  get_new_blocks   -- drains probation (oldest-freed first) before falling
                       through to the protected segment (free_block_queue's
                       native LRU order); resets reuse_count to 0 for every
                       block redispensed here (fresh content, clean slate)
  evict_blocks     -- (external/connector-driven cache invalidation, not
                       exercised by flat single-node serving but patched
                       for completeness) discards stale reuse/probation
                       state for any block whose cache identity is removed

Gated behind KVDEFENSE_ENABLE=1 (see kvdefense_bootstrap.py). Unset the var
and the identical vllm/dynamo binary runs completely unpatched -- no file
under site-packages/vllm/ is ever modified.
"""

import collections
import os
import time

KVDEFENSE_LOG_PATH = os.environ.get(
    "KVDEFENSE_LOG", os.path.expanduser("~/LG2026/KVDEFENSE/kvdefense_patch.log")
)

_already_patched = False


def _log(msg: str) -> None:
    line = f"[KVDEFENSE {time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        os.makedirs(os.path.dirname(KVDEFENSE_LOG_PATH), exist_ok=True)
        with open(KVDEFENSE_LOG_PATH, "a") as f:
            f.write(line + "\n")
    except OSError:
        pass


def patch_block_pool() -> None:
    """Idempotent. Call at import time (before any BlockPool is constructed
    -- in practice this runs at process start via the .pth bootstrap, well
    before the engine builds its KV cache, so ordering is not a concern)."""
    global _already_patched
    if _already_patched:
        _log("patch_block_pool() called again, already patched, skipping")
        return

    from vllm.v1.core.block_pool import BlockPool

    _orig_init = BlockPool.__init__
    _orig_touch = BlockPool.touch
    _orig_free_blocks = BlockPool.free_blocks
    _orig_evict_blocks = BlockPool.evict_blocks
    _orig_reset_prefix_cache = BlockPool.reset_prefix_cache

    def patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        # block_id -> number of times this block's CURRENT cached content
        # has been reused (touch()ed) since it was last (re)dispensed by
        # get_new_blocks. Absence == 0.
        self._kvdefense_reuse_count: dict[int, int] = {}
        # Ordered set (dict used as such) of block_ids that are currently
        # free (ref_cnt == 0), still cached (have a hash), and have never
        # been reused -- i.e. the probation segment. Insertion order ==
        # free-order, so popitem(last=False) gives FIFO/LRU-within-segment.
        self._kvdefense_probation_ids: "collections.OrderedDict[int, None]" = (
            collections.OrderedDict()
        )
        self._kvdefense_stats = {"evictions_from_probation": 0, "evictions_from_protected": 0}
        _log(
            f"BlockPool initialized with segmented eviction active "
            f"(num_gpu_blocks={getattr(self, 'num_gpu_blocks', '?')})"
        )

    def patched_touch(self, blocks) -> None:
        for block in blocks:
            if block.is_null:
                continue
            bid = block.block_id
            was_free = block.ref_cnt == 0
            self._kvdefense_reuse_count[bid] = self._kvdefense_reuse_count.get(bid, 0) + 1
            if was_free:
                # It's about to be pulled out of the free queue by the
                # original touch() below; drop the matching probation
                # entry now so probation_ids never points at a block that
                # is no longer actually in free_block_queue. Once reused,
                # a block does not re-enter probation until it is next
                # redispensed for new content (see patched_get_new_blocks).
                self._kvdefense_probation_ids.pop(bid, None)
        _orig_touch(self, blocks)

    def patched_free_blocks(self, ordered_blocks) -> None:
        # Snapshot which blocks these are before the original call mutates
        # ref_cnt, so we can tell afterward which ones actually became free.
        candidates = list(ordered_blocks)
        _orig_free_blocks(self, candidates)
        for block in candidates:
            if block.is_null or block.ref_cnt != 0 or block.block_hash is None:
                continue
            if self._kvdefense_reuse_count.get(block.block_id, 0) == 0:
                # Newly free, still cached, never reused -- enters
                # probation. (Re-)insertion via pop+set keeps it at the
                # freshest (tail/MRU-of-probation) position if somehow
                # already present, though in practice this path only runs
                # once per free.
                self._kvdefense_probation_ids.pop(block.block_id, None)
                self._kvdefense_probation_ids[block.block_id] = None

    def patched_get_new_blocks(self, num_blocks: int):
        if num_blocks > self.get_num_free_blocks():
            raise ValueError(f"Cannot get {num_blocks} free blocks from the pool")

        ret = []

        # Tier 0 (unchanged from vanilla vLLM): blocks with no hash at all
        # carry zero cache value and must be evicted before anything with
        # content, hashed or not. free_blocks() always prepend_n()s these
        # to the very front of free_block_queue, so they form a contiguous
        # prefix at the head -- drain that prefix first. Once the head is
        # no longer hashless, there cannot be another hashless block
        # further back (prepend always inserts ahead of everything).
        q = self.free_block_queue
        while len(ret) < num_blocks:
            head = q.fake_free_list_head.next_free_block
            if head is None or head is q.fake_free_list_tail or head.block_hash is not None:
                break
            ret.append(q.popleft())

        # Tier 1: probation (reuse_count == 0), oldest-freed first, before
        # ever touching the protected (reused >= 1) segment.
        while len(ret) < num_blocks and self._kvdefense_probation_ids:
            bid, _ = self._kvdefense_probation_ids.popitem(last=False)
            block = self.blocks[bid]
            if block.ref_cnt != 0 or block.is_null:
                # Stale entry (shouldn't happen given the touch()-side
                # cleanup above, but never trust it blindly on a hot path
                # that must not crash real inference) -- skip, don't pull
                # from free_block_queue.
                continue
            self.free_block_queue.remove(block)
            ret.append(block)
            self._kvdefense_stats["evictions_from_probation"] += 1

        # Tier 2: protected (reused >= 1), native LRU order -- whatever is
        # left in free_block_queue at this point is guaranteed hash-bearing
        # (tier 0 drained above) and reuse_count >= 1 (tier 1 drained above,
        # and probation membership is exhaustive for reuse_count == 0 free
        # blocks by construction).
        if len(ret) < num_blocks:
            remaining = self.free_block_queue.popleft_n(num_blocks - len(ret))
            self._kvdefense_stats["evictions_from_protected"] += len(remaining)
            ret.extend(remaining)

        if self.enable_caching:
            for block in ret:
                self._maybe_evict_cached_block(block)
                assert block.ref_cnt == 0
                block.ref_cnt += 1
                # Fresh dispense -- whatever this block's history was, it
                # is about to hold entirely new content. Clean slate.
                self._kvdefense_reuse_count[block.block_id] = 0
                if self.metrics_collector:
                    self.metrics_collector.on_block_allocated(block)
        else:
            for block in ret:
                assert block.ref_cnt == 0
                block.ref_cnt += 1
                self._kvdefense_reuse_count[block.block_id] = 0
                if self.metrics_collector:
                    self.metrics_collector.on_block_allocated(block)
        return ret

    def patched_evict_blocks(self, block_ids) -> None:
        _orig_evict_blocks(self, block_ids)
        for bid in block_ids:
            self._kvdefense_reuse_count.pop(bid, None)
            self._kvdefense_probation_ids.pop(bid, None)

    def patched_reset_prefix_cache(self) -> bool:
        ok = _orig_reset_prefix_cache(self)
        if ok:
            # Mirrors the original's own wipe of cached_block_hash_to_block:
            # every block just lost its hash, so no block_id may remain
            # tagged as "cached but never reused" (tier 1) -- it is now
            # tier 0 (no hash at all) like everything else post-reset.
            self._kvdefense_reuse_count.clear()
            self._kvdefense_probation_ids.clear()
        return ok

    BlockPool.__init__ = patched_init
    BlockPool.touch = patched_touch
    BlockPool.free_blocks = patched_free_blocks
    BlockPool.get_new_blocks = patched_get_new_blocks
    BlockPool.evict_blocks = patched_evict_blocks
    BlockPool.reset_prefix_cache = patched_reset_prefix_cache

    _already_patched = True
    _log(
        "patched BlockPool.__init__/touch/free_blocks/get_new_blocks/evict_blocks "
        "-- segmented (probation/protected) eviction ACTIVE"
    )


if __name__ == "__main__":
    patch_block_pool()
