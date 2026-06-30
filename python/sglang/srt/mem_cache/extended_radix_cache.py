from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Set

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    EvictParams,
    EvictResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.kv_connector import BaseKVConnector, LoadOperation
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey, TreeNode
from sglang.srt.mem_cache.base_prefix_cache import InitLoadBackParams, InsertParams, MatchPrefixParams
if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams

logger = logging.getLogger(__name__)


class ExtendedRadixCache(BasePrefixCache):
    """RadixCache decorator with external KV storage connector.

    Wraps any BasePrefixCache implementation (RadixCache, SWARadixCache, etc.)
    and adds external KV storage capabilities via a BaseKVConnector.
    """

    def __init__(
        self,
        params: CacheInitParams,
        connector: Optional[BaseKVConnector] = None,
        inner_cache: Optional[BasePrefixCache] = None,
    ):
        # Use provided inner cache, or create a default RadixCache
        if inner_cache is not None:
            self._inner_radixtree = inner_cache
        else:
            self._inner_radixtree = RadixCache(params)
        self._connector = connector

        self._load_task_id_counter = 0
        self._load_queue: List[LoadOperation] = []
        self._ongoing_load_tasks: Dict[int, List[TreeNode]] = {}
        self._ongoing_store_tasks: Dict[int, TreeNode] = {}

    # -- Forward PrefixCacheTrait properties to inner cache --

    @property
    def req_to_token_pool(self):
        return self._inner_radixtree.req_to_token_pool

    @req_to_token_pool.setter
    def req_to_token_pool(self, value):
        self._inner_radixtree.req_to_token_pool = value

    @property
    def token_to_kv_pool_allocator(self):
        return self._inner_radixtree.token_to_kv_pool_allocator

    @token_to_kv_pool_allocator.setter
    def token_to_kv_pool_allocator(self, value):
        self._inner_radixtree.token_to_kv_pool_allocator = value

    @property
    def page_size(self):
        return self._inner_radixtree.page_size

    @page_size.setter
    def page_size(self, value):
        self._inner_radixtree.page_size = value

    @property
    def disable(self):
        return self._inner_radixtree.disable

    @property
    def device(self):
        return self._inner_radixtree.device

    @property
    def metrics_collector(self):
        return self._inner_radixtree.metrics_collector

    @metrics_collector.setter
    def metrics_collector(self, value):
        self._inner_radixtree.metrics_collector = value

    @property
    def layer_done_counter(self):
        if self._connector is None:
            return None
        return self._connector.layer_done_counter

    # -- Core methods with connector logic --

    def reset(self):
        self._ongoing_store_tasks.clear()
        self._ongoing_load_tasks.clear()
        self._load_queue.clear()

        if self._connector is not None:
            self._connector.reset()

        self._inner_radixtree.reset()

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        device_match_result = self._inner_radixtree.match_prefix(params)
        if self._connector is None:
            return device_match_result

        key = params.key
        device_indices: torch.Tensor = device_match_result.device_indices
        last_device_node = device_match_result.last_device_node

        uncached_len = len(key) - device_indices.numel()
        if uncached_len <= 0:
            if params.req is not None:
                params.req.cached_tokens_extended_device = 0
            return device_match_result

        # ``key.token_ids`` may be longer than ``len(key)`` in EAGLE bigram
        # mode (N raw tokens make N-1 bigrams).  The connector consumes
        # token_ids and a parallel mask, so both must have the same length —
        # use ``len(key)`` (the logical bigram count) as the source of truth
        # and slice token_ids to match.
        n = len(key)
        token_ids_for_connector = key.token_ids[:n] if hasattr(key, 'token_ids') else key.token_ids
        token_mask = torch.zeros(n, dtype=torch.bool)
        token_mask[device_indices.numel() :] = True

        new_hit_length = self._connector.get_new_hit_length(
            token_ids=token_ids_for_connector,
            token_mask=token_mask,
            update_state_for_load=params.update_connector_state,
            rid=params.req.rid if params.req is not None else None,
            device_indices=device_indices,
        )

        if params.req is not None:
            params.req.cached_tokens_extended_device = new_hit_length

        return MatchResult(
            device_indices=device_indices,
            last_device_node=last_device_node,
            last_host_node=last_device_node,
            best_match_node=last_device_node,
            host_hit_length=new_hit_length,
        )

    def init_load_back(
        self,
        params: InitLoadBackParams
    ):
        """Reserve GPU slots, build a new tree node, and queue the H2D op.

        Returns (new_indices, last_node) tuple — the schedule_policy caller
        unpacks this directly:
            new_indices, req.last_node = tree_cache.init_load_back(...)
        On any early-exit path we return (empty_tensor, req.last_node) to
        keep that contract intact.
        """
        req = params.req
        mem_quota = params.mem_quota

        # Empty tensor sized to req.prefix_indices' device, so torch.cat works.
        empty_indices = torch.empty(
            (0,), dtype=torch.int64, device=self._inner_radixtree.device
        )

        if self._connector is None:
            return empty_indices, req.last_node

        host_hit_length = req.host_hit_length

        if host_hit_length <= 0 or (
            mem_quota is not None and host_hit_length > mem_quota
        ):
            self._connector.release_load_state(req.rid)
            return empty_indices, req.last_node

        # Allocate GPU slots for host_hit_length tokens. Three allocator regimes:
        #   * page_size == 1: plain RadixCache / SWARadixCache w/ token-level alloc.
        #     `.alloc(N)` works directly.
        #   * page_size  > 1, no SWA: DSv4 paged allocator without window — use
        #     `alloc_extend()` for the whole hit length.
        #   * page_size  > 1, hybrid SWA (DSv4): full layers get the whole hit,
        #     but SWA layers only get the trailing window. We MUST call
        #     `alloc_extend_swa_tail(extend_num_tokens=H, swa_tail_len=window)`
        #     because the SWA pool is sized to ~window-per-request and
        #     never has room for H tokens of SWA data — and FlexKV only stored
        #     `window` tokens of SWA data on the host side anyway.
        allocator = self._inner_radixtree.token_to_kv_pool_allocator
        page_size = getattr(allocator, "page_size", 1) or 1
        # Pull window size from the connector if it exposes one (FlexKV does).
        window_size = getattr(self._connector, "_swa_window_size", 0) or 0
        # Detect hybrid-SWA paged allocator by the alloc_extend_swa_tail method.
        has_swa_tail = hasattr(allocator, "alloc_extend_swa_tail")

        def _swa_tail_len() -> int:
            # SWA slots to allocate/map for the trailing window. Use CEIL to a
            # page so the mapped tail node is >= sliding_window_size tokens.
            # _insert_helper splits the unmapped head into a `swa_tombstone`
            # node; if the mapped tail were < window, inc_lock_ref would not
            # finish locking the window inside the tail and would walk up into
            # the tombstone head, hitting `assert not swa_tombstone`. Cap at
            # host_hit_length (which is already page-aligned) and re-floor to a
            # page as a defensive guard.
            tail = min(
                ((window_size + page_size - 1) // page_size) * page_size,
                host_hit_length,
            )
            tail = (tail // page_size) * page_size
            return tail if tail > 0 else page_size

        def _alloc_paged():
            # prefix_lens is the device-cached length so far. host_hit_length is
            # appended after that.
            gpu_prefix_len = int(req.prefix_indices.numel())
            seq_len = gpu_prefix_len + host_hit_length
            device = self._inner_radixtree.device
            prefix_lens = torch.tensor([gpu_prefix_len], dtype=torch.int64, device=device)
            prefix_lens_cpu = torch.tensor([gpu_prefix_len], dtype=torch.int64, device="cpu")
            seq_lens = torch.tensor([seq_len], dtype=torch.int64, device=device)
            seq_lens_cpu = torch.tensor([seq_len], dtype=torch.int64, device="cpu")
            # last_loc: previous token's slot index, or -1 sentinel when empty.
            if gpu_prefix_len > 0:
                last_loc = req.prefix_indices[-1:].to(device=device, dtype=torch.int64)
            else:
                last_loc = torch.tensor([-1], dtype=torch.int64, device=device)
            if has_swa_tail and window_size > 0:
                # Hybrid SWA: only allocate the trailing window of SWA slots
                # regardless of how many full slots we need (see _swa_tail_len
                # for the ceil-to-page rationale vs. inc_lock_ref).
                swa_tail_len = _swa_tail_len()
                return allocator.alloc_extend_swa_tail(
                    prefix_lens,
                    prefix_lens_cpu,
                    seq_lens,
                    seq_lens_cpu,
                    last_loc,
                    host_hit_length,
                    swa_tail_len,
                )
            return allocator.alloc_extend(
                prefix_lens,
                prefix_lens_cpu,
                seq_lens,
                seq_lens_cpu,
                last_loc,
                host_hit_length,
            )

        def _do_alloc():
            if page_size == 1:
                return allocator.alloc(host_hit_length)
            # Page-aligned paged allocator path
            if host_hit_length % page_size != 0:
                logger.warning(
                    "[FlexKV] host_hit_length=%d not page-aligned (page_size=%d), "
                    "skipping H2D for rid=%s",
                    host_hit_length, page_size, req.rid,
                )
                return None
            return _alloc_paged()

        swa_alloc_need = host_hit_length
        if has_swa_tail and window_size > 0 and page_size > 1:
            swa_alloc_need = _swa_tail_len()

        device_indices = _do_alloc()
        if device_indices is None:
            from sglang.srt.mem_cache.common import evict_from_tree_cache

            evict_from_tree_cache(
                self,
                host_hit_length,
                swa_num_tokens=swa_alloc_need,
            )
            device_indices = _do_alloc()
        if device_indices is None:
            from sglang.srt.mem_cache.common import available_and_evictable_str

            logger.warning(
                "Failed to allocate %d GPU slots for external load (swa_need=%d). %s",
                host_hit_length,
                swa_alloc_need,
                available_and_evictable_str(self),
            )
            self._connector.release_load_state(req.rid)
            return empty_indices, req.last_node

        gpu_cached_len = len(req.prefix_indices)
        # Sanity-check the effective length that ``_inner_radixtree.insert``
        # will keep in the tree.  insert() applies two transforms:
        #   1. EAGLE bigram view: ``len(key) -> len(key) - 1`` when is_eagle.
        #   2. page_aligned trim: ``len(key) -> (len(key) // page_size) * page_size``.
        # If ``effective_len`` is 0, the insert is a no-op and re-match would
        # land on root; bail cleanly to avoid wasting a transfer.
        #
        # We do NOT pre-trim ``device_indices`` to ``effective_len``: the
        # connector already prepared the FlexKV transfer graph with
        # ``host_hit_length`` source blocks, and shrinking the destination
        # mapping would crash with ``src_block_ids.size != dst_block_ids.size``
        # in the worker. Instead we mirror normal sglang prefill: pass the
        # FULL ``device_indices`` to the load op, let ``insert`` do its
        # internal trim, and rely on the trailing slots being soft-locked
        # by ``req.prefix_indices`` (the same way they are after a normal
        # paged prefill — those slots are "allocator-allocated, not
        # tree-tracked" and stay alive for the lifetime of the request).
        is_eagle = bool(getattr(self._inner_radixtree, 'is_eagle', False))
        effective_len = host_hit_length - (1 if is_eagle else 0)
        if page_size > 1:
            # TEMP FIX (999-temp-fix-init-load-back.patch): the original
            # implementation used floor-page-align, which under EAGLE bigram
            # always trims a host_hit_length that's already a page multiple
            # to 0:
            #     hit_length=256, page_size=256, eagle=True
            #     → effective_len = 256 - 1 = 255
            #     → floor(255/256)*256 = 0
            #     → "trims to 0" warning, rolls back alloc, H2D never fires.
            #
            # The store path (cache_finished_req above) page-aligns
            # ``len(token_ids)`` BEFORE applying EAGLE's bigram view, so
            # for hit_length=256 it stores exactly 256 tokens. For load to
            # match store, we want effective_len to round to the same
            # 256-token boundary.
            #
            # Fix: under EAGLE, ceil-page-align AFTER the bigram trim,
            # capped at host_hit_length so we never exceed what FlexKV
            # actually has on the host side.
            if is_eagle and effective_len > 0:
                effective_len = min(
                    ((effective_len + page_size - 1) // page_size) * page_size,
                    host_hit_length,
                )
            else:
                effective_len = (effective_len // page_size) * page_size
        if effective_len <= 0:
            logger.warning(
                "[FlexKV] init_load_back: host_hit_length=%d trims to 0 after "
                "is_eagle=%s + page_align(%d) for rid=%s; rolling back alloc.",
                host_hit_length, is_eagle, page_size, req.rid,
            )
            try:
                allocator.free(device_indices)
            except Exception:
                pass
            # release_load_state may itself raise (FlexKV-side bugs); the GPU
            # slots are already freed above, so demote the failure to a warning
            # rather than crashing the whole rank on this optimisation-only path.
            try:
                self._connector.release_load_state(req.rid)
            except Exception as _e:
                logger.warning(
                    "[FlexKV] init_load_back: release_load_state raised after "
                    "effective_len<=0 rollback for rid=%s: %s. Continuing.",
                    req.rid, _e,
                )
            return empty_indices, req.last_node

        # DECOUPLED H2D RESTORE (replaces the previous insert + tail_all_evict
        # guard approach).
        #
        # Root cause being fixed: page_size(256) == sliding_window_size(256) +
        # EAGLE bigram means insert()'s internal `bigram(-1) + floor page-align`
        # always drops exactly the last page — which is the ONLY page that
        # alloc_extend_swa_tail mapped to the SWA pool. The inserted value is
        # therefore entirely below the SWA eviction frontier, hits
        # `tail_all_evicted`, gets freed, and produces NO leaf — while the caller
        # still keeps those same slots in req.prefix_indices and the H2D op. That
        # double ownership is what the old guard aborted H2D to avoid (which
        # killed the cache hit rate, since every multi-page EAGLE hit tripped it).
        #
        # Instead of inserting the restored prefix into the radix tree HERE, we
        # treat it exactly like normal sglang prefill treats freshly-computed
        # tokens: leave the slots req-owned ("uncached") and let the already
        # stable cache_unfinished_req / cache_finished_req path insert them
        # canonically — the restored full KV becomes swa_tombstone nodes (full KV
        # kept, SWA window already slid out by _evict_swa) and the current
        # sliding window becomes a homogeneous non-tombstone leaf. We only:
        #   * fire the H2D op so the hit is actually restored,
        #   * inc_lock the existing matched prefix node for the async H2D
        #     lifetime (paired with dec in _check_load_completion),
        #   * persist swa_evicted_seqlen so the restored unmapped head is never
        #     handed to free_swa and is later inserted as a tombstone, and
        #   * keep cache_protected_len at the pre-restore prefix length so
        #     cache_*_req uses the correct prev_prefix_len and dedup-frees any
        #     cross-request duplicate slots.
        #
        # swa_evicted_seqlen marks the boundary between the restored unmapped
        # head [gpu_cached_len, gpu_cached_len + effective_len - swa_tail_len) and
        # the mapped tail. It is ABSOLUTE from sequence start and in SLOT units
        # (matches schedule_batch._evict_swa and the cache_finished_req path).
        swa_evicted_seqlen = 0
        if has_swa_tail and window_size > 0:
            swa_evicted_seqlen = gpu_cached_len + max(
                0, effective_len - _swa_tail_len()
            )

        # Persist the SWA eviction frontier onto the req so the restored unmapped
        # head is handled correctly later:
        #   * _evict_swa frees req_to_token[swa_evicted_seqlen:new] via free_swa;
        #     starting the frontier at the mapped-tail boundary keeps the
        #     restored head (full->swa mapping == 0) out of that range. (free_swa
        #     itself filters mapping>0, but this also avoids redundant churn.)
        #   * cache_unfinished_req / cache_finished_req pass req.swa_evicted_seqlen
        #     to insert(), so the restored head is split into a swa_tombstone node
        #     (full KV kept, no SWA) instead of a heterogeneous non-tombstone
        #     node — preserving the `_swa_slots_in_value == len(value)` invariant.
        # Use max() so we never walk the frontier backwards past an _evict_swa
        # update.
        if swa_evicted_seqlen > 0:
            req.swa_evicted_seqlen = max(req.swa_evicted_seqlen, swa_evicted_seqlen)

        # DECOUPLED: do NOT insert the restored prefix into the radix tree here.
        # Lock the existing matched prefix node (req.last_node) for the async H2D
        # lifetime. _check_load_completion() unconditionally dec_lock_ref's the
        # node we stash in the LoadOperation, so we MUST take a paired inc here.
        # This mirrors the previous root_fallback path, generalised to every hit:
        # the restored slots stay req-owned/uncached and are inserted into the
        # tree later by cache_unfinished_req / cache_finished_req (the stable
        # normal-prefill path), so there is no in-tree node to create here and no
        # tail_all_evict free to double-own device_indices.
        new_node = req.last_node
        inc_result = self._flexkv_inc_lock(
            new_node,
            "init_load_back:decoupled",
            rid=req.rid,
            host_hit_length=host_hit_length,
        )
        load_swa_uuid = getattr(inc_result, 'swa_uuid_for_lock', None)

        # Keep cache_protected_len at the pre-restore prefix length. The restored
        # device_indices are appended to req.prefix_indices below (the forward
        # pass needs them and must not recompute them), but they are NOT in the
        # radix tree, so they are req-owned "uncached" slots. schedule_policy
        # would otherwise set cache_protected_len = len(prefix_indices) (the FULL
        # length); that lie is exactly what made the earlier skip-insert attempt
        # leak — cache_*_req then used prev_prefix_len == full length and its
        # dedup loop neither inserted nor freed cross-request duplicate slots.
        # Pin the correct value and signal schedule_policy not to overwrite it.
        req.cache_protected_len = gpu_cached_len
        req._flexkv_uncached_restore = True

        # NOTE: device_indices may be longer than match_result.device_indices
        # by up to one page when is_eagle + page_size > 1 — that's normal
        # sglang behavior (the trailing partial page is not in the tree but
        # is held alive by req.prefix_indices for the request's forward).
        # The full device_indices is passed to the connector load op (so
        # FlexKV can H2D into all allocated slots, matching its src graph)
        # and to req.prefix_indices (so model forward sees them all).
        # ``new_node`` only reflects the in-tree slots, which is correct
        # for the lock_ref protection.

        self._load_queue.append(
            LoadOperation(
                rid=req.rid,
                device_indices=device_indices,
                node=new_node,
            )
        )
        # Track the (node, swa_uuid) pair keyed by op identity so
        # ready_to_load_host_cache can pair load_ops with their swa_uuids
        # when stashing into _ongoing_load_tasks. Using id() of the
        # LoadOperation works because load_ops live until ready_to_load
        # transfers them to _ongoing_load_tasks (then we drop this).
        if not hasattr(self, '_load_queue_swa_uuids'):
            self._load_queue_swa_uuids: Dict[int, Optional[int]] = {}
        self._load_queue_swa_uuids[id(self._load_queue[-1])] = load_swa_uuid

        req.prefix_indices = torch.cat([req.prefix_indices, device_indices])
        req.last_node = new_node
        # Match the (new_indices, last_node) contract that schedule_policy.add_one_req
        # expects.
        return device_indices, new_node

    def ready_to_load_host_cache(self) -> int:
        if self._connector is None or not self._load_queue:
            return -1

        task_id = self._load_task_id_counter
        self._load_task_id_counter += 1

        self._connector.start_load_kv(task_id, self._load_queue)

        # Pair each (node, swa_uuid) so _check_load_completion can pass the
        # right swa_uuid_for_lock to dec_lock_ref. Without the swa_uuid,
        # SWARadixCache.dec_lock_ref walks the SWA chain to root which can
        # underflow swa_lock_ref or hit ``dec_lock_ref on swa_tombstone node``.
        uuid_map = getattr(self, '_load_queue_swa_uuids', {})
        nodes_with_uuid = [
            (op.node, uuid_map.pop(id(op), None)) for op in self._load_queue
        ]
        self._ongoing_load_tasks[task_id] = nodes_with_uuid

        self._load_queue.clear()
        return task_id

    def cache_finished_req(self, req: Req, is_insert: bool = True, **kwargs):
        # Save kv_committed_len before super() pops it (pop_committed_kv_cache
        # sets kv_committed_freed=True and cannot be called again).
        kv_committed_len = req.kv_committed_len

        token_ids = None
        cache_to_connector = False
        if self._connector is not None and is_insert:
            req_id = req.req_pool_idx
            token_ids = (req.origin_input_ids + req.output_ids)[:kv_committed_len]
            # Truncate to page boundary (page_align_floor equivalent)
            page_aligned_len = (len(token_ids) // self.page_size) * self.page_size
            token_ids = token_ids[:page_aligned_len]
            if len(token_ids) > 0 and req_id is not None:
                cache_to_connector = True

        # Let the inner radix tree do insert + free duplicates + dec_lock_ref.
        self._inner_radixtree.cache_finished_req(req, is_insert=is_insert, **kwargs)

        if not cache_to_connector:
            return

        self._flexkv_store_prefix(token_ids, req, tag="cache_finished_req")

    def _flexkv_store_prefix(self, token_ids, req, tag: str) -> None:
        """Re-match ``token_ids`` to the resolved leaf node and launch an async
        D2H store of its kv_indices to the external connector.

        The node is pinned via inc_lock_ref BEFORE the async transfer so evict
        cannot free the pages while D2H is in flight; the pin is released in
        _check_store_completion. On SWARadixCache this pins BOTH full_lock_ref
        and a bounded swa_lock_ref window (returning swa_uuid_for_lock, which is
        threaded back to dec_lock_ref). Shared by cache_finished_req and the
        prefill-end cache_unfinished_req store.
        """
        # Re-match the tree to get the actual leaf node and its kv_indices AFTER
        # insert. These kv_indices are tree node values (not the req_to_token
        # snapshot), so they are protected by lock_ref and won't be freed by the
        # allocator while D2H transfer is in flight.
        radix_key = RadixKey(token_ids, req.extra_key)
        match_result = self._inner_radixtree.match_prefix(
            MatchPrefixParams(key=radix_key)
        )
        new_last_node = match_result.last_device_node
        if new_last_node is None or new_last_node is self._inner_radixtree.root_node:
            return

        kv_indices = match_result.device_indices
        if kv_indices is None or kv_indices.numel() == 0:
            return

        # The radix tree may return fewer indices than tokens passed in (EAGLE
        # bigram mode; page-alignment in match_prefix). We cannot store more
        # tokens than we have indices for, so truncate token_ids to match.
        if len(token_ids) > kv_indices.numel():
            token_ids = token_ids[: kv_indices.numel()]
        if len(token_ids) == 0:
            return
        if len(token_ids) != kv_indices.numel():
            logger.warning(
                "[FlexKV] %s: irrecoverable length mismatch! "
                "len(token_ids)=%d, kv_indices.numel()=%d, skipping store",
                tag, len(token_ids), kv_indices.numel(),
            )
            return

        # Lock the resolved tree node BEFORE starting async D2H transfer so that
        # evict cannot free these pages while transfer is in flight. On
        # SWARadixCache, inc_lock_ref locks both full_lock_ref and a bounded
        # prefix of swa_lock_ref (up to sliding_window_size from the leaf). It
        # returns IncLockRefResult.swa_uuid_for_lock which MUST be passed to
        # dec_lock_ref so the SWA-side release walks the exact same range. Plain
        # RadixCache returns the result but ignores the SWA fields, so this also
        # works there.
        inc_result = self._flexkv_inc_lock(
            new_last_node,
            f"{tag}:flexkv_store",
            rid=req.rid,
            token_count=len(token_ids),
            node_id=getattr(new_last_node, "id", None),
        )
        swa_uuid_for_lock = getattr(inc_result, 'swa_uuid_for_lock', None)

        task_id = self._load_task_id_counter
        self._load_task_id_counter += 1

        self._connector.start_store_kv(
            task_id=task_id,
            token_ids=token_ids,
            kv_indices=kv_indices,
        )

        self._ongoing_store_tasks[task_id] = (new_last_node, swa_uuid_for_lock)

    def evict(self, params: EvictParams) -> EvictResult:
        return self._inner_radixtree.evict(params)

    def check_kv_events(self):
        if self._connector is None:
            return
        self._check_store_completion()
        self._check_load_completion()

    # Alias for compatibility with scheduler (which calls check_hicache_events)
    def check_hicache_events(self):
        self.check_kv_events()

    def prefetch(self, req: Req) -> None:
        if self._connector is None:
            return
        token_ids = (req.origin_input_ids + req.output_ids)[:-1]
        self._connector.prefetch(req.rid, token_ids)

    def check_prefetch_progress(self, req_id: str) -> bool:
        if self._connector is None:
            return True
        return self._connector.check_prefetch_progress(req_id)

    def pop_prefetch_loaded_tokens(self, req_id: str) -> int:
        if self._connector is None:
            return 0
        return self._connector.pop_prefetch_loaded_tokens(req_id)

    def release_aborted_request(self, req_id: str) -> None:
        if self._connector is None:
            return
        self._connector.cancel_prefetch(req_id)

    # -- Private helpers --

    def _flexkv_inc_lock(
        self,
        node: TreeNode,
        tag: str,
        **ctx,
    ):
        return self._inner_radixtree.inc_lock_ref(node)

    def _flexkv_dec_lock(
        self,
        node: TreeNode,
        swa_uuid,
        tag: str,
        **ctx,
    ):
        try:
            from sglang.srt.mem_cache.base_prefix_cache import DecLockRefParams

            params = DecLockRefParams(swa_uuid_for_lock=swa_uuid)
            return self._inner_radixtree.dec_lock_ref(node, params)
        except (ImportError, TypeError):
            return self._inner_radixtree.dec_lock_ref(node)

    def _check_store_completion(self) -> None:
        completed_ids = self._connector.check_completed_store_tasks()
        for task_id in completed_ids:
            entry = self._ongoing_store_tasks.pop(task_id, None)
            if entry is None:
                continue
            # Backward-compat: entry may be a bare node OR (node, swa_uuid)
            if isinstance(entry, tuple):
                node, swa_uuid = entry
            else:
                node, swa_uuid = entry, None
            self._flexkv_dec_lock(
                node,
                swa_uuid,
                "check_store_completion",
                task_id=task_id,
                node_id=getattr(node, "id", None),
            )

    def _check_load_completion(self) -> None:
        completed_ids = self._connector.check_completed_load_tasks()
        for task_id in completed_ids:
            nodes = self._ongoing_load_tasks.pop(task_id, None)
            if nodes is None:
                continue
            for entry in nodes:
                # Backward-compat: entry may be a bare node OR (node, swa_uuid)
                if isinstance(entry, tuple):
                    node, swa_uuid = entry
                else:
                    node, swa_uuid = entry, None
                self._flexkv_dec_lock(
                    node,
                    swa_uuid,
                    "check_load_completion",
                    task_id=task_id,
                    node_id=getattr(node, "id", None),
                )

    # -- Pass-through methods --

    def insert(self, *args, **kwargs):
        return self._inner_radixtree.insert(*args, **kwargs)

    def inc_lock_ref(self, *args, **kwargs):
        return self._inner_radixtree.inc_lock_ref(*args, **kwargs)

    def dec_lock_ref(self, *args, **kwargs):
        return self._inner_radixtree.dec_lock_ref(*args, **kwargs)

    def cache_unfinished_req(self, req: Req, chunked: bool = False, **kwargs):
        # Inner cache inserts the committed prefix and pins it for the ongoing
        # request: SWARadixCache.cache_unfinished_req calls inc_lock_ref on the
        # new leaf, which locks full_lock_ref on [leaf, root] AND swa_lock_ref
        # within the sliding window, storing swa_uuid_for_lock on the req. That
        # pin is held through decode until cache_finished_req releases it, so
        # decode cannot evict / overwrite this prefix's SWA window.
        self._inner_radixtree.cache_unfinished_req(req, chunked=chunked, **kwargs)

        # Prefill-end (optimized) store: persist the prompt prefix's full + SWA
        # KV to the external connector ONCE, at the prefill->decode boundary.
        # Gated on `not chunked`: cache_unfinished_req also fires at every prefill
        # chunk boundary (chunked=True), where storing a partial prefix would be
        # wasted work / risk attaching to a partial-prefix node. Only the final,
        # non-chunked call corresponds to "prompt fully prefilled".
        if self._connector is None or chunked:
            return

        # Committed prefix length is what the inner cache just inserted/pinned.
        protected_len = getattr(req, "cache_protected_len", 0)
        if protected_len <= 0:
            return
        token_ids = req.fill_ids[:protected_len]
        page_aligned_len = (len(token_ids) // self.page_size) * self.page_size
        token_ids = token_ids[:page_aligned_len]
        if len(token_ids) == 0 or req.req_pool_idx is None:
            return

        self._flexkv_store_prefix(token_ids, req, tag="cache_unfinished_req")

    def evictable_size(self):
        inner = self._inner_radixtree
        if inner.supports_swa():
            return inner.full_evictable_size()
        return inner.evictable_size()

    def protected_size(self):
        inner = self._inner_radixtree
        if inner.supports_swa():
            return inner.full_protected_size()
        return inner.protected_size()

    def available_and_evictable_str(self) -> str:
        inner = self._inner_radixtree
        if hasattr(inner, "available_and_evictable_str"):
            return inner.available_and_evictable_str()
        return super().available_and_evictable_str()

    # NOTE: BasePrefixCache gives non-raising DEFAULT implementations for the
    # methods below (the size getters return 0, supports_* return False).
    # Because the base class defines them, normal attribute lookup resolves
    # them BEFORE __getattr__ fires, so __getattr__ never delegates them to the
    # inner cache and the inner's real value is silently shadowed by the base
    # default. Every method here is overridden by at least one inner cache we
    # wrap (SWARadixCache / MambaRadixCache / ...), so it MUST be forwarded
    # explicitly.
    #
    # This shadowing previously caused a phantom "pool memory leak detected"
    # crash: under hybrid-SWA, invariant_checker and pool_stats_observer read
    # tree_cache.full_protected_size()/swa_protected_size()/full_evictable_size()
    # /swa_evictable_size(), got the base-class 0 instead of the inner
    # SWARadixCache's locked sizes, and mis-counted the locked pages as leaked.
    def full_evictable_size(self):
        return self._inner_radixtree.full_evictable_size()

    def swa_evictable_size(self):
        return self._inner_radixtree.swa_evictable_size()

    def full_protected_size(self):
        return self._inner_radixtree.full_protected_size()

    def swa_protected_size(self):
        return self._inner_radixtree.swa_protected_size()

    def supports_swa(self) -> bool:
        return self._inner_radixtree.supports_swa()

    def supports_mamba(self) -> bool:
        return self._inner_radixtree.supports_mamba()

    def is_chunk_cache(self) -> bool:
        return self._inner_radixtree.is_chunk_cache()

    def flush_write_through_acks(self) -> None:
        return self._inner_radixtree.flush_write_through_acks()

    def sanity_check(self, *args, **kwargs):
        # An in-flight async store (FlexKV D2H) holds full_lock_ref/swa_lock_ref
        # on the stored leaf AND every ancestor up to root (see
        # SWARadixCache.inc_lock_ref, which locks the [leaf, root) path) until
        # the transfer completes and _check_store_completion runs dec_lock_ref.
        # The scheduler can go idle and run sanity_check while a store is still
        # in flight (is_fully_idle does not account for connector store tasks),
        # so those nodes are legitimately locked at idle. Collect their ids and
        # exempt them from SWARadixCache's "must be unlocked when idle" assert,
        # mirroring how UnifiedRadixCache.sanity_check tolerates
        # ongoing_write_through / ongoing_load_back nodes.
        inner = self._inner_radixtree
        if not self._ongoing_store_tasks or not inner.supports_swa():
            return inner.sanity_check(*args, **kwargs)

        exempt_node_ids: Set[int] = set()
        root_node = getattr(inner, "root_node", None)
        for entry in self._ongoing_store_tasks.values():
            # main2 stores (node, swa_uuid) tuples; tolerate a bare node too.
            node = entry[0] if isinstance(entry, tuple) else entry
            walker = node
            while walker is not None and walker is not root_node:
                exempt_node_ids.add(walker.id)
                walker = walker.parent
        return inner.sanity_check(exempt_node_ids=exempt_node_ids)

    def total_size(self):
        return self._inner_radixtree.total_size()

    def pretty_print(self):
        return self._inner_radixtree.pretty_print()

    def all_values_flatten(self):
        return self._inner_radixtree.all_values_flatten()

    def take_events(self):
        return self._inner_radixtree.take_events()

    def __getattr__(self, name):
        return getattr(self._inner_radixtree, name)
