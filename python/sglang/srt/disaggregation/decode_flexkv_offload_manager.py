from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Dict, Set

from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.radix_cache import page_align_keys

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class DecodeFlexKVOffloadManager:
    """Periodically store decode-generated KV into the FlexKV connector.

    The running request keeps owning usable KV slots. The manager only advances
    page-aligned prefixes into the radix cache and asks the connector to store
    the matched radix KV asynchronously.
    """

    def __init__(
        self,
        tree_cache: BasePrefixCache,
        server_args: ServerArgs,
    ) -> None:
        self.tree_cache = tree_cache
        self.server_args = server_args
        self.enabled = bool(
            int(os.getenv("SGLANG_FLEXKV_DECODE_OFFLOAD_ENABLE", "1"))
        )
        self.interval_tokens = max(
            1,
            int(os.getenv("SGLANG_FLEXKV_DECODE_OFFLOAD_INTERVAL_TOKENS", "128")),
        )
        self.max_inflight = max(
            1,
            int(os.getenv("SGLANG_FLEXKV_DECODE_OFFLOAD_MAX_INFLIGHT", "2")),
        )
        self.offloaded_lens: Dict[str, int] = {}
        self._inflight_store_rids: Set[str] = set()

        if self.enabled:
            logger.info(
                "Enable decode FlexKV offload: interval_tokens=%d max_inflight=%d",
                self.interval_tokens,
                self.max_inflight,
            )

    def check_offload_progress(self) -> None:
        check_kv_events = getattr(self.tree_cache, "check_kv_events", None)
        if callable(check_kv_events):
            check_kv_events()

        store_inflight = getattr(self.tree_cache, "store_inflight_count", None)
        if callable(store_inflight) and store_inflight() == 0:
            self._inflight_store_rids.clear()

    def offload_kv_cache(self, req: Req) -> bool:
        if not self.enabled:
            return False
        if not req.rid or req.req_pool_idx is None:
            return False
        if len(req.output_ids) == 0 or req.kv_committed_len <= 0:
            return False

        self.check_offload_progress()
        if req.rid in self._inflight_store_rids:
            return False

        inflight_count = getattr(self.tree_cache, "store_inflight_count", None)
        if callable(inflight_count) and inflight_count() >= self.max_inflight:
            return False

        token_ids = req.origin_input_ids + req.output_ids
        store_len = min(len(token_ids), req.kv_committed_len)
        store_len = store_len // self.tree_cache.page_size * self.tree_cache.page_size
        if store_len <= 0:
            return False

        base_len = (
            req.cache_protected_len
            // self.tree_cache.page_size
            * self.tree_cache.page_size
        )
        last_store_len = self.offloaded_lens.setdefault(req.rid, base_len)
        if store_len - last_store_len < self.interval_tokens:
            return False

        store_token_ids = token_ids[:store_len]
        old_fill_ids = req.fill_ids
        req.fill_ids = store_token_ids
        try:
            # Insert the page-aligned running prefix into the local radix cache.
            # This protects the KV slots via radix locks while async D2H/remote
            # store is in flight and also deduplicates against existing L1.
            self.tree_cache.cache_unfinished_req(req)

            aligned_token_ids = page_align_keys(store_token_ids, self.tree_cache.page_size)
            if not aligned_token_ids or len(aligned_token_ids) <= last_store_len:
                return False

            start_store = getattr(self.tree_cache, "start_store_kv_for_token_ids", None)
            if not callable(start_store):
                self.offloaded_lens[req.rid] = len(aligned_token_ids)
                return False

            launched = start_store(
                req,
                aligned_token_ids,
                event="decode_unfinished",
            )
            self.offloaded_lens[req.rid] = len(aligned_token_ids)
            if launched:
                self._inflight_store_rids.add(req.rid)
                logger.info(
                    "[FlexKV-DecodeOffload] event=launch rid=%s tokens=%d "
                    "last_store_len=%d",
                    req.rid,
                    len(aligned_token_ids),
                    last_store_len,
                )
            else:
                logger.warning(
                    "[FlexKV-DecodeOffload] event=skip rid=%s tokens=%d "
                    "reason=store_not_launched",
                    req.rid,
                    len(aligned_token_ids),
                )
            return launched
        finally:
            req.fill_ids = old_fill_ids

    def finalize_request(self, req: Req) -> None:
        if not req.rid:
            return
        self.offloaded_lens.pop(req.rid, None)
        self._inflight_store_rids.discard(req.rid)

    def reset(self) -> None:
        self.offloaded_lens.clear()
        self._inflight_store_rids.clear()
