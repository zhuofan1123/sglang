"""External-cache integration helpers for decode-side PD disaggregation.

For HiRadix this maps to HiCache host/storage loadback. For FlexKV this treats
the connector as an opaque external-cache manager: FlexKV owns its internal L2/L3
policy, while decode only consumes the connector-reported hit length.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, List, Optional

import torch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.mem_cache.base_prefix_cache import InitLoadBackParams

if TYPE_CHECKING:
    from sglang.srt.disaggregation.decode import DecodeRequest
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)


@dataclass
class DecodePrefixMatch:
    prefix_indices: torch.Tensor
    external_hit_length: int
    l3_storage_hit_length: int
    last_device_node: Any
    last_host_node: Any = None
    prefetch_registered: bool = False

    @property
    def l1_prefix_len(self) -> int:
        return len(self.prefix_indices)

    @property
    def decode_prefix_len(self) -> int:
        return self.l1_prefix_len + self.external_hit_length + self.l3_storage_hit_length

    @property
    def needs_local_restore(self) -> bool:
        return self.decode_prefix_len > self.l1_prefix_len

    @property
    def restore_token_count(self) -> int:
        return self.decode_prefix_len - self.l1_prefix_len


class HiCacheRestoreResult(Enum):
    PENDING = "pending"
    READY = "ready"
    FAILED = "failed"


class DecodeHiCachePreallocMixin:
    """HiCache hooks for DecodePreallocQueue."""

    def _decode_external_hit_budget(self, allocatable_tokens: int) -> Optional[int]:
        """Return max FlexKV external-hit tokens this admission may promise.

        HiRadix owns L2/L3 state in-tree, so this guard only applies to opaque
        connector-backed caches such as FlexKV. The budget is applied before
        connector prefix match, ensuring the pending connector load task and the
        later slot mapping have the same bounded length.
        """
        if (
            not self.scheduler.enable_decode_hicache
            or getattr(self.tree_cache, "_connector", None) is None
        ):
            return None

        policy = os.getenv(
            "SGLANG_FLEXKV_DECODE_EXTERNAL_HIT_BUDGET_POLICY", "ratio"
        ).strip().lower()
        if policy in ("off", "disable", "disabled", "none"):
            return None
        if policy in ("zero", "none_on_pressure"):
            return 0

        try:
            ratio = float(
                os.getenv("SGLANG_FLEXKV_DECODE_EXTERNAL_HIT_BUDGET_RATIO", "0.5")
            )
        except ValueError:
            ratio = 0.5
        ratio = min(max(ratio, 0.0), 1.0)

        try:
            safety_tokens = int(
                os.getenv(
                    "SGLANG_FLEXKV_DECODE_EXTERNAL_HIT_SAFETY_TOKENS",
                    str(getattr(self, "num_reserved_decode_tokens", 0)),
                )
            )
        except ValueError:
            safety_tokens = getattr(self, "num_reserved_decode_tokens", 0)

        restore_budget = max(0, allocatable_tokens - max(safety_tokens, 0))
        restore_budget = int(restore_budget * ratio)

        page_size = getattr(self.token_to_kv_pool_allocator, "page_size", 1)
        if page_size > 1:
            restore_budget = restore_budget // page_size * page_size

        try:
            min_tokens = int(
                os.getenv(
                    "SGLANG_FLEXKV_DECODE_EXTERNAL_HIT_MIN_TOKENS",
                    str(max(page_size, 1)),
                )
            )
        except ValueError:
            min_tokens = max(page_size, 1)
        if restore_budget < max(min_tokens, 1):
            return 0
        return restore_budget

    def _build_decode_prefix_match(self, req: Req, result: Any) -> DecodePrefixMatch:
        prefix_indices = req.prefix_indices
        external_hit_length = result.host_hit_length
        l3_storage_hit_length = 0
        last_host_node = result.last_host_node
        has_connector = getattr(self.tree_cache, "_connector", None) is not None

        if not self.scheduler.enable_decode_hicache:
            external_hit_length = 0
        if last_host_node is None:
            external_hit_length = 0

        # The legacy HiCache path can refuse small loadbacks via
        # load_back_threshold. Do not promise those tokens to prefill unless the
        # decode node can restore them locally.
        load_back_threshold = getattr(self.tree_cache, "load_back_threshold", 0)
        if 0 < external_hit_length < load_back_threshold:
            external_hit_length = 0

        if (
            self.scheduler.enable_decode_hicache
            and not has_connector
            and external_hit_length > 0
            and hasattr(self.tree_cache, "query_storage_hit_length")
            and last_host_node is not None
        ):
            try:
                matched_len = len(prefix_indices) + external_hit_length
                suffix_tokens = req.origin_input_ids[matched_len:]
                last_hash = (
                    last_host_node.get_last_hash_value()
                    if hasattr(last_host_node, "get_last_hash_value")
                    else None
                )
                prefix_keys = (
                    last_host_node.get_prefix_hash_values(last_host_node.parent)
                    if getattr(self.tree_cache, "hicache_storage_pass_prefix_keys", False)
                    and hasattr(last_host_node, "get_prefix_hash_values")
                    else None
                )
                l3_storage_hit_length = self.tree_cache.query_storage_hit_length(
                    last_host_node,
                    suffix_tokens,
                    last_hash,
                    prefix_keys,
                )
            except Exception as e:
                logger.warning(
                    "Decode HiCache storage hit query failed for rid=%s: %s",
                    req.rid,
                    e,
                )
                l3_storage_hit_length = 0

        return DecodePrefixMatch(
            prefix_indices=prefix_indices,
            external_hit_length=external_hit_length,
            l3_storage_hit_length=l3_storage_hit_length,
            last_device_node=result.last_device_node,
            last_host_node=last_host_node,
        )

    def _start_hicache_prefetch(
        self, req: Req, prefix_match: Optional[DecodePrefixMatch]
    ) -> None:
        if (
            prefix_match is None
            or prefix_match.l3_storage_hit_length <= 0
            or prefix_match.last_host_node is None
            or not hasattr(self.tree_cache, "prefetch_from_storage")
        ):
            return

        try:
            node = prefix_match.last_host_node
            matched_len = prefix_match.l1_prefix_len + prefix_match.external_hit_length
            suffix = req.origin_input_ids[
                matched_len : matched_len + prefix_match.l3_storage_hit_length
            ]
            last_hash = (
                node.get_last_hash_value()
                if hasattr(node, "get_last_hash_value")
                else None
            )
            prefix_keys = (
                node.get_prefix_hash_values(node.parent)
                if getattr(self.tree_cache, "hicache_storage_pass_prefix_keys", False)
                and hasattr(node, "get_prefix_hash_values")
                else None
            )
            self.tree_cache.prefetch_from_storage(
                req.rid, node, suffix, last_hash, prefix_keys
            )
            prefix_match.prefetch_registered = req.rid in getattr(
                self.tree_cache, "ongoing_prefetch", {}
            )
        except Exception as e:
            logger.warning(
                "Decode HiCache storage prefetch failed for rid=%s: %s",
                req.rid,
                e,
            )
            prefix_match.l3_storage_hit_length = 0
            prefix_match.prefetch_registered = False

    def _hicache_pending_restore_tokens(self) -> int:
        if not self.scheduler.enable_decode_hicache:
            return 0
        return sum(
            dr.prefix_match.restore_token_count
            for dr in self.transfer_queue.queue
            if dr.prefix_match is not None
            and dr.hicache_restore_status == HiCacheRestoreResult.PENDING
            and dr.hicache_restored_node is None
        )


class HiCacheRestoreGatedKVReceiver:
    """Gate KV transfer Success until the local HiCache restore is ready."""

    def __init__(self, decode_req: DecodeRequest):
        self.decode_req = decode_req

    def poll(self) -> KVPoll:
        poll = self.decode_req.kv_receiver.poll()
        if (
            poll == KVPoll.Success
            and self.decode_req.hicache_restore_status == HiCacheRestoreResult.PENDING
        ):
            return KVPoll.Transferring
        return poll


class DecodeHiCacheTransferMixin:
    """HiCache hooks for DecodeTransferQueue."""

    def _is_hicache_load_done(self, consumer_index: int) -> bool:
        if consumer_index < 0:
            return True
        if hasattr(self.tree_cache, "is_load_back_event_done"):
            return self.tree_cache.is_load_back_event_done(consumer_index)

        counter = getattr(
            getattr(self.tree_cache, "cache_controller", None),
            "layer_done_counter",
            getattr(self.tree_cache, "layer_done_counter", None),
        )
        if counter is None:
            return True
        event = counter.events[consumer_index]
        if hasattr(event, "finish_event"):
            return event.finish_event.query()
        if hasattr(event, "_finished"):
            return event._finished
        return True

    def _next_hicache_load_slot_free(self) -> bool:
        counter = getattr(
            getattr(self.tree_cache, "cache_controller", None),
            "layer_done_counter",
            getattr(self.tree_cache, "layer_done_counter", None),
        )
        if counter is None:
            return True
        if hasattr(counter, "is_next_slot_ready"):
            return counter.is_next_slot_ready()
        next_index = (counter.producer_index + 1) % counter.num_counters
        event = counter.events[next_index]
        if hasattr(event, "finish_event"):
            return event.finish_event.query()
        if hasattr(event, "_finished"):
            return event._finished
        return True

    def _clean_hicache_prefetch_resources(self, decode_req: DecodeRequest) -> None:
        if (
            decode_req.prefix_match is not None
            and (
                decode_req.prefix_match.prefetch_registered
                or decode_req.prefix_match.needs_local_restore
            )
            and hasattr(self.tree_cache, "release_aborted_request")
        ):
            self.tree_cache.release_aborted_request(decode_req.req.rid)
        if (
            decode_req.hicache_load_consumer_index >= 0
            and hasattr(self.tree_cache, "cancel_load_back")
        ):
            self.tree_cache.cancel_load_back(decode_req.hicache_load_consumer_index)
            decode_req.hicache_load_consumer_index = -1
        if (
            decode_req.hicache_restored_node is not None
            and (
                decode_req.prefix_match is None
                or decode_req.hicache_restored_node
                is not decode_req.prefix_match.last_device_node
            )
        ):
            self.tree_cache.dec_lock_ref(decode_req.hicache_restored_node)
            decode_req.hicache_restored_node = None

    def _try_hicache_queue_load_back(self, dr: DecodeRequest) -> bool:
        pm = dr.prefix_match
        assert pm is not None

        if pm.l3_storage_hit_length > 0 and hasattr(
            self.tree_cache, "check_prefetch_progress"
        ):
            if not self.tree_cache.check_prefetch_progress(dr.req.rid):
                return False
            if hasattr(self.tree_cache, "pop_prefetch_loaded_tokens"):
                dr.req.storage_hit_length = self.tree_cache.pop_prefetch_loaded_tokens(
                    dr.req.rid
                )

        original_last_node = dr.req.last_node
        old_prefix_len = len(dr.req.prefix_indices)
        load_result = self.tree_cache.init_load_back(
            InitLoadBackParams(
                last_host_node=pm.last_host_node,
                host_hit_length=pm.restore_token_count,
                req=dr.req,
            )
        )
        if load_result is False:
            dr.req.last_node = original_last_node
            dr.hicache_restore_status = HiCacheRestoreResult.FAILED
            return False

        if isinstance(load_result, tuple):
            new_indices, restored_node = load_result
            if len(new_indices) > 0:
                new_indices = new_indices.to(
                    dtype=torch.int64, device=dr.req.prefix_indices.device
                )
                dr.req.prefix_indices = torch.cat([dr.req.prefix_indices, new_indices])
        else:
            restored_node = dr.req.last_node
        # Keep req.last_node pointing to the originally locked device prefix
        # until commit. Failure cleanup relies on release_kv_cache to dec that
        # original lock, while _clean_hicache_prefetch_resources releases the
        # newly restored node.
        dr.req.last_node = original_last_node

        restored_indices = dr.req.prefix_indices[pm.l1_prefix_len : pm.decode_prefix_len]
        if len(restored_indices) < pm.restore_token_count:
            logger.warning(
                "Decode HiCache loadback failed for rid=%s: restored=%d expected=%d",
                dr.req.rid,
                len(restored_indices),
                pm.restore_token_count,
            )
            dr.hicache_restore_status = HiCacheRestoreResult.FAILED
            return False

        dr.hicache_restored_kv_indices = restored_indices
        dr.hicache_restored_node = restored_node
        if restored_node is not None and restored_node is not pm.last_device_node:
            self.tree_cache.inc_lock_ref(restored_node)
        if len(dr.req.prefix_indices) == old_prefix_len:
            dr.hicache_restore_status = HiCacheRestoreResult.READY
            return False
        return True

    def _process_hicache_local_restores(self, decode_reqs: List[DecodeRequest]) -> None:
        active: List[DecodeRequest] = []
        for dr in decode_reqs:
            if dr.hicache_restore_status != HiCacheRestoreResult.PENDING:
                continue
            pm = dr.prefix_match
            if pm is None or not pm.needs_local_restore:
                dr.hicache_restore_status = HiCacheRestoreResult.READY
                continue
            active.append(dr)

        for dr in active:
            if (
                dr.hicache_load_consumer_index >= 0
                and self._is_hicache_load_done(dr.hicache_load_consumer_index)
            ):
                dr.hicache_restore_status = HiCacheRestoreResult.READY

        if not self._next_hicache_load_slot_free():
            return

        queued = [
            dr
            for dr in active
            if dr.hicache_restore_status == HiCacheRestoreResult.PENDING
            and dr.hicache_restored_node is None
            and self._try_hicache_queue_load_back(dr)
        ]
        if not queued:
            return

        consumer_index = self.tree_cache.ready_to_load_host_cache()
        if consumer_index < 0:
            for dr in queued:
                dr.hicache_restore_status = HiCacheRestoreResult.READY
            return
        for dr in queued:
            dr.hicache_load_consumer_index = consumer_index

    def _commit_hicache_local_restore_to_req(self, decode_req: DecodeRequest) -> None:
        prefix_match = decode_req.prefix_match
        if prefix_match is None or not prefix_match.needs_local_restore:
            return

        self.tree_cache.dec_lock_ref(prefix_match.last_device_node)
        self.tree_cache.req_to_token_pool.write(
            (
                decode_req.req.req_pool_idx,
                slice(prefix_match.l1_prefix_len, prefix_match.decode_prefix_len),
            ),
            decode_req.hicache_restored_kv_indices,
        )
        restored_kv_indices = decode_req.hicache_restored_kv_indices
        prefix_indices = prefix_match.prefix_indices.to(
            dtype=torch.int64, device=restored_kv_indices.device
        )
        decode_req.req.prefix_indices = torch.cat(
            [prefix_indices, restored_kv_indices]
        )
        decode_req.req.last_node = decode_req.hicache_restored_node
        decode_req.prefix_match = None
