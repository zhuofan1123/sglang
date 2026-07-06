"""
Life cycle of a request in the decode server

1. PreallocQueue:
    a. Initialize a receiver for each request
    b. The request handshakes first, and pre-allocate kv once there is available kv.
    c. Move the request to TransferQueue.

2. TransferQueue:
    a. Poll the receiver to check the transfer state
    b. If the transfer has finished, move the request to waiting queue

3. WaitingQueue:
    a. Use the requests in the queue to construct a PrebuiltExtendBatch
    b. Skip the prefill forward but only populate metadata

4. RunningBatch:
    a. Merge the resolved PrebuiltExtendBatch into running batch to run decoding
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from http import HTTPStatus
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch
from torch.distributed import ProcessGroup

from sglang.srt.configs.mamba_utils import Mamba2CacheParams
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.common.conn import CommonKVManager, CommonKVReceiver
from sglang.srt.disaggregation.decode_hicache_mixin import (
    DecodeHiCachePreallocMixin,
    DecodeHiCacheTransferMixin,
    DecodePrefixMatch,
    HiCacheRestoreGatedKVReceiver,
    HiCacheRestoreResult,
)
from sglang.srt.disaggregation.utils import (
    FAKE_BOOTSTRAP_HOST,
    DisaggregationMode,
    KVClassType,
    MetadataBuffers,
    ReqToMetadataIdxAllocator,
    TransferBackend,
    get_kv_class,
    is_mla_backend,
    kv_to_page_indices,
    poll_and_all_reduce,
    prepare_abort,
)
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.managers.schedule_batch import FINISH_ABORT, ScheduleBatch
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    EvictParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.mem_cache.memory_pool import (
    HybridLinearKVPool,
    HybridReqToTokenPool,
    KVCache,
    NSATokenToKVPool,
    ReqToTokenPool,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.observability.req_time_stats import (
    set_schedule_time_batch,
    set_time_batch,
)
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter
from sglang.srt.utils import get_num_new_pages

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.server_args import ServerArgs

CLIP_MAX_NEW_TOKEN = envs.SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION.get()


def _is_fake_transfer(req: Req, server_args: ServerArgs) -> bool:
    return req.bootstrap_host == FAKE_BOOTSTRAP_HOST or (
        req.bootstrap_host is None
        and server_args.disaggregation_transfer_backend == "fake"
    )


def _bootstrap_addr(req: Req) -> str:
    # FIXME: make a property of a req
    return f"{req.bootstrap_host}:{req.bootstrap_port}"


class DecodeReqToTokenPool:
    """
    The difference of DecodeReqToTokenPool and ReqToTokenPool is that
    DecodeReqToTokenPool subscribes memory for pre-allocated requests.

    In ReqToTokenPool, if `--max-running-requests` is 8,
    #pre-allocated + #transfer + #running <= 8, but there are in fact more memory can carry pre-allocated requests.

    In DecodeReqToTokenPool, if `--max-running-requests` is 8,
    #running <= 8, #pre-allocated + #transfer <= pre_alloc_size, so we can use the free memory to pre-allocate requests to unblock prefill.
    """

    def __init__(
        self,
        size: int,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,
        pre_alloc_size: int,
    ):
        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )

        self.size = size
        self.max_context_len = max_context_len
        self.device = device
        self.pre_alloc_size = pre_alloc_size
        with memory_saver_adapter.region(tag=GPU_MEMORY_TYPE_KV_CACHE):
            self.req_to_token = torch.zeros(
                (size + pre_alloc_size, max_context_len),
                dtype=torch.int32,
                device=device,
            )

        self.free_slots = list(range(size + pre_alloc_size))

    def write(self, indices, values):
        self.req_to_token[indices] = values

    def available_size(self):
        return len(self.free_slots)

    def alloc(self, reqs: List["Req"]) -> Optional[List[int]]:
        # Indices of reqs that already have a req_pool_idx and will reuse
        # their existing slot (e.g. chunked prefill continuing across chunks).
        reusing = [i for i, r in enumerate(reqs) if r.req_pool_idx is not None]
        assert (
            len(reusing) <= 1
        ), "only one chunked request may reuse req_pool_idx in a batch"
        assert all(
            reqs[i].is_chunked > 0 or reqs[i].kv_committed_len > 0 for i in reusing
        ), "reusing request must be chunked or have committed KV"

        need_size = len(reqs) - len(reusing)
        if need_size > len(self.free_slots):
            return None
        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        offset = 0
        for r in reqs:
            if r.req_pool_idx is None:
                r.req_pool_idx = select_index[offset]
                offset += 1
        return [r.req_pool_idx for r in reqs]

    def free(self, req: "Req"):
        assert req.req_pool_idx is not None, "request must have req_pool_idx"
        self.free_slots.append(req.req_pool_idx)
        req.req_pool_idx = None

    def clear(self):
        self.free_slots = list(range(self.size + self.pre_alloc_size))


class HybridMambaDecodeReqToTokenPool(HybridReqToTokenPool):

    def __init__(
        self,
        size: int,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,
        cache_params: "Mamba2CacheParams",
        speculative_num_draft_tokens: int,
        enable_mamba_extra_buffer: bool,
        pre_alloc_size: int,
        enable_overlap_schedule: bool,
        mamba_size: int = None,
    ):
        DecodeReqToTokenPool.__init__(
            self,
            size=size,
            max_context_len=max_context_len,
            device=device,
            enable_memory_saver=enable_memory_saver,
            pre_alloc_size=pre_alloc_size,
        )

        self.mamba_ping_pong_track_buffer_size = 2 if enable_overlap_schedule else 1
        self.enable_mamba_extra_buffer = enable_mamba_extra_buffer
        self.enable_memory_saver = enable_memory_saver
        effective_mamba_size = (
            mamba_size if mamba_size is not None else size
        ) + pre_alloc_size
        # TODO: Support PP
        self.start_layer = 0
        self.layer_transfer_counter = None
        self._init_mamba_pool(
            size=effective_mamba_size,
            mamba_spec_state_size=size + pre_alloc_size,
            cache_params=cache_params,
            device=device,
            enable_mamba_extra_buffer=self.enable_mamba_extra_buffer,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
        )

    def clear(self):
        self.free_slots = list(range(self.size + self.pre_alloc_size))
        self.mamba_pool.clear()


@dataclass
class DecodeRequest:
    req: Req
    kv_receiver: CommonKVReceiver
    waiting_for_input: bool = False
    metadata_buffer_index: int = -1
    prefix_match: Optional[DecodePrefixMatch] = None
    hicache_restore_status: HiCacheRestoreResult = HiCacheRestoreResult.PENDING
    hicache_restored_node: Any = None
    hicache_restored_kv_indices: Optional[torch.Tensor] = None
    hicache_load_consumer_index: int = -1

    @property
    def seqlen(self) -> int:
        return self.req.seqlen


class DecodePreallocQueue(DecodeHiCachePreallocMixin):
    """
    Store the requests that are preallocating.
    """

    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        draft_token_to_kv_pool: Optional[KVCache],
        req_to_metadata_buffer_idx_allocator: ReqToMetadataIdxAllocator,
        metadata_buffers: MetadataBuffers,
        scheduler: Scheduler,
        transfer_queue: DecodeTransferQueue,
        tree_cache: BasePrefixCache,
        gloo_group: ProcessGroup,
        tp_rank: int,
        tp_size: int,
        dp_size: int,
        gpu_id: int,
        bootstrap_port: int,
        max_total_num_tokens: int,
        pp_rank: int,
        num_reserved_decode_tokens: int,
        transfer_backend: TransferBackend,
    ):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.token_to_kv_pool = token_to_kv_pool_allocator.get_kvcache()
        self.draft_token_to_kv_pool = draft_token_to_kv_pool
        self.is_mla_backend = is_mla_backend(self.token_to_kv_pool)
        self.metadata_buffers = metadata_buffers
        self.req_to_metadata_buffer_idx_allocator = req_to_metadata_buffer_idx_allocator
        self.scheduler = scheduler
        self.transfer_queue = transfer_queue
        self.tree_cache = tree_cache  # this is always a chunk cache
        self.gloo_group = gloo_group
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.dp_size = dp_size
        self.gpu_id = gpu_id
        self.bootstrap_port = bootstrap_port
        self.max_total_num_tokens = max_total_num_tokens
        self.pp_rank = pp_rank
        self.num_reserved_decode_tokens = num_reserved_decode_tokens
        self.transfer_backend = transfer_backend
        # Queue for requests pending pre-allocation
        self.queue: List[DecodeRequest] = []
        self.retracted_queue: List[Req] = []
        self.pending_reqs: List[Req] = []
        self._ensure_retry_count: Dict[str, int] = {}
        self._max_ensure_retries: int = 20  # scheduling cycles
        self._ensure_last_attempt_time: Dict[str, float] = {}
        self._ensure_retry_interval: float = 1.0  # seconds
        self.kv_manager = self._init_kv_manager()

        if self.scheduler.tp_worker.is_hybrid_swa:
            # FIXME: current SWA allocation allocate full kv cache size in prefill
            self.max_total_num_tokens = min(
                self.max_total_num_tokens,
                self.scheduler.tp_worker.model_runner.swa_max_total_num_tokens,
            )

    def _init_kv_manager(self) -> CommonKVManager:
        kv_args_class = get_kv_class(self.transfer_backend, KVClassType.KVARGS)
        kv_args = kv_args_class()

        attn_tp_size = get_attention_tp_size()
        kv_args.engine_rank = self.tp_rank % (attn_tp_size)

        kv_args.pp_rank = self.pp_rank
        kv_args.system_dp_rank = self.scheduler.dp_rank
        kv_data_ptrs, kv_data_lens, kv_item_lens = (
            self.token_to_kv_pool.get_contiguous_buf_infos()
        )
        if self.draft_token_to_kv_pool is not None:
            # We should also transfer draft model kv cache. The indices are
            # always shared with a target model.
            draft_kv_data_ptrs, draft_kv_data_lens, draft_kv_item_lens = (
                self.draft_token_to_kv_pool.get_contiguous_buf_infos()
            )
            kv_data_ptrs += draft_kv_data_ptrs
            kv_data_lens += draft_kv_data_lens
            kv_item_lens += draft_kv_item_lens

        kv_args.kv_data_ptrs = kv_data_ptrs
        kv_args.kv_data_lens = kv_data_lens
        kv_args.kv_item_lens = kv_item_lens
        kv_args.page_size = self.token_to_kv_pool.page_size

        kv_args.aux_data_ptrs, kv_args.aux_data_lens, kv_args.aux_item_lens = (
            self.metadata_buffers.get_buf_infos()
        )

        if hasattr(self.token_to_kv_pool, "get_state_buf_infos"):
            state_data_ptrs, state_data_lens, state_item_lens = (
                self.token_to_kv_pool.get_state_buf_infos()
            )
            kv_args.state_data_ptrs = state_data_ptrs
            kv_args.state_data_lens = state_data_lens
            kv_args.state_item_lens = state_item_lens

            if isinstance(self.token_to_kv_pool, SWAKVPool):
                kv_args.state_type = "swa"
            elif isinstance(self.token_to_kv_pool, HybridLinearKVPool):
                kv_args.state_type = "mamba"
                # Get state dimension info for cross-TP slice transfer
                if hasattr(self.token_to_kv_pool, "get_state_dim_per_tensor"):
                    kv_args.state_dim_per_tensor = (
                        self.token_to_kv_pool.get_state_dim_per_tensor()
                    )
            elif isinstance(self.token_to_kv_pool, NSATokenToKVPool):
                kv_args.state_type = "nsa"
            else:
                kv_args.state_type = "none"
        else:
            kv_args.state_data_ptrs = []
            kv_args.state_data_lens = []
            kv_args.state_item_lens = []
            kv_args.state_type = "none"

        kv_args.ib_device = self.scheduler.server_args.disaggregation_ib_device
        kv_args.gpu_id = self.scheduler.gpu_id
        kv_manager_class = get_kv_class(self.transfer_backend, KVClassType.MANAGER)
        kv_manager = kv_manager_class(
            kv_args,
            DisaggregationMode.DECODE,
            self.scheduler.server_args,
            self.is_mla_backend,
        )
        return kv_manager

    def add(self, req: Req, is_retracted: bool = False) -> None:
        """Add a request to the pending queue."""
        if self._check_if_req_exceed_kv_capacity(req):
            return

        if is_retracted:
            req.retraction_mb_id = None
            self.retracted_queue.append(req)
        else:
            # NOTE: fake transfer does not need to resolve prefill dp rank in the pending queue
            if _is_fake_transfer(req, self.scheduler.server_args):
                self._create_receiver_and_enqueue(req, 0)
                return

            # Fast path: cache-only lookup, no network calls
            prefill_dp_rank = self._resolve_prefill_dp_rank(req)
            if prefill_dp_rank is not None:
                self._create_receiver_and_enqueue(req, prefill_dp_rank)
            else:
                self.pending_reqs.append(req)

    def _resolve_prefill_dp_rank(self, req: Req) -> Optional[int]:
        if req.disagg_prefill_dp_rank is not None:
            return req.disagg_prefill_dp_rank

        prefill_info = self.kv_manager.prefill_info_table.get(_bootstrap_addr(req))
        if prefill_info is None:
            return None

        if prefill_info.dp_size == 1:
            return 0

        if prefill_info.follow_bootstrap_room:
            return req.bootstrap_room % prefill_info.dp_size

        return None

    def _create_receiver_and_enqueue(self, req: Req, prefill_dp_rank: int) -> None:
        backend = (
            TransferBackend.FAKE
            if _is_fake_transfer(req, self.scheduler.server_args)
            else self.transfer_backend
        )
        kv_receiver_class = get_kv_class(backend, KVClassType.RECEIVER)

        kv_receiver = kv_receiver_class(
            mgr=self.kv_manager,
            bootstrap_addr=_bootstrap_addr(req),
            bootstrap_room=req.bootstrap_room,
            prefill_dp_rank=prefill_dp_rank,
        )

        self.queue.append(
            DecodeRequest(req=req, kv_receiver=kv_receiver, waiting_for_input=False)
        )

    def _match_prefix_and_lock(
        self, req: Req, external_hit_budget: Optional[int] = None
    ) -> DecodePrefixMatch:
        max_prefix_len = len(req.origin_input_ids)
        if req.return_logprob and req.logprob_start_len >= 0:
            max_prefix_len = min(max_prefix_len, req.logprob_start_len)
        token_ids = req.origin_input_ids[:max_prefix_len]

        if external_hit_budget is not None:
            req.decode_external_hit_budget = external_hit_budget
        try:
            match_result = self.tree_cache.match_prefix(
                MatchPrefixParams(
                    key=RadixKey(token_ids=token_ids, extra_key=req.extra_key),
                    req=req,
                    cow_mamba=self.tree_cache.supports_mamba(),
                    update_connector_state=True,
                )
            )
        finally:
            if external_hit_budget is not None and hasattr(
                req, "decode_external_hit_budget"
            ):
                delattr(req, "decode_external_hit_budget")
        req.prefix_indices = match_result.device_indices.to(
            dtype=torch.int64, device=self.token_to_kv_pool_allocator.device
        )
        (
            req.last_node,
            req.last_host_node,
            req.host_hit_length,
            req.mamba_branching_seqlen,
        ) = (
            match_result.last_device_node,
            match_result.last_host_node,
            match_result.host_hit_length,
            match_result.mamba_branching_seqlen,
        )
        if match_result.cache_protected_len is not None:
            req.cache_protected_len = match_result.cache_protected_len
        else:
            req.cache_protected_len = len(req.prefix_indices)

        self.tree_cache.inc_lock_ref(match_result.last_device_node)
        return self._build_decode_prefix_match(req, match_result)

    def _release_prefix_match(
        self, req: Req, prefix_match: Optional[DecodePrefixMatch]
    ) -> None:
        if prefix_match is not None:
            self.tree_cache.dec_lock_ref(prefix_match.last_device_node)
            if prefix_match.needs_local_restore and hasattr(
                self.tree_cache, "release_aborted_request"
            ):
                self.tree_cache.release_aborted_request(req.rid)

    def _check_if_req_exceed_kv_capacity(self, req: Req) -> bool:
        if len(req.origin_input_ids) > self.max_total_num_tokens:
            message = f"Request {req.rid} exceeds the maximum number of tokens: {len(req.origin_input_ids)} > {self.max_total_num_tokens}"
            logger.error(message)
            prepare_abort(req, message, status_code=HTTPStatus.BAD_REQUEST)
            self.scheduler.stream_output([req], req.return_logprob)
            return True
        return False

    def extend(self, reqs: List[Req], is_retracted: bool = False) -> None:
        """Add a request to the pending queue."""
        for req in reqs:
            self.add(req, is_retracted=is_retracted)

    def resume_retracted_reqs(
        self, rids_to_check: Optional[List[str]] = None
    ) -> List[Req]:
        # TODO refactor the scheduling part, reuse with the unified engine logic as much as possible

        # allocate memory
        resumed_reqs = []
        indices_to_remove = set()
        allocatable_tokens = self._allocatable_tokens(count_retracted=False)

        for i, req in enumerate(self.retracted_queue):
            if rids_to_check is not None and req.rid not in rids_to_check:
                continue

            if self.req_to_token_pool.available_size() <= 0:
                break

            fill_len = len(req.origin_input_ids) + max(len(req.output_ids) - 1, 0)
            required_alloc_tokens = self._required_alloc_tokens(
                fill_len=fill_len,
                prefix_len=0,
            )
            required_tokens_for_request = (
                required_alloc_tokens + self.num_reserved_decode_tokens
            )
            if required_tokens_for_request > allocatable_tokens:
                break

            kv_loc = self._pre_alloc(req, fail_on_oom=True)
            if kv_loc is None:
                logger.warning(
                    "Skip resuming retracted decode request due to KV pressure: "
                    "rid=%s fill_len=%d required_alloc_tokens=%d "
                    "allocatable_tokens=%d available_size=%d evictable_size=%d",
                    req.rid,
                    fill_len,
                    required_alloc_tokens,
                    allocatable_tokens,
                    self.token_to_kv_pool_allocator.available_size(),
                    self.tree_cache.evictable_size(),
                )
                break

            resumed_reqs.append(req)
            indices_to_remove.add(i)
            req.is_retracted = False
            allocatable_tokens -= required_tokens_for_request

            # load from cpu, release the cpu copy
            req.load_kv_cache(self.req_to_token_pool, self.token_to_kv_pool_allocator)

        self.retracted_queue = [
            entry
            for i, entry in enumerate(self.retracted_queue)
            if i not in indices_to_remove
        ]

        return resumed_reqs

    def _update_handshake_waiters(
        self, rids_to_check: Optional[List[str]] = None
    ) -> None:
        if not self.queue:
            return

        if all(decode_req.waiting_for_input for decode_req in self.queue):
            return

        polls = poll_and_all_reduce(
            [decode_req.kv_receiver for decode_req in self.queue], self.gloo_group
        )

        for i, (decode_req, poll) in enumerate(zip(self.queue, polls)):
            if rids_to_check is not None and decode_req.req.rid not in rids_to_check:
                continue

            if poll == KVPoll.Bootstrapping:
                pass
            elif poll == KVPoll.WaitingForInput:
                decode_req.waiting_for_input = True
                decode_req.req.time_stats.set_bootstrap_done_time()
            elif poll == KVPoll.Failed:
                error_message = f"Decode handshake failed for request rank={self.tp_rank} {decode_req.req.rid=} {decode_req.req.bootstrap_room=}"
                try:
                    decode_req.kv_receiver.failure_exception()
                except Exception as e:
                    error_message += f" with exception {e}"
                logger.error(error_message)
                prepare_abort(
                    decode_req.req,
                    error_message,
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
                if self.scheduler.enable_metrics:
                    self.scheduler.metrics_collector.increment_bootstrap_failed_reqs()
            else:
                raise ValueError(f"Unexpected poll case: {poll}")

    def _ensure_prefill_info(
        self, addr_to_reqs: Dict[str, List[Req]]
    ) -> Tuple[Dict[str, List[Req]], List[Req]]:
        """Non-blocking ensure parallel info for each addr.
        Returns (ready_addrs, remaining_reqs)."""
        ready: Dict[str, List[Req]] = {}
        remaining: List[Req] = []

        now = time.monotonic()
        for bootstrap_addr, reqs in addr_to_reqs.items():
            last_attempt = self._ensure_last_attempt_time.get(bootstrap_addr)
            if last_attempt is not None and (
                now - last_attempt < self._ensure_retry_interval
            ):
                remaining.extend(reqs)
                continue

            self._ensure_last_attempt_time[bootstrap_addr] = now

            if self.kv_manager.try_ensure_parallel_info(bootstrap_addr):
                if bootstrap_addr in self._ensure_retry_count:
                    del self._ensure_retry_count[bootstrap_addr]
                if bootstrap_addr in self._ensure_last_attempt_time:
                    del self._ensure_last_attempt_time[bootstrap_addr]
                ready[bootstrap_addr] = reqs
                continue

            count = self._ensure_retry_count.get(bootstrap_addr, 0) + 1
            self._ensure_retry_count[bootstrap_addr] = count

            if count >= self._max_ensure_retries:
                error_msg = f"Could not fetch prefill parallel info from {bootstrap_addr} after {count} attempts"
                logger.error(error_msg)
                for req in reqs:
                    prepare_abort(
                        req, error_msg, status_code=HTTPStatus.INTERNAL_SERVER_ERROR
                    )
                    if self.scheduler.enable_metrics:
                        self.scheduler.metrics_collector.increment_bootstrap_failed_reqs()
                    self.scheduler.stream_output([req], req.return_logprob)
                del self._ensure_retry_count[bootstrap_addr]
                del self._ensure_last_attempt_time[bootstrap_addr]
            else:
                remaining.extend(reqs)

        return ready, remaining

    def _resolve_pending_reqs(self) -> None:
        """Batch-resolve prefill_dp_ranks for pending requests and create receivers."""
        if not self.pending_reqs:
            return

        # Group pending requests by bootstrap_addr
        addr_to_reqs: Dict[str, List[Req]] = {}
        for req in self.pending_reqs:
            addr = _bootstrap_addr(req)
            addr_to_reqs.setdefault(addr, []).append(req)

        # Pass 1: ensure parallel info for each addr
        ready_addrs, remaining = self._ensure_prefill_info(addr_to_reqs)

        # Pass 2: resolve dp rank for addrs whose info is available
        resolved = []
        for bootstrap_addr, reqs in ready_addrs.items():
            need_query: List[Req] = []
            for req in reqs:
                prefill_dp_rank = self._resolve_prefill_dp_rank(req)
                if prefill_dp_rank is not None:
                    resolved.append((req, prefill_dp_rank))
                else:
                    need_query.append(req)

            if need_query:
                rooms = [req.bootstrap_room for req in need_query]
                room_to_rank = CommonKVReceiver.query_prefill_dp_ranks(
                    bootstrap_addr, rooms
                )
                for req in need_query:
                    prefill_dp_rank = room_to_rank.get(str(req.bootstrap_room))
                    if prefill_dp_rank is not None:
                        resolved.append((req, int(prefill_dp_rank)))
                    else:
                        remaining.append(req)

        self.pending_reqs = remaining

        for req, prefill_dp_rank in resolved:
            self._create_receiver_and_enqueue(req, prefill_dp_rank)

    def pop_preallocated(
        self, rids_to_check: Optional[List[str]] = None
    ) -> Tuple[List[DecodeRequest], List[DecodeRequest]]:
        """Pop the preallocated requests from the pending queue (FIFO)."""
        self._resolve_pending_reqs()
        self._update_handshake_waiters(rids_to_check)

        failed_reqs = []
        preallocated_reqs = []
        indices_to_remove = set()

        # We need to make sure that the sum of inflight tokens and allocatable tokens is greater than maximum input+output length of each inflight request
        # Otherwise it is possible for one request running decode out of memory, while all other requests are in the transfer queue that cannot be retracted.
        retractable_tokens = sum(
            len(r.origin_input_ids) + len(r.output_ids)
            for r in self.scheduler.running_batch.reqs
        )
        allocatable_tokens = self._allocatable_tokens(
            retractable_tokens=retractable_tokens, count_retracted=True
        )
        # First, remove all failed requests from the queue
        for i, decode_req in enumerate(self.queue):
            if rids_to_check is not None and decode_req.req.rid not in rids_to_check:
                continue
            if isinstance(decode_req.req.finished_reason, FINISH_ABORT):
                self.scheduler.stream_output(
                    [decode_req.req], decode_req.req.return_logprob
                )
                failed_reqs.append(decode_req)
                indices_to_remove.add(i)

        # Then, preallocate the remaining requests if possible
        for i, decode_req in enumerate(self.queue):
            if rids_to_check is not None and decode_req.req.rid not in rids_to_check:
                continue

            if i in indices_to_remove:
                continue

            if not decode_req.waiting_for_input:
                continue

            if self.req_to_token_pool.available_size() <= 0:
                break

            if self.req_to_metadata_buffer_idx_allocator.available_size() <= 0:
                break

            # Memory estimation: don't add if the projected memory cannot be met
            # TODO: add new_token ratio
            origin_input_len = len(decode_req.req.origin_input_ids)
            prefix_match: Optional[DecodePrefixMatch] = None
            if self.scheduler.server_args.disaggregation_decode_enable_radix_cache:
                external_hit_budget = self._decode_external_hit_budget(
                    allocatable_tokens
                )
                prefix_match = self._match_prefix_and_lock(
                    decode_req.req, external_hit_budget=external_hit_budget
                )
                prefix_indices = prefix_match.prefix_indices
                prefix_len = prefix_match.l1_prefix_len
                total_prefix_len = prefix_match.decode_prefix_len
                fill_len = origin_input_len + max(len(decode_req.req.output_ids) - 1, 0)
                required_alloc_tokens = self._required_alloc_tokens(
                    fill_len=fill_len,
                    prefix_len=prefix_len,
                )
                allocatable_tokens = self._allocatable_tokens(
                    retractable_tokens=retractable_tokens,
                    count_retracted=True,
                )
            else:
                prefix_indices = None
                prefix_len = 0
                total_prefix_len = 0
                required_alloc_tokens = origin_input_len

            required_tokens_for_request = (
                required_alloc_tokens + self.num_reserved_decode_tokens
            )

            if (
                max(
                    required_tokens_for_request,
                    origin_input_len
                    - prefix_len
                    + min(
                        decode_req.req.sampling_params.max_new_tokens,
                        CLIP_MAX_NEW_TOKEN,
                    )
                    - retractable_tokens,
                )
                > allocatable_tokens
            ):
                self._release_prefix_match(decode_req.req, prefix_match)
                break
            if required_tokens_for_request > allocatable_tokens:
                self._release_prefix_match(decode_req.req, prefix_match)
                break

            kv_loc = self._pre_alloc(
                decode_req.req,
                prefix_indices,
                prefix_len,
                total_prefix_len,
                fail_on_oom=True,
            )
            if kv_loc is None:
                logger.warning(
                    "Skip preallocating decode request due to KV pressure: "
                    "rid=%s origin_input_len=%d prefix_len=%d total_prefix_len=%d "
                    "required_alloc_tokens=%d allocatable_tokens=%d "
                    "available_size=%d evictable_size=%d",
                    decode_req.req.rid,
                    origin_input_len,
                    prefix_len,
                    total_prefix_len,
                    required_alloc_tokens,
                    allocatable_tokens,
                    self.token_to_kv_pool_allocator.available_size(),
                    self.tree_cache.evictable_size(),
                )
                self._release_prefix_match(decode_req.req, prefix_match)
                break

            allocatable_tokens -= required_tokens_for_request
            decode_req.prefix_match = prefix_match
            if self.scheduler.enable_decode_hicache:
                self._start_hicache_prefetch(decode_req.req, prefix_match)
            decode_req.req.cache_protected_len = total_prefix_len

            kv_indices = (
                self.req_to_token_pool.req_to_token[decode_req.req.req_pool_idx][
                    total_prefix_len:origin_input_len
                ]
                .cpu()
                .numpy()
            )
            page_size = self.token_to_kv_pool_allocator.page_size

            # Prepare extra pool indices for hybrid models
            if isinstance(self.token_to_kv_pool, HybridLinearKVPool):
                # Mamba hybrid model: single mamba state index
                state_indices = [
                    self.req_to_token_pool.req_index_to_mamba_index_mapping[
                        decode_req.req.req_pool_idx
                    ]
                    .cpu()
                    .numpy()
                ]
            elif isinstance(self.token_to_kv_pool, SWAKVPool):
                # SWA hybrid model: send decode-side SWA window indices
                seq_len = len(decode_req.req.origin_input_ids)
                window_size = self.scheduler.sliding_window_size

                window_start = max(0, seq_len - window_size)
                window_start = (window_start // page_size) * page_size
                window_kv_indices_full = self.req_to_token_pool.req_to_token[
                    decode_req.req.req_pool_idx, window_start:seq_len
                ]

                # Translate to SWA pool indices
                window_kv_indices_swa = (
                    self.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
                        window_kv_indices_full
                    )
                )
                state_indices = window_kv_indices_swa.cpu().numpy()
                state_indices = kv_to_page_indices(state_indices, page_size)
            elif isinstance(self.token_to_kv_pool, NSATokenToKVPool):
                seq_len = len(decode_req.req.origin_input_ids)
                kv_indices_full = self.req_to_token_pool.req_to_token[
                    decode_req.req.req_pool_idx, :seq_len
                ]
                state_indices = kv_indices_full.cpu().numpy()
                state_indices = kv_to_page_indices(state_indices, page_size)
            else:
                state_indices = None

            decode_req.metadata_buffer_index = (
                self.req_to_metadata_buffer_idx_allocator.alloc()
            )
            assert decode_req.metadata_buffer_index is not None
            page_indices = kv_to_page_indices(kv_indices, page_size)
            decode_req.kv_receiver.init(
                page_indices,
                decode_req.metadata_buffer_index,
                state_indices,
                decode_prefix_len=total_prefix_len,
            )
            preallocated_reqs.append(decode_req)
            indices_to_remove.add(i)
            decode_req.req.time_stats.set_decode_transfer_queue_entry_time()

        self.queue = [
            entry for i, entry in enumerate(self.queue) if i not in indices_to_remove
        ]

        return preallocated_reqs, failed_reqs

    @property
    def num_tokens_pre_allocated(self):
        return sum(
            len(decode_req.req.fill_ids) for decode_req in self.transfer_queue.queue
        )

    def _allocatable_tokens(
        self, retractable_tokens: Optional[int] = None, count_retracted: bool = True
    ) -> int:
        need_space_for_single_req = (
            max(
                [
                    min(x.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKEN)
                    + len(x.origin_input_ids)
                    - retractable_tokens
                    for x in self.scheduler.running_batch.reqs
                ]
            )
            if retractable_tokens is not None
            and len(self.scheduler.running_batch.reqs) > 0
            else 0
        )
        available_size = self.token_to_kv_pool_allocator.available_size()
        if self.scheduler.server_args.disaggregation_decode_enable_radix_cache:
            available_size += self.tree_cache.evictable_size()
        allocatable_tokens = available_size - max(
            # preserve some space for future decode
            self.num_reserved_decode_tokens
            * (
                len(self.scheduler.running_batch.reqs)
                + len(self.transfer_queue.queue)
                + len(self.scheduler.waiting_queue)
            ),
            # make sure each request can finish if reach max_tokens with all other requests retracted
            need_space_for_single_req,
        )

        # Note: if the last prebuilt extend just finishes, and we enter `pop_preallocated` immediately in the next iteration
        #       the extend batch is not in any queue, so we need to explicitly add the tokens slots here
        if (
            self.scheduler.last_batch
            and self.scheduler.last_batch.forward_mode.is_prebuilt()
        ):
            allocatable_tokens -= self.num_reserved_decode_tokens * len(
                self.scheduler.last_batch.reqs
            )

        if count_retracted:
            allocatable_tokens -= sum(
                [
                    len(req.origin_input_ids)
                    + len(req.output_ids)
                    + self.num_reserved_decode_tokens
                    for req in self.retracted_queue
                ]
            )
        if self.scheduler.enable_decode_hicache:
            allocatable_tokens -= self._hicache_pending_restore_tokens()
        return allocatable_tokens

    def _required_alloc_tokens(self, fill_len: int, prefix_len: int) -> int:
        page_size = self.token_to_kv_pool_allocator.page_size
        if page_size == 1:
            return fill_len - prefix_len

        num_new_pages = get_num_new_pages(
            seq_lens=torch.tensor([fill_len], dtype=torch.int64),
            prefix_lens=torch.tensor([prefix_len], dtype=torch.int64),
            page_size=page_size,
        )
        return num_new_pages * page_size

    def _pre_alloc(
        self,
        req: Req,
        prefix_indices: Optional[torch.Tensor] = None,
        prefix_len: Optional[int] = None,
        total_prefix_len: Optional[int] = None,
        fail_on_oom: bool = False,
    ) -> Optional[torch.Tensor]:
        """Pre-allocate req_to_token and token_kv_pool for decode-side PD."""
        if prefix_len is None:
            prefix_len = 0
        if total_prefix_len is None:
            total_prefix_len = prefix_len

        device = self.token_to_kv_pool_allocator.device
        if prefix_len > 0:
            prefix_indices = prefix_indices.to(dtype=torch.int64, device=device)
        else:
            prefix_indices = torch.empty((0,), dtype=torch.int64, device=device)

        req_pool_indices = self.req_to_token_pool.alloc([req])

        if req_pool_indices is None:
            if fail_on_oom:
                return None
            assert (
                req_pool_indices is not None
            ), "req_pool_indices is full! There is a bug in memory estimation."

        # Alloc all tokens for the prebuilt req (except for the reserved input token for decoding)
        fill_len = len(req.origin_input_ids) + max(len(req.output_ids) - 1, 0)
        req.kv_allocated_len = fill_len
        req.kv_committed_len = fill_len

        if prefix_len > 0:
            self.req_to_token_pool.write(
                (req.req_pool_idx, slice(0, prefix_len)), prefix_indices
            )

        delta_len = fill_len - total_prefix_len
        required_alloc_tokens = self._required_alloc_tokens(
            fill_len=fill_len,
            prefix_len=total_prefix_len,
        )

        if (
            self.scheduler.server_args.disaggregation_decode_enable_radix_cache
            and self.token_to_kv_pool_allocator.available_size() < required_alloc_tokens
        ):
            num_to_evict = (
                required_alloc_tokens - self.token_to_kv_pool_allocator.available_size()
            )
            self.tree_cache.evict(EvictParams(num_tokens=num_to_evict))

        if self.token_to_kv_pool_allocator.page_size == 1:
            kv_loc = self.token_to_kv_pool_allocator.alloc(delta_len)
        else:
            last_loc = (
                prefix_indices[-1:].to(dtype=torch.int64, device=device)
                if prefix_len > 0
                else torch.tensor([-1], dtype=torch.int64, device=device)
            )
            kv_loc = self.token_to_kv_pool_allocator.alloc_extend(
                prefix_lens=torch.tensor(
                    [total_prefix_len], dtype=torch.int64, device=device
                ),
                prefix_lens_cpu=torch.tensor([total_prefix_len], dtype=torch.int64),
                seq_lens=torch.tensor([fill_len], dtype=torch.int64, device=device),
                seq_lens_cpu=torch.tensor([fill_len], dtype=torch.int64),
                last_loc=last_loc,
                extend_num_tokens=delta_len,
            )

        if kv_loc is None:
            self.req_to_token_pool.free(req)
            req.kv_allocated_len = 0
            req.kv_committed_len = 0
            if fail_on_oom:
                return None
            assert (
                kv_loc is not None
            ), "KV cache is full! There is a bug in memory estimation."

        self.req_to_token_pool.write(
            (
                req.req_pool_idx,
                slice(total_prefix_len, total_prefix_len + len(kv_loc)),
            ),
            kv_loc,
        )

        # populate metadata
        req.fill_ids = req.origin_input_ids + req.output_ids
        req.fill_len = req.kv_committed_len
        req.prefix_indices = prefix_indices
        req.set_extend_input_len(req.fill_len - total_prefix_len)

        return kv_loc


class DecodeTransferQueue(DecodeHiCacheTransferMixin):
    """
    Store the requests that is polling kv
    """

    def __init__(
        self,
        gloo_group: ProcessGroup,
        req_to_metadata_buffer_idx_allocator: ReqToMetadataIdxAllocator,
        tp_rank: int,
        metadata_buffers: MetadataBuffers,
        scheduler: Scheduler,
        tree_cache: BasePrefixCache,
    ):
        self.queue: List[DecodeRequest] = []
        self.gloo_group = gloo_group
        self.req_to_metadata_buffer_idx_allocator = req_to_metadata_buffer_idx_allocator
        self.tp_rank = tp_rank
        self.metadata_buffers = metadata_buffers
        self.scheduler = scheduler
        self.tree_cache = tree_cache
        self.spec_algorithm = scheduler.spec_algorithm

    def add(self, decode_req: DecodeRequest) -> None:
        self.queue.append(decode_req)

    def extend(self, decode_reqs: List[DecodeRequest]) -> None:
        self.queue.extend(decode_reqs)

    def _commit_transfer_to_req(self, decode_req: DecodeRequest) -> bool:
        """
        Returns:
            True if the request should be removed from the queue (success or corruption)
            False if metadata not ready yet (keep in queue for next poll)
        """
        idx = decode_req.metadata_buffer_index
        (
            output_id,
            cached_tokens,
            output_token_logprobs_val,
            output_token_logprobs_idx,
            output_top_logprobs_val,
            output_top_logprobs_idx,
            output_topk_p,
            output_topk_index,
            output_hidden_states,
            output_bootstrap_room,
        ) = self.metadata_buffers.get_buf(idx)

        # Validate bootstrap_room to detect context corruption
        actual_room = output_bootstrap_room[0].item()
        expected_room = (
            decode_req.req.bootstrap_room
            if decode_req.req.bootstrap_room is not None
            else 0
        )

        if _is_fake_transfer(decode_req.req, self.scheduler.server_args):
            pass
        elif actual_room == 0:
            # Case 1: Metadata not ready yet (actual_room == 0)
            # Keep request in queue and wait for next poll
            return False
        elif actual_room != expected_room:
            # Case 2: Real corruption detected (mismatch)
            # Abort the request and remove from the queue
            error_msg = (
                f"Context corruption detected: Request {decode_req.req.rid} "
                f"(bootstrap_room={expected_room}) received metadata from "
                f"bootstrap_room={actual_room}. "
                f"Metadata buffer index: {idx}. "
                f"This indicates metadata buffer index collision."
            )
            logger.error(error_msg)
            prepare_abort(
                decode_req.req,
                "Metadata corruption detected - bootstrap_room mismatch",
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            decode_req.kv_receiver.clear()
            decode_req.kv_receiver = None
            return True

        self._commit_hicache_local_restore_to_req(decode_req)

        # Case 3: Success - commit the transfer
        decode_req.req.output_ids.append(output_id[0].item())
        decode_req.req.cached_tokens = cached_tokens[0].item()
        decode_req.req.already_computed = decode_req.req.cached_tokens
        decode_req.req.cached_tokens_device = cached_tokens[1].item()
        decode_req.req.cached_tokens_host = cached_tokens[2].item()
        decode_req.req.cached_tokens_storage = cached_tokens[3].item()
        decode_req.req.mm_image_tokens = cached_tokens[4].item()
        decode_req.req.mm_audio_tokens = cached_tokens[5].item()
        decode_req.req.mm_video_tokens = cached_tokens[6].item()
        if not self.spec_algorithm.is_none():
            decode_req.req.output_topk_p = output_topk_p
            decode_req.req.output_topk_index = output_topk_index
            decode_req.req.hidden_states_tensor = output_hidden_states

        if decode_req.req.return_logprob:
            decode_req.req.output_token_logprobs_val.append(
                output_token_logprobs_val[0].item()
            )
            decode_req.req.output_token_logprobs_idx.append(
                output_token_logprobs_idx[0].item()
            )
            decode_req.req.output_top_logprobs_val.append(
                output_top_logprobs_val[: decode_req.req.top_logprobs_num].tolist()
            )
            decode_req.req.output_top_logprobs_idx.append(
                output_top_logprobs_idx[: decode_req.req.top_logprobs_num].tolist()
            )

        decode_req.kv_receiver.clear()
        decode_req.kv_receiver = None
        decode_req.req.time_stats.set_wait_queue_entry_time()
        return True

    def _poll_with_metadata_gate(self) -> List[int]:
        pollers = (
            [HiCacheRestoreGatedKVReceiver(dr) for dr in self.queue]
            if self.scheduler.enable_decode_hicache
            else [dr.kv_receiver for dr in self.queue]
        )
        return poll_and_all_reduce(
            pollers,
            self.gloo_group,
            decode_reqs=self.queue,
            metadata_buffers=self.metadata_buffers,
            server_args=self.scheduler.server_args,
        )

    def pop_transferred(self, rids_to_check: Optional[List[str]] = None) -> List[Req]:
        if not self.queue:
            return []
        if self.scheduler.enable_decode_hicache:
            self._process_hicache_local_restores(
                [
                    decode_req
                    for decode_req in self.queue
                    if rids_to_check is None or decode_req.req.rid in rids_to_check
                ]
            )

        polls = self._poll_with_metadata_gate()

        transferred_reqs = []
        indices_to_remove = set()
        for i, (decode_req, poll) in enumerate(zip(self.queue, polls)):
            if rids_to_check is not None and decode_req.req.rid not in rids_to_check:
                continue
            hicache_restore_status = decode_req.hicache_restore_status
            if (
                poll == KVPoll.Failed
                or hicache_restore_status == HiCacheRestoreResult.FAILED
            ):
                error_message = f"Decode transfer failed for request rank={self.tp_rank} {decode_req.req.rid=} {decode_req.req.bootstrap_room=}"
                if poll == KVPoll.Failed:
                    try:
                        decode_req.kv_receiver.failure_exception()
                    except Exception as e:
                        error_message += f" with exception {e}"
                self._clean_hicache_prefetch_resources(decode_req)
                logger.error(error_message)
                prepare_abort(
                    decode_req.req,
                    error_message,
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
                self.scheduler.stream_output(
                    [decode_req.req], decode_req.req.return_logprob
                )
                # release pre-allocated kv cache, but don't insert into the tree since it's failed
                release_kv_cache(decode_req.req, self.tree_cache, is_insert=False)
                decode_req.kv_receiver.clear()
                decode_req.kv_receiver = None
                indices_to_remove.add(i)
                if self.scheduler.enable_metrics:
                    self.scheduler.metrics_collector.increment_transfer_failed_reqs()
                continue
            elif poll == KVPoll.Success:
                if (
                    self.scheduler.enable_decode_hicache
                    and hicache_restore_status == HiCacheRestoreResult.PENDING
                ):
                    continue
                should_remove = self._commit_transfer_to_req(decode_req)
                if should_remove:
                    indices_to_remove.add(i)
                    # Check if request was aborted due to corruption
                    if isinstance(decode_req.req.finished_reason, FINISH_ABORT):
                        self.scheduler.stream_output(
                            [decode_req.req], decode_req.req.return_logprob
                        )
                        self._clean_hicache_prefetch_resources(decode_req)
                        release_kv_cache(
                            decode_req.req, self.tree_cache, is_insert=False
                        )
                        if self.scheduler.enable_metrics:
                            self.scheduler.metrics_collector.increment_transfer_failed_reqs()
                    else:
                        transferred_reqs.append(decode_req.req)
            elif poll in [
                KVPoll.Bootstrapping,
                KVPoll.WaitingForInput,
                KVPoll.Transferring,
            ]:
                pass
            else:
                raise ValueError(f"Unexpected poll case: {poll}")

        for i in indices_to_remove:
            idx = self.queue[i].metadata_buffer_index
            assert idx != -1
            self.metadata_buffers.bootstrap_room[idx] = 0
            self.req_to_metadata_buffer_idx_allocator.free(idx)

        self.queue = [
            entry for i, entry in enumerate(self.queue) if i not in indices_to_remove
        ]

        return transferred_reqs


class SchedulerDisaggregationDecodeMixin:

    @torch.no_grad()
    def event_loop_normal_disagg_decode(self: Scheduler):
        """A normal scheduler loop for decode worker in disaggregation mode."""

        while True:
            # Receive requests
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            # polling and allocating kv cache
            self.process_decode_queue()

            # Get the next batch to run
            batch = self.get_next_disagg_decode_batch_to_run()
            self.cur_batch = batch

            # Launch the current batch
            if batch:
                result = self.run_batch(batch)
                self.process_batch_result(batch, result)
            else:
                # When the server is idle, do self-check and re-init some states
                self.self_check_during_idle()

            # Update last_batch
            self.last_batch = batch

    @torch.no_grad()
    def event_loop_overlap_disagg_decode(self: Scheduler):
        self.result_queue = deque()
        self.last_batch: Optional[ScheduleBatch] = None

        while True:
            # Receive requests
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            # polling and allocating kv cache
            self.process_decode_queue()

            # Get the next batch to run
            batch = self.get_next_disagg_decode_batch_to_run()
            self.cur_batch = batch

            # Launch the current batch
            if batch:
                batch_result = self.run_batch(batch)
                self.result_queue.append((batch.copy(), batch_result))
            else:
                batch_result = None

            # Process the last batch
            if self.last_batch:
                tmp_batch, tmp_result = self.result_queue.popleft()
                self.process_batch_result(tmp_batch, tmp_result)
            elif batch is None:
                self.self_check_during_idle()

            # Run sample of the current batch
            # It depends on the result of the last batch (e.g., grammar), so we run it after the last batch is processed.
            self.launch_batch_sample_if_needed(batch_result)

            # Update last_batch
            self.last_batch = batch

    def _run_batch_prebuilt(
        self: Scheduler, batch: ScheduleBatch
    ) -> GenerationBatchResult:
        if batch.inner_idle_batch is not None:
            idle_batch = batch.inner_idle_batch
            # Reset the inner idle batch to avoid reusing it.
            batch.inner_idle_batch = None
            return self.run_batch(idle_batch)

        return GenerationBatchResult()

    def get_next_disagg_decode_batch_to_run(
        self: Scheduler,
    ) -> Optional[ScheduleBatch]:
        """Process prebuilt batch and schedule the next decode batch."""
        # Process pending prebuilt batch: output processing + filter + merge
        new_prebuilt_batch = self.get_new_prebuilt_batch()
        if new_prebuilt_batch:
            assert self.chunked_req is None
            self.process_batch_result_prebuilt(new_prebuilt_batch)
            new_prebuilt_batch.filter_batch()
            if not new_prebuilt_batch.is_empty():
                if self.running_batch.is_empty():
                    self.running_batch = new_prebuilt_batch
                else:
                    self.running_batch.merge_batch(new_prebuilt_batch)

        # Schedule decode batch
        if self.running_batch.is_empty():
            ret = None
        else:
            self.running_batch = self.update_running_batch(self.running_batch)
            ret = self.running_batch if not self.running_batch.is_empty() else None

        ret = self.maybe_prepare_mlp_sync_batch(ret)
        if ret:
            set_schedule_time_batch(ret)
        return ret

    def get_new_prebuilt_batch(self: Scheduler) -> Optional[ScheduleBatch]:
        """Create a schedulebatch for fake completed prefill"""
        if self.grammar_manager.has_waiting_grammars():
            ready_grammar_requests = self.grammar_manager.get_ready_grammar_requests()
            for req in ready_grammar_requests:
                self._add_request_to_queue(req)

        if len(self.waiting_queue) == 0:
            return None

        curr_batch_size = self.running_batch.batch_size()

        batch_size = min(self.req_to_token_pool.size, self.max_running_requests)

        num_not_used_batch = batch_size - curr_batch_size

        # pop req from waiting queue
        can_run_list: List[Req] = []
        waiting_queue: List[Req] = []

        for i in range(len(self.waiting_queue)):
            req = self.waiting_queue[i]
            # we can only add at least `num_not_used_batch` new batch to the running queue
            if i < num_not_used_batch:
                can_run_list.append(req)
                req.init_next_round_input(self.tree_cache)
            else:
                waiting_queue.append(req)

        self.waiting_queue = waiting_queue
        if len(can_run_list) == 0:
            return None

        set_time_batch(can_run_list, "set_forward_entry_time")

        # construct a schedule batch with those requests and mark as decode
        new_batch = ScheduleBatch.init_new(
            can_run_list,
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.spec_algorithm,
        )

        # construct fake completed prefill
        new_batch.prepare_for_prebuilt()
        new_batch.process_prebuilt(self.server_args, self.future_map)

        return new_batch

    def process_decode_queue(self: Scheduler):
        if self.enable_hierarchical_cache or self.enable_kv_connector:
            self.tree_cache.check_kv_events()

        if self.server_args.disaggregation_decode_enable_offload_kvcache:
            self.decode_offload_manager.check_offload_progress()
        if getattr(self, "decode_flexkv_offload_manager", None) is not None:
            self.decode_flexkv_offload_manager.check_offload_progress()

        # try to resume retracted requests if there are enough space for another `num_reserved_decode_tokens` decode steps
        resumed_reqs = self.disagg_decode_prealloc_queue.resume_retracted_reqs()
        self.waiting_queue.extend(resumed_reqs)
        if len(self.disagg_decode_prealloc_queue.retracted_queue) > 0:
            # if there are still retracted requests, we do not allocate new requests
            return

        if not hasattr(self, "polling_count"):
            self.polling_count = 0
            self.polling_interval = (
                self.server_args.disaggregation_decode_polling_interval
            )

        self.polling_count = (self.polling_count + 1) % self.polling_interval

        if self.polling_count % self.polling_interval == 0:
            req_conns, _ = self.disagg_decode_prealloc_queue.pop_preallocated()
            self.disagg_decode_transfer_queue.extend(req_conns)
            transferred_reqs = (
                self.disagg_decode_transfer_queue.pop_transferred()
            )  # the requests which kv has arrived
            self.waiting_queue.extend(transferred_reqs)
