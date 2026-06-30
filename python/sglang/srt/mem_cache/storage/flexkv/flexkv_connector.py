import ctypes
import logging
import os
import socket
import struct
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.mem_cache.kv_connector import BaseKVConnector, LoadOperation
from sglang.srt.mem_cache.storage.flexkv.flexkv_comm import (
    CMD_PUT_META,
    CMD_LAYERWISE,
    CMD_STORE_COMPLETE,
    FlexKVLayerLoadingEvent,
    FlexKVLayerDoneCounter,
    FlexKVComm,
    send_fds,
)

try:
    from flexkv.common.config import LayerGroupSpec, recompute_cache_block_counts
    from flexkv.common.request import KVResponseStatus
    from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
    from flexkv.integration.config import FlexKVConfig
    from flexkv.kvmanager import KVManager
    from flexkv.server.client import KVTPClient
    from flexkv.transfer.layerwise import build_layerwise_eventfd_socket_path
    from flexkv.transfer_manager import TransferManagerOnRemote
except ImportError as e:
    raise RuntimeError("FlexKV is not installed. Please install it.") from e

logger = logging.getLogger(__name__)

# ---- CUDA Runtime (via ctypes) ----
def load_cudart():
    candidates = [
        "libcudart.so",
        "libcudart.so.12",
        "libcudart.so.11.0",
        "/usr/local/cuda/lib64/libcudart.so",
    ]
    for lib in candidates:
        try:
            return ctypes.CDLL(lib)
        except OSError:
            continue
    return None


cudart = load_cudart()

if cudart:
    cudart.cudaLaunchHostFunc.argtypes = [
        ctypes.c_void_p,
        ctypes.CFUNCTYPE(None, ctypes.c_void_p),
        ctypes.c_void_p,
    ]
    cudart.cudaLaunchHostFunc.restype = ctypes.c_int


# ---- FlexKV Connector ----


class FlexKVConnector(BaseKVConnector):
    """KV cache connector backed by FlexKV's distributed cache system.

    Implements ``BaseKVConnector`` so it can be used with
    ``ExtendedRadixCache`` via ``--kv-connector-cls``.
    """

    def __init__(
        self,
        params: Any,
        server_args: Any,
        tp_rank: int = 0,
        dp_rank: Optional[int] = 0,
        attn_cp_rank: Optional[int] = 0,
        pp_group: Any = None,
        attn_tp_group: Any = None,
        attn_cp_group: Any = None,
    ):
        super().__init__(
            params=params,
            server_args=server_args,
            tp_rank=tp_rank,
            dp_rank=dp_rank,
            attn_cp_rank=attn_cp_rank,
            pp_group=pp_group,
            attn_tp_group=attn_tp_group,
            attn_cp_group=attn_cp_group,
        )

        # ---- Primitive variables (from params / constructor) ----
        self.page_size = params.page_size
        kvcache = params.token_to_kv_pool_allocator.get_kvcache()

        sglang_model_config = ModelConfig.from_server_args(server_args)

        # ---- Initialize FlexKV config ----
        self.flexkv_config = FlexKVConfig.from_env()
        rank_info = self.flexkv_config.post_init_from_sglang_config(
            sglang_config=sglang_model_config,
            server_args=server_args,
            page_size=self.page_size,
            tp_rank=tp_rank,
            pp_rank=params.pp_rank,
            dp_rank=dp_rank,
            attn_cp_rank=attn_cp_rank,
        )

        model_config = self.flexkv_config.model_config
        cache_config = self.flexkv_config.cache_config
        self.rank_info = rank_info

        # Structured logging label
        self._rank_label = f" [model_config={model_config}, rank_info={rank_info}]"

        # ---- Communication / sync context ----
        self._sync_ctx = FlexKVComm(
            rank_info=rank_info,
            world_rank=(
                torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            ),
            pp_group=pp_group,
            attn_tp_group=attn_tp_group,
            attn_cp_group=attn_cp_group,
        )
        logger.debug(
            f"[FlexKV] sync_context{self._rank_label}: "
            f"is_sync_leader={self._sync_ctx.is_sync_leader}, "
            f"needs_sync={self._sync_ctx.needs_sync}, "
            f"is_pp_active={self._sync_ctx.is_pp_active}"
        )


        # ---- Align block counts on unified group (single all_reduce MIN) ----
        for _attr in ("num_cpu_blocks", "num_ssd_blocks", "num_remote_blocks"):
            _orig = getattr(self.flexkv_config.cache_config, _attr)
            if _orig is None or _orig <= 0:
                continue
            _aligned = self._sync_ctx.all_reduce_min(_orig)
            logger.debug(
                f"[FlexKV] Block count alignment{self._rank_label}: "
                f"attr={_attr}, {_orig} -> {_aligned}"
            )
            if _aligned != _orig:
                logger.info(
                    f"[FlexKV] Block count MIN alignment '{_attr}': "
                    f"{_orig} -> {_aligned}"
                )
            setattr(self.flexkv_config.cache_config, _attr, _aligned)

        if model_config.nnodes > 1:
            logger.info(
                f"[FlexKV] Multi-node detected{self._rank_label}: "
                f"model_config={model_config}, rank_info={rank_info}"
            )

        # Build unified kv_caches list (MLA vs MHA vs DSv4 multi-pool)
        # ``self._is_dsv4`` and ``self._dsv4_layer_groups_info`` are populated only
        # when a ``DeepSeekV4TokenToKVPool`` is detected (its sub-pools each store
        # KV in 2D packed buffers ``[num_pages, bytes_per_page_padded]`` with a
        # different compression ratio per group: c4 (CSA) at 4x, c128 (HCA) at
        # 128x, and an optional c4 indexer at 4x).  The SWA sub-pool is handled
        # separately further down (matches existing SWA path).
        self._is_dsv4 = False
        self._dsv4_layer_groups_info: List[Dict[str, Any]] = []
        self._dsv4_kvcache = None

        indexer_buffers = getattr(kvcache, "index_k_with_scale_buffer", None)
        if indexer_buffers is not None and len(indexer_buffers) > 0:
            logger.info(
                f"[FlexKV] Detected sparse attention indexer cache with "
                f"{len(indexer_buffers)} indexer layers, "
                f"shape={indexer_buffers[0].shape}"
            )

        if hasattr(kvcache, "c4_kv_pool"):
            # ---- DeepSeek V4 multi-pool path ----
            # DSv4 splits the KV cache across several sub-pools, each with its
            # own page_size and compression ratio.  We register each sub-pool
            # as its own ``LayerGroupSpec`` (FlexKV multi-group registration).
            # The control plane (slot_mapping in full-pool token index space)
            # is handed to FlexKV unchanged; FlexKV's per-group ``compress_ratio``
            # in ``LayerGroupSpec`` is what derives sub-mappings during transfer.
            self._is_dsv4 = True
            self._dsv4_kvcache = kvcache

            # PP support: ``compression_ratios`` is the FULL model's; the
            # sub-pool ``kv_buffer`` lists are sized to the current PP stage
            # only.  Compute layer ids within the stage range so they match
            # the buffer count, but emit them as absolute layer ids (FlexKV's
            # LayerGroupSpec invariant: ``layer_indices ⊂ [0, num_layers)``
            # where ``num_layers`` is the full-model count).
            compression_ratios = kvcache.compression_ratios
            stage_start = getattr(kvcache, "_stage_start", 0)
            stage_end = getattr(kvcache, "_stage_end", len(compression_ratios))
            c4_layer_ids = [
                i for i in range(stage_start, stage_end)
                if compression_ratios[i] == 4
            ]
            c128_layer_ids = [
                i for i in range(stage_start, stage_end)
                if compression_ratios[i] == 128
            ]

            # Each entry: dict with keys
            #   name: human label (logging)
            #   ratio: compress_ratio (4 / 128)
            #   layer_ids: original-layer ids belonging to this group
            #   buffers: list of 2D GPU tensors [num_pages, bytes_per_page_padded]
            #   bytes_per_token: packed bytes per token in this sub-pool
            #   sub_page_size: per-token slots per page IN THE SUB-POOL
            #                  (= full-pool page_size // compress_ratio)
            self._dsv4_layer_groups_info = []
            self._dsv4_layer_groups_info.append({
                "name": "c4",
                "ratio": 4,
                "layer_ids": c4_layer_ids,
                "buffers": kvcache.c4_kv_pool.kv_buffer,
                "bytes_per_token": kvcache.c4_kv_pool.get_bytes_per_token(),
                "sub_page_size": kvcache.c4_kv_pool.page_size,
                "dtype": torch.uint8,
            })
            self._dsv4_layer_groups_info.append({
                "name": "c128",
                "ratio": 128,
                "layer_ids": c128_layer_ids,
                "buffers": kvcache.c128_kv_pool.kv_buffer,
                "bytes_per_token": kvcache.c128_kv_pool.get_bytes_per_token(),
                "sub_page_size": kvcache.c128_kv_pool.page_size,
                "dtype": torch.uint8,
            })

            # c4_indexer_kv_pool is optional (older DSv4 configs may not have
            # the indexer split).  It uses the same compress_ratio=4 layer set
            # as c4_kv_pool but stores indexer K (uint8 packed) instead of KV.
            if hasattr(kvcache, "c4_indexer_kv_pool"):
                indexer_pool = kvcache.c4_indexer_kv_pool
                indexer_buffers_pool = getattr(
                    indexer_pool, "index_k_with_scale_buffer", None
                )
                if indexer_buffers_pool and len(indexer_buffers_pool) > 0:
                    # Indexer page bytes = page_size * head_dim
                    #                    + page_size * (head_dim/quant_block) * 4
                    # Compute per-token bytes from buffer shape so we don't
                    # duplicate the formula in DeepSeekV4IndexerPool.
                    sample = indexer_buffers_pool[0]
                    bytes_per_page = sample.shape[1]
                    bytes_per_token_idx = bytes_per_page // indexer_pool.page_size
                    self._dsv4_layer_groups_info.append({
                        "name": "c4_indexer",
                        "ratio": 4,
                        "layer_ids": list(c4_layer_ids),
                        "buffers": indexer_buffers_pool,
                        "bytes_per_token": bytes_per_token_idx,
                        "sub_page_size": indexer_pool.page_size,
                        "dtype": torch.uint8,
                    })

            # Build the flat ``kv_caches`` list expected by the registration
            # path (concatenation of all non-empty group buffers; ordering
            # must match ``handles_per_group`` at registration time).  Empty
            # groups (e.g. a PP stage with no c4 layers) are skipped.
            self._dsv4_layer_groups_info = [
                gi for gi in self._dsv4_layer_groups_info if gi["buffers"]
            ]
            if not self._dsv4_layer_groups_info:
                raise RuntimeError(
                    "[FlexKV] DSv4 detected but no non-empty sub-pool groups "
                    "found on this PP stage. Check stage_start/stage_end and "
                    "compression_ratios."
                )
            kv_caches = []
            for gi in self._dsv4_layer_groups_info:
                kv_caches.extend(gi["buffers"])

            # DSv4 uses a separate SWA sub-pool with its own page_size; the
            # existing SWA path (below, see ``self._swa_kv_pool``) handles it.
            # Suppress the legacy NSA-indexer code path: DSv4's indexer is
            # registered via a layer_group instead.
            indexer_buffers = None

            logger.info(
                f"[FlexKV] Detected DeepSeekV4TokenToKVPool: "
                f"groups={[(g['name'], g['ratio'], len(g['layer_ids'])) for g in self._dsv4_layer_groups_info]}, "
                f"total_layers={len(compression_ratios)}, "
                f"page_size_full={self.page_size}, "
                f"swa_pool={'present' if hasattr(kvcache, 'swa_kv_pool') else 'absent'}"
            )
        elif hasattr(kvcache, "kv_buffer"):
            # MLA: K and V share the same buffer, register once per layer
            kv_caches = kvcache.kv_buffer
        elif hasattr(kvcache, "k_buffer"):
            # MHA: separate K and V buffers, concat as [k_layers..., v_layers...]
            kv_caches = kvcache.k_buffer + kvcache.v_buffer
        else:
            # Other multi-pool layouts are not yet supported by FlexKV's
            # transfer engine.
            raise NotImplementedError(
                f"FlexKV does not yet support KV cache type {type(kvcache).__name__}. "
                f"Supported: 'kv_buffer' (MLA/NSA), 'k_buffer'/'v_buffer' (MHA), "
                f"or 'c4_kv_pool' (DeepSeek V4 multi-pool)."
            )

        # ---- Node B: Launch TransferManagerOnRemote ----
        self._remote_process = None
        if model_config.nnodes > 1 and rank_info.node_rank > 0 and rank_info.local_rank == 0:
            logger.debug(
                f"[FlexKV] Launching TransferManagerOnRemote{self._rank_label}: "
                f"master_host={self.flexkv_config.model_config.master_host}, "
                f"master_ports={self.flexkv_config.model_config.master_ports}")
            self._remote_process = TransferManagerOnRemote.create_process(
                master_host=self.flexkv_config.model_config.master_host,
                master_ports=self.flexkv_config.model_config.master_ports,
            )
            logger.info(
                f"[FlexKV] Launched TransferManagerOnRemote on node_rank={rank_info.node_rank}"
                f"{self._rank_label}"
            )

        # Size CPU/SSD pools from real layer_groups before CacheEngine / Transfer
        # subprocess start.  DSv4 and NSA indexer layouts are known here from
        # GPU buffers; without this step mempool uses the uniform estimate
        # (too many logical blocks vs the physical StorageEngine buffer).
        self._apply_layer_groups_for_cache_sizing(kv_caches, indexer_buffers)

        # ---- SWA (Sliding Window Attention) GPU pool detection ----
        # MUST run BEFORE KVManager is constructed: in server_client_mode the
        # cache_config is pickled into the KVServer subprocess at KVManager()
        # time, so cache_config.swa has to be populated first or the server's
        # GlobalCacheEngine builds no swa_index and get_match_swa's RPC returns
        # an empty return_mask_swa. (The auto-derive below is a fallback for when
        # upstream config didn't set cache_config.swa; with the old ordering it
        # fired after KVManager and never reached the engine — in-process too.)
        self._kvcache = kvcache  # Store full kvcache for translate_loc_from_full_to_swa
        self._swa_kv_pool = getattr(kvcache, 'swa_kv_pool', None)

        # Auto-derive SWA config: if a SWA GPU pool is present but
        # cache_config.swa is None / disabled, build a sensible default so
        # swa_available() / swa_put() / swa_get() can actually work.
        # Without this, swa_available() always returns False and the
        # connector silently disables H2D restoration entirely.
        if self._swa_kv_pool is not None and (
            cache_config.swa is None or not cache_config.swa.enabled
        ):
            try:
                from flexkv.common.config import SWAPoolConfig
                # Derive params from the GPU pool itself:
                #   - window_size from kvcache (DSv4 swa_window_size = 128)
                #   - num_swa_layers from len(swa_kv_pool.kv_buffer)
                #   - bytes_per_token from get_bytes_per_token() (= 584)
                derived_window = (
                    getattr(kvcache, 'swa_window_size', None)
                    or getattr(self._swa_kv_pool, 'window_size', None)
                    or getattr(self._swa_kv_pool, 'page_size', None)
                    or 128
                )
                derived_layers = len(getattr(self._swa_kv_pool, 'kv_buffer', []))
                derived_bpt = (
                    self._swa_kv_pool.get_bytes_per_token()
                    if hasattr(self._swa_kv_pool, 'get_bytes_per_token') else 584
                )
                cache_config.swa = SWAPoolConfig(
                    enabled=True,
                    num_slots=1024,
                    window_size=derived_window,
                    num_swa_layers=derived_layers or 61,
                    bytes_per_token_per_layer=derived_bpt,
                    evict_ratio=0.1,
                    pin_memory=True,
                )
                logger.info(
                    f"[FlexKV-SWA] Auto-derived SWAPoolConfig: "
                    f"window_size={derived_window}, num_layers={derived_layers}, "
                    f"bytes_per_token={derived_bpt}"
                )
            except Exception as e:
                logger.warning(
                    f"[FlexKV-SWA] Failed to auto-derive SWAPoolConfig: {e}. "
                    f"H2D restoration will be disabled."
                )

        if self._sync_ctx.is_sync_leader:
            self.kv_manager = KVManager(
                model_config=model_config,
                cache_config=cache_config,
                dp_client_id=rank_info.dp_client_id,
                server_recv_port=self.flexkv_config.server_recv_port,
                gpu_register_port=self.flexkv_config.gpu_register_port,
            )
            self.kv_manager.start()
            logger.info(
                f"[FlexKV] Creating KVManager{self._rank_label}: "
                f"server_recv_port={self.flexkv_config.server_recv_port}, "
                f"gpu_register_port={self.flexkv_config.gpu_register_port}")

        # ---- GPU Registration Routing ----
        self.dp_client_id = rank_info.dp_client_id
        self.pp_rank = rank_info.pp_rank
        self.tp_client = KVTPClient(
            self.flexkv_config.gpu_register_port,
            dp_client_id=self.dp_client_id,
            pp_rank=self.pp_rank,
            device_id=rank_info.local_rank,
        )

        # ---- MTP piggyback: attach draft pool BEFORE registration ----
        # FlexKV's TransferManager rejects re-registering a known device_id
        # (see _handle_gpu_blocks_registration: "first registration wins"),
        # so the draft pool MUST be folded into the layer_groups list before
        # we call _register_with_retry() below. The method extends
        # self._dsv4_layer_groups_info in-place and returns the additional
        # GPU buffers to append to kv_caches; on any unsupported case it
        # returns [] and logs a warning, leaving kv_caches unchanged.
        #
        # NOTE: ``self._draft_kv_pool`` is also assigned later in __init__
        # (in the SWA detection block) for code-clarity / locality with the
        # SWA pool fields. We pre-initialize it here so that
        # ``_attach_draft_pool`` can read it without an AttributeError.
        # The later assignment writes the same value (idempotent).
        self._draft_kv_pool = getattr(params, 'draft_token_to_kv_pool', None)
        self._draft_swa_layer_group = None  # populated by _attach_draft_pool()
        kv_caches = list(kv_caches) + self._attach_draft_pool()

        # ---- GPU Registration (with retry) ----
        self._register_with_retry(kv_caches, indexer_buffers)
        logger.info(
            f"[FlexKV] KVTPClient registered to server{self._rank_label}: "
            f"gpu_register_port={self.flexkv_config.gpu_register_port}")

        self.enable_layerwise_transfer = bool(
            int(os.getenv("FLEXKV_ENABLE_LAYERWISE_TRANSFER", "0"))
        )

        self.layerwise_eventfd_socket = build_layerwise_eventfd_socket_path(
            dp_client_id=self.dp_client_id,
            pp_rank=self.pp_rank,
            model_config=model_config,
        )
        logger.info(
            f"[FlexKV] Eventfd socket path configured{self._rank_label}: "
            f"socket={self.layerwise_eventfd_socket}, "
            f"layerwise_transfer={self.enable_layerwise_transfer}")
        self.layerwise_eventfd_connect_max_retries = max(
            360,
            int(os.getenv("FLEXKV_LAYERWISE_EVENTFD_CONNECT_MAX_RETRIES", "0")),
        )
        self._layer_done_counter: Optional[FlexKVLayerDoneCounter] = None
        self._worker_connected = False

        self._init_layer_transfer_components()

        if self._layer_done_counter is not None and kvcache is not None:
            kvcache.register_layer_transfer_counter(self._layer_done_counter)

        # rid -> flexkv_task_id (pending loads awaiting start_load_kv)
        self._pending_loads: Dict[str, int] = {}
        # ext_task_id -> producer_id (layerwise loads in flight)
        self._ongoing_loads: Dict[int, int] = {}
        # ext_task_ids whose load has completed
        self._completed_loads: List[int] = []
        # ext_task_id -> flexkv_task_id (stores in flight)
        self._ongoing_stores: Dict[int, int] = {}
        # ext_task_ids whose store has completed or was skipped
        self._completed_stores: List[int] = []
        # flexkv task ids for periodic drain to prevent pipe deadlock
        self._load_fkv_tids: List[int] = []
        # rid -> flexkv_task_id (prefetch in flight)
        self._ongoing_prefetches: Dict[str, int] = {}
        self._prefetch_enabled = bool(
            cache_config.enable_ssd
            or cache_config.enable_remote
            or cache_config.enable_kv_sharing
        )

        # ---- SWA GPU pool fields derived from cache_config.swa ----
        # NOTE: SWA pool detection and cache_config.swa auto-derive already ran
        # ABOVE, before KVManager construction (so server_client_mode pickles the
        # SWA config into the KVServer subprocess). self._kvcache / self._swa_kv_pool
        # are already set there. Here we only read back the (possibly auto-derived)
        # cache_config.swa into the connector's convenience fields.
        self._swa_window_size = (
            cache_config.swa.window_size
            if cache_config.swa is not None and cache_config.swa.enabled
            else 0
        )
        self._swa_bytes_per_token_per_layer = (
            cache_config.swa.bytes_per_token_per_layer
            if cache_config.swa is not None and hasattr(cache_config.swa, 'bytes_per_token_per_layer')
            else 0
        )
        self._device = kv_caches[0].device if kv_caches else torch.device("cuda")
        if self._swa_kv_pool is not None:
            logger.info(
                f"[FlexKV-SWA] Detected SWA KV pool on kvcache, "
                f"window_size={self._swa_window_size}, "
                f"bytes_per_token_per_layer={self._swa_bytes_per_token_per_layer}, "
                f"swa_enabled_in_cache_config={cache_config.swa is not None and cache_config.swa.enabled}"
            )

        if self._sync_ctx.is_sync_leader:
            wait_count = 0
            while not self.kv_manager.is_ready():
                time.sleep(10)
                wait_count += 1
                # Collect diagnostic info for debugging
                diag_parts = []
                # Check IPC socket file existence
                gpu_port = self.flexkv_config.gpu_register_port
                if gpu_port.startswith("ipc://"):
                    ipc_path = gpu_port[len("ipc://"):]
                    ipc_exists = os.path.exists(ipc_path)
                    diag_parts.append(f"ipc_socket={ipc_path} exists={ipc_exists}")
                # Check TransferManager subprocess status
                task_engine = getattr(self.kv_manager, 'kv_task_engine', None)
                if task_engine is not None:
                    for i, th in enumerate(getattr(task_engine, 'transfer_handles', [])):
                        handle = getattr(th, '_handle', None)
                        if handle is not None:
                            parts = []
                            start_evt = getattr(handle, 'start_event', None)
                            ready_evt = getattr(handle, 'ready_event', None)
                            proc = getattr(handle, 'process', None)
                            if start_evt is not None:
                                parts.append(f"started={start_evt.is_set()}")
                            if ready_evt is not None:
                                parts.append(f"ready={ready_evt.is_set()}")
                            if proc is not None:
                                parts.append(f"alive={proc.is_alive()}")
                            if parts:
                                diag_parts.append(f"transfer_handle[{i}]: {', '.join(parts)}")
                diag_str = "; ".join(diag_parts) if diag_parts else "no diagnostics available"
                logger.info(
                    f"[FlexKV] Waiting for FlexKV to be ready{self._rank_label}... "
                    f"(waited {wait_count * 10}s, {diag_str})"
                )
            logger.info(f"[FlexKV] FlexKV is ready{self._rank_label}")
        elif model_config.nnodes > 1 and rank_info.node_rank > 0:
            # Node B: no KVManager to wait for, GPU registration retry handles readiness
            logger.info(f"[FlexKV] Node B skipping is_ready wait{self._rank_label}")

        logger.info(
            f"[FlexKV] Connector initialized{self._rank_label}: "
            f"layerwise_transfer={self.enable_layerwise_transfer}, "
            f"prefetch_enabled={self._prefetch_enabled}, "
            f"model_config={model_config}, rank_info={rank_info}"
        )

    # ---- BaseKVConnector abstract methods ----

    def get_new_hit_length(
        self,
        token_ids: List[int],
        token_mask: torch.Tensor,
        update_state_for_load: bool = False,
        rid: Optional[str] = None,
        device_indices: Optional[torch.Tensor] = None,
    ) -> int:
        hit_length = 0
        flexkv_task_id = -1

        # INFO: TP/CP group is strictly synchronous, so TP/CP ranks are symmetric. This means they
        #       have identical dst GPU blocks. Hence, let TP/CP rank 0 do prefix matching on the
        #       TP/CP group's behalf and broadcast the result to the rest of the group.
        if self._sync_ctx.is_sync_leader:
            token_ids_np = np.array(token_ids, dtype=np.int64)
            # SWA-aware matching: when a SWA GPU pool is present and the FlexKV
            # side exposes get_match_swa, use it so the matched prefix also gets
            # an SWA reuse window (return_mask_swa). Full-KV reuse is driven by
            # ``token_mask`` (== full_mask) exactly like get_match and is NOT
            # gated on SWA. Works in BOTH deployment forms: in-process and
            # server_client_mode (the SWA RPC is implemented). The try/except
            # below still falls back to single-mask get_match if the RPC errors,
            # so a server-side problem never crashes the scheduler.
            #
            # NOTE: swa_mask is reserved for the future SWA H2D restore path and
            # is currently ignored by FlexKV's get_match_swa, so we do NOT build
            # gpu_swa_mask here (it would be a wasted GPU gather + D2H). When the
            # SWA data plane lands, construct it from
            # full_to_swa_index_mapping[device_indices] == 0 (0 = not resident).
            use_swa_match = (
                self._swa_kv_pool is not None
                and hasattr(self.kv_manager, "get_match_swa")
            )
            if use_swa_match:
                try:
                    result = self.kv_manager.get_match_swa(
                        token_ids=token_ids_np,
                        full_mask=token_mask,
                        swa_mask=None,
                        update_state_for_load=True,
                    )
                    flexkv_task_id, return_mask_full, _return_mask_swa = result
                    # FlexKV already truncates the Full-KV transfer to the
                    # SWA-reusable prefix (usable = min(full_hit, swa_hit)) before
                    # building the graph, so return_mask_full is SWA-bounded and
                    # this hit_length never over-reads full KV past where the SWA
                    # window can be restored. (_return_mask_swa is the trailing SWA
                    # window; unused here until the connector grows a gpu_swa_mask
                    # to extend reuse — see design doc §5.5.)
                    hit_length = (
                        int(return_mask_full.sum()) if return_mask_full is not None else 0
                    )
                except Exception as swa_err:  # noqa: BLE001 — fall back, never crash scheduler
                    logger.warning(
                        "[FlexKV] get_match_swa failed (%s); falling back to get_match",
                        swa_err,
                    )
                    result = self.kv_manager.get_match(
                        token_ids=token_ids_np, token_mask=token_mask,
                    )
                    if result is None:
                        flexkv_task_id, hit_length = -1, 0
                    else:
                        flexkv_task_id, matched_mask = result
                        hit_length = int(matched_mask.sum()) if matched_mask is not None else 0
            else:
                result = self.kv_manager.get_match(
                    token_ids=token_ids_np,
                    token_mask=token_mask,
                )
                # get_match returns None when the FlexKV server encounters an
                # error (e.g. in server_client_mode).  Guard against unpacking a
                # None result to avoid crashing the scheduler.
                if result is None:
                    logger.warning("[FlexKV] get_match returned None, treating as no hit")
                    flexkv_task_id = -1
                    hit_length = 0
                else:
                    flexkv_task_id, matched_mask = result
                    hit_length = int(matched_mask.sum()) if matched_mask is not None else 0
            if not update_state_for_load and flexkv_task_id >= 0:
                # Only cancel if the task actually has pending work.  When
                # hit_length == 0 the transfer graph is empty and the task was
                # already marked COMPLETED synchronously inside get_match →
                # _process_empty_graph, so cancelling would be a no-op that
                # triggers a spurious "already completed" warning.
                if hit_length > 0:
                    self.kv_manager.cancel([flexkv_task_id])
            else:
                ## GPU hit length is the zero length of token masks
                gpu_hit_length = torch.logical_not(token_mask).sum()
                logger.debug(f"[FlexKV Connector] gpu hit length: {gpu_hit_length}, Flexkv hit length: {hit_length}")

        if self._sync_ctx.needs_sync:
            data = self._sync_ctx.scatter(
                {"hit_length": hit_length, "task_id": flexkv_task_id},
            )
            hit_length = data["hit_length"]
            flexkv_task_id = data["task_id"]

        # Page-align host_hit_length: ensure GET loads complete pages
        if hit_length > 0 and self.page_size > 1:
            aligned_hit = (hit_length // self.page_size) * self.page_size
            if aligned_hit < hit_length:
                logger.debug(
                    "[FlexKV] get_new_hit_length: host_hit_length page_align %d -> %d (page_size=%d)",
                    hit_length, aligned_hit, self.page_size,
                )
                hit_length = aligned_hit

        if update_state_for_load and rid is not None and hit_length > 0:
            self._pending_loads[rid] = flexkv_task_id
        elif update_state_for_load and flexkv_task_id >= 0 and self._sync_ctx.is_sync_leader:
            # Task was not cancelled earlier, but won't be used — cancel it now
            # to avoid resource leak (e.g. hit_length page-aligned to 0, or rid is None).
            # Skip cancel when hit_length == 0: the task's transfer graph was
            # empty and _process_empty_graph already marked it COMPLETED.
            if hit_length > 0:
                self.kv_manager.cancel([flexkv_task_id])
        return hit_length

    def release_load_state(self, rid: str) -> None:
        fkv_tid = self._pending_loads.pop(rid, -1)
        if fkv_tid >= 0 and self._sync_ctx.is_sync_leader:
            self.kv_manager.cancel([fkv_tid])

    def start_load_kv(
        self,
        task_id: int,
        load_ops: List[LoadOperation],
    ) -> None:
        flexkv_task_ids: List[int] = []
        slot_mappings: List[torch.Tensor] = []

        for op in load_ops:
            fkv_tid = self._pending_loads.pop(op.rid, -1)
            if fkv_tid < 0:
                continue
            flexkv_task_ids.append(fkv_tid)
            indices = op.device_indices
            slot_mapping_cpu = indices.cpu() if indices.is_cuda else indices
            slot_mapping_cpu = slot_mapping_cpu.to(torch.int64)
            slot_mappings.append(slot_mapping_cpu)
            try:
                from flexkv.common.debug import summarize_block_ids_from_slots

                block_stats = summarize_block_ids_from_slots(
                    slot_mapping_cpu, self.page_size
                )
                logger.info(
                    f"[FlexKV-SEGV-DEBUG] start_load_kv rid={op.rid}, fkv_tid={fkv_tid}, "
                    f"device_indices_count={int(indices.numel())}, "
                    f"slot_min={block_stats.get('slot_min', 'n/a')}, "
                    f"slot_max={block_stats.get('slot_max', 'n/a')}, "
                    f"block_count={block_stats.get('block_count', 0)}, "
                    f"block_min={block_stats.get('block_min', 'n/a')}, "
                    f"block_max={block_stats.get('block_max', 'n/a')}"
                )
            except Exception as e:
                logger.warning(
                    f"[FlexKV-SEGV-DEBUG] start_load_kv stats failed rid={op.rid}: {e}"
                )

        logger.debug(f"[FlexKV] start_load_kv: resolved {len(flexkv_task_ids)} flexkv tasks")
        if not flexkv_task_ids:
            self._completed_loads.append(task_id)
            return

        if self._sync_ctx.should_send_slot_mapping_to_remote:
            logger.debug(f"[FlexKV] start_load_kv: sending slot_mapping for cross-node pp_receiver")
            for fkv_tid, slot_map in zip(flexkv_task_ids, slot_mappings):
                self.send_slot_mapping_to_remote(fkv_tid, slot_map)

        if self.enable_layerwise_transfer and self._layer_done_counter is not None:
            # PP1+: receive counter_id from PP0 first
            if self._sync_ctx.is_pp_receiver:
                payload = self._sync_ctx.scatter_pp(None)
                if payload.get("cmd") != CMD_LAYERWISE:
                    raise RuntimeError(f"Tag mismatch: expected {CMD_LAYERWISE}, got {payload.get('cmd')}")
                producer_id = payload["counter_id"]
                self._layer_done_counter.register_task_with_explicit_counter_id(task_id, producer_id)
            else:
                # Original logic: every rank independently updates producer
                producer_id = self._layer_done_counter.update_producer()
                self._layer_done_counter.events[producer_id].reset_for_new_transfer()
                self._layer_done_counter.register_task(task_id, producer_id)

            # Pre-fire eventfds for layers FlexKV's multi-group worker won't
            # touch (DSv4 dense layers, compress_ratio=0). SWA KV restore for
            # those layers now rides the FlexKV transfer engine (data plane;
            # see get_match_swa), so the eventfd here is purely a "ready" signal
            # that releases sglang's per-layer wait_until for those layers so
            # forward can progress through them. See ``_signal_dense_layers_ready``.
            self._signal_dense_layers_ready(producer_id)

            # PP0 sync leader: send counter_id to PP1+
            if self._sync_ctx.is_pp_sender:
                self._sync_ctx.scatter_pp(
                    {"cmd": CMD_LAYERWISE, "fkv_task_id": flexkv_task_ids[0], "counter_id": producer_id},
                )

            if self._sync_ctx.is_sync_leader:
                # [FLEXKV-DEBUG-ISOLATE] main c4/c128/indexer H2D path.
                logger.info(
                    f"[FLEXKV-DEBUG] MAIN H2D (layerwise) launch task_ids={flexkv_task_ids}, "
                    f"slot_counts={[int(s.numel()) for s in slot_mappings]}, "
                    f"counter_id={producer_id}"
                )
                self.kv_manager.launch(
                    task_ids=flexkv_task_ids,
                    slot_mappings=slot_mappings,
                    as_batch=True,
                    layerwise_transfer=True,
                    counter_id=producer_id,
                )
                self._load_fkv_tids.extend(flexkv_task_ids)
            self._ongoing_loads[task_id] = producer_id
        else:
            if self._sync_ctx.is_sync_leader:
                self.kv_manager.launch(
                    task_ids=flexkv_task_ids,
                    slot_mappings=slot_mappings,
                    as_batch=True,
                    layerwise_transfer=False,
                )
                response = self.kv_manager.wait(flexkv_task_ids, timeout=30.0)
                if not all(
                    tid in response and response[tid].status == KVResponseStatus.SUCCESS
                    for tid in flexkv_task_ids
                ):
                    logger.warning(
                        "[FlexKV] Some tasks failed in non-layerwise transfer"
                    )

            if self._sync_ctx.needs_sync:
                self._sync_ctx.barrier()

            self._completed_loads.append(task_id)

    def check_completed_load_tasks(self) -> List[int]:
        if self._sync_ctx.is_sync_leader and len(self._load_fkv_tids) >= 100:
            self.kv_manager.try_wait(task_ids=self._load_fkv_tids)
            self._load_fkv_tids.clear()

        if self._layer_done_counter is not None:
            for ext_tid, producer_id in list(self._ongoing_loads.items()):
                if self._layer_done_counter.events[producer_id]._finished:
                    self._completed_loads.append(ext_tid)
                    del self._ongoing_loads[ext_tid]

        result = list(self._completed_loads)
        self._completed_loads.clear()
        return result

    def start_store_kv(
        self,
        task_id: int,
        token_ids: List[int],
        kv_indices: torch.Tensor,
    ) -> None:

        def _send_pp_put_meta(fkv_task_id: int, unmatched_mask):
            if not self._sync_ctx.is_pp_active:
                return
            mask_list = (
                unmatched_mask.cpu().tolist()
                if hasattr(unmatched_mask, "is_cuda") and unmatched_mask.is_cuda
                else (unmatched_mask.tolist() if hasattr(unmatched_mask, "tolist") else [])
            )
            self._sync_ctx.scatter_pp(
                {"cmd": CMD_PUT_META, "fkv_task_id": fkv_task_id, "unmatched_mask": mask_list},
            )

        if not self._sync_ctx.is_sync_leader:
            if self._sync_ctx.is_pp_receiver:
                logger.debug(
                    f"[FlexKV-Connector] start_store_kv: PP1+ scatter recv PUT_META"
                )
                payload = self._sync_ctx.scatter_pp(None)
                if payload.get("cmd") != CMD_PUT_META:
                    raise RuntimeError(f"Tag mismatch: expected {CMD_PUT_META}, got {payload.get('cmd')}")
                fkv_task_id = payload["fkv_task_id"]
                unmatched_mask = torch.tensor(payload["unmatched_mask"])
                if unmatched_mask.sum() > 0 and fkv_task_id >= 0:
                    if self._sync_ctx.should_send_slot_mapping_to_remote:
                        filtered = kv_indices[unmatched_mask]
                        slot_mapping = filtered.cpu() if filtered.is_cuda else filtered
                        slot_mapping = slot_mapping.to(torch.int64)
                        self.send_slot_mapping_to_remote(fkv_task_id, slot_mapping)
                    self._ongoing_stores[task_id] = fkv_task_id
                else:
                    self._completed_stores.append(task_id)
            return

        try:
            token_ids_np = np.array(token_ids, dtype=np.int64)
            assert len(token_ids) == len(kv_indices), (
                f"len(token_ids)={len(token_ids)} != len(kv_indices)={len(kv_indices)}, "
                f"task_id={task_id}, page_size={self.page_size}, "
                f"kv_indices_shape={kv_indices.shape if hasattr(kv_indices, 'shape') else 'N/A'}"
            )

            # Page-align token_ids and kv_indices BEFORE put_match so that
            # put_match allocates dst_block_ids consistent with the slot_mapping
            # we will later pass to launch().
            original_len = len(token_ids_np)
            if self.page_size > 1:
                aligned_len = (original_len // self.page_size) * self.page_size
                if aligned_len == 0:
                    _send_pp_put_meta(fkv_task_id=-1, unmatched_mask=[])
                    self._completed_stores.append(task_id)
                    return
                if aligned_len < original_len:
                    token_ids_np = token_ids_np[:aligned_len]
                    kv_indices = kv_indices[:aligned_len]
            result = self.kv_manager.put_match(
                token_ids=token_ids_np,
                token_mask=None,
            )
            # put_match returns None when the FlexKV server encounters an
            # error (e.g. in server_client_mode).  Treat as a failed store.
            if result is None:
                logger.warning("[FlexKV] put_match returned None, skipping store for task %d", task_id)
                _send_pp_put_meta(fkv_task_id=-1, unmatched_mask=[])
                self._completed_stores.append(task_id)
                return
            fkv_task_id, unmatched_mask = result

            logger.debug(
                f"[FlexKV] start_store_kv: tokens={len(token_ids)}, "
                f"kv_indices={len(kv_indices)}, fkv_task_id={fkv_task_id}, "
                f"unmatched={unmatched_mask.sum().item() if hasattr(unmatched_mask, 'sum') else len(unmatched_mask)}"
            )

            _send_pp_put_meta(fkv_task_id, unmatched_mask)

            if unmatched_mask.sum() > 0:
                filtered = kv_indices[unmatched_mask]
                slot_mapping = filtered.cpu() if filtered.is_cuda else filtered
                slot_mapping = slot_mapping.to(torch.int64)
                try:
                    from flexkv.common.debug import summarize_block_ids_from_slots

                    store_stats = summarize_block_ids_from_slots(
                        slot_mapping, self.page_size
                    )
                    logger.info(
                        f"[FlexKV-SEGV-DEBUG] start_store_kv launch D2H "
                        f"task_id={task_id}, fkv_task_id={fkv_task_id}, "
                        f"token_count={len(token_ids_np)}, "
                        f"unmatched_count={int(unmatched_mask.sum())}, "
                        f"slot_min={store_stats.get('slot_min', 'n/a')}, "
                        f"slot_max={store_stats.get('slot_max', 'n/a')}, "
                        f"block_count={store_stats.get('block_count', 0)}, "
                        f"block_min={store_stats.get('block_min', 'n/a')}, "
                        f"block_max={store_stats.get('block_max', 'n/a')}"
                    )
                except Exception as log_err:
                    logger.warning(
                        f"[FlexKV-SEGV-DEBUG] start_store_kv stats failed task_id={task_id}: {log_err}"
                    )

                self.kv_manager.launch(
                    task_ids=[fkv_task_id], slot_mappings=[slot_mapping]
                )
                self._ongoing_stores[task_id] = fkv_task_id
            else:
                self._completed_stores.append(task_id)

            # SWA store now rides the FlexKV transfer engine (data plane); see get_match_swa.
        except Exception as e:
            logger.error("[FlexKV] start_store_kv failed: %s", e, exc_info=True)
            _send_pp_put_meta(fkv_task_id=-1, unmatched_mask=[])
            self._completed_stores.append(task_id)

    def check_completed_store_tasks(self) -> List[int]:
        completed_ext_ids = list(self._completed_stores)
        self._completed_stores.clear()

        completed_dict = {}
        if self._sync_ctx.is_sync_leader and self._ongoing_stores:
            fk_to_ext = {v: k for k, v in self._ongoing_stores.items()}
            completed_dict = self.kv_manager.try_wait(task_ids=list(fk_to_ext.keys()))
            for fk_tid in completed_dict:
                ext_tid = fk_to_ext[fk_tid]
                completed_ext_ids.append(ext_tid)
                del self._ongoing_stores[ext_tid]

        if self._sync_ctx.is_pp_sender:
            self._sync_ctx.scatter_pp(
                {"cmd": CMD_STORE_COMPLETE, "completed_fk_ids": list(completed_dict)},
            )
        elif self._sync_ctx.is_pp_receiver:
            payload = self._sync_ctx.scatter_pp(None)
            if payload.get("cmd") != CMD_STORE_COMPLETE:
                raise RuntimeError(f"Tag mismatch: expected {CMD_STORE_COMPLETE}, got {payload.get('cmd')}")
            fk_ids = payload["completed_fk_ids"]
            if fk_ids and self._ongoing_stores:
                fk_to_ext = {v: k for k, v in self._ongoing_stores.items()}
                for fk_tid in fk_ids:
                    if fk_tid in fk_to_ext:
                        ext_tid = fk_to_ext[fk_tid]
                        completed_ext_ids.append(ext_tid)
                        del self._ongoing_stores[ext_tid]

        if self._sync_ctx.needs_sync:
            completed_ext_ids = self._sync_ctx.scatter(completed_ext_ids)

        return completed_ext_ids

    # ---- Optional overrides ----

    def prefetch(self, rid: str, token_ids: List[int]) -> None:
        if not self._prefetch_enabled:
            return
        if not rid:
            return

        prefetch_task_id = -1
        if self._sync_ctx.is_sync_leader:
            token_ids_np = np.array(token_ids, dtype=np.int64)
            prefetch_task_id = self.kv_manager.prefetch_async(
                token_ids=token_ids_np,
            )
            logger.debug(f"[FlexKV] prefetch: launched task_id={prefetch_task_id}")

        if self._sync_ctx.needs_sync:
            data = self._sync_ctx.scatter(
                {"task_id": prefetch_task_id},
            )
            prefetch_task_id = data["task_id"]

        if prefetch_task_id >= 0:
            self._ongoing_prefetches[rid] = prefetch_task_id

    def check_prefetch_progress(self, rid: str) -> bool:
        if not self._prefetch_enabled:
            return True

        prefetch_task_id = self._ongoing_prefetches.get(rid, -1)
        if prefetch_task_id < 0:
            return True

        is_completed = False
        if self._sync_ctx.is_sync_leader:
            completed = self.kv_manager.try_wait(task_ids=[prefetch_task_id])
            if prefetch_task_id in completed:
                status = completed[prefetch_task_id].status
                if status != KVResponseStatus.SUCCESS:
                    logger.warning(
                        "[FlexKV] prefetch task %d for rid=%s finished with status=%s",
                        prefetch_task_id,
                        rid,
                        status,
                    )
                is_completed = True

        if self._sync_ctx.needs_sync:
            data = self._sync_ctx.scatter(
                {"is_completed": is_completed, "loaded_tokens": 0},
            )
            is_completed = data["is_completed"]

        if is_completed:
            self._ongoing_prefetches.pop(rid, None)
        return is_completed

    def pop_prefetch_loaded_tokens(self, rid: str) -> int:
        # TODO: Implement this
        return 0

    def cancel_prefetch(self, rid: str) -> None:
        self._pending_loads.pop(rid, None)
        prefetch_task_id = self._ongoing_prefetches.pop(rid, -1)
        if self._sync_ctx.is_sync_leader and prefetch_task_id >= 0:
            # Flexkv not support cancel prefetch task yet
            pass

    @property
    def layer_done_counter(self) -> Any:
        return self._layer_done_counter

    def register_layer_transfer_counter(self, kvcache: Any) -> None:
        if self._layer_done_counter is not None:
            kvcache.register_layer_transfer_counter(self._layer_done_counter)

    def reset(self) -> None:
        if self._sync_ctx.is_sync_leader and self._pending_loads:
            pending_tids = [tid for tid in self._pending_loads.values() if tid >= 0]
            if pending_tids:
                self.kv_manager.cancel(pending_tids)
        self._pending_loads.clear()
        self._ongoing_prefetches.clear()
        self._ongoing_loads.clear()
        self._completed_loads.clear()
        self._load_fkv_tids.clear()

        if self._sync_ctx.is_sync_leader:
            for fk_tid in list(self._ongoing_stores.values()):
                if fk_tid >= 0:
                    self._wait_flexkv_task(fk_tid)
        self._ongoing_stores.clear()
        self._completed_stores.clear()

        if self._layer_done_counter is not None:
            self._layer_done_counter.reset()

    def shutdown(self) -> None:
        if self._sync_ctx.is_sync_leader:
            self.kv_manager.shutdown()

        # Shutdown TransferManagerOnRemote process on Node B
        if self._remote_process is not None:
            try:
                self._remote_process.terminate()
                self._remote_process.join(timeout=5.0)
                if self._remote_process.is_alive():
                    logger.warning(
                        f"[FlexKV] TransferManagerOnRemote did not terminate gracefully, "
                        f"killing{self._rank_label}")
                    self._remote_process.kill()
                    self._remote_process.join()
            except Exception as e:
                logger.warning(
                    f"[FlexKV] Error shutting down TransferManagerOnRemote{self._rank_label}: {e}")
            self._remote_process = None

    # ---- Private helpers ----

    def _wait_flexkv_task(self, fk_task_id: int, timeout: float = 20.0) -> bool:
        if fk_task_id < 0 or not self._sync_ctx.is_sync_leader:
            return True
        try:
            response = self.kv_manager.wait([fk_task_id], timeout=timeout)
            return (
                fk_task_id in response
                and response[fk_task_id].status == KVResponseStatus.SUCCESS
            )
        except Exception as e:
            logger.error("[FlexKV] wait task failed: %s", e, exc_info=True)
            return False

    def _build_dsv4_layer_group_specs(self) -> List[LayerGroupSpec]:
        """Build LayerGroupSpec list from detected DSv4 sub-pool metadata."""
        layer_groups: List[LayerGroupSpec] = []
        for gi in self._dsv4_layer_groups_info:
            buf = gi["buffers"][0]
            bytes_per_page_padded = buf.shape[1]
            sub_page_size = gi["sub_page_size"]
            if bytes_per_page_padded % sub_page_size != 0:
                raise RuntimeError(
                    f"[FlexKV-DSv4] group '{gi['name']}': "
                    f"bytes_per_page_padded={bytes_per_page_padded} is not "
                    f"divisible by sub_page_size={sub_page_size}"
                )
            effective_head_size = bytes_per_page_padded // sub_page_size
            layer_groups.append(LayerGroupSpec(
                num_layers=len(gi["layer_ids"]),
                num_kv_heads=1,
                head_size=effective_head_size,
                layer_indices=list(gi["layer_ids"]),
                compress_ratio=gi["ratio"],
                dtype=gi["dtype"],
            ))
        return layer_groups

    def _build_mla_indexer_layer_group_specs(
        self,
        kv_caches: List[torch.Tensor],
        indexer_buffers: List[torch.Tensor],
    ) -> List[LayerGroupSpec]:
        """Main MLA KV + sparse-attention indexer as two layer groups."""
        _, num_kv_heads, head_size = kv_caches[0].shape
        main_layer_group = LayerGroupSpec(
            num_layers=self.rank_info.num_layers_per_pp_stage,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            layer_indices=list(range(self.rank_info.num_layers_per_pp_stage)),
            compress_ratio=1,
            dtype=kv_caches[0].dtype,
        )
        indexer_tensor = indexer_buffers[0]
        indexer_layer_group = LayerGroupSpec(
            num_layers=len(indexer_buffers),
            num_kv_heads=1,
            head_size=indexer_tensor.shape[1],
            layer_indices=list(range(len(indexer_buffers))),
            compress_ratio=1,
            dtype=indexer_buffers[0].dtype,
        )
        return [main_layer_group, indexer_layer_group]

    def _apply_layer_groups_for_cache_sizing(
        self,
        kv_caches: List[torch.Tensor],
        indexer_buffers: Optional[List[torch.Tensor]],
    ) -> None:
        """Set layer_groups and recompute num_cpu/ssd_blocks before KVManager."""
        model_config = self.flexkv_config.model_config
        cache_config = self.flexkv_config.cache_config

        old_cpu = cache_config.num_cpu_blocks

        if model_config.layer_groups is None:
            if self._is_dsv4:
                model_config.layer_groups = self._build_dsv4_layer_group_specs()
            elif indexer_buffers is not None and len(indexer_buffers) > 0:
                model_config.layer_groups = self._build_mla_indexer_layer_group_specs(
                    kv_caches, indexer_buffers)
            else:
                return

        if not recompute_cache_block_counts(model_config, cache_config):
            return

        logger.info(
            f"[FlexKV] Cache block counts aligned to layer_groups{self._rank_label}: "
            f"num_cpu_blocks {old_cpu} -> {cache_config.num_cpu_blocks}, "
            f"num_groups={len(model_config.layer_groups)}"
        )

        for _attr in ("num_cpu_blocks", "num_ssd_blocks", "num_remote_blocks"):
            _orig = getattr(cache_config, _attr)
            if _orig is None or _orig <= 0:
                continue
            _aligned = self._sync_ctx.all_reduce_min(_orig)
            if _aligned != _orig:
                logger.info(
                    f"[FlexKV] Block count MIN alignment '{_attr}' after "
                    f"layer_groups recompute{self._rank_label}: "
                    f"{_orig} -> {_aligned}"
                )
            setattr(cache_config, _attr, _aligned)

    def _register_with_retry(
        self,
        kv_caches: List[torch.Tensor],
        indexer_buffers: Optional[List[torch.Tensor]] = None,
        max_retries: int = 360,
    ) -> None:
        """Register GPU with retry for Node B (wait for TransferManagerOnRemote).

        Node B's non-leader ranks may attempt to register before
        TransferManagerOnRemote has finished initializing. This method
        retries the registration up to ``max_retries`` times (default 360,
        i.e. 6 minutes at 1 s intervals).
        """
        for attempt in range(max_retries):
            try:
                self._register_to_server(kv_caches, indexer_buffers)
                return
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                if attempt % 30 == 0:
                    logger.info(
                        f"[FlexKV] GPU register retry{self._rank_label}: "
                        f"attempt={attempt+1}/{max_retries}, error={e}"
                    )
                time.sleep(1.0)

    def _attach_draft_pool(self) -> List[torch.Tensor]:
        """Fold the speculative-decoding draft pool into ``_dsv4_layer_groups_info``.

        Called from ``__init__`` after target ``_dsv4_layer_groups_info`` is
        built but BEFORE ``_register_with_retry`` runs. Returns the list of
        draft GPU buffers to append to ``kv_caches``; empty list on any
        unsupported case (and a warning is logged).

        Why here and not in a public setter:
            FlexKV's ``TransferManager._handle_gpu_blocks_registration``
            rejects re-registering an already-known device_id, and
            ``model_config.layer_groups`` follows "first registration wins"
            semantics. So the draft pool MUST be folded into the very first
            ``register_to_server`` call. Set-after-init would not work.

        Simplified-version supported scope (see
        ``MTP_PIGGYBACK_DESIGN.md`` §2.4):
            * draft pool is a ``DeepSeekV4TokenToKVPool`` (DSv4 NextN).
            * draft has exactly 1 SWA layer (the standard NextN topology).
            * Any other case returns ``[]`` and degrades silently to
              "no piggyback" (i.e. accept_length will degrade after a
              FlexKV cache hit, but correctness is preserved).
        """
        if self._draft_kv_pool is None:
            return []

        # We only support the DSv4 multi-pool layout for now (no MLA/MHA
        # draft support in the simplified version). Detect by presence of
        # the ``swa_kv_pool`` attribute, matching how the target side does
        # DSv4 detection above.
        if not hasattr(self._draft_kv_pool, 'swa_kv_pool'):
            logger.warning(
                "[FlexKV-MTP] Draft pool type %s lacks 'swa_kv_pool'; "
                "MTP piggyback unsupported, falling back to no piggyback",
                type(self._draft_kv_pool).__name__,
            )
            return []

        draft_swa = self._draft_kv_pool.swa_kv_pool
        if draft_swa is None:
            logger.warning("[FlexKV-MTP] Draft pool has no swa_kv_pool; skipping piggyback")
            return []

        draft_buffers = getattr(draft_swa, 'kv_buffer', None)
        if draft_buffers is None or len(draft_buffers) == 0:
            logger.warning(
                "[FlexKV-MTP] Draft swa_kv_pool has no kv_buffer; skipping piggyback"
            )
            return []

        num_draft_layers = len(draft_buffers)
        if num_draft_layers != 1:
            # Multi-layer draft (e.g. EAGLE-2) is out of scope for the
            # simplified version. Skip — accept_length will degrade after
            # FlexKV cache hit, but model correctness is preserved.
            logger.warning(
                "[FlexKV-MTP] Draft has %d swa layers; simplified version "
                "supports only 1-layer NextN. Skipping piggyback.",
                num_draft_layers,
            )
            return []

        if not hasattr(self, '_dsv4_layer_groups_info'):
            # Target is not DSv4 (e.g. MLA/MHA). Simplified version requires
            # DSv4 on both sides because the layer-groups wiring is DSv4-
            # specific (compress_ratio / sub_page_size / packed bytes).
            logger.warning(
                "[FlexKV-MTP] Target is not DSv4 multi-pool; simplified version "
                "requires DSv4 on both target and draft. Skipping piggyback.",
            )
            return []

        # Layer-id namespace: target uses 0..target_num_layers-1; we put
        # draft at target_num_layers..+num_draft_layers-1 so layer_ids
        # don't collide. The actual numeric value doesn't matter to the
        # transfer engine (LayerGroup is opaque), only that it's unique
        # across all groups.
        target_num_layers = self.rank_info.num_layers_per_pp_stage
        draft_layer_ids = [target_num_layers + i for i in range(num_draft_layers)]

        # Build the LayerGroup descriptor mirroring c4/c128/c4_indexer
        # entries — the rest of the registration / store / load path
        # iterates self._dsv4_layer_groups_info uniformly.
        # Note: ratio=1 (raw, uncompressed). DSv4 NextN uses
        # COMPRESS_RATIO_NEXTN_LAYER=0 internally, but FlexKV's
        # LayerGroupSpec requires compress_ratio >= 1; ratio=1 is the
        # natural representation for "no compression".
        bytes_per_token = (
            draft_swa.get_bytes_per_token()
            if hasattr(draft_swa, 'get_bytes_per_token')
            else None
        )
        sub_page_size = getattr(draft_swa, 'page_size', self.page_size)

        self._draft_swa_layer_group = {
            "name": "draft_swa",
            "ratio": 1,
            "layer_ids": draft_layer_ids,
            "buffers": list(draft_buffers),
            "bytes_per_token": bytes_per_token,
            "sub_page_size": sub_page_size,
            "dtype": draft_buffers[0].dtype,
        }
        self._dsv4_layer_groups_info.append(self._draft_swa_layer_group)

        logger.info(
            f"[FlexKV-MTP] Draft pool attached{self._rank_label}: "
            f"num_layers={num_draft_layers}, "
            f"bytes_per_token={bytes_per_token}, "
            f"sub_page_size={sub_page_size}, "
            f"layer_ids={draft_layer_ids}, "
            f"draft_pool_type={type(self._draft_kv_pool).__name__}"
        )
        return list(draft_buffers)

    def _register_to_server(
        self,
        kv_caches: List[torch.Tensor],
        indexer_buffers: Optional[List[torch.Tensor]] = None,
    ) -> None:
        """Register GPU KV cache buffers to FlexKV server.

        Args:
            kv_caches: Unified KV cache tensor list.
                - MLA: num_layer tensors (K and V share the same buffer).
                - MHA: 2 * num_layer tensors (K buffers followed by V buffers).
                - DSv4: concatenation of all sub-pool buffers (c4 first, then
                  c128, then optional c4_indexer); each sub-pool tensor is 2D
                  ``[num_pages, bytes_per_page_padded]`` uint8.
            indexer_buffers: Optional sparse attention indexer buffers (NSA path).
                Mutually exclusive with the DSv4 path (DSv4 routes its indexer
                via ``self._dsv4_layer_groups_info`` instead).
        """
        assert len(kv_caches) > 0

        # ---- DSv4 multi-pool path ----
        if self._is_dsv4:
            self._register_to_server_dsv4(kv_caches)
            return

        assert kv_caches[0].ndim == 3, f"Expected 3D tensor, got shape={kv_caches[0].shape}"

        is_mla = self.flexkv_config.model_config.use_mla
        num_blocks, num_kv_heads, head_size = kv_caches[0].shape

        # GPU layout uses page_size as tokens_per_block so that the transfer
        # engine's block_stride covers an entire page of tokens.  The physical
        # GPU tensor shape is [num_blocks, num_kv_heads, head_size] where each
        # slot stores 1 token, but we present it to FlexKV as
        # [num_blocks/page_size, page_size, num_kv_heads, head_size] so that
        # block_id * block_stride correctly addresses the start of a page.

        gpu_layout = KVCacheLayout(
            type=KVCacheLayoutType.LAYERFIRST,
            num_layer=self.rank_info.num_layers_per_pp_stage,
            num_block=num_blocks // self.page_size,
            tokens_per_block=self.page_size,
            num_head=num_kv_heads,
            head_size=head_size,
            is_mla=is_mla,
        )

        # Build indexer layout if indexer buffers are present
        indexer_layout = None
        if indexer_buffers is not None and len(indexer_buffers) > 0:
            indexer_tensor = indexer_buffers[0]
            assert indexer_tensor.ndim == 2, (
                f"Expected 2D indexer tensor (num_pages, page_stride_size), "
                f"got shape={indexer_tensor.shape}"
            )
            # sglang's NSA indexer buffer is 2D: (num_pages, page_stride_size),
            # where page_stride_size = page_size * (index_head_dim + scale_bytes).
            # All tokens within a page are flattened into a single contiguous
            # vector, so from FlexKV's perspective each page is one indivisible
            # block with tokens_per_block=1.  The resulting block_stride
            # (= 1 * 1 * page_stride_size) correctly addresses each page.
            indexer_layout = KVCacheLayout(
                type=KVCacheLayoutType.LAYERFIRST,
                num_layer=len(indexer_buffers),
                num_block=indexer_tensor.shape[0],
                tokens_per_block=1,
                num_head=1,
                head_size=indexer_tensor.shape[1],
                is_mla=True,
            )
            logger.debug(
                "[FlexKV] Indexer layout: num_layer=%d, num_block=%d, "
                "tokens_per_block=%d, head_size=%d",
                len(indexer_buffers), indexer_tensor.shape[0],
                1, indexer_tensor.shape[1],
            )
            # Consistency check: indexer num_block should equal main KV num_block
            # (1:1 mapping since tokens_per_block = page_size)
            indexer_config = self.flexkv_config.cache_config.indexer
            if indexer_config is not None:
                expected_indexer_blocks = num_blocks // self.page_size
                assert indexer_tensor.shape[0] == expected_indexer_blocks, (
                    f"[FlexKV] Indexer num_block mismatch: indexer has {indexer_tensor.shape[0]} pages, "
                    f"but main KV has {num_blocks} slots / page_size {self.page_size} "
                    f"= {expected_indexer_blocks} expected blocks"
                )

        # Register KV caches to FlexKV server.
        # If NSA indexer buffers are present, register them as an additional
        # layer-group (matches the multi-group registration API; FlexKV's
        # register_to_server() does NOT accept ad-hoc indexer_buffers/
        # indexer_layout kwargs — only kv_layout + layer_groups + gpu_layouts +
        # handles_per_group).
        if indexer_buffers is not None and len(indexer_buffers) > 0 and indexer_layout is not None:
            # Build a 2-group registration: main KV + indexer.
            main_layer_group = LayerGroupSpec(
                num_layers=self.rank_info.num_layers_per_pp_stage,
                num_kv_heads=num_kv_heads,
                head_size=head_size,
                layer_indices=list(range(self.rank_info.num_layers_per_pp_stage)),
                compress_ratio=1,
                dtype=kv_caches[0].dtype,
            )
            indexer_layer_group = LayerGroupSpec(
                num_layers=len(indexer_buffers),
                num_kv_heads=indexer_layout.num_head,
                head_size=indexer_layout.head_size,
                layer_indices=list(range(len(indexer_buffers))),
                compress_ratio=1,
                dtype=indexer_buffers[0].dtype,
            )
            # NOTE: Do NOT set self.flexkv_config.model_config.layer_groups here.
            # ModelConfig is frozen after FlexKVConfig.from_env() / post_init_*.
            # The server-side TransferManager picks up layer_groups from the
            # RegisterTPClientRequest and assigns it on its OWN model_config copy
            # (transfer_manager.py:85-86). Setting it here would just raise
            # AttributeError("ModelConfig is frozen") on every rank.
            self.tp_client.register_to_server(
                kv_caches=list(kv_caches) + list(indexer_buffers),
                kv_layout=gpu_layout,
                layer_groups=[main_layer_group, indexer_layer_group],
                gpu_layouts=[gpu_layout, indexer_layout],
                handles_per_group=[list(kv_caches), list(indexer_buffers)],
            )
            logger.info(
                "[FlexKV] Registered KV caches + NSA indexer to server "
                "(2 layer-groups)"
            )
        else:
            self.tp_client.register_to_server(
                kv_caches=kv_caches,
                kv_layout=gpu_layout,
            )
            logger.info("[FlexKV] Registered KV caches to server")

    def _register_to_server_dsv4(
        self,
        kv_caches: List[torch.Tensor],
    ) -> None:
        """DeepSeek V4 multi-pool registration path.

        DSv4's GPU KV layout differs structurally from MLA/MHA:

          * KV is split across multiple sub-pools (c4, c128, optional
            c4_indexer); each sub-pool has its own ``page_size``:

                page_size_full           = self.page_size  (e.g. 128)
                c4_kv_pool.page_size     = page_size_full // 4
                c128_kv_pool.page_size   = page_size_full // 128

            Per-token write is at ``compressed_loc = full_loc // ratio``
            for tokens where ``(full_loc + 1) % ratio == 0``. **At the page
            level this is a 1:1 mapping**: full_page_id == sub_pool_page_id
            because consecutive ``page_size_full`` full-pool tokens always
            land in exactly ``page_size_full // ratio`` consecutive sub-pool
            tokens, i.e. one sub-pool page.

          * Each sub-pool's ``kv_buffer[layer]`` is a *2D* uint8 tensor of
            shape ``[num_pages, bytes_per_page_padded]`` where each row
            packs ``sub_page_size`` tokens of (nope_fp8 + rope_bf16 + scale)
            into a single contiguous byte vector, and
            ``bytes_per_page_padded = ceil_div(sub_page_size * 584, 576) * 576``
            (576-byte alignment).

        We register each sub-pool as one ``LayerGroupSpec`` with:
            compress_ratio = sub-pool ratio (4 or 128)
            num_kv_heads   = 1   (DSv4 packs heads into the byte stream)
            head_size      = effective_per_token_bytes
                            = bytes_per_page_padded // sub_page_size
            dtype          = torch.uint8

        and a per-group ``KVCacheLayout`` with:
            num_block        = sub-pool num_pages
            tokens_per_block = sub_page_size
            num_head         = 1
            head_size        = effective_per_token_bytes
            is_mla           = True

        With this layout, FlexKV's GPU block_stride
        (= tokens_per_block * num_kv_heads * head_size * dtype_size
           = sub_page_size * (bytes_per_page_padded / sub_page_size)
           = bytes_per_page_padded)
        exactly matches the sglang DSv4 GPU buffer's actual page stride —
        no per-token offset translation needed at the byte layout level.

        The slot_mapping passed to ``kv_manager.launch()`` is in full-pool
        token index space; FlexKV converts it to full-pool page-ids via
        ``slot[::page_size_full] // page_size_full`` and uses those page-ids
        directly to index each sub-pool buffer, which is correct because
        the page-level mapping is 1:1 (see above).

        Limitations:
            * If ``bytes_per_page_padded`` is not divisible by
              ``sub_page_size``, the alignment-padding split is ambiguous
              and we raise. (Holds for typical DSv4 configs because
              gcd(sub_page_size, 576) typically lets the padded bytes
              divide evenly per token.)
            * ``cache_config.tokens_per_block`` (= ``page_size_full``) must
              be divisible by every group's compress_ratio. FlexKV's
              KVCacheLayout enforces this.
            * Indexer compression-state pools and SWA pool are NOT covered
              by this registration; SWA goes through the existing
              ``self._swa_kv_pool`` path; indexer compress states stay
              GPU-only.

        Args:
            kv_caches: concatenation of all DSv4 sub-pool buffers, in the
                same group order as ``self._dsv4_layer_groups_info``.
        """
        if not self._dsv4_layer_groups_info:
            raise RuntimeError(
                "[FlexKV] DSv4 detected but layer_groups_info is empty"
            )

        is_mla = self.flexkv_config.model_config.use_mla
        page_size_full = self.page_size

        # Build per-group LayerGroupSpec + KVCacheLayout entries.
        # ``layer_indices`` here are *original-layer ids* (from
        # ``compression_ratios``).  Multiple groups can share the same
        # original layer (e.g. c4 main KV and c4 indexer both attach to
        # the same CSA layer); FlexKV's ``LayerMemberMap`` handles this.
        layer_groups: List[LayerGroupSpec] = []
        gpu_layouts: List[KVCacheLayout] = []
        handles_per_group: List[List[torch.Tensor]] = []
        all_gpu_blocks: List[torch.Tensor] = []

        for gi in self._dsv4_layer_groups_info:
            buffers = gi["buffers"]
            if not buffers:
                logger.warning(
                    f"[FlexKV-DSv4] Skipping empty layer group '{gi['name']}'"
                )
                continue
            buf = buffers[0]
            assert buf.ndim == 2, (
                f"[FlexKV-DSv4] group '{gi['name']}' buffer must be 2D "
                f"[num_pages, bytes_per_page_padded], got shape={buf.shape}"
            )

            num_pages = buf.shape[0]
            bytes_per_page_padded = buf.shape[1]
            sub_page_size = gi["sub_page_size"]
            bytes_per_token = gi["bytes_per_token"]
            ratio = gi["ratio"]

            # Effective per-token width including alignment padding split
            # evenly across the tokens in a page. This is what FlexKV will
            # use as ``head_size`` so that one block_stride covers exactly
            # one packed page.
            #
            # NOTE: For DSv4 the resulting head_size (e.g. c4: 37440/64=585)
            # may not be a 16-byte multiple. FlexKV's transfer.cu uses float4
            # (16-byte) vectorized copies, which require chunk_size aligned
            # to 16 bytes. We address this in the FlexKV worker by falling
            # back to a non-vectorized path when stride alignment fails.
            if bytes_per_page_padded % sub_page_size != 0:
                raise RuntimeError(
                    f"[FlexKV-DSv4] group '{gi['name']}': "
                    f"bytes_per_page_padded={bytes_per_page_padded} is not "
                    f"divisible by sub_page_size={sub_page_size}. This breaks "
                    f"FlexKV's stride math (each token must occupy a fixed "
                    f"byte width). Check DeepSeekV4SingleKVPool.create_buffer "
                    f"alignment constants (576-byte multiple)."
                )
            effective_head_size = bytes_per_page_padded // sub_page_size

            # cache_config.tokens_per_block (= page_size_full) must be
            # divisible by ratio — FlexKV's KVCacheLayout enforces this for
            # the CPU multi-group BLOCKFIRST layout.
            if page_size_full % ratio != 0:
                raise RuntimeError(
                    f"[FlexKV-DSv4] group '{gi['name']}': page_size_full="
                    f"{page_size_full} not divisible by compress_ratio={ratio}. "
                    f"Choose a sglang page_size that is a multiple of every "
                    f"DSv4 compression ratio (4 and 128)."
                )

            # Useful (non-padded) bytes per page — informational only.
            useful_bytes = sub_page_size * bytes_per_token
            if useful_bytes != bytes_per_page_padded:
                logger.info(
                    f"[FlexKV-DSv4] group '{gi['name']}' has page padding: "
                    f"useful={useful_bytes}B, padded={bytes_per_page_padded}B "
                    f"(effective head_size={effective_head_size}B/token)"
                )

            # Verify all layers in this group have identical shapes
            for li, b in enumerate(buffers):
                if b.shape != buf.shape:
                    raise RuntimeError(
                        f"[FlexKV-DSv4] group '{gi['name']}' layer {li} has "
                        f"shape {b.shape}, expected {buf.shape}"
                    )

            layer_groups.append(LayerGroupSpec(
                num_layers=len(gi["layer_ids"]),
                num_kv_heads=1,
                head_size=effective_head_size,
                layer_indices=list(gi["layer_ids"]),
                compress_ratio=ratio,
                dtype=gi["dtype"],
            ))
            gpu_layouts.append(KVCacheLayout(
                type=KVCacheLayoutType.LAYERFIRST,
                num_layer=len(gi["layer_ids"]),
                num_block=num_pages,
                tokens_per_block=sub_page_size,
                num_head=1,
                head_size=effective_head_size,
                is_mla=True,
            ))
            handles_per_group.append(list(buffers))
            all_gpu_blocks.extend(buffers)

            logger.info(
                f"[FlexKV-DSv4] Registered group '{gi['name']}': "
                f"compress_ratio={ratio}, num_layers={len(gi['layer_ids'])}, "
                f"num_pages={num_pages}, sub_page_size={sub_page_size}, "
                f"bytes_per_page_padded={bytes_per_page_padded}, "
                f"effective_head_size={effective_head_size}, "
                f"useful_bytes_per_token={bytes_per_token}, "
                f"layer_indices={gi['layer_ids'][:4]}"
                f"{'...' if len(gi['layer_ids']) > 4 else ''}"
            )

        if len(all_gpu_blocks) != len(kv_caches):
            raise RuntimeError(
                f"[FlexKV-DSv4] flat buffer count mismatch: "
                f"groups built {len(all_gpu_blocks)} buffers, "
                f"but kv_caches has {len(kv_caches)}"
            )

        # NOTE: Do NOT set self.flexkv_config.model_config.layer_groups here.
        # ModelConfig is frozen after FlexKVConfig.from_env() / post_init_*.
        # The server-side TransferManager picks it up from the registration
        # request and assigns it on its own model_config copy
        # (transfer_manager.py:85-86). Setting it here raises
        # AttributeError("ModelConfig is frozen") on every rank.

        # Primary layout (kv_layout arg).
        # IMPORTANT: TransferManager derives ``num_layers_per_pp_stage`` from
        # ``primary_layout.num_layer`` (transfer_manager.py:148), which then
        # propagates to cpu_kv_layout.num_layer and is used as the original
        # layer-id index space when building LayerMemberMap.
        # Per-group GPU layouts have GROUP-LOCAL num_layer (e.g. c4 group has
        # only 21 layers, c128 has 20), but ``layer_indices`` are absolute
        # layer ids in [0, total_stage_layers).  So ``primary_layout`` MUST
        # carry the FULL stage layer count, not the first group's count.
        primary_layout = KVCacheLayout(
            type=gpu_layouts[0].type,
            num_layer=self.rank_info.num_layers_per_pp_stage,
            num_block=gpu_layouts[0].num_block,
            tokens_per_block=gpu_layouts[0].tokens_per_block,
            num_head=gpu_layouts[0].num_head,
            head_size=gpu_layouts[0].head_size,
            is_mla=gpu_layouts[0].is_mla,
        )

        logger.info(
            f"[FlexKV-DSv4] Submitting registration: "
            f"num_groups={len(layer_groups)}, "
            f"total_buffers={len(all_gpu_blocks)}, "
            f"page_size_full={page_size_full}, is_mla={is_mla}"
        )
        self.tp_client.register_to_server(
            kv_caches=all_gpu_blocks,
            kv_layout=primary_layout,
            layer_groups=layer_groups,
            gpu_layouts=gpu_layouts,
            handles_per_group=handles_per_group,
        )
        logger.info(
            "[FlexKV-DSv4] Registered DSv4 multi-pool KV caches to server"
        )

    def _init_layer_transfer_components(self):
        if not self.enable_layerwise_transfer:
            self._layer_done_counter = None
            self._worker_connected = False
            logger.debug(f"[FlexKV] Layerwise transfer disabled{self._rank_label}")
            return

        self._layer_done_counter = FlexKVLayerDoneCounter(self.rank_info.num_layers_per_pp_stage)

        # Compute the set of layer indices (PP-stage-local) whose layerwise
        # eventfds FlexKV's C++ multi-group worker will NOT fire.
        #
        # Reason: the multi-group worker iterates ``layer_members[orig]`` and
        # only schedules a layer_done callback when that list is non-empty.
        # Layers whose ``compress_ratio == 0`` (DSv4 dense MLA layers, e.g.
        # layer 0 and 1 in DSv4-Flash) are NOT in any LayerGroupSpec
        # (c4 / c128 / c4_indexer all require ratio in {4, 128}), so their
        # ``layer_members`` are empty and FlexKV omits the eventfd write.
        #
        # However, sglang's DSv4 forward calls
        # ``get_swa_key_buffer_radix(layer_id)`` for ALL 43 attention layers
        # — including the dense ones — which triggers
        # ``layer_transfer_counter.wait_until(layer_id)`` →
        # ``eventfd_read``. With no writer, the read blocks forever and
        # the scheduler hangs.
        #
        # The fix is to pre-fire the eventfds of these dense layers right
        # after each ``update_producer`` call, so that ``wait_until`` for
        # those layers returns immediately. The dense layers carry only SWA
        # KV, whose restore is currently disabled (pending the data-plane
        # sidecar; see get_match_swa), so the eventfd is purely a "ready"
        # signal here and pre-firing keeps forward from hanging.
        self._dense_layer_local_ids: List[int] = []
        if getattr(self, "_is_dsv4", False) and self._dsv4_kvcache is not None:
            compression_ratios = getattr(self._dsv4_kvcache, "compression_ratios", None)
            stage_start = getattr(self._dsv4_kvcache, "_stage_start", 0)
            stage_end = getattr(
                self._dsv4_kvcache, "_stage_end",
                len(compression_ratios) if compression_ratios is not None else 0,
            )
            if compression_ratios is not None:
                for absolute_layer in range(stage_start, stage_end):
                    if compression_ratios[absolute_layer] == 0:
                        self._dense_layer_local_ids.append(absolute_layer - stage_start)
        if self._dense_layer_local_ids:
            logger.info(
                f"[FlexKV] Detected %d dense layer(s) (PP-stage-local ids=%s) "
                f"whose layerwise eventfds will be pre-fired per H2D so "
                f"sglang's wait_until does not hang. Total layers in stage=%d.",
                len(self._dense_layer_local_ids),
                self._dense_layer_local_ids,
                self.rank_info.num_layers_per_pp_stage,
            )

        self._send_eventfds_to_worker()
        logger.info(f"[FlexKV] Initialized layerwise transfer{self._rank_label}")

    def _signal_dense_layers_ready(self, producer_id: int) -> None:
        """Pre-fire eventfds for layers FlexKV does not manage.

        FlexKV's multi-group worker only fires eventfds for layers that have
        at least one ``LayerGroupSpec`` member (c4/c128/c4_indexer). DSv4
        dense layers (compress_ratio=0) have empty ``layer_members`` and get
        no eventfd write, so sglang's per-layer ``wait_until`` would block.
        These dense layers carry only SWA KV, whose restore now rides the
        FlexKV transfer engine (data plane; see get_match_swa), so the eventfd
        here is purely a "ready" signal and pre-firing it keeps forward from
        hanging.

        Must be called AFTER ``reset_for_new_transfer`` on the same counter
        slot. Writes ``1`` (not 2) so that with sglang's ``wait_remaining=1``
        the eventfd counter stays balanced across rounds (one write per
        round, one read per round, no accumulation).
        """
        if not self._dense_layer_local_ids:
            return
        event = self._layer_done_counter.events[producer_id]
        # eventfd ABI: write 8-byte little-endian uint64 to increment the
        # semaphore counter.
        for layer in self._dense_layer_local_ids:
            fd = event.load_event_fds[layer]
            if fd < 0:
                continue
            try:
                os.write(fd, (1).to_bytes(8, byteorder="little"))
            except OSError as e:
                logger.warning(
                    f"[FlexKV] Failed to pre-fire dense layer eventfd: "
                    f"layer={layer}, counter={producer_id}, fd={fd}, err={e}"
                )

    def send_slot_mapping_to_remote(self, task_id: int, slot_mapping: torch.Tensor) -> None:
        """Send slot_mapping to TransferManagerOnRemote via existing ZMQ channel (PP1 side only).

        In cross-node PP mode, PP1's FlexKVConnector sends slot_mapping
        via KVTPClient.send_to_server -> TransferManagerOnRemote.command_socket,
        so that the remote side can call set_gpu_blocks() with its own local GPU block_ids.
        """
        slot_mapping_np = slot_mapping.cpu().to(torch.int64).numpy() if slot_mapping.is_cuda else slot_mapping.numpy()
        self.tp_client.set_slot_mapping(task_id, slot_mapping_np)
        logger.debug(
            f"[FlexKV] send_slot_mapping_to_remote: "
            f"sent task_id={task_id} to TransferManagerOnRemote"
        )

    def _send_eventfds_to_worker(
        self, retry_interval: float = 1.0
    ):
        max_retries = self.layerwise_eventfd_connect_max_retries
        # Allow up to 3 full connect+send attempts before giving up.
        max_send_retries = 3
        logger.info(
            f"[FlexKV] Attempting eventfd connection{self._rank_label}: "
            f"socket={self.layerwise_eventfd_socket}, max_retries={max_retries}")

        last_error = None
        for send_attempt in range(max_send_retries):
            sock = None
            try:
                # Phase 1: Connect to the worker socket (retry until ready).
                for attempt in range(max_retries):
                    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    try:
                        sock.connect(self.layerwise_eventfd_socket)
                        logger.info(
                            f"[FlexKV] Eventfd connected{self._rank_label}: "
                            f"socket={self.layerwise_eventfd_socket}, "
                            f"attempts={attempt + 1}"
                            f"{f', send_retry={send_attempt}' if send_attempt > 0 else ''}"
                        )
                        break
                    except (FileNotFoundError, ConnectionRefusedError) as e:
                        sock.close()
                        sock = None
                        if attempt == max_retries - 1:
                            logger.error(
                                f"[FlexKV] Eventfd connection failed{self._rank_label}: "
                                f"socket={self.layerwise_eventfd_socket}, "
                                f"attempts={max_retries}, error={type(e).__name__}"
                            )
                            raise RuntimeError(
                                f"[FlexKV] Failed to connect to eventfd socket "
                                f"{self.layerwise_eventfd_socket} after {max_retries} attempts"
                            )
                        if attempt % 10 == 0:
                            socket_exists = os.path.exists(self.layerwise_eventfd_socket)
                            logger.debug(
                                f"[FlexKV] Eventfd connect retry{self._rank_label}: "
                                f"socket={self.layerwise_eventfd_socket}, "
                                f"attempt={attempt + 1}/{max_retries}, "
                                f"error={type(e).__name__}, socket_exists={socket_exists}"
                            )
                        time.sleep(retry_interval)

                if sock is None:
                    raise RuntimeError(
                        f"[FlexKV] Eventfd socket unavailable after {max_retries} attempts: "
                        f"{self.layerwise_eventfd_socket}"
                    )

                # Phase 2: Send metadata + eventfds over the connected socket.
                # UDS is node-local, so use _per_node TP rank/size so that
                # LayerwiseWorker builds the correct eventfd tensor shape.
                num_counters = self._layer_done_counter.num_counters
                model_config = self.flexkv_config.model_config
                rank_info = self.rank_info
                # Send 16-byte metadata: tp_rank_per_node, tp_size_per_node, num_layers, num_counters
                metadata = struct.pack(
                    "iiii",
                    rank_info.tp_rank_per_node,
                    model_config.tp_size_per_node,
                    rank_info.num_layers_per_pp_stage,
                    num_counters,
                )
                sock.sendall(metadata)
                logger.debug(
                    f"[FlexKV] Eventfd metadata sent{self._rank_label}: "
                    f"tp_rank_per_node={rank_info.tp_rank_per_node}, "
                    f"tp_size_per_node={model_config.tp_size_per_node}, "
                    f"num_layers={rank_info.num_layers_per_pp_stage}, num_counters={num_counters}"
                )

                for counter_id in range(num_counters):
                    fds = self._layer_done_counter.events[counter_id].load_event_fds
                    send_fds(sock, fds, struct.pack("i", counter_id))
                    logger.debug(
                        f"[FlexKV] Eventfd fds sent{self._rank_label}: "
                        f"counter_id={counter_id}, num_fds={len(fds)}"
                    )

                # Wait for ACK from server to confirm fds were received.
                sock.settimeout(30.0)
                try:
                    ack = sock.recv(1)
                except socket.timeout:
                    raise RuntimeError("Timed out waiting for ACK from FlexKV worker")
                if not ack or ack[0] != 1:
                    raise RuntimeError(
                        f"FlexKV worker NACK'd eventfd transfer (ack={ack!r})"
                    )

                self._worker_connected = True
                logger.info(
                    f"[FlexKV] Eventfd setup complete{self._rank_label}: "
                    f"socket={self.layerwise_eventfd_socket}, "
                    f"counters={num_counters}, layers={rank_info.num_layers_per_pp_stage}"
                )
                return
            except Exception as e:
                last_error = e
                logger.warning(
                    f"[FlexKV] Failed to send eventfds{self._rank_label} "
                    f"(send_attempt {send_attempt + 1}/{max_send_retries}): "
                    f"socket={self.layerwise_eventfd_socket}, error={e}. "
                    f"Will reconnect and retry..."
                )
            finally:
                if sock is not None:
                    sock.close()
                # Brief pause before reconnecting.
                time.sleep(retry_interval)

        # All send retries exhausted.
        logger.error(
            f"[FlexKV] Failed to send eventfds{self._rank_label} after "
            f"{max_send_retries} attempts: "
            f"socket={self.layerwise_eventfd_socket}, last_error={last_error}",
            exc_info=True,
        )
        raise RuntimeError(
            f"[FlexKV] Failed to send eventfds to {self.layerwise_eventfd_socket} "
            f"after {max_send_retries} attempts: {last_error}"
        )
