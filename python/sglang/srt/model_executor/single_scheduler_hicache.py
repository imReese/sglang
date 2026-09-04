# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""HiCache L2 bridge for the single-Scheduler / multi-ModelRunner topology.

The Scheduler keeps the existing HiRadixCache control plane and canonical
integer slot ids. Each ModelRunner owns the real device KV pool, host pool,
and HiCacheController data path. This bridge only turns the old in-process
D<->H controller calls into synchronous fan-out calls to the eight runners.

This first slice intentionally supports L2 only (no storage/L3) and the legacy
flat MHA/MLA HiRadixCache layouts. DSA/DeepSeek-V4 sidecar pools and hybrid
SWA/SSM remain fail-closed until their extra pools are mirrored explicitly.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Callable, Optional

import torch

from sglang.srt.managers.cache_controller import CacheOperation, HiCacheController
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.hicache_storage import PoolName, PrefetchTimeoutConfig
from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, MLATokenToKVPool
from sglang.srt.mem_cache.pool_host.common import get_allocator_type
from sglang.srt.mem_cache.pool_host.mha import get_mha_host_pool_cls
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.runtime_context import get_memory


class SingleSchedulerHiCacheError(RuntimeError):
    pass


class _ImmediateEvent:
    """Event-shaped completion used after a synchronous 8-rank fan-out."""

    def query(self) -> bool:
        return True

    def synchronize(self) -> None:
        return None

    def elapsed_time(self, _other) -> float:
        return 0.0


@dataclass(slots=True)
class _ImmediateAck:
    node_ids: list[int]
    num_tokens: int
    num_bytes: int
    num_tokens_by_pool: dict[str, int]
    timing_enabled: bool = False
    start_event: object = field(init=False)
    finish_event: object = field(init=False)

    def __post_init__(self) -> None:
        self.start_event = _ImmediateEvent()
        self.finish_event = _ImmediateEvent()


class RemoteHostPool:
    """Scheduler-side host-slot metadata; the bytes live in ModelRunner processes."""

    def __init__(
        self,
        *,
        size: int,
        logical_size: int,
        page_size: int,
        size_per_token: int,
    ) -> None:
        self.size = int(size)
        self.logical_size = int(logical_size)
        self.page_size = int(page_size)
        self.size_per_token = int(size_per_token)
        self.layout = "remote"
        self._allocated: set[int] = set()
        self._controller = None

    def bind(self, controller) -> None:
        self._controller = controller

    def mark_allocated(self, indices: torch.Tensor) -> None:
        self._allocated.update(map(int, indices.to("cpu").tolist()))

    def mark_freed(self, indices: torch.Tensor) -> None:
        for idx in indices.to("cpu").tolist():
            self._allocated.discard(int(idx))

    def available_size(self) -> int:
        return max(0, self.logical_size - len(self._allocated))

    def clear(self) -> None:
        self._allocated.clear()

    def free(self, indices: torch.Tensor) -> None:
        if self._controller is None:
            self.mark_freed(indices)
            return
        self._controller.evict_host(indices)

    def destroy(self) -> None:
        self._allocated.clear()


class RemoteHiCacheController:
    """HiRadixCache-compatible L2 controller backed by eight runner controllers."""

    def __init__(
        self,
        *,
        fanout: Callable,
        device_allocator,
        host_size: int,
        host_logical_size: int,
        host_page_size: int,
        host_size_per_token: int,
        write_policy: str,
    ) -> None:
        self._fanout = fanout
        self.mem_pool_device_allocator = device_allocator
        self.mem_pool_host = RemoteHostPool(
            size=host_size,
            logical_size=host_logical_size,
            page_size=host_page_size,
            size_per_token=host_size_per_token,
        )
        self.mem_pool_host.bind(self)
        self.write_policy = write_policy
        self.enable_storage = False
        self.storage_backend = None
        self.storage_backend_type = None
        self.ack_write_queue = []
        self.ack_load_queue = []
        self.layer_done_counter = SimpleNamespace(events=[])

    def _ack(self, *, node_id: int, num_tokens: int) -> _ImmediateAck:
        return _ImmediateAck(
            node_ids=[node_id],
            num_tokens=num_tokens,
            num_bytes=num_tokens * self.mem_pool_host.size_per_token,
            num_tokens_by_pool={PoolName.KV.value: num_tokens},
        )

    def reset(self) -> None:
        self._fanout("hicache_clear")
        self.ack_write_queue.clear()
        self.ack_load_queue.clear()
        self.mem_pool_host.clear()

    def write(
        self,
        device_indices: torch.Tensor,
        priority: Optional[int] = None,
        node_id: int = -1,
        **kwargs,
    ) -> Optional[torch.Tensor]:
        del priority
        if kwargs:
            raise SingleSchedulerHiCacheError(
                "single-Scheduler HiCache L2 does not support auxiliary/sidecar pools yet"
            )
        replies = self._fanout(
            "hicache_write",
            device_indices=device_indices.to("cpu", dtype=torch.int64),
            node_id=node_id,
        )
        host_indices = [reply for reply in replies if reply is not None]
        if not host_indices:
            return None
        if len(host_indices) != len(replies):
            raise SingleSchedulerHiCacheError(
                "HiCache host allocation diverged across ModelRunner ranks"
            )
        first = host_indices[0].to("cpu", dtype=torch.int64)
        for other in host_indices[1:]:
            if not torch.equal(first, other.to("cpu", dtype=torch.int64)):
                raise SingleSchedulerHiCacheError(
                    "HiCache canonical host indices diverged across ModelRunner ranks"
                )
        self.mem_pool_host.mark_allocated(first)
        self.ack_write_queue.append(self._ack(node_id=node_id, num_tokens=len(first)))
        return first

    def load(
        self,
        host_indices: torch.Tensor,
        priority: Optional[int] = None,
        node_id: int = -1,
        **kwargs,
    ) -> Optional[torch.Tensor]:
        del priority
        if kwargs:
            raise SingleSchedulerHiCacheError(
                "single-Scheduler HiCache L2 does not support auxiliary/sidecar pools yet"
            )
        host_indices = host_indices.to("cpu", dtype=torch.int64)
        device_indices = self.mem_pool_device_allocator.alloc(len(host_indices))
        if device_indices is None:
            return None
        device_indices = device_indices.to("cpu", dtype=torch.int64)
        try:
            self._fanout(
                "hicache_load",
                host_indices=host_indices,
                device_indices=device_indices,
                node_id=node_id,
            )
        except Exception:
            self.mem_pool_device_allocator.free(device_indices)
            raise
        self.ack_load_queue.append(
            self._ack(node_id=node_id, num_tokens=len(device_indices))
        )
        return device_indices

    def start_loading(self) -> int:
        return -1

    def evict_device(self, device_indices: torch.Tensor) -> int:
        self.mem_pool_device_allocator.free(device_indices)
        return len(device_indices)

    def evict_host(self, host_indices: torch.Tensor, backup_only: bool = True) -> int:
        if not backup_only:
            raise ValueError("Other eviction policies are not supported yet.")
        host_indices = host_indices.to("cpu", dtype=torch.int64)
        self._fanout("hicache_free_host", host_indices=host_indices)
        self.mem_pool_host.mark_freed(host_indices)
        return len(host_indices)

    def get_attn_cp_rank_and_size(self) -> tuple[int, int]:
        return 0, 1


def init_runner_hicache_l2(runner, server_args):
    """Create the real rank-local HiCacheController and host KV pool."""

    if not server_args.enable_hierarchical_cache:
        return None, None
    if server_args.hicache_storage_backend is not None:
        raise SingleSchedulerHiCacheError(
            "single-Scheduler HiCache currently supports L2 only; "
            "hicache_storage_backend/L3 is not supported"
        )

    allocator = runner.token_to_kv_pool_allocator
    kv_cache = allocator.get_kvcache()
    page_size = allocator.page_size
    allocator_type = get_allocator_type(server_args)

    if isinstance(kv_cache, MHATokenToKVPool):
        host_pool = get_mha_host_pool_cls(kv_cache)(
            kv_cache,
            get_memory().hicache_ratio,
            server_args.hicache_size,
            page_size,
            server_args.hicache_mem_layout,
            allocator_type=allocator_type,
        )
    elif isinstance(kv_cache, MLATokenToKVPool):
        host_pool = MLATokenToKVPoolHost(
            kv_cache,
            get_memory().hicache_ratio,
            server_args.hicache_size,
            page_size,
            server_args.hicache_mem_layout,
            allocator_type=allocator_type,
            dcp_size=1,
            dcp_rank=0,
        )
    else:
        raise SingleSchedulerHiCacheError(
            "single-Scheduler HiCache L2 currently supports flat MHA/MLA KV pools; "
            f"got {type(kv_cache).__name__}. DSA/DeepSeek-V4 and hybrid sidecar "
            "pools need their extra pools mirrored in a follow-up."
        )

    controller = HiCacheController(
        allocator,
        host_pool,
        page_size,
        tp_group=None,
        load_cache_event=threading.Event(),
        attn_cp_group=None,
        attn_tp_group=None,
        pp_group=None,
        write_policy=server_args.hicache_write_policy,
        io_backend=server_args.hicache_io_backend,
        storage_backend=None,
        model_name=server_args.served_model_name,
        enable_storage_metrics=False,
    )
    info = dict(
        host_size=int(host_pool.size),
        host_logical_size=int(getattr(host_pool, "logical_size", host_pool.size)),
        host_page_size=int(host_pool.page_size),
        host_size_per_token=int(host_pool.size_per_token),
    )
    return controller, info


def execute_runner_hicache_command(
    controller,
    *,
    kind: str,
    host_indices: Optional[torch.Tensor],
    device_indices: Optional[torch.Tensor],
    node_id: int,
) -> Optional[torch.Tensor]:
    """Execute one physical L2 operation in a ModelRunner process."""

    if controller is None:
        raise SingleSchedulerHiCacheError("HiCache command received without HiCache")

    if kind == "hicache_write":
        assert device_indices is not None
        device_indices = device_indices.to(controller.device, dtype=torch.int64)
        allocated = controller.write(device_indices=device_indices, node_id=node_id)
        if allocated is None:
            return None
        ack = controller.ack_write_queue.pop(0)
        ack.finish_event.synchronize()
        return allocated.to("cpu", dtype=torch.int64)

    if kind == "hicache_load":
        assert host_indices is not None and device_indices is not None
        op = CacheOperation(
            host_indices.to("cpu", dtype=torch.int64),
            device_indices.to(controller.device, dtype=torch.int64),
            node_id,
        )
        controller.load_queue.append(op)
        controller.start_loading()
        ack = controller.ack_load_queue.pop(0)
        ack.finish_event.synchronize()
        return None

    if kind == "hicache_free_host":
        assert host_indices is not None
        controller.evict_host(host_indices.to("cpu", dtype=torch.int64))
        return None

    if kind == "hicache_clear":
        controller.reset()
        controller.mem_pool_host.clear()
        return None

    raise SingleSchedulerHiCacheError(f"unknown HiCache runner command: {kind}")


def build_scheduler_hiradix_cache(
    *,
    server_args,
    req_to_token_pool,
    token_to_kv_pool_allocator,
    remote_controller: RemoteHiCacheController,
    page_size: int,
    disable: bool,
    enable_metrics: bool,
):
    """Build the existing HiRadixCache as a Scheduler-only control plane."""

    params = CacheInitParams(
        disable=disable,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        page_size=page_size,
        is_eagle=False,
        tp_cache_group=None,
        attn_cp_cache_group=None,
        attn_tp_cache_group=None,
        pp_cache_group=None,
        eviction_policy=get_memory().radix_eviction_policy,
        enable_metrics=enable_metrics,
        enable_kv_cache_events=False,
        enable_session_radix_cache=False,
    )

    cache = HiRadixCache.__new__(HiRadixCache)
    cache._enable_metrics_flag = enable_metrics
    cache.page_size = page_size
    cache.kv_cache = None
    cache.token_to_kv_pool_host = remote_controller.mem_pool_host
    cache.tp_group = None
    cache.attn_cp_group = None
    cache.attn_tp_group = None
    cache.pp_group = None
    cache.tp_world_size = 1
    cache.pp_rank = 0
    cache.pp_size = 1
    cache.enable_storage = False
    cache.enable_storage_metrics = False
    cache.storage_metrics_collector = None
    cache.extra_metric_labels = server_args.extra_metric_labels
    cache.prefetch_threshold = 256
    cache.prefetch_timeout_config = PrefetchTimeoutConfig()
    cache.hicache_storage_pass_prefix_keys = False
    cache.is_prefetch_timeout = cache._prefetch_timeout_check_linear_func
    cache.prefetch_stop_policy = server_args.hicache_storage_prefetch_policy
    cache.load_cache_event = threading.Event()
    cache.cache_controller = remote_controller
    cache.ongoing_write_through = {}
    cache.ongoing_load_back = {}
    cache.ongoing_prefetch = {}
    cache.ongoing_backup = {}
    cache.prefetch_loaded_tokens_by_reqid = {}
    cache.work_list = []
    cache.write_through_threshold = (
        1 if server_args.hicache_write_policy == "write_through" else 2
    )
    cache.load_back_threshold = 10
    cache.evictable_host_leaves = set()

    RadixCache.__init__(cache, params=params)
    return cache
