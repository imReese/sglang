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
"""Thin process executor for the single-Scheduler TP8/EP8 refactor.

The Scheduler owns request/cache metadata and chooses canonical integer request
and KV slot ids once. Eight rank-local ModelRunner processes own all CUDA/NCCL
state and mirror those integer ids into their local ReqToTokenPool before each
forward. The Scheduler itself never joins the model distributed world.

The first slice is synchronous and supports normal generation plus the standard
HiCache L2 data path. Token and paged KV allocators preserve the model/backend
resolved page size; HiCache physical host/device bytes stay rank-local.
"""

from __future__ import annotations

import dataclasses
import logging
import traceback

import setproctitle
from dataclasses import dataclass, field
from multiprocessing.connection import Connection
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Optional

import torch

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.debug_utils.pr_fix_toggle import maybe_revert_pr_fix
from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.layers.moe import initialize_moe_config
from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config
from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
from sglang.srt.layers.quantization.unquant import initialize_bf16_gemm_config
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.mem_cache.allocator import (
    PagedTokenToKVPoolAllocator,
    TokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.model_executor.single_scheduler_hicache import (
    RemoteHiCacheController,
    execute_runner_hicache_command,
    init_runner_hicache_l2,
)
from sglang.srt.plugins import load_plugins
from sglang.srt.runtime_context import get_schedule, publish
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import (
    configure_logger,
    get_available_gpu_memory,
    kill_itself_when_parent_died,
    set_random_seed,
)

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner


logger = logging.getLogger(__name__)


class SingleSchedulerExecutorError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RunnerInitInfo:
    rank: int
    max_total_num_tokens: int = 0
    max_running_requests: int = 0
    req_pool_size: int = 0
    max_context_len: int = 0
    allocator_need_sort: bool = False
    allocator_dtype: torch.dtype = torch.int64
    allocator_page_size: int = 1
    weight_load_time: float = 0.0
    weight_load_mem_usage: float = 0.0
    graph_memory_usage: dict[str, float] = field(default_factory=dict)
    graph_time_usage: dict[str, float] = field(default_factory=dict)
    available_gpu_memory_gb: float = 0.0
    hicache_enabled: bool = False
    hicache_host_size: int = 0
    hicache_host_logical_size: int = 0
    hicache_host_page_size: int = 1
    hicache_host_size_per_token: int = 0
    error: Optional[str] = None
    remote_traceback: Optional[str] = None


@dataclass(frozen=True, slots=True)
class _RunnerReq:
    """Only the request fields ForwardBatch.init_new reads in this pilot."""

    rid: str
    lora_id: None = None
    grammar: None = None
    bootstrap_room: None = None
    token_type_ids: None = None
    dimensions: None = None
    return_pooled_hidden_states: bool = False
    multi_item_delimiter_indices: None = None


@dataclass(slots=True)
class ModelRunnerBatch:
    """Picklable execution snapshot derived from one Scheduler-owned batch.

    It is deliberately not a second scheduling model. It only strips process-
    local objects from ScheduleBatch and converts its existing execution fields
    to CPU tensors. For EXTEND, canonical req->KV rows are included so runners
    can mirror prefix-cache mappings. For DECODE, only the newly allocated slot
    needs to be written at seq_len-1.
    """

    forward_mode: ForwardMode
    rids: tuple[str, ...]
    input_ids: torch.Tensor
    req_pool_indices: torch.Tensor
    seq_lens: torch.Tensor
    orig_seq_lens: torch.Tensor
    out_cache_loc: torch.Tensor
    seq_lens_sum: int
    prefix_lens: Optional[list[int]]
    extend_lens: Optional[list[int]]
    extend_num_tokens: Optional[int]
    sampling_info: SamplingBatchInfo
    kv_rows: Optional[tuple[torch.Tensor, ...]] = None

    @classmethod
    def from_schedule_batch(cls, batch: ScheduleBatch) -> "ModelRunnerBatch":
        if not batch.spec_algorithm.is_none():
            raise SingleSchedulerExecutorError(
                "single-Scheduler runner split does not support speculative decoding yet"
            )
        if batch.forward_mode not in (ForwardMode.EXTEND, ForwardMode.DECODE):
            raise SingleSchedulerExecutorError(
                f"unsupported forward mode in first runner split: {batch.forward_mode}"
            )
        if batch.return_logprob or batch.return_hidden_states:
            raise SingleSchedulerExecutorError(
                "return_logprob/return_hidden_states are not supported in the first runner split"
            )
        if batch.has_grammar:
            raise SingleSchedulerExecutorError(
                "grammar is not supported in the first runner split"
            )
        if any(req.lora_id is not None for req in batch.reqs):
            raise SingleSchedulerExecutorError(
                "LoRA is not supported in the first runner split"
            )
        if any(req.custom_logit_processor is not None for req in batch.reqs):
            raise SingleSchedulerExecutorError(
                "custom logit processors are not supported in the first runner split"
            )
        if any(req.return_sampling_mask for req in batch.reqs):
            raise SingleSchedulerExecutorError(
                "return_sampling_mask is not supported in the first runner split"
            )
        if any(req.return_routed_experts for req in batch.reqs):
            raise SingleSchedulerExecutorError(
                "return_routed_experts is not supported in the first runner split"
            )
        if any(req.return_indexer_topk for req in batch.reqs):
            raise SingleSchedulerExecutorError(
                "return_indexer_topk is not supported in the first runner split"
            )
        if batch.multimodal_inputs and any(x is not None for x in batch.multimodal_inputs):
            raise SingleSchedulerExecutorError(
                "multimodal inputs are not supported in the first runner split"
            )
        if batch.sampling_info is None:
            raise SingleSchedulerExecutorError("batch has no sampling_info")

        def cpu(t: torch.Tensor) -> torch.Tensor:
            return t.detach().to("cpu").contiguous()

        sampling_info = batch.sampling_info.copy_for_forward()
        for f in dataclasses.fields(sampling_info):
            value = getattr(sampling_info, f.name)
            if isinstance(value, torch.Tensor):
                setattr(sampling_info, f.name, cpu(value))
        sampling_info.device = "cpu"

        req_pool_indices = cpu(batch.req_pool_indices).to(torch.int64)
        seq_lens = cpu(batch.seq_lens).to(torch.int64)
        out_cache_loc = cpu(batch.out_cache_loc).to(torch.int64)
        orig_seq_lens = cpu(batch.orig_seq_lens).to(torch.int32)
        input_ids = cpu(batch.input_ids).to(torch.int64)

        kv_rows = None
        if batch.forward_mode.is_extend():
            rows = []
            for req_slot, seq_len in zip(req_pool_indices.tolist(), seq_lens.tolist()):
                rows.append(
                    batch.req_to_token_pool.req_to_token[
                        req_slot, :seq_len
                    ].detach().to("cpu").to(torch.int32).contiguous()
                )
            kv_rows = tuple(rows)

        return cls(
            forward_mode=batch.forward_mode,
            rids=tuple(req.rid for req in batch.reqs),
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            orig_seq_lens=orig_seq_lens,
            out_cache_loc=out_cache_loc,
            seq_lens_sum=(
                int(batch.seq_lens_sum)
                if batch.seq_lens_sum is not None
                else int(seq_lens.sum().item())
            ),
            prefix_lens=(list(batch.prefix_lens) if batch.prefix_lens is not None else None),
            extend_lens=(list(batch.extend_lens) if batch.extend_lens is not None else None),
            extend_num_tokens=batch.extend_num_tokens,
            sampling_info=sampling_info,
            kv_rows=kv_rows,
        )


@dataclass(frozen=True, slots=True)
class RunnerCommand:
    kind: str
    batch: Optional[ModelRunnerBatch] = None
    host_indices: Optional[torch.Tensor] = None
    device_indices: Optional[torch.Tensor] = None
    node_id: int = -1


@dataclass(frozen=True, slots=True)
class RunnerReply:
    rank: int
    result: Optional[GenerationBatchResult] = None
    host_indices: Optional[torch.Tensor] = None
    error: Optional[str] = None
    remote_traceback: Optional[str] = None


class _MetadataKVCache:
    mem_usage = 0

    def maybe_get_custom_mem_pool(self):
        return None


class ModelExecutor:
    """Very thin Scheduler -> eight ModelRunner process dispatcher."""

    def __init__(
        self,
        *,
        connections: list[Connection],
        model_config: ModelConfig,
        server_args: ServerArgs,
    ) -> None:
        if len(connections) != 8:
            raise SingleSchedulerExecutorError(
                f"DP1/TP8/EP8 pilot requires 8 ModelRunner connections, got {len(connections)}"
            )
        self._connections = tuple(connections)
        infos = tuple(conn.recv() for conn in self._connections)
        if not all(isinstance(info, RunnerInitInfo) for info in infos):
            raise SingleSchedulerExecutorError("invalid ModelRunner startup handshake")
        infos = tuple(sorted(infos, key=lambda x: x.rank))
        if tuple(info.rank for info in infos) != tuple(range(8)):
            raise SingleSchedulerExecutorError(
                f"runner ranks must be exactly 0..7, got {[info.rank for info in infos]}"
            )
        failed = [info for info in infos if info.error]
        if failed:
            info = failed[0]
            detail = info.error
            if info.remote_traceback:
                detail += f"\nRemote traceback:\n{info.remote_traceback}"
            raise SingleSchedulerExecutorError(
                f"ModelRunner rank {info.rank} failed during startup: {detail}"
            )

        self.runner_infos = infos
        leader = infos[0]
        geometry = {
            (
                info.max_total_num_tokens,
                info.max_running_requests,
                info.req_pool_size,
                info.max_context_len,
                info.allocator_page_size,
            )
            for info in infos
        }
        if len(geometry) != 1:
            raise SingleSchedulerExecutorError(
                f"ModelRunner pool geometry diverged across ranks: {sorted(geometry)}"
            )
        if any(info.allocator_need_sort != leader.allocator_need_sort for info in infos):
            raise SingleSchedulerExecutorError("runner allocator ordering diverged")

        self.req_to_token_pool = ReqToTokenPool(
            size=leader.req_pool_size,
            max_context_len=leader.max_context_len,
            device="cpu",
            enable_memory_saver=False,
        )
        if leader.allocator_page_size == 1:
            self.token_to_kv_pool_allocator = TokenToKVPoolAllocator(
                size=leader.max_total_num_tokens,
                dtype=leader.allocator_dtype,
                device="cpu",
                kvcache=_MetadataKVCache(),
                need_sort=leader.allocator_need_sort,
            )
        else:
            self.token_to_kv_pool_allocator = PagedTokenToKVPoolAllocator(
                size=leader.max_total_num_tokens,
                page_size=leader.allocator_page_size,
                dtype=leader.allocator_dtype,
                device="cpu",
                kvcache=_MetadataKVCache(),
                need_sort=leader.allocator_need_sort,
            )

        self.model_config = model_config
        self.server_args = server_args
        self.is_hybrid_swa = False
        self.sliding_window_size = None
        self.weight_load_time = max(info.weight_load_time for info in infos)
        self.graph_memory_usage = dict(leader.graph_memory_usage)
        self.graph_time_usage = dict(leader.graph_time_usage)
        self.model_runner = SimpleNamespace(
            model_config=model_config,
            mtp_draft_device_pools=(),
            canary_manager=None,
            prefill_aware_swa=False,
            weight_load_mem_usage=leader.weight_load_mem_usage,
            server_args=server_args,
            ngram_embedding_manager=SimpleNamespace(enabled=False),
            attn_backend=SimpleNamespace(),
        )

        self.hicache_controller = None
        if server_args.enable_hierarchical_cache:
            if not all(info.hicache_enabled for info in infos):
                raise SingleSchedulerExecutorError(
                    "HiCache was requested but at least one ModelRunner did not initialize L2"
                )
            host_geometry = {
                (
                    info.hicache_host_size,
                    info.hicache_host_logical_size,
                    info.hicache_host_page_size,
                    info.hicache_host_size_per_token,
                )
                for info in infos
            }
            if len(host_geometry) != 1:
                raise SingleSchedulerExecutorError(
                    f"HiCache host-pool geometry diverged across ranks: {sorted(host_geometry)}"
                )
            self.hicache_controller = RemoteHiCacheController(
                fanout=self._run_hicache_command,
                device_allocator=self.token_to_kv_pool_allocator,
                host_size=leader.hicache_host_size,
                host_logical_size=leader.hicache_host_logical_size,
                host_page_size=leader.hicache_host_page_size,
                host_size_per_token=leader.hicache_host_size_per_token,
                write_policy=server_args.hicache_write_policy,
            )

    @property
    def leader_info(self) -> RunnerInitInfo:
        return self.runner_infos[0]

    def get_memory_pool(self):
        return self.req_to_token_pool, self.token_to_kv_pool_allocator

    def get_tokens_per_layer_info(self):
        return self.leader_info.max_total_num_tokens, None

    def get_pad_input_ids_func(self):
        return None

    def register_hicache_layer_transfer_counter(self, _counter) -> None:
        return None

    def get_hicache_controller(self):
        return self.hicache_controller

    def _collect_replies(self) -> list[RunnerReply]:
        replies = [conn.recv() for conn in self._connections]
        for expected_rank, reply in enumerate(replies):
            if not isinstance(reply, RunnerReply) or reply.rank != expected_rank:
                raise SingleSchedulerExecutorError(
                    f"invalid reply for runner rank {expected_rank}: {reply!r}"
                )
            if reply.error is not None:
                detail = reply.error
                if reply.remote_traceback:
                    detail += f"\nRemote traceback:\n{reply.remote_traceback}"
                raise SingleSchedulerExecutorError(
                    f"ModelRunner rank {expected_rank} failed: {detail}"
                )
        return replies

    def _run_hicache_command(
        self,
        kind: str,
        *,
        host_indices: Optional[torch.Tensor] = None,
        device_indices: Optional[torch.Tensor] = None,
        node_id: int = -1,
    ) -> list[Optional[torch.Tensor]]:
        command = RunnerCommand(
            kind=kind,
            host_indices=host_indices,
            device_indices=device_indices,
            node_id=node_id,
        )
        for conn in self._connections:
            conn.send(command)
        return [reply.host_indices for reply in self._collect_replies()]

    def run_batch(self, batch: ScheduleBatch) -> GenerationBatchResult:
        wire_batch = ModelRunnerBatch.from_schedule_batch(batch)
        command = RunnerCommand(kind="execute", batch=wire_batch)
        for conn in self._connections:
            conn.send(command)

        replies = self._collect_replies()
        result = replies[0].result
        if result is None:
            raise SingleSchedulerExecutorError("rank0 returned no generation result")
        return result

    def shutdown(self) -> None:
        for conn in self._connections:
            try:
                conn.send(RunnerCommand(kind="shutdown"))
            except (BrokenPipeError, EOFError, OSError):
                pass


def _move_sampling_info_to_device(
    info: SamplingBatchInfo, device: str
) -> SamplingBatchInfo:
    for f in dataclasses.fields(info):
        value = getattr(info, f.name)
        if isinstance(value, torch.Tensor):
            setattr(info, f.name, value.to(device, non_blocking=True))
    info.device = device
    return info


def _materialize_batch(batch: ModelRunnerBatch, runner: ModelRunner) -> ScheduleBatch:
    device = runner.device
    req_pool_indices = batch.req_pool_indices.to(device, non_blocking=True)
    seq_lens = batch.seq_lens.to(device, non_blocking=True)
    out_cache_loc = batch.out_cache_loc.to(device, non_blocking=True)

    if batch.forward_mode.is_extend():
        assert batch.kv_rows is not None
        for req_slot, row in zip(batch.req_pool_indices.tolist(), batch.kv_rows):
            runner.req_to_token_pool.write(
                (req_slot, slice(0, row.numel())),
                row.to(device=device, dtype=torch.int32, non_blocking=True),
            )
    else:
        runner.req_to_token_pool.write(
            (req_pool_indices, seq_lens - 1), out_cache_loc.to(torch.int32)
        )

    reqs = [_RunnerReq(rid=rid) for rid in batch.rids]
    sampling_info = _move_sampling_info_to_device(batch.sampling_info, device)

    return ScheduleBatch(
        reqs=reqs,
        req_to_token_pool=runner.req_to_token_pool,
        token_to_kv_pool_allocator=runner.token_to_kv_pool_allocator,
        tree_cache=None,
        model_config=runner.model_config,
        enable_overlap=False,
        device=device,
        input_ids=batch.input_ids.to(device, non_blocking=True),
        req_pool_indices=req_pool_indices,
        req_pool_indices_cpu=batch.req_pool_indices,
        seq_lens=seq_lens,
        seq_lens_cpu=batch.seq_lens,
        orig_seq_lens=batch.orig_seq_lens.to(device, non_blocking=True),
        out_cache_loc=out_cache_loc,
        forward_mode=batch.forward_mode,
        return_logprob=False,
        is_prefill_only=False,
        spec_algorithm=SpeculativeAlgorithm.NONE,
        return_hidden_states=False,
        return_hidden_states_mode=CaptureHiddenMode.NULL,
        has_grammar=False,
        seq_lens_sum=batch.seq_lens_sum,
        extend_num_tokens=batch.extend_num_tokens,
        prefix_lens=batch.prefix_lens,
        extend_lens=batch.extend_lens,
        extend_logprob_start_lens=(
            [0] * len(reqs) if batch.forward_mode.is_extend() else None
        ),
        multimodal_inputs=[None] * len(reqs),
        sampling_info=sampling_info,
    )


def _execute_batch(
    batch: ModelRunnerBatch,
    runner: ModelRunner,
    *,
    return_result: bool,
) -> Optional[GenerationBatchResult]:
    local_batch = _materialize_batch(batch, runner)
    forward_batch = ForwardBatch.init_new(
        local_batch,
        runner,
        return_hidden_states_before_norm=False,
    )
    output = runner.forward(forward_batch)
    next_token_ids = runner.sample(output.logits_output, forward_batch)
    if not return_result:
        return None

    next_token_ids = next_token_ids.detach().to("cpu")
    return GenerationBatchResult(
        logits_output=None,
        next_token_ids=next_token_ids,
        can_run_cuda_graph=output.can_run_graph,
    )


def _init_model_runner_process_config(
    server_args: ServerArgs, model_config: ModelConfig
) -> None:
    config_to_check = getattr(
        model_config.hf_config, "text_config", model_config.hf_config
    )
    moe_topk_attrs = (
        "num_experts_per_tok",
        "num_experts_per_token",
        "top_k_experts",
        "moe_top_k",
        "moe_topk",
    )
    if any(hasattr(config_to_check, attr) for attr in moe_topk_attrs):
        initialize_moe_config(server_args)
    initialize_fp8_gemm_config(server_args)
    initialize_fp4_gemm_config(server_args)
    initialize_bf16_gemm_config(server_args)
    maybe_revert_pr_fix()


def run_model_runner_process(
    server_args: ServerArgs,
    *,
    rank: int,
    gpu_id: int,
    nccl_port: int,
    connection: Connection,
) -> None:
    """Own exactly one rank-local ModelRunner and no Scheduler."""

    try:
        kill_itself_when_parent_died()
        setproctitle.setproctitle(f"sglang::model_runner_TP{rank}_EP{rank}")
        load_plugins()
        publish(server_args, role="scheduler")
        configure_logger(server_args, prefix=f" TP{rank} EP{rank}")
        model_config = ModelConfig.from_server_args(server_args)
        _init_model_runner_process_config(server_args, model_config)
        from sglang.srt.model_executor.model_runner import ModelRunner

        ps = ParallelState(
            tp_rank=rank,
            tp_size=8,
            pp_rank=0,
            pp_size=1,
            dp_rank=None,
            dp_size=1,
            attn_tp_rank=rank,
            attn_tp_size=8,
            attn_cp_rank=0,
            attn_cp_size=1,
            attn_dcp_rank=0,
            attn_dcp_size=1,
            attn_dp_rank=0,
            attn_dp_size=1,
            moe_ep_rank=rank,
            moe_ep_size=8,
            moe_dp_rank=0,
            moe_dp_size=1,
            gpu_id=gpu_id,
        )
        runner = ModelRunner(
            model_config=model_config,
            mem_fraction_static=get_schedule().mem_fraction_static,
            gpu_id=gpu_id,
            ps=ps,
            nccl_port=nccl_port,
            server_args=server_args,
        )
        runner.alloc_memory_pool()
        runner.init_attention_backends()
        runner.init_cuda_graphs()
        if runner.token_to_kv_pool.post_capture_active:
            runner.post_capture_resize_kv_pool()
        set_random_seed(server_args.random_seed)

        allocator = runner.token_to_kv_pool_allocator
        if not isinstance(
            allocator, (TokenToKVPoolAllocator, PagedTokenToKVPoolAllocator)
        ):
            raise SingleSchedulerExecutorError(
                "the first runner split supports token or paged KV allocators, got "
                f"{type(allocator).__name__}"
            )

        hicache_controller, hicache_info = init_runner_hicache_l2(runner, server_args)
        connection.send(
            RunnerInitInfo(
                rank=rank,
                max_total_num_tokens=runner.token_to_kv_pool_allocator.size,
                max_running_requests=runner.max_running_requests,
                req_pool_size=runner.req_to_token_pool.size,
                max_context_len=runner.req_to_token_pool.max_context_len,
                allocator_need_sort=runner.token_to_kv_pool_allocator.need_sort,
                allocator_dtype=allocator.dtype,
                allocator_page_size=allocator.page_size,
                weight_load_time=runner.weight_load_time,
                weight_load_mem_usage=getattr(runner, "weight_load_mem_usage", 0.0),
                graph_memory_usage=dict(runner.graph_memory_usage),
                graph_time_usage=dict(runner.graph_time_usage),
                available_gpu_memory_gb=get_available_gpu_memory(
                    runner.device, gpu_id, empty_cache=False
                ),
                hicache_enabled=hicache_controller is not None,
                hicache_host_size=(hicache_info or {}).get("host_size", 0),
                hicache_host_logical_size=(hicache_info or {}).get(
                    "host_logical_size", 0
                ),
                hicache_host_page_size=(hicache_info or {}).get("host_page_size", 1),
                hicache_host_size_per_token=(hicache_info or {}).get(
                    "host_size_per_token", 0
                ),
            )
        )
    except Exception as exc:
        connection.send(
            RunnerInitInfo(
                rank=rank,
                error=f"{type(exc).__name__}: {exc}",
                remote_traceback=traceback.format_exc(),
            )
        )
        return

    while True:
        try:
            command = connection.recv()
        except EOFError:
            return
        if not isinstance(command, RunnerCommand):
            connection.send(
                RunnerReply(
                    rank=rank,
                    error=f"unexpected command type: {type(command)!r}",
                )
            )
            continue
        if command.kind == "shutdown":
            return

        try:
            if command.kind == "execute":
                if command.batch is None:
                    raise SingleSchedulerExecutorError("execute command has no batch")
                result = _execute_batch(
                    command.batch, runner, return_result=(rank == 0)
                )
                connection.send(RunnerReply(rank=rank, result=result))
                continue

            if command.kind.startswith("hicache_"):
                host_indices = execute_runner_hicache_command(
                    hicache_controller,
                    kind=command.kind,
                    host_indices=command.host_indices,
                    device_indices=command.device_indices,
                    node_id=command.node_id,
                )
                connection.send(RunnerReply(rank=rank, host_indices=host_indices))
                continue

            raise SingleSchedulerExecutorError(
                f"invalid runner command: {command.kind!r}"
            )
        except Exception as exc:
            connection.send(
                RunnerReply(
                    rank=rank,
                    error=f"{type(exc).__name__}: {exc}",
                    remote_traceback=traceback.format_exc(),
                )
            )
