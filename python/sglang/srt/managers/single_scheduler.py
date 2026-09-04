# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""One CPU Scheduler driving eight out-of-process ModelRunners.

This module keeps SGLang's existing Scheduler/Req/ScheduleBatch/RadixCache
logic and changes the execution ownership underneath it:

    1 Scheduler process (CPU canonical metadata, no model/NCCL)
              |
         ModelExecutor
       / / / / | \\ \\ \\ \
    MR0 ...             ... MR7
    TP0/EP0             TP7/EP7

The first slice is synchronous. Standard HiCache L2 is Scheduler-owned at the
control plane while the physical D<->H transfers stay inside ModelRunner ranks.
"""

from __future__ import annotations

import logging
import os
import signal
import time
from types import SimpleNamespace
from typing import Optional

import psutil
import setproctitle
import torch

from sglang.srt.configs.hybrid_arch import (
    hybrid_gdn_config,
    hybrid_lightning_config,
    kimi_linear_config,
    linear_attn_model_spec,
    mamba2_config,
)
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import ShutdownReq
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.request_receiver import (
    SchedulerRequestReceiver,
)
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.single_scheduler_executor import (
    ModelExecutor,
    SingleSchedulerExecutorError,
)
from sglang.srt.model_executor.single_scheduler_hicache import (
    build_scheduler_hiradix_cache,
)
from sglang.srt.runtime_context import (
    get_context,
    get_observability,
    get_parallel,
    get_schedule,
    publish,
)
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import (
    configure_logger,
    get_exception_traceback,
    kill_itself_when_parent_died,
    set_random_seed,
)

logger = logging.getLogger(__name__)


class _LogicalGroup:
    """Scheduler-only topology descriptor; never used for a collective."""

    def __init__(self, world_size: int):
        self.world_size = world_size
        self.rank = 0
        self.rank_in_group = 0
        self.ranks = tuple(range(world_size))
        self.first_rank = 0
        self.cpu_group = None
        self.device_group = None
        self.device = torch.device("cpu")
        self.device_module = torch


class SingleSchedulerRequestReceiver(SchedulerRequestReceiver):
    """The sole Scheduler receives ingress directly; there are no peers to fan out to."""

    def _broadcast_reqs_across_ranks(self, recv_reqs):
        return recv_reqs or []

    def _finalize_shm_features(self, recv_reqs):
        if recv_reqs:
            from sglang.srt.managers.mm_utils import unwrap_shm_features

            for req in recv_reqs:
                unwrap_shm_features(req)


def validate_single_scheduler_server_args(server_args: ServerArgs) -> None:
    """Fail closed around the first real DP1/TP8/EP8 process-split milestone."""

    checks = [
        (server_args.tp_size == 8, "--tp-size must be 8"),
        (server_args.pp_size == 1, "--pp-size must be 1"),
        (getattr(server_args, "dp_size", 1) == 1, "--dp-size must be 1"),
        (getattr(server_args, "ep_size", 1) == 8, "--ep-size must be 8"),
        (server_args.moe_dp_size == 1, "--moe-dp-size must be 1"),
        (server_args.attn_cp_size == 1, "--attn-cp-size must be 1"),
        (server_args.dcp_size == 1, "--dcp-size must be 1"),
        (
            server_args.speculative_algorithm is None,
            "speculative decoding is not supported in the first runner split",
        ),
        (
            server_args.disable_overlap_schedule,
            "single-Scheduler model core must run synchronously in the first runner split",
        ),
        (
            getattr(server_args, "hicache_storage_backend", None) is None,
            "single-Scheduler HiCache currently supports L2 only; L3 storage is not supported",
        ),
        (not server_args.enable_lora, "LoRA is not supported in the first runner split"),
        (
            not getattr(server_args, "enable_mixed_chunk", False),
            "mixed prefill/decode batches are not supported in the first runner split",
        ),
        (
            getattr(server_args, "disaggregation_mode", "null") == "null",
            "PD disaggregation is not supported in the first runner split",
        ),
        (
            not getattr(server_args, "enable_dp_attention", False),
            "DP attention is not supported in the first runner split",
        ),
        (
            not getattr(server_args, "is_startup_weight_load_overlap", False),
            "startup weight-load overlap is not supported in the first runner split",
        ),
        (
            not getattr(server_args, "enable_hisparse", False),
            "HiSparse is not supported in the first runner split",
        ),
        (
            not getattr(server_args, "enable_unified_memory", False),
            "unified KV memory is not supported in the first runner split",
        ),
        (
            not getattr(server_args, "enable_prefill_delayer", False),
            "prefill delayer is not supported in the first runner split",
        ),
        (
            getattr(server_args, "num_continuous_decode_steps", 1) == 1,
            "continuous multi-step decode is not supported in the first runner split",
        ),
        (
            not getattr(server_args, "use_ray", False),
            "Ray is not supported in the first runner split",
        ),
        (
            getattr(server_args, "weight_cache_mode", "off") == "off",
            "weight cache mode is not supported in the first runner split",
        ),
        (
            not getattr(server_args, "enable_session_radix_cache", False),
            "session radix cache is not supported in the first runner split",
        ),
        (
            not getattr(server_args, "enable_streaming_session", False),
            "streaming sessions are not supported in the first runner split",
        ),
        (
            not getattr(server_args, "enable_eplb", False),
            "EPLB is not supported in the first runner split",
        ),
        (
            getattr(server_args, "elastic_ep_backend", None) is None,
            "elastic EP is not supported in the first runner split",
        ),
        (
            not getattr(server_args, "enable_elastic_expert_backup", False),
            "elastic expert backup is not supported in the first runner split",
        ),
        (
            not getattr(server_args, "enable_pdmux", False),
            "PD multiplexing is not supported in the first runner split",
        ),
        (
            getattr(server_args, "kv_events_config", None) is None,
            "KV event publishing is not supported in the first runner split",
        ),
        (
            not getattr(server_args, "enable_deterministic_inference", False),
            "deterministic inference is not supported in the first runner split",
        ),
        (
            not getattr(server_args, "enable_lmcache", False),
            "LMCache is not supported in the first runner split",
        ),
        (
            not getattr(server_args, "enable_flexkv", False),
            "FlexKV is not supported in the first runner split",
        ),
        (
            getattr(server_args, "radix_cache_backend", None) is None,
            "custom radix-cache backends are not supported in the first runner split",
        ),
    ]
    errors = [message for ok, message in checks if not ok]
    if errors:
        raise ValueError("single-Scheduler ModelRunner pilot: " + "; ".join(errors))
    if envs.SGLANG_RUST_SERVER.get():
        raise ValueError("Rust server mode is not supported in the first runner split")


class SingleScheduler(Scheduler):
    """Existing Scheduler logic backed by CPU canonical KV/request metadata."""

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        runner_connections,
        pipe_writer=None,
    ) -> None:
        validate_single_scheduler_server_args(server_args)
        self._runner_connections = list(runner_connections)
        self._runner_shutdown = False
        self._single_scheduler_hicache_enabled = bool(
            server_args.enable_hierarchical_cache
        )
        super().__init__(
            server_args=server_args,
            port_args=port_args,
            gpu_id=0,
            tp_rank=0,
            moe_ep_rank=0,
            pp_rank=0,
            attn_cp_rank=0,
            moe_dp_rank=0,
            dp_rank=None,
        )

    def init_moe_gemm_config(self):
        self.require_mlp_sync = False

    def init_mamba_backend(self) -> None:
        return None

    def maybe_init_hccl_dp_prewarm(self) -> None:
        return None

    def init_model_worker(self):
        if not self.model_config.is_generation:
            raise ValueError(
                "the first single-Scheduler runner split supports generation models only"
            )
        if self.model_config.is_multimodal:
            raise ValueError(
                "multimodal models are not supported in the first single-Scheduler runner split"
            )
        if self.model_config.is_hybrid_swa:
            raise ValueError(
                "hybrid SWA is not supported in the first single-Scheduler runner split"
            )
        linear_spec = linear_attn_model_spec(self.model_config)
        if (
            hybrid_gdn_config(self.model_config) is not None
            or mamba2_config(self.model_config) is not None
            or kimi_linear_config(self.model_config) is not None
            or hybrid_lightning_config(self.model_config) is not None
            or linear_spec is not None
        ):
            raise ValueError(
                "hybrid SSM/linear-attention models are not supported in the first "
                "single-Scheduler runner split"
            )

        executor = ModelExecutor(
            connections=self._runner_connections,
            model_config=self.model_config,
            server_args=self.server_args,
        )
        self.model_executor = executor
        self.tp_worker = executor
        self.model_worker = executor
        self.draft_worker = None
        self.external_corpus_manager = None

        info = executor.leader_info
        self.max_total_num_tokens = info.max_total_num_tokens
        self.max_prefill_tokens = get_schedule().max_prefill_tokens
        self.max_running_requests = info.max_running_requests
        self.max_queued_requests = get_schedule().max_queued_requests
        self.max_req_len = min(
            self.model_config.context_len - 1,
            info.max_total_num_tokens - 1,
        )
        self.max_req_input_len = self.max_req_len - 5
        self.random_seed = self.server_args.random_seed
        self.device = "cpu"
        self.forward_stream = None
        self.kv_cache_allocation_time = 0.0
        self.startup_available_gpu_memory_gb = info.available_gpu_memory_gb
        self.min_free_slots_delayer = None

        self.tp_group = get_parallel().tp_group
        self.tp_cpu_group = None
        self.attn_tp_group = get_parallel().attn_tp_group
        self.attn_tp_cpu_group = None
        self.attn_cp_group = get_parallel().attn_cp_group
        self.attn_cp_cpu_group = None
        self.pp_group = get_parallel().pp_group
        self.world_group = get_parallel().world_group
        self.dp_tp_group = self.tp_group
        self.dp_tp_cpu_group = None

        self.pad_input_ids_func = None
        set_random_seed(self.random_seed)

        # The stock kv_cache_builder assumes HiCache control/data live in this
        # process. Build its ordinary CPU radix metadata first; init_hisparse_coordinator
        # replaces that empty startup tree with the remote HiRadixCache control plane
        # before any request can be scheduled.
        if self._single_scheduler_hicache_enabled:
            if executor.get_hicache_controller() is None:
                raise ValueError("HiCache was enabled but ModelRunner L2 did not initialize")
            self.enable_hierarchical_cache = False

    def emit_metrics_constants(self) -> None:
        return None

    def init_hisparse_coordinator(self) -> None:
        self.hisparse_coordinator = None
        if not self._single_scheduler_hicache_enabled:
            return

        controller = self.model_executor.get_hicache_controller()
        if controller is None:
            raise RuntimeError("single-Scheduler HiCache controller is missing")
        self.tree_cache = build_scheduler_hiradix_cache(
            server_args=self.server_args,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            remote_controller=controller,
            page_size=self.page_size,
            disable=self.disable_radix_cache,
            enable_metrics=get_observability().enable_metrics,
        )
        self.enable_hierarchical_cache = True

    def init_overlap(self):
        self._next_token_ids = torch.zeros(
            self.req_to_token_pool.req_to_token.shape[0], dtype=torch.int64
        )
        self.future_map = None
        self._confidence_budget_prepare = None
        self.device_module = torch

    def maybe_init_ngram_embedding(self):
        self.ngram_embedding_manager = SimpleNamespace(enabled=False)
        self.use_ngram_embedding = False

    def init_request_receiver(self) -> None:
        self.request_receiver = SingleSchedulerRequestReceiver(
            recv_from_tokenizer=self.recv_from_tokenizer,
            recv_from_rpc=self.ipc_channels.recv_from_rpc,
            recv_skipper=self.recv_skipper,
            input_blocker=self.input_blocker,
            mm_receiver=self.mm_receiver,
            ps=self.ps,
            tp_group=self.tp_group,
            tp_cpu_group=None,
            attn_tp_group=self.attn_tp_group,
            attn_tp_cpu_group=None,
            attn_cp_group=self.attn_cp_group,
            attn_cp_cpu_group=None,
            world_group=self.world_group,
            server_args=self.server_args,
            model_config=self.model_config,
            max_recv_per_poll=self.max_recv_per_poll,
            stream_output=lambda *a, **kw: self.output_streamer.stream_output(*a, **kw),
            get_last_batch=lambda: self.last_batch,
            scripted_scheduler_hook=self.scripted_scheduler_hook,
        )

    def init_dp_attn_adapter(self) -> None:
        self.dp_attn_adapter = None

    def run_event_loop(self) -> None:
        self.event_loop_normal()

    def _resolve_runner_inputs(self, batch: ScheduleBatch) -> None:
        if batch.prefill_input_ids_cpu is not None:
            if batch.mix_running_indices is not None:
                raise SingleSchedulerExecutorError(
                    "mixed/chunked prefill is not supported in the first runner split"
                )
            batch.input_ids = batch.prefill_input_ids_cpu.to("cpu")
            batch.prefill_input_ids_cpu = None
        elif batch.input_ids is None:
            indices = (
                batch.req_pool_indices_cpu
                if batch.req_pool_indices_cpu is not None
                else batch.req_pool_indices.to("cpu")
            )
            batch.input_ids = self._next_token_ids[indices]

    def run_batch(
        self,
        batch: ScheduleBatch,
        pp_proxy_tensors=None,
    ) -> GenerationBatchResult:
        if pp_proxy_tensors is not None:
            raise SingleSchedulerExecutorError("PP is not supported in the first runner split")
        if not self.is_generation:
            raise SingleSchedulerExecutorError(
                "embedding/reward models are not supported in the first runner split"
            )

        self.forward_ct += 1
        batch.forward_iter = self.forward_ct
        batch.launch_ts = time.monotonic()
        batch.after_idle_gap = self._sched_idled
        self._sched_idled = False
        self.cur_batch_for_debug = batch
        self.profiler_manager._profile_batch_predicate(batch)

        self._resolve_runner_inputs(batch)
        result = self.model_executor.run_batch(batch)

        if result.has_sampled_token_ids:
            indices = (
                batch.req_pool_indices_cpu
                if batch.req_pool_indices_cpu is not None
                else batch.req_pool_indices.to("cpu")
            )
            self._next_token_ids[indices] = result.next_token_ids.to(torch.int64)
            batch.input_ids = None

        if batch.return_logprob or batch.return_hidden_states:
            result.extend_input_len_per_req = [
                req.extend_range.length if req.extend_range is not None else 0
                for req in batch.reqs
            ]
        else:
            result.extend_input_len_per_req = None
        result.extend_logprob_start_len_per_req = (
            batch.extend_logprob_start_lens if batch.return_logprob else None
        )
        return result

    def flush_cache(self, empty_cache: bool = True):
        return super().flush_cache(empty_cache=False)

    def handle_shutdown(self, recv_req: ShutdownReq):
        self._shutdown_runners()
        return super().handle_shutdown(recv_req)

    def _shutdown_runners(self) -> None:
        if self._runner_shutdown:
            return
        self._runner_shutdown = True
        self.model_executor.shutdown()

    def release_host_resources(self) -> None:
        self._shutdown_runners()
        super().release_host_resources()


def _install_scheduler_cpu_attention_override() -> None:
    context = get_context()
    fields = {}
    for name in (
        "attention_backend",
        "prefill_attention_backend",
        "decode_attention_backend",
    ):
        try:
            context.config_leaf(name)
        except (AttributeError, ValueError):
            continue
        fields[name] = "torch_native"
    if fields:
        context.override("single_scheduler.cpu_metadata", **fields)


def _parallel_overrides():
    world = _LogicalGroup(8)
    tp = _LogicalGroup(8)
    pp = _LogicalGroup(1)
    moe_ep = _LogicalGroup(8)
    one = _LogicalGroup(1)
    return dict(
        world_size=8,
        world_rank=0,
        tp_size=8,
        tp_rank=0,
        pp_size=1,
        pp_rank=0,
        moe_ep_size=8,
        moe_ep_rank=0,
        moe_dp_size=1,
        moe_dp_rank=0,
        moe_tp_size=1,
        moe_tp_rank=0,
        attn_tp_size=8,
        attn_tp_rank=0,
        attn_cp_size=1,
        attn_cp_rank=0,
        dcp_enabled=False,
        dcp_size=1,
        dcp_rank=0,
        attn_dcp_size=1,
        attn_dcp_rank=0,
        attn_dp_size=1,
        attn_dp_rank=0,
        world_group=world,
        tp_group=tp,
        pp_group=pp,
        moe_ep_group=moe_ep,
        moe_dp_group=one,
        moe_tp_group=one,
        attn_tp_group=tp,
        attn_cp_group=one,
        dcp_group=one,
    )


def run_single_scheduler_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    runner_connections,
    pipe_writer,
) -> None:
    """Process entry: one Scheduler, not a member of the TP/EP model world."""

    kill_itself_when_parent_died()
    publish(server_args, role="scheduler")
    validate_single_scheduler_server_args(server_args)
    _install_scheduler_cpu_attention_override()
    setproctitle.setproctitle("sglang::scheduler")
    configure_logger(server_args, prefix=" SCHED")
    parent_process = psutil.Process().parent()

    scheduler = None
    try:
        with get_parallel().override(**_parallel_overrides()):
            scheduler = SingleScheduler(
                server_args=server_args,
                port_args=port_args,
                runner_connections=runner_connections,
            )
            pipe_writer.send(scheduler.get_init_info())
            scheduler.run_event_loop()
    except Exception:
        traceback = get_exception_traceback()
        logger.error("Single Scheduler hit an exception: %s", traceback)
        try:
            parent_process.send_signal(signal.SIGQUIT)
        except Exception:
            pass
    finally:
        if scheduler is not None:
            try:
                scheduler.metrics_reporter._shutdown_fpm()
            except Exception:
                pass
            if getattr(scheduler, "gracefully_exit", False):
                scheduler.release_host_resources()
