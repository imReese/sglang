"""Launch the inference server."""

import asyncio
import copy
import logging
import multiprocessing as mp
import os
import sys
import warnings

from sglang.srt.plugins import load_plugins
from sglang.srt.server_args import prepare_server_args
from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.common import suppress_noisy_warnings

suppress_noisy_warnings()

logger = logging.getLogger(__name__)


def _use_single_scheduler_model_runners(server_args) -> bool:
    """Select the 1-Scheduler/8-ModelRunner topology from existing topology args."""

    return (
        server_args.tp_size == 8
        and getattr(server_args, "ep_size", 1) == 8
        and server_args.pp_size == 1
        and getattr(server_args, "dp_size", 1) == 1
        and getattr(server_args, "moe_dp_size", 1) == 1
        and server_args.attn_cp_size == 1
        and server_args.dcp_size == 1
    )


def _single_scheduler_model_core_args(server_args):
    """Build the private model-core config for the first process-split slice.

    The public CLI/config remains untouched.  The first 1+8 milestone is
    intentionally synchronous, so only the model-core copy disables overlap.
    Once async dispatch is wired, this override disappears without changing the
    launch contract.
    """

    model_core_args = copy.copy(server_args)
    object.__setattr__(model_core_args, "disable_overlap_schedule", True)
    return model_core_args


def _spawn_single_scheduler_model_core(server_args, port_args):
    """Spawn exactly one Scheduler and eight sibling ModelRunner processes."""

    from sglang.srt.managers.single_scheduler import (
        run_single_scheduler_process,
        validate_single_scheduler_server_args,
    )
    from sglang.srt.model_executor.single_scheduler_executor import (
        run_model_runner_process,
    )
    from sglang.srt.utils import maybe_reindex_device_id, numa_utils
    from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

    model_core_args = _single_scheduler_model_core_args(server_args)
    validate_single_scheduler_server_args(model_core_args)

    if model_core_args.nnodes != 1 or model_core_args.node_rank != 0:
        raise ValueError(
            "the first single-Scheduler runner split supports one node only"
        )

    memory_saver_adapter = TorchMemorySaverAdapter.create(
        enable=model_core_args.enable_memory_saver
    )
    runner_procs = []
    scheduler_connections = []

    for rank in range(8):
        scheduler_conn, runner_conn = mp.Pipe(duplex=True)
        gpu_id = model_core_args.base_gpu_id + rank * model_core_args.gpu_id_step
        with maybe_reindex_device_id(gpu_id) as child_gpu_id:
            proc = mp.Process(
                target=run_model_runner_process,
                kwargs=dict(
                    server_args=model_core_args,
                    rank=rank,
                    gpu_id=child_gpu_id,
                    nccl_port=port_args.nccl_port,
                    connection=runner_conn,
                ),
                name=f"sglang-model-runner-{rank}",
            )
            with (
                memory_saver_adapter.configure_subprocess(),
                numa_utils.configure_subprocess(model_core_args, child_gpu_id),
            ):
                proc.start()
        runner_conn.close()
        runner_procs.append(proc)
        scheduler_connections.append(scheduler_conn)

    ready_reader, ready_writer = mp.Pipe(duplex=False)
    scheduler_proc = mp.Process(
        target=run_single_scheduler_process,
        args=(
            model_core_args,
            port_args,
            scheduler_connections,
            ready_writer,
        ),
        name="sglang-single-scheduler",
    )
    scheduler_proc.start()
    ready_writer.close()
    for connection in scheduler_connections:
        connection.close()

    return ready_reader, [scheduler_proc, *runner_procs]


def _launch_single_scheduler_process_only(server_args, port_args=None):
    """smg-grpc-servicer compatible launcher for the same nine-process core."""

    from sglang.srt.entrypoints.engine import _wait_for_scheduler_ready
    from sglang.srt.server_args import PortArgs

    if port_args is None:
        port_args = PortArgs.init_new(server_args)

    ready_reader, model_core_procs = _spawn_single_scheduler_model_core(
        server_args, port_args
    )
    scheduler_infos = _wait_for_scheduler_ready([ready_reader], model_core_procs)
    return scheduler_infos[0], port_args, model_core_procs


def _launch_single_scheduler_http_server(server_args) -> None:
    """Reuse the normal HTTP/tokenizer lifecycle with a nine-process model core."""

    from sglang.srt.entrypoints.engine import (
        Engine,
        SchedulerInitResult,
        _wait_for_scheduler_ready,
        init_tokenizer_manager,
    )
    from sglang.srt.entrypoints.http_server import (
        _execute_server_warmup,
        _setup_and_run_http_server,
    )
    from sglang.srt.managers.detokenizer_manager import run_detokenizer_process

    class _SingleSchedulerEngine(Engine):
        @classmethod
        def _launch_scheduler_processes(
            cls,
            server_args,
            port_args,
            run_scheduler_process_func,
            *,
            placement_group=None,
        ):
            del run_scheduler_process_func
            if placement_group is not None:
                raise ValueError(
                    "Ray placement groups are not supported by the first "
                    "single-Scheduler runner split"
                )

            ready_reader, model_core_procs = _spawn_single_scheduler_model_core(
                server_args, port_args
            )
            all_child_pids = [proc.pid for proc in model_core_procs]
            scheduler_infos = []

            def wait_for_ready():
                infos = _wait_for_scheduler_ready([ready_reader], model_core_procs)
                scheduler_infos.extend(infos)

            def block_until_scheduler_exits():
                for proc in model_core_procs:
                    proc.join()
                    if proc.exitcode not in (0, None):
                        logger.error(
                            "%s (pid=%s) terminated with exit code %s",
                            proc.name,
                            proc.pid,
                            proc.exitcode,
                        )

            return (
                SchedulerInitResult(
                    scheduler_infos=scheduler_infos,
                    all_child_pids=all_child_pids,
                    wait_for_ready=wait_for_ready,
                    block_until_scheduler_exits=block_until_scheduler_exits,
                ),
                model_core_procs,
            )

    (
        tokenizer_manager,
        template_manager,
        port_args,
        scheduler_init_result,
        subprocess_watchdog,
        _weight_cache_daemon_procs,
    ) = _SingleSchedulerEngine._launch_subprocesses(
        server_args=server_args,
        init_tokenizer_manager_func=init_tokenizer_manager,
        run_scheduler_process_func=lambda *args, **kwargs: None,
        run_detokenizer_process_func=run_detokenizer_process,
    )

    _setup_and_run_http_server(
        server_args,
        tokenizer_manager,
        template_manager,
        port_args,
        scheduler_init_result.scheduler_infos,
        subprocess_watchdog,
        execute_warmup_func=_execute_server_warmup,
    )


def run_server(server_args):
    """Run the server based on the gRPC flags and server_args.encoder_only."""
    if server_args.encoder_only:
        # For encoder disaggregation
        if server_args.smg_grpc_mode or server_args.grpc_mode:
            from sglang.srt.disaggregation.encode_grpc_server import (
                serve_grpc_encoder,
            )

            asyncio.run(serve_grpc_encoder(server_args))
        else:
            from sglang.srt.disaggregation.encode_server import launch_server

            launch_server(server_args)
    elif server_args.smg_grpc_mode:
        # Legacy SMG gRPC server (--smg-grpc-mode, or the deprecated --grpc-mode
        # which __post_init__ folds into smg_grpc_mode). Reuse the same 1+8 model
        # core when the existing TP/EP topology selects it.
        from sglang.srt.entrypoints.grpc_server import serve_grpc

        scheduler_launcher = (
            _launch_single_scheduler_process_only
            if _use_single_scheduler_model_runners(server_args)
            else None
        )
        asyncio.run(serve_grpc(server_args, scheduler_launcher=scheduler_launcher))
    elif server_args.use_ray:
        # Ray mode: HTTP mode with Ray backend.
        try:
            from sglang.srt.ray.http_server import launch_server
        except ImportError:
            raise ImportError(
                "Ray is required for --use-ray mode. "
                "Install it with: pip install 'sglang[ray]'"
            )

        launch_server(server_args)
    elif _use_single_scheduler_model_runners(server_args):
        _launch_single_scheduler_http_server(server_args)
    else:
        # Default mode: HTTP mode.
        from sglang.srt.entrypoints.http_server import launch_server

        launch_server(server_args)


if __name__ == "__main__":
    warnings.warn(
        "'python -m sglang.launch_server' is still supported, but "
        "'sglang serve' is the recommended entrypoint.\n"
        "  Example: sglang serve --model-path <model> [options]",
        UserWarning,
        stacklevel=1,
    )

    load_plugins()

    server_args = prepare_server_args(sys.argv[1:])

    try:
        run_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
