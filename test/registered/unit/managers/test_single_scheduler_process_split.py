import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci

from sglang.launch_server import (
    _single_scheduler_model_core_args,
    _use_single_scheduler_model_runners,
)
from sglang.srt.managers.single_scheduler import (
    _parallel_overrides,
    validate_single_scheduler_server_args,
)
from sglang.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator
from sglang.srt.model_executor.single_scheduler_executor import (
    ModelExecutor,
    SingleSchedulerExecutorError,
)

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _args(**overrides):
    values = dict(
        tp_size=8,
        pp_size=1,
        dp_size=1,
        ep_size=8,
        moe_dp_size=1,
        attn_cp_size=1,
        dcp_size=1,
        speculative_algorithm=None,
        disable_overlap_schedule=True,
        enable_hierarchical_cache=False,
        enable_lora=False,
        page_size=1,
        enable_mixed_chunk=False,
        disaggregation_mode="null",
        enable_dp_attention=False,
        is_startup_weight_load_overlap=False,
        enable_hisparse=False,
        enable_unified_memory=False,
        enable_prefill_delayer=False,
        num_continuous_decode_steps=1,
        use_ray=False,
        weight_cache_mode="off",
        enable_session_radix_cache=False,
        enable_streaming_session=False,
        enable_eplb=False,
        elastic_ep_backend=None,
        enable_elastic_expert_backup=False,
        enable_pdmux=False,
        kv_events_config=None,
        enable_deterministic_inference=False,
        enable_lmcache=False,
        enable_flexkv=False,
        radix_cache_backend=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class _MetadataKVCache:
    mem_usage = 0

    def maybe_get_custom_mem_pool(self):
        return None


class TestSingleSchedulerProcessSplit(unittest.TestCase):
    def test_pilot_topology_is_accepted(self):
        validate_single_scheduler_server_args(_args())

    def test_existing_launch_args_select_single_scheduler(self):
        user_args = _args(disable_overlap_schedule=False, page_size=256)
        self.assertTrue(_use_single_scheduler_model_runners(user_args))

        model_core_args = _single_scheduler_model_core_args(user_args)
        self.assertFalse(user_args.disable_overlap_schedule)
        self.assertTrue(model_core_args.disable_overlap_schedule)
        self.assertEqual(model_core_args.page_size, 256)
        validate_single_scheduler_server_args(model_core_args)

    def test_topology_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "--tp-size must be 8"):
            validate_single_scheduler_server_args(_args(tp_size=4))

    def test_scheduler_logical_groups_have_no_collective_handles(self):
        overrides = _parallel_overrides()
        self.assertEqual(overrides["tp_size"], 8)
        self.assertEqual(overrides["moe_ep_size"], 8)
        for name in (
            "world_group",
            "tp_group",
            "pp_group",
            "moe_ep_group",
            "moe_dp_group",
            "moe_tp_group",
            "attn_tp_group",
            "attn_cp_group",
            "dcp_group",
        ):
            group = overrides[name]
            self.assertIsNone(group.cpu_group)
            self.assertIsNone(group.device_group)

    def test_executor_requires_exactly_eight_model_runners(self):
        with self.assertRaisesRegex(SingleSchedulerExecutorError, "requires 8"):
            ModelExecutor(connections=[], model_config=None, server_args=None)

    def test_paged_allocator_runs_on_cpu_metadata(self):
        allocator = PagedTokenToKVPoolAllocator(
            size=16,
            page_size=4,
            dtype=torch.int64,
            device="cpu",
            kvcache=_MetadataKVCache(),
            need_sort=False,
        )

        extend = allocator.alloc_extend(
            prefix_lens=torch.tensor([0], dtype=torch.int64),
            prefix_lens_cpu=torch.tensor([0], dtype=torch.int64),
            seq_lens=torch.tensor([3], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([3], dtype=torch.int64),
            last_loc=torch.tensor([-1], dtype=torch.int64),
            extend_num_tokens=3,
        )
        self.assertEqual(extend.tolist(), [4, 5, 6])

        decode_same_page = allocator.alloc_decode(
            seq_lens=torch.tensor([4], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([4], dtype=torch.int64),
            last_loc=torch.tensor([6], dtype=torch.int64),
        )
        self.assertEqual(decode_same_page.tolist(), [7])

        decode_new_page = allocator.alloc_decode(
            seq_lens=torch.tensor([5], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([5], dtype=torch.int64),
            last_loc=torch.tensor([7], dtype=torch.int64),
        )
        self.assertEqual(decode_new_page.tolist(), [8])


if __name__ == "__main__":
    unittest.main()
