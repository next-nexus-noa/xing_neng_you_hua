import os
import unittest
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, KVCacheTensor
from vllm.v1.worker.adaptive_ubatch import AdaptiveUBatchDecision
from vllm.v1.worker.scom_ubatch import plan_scom_groups

from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


class TestScomPPGroupingConfig(unittest.TestCase):
    @patch.dict(
        os.environ,
        {
            "VLLM_ASCEND_PP_MICROBATCH_GROUPING": "scom",
            "VLLM_ASCEND_PP_SCOM_MIN_GAIN_PCT": "4",
            "VLLM_ASCEND_PP_SCOM_SHAPE_BUCKETS": (
                "128,256,512,1024"
            ),
            "VLLM_ASCEND_PP_SCOM_OPTIMIZE_CAPACITIES": "1",
            "VLLM_ASCEND_PP_SCOM_ALLOW_BUCKET_CROSSING": "0",
            "VLLM_ASCEND_PP_SCOM_CAPACITY_QUANTUM": "64",
            "VLLM_ASCEND_PP_SCOM_MAX_CAPACITY_CANDIDATES": "12",
            "VLLM_ASCEND_PP_SCOM_MAX_SWAPS": "3",
        },
    )
    def test_scom_config_is_resolved_from_central_envs(self):
        self.assertEqual(
            NPUModelRunner._pp_microbatch_grouping_mode(),
            "scom",
        )
        self.assertEqual(
            NPUModelRunner._scom_grouping_config(),
            (
                0.04,
                (128, 256, 512, 1024),
                True,
                False,
                64,
                12,
                3,
            ),
        )

    @patch.dict(
        os.environ,
        {"VLLM_ASCEND_PP_MICROBATCH_GROUPING": "invalid"},
    )
    def test_invalid_grouping_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            NPUModelRunner._pp_microbatch_grouping_mode()

    def test_exact_scom_plan_is_reused_from_runner_cache(self):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.input_batch = SimpleNamespace(
            num_computed_tokens_cpu=[3072, 0],
        )
        runner.compute_aware_quantum = 8
        runner.scom_capacity_quantum = 64
        runner.scom_min_predicted_gain = 0.02
        runner.scom_shape_buckets = (
            128,
            256,
            512,
            1024,
            2048,
        )
        runner.scom_optimize_capacities = True
        runner.scom_allow_bucket_crossing = False
        runner.scom_max_capacity_candidates = 8
        runner.scom_max_swaps = 4
        runner._scom_plan_cache = OrderedDict()
        pp_group = SimpleNamespace(
            is_first_rank=True,
            world_size=1,
        )

        with (
            patch(
                "vllm_ascend.worker.model_runner_v1.get_pp_group",
                return_value=pp_group,
            ),
            patch(
                "vllm_ascend.worker.model_runner_v1.plan_scom_groups",
                wraps=plan_scom_groups,
            ) as planner,
        ):
            first = runner._select_scom_grouping(
                num_scheduled_tokens_np=np.array([1024, 1024]),
                num_reqs=2,
                num_ubatches=2,
            )
            second = runner._select_scom_grouping(
                num_scheduled_tokens_np=np.array([1024, 1024]),
                num_reqs=2,
                num_ubatches=2,
            )

        self.assertEqual(planner.call_count, 1)
        self.assertFalse(first.cache_hit)
        self.assertTrue(second.cache_hit)


class TestNPUModelRunnerKVCache(unittest.TestCase):
    def _build_runner(self):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.device = torch.device("cpu")
        runner.use_sparse = False
        runner.use_sparse_c8_indexer = False
        runner.use_compress = False
        runner.use_hybrid_blocks = False
        runner.hybrid_with_attn_and_mamba = False
        runner.runner_only_attn_layers = set()
        runner.is_kv_consumer = False
        runner.vllm_config = MagicMock()
        runner.vllm_config.kv_transfer_config = None
        runner.model_config = MagicMock()
        runner.model_config.use_mla = True
        backend = MagicMock()
        backend.get_kv_cache_shape.side_effect = lambda num_blocks, block_size, num_kv_heads, head_size: (
            2,
            num_blocks,
            block_size,
            num_kv_heads,
            head_size,
        )
        runner.attn_backend = backend
        return runner

    def test_allocate_kv_cache_uses_layer_spec_for_draft_gqa(self):
        runner = self._build_runner()
        kv_cache_spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=64,
            head_size_v=64,
            dtype=torch.float16,
        )
        kv_cache_config = KVCacheConfig(
            num_blocks=2,
            kv_cache_tensors=[KVCacheTensor(size=kv_cache_spec.page_size_bytes * 2, shared_by=["draft_attn"])],
            kv_cache_groups=[KVCacheGroupSpec(layer_names=["draft_attn"], kv_cache_spec=kv_cache_spec)],
        )

        kv_cache_raw_tensors = runner._allocate_kv_cache_tensors(kv_cache_config)
        k_cache_raw, v_cache_raw = kv_cache_raw_tensors["draft_attn"]

        self.assertEqual(k_cache_raw.numel(), kv_cache_spec.page_size_bytes)
        self.assertEqual(v_cache_raw.numel(), kv_cache_spec.page_size_bytes)

    def test_reshape_kv_cache_uses_layer_spec_for_draft_gqa(self):
        runner = self._build_runner()
        kv_cache_spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=64,
            head_size_v=64,
            dtype=torch.float16,
        )
        kv_cache_config = KVCacheConfig(
            num_blocks=2,
            kv_cache_tensors=[KVCacheTensor(size=kv_cache_spec.page_size_bytes * 2, shared_by=["draft_attn"])],
            kv_cache_groups=[KVCacheGroupSpec(layer_names=["draft_attn"], kv_cache_spec=kv_cache_spec)],
        )
        kv_cache_raw_tensors = runner._allocate_kv_cache_tensors(kv_cache_config)
        runner._kv_cache_spec_attn_group_iterator = lambda: [
            SimpleNamespace(
                kv_cache_spec=kv_cache_spec,
                backend=runner.attn_backend,
                layer_names=["draft_attn"],
            )
        ]

        kv_caches = runner._reshape_kv_cache_tensors(kv_cache_config, kv_cache_raw_tensors)
        k_cache, v_cache = kv_caches["draft_attn"]

        self.assertEqual(k_cache.shape, (2, 16, 8, 64))
        self.assertEqual(v_cache.shape, (2, 16, 8, 64))


class TestNPUModelRunnerOutputTokenIds(unittest.TestCase):
    def _build_runner(self):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.device = torch.device("cpu")
        runner.vllm_config = MagicMock()
        runner.model_config = MagicMock()
        runner.use_compress = False
        return runner

    @patch("vllm_ascend.worker.model_runner_v1.lmhead_tp_enable")
    def test_sample_updates_output_token_ids_before_sampler(self, mock_lmhead_tp_enable):
        """Verify output_token_ids are updated before sampler is called"""
        mock_lmhead_tp_enable.return_value = False

        # Build input batch with historical sampled tokens
        input_batch = MagicMock()
        input_batch.sampling_metadata.output_token_ids = [
            [1, 2, 3, -1],
            [4, 5, -1],
        ]
        input_batch.num_reqs = 2
        input_batch.top_k_cpu = None
        input_batch.prev_req_id_to_index = {
            "req0": 0,
            "req1": 1,
        }
        input_batch.sampled_token_ids_cpu = torch.tensor([6, 7])
        input_batch.async_copy_ready_event = MagicMock()
        input_batch.async_copy_ready_event.synchronize = MagicMock()

        # Simulate the real behavior of InputBatch.update_async_output_token_ids
        def mock_update_output_token_ids():
            output_token_ids = input_batch.sampling_metadata.output_token_ids
            sampled_ids = input_batch.sampled_token_ids_cpu.tolist()

            for index, req_id in enumerate(input_batch.prev_req_id_to_index):
                prev_index = input_batch.prev_req_id_to_index[req_id]
                req_output = output_token_ids[index]
                if req_output and req_output[-1] == -1:
                    req_output[-1] = sampled_ids[prev_index]

        input_batch.update_async_output_token_ids.side_effect = mock_update_output_token_ids

        # Build runner and inject dependencies
        runner = self._build_runner()
        runner.input_batch = input_batch
        runner.sampler = MagicMock(return_value=MagicMock())

        # Call sample method
        logits = torch.randn(2, 32000)
        runner._sample(logits=logits, spec_decode_metadata=None)

        # Verify sampler and update_async_output_token_ids were called
        runner.sampler.assert_called_once()
        input_batch.update_async_output_token_ids.assert_called_once()

        # Verify output_token_ids were updated before sampler is called
        call_kwargs = runner.sampler.call_args[1]
        actual_sampling_metadata = call_kwargs["sampling_metadata"]
        actual_output_token_ids = actual_sampling_metadata.output_token_ids
        self.assertEqual(actual_output_token_ids[0], [1, 2, 3, 6])
        self.assertEqual(actual_output_token_ids[1], [4, 5, 7])


class TestAdaptivePPFeedback(unittest.TestCase):
    def _build_runner(self):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.parallel_config = SimpleNamespace(
            adaptive_ubatch_min_observations=2,
            adaptive_ubatch_feedback_interval_steps=4,
        )
        runner.adaptive_ubatch_controller = MagicMock()
        runner._adaptive_pp_feedback_step = 0
        runner._pending_adaptive_pp_decision = None
        runner._pending_adaptive_pp_elapsed_ms = None
        runner._pending_adaptive_pp_npu_events = None
        runner._pending_adaptive_pp_send_wait_ms = 0.0
        runner._pending_adaptive_pp_reduction = None
        runner._active_adaptive_pp_decision = None
        runner._active_adaptive_pp_step_start = None
        runner._active_adaptive_pp_npu_start_event = None
        return runner

    @patch("vllm_ascend.worker.model_runner_v1.dist.broadcast")
    def test_mirrored_decision_adds_no_per_step_collective(
        self,
        mock_broadcast,
    ):
        runner = self._build_runner()
        decision = AdaptiveUBatchDecision(
            num_ubatches=2,
            predicted_gain_pct=8.0,
            reason="candidate_calibration",
            bucket_key=("medium", "prefill", "large"),
            total_tokens=2048,
            online=True,
            num_reqs=16,
            previous_m=1,
            switched=True,
            candidate_scores=(
                {"m": 1, "robust_ms": 100.0},
                {"m": 2, "robust_ms": 92.0},
            ),
        )

        mirrored = runner._use_mirrored_adaptive_pp_microbatch_decision(
            decision
        )

        self.assertIs(mirrored, decision)
        mock_broadcast.assert_not_called()

    def test_single_candidate_skips_feedback_measurement(self):
        runner = self._build_runner()
        runner._pp_microbatch_configured_count = MagicMock(return_value=4)
        decision = SimpleNamespace(
            candidate_scores=({"m": 1, "robust_ms": 100.0},),
            switched=True,
            reason="candidate_calibration",
        )

        self.assertFalse(
            runner._should_measure_adaptive_pp_critical_path(decision)
        )
        self.assertEqual(runner._adaptive_pp_feedback_step, 0)

    def test_mirrored_non_first_controller_observes_reduced_feedback(self):
        runner = self._build_runner()
        decision = MagicMock()

        runner._observe_adaptive_pp_microbatch_result(
            decision,
            forward_ms=125.0,
        )

        runner.adaptive_ubatch_controller.observe.assert_called_once_with(
            decision,
            forward_ms=125.0,
            next_waiting_count=None,
            next_running_count=None,
            next_oldest_wait_ms=None,
            next_pending_first_token_count=None,
            next_oldest_first_token_wait_ms=None,
            next_pending_prefill_tokens=None,
            completed_first_token_count=None,
        )

    def test_measurement_schedule_covers_warmup_switches_and_interval(self):
        runner = self._build_runner()
        stable = SimpleNamespace(switched=False, reason="keep_current_best")
        switched = SimpleNamespace(switched=True, reason="robust_cost_improvement")
        calibration = SimpleNamespace(
            switched=False,
            reason="candidate_calibration",
        )
        probation = SimpleNamespace(
            switched=False,
            reason="candidate_probation",
            probation=True,
        )
        contextual_exploration = SimpleNamespace(
            switched=False,
            reason="contextual_exploration",
        )

        self.assertFalse(runner._should_measure_adaptive_pp_critical_path(stable))
        self.assertFalse(runner._should_measure_adaptive_pp_critical_path(stable))
        self.assertTrue(runner._should_measure_adaptive_pp_critical_path(switched))
        self.assertTrue(runner._should_measure_adaptive_pp_critical_path(stable))
        self.assertFalse(runner._should_measure_adaptive_pp_critical_path(stable))
        self.assertTrue(
            runner._should_measure_adaptive_pp_critical_path(calibration)
        )
        self.assertTrue(
            runner._should_measure_adaptive_pp_critical_path(probation)
        )
        self.assertTrue(
            runner._should_measure_adaptive_pp_critical_path(
                contextual_exploration
            )
        )

    @patch("vllm_ascend.worker.model_runner_v1.get_pp_group")
    @patch("vllm_ascend.worker.model_runner_v1.dist.all_reduce")
    def test_flush_uses_cross_stage_max_without_changing_trace_format(
        self,
        mock_all_reduce,
        mock_get_pp_group,
    ):
        runner = self._build_runner()
        decision = MagicMock()
        runner._pending_adaptive_pp_decision = decision
        runner._pending_adaptive_pp_elapsed_ms = 50.0
        mock_get_pp_group.return_value = SimpleNamespace(
            world_size=2,
            cpu_group=MagicMock(),
            is_first_rank=True,
        )

        def set_cross_stage_max(tensor, **_kwargs):
            tensor.fill_(125.0)

        mock_all_reduce.side_effect = set_cross_stage_max
        runner._flush_adaptive_pp_microbatch_result()

        runner.adaptive_ubatch_controller.observe.assert_called_once_with(
            decision,
            forward_ms=125.0,
            next_waiting_count=None,
            next_running_count=None,
            next_oldest_wait_ms=None,
            next_pending_first_token_count=None,
            next_oldest_first_token_wait_ms=None,
            next_pending_prefill_tokens=None,
            completed_first_token_count=None,
        )
        self.assertIsNone(runner._pending_adaptive_pp_decision)
        self.assertIsNone(runner._pending_adaptive_pp_elapsed_ms)

    @patch("vllm_ascend.worker.model_runner_v1.get_pp_group")
    @patch("vllm_ascend.worker.model_runner_v1.dist.all_reduce")
    def test_feedback_max_is_started_async_before_flush(
        self,
        mock_all_reduce,
        mock_get_pp_group,
    ):
        runner = self._build_runner()
        decision = MagicMock()
        runner._pending_adaptive_pp_decision = decision
        runner._pending_adaptive_pp_elapsed_ms = 50.0
        work = MagicMock()
        mock_all_reduce.return_value = work
        mock_get_pp_group.return_value = SimpleNamespace(
            world_size=2,
            cpu_group=MagicMock(),
            is_first_rank=True,
        )

        runner._start_adaptive_pp_feedback_reduction()

        mock_all_reduce.assert_called_once()
        self.assertTrue(
            mock_all_reduce.call_args.kwargs["async_op"]
        )
        elapsed, pending_work = runner._pending_adaptive_pp_reduction
        elapsed.fill_(125.0)
        self.assertIs(pending_work, work)

        runner._flush_adaptive_pp_microbatch_result()

        work.wait.assert_called_once_with()
        runner.adaptive_ubatch_controller.observe.assert_called_once_with(
            decision,
            forward_ms=125.0,
            next_waiting_count=None,
            next_running_count=None,
            next_oldest_wait_ms=None,
            next_pending_first_token_count=None,
            next_oldest_first_token_wait_ms=None,
            next_pending_prefill_tokens=None,
            completed_first_token_count=None,
        )
        self.assertIsNone(runner._pending_adaptive_pp_reduction)

    @patch("vllm_ascend.worker.model_runner_v1.get_pp_group")
    def test_flush_combines_npu_time_and_deferred_send_tail(
        self,
        mock_get_pp_group,
    ):
        runner = self._build_runner()
        decision = MagicMock()
        start_event = MagicMock()
        end_event = MagicMock()
        start_event.elapsed_time.return_value = 200.0
        runner._pending_adaptive_pp_decision = decision
        runner._pending_adaptive_pp_elapsed_ms = 125.0
        runner._pending_adaptive_pp_npu_events = (start_event, end_event)
        runner._pending_adaptive_pp_send_wait_ms = 25.0
        mock_get_pp_group.return_value = SimpleNamespace(
            world_size=1,
            is_first_rank=True,
        )

        runner._flush_adaptive_pp_microbatch_result()

        runner.adaptive_ubatch_controller.observe.assert_called_once_with(
            decision,
            forward_ms=225.0,
            next_waiting_count=None,
            next_running_count=None,
            next_oldest_wait_ms=None,
            next_pending_first_token_count=None,
            next_oldest_first_token_wait_ms=None,
            next_pending_prefill_tokens=None,
            completed_first_token_count=None,
        )
        start_event.elapsed_time.assert_called_once_with(end_event)

    @patch("vllm_ascend.worker.model_runner_v1.torch.npu.is_available")
    @patch("vllm_ascend.worker.model_runner_v1.time.perf_counter")
    def test_full_step_measurement_is_queued_after_sampling(
        self,
        mock_perf_counter,
        mock_npu_available,
    ):
        runner = self._build_runner()
        decision = MagicMock()
        runner._active_adaptive_pp_decision = decision
        runner._active_adaptive_pp_step_start = 10.0
        mock_perf_counter.return_value = 10.25
        mock_npu_available.return_value = False

        runner._finish_adaptive_pp_step_measurement()

        self.assertIs(runner._pending_adaptive_pp_decision, decision)
        self.assertEqual(runner._pending_adaptive_pp_elapsed_ms, 250.0)
        self.assertIsNone(runner._active_adaptive_pp_decision)
        self.assertIsNone(runner._active_adaptive_pp_step_start)

    def test_deferred_send_wait_is_charged_to_pending_step(self):
        runner = self._build_runner()
        runner._pending_adaptive_pp_elapsed_ms = 250.0

        runner._add_pending_adaptive_pp_send_wait(75.0)

        self.assertEqual(runner._pending_adaptive_pp_elapsed_ms, 250.0)
        self.assertEqual(runner._pending_adaptive_pp_send_wait_ms, 75.0)

    def test_deferred_send_wait_is_ignored_without_pending_measurement(self):
        runner = self._build_runner()

        runner._add_pending_adaptive_pp_send_wait(75.0)

        self.assertIsNone(runner._pending_adaptive_pp_elapsed_ms)
        self.assertEqual(runner._pending_adaptive_pp_send_wait_ms, 0.0)

    def test_finish_is_safe_when_subclass_skips_state_initialization(self):
        runner = self._build_runner()
        del runner._active_adaptive_pp_decision
        del runner._active_adaptive_pp_step_start

        runner._finish_adaptive_pp_step_measurement()

        self.assertIsNone(runner._active_adaptive_pp_decision)
        self.assertIsNone(runner._active_adaptive_pp_step_start)
        self.assertIsNone(runner._pending_adaptive_pp_decision)
        self.assertIsNone(runner._pending_adaptive_pp_elapsed_ms)


class TestNPUModelRunnerDebugger(unittest.TestCase):
    def _build_runner(self, debugger=None):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.debugger = debugger or MagicMock()
        runner.model = MagicMock()
        runner.model_config = MagicMock()
        runner.model_config.enforce_eager = False
        runner._debugger_started = True
        runner._debugger_step_dummy_data_before_execute = False
        runner.use_compress = False
        return runner

    def test_finalize_dump_data_stops_stop_capable_debugger(self):
        runner = self._build_runner()

        runner._finalize_dump_data()

        runner.debugger.stop.assert_called_once_with()
        runner.debugger.step.assert_called_once_with()
        self.assertFalse(runner._debugger_started)

    def test_finalize_dump_data_steps_graph_debugger_without_stop(self):
        debugger = MagicMock(spec=["start", "step"])
        runner = self._build_runner(debugger)

        runner._finalize_dump_data()

        debugger.step.assert_called_once_with()
        self.assertTrue(runner._debugger_started)

    def test_start_dump_data_noop_when_already_started(self):
        runner = self._build_runner(MagicMock(spec=["start", "step"]))

        runner._start_dump_data()

        runner.debugger.start.assert_not_called()
        runner.debugger.step.assert_not_called()
        self.assertTrue(runner._debugger_started)


if __name__ == "__main__":
    unittest.main()
