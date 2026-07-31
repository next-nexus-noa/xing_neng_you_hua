# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace

import torch

from vllm_ascend.attention.utils import (
    split_decodes_and_prefills,
)


def test_compute_aware_group_preserves_short_prefill_phase() -> None:
    metadata = SimpleNamespace(
        prefill_context_parallel_metadata=None,
        max_query_len=1,
        num_reqs=2,
        num_actual_tokens=2,
        query_start_loc_cpu=torch.tensor([0, 1, 2]),
        is_prefilling=torch.tensor([False, True]),
        compute_aware_grouping=True,
    )

    assert split_decodes_and_prefills(
        metadata,
        decode_threshold=1,
    ) == (1, 1, 1, 1)


def test_standard_group_keeps_query_length_classification() -> None:
    metadata = SimpleNamespace(
        prefill_context_parallel_metadata=None,
        max_query_len=1,
        num_reqs=2,
        num_actual_tokens=2,
        query_start_loc_cpu=torch.tensor([0, 1, 2]),
        is_prefilling=torch.tensor([False, True]),
        compute_aware_grouping=False,
    )

    assert split_decodes_and_prefills(
        metadata,
        decode_threshold=1,
    ) == (2, 0, 2, 0)
