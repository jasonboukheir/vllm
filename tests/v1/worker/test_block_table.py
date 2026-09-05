# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.worker.block_table import (
    BlockTable,
    MultiGroupBlockTable,
    SlotMappingMode,
)


def _make_cpu_block_table(*, track_row_versions: bool = False) -> BlockTable:
    return BlockTable(
        block_size=16,
        max_num_reqs=4,
        max_num_blocks_per_req=8,
        max_num_batched_tokens=64,
        pin_memory=False,
        device=torch.device("cpu"),
        kernel_block_size=16,
        cp_kv_cache_interleave_size=1,
        slot_mapping_mode=SlotMappingMode.NONE,
        track_row_versions=track_row_versions,
    )


def test_row_version_tracking_is_opt_in():
    block_table = _make_cpu_block_table()

    block_table.add_row([7, 8], row_idx=0)
    block_table.append_row([9], row_idx=0)
    block_table.move_row(0, 1)
    block_table.swap_row(0, 1)
    block_table.clear_row(0)
    block_table.clear()

    assert not hasattr(block_table, "row_versions")
    with pytest.raises(RuntimeError, match="row version tracking is disabled"):
        block_table.get_row_versions(2)


def test_row_mutations_are_versioned_when_enabled():
    block_table = _make_cpu_block_table(track_row_versions=True)

    block_table.add_row([7, 8], row_idx=0)
    block_table.add_row([4, 5], row_idx=1)
    assert block_table.get_row_versions(2).tolist() == [1, 1]

    block_table.append_row([9], row_idx=0)
    assert block_table.get_row_versions(2).tolist() == [2, 1]

    block_table.move_row(1, 0)
    assert block_table.get_row_versions(2).tolist() == [3, 2]

    block_table.swap_row(0, 1)
    assert block_table.get_row_versions(2).tolist() == [4, 3]

    block_table.clear_row(0)
    assert block_table.get_row_versions(2).tolist() == [5, 3]

    block_table.add_row([], row_idx=1)
    assert block_table.get_row_versions(2).tolist() == [5, 4]

    block_table.clear()
    assert block_table.get_row_versions(2).tolist() == [6, 5]


@pytest.mark.parametrize("track_row_versions", [False, True])
def test_multigroup_propagates_row_version_capability(track_row_versions):
    block_tables = MultiGroupBlockTable(
        max_num_reqs=2,
        max_num_batched_tokens=32,
        pin_memory=False,
        device=torch.device("cpu"),
        block_sizes=[16, 32],
        kernel_block_sizes=[16, 32],
        max_num_blocks=[4, 2],
        slot_mapping_modes=[SlotMappingMode.NONE, SlotMappingMode.NONE],
        track_row_versions=track_row_versions,
    )

    assert all(
        hasattr(block_table, "row_versions") is track_row_versions
        for block_table in block_tables.block_tables
    )
