# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.worker.block_table import (
    BlockTable,
    SlotMappingMode,
)


def _make_cpu_block_table() -> BlockTable:
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
    )


def test_cpu_block_table_row_lifecycle():
    table = _make_cpu_block_table()
    table.add_row([7, 9], 0)
    table.append_row([11], 0)
    assert table.num_blocks_per_row[0] == 3
    assert table.block_table.np[0, :3].tolist() == [7, 9, 11]

    table.move_row(0, 1)
    assert table.num_blocks_per_row[:2].tolist() == [0, 3]
    assert table.block_table.np[0, :3].tolist() == [0, 0, 0]
    assert table.block_table.np[1, :3].tolist() == [7, 9, 11]

    table.clear_row(1)
    assert table.num_blocks_per_row[1] == 0
    assert table.block_table.np[1, :3].tolist() == [0, 0, 0]
