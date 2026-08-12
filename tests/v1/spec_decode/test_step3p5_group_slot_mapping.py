from types import SimpleNamespace

import torch

from vllm.v1.spec_decode.step3p5 import _compute_group_slot_mapping


def _group(block_size: int):
    builder = SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=block_size))
    return SimpleNamespace(get_metadata_builder=lambda: builder)


def test_group_slot_mapping_uses_each_groups_block_size():
    positions = torch.tensor([15, 16, 47, 48, 127, 128])
    exceeds = torch.zeros_like(positions, dtype=torch.bool)
    block_table = torch.tensor(
        [
            [10, 11, 12, 13, 14, 15, 16, 17, 18],
            [20, 21, 22, 23, 24, 25, 26, 27, 28],
            [30, 31, 32, 33, 34, 35, 36, 37, 38],
            [40, 41, 42, 43, 44, 45, 46, 47, 48],
            [50, 51, 52, 53, 54, 55, 56, 57, 58],
            [60, 61, 62, 63, 64, 65, 66, 67, 68],
        ]
    )

    recurrent = _compute_group_slot_mapping(
        _group(16), block_table, positions, exceeds, len(positions)
    )
    attention = _compute_group_slot_mapping(
        _group(128), block_table, positions, exceeds, len(positions)
    )

    assert recurrent.tolist() == [175, 336, 527, 688, 927, 1088]
    assert attention.tolist() == [1295, 2576, 3887, 5168, 6527, 7808]
    assert recurrent[3] != attention[3]


def test_group_slot_mapping_masks_positions_past_model_limit():
    positions = torch.tensor([15, 16])
    exceeds = torch.tensor([False, True])
    block_table = torch.tensor([[3, 4], [5, 6]])

    result = _compute_group_slot_mapping(
        _group(16), block_table, positions, exceeds, len(positions)
    )

    assert result.tolist() == [63, -1]
