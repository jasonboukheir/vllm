# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import torch

from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCachePoolSpec,
)
from vllm.v1.worker.utils import copy_kv_cache_blocks_inplace


def test_block_copy_is_scoped_to_owning_independent_pool():
    attention = torch.arange(10, dtype=torch.int32).view(5, 2)
    mamba = torch.arange(12, dtype=torch.int32).view(3, 4)
    attention_before = attention.clone()
    mamba_source = mamba[1].clone()
    config = KVCacheConfig(
        num_blocks=3,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["attention"], Mock()),
            KVCacheGroupSpec(["mamba"], Mock()),
        ],
        kv_cache_pools=[
            KVCachePoolSpec(num_blocks=5, group_ids=[0]),
            KVCachePoolSpec(num_blocks=3, group_ids=[1]),
        ],
    )
    context = {
        "attention": SimpleNamespace(kv_cache=attention),
        "mamba": SimpleNamespace(kv_cache=[mamba]),
    }

    copy_kv_cache_blocks_inplace(
        config,
        context,
        [KVCacheBlockCopy(src_block_id=1, dst_block_id=2, group_id=1)],
    )

    torch.testing.assert_close(mamba[2], mamba_source)
    torch.testing.assert_close(attention, attention_before)
