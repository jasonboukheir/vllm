# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCachePoolSpec,
    MambaSpec,
)
from vllm.v1.worker.utils import (
    copy_kv_cache_blocks_inplace,
    zero_mamba_block_ids,
)


def _config() -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=3,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["attention"],
                FullAttentionSpec(
                    block_size=1,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.int32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=1,
                    shapes=((4,),),
                    dtypes=(torch.int32,),
                    mamba_cache_mode="none",
                ),
            ),
        ],
        kv_cache_pools=[
            KVCachePoolSpec(num_blocks=5, group_ids=[0]),
            KVCachePoolSpec(num_blocks=3, group_ids=[1]),
        ],
    )


def test_block_copy_is_scoped_to_owning_independent_pool():
    attention = torch.arange(10, dtype=torch.int32).view(5, 2)
    mamba = torch.arange(12, dtype=torch.int32).view(3, 4)
    attention_before = attention.clone()
    mamba_source = mamba[1].clone()
    context = {
        "attention": SimpleNamespace(kv_cache=attention),
        "mamba": SimpleNamespace(kv_cache=[mamba]),
    }

    copy_kv_cache_blocks_inplace(
        _config(),
        context,
        [KVCacheBlockCopy(src_block_id=1, dst_block_id=2, group_id=1)],
    )

    torch.testing.assert_close(mamba[2], mamba_source)
    torch.testing.assert_close(attention, attention_before)


def test_mamba_zeroing_does_not_cross_pool_namespace():
    attention = torch.full((5, 2), 7, dtype=torch.int32)
    mamba = torch.full((3, 4), 9, dtype=torch.int32)
    context = {
        "attention": SimpleNamespace(kv_cache=attention),
        "mamba": SimpleNamespace(kv_cache=[mamba]),
    }

    zero_mamba_block_ids(_config(), context, [(1, [1])], torch.device("cpu"))

    assert not torch.any(mamba[1])
    assert torch.all(mamba[0] == 9)
    assert torch.all(attention == 7)
