# SPDX-License-Identifier: Apache-2.0
"""CPU-only contracts for KVarN configuration and cache accounting."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.quantization.kvarn.config import (
    KVARN_PRESETS,
    KVarNConfig,
)
from vllm.platforms.xpu import XPUPlatform
from vllm.v1.attention.backends.kvarn_attn import (
    KVarNAttentionBackend,
    KVarNAttentionImpl,
    _reconcile_kvarn_sink_ownership,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.backends.turboquant_attn import TurboQuantAttentionBackend
from vllm.v1.attention.selector import AttentionSelectorConfig
from vllm.v1.core.kv_cache_utils import unify_kv_cache_spec_page_size
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheLayout,
    KVCacheTensor,
    compute_layer_kv_cache_shape_bytes,
    compute_layout_strides,
    create_kv_cache_views,
    get_kv_quant_mode,
    is_quantized_kv_cache,
)


@pytest.mark.parametrize(
    ("name", "key_bits", "value_bits", "group"),
    [
        ("kvarn_k4v2_g128", 4, 2, 128),
        ("kvarn_k4v4_g128", 4, 4, 128),
        ("kvarn_k4v2_g64", 4, 2, 64),
        ("kvarn_k4v4_g64", 4, 4, 64),
    ],
)
def test_presets_are_an_auditable_fixed_contract(name, key_bits, value_bits, group):
    assert KVARN_PRESETS[name] == {
        "key_bits": key_bits,
        "value_bits": value_bits,
        "group": group,
    }
    config = KVarNConfig.from_cache_dtype(name, head_dim=256)
    assert (config.key_bits, config.value_bits, config.group) == (
        key_bits,
        value_bits,
        group,
    )


def test_unknown_preset_fails_closed():
    with pytest.raises(ValueError, match="Unknown KVarN cache dtype"):
        KVarNConfig.from_cache_dtype("kvarn_k3v2_g128", head_dim=256)


@pytest.mark.parametrize("cache_dtype", KVARN_PRESETS)
def test_kvarn_presets_are_registered_as_quantized_cache_modes(cache_dtype):
    assert get_kv_quant_mode(cache_dtype).is_kvarn
    assert is_quantized_kv_cache(cache_dtype)


@pytest.mark.parametrize(
    ("name", "raw_tile_bytes", "aligned_tile_bytes", "slot_bytes"),
    [
        ("kvarn_k4v2_g128", 26880, 32768, 256),
        ("kvarn_k4v4_g128", 35072, 65536, 512),
        ("kvarn_k4v4_g128_compact", 35072, 35072, 274),
        ("kvarn_k4v2_g64", 14208, 16384, 256),
        ("kvarn_k4v4_g64", 18304, 32768, 512),
    ],
)
def test_head_dim_256_layout_is_contiguous_and_page_aligned(
    name, raw_tile_bytes, aligned_tile_bytes, slot_bytes
):
    config = KVarNConfig.from_cache_dtype(name, head_dim=256)
    assert config.tile_bytes == raw_tile_bytes
    assert config.tile_bytes_aligned == aligned_tile_bytes
    assert config.tile_bytes_aligned // config.group == slot_bytes
    assert config.tile_bytes_aligned >= config.tile_bytes

    regions = [
        (config.k_packed_offset, config.k_packed_bytes),
        (config.k_s_col_offset, config.head_dim * 2),
        (config.k_zp_offset, config.head_dim * 2),
        (config.k_s_row_offset, config.group * 2),
        (config.v_packed_offset, config.v_packed_bytes),
        (config.v_s_col_offset, config.head_dim * 2),
        (config.v_s_row_offset, config.group * 2),
        (config.v_zp_offset, config.group * 2),
    ]
    assert regions[0][0] == 0
    for (offset, size), (next_offset, _) in zip(regions, regions[1:]):
        assert offset + size == next_offset
    assert regions[-1][0] + regions[-1][1] == config.tile_bytes


def test_compact_d256_k4v4_record_has_no_power_of_two_padding():
    config = KVarNConfig.from_cache_dtype("kvarn_k4v4_g128_compact", head_dim=256)
    assert config.compact_records
    assert config.record_bytes == config.tile_bytes == 35_072
    assert 4 * config.record_bytes == 140_288
    spec = KVarNAttentionBackend.customize_spec(
        FullAttentionSpec(
            block_size=128,
            num_kv_heads=4,
            head_size=256,
            dtype=torch.bfloat16,
            kv_quant_mode=get_kv_quant_mode("kvarn_k4v4_g128_compact"),
        )
    )
    assert spec.page_size_bytes == 140_288
    assert spec.state_content_size_bytes == 35_072


class _ModelConfig:
    def __init__(self, attention_layers, total_layers=64):
        self.attention_layers = attention_layers
        self.total_layers = total_layers

    def get_num_layers_by_block_type(self, parallel_config, block_type):
        assert block_type == "attention"
        if self.attention_layers is None:
            raise RuntimeError("block-type metadata unavailable")
        return self.attention_layers

    def get_num_layers(self, parallel_config):
        return self.total_layers


def test_hybrid_pool_counts_only_full_attention_layers():
    model = _ModelConfig(attention_layers=16, total_layers=64)
    assert KVarNConfig.num_kvarn_layers(model, SimpleNamespace()) == 16


def test_layer_count_falls_back_when_block_metadata_is_unavailable():
    model = _ModelConfig(attention_layers=None, total_layers=64)
    assert KVarNConfig.num_kvarn_layers(model, SimpleNamespace()) == 64


def test_attention_free_pipeline_rank_does_not_allocate_a_pool():
    model = _ModelConfig(attention_layers=0, total_layers=16)
    assert KVarNConfig.num_kvarn_layers(model, SimpleNamespace()) == 0


@pytest.mark.parametrize("cache_dtype", KVARN_PRESETS)
def test_xpu_routes_every_kvarn_preset_to_kvarn_backend(cache_dtype):
    selector = AttentionSelectorConfig(
        head_size=256,
        dtype=torch.bfloat16,
        kv_cache_dtype=cache_dtype,
        block_size=128,
    )
    assert (
        XPUPlatform.get_attn_backend_cls(AttentionBackendEnum.FLASH_ATTN, selector)
        == AttentionBackendEnum.KVARN.get_path()
    )


def test_kvarn_current_layout_uses_one_packed_record_per_tile():
    base = FullAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=256,
        head_size_v=256,
        dtype=torch.uint8,
        kv_quant_mode=get_kv_quant_mode("kvarn_k4v4_g128"),
    )
    kvarn = KVarNAttentionBackend.customize_spec(base)
    config = KVarNConfig.from_cache_dtype("kvarn_k4v4_g128", head_dim=256)
    assert kvarn.tokens_per_state == config.group
    assert kvarn.state_content_size_bytes == config.tile_bytes_aligned
    assert compute_layer_kv_cache_shape_bytes(kvarn, 3) == (
        3,
        1,
        1,
        config.tile_bytes_aligned,
    )

    layer_stride, block_stride, _, _, _ = compute_layout_strides(
        kvarn, 3, 1, KVCacheLayout.LBHNC
    )
    tensor = KVCacheTensor(
        size=kvarn.page_size_bytes * 3,
        layers=["kvarn"],
        layer_stride=layer_stride,
        block_stride=block_stride,
    )
    raw = torch.empty(tensor.size, dtype=torch.int8)
    (view,) = create_kv_cache_views(raw, kvarn, 3, KVCacheLayout.LBHNC, tensor)
    assert view.dtype == torch.uint8
    assert view.shape == (3, 1, 1, config.tile_bytes_aligned)
    records = KVarNAttentionImpl._record_cache_view(view)
    assert records.shape == (3, 1, config.tile_bytes_aligned)
    assert records.stride() == view.squeeze(2).stride()


def test_kvarn_forward_profiles_with_empty_cache_sentinel():
    impl = object.__new__(KVarNAttentionImpl)
    impl.kvarn_config = SimpleNamespace(record_bytes=35072)
    impl.num_heads = 2
    impl.num_kv_heads = 1
    impl.head_size = 4
    pool_calls = []
    impl._ensure_pool = lambda device, num_blocks_hint: pool_calls.append(
        (device, num_blocks_hint)
    )
    impl._prefill_first_chunk = (
        lambda q, k, v, metadata, cache: torch.full_like(q, 2)
    )
    metadata = SimpleNamespace(
        num_actual_tokens=3,
        is_prefill=True,
        num_decodes=0,
        num_decode_tokens=0,
        has_cached_multiquery=False,
        vq_seqlen=None,
    )
    query = torch.zeros(3, 8, dtype=torch.float16)
    key = torch.zeros(3, 4, dtype=torch.float16)
    value = torch.zeros_like(key)
    empty_cache = torch.empty(0, dtype=torch.uint8)

    output = impl.forward(None, query, key, value, empty_cache, metadata)

    assert pool_calls == [(query.device, 0)]
    assert output.shape == query.shape
    assert torch.equal(output, torch.full_like(query, 2))
    assert not hasattr(impl, "_kv_cache_ref")


def test_immediately_recycled_sink_is_delabeled_outside_new_row_zero():
    """LIFO reuse must not carry a finished request's sink role into history."""
    old_sink = 5
    new_sink = 7
    sinks = {old_sink, new_sink}
    is_sink_t = torch.zeros(16, dtype=torch.bool)
    # The failure requires immediate reuse, before an absence step can retire it.
    retired_sinks = {}
    is_sink_t[list(sinks)] = True

    _reconcile_kvarn_sink_ownership(
        sinks=sinks,
        retired_sinks=retired_sinks,
        blocks_needed={new_sink, old_sink},
        row0_set={new_sink},
        is_sink_t=is_sink_t,
    )

    assert sinks == {new_sink}
    assert not is_sink_t[old_sink]
    assert is_sink_t[new_sink]


def test_kvarn_page_unification_scales_by_whole_quantization_tiles():
    kvarn = KVarNAttentionBackend.customize_spec(
        FullAttentionSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=256,
            head_size_v=256,
            dtype=torch.bfloat16,
            kv_quant_mode=get_kv_quant_mode("kvarn_k4v4_g128"),
        )
    )
    native = FullAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=256,
        head_size_v=256,
        dtype=torch.bfloat16,
    )
    unified = unify_kv_cache_spec_page_size({"kvarn": kvarn, "native": native})
    assert unified["kvarn"].block_size % kvarn.tokens_per_state == 0
    assert unified["kvarn"].page_size_bytes == unified["native"].page_size_bytes


def test_turboquant_page_unification_retains_scaling_behavior():
    turboquant = TurboQuantAttentionBackend.customize_spec(
        FullAttentionSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=256,
            head_size_v=256,
            dtype=torch.bfloat16,
            kv_quant_mode=get_kv_quant_mode("turboquant_4bit_nc"),
        )
    )
    native = FullAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=256,
        head_size_v=256,
        dtype=torch.bfloat16,
    )
    unified = unify_kv_cache_spec_page_size(
        {"turboquant": turboquant, "native": native}
    )
    assert unified["turboquant"].block_size >= turboquant.block_size
    assert unified["turboquant"].page_size_bytes == unified["native"].page_size_bytes
