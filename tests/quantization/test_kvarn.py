# SPDX-License-Identifier: Apache-2.0
"""CPU-only contracts for KVarN configuration and cache accounting."""

import math
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
    KVarNMetadataBuilder,
    _cast_kvarn_activations,
    _get_kvarn_cpu_lengths,
    expand_kvarn_block_table,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.selector import AttentionSelectorConfig
from vllm.v1.core.kv_cache_utils import unify_kv_cache_spec_page_size
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVQuantMode,
    TQFullAttentionSpec,
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


def test_pure_decode_casts_only_query_after_cache_update():
    query = torch.randn(2, 3, dtype=torch.bfloat16)
    key = torch.randn(2, 3, dtype=torch.bfloat16)
    value = torch.randn(2, 3, dtype=torch.bfloat16)

    cast_query, cast_key, cast_value = _cast_kvarn_activations(
        query, key, value, query_only=True
    )

    assert cast_query.dtype == torch.float16
    assert cast_key is key
    assert cast_value is value


def test_prefill_casts_all_activations_for_fp16_kvarn_compute():
    tensors = [torch.randn(2, 3, dtype=torch.bfloat16) for _ in range(3)]
    cast = _cast_kvarn_activations(*tensors, query_only=False)
    assert all(tensor.dtype == torch.float16 for tensor in cast)




def test_compact_k4v4_preset_changes_only_the_physical_record_stride():
    """Compact is a format-bearing preset, never an ambient mode switch."""
    padded = KVarNConfig.from_cache_dtype("kvarn_k4v4_g128", head_dim=256)
    compact = KVarNConfig.from_cache_dtype("kvarn_k4v4_g128_compact", head_dim=256)

    assert (compact.key_bits, compact.value_bits, compact.group) == (4, 4, 128)
    assert compact.tile_bytes == padded.tile_bytes == 35072
    assert padded.tile_bytes_aligned == 65536
    assert compact.tile_bytes_aligned == compact.tile_bytes == 35072
    assert compact.tile_bytes_aligned // compact.group == 274
    for field in (
        "k_packed_offset",
        "k_s_col_offset",
        "k_zp_offset",
        "k_s_row_offset",
        "v_packed_offset",
        "v_s_col_offset",
        "v_s_row_offset",
        "v_zp_offset",
    ):
        assert getattr(compact, field) == getattr(padded, field)


def test_compact_k4v4_preset_has_exact_memory_ratio():
    padded = KVarNConfig.from_cache_dtype("kvarn_k4v4_g128", head_dim=256)
    compact = KVarNConfig.from_cache_dtype("kvarn_k4v4_g128_compact", head_dim=256)
    # 35072 / 65536 reduces exactly to 137 / 256: 46.484375% fewer bytes.
    assert compact.tile_bytes_aligned * 256 == padded.tile_bytes_aligned * 137
    assert (padded.tile_bytes_aligned - compact.tile_bytes_aligned) * 256 == (
        padded.tile_bytes_aligned * 119
    )


def test_compact_preset_allows_stride_aware_native_decoder(monkeypatch):
    monkeypatch.setenv("KVARN_NATIVE_XPU_DECODE", "1")
    impl = KVarNAttentionImpl(
        num_heads=24,
        head_size=256,
        scale=1 / 16,
        num_kv_heads=4,
        kv_cache_dtype="kvarn_k4v4_g128_compact",
    )
    assert impl.kvarn_config.record_bytes == 35072


@pytest.mark.parametrize("cache_dtype", KVARN_PRESETS)
def test_kvarn_presets_are_registered_as_quantized_cache_modes(cache_dtype):
    """The runner must not substitute ``auto`` for a KVarN cache shape."""
    assert get_kv_quant_mode(cache_dtype) == KVQuantMode.KVARN
    assert is_quantized_kv_cache(cache_dtype)


@pytest.mark.parametrize(
    ("name", "raw_tile_bytes", "aligned_tile_bytes", "slot_bytes"),
    [
        ("kvarn_k4v2_g128", 26880, 32768, 256),
        ("kvarn_k4v4_g128", 35072, 65536, 512),
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


def test_kvarn_page_unification_scales_by_whole_quantization_tiles():
    kvarn = TQFullAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=256,
        head_size_v=256,
        dtype=torch.bfloat16,
        kv_quant_mode=KVQuantMode.KVARN,
        tq_slot_size=256,
    )
    native = FullAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=256,
        head_size_v=256,
        dtype=torch.bfloat16,
    )
    unified = unify_kv_cache_spec_page_size({"kvarn": kvarn, "native": native})
    assert unified["kvarn"].block_size == 512
    assert unified["kvarn"].page_size_bytes == unified["native"].page_size_bytes


def test_turboquant_page_unification_retains_scaling_behavior():
    turboquant = TQFullAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=256,
        head_size_v=256,
        dtype=torch.bfloat16,
        tq_slot_size=256,
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
    assert unified["turboquant"].block_size == 512
    assert unified["turboquant"].page_size_bytes == unified["native"].page_size_bytes


def test_macro_block_table_expands_to_stable_tile_ids():
    macro = torch.tensor([[4, 7, -1], [2, -1, -1]], dtype=torch.int32)
    expanded = expand_kvarn_block_table(macro, tiles_per_block=3)
    assert expanded.tolist() == [
        [12, 13, 14, 21, 22, 23, -1, -1, -1],
        [6, 7, 8, -1, -1, -1, -1, -1, -1],
    ]


def test_macro_cache_shape_exposes_one_record_per_tile():
    shape = KVarNAttentionBackend.get_kv_cache_shape(
        num_blocks=5,
        block_size=1664,
        num_kv_heads=4,
        head_size=256,
        cache_dtype_str="kvarn_k4v4_g128",
    )
    assert shape == (65, 4, 65536)


def test_compact_macro_cache_shape_uses_active_record_stride():
    shape = KVarNAttentionBackend.get_kv_cache_shape(
        num_blocks=5,
        block_size=1664,
        num_kv_heads=4,
        head_size=256,
        cache_dtype_str="kvarn_k4v4_g128_compact",
    )
    assert shape == (65, 4, 35072)
    assert math.prod(shape) == 65 * 4 * 35072

    # The allocator accounts in 274-byte logical token slots, while the
    # backend materializes one 35,072-byte record per 128-token tile.  Those
    # two views must stay byte-identical even when page unification grows a
    # physical block to several quantization tiles.
    spec = TQFullAttentionSpec(
        block_size=1664,
        num_kv_heads=4,
        head_size=256,
        head_size_v=256,
        dtype=torch.uint8,
        kv_quant_mode=KVQuantMode.KVARN,
        tq_slot_size=274,
    )
    assert math.prod(shape) == 5 * spec.page_size_bytes


def test_metadata_builder_uses_physical_spec_not_global_logical_block_size():
    spec = TQFullAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=256,
        head_size_v=256,
        dtype=torch.uint8,
        kv_quant_mode=KVQuantMode.KVARN,
        tq_slot_size=512,
    )
    config = SimpleNamespace(
        speculative_config=None,
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        cache_config=SimpleNamespace(
            block_size=16,
            cache_dtype="kvarn_k4v4_g128",
        ),
        model_config=SimpleNamespace(
            max_model_len=8192,
            get_head_size=lambda: 256,
        ),
    )

    builder = KVarNMetadataBuilder(spec, ["attn"], config, torch.device("cpu"))

    assert builder._group == 128
    assert builder._tiles_per_block == 1


def test_metadata_builder_prefers_host_native_lengths_over_staging_tensors():
    class PoisonTensor:
        def tolist(self):
            raise AssertionError("staging tensor read would serialize device work")

    metadata = SimpleNamespace(
        seq_lens_cpu_list=[131, 259],
        query_lens_cpu_list=[3, 3],
        seq_lens_cpu=PoisonTensor(),
        query_start_loc_cpu=PoisonTensor(),
        seq_lens=PoisonTensor(),
    )

    assert _get_kvarn_cpu_lengths(metadata) == ([131, 259], [3, 3])


def test_native_layer_filter_matches_components_not_numeric_prefixes():
    from vllm.v1.attention.ops.triton_kvarn_decode import (
        kvarn_native_layer_selected,
    )

    assert kvarn_native_layer_selected("model.layers.3.self_attn", "layers.3")
    assert not kvarn_native_layer_selected(
        "model.layers.31.self_attn", "layers.3"
    )
    assert kvarn_native_layer_selected(
        "model.layers.31.self_attn", "layers.3, layers.31"
    )
    assert kvarn_native_layer_selected("model.layers.7.self_attn", "")


@pytest.mark.parametrize(
    ("is_prefill", "num_decodes", "has_cached_multiquery", "expected"),
    [
        (True, 0, False, False),
        (True, 0, True, True),
        (True, 1, False, True),
        (False, 0, False, True),
    ],
)
def test_kvarn_preserves_model_dtype_only_for_fresh_prefill(
    is_prefill, num_decodes, has_cached_multiquery, expected
):
    from vllm.v1.attention.backends.kvarn_attn import (
        _kvarn_attention_requires_fp16,
    )

    assert (
        _kvarn_attention_requires_fp16(
            is_prefill=is_prefill,
            num_decodes=num_decodes,
            has_cached_multiquery=has_cached_multiquery,
        )
        is expected
    )
