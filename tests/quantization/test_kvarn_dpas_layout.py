# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for the feature-flagged Xe2 KVarN payload layout."""

from vllm.v1.attention.ops.triton_kvarn_flush import (
    kvarn_dpas_k_coord,
    kvarn_dpas_v_coord,
)


def test_kvarn_dpas_k_fragment_coordinates_are_bijective():
    coords = {
        kvarn_dpas_k_coord(lane, slot)
        for lane in range(16)
        for slot in range(64)
    }
    assert coords == {(token, dim) for token in range(16) for dim in range(64)}


def test_kvarn_dpas_v_fragment_coordinates_are_bijective():
    coords = {
        kvarn_dpas_v_coord(lane, slot)
        for lane in range(16)
        for slot in range(32)
    }
    assert coords == {(dim, token) for dim in range(32) for token in range(16)}
