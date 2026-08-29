# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest

from tools.kvarn_compare_logits import compare, load_artifact


def _write_artifact(path, *, prompt, forced, rows, full_logits=False):
    width = max(len(row) for row in rows)
    token_ids = np.full((len(rows), width), -1, dtype=np.int32)
    logits = np.full((len(rows), width), np.nan, dtype=np.float32)
    for index, row in enumerate(rows):
        for column, (token_id, value) in enumerate(row.items()):
            token_ids[index, column] = token_id
            logits[index, column] = value
    np.savez(
        path,
        prompt_token_ids=np.asarray(prompt, dtype=np.int32),
        forced_token_ids=np.asarray(forced, dtype=np.int32),
        logit_token_ids=token_ids,
        raw_logits=logits,
        full_logits=np.asarray(full_logits),
    )


def test_compare_reports_agreement_errors_and_context_drift(tmp_path):
    reference_path = tmp_path / "bf16.npz"
    candidate_path = tmp_path / "kvarn.npz"
    _write_artifact(
        reference_path,
        prompt=[1, 2, 3],
        forced=[7, 8],
        rows=[{7: 2.0, 9: 2.0, 4: 0.0}, {8: 3.0, 5: 2.0, 6: 1.0}],
    )
    _write_artifact(
        candidate_path,
        prompt=[1, 2, 3],
        forced=[7, 8],
        rows=[{9: 2.0, 7: 1.9995, 4: 0.1}, {8: 2.5, 6: 2.0, 5: 1.0}],
    )

    result = compare(
        load_artifact(reference_path),
        load_artifact(candidate_path),
        tie_tolerance=0.001,
        boundaries=(4,),
    )

    assert result["decode_steps"] == 2
    assert result["top1_agreement_rate"] == 0.5
    assert result["tie_aware_top1_agreement_rate"] == 1.0
    assert result["top5_exact_agreement_rate"] == 1.0
    assert result["top5_mean_jaccard"] == 1.0
    assert result["mean_intersection_coverage"] == 1.0
    assert result["selected_token_delta"]["count"] == 2
    assert result["selected_token_delta"]["max_abs"] == pytest.approx(0.5)
    assert set(result["drift_by_context"]) == {"1-4", "5+"}


def test_compare_rejects_different_forced_sequence(tmp_path):
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    _write_artifact(first, prompt=[1], forced=[2], rows=[{2: 1.0}])
    _write_artifact(second, prompt=[1], forced=[3], rows=[{3: 1.0}])

    with pytest.raises(ValueError, match="forced token IDs differ"):
        compare(
            load_artifact(first),
            load_artifact(second),
            tie_tolerance=0.0,
            boundaries=(4096,),
        )


def test_compare_requires_selected_token_logits(tmp_path):
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    _write_artifact(first, prompt=[1], forced=[2], rows=[{2: 1.0, 4: 0.0}])
    _write_artifact(second, prompt=[1], forced=[2], rows=[{4: 0.0, 5: -1.0}])

    with pytest.raises(ValueError, match="forced token 2 is absent"):
        compare(
            load_artifact(first),
            load_artifact(second),
            tie_tolerance=0.0,
            boundaries=(4096,),
        )
