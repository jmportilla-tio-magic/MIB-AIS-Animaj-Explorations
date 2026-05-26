from __future__ import annotations

import numpy as np

from improved_ais.data.preprocessing import apply_relative_root_translation, invalid_scene_keys_from_ranges
from improved_ais.eval_official import _official_shifted_distance


def test_shifted_distance_uses_elementwise_l1_sum():
    target = np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    pred = np.zeros_like(target)
    score = _official_shifted_distance(
        target=target,
        pred=pred,
        animation_keyframes=np.ones(3, dtype=bool),
        unmasked_frames=[0, 2],
    )
    assert score == 7.0


def test_relative_root_translation_matches_upstream_indices():
    vectors = np.asarray([[10.0, 20.0, 30.0, 1.0], [13.0, 18.0, 35.0, 2.0]], dtype=np.float32)
    out = apply_relative_root_translation(vectors)
    np.testing.assert_allclose(out, [[0.0, 0.0, 0.0, 1.0], [3.0, -2.0, 5.0, 2.0]])


def test_ranges_filter_uses_x_main_translation_threshold(tmp_path):
    ranges = tmp_path / "ranges.csv"
    ranges.write_text(
        "episode_id,scene_id,x_main_CTRL:translateX,x_main_CTRL:translateY,x_main_CTRL:translateZ\n"
        "ep1,scn1,99,0,0\n"
        "ep2,scn2,101,0,0\n",
        encoding="utf-8",
    )
    assert invalid_scene_keys_from_ranges(ranges, threshold=100) == {("ep2", "scn2")}
