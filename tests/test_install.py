from __future__ import annotations

import torch

from improved_ais.checkpoint import load_improved_model


def test_release_weights_forward_shape():
    model = load_improved_model("weights", device="cpu")
    model.eval()
    with torch.no_grad():
        input_seq = torch.zeros(1, 224, 597)
        input_seq[..., -1] = 1.0
        prev_pose = torch.zeros(1, 224, 596)
        next_pose = torch.zeros(1, 224, 596)
        temporal = torch.zeros(1, 224, 4)
        out = model(input_seq, prev_pose, next_pose, temporal_features=temporal)
    assert tuple(out["pred"].shape) == (1, 224, 596)
