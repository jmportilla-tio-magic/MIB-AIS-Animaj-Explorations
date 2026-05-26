from __future__ import annotations

from pathlib import Path


def main() -> None:
    import numpy as np
    import torch

    from improved_ais.checkpoint import load_improved_model

    weights = Path.cwd() / "weights"
    if not (weights / "model.safetensors").exists():
        weights = Path(__file__).resolve().parents[2] / "weights"
    model = load_improved_model(weights, device="cpu")
    model.eval()
    with torch.no_grad():
        batch = 1
        length = 224
        pose_dim = 596
        input_seq = torch.zeros(batch, length, pose_dim + 1)
        input_seq[..., -1] = 1.0
        prev_pose = torch.zeros(batch, length, pose_dim)
        next_pose = torch.zeros(batch, length, pose_dim)
        temporal = torch.from_numpy(np.zeros((batch, length, 4), dtype=np.float32))
        out = model(input_seq, prev_pose, next_pose, temporal_features=temporal)
    if tuple(out["pred"].shape) != (1, 224, 596):
        raise SystemExit(f"unexpected output shape: {tuple(out['pred'].shape)}")
    print("improved_ais install ok")
    print(f"torch={torch.__version__}")
    print(f"weights={weights / 'model.safetensors'}")


if __name__ == "__main__":
    main()
