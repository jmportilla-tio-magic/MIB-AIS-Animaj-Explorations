from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn


class ImprovedAISBiLSTM(nn.Module):
    """AIS-BiLSTM variant for scratch experiments.

    Adds explicit temporal progress conditioning and a grouped beta gate. The
    grouped gate reduces per-dimension switching noise by predicting beta for
    contiguous controller-dimension groups, then expanding it to pose space.
    """

    uses_temporal_features = True

    def __init__(
        self,
        *,
        pose_dim: int = 596,
        input_dim: int = 597,
        temporal_dim: int = 4,
        hidden_size: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3003,
        synthesis_hidden: int = 1024,
        beta_groups: int = 64,
    ) -> None:
        super().__init__()
        self.pose_dim = int(pose_dim)
        self.input_dim = int(input_dim)
        self.temporal_dim = int(temporal_dim)
        self.beta_groups = max(1, min(int(beta_groups), self.pose_dim))

        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        head_dim = hidden_size * 2 + self.temporal_dim
        self.layer_alpha = nn.Sequential(nn.Linear(head_dim, self.pose_dim))
        self.layer_beta_group = nn.Sequential(nn.Linear(head_dim, self.beta_groups))
        self.layer_synthesis = nn.Sequential(
            nn.Linear(head_dim, synthesis_hidden),
            nn.ReLU(),
            nn.Linear(synthesis_hidden, self.pose_dim),
        )

        group_ids = torch.arange(self.pose_dim) * self.beta_groups // self.pose_dim
        self.register_buffer("beta_group_ids", group_ids.long(), persistent=False)

    def forward(
        self,
        input_seq: torch.Tensor,
        prev_pose: torch.Tensor,
        next_pose: torch.Tensor,
        *,
        temporal_features: torch.Tensor | None = None,
        observed_mask: torch.Tensor | None = None,
        keypose_values: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        h, _ = self.lstm(input_seq)
        if temporal_features is None:
            temporal_features = h.new_zeros((*h.shape[:2], self.temporal_dim))
        h = torch.cat([h, temporal_features], dim=-1)

        alpha = torch.sigmoid(self.layer_alpha(h))
        p_interp = (1.0 - alpha) * prev_pose + alpha * next_pose
        p_synth = self.layer_synthesis(h)

        beta_group = torch.sigmoid(self.layer_beta_group(h))
        beta = beta_group.index_select(-1, self.beta_group_ids)
        pred = (1.0 - beta) * p_interp + beta * p_synth

        if observed_mask is not None and keypose_values is not None:
            pred = torch.where(observed_mask[..., None].bool(), keypose_values, pred)
        return {
            "pred": pred,
            "alpha": alpha,
            "beta": beta,
            "beta_group": beta_group,
            "p_interp": p_interp,
            "p_synth": p_synth,
        }


def build_improved_ais_bilstm(config: dict | None = None, *, device: str = "cpu") -> ImprovedAISBiLSTM:
    config = config or {}
    model = ImprovedAISBiLSTM(
        pose_dim=int(config.get("pose_dim", 596)),
        input_dim=int(config.get("input_dim", 597)),
        temporal_dim=int(config.get("temporal_dim", 4)),
        hidden_size=int(config.get("hidden_size", 512)),
        num_layers=int(config.get("num_layers", 2)),
        dropout=float(config.get("dropout", 0.3003)),
        synthesis_hidden=int(config.get("synthesis_hidden", 1024)),
        beta_groups=int(config.get("beta_groups", 64)),
    )
    return model.to(device)


def improved_model_config_from_path(checkpoint_path: str | Path) -> dict[str, Any] | None:
    path = Path(checkpoint_path)
    base = path if path.is_dir() else path.parent
    model_config = base / "model_config.yaml"
    training_config = base / "training_config.yaml"
    if model_config.exists():
        with model_config.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get("model", cfg)
    if training_config.exists():
        with training_config.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        model_cfg = cfg.get("model", {})
        if str(model_cfg.get("type", "")) == "improved_ais_bilstm":
            return model_cfg
    tensor_path = path / "model.safetensors" if path.is_dir() else path
    if tensor_path.suffix == ".safetensors" and tensor_path.exists():
        try:
            from safetensors import safe_open

            with safe_open(str(tensor_path), framework="pt", device="cpu") as f:
                raw = f.metadata().get("model_config") if f.metadata() else None
            model_cfg = json.loads(raw) if raw else None
            if isinstance(model_cfg, dict) and str(model_cfg.get("type", "")) == "improved_ais_bilstm":
                return model_cfg
        except Exception:
            pass
    return None


def load_improved_ais_bilstm(checkpoint_path: str | Path, *, device: str = "cpu"):
    from safetensors.torch import load_file

    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.is_dir():
        tensor_path = checkpoint_path / "model.safetensors"
    else:
        tensor_path = checkpoint_path

    model_cfg = improved_model_config_from_path(checkpoint_path)
    state = load_file(str(tensor_path), device="cpu" if str(device).startswith("mps") else device)
    model_cfg = model_cfg or {}
    if str(model_cfg.get("type", "improved_ais_bilstm")) != "improved_ais_bilstm":
        raise RuntimeError(f"not an improved_ais_bilstm checkpoint: {checkpoint_path}")
    model = build_improved_ais_bilstm(model_cfg, device=device)
    model.load_state_dict(state)
    model.eval()
    return model
