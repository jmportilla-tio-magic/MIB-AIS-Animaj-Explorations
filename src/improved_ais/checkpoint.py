from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from improved_ais.models.improved_ais import build_improved_ais_bilstm, improved_model_config_from_path


@dataclass(frozen=True)
class CheckpointSpec:
    label: str
    path: Path


def parse_checkpoint(value: str) -> CheckpointSpec:
    if "=" in value:
        label, path = value.split("=", 1)
        return CheckpointSpec(label=label, path=Path(path))
    path = Path(value)
    label = path.parent.name if path.name in {"best.pt", "last.pt", "model.safetensors"} else path.stem
    return CheckpointSpec(label=label, path=path)


def resolve_device(requested: str):
    import torch

    requested = requested.lower()
    if requested == "auto":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if requested == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS was requested but is not available.")
        return torch.device("mps")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError(f"unsupported device: {requested}")


def _load_pt_model(path: Path, *, device: str):
    import torch

    try:
        state = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(path, map_location=device)
    config = state.get("config", {}) if isinstance(state, dict) else {}
    model_cfg = config.get("model", {})
    if str(model_cfg.get("type", "improved_ais_bilstm")) != "improved_ais_bilstm":
        raise RuntimeError(f"checkpoint is not an improved_ais_bilstm training checkpoint: {path}")
    model = build_improved_ais_bilstm(model_cfg, device=device)
    model.load_state_dict(state["model"])
    model.eval()
    return model


def _load_safetensors_model(path: Path, *, device: str):
    from safetensors.torch import load_file

    tensor_path = path / "model.safetensors" if path.is_dir() else path
    model_cfg = improved_model_config_from_path(path)
    if model_cfg is None:
        cfg_path = tensor_path.parent / "model_config.yaml"
        if cfg_path.exists():
            with cfg_path.open("r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            model_cfg = raw.get("model", raw)
    model_cfg = model_cfg or {"type": "improved_ais_bilstm"}
    model = build_improved_ais_bilstm(model_cfg, device=device)
    state = load_file(str(tensor_path), device="cpu" if str(device).startswith("mps") else device)
    model.load_state_dict(state)
    model.eval()
    return model


def load_improved_model(checkpoint: str | Path, *, device: str = "cpu"):
    path = Path(checkpoint)
    if path.is_dir() or path.suffix == ".safetensors":
        return _load_safetensors_model(path, device=device)
    if path.suffix == ".pt":
        return _load_pt_model(path, device=device)
    raise ValueError(f"unsupported checkpoint path: {path}")
