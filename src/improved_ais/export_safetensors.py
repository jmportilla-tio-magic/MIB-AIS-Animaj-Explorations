from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import yaml


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_training_checkpoint(path: Path) -> dict[str, Any]:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required to export a training checkpoint.") from exc

    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict) or "model" not in state:
        raise RuntimeError(f"expected a standalone improved AIS training checkpoint with a 'model' state dict: {path}")
    return state


def export_safetensors(*, checkpoint: Path, config: Path, output_dir: Path) -> dict[str, Path]:
    try:
        from safetensors.torch import save_file
    except ModuleNotFoundError as exc:
        raise RuntimeError("safetensors is required. Install with `pip install -e .[hf]`.") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    state = _load_training_checkpoint(checkpoint)
    training_config = _load_yaml(config)
    model_config = dict(training_config.get("model", {"type": "improved_ais_bilstm"}))
    if str(model_config.get("type", "improved_ais_bilstm")) != "improved_ais_bilstm":
        raise RuntimeError("only improved_ais_bilstm checkpoints can be exported by this package")

    tensors = {key: value.detach().cpu().contiguous() for key, value in state["model"].items()}
    metadata = {
        "format": "pt",
        "model_type": "improved_ais_bilstm",
        "model_config": json.dumps(model_config, sort_keys=True),
    }
    model_path = output_dir / "model.safetensors"
    model_config_path = output_dir / "model_config.yaml"
    training_config_path = output_dir / "training_config.yaml"
    save_file(tensors, str(model_path), metadata=metadata)
    with model_config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump({"model": model_config}, f, sort_keys=False)
    shutil.copyfile(config, training_config_path)
    return {
        "checkpoint": model_path,
        "model_config": model_config_path,
        "training_config": training_config_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export an improved AIS .pt checkpoint to release safetensors.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    outputs = export_safetensors(checkpoint=args.checkpoint, config=args.config, output_dir=args.output_dir)
    print(yaml.safe_dump({key: str(value) for key, value in outputs.items()}, sort_keys=True).strip())


if __name__ == "__main__":
    main()
