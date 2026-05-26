from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from improved_ais.data import AnimajWindowDataset, torch_collate_ais


@dataclass(frozen=True)
class WindowDatasetSpec:
    name: str
    role: str
    controller_h5: str
    block_keyframes_h5: str
    stride: int
    max_clips: int | None = None
    max_windows: int | None = None


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _merge_cli_config(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    cfg = dict(config)
    cfg.setdefault("data", {})
    cfg.setdefault("training", {})
    cfg.setdefault("eval", {})
    if args.max_train_clips is not None:
        cfg["data"]["max_train_clips"] = args.max_train_clips
    if args.max_val_clips is not None:
        cfg["data"]["max_val_clips"] = args.max_val_clips
    if args.device is not None:
        cfg["training"]["device"] = args.device
    if args.init_checkpoint is not None:
        cfg["training"]["init_checkpoint"] = str(args.init_checkpoint)
    if args.output_dir is not None:
        cfg["training"]["output_dir"] = str(args.output_dir)
    if getattr(args, "max_steps", None) is not None:
        cfg["training"]["max_steps"] = args.max_steps
    if args.eval_only:
        cfg["training"]["eval_only"] = True
    return cfg


def _require_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required. Install with `pip install -e .[train,hf]`.") from exc
    return torch


def resolve_device(requested: str):
    torch = _require_torch()
    requested = requested.lower()
    if requested == "auto":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if requested == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS was requested but is not available in this PyTorch install.")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available in this PyTorch install.")
        return torch.device("cuda")
    raise ValueError(f"unsupported device: {requested}")


def _make_loader(dataset, *, batch_size: int, shuffle: bool, num_workers: int):
    torch = _require_torch()
    if len(dataset) == 0:
        raise RuntimeError("dataset has zero windows; check clip_length, stride, and HDF5 paths")
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=torch_collate_ais,
        drop_last=False,
    )


def _dataset_specs_from_config(config: dict[str, Any]) -> dict[str, list[WindowDatasetSpec]]:
    data_cfg = config["data"]
    eval_cfg = config.get("eval", {})
    clip_length = int(data_cfg.get("clip_length", 224))

    def from_items(role: str, items: list[dict[str, Any]]) -> list[WindowDatasetSpec]:
        specs = []
        for idx, item in enumerate(items):
            name = str(item.get("name") or ("val" if role == "selection" and idx == 0 else f"{role}_{idx + 1}"))
            specs.append(
                WindowDatasetSpec(
                    name=name,
                    role=role,
                    controller_h5=str(item["controller_h5"]),
                    block_keyframes_h5=str(item["block_keyframes_h5"]),
                    stride=int(item.get("stride", data_cfg.get("val_stride" if role != "train" else "stride", clip_length))),
                    max_clips=item.get("max_clips"),
                    max_windows=item.get("max_windows"),
                )
            )
        return specs

    if data_cfg.get("train_sets"):
        train_specs = from_items("train", list(data_cfg["train_sets"]))
    else:
        train_specs = [
            WindowDatasetSpec(
                name="train",
                role="train",
                controller_h5=str(data_cfg["train_controller_h5"]),
                block_keyframes_h5=str(data_cfg["train_block_keyframes_h5"]),
                stride=int(data_cfg.get("stride", clip_length)),
                max_clips=data_cfg.get("max_train_clips"),
                max_windows=data_cfg.get("max_train_windows"),
            )
        ]

    if data_cfg.get("selection_sets"):
        selection_specs = from_items("selection", list(data_cfg["selection_sets"]))
    else:
        selection_specs = [
            WindowDatasetSpec(
                name="val",
                role="selection",
                controller_h5=str(data_cfg["val_controller_h5"]),
                block_keyframes_h5=str(data_cfg["val_block_keyframes_h5"]),
                stride=int(data_cfg.get("val_stride", clip_length)),
                max_clips=data_cfg.get("max_val_clips"),
                max_windows=data_cfg.get("max_val_windows"),
            )
        ]

    holdout_specs = from_items("holdout", list(data_cfg.get("holdout_sets", []) or []))
    benchmark_specs = from_items("benchmark", list(eval_cfg.get("sets", []) or []))
    return {"train": train_specs, "selection": selection_specs, "holdout": holdout_specs, "benchmark": benchmark_specs}


def _preprocessing_config(config: dict[str, Any]) -> dict[str, Any]:
    preprocessing = config.get("preprocessing", {})
    return {
        "relative_root_translation": bool(preprocessing.get("relative_root_translation", True)),
        "range_filter_threshold": preprocessing.get("range_filter_threshold", 100.0),
    }


def _make_window_dataset(
    spec: WindowDatasetSpec,
    *,
    clip_length: int,
    relative_root_translation: bool = True,
    range_filter_threshold: float | None = 100.0,
):
    return AnimajWindowDataset(
        spec.controller_h5,
        spec.block_keyframes_h5,
        clip_length=clip_length,
        stride=spec.stride,
        max_clips=spec.max_clips,
        max_windows=spec.max_windows,
        relative_root_translation=relative_root_translation,
        range_filter_threshold=range_filter_threshold,
    )


def _build_train_dataset(config: dict[str, Any], specs: list[WindowDatasetSpec]):
    torch = _require_torch()
    clip_length = int(config["data"].get("clip_length", 224))
    preprocessing = _preprocessing_config(config)
    datasets = [_make_window_dataset(spec, clip_length=clip_length, **preprocessing) for spec in specs]
    if len(datasets) == 1:
        return datasets[0]
    return torch.utils.data.ConcatDataset(datasets)


def _build_eval_loaders(
    config: dict[str, Any],
    specs_by_role: dict[str, list[WindowDatasetSpec]],
    *,
    batch_size: int,
    num_workers: int,
) -> tuple[dict[str, Any], dict[str, str]]:
    data_cfg = config["data"]
    clip_length = int(data_cfg.get("clip_length", 224))
    preprocessing = _preprocessing_config(config)

    loaders = {}
    roles = {}
    for role in ("selection", "holdout", "benchmark"):
        for spec in specs_by_role.get(role, []):
            dataset = _make_window_dataset(spec, clip_length=clip_length, **preprocessing)
            loaders[spec.name] = _make_loader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            roles[spec.name] = spec.role
    return loaders, roles


def _spectral_l1(pred, target):
    torch = _require_torch()
    pred_power = torch.log1p(torch.fft.rfft(pred, dim=1).abs() ** 2)
    target_power = torch.log1p(torch.fft.rfft(target, dim=1).abs() ** 2)
    return torch.mean(torch.abs(pred_power - target_power))


def _gate_tv(out):
    loss = None
    for key in ("alpha", "beta"):
        value = out.get(key)
        if value is None or value.shape[1] < 2:
            continue
        term = (value[:, 1:] - value[:, :-1]).abs().mean()
        loss = term if loss is None else loss + term
    if loss is None:
        pred = out["pred"]
        return pred.new_tensor(0.0)
    return loss


def _batch_loss(model, criterion, batch, loss_cfg: dict[str, Any], *, hard_copy: bool):
    from improved_ais.losses import acceleration_l1, velocity_l1

    target = batch["target"]
    model_kwargs = {
        "observed_mask": batch["observed_mask"] if hard_copy else None,
        "keypose_values": target if hard_copy else None,
    }
    if bool(getattr(model, "uses_temporal_features", False)):
        model_kwargs["temporal_features"] = batch.get("temporal_features")
    out = model(batch["input_seq"], batch["prev_pose"], batch["next_pose"], **model_kwargs)
    pred = out["pred"]
    loss = criterion(pred, target)
    components = {"weighted_l1": float(loss.detach().cpu().item())}

    weights = {
        "velocity": float(loss_cfg.get("velocity_weight", 0.0)),
        "acceleration": float(loss_cfg.get("acceleration_weight", 0.0)),
        "spectral": float(loss_cfg.get("spectral_weight", 0.0)),
        "gate_tv": float(loss_cfg.get("gate_tv_weight", 0.0)),
        "keypose": float(loss_cfg.get("keypose_weight", 0.0)),
    }
    if weights["velocity"]:
        term = velocity_l1(pred, target)
        components["velocity"] = float(term.detach().cpu().item())
        loss = loss + weights["velocity"] * term
    if weights["acceleration"]:
        term = acceleration_l1(pred, target)
        components["acceleration"] = float(term.detach().cpu().item())
        loss = loss + weights["acceleration"] * term
    if weights["spectral"]:
        term = _spectral_l1(pred, target)
        components["spectral"] = float(term.detach().cpu().item())
        loss = loss + weights["spectral"] * term
    if weights["gate_tv"]:
        term = _gate_tv(out)
        components["gate_tv"] = float(term.detach().cpu().item())
        loss = loss + weights["gate_tv"] * term
    if weights["keypose"]:
        observed = batch["observed_mask"].bool()
        term = (pred[observed] - target[observed]).abs().mean()
        components["keypose"] = float(term.detach().cpu().item())
        loss = loss + weights["keypose"] * term
    components["total"] = float(loss.detach().cpu().item())
    return loss, out, components


def _move_batch(batch: dict[str, Any], device):
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}


def _write_metrics_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = _read_csv_rows(path)
    fieldnames = list(row.keys())
    for existing_row in existing:
        for key in existing_row:
            if key not in fieldnames:
                fieldnames.append(key)
    for key in row:
        if key not in fieldnames:
            fieldnames.append(key)
    if existing:
        normalized = [{key: existing_row.get(key, "") for key in fieldnames} for existing_row in existing]
        normalized.append({key: row.get(key, "") for key in fieldnames})
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(normalized)
        return

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_progress(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp_path.replace(path)


def _metric_is_finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _initial_best_metric(mode: str) -> float:
    if mode == "max":
        return float("-inf")
    if mode == "min":
        return float("inf")
    raise ValueError(f"unsupported metric mode: {mode}")


def _metric_is_better(value: float, best_value: float, *, mode: str, min_delta: float = 0.0) -> bool:
    if mode == "min":
        return value < best_value - min_delta
    if mode == "max":
        return value > best_value + min_delta
    raise ValueError(f"unsupported metric mode: {mode}")


class EarlyStopping:
    """Patience-based early stopping over periodic validation metrics."""

    def __init__(
        self,
        *,
        enabled: bool,
        monitor: str,
        mode: str,
        patience: int,
        min_delta: float,
        min_steps: int,
        min_validation_checks: int,
        monitor_role: str = "selection",
        warning: str = "",
        best_value: float | None = None,
        best_step: int | None = None,
        validation_checks: int = 0,
        no_improvement_count: int = 0,
    ) -> None:
        self.enabled = bool(enabled)
        self.monitor = monitor
        self.mode = mode
        self.patience = max(0, int(patience))
        self.min_delta = max(0.0, float(min_delta))
        self.min_steps = max(0, int(min_steps))
        self.min_validation_checks = max(1, int(min_validation_checks))
        self.monitor_role = monitor_role
        self.warning = warning
        self.best_value = best_value if best_value is not None else _initial_best_metric(mode)
        self.best_step = best_step
        self.validation_checks = max(0, int(validation_checks))
        self.no_improvement_count = max(0, int(no_improvement_count))
        self.should_stop = False
        self.stop_reason = ""

    @classmethod
    def from_config(
        cls,
        training_cfg: dict[str, Any],
        *,
        default_monitor: str,
        default_mode: str,
        metric_roles: dict[str, str] | None = None,
        best_value: float | None = None,
        state: dict[str, Any] | None = None,
    ) -> "EarlyStopping":
        cfg = training_cfg.get("early_stopping", {}) or {}
        state = state or {}
        monitor = str(cfg.get("monitor", state.get("monitor", default_monitor)))
        mode = str(cfg.get("mode", state.get("mode", default_mode))).lower()
        metric_roles = metric_roles or {}
        monitor_role = metric_roles.get(monitor, str(state.get("monitor_role", "selection")))
        warning = ""
        if monitor_role == "holdout":
            warning = f"early stopping monitors holdout metric {monitor!r}; final holdout results are selection-tuned"
        return cls(
            enabled=bool(cfg.get("enabled", False)),
            monitor=monitor,
            mode=mode,
            patience=int(cfg.get("patience", state.get("patience", 5))),
            min_delta=float(cfg.get("min_delta", state.get("min_delta", 0.0))),
            min_steps=int(cfg.get("min_steps", state.get("min_steps", 0))),
            min_validation_checks=int(cfg.get("min_validation_checks", state.get("min_validation_checks", 1))),
            monitor_role=monitor_role,
            warning=warning or str(state.get("warning", "")),
            best_value=state.get("best_value", best_value),
            best_step=state.get("best_step"),
            validation_checks=int(state.get("validation_checks", 0)),
            no_improvement_count=int(state.get("no_improvement_count", 0)),
        )

    def update(self, *, step: int, metrics: dict[str, float]) -> dict[str, Any]:
        value = metrics.get(self.monitor)
        if not self.enabled:
            return self.state()
        if not _metric_is_finite(value):
            self.stop_reason = f"monitored metric {self.monitor!r} is missing or non-finite"
            return self.state()

        value = float(value)
        self.validation_checks += 1
        improved = _metric_is_better(value, float(self.best_value), mode=self.mode, min_delta=self.min_delta)
        if improved:
            self.best_value = value
            self.best_step = int(step)
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1

        can_stop = int(step) >= self.min_steps and self.validation_checks >= self.min_validation_checks
        self.should_stop = bool(can_stop and self.no_improvement_count >= self.patience)
        if self.should_stop:
            self.stop_reason = (
                f"{self.monitor} did not improve by at least {self.min_delta:g} "
                f"for {self.no_improvement_count} validation checks"
            )
        else:
            self.stop_reason = ""
        state = self.state()
        state["latest_value"] = value
        state["improved"] = improved
        return state

    def state(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "monitor": self.monitor,
            "mode": self.mode,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "min_steps": self.min_steps,
            "min_validation_checks": self.min_validation_checks,
            "monitor_role": self.monitor_role,
            "warning": self.warning,
            "validation_checks": self.validation_checks,
            "no_improvement_count": self.no_improvement_count,
            "best_value": float(self.best_value),
            "best_step": self.best_step,
            "should_stop": self.should_stop,
            "stop_reason": self.stop_reason,
        }


class TrainingProgress:
    def __init__(
        self,
        *,
        output_dir: Path,
        start_step: int,
        max_steps: int,
        enabled: bool = True,
        use_tqdm: bool = True,
        log_every_steps: int = 25,
    ) -> None:
        self.output_dir = output_dir
        self.progress_path = output_dir / "progress.json"
        self.start_step = int(start_step)
        self.max_steps = int(max_steps)
        self.enabled = bool(enabled)
        self.log_every_steps = max(1, int(log_every_steps))
        self.started_at = time.time()
        self.last_log_step = self.start_step
        self.bar = None

        if self.enabled and use_tqdm:
            try:
                from tqdm.auto import tqdm

                self.bar = tqdm(total=self.max_steps, initial=self.start_step, desc="AIS training", unit="step")
            except Exception:
                self.bar = None

    def update(
        self,
        *,
        step: int,
        train_loss: float | None = None,
        val_metrics: dict[str, float] | None = None,
        early_stopping: dict[str, Any] | None = None,
        phase: str = "train",
    ) -> None:
        elapsed = time.time() - self.started_at
        completed = max(0, int(step) - self.start_step)
        steps_per_sec = completed / elapsed if elapsed > 0 and completed > 0 else 0.0
        remaining = max(0, self.max_steps - int(step))
        eta_seconds = remaining / steps_per_sec if steps_per_sec > 0 else None
        payload = {
            "phase": phase,
            "step": int(step),
            "max_steps": self.max_steps,
            "percent": round(100.0 * int(step) / max(1, self.max_steps), 3),
            "elapsed_seconds": round(elapsed, 3),
            "eta_seconds": None if eta_seconds is None else round(eta_seconds, 3),
            "steps_per_second": round(steps_per_sec, 6),
            "train_loss": train_loss,
            "val_metrics": val_metrics or {},
            "early_stopping": early_stopping or {},
        }
        _write_progress(self.progress_path, payload)

        if self.bar is not None:
            delta = int(step) - int(self.bar.n)
            if delta > 0:
                self.bar.update(delta)
            postfix = {}
            if train_loss is not None:
                postfix["loss"] = f"{train_loss:.4f}"
            if val_metrics:
                postfix["val"] = f"{val_metrics.get('val_loss', float('nan')):.4f}"
            if postfix:
                self.bar.set_postfix(postfix)
            return

        if self.enabled and (int(step) - self.last_log_step >= self.log_every_steps or int(step) >= self.max_steps):
            eta_text = "unknown" if eta_seconds is None else f"{eta_seconds / 60.0:.1f}m"
            loss_text = "" if train_loss is None else f" train_loss={train_loss:.6f}"
            val_text = "" if not val_metrics else f" val_loss={val_metrics.get('val_loss', float('nan')):.6f}"
            print(
                f"[ais-train] step={step}/{self.max_steps} "
                f"({payload['percent']:.2f}%) speed={steps_per_sec:.3f} step/s eta={eta_text}"
                f"{loss_text}{val_text}",
                flush=True,
            )
            self.last_log_step = int(step)

    def close(self) -> None:
        if self.bar is not None:
            self.bar.close()


def _format_float(value: Any) -> str:
    try:
        if value in ("", None):
            return ""
        if isinstance(value, bool):
            return str(value).lower()
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def _svg_line_chart(rows: list[dict[str, str]], keys: list[str], *, width: int = 900, height: int = 240) -> str:
    points_by_key = {}
    all_values = []
    all_steps = []
    present_keys = []
    for key in keys:
        points = []
        for row in rows:
            try:
                if row.get(key, "") == "":
                    continue
                step = float(row["step"])
                value = float(row[key])
            except (KeyError, TypeError, ValueError):
                continue
            points.append((step, value))
            all_steps.append(step)
            all_values.append(value)
        points_by_key[key] = points
        if points:
            present_keys.append(key)
    if not all_values:
        return "<p>No metric points yet.</p>"

    if len(all_steps) < 2:
        # Avoid collapse if training is still bootstrapping.
        all_steps = [all_steps[0], all_steps[0] + 1.0]

    pad_left = 56
    pad_right = 20
    pad_top = 24
    pad_bottom = 44
    min_step, max_step = min(all_steps), max(all_steps)
    min_val, max_val = min(all_values), max(all_values)
    if max_step == min_step:
        max_step += 1.0
    if max_val == min_val:
        max_val += 1.0
    value_range = max_val - min_val
    pad_value = 0.08 * value_range if value_range > 0 else 1.0
    min_val -= pad_value
    max_val += pad_value
    value_range = max_val - min_val

    plot_left = pad_left
    plot_right = width - pad_right
    plot_top = pad_top
    plot_bottom = height - pad_bottom
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top

    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2"]

    def tick_positions(start: float, stop: float, count: int) -> list[float]:
        if count <= 1:
            return [start]
        step = (stop - start) / (count - 1)
        return [start + i * step for i in range(count)]

    def nice_str(value: float) -> str:
        abs_v = abs(value)
        if abs_v >= 1:
            return f"{value:.4g}"
        if abs_v >= 0.01:
            return f"{value:.4f}"
        return f"{value:.3e}"

    def format_step(value: float) -> str:
        if float(value).is_integer():
            return f"{int(value)}"
        return f"{value:.1f}"

    def map_point(point):
        step, value = point
        x = plot_left + (step - min_step) / (max_step - min_step) * plot_width
        y = plot_bottom - (value - min_val) / (max_val - min_val) * plot_height
        return x, y

    x_ticks = tick_positions(min_step, max_step, 7)
    y_ticks = tick_positions(min_val, max_val, 6)

    lines = [
        f'<div class="chart">',
        f'<svg viewBox="0 0 {width} {height}" role="img" preserveAspectRatio="none">',
        # Plot area
        f'<rect x="{plot_left}" y="{plot_top}" width="{plot_width}" height="{plot_height}" fill="#f9fafb" rx="4"/>',

        # Axes
        f'<line x1="{plot_left}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_bottom}" stroke="#374151" stroke-width="1.5"/>',
        f'<line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_bottom}" stroke="#374151" stroke-width="1.5"/>',
        f'<text x="{plot_left - 46}" y="{plot_top + 12}" font-size="11" fill="#4b5563">Value</text>',
        f'<text x="{(plot_left + plot_right) / 2 - 20}" y="{height - 10}" font-size="11" fill="#4b5563">Step</text>',
    ]

    # Grid and tick labels
    for tick in x_ticks:
        x = plot_left + (tick - min_step) / (max_step - min_step) * plot_width
        lines.append(f'<line x1="{x:.2f}" y1="{plot_top}" x2="{x:.2f}" y2="{plot_bottom}" stroke="#e5e7eb" stroke-width="1"/>')
        lines.append(f'<text x="{x:.2f}" y="{plot_bottom + 20}" text-anchor="middle" font-size="10" fill="#6b7280">{format_step(tick)}</text>')
    for tick in y_ticks:
        y = plot_bottom - (tick - min_val) / (max_val - min_val) * plot_height
        lines.append(f'<line x1="{plot_left}" y1="{y:.2f}" x2="{plot_right}" y2="{y:.2f}" stroke="#e5e7eb" stroke-width="1"/>')
        lines.append(f'<text x="{plot_left - 8}" y="{y + 3:.2f}" text-anchor="end" font-size="10" fill="#6b7280">{nice_str(tick)}</text>')

    for idx, key in enumerate(present_keys):
        mapped = [map_point(point) for point in points_by_key.get(key, [])]
        if not mapped:
            continue
        color = colors[idx % len(colors)]
        if len(mapped) == 1:
            x, y = mapped[0]
            lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="{color}"/>')
        else:
            points = " ".join(f"{x:.2f},{y:.2f}" for x, y in mapped)
            lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
        legend_x = plot_left + (idx * 140)
        legend_y = plot_top - 8
        lines.append(f'<circle cx="{legend_x + 6}" cy="{legend_y:.2f}" r="4" fill="{color}"/>')
        lines.append(f'<text x="{legend_x + 14}" y="{legend_y + 4:.2f}" font-size="11" fill="#374151">{key}</text>')
    lines.append("</svg>")
    lines.append("</div>")
    return "\n".join(lines)


def _render_dashboard(output_dir: Path, *, title: str = "AIS Training Dashboard") -> None:
    metrics_rows = _read_csv_rows(output_dir / "metrics.csv")
    latest = metrics_rows[-1] if metrics_rows else {}
    progress = {}
    manifest = {}
    progress_path = output_dir / "progress.json"
    manifest_path = output_dir / "checkpoints" / "manifest.json"
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    metric_keys = sorted({key for row in metrics_rows for key in row if key.endswith(("_loss", "_l1", "_drift"))})
    preferred = ["train_loss", "val_loss", "val_l1", "val_keypose_drift"]
    metric_keys = [key for key in preferred if key in metric_keys] + [key for key in metric_keys if key not in preferred]
    loss_keys = [key for key in ("train_loss", "val_loss") if key in metric_keys]
    drift_keys = [key for key in ("train_keypose_drift", "val_keypose_drift") if key in metric_keys]
    latest_rows = "".join(
        f"<tr><th>{key}</th><td>{_format_float(latest.get(key, ''))}</td></tr>"
        for key in ["step", "phase", *metric_keys]
    )
    checkpoint_items = "".join(f"<li><code>{path}</code></li>" for path in manifest.get("kept_checkpoints", []))
    checkpoint_items = checkpoint_items or "<li>No retained periodic checkpoints yet.</li>"
    percent = float(progress.get("percent", 0.0) or 0.0)
    early_stopping = manifest.get("early_stopping") or progress.get("early_stopping") or {}
    early_stopping_rows = ""
    if early_stopping:
        early_stopping_rows = "".join(
            f"<tr><th>{key}</th><td>{_format_float(early_stopping.get(key, ''))}</td></tr>"
            for key in [
                "enabled",
                "monitor",
                "mode",
                "monitor_role",
                "patience",
                "min_delta",
                "validation_checks",
                "no_improvement_count",
                "best_value",
                "best_step",
                "should_stop",
                "warning",
                "stop_reason",
            ]
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="15">
  <title>{title}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #111827; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }}
    .card {{ border: 1px solid #d1d5db; border-radius: 8px; padding: 16px; background: #fff; }}
    .chart {{ border: 1px solid #e5e7eb; border-radius: 8px; padding: 8px; background: #fff; overflow-x: auto; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; padding: 6px 8px; text-align: left; font-size: 14px; }}
    th {{ color: #4b5563; }}
    code {{ font-size: 12px; overflow-wrap: anywhere; }}
    .bar {{ width: 100%; height: 12px; background: #e5e7eb; border-radius: 999px; overflow: hidden; }}
    .fill {{ height: 100%; width: {percent:.2f}%; background: #2563eb; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <p>Auto-refreshes every 15 seconds. Run folder: <code>{output_dir}</code></p>
  <div class="grid">
    <section class="card">
      <h2>Progress</h2>
      <div class="bar"><div class="fill"></div></div>
      <table>
        <tr><th>Step</th><td>{progress.get("step", "")} / {progress.get("max_steps", "")}</td></tr>
        <tr><th>Percent</th><td>{progress.get("percent", "")}%</td></tr>
        <tr><th>Elapsed</th><td>{_format_float(progress.get("elapsed_seconds", ""))} s</td></tr>
        <tr><th>ETA</th><td>{_format_float(progress.get("eta_seconds", ""))} s</td></tr>
        <tr><th>Speed</th><td>{_format_float(progress.get("steps_per_second", ""))} step/s</td></tr>
      </table>
    </section>
    <section class="card">
      <h2>Latest Metrics</h2>
      <table>{latest_rows or '<tr><td>No metrics yet.</td></tr>'}</table>
    </section>
    <section class="card">
      <h2>Checkpoints</h2>
      <table>
        <tr><th>Best metric</th><td>{manifest.get("best_metric_name", "")}</td></tr>
        <tr><th>Best value</th><td>{_format_float(manifest.get("best_metric_value", ""))}</td></tr>
        <tr><th>Best path</th><td><code>{manifest.get("best_checkpoint", "")}</code></td></tr>
        <tr><th>Last path</th><td><code>{manifest.get("last_checkpoint", "")}</code></td></tr>
      </table>
      <h3>Retained periodic checkpoints</h3>
      <ul>{checkpoint_items}</ul>
    </section>
    <section class="card">
      <h2>Early Stopping</h2>
      <table>{early_stopping_rows or '<tr><td>Disabled.</td></tr>'}</table>
    </section>
  </div>
  <section class="card" style="margin-top:16px;">
    <h2>Overfitting Curves</h2>
    <h3>Loss</h3>
    {_svg_line_chart(metrics_rows, loss_keys or ["train_loss", "val_loss"])}
    <h3>Keypose Drift</h3>
    {_svg_line_chart(metrics_rows, drift_keys or ["train_keypose_drift", "val_keypose_drift"])}
  </section>
</body>
</html>
"""
    (output_dir / "dashboard.html").write_text(html, encoding="utf-8")


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    torch = _require_torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def _checkpoint_payload(
    *,
    model,
    criterion,
    optimizer,
    step: int,
    config: dict[str, Any],
    best_metric_name: str,
    best_metric_value: float,
    early_stopping_state: dict[str, Any] | None = None,
    dataset_roles: dict[str, str] | None = None,
) -> dict[str, Any]:
    torch = _require_torch()
    rng = {"torch": torch.get_rng_state(), "numpy": np.random.get_state()}
    if hasattr(torch, "mps") and hasattr(torch.mps, "get_rng_state"):
        try:
            rng["mps"] = torch.mps.get_rng_state()
        except Exception:
            pass
    return {
        "model": model.state_dict(),
        "criterion": criterion.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "config": config,
        "best_metric_name": best_metric_name,
        "best_metric_value": best_metric_value,
        "early_stopping": early_stopping_state or {},
        "dataset_roles": dataset_roles or {},
        "rng": rng,
    }


def _save_checkpoint(
    path: Path,
    *,
    model,
    criterion,
    optimizer,
    step: int,
    config: dict[str, Any],
    best_metric_name: str = "val_loss",
    best_metric_value: float = float("inf"),
    early_stopping_state: dict[str, Any] | None = None,
    dataset_roles: dict[str, str] | None = None,
) -> None:
    _atomic_torch_save(
        _checkpoint_payload(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            step=step,
            config=config,
            best_metric_name=best_metric_name,
            best_metric_value=best_metric_value,
            early_stopping_state=early_stopping_state,
            dataset_roles=dataset_roles,
        ),
        path,
    )


def _update_checkpoint_manifest(
    output_dir: Path,
    *,
    best_path: Path,
    last_path: Path,
    best_metric_name: str,
    best_metric_mode: str,
    best_metric_value: float,
    kept_checkpoints: list[Path],
    early_stopping_state: dict[str, Any] | None = None,
    dataset_roles: dict[str, str] | None = None,
) -> None:
    _write_progress(
        output_dir / "checkpoints" / "manifest.json",
        {
            "best_metric_name": best_metric_name,
            "best_metric_mode": best_metric_mode,
            "best_metric_value": best_metric_value,
            "best_checkpoint": str(best_path),
            "last_checkpoint": str(last_path),
            "kept_checkpoints": [str(path) for path in kept_checkpoints],
            "early_stopping": early_stopping_state or {},
            "dataset_roles": dataset_roles or {},
        },
    )


def _save_training_checkpoints(
    *,
    output_dir: Path,
    model,
    criterion,
    optimizer,
    step: int,
    config: dict[str, Any],
    metric_value: float | None,
    best_metric_name: str,
    best_metric_mode: str,
    best_metric_value: float,
    keep_every_n: int,
    save_periodic: bool,
    early_stopping_state: dict[str, Any] | None = None,
    dataset_roles: dict[str, str] | None = None,
    is_final: bool = False,
) -> float:
    checkpoint_dir = output_dir / "checkpoints"
    best_path = checkpoint_dir / "best.pt"
    last_path = checkpoint_dir / "last.pt"
    if metric_value is not None and _metric_is_finite(metric_value):
        candidate = float(metric_value)
    else:
        candidate = None
    save_best = candidate is not None and _metric_is_better(candidate, best_metric_value, mode=best_metric_mode)
    if save_best:
        best_metric_value = float(candidate)
    _save_checkpoint(
        last_path,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        step=step,
        config=config,
        best_metric_name=best_metric_name,
        best_metric_value=best_metric_value,
        early_stopping_state=early_stopping_state,
        dataset_roles=dataset_roles,
    )
    if save_best:
        _save_checkpoint(
            best_path,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            step=step,
            config=config,
            best_metric_name=best_metric_name,
            best_metric_value=best_metric_value,
            early_stopping_state=early_stopping_state,
            dataset_roles=dataset_roles,
        )
    if save_periodic or is_final:
        _save_checkpoint(
            checkpoint_dir / f"step_{step:08d}.pt",
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            step=step,
            config=config,
            best_metric_name=best_metric_name,
            best_metric_value=best_metric_value,
            early_stopping_state=early_stopping_state,
            dataset_roles=dataset_roles,
        )
    periodic = sorted(checkpoint_dir.glob("step_*.pt"))
    if keep_every_n >= 0:
        stale = periodic[:-keep_every_n] if keep_every_n else periodic
        for path in stale:
            path.unlink(missing_ok=True)
    _update_checkpoint_manifest(
        output_dir,
        best_path=best_path,
        last_path=last_path,
        best_metric_name=best_metric_name,
        best_metric_mode=best_metric_mode,
        best_metric_value=best_metric_value,
        kept_checkpoints=sorted(checkpoint_dir.glob("step_*.pt")),
        early_stopping_state=early_stopping_state,
        dataset_roles=dataset_roles,
    )
    return best_metric_value


def _restore_rng(state: dict[str, Any]) -> None:
    torch = _require_torch()
    rng = state.get("rng", {})
    if "torch" in rng:
        torch.set_rng_state(rng["torch"])
    if "numpy" in rng:
        np.random.set_state(rng["numpy"])
    if "mps" in rng and hasattr(torch, "mps") and hasattr(torch.mps, "set_rng_state"):
        try:
            torch.mps.set_rng_state(rng["mps"])
        except Exception:
            pass


def _load_resume(path: Path, *, model, criterion, optimizer, device) -> tuple[int, float, dict[str, Any]]:
    torch = _require_torch()
    try:
        state = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(path, map_location=device)
    model.load_state_dict(state["model"])
    criterion.load_state_dict(state["criterion"])
    optimizer.load_state_dict(state["optimizer"])
    _restore_rng(state)
    return int(state.get("step", 0)), float(state.get("best_metric_value", float("inf"))), dict(state.get("early_stopping", {}))


def _evaluate(
    model,
    criterion,
    loader,
    loss_cfg: dict[str, Any],
    device,
    *,
    hard_copy: bool,
    prefix: str = "val",
) -> dict[str, float]:
    from improved_ais.metrics import keypose_drift, l1, npss, stl1

    torch = _require_torch()
    model.eval()
    losses = []
    l1_values = []
    keypose_values = []
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)
            loss, out, _ = _batch_loss(model, criterion, batch, loss_cfg, hard_copy=hard_copy)
            pred = out["pred"].detach().cpu().numpy()
            target = batch["target"].detach().cpu().numpy()
            observed = batch["observed_mask"].detach().cpu().numpy()
            losses.append(float(loss.detach().cpu().item()))
            for i in range(pred.shape[0]):
                keys = np.flatnonzero(observed[i] > 0.5)
                l1_values.append(l1(pred[i], target[i]))
                keypose_values.append(keypose_drift(pred[i], target[i, keys], keys))
    return {
        f"{prefix}_loss": float(np.mean(losses)) if losses else float("nan"),
        f"{prefix}_l1": float(np.mean(l1_values)) if l1_values else float("nan"),
        f"{prefix}_keypose_drift": float(np.mean(keypose_values)) if keypose_values else float("nan"),
    }


def _generate_validation_report(config: dict[str, Any], output_dir: Path, checkpoint_path: Path, device: str) -> None:
    del config, output_dir, checkpoint_path, device
    return


def _evaluate_all(model, criterion, loaders: dict[str, Any], loss_cfg: dict[str, Any], device, *, hard_copy: bool) -> dict[str, float]:
    metrics = {}
    for name, loader in loaders.items():
        metrics.update(_evaluate(model, criterion, loader, loss_cfg, device, hard_copy=hard_copy, prefix=name))
    return metrics


def _build_train_model(config: dict[str, Any], *, device: str):
    model_cfg = config.get("model", {})
    model_type = str(model_cfg.get("type", "improved_ais_bilstm"))
    if model_type == "improved_ais_bilstm":
        from improved_ais.models.improved_ais import build_improved_ais_bilstm

        return build_improved_ais_bilstm(model_cfg, device=device)
    raise ValueError(f"unsupported model.type: {model_type}")


def run_training(config: dict[str, Any], *, resume: Path | None = None) -> dict[str, Any]:
    torch = _require_torch()
    from improved_ais.losses import WeightedL1

    training_cfg = config["training"]
    model_cfg = config.get("model", {})
    model_type = str(model_cfg.get("type", "improved_ais_bilstm"))
    loss_cfg = config.get("loss", {})
    device = resolve_device(str(training_cfg.get("device", "auto")))
    seed = int(config.get("seed", 1234))
    torch.manual_seed(seed)
    np.random.seed(seed)

    output_dir = Path(training_cfg.get("output_dir", "runs/ais_repro_local"))
    output_dir.mkdir(parents=True, exist_ok=True)
    config_snapshot = output_dir / "config.yaml"
    with config_snapshot.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    specs_by_role = _dataset_specs_from_config(config)
    dataset_roles = {
        spec.name: spec.role
        for specs in specs_by_role.values()
        for spec in specs
    }
    train_dataset = _build_train_dataset(config, specs_by_role["train"])
    train_loader = _make_loader(
        train_dataset,
        batch_size=int(training_cfg.get("batch_size", 8)),
        shuffle=True,
        num_workers=int(training_cfg.get("num_workers", 0)),
    )
    eval_loaders, eval_roles = _build_eval_loaders(
        config,
        specs_by_role,
        batch_size=int(training_cfg.get("eval_batch_size", training_cfg.get("batch_size", 8))),
        num_workers=int(training_cfg.get("num_workers", 0)),
    )

    model = _build_train_model(config, device=str(device))
    criterion = WeightedL1(596).to(device)
    init_checkpoint = training_cfg.get("init_checkpoint")
    if init_checkpoint:
        raise RuntimeError("--init-checkpoint is not supported in this standalone improved-architecture trainer.")

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=float(training_cfg.get("lr", config.get("optimizer", {}).get("lr", 1e-4))),
        weight_decay=float(training_cfg.get("weight_decay", config.get("optimizer", {}).get("weight_decay", 1e-3))),
    )
    best_path = output_dir / "checkpoints" / "best.pt"
    last_path = output_dir / "checkpoints" / "last.pt"
    metrics_path = output_dir / "metrics.csv"
    max_steps = int(training_cfg.get("max_steps", 1000))
    grad_accum = int(training_cfg.get("gradient_accumulation_steps", 1))
    val_every = int(training_cfg.get("val_every_steps", 100))
    save_every = int(training_cfg.get("save_every_steps", val_every))
    progress_every = int(training_cfg.get("progress_every_steps", 25))
    progress_enabled = bool(training_cfg.get("progress", True))
    use_tqdm = bool(training_cfg.get("progress_bar", True))
    hard_copy_train = bool(training_cfg.get("hard_copy_keyposes_in_forward", False))
    hard_copy_eval = True
    grad_clip = float(training_cfg.get("grad_clip_norm", 0.0))
    train_log_every = int(training_cfg.get("train_log_every_steps", progress_every))
    dashboard_enabled = bool(training_cfg.get("dashboard", True))
    checkpoint_cfg = training_cfg.get("checkpoint", {})
    best_metric_name = str(checkpoint_cfg.get("best_metric", "val_loss"))
    best_metric_mode = str(checkpoint_cfg.get("mode", checkpoint_cfg.get("best_metric_mode", "min"))).lower()
    keep_every_n = int(checkpoint_cfg.get("keep_every_n", 3))
    save_periodic = bool(checkpoint_cfg.get("save_periodic", True))
    start_step = 0
    best_val = _initial_best_metric(best_metric_mode)
    early_stopping_resume: dict[str, Any] = {}
    if resume is not None:
        start_step, best_val, early_stopping_resume = _load_resume(
            resume, model=model, criterion=criterion, optimizer=optimizer, device=device
        )
    metric_roles = {
        f"{dataset_name}_{metric}": role
        for dataset_name, role in eval_roles.items()
        for metric in ("loss", "l1", "keypose_drift")
    }
    early_stopping = EarlyStopping.from_config(
        training_cfg,
        default_monitor=best_metric_name,
        default_mode=best_metric_mode,
        metric_roles=metric_roles,
        best_value=best_val,
        state=early_stopping_resume,
    )
    early_stopping_state = early_stopping.state()

    step = start_step
    start_time = time.time()
    progress = TrainingProgress(
        output_dir=output_dir,
        start_step=start_step,
        max_steps=max_steps,
        enabled=progress_enabled,
        use_tqdm=use_tqdm,
        log_every_steps=progress_every,
    )

    if bool(training_cfg.get("eval_only", False)):
        eval_metrics = _evaluate_all(model, criterion, eval_loaders, loss_cfg, device, hard_copy=hard_copy_eval)
        row = {"step": step, "phase": "eval_only", "seconds": 0.0, **eval_metrics}
        _write_metrics_row(metrics_path, row)
        progress.update(step=step, val_metrics=eval_metrics, early_stopping=early_stopping_state, phase="eval_only")
        progress.close()
        best_val = _save_training_checkpoints(
            output_dir=output_dir,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            step=step,
            config=config,
            metric_value=eval_metrics.get(best_metric_name),
            best_metric_name=best_metric_name,
            best_metric_mode=best_metric_mode,
            best_metric_value=best_val,
            keep_every_n=keep_every_n,
            save_periodic=False,
            early_stopping_state=early_stopping_state,
            dataset_roles=dataset_roles,
            is_final=True,
        )
        if dashboard_enabled:
            _render_dashboard(output_dir)
        _generate_validation_report(config, output_dir, last_path, str(device))
        return row

    try:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        stop_training = False
        while step < max_steps:
            for batch in train_loader:
                if step >= max_steps:
                    break
                batch = _move_batch(batch, device)
                loss, _, components = _batch_loss(model, criterion, batch, loss_cfg, hard_copy=hard_copy_train)
                (loss / grad_accum).backward()
                if (step + 1) % grad_accum == 0:
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(criterion.parameters()), grad_clip)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                step += 1

                train_loss = components["total"]
                if step % val_every == 0 or step == max_steps:
                    eval_metrics = _evaluate_all(model, criterion, eval_loaders, loss_cfg, device, hard_copy=hard_copy_eval)
                    early_stopping_state = early_stopping.update(step=step, metrics=eval_metrics)
                    stop_training = bool(early_stopping_state.get("should_stop", False))
                    phase = "early_stop" if stop_training else "train"
                    row = {
                        "step": step,
                        "phase": phase,
                        "seconds": round(time.time() - start_time, 3),
                        "train_loss": train_loss,
                        **eval_metrics,
                        "early_stop_no_improve": early_stopping_state.get("no_improvement_count", ""),
                        "early_stop_best": early_stopping_state.get("best_value", ""),
                        "early_stop_best_step": early_stopping_state.get("best_step", ""),
                        "early_stop_monitor_role": early_stopping_state.get("monitor_role", ""),
                        "early_stop_warning": early_stopping_state.get("warning", ""),
                    }
                    _write_metrics_row(metrics_path, row)
                    progress.update(
                        step=step,
                        train_loss=train_loss,
                        val_metrics=eval_metrics,
                        early_stopping=early_stopping_state,
                        phase=phase,
                    )
                    model.train()
                    best_val = _save_training_checkpoints(
                        output_dir=output_dir,
                        model=model,
                        criterion=criterion,
                        optimizer=optimizer,
                        step=step,
                        config=config,
                        metric_value=eval_metrics.get(best_metric_name),
                        best_metric_name=best_metric_name,
                        best_metric_mode=best_metric_mode,
                        best_metric_value=best_val,
                        keep_every_n=keep_every_n,
                        save_periodic=(step % save_every == 0),
                        early_stopping_state=early_stopping_state,
                        dataset_roles=dataset_roles,
                        is_final=(step == max_steps or stop_training),
                    )
                    if dashboard_enabled:
                        _render_dashboard(output_dir)
                    if stop_training:
                        break
                else:
                    progress.update(step=step, train_loss=train_loss, early_stopping=early_stopping_state)
                    if step % train_log_every == 0:
                        _write_metrics_row(
                            metrics_path,
                            {
                                "step": step,
                                "phase": "train_step",
                                "seconds": round(time.time() - start_time, 3),
                                "train_loss": train_loss,
                            },
                        )
                        if dashboard_enabled:
                            _render_dashboard(output_dir)
                if (step % save_every == 0 or step == max_steps) and step % val_every != 0:
                    best_val = _save_training_checkpoints(
                        output_dir=output_dir,
                        model=model,
                        criterion=criterion,
                        optimizer=optimizer,
                        step=step,
                        config=config,
                        metric_value=None,
                        best_metric_name=best_metric_name,
                        best_metric_mode=best_metric_mode,
                        best_metric_value=best_val,
                        keep_every_n=keep_every_n,
                        save_periodic=True,
                        early_stopping_state=early_stopping_state,
                        dataset_roles=dataset_roles,
                        is_final=(step == max_steps),
                    )
                    if dashboard_enabled:
                        _render_dashboard(output_dir)
                if stop_training:
                    break
            if stop_training:
                break
    finally:
        progress.close()

    if not best_path.exists() and last_path.exists():
        shutil.copyfile(last_path, best_path)
        _update_checkpoint_manifest(
            output_dir,
            best_path=best_path,
            last_path=last_path,
            best_metric_name=best_metric_name,
            best_metric_mode=best_metric_mode,
            best_metric_value=best_val,
            kept_checkpoints=sorted((output_dir / "checkpoints").glob("step_*.pt")),
            early_stopping_state=early_stopping_state,
            dataset_roles=dataset_roles,
        )
    if dashboard_enabled:
        _render_dashboard(output_dir)
    _generate_validation_report(config, output_dir, best_path if best_path.exists() else last_path, str(device))
    return {
        "step": step,
        "output_dir": str(output_dir),
        "checkpoint": str(best_path if best_path.exists() else last_path),
        "early_stopped": bool(early_stopping_state.get("should_stop", False)),
        "early_stop_reason": early_stopping_state.get("stop_reason", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or evaluate the official AIS-BiLSTM on Animaj HDF5 data.")
    parser.add_argument("--config", type=Path, default=Path("configs/train/ais_repro.yaml"))
    parser.add_argument("--max-train-clips", type=int, default=None)
    parser.add_argument("--max-val-clips", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()
    try:
        config = _merge_cli_config(_load_yaml(args.config), args)
        result = run_training(config, resume=args.resume)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    print(yaml.safe_dump(result, sort_keys=True).strip())


if __name__ == "__main__":
    main()
