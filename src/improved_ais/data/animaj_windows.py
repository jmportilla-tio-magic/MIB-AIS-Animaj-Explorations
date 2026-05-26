from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from improved_ais.data.preprocessing import (
    apply_relative_root_translation,
    dataset_root_from_controller_h5,
    invalid_scene_keys_from_ranges,
)
from improved_ais.data.window import make_training_sample


@dataclass(frozen=True)
class WindowSpec:
    clip_id: str
    start: int
    length: int


def _clip_sort_key(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def _window_starts(length: int, clip_length: int, stride: int) -> list[int]:
    if length < clip_length:
        return []
    starts = list(range(0, length - clip_length + 1, stride))
    final = length - clip_length
    if not starts or starts[-1] != final:
        starts.append(final)
    return starts


class AnimajWindowIndex:
    """Deterministic fixed-length window index for public Animaj HDF5 files."""

    def __init__(
        self,
        controller_h5: str | Path,
        block_keyframes_h5: str | Path,
        *,
        clip_length: int = 224,
        stride: int = 56,
        max_clips: int | None = None,
        max_windows: int | None = None,
        relative_root_translation: bool = True,
        range_filter_threshold: float | None = 100.0,
    ) -> None:
        self.controller_h5 = Path(controller_h5)
        self.block_keyframes_h5 = Path(block_keyframes_h5)
        self.clip_length = int(clip_length)
        self.stride = int(stride)
        self.relative_root_translation = bool(relative_root_translation)
        self.range_filter_threshold = range_filter_threshold
        if self.clip_length < 2:
            raise ValueError("clip_length must be at least 2")
        if self.stride < 1:
            raise ValueError("stride must be at least 1")
        if not self.controller_h5.exists():
            raise FileNotFoundError(f"controller HDF5 not found: {self.controller_h5}")
        if not self.block_keyframes_h5.exists():
            raise FileNotFoundError(f"block-keyframe HDF5 not found: {self.block_keyframes_h5}")

        self.windows: list[WindowSpec] = []
        dataset_root = dataset_root_from_controller_h5(self.controller_h5)
        invalid_scenes = invalid_scene_keys_from_ranges(
            dataset_root / "ranges.csv" if dataset_root is not None else None,
            threshold=self.range_filter_threshold,
        )

        with h5py.File(self.controller_h5, "r") as controller_file, h5py.File(self.block_keyframes_h5, "r") as block_file:
            clip_ids = sorted(controller_file.keys(), key=_clip_sort_key)
            if max_clips is not None:
                clip_ids = clip_ids[: int(max_clips)]
            for clip_id in clip_ids:
                if clip_id not in block_file or "vectors" not in controller_file[clip_id]:
                    continue
                episode_id = _read_h5_text(controller_file[clip_id], "episode_id")
                scene_id = _read_h5_text(controller_file[clip_id], "scene_id")
                if (episode_id, scene_id) in invalid_scenes:
                    continue
                length = int(controller_file[clip_id]["vectors"].shape[0])
                for start in _window_starts(length, self.clip_length, self.stride):
                    self.windows.append(WindowSpec(clip_id=clip_id, start=start, length=self.clip_length))
                    if max_windows is not None and len(self.windows) >= int(max_windows):
                        return

    def __len__(self) -> int:
        return len(self.windows)


class AnimajWindowDataset:
    """Torch-compatible dataset returning masked AIS training samples."""

    def __init__(
        self,
        controller_h5: str | Path,
        block_keyframes_h5: str | Path,
        *,
        clip_length: int = 224,
        stride: int = 56,
        max_clips: int | None = None,
        max_windows: int | None = None,
        relative_root_translation: bool = True,
        range_filter_threshold: float | None = 100.0,
    ) -> None:
        self.index = AnimajWindowIndex(
            controller_h5,
            block_keyframes_h5,
            clip_length=clip_length,
            stride=stride,
            max_clips=max_clips,
            max_windows=max_windows,
            relative_root_translation=relative_root_translation,
            range_filter_threshold=range_filter_threshold,
        )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int) -> dict[str, np.ndarray | str | int]:
        spec = self.index.windows[item]
        start = spec.start
        end = start + spec.length
        with h5py.File(self.index.controller_h5, "r") as controller_file, h5py.File(
            self.index.block_keyframes_h5, "r"
        ) as block_file:
            cg = controller_file[spec.clip_id]
            bg = block_file[spec.clip_id]
            vectors = np.asarray(cg["vectors"][start:end], dtype=np.float32)
            block_mask = np.asarray(bg["block_keyframes_vectors"][start:end], dtype=bool)
        if self.index.relative_root_translation:
            vectors = apply_relative_root_translation(vectors)

        key_indices = np.flatnonzero(block_mask)
        key_indices = np.asarray(sorted(set([0, spec.length - 1, *map(int, key_indices)])), dtype=np.int64)
        sample = make_training_sample(vectors, key_indices)
        return {
            "target": sample.target,
            "input_seq": sample.input_seq,
            "missing_mask": sample.missing_mask,
            "observed_mask": sample.observed_mask,
            "prev_pose": sample.prev_pose,
            "next_pose": sample.next_pose,
            "phase": sample.phase,
            "segment_len": sample.segment_len,
            "dist_prev": sample.dist_prev,
            "dist_next": sample.dist_next,
            "indices": sample.indices,
            "clip_id": spec.clip_id,
            "start": start,
        }


def torch_collate_ais(batch):
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required for AIS training batches.") from exc

    tensor_keys = [
        "target",
        "input_seq",
        "missing_mask",
        "observed_mask",
        "prev_pose",
        "next_pose",
        "phase",
        "segment_len",
        "dist_prev",
        "dist_next",
    ]
    out = {key: torch.from_numpy(np.stack([item[key] for item in batch], axis=0)).float() for key in tensor_keys}
    out["temporal_features"] = torch.cat(
        [out["phase"], out["segment_len"], out["dist_prev"], out["dist_next"]],
        dim=-1,
    )
    out["clip_id"] = [str(item["clip_id"]) for item in batch]
    out["start"] = torch.tensor([int(item["start"]) for item in batch], dtype=torch.long)
    return out


def _read_h5_text(group: h5py.Group, key: str) -> str:
    if key not in group:
        return ""
    value = group[key][()]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)
