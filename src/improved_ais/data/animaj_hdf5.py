from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np


@dataclass(frozen=True)
class AnimajClip:
    clip_id: str
    vectors: np.ndarray
    block_keyframes: np.ndarray
    animation_keyframes: np.ndarray
    episode_id: str | None = None
    scene_id: str | None = None

    @property
    def key_indices(self) -> np.ndarray:
        idx = np.flatnonzero(self.block_keyframes)
        if len(idx) == 0:
            idx = np.asarray([0, len(self.block_keyframes) - 1], dtype=np.int64)
        idx = np.asarray(sorted(set([0, len(self.block_keyframes) - 1, *map(int, idx)])), dtype=np.int64)
        return idx


def _read_scalar(group: h5py.Group, name: str) -> str | None:
    if name not in group:
        return None
    value = group[name][()]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def iter_animaj_hdf5(
    controller_h5: str | Path,
    block_keyframes_h5: str | Path,
    *,
    max_clips: int | None = None,
) -> Iterator[AnimajClip]:
    controller_h5 = Path(controller_h5)
    block_keyframes_h5 = Path(block_keyframes_h5)
    with h5py.File(controller_h5, "r") as controller_file, h5py.File(block_keyframes_h5, "r") as block_file:
        clip_ids = sorted(controller_file.keys(), key=lambda x: int(x) if x.isdigit() else x)
        if max_clips is not None:
            clip_ids = clip_ids[:max_clips]
        for clip_id in clip_ids:
            if clip_id not in block_file:
                continue
            cg = controller_file[clip_id]
            bg = block_file[clip_id]
            vectors = np.asarray(cg["vectors"], dtype=np.float32)
            animation_keyframes = np.asarray(cg.get("animation_keyframes", np.zeros(len(vectors))), dtype=bool)
            block_keyframes = np.asarray(bg["block_keyframes_vectors"], dtype=bool)
            if len(block_keyframes) != len(vectors):
                raise ValueError(f"clip {clip_id} has mismatched vector/keyframe lengths")
            yield AnimajClip(
                clip_id=clip_id,
                vectors=vectors,
                block_keyframes=block_keyframes,
                animation_keyframes=animation_keyframes,
                episode_id=_read_scalar(cg, "episode_id"),
                scene_id=_read_scalar(cg, "scene_id"),
            )
