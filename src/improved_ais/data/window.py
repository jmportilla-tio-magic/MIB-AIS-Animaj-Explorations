from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class TrainingSample:
    target: np.ndarray
    input_seq: np.ndarray
    missing_mask: np.ndarray
    observed_mask: np.ndarray
    prev_pose: np.ndarray
    next_pose: np.ndarray
    phase: np.ndarray
    segment_len: np.ndarray
    dist_prev: np.ndarray
    dist_next: np.ndarray
    indices: np.ndarray
    scene_id: str | None = None


def sanitize_key_indices(indices: Iterable[int], length: int) -> np.ndarray:
    if length < 2:
        raise ValueError("length must be at least 2")
    unique = {0, length - 1}
    unique.update(int(i) for i in indices if 0 <= int(i) < length)
    return np.asarray(sorted(unique), dtype=np.int64)


def surrounding_keyposes(target: np.ndarray, indices: Iterable[int]) -> tuple[np.ndarray, ...]:
    target = np.asarray(target, dtype=np.float32)
    length = target.shape[0]
    key_indices = sanitize_key_indices(indices, length)

    prev_idx = np.empty(length, dtype=np.int64)
    next_idx = np.empty(length, dtype=np.int64)
    cursor = 0
    for t in range(length):
        while cursor + 1 < len(key_indices) and key_indices[cursor + 1] <= t:
            cursor += 1
        prev_idx[t] = key_indices[cursor]
        next_idx[t] = key_indices[min(cursor + 1, len(key_indices) - 1)]

    prev_pose = target[prev_idx]
    next_pose = target[next_idx]
    gap = np.maximum(next_idx - prev_idx, 1).astype(np.float32)
    phase = ((np.arange(length) - prev_idx) / gap).astype(np.float32)[:, None]
    segment_len = (gap / float(length)).astype(np.float32)[:, None]
    dist_prev = ((np.arange(length) - prev_idx) / float(length)).astype(np.float32)[:, None]
    dist_next = ((next_idx - np.arange(length)) / float(length)).astype(np.float32)[:, None]
    return prev_pose, next_pose, phase, segment_len, dist_prev, dist_next


def make_training_sample(
    target: np.ndarray,
    indices: Iterable[int],
    *,
    include_features: bool = False,
    scene_id: str | None = None,
) -> TrainingSample:
    """Build the masked AIS training sample from a dense clip.

    Observed frames are encoded as ``[pose, 0]`` and missing frames as
    ``[0, 1]``. Optional deterministic features are added for later
    Transformer/Curve-AIS experiments without changing the supervision.
    """

    target = np.asarray(target, dtype=np.float32)
    if target.ndim != 2:
        raise ValueError("target must have shape [N, D]")

    length = target.shape[0]
    key_indices = sanitize_key_indices(indices, length)
    observed_mask = np.zeros(length, dtype=np.float32)
    observed_mask[key_indices] = 1.0
    missing_mask = 1.0 - observed_mask

    masked_pose = np.zeros_like(target)
    masked_pose[key_indices] = target[key_indices]

    prev_pose, next_pose, phase, segment_len, dist_prev, dist_next = surrounding_keyposes(
        target, key_indices
    )

    features = [masked_pose, missing_mask[:, None]]
    if include_features:
        features.extend([prev_pose, next_pose, phase, segment_len, dist_prev, dist_next])

    input_seq = np.concatenate(features, axis=-1).astype(np.float32)
    return TrainingSample(
        target=target,
        input_seq=input_seq,
        missing_mask=missing_mask,
        observed_mask=observed_mask,
        prev_pose=prev_pose.astype(np.float32),
        next_pose=next_pose.astype(np.float32),
        phase=phase,
        segment_len=segment_len,
        dist_prev=dist_prev,
        dist_next=dist_next,
        indices=key_indices,
        scene_id=scene_id,
    )
