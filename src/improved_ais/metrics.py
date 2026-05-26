from __future__ import annotations

import numpy as np


def l1(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(target))))


def keypose_drift(pred: np.ndarray, input_keyposes: np.ndarray, key_indices: np.ndarray) -> float:
    pred = np.asarray(pred)
    input_keyposes = np.asarray(input_keyposes)
    key_indices = np.asarray(key_indices, dtype=np.int64)
    return float(np.mean(np.abs(pred[key_indices] - input_keyposes)))


def npss(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if pred.shape != target.shape:
        raise ValueError("pred and target must have the same shape")
    pred_coeff = np.fft.rfft(pred, axis=0)
    target_coeff = np.fft.rfft(target, axis=0)
    pred_power = np.abs(pred_coeff) ** 2
    target_power = np.abs(target_coeff) ** 2
    dim_power = np.sum(target_power, axis=0)
    pred_cdf = np.cumsum(pred_power / np.maximum(np.sum(pred_power, axis=0, keepdims=True), 1e-12), axis=0)
    target_cdf = np.cumsum(target_power / np.maximum(dim_power[None, :], 1e-12), axis=0)
    emd = np.sum(np.abs(pred_cdf - target_cdf), axis=0)
    weights = dim_power / max(float(np.sum(dim_power)), 1e-12)
    return float(np.sum(emd * weights))


def stl1(pred: np.ndarray, target: np.ndarray, key_indices: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    keys = np.asarray(sorted(set(int(i) for i in key_indices)), dtype=np.int64)
    if pred.shape != target.shape:
        raise ValueError("pred and target must have the same shape")
    if len(keys) < 2:
        return float(np.mean(np.abs(pred - target)))

    total = 0.0
    total_frames = 0
    for start, end in zip(keys[:-1], keys[1:]):
        length = int(end - start)
        if length <= 0:
            continue
        target_seg = target[start : start + length]
        best = None
        for delta in range(-length, length + 1):
            ps = start + delta
            pe = ps + length
            if ps < 0 or pe > pred.shape[0]:
                continue
            err = float(np.sum(np.abs(target_seg - pred[ps:pe])))
            if best is None or err < best:
                best = err
        if best is not None:
            total += best
            total_frames += length
    return float(total / max(1, total_frames))
