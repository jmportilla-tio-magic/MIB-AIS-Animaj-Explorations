# Improved AIS-BiLSTM Release

This repository is a standalone release of an improved AIS-BiLSTM model for controller-value motion in-betweening on the public Animaj AIS/MIB HDF5 dataset. It contains the minimum code needed to load the release weights, train the architecture from scratch, export a final `model.safetensors`, and evaluate with local official-style protocols.

The included release checkpoint is:

```text
weights/model.safetensors
```

It is intentionally self-contained. It does not depend on the original `animaj-lab/mib-ais` repository and it is not load-compatible with that repository without code changes, because this architecture adds temporal conditioning and a grouped beta gate that are not present in the original AIS module.

## Repository Layout

```text
.
├── configs/
│   └── release.yaml
├── reports/
│   ├── official_protocol_clip_metrics.csv
│   └── official_protocol_summary_metrics.csv
├── src/improved_ais/
│   ├── checkpoint.py
│   ├── download_data.py
│   ├── eval_official.py
│   ├── export_safetensors.py
│   ├── train.py
│   ├── data/
│   └── models/
├── tests/
│   └── test_install.py
├── weights/
│   ├── model.safetensors
│   ├── model_config.yaml
│   └── training_config.yaml
└── pyproject.toml
```

The `runs/` and `artifacts/` directories are created locally when you train or download data. They are ignored by Git.

## Installation

Use Python 3.10 or newer. A GPU is optional; the code supports CPU, Apple MPS, and CUDA when available.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[train,hf,dev]"
```

If you only want to load the safetensors checkpoint and run inference from your own Python code, install:

```bash
pip install -e .
pip install torch
```

Verify the package and included weights:

```bash
python -m improved_ais.check_install
```

Expected output includes:

```text
improved_ais install ok
weights=.../weights/model.safetensors
```

## Model Theory

Motion in-betweening reconstructs dense animation controller trajectories from sparse observed keyframes. Each clip is a sequence of controller vectors:

```text
target: [frames, 596]
```

The public Animaj data provides dense controller values and block-keyframe masks. Observed keyframes are kept, missing frames are predicted.

### Original AIS Idea

AIS combines two prediction paths:

1. Learned interpolation between previous and next observed keyposes.
2. Direct synthesis from a bidirectional recurrent hidden state.

For each frame, the model predicts:

```text
alpha = interpolation gate
beta  = synthesis/interpolation blend gate
```

The original AIS-style output can be summarized as:

```text
p_interp = (1 - alpha) * previous_keypose + alpha * next_keypose
p_synth  = synthesis(hidden)
pred     = (1 - beta) * p_interp + beta * p_synth
```

This is useful because many in-between frames should stay close to interpolation, while difficult timing or pose transitions need synthesis.

### Novel Architecture Changes

This standalone release uses `improved_ais_bilstm`, which keeps the AIS principle but changes the conditioning and beta gate.

The input to the recurrent trunk remains:

```text
[masked_pose_596, missing_flag_1] = 597 dimensions
```

The model also computes deterministic temporal features for each frame:

```text
phase        = normalized progress between previous and next keypose
segment_len  = keypose gap length normalized by clip length
dist_prev    = normalized distance from previous keypose
dist_next    = normalized distance to next keypose
```

Those four features are concatenated to the BiLSTM hidden state before the AIS heads:

```text
head_input = [bilstm_hidden_1024, temporal_features_4] = 1028 dimensions
```

The beta gate is grouped instead of per-controller:

```text
beta_group: [frames, 64]
beta:       [frames, 596] after deterministic expansion
```

The grouped beta gate reduces high-frequency per-controller gate noise while still allowing different controller regions to choose interpolation or synthesis differently.

## Training Objective

Training optimizes the complete reconstructed sequence. Observed keyframes are optionally hard-copied during evaluation so the score focuses on in-between quality.

The loss combines:

```text
weighted_l1      learned per-controller-dimension L1
velocity_l1      first-derivative consistency
acceleration_l1  second-derivative smoothness
spectral_l1      frequency-domain trajectory shape
gate_tv          temporal smoothness for alpha/beta gates
keypose_l1       observed-keypose reconstruction pressure
```

The release config uses:

```yaml
loss:
  velocity_weight: 0.05
  acceleration_weight: 0.01
  spectral_weight: 0.001
  gate_tv_weight: 0.001
  keypose_weight: 10.0
```

## Dataset

Download the public HDF5 files from Hugging Face:

```bash
python -m improved_ais.download_data --output-dir artifacts/hf_dataset
```

The downloader fetches:

```text
in_house_dataset/.../train.h5
in_house_dataset/.../test.h5
prod_test_dataset/.../test.h5
```

The default config expects paths under:

```text
artifacts/hf_dataset/
```

## Load the Included Safetensors Model

```python
import numpy as np
import torch

from improved_ais.checkpoint import load_improved_model

model = load_improved_model("weights", device="cpu")
model.eval()

input_seq = torch.zeros(1, 224, 597)
input_seq[..., -1] = 1.0
prev_pose = torch.zeros(1, 224, 596)
next_pose = torch.zeros(1, 224, 596)
temporal_features = torch.from_numpy(np.zeros((1, 224, 4), dtype=np.float32))

with torch.no_grad():
    out = model(input_seq, prev_pose, next_pose, temporal_features=temporal_features)

print(out["pred"].shape)
```

## Evaluate the Release

Quick production check:

```bash
python -m improved_ais.eval_official \
  --config configs/release.yaml \
  --checkpoint release=weights \
  --test-set production \
  --max-clips 20 \
  --device auto \
  --output-dir reports/release_production_quick
```

Full official-style local evaluation:

```bash
python -m improved_ais.eval_official \
  --config configs/release.yaml \
  --checkpoint release=weights \
  --test-set all \
  --max-clips 0 \
  --device auto \
  --output-dir reports/release_full
```

Outputs:

```text
official_protocol_clip_metrics.csv
official_protocol_summary_metrics.csv
official_protocol_manifest.json
official_protocol_dashboard.html
```

The primary paper-style metrics are:

```text
shifted_distance  lower is better
npss              lower is better
```

The evaluator also writes `full_l1` and `missing_l1`.

## Training

The release config trains the final model on both the in-house train split and the in-house selection/test split, then reserves the production split for final reporting.

```bash
python -m improved_ais.train \
  --config configs/release.yaml \
  --device auto \
  --output-dir runs/release
```

For a CPU smoke run:

```bash
python -m improved_ais.train \
  --config configs/release.yaml \
  --max-train-clips 2 \
  --max-val-clips 1 \
  --max-steps 2 \
  --device cpu \
  --output-dir runs/smoke
```

Training checkpoints are written as `.pt` files:

```text
runs/release/checkpoints/best.pt
runs/release/checkpoints/last.pt
runs/release/checkpoints/step_XXXXXXXX.pt
```

These files include optimizer, criterion, config, metric, RNG, and early-stopping state. They are useful for resume/debug work but are not the final public release format.

## Two-Stage Release Protocol

For a new release, use a two-stage protocol.

Stage 1: find a step budget with selection metrics and early stopping. Enable early stopping in a copy of the config and train on only the training split.

Stage 2: retrain with early stopping disabled for the selected number of steps using both training and selection data.

This release used a 10,000-step final budget:

```yaml
training:
  max_steps: 10000
  early_stopping:
    enabled: false
```

This protocol separates model selection from final fitting. Production metrics should be treated as final holdout reporting, not as the signal used to choose the stopping point.

## Export Safetensors

Convert a training checkpoint to the public release format:

```bash
python -m improved_ais.export_safetensors \
  --checkpoint runs/release/checkpoints/best.pt \
  --config runs/release/config.yaml \
  --output-dir weights
```

This writes:

```text
weights/model.safetensors
weights/model_config.yaml
weights/training_config.yaml
```

Verify the export:

```bash
python -m improved_ais.check_install
```

## Hugging Face Upload

After exporting, upload the standalone weights folder and model card files:

```bash
hf auth login
hf repo create your-username/improved-ais-release --repo-type model --exist-ok

hf upload your-username/improved-ais-release weights . \
  --repo-type model \
  --commit-message "Upload improved AIS release weights"
```

You can also upload the full repository directory if you want the training/evaluation code on the Hub:

```bash
hf upload your-username/improved-ais-release . . \
  --repo-type model \
  --exclude "artifacts/*" "runs/*" ".venv/*"
```

## Compatibility Notes

The included `model.safetensors` is not in the original `mib-ais` Lightning export layout. The original layout expects:

```text
model.safetensors with model/model.* keys
model.json
trainable_controllers.yaml
```

This release instead stores the improved architecture state dict directly and records architecture metadata in `model_config.yaml` and safetensors metadata. That is deliberate: the original `mib-ais` model does not contain the 1028-dimensional temporal-conditioned heads or grouped beta gate used here.

Use this package to load and evaluate the release.

## Reproducibility Checklist

1. Install package:

```bash
pip install -e ".[train,hf,dev]"
```

2. Verify weights:

```bash
python -m improved_ais.check_install
```

3. Download data:

```bash
python -m improved_ais.download_data --output-dir artifacts/hf_dataset
```

4. Run a smoke train:

```bash
python -m improved_ais.train --config configs/release.yaml --max-train-clips 2 --max-val-clips 1 --max-steps 2 --device cpu --output-dir runs/smoke
```

5. Run a quick eval:

```bash
python -m improved_ais.eval_official --config configs/release.yaml --checkpoint release=weights --test-set production --max-clips 2 --device cpu --output-dir reports/tmp/production_smoke
```

6. Run tests:

```bash
pytest
```
