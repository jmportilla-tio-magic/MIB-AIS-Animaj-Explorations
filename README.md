# Improved AIS-BiLSTM Release

This repository is a standalone release of an improved AIS-BiLSTM model for controller-value motion in-betweening on the public Animaj AIS/MIB HDF5 dataset. It includes the code needed to load the release weights, train the architecture, export a final `model.safetensors`, and evaluate with local official-style protocols.

The included release checkpoint is:

```text
weights/model.safetensors
```

It is intentionally self-contained. It does not depend on the original `animaj-lab/mib-ais` repository and it is not load-compatible with that repository without code changes, because this architecture adds temporal conditioning and a grouped beta gate that are not present in the original AIS module.

For reviewers, the shortest path is:

```bash
pip install -e ".[train,hf,dev]"
python -m improved_ais.check_install
python -m improved_ais.download_data --output-dir artifacts/hf_dataset
python -m improved_ais.eval_official \
  --config configs/release.yaml \
  --checkpoint release=weights \
  --test-set all \
  --max-clips 0 \
  --device auto \
  --output-dir reports/release_full
pytest
```

## Repository Layout

```text
.
├── configs/
│   └── release.yaml
├── reports/
│   ├── release_full/
│   │   ├── official_protocol_clip_metrics.csv
│   │   ├── official_protocol_summary_metrics.csv
│   │   ├── official_protocol_manifest.json
│   │   └── official_protocol_dashboard.html
│   ├── official_protocol_clip_metrics.csv
│   └── official_protocol_summary_metrics.csv
├── docs/
│   └── comparison_with_mib_ais.md
├── src/improved_ais/
│   ├── checkpoint.py
│   ├── download_data.py
│   ├── eval_official.py
│   ├── export_safetensors.py
│   ├── train.py
│   ├── data/
│   └── models/
├── tests/
│   ├── test_eval_metrics.py
│   └── test_install.py
├── weights/
│   ├── model.safetensors
│   ├── model_config.yaml
│   └── training_config.yaml
└── pyproject.toml
```

The `reports/release_full/` directory is the canonical full release evaluation bundle. The root `reports/official_protocol_*.csv` files are convenience mirrors of the CSVs from that bundle for direct inspection. The `runs/` and `artifacts/` directories are created locally when you train or download data. They are ignored by Git.

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

Run the test suite:

```bash
pytest
```

## Model Architecture

Motion in-betweening reconstructs dense animation controller trajectories from sparse observed keyframes. Each clip is a sequence of controller vectors:

```text
target: [frames, 596]
```

The public Animaj data provides dense controller values and block-keyframe masks. Observed keyframes are kept, missing frames are predicted.

### AIS Formulation

AIS combines two prediction paths:

1. Learned interpolation between previous and next observed keyposes.
2. Direct synthesis from a bidirectional recurrent hidden state.

For each frame, the model predicts:

```text
alpha = interpolation gate
beta  = synthesis/interpolation blend gate
```

Using this implementation's alpha convention, the AIS-style output is:

```text
p_interp = (1 - alpha) * previous_keypose + alpha * next_keypose
p_synth  = synthesis(hidden)
pred     = (1 - beta) * p_interp + beta * p_synth
```

This decomposition preserves the strong baseline behavior of interpolation while allowing the recurrent synthesis path to model transitions that require timing- or pose-dependent corrections.

### Changes Relative to `mib-ais`

This standalone release uses `improved_ais_bilstm`, which keeps the AIS principle from the original `mib-ais` LSTM model while changing the AIS head. The recurrent trunk remains a 2-layer bidirectional LSTM over the standard masked controller input:

```text
[masked_pose_596, missing_flag_1] = 597 dimensions
```

The upstream `mib-ais` AIS LSTM predicts alpha, beta, and synthesis directly from the BiLSTM hidden state:

```text
base_head_input = bilstm_hidden_1024
alpha: [frames, 596]
beta:  [frames, 596]
```

This release adds deterministic temporal features for each frame:

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

The beta gate is also grouped instead of predicted independently for every controller dimension:

```text
beta_group: [frames, 64]
beta:       [frames, 596] after deterministic expansion
```

The grouped beta gate reduces high-frequency per-controller gate variation while still allowing different controller regions to choose interpolation or synthesis differently. The checkpoint is therefore not weight-compatible with the upstream `mib-ais` AIS module, but it uses the same controller-vector dimensionality, masked-input convention, public HDF5 data, and primary evaluation metrics.

For a detailed parity audit against the local upstream checkout, see `docs/comparison_with_mib_ais.md`.

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

The checkpoint loader reads either a weights directory or a direct `.safetensors` file. The model returns a dictionary containing `pred`, `alpha`, `beta`, `beta_group`, `p_interp`, and `p_synth`.

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

Production subset evaluation:

```bash
python -m improved_ais.eval_official \
  --config configs/release.yaml \
  --checkpoint release=weights \
  --test-set production \
  --max-clips 20 \
  --device auto \
  --output-dir reports/release_production_subset
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
reports/release_full/official_protocol_clip_metrics.csv
reports/release_full/official_protocol_summary_metrics.csv
reports/release_full/official_protocol_manifest.json
reports/release_full/official_protocol_dashboard.html
```

The root `reports/official_protocol_clip_metrics.csv` and `reports/official_protocol_summary_metrics.csv` files mirror the latest full-release CSVs for convenience. Use `reports/release_full/` when you need the manifest or dashboard associated with the run.

The primary protocol metrics are:

```text
shifted_distance  lower is better
npss              lower is better
```

The evaluator also writes `full_l1` and `missing_l1`.

Report files have distinct roles:

```text
clip_metrics.csv     one row per evaluated clip and model
summary_metrics.csv  aggregate mean, median, and standard deviation by test set and model
manifest.json        checkpoint paths, preprocessing settings, random seed, filter counts
dashboard.html       local interactive view over the CSV/JSON report bundle
```

## Fair Comparison Protocol

The local evaluator mirrors the public `mib-ais` test-set structure:

```text
held_out_algorithmic  in-house test split with block keyframes
held_out_random       in-house test split with 90% random masking
production            production test split with block keyframes
```

For release reporting, use `--test-set all --max-clips 0` and treat `production` as a final holdout. The metric implementations for `shifted_distance` and `npss` are ported from the upstream repository so this checkpoint can be scored without requiring the upstream Lightning module or visualization stack.

The evaluator also mirrors the upstream preprocessing used by the `mib-ais` datamodule:

```text
relative_root_translation  subtract frame-0 x_main_CTRL translateX/Y/Z from each sequence
range_filter_threshold     exclude scenes where x_main_CTRL translation range exceeds 100
random-mask replay         write unmasked_frame_indices for every scored clip
```

These settings are recorded in `official_protocol_manifest.json`.

To evaluate the upstream release checkpoint from the original repository:

```bash
cd ../mib-ais
uv sync --frozen --all-groups
uv run python -m motion_inbetweening.scripts.test AnimajSAS/AIS_BI_LSTM_v0 --test-set all
```

The upstream command uses the original Lightning datamodule and logger. For an exact same-mask side-by-side comparison inside this repository's CSV reports, wrap the upstream `Seq2SeqModule` behind the prediction interface used by `improved_ais.eval_official` and replay the `unmasked_frame_indices` column from `official_protocol_clip_metrics.csv`.

To evaluate this release checkpoint:

```bash
cd ../MIB-AIS-Animaj-Explorations
python -m improved_ais.eval_official \
  --config configs/release.yaml \
  --checkpoint release=weights \
  --test-set all \
  --max-clips 0 \
  --device auto \
  --output-dir reports/release_full
```

## Training

The release config trains the final model on both the in-house train split and the in-house selection/test split, then reserves the production split for final reporting.

```bash
python -m improved_ais.train \
  --config configs/release.yaml \
  --device auto \
  --output-dir runs/release
```

For a small CPU installation check:

```bash
python -m improved_ais.train \
  --config configs/release.yaml \
  --max-train-clips 2 \
  --max-val-clips 1 \
  --max-steps 2 \
  --device cpu \
  --output-dir runs/check
```

Training checkpoints are written as `.pt` files:

```text
runs/release/checkpoints/best.pt
runs/release/checkpoints/last.pt
runs/release/checkpoints/step_XXXXXXXX.pt
```

These files include optimizer, criterion, config, metric, RNG, and early-stopping state. They are intended for training continuation and experiment inspection; the public release artifact is the exported safetensors package.

## Model Selection and Reporting

The release workflow separates model selection from final holdout reporting. Hyperparameters, checkpoint selection criteria, and training duration should be chosen using the in-house training and selection splits only. The production split is reserved for final reporting and should not be used as a tuning signal.

The included `configs/release.yaml` records the final training configuration used for the packaged checkpoint. For new runs, keep the production split out of any model-selection loop, then evaluate the exported checkpoint once with `improved_ais.eval_official`.

This makes the reported production metrics a holdout estimate of release quality rather than a criterion optimized during development.

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

This release instead stores the improved architecture state dict directly and records architecture metadata in `model_config.yaml`, `training_config.yaml`, and safetensors metadata. That is deliberate: the original `mib-ais` model does not contain the 1028-dimensional temporal-conditioned heads or grouped beta gate used here.

Use this package to load and evaluate the release checkpoint. To run it from the original repository, a small integration layer would need to instantiate `ImprovedAISBiLSTM` and map the upstream datamodule batch format into this model's `input_seq`, `prev_pose`, `next_pose`, and `temporal_features` tensors.

## Author Handoff

Files intended for handoff:

```text
weights/model.safetensors       release checkpoint
weights/model_config.yaml       architecture metadata
weights/training_config.yaml    training and preprocessing metadata
configs/release.yaml            runnable local config
reports/release_full/           latest full evaluation bundle
```

The model output is a `[frames, 596]` controller-value trajectory in the same vector order as the public HDF5 data. The checkpoint is safetensors, but it is not in the upstream Lightning export layout; load it with `improved_ais.checkpoint.load_improved_model` or add an integration layer in `mib-ais` that instantiates `ImprovedAISBiLSTM`.

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

4. Run a small training check:

```bash
python -m improved_ais.train --config configs/release.yaml --max-train-clips 2 --max-val-clips 1 --max-steps 2 --device cpu --output-dir runs/check
```

5. Run a small evaluation check:

```bash
python -m improved_ais.eval_official --config configs/release.yaml --checkpoint release=weights --test-set production --max-clips 2 --device cpu --output-dir reports/tmp/production_check
```

6. Run tests:

```bash
pytest
```
