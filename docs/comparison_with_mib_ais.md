# Comparison With `mib-ais`

This note records the compatibility audit against the local upstream checkout at `/Users/jmportilla/github-repos/mib-ais`. It is intended to make the release handoff explicit for reviewers who want to run the checkpoint, compare metrics, or adapt the model to the original codebase.

## Upstream Reference Points

The relevant upstream files are:

```text
motion_inbetweening/domain/models/explicit_interpolation_extrapolation_residual_lstm.py
motion_inbetweening/domain/models/ais_layer.py
motion_inbetweening/infra/configs/train.py
motion_inbetweening/infra/loading/safetensors.py
motion_inbetweening/scripts/test.py
motion_inbetweening/metric/shifted_distance.py
motion_inbetweening/metric/npss.py
motion_inbetweening/app_services/online_preprocessing/relative_sequence.py
motion_inbetweening/app_services/online_preprocessing/filtering.py
```

The upstream best LSTM configuration uses an AIS last layer with:

```text
pose_dim: 596
input: [masked_pose_596, missing_flag_1]
BiLSTM hidden size: 512
BiLSTM layers: 2
dropout: 0.3003
alpha head: hidden_1024 -> 596
beta head: hidden_1024 -> 596
synthesis head: hidden_1024 -> hidden_1024 -> 596
```

## Architecture Parity

This repo intentionally preserves:

- The 596-dimensional Pocoyo controller-vector output.
- The masked-input convention `[masked_pose, missing_flag]`.
- A 2-layer bidirectional LSTM trunk with hidden size 512 and dropout 0.3003.
- The AIS decomposition into learned interpolation, synthesis, and a beta blend.
- The public HDF5 dataset layout from `AnimajSAS/mib_rig_controllers_values`.

This repo intentionally changes:

- The AIS heads receive `hidden_1024 + temporal_features_4`, for a 1028-dimensional head input.
- The temporal features are deterministic: phase, segment length, distance from previous keypose, and distance to next keypose.
- The beta gate is predicted as 64 groups and expanded to 596 controller dimensions.
- The checkpoint stores the improved model state dict directly, rather than an upstream Lightning `Seq2SeqModule` state dict.

One notation detail: the upstream code computes interpolation as `alpha * previous + (1 - alpha) * next`; this repo uses `(1 - alpha) * previous + alpha * next`. Since alpha is learned, this is a convention difference rather than a change to the interpolation family, but it is another reason the weights are not directly interchangeable.

## Checkpoint Format

The upstream safetensors loader expects:

```text
model.safetensors
model.json
trainable_controllers.yaml
```

with tensor names prefixed by `model/` and optional normalizer tensors prefixed by `normalizer/`.

This release provides:

```text
weights/model.safetensors
weights/model_config.yaml
weights/training_config.yaml
```

The checkpoint is a valid safetensors artifact, but it is not load-compatible with `motion_inbetweening.infra.loading.safetensors.load_safetensors` without an integration layer. Reviewers should run this repository directly, or add upstream integration code that instantiates `improved_ais.models.improved_ais.ImprovedAISBiLSTM` and prepares the same inputs used by `improved_ais.data.window.make_training_sample`.

## Evaluation Parity

The local evaluator mirrors the upstream test script's three public test sets:

```text
held_out_algorithmic: in_house_dataset with block keyframes
held_out_random:      in_house_dataset with random_uniform masking at ratio 0.9
production:           prod_test_dataset with block keyframes
```

The local evaluator also mirrors the upstream preprocessing path:

- `x_main_CTRL` translation is represented relative to the sequence start. In the public controller vector order this corresponds to indices `0, 1, 2`.
- Scenes are filtered with `ranges.csv` when any `x_main_CTRL:translateX/Y/Z` range exceeds `100`, matching the upstream `RelativeControllerTransformationConfig` thresholds.
- Random-mask evaluations record `unmasked_frame_indices` in the clip CSV so the exact masks can be replayed across models.

The primary metrics are ported from upstream:

- `shifted_distance` follows `motion_inbetweening/metric/shifted_distance.py` and uses elementwise L1 distance over shifted candidate motions.
- `npss` follows `motion_inbetweening/metric/npss.py` with full FFT power spectra and target-power weighting.

Known differences from the upstream test runner:

- This repo reads the public HDF5 files directly instead of using the upstream Lightning datamodule.
- This repo writes CSV, JSON, and HTML reports locally instead of logging through Lightning or MLflow.
- The upstream random mask generator uses PyTorch randomness; this repo uses a NumPy generator with the seed recorded in the manifest and exact unmasked frame indices recorded in the clip CSV.
- Visualization is not included; the original authors' Maya-based renderer or future open renderer should be used for video generation.

## Side-by-Side Commands

Evaluate the upstream Hugging Face checkpoint from the original repository:

```bash
cd /Users/jmportilla/github-repos/mib-ais
uv sync --frozen --all-groups
uv run python -m motion_inbetweening.scripts.test AnimajSAS/AIS_BI_LSTM_v0 --test-set all
```

Evaluate this release checkpoint from this repository:

```bash
cd /Users/jmportilla/github-repos/MIB-AIS-Animaj-Explorations
python -m improved_ais.eval_official \
  --config configs/release.yaml \
  --checkpoint release=weights \
  --test-set all \
  --max-clips 0 \
  --device auto \
  --output-dir reports/release_full
```

For final reporting, run:

```bash
python -m improved_ais.eval_official \
  --config configs/release.yaml \
  --checkpoint release=weights \
  --test-set all \
  --max-clips 0 \
  --device auto \
  --output-dir reports/release_full
```

The production split should be treated as holdout reporting only.
