from __future__ import annotations

import argparse
from pathlib import Path


DATASET_REPO = "AnimajSAS/mib_rig_controllers_values"
CONTROLLER_HASH = "ef25a8e5ecd2fa86420741e5646cd785aa025c44c06f4118aab77f558c8f6981"

PUBLIC_FILES = [
    f"in_house_dataset/vectorized_controller_values/{CONTROLLER_HASH}/train.h5",
    "in_house_dataset/vectorized_block_keyframes/train.h5",
    f"in_house_dataset/vectorized_controller_values/{CONTROLLER_HASH}/test.h5",
    "in_house_dataset/vectorized_block_keyframes/test.h5",
    "in_house_dataset/ranges.csv",
    f"prod_test_dataset/vectorized_controller_values/{CONTROLLER_HASH}/test.h5",
    "prod_test_dataset/vectorized_block_keyframes/test.h5",
    "prod_test_dataset/ranges.csv",
]


def download_data(*, output_dir: Path, files: list[str] | None = None) -> list[Path]:
    try:
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError as exc:
        raise RuntimeError("huggingface_hub is required. Install with `pip install -e .[hf]`.") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for file in files or PUBLIC_FILES:
        path = hf_hub_download(DATASET_REPO, file, repo_type="dataset", local_dir=output_dir)
        downloaded.append(Path(path))
    return downloaded


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the public Animaj AIS HDF5 files used by this package.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/hf_dataset"))
    args = parser.parse_args()
    for path in download_data(output_dir=args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
