"""Generate SAR edge and block priors as GeoTIFF files."""

import argparse
from pathlib import Path

import rasterio
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from datasets.common import get_train_val_test_filelists, normalize_sar, read_tif
from utils.canny import extract_canny
from utils.rtv import tsmooth


class SarDataset(Dataset):
    """Load SAR images referenced by the prior-generation CSV layout."""

    def __init__(self, data_root, rows):
        self.data_root = Path(data_root)
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        sar_path = self.data_root.joinpath(*row[1:6])
        sar = normalize_sar(read_tif(sar_path))
        block = tsmooth(sar.transpose(1, 2, 0), lambda_=0.008, sigma=3)
        block = torch.from_numpy(block.transpose(2, 0, 1))
        sar_tensor = torch.from_numpy(sar)
        edge = torch.cat(
            (
                torch.from_numpy(extract_canny(sar_tensor[0:1])),
                torch.from_numpy(extract_canny(sar_tensor[1:])),
            ),
            dim=0,
        )
        return {
            "SAR_edge": edge,
            "SAR_block": block,
            "s1_path": str(sar_path),
            "file_name": row[5],
        }


def save_sar_as_tif(data, output_path, reference_path):
    """Save a two-channel prior while preserving source georeferencing."""
    if isinstance(data, torch.Tensor):
        data = data.squeeze(0).numpy()
    if data.shape[0] != 2:
        raise ValueError(f"Expected two SAR channels, got shape {data.shape}")

    with rasterio.open(reference_path) as source:
        profile = source.profile.copy()
    profile.update(count=2, dtype=data.dtype)
    with rasterio.open(output_path, "w", **profile) as destination:
        destination.write(data)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate PDF-OS-CR SAR priors")
    parser.add_argument("--input-data-folder", required=True)
    parser.add_argument("--data-list-filepath", required=True)
    parser.add_argument("--block-output-dir", required=True)
    parser.add_argument("--edge-output-dir", required=True)
    return parser.parse_args()


def main():
    opts = parse_args()
    train_rows, _, _ = get_train_val_test_filelists(opts.data_list_filepath)
    if not train_rows:
        raise ValueError("The manifest contains no training (1) samples.")

    block_dir = Path(opts.block_output_dir).expanduser().resolve()
    edge_dir = Path(opts.edge_output_dir).expanduser().resolve()
    block_dir.mkdir(parents=True, exist_ok=True)
    edge_dir.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(SarDataset(opts.input_data_folder, train_rows), batch_size=1)
    for sample in tqdm(loader, desc="Generating priors", unit="image"):
        reference = sample["s1_path"][0]
        file_name = sample["file_name"][0]
        save_sar_as_tif(sample["SAR_block"], block_dir / file_name, reference)
        save_sar_as_tif(sample["SAR_edge"], edge_dir / file_name, reference)


if __name__ == "__main__":
    main()

