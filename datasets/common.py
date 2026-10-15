"""Shared data loading and preprocessing for the supported datasets."""

import csv
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

SAR_MIN = (-25.0, -32.5)
SAR_MAX = (0.0, 0.0)
OPTICAL_SCALE = 10000.0


def read_tif(path):
    """Read all bands from a GeoTIFF and replace invalid values."""
    with rasterio.open(path, "r", driver="GTiff") as src:
        image = src.read()
    image[np.isnan(image)] = np.nanmean(image)
    return image.astype("float32")


def read_rgb_tif(path):
    """Read the three optical bands used by the released model."""
    return read_tif(path)[1:4]


def normalize_sar(image):
    """Normalize VV and VH channels to [0, 1]."""
    for channel, (lower, upper) in enumerate(zip(SAR_MIN, SAR_MAX)):
        image[channel] = np.clip(image[channel], lower, upper)
        image[channel] = (image[channel] - lower) / (upper - lower)
    return image


def normalize_optical(image):
    """Normalize Sentinel-2 reflectance values to [0, 1]."""
    return np.clip(image, 0, OPTICAL_SCALE) / OPTICAL_SCALE


def get_train_val_test_filelists(list_path):
    """Read the train (1), validation (2), and test (3) CSV splits."""
    splits = {"1": [], "2": [], "3": []}
    with Path(list_path).open("r", newline="") as csv_file:
        for row in csv.reader(csv_file):
            if row and row[0] in splits:
                splits[row[0]].append(row)
    return splits["1"], splits["2"], splits["3"]


def set_input(batch):
    """Return tensors in the order expected by the training loop."""
    return (
        batch["SAR_data"],
        batch["cloudy_data"],
        batch["groundtruth_data"],
        batch["SAR_edge"],
        batch["SAR_block"],
    )


class BaseAlignedDataset(Dataset):
    """Common optical-SAR sample loading; subclasses define row layouts."""

    def __init__(self, opts, filelist):
        self.opts = opts
        self.filelist = filelist
        self.data_root = Path(opts.input_data_folder)

    def __len__(self):
        return len(self.filelist)

    def build_paths(self, row):
        raise NotImplementedError

    def __getitem__(self, index):
        paths = self.build_paths(self.filelist[index])

        sar = normalize_sar(read_tif(paths["sar"]))
        cloudy = normalize_optical(read_rgb_tif(paths["cloudy"]))
        ground_truth = normalize_optical(read_rgb_tif(paths["ground_truth"]))
        sar_edge = read_tif(paths["sar_edge"])
        sar_block = read_tif(paths["sar_block"])

        sample = {
            "cloudy_data": torch.from_numpy(cloudy),
            "groundtruth_data": torch.from_numpy(ground_truth),
            "SAR_data": torch.from_numpy(sar),
            "SAR_edge": torch.from_numpy(sar_edge),
            "SAR_block": torch.from_numpy(sar_block),
            "cloudy_path": str(paths["cloudy"]),
            "groundtruth_path": str(paths["ground_truth"]),
            "s1_path": str(paths["sar"]),
            "file_name": paths["file_name"],
        }

        return sample
