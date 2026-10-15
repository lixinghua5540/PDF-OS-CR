"""Evaluate a pretrained PDF-OS-CR generator."""

import csv
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from datasets.registry import get_dataset
from models.generator import Generator
from options.test_options import TestOptions
from utils.metrics import LPIPS, MAE, PSNR, SSIM


def require_path(path, kind):
    """Fail early with the resolved missing input path."""
    resolved = Path(path).expanduser().resolve()
    exists = resolved.is_dir() if kind == "directory" else resolved.is_file()
    if not exists:
        raise FileNotFoundError(f"Required {kind} not found: {resolved}")
    return resolved


def clean_state_dict(state_dict):
    """Remove keys added by optional THOP profiling."""
    cleaned = {
        key: value
        for key, value in state_dict.items()
        if not key.endswith(("total_ops", "total_params"))
    }
    print(f"Removed {len(state_dict) - len(cleaned)} profiling key(s).")
    return cleaned


def center_crop(image, size):
    """Crop a BCHW tensor around its spatial center."""
    _, _, height, width = image.shape
    start_h = (height - size) // 2
    start_w = (width - size) // 2
    return image[:, :, start_h:start_h + size, start_w:start_w + size]


def main():
    opts = TestOptions().parse
    require_path(opts.input_data_folder, "directory")
    require_path(opts.data_list_filepath, "file")
    checkpoint_path = require_path(opts.pre_trained, "file")

    result_root = Path(opts.result_root)
    result_root.mkdir(parents=True, exist_ok=True)
    metrics_path = result_root / "metrics.csv"

    is_cuda = torch.cuda.is_available()
    if is_cuda:
        cudnn.enabled = True
        cudnn.benchmark = True
        torch.cuda.set_device(opts.gpu_id)
        print("CUDA is available.")

    generator = Generator(optical_channels=5, sar_channels=4, out_channels=3)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    generator.load_state_dict(clean_state_dict(checkpoint["generator"]), strict=True)
    print(f"Loaded checkpoint: {checkpoint_path}")
    if is_cuda:
        generator = generator.cuda()

    dataset_class, read_splits = get_dataset(opts.dataset_name)
    _, _, test_files = read_splits(opts.data_list_filepath)
    if not test_files:
        raise ValueError("The selected manifest contains no test (3) samples.")
    test_data = dataset_class(opts, test_files)
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=opts.batch_size,
        shuffle=False,
        num_workers=opts.num_workers,
        drop_last=False,
    )

    if not metrics_path.exists():
        with metrics_path.open("w", newline="") as csv_file:
            csv.writer(csv_file).writerow(["PSNR", "SSIM", "MAE", "LPIPS"])

    print("Start testing...")
    generator.eval()
    with torch.no_grad():
        for inputs in tqdm(test_loader, desc="Testing", ncols=100):
            sar = inputs["SAR_data"]
            cloudy = inputs["cloudy_data"]
            ground_truth = inputs["groundtruth_data"]
            sar_edge = inputs["SAR_edge"]
            sar_block = inputs["SAR_block"]
            if is_cuda:
                sar = sar.cuda()
                cloudy = cloudy.cuda()
                ground_truth = ground_truth.cuda()
                sar_edge = sar_edge.cuda()
                sar_block = sar_block.cuda()

            output = generator(cloudy, sar_block, torch.cat((sar_edge, sar), dim=1))
            cropped_output = center_crop(output, 128)
            cropped_ground_truth = center_crop(ground_truth, 128)
            row = [
                PSNR(cropped_output, cropped_ground_truth),
                SSIM(cropped_output, cropped_ground_truth).item(),
                MAE(cropped_output, cropped_ground_truth).item(),
                LPIPS(cropped_output, cropped_ground_truth),
            ]
            with metrics_path.open("a", newline="") as csv_file:
                csv.writer(csv_file).writerow(row)


if __name__ == "__main__":
    main()
