"""Train PDF-OS-CR on SEN12MS-CR or LuojiaSET-OSFCR."""

import os
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim

from datasets.registry import get_dataset
from models.discriminator import Discriminator
from models.generator import Generator
from options.train_options import TrainOptions
from trainer import train
from utils.ddp import data_sampler
from utils.distributed import synchronize


def require_path(path, kind):
    """Fail early with the resolved missing input path."""
    resolved = Path(path).expanduser().resolve()
    exists = resolved.is_dir() if kind == "directory" else resolved.is_file()
    if not exists:
        raise FileNotFoundError(f"Required {kind} not found: {resolved}")
    return resolved


def main():
    opts = TrainOptions().parse
    require_path(opts.input_data_folder, "directory")
    require_path(opts.data_list_filepath, "file")
    Path(opts.save_dir).mkdir(parents=True, exist_ok=True)

    dataset_class, read_splits = get_dataset(opts.dataset_name)
    train_files, val_files, _ = read_splits(opts.data_list_filepath)
    if not train_files or not val_files:
        raise ValueError(
            "The selected manifest must contain non-empty training (1) "
            "and validation (2) splits."
        )

    is_cuda = torch.cuda.is_available()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    opts.distributed = is_cuda and world_size > 1
    if is_cuda:
        cudnn.enabled = True
        cudnn.benchmark = True
        print(f"CUDA is available ({world_size} process(es)).")
        if opts.distributed:
            torch.cuda.set_device(opts.local_rank)
            torch.distributed.init_process_group(backend="nccl", init_method="env://")
            synchronize()
        else:
            torch.cuda.set_device(opts.gpu_id)

    train_data = dataset_class(opts, train_files)
    val_data = dataset_class(opts, val_files)
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=opts.batch_size,
        sampler=data_sampler(train_data, shuffle=True, distributed=opts.distributed),
        num_workers=opts.num_workers,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=opts.batch_size,
        sampler=data_sampler(val_data, shuffle=True, distributed=opts.distributed),
        num_workers=opts.num_workers,
        drop_last=True,
    )

    generator = Generator(optical_channels=5, sar_channels=4, out_channels=3)
    discriminator = Discriminator(optical_channels=3, sar_channels=4)
    if is_cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()

    if opts.finetune:
        print("Fine-tuning mode enabled.")
        learning_rate = opts.lr_finetune
        generator.freeze_ec_bn = True
    else:
        learning_rate = opts.gen_lr

    generator_optim = optim.Adam(
        filter(lambda parameter: parameter.requires_grad, generator.parameters()),
        lr=learning_rate,
    )
    discriminator_optim = optim.Adam(
        discriminator.parameters(), lr=learning_rate * opts.D2G_lr
    )

    opts.start_epoch = 0
    if opts.pre_trained:
        checkpoint_path = require_path(opts.pre_trained, "file")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        opts.start_epoch = checkpoint.get("epoch", 0)
        generator.load_state_dict(checkpoint["generator"])
        discriminator.load_state_dict(checkpoint["discriminator"])
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print("Training from scratch.")

    if opts.distributed:
        generator = nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[opts.local_rank],
            output_device=opts.local_rank,
            broadcast_buffers=False,
        )
        discriminator = nn.parallel.DistributedDataParallel(
            discriminator,
            device_ids=[opts.local_rank],
            output_device=opts.local_rank,
            broadcast_buffers=False,
        )

    train(
        opts,
        train_loader,
        val_loader,
        generator,
        discriminator,
        generator_optim,
        discriminator_optim,
        is_cuda,
    )


if __name__ == "__main__":
    main()
