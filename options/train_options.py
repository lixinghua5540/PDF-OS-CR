"""Training command-line options."""

import argparse
import pprint

from options.common import parse_bool, repo_path

# Edit these paths before training. Repository-relative paths are resolved
# from the project root; dataset roots may be absolute paths.
TRAIN_PATHS = {
    "SEN12MS-CR": {
        "input_data_folder": r"/path/to/SEN12MS-CR",
        "data_list_filepath": "data/data_SEN12MS.csv",
        "save_dir": "ckpt/SEN12MS-CR",
    },
    "LuoJiaSET-OSFCR": {
        "input_data_folder": r"/path/to/LuojiaSET-OSFCR",
        "data_list_filepath": "data/data_LuoJia.csv",
        "save_dir": "ckpt/LuojiaSET-OSFCR",
    },
}
LOG_DIR = "runs/log"
PRETRAINED_CHECKPOINT = ""


class TrainOptions:
    def __init__(self):
        parser = argparse.ArgumentParser(description="Train PDF-OS-CR")
        parser.add_argument(
            "--dataset_name",
            choices=["SEN12MS-CR", "LuoJiaSET-OSFCR"],
            default="SEN12MS-CR",
        )
        parser.add_argument("--epochs", type=int)

        parser.add_argument("--num_workers", type=int, default=4)
        parser.add_argument("--batch_size", type=int, default=6)
        parser.add_argument("--gen_lr", type=float, default=0.0002)
        parser.add_argument("--D2G_lr", type=float, default=0.1)
        parser.add_argument("--lr_finetune", type=float, default=0.00005)
        parser.add_argument("--finetune", type=parse_bool, default=False)
        parser.add_argument("--save_epoch", type=int, default=1)
        parser.add_argument("--gpu_id", type=int, default=0)

        parser.add_argument("--L1_LOSS", type=float, default=100.0)
        parser.add_argument("--ADVERSARIAL_LOSS", type=float, default=0.1)
        parser.add_argument("--MSSSIM_LOSS", type=float, default=20.0)
        parser.add_argument("--local_rank", type=int, default=0)

        self.opts = parser.parse_args()
        paths = TRAIN_PATHS[self.opts.dataset_name]
        self.opts.input_data_folder = repo_path(paths["input_data_folder"])
        self.opts.data_list_filepath = repo_path(paths["data_list_filepath"])
        self.opts.save_dir = repo_path(paths["save_dir"])
        self.opts.log_dir = repo_path(LOG_DIR)
        self.opts.pre_trained = repo_path(PRETRAINED_CHECKPOINT)
        if self.opts.epochs is None:
            self.opts.epochs = 50 if self.opts.dataset_name == "SEN12MS-CR" else 100

    @property
    def parse(self):
        pprint.pprint(vars(self.opts))
        return self.opts
