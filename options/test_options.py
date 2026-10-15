"""Evaluation command-line options."""

import argparse
import pprint

from options.common import repo_path

# Edit these paths before evaluation. Repository-relative paths are resolved
# from the project root; dataset roots may be absolute paths.
TEST_PATHS = {
    "SEN12MS-CR": {
        "input_data_folder": r"/path/to/SEN12MS-CR",
        "data_list_filepath": "data/data_SEN12MS.csv",
        "pre_trained": "ckpt/SEN12MS-CR/best_model.pt",
        "result_root": "results/SEN12MS-CR",
    },
    "LuoJiaSET-OSFCR": {
        "input_data_folder": r"/path/to/LuojiaSET-OSFCR",
        "data_list_filepath": "data/data_LuoJia.csv",
        "pre_trained": "ckpt/LuojiaSET-OSFCR/best_model.pt",
        "result_root": "results/LuojiaSET-OSFCR",
    },
}


class TestOptions:
    def __init__(self):
        parser = argparse.ArgumentParser(description="Evaluate PDF-OS-CR")
        parser.add_argument(
            "--dataset_name",
            choices=["SEN12MS-CR", "LuoJiaSET-OSFCR"],
            default="LuoJiaSET-OSFCR",
        )
        parser.add_argument("--num_workers", type=int, default=4)
        parser.add_argument("--batch_size", type=int, default=1)
        parser.add_argument("--gpu_id", type=int, default=0)

        self.opts = parser.parse_args()
        paths = TEST_PATHS[self.opts.dataset_name]
        self.opts.input_data_folder = repo_path(paths["input_data_folder"])
        self.opts.data_list_filepath = repo_path(paths["data_list_filepath"])
        self.opts.pre_trained = repo_path(paths["pre_trained"])
        self.opts.result_root = repo_path(paths["result_root"])

    @property
    def parse(self):
        pprint.pprint(vars(self.opts))
        return self.opts
