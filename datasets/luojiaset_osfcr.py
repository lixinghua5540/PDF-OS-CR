"""LuojiaSET-OSFCR dataset definition."""

from datasets.common import BaseAlignedDataset, get_train_val_test_filelists


class AlignedDataset(BaseAlignedDataset):
    """Load LuojiaSET-OSFCR SAR, optical, edge, and block images."""

    def build_paths(self, row):
        file_name = row[7]
        scene = row[1]
        return {
            "sar": self.data_root / scene / row[2] / file_name,
            "cloudy": self.data_root / scene / row[3] / file_name,
            "ground_truth": self.data_root / scene / row[4] / file_name,
            "sar_edge": self.data_root / scene / row[5] / file_name,
            "sar_block": self.data_root / scene / row[6] / file_name,
            "file_name": file_name,
        }


__all__ = ["AlignedDataset", "get_train_val_test_filelists"]

