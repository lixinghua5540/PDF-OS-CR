"""SEN12MS-CR dataset definition."""

from datasets.common import BaseAlignedDataset, get_train_val_test_filelists


class AlignedDataset(BaseAlignedDataset):
    """Load SEN12MS-CR SAR, optical, edge, and block images."""

    def build_paths(self, row):
        file_name = row[6]
        return {
            "sar": self.data_root / row[1] / file_name,
            "cloudy": self.data_root / row[2] / file_name,
            "ground_truth": self.data_root / row[3] / file_name,
            "sar_edge": self.data_root / row[4] / file_name,
            "sar_block": self.data_root / row[5] / file_name,
            "file_name": file_name,
        }


__all__ = ["AlignedDataset", "get_train_val_test_filelists"]

