"""Dataset selection by the public command-line name."""

from datasets.common import get_train_val_test_filelists
from datasets.luojiaset_osfcr import AlignedDataset as LuojiaSetDataset
from datasets.sen12ms_cr import AlignedDataset as Sen12msDataset

DATASETS = {
    "SEN12MS-CR": (Sen12msDataset, get_train_val_test_filelists),
    "LuoJiaSET-OSFCR": (LuojiaSetDataset, get_train_val_test_filelists),
}


def get_dataset(name):
    """Return the dataset class and split reader for a public dataset name."""
    try:
        return DATASETS[name]
    except KeyError as exc:
        supported = ", ".join(DATASETS)
        raise ValueError(f"Unknown dataset '{name}'. Supported datasets: {supported}") from exc


__all__ = ["get_dataset"]
