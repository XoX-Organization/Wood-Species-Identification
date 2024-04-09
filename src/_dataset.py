
from datasets import load_dataset
from datasets.dataset_dict import DatasetDict
from datasets.table import ConcatenationTable
from sklearn.model_selection import train_test_split
from tensorflow.data import Dataset as TFDataset
from tensorflow.keras.utils import img_to_array
from typing import Dict, Tuple

import numpy as np


class Dataset:
    HUGGINGFACE_REPO = r"LynBean/wood-species-identification"

    @staticmethod
    def _load() -> DatasetDict:
        return load_dataset(
            Dataset.HUGGINGFACE_REPO,
            num_proc=20,
        )

    @staticmethod
    def _process_table(table: ConcatenationTable) -> Tuple[np.ndarray, np.ndarray]:
        return (
            np.array([img_to_array(img) for img in table["image"]]),
            np.array(table["label"]),
        )

    @staticmethod
    def get(*, validation_split: int=0.2) -> Tuple[TFDataset, TFDataset, TFDataset]:
        """Returns a tuple of train, validation, and test datasets.
        Each dataset contains an attribute `class_names` that contains the class names.
        """
        dd: DatasetDict = Dataset._load()

        split_list = Dataset._process_table(dd["train"])

        train_images, val_images, train_labels, val_labels = train_test_split(
            split_list[0],
            split_list[1],
            test_size=validation_split,
            random_state=42,
        )

        train_list = (
            train_images,
            train_labels,
        )
        val_list = (
            val_images,
            val_labels,
        )

        test_list = Dataset._process_table(dd["test"])

        train_ds = TFDataset.from_tensor_slices(train_list)
        val_ds = TFDataset.from_tensor_slices(val_list)
        test_ds = TFDataset.from_tensor_slices(test_list)

        train_ds.class_names = dd["train"].features["label"].names
        val_ds.class_names = dd["train"].features["label"].names
        test_ds.class_names = dd["test"].features["label"].names

        return train_ds, val_ds, test_ds
