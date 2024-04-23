
from datasets import load_dataset
from datasets.dataset_dict import DatasetDict
from datasets.table import ConcatenationTable
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.data import Dataset as TFDataset
from tensorflow.keras.utils import img_to_array
from typing import Dict, Tuple

import numpy as np


class Dataset:
    """
    A class to handle loading, processing, and splitting of the wood species identification dataset from Hugging Face.

    This class provides functionality to load the dataset from the specified Hugging Face repository, process the images and labels, and split the dataset into training, validation, and test sets.
    """
    DATASET_DIR = Path("caches/datasets").resolve()
    HUGGINGFACE_REPO = r"LynBean/wood-species-identification"

    @classmethod
    def _load(cls) -> DatasetDict:
        """Loads the dataset from the Hugging Face repository, caching it locally.
        """
        return load_dataset(
            cls.HUGGINGFACE_REPO,
            cache_dir=cls.DATASET_DIR,
            num_proc=20,
        )

    @classmethod
    def _process_table(cls, table: ConcatenationTable) -> Tuple[np.ndarray, np.ndarray]:
        """Processes a table from the dataset, converting images to arrays and extracting labels.
        """
        return (
            np.array([img_to_array(img) for img in table["image"]]),
            np.array(table["label"]),
        )

    @classmethod
    def get(cls, *, validation_split: int=0.3) -> Tuple[TFDataset, TFDataset, TFDataset]:
        """Returns the training, validation, and test datasets as TensorFlow datasets, with an optional validation split ratio.
        """
        dd: DatasetDict = cls._load()

        split_list = cls._process_table(dd["train"])

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

        test_list = cls._process_table(dd["test"])

        train_ds = TFDataset.from_tensor_slices(train_list)
        val_ds = TFDataset.from_tensor_slices(val_list)
        test_ds = TFDataset.from_tensor_slices(test_list)

        train_ds.class_names = dd["train"].features["label"].names
        val_ds.class_names = dd["train"].features["label"].names
        test_ds.class_names = dd["test"].features["label"].names

        return train_ds, val_ds, test_ds
