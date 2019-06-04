"""
Module for DataRepositories which can get datasets
"""
from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
import pandas as pd
from .dataset import CombinedDataset


class DatasetRepositoryBase(ABC):
    """
    Abstract class for getting datasets
    """

    @abstractmethod
    def load_dataset(self) -> CombinedDataset:
        """
        Loads the dataset
        """


class DatasetRepository:
    def __init__(self, datastore_root: Path, width: int, height: int):
        self.datastore_root = datastore_root
        self.width = width
        self.height = height

    def load_dataset(self) -> CombinedDataset:
        pathlist = Path(self.datastore_root).glob('*.csv')
        for path in pathlist: 
            
        result = pd.read_csv(self.datastore_root)
        result.reshape(width, height)
        return result


class DummyDatasetRepository(DatasetRepositoryBase):
    """
    Returns dummy data
    renders should be 0.2 darker than the images, with a random factor applied
    """

    def __init__(self, samples: int):
        self.samples = samples
        self.width = 20
        self.height = 20

    def load_dataset(self) -> CombinedDataset:
        names = [f"{i}" for i in range(self.samples)]
        renders = [np.random.rand(self.width, self.height, 5) for _ in range(self.samples)]
        images = [(renders[i][:, :, :3] * 0.8 + (np.random.rand(self.width, self.height, 3) * 0.2)) * 0.8 + 0.2
                  for i in range(self.samples)]

        dataset = CombinedDataset(dataset_path=Path(), names=names, images=images, renders=renders)
        return dataset
