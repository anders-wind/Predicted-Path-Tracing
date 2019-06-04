"""
Module for DataRepositories which can get datasets
"""
from typing import Tuple
from abc import ABC, abstractmethod
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from .dataset_repository import DatasetRepositoryBase
from .dataset import CombinedDataset


class DatasetServiceBase(ABC):
    """
    Abstract class for getting datasets
    """

    @abstractmethod
    def get_dataset(self) -> CombinedDataset:
        """
        get a dataset
        """

    @abstractmethod
    def get_training_and_test_loaders(self, dataset: CombinedDataset) -> Tuple[DataLoader, DataLoader]:
        """
        get a dataset
        """


class DatasetService(ABC):
    """
    Abstract class for getting datasets
    """

    def __init__(self, dataset_repository: DatasetRepositoryBase):
        self.dataset_repository = dataset_repository

    def get_dataset(self) -> CombinedDataset:
        """
        get a dataset
        """
        combined_data = self.dataset_repository.load_dataset()
        return combined_data

    def get_training_and_test_loaders(
            self,
            dataset: CombinedDataset,
            batch_size: int = 1,
            validation_split: float = 0.25,
            shuffle_dataset: bool = True,
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Returns training and test data loaders.
        """
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))
        if shuffle_dataset:
            np.random.shuffle(indices)

        train_indices, test_indices = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_indices)
        test_sampler = SubsetRandomSampler(test_indices)

        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)
        return train_loader, test_loader
