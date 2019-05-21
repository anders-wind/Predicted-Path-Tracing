"""
Module for DataRepositories which can get datasets
"""
from abc import ABC, abstractmethod
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
    def get_data_loader(self) -> CombinedDataset:
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
