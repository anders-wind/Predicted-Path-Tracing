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

    def _open_and_read(self, path: Path) -> str:
        with open(path, "r") as file:
            return file.read()

    def load_dataset(self) -> CombinedDataset:
        """
        Loads the dataset
        """
        datapoint_paths = Path(self.datastore_root).glob('*.dp')
        datastore_root_string = str(self.datastore_root) + "/"
        images = []
        renders = []
        names = []
        for path in datapoint_paths:
            file_names = self._open_and_read(path).split(", ")
            target_file_name = Path(datastore_root_string + file_names[0])
            render_file_names = [Path(datastore_root_string + x) for x in file_names[1:]]

            target = pd.read_csv(target_file_name, dtype={'x': np.float32, 'y': np.float32, 'z': np.float32}).values
            target = np.asarray(target, dtype=np.float32)
            target = target.reshape(self.height, self.width, 3)
            for render_file in render_file_names:
                render = pd.read_csv(
                    render_file, dtype={
                        'x': np.float32,
                        'y': np.float32,
                        'z': np.float32,
                        'v': np.float32,
                        'w': np.float32
                    }).values
                render = np.asarray(render, dtype=np.float32)
                render = render.reshape(self.height, self.width, 5)
                renders.append(render)
                images.append(target)
                names.append(str(render_file))

        dataset = CombinedDataset(dataset_path=Path(), names=names, images=images, renders=renders)
        return dataset


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


# repo = DatasetRepository(Path("/home/anders/Documents/datasets/ppt/640x360_run02"), 640, 380)
