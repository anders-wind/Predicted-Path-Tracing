"""
Module for the combined dataset
"""

from typing import List
from pathlib import Path
from dataclasses import dataclass
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class CombinedDataPoint():
    """
    A combination of input and output.
    """
    name: str
    image: np.ndarray  # 3d [H,W,channel(r,g,b)]
    render: np.ndarray  # 3d [H,W,channel(r,g,b,depth,completion)]

    def __init__(self, name: str, image: np.ndarray, render: np.ndarray):
        assert len(render.shape) == 3
        assert render.shape[2] == 5
        assert len(image.shape) == 3
        assert image.shape[2] == 3

        self.name = name
        self.render = render
        self.image = image

    def get_render_image(self):
        """
        Returns the image representation of the render (channel has 3 colors)
        """
        return self.render[:, :, :3]

    def show_render(self) -> plt.imshow:
        """
        Show the render
        """
        return plt.imshow(X=self.get_render_image())

    def show_image(self) -> plt.imshow:
        """
        Show the image
        """
        return plt.imshow(X=self.image)


@dataclass
class CombinedDataTensor():
    """
    A combination of input and output.
    """
    name: str
    image: Tensor  # 3d [H,W,channel(r,g,b)]
    render: Tensor  # 3d [H,W,channel(r,g,b,depth,completion)]


@dataclass
class CombinedDataset(Dataset):
    """
    CombinedDataset is a collection of images, renders and names
    """

    dataset_path: Path
    names: List[str]
    images: List[np.ndarray]
    renders: List[np.ndarray]

    def __init__(self,
                 dataset_path: Path,
                 names: List[str],
                 images: List[np.ndarray],
                 renders: List[np.ndarray],
                 transform: transforms = None):
        self.dataset_path = dataset_path
        self.names = names
        self.images = images
        self.renders = renders
        self.transform = transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx: int):
        item = CombinedDataPoint(
            name=self.names[idx],
            image=self.images[idx],
            render=self.renders[idx],
        )

        if self.transform:
            item = self.transform(item)
        return item
