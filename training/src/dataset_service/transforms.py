"""
Transforms for the datasets
"""
import torch
from .dataset import CombinedDataPoint, CombinedDataTensor


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample: CombinedDataPoint) -> CombinedDataTensor:
        image = sample.image
        render = sample.render
        name = sample.name

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return CombinedDataTensor(
            name=name,
            image=torch.from_numpy(image),
            render=torch.from_numpy(render),
        )
