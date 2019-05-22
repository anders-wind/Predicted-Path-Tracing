"""
Transforms for the datasets
"""
from typing import Dict
import torch


class Transposer():
    """
    Transposes the ndarrays
    from:   H x W x C
    to:     C X H X W
    """

    def __call__(self, sample: Dict) -> Dict:
        name = sample["name"]
        image = sample["image"]
        render = sample["render"]

        return dict(
            name=name,
            image=image.transpose((2, 0, 1)),
            render=render.transpose((2, 0, 1)),
        )


class ToTensor():
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample: Dict) -> Dict:
        image = sample["image"]
        render = sample["render"]
        name = sample["name"]

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # Apparently?

        return dict(
            name=name,
            image=torch.from_numpy(image).float().to(0),
            render=torch.from_numpy(render).float().to(0),
        )
