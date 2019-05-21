"""
Transforms for the datasets
"""
from typing import Dict
import torch


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample: Dict) -> Dict:
        image = sample["image"]
        render = sample["render"]
        name = sample["name"]

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # Apparently?

        image = image.transpose((2, 0, 1))
        render = render.transpose((2, 0, 1))
        return dict(
            name=name,
            image=torch.from_numpy(image),
            render=torch.from_numpy(render),
        )
