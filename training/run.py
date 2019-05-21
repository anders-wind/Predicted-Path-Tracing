"""
The runner module
"""
from torch.utils.data import DataLoader
from torchvision import transforms
from src.dataset_service.dataset_repository import DummyDatasetRepository
from src.dataset_service.dataset_service import DatasetService
from src.dataset_service.transforms import ToTensor
from src.dataset_service.dataset import CombinedDataPoint


def run():
    """
    Main run method
    """
    data_repository = DummyDatasetRepository(1)
    data_service = DatasetService(data_repository)
    dataset = data_service.get_dataset()

    print("original")
    for sample in dataset:
        sample = CombinedDataPoint(**sample)
        print(f"name: {sample.name}, image: {sample.image.shape}, render: {sample.render.shape}")

    # transforms and dataloader
    dataset.set_transform(transforms.Compose([ToTensor()]))
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    print("transformed")
    for sample in dataset:
        print(f"name: {sample['name']}, image: {sample['image'].shape}, render: {sample['render'].shape}")

    print("dataloader")
    for _, sample_batched in enumerate(dataloader):
        name_batch = sample_batched["name"]
        image_batch = sample_batched["image"]
        render_batch = sample_batched["render"]
        for name, image, render in zip(name_batch, image_batch, render_batch):
            print(f"name: {name}, image: {image.shape}, render: {render.shape}")


if __name__ == "__main__":
    run()
