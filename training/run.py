"""
The runner module
"""
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from src.dataset_service.dataset_repository import DummyDatasetRepository
from src.dataset_service.dataset_service import DatasetService
from src.dataset_service.transforms import ToTensor, Transposer
# from src.dataset_service.dataset import CombinedDataPoint
from src.training_service.training_service import TrainingService


def print_shapes(dataset, dataloader):
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


def run():
    """
    Main run method
    """
    data_repository = DummyDatasetRepository(32000)
    data_service = DatasetService(data_repository)
    dataset = data_service.get_dataset()

    # print("original")
    # for sample in dataset:
    #     sample = CombinedDataPoint(**sample)
    #     print(f"name: {sample.name}, image: {sample.image.shape}, render: {sample.render.shape}")

    # transforms and dataloader
    dataset.set_transform(transforms.Compose([Transposer(), ToTensor()]))
    train_loader, test_loader = data_service.get_training_and_test_loaders(dataset)

    # print_shapes(dataset, dataloader)

    training_service = TrainingService(epochs=2)
    net = training_service.train(train_loader, test_loader)

    predicted = net.forward_single(dataset[0]["render"])
    figure, axarr = plt.subplots(3)
    axarr[0].imshow(dataset[0]["image"].detach().cpu().numpy().transpose((1, 2, 0)))
    axarr[1].imshow(dataset[0]["render"].detach().cpu().numpy().transpose((1, 2, 0))[:, :, :3])
    axarr[2].imshow(predicted.detach().cpu().numpy().transpose((1, 2, 0)))
    figure.show()


if __name__ == "__main__":
    run()
