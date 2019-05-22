"""
The runner module
"""
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from src.dataset_service.dataset_repository import DummyDatasetRepository
from src.dataset_service.dataset_service import DatasetService
from src.dataset_service.transforms import ToTensor, Transposer
from src.training_service.training_service import TrainingService


def print_dataset_shapes(dataset):
    """
    Prints the shapes of the elements in dataset
    """
    print("transformed")
    for sample in dataset:
        print(f"name: {sample['name']}, image: {sample['image'].shape}, render: {sample['render'].shape}")


def print_loader_shapes(dataloader):
    """
    Prints the shape of the elements in the dataloader
    """
    print("dataloader")
    for _, sample_batched in enumerate(dataloader):
        name_batch = sample_batched["name"]
        image_batch = sample_batched["image"]
        render_batch = sample_batched["render"]
        for name, image, render in zip(name_batch, image_batch, render_batch):
            print(f"name: {name}, image: {image.shape}, render: {render.shape}")


def show_images(net, input_render, target_image):
    """
    Plots the input_render, the target_image, and the prediction image
    """
    predicted = net.forward_single(input_render)
    figure, axarr = plt.subplots(ncols=3)
    axarr[0].imshow(input_render.detach().cpu().numpy().transpose((1, 2, 0))[:, :, :3], label="input")
    axarr[0].title.set_text('input')
    axarr[1].imshow(target_image.detach().cpu().numpy().transpose((1, 2, 0)), label="target")
    axarr[1].title.set_text('target')
    axarr[2].imshow(predicted.detach().cpu().numpy().transpose((1, 2, 0)), label="prediction")
    axarr[2].title.set_text('prediction')
    figure.show()


def run():
    """
    Main run method
    """
    np.random.seed(42)

    # ========= Create Services
    data_repository = DummyDatasetRepository(32000)
    data_service = DatasetService(data_repository)
    training_service = TrainingService(epochs=2)

    # ========= Create data
    dataset = data_service.get_dataset()
    # print_dataset_shapes(dataset)
    dataset.set_transform(transforms.Compose([Transposer(), ToTensor()]))
    # print_dataset_shapes(dataset)
    train_loader, test_loader = data_service.get_training_and_test_loaders(dataset)
    # print_loader_shapes(train_loader)

    # ========= Train
    net = training_service.train(train_loader, test_loader)

    # ========= Show Results
    show_images(net, input_render=dataset[0]["render"], target_image=dataset[0]["image"])


if __name__ == "__main__":
    run()
