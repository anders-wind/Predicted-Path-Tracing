"""
The runner module
"""
from src.dataset_service.dataset_repository import DummyDatasetRepository
from src.dataset_service.dataset_service import DatasetService


def run():
    """
    Main run method
    """
    data_repository = DummyDatasetRepository(4)
    data_service = DatasetService(data_repository)
    dataset = data_service.get_dataset()

    for datapoint in dataset:
        print(datapoint.image)
        print(datapoint.render)


if __name__ == "__main__":
    run()
