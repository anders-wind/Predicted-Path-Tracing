"""
Service for training and testing models
"""
import sys
from dataclasses import dataclass
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from src.ml_models.conv_model import FirstNet, SimpleNet, PyramidNet


@dataclass
class TestResults():
    """
    Test results
    """
    total_loss: float


class TrainingService():
    """
    A training service which allows for training and testing of models
    """

    def __init__(self, epochs: int = 100, number_of_nets: int = 1):
        self.epochs = epochs
        self.number_of_nets = number_of_nets

    def train(self, train_loader: DataLoader, test_loader: DataLoader):
        """
        Train a NN on the data
        """
        best_net = None
        best_loss = sys.float_info.max

        for _ in range(self.number_of_nets):
            # net = FirstNet()
            net = SimpleNet()
            # net = PyramidNet()
            self._optimize_net(net, train_loader)
            test_results = self._evaluate_net(net, test_loader)
            if test_results.total_loss < best_loss:
                best_net = net
                best_loss = test_results.total_loss
                print(f"New best network {best_loss}")

        print(f"Best loss: {best_loss}")
        return best_net

    def _optimize_net(self, net: nn.Module, train_loader: DataLoader):
        print('Started Training')
        net.train()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(net.parameters(), lr=1e-4)
        run_loss_iterations = 1

        for epoch in range(self.epochs):
            running_loss = 0.0
            for i, data in enumerate(train_loader):
                inputs, outputs = data["render"], data["image"]
                optimizer.zero_grad()
                y_hat = net.forward(inputs)
                loss = criterion(y_hat, outputs)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if (i + 1) % (run_loss_iterations) == 0:
                    print('[%d, %5d] loss: %.3f' % (epoch, i, running_loss / run_loss_iterations))
                    running_loss = 0.0

            print('Finished Epoch: ', epoch)

        print('Finished Training')

    def _evaluate_net(self, net: nn.Module, test_loader: DataLoader) -> TestResults:
        print('Started Evaluating')
        net.eval()
        criterion = nn.MSELoss()
        running_loss = 0.0
        with torch.no_grad():
            for data in test_loader:
                inputs, outputs = data["render"], data["image"]
                print(outputs[0])
                y_hat = net.forward(inputs)
                loss = criterion(y_hat, outputs)

                running_loss += loss.item()
        print('Finished Evaluating')
        return TestResults(running_loss)
