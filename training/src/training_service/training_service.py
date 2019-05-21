"""
Service for training and testing models
"""
import torch.optim as optim
import torch.nn as nn
from src.ml_models.conv_model import FirstNet


class TrainingService():
    """
    A training service which allows for training and testing of models
    """

    def __init__(self, epochs: int):
        self.epochs = epochs

    def train(self, train_loader):
        """
        Train a NN on the data
        """
        net = FirstNet()
        criterion = nn.MSELoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

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
                if i % 10 == 0:
                    print('[%d, %5d] loss: %.3f' % (epoch, i, running_loss / 10))
                    running_loss = 0.0

            print('Finished Epoch: ', epoch)

        print('Finished Training')

        return net
