import numpy as np

from tqdm import tqdm
from torch import manual_seed
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.data_processing.cifar_10_helper import read_cifar_10
from global_fun import *

module_logger = module_logging(__file__, True)

import torchvision
import torchvision.transforms as transforms


class AlexNet(torch.nn.Module):
    f"""
    .. image:: {Path(__file__).with_suffix('.png')}

    Test

    :param seed: Random Seed for torch and numpy
    :type seed: int
    :param num_of_classes:
    :type num_of_classes: int
    """

    def __init__(self, seed: int = 42, num_of_classes: int = 10):
        super().__init__()

        np.random.seed(seed)
        manual_seed(seed)

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Linear(256, num_of_classes)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = F.softmax(x, dim=0)
        return x


if __name__ == '__main__':
    cuda = torch.device('cuda')
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                            download=True,
                                            transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                           download=True,
                                           transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    an = AlexNet()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(an.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(tqdm(trainloader, 0)):
            # get the inputs
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = an(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                module_logger.info('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    module_logger.info('Finished Training')

    torch.save(an,
               proj_dir / 'models' / 'alex_net_cifar_10.pth')

    # output = torch.onnx.export(an,
    #                            torch.from_numpy(testset.test_data),
    #                            proj_dir / 'models' / 'alex_net_cifar_10.onnx',
    #                            verbose=False)
