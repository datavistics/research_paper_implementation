import numpy as np

from tqdm import tqdm
from torch import manual_seed
import torch.optim as optim
import torch
import torch.nn.functional as F
from src.data_processing.cifar_10_helper import read_cifar_10
from global_fun import module_logging

module_logger = module_logging(__file__, True)

import torchvision
import torchvision.transforms as transforms


class AlexNet(torch.nn.Module):
    """
    :param seed: Random Seed for torch and numpy
    :type seed: int
    """

    def __init__(self, seed: int = 42):
        super().__init__()

        np.random.seed(seed)
        manual_seed(seed)

        self.conv_1 = torch.nn.Conv2d(11, 11, kernel_size=96, stride=4, padding=0)
        self.pool_1 = torch.nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv_2 = torch.nn.Conv2d(5, 5, kernel_size=256, stride=1, padding=2)
        self.pool_2 = torch.nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv_3 = torch.nn.Conv2d(3, 3, kernel_size=384, stride=1, padding=1)

        self.conv_4 = torch.nn.Conv2d(3, 3, kernel_size=384, stride=1, padding=1)

        self.conv_5 = torch.nn.Conv2d(3, 3, kernel_size=256, stride=1, padding=1)
        self.pool_5 = torch.nn.MaxPool2d(kernel_size=3, stride=2)

        self.flat_1 = torch.nn.Linear(9216, 4096)
        self.flat_2 = torch.nn.Linear(4096, 4096)
        self.flat_3 = torch.nn.Linear(4096, 1000)

        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = F.relu(self.pool_1(x))

        x = F.relu(self.conv_2(x))
        x = F.relu(self.pool_2(x))

        x = F.relu(self.conv_3(x))

        x = F.relu(self.pool_4(x))

        x = F.relu(self.conv_5(x))
        x = F.relu(self.pool_5(x))

        x = x.view(-1, 9216)

        x = self.dropout(F.relu(self.flat_1(x)))
        x = self.dropout(F.relu(self.flat_2(x)))

        x = F.softmax(self.flat_3(x))

        return x


if __name__ == '__main__':

    cuda = torch.device('cuda')
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True,)
                                            # transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True,)
                                           # transform=transform)
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
