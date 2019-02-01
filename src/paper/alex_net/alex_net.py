import torch
import numpy as np

import torch.nn.Functional as F


class AlexNet(torch.nn.module):

    def __init__(self, seed: int=42):
        super(AlexNet, self).__init()

        np.random.seed(seed)
        torch.manual_seed(seed)

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
        self.flat_2 = torch.nn.Linear(4096, 1000)

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

        x = F.relu(self.flat_1(x))
        x = F.relu(self.flat_2(x))

        x = F.softmax(self.flat_3(x))

        return x
