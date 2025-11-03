import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model=Sequential(
            Conv2d(3, 32, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(32, 32, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(32, 64, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10),
        )

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    net = Net()
    print(net)
    input = torch.ones(64, 3, 32 , 32)
    output = net(input)
    print(output)
