import torch
import torch.nn.functional as F
from torch import nn

input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]])

kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])

input = torch.reshape(input, (1, 1, 5, 5))
kernel = torch.reshape(kernel, (1, 1, 3, 3))


print(F.conv2d(input, kernel))
print(F.conv2d(input, kernel,stride=2))
print(F.conv2d(input, kernel,stride=3,padding=1))

class hanbingnet(nn.Module):
    def __init__(self):
        super(hanbingnet, self).__init__()

    def forward(self, x):
        return  torch.conv2d(x, kernel)

if __name__ == '__main__':
    input = torch.tensor([[1, 2, 0, 3, 1],
                          [0, 1, 2, 3, 1],
                          [1, 2, 1, 0, 0],
                          [5, 2, 3, 1, 1],
                          [2, 1, 0, 1, 1]])

    net = hanbingnet()
    print(net(input.reshape(1, 1, 5, 5)))