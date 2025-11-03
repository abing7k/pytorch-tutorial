import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torchvision import transforms

dataset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transforms.ToTensor())
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)



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
    # print(net)
    # input = torch.ones(64, 3, 32 , 32)
    # output = net(input)
    # print(output)
    inputs = torch.tensor([1,2,3],dtype=torch.float32)
    targets = torch.tensor([1,2,5],dtype=torch.float32)
    print(nn.MSELoss()(inputs, targets))
    print(nn.L1Loss()(inputs, targets))


    for data in dataloader:
        images, labels = data
        outputs = net(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        print(loss)
        break
