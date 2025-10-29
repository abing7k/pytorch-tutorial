import torchvision
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter


class MaxPool(nn.Module):
    def __init__(self):
        super(MaxPool, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, ceil_mode=True)

    def forward(self, x):
        return self.maxpool(x)


if __name__ == '__main__':
    model = MaxPool()
    dataset = torchvision.datasets.CIFAR10(root='../data', download=True,train=False, transform=torchvision.transforms.ToTensor())
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=64, shuffle=False)
    writer = SummaryWriter("../log/maxpool")

    step = 0
    for data in dataloader:
        images, labels = data
        writer.add_images('input', img_tensor=images, global_step=step)
        writer.add_images('output', img_tensor=model(images), global_step=step)
        step += 1

    writer.close()

