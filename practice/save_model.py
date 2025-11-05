import torchvision
import torch
from torch import nn

model = torchvision.models.vgg16(pretrained=False)

torch.save(model,"vgg16.pth")

torch.save(model.state_dict(),"vgg16-2.pth")

model2 = torch.load('vgg16.pth')
# print(model2)

print(torch.load('vgg16-2.pth'))