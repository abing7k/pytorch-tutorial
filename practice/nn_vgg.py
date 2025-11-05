import torchvision
from torch import nn

model1 = torchvision.models.vgg16(pretrained=True)

model2 = torchvision.models.vgg16(pretrained=True)

model3 = model1.classifier[6] = nn.Linear(4096, 10)
model4 = model2.add_module("Linear", nn.Linear(1000, 10))

print(model1)
print(model2)
print(model3)
print(model4)