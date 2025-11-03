import torch
import torchvision
import time
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torchvision import transforms

# ✅ 设备自动检测
device = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)
print(f"🚀 使用设备: {device}")

# ✅ 数据预处理（归一化）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 映射到 [-1,1]
])

# ✅ 加载 CIFAR10 数据集
dataset = torchvision.datasets.CIFAR10(
    root='../data', train=True, download=True, transform=transform
)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

# ✅ 定义网络结构
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = Sequential(
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
    net = Net().to(device)  # ✅ 模型放到设备上
    net.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    num_epochs = 100
    print("📘 开始训练...\n")

    for epoch in range(num_epochs):
        start_time = time.time()  # ✅ 记录开始时间
        running_loss = 0.0

        for images, labels in dataloader:
            # ✅ 数据放到设备上
            images, labels = images.to(device), labels.to(device)

            outputs = net(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        end_time = time.time()  # ✅ 记录结束时间
        epoch_time = end_time - start_time

        print(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {running_loss:.3f} | Time: {epoch_time:.2f}s")

    print("\n✅ 训练完成！")
