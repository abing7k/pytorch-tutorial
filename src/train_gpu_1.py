# -*- coding: utf-8 -*-
# 作者：小土堆（修改版：支持 CUDA / MPS / CPU）
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader

# -----------------------------
# 自动检测设备（CUDA → MPS → CPU）
# -----------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("✅ Using GPU (CUDA):", torch.cuda.get_device_name(0))
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ Using Apple MPS backend (Metal)")
else:
    device = torch.device("cpu")
    print("⚙️ Using CPU")

# -----------------------------
# 准备数据集
# -----------------------------
train_data = torchvision.datasets.CIFAR10(
    root="../data", train=True, transform=torchvision.transforms.ToTensor(), download=True
)
test_data = torchvision.datasets.CIFAR10(
    root="../data", train=False, transform=torchvision.transforms.ToTensor(), download=True
)

train_data_size = len(train_data)
test_data_size = len(test_data)
print(f"训练数据集的长度为：{train_data_size}")
print(f"测试数据集的长度为：{test_data_size}")

train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# -----------------------------
# 定义网络模型
# -----------------------------
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.model(x)

tudui = Tudui().to(device)

# -----------------------------
# 损失函数与优化器
# -----------------------------
loss_fn = nn.CrossEntropyLoss().to(device)
learning_rate = 1e-2
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate)

# -----------------------------
# 训练参数
# -----------------------------
total_train_step = 0
total_test_step = 0
epoch = 10
writer = SummaryWriter("../log/logs_train")

# -----------------------------
# 训练与测试循环
# -----------------------------
for i in range(epoch):
    print(f"-------第 {i + 1} 轮训练开始-------")

    # 训练模式
    tudui.train()
    for imgs, targets in train_dataloader:
        imgs, targets = imgs.to(device), targets.to(device)

        # 前向传播
        outputs = tudui(imgs)
        loss = loss_fn(outputs, targets)

        # 反向传播与优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            print(f"训练次数：{total_train_step}, Loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试模式
    tudui.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for imgs, targets in test_dataloader:
            imgs, targets = imgs.to(device), targets.to(device)
            outputs = tudui(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            total_accuracy += (outputs.argmax(1) == targets).sum()

    avg_loss = total_test_loss / len(test_dataloader)
    accuracy = total_accuracy / test_data_size
    print(f"整体测试集上的Loss: {avg_loss:.4f}")
    print(f"整体测试集上的正确率: {accuracy:.4f}")

    writer.add_scalar("test_loss", avg_loss, total_test_step)
    writer.add_scalar("test_accuracy", accuracy, total_test_step)
    total_test_step += 1

    torch.save(tudui.state_dict(), f"tudui_{i+1}.pth")
    print("✅ 模型已保存")

writer.close()
