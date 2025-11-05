import torch
import torchvision.datasets
from torch import nn, optim
from torchvision import transforms

transforms = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

datasets_train = torchvision.datasets.CIFAR10("../data", download=True, transform=transforms, train=True)
dataloader_train = torch.utils.data.DataLoader(datasets_train, batch_size=64, shuffle=True)
datasets_test = torchvision.datasets.CIFAR10("../data", download=True, transform=transforms, train=False)
dataloader_test = torch.utils.data.DataLoader(datasets_test, batch_size=64, shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.model(x)


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def train(epoch: int, lr: float):
    global loss
    total_train_step = 0

    net = Net().to(get_device())

    optimizer = optim.Adam(net.parameters(), lr)
    loss_fn = nn.CrossEntropyLoss().to(get_device())
    for i in range(epoch):
        print('Epoch: %d' % i)
        net.train()
        for data in dataloader_train:
            images, labels = data
            images, labels = images.to(get_device()), labels.to(get_device())
            output = net(images)
            loss = loss_fn(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_step += 1
            if total_train_step % 100 == 0:
                print('Train loss: %.3f' % loss.item(), "total_train_step: %d" % total_train_step)

        # test
        net.eval()

        total_test_loss = 0
        total_correct = 0
        total_num = len(datasets_test)

        with torch.no_grad():
            for data in dataloader_test:
                images, labels = data
                images, labels = images.to(get_device()), labels.to(get_device())
                output = net(images)
                loss = loss_fn(output, labels)
                total_test_loss += loss.item()
                total_correct += (output.argmax(1) == labels).sum().item()

        # 平均损失：用总损失除以 batch 数
        avg_test_loss = total_test_loss / len(dataloader_test)

        # 准确率：用正确样本数除以测试集样本总数
        accuracy = total_correct / total_num * 100

        print(f'Test loss: {avg_test_loss:.3f}')
        print(f'Test accuracy: {accuracy:.2f}%')


if __name__ == '__main__':
    print(get_device())
    train(20, 0.001)
