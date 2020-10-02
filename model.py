from torch import nn


class LeNet(nn.Module):
    def __init__(self, output_size):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)  # (32, 158, 98)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2)  # (32, 79, 49)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)  # (64, 78, 48)
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2)  # (64, 39, 24)
        self.fc1 = nn.Linear(59904, 1000)
        self.fc2 = nn.Linear(1000, output_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.batch_norm1(self.conv1(x))
        x = self.activation(x)
        x = self.max_pool1(x)
        x = self.batch_norm2(self.conv2(x))
        x = self.activation(x)
        x = self.max_pool2(x)
        x = x.view(x.size(0), -1)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        x = nn.functional.log_softmax(x)
        return x
