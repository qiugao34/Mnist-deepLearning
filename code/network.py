import torch
from torch import nn

class SimpleNN(nn.Module):
    """简单的神经网络模型实现MNIST手写数字识别
       该模型包含：
         - 输入层：28x28灰度图
         - 卷积层1：Conv2d(1, 32, 3x3) -> BatchNorm2d(32) -> ReLU -> MaxPool2d(2)
         - 卷积层2：Conv2d(32, 64, 3x3) -> BatchNorm2d(64) -> ReLU -> MaxPool2d(2)
         - 全连接层1：Linear(64x5x5, 128) -> ReLU
         - 全连接层2：Linear(128, 10) -> Softmax
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.Linear1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU()
        )
        self.Linear2 = nn.Linear(128, 10)
        
    def forward(self, X):
        X = self.conv1(X)
        X = self.conv2(X)
        X = self.Linear1(X)
        X = self.Linear2(X)
        return X