import torch
from torch import nn
import numpy as np
import random
import matplotlib.pyplot as plt

# 优化器字典
optimizers = {
    "SGD": torch.optim.SGD, 
    "Adagrad": torch.optim.Adagrad, 
    "RMSProp": torch.optim.RMSprop, 
    "Adam": torch.optim.Adam
}

def trainNet(net, train_iter, test_iter, optimizer, lr, num_epochs, device):
    """训练神经网络模型"""
    # 初始化权重
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            torch.manual_seed(42)  # 设置随机种子
            nn.init.kaiming_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    net.apply(init_weights)
    net.to(device)
    # 定义优化器和损失函数
    optimizer = optimizer(net.parameters(), lr)
    loss = nn.CrossEntropyLoss(reduction="mean")
    
    print("start training...")
    # 训练准确率、测试准确率、损失值
    train_accuracy_rate, test_accuracy_rate, loss_value = [], [], []
    for epoch in range(num_epochs):
        net.train() # 训练模式
        epoch_loss = 0
        for X, y in train_iter:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                epoch_loss += l.item() * y.size(0) # 累加损失值

        loss_value.append(epoch_loss / len(train_iter.dataset)) # 添加损失值
        train_accuracy_rate.append(accuracy(net, train_iter)) # 添加训练准确率
        test_accuracy_rate.append(accuracy(net, test_iter)) # 添加测试准确率
        
        print(f"Epoch: {epoch + 1}/{num_epochs} Loss: {loss_value[-1]:.4f} ",
              f"Train Acc: {train_accuracy_rate[-1]:.2%} Test Acc: {test_accuracy_rate[-1]:.2%}\n")
        
    return train_accuracy_rate, test_accuracy_rate, loss_value

def accuracy(net, data_iter, device=None):
    """计算模型在数据集上的准确率"""
    if device is None and isinstance(net, nn.Module):
        device = next(iter(net.parameters())).device
    net.eval()
    accu = 0
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            accu += (torch.argmax(net(X), dim=1).to(y.dtype) == y).sum().item()
    
    return accu / len(data_iter.dataset)

def predict(net, X):
    """根据模型net预测"""
    net.eval()
    with torch.no_grad():
        X = X.to(next(iter(net.parameters())).device)
        pred = torch.argmax(net(X), dim=1)
    return pred

def show_image(images, labels, preds=None, num=8, save_path=None):
    """
    可视化图片及预测结果
    """
    images = images.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    if preds is not None:
        preds = preds.cpu().detach().numpy()
    rows = (num + 3) // 4
    plt.figure(figsize=(8, 2 * rows))
    for i in range(num):
        plt.subplot(rows, 4, i + 1)
        plt.imshow(images[i][0], cmap='gray')
        title = f"True:{labels[i]}"
        if preds is not None:
            title += f"\nPred:{preds[i]}"
        plt.title(title)
        plt.axis('off')
    plt.subplots_adjust(hspace=0.5)
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_metrics(X_data, Y_data, titles, X_labels, y_labels, x_lim=None, y_lim=None, legends=None, save_path=None):
    """可视化训练过程中的指标"""
    for i in range(len(X_data)):
        plt.subplot((len(X_data) + 1) // 2, 2, i + 1)
        if isinstance(X_data[i][0], list):
            for j in range(len(X_data[i])):
                plt.plot(X_data[i][j], Y_data[i][j])
            if legends and isinstance(legends[i], list):
                plt.legend(legends[i])
        else:
            plt.plot(X_data[i], Y_data[i])
        plt.title(titles[i])
        plt.xlabel(X_labels[i])
        plt.ylabel(y_labels[i])
        if x_lim and x_lim[i]:
            plt.xlim(x_lim[i])
        if y_lim and y_lim[i]:
            plt.ylim(y_lim[i])
    plt.subplots_adjust(hspace=0.5, wspace=0.3)
    if save_path:
        plt.savefig(save_path)
    plt.show()

def set_global_seed(seed=42):
    # 基础随机性
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # CUDA 相关
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多 GPU 时
    torch.backends.cudnn.deterministic = True  # 确定性算法
    torch.backends.cudnn.benchmark = False     # 禁用自动优化

if __name__ == "__main__":
    from network import SimpleNN
    from load_mnist import load_mnist
    set_global_seed(42)  # 在模型定义前调用
    
    X_data = [[list(range(1 + i, 11 + i)) for i in range(3)]]
    Y_data = [[list(range(1, 11)) for _ in range(3)]]
    legends = [["Train Accuracy", "Test Accuracy", "Loss"]]
    titles = ["Train Accuracy", "Test Accuracy", "Loss"]
    X_labels = ["Epochs", "Epochs", "Epochs"]
    y_labels = ["Train Accuracy", "Test Accuracy", "Loss"]
    plot_metrics(
        X_data, Y_data, titles, X_labels, y_labels, legends=legends,
    )
    