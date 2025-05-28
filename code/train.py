import torch
import argparse
from load_mnist import load_mnist
from network import SimpleNN
import functions as funcs

# 设置全局随机种子
funcs.set_global_seed(42) 
# 优化器字典
optimizers = {
    "SGD": torch.optim.SGD, 
    "Adagrad": torch.optim.Adagrad, 
    "RMSProp": torch.optim.RMSprop, 
    "Adam": torch.optim.Adam
}
# 接收命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
parser.add_argument('--lr', type=float, nargs='+', default=[0.01], help='学习率')
parser.add_argument('--optimizer', type=str, default='SGD', choices=optimizers.keys(), help='优化器类型')
parser.add_argument('--batch_size', type=int, default=64, help='批量大小')
parser.add_argument('--train_size', type=int, default=5000, help='训练集大小')
parser.add_argument('--test_size', type=int, default=1000, help='测试集大小')
parser.add_argument('--device', type=str, default='cuda:0', help='训练设备')
parser.add_argument('--save_model', type=bool, default=True, help='是否保存模型')
args = parser.parse_args()

# 配置参数
train_size, test_size, batch_size = args.train_size, args.test_size, args.batch_size
num_epochs, learning_rates, device = args.epochs, args.lr, torch.device(args.device)
save_model = args.save_model
optimizer_name = args.optimizer
train_accs, test_accs, losses = [], [], [] # 保存不同学习率下训练的指标

# 加载数据
train_iter, test_iter = load_mnist(train_size, test_size, batch_size, 42)

if not isinstance(learning_rates, list):
    learning_rates = [learning_rates]

# 训练不同学习率下的模型
for lr in learning_rates:
    model_path1 = f"../model/{optimizer_name}_lr={lr}.pth" # 模型保存路径
    net = SimpleNN() # 定义模型

    # 模型训练
    train_acc, test_acc, loss = funcs.trainNet(
        net, train_iter, test_iter, optimizers[optimizer_name], lr, num_epochs, device
    )
    train_accs.append(train_acc)
    test_accs.append(test_acc)
    losses.append(loss)
    # 模型保存
    if save_model:
        torch.save(net.state_dict(), model_path1)
        
# 画出Train Accuracy、Test Accuracy、Loss在训练过程中的变化
funcs.plot_metrics(
    [[list(range(1, num_epochs + 1))] * len(learning_rates)] * 3,
    [train_accs, test_accs, losses],
    ["Train Accuracy", "Test Accuracy", "Loss"],
    ["Epochs"] * 3,
    ["Train Accuracy", "Test Accuracy", "Loss"],
    x_lim=[(1, num_epochs)] * 3,
    y_lim=[(0, 1), (0, 1), None],
    legends=[[f"lr={lr}" for lr in learning_rates]] * 3,
    save_path=f"../figure/{optimizer_name}_metrics.png"
)

# 写入训练日志中
with open("train_log.txt", "a", encoding="utf-8") as f:
    f.write(f"num_epochs: {num_epochs}\n")
    f.write(f"optimizer: {optimizer_name}\n")
    f.write(f"batch_size: {batch_size}\n")
    f.write(f"train_size: {train_size}\n")
    f.write(f"test_size: {test_size}\n")
    f.write(f"device: {str(device)}\n")
    for k in range(len(learning_rates)):
        f.write(f"learning_rate: {learning_rates[k]}\n")
        f.write(f"ultimate_train_accuracy: {train_accs[k][-1]}\n")
        f.write(f"ultimate_test_accuracy: {test_accs[k][-1]}\n")
        for i in range(num_epochs):
            f.write(f"Epoch: {i + 1}/{num_epochs} Loss: {losses[k][i]:.4f} ")
            f.write(f"Train Acc: {train_accs[k][i]:.2%} Test Acc: {test_accs[k][i]:.2%}\n")
    f.write("\n")