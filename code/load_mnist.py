import numpy as np
import random
import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader

def seed_worker(worker_id):
    worker_seed = 42  # 固定工作进程的随机种子
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def load_data(data_size, batch_size, train=True):
    """加载单个MNIST数据集"""
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST(root="../data", train=train, transform=transform, download=True)
    data_indices = torch.randperm(len(dataset))[: data_size]
    
    data_subset = Subset(dataset, data_indices)
    return DataLoader(data_subset, batch_size=batch_size, shuffle=True, num_workers=0, worker_init_fn=seed_worker)

def load_mnist(train_size, test_size, batch_size, seed):
    """加载MNIST数据"""
    torch.manual_seed(seed)
    return (
        load_data(train_size, batch_size, train=True),
        load_data(test_size, batch_size, train=False)
    )
    
if __name__ == "__main__":
    # 第一次运行
    # set_seed(42)

    _, test_loader1 = load_mnist(5000, 1000, 64, 42)
    first_order = [batch[1].tolist() for batch in test_loader1]  # 获取所有标签顺序

    # 第二次运行（重置 DataLoader）
    # set_seed(42)
    _, test_loader2 = load_mnist(5000, 1000, 64, 42)
    second_order = [batch[1].tolist() for batch in test_loader2]

    print("顺序是否一致:", first_order == second_order)  # 应输出 True