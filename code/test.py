import torch
import argparse
from network import SimpleNN
import functions as funcs
from load_mnist import load_data

"""
根据导入的模型进行进行测试与预测，打印出模型在测试集上的准确率
预测图片数量为的show_num大小(show_num < 64)
最后可视化预测结果
"""

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='SGD', help='导入模型名称')
parser.add_argument('--test_size', type=int, default=1000, help='测试集大小')
parser.add_argument('--show_num', type=int, default=16, help='可视化图片数量')
parser.add_argument('--device', type=str, default='cuda:0', help='训练设备')
parser.add_argument('--is_save', type=bool, default=False, help='是否保存预测结果')
args = parser.parse_args()

load_model_path = f"../model/{args.model_name}.pth" # 导入模型路径
if args.is_save:
    save_image_path = f"../figure/{args.model_name}_sample.png" # 保存预测结果路径
else:
    save_image_path = None
    
# 用于预测的数据
test_iter = load_data(args.test_size, 64, train=False)

# 导入模型
net = SimpleNN()
net.load_state_dict(torch.load(load_model_path))
net.to(args.device)

test_acc = funcs.accuracy(net, test_iter, device=args.device)  # 测试模型准确率
print(f"Model {args.model_name} accuracy: {test_acc} and test size: {args.test_size}")

# 随机选择show_num张图片进行预测
num = args.show_num
images, labels = next(iter(test_iter))
idx = torch.randperm(images.size(0))[:num]
images, labels = images[idx], labels[idx] 
preds1 = funcs.predict(net, images)

funcs.show_image(images, labels, preds1, num, save_path=save_image_path)