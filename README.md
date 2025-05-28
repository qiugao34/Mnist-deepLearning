### 环境配置
```
Python>=3.9.19
matplotlib==3.10.3
numpy==2.2.6
torch==2.0.1+cu118
torchvision==0.15.2
```

### 运行说明
#### 训练
首先进入`code`目录下
```
cd code
```
直接使用命令行运行`train.py`训练模型，例如：
```
python train.py --optimizer SGD --lr 0.1 0.01 0.001
```
其中命令行参数有
```
--epochs default=10 # 训练轮数
--lr default=[0.01] # 学习率，可输入多个学习率进行对比
--optimizer default='SGD' # 优化器名称（可选：SGD、Adagrad、RMSProp、Adam）
--batch_size default=64 # 批量大小
--train_size default=5000 # 训练集大小
--test_size default=1000 # 测试集大小
--device default='cuda:0' # 训练设备
--save_model default=True # 是否保存模型
```
最后训练产生的模型保存在`../model/`目录下，名称为`{optimizer_name}_lr={lr}.pth`，训练日志保存在`train_log.txt`文件中，训练过程中的指标变化图保存在`../figure/`目录下，名称为`{optimizer_name}_metrics.png`，指标有训练准确率、测试准确率、训练过程损失值。

#### 预测或者测试
使用测试集中的`batch_size=64`大小的一个批次中的数据进行预测，并将预测结果可视化
基本命令如下：
```
python test.py --test_size 5000
```
其中命令行参数有
```
--model_name default='SGD' # 模型名称
--test_size default=1000 # 测试集大小
--show_num default=16 # 可视化预测结果与原始图片数量，需小于64
--device default='cuda:0' # 测试时使用的设备
--is_save default=False # 是否保存预测可视化结果
```
如果选择保存预测可视化结果，则会在`../figure/`目录下生成一个名为`{optimizer_name}_sample.png`的文件。

### 其他
`code/net_work.py`文件定义卷积神经网络\
`code/load_mnist.py`文件定义加载MNIST数据集的函数\
`code/functions.py`定义了训练和测试过程使用的函数\
`code/train_log.txt`文件记录了训练过程中的日志信息\
`data`目录下存放MNIST数据集\
`model`目录下存放训练好的模型\
`figure`目录下存放训练过程中的指标变化图和预测结果可视化图