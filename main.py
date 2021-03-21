from utils.resnet import ResNet
from utils.dataTool import train_val_test_loader, plotArea
from utils.trainer import train_test_model, evaluate
from utils.baseline import testLastHourRegression, testArima
from config import Config
import torch
import numpy as np

# 全局设置
torch.set_default_tensor_type(torch.DoubleTensor)
config = Config()
# 读入训练集，验证集，测试集
train_loader, val_loader = train_val_test_loader(config)
# 定义网络，训练并测试
net = ResNet(config).to(config.device)
train_test_model(net, train_loader, val_loader, config)
# 测试baseline
testLastHourRegression(val_loader, config.device, config.metric, hour_type='last_day')
testLastHourRegression(val_loader, config.device, config.metric, hour_type='last_week')
