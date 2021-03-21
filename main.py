from utils.resnet import ResNet
from utils.dataTool import train_val_test_loader, plotArea
from utils.trainer import train_test_model, evaluate
from utils.baseline import testLastHourRegression, testArima
from config import Config
import torch

# 全局设置
torch.set_default_tensor_type(torch.DoubleTensor)
config = Config()
# 读入训练集，验证集
train_loader, test_loader = train_val_test_loader(config)
# 定义网络，训练并测试
net = ResNet(config).to(config.device)
train_test_model(net, train_loader, test_loader, config)
# 测试baseline
testLastHourRegression(test_loader, config.device, config.metric, hour_type='last_day') # 前一天
testLastHourRegression(test_loader, config.device, config.metric, hour_type='last_week')  # 前一周
testArima(config)  # arima
