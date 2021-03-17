from utils.resnet import ResNet
from utils.dataTool import train_val_test_loader, plotTrainValLoss
from utils.trainer import train_test_model
from config import Config
import torch

# 全局设置
torch.set_default_tensor_type(torch.DoubleTensor)
config = Config()
# 读入训练集，验证集，测试集
train_loader, val_loader, test_loader = train_val_test_loader(config)
# 定义网络，训练并测试
net = ResNet(config).to(config.device)
train_test_model(net, train_loader, val_loader, test_loader, config)
plotTrainValLoss()





# recent = torch.randn(size=(32, config.recent_len, 52, 77))
# period = torch.randn(size=(32, config.period_len, 52, 77))
# trend = torch.randn(size=(32, config.trend_len, 52, 77))
# meta = torch.randn(size=(32, config.ext_dim,))
# print(net(recent, period, trend, meta).shape)
