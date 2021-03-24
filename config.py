from torch.nn import MSELoss
from torch.nn.init import sparse_, orthogonal_, kaiming_normal_

import torch


class Config:
    def __init__(self):
        self.y = 52  # 纬度
        self.x = 77  # 经度

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pred_len = 1
        # 训练，验证，测试集划分
        self.train_size = 0.8
        self.test_size = 0.2
        # 邻近性因子，周期性因子，趋势性因子长度
        self.recent_len = 3
        self.period_len = 1
        self.trend_len = 1
        # 偏移量day, week设置
        self.day = 24
        self.week = 24 * 7
        # 神经网络训练相关超参
        self.epochs = 1
        self.batch_size = 32
        self.num_linear_units = 10
        self.init_method = kaiming_normal_
        self.ext_dim = 13
        self.learning_rate = 5e-3
        self.metric = MSELoss(reduction='sum')
        self.optimizer = torch.optim.Adam
        self.patience = 15
        # baseline
        self.sample_rate = 0.25  # arima
