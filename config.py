from torch.nn import MSELoss
import torch


class Config:
    def __init__(self):
        self.y = 52
        self.x = 77

        self.pred_len = 1
        # 训练，验证，测试集划分
        self.train_size = 0.6
        self.val_size = 0.2
        self.test_size = 0.2

        self.recent_len = 4
        self.period_len = 3
        self.trend_len = 1
        # 偏移量day, week设置
        self.day = 24
        self.week = 24 * 7

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.epochs = 5
        self.batch_size = 32
        self.num_linear_units = 10
        self.ext_dim = 13
        self.learning_rate = 1e-3
        self.loss_func = MSELoss()
        self.optimizer = torch.optim.Adam
        self.patience = 5
