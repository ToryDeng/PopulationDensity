from torch import nn
import torch
import torch.nn.functional as F


class ResNet(nn.Module):
    def __init__(self, config=None):
        super(ResNet, self).__init__()
        self.config = config

        self.recent = ResPath(config.recent_len, config.pred_len)
        self.period = ResPath(config.period_len, config.pred_len)
        self.trend = ResPath(config.trend_len, config.pred_len)

        self.y = config.y
        self.x = config.x
        self.device = config.device

        self.ext = nn.Sequential(
            nn.Linear(in_features=config.ext_dim, out_features=config.num_linear_units),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=config.num_linear_units, out_features=config.y * config.x))
        self.w1 = nn.Parameter(self._weightInit(config.init_method))
        self.w2 = nn.Parameter(self._weightInit(config.init_method))
        self.w3 = nn.Parameter(self._weightInit(config.init_method))
        self.w4 = nn.Parameter(self._weightInit(config.init_method))

    def forward(self, X):
        recent_data, period_data, trend_data, feature_data, strength_data = \
            X[0].to(self.device), X[1].to(self.device), X[2].to(self.device), X[3].to(self.device), X[4].to(self.device)
        recent_out = self.recent(recent_data).view(-1, self.y, self.x)
        period_out = self.period(period_data).view(-1, self.y, self.x)
        trend_out = self.trend(trend_data).view(-1, self.y, self.x)
        ext_out = self.ext(feature_data)
        ext_out = ext_out.view(-1, self.y, self.x)
        main_out = F.softsign(
            torch.mul(recent_out, self.w1) + torch.mul(period_out, self.w2) +
            torch.mul(trend_out, self.w3) + torch.mul(strength_data, self.w4) + ext_out
        )
        return main_out

    def _weightInit(self, method):
        return method(torch.Tensor(self.y, self.x))


class ResPath(nn.Module):
    def __init__(self, in_flow, out_flow):
        super(ResPath, self).__init__()
        self.unit = nn.Sequential(
            nn.Conv2d(in_flow, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True),
            ResUnit(), ResUnit(), ResUnit(), ResUnit(),

            nn.ReLU(),
            nn.Conv2d(64, out_flow, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True),
        )

    def forward(self, x):
        return self.unit(x)


class ResUnit(nn.Module):
    def __init__(self, in_flow=64, out_flow=64):
        super(ResUnit, self).__init__()
        self.left = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_flow, out_flow, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True),
        )

    def forward(self, x):
        out = self.left(x)
        out = self.left(out)
        out += x
        return out
