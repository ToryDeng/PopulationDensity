from torch import nn
import torch


class ResNet(nn.Module):
    def __init__(self, config=None):
        super(ResNet, self).__init__()
        self.config = config

        self.recent = ResPath(config.recent_len, config.pred_len)
        self.period = ResPath(config.period_len, config.pred_len)
        self.trend = ResPath(config.trend_len, config.pred_len)

        self.ext = nn.Sequential(
            nn.Linear(in_features=config.ext_dim, out_features=config.num_linear_units),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=config.num_linear_units, out_features=config.y * config.x))
        self.w1 = nn.Parameter(-torch.ones((config.y, config.x), requires_grad=True, device=config.device))
        self.w2 = nn.Parameter(-torch.ones((config.y, config.x), requires_grad=True, device=config.device))
        self.w3 = nn.Parameter(-torch.ones((config.y, config.x), requires_grad=True, device=config.device))

    def forward(self, recent_data, period_data, trend_data, feature_data):
        recent_out = self.recent(recent_data).view(-1, self.config.y, self.config.x)
        period_out = self.period(period_data).view(-1, self.config.y, self.config.x)
        trend_out = self.trend(trend_data).view(-1, self.config.y, self.config.x)
        ext_out = self.ext(feature_data)
        ext_out = ext_out.view(-1, self.config.y, self.config.x)
        main_out = torch.tanh(
            torch.mul(recent_out, self.w1) + torch.mul(period_out, self.w2) + torch.mul(trend_out, self.w3) + ext_out
        )
        return main_out


class ResPath(nn.Module):
    def __init__(self, in_flow, out_flow):
        super(ResPath, self).__init__()
        self.unit = nn.Sequential(
            nn.Conv2d(in_flow, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            ResUnit(),
            ResUnit(),
            ResUnit(),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_flow, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False))

    def forward(self, x):
        return self.unit(x)


class ResUnit(nn.Module):
    def __init__(self, in_flow=64, out_flow=64):
        super(ResUnit, self).__init__()
        self.left = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_flow, out_flow, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        )

    def forward(self, x):
        out = self.left(x)
        out = self.left(out)
        out += x
        return out
