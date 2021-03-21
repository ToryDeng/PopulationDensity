from torch.utils.data import Dataset
import numpy as np
import torch


def toTensor(lst):
    return torch.tensor(np.array(lst))


class MyDataset(Dataset):
    def __init__(self, config, data_type='train'):
        self.flow = np.load('data/grid_graph_flow.npy')[8 * 24:, :, :]
        self.meta = np.load('data/meta_data.npy')[8 * 24:, :]
        self.strength = np.load('data/norm_strength_flow.npy')
        self.min = self.flow.min()
        self.max = self.flow.max()
        self.mean = self.flow.mean()
        self.std = self.flow.std()
        self.flow = (self.flow - self.min) / (self.max - self.min)  # min-max

        # self.flow = (self.flow - self.mean) / self.std
        self.recent_len = config.recent_len
        self.period_len = config.period_len
        self.trend_len = config.trend_len
        self.day = config.day
        self.week = config.week

        self.data_type = data_type
        self.sample_num = self.flow.shape[0] - self.week * self.trend_len
        self.train_len = int(self.sample_num * config.train_size)
        self.val_len = self.sample_num - self.train_len
        # self.test_len = self.sample_num - self.train_len - self.val_len

    def __len__(self):
        if self.data_type == 'train':
            return self.train_len
        elif self.data_type == 'val':
            return self.val_len
        # elif self.data_type == 'test':
        #     return self.test_len
        else:
            print('ERROR!')
            return None

    def __getitem__(self, index):
        if self.data_type == 'train':
            y_idx = self.week * self.trend_len + index
        elif self.data_type == 'val':
            y_idx = self.week * self.trend_len + self.train_len + index
        # elif self.data_type == 'test':
        #     y_idx = self.week * self.trend_len + self.train_len + self.val_len + index
        else:
            print('ERROR!')
            return None
        recent_list, period_list, trend_list = [], [], []
        y = torch.from_numpy(self.flow[y_idx])
        y_meta = torch.from_numpy(self.meta[y_idx])
        y_strength = torch.from_numpy((self.strength[y_idx % 24]))
        for i in range(self.recent_len):
            recent_list.append(self.flow[y_idx - (i + 1)])
        for j in range(self.period_len):
            period_list.append(self.flow[y_idx - (j + 1) * self.day])
        for k in range(self.trend_len):
            trend_list.append(self.flow[y_idx - (k + 1) * self.week])
        x = [toTensor(recent_list), toTensor(period_list), toTensor(trend_list),
             toTensor(y_meta), toTensor(y_strength)]
        return x, y
