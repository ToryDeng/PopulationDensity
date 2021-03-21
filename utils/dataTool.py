import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from utils.resnet import ResNet
from utils.myDataset import MyDataset
from torch.utils.data import DataLoader
from tqdm import tqdm


def convertGridStrength():
    grid_strength = pd.read_csv('data/datafountain_competition_od.txt', delimiter='\t', header=None)
    grid_strength.columns = ['hour', 'start_grid_x', 'start_grid_y', 'end_grid_x', 'end_grid_y', 'index']
    for col in ['start_grid_x', 'start_grid_y', 'end_grid_x', 'end_grid_y']:
        grid_strength[col] = np.around(grid_strength[col].values, 2)
    x = np.sort(np.unique(grid_strength['start_grid_x']))
    y = np.sort(np.unique(grid_strength['start_grid_y']))
    grid_strength_flow = np.zeros((2, 24, x.shape[0], y.shape[0]))
    for i in tqdm(range(grid_strength.shape[0])):
        start_idx = np.concatenate([np.argwhere(np.isin(x, grid_strength.iloc[i, 1]))[0], np.argwhere(np.isin(y, grid_strength.iloc[i, 2]))[0]])
        end_idx = np.concatenate([np.argwhere(np.isin(x, grid_strength.iloc[i, 3]))[0], np.argwhere(np.isin(y, grid_strength.iloc[i, 4]))[0]])
        # start离开-end到达
        grid_strength_flow[0, grid_strength.iloc[i, 0], start_idx[0], start_idx[1]] += grid_strength.iloc[i, 5]
        grid_strength_flow[1, grid_strength.iloc[i, 0], end_idx[0], end_idx[1]] += grid_strength.iloc[i, 5]
    np.save('data/grid_strength_flow.npy', grid_strength_flow[0] - grid_strength_flow[1])


def loadGridStrength():
    strength = np.load('data/grid_strength_flow.npy').swapaxes(1, 2)
    strength = (strength - strength.min()) / (strength.max() - strength.min())
    np.save('data/norm_strength_flow.npy', strength)

def loadMigrationIndex():
    migra_data = pd.read_csv('data/migration_index.csv', header=None)
    migra_data.columns = ['日期', '出发省份', '出发城市', '到达省份', '到达城市', '迁徙指数']
    from_beijing_data = migra_data[migra_data['出发省份'] == '北京市']
    to_beijing_data = migra_data[migra_data['到达省份'] == '北京市']
    sum_by_day = from_beijing_data.groupby(by='日期')['迁徙指数'].sum() - to_beijing_data.groupby(by='日期')['迁徙指数'].sum()
    norm_sum = (sum_by_day - sum_by_day.min()) / (sum_by_day.max() - sum_by_day.min())
    migration_feature = np.zeros(shape=(norm_sum.shape[0] * 24, 1))
    for i in range(norm_sum.shape[0]):
        migration_feature[i * 24:(i + 1) * 24, 0] = norm_sum.iloc[i]
    return migration_feature


def loadTimeFeature():
    time_data = pd.read_csv('data/isHoliday.csv', encoding='gbk')
    time_data['日期'] = pd.to_datetime(time_data['日期'])
    time_data['周几'] = time_data['日期'].map(lambda x: x.weekday())
    time_data['是否工作日'] = time_data['周几'].map(lambda x: 1 if x not in [6, 7] else 0)

    time_feature = np.zeros(shape=(time_data.shape[0] * 24, 9))
    for i in range(time_data.shape[0]):
        time_feature[i * 24:(i + 1) * 24, time_data.iloc[i, 2]] = 1
        time_feature[i * 24:(i + 1) * 24, 7] = time_data.iloc[i, 3]
        time_feature[i * 24:(i + 1) * 24, 8] = time_data.iloc[i, 1]
    return time_feature


def loadWeatherFeature():
    weather_data = pd.read_csv('data/weather.csv', encoding='gbk')
    weather_data['日期'] = pd.to_datetime(weather_data['日期'])
    weather_feature = np.zeros(shape=(weather_data.shape[0] * 24, 3))
    for i in range(weather_data.shape[0]):
        weather_feature[i * 24:(i + 1) * 24, :] = weather_data.iloc[i, 1:].values
    return weather_feature


def integrateMetaData():
    time_feature, weather_feature, migration_feature = loadTimeFeature(), loadWeatherFeature(), loadMigrationIndex()
    integrated_data = np.concatenate([time_feature, weather_feature, migration_feature], axis=1)
    np.save('data/meta_data.npy', integrated_data)
    print('数据保存完成！')


def convertDataToFlow():
    """
    原经纬度数据保留两位小数，转换成ndarray

    :return: None
    """
    jan_data = pd.read_csv('data/shortstay_20200117_20200131.csv', delimiter='\t', header=None,
                           dtype={0: np.int32, 1: np.int32, 2: np.float32, 3: np.float32, 4: np.float32})
    feb_data = pd.read_csv('data/shortstay_20200201_20200215.csv', delimiter='\t', header=None,
                           dtype={0: np.int32, 1: np.int32, 2: np.float32, 3: np.float32, 4: np.float32})
    stay_data = pd.concat([jan_data, feb_data]).reset_index(drop=True)
    stay_data.columns = ['日期', '小时', '经度', '纬度', '人流量指数']
    print("经度范围：东经{}°-东经{}°\n纬度范围：北纬{}°-北纬{}°".format(
        stay_data['经度'].min(), stay_data['经度'].max(), stay_data['纬度'].min(), stay_data['纬度'].max())
    )
    stay_data['经度'] = np.around(stay_data['经度'].values, 2)
    stay_data['纬度'] = np.around(stay_data['纬度'].values, 2)

    x = np.sort(np.unique(stay_data['经度'].values))
    y = np.sort(np.unique(stay_data['纬度'].values))
    # 20200117-20200215 30 days, 0-23 hours
    grid_graph_flow = np.zeros(shape=(30 * 24, y.shape[0], x.shape[0]), dtype=np.float64)
    dates, hours = np.sort(stay_data['日期'].unique()), np.arange(0, 24)
    print("纬度(y)网格数：{}， 经度(x)网格数：{}".format(y.shape[0], x.shape[0]))
    for i in tqdm(range(stay_data.shape[0])):
        time_idx = np.argwhere(np.isin(dates, stay_data.loc[i, '日期'])) * 24 + np.argwhere(
            np.isin(hours, stay_data.loc[i, '小时']))
        x_idx, y_idx = np.argwhere(np.isin(x, stay_data.loc[i, '经度'])), np.argwhere(np.isin(y, stay_data.loc[i, '纬度']))
        idx = np.squeeze(np.concatenate([time_idx, y_idx, x_idx]))
        grid_graph_flow[idx[0], idx[1], idx[2]] += stay_data.loc[i, '人流量指数']
    np.save('data/grid_graph_flow.npy', grid_graph_flow)
    print("数据转换完成！")


def train_val_test_loader(config):
    myTrainDataset = MyDataset(config=config, data_type='train')
    train_loader = DataLoader(dataset=myTrainDataset, batch_size=config.batch_size, shuffle=False)

    myTestDataset = MyDataset(config=config, data_type='test')  # 一次读入
    test_loader = DataLoader(dataset=myTestDataset, batch_size=myTestDataset.test_len, shuffle=False)

    return train_loader, test_loader


def plotTrainValLoss():
    plt.style.use("ggplot")
    train_losses = np.load('results/train_losses.npy', allow_pickle=True)
    val_losses = np.load('results/val_losses.npy', allow_pickle=True)
    concat_losses = np.concatenate([train_losses[:, np.newaxis], val_losses[:, np.newaxis]], axis=1)
    pd.DataFrame(data=concat_losses, columns=['Train Loss', 'Val Loss']).plot(
        logy=True, figsize=(10, 6), xlabel='Batch', ylabel='Loss')
    plt.savefig('results/train_val_loss_decrease.jpg', dpi=150, bbox_inches='tight')


def plotArea(i, j):
    flow = np.load('data/grid_graph_flow.npy')
    plt.plot(range(flow.shape[0]-192), flow[192:, i, j])
    plt.show()


def loadModel(config):
    net = ResNet(config)
    net.load_state_dict(torch.load('results/checkpoint.pt'))
    return net
