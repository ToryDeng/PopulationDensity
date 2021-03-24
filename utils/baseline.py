import pmdarima as pm
import numpy as np
from itertools import product
import warnings
from tqdm.contrib.itertools import product


def testLastHourRegression(test_iter, device, criterion, hour_type):
    for X, y_true in test_iter:
        if hour_type == 'last_week':
            y_pred = X[2][:, 0]
            print("以上周同天同小时预测的MSE：{:e}".format(criterion(y_pred.to(device), y_true.to(device)) / y_true.shape[0]))
        elif hour_type == 'last_day':
            y_pred = X[1][:, 0]
            print("以昨天同小时预测的MSE：{:e}".format(criterion(y_pred.to(device), y_true.to(device)) / y_true.shape[0]))
        else:
            print('ERROR!')
            return None


def onePlaceArima(train, test, test_len):
    model = pm.auto_arima(train, trace=False, suppress_warnings=True, seasonal=True, m=24, n_jobs=-1)
    pred = model.predict(test_len)
    return np.sum(np.square(test - pred))


def testArima(config):
    warnings.filterwarnings('ignore')
    flow = np.load('data/grid_graph_flow.npy')[8 * 24:, :, :]
    flow = (flow - np.min(flow)) / (np.max(flow) - np.min(flow))
    sample_num = flow.shape[0] - config.week * config.trend_len
    train_len = int(sample_num * config.train_size)
    test_len = sample_num - train_len

    train_flow, test_flow = flow[:-test_len], flow[-test_len:]
    err_sum = 0.0
    random_y = np.random.choice(train_flow.shape[1], int(train_flow.shape[1] * config.sample_rate), replace=False)
    random_x = np.random.choice(train_flow.shape[2], int(train_flow.shape[2] * config.sample_rate), replace=False)

    for i, j in product(random_y, random_x):
        err_sum += onePlaceArima(train_flow[:, i, j], test_flow[:, i, j], test_len)
    print("ARIMA预测MSE：{:e}".format(err_sum / (test_len * config.sample_rate)))
