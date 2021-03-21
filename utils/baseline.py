import torch
import pmdarima as pm
import numpy as np
from itertools import product
import multiprocessing
from tqdm.contrib.itertools import product


def testLastHourRegression(test_iter, device, criterion, hour_type):
    for X, y_true in test_iter:
        if hour_type == 'last_week':
            y_pred = X[2][:, 0]
            print(criterion(y_pred.to(device), y_true.to(device)))
            print("以上周同天同小时预测的MSE：{:e}".format(criterion(y_pred.to(device), y_true.to(device)) / y_true.shape[0]))
        elif hour_type == 'last_day':
            y_pred = X[1][:, 0]
            print("以昨天同小时预测的MSE：{:e}".format(criterion(y_pred.to(device), y_true.to(device)) / y_true.shape[0]))
        else:
            print('ERROR!')
            return None


def onePlaceArima(train, test, test_len):
    model = pm.auto_arima(train, trace=False, suppress_warnings=True)
    pred = model.predict(test_len)
    return np.sum(np.square(test - pred))


def testArima(config):
    flow = np.load('data/grid_graph_flow.npy')
    flow = (flow - np.min(flow)) / (np.max(flow) - np.min(flow))
    sample_num = flow.shape[0] - config.week * config.trend_len
    train_len = int(sample_num * config.train_size)
    test_len = sample_num - train_len

    train_flow, test_flow = flow[:-test_len], flow[-test_len:]
    data = [(train_flow[:, i, j], test_flow[:, i, j], test_len) for i, j in
            product(range(train_flow.shape[1]), range(train_flow.shape[2]))]
    err_sum = 0.0

    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=int(cores / 2))
    for err in pool.imap(onePlaceArima, data):
        print(err)
        err_sum += err
    print(err_sum / test_len)



