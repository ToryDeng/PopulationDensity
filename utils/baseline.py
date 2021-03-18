import torch


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




