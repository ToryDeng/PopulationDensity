import torch


def testLastSameDayRegression(test_iter, device):
    err_sum, n = 0.0, 0
    for sample in test_iter:
        X, y_true = sample[0], sample[1]
        y_pred = X[2][0]
        err_sum += torch.sum(torch.square(y_pred.to(device) - y_true.to(device))).item()
        n += y_true.shape[0] * y_true.shape[-1] * y_true.shape[-2]
    print("以上周同天预测的MSE：{:e}".format(err_sum / n))
