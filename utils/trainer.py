from utils.earlyStopping import EarlyStopping
from datetime import datetime
import torch
import numpy as np


def train_test_model(net, train_data, val_data, test_data, config):
    """
    训练并测试网络，最后保存训练损失与验证损失

    :param net: 神经网络
    :param train_data: 训练数据
    :param val_data: 验证数据
    :param test_data: 测试数据
    :param config: 全局设置文件
    :return: None
    """
    train_losses, val_losses = [], []
    start_time = datetime.now()
    early_stopping = EarlyStopping(patience=5, verbose=False)
    criterion = config.loss_func
    device = config.device
    optimizer = config.optimizer(net.parameters(), lr=config.learning_rate)

    for epoch in range(config.epochs):
        net.train()
        for i, data in enumerate(train_data):
            X, y_true = data[0], data[1].to(device)
            y_pred = net(X[0].to(device), X[1].to(device), X[2].to(device), X[3].to(device))

            train_loss = criterion(y_true, y_pred)
            train_losses.append(train_loss.item())
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            val_loss = evaluate(net, val_data, device)
            val_losses.append(val_loss)
            print("Epoch: {:>2d}, Batch: {:>2d} | Training Loss: {:.5f} | Val Loss: {:.5f}".format(
                epoch + 1, i, train_loss, val_loss
            ))
            early_stopping(val_loss, net)
            if early_stopping.early_stop:
                print("Early stopping")
                # 结束模型训练
                break
        if early_stopping.early_stop:
            break
    test_loss = evaluate(net, test_data, device)
    end_time = datetime.now()
    print("Test Loss:{:e} | Total Running Time: {}".format(test_loss, end_time - start_time))
    np.save('results/train_losses.npy', np.array(train_losses))
    np.save('results/val_losses.npy', np.array(val_losses))


def evaluate(net, data_iter, device):
    """
    验证/测试函数

    :param net: 网络
    :param data_iter: 数据生成器
    :param device: 工作设备
    :return: 均方误差
    """
    err_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval()
                y_pred = net(X[0].to(device), X[1].to(device), X[2].to(device), X[3].to(device))
                err_sum += torch.sum(torch.square(y_pred - y.to(device))).item()
                n += y.shape[0] * y.shape[-1] * y.shape[-2]
                net.train()
        return err_sum / n



