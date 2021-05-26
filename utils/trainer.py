from utils.earlyStopping import EarlyStopping
from torch.optim.lr_scheduler import CosineAnnealingLR
from datetime import datetime
import torch
import numpy as np


def train_test_model(net, train_data, test_data, config):
    """
    训练并测试网络，最后保存训练损失与验证损失

    :param net: 神经网络
    :param train_data: 训练数据
    :param test_data: 测试数据
    :param config: 全局设置文件
    :return: None
    """
    train_losses, test_losses = [], []
    start_time = datetime.now()
    early_stopping = EarlyStopping(patience=config.patience, verbose=False)
    metric = config.metric
    device = config.device
    optimizer = config.optimizer(net.parameters(), lr=config.learning_rate)
    scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=30, eta_min=0)

    for epoch in range(config.epochs):
        net.train()
        for i, data in enumerate(train_data):
            X, y_true = data[0], data[1].to(device)
            y_pred = net(X)

            train_loss = metric(y_true, y_pred) / y_true.shape[0]
            train_losses.append(train_loss.item())
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            if (i + 1) % 2 == 0:
                print("Epoch: {:>2d}, Batch: {:>2d} | Training Loss: {:.5f}".format(
                    epoch + 1, i + 1, train_loss
                ))
        scheduler.step()
    test_loss = evaluate(net, test_data, device, metric)
    # early_stopping(val_loss, net)
    end_time = datetime.now()
    print("Test Loss: {:e} | Total Running Time: {}".format(test_loss, end_time - start_time))
    # if early_stopping.early_stop:
    #     print("Early stopping")
    #     # 结束模型训练
    #     break
    torch.save(net, 'results/DeepST-ResNet.pkl')
    np.save('results/train_losses.npy', np.array(train_losses))


def evaluate(net, data_iter, device, method):
    """
    验证/测试函数

    :param method: 评估指标
    :param net: 网络
    :param data_iter: 数据生成器
    :param device: 工作设备
    :return: 均方误差
    """
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval()
                y_pred = net(X)
                y_pred[y_pred < 0] = 0
                net.train()
                return method(y_pred, y.to(device)) / y.shape[0]
