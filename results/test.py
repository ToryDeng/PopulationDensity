import torch
import numpy as np

loss = torch.nn.MSELoss()
a = np.array([[1, 2], [4, 4]])
b = np.array([[2, 3], [4, 5]])
input = torch.autograd.Variable(torch.from_numpy(a))
target = torch.autograd.Variable(torch.from_numpy(b))
output = loss(input.float(), target.float())
print(output)
