# import torch
# from torch.distributions.laplace import Laplace

import numpy as np

exponent_eps_list = [-3.0 + i * 0.2 for i in range(10)]

print(exponent_eps_list)

# x = np.linspace(-3.0, -1.0, num=10)
# print(x)

# for exponent_eps in range(-3, -1, )

# m = Laplace(torch.tensor([0.0]), torch.tensor([1.0]))
# x = m.sample()  # Laplace distributed with loc=0, scale=1

# print(x.shape)
# print(x)

# Delta = 3 / 1e4

# x = torch.randn(2, 3)
# a = torch.cat((x, x, x), 0)
# b = torch.cat((x, x, x), 1)

# print(a.shape)
# print(b.shape)
