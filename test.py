import torch

x = torch.randn(2, 3)
a = torch.cat((x, x, x), 0)
b = torch.cat((x, x, x), 1)

print(a.shape)
print(b.shape)
