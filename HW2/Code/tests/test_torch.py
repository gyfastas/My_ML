import torch
import torchvision

x = torch.randn((4,4),requires_grad = True)
y = x*2
z = y.sum()
z.backward()
grad = x.grad
gradz = z.grad
print(x)
print(grad)
print(gradz)