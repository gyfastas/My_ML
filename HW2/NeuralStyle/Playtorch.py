import torchvision
import  torch
from torchvision.utils import save_image


k = torch.zeros([250,250,3])
k[10:240][10:240] = torch.Tensor([255,255,0])
k = k.type(torch.ByteTensor)

save_image(k,'play.png')
