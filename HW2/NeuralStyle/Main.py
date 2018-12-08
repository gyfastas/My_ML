import torchvision
import torch
import torch.nn as nn
import PIL
import torch.nn.functional as F
from torchvision.transforms import transforms
import models
from models.Metanet import *
from Config import Config

#load configs
configs = Config()

'''
tensor normalization demonstrated in Pytorch doc
'''
cnn_normalization_mean = [0.485, 0.456, 0.406]
cnn_normalization_std = [0.229, 0.224, 0.225]
tensor_normalizer = transforms.Normalize(mean=cnn_normalization_mean, std=cnn_normalization_std)


'''
Load Data
'''
data_transform = transforms.Compose([
    transforms.RandomResizedCrop(configs.img_size, scale=(256/480, 1), ratio=(1, 1)),
    transforms.ToTensor(),
    tensor_normalizer
])

style_dataset = torchvision.datasets.ImageFolder(configs.content_root, transform=data_transform)
content_dataset = torchvision.datasets.ImageFolder(configs.style_root, transform=data_transform)


def train(**kwargs):
    opt = Config()
    for k_,v_ in kwargs.items():
        setattr(opt,k_,v_)

    device = torch.device('cuda') if opt.use_gpu else torch.device('cpu')


    for batch, (content_images, _) in pbar:
        # 每 20 个 batch 随机挑选一张新的风格图像，计算其特征
        if batch % 20 == 0:
            style_image = random.choice(style_dataset)[0].unsqueeze(0).to(device)
            style_features = vgg16(style_image)
            style_mean_std = mean_std(style_features)

        # 检查纯色
        x = content_images.cpu().numpy()
        if (x.min(-1).min(-1) == x.max(-1).max(-1)).any():
            continue

        optimizer.zero_grad()

        # 使用风格图像生成风格模型
        weights = metanet(mean_std(style_features))
        transform_net.set_weights(weights, 0)

        # 使用风格模型预测风格迁移图像
        content_images = content_images.to(device)
        transformed_images = transform_net(content_images)

        # 使用 vgg16 计算特征
        content_features = vgg16(content_images)
        transformed_features = vgg16(transformed_images)
        transformed_mean_std = mean_std(transformed_features)

        # content loss
        content_loss = content_weight * F.mse_loss(transformed_features[2], content_features[2])

        # style loss
        style_loss = style_weight * F.mse_loss(transformed_mean_std,
                                               style_mean_std.expand_as(transformed_mean_std))

        # total variation loss
        y = transformed_images
        tv_loss = tv_weight * (torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) +
                               torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :])))

        # 求和
        loss = content_loss + style_loss + tv_loss

        loss.backward()
        optimizer.step()



