import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class MapEncoder(nn.Module):
    def __init__(self, input_channels=1, feature_dim=128, use_pretrained=True):
        super().__init__()
        
        # 1. 加载 ResNet18
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if use_pretrained else None
        self.backbone = models.resnet18(weights=weights)
        
        # 2. 修改第一层卷积 (适配单通道输入)
        if input_channels != 3:
            original_conv1 = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(
                input_channels, 
                original_conv1.out_channels,
                kernel_size=original_conv1.kernel_size,
                stride=original_conv1.stride,
                padding=original_conv1.padding,
                bias=False
            )
            if use_pretrained:
                with torch.no_grad():
                    self.backbone.conv1.weight.data = original_conv1.weight.data.mean(dim=1, keepdim=True)

        # 3. 移除 FC，改用 1x1 Conv 降维保留空间结构
        # ResNet18 layer4 output is 512 channels
        self.proj_conv = nn.Conv2d(512, feature_dim, kernel_size=1)

    def forward(self, x):
        # Explicitly forward through ResNet layers to keep spatial dim
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x) # (B, 512, H/32, W/32)
        
        # Project to target dim
        x = self.proj_conv(x) # (B, 128, H/32, W/32)
        
        return x
