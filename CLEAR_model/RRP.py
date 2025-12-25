import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as fn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.resnet import resnet50
from typing import Optional
from torchvision.transforms import Compose
import cv2
from collections import OrderedDict
from CLEAR_model.depth_anything_v2.dinov2 import DINOv2
from CLEAR_model.depth_anything_v2.util.blocks import FeatureFusionBlock, _make_scratch
from CLEAR_model.depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet
from CLEAR_model.network_utils import *
import math

class RRP(nn.Module):
    def __init__(self, encoder="vits", fH=30, fW=40, embed_dim=128, **kwargs):
        super().__init__()
        # 特征提取
        self.feature_extractor = RRPFeatureExtractor(
            encoder=encoder, 
            fH=fH, 
            fW=fW, 
            embed_dim=embed_dim,
            **kwargs # 传入 num_heads 等其他参数
        )
        # 深度和不确定性头
        self.prediction_head = DepthUncertaintyHead(
            in_features=embed_dim
        )

    def forward(self, obs_img, mask=None):

        # 1. 特征提取
        # x_feat: (B, 40, 128), attn_w: (B, 40, 1200)
        x_feat, attn_w = self.feature_extractor(obs_img, mask)
        
        # 2. 预测深度和不确定性
        # d_hat: (B, 40), b_hat: (B, 40)
        d_hat, b_hat = self.prediction_head(x_feat)
        
        return d_hat, b_hat, attn_w
    
class DepthUncertaintyHead(nn.Module):
    def __init__(self, in_features=128):
        super().__init__()
        # 论文中提到的两个并行全连接层 
        # 预测深度 d_t
        self.depth_head = nn.Linear(in_features, 1)
        # 预测不确定性 b_t
        self.uncertainty_head = nn.Linear(in_features, 1)

    def forward(self, x):
        # 1. 预测深度 
        # (B, 40, 128) -> (B, 40, 1) -> (B, 40)
        d_hat = self.depth_head(x).squeeze(-1)
        # (B, 40, 128) -> (B, 40, 1) -> (B,  40)

        b_hat = F.softplus(self.uncertainty_head(x)).squeeze(-1)
        
        return d_hat, b_hat

class RRPFeatureExtractor(nn.Module):
    def __init__(self, encoder="vits", fH=30, fW=40, embed_dim=128, pos_embed_dim=32, num_heads=8, target_size=(23, 40), checkpoint_path=None):
        super().__init__()
        
        self.fH = fH 
        self.fW = fW
        self.embed_dim = embed_dim
        self.pos_embed_dim = pos_embed_dim
        self.intermediate_layer_idx = 11 

        params = torch.load(checkpoint_path, map_location='cpu')
        params = torch.load(checkpoint_path, map_location='cpu')
        self.pretrained = DINOv2(model_name=encoder)

        pretrained_dict = OrderedDict()
        for k, v in params.items():
            if k.startswith('pretrained.'):
                pretrained_dict[k[11:]] = v
                
        self.pretrained.load_state_dict(pretrained_dict, strict=True)
        
        # 冻结试试
        for param in self.pretrained.parameters():
            param.requires_grad = False
        # 解冻最后一层
        # if hasattr(self.pretrained, 'blocks'):
        #     for param in self.pretrained.blocks[self.intermediate_layer_idx].parameters():
        #         param.requires_grad = True
        # if hasattr(self.pretrained, 'norm'):
        #     for param in self.pretrained.norm.parameters():
        #         param.requires_grad = True
                
        # (B, 384, 30, 40) -> (B, 128, 30, 40)
        self.conv = ConvBnReLU(
            in_channels=384, out_channels=self.embed_dim, kernel_size=3, padding=1, stride=1
        )
        
        # 垂直注意力池化
        self.vertical_pool = VerticalAttentionPooling(in_channels=self.embed_dim)
        
        # 2D 位置编码 (用于 K 和 V)
        self.pos_mlp_2d = nn.Sequential(
            nn.Linear(2, self.pos_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_embed_dim, self.pos_embed_dim)
        )
        # 1D 位置编码 (用于 Q)
        self.pos_mlp_1d = nn.Sequential(
            nn.Linear(1, self.pos_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_embed_dim, self.pos_embed_dim)
        )
        
        # --- 注意力机制的 Q, K, V 投影层 ---
        total_embed_dim = self.embed_dim + self.pos_embed_dim
        self.q_proj = nn.Linear(total_embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(total_embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(total_embed_dim, self.embed_dim)
        
        self.attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim, 
            num_heads=num_heads, 
            batch_first=True
        )
        
        self.target_size = target_size

    def forward(self, obs_img, mask=None):
        B, C, H, W = obs_img.shape
        patch_size = 14
        
        target_h_pad = int(math.ceil(H / patch_size) * patch_size)
        target_w_pad = int(math.ceil(W / patch_size) * patch_size)
        
        target_h = int(math.ceil(H / patch_size) * patch_size)
        target_w = int(math.ceil(W / patch_size) * patch_size)
        pad_h = target_h - H
        pad_w = target_w - W
        img_padded = F.pad(obs_img, (0, pad_w, 0, pad_h))
        # 1. 拿最后一层的特征
        # (B, 3, H, W) -> (B, 384, 37, 49)
        features = self.pretrained.get_intermediate_layers(
            img_padded, 
            [self.intermediate_layer_idx], 
            return_class_token=True
        )
        
        # features[0] is a tuple (patch_tokens, class_token)
        features_tensor, class_token = features[0] 
        # features_tensor: (B, 37 x 49, 384)
        # class_token: (B, 384)

        features_tensor = features_tensor.permute(0, 2, 1) # (B, 37 x 49, 384) -> (B, 384, 37 x 49)
        
        grid_h = target_h_pad // patch_size
        grid_w = target_w_pad // patch_size  # 644 / 14 = 46
        
        features_tensor_2d = features_tensor.reshape(B, 384, grid_h, grid_w)
        
        # 双线性插值 (B, 384, 37, 49) -> (B, 384, 30, 40)
        interpolated_features = F.interpolate(
            features_tensor_2d, 
            size=self.target_size, 
            mode='bilinear', 
            align_corners=False
        )
        
        # 2. 卷积降维
        # (B, 384, 30, 40) -> (B, 128, 30, 40)
        x_2d = self.conv(interpolated_features) 
        
        # 3. 准备 Query (1D)
        # (B, 128, 30, 40) -> (B, 128, 40)
        # query = x_2d.mean(dim=2) 
        x_weighted, pooling_weights = self.vertical_pool(x_2d)  #垂直注意力池化
        query = x_weighted.permute(0, 2, 1) # B 40 128
        
        
        # query = query.permute(0, 2, 1) # (B, 40, 128)
        
        # 4. 准备 Key 和 Value (2D)
        x_2d = x_2d.view(B, self.embed_dim, -1) # (B, 128, 30*40)
        x_2d = x_2d.permute(0, 2, 1) # (B, 1200, 128)
        
        # 5. 计算并附加 2D 位置编码 (用于 K, V)
        pos_x = torch.linspace(0, 1, self.target_size[1], device=x_2d.device) - 0.5
        pos_y = torch.linspace(0, 1, self.target_size[0], device=x_2d.device) - 0.5
        pos_grid_2d_y, pos_grid_2d_x = torch.meshgrid(pos_y, pos_x, indexing="ij")
        pos_grid_2d = torch.stack((pos_grid_2d_x, pos_grid_2d_y), dim=-1) # (30, 40, 2)
        
        pos_enc_2d = self.pos_mlp_2d(pos_grid_2d) # (30, 40, 32)
        pos_enc_2d = pos_enc_2d.reshape((1, -1, self.pos_embed_dim)) # (1, 1200, 32)
        pos_enc_2d = pos_enc_2d.repeat((B, 1, 1))
        
        x_2d = torch.cat((x_2d, pos_enc_2d), dim=-1) # (B, 1200, 128+32)
        
        # 6. 计算并附加 1D 位置编码 (用于 Q)
        pos_v = torch.linspace(0, 1, self.target_size[1], device=query.device) - 0.5 # (40,)
        pos_enc_1d = self.pos_mlp_1d(pos_v.reshape((-1, 1))) # (40, 32)
        pos_enc_1d = pos_enc_1d.reshape((1, -1, self.pos_embed_dim)).repeat((B, 1, 1)) # (B, 40, 32)
        
        query = torch.cat((query, pos_enc_1d), dim=-1) # (B, 40, 128+32)
        
        # 7. 投影 Q, K, V
        query = self.q_proj(query) # (B, 40, 128)
        key = self.k_proj(x_2d)   # (B, 1200, 128)
        value = self.v_proj(x_2d) # (B, 1200, 128)
        
        # 8. 处理掩码 (Mask) [cite: 153]
        attn_mask = None
        if mask is not None:
            # (B, H, W) -> (B, fH, fW)
            mask = fn.resize(mask, self.target_size, fn.InterpolationMode.NEAREST).type(torch.bool)
            # True 表示 *无效* 区域, 在注意力中应被掩盖
            mask = torch.logical_not(mask)
            mask = mask.reshape((B, 1, -1)) # (B, 1, fH*fW)
            # (N, fW, fH*fW)
            attn_mask = mask.repeat(1, grid_w, 1)

        # 9. 执行掩码注意力
        # x_out: (B, 40, 128), attn_w: (B, 40, 1200)
        x_out, attn_w = self.attn(query, key, value, attn_mask=attn_mask)
        
        return x_out, attn_w, class_token
class VerticalAttentionPooling(nn.Module):
    def __init__(self, in_channels, hidden_channels=64):
        super().__init__()
        # 一个轻量级的 MLP，用来给每个像素打分
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1)
        )

    def forward(self, x, mask=None):
        """
        x: (B, C, H, W) - 输入特征图
        """
        # 1. 计算每个像素的得分 (B, 1, H, W)
        scores = self.net(x)
    
        # 2. 在高度 H (dim=2) 维度上进行 Softmax
        # 这样每一列 (column) 的权重之和为 1
        attn_weights = F.softmax(scores, dim=2) 
        
        # 3. 加权求和: (B, C, H, W) * (B, 1, H, W) -> sum(dim=2) -> (B, C, W)
        x_weighted = (x * attn_weights).sum(dim=2)
        
        return x_weighted, attn_weights