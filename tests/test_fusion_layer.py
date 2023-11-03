import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from models.query_support_feature_fusion import BiAttentionBlock
d_model = 256
dim_feedforward = 2048
nhead = 8
fusion_dropout = 0.0
fusion_droppath = 0.1
feature_fusion_layer = BiAttentionBlock(
                q_dim=d_model,
                s_dim=d_model,
                embed_dim=dim_feedforward // 2,
                num_heads=nhead // 2,
                dropout=fusion_dropout,
                drop_path=fusion_droppath,
            )
support_imgs = torch.randn(3,400,256)
query_imgs = torch.randn(3,300,256)
output, memory_support_img = feature_fusion_layer(
                        v=query_imgs,
                        l=support_imgs,
                        attention_mask_v=None,
                        attention_mask_l=None,
                    )
print(f'output:{output.shape},memory_support_img：{memory_support_img.shape}')