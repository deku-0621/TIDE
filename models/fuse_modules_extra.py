from math import sqrt

import torch
import torch.nn as nn
import torch
import torch.nn as nn


class AsymmetricBatchCrossAttention(nn.Module):
    def __init__(self, q_dim, s_dim, embed_dim, num_heads=4, dropout=0.1):
        super(AsymmetricBatchCrossAttention, self).__init__()

        # 维度必须能被num_head 整除
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim
        assert q_dim == s_dim
        self.dim = q_dim

        assert (
                self.head_dim * self.num_heads == self.embed_dim
        )

        self.scale = self.head_dim ** (-0.5)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)

        self.q = nn.Linear(self.dim, self.embed_dim)
        self.k = nn.Linear(self.dim, self.embed_dim)
        self.v = nn.Linear(self.dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, tgt_len: int, bsz: int):
        return tensor.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, x, y):
        # x: tensor of shape (batch, n, dim_in)
        bsz_x, tgt_len_x, c_x = x.size()
        bsz_y, tgt_len_y, c_y = y.size()

        q_x = self._shape(self.q(x), -1, bsz_x)
        q_y = self._shape(self.q(y), -1, bsz_y)

        k_x = self._shape(self.k(x), -1, bsz_x)
        k_y = self._shape(self.k(y), -1, bsz_y)

        v_x = self._shape(self.v(x), -1, bsz_x)
        v_y = self._shape(self.v(y), -1, bsz_y)

        if bsz_x == 1:
            k_y_avg = k_y.mean(0, True)
            v_y_avg = v_y.mean(0, True)
            k_cat_x = torch.cat((k_x, k_y_avg), dim=2)
            v_cat_x = torch.cat((v_x, v_y_avg), dim=2)
        elif bsz_y == 1:
            k_y_ext = k_y.repeat(bsz_x, 1, 1, 1)
            v_y_ext = v_y.repeat(bsz_x, 1, 1, 1)
            k_cat_x = torch.cat((k_x, k_y_ext), dim=2)
            v_cat_x = torch.cat((v_x, v_y_ext), dim=2)

        attn_x = (q_x @ k_cat_x.transpose(-2, -1)) * self.scale
        attn_x = attn_x.softmax(dim=-1)
        attn_x = self.attn_drop(attn_x)

        x = (attn_x @ v_cat_x).transpose(1, 2).reshape(bsz_x, tgt_len_x, c_x)
        x = self.proj(x)
        x = self.proj_drop(x)

        if bsz_x == 1:
            k_x_ext = k_x.repeat(bsz_y, 1, 1, 1)
            v_x_ext = v_x.repeat(bsz_y, 1, 1, 1)
            k_cat_y = torch.cat((k_x_ext, k_y), dim=2)
            v_cat_y = torch.cat((v_x_ext, v_y), dim=2)
        elif bsz_y == 1:
            k_x_avg = k_x.mean(0, True)
            v_x_avg = v_x.mean(0, True)
            k_cat_y = torch.cat((k_x_avg, k_y), dim=2)
            v_cat_y = torch.cat((v_x_avg, v_y), dim=2)

        attn_y = (q_y @ k_cat_y.transpose(-2, -1)) * self.scale
        attn_y = attn_y.softmax(dim=-1)
        attn_y = self.attn_drop(attn_y)

        y = (attn_y @ v_cat_y).transpose(1, 2).reshape(bsz_y, tgt_len_y, c_y)
        y = self.proj(y)
        y = self.proj_drop(y)

        return x, y


if __name__ == "__main__":
    a = torch.rand(1, 10, 256)
    b = torch.rand(2, 10, 256)
    attn = AsymmetricBatchCrossAttention(256, 256, 256, 4)
    out1, out2 = attn(a, b)
    print(out1.shape,  out2.shape)
