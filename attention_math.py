import math
from typing import Any
from typing import Literal

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F


def main():
    num_heads = 2
    input_tensor = torch.rand(4, 4, 16)
    print("input shape: %s", input_tensor.shape)
    print("-------first data row-------")
    print(input_tensor[0])
    s_attn = nn.Linear(16, 16 * 3, bias=False)
    B, T, C = input_tensor.size()
    q, k, v = s_attn(input_tensor).split(16, dim=2)
    print("-------first data row after attn linear-------")
    print(q[0])
    print("query shape: %s", q.shape)
    q = q.view(B, T, num_heads, C // num_heads).transpose(1, 2)
    k = k.view(B, T, num_heads, C // num_heads).transpose(1, 2)
    v = v.view(B, T, num_heads, C // num_heads).transpose(1, 2)
    print("-------first data row Query-------")
    print(q[0])
    print(q.shape)
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    print(att[0])
    print("q @ k.T shape: %s", att.shape)
    att = F.softmax(att, dim=-1)
    print(att[0])
    attn_dropout = nn.Dropout(0.1)
    att = attn_dropout(att)
    print(att[0])
    y = att @ v
    print("att @ v shape: %s", y.shape)
    y = y.transpose(1, 2).contiguous().view(B, T, C)
    print("final y shape: %s", y.shape)


if __name__ == "__main__":
    main()