import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention, GRU, Linear, LayerNorm, Dropout


class FFN(nn.Module):
    def __init__(self, d_model, bidirectional=True, dropout=0):
        super().__init__()
        self.gru = GRU(d_model, d_model * 2, 1, bidirectional=bidirectional)
        if bidirectional:
            self.linear = Linear(d_model * 2 * 2, d_model)
        else:
            self.linear = Linear(d_model * 2, d_model)
        self.dropout = Dropout(dropout)

    def forward(self, x):
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)
        x = self.linear(x)

        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, bidirectional=True, dropout=0):
        super().__init__()

        self.norm1 = LayerNorm(d_model)
        self.attention = MultiheadAttention(
            d_model, n_heads, dropout=dropout
        )  # batch_first=True?
        self.dropout1 = Dropout(dropout)

        self.norm2 = LayerNorm(d_model)
        self.ffn = FFN(d_model, bidirectional=bidirectional)
        self.dropout2 = Dropout(dropout)

        self.norm3 = LayerNorm(d_model)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        xt = self.norm1(x)
        xt, _ = self.attention(
            xt, xt, xt, attn_mask=attn_mask, key_padding_mask=key_padding_mask
        )
        x = x + self.dropout1(xt)

        xt = self.norm2(x)
        xt = self.ffn(xt)
        x = x + self.dropout2(xt)

        x = self.norm3(x)

        return x
