import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import torch.nn.init as init
from torch import Tensor
from typing import Optional, Tuple
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import TensorDataset, DataLoader



class SelfAttentionPooling(nn.Module):
    """
    Implementation of SelfAttentionPooling 
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf
    """
    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = nn.Linear(input_dim, 1)
        
    def forward(self, batch_rep):
        """
        input:
            batch_rep : size (N, T, H), N: batch size, T: sequence length, H: Hidden dimension
        
        attention_weight:
            att_w : size (N, T, 1)
        
        return:
            utter_rep: size (N, H)
        """
        softmax = nn.functional.softmax
        att_w = softmax(self.W(batch_rep).squeeze(-1)).unsqueeze(-1)
        utter_rep = torch.sum(batch_rep * att_w, dim=1)

        return utter_rep, att_w.squeeze()
    
class Conv2dSubsampling(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None):
        """Construct an Conv2dSubsampling object."""
        super(Conv2dSubsampling, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim),
            RelPositionalEncoding(odim, dropout_rate),
        )

    def forward(self, x, x_mask):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 4.

        """
        x = x.transpose(1,2)
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x
        return x, x_mask[:, :, :-2:2][:, :, :-2:2]
    
class RelPositionalEncoding(torch.nn.Module):

    def __init__(self, d_model, dropout_rate, max_len=5000):
        """Construct an PositionalEncoding object."""
        super(RelPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x):
        """Reset the positional encodings."""
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1) * 2 - 1:
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return

        pe_positive = torch.zeros(x.size(1), self.d_model)
        pe_negative = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor):
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).

        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).

        """
        self.extend_pe(x)
        x = x * self.xscale
        pos_emb = self.pe[
            :,
            self.pe.size(1) // 2 - x.size(1) + 1 : self.pe.size(1) // 2 + x.size(1),
        ]
        return self.dropout(x), self.dropout(pos_emb)

    
class Linear(nn.Module):
    """
    Wrapper class of torch.nn.Linear
    Weight initialize by xavier initialization and bias initialize to zeros.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.xavier_uniform_(self.linear.weight)
        if bias:
            init.zeros_(self.linear.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)

class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention proposed in "Attention Is All You Need"
    Compute the dot products of the query with all keys, divide each by sqrt(dim),
    and apply a softmax function to obtain the weights on the values

    Args: dim, mask
        dim (int): dimension of attention
        mask (torch.Tensor): tensor containing indices to be masked

    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): tensor containing projection vector for decoder.
        - **key** (batch, k_len, d_model): tensor containing projection vector for encoder.
        - **value** (batch, v_len, d_model): tensor containing features of the encoded input sequence.
        - **mask** (-): tensor containing indices to be masked

    Returns: context, attn
        - **context**: tensor containing the context vector from attention mechanism.
        - **attn**: tensor containing the attention (alignment) from the encoder outputs.
    """
    def __init__(self, dim: int) -> None:
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim

        if mask is not None:
            score.masked_fill_(mask, -1e9)

        attn = F.softmax(score, -1)
        context = torch.bmm(attn, value)
        return context, attn


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention proposed in "Attention Is All You Need"
    Instead of performing a single attention function with d_model-dimensional keys, values, and queries,
    project the queries, keys and values h times with different, learned linear projections to d_head dimensions.
    These are concatenated and once again projected, resulting in the final values.
    Multi-head attention allows the model to jointly attend to information from different representation
    subspaces at different positions.

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W_o
        where head_i = Attention(Q · W_q, K · W_k, V · W_v)

    Args:
        d_model (int): The dimension of keys / values / quries (default: 512)
        num_heads (int): The number of attention heads. (default: 8)

    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): tensor containing projection vector for decoder.
        - **key** (batch, k_len, d_model): tensor containing projection vector for encoder.
        - **value** (batch, v_len, d_model): tensor containing features of the encoded input sequence.
        - **mask** (-): tensor containing indices to be masked

    Returns: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the attended output features.
        - **attn** (batch * num_heads, v_len): tensor containing the attention (alignment) from the encoder outputs.
    """
    def __init__(self, d_model: int = 384, num_heads: int = 8) -> None:
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0, "hidden_dim % num_heads should be zero."

        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads
        self.query_proj = Linear(d_model, self.d_head * num_heads)
        self.key_proj = Linear(d_model, self.d_head * num_heads)
        self.value_proj = Linear(d_model, self.d_head * num_heads)
        self.sqrt_dim = np.sqrt(d_model)
        self.scaled_dot_attn = ScaledDotProductAttention(self.d_head)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        batch_size = value.size(0)

        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head)  # BxQ_LENxNxD
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head)        # BxK_LENxNxD
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head)  # BxV_LENxNxD

        query = query.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_head)  # BNxQ_LENxD
        key = key.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_head)      # BNxK_LENxD
        value = value.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_head)  # BNxV_LENxD

        if mask is not None:
            mask = mask.repeat(self.num_heads, 1, 1)

        context, attn = self.scaled_dot_attn(query, key, value, mask)
        context = context.view(self.num_heads, batch_size, -1, self.d_head)
        context = context.permute(1, 2, 0, 3).contiguous().view(batch_size, -1, self.num_heads * self.d_head)  # BxTxND

        return context, attn
    
class PositionWiseFeedForwardNet(nn.Module):
    """
    Position-wise Feedforward Networks proposed in "Attention Is All You Need".
    Fully connected feed-forward network, which is applied to each position separately and identically.
    This consists of two linear transformations with a ReLU activation in between.
    Another way of describing this is as two convolutions with kernel size 1.
    """
    def __init__(self, d_model: int = 384, d_ff: int = 2048, dropout_p: float = 0.3, ffnet_style: str = 'ff') -> None:
        super(PositionWiseFeedForwardNet, self).__init__()
        self.ffnet_style = ffnet_style.lower()
        if self.ffnet_style == 'ff':
            self.feed_forward = nn.Sequential(
                Linear(d_model, d_ff),
                nn.Dropout(dropout_p),
                nn.ReLU(),
                Linear(d_ff, d_model),
                nn.Dropout(dropout_p),
            )

        elif self.ffnet_style == 'conv':
            self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
            self.relu = nn.ReLU()
            self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)

        else:
            raise ValueError("Unsupported mode: {0}".format(self.mode))

    def forward(self, inputs: Tensor) -> Tensor:
        if self.ffnet_style == 'conv':
            output = self.conv1(inputs.transpose(1, 2))
            output = self.relu(output)
            return self.conv2(output).transpose(1, 2)

        return self.feed_forward(inputs)


class ACME(nn.Module):
    def __init__(self, num_layers: int = 2, d_model: int = 384, num_heads: int = 8, d_ff: int = 2048):
        super(ACME, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads without remainder."

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleDict({
                    'mhsa_asr': MultiHeadAttention(d_model, num_heads),
                    'mhsa_psd': MultiHeadAttention(d_model, num_heads),
                    'mhsa_cross_a': MultiHeadAttention(d_model, num_heads),
                    'mhsa_cross_b': MultiHeadAttention(d_model, num_heads),
                    'ffn_asr': PositionWiseFeedForwardNet(d_model, d_ff),
                    'ffn_psd': PositionWiseFeedForwardNet(d_model, d_ff),
                    'norm_asr': nn.LayerNorm(d_model),
                    'norm_psd': nn.LayerNorm(d_model),
                    'norm_cross_asr': nn.LayerNorm(d_model),
                    'norm_cross_psd': nn.LayerNorm(d_model),
                    'norm_asr2': nn.LayerNorm(d_model),
                    'norm_psd2': nn.LayerNorm(d_model),
                })
            )

    def forward(self, asr_embed, psd_embed) -> Tensor:
        a_output, b_output = asr_embed, psd_embed

        for layer in self.layers:
            a_context, _ = layer['mhsa_asr'](a_output, a_output, a_output)
            b_context, _ = layer['mhsa_psd'](b_output, b_output, b_output)
            a_output2 = layer['norm_asr'](a_output + a_context)
            b_output2 = layer['norm_psd'](b_output + b_context)

            c_context, _ = layer['mhsa_cross_a'](a_output2, b_output2, b_output2)
            d_context, _ = layer['mhsa_cross_b'](b_output2, a_output2, a_output2)
            a_output3 = layer['norm_cross_asr'](a_output + c_context + a_output2)
            b_output3 = layer['norm_cross_psd'](b_output + d_context + b_output2)

            a_output = layer['norm_asr2'](layer['ffn_asr'](a_output3) + a_output3 + a_output)
            b_output = layer['norm_psd2'](layer['ffn_psd'](b_output3) + b_output3 + b_output)

        return torch.cat((a_output, b_output), dim=1)
    
    
class ACME2(nn.Module):
    def __init__(self, num_layers: int = 2, d_model: int = 384, num_heads: int = 8, d_ff: int = 2048):
        super(ACME2, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads without remainder."

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleDict({
                    'mhsa_asr': MultiHeadAttention(d_model, num_heads),
                    'mhsa_psd': MultiHeadAttention(d_model, num_heads),
                    'mhsa_cross_a': MultiHeadAttention(d_model, num_heads),
                    'mhsa_cross_b': MultiHeadAttention(d_model, num_heads),
                    'ffn_asr': PositionWiseFeedForwardNet(d_model, d_ff),
                    'ffn_psd': PositionWiseFeedForwardNet(d_model, d_ff),
                    'norm_asr': nn.LayerNorm(d_model),
                    'norm_psd': nn.LayerNorm(d_model),
                    'norm_cross_asr': nn.LayerNorm(d_model),
                    'norm_cross_psd': nn.LayerNorm(d_model),
                    'norm_asr2': nn.LayerNorm(d_model),
                    'norm_psd2': nn.LayerNorm(d_model),
                })
            )

    def forward(self, asr_embed, psd_embed) -> Tensor:
        a_output, b_output = asr_embed, psd_embed
        
        for layer in self.layers:
            a_context, _ = layer['mhsa_asr'](a_output, a_output, a_output)
            b_context, _ = layer['mhsa_psd'](b_output, b_output, b_output)
            a_output2 = layer['norm_asr'](a_output + a_context)
            b_output2 = layer['norm_psd'](b_output + b_context)

            a_output3, _ = layer['mhsa_cross_a'](a_output2, b_output2, b_output2)
            b_output3, _ = layer['mhsa_cross_b'](b_output2, a_output2, a_output2)
            # a_output3 = layer['norm_cross_asr'](a_output + c_context + a_output2)
            # b_output3 = layer['norm_cross_psd'](b_output + d_context + b_output2)

            a_output = layer['norm_asr2'](layer['ffn_asr'](a_output3) + a_output3 + a_output)
            b_output = layer['norm_psd2'](layer['ffn_psd'](b_output3) + b_output3 + b_output)

        return torch.cat((a_output, b_output), dim=1)
    
    
class MMCA(nn.Module):
    def __init__(self, num_layers: int = 2, d_model: int = 384, num_heads: int = 8, d_ff: int = 2048):
        super(MMCA, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads without remainder."

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleDict({
                    'mhsa_asr': MultiHeadAttention(d_model, num_heads),
                    'mhsa_psd': MultiHeadAttention(d_model, num_heads),
                    'mhsa_cross_a': MultiHeadAttention(d_model, num_heads),
                    'mhsa_cross_b': MultiHeadAttention(d_model, num_heads),
                    'ffn_asr': PositionWiseFeedForwardNet(d_model, d_ff),
                    'ffn_psd': PositionWiseFeedForwardNet(d_model, d_ff),
                    'norm_asr': nn.LayerNorm(d_model),
                    'norm_psd': nn.LayerNorm(d_model),
                    'norm_cross_asr': nn.LayerNorm(d_model),
                    'norm_cross_psd': nn.LayerNorm(d_model),
                    # 'norm_asr2': nn.LayerNorm(d_model),
                    # 'norm_psd2': nn.LayerNorm(d_model),
                })
            )

    def forward(self, asr_embed, psd_embed) -> Tensor:
        a_output, b_output = asr_embed, psd_embed

        for layer in self.layers:

            c_context, _ = layer['mhsa_cross_a'](a_output, b_output, b_output)
            d_context, _ = layer['mhsa_cross_b'](b_output, a_output, a_output)
            a_output2 = layer['norm_cross_asr'](a_output + c_context)
            b_output2 = layer['norm_cross_psd'](b_output + d_context)
            a_output3, _ = layer['mhsa_asr'](a_output2, a_output2, a_output2)
            b_output3, _ = layer['mhsa_psd'](b_output2, b_output2, b_output2)
            # a_output3 = layer['norm_asr2'](a_output + a_output3 + a_output2)
            # b_output3 = layer['norm_psd2'](b_output + b_output3 + b_output2)

            a_output = layer['norm_asr'](layer['ffn_asr'](a_output3) + a_output3 + a_output)
            b_output = layer['norm_psd'](layer['ffn_psd'](b_output3) + b_output3 + b_output)
            
        return torch.cat((a_output, b_output), dim=1)
    
class MMCA2(nn.Module):
    def __init__(self, num_layers: int = 2, d_model: int = 384, num_heads: int = 8, d_ff: int = 2048):
        super(MMCA2, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads without remainder."

        self.mhsa_cross_a = MultiHeadAttention(d_model, num_heads)
        self.mhsa_cross_b = MultiHeadAttention(d_model, num_heads)
        self.norm_cross_asr = nn.LayerNorm(d_model)
        self.norm_cross_psd = nn.LayerNorm(d_model)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleDict({
                    'mhsa_asr': MultiHeadAttention(d_model, num_heads),
                    'mhsa_psd': MultiHeadAttention(d_model, num_heads),
                    'ffn_asr': PositionWiseFeedForwardNet(d_model, d_ff),
                    'ffn_psd': PositionWiseFeedForwardNet(d_model, d_ff),
                    'norm_asr': nn.LayerNorm(d_model),
                    'norm_psd': nn.LayerNorm(d_model),
                    'norm_asr2': nn.LayerNorm(d_model),
                    'norm_psd2': nn.LayerNorm(d_model),
                })
            )

    def forward(self, asr_embed, psd_embed) -> Tensor:
        a_output, b_output = asr_embed, psd_embed
        
        c_context, _ = self.mhsa_cross_a(a_output, b_output, b_output)
        d_context, _ = self.mhsa_cross_b(b_output, a_output, a_output)
        
        a_output2 = self.norm_cross_asr(a_output + c_context)
        b_output2 = self.norm_cross_psd(b_output + d_context)
        
        for layer in self.layers:
            a_output3, _ = layer['mhsa_asr'](a_output2, a_output2, a_output2)
            b_output3, _ = layer['mhsa_psd'](b_output2, b_output2, b_output2)
            a_output3 = layer['norm_asr2'](a_output + a_output3 + a_output2)
            b_output3 = layer['norm_psd2'](b_output + b_output3 + b_output2)
            a_output2 = layer['norm_asr'](layer['ffn_asr'](a_output3) + a_output3 + a_output)
            b_output2 = layer['norm_psd'](layer['ffn_psd'](b_output3) + b_output3 + b_output)

        return torch.cat((a_output2, b_output2), dim=1)
    

class MMCA3(nn.Module):
    def __init__(self, num_layers: int = 2, d_model: int = 384, num_heads: int = 8, d_ff: int = 2048, dropout_rate: float = 0.1):
        super(MMCA3, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads without remainder."

        self.mhsa_cross_a = MultiHeadAttention(d_model, num_heads)
        self.mhsa_cross_b = MultiHeadAttention(d_model, num_heads)
        self.norm_cross_asr = nn.LayerNorm(d_model)
        self.norm_cross_psd = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleDict({
                    'mhsa_asr': MultiHeadAttention(d_model, num_heads),
                    'mhsa_psd': MultiHeadAttention(d_model, num_heads),
                    'ffn_asr': PositionWiseFeedForwardNet(d_model, d_ff),
                    'ffn_psd': PositionWiseFeedForwardNet(d_model, d_ff),
                    'norm_asr': nn.LayerNorm(d_model),
                    'norm_psd': nn.LayerNorm(d_model),
                    'norm_asr2': nn.LayerNorm(d_model),
                    'norm_psd2': nn.LayerNorm(d_model),
                })
            )

    def forward(self, asr_embed, psd_embed) -> Tensor:
        a_output, b_output = asr_embed, psd_embed
        
        # Cross-attention with dropout
        c_context, _ = self.mhsa_cross_a(a_output, b_output, b_output)
        c_context = self.dropout(c_context)  # Dropout after cross-attention
        d_context, _ = self.mhsa_cross_b(b_output, a_output, a_output)
        d_context = self.dropout(d_context)  # Dropout after cross-attention
        
        a_output2 = self.norm_cross_asr(a_output + c_context)
        b_output2 = self.norm_cross_psd(b_output + d_context)
        
        for layer in self.layers:
            # Self-attention with dropout
            a_output3, _ = layer['mhsa_asr'](a_output2, a_output2, a_output2)
            a_output3 = self.dropout(a_output3)  # Dropout after self-attention
            b_output3, _ = layer['mhsa_psd'](b_output2, b_output2, b_output2)
            b_output3 = self.dropout(b_output3)  # Dropout after self-attention
            
            # Residual connection and layer norm
            a_output3 = layer['norm_asr2'](a_output + a_output3 + a_output2)
            b_output3 = layer['norm_psd2'](b_output + b_output3 + b_output2)
            
            # Feedforward with dropout
            a_output2 = layer['norm_asr'](layer['ffn_asr'](a_output3) + a_output3 + a_output)
            a_output2 = self.dropout(a_output2)  # Dropout after feedforward network
            b_output2 = layer['norm_psd'](layer['ffn_psd'](b_output3) + b_output3 + b_output)
            b_output2 = self.dropout(b_output2)  # Dropout after feedforward network

        return torch.cat((a_output2, b_output2), dim=1)

    
    
class BertRNN(nn.Module):
    def __init__(self, nlayer, nclass, dropout=0.5, nfinetune=0, speaker_info='none', topic_info='none', emb_batch=0):
        super(BertRNN, self).__init__()

        from transformers import AutoModel
        # self.bert = AutoModel.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens")
        self.bert = AutoModel.from_pretrained('roberta-base')
        nhid = self.bert.config.hidden_size

        for param in self.bert.parameters():
            param.requires_grad = False
        n_layers = 12
        if nfinetune > 0:
            for param in self.bert.pooler.parameters():
                param.requires_grad = True
            for i in range(n_layers-1, n_layers-1-nfinetune, -1):
                for param in self.bert.encoder.layer[i].parameters():
                    param.requires_grad = True

        # classifying act tag
        self.encoder = nn.GRU(nhid, nhid//2, num_layers=nlayer, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(nhid, nclass)

        # making use of speaker info
        self.speaker_emb = nn.Embedding(100, nhid)

        # making use of topic info
        self.topic_emb = nn.Embedding(10, nhid)

        self.dropout = nn.Dropout(p=dropout)
        self.nclass = nclass
        self.speaker_info = speaker_info
        self.topic_info = topic_info
        self.emb_batch = emb_batch

    def forward(self, input_ids, attention_mask, chunk_lens, speaker_ids, topic_labels):
        # pdb.set_trace()
        chunk_lens = chunk_lens.to('cpu')
        batch_size, chunk_size, seq_len = input_ids.shape
        speaker_ids = speaker_ids.reshape(-1)   # (batch_size, chunk_size) --> (batch_size*chunk_size)
        chunk_lens = chunk_lens.reshape(-1)   # (batch_size, chunk_size) --> (batch_size*chunk_size)
        topic_labels = topic_labels.reshape(-1)   # (batch_size, chunk_size) --> (batch_size*chunk_size)

        # pdb.set_trace()

        input_ids = input_ids.reshape(-1, seq_len)  # (bs*chunk_size, emb_dim)
        attention_mask = attention_mask.reshape(-1, seq_len)


        if self.training or self.emb_batch == 0:
            embeddings = self.bert(input_ids, attention_mask=attention_mask,
                        output_hidden_states=True)[0][:, 0]  # (bs*chunk_size, emb_dim)
            
        else:
            embeddings_ = []
            dataset2 = TensorDataset(input_ids, attention_mask)
            loader = DataLoader(dataset2, batch_size=self.emb_batch)
            for _, batch in enumerate(loader):
                embeddings = self.bert(batch[0], attention_mask=batch[1], output_hidden_states=True)[0][:, 0]
                embeddings_.append(embeddings)
            embeddings = torch.cat(embeddings_, dim=0)

        nhid = embeddings.shape[-1]

        if self.speaker_info == 'emb_cls':
            speaker_embeddings = self.speaker_emb(speaker_ids)  # (bs*chunk_size, emb_dim)
            embeddings = embeddings + speaker_embeddings    # (bs*chunk_size, emb_dim)
        if self.topic_info == 'emb_cls':
            topic_embeddings = self.topic_emb(topic_labels)     # (bs*chunk_size, emb_dim)
            embeddings = embeddings + topic_embeddings  # (bs*chunk_size, emb_dim)

        # reshape BERT embeddings to fit into RNN
        embeddings = embeddings.reshape(-1, chunk_size, nhid)  # (bs, chunk_size, emd_dim)
        embeddings = embeddings.permute(1, 0, 2)  # (chunk_size, bs, emb_dim)

        # sequence modeling of act tags using RNN
        embeddings = pack_padded_sequence(embeddings, chunk_lens, enforce_sorted=False)
        self.encoder.flatten_parameters()
        outputs, _ = self.encoder(embeddings)
        outputs, _ = pad_packed_sequence(outputs)  # (chunk_size/chunk_len, bs, emb_dim)
        if outputs.shape[0] < chunk_size:
            outputs_padding = torch.zeros(chunk_size - outputs.shape[0], batch_size, nhid, device=outputs.device)
            outputs = torch.cat([outputs, outputs_padding], dim=0)  # (chunk_len, bs, emb_dim)
        outputs = self.dropout(outputs).squeeze(0) # (bs, emb_dim)


        return outputs
